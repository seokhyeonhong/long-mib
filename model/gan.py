import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import PeriodicPositionalEmbedding
from .transformer import MultiHeadContextualBiasedAttention
from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet, MultiHeadBiasedAttention

def get_mask(batch, context_frames):
    B, T, D = batch.shape

    # 0 for unknown frames, 1 for known frames
    batch_mask = torch.ones(B, T, 1, dtype=batch.dtype, device=batch.device)
    batch_mask[:, context_frames:-1, :] = 0
    return batch_mask

class TwoStageGAN(nn.Module):
    def __init__(self, d_motion, config):
        super(TwoStageGAN, self).__init__()
        self.d_motion = d_motion
        self.config = config

        self.G_ctx = Generator(d_motion, config, is_context=True)
        self.G_det = Generator(d_motion, config, is_context=False)

        self.D_ctx = Discriminator(d_motion, config)
        self.D_det = Discriminator(d_motion, config)

    def generate(self, motion, traj):
        ctx_motion = self.G_ctx.forward(motion, traj)
        det_motion = self.G_det.forward(ctx_motion, traj)
        det_motion, det_contact = torch.split(det_motion, [self.d_motion, 4], dim=-1)

        return ctx_motion, det_motion, det_contact
    
    def discriminate(self, ctx_motion, det_motion):
        ctx_short_score, ctx_long_score = self.D_ctx.forward(ctx_motion)
        det_short_score, det_long_score = self.D_det.forward(det_motion)

        return ctx_short_score, ctx_long_score, det_short_score, det_long_score

class Generator(nn.Module):
    def __init__(self, d_motion, config, is_context=False):
        super(Generator, self).__init__()
        self.d_motion = d_motion
        self.config = config
        self.is_context = is_context

        self.d_model        = config.d_model
        self.n_layers       = config.n_layers
        self.n_heads        = config.n_heads
        self.d_head         = self.d_model // self.n_heads
        self.d_ff           = config.d_ff
        self.pre_layernorm  = config.pre_layernorm
        self.dropout        = config.dropout
        self.period         = config.fps // 2

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")

        # encoders
        self.motion_encoder = nn.Sequential(
            nn.Linear(self.d_motion + 5 + 1, self.d_model), # (motion, traj(=5), mask(=1))
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
        )

        # periodic positional embedding
        self.embedding = PeriodicPositionalEmbedding(self.d_model, period=self.period)

        # Transformer layers
        self.attn_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadBiasedAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # layer norm
        self.layer_norm = nn.LayerNorm(self.d_model)

        # output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_motion if self.is_context else self.d_motion + 4),
        )
    
    def get_bias(self, length, device="cuda"):
        frame_from = torch.arange(length, device=device).view(length, 1)
        frame_to   = torch.arange(length, device=device).view(1, length)

        bias       = -(torch.abs(frame_to - frame_from) // self.period).float()
        return bias
    
    def get_mask(self, batch):
        B, T, D = batch.shape

        # 0 for unknown frames, 1 for known frames
        batch_mask = torch.ones(B, T, 1, dtype=batch.dtype, device=batch.device)
        batch_mask[:, self.config.context_frames:-1, :] = 0

        # True for unknown frames, False for known frames
        atten_mask = torch.zeros(T, T, dtype=torch.bool, device=batch.device)
        atten_mask[:, self.config.context_frames:-1] = True

        return batch_mask, atten_mask

    def forward(self, motion, traj):
        B, T, D = motion.shape

        """ 1. Get mask and bias """
        original_motion = motion.clone()
        batch_mask, atten_mask = self.get_mask(motion)
        atten_bias = self.get_bias(T, device=motion.device)

        """ 2. Motion encoder """
        if self.is_context:
            x = torch.cat([motion*batch_mask, traj, batch_mask], dim=-1)
        else:
            x = torch.cat([motion, traj, batch_mask], dim=-1)
        x = self.motion_encoder(x)

        """ 3. Add random noise """
        z = torch.randn(B, T, self.d_model, device=x.device, dtype=x.dtype)
        x = x + z

        """ 4. Add positional embedding """
        pos = torch.arange(T, device=x.device, dtype=x.dtype)
        x = x + self.embedding(pos)
        
        """ 5. Transformer layers """
        for i in range(self.n_layers):
            x = self.attn_layers[i](x, x, atten_bias, mask=atten_mask if self.is_context else None)
            x = self.pffn_layers[i](x)
        
        """ 6. Output layer """
        if self.pre_layernorm:
            x = self.layer_norm(x)

        x = self.output_layer(x)

        """ 7. Final output """
        x[..., :self.d_motion] = original_motion * batch_mask + x[..., :self.d_motion] * (1 - batch_mask)
        if not self.is_context:
            x[..., self.d_motion:] = torch.sigmoid(x[..., self.d_motion:])

        return x

class Discriminator(nn.Module):
    def __init__(self, d_motion, config):
        super(Discriminator, self).__init__()
        self.d_motion = d_motion
        self.config = config

        self.d_model  = config.d_model
        self.n_layers = config.n_layers
        self.n_heads  = config.n_heads
        self.d_head   = self.d_model // self.n_heads
        self.d_ff     = config.d_ff
        self.dropout  = config.dropout
        self.period   = config.fps

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")

        # short discriminator
        self.short_discriminator = nn.Sequential(
            nn.Conv1d(self.d_motion, self.d_model, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(self.d_model, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

        # long discriminator
        self.long_discriminator = nn.Sequential(
            nn.Conv1d(self.d_motion, self.d_model, kernel_size=15, padding=7),
            nn.LeakyReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=15, padding=7),
            nn.LeakyReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=15, padding=7),
            nn.LeakyReLU(),
            nn.Conv1d(self.d_model, 1, kernel_size=15, padding=7),
            nn.Sigmoid()
        )

    def forward(self, motion):
        B, T, D = motion.shape

        """ 1. Short discriminator """
        short_score = self.short_discriminator(motion.transpose(1, 2)).squeeze(-1)

        """ 2. Long discriminator """
        long_score = self.long_discriminator(motion.transpose(1, 2)).squeeze(-1)
        
        return short_score, long_score