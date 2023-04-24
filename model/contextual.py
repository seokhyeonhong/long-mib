import numpy as np
import math
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

class ContextualTransformer(nn.Module):
    def __init__(self, d_motion, config, is_context=False):
        super(ContextualTransformer, self).__init__()
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
        self.decoder = nn.Sequential(
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
    
    def get_mask(self, length, device="cuda"):
        # 0 for unknown frames, 1 for known frames
        batch_mask = torch.ones(length, 1, dtype=torch.float32, device=device)
        batch_mask[self.config.context_frames:-1] = 0

        # True for known frames, False for unknown frames
        constrained_frames = torch.ones(length, dtype=torch.bool, device=device)
        constrained_frames[self.config.context_frames:-1] = False

        return batch_mask, constrained_frames
    
    def get_atten_mask(self, constrained_frames, dist_threshold):
        T = len(constrained_frames)

        frame_from = torch.arange(T, device=constrained_frames.device).view(T, 1)
        frame_to   = torch.arange(T, device=constrained_frames.device).view(1, T)
        attn_mask  = torch.abs(frame_to - frame_from) > dist_threshold
        attn_mask[:, ~constrained_frames] = True

        return attn_mask

    def get_noise_weight(self, length):
        t0 = torch.arange(0, length,  1, dtype=torch.float32) - (self.config.context_frames-1)
        t1 = torch.arange(length, 0, -1, dtype=torch.float32) - 1
        t  = torch.clip(torch.min(t0, t1) / self.config.fps, 0, 1)
        return t
    
    def forward(self, motion, traj, mask=None):
        B, T, D = motion.shape

        """ 1. Get bias and mask """
        original_motion = motion.clone()
        atten_bias = self.get_bias(T, device=motion.device)
        if self.is_context:
            mask, constrained_frames = self.get_mask(T, device=motion.device)
            mask = mask.unsqueeze(0).repeat(B, 1, 1)
            motion = motion * mask

        """ 2. Motion encoder """
        x = self.motion_encoder(torch.cat([motion, traj, mask], dim=-1))

        """ 3. Add positional embedding """
        pos = torch.arange(T, device=x.device, dtype=x.dtype)
        x = x + self.embedding(pos)
        
        """ 4. Transformer layers """
        if self.is_context:
            num_transition_frames = T - self.config.context_frames - 1
            kernel_size = math.ceil(num_transition_frames / 2 / self.n_layers) * 2 + 1
            conv_kernel = torch.ones(kernel_size, device=constrained_frames.device, dtype=torch.float).view(1, 1, kernel_size)
            for i in range(self.n_layers):
                atten_mask = self.get_atten_mask(constrained_frames, kernel_size // 2)
                x = self.attn_layers[i](x, x, atten_bias, mask=atten_mask)
                x = self.pffn_layers[i](x)
                constrained_frames = F.conv1d(constrained_frames.float().view(1, 1, T), conv_kernel, padding=kernel_size // 2).view(T).bool()
        else:
            for i in range(self.n_layers):
                x = self.attn_layers[i](x, x, atten_bias)
                x = self.pffn_layers[i](x)
        
        """ 6. Layer norm """
        if self.pre_layernorm:
            x = self.layer_norm(x)

        """ 7. Output """
        x = self.decoder(x)

        # unmask
        x[..., :self.d_motion] = original_motion * mask + x[..., :self.d_motion] * (1 - mask)
        if not self.is_context:
            x[..., self.d_motion:] = torch.sigmoid(x[..., self.d_motion:])

        return x, mask