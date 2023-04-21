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

class TwoStageVAE(nn.Module):
    def __init__(self, d_motion, config):
        super(TwoStageVAE, self).__init__()
        self.d_motion = d_motion
        self.config = config

        self.encoder = Encoder(d_motion, config)
        self.ctx_decoder = ContextDecoder(d_motion, config)
        self.det_decoder = DetailDecoder(d_motion, config)
    
    def forward(self, motion, traj):
        B, T, D = motion.shape

        # encoder
        mean, logvar = self.encoder.forward(motion, traj)

        # repeat mean and logvar
        mean = mean.unsqueeze(1).repeat(1, T, 1)
        logvar = logvar.unsqueeze(1).repeat(1, T, 1)
        z = self.encoder.reparameterize(mean, logvar)

        # context decoder
        recon_ctx, mask = self.ctx_decoder.forward(motion, traj, z)

        # detail decoder
        recon_det, contact = self.det_decoder.forward(recon_ctx, traj, mask)

        return recon_ctx, recon_det, contact, mean, logvar
    
    def sample(self, motion, traj):
        B, T, D = motion.shape
        z = torch.randn(B, T, self.config.d_model, device=motion.device)
        recon_ctx, mask = self.ctx_decoder.forward(motion, traj, z)
        recon_det, contact = self.det_decoder.forward(recon_ctx, traj, mask)
        return recon_ctx, recon_det, contact

class Encoder(nn.Module):
    def __init__(self, d_motion, config):
        super(Encoder, self).__init__()
        self.d_motion = d_motion
        self.config = config

        self.d_model        = config.d_model
        self.n_layers       = config.n_layers
        self.n_heads        = config.n_heads
        self.d_head         = self.d_model // self.n_heads
        self.d_ff           = config.d_ff
        self.pre_layernorm  = config.pre_layernorm
        self.dropout        = config.dropout

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")

        # input token for mean and logvar of VAE
        self.mean_token = nn.Parameter(torch.zeros(1, 1, self.d_model), requires_grad=False)
        self.logvar_token = nn.Parameter(torch.zeros(1, 1, self.d_model), requires_grad=False)

        # encoders
        self.conv = nn.Sequential(
            nn.Conv1d(self.d_motion + 5, self.d_model, kernel_size=7, padding=3), # (motion, traj)
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=7, padding=3),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
        )

        # Transformer layers
        self.attn_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # layer norm
        self.layer_norm = nn.LayerNorm(self.d_model)

        # output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, motion, traj):
        B, T, D = motion.shape

        """ 1. Convolutional encoder """
        x = torch.cat([motion, traj], dim=-1) # (B, T, D+5)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2) # (B, T, d_model)

        """ 2. Concat mean and logvar tokens """
        mean = self.mean_token.repeat(B, 1, 1)
        logvar = self.logvar_token.repeat(B, 1, 1)
        x = torch.cat([mean, logvar, x], dim=1)
        T += 2
        
        """ 3. Transformer layers """
        for i in range(self.n_layers):
            x = self.attn_layers[i](x, x)
            x = self.pffn_layers[i](x)
        
        """ 4. Layer norm """
        if self.pre_layernorm:
            x = self.layer_norm(x)

        """ 5. Output layer """
        x = self.output_layer(x)
        if not self.pre_layernorm:
            x = self.layer_norm(x)

        mean, logvar = x[:, 0], x[:, 1]
        return mean, logvar

class ContextDecoder(nn.Module):
    def __init__(self, d_motion, config):
        super(ContextDecoder, self).__init__()
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

        # encoders
        self.encoder = nn.Sequential(
            nn.Linear(self.d_motion + 1 + 5, self.d_model), # (motion, mask, traj)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        
        # embedding
        self.embedding = PeriodicPositionalEmbedding(self.d_model, self.period)

        # Transformer layers
        self.attn_layers  = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadContextualBiasedAttention(self.d_model, self.d_head, self.n_heads, self.period, dropout=self.dropout))
            self.cross_layers.append(MultiHeadContextualBiasedAttention(self.d_model, self.d_head, self.n_heads, self.period, dropout=self.dropout))
            self.pffn_layers.append(nn.Sequential(nn.Linear(self.d_model*2, self.d_ff),
                                                  nn.ReLU(),
                                                  nn.Dropout(self.dropout),
                                                  nn.Linear(self.d_ff, self.d_model),
                                                  nn.Dropout(self.dropout)))
        
        # output
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion),
        )
    
    def get_self_bias(self, T, device="cuda"):
        # temporal distance
        frame_from = torch.arange(T, dtype=torch.float32, device=device).view(T, 1)
        frame_to   = torch.arange(T, dtype=torch.float32, device=device).view(1, T)

        # mask: True to fill -inf
        mask = (frame_to - frame_from) > 0
        mask[:, :self.config.context_frames] = False
        mask[:, -1] = False

        # temporal distance
        frame_bias = -(torch.abs(frame_to - frame_from) // self.period)
        frame_bias = frame_bias.masked_fill(mask, -1e9)
        return frame_bias
    
    def get_cross_bias(self, T, device="cuda"):
        # diagonal zero else -inf
        mask = 1 - torch.eye(T, dtype=torch.float32, device=device)
        mask = mask.masked_fill(mask==1, -1e9)
        return mask

    def forward(self, motion, traj, z):
        B, T, D = motion.shape

        # original motion
        original_motion = motion.clone()

        # get mask
        batch_mask = get_mask(motion, self.config.context_frames)
        x = torch.cat([motion*batch_mask, traj, batch_mask], dim=-1)

        # encoder
        x = self.encoder(x) # (B, T, d_model)
        
        # keyframe position encoding
        p_kf = self.embedding(torch.arange(T, dtype=torch.float32).to(x.device)) # (T, d_model)
        x = x + p_kf # (B, T, d_model)

        # Transformer layers
        self_bias = self.get_self_bias(T, device=x.device)
        cross_bias = self.get_cross_bias(T, device=x.device)

        for attn_layer, cross_layer, pffn_layer in zip(self.attn_layers, self.cross_layers, self.pffn_layers):
            out_self  = attn_layer(x, x, self_bias)
            out_cross = cross_layer(out_self, z, cross_bias)
            out_pffn  = pffn_layer(torch.cat([x, out_cross], dim=-1))
            x = x + out_pffn
        
        # decoder
        x = self.decoder(x)

        # output
        x = original_motion * batch_mask + x * (1 - batch_mask)

        return x, batch_mask

class DetailDecoder(nn.Module):
    def __init__(self, d_motion, config):
        super(DetailDecoder, self).__init__()
        self.d_motion = d_motion
        self.config = config

        self.d_model       = config.d_model
        self.n_layers      = config.n_layers
        self.n_heads       = config.n_heads
        self.d_head        = self.d_model // self.n_heads
        self.d_ff          = config.d_ff
        self.pre_layernorm = config.pre_layernorm
        self.dropout       = config.dropout
        self.period        = config.fps

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")

        # encoders
        self.encoder = nn.Sequential(
            nn.Linear(self.d_motion + 1 + 5, self.d_model), # (motion, mask, traj)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        
        # embedding
        self.embedding = PeriodicPositionalEmbedding(self.d_model, self.period)

        # Transformer layers
        self.attn_layers  = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadBiasedAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
        
        # head-specific weight
        self.head_specific_sclae = nn.Parameter(torch.pow(2, -torch.arange(1, self.n_heads + 1, dtype=torch.float32)), requires_grad=False)

        # layer normalization
        self.layernorm = nn.LayerNorm(self.d_model)

        # output
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion + 4), # (motion, contact)
        )
    
    def get_bias(self, T, device="cuda"):
        # temporal distance
        frame_from = torch.arange(T, dtype=torch.float32, device=device).view(T, 1)
        frame_to   = torch.arange(T, dtype=torch.float32, device=device).view(1, T)
        frame_bias = -(torch.abs(frame_to - frame_from) // self.period) # (T, T)
        frame_bias = frame_bias.unsqueeze(0) * self.head_specific_sclae.view(self.n_heads, 1, 1)
        return frame_bias

    def forward(self, motion, traj, mask):
        B, T, D = motion.shape

        # original motion
        original_motion = motion.clone()

        # encoder
        x = self.encoder(torch.cat([motion, traj, mask], dim=-1)) # (B, T, d_model)
        
        # keyframe position encoding
        p_kf = self.embedding(torch.arange(T, dtype=torch.float32).to(x.device)) # (T, d_model)
        x = x + p_kf # (B, T, d_model)

        # Transformer layers
        bias = self.get_bias(T, device=x.device)
        for attn_layer, pffn_layer in zip(self.attn_layers, self.pffn_layers):
            x = attn_layer(x, x, bias)
            x = pffn_layer(x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layernorm(x)

        x = self.decoder(x)

        # output
        motion, contact = torch.split(x, [self.d_motion, 4], dim=-1)
        motion = original_motion * mask + motion * (1 - mask)
        contact = torch.sigmoid(contact)

        return motion, contact