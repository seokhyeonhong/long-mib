import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymovis.ops import rotation
from pymovis.learning.mlp import MultiLinear
from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet, LocalMultiHeadAttention

def get_mask(batch, context_frames):
    B, T, D = batch.shape

    # 0 for unknown frames, 1 for known frames
    batch_mask = torch.ones(B, T, 1, dtype=batch.dtype, device=batch.device)
    batch_mask[:, context_frames:-1, :] = 0
    return batch_mask

def get_keyframe_relative_position(window_length, context_frames):
    position = torch.arange(window_length, dtype=torch.float32)
    dist_ctx = position - (context_frames - 1) # distance to the last context frame
    dist_tgt = position - (window_length - 1)  # distance to the target frame

    p_kf = torch.stack([dist_ctx, dist_tgt], dim=-1) # (T, 2)

    return p_kf

def _get_interpolated_motion(local_R, root_p, keyframes):
    R, p = local_R.clone(), root_p.clone()
    for i in range(len(keyframes) - 1):
        kf1, kf2 = keyframes[i], keyframes[i+1]
        t = torch.arange(0, 1, 1/(kf2-kf1), dtype=R.dtype, device=R.device).unsqueeze(-1)
        
        # interpolate joint orientations
        R1 = R[:, kf1].unsqueeze(1)
        R2 = R[:, kf2].unsqueeze(1)
        R_diff = torch.matmul(R1.transpose(-1, -2), R2)

        angle_diff, axis_diff = rotation.R_to_A(R_diff)
        angle_diff = t * angle_diff
        axis_diff = axis_diff.repeat(1, len(t), 1, 1)
        R_diff = rotation.A_to_R(angle_diff, axis_diff)

        R[:, kf1:kf2] = torch.matmul(R1, R_diff)

        # interpolate root positions
        p1 = p[:, kf1].unsqueeze(1)
        p2 = p[:, kf2].unsqueeze(1)
        p[:, kf1:kf2] = p1 + t * (p2 - p1)
    
    R6 = rotation.R_to_R6(R).reshape(R.shape[0], R.shape[1], -1)
    return torch.cat([R6, p], dim=-1)

def _get_random_keyframes(t_ctx, t_max, t_total):
    keyframes = [t_ctx-1]

    transition_start = t_ctx
    while transition_start + t_max < t_total - 1:
        transition_end = min(transition_start + t_max, t_total - 1)
        kf = random.randint(transition_start + 5, transition_end)
        keyframes.append(kf)
        transition_start = kf

    if keyframes[-1] != t_total - 1:
        keyframes.append(t_total - 1)
    
    return keyframes

def _get_mask_by_keyframe(x, t_ctx, keyframes=None):
    B, T, D = x.shape
    mask = torch.zeros(B, T, 1, dtype=x.dtype, device=x.device)
    mask[:, :t_ctx] = 1
    mask[:, -1] = 1
    if keyframes is not None:
        mask[:, keyframes] = 1
    return mask

class KeyframeVAE(nn.Module):
    def __init__(self, d_motion, config):
        super(KeyframeVAE, self).__init__()
        self.d_motion = d_motion
        self.config = config

        self.encoder = KeyframeEncoder(d_motion, config)
        self.decoder = KeyframeDecoder(d_motion, config)

class KeyframeEncoder(nn.Module):
    def __init__(self, d_motion, config):
        super(KeyframeEncoder, self).__init__()
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
        self.mean_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.logvar_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # encoders
        self.encoder = nn.Sequential(
            nn.Linear(self.d_motion + 3, self.d_model), # (motion, mask, mu, logvar)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        self.keyframe_pos_encoder = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(self.dropout),
        )
        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_head),
            nn.Dropout(self.dropout),
        )

        # Transformer layers
        self.layer_norm  = nn.LayerNorm(self.d_model)
        self.attn_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
        
        # output
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion - 3), # except trajectory features
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        B, T, D = x.shape

        # split
        motion, traj = torch.split(x, [D-3, 3], dim=-1)

        # get mask
        batch_mask = get_mask(x, self.config.context_frames, ratio_constrained=0.0, prob_constrained=0.0)
        x = torch.cat([motion*batch_mask, traj, batch_mask], dim=-1)

        # concat mean and logvar tokens
        x = torch.cat([self.mean_token.repeat(B, 1, 1), self.logvar_token.repeat(B, 1, 1), x], dim=1)
        T += 2

        # encoder
        x = self.encoder(x)
        
        # keyframe position encoding
        keyframe_pos = get_keyframe_relative_position(T, self.config.context_frames+2)
        keyframe_pos = self.keyframe_pos_encoder(keyframe_pos)
        x = x + keyframe_pos

        # relative position encoding
        lookup_table = torch.arange(-T+1, T, dtype=torch.float32).unsqueeze(-1).to(x.device) # (2T-1, 1)
        lookup_table = self.relative_pos_encoder(lookup_table) # (2T-1, d_head)

        # Transformer layers
        for attn_layer, pffn_layer in zip(self.attn_layers, self.pffn_layers):
            x = attn_layer(x, x, lookup_table=lookup_table, mask=None)
            x = pffn_layer(x)
        
        # output
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)
        mean, logvar = x[:, 0], x[:, 1]
        return mean, logvar

class KeyframeDecoder(nn.Module):
    def __init__(self, d_motion, config):
        super(KeyframeDecoder, self).__init__()
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

        # encoders
        self.encoder = nn.Sequential(
            nn.Linear(self.d_motion + 1, self.d_model), # (motion, mask)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        self.keyframe_pos_encoder = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(self.dropout),
        )
        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_head),
            nn.Dropout(self.dropout),
        )

        # Transformer layers
        self.layer_norm  = nn.LayerNorm(self.d_model)
        self.attn_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
        
        # output
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion - 3), # except trajectory features
        )

    def forward(self, x, z):
        B, T, D = x.shape

        # split
        motion, traj = torch.split(x, [D-3, 3], dim=-1)

        # get mask
        batch_mask = get_mask(x, self.config.context_frames, ratio_constrained=0.0, prob_constrained=0.0)
        x = torch.cat([motion*batch_mask, traj, batch_mask], dim=-1)

        # encoder
        x = self.encoder(x)
        
        # keyframe position encoding
        keyframe_pos = get_keyframe_relative_position(T, self.config.context_frames)
        keyframe_pos = self.keyframe_pos_encoder(keyframe_pos)
        x = x + keyframe_pos

        # relative position encoding
        lookup_table = torch.arange(-T+1, T, dtype=torch.float32).unsqueeze(-1).to(x.device) # (2T-1, 1)
        lookup_table = self.relative_pos_encoder(lookup_table) # (2T-1, d_head)

        # Transformer layers
        for attn_layer, pffn_layer in zip(self.attn_layers, self.pffn_layers):
            x = attn_layer(x, z, lookup_table=lookup_table, mask=None)
            x = pffn_layer(x)
        
        # output
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)
        return x

class KeyframeTransformer(nn.Module):
    def __init__(self, d_motion, config):
        super(KeyframeTransformer, self).__init__()
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
        
        # encoders
        self.encoder = nn.Sequential(
            nn.Linear(self.d_motion + 1, self.d_model), # (motion, mask)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        self.keyframe_pos_encoder = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(self.dropout),
        )
        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_head),
            nn.Dropout(self.dropout),
        )

        # Transformer layers
        self.layer_norm  = nn.LayerNorm(self.d_model)
        self.attn_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion - 3), # except trajectory features
        )
    
    def forward(self, x):
        B, T, D = x.shape
        
        motion, traj = torch.split(x, [D-3, 3], dim=-1)

        # mask
        batch_mask = get_mask(x, self.config.context_frames)
        x = self.encoder(torch.cat([motion * batch_mask, traj, batch_mask], dim=-1))

        # add keyframe positional embedding
        keyframe_pos = get_keyframe_relative_position(T, self.config.context_frames).to(x.device)
        keyframe_pos = self.keyframe_pos_encoder(keyframe_pos)
        x = x + keyframe_pos

        # relative distance
        lookup_table = torch.arange(-T+1, T, dtype=torch.float32).unsqueeze(-1).to(x.device) # (2T-1, 1)
        lookup_table = self.relative_pos_encoder(lookup_table) # (2T-1, d_head)

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = self.attn_layers[i](x, x, mask=None, lookup_table=lookup_table)
            x = self.pffn_layers[i](x)

        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)
        return x

class InterpolationTransformerGlobal(nn.Module):
    def __init__(self, d_motion, config):
        super(InterpolationTransformerGlobal, self).__init__()
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
        
        # encoders
        self.motion_encoder = nn.Sequential(
            nn.Linear(self.d_motion + 1, self.d_model), # (motion, mask)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_head),
            nn.Dropout(self.dropout),
        )

        # Transformer layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.atten_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion - 3), # except trajectory features
        )
    
    def get_random_keyframes(self, total_frames):
        return _get_random_keyframes(self.config.context_frames, self.config.fps, total_frames)
    
    def get_mask_by_keyframe(self, x, keyframes):
        return _get_mask_by_keyframe(x, self.config.context_frames, keyframes)

    def get_interpolated_motion(self, local_R, root_p, keyframes):
        return _get_interpolated_motion(local_R, root_p, keyframes)
    
    def forward(self, x, keyframes):
        B, T, D = x.shape

        original_motion = x[..., :-3].clone()
        
        # mask
        mask = self.get_mask_by_keyframe(x, keyframes)
        x = self.motion_encoder(torch.cat([x, mask], dim=-1))
        
        # relative distance range: [-T+1, ..., T-1], 2T-1 values in total
        rel_dist = torch.arange(-T+1, T, dtype=torch.float32).unsqueeze(-1).to(x.device) # (2T-1, 1)
        lookup_table = self.relative_pos_encoder(rel_dist) # (2T-1, d_model)

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, lookup_table=lookup_table, mask=None)
            x = self.pffn_layers[i](x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x) # residual of the input motion

        motion = original_motion + x

        return motion

class InterpolationTransformerLocal(nn.Module):
    def __init__(self, d_motion, config):
        super(InterpolationTransformerLocal, self).__init__()
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
        
        # encoders
        self.motion_encoder = nn.Sequential(
            nn.Linear(self.d_motion + 1, self.d_model), # (motion, mask)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_head),
            nn.Dropout(self.dropout),
        )

        # Transformer layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.atten_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(LocalMultiHeadAttention(self.d_model, self.d_head, self.n_heads, self.config.fps+1, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion - 3), # except trajectory features
        )
    
    def get_random_keyframes(self, total_frames):
        return _get_random_keyframes(self.config.context_frames, self.config.fps, total_frames)

    def get_mask_by_keyframe(self, x, keyframes):
        return _get_mask_by_keyframe(x, self.config.context_frames, keyframes)

    def get_interpolated_motion(self, local_R, root_p, keyframes):
        return _get_interpolated_motion(local_R, root_p, keyframes)
    
    def forward(self, x, keyframes):
        B, T, D = x.shape

        # mask
        mask = self.get_mask_by_keyframe(x, keyframes)
        x = self.motion_encoder(torch.cat([x, mask], dim=-1))

        # relative position
        half_len = self.config.fps // 2
        rel_pos = torch.arange(-half_len, half_len+1, dtype=x.dtype, device=x.device).unsqueeze(-1) # (2*half_len+1, 1)
        lookup_table = self.relative_pos_encoder(rel_pos) # (2*half_len+1, d_head)
        lookup_table = F.pad(lookup_table, (0, 0, T-half_len-1, T-half_len-1), mode='constant', value=0) # (2*T-1, d_head)

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, lookup_table=lookup_table)
            x = self.pffn_layers[i](x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)

        return x

class RefineDiscriminator(nn.Module):
    def __init__(self, d_motion, config):
        super(RefineDiscriminator, self).__init__()
        self.d_motion = d_motion
        self.config   = config

        self.d_model        = config.d_model

        self.short_term_disc = nn.Sequential(
            nn.Conv1d(self.d_motion - 3, self.d_model, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv1d(self.d_model, 1, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
        )

        self.long_term_disc = nn.Sequential(
            nn.Conv1d(self.d_motion - 3, self.d_model, kernel_size=15, stride=1, padding=7),
            nn.PReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=15, stride=1, padding=7),
            nn.PReLU(),
            nn.Conv1d(self.d_model, 1, kernel_size=15, stride=1, padding=7),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, T, D = x.shape

        # transpose
        x_ = x.transpose(1, 2)

        # short-term discriminator
        x_short = self.short_term_disc(x_)

        # long-term discriminator
        x_long = self.long_term_disc(x_)

        # concat
        x = torch.cat([x_short, x_long], dim=1)

        return x
    
class RefineGAN(nn.Module):
    def __init__(self, d_motion, config):
        super(RefineGAN, self).__init__()
        self.d_motion = d_motion
        self.config   = config
        
        self.generator = InterpolationTransformerLocal(d_motion, config)
        self.discriminator = RefineDiscriminator(d_motion, config)

    def get_random_keyframes(self, total_frames):
        return _get_random_keyframes(self.config.context_frames, self.config.fps, total_frames)

    def get_mask_by_keyframe(self, x, keyframes):
        return _get_mask_by_keyframe(x, self.config.context_frames, keyframes)

    def get_interpolated_motion(self, local_R, root_p, keyframes):
        return _get_interpolated_motion(local_R, root_p, keyframes)

    def generate(self, x_interp, keyframes):
        return self.generator.forward(x_interp, keyframes)
    
    def discriminate(self, x):
        return self.discriminator.forward(x)

class KeyframeGenerator(nn.Module):
    def __init__(self, d_motion, config):
        super(KeyframeGenerator, self).__init__()
        self.d_motion       = d_motion
        self.config         = config
    
        self.d_model        = config.d_model
        self.n_layers       = config.n_layers
        self.n_heads        = config.n_heads
        self.d_head         = self.d_model // self.n_heads
        self.d_ff           = config.d_ff
        self.pre_layernorm  = config.pre_layernorm
        self.dropout        = config.dropout

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")
        
        # encoders
        self.noise_encoder = nn.Sequential(
            nn.Linear(self.d_motion, self.d_model),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(self.d_motion + 4, self.d_model), # (motion, mask(=1), trajectory(=3))
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )

        # relative positional encoder
        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_head),
            nn.Dropout(self.dropout),
        )

        # Transformer layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.atten_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.cross_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_motion + 1), # (motion, keyframe score(=1))
        )
    
    def forward(self, GT_motion):
        B, T, D = GT_motion.shape
        
        # noise
        z = torch.randn(B, T, self.d_motion).to(GT_motion.device)
        z = self.noise_encoder(z)

        # context
        mask = get_mask(GT_motion, self.config.context_frames, ratio_constrained=0.0, prob_constrained=0.0)
        motion, traj = torch.split(GT_motion, [D-3, 3], dim=-1)
        context = torch.cat([motion*mask, traj, mask], dim=-1)
        context = self.context_encoder(context)

        # relative positional encoding
        rel_pos = torch.arange(-T+1, T, dtype=torch.float32).to(GT_motion.device)
        rel_pos = self.relative_pos_encoder(rel_pos.unsqueeze(-1)) # (2T-1, d_head)

        # Transformer layers
        for i in range(self.n_layers):
            z = self.atten_layers[i](z, z, lookup_table=rel_pos)
            z = self.cross_layers[i](z, context, lookup_table=rel_pos)
            z = self.pffn_layers[i](z)

        if self.pre_layernorm:
            z = self.layer_norm(z)

        # decoder
        z = self.decoder(z)

        motion, keyframe_score = torch.split(z, [self.d_motion, 1], dim=-1)
        keyframe_score = torch.sigmoid(keyframe_score)

        return motion, keyframe_score

class KeyframeDiscriminator(nn.Module):
    def __init__(self, d_motion, config):
        super(KeyframeDiscriminator, self).__init__()
        self.d_motion = d_motion
        self.config   = config
        
        self.d_model        = config.d_model
        self.n_layers       = config.n_layers
        self.n_heads        = config.n_heads
        self.d_head         = self.d_model // self.n_heads
        self.d_ff           = config.d_ff
        self.pre_layernorm  = config.pre_layernorm
        self.dropout        = config.dropout

        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, but d_model={self.d_model} and num_heads={self.n_heads}")
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.d_motion, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
        )

        # local discriminator - 1D convolution
        self.local_disc = nn.ModuleList()
        for _ in range(self.n_layers):
            self.local_disc.append(nn.Sequential(
                nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
                nn.SiLU()))
        self.local_disc.append(nn.Conv1d(self.d_model, 1, kernel_size=3, padding=1))
        self.local_disc.append(nn.Sigmoid())
        self.local_disc = nn.Sequential(*self.local_disc)

        # global discriminator - Transformer
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_head),
        )

        self.global_disc_atten = nn.ModuleList()
        self.global_disc_pffn  = nn.ModuleList()
        for _ in range(self.n_layers):
            self.global_disc_atten.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.global_disc_pffn.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
        self.global_disc_linear = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid())

    def forward(self, x):
        B, T, D = x.shape

        # encoder
        x = self.encoder(x)

        """ Local discriminator """
        local = self.local_disc(x.transpose(1, 2)).transpose(1, 2)

        """ Global discriminator """
        # relative positional encoding
        rel_pos = torch.arange(-T+1, T, dtype=torch.float32).to(x.device)
        rel_pos = self.relative_pos_encoder(rel_pos.unsqueeze(-1)) # (2T-1, d_head)

        for i in range(self.n_layers):
            x = self.global_disc_atten[i](x, x, lookup_table=rel_pos)
            x = self.global_disc_pffn[i](x)

        if not self.pre_layernorm:
            x = self.layer_norm(x)

        global_ = self.global_disc_linear(x)

        scores = torch.cat([local, global_], dim=-1)
        return scores

class KeyframeGAN(nn.Module):
    def __init__(self, d_motion, config):
        super(KeyframeGAN, self).__init__()
        
        self.generator = KeyframeGenerator(d_motion, config)
        self.discriminator = KeyframeDiscriminator(d_motion, config)

    def generate(self, x):
        return self.generator(x)
    
    def discriminate(self, x):
        return self.discriminator(x)

class KeyframeContrastive(nn.Module):
    def __init__(self, d_motion, config):
        super(KeyframeContrastive, self).__init__()
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
        
        # encoders
        self.encoder = nn.Sequential(
            nn.Linear(self.d_motion + 1, self.d_model), # (motion, mask)
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        self.keyframe_pos_encoder = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(self.dropout),
        )
        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_head),
            nn.Dropout(self.dropout),
        )

        # Transformer layers
        self.layer_norm  = nn.LayerNorm(self.d_model)
        self.attn_layers = nn.ModuleList()
        self.pffn_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion - 3 + 1), # except trajectory features but include keyframe score
        )
    
    def forward(self, x_positive, x_negative):
        B, T, D = x_positive.shape
        
        motion, traj = torch.split(x_positive, [D-3, 3], dim=-1)

        # anchor
        batch_mask = get_mask(x_positive, self.config.context_frames)
        x_anchor = torch.cat([motion * batch_mask, traj, batch_mask], dim=-1)

        # concat mask feature
        x_positive = torch.cat([x_positive, torch.ones_like(batch_mask)], dim=-1)
        x_negative = torch.cat([x_negative, torch.ones_like(batch_mask)], dim=-1)

        # encoder
        x_anchor   = self.encoder(x_anchor)
        x_positive = self.encoder(x_positive)
        x_negative = self.encoder(x_negative)

        # add keyframe positional embedding
        keyframe_pos = get_keyframe_relative_position(T, self.config.context_frames).to(x_anchor.device)
        keyframe_pos = self.keyframe_pos_encoder(keyframe_pos)

        # relative distance
        lookup_table = torch.arange(-T+1, T, dtype=torch.float32).unsqueeze(-1).to(x_anchor.device) # (2T-1, 1)
        lookup_table = self.relative_pos_encoder(lookup_table) # (2T-1, d_head)

        def propagate_transformer(x_in):
            x_out = x_in + keyframe_pos

            # Transformer encoder layers
            for i in range(self.n_layers):
                x_out = self.attn_layers[i](x_out, x_out, mask=None, lookup_table=lookup_table)
                x_out = self.pffn_layers[i](x_out)

            # decoder
            if self.pre_layernorm:
                x_out = self.layer_norm(x_out)
            
            return x_out
        
        x_anchor   = propagate_transformer(x_anchor)
        x_positive = propagate_transformer(x_positive)
        x_negative = propagate_transformer(x_negative)
        
        x_out = self.decoder(x_anchor)
        motion, kf_score = torch.split(x_out, [D-3, 1], dim=-1)
        kf_score = torch.sigmoid(kf_score)
        return motion, kf_score, x_anchor, x_positive, x_negative