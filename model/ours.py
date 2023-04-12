import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymovis.ops import rotation
from pymovis.learning.mlp import MultiLinear
from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet, LocalMultiHeadAttention

def get_mask(batch, context_frames, ratio_constrained=0.1, prob_constrained=0.5):
    B, T, D = batch.shape

    # 0 for unknown frames, 1 for known frames
    batch_mask = torch.ones(B, T, 1, dtype=batch.dtype, device=batch.device)
    batch_mask[:, context_frames:-1, :] = 0

    # mask out random partial frames
    constrained_frames = np.arange(context_frames, T-1)
    constrained_frames = np.random.choice(constrained_frames, int(len(constrained_frames) * ratio_constrained), replace=False)
    for t in constrained_frames:
        if np.random.rand() < prob_constrained:
            batch_mask[:, t, :] = 1
            
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
    
    def forward(self, x, ratio_constrained=0.1, prob_constrained=0.5):
        B, T, D = x.shape
        
        motion, traj = torch.split(x, [D-3, 3], dim=-1)

        # mask
        batch_mask = get_mask(x, self.config.context_frames, ratio_constrained, prob_constrained)
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

class RefineEncoder(nn.Module):
    def __init__(self, d_motion, config):
        super(RefineEncoder, self).__init__()
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

        # mean and logvar token
        self.mean_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.logvar_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # encoders
        self.motion_encoder = nn.Sequential(
            nn.Linear(self.d_motion, self.d_model),
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
    
    def forward(self, x):
        B, T, D = x.shape

        # encoder
        x = self.motion_encoder(x)

        # mean and logvar token
        mean_token   = self.mean_token.repeat(B, 1, 1)
        logvar_token = self.logvar_token.repeat(B, 1, 1)
        x = torch.cat([mean_token, logvar_token, x], dim=1)
        T += 2

        # relative position
        half_len = self.config.fps // 2
        rel_pos = torch.arange(-half_len, half_len+1, dtype=x.dtype, device=x.device).unsqueeze(-1) # (2*half_len+1, 1)
        lookup_table = self.relative_pos_encoder(rel_pos) # (2*half_len+1, d_head)
        lookup_table = F.pad(lookup_table, (0, 0, T-half_len-1, T-half_len-1), mode='constant', value=0) # (2*T-1, d_head)

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, lookup_table=lookup_table)
            x = self.pffn_layers[i](x)
        
        if self.pre_layernorm:
            x = self.layer_norm(x)
            
        mean, logvar = x[:, 0], x[:, 1]
        return mean, logvar

class RefineDecoder(nn.Module):
    def __init__(self, d_motion, config):
        super(RefineDecoder, self).__init__()
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

    def get_mask_by_keyframe(self, x, keyframes):
        return _get_mask_by_keyframe(x, self.config.context_frames, keyframes)
    
    def forward(self, x, z, keyframes):
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
            x = self.atten_layers[i](x, z, lookup_table=lookup_table)
            x = self.pffn_layers[i](x)

        if self.pre_layernorm:
            x = self.layer_norm(x)

        x = self.decoder(x)
        return x

class RefineVAE(nn.Module):
    def __init__(self, d_motion, config):
        super(RefineVAE, self).__init__()

        self.d_motion = d_motion
        self.config   = config
        
        self.encoder = RefineEncoder(d_motion, config)
        self.decoder = RefineDecoder(d_motion, config)
    
    def get_random_keyframes(self, total_frames):
        return _get_random_keyframes(self.config.context_frames, self.config.fps, total_frames)
    
    def get_interpolated_motion(self, local_R, root_p, keyframes):
        return _get_interpolated_motion(local_R, root_p, keyframes)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, x_interp, keyframes):
        B, T, D = x.shape
        
        # encode
        mean, logvar = self.encoder.forward(x)
        mean_ = mean[:, None].repeat(1, T, 1)
        logvar_ = logvar[:, None].repeat(1, T, 1)
        z = self.reparameterize(mean_, logvar_)

        # decode
        x_recon = self.decoder.forward(x_interp, z, keyframes)

        return x_recon, mean, logvar