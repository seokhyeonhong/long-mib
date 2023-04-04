import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymovis.ops import rotation
from pymovis.learning.transformer import RelativeMultiHeadAttention, PoswiseFeedForwardNet, LocalMultiHeadAttention
from pymovis.learning.embedding import RelativeSinusoidalPositionalEmbedding

def get_mask(batch, context_frames, ratio_constrained=0.1, prob_constrained=0.5):
    B, T, D = batch.shape

    # 0 for unknown frames, 1 for known frames
    batch_mask = torch.ones_like(batch)
    batch_mask[:, context_frames:-1, :] = 0
    
    # False for known frames, True for unknown frames
    attn_mask = torch.zeros(1, T, T, dtype=torch.bool, device=batch.device)
    attn_mask[:, :, context_frames:-1] = True

    # mask out random partial frames
    constrained_frames = np.arange(context_frames, T-1)
    constrained_frames = np.random.choice(constrained_frames, int(len(constrained_frames) * ratio_constrained), replace=False)
    for t in constrained_frames:
        if np.random.rand() < prob_constrained:
            batch_mask[:, t, :] = 1
            attn_mask[:, :, t] = False
            
    return batch_mask, attn_mask

def get_keyframe_relative_position(window_length, context_frames):
    position = torch.arange(window_length, dtype=torch.float32)
    dist_ctx = position - (context_frames - 1) # distance to the last context frame
    dist_tgt = position - (window_length - 1)  # distance to the target frame

    p_kf = torch.stack([dist_ctx, dist_tgt], dim=-1) # (T, 2)

    return p_kf

class TrajectoryTransformer(nn.Module):
    def __init__(self, d_motion, config):
        super(TrajectoryTransformer, self).__init__()
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
            nn.Linear(self.d_motion * 2, self.d_model), # (motion, mask)
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
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.atten_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(RelativeMultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion - 5), # except trajectory features
        )
    
    def forward(self, x, ratio_constrained=0.1, prob_constrained=0.5):
        B, T, D = x.shape
        
        # mask
        batch_mask, _ = get_mask(x, self.config.context_frames, ratio_constrained, prob_constrained)
        batch_mask[..., -5:] = 1 # no mask for trajectory
        masked_x = x * batch_mask
        x = self.encoder(torch.cat([masked_x, batch_mask], dim=-1))

        # add keyframe positional embedding
        keyframe_pos = get_keyframe_relative_position(T, self.config.context_frames).to(x.device)
        x = x + self.keyframe_pos_encoder(keyframe_pos)

        # relative distance range: [-T+1, ..., T-1], 2T-1 values in total
        rel_dist = torch.arange(-T+1, T, dtype=torch.float32).unsqueeze(-1).to(x.device) # (2T-1, 1)
        lookup_table = self.relative_pos_encoder(rel_dist) # (2T-1, d_model)

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, lookup_table=lookup_table, mask=None) # self-attention
            x = self.pffn_layers[i](x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)

        return x, batch_mask
    
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
            nn.Linear(self.d_motion * 2, self.d_model), # (motion, mask)
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
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.atten_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(RelativeMultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion - 5), # except trajectory features
        )
    
    def forward(self, x, ratio_constrained=0.1, prob_constrained=0.5):
        B, T, D = x.shape
        
        # mask
        batch_mask, _ = get_mask(x, self.config.context_frames, ratio_constrained=ratio_constrained, prob_constrained=prob_constrained)
        batch_mask[..., -5:] = 1 # no mask for trajectory
        masked_x = x * batch_mask
        x = self.encoder(torch.cat([masked_x, batch_mask], dim=-1))

        # add keyframe positional embedding
        keyframe_pos = get_keyframe_relative_position(T, self.config.context_frames).to(x.device)
        x = x + self.keyframe_pos_encoder(keyframe_pos)

        # relative distance
        rel_dist = torch.arange(-T+1, T, dtype=torch.float32).unsqueeze(-1).to(x.device) # (2T-1, 1)
        lookup_table = self.relative_pos_encoder(rel_dist) # (2T-1, d_model)

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, lookup_table, mask=None)
            x = self.pffn_layers[i](x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)

        return x, batch_mask

class KeyframeTransformerLocal(nn.Module):
    def __init__(self, d_motion, config):
        super(KeyframeTransformerLocal, self).__init__()
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
            nn.Linear(self.d_motion * 2, self.d_model), # (motion, mask)
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
            nn.Linear(self.d_model, self.d_motion - 5), # except trajectory features
        )
    
    def forward(self, x, ratio_constrained=0.1, prob_constrained=0.5):
        B, T, D = x.shape
        
        # mask
        batch_mask, _ = get_mask(x, self.config.context_frames, ratio_constrained=ratio_constrained, prob_constrained=prob_constrained)
        batch_mask[..., -5:] = 1 # no mask for trajectory
        masked_x = x * batch_mask
        x = self.encoder(torch.cat([masked_x, batch_mask], dim=-1))

        # add keyframe positional embedding
        keyframe_pos = get_keyframe_relative_position(T, self.config.context_frames).to(x.device)
        x = x + self.keyframe_pos_encoder(keyframe_pos)

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x)
            x = self.pffn_layers[i](x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)

        return x, batch_mask

class InterpolationTransformer(nn.Module):
    def __init__(self, d_motion, config):
        super(InterpolationTransformer, self).__init__()
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
            nn.Linear(self.d_motion - 5, self.d_model),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )
        self.traj_encoder = nn.Sequential(
            nn.Linear(5, self.d_model),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Dropout(self.dropout),
        )
        self.keyframe_pos_encoder = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(self.dropout),
        )

        # Transformer layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.atten_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        self.pffn_layers  = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.atten_layers.append(LocalMultiHeadAttention(self.d_model, self.d_head, self.n_heads, self.config.fps+1, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.cross_layers.append(LocalMultiHeadAttention(self.d_model, self.d_head, self.n_heads, self.config.fps+1, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_motion - 1), # except trajectory features but include contacts
        )
    
    def get_random_keyframes(self, total_frames):
        keyframes = [self.config.context_frames-1]

        transition_start = self.config.context_frames
        while transition_start+self.config.fps < total_frames-1:
            transition_end = min(transition_start + self.config.fps, total_frames-1)
            kf = random.randint(transition_start + 5, transition_end)
            keyframes.append(kf)
            transition_start = kf

        if keyframes[-1] != total_frames - 1:
            keyframes.append(total_frames - 1)
        
        return keyframes

    def get_keyframe_relative_position(self, x, keyframes):
        B, T, D = x.shape
        pos_from_prev = torch.zeros((B, T, 1), dtype=x.dtype, device=x.device)
        pos_from_next = torch.zeros((B, T, 1), dtype=x.dtype, device=x.device)

        for i in range(len(keyframes) - 1):
            kf1, kf2 = keyframes[i], keyframes[i+1]
            pos_from_prev[:, kf1:kf2] = torch.arange(kf2-kf1).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)
            pos_from_next[:, kf1+1:kf2] = torch.arange(-kf2+kf1+1, 0).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)
        
        res = torch.cat([pos_from_prev, pos_from_next], dim=-1)
        return res
    
    def get_interpolated_motion(self, local_R, root_p, keyframes):
        R, p = local_R.clone(), root_p.clone()
        for i in range(len(keyframes) - 1):
            kf1, kf2 = keyframes[i], keyframes[i+1]
            t = torch.arange(0, 1, 1/(kf2-kf1), dtype=R.dtype, device=R.device).unsqueeze(-1)
            
            # interpolate joint orientations
            R1 = R[:, kf1].unsqueeze(1)
            R2 = R[:, kf2].unsqueeze(1)
            R_diff = torch.matmul(R1.transpose(-1, -2), R2)
            angle_diff, axis_diff = rotation.R_to_A(R_diff)
            # angle_diff += torch.randn_like(angle_diff) * 0.01
            angle_diff = t * angle_diff
            axis_diff = axis_diff.repeat(1, len(t), 1, 1)
            R_diff = rotation.A_to_R(angle_diff, axis_diff)

            R[:, kf1:kf2] = torch.matmul(R1, R_diff)

            # interpolate root positions
            p1 = p[:, kf1].unsqueeze(1)
            p2 = p[:, kf2].unsqueeze(1)
            p_diff = p2 - p1
            # p_diff += torch.randn_like(p_diff) * 0.01
            p[:, kf1:kf2] = p1 + t * p_diff
        
        R6 = rotation.R_to_R6(R).reshape(R.shape[0], R.shape[1], -1)
        return torch.cat([R6, p], dim=-1)
    
    def forward(self, motion_interpolated, keyframes, traj):
        B, T, D = motion_interpolated.shape

        # encoder
        motion = self.motion_encoder(motion_interpolated)
        traj   = self.traj_encoder(traj)

        # time embedding
        kf_pos = self.get_keyframe_relative_position(motion_interpolated, keyframes)
        kf_pos = self.keyframe_pos_encoder(kf_pos)

        # additive time embedding
        x = motion + kf_pos
        context = traj + kf_pos

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x)
            x = self.cross_layers[i](x, context)
            x = self.pffn_layers[i](x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)

        motion_residual, contact = torch.split(x, [self.d_motion-5, 4], dim=-1)

        return motion_residual + motion_interpolated, torch.sigmoid(contact)