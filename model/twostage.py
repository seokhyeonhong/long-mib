import numpy as np
import torch
import torch.nn as nn

from pymovis.learning.transformer import MultiHeadAttention, PoswiseFeedForwardNet
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

class ContextTransformer(nn.Module):
    def __init__(self, d_motion, config):
        super(ContextTransformer, self).__init__()
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
            self.atten_layers.append(MultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion),
        )
    
    def forward(self, x, ratio_constrained=0.1, prob_constrained=0.3):
        B, T, D = x.shape

        original_x = x.clone()
        
        # mask
        batch_mask, atten_mask = get_mask(x, self.config.context_frames, ratio_constrained, prob_constrained)
        masked_x = x * batch_mask

        # linear interpolation of root position
        t = torch.linspace(0, 1, T-self.config.context_frames+1, device=x.device).unsqueeze(0).unsqueeze(-1) # (1, T, 1)
        delta = (masked_x[:, -1, -3:] - masked_x[:, self.config.context_frames-1, -3:]).unsqueeze(1) # (B, 1, 3)
        masked_x[:, self.config.context_frames-1:, -3:] = masked_x[:, self.config.context_frames-1:self.config.context_frames, -3:] + t * delta
        x = self.encoder(torch.cat([masked_x, batch_mask], dim=-1))

        # add keyframe positional embedding
        keyframe_pos = get_keyframe_relative_position(T, self.config.context_frames).to(x.device)
        x = x + self.keyframe_pos_encoder(keyframe_pos)

        # relative distance range: [-T+1, ..., T-1], 2T-1 values in total
        rel_dist = torch.arange(-T+1, T, dtype=torch.float32).unsqueeze(-1).to(x.device) # (2T-1, 1)
        lookup_table = self.relative_pos_encoder(rel_dist) # (2T-1, d_model)

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, mask=atten_mask, lookup_table=lookup_table) # self-attention
            x = self.pffn_layers[i](x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)

        x = batch_mask * original_x + (1 - batch_mask) * x

        return x, batch_mask

class DetailTransformer(nn.Module):
    def __init__(self, d_motion, config):
        super(DetailTransformer, self).__init__()
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
            nn.Linear(self.d_model, self.d_motion + 4),
        )
    
    def forward(self, x, batch_mask):
        B, T, D = x.shape

        original_x = x.clone()
        
        # mask
        x = self.encoder(torch.cat([x, batch_mask], dim=-1))

        # relative distance range: [-T+1, ..., T-1], 2T-1 values in total
        rel_dist = torch.arange(-T+1, T, dtype=torch.float32).unsqueeze(-1).to(x.device) # (2T-1, 1)
        lookup_table = self.relative_pos_encoder(rel_dist) # (2T-1, d_model)

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, x, mask=None, lookup_table=lookup_table)
            x = self.pffn_layers[i](x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)
        motion, contact = torch.split(x, [self.d_motion, 4], dim=-1)

        motion = batch_mask * original_x + (1 - batch_mask) * motion
        contact = torch.sigmoid(contact)

        return motion, contact