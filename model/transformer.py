import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymovis.learning.transformer import RelativeMultiHeadAttention, PoswiseFeedForwardNet

class SparseRelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, dropout=0.1, pre_layernorm=True):
        super(SparseRelativeMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.pre_layernorm = pre_layernorm

        self.W_q = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_out = nn.Linear(n_head * d_head, d_model)

        self.atten_scale = 1 / (d_head ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def skew(self, QE_t):
        B, H, T, X = QE_t.shape # (B, H, T, 2T-1)

        QE_t = F.pad(QE_t, (0, 1)).view(B, H, 2*T*T)
        QE_t = F.pad(QE_t, (0, T-1)).view(B, H, T+1, 2*T - 1)
        return QE_t[:, :, :T, -T:]

    def forward(self, Q, K, V, lookup_table, valid_frames, mask=None):
        K = K[:, valid_frames]
        V = V[:, valid_frames]

        B, T1, D = Q.shape
        _, T2, _ = K.shape

        if self.pre_layernorm:
            Q = self.layer_norm(Q)

        # linear projection to
        Q = self.W_q(Q) # (B, T1, n_head*d_head)
        K = self.W_k(K) # (B, T2, n_head*d_head)
        V = self.W_v(V) # (B, T2, n_head*d_head)

        # split heads
        Q = Q.view(B, T1, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T1, d_head)
        K = K.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)
        V = V.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)

        # attention score
        atten_score = torch.matmul(Q, K.transpose(-2, -1)) # (B, n_head, T1, T2)
        rel_atten_score = self.skew(torch.matmul(Q, lookup_table.transpose(-2, -1))) # (B, n_head, T1, T1)
        rel_atten_score = rel_atten_score[..., valid_frames]
        atten_score = (atten_score + rel_atten_score) * self.atten_scale # TODO: Fix this line for atten_score and rel_atten_score are not the same shape

        if mask is not None:
            atten_score.masked_fill_(mask, -torch.finfo(atten_score.dtype).max)

        # attention
        attention = F.softmax(atten_score, dim=-1)
        attention = torch.matmul(attention, V).transpose(1, 2).contiguous().view(B, -1, self.n_head * self.d_head) # (B, T1, n_head*d_head)

        # output
        out = self.W_out(attention) # (B, T1, d_model)
        out = self.dropout(out)
        
        if self.pre_layernorm:
            return out + attention
        else:
            return self.layer_norm(out + attention)
        
def get_mask(batch, context_frames, ratio_constrained=0.1, prob_constrained=0.5):
    B, T, D = batch.shape

    # 0 for unknown frames, 1 for known frames
    batch_mask = torch.ones_like(batch)
    batch_mask[:, context_frames:-1, :] = 0
    
    # mask out random partial frames
    constrained_frames = np.arange(context_frames, T-1)
    constrained_frames = np.random.choice(constrained_frames, int(len(constrained_frames) * ratio_constrained), replace=False)
    constrained_frames = np.sort(constrained_frames)
    for t in constrained_frames:
        if np.random.rand() < prob_constrained:
            batch_mask[:, t, :] = 1
            
    return batch_mask, constrained_frames


def get_keyframe_relative_position(window_length, context_frames):
    position = torch.arange(window_length, dtype=torch.float32)
    dist_ctx = position - (context_frames - 1) # distance to the last context frame
    dist_tgt = position - (window_length - 1)  # distance to the target frame

    p_kf = torch.stack([dist_ctx, dist_tgt], dim=-1) # (T, 2)

    return p_kf

class SparseTransformer(nn.Module):
    def __init__(self, d_motion, config):
        super(SparseTransformer, self).__init__()
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
            self.atten_layers.append(SparseRelativeMultiHeadAttention(self.d_model, self.d_head, self.n_heads, dropout=self.dropout, pre_layernorm=self.pre_layernorm))
            self.pffn_layers.append(PoswiseFeedForwardNet(self.d_model, self.d_ff, dropout=self.dropout, pre_layernorm=self.pre_layernorm))

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.PReLU(),
            nn.Linear(self.d_model, self.d_motion),
        )
    
    def forward(self, x, ratio_constrained=0.1, prob_constrained=0.3):
        B, T, D = x.shape
        
        # mask
        batch_mask, constrained_frames = get_mask(x, self.config.context_frames, ratio_constrained, prob_constrained)
        x = x * batch_mask
        x = self.encoder(torch.cat([x, batch_mask], dim=-1))

        # add keyframe positional embedding
        keyframe_pos = get_keyframe_relative_position(T, self.config.context_frames).to(x.device)
        x = x + self.keyframe_pos_encoder(keyframe_pos)

        # relative distance range: [-T+1, ..., -1, 0, 1, ..., T-1], 2T-1 values in total
        rel_dist = torch.arange(-T+1, T, dtype=torch.float32).to(x.device) # (2T-1)
        lookup_table = self.relative_pos_encoder(rel_dist.unsqueeze(-1)) # (2T-1, d_model)

        constrained_frames = np.concatenate([np.arange(self.config.context_frames), constrained_frames, np.arange(T-1, T)])

        # Transformer encoder layers
        for i in range(self.n_layers):
            x = x + self.atten_layers[i](x, x, x, lookup_table, constrained_frames, mask=None) # self-attention
            x = x + self.pffn_layers[i](x)
        
        # decoder
        if self.pre_layernorm:
            x = self.layer_norm(x)
        
        x = self.decoder(x)

        return x, batch_mask