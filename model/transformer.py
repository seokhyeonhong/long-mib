import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1, pre_lnorm=True):
        """
        Positionwise feed-forward network.

        Args:
            d_model(int): Dimension of the input and output.
            d_inner (int): Dimension of the middle layer(bottleneck).
            dropout (float, optional): Dropout value. Defaults to 0.1.
            pre_lnorm (bool, optional):
                Apply layer norm before rest of calculation. Defaults to True.
                In original Transformer paper (pre_lnorm=False):
                    LayerNorm(x + Sublayer(x))
                In tensor2tensor implementation (pre_lnorm=True):
                    x + Sublayer(LayerNorm(x))
        """
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm

        self.layer_norm = nn.LayerNorm(d_model)
        self.network = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        if self.pre_lnorm:
            return x + self.network(self.layer_norm(x))
        else:
            return self.layer_norm(x + self.network(x))


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout=0.1,
                 pre_lnorm=True, bias=False):
        """
        Multi-headed attention of vanilla transformer with memory mechanism.

        Args:
            n_head (int): Number of heads.
            d_model (int): Input dimension.
            d_head (int): Head dimension.
            dropout (float, optional): Dropout value. Defaults to 0.1.
            pre_lnorm (bool, optional):
                Apply layer norm before rest of calculation. Defaults to True.
                In original Transformer paper (pre_lnorm=False):
                    LayerNorm(x + Sublayer(x))
                In tensor2tensor implementation (pre_lnorm=True):
                    x + Sublayer(LayerNorm(x))
            bias (bool, optional):
                Add bias to q, k, v and output projections. Defaults to False.

        """
        super(MultiHeadedAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm
        self.bias = bias
        self.atten_scale = 1 / math.sqrt(self.d_model)

        self.q_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.k_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.v_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.out_linear = nn.Linear(n_head * d_head, d_model, bias=bias)

        self.droput_layer = nn.Dropout(dropout)
        self.atten_dropout_layer = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, hidden, memory=None, mask=None):
        """
        Args:
            hidden (Tensor): Input embedding or hidden state of previous layer.
                Shape: (batch, seq, dim)
            memory (Tensor): Memory tensor of previous layer.
                Shape: (batch, mem_len, dim)
            mask (BoolTensor, optional): Attention mask.
                Set item value to True if you DO NOT want keep certain
                attention score, otherwise False. Defaults to None.
                Shape: (seq, seq+mem_len).
        """
        if memory is None:
            combined = hidden
        else:
            combined = torch.cat([memory, hidden], dim=1)

        if self.pre_lnorm:
            hidden = self.layer_norm(hidden)
            combined = self.layer_norm(combined)

        # shape: (batch, q/k/v_len, dim)
        q = self.q_linear(hidden)
        k = self.k_linear(combined)
        v = self.v_linear(combined)

        # reshape to (batch, q/k/v_len, n_head, d_head)
        q = q.reshape(q.shape[0], q.shape[1], self.n_head, self.d_head)
        k = k.reshape(k.shape[0], k.shape[1], self.n_head, self.d_head)
        v = v.reshape(v.shape[0], v.shape[1], self.n_head, self.d_head)

        # transpose to (batch, n_head, q/k/v_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # (batch, n_head, q_len, k_len)
        atten_score = torch.matmul(q, k.transpose(-1, -2)) * self.atten_scale
        if mask is not None:
            # apply attention mask
            atten_score = atten_score.masked_fill(mask, float("-inf"))
        atten_score = atten_score.softmax(dim=-1)
        atten_score = self.atten_dropout_layer(atten_score)

        # (batch, n_head, q_len, d_head)
        atten_vec = torch.matmul(atten_score, v)
        # (batch, q_len, n_head*d_head)
        atten_vec = atten_vec.transpose(1, 2).flatten(start_dim=-2)

        # linear projection
        output = self.droput_layer(self.out_linear(atten_vec))

        if self.pre_lnorm:
            return hidden + output
        else:
            return self.layer_norm(hidden + output)


class RelMultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout=0.1,
                 pre_lnorm=True, bias=False):
        """
        Multi-headed attention with relative positional encoding and
        memory mechanism.

        Args:
            n_head (int): Number of heads.
            d_model (int): Input dimension.
            d_head (int): Head dimension.
            dropout (float, optional): Dropout value. Defaults to 0.1.
            pre_lnorm (bool, optional):
                Apply layer norm before rest of calculation. Defaults to True.
                In original Transformer paper (pre_lnorm=False):
                    LayerNorm(x + Sublayer(x))
                In tensor2tensor implementation (pre_lnorm=True):
                    x + Sublayer(LayerNorm(x))
            bias (bool, optional):
                Add bias to q, k, v and output projections. Defaults to False.

        """
        super(RelMultiHeadedAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm
        self.bias = bias
        self.atten_scale = 1 / math.sqrt(self.d_model)

        self.q_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.k_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.v_linear = nn.Linear(d_model, n_head * d_head, bias=bias)
        self.out_linear = nn.Linear(n_head * d_head, d_model, bias=bias)

        self.droput_layer = nn.Dropout(dropout)
        self.atten_dropout_layer = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

    def skew(self, qpos, mem_len):
        padding_shape = list(qpos.shape)
        padding_shape[-1] = 1
        padding = torch.full(padding_shape, float("-inf"), device=qpos.device)

        relpos = torch.cat([qpos, padding], dim=-1)
        relpos = relpos.flatten(start_dim=-2)
        relpos = relpos[..., :qpos.shape[-1] * qpos.shape[-2]]
        relpos = relpos.reshape(*qpos.shape)

        q_len = qpos.shape[-2]
        k_len = int((qpos.shape[-1] + 1) / 2)

        assert q_len + mem_len == k_len

        zero_pos_idx = k_len - 1
        start_idx = zero_pos_idx - mem_len
        end_idx = start_idx + k_len

        relpos = relpos[..., start_idx:end_idx]

        return relpos

    def forward(self, hidden, pos_emb, memory=None, mask=None,
                extra_atten_score=None):
        """
        Args:
            hidden (Tensor): Input embedding or hidden state of previous layer.
                Shape: (batch, seq, dim)
            pos_emb (Tensor): Relative positional embedding lookup table.
                Shape: (batch, (seq+mem_len)*2-1, d_head)
                pos_emb[:, seq+mem_len]

            memory (Tensor): Memory tensor of previous layer.
                Shape: (batch, mem_len, dim)
            mask (BoolTensor, optional): Attention mask.
                Set item value to True if you DO NOT want keep certain
                attention score, otherwise False. Defaults to None.
                Shape: (seq, seq+mem_len).
        """
        if memory is None:
            combined = hidden
            mem_len = 0
        else:
            combined = torch.cat([memory, hidden], dim=1)
            mem_len = memory.shape[1]

        if self.pre_lnorm:
            hidden = self.layer_norm(hidden)
            combined = self.layer_norm(combined)

        # shape: (batch, q/k/v_len, dim)
        q = self.q_linear(hidden)
        k = self.k_linear(combined)
        v = self.v_linear(combined)

        # reshape to (batch, q/k/v_len, n_head, d_head)
        q = q.reshape(q.shape[0], q.shape[1], self.n_head, self.d_head)
        k = k.reshape(k.shape[0], k.shape[1], self.n_head, self.d_head)
        v = v.reshape(v.shape[0], v.shape[1], self.n_head, self.d_head)

        # transpose to (batch, n_head, q/k/v_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # add n_head dimension for relative positional embedding lookup table
        # (batch, n_head, k/v_len*2-1, d_head)
        pos_emb = pos_emb[:, None]

        # (batch, n_head, q_len, k_len)
        atten_score = torch.matmul(q, k.transpose(-1, -2))

        qpos = torch.matmul(q, pos_emb.transpose(-1, -2))
        # DEBUG
        # ones = torch.zeros(q.shape)
        # ones[:, :, :, 0] = 1.0
        # qpos = torch.matmul(ones, pos_emb.transpose(-1, -2))
        atten_score = atten_score + self.skew(qpos, mem_len)
        atten_score = atten_score * self.atten_scale

        if extra_atten_score is not None:
            atten_score = atten_score + extra_atten_score

        if mask is not None:
            # apply attention mask
            atten_score = atten_score.masked_fill(mask, float("-inf"))
        atten_score = atten_score.softmax(dim=-1)
        atten_score = self.atten_dropout_layer(atten_score)

        # (batch, n_head, q_len, d_head)
        atten_vec = torch.matmul(atten_score, v)
        # (batch, q_len, n_head*d_head)
        atten_vec = atten_vec.transpose(1, 2).flatten(start_dim=-2)

        # linear projection
        output = self.droput_layer(self.out_linear(atten_vec))

        if self.pre_lnorm:
            return hidden + output
        else:
            return self.layer_norm(hidden + output)

# class ConvBlockAttention(nn.Module):
#     def __init__(self, d_model, kernel_size, dropout=0.1, pre_lnorm=True):
#         super(ConvBlockAttention, self).__init__()
#         self.d_model = d_model
#         # self.d_head = d_head >> later use for multi-head
#         self.dropout = dropout
#         self.pre_lnorm = pre_lnorm

#         # conv block (B, T, D) -> (B, T, D)
#         self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        
#         # feature attention (B, T, D) -> (B, 1, D)
#         self.feat_linear = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, d_model),
#         )

#         # temporal attention (B, T, D) -> (B, T, 1)
#         self.temp_conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)

#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model)

#     def forward(self, x, *args, **kwargs):
#         # x.shape: (B, T, D)
#         if self.pre_lnorm:
#             x_in = self.layer_norm(x)
#         else:
#             x_in = x

#         # conv block (B, T, D) -> (B, T, D)
#         x = self.conv(x.transpose(1, 2)).transpose(1, 2)

#         # feature attention
#         feat_max = self.feat_linear(torch.max(x, dim=1, keepdim=True).values) # (B, 1, D)
#         feat_avg = self.feat_linear(torch.mean(x, dim=1, keepdim=True))
#         feat_attn = F.sigmoid(feat_max + feat_avg) # (B, 1, D)
#         x = x * feat_attn

#         # temporal attention
#         temp_max = torch.max(x, dim=2, keepdim=True).values # (B, T, 1)
#         temp_avg = torch.mean(x, dim=2, keepdim=True)
#         temp_out = torch.cat([temp_max, temp_avg], dim=2) # (B, T, 2)
#         temp_attn = F.sigmoid(self.temp_conv(temp_out.transpose(1, 2))).transpose(1, 2) # (B, T, 1)
#         x = x * temp_attn

#         # residual
#         x = x_in + self.dropout(x)

#         if self.pre_lnorm:
#             return x
#         else:
#             return self.layer_norm(x)