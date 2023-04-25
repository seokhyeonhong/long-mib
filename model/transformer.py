import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi Head Contextual Periodic Biased Attention
class MultiHeadContextualBiasedAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, dropout=0.1, pre_layernorm=False):
        super(MultiHeadContextualBiasedAttention, self).__init__()
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
    
    def forward(self, x, context, bias, mask=None):
        B, T1, D = x.shape
        _, T2, _ = context.shape

        if self.pre_layernorm:
            x = self.layer_norm(x)

        # linear projection
        q = self.W_q(x) # (B, T1, n_head*d_head)
        k = self.W_k(context) # (B, T2, n_head*d_head)
        v = self.W_v(context) # (B, T2, n_head*d_head)

        # split heads
        q = q.view(B, T1, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T1, d_head)
        k = k.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)
        v = v.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)

        # attention score
        atten_score = torch.matmul(q, k.transpose(-2, -1)) # (B, n_head, T1, T2)
        atten_score = (atten_score + bias) * self.atten_scale # (B, n_head, T1, T2)

        # mask
        if mask is not None:
            atten_score.masked_fill_(mask, -1e9)
        
        # attention
        attention = F.softmax(atten_score, dim=-1) # (B, n_head, T1, T2)
        if mask is not None:
            attention_copy = attention.clone()
            attention_copy.masked_fill_(mask, 0)
            attention = attention_copy
        attention = torch.matmul(attention, v).transpose(1, 2).contiguous().view(B, -1, self.n_head * self.d_head) # (B, T1, n_head*d_head)

        # output
        output = self.W_out(attention) # (B, T1, d_model)
        output = self.dropout(output)

        return output
    
class ModifiedPoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(ModifiedPoswiseFeedForwardNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.layers(x)