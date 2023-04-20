import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi Head Contextual Periodic Biased Attention
class MultiHeadContextualBiasedAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, period, dropout=0.1):
        super(MultiHeadContextualBiasedAttention, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.period = period

        self.W_q = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_head * d_head, bias=False)
        self.W_out = nn.Linear(n_head * d_head, d_model)

        self.atten_scale = 1 / (d_head ** 0.5)
        self.dropout = nn.Dropout(dropout)

        self.head_specific_sclae = nn.Parameter(torch.pow(2, -torch.arange(1, n_head + 1, dtype=torch.float32)), requires_grad=False)
    
    def forward(self, x, context, bias):
        B, T1, D = x.shape
        _, T2, _ = context.shape

        # linear projection
        q = self.W_q(x) # (B, T1, n_head*d_head)
        k = self.W_k(context) # (B, T2, n_head*d_head)
        v = self.W_v(context) # (B, T2, n_head*d_head)

        # split heads
        q = q.view(B, T1, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T1, d_head)
        k = k.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)
        v = v.view(B, T2, self.n_head, self.d_head).transpose(1, 2) # (B, n_head, T2, d_head)

        # attention score
        atten_score = torch.matmul(q, k.transpose(-2, -1)) * self.atten_scale # (B, n_head, T1, T2)

        # attention
        bias = bias.view(1, T1, T2) * self.head_specific_sclae.view(self.n_head, 1, 1) # (n_head, T1, T2)
        attention = F.softmax(atten_score + bias, dim=-1) # (B, n_head, T1, T2)
        attention = torch.matmul(attention, v).transpose(1, 2).contiguous().view(B, -1, self.n_head * self.d_head) # (B, T1, n_head*d_head)

        # output
        output = self.W_out(attention) # (B, T1, d_model)
        output = self.dropout(output)
        
        return output

class ModifiedPoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(ModifiedPoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        output = self.fc1(x) # (B, T, d_ff)
        output = F.relu(output) # (B, T, d_ff)
        output = self.dropout(output)
        output = self.fc2(output) # (B, T, d_model)
        output = self.dropout(output)
        return output