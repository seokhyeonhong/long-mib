import torch
import torch.nn as nn

class PeriodicPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, period=30):
        super(PeriodicPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.period        = period
        
        if embedding_dim % 2 != 0:
            raise ValueError(f"PeriodicPositionalEmbedding: embedding_dim must be even, but got {embedding_dim}")

        pos = torch.arange(0, period, step=1, dtype=torch.float32).unsqueeze(1)
        div_term = 1.0 / torch.pow(10000, torch.arange(0, embedding_dim, step=2, dtype=torch.float32) / embedding_dim)

        embedding = torch.empty((period, embedding_dim))
        embedding[:, 0::2] = torch.sin(pos * div_term)
        embedding[:, 1::2] = torch.cos(pos * div_term)
        self.embedding = nn.Parameter(embedding, requires_grad=False)
        
    def forward(self, position):
        # position: (B, T)
        position = torch.fmod(torch.fmod(position, self.period) + self.period, self.period).long() # positive modulo
        return self.embedding[position] # (B, T, embedding_dim)