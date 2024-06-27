import torch
import torch.nn as nn

from model.multilinear import MultiLinear

class Encoder(nn.Module):
    def __init__(self, d_motion, d_hidden, d_latent):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_motion * 2, d_hidden),
            nn.ELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ELU(),
            nn.Linear(d_hidden, d_latent * 2)
        )

    def forward(self, x_curr, x_next):
        x = torch.cat([x_curr, x_next], dim=-1)
        h = self.layers(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar
    

class Decoder(nn.Module):
    def __init__(self, d_motion, d_latent, d_hidden, d_phase, num_experts):
        super(Decoder, self).__init__()
        self.gating = Gating(d_latent, d_hidden, d_phase, num_experts)
        self.decoder_layers = nn.ModuleList([
            MultiLinear(num_experts, d_motion + 3 + d_latent, d_hidden), # 3 for hip velocity
            MultiLinear(num_experts, d_hidden + d_latent, d_hidden),
            MultiLinear(num_experts, d_hidden + d_latent, d_hidden),
        ])
        self.out = MultiLinear(num_experts, d_hidden + d_latent, d_motion)
        self.act = nn.ELU()

        self.num_experts = num_experts

    def forward(self, z, p_next, v_hip_next, x_curr):
        expert_weights = self.gating(z, p_next) # (B, num_experts)

        x = torch.cat([v_hip_next, x_curr], dim=-1)
        for layer in self.decoder_layers:
            x = torch.cat([z, x], dim=-1)

            B, T, D = x.shape
            x = layer(x.reshape(1, B*T, D)) # (num_experts, B*T, D)
            x = x.transpose(0, 1).reshape(B, T, self.num_experts, -1)
            x = torch.sum(x * expert_weights.unsqueeze(-1), dim=-2)
            x = self.act(x)

        # last layer
        x = torch.cat([z, x], dim=-1)
        B, T, D = x.shape
        x = self.out(x.reshape(1, B*T, D)) # (num_experts, B, D)
        x = x.transpose(0, 1).reshape(B, T, self.num_experts, -1)
        x = torch.sum(x * expert_weights.unsqueeze(-1), dim=-2)

        return x


class Gating(nn.Module):
    def __init__(self, d_latent, d_hidden, d_phase, num_experts):
        super(Gating, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_latent + d_phase, d_hidden),
            nn.Linear(d_latent + d_hidden, d_hidden),
            nn.Linear(d_latent + d_hidden, num_experts),
        ])
        self.act = nn.ModuleList([
            nn.ELU(),
            nn.ELU(),
            nn.Softmax(dim=-1),
        ])

    def forward(self, z, p_next):
        h = p_next
        for i in range(len(self.layers)):
            h = torch.cat([z, h], dim=-1)
            h = self.layers[i](h)
            h = self.act[i](h)
        return h


class MotionVAE(nn.Module):
    def __init__(self, config, d_motion, d_phase):
        super(MotionVAE, self).__init__()
        self.encoder = Encoder(d_motion, config.d_hidden, config.d_latent)
        self.decoder = Decoder(d_motion, config.d_latent, config.d_hidden, d_phase, config.num_expert)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_curr, x_next, p_next, v_hip_next):
        mu, logvar = self.encoder.forward(x_curr, x_next)
        z = self.reparametrize(mu, logvar)
        x = self.decoder.forward(z, p_next, v_hip_next, x_curr)
        return x, mu, logvar
    
    def decode(self, z, p_next, v_hip_next, x_curr):
        return self.decoder.forward(z, p_next, v_hip_next, x_curr)