import torch
import torch.nn as nn
import torch.nn.functional as F


class PLU(nn.Module):
    def __init__(self, alpha=0.1, c=1.0):
        super(PLU, self).__init__()
        self.alpha = alpha
        self.c = c
    
    def forward(self, x):
        out = torch.max(self.alpha * (x + self.c) - self.c, torch.min(self.alpha * (x - self.c) + self.c, x))
        return out


class TimeToArrivalEmbedding(nn.Module):
    def __init__(self, dim, context_len, max_trans=30, base=10000):
        super(TimeToArrivalEmbedding, self).__init__()
        self.dim = dim
        self.base = base
        self.context_len = context_len
        self.max_trans = max_trans

        inv_freq = 1 / (base ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tta):
        """
        Args:
            tta (int): number of frames away from target frame
        Returns:
            tensor: time-to-arrival embedding, shape: (dim,)
        """
        tta = min(tta, self.context_len + self.max_trans - 5)

        inv_term = tta * self.inv_freq

        # interleave sin and cos values
        pe = torch.stack([inv_term.sin(), inv_term.cos()], dim=-1)
        pe = pe.flatten(start_dim=-2)

        # ":self.dim" on last dimesion makes sure the dimension of
        # positional encoding is as required when self.dim is
        # an odd number.
        return pe[..., :self.dim]


class RmiGenerator(nn.Module):
    def __init__(self, config, num_joints):
        super(RmiGenerator, self).__init__()
        self.config = config
        self.num_joints = num_joints

        # encoders
        self.state_enc = nn.Sequential(
            nn.Linear(num_joints * 6 + 7, config.d_enc_hidden), # 6J for local rotations, 3 for root velocity, 4 for contact
            PLU(),
            nn.Linear(config.d_enc_hidden, config.d_enc_output),
            PLU(),
        )
        self.offset_enc = nn.Sequential(
            nn.Linear(num_joints * 6 + 3, config.d_enc_hidden), # 6J for local rotation offsets, 3 for root position offsets
            PLU(),
            nn.Linear(config.d_enc_hidden, config.d_enc_output),
            PLU(),
        )
        self.target_enc = nn.Sequential(
            nn.Linear(num_joints * 6, config.d_enc_hidden), # 6J for target rotations
            PLU(),
            nn.Linear(config.d_enc_hidden, config.d_enc_output),
            PLU(),
        )

        # LSTM
        self.lstm = nn.LSTM(config.d_enc_output * 3, config.d_lstm_hidden, config.d_lstm_layers)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(config.d_lstm_hidden, config.d_dec_hidden1),
            PLU(),
            nn.Linear(config.d_dec_hidden1, config.d_dec_hidden2),
            PLU(),
            nn.Linear(config.d_dec_hidden2, num_joints * 6 + 7), # 6J for local rotations, 3 for root velocity, 4 for contact
        )

        # embedding
        self.tta_emb = TimeToArrivalEmbedding(config.d_enc_output, config.context_frames)
    
    def init_hidden(self, batch_size, device="cuda"):
        self.h = torch.zeros(self.config.d_lstm_layers, batch_size, self.config.d_lstm_hidden).to(device)
        self.c = torch.zeros(self.config.d_lstm_layers, batch_size, self.config.d_lstm_hidden).to(device)
    
    def forward(self, local_rots, root_pos, root_vel, contact, target_rot, target_root_pos, tta):
        # input
        state = torch.cat([local_rots, root_vel, contact], dim=-1)
        offset = torch.cat([target_root_pos - root_pos, target_rot - local_rots], dim=-1)

        # embedding
        h_state = self.state_enc(state)
        h_offset = self.offset_enc(offset)
        h_target = self.target_enc(target_rot)

        # time-to-arrival embedding
        z_tta = self.tta_emb(tta)
        h_state += z_tta
        h_offset += z_tta
        h_target += z_tta

        ot = torch.cat([h_offset, h_target], dim=-1)
        
        # target embedding
        if tta >= 30:
            stdev = 1
        elif tta >= 5:
            stdev = (tta - 5) / 25
        else:
            stdev = 0

        z_target = torch.normal(torch.zeros_like(ot), torch.ones_like(ot) * self.config.target_std) * stdev
        lstm_in = torch.cat([h_state, ot + z_target], dim=-1)
        lstm_in = lstm_in.unsqueeze(0) # (1, B, D)

        # LSTM
        lstm_out, (self.h, self.c) = self.lstm(lstm_in, (self.h, self.c))

        # decoder
        out = self.dec(lstm_out.squeeze(0))

        # output
        lr, rv, c = torch.split(out, [self.num_joints * 6, 3, 4], dim=-1)
        lr = lr + local_rots
        rp = root_pos + rv
        c = F.sigmoid(c)

        return lr, rp, c


class RmiDiscriminator(nn.Module):
    def __init__(self, config, num_joints, window_size):
        super(RmiDiscriminator, self).__init__()
        self.config = config

        self.layers = nn.Sequential(
            nn.Conv1d(6 * num_joints + 3, config.d_disc_hidden1, kernel_size=window_size, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(config.d_disc_hidden1, config.d_disc_hidden2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(config.d_disc_hidden2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # prediction
        # 0: real data, 1: fake data
        return self.layers(x.transpose(1, 2)).mean(dim=-1)