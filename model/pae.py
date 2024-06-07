import torch
import torch.nn as nn

from .multilinear import MultiLinear

"""
Periodie AutoEncoder - DeepPhase
"""
class PAE(nn.Module):
    def __init__(self, input_channels, phase_channels, num_frames, time_duration):
        super(PAE, self).__init__()
        self.input_channels = input_channels
        self.phase_channels = phase_channels
        self.num_frames = num_frames
        self.time_duration = time_duration

        # constant parameters
        self.two_pi = nn.Parameter(torch.tensor([2.0 * torch.pi], dtype=torch.float32), requires_grad=False)
        self.args = nn.Parameter(torch.linspace(-self.time_duration/2, self.time_duration/2, self.num_frames, dtype=torch.float32), requires_grad=False)
        self.freqs = nn.Parameter(torch.fft.rfftfreq(self.num_frames)[1:] * self.num_frames / self.time_duration, requires_grad=False) # Remove DC frequency

        # conv encoder
        interim_channels = input_channels // 3
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, interim_channels, self.num_frames, stride=1, padding=(self.num_frames - 1) // 2),
            nn.BatchNorm1d(interim_channels),
            nn.Tanh(),
            nn.Conv1d(interim_channels, phase_channels, self.num_frames, stride=1, padding=(self.num_frames - 1) // 2),
            nn.BatchNorm1d(phase_channels),
            nn.Tanh(),
        )

        # phase
        self.fc = MultiLinear(self.phase_channels, self.num_frames, 2)

        # conv decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(phase_channels, interim_channels, self.num_frames, stride=1, padding=(self.num_frames - 1) // 2),
            nn.BatchNorm1d(interim_channels),
            nn.Tanh(),
            nn.Conv1d(interim_channels, input_channels, self.num_frames, stride=1, padding=(self.num_frames - 1) // 2),
        )
    
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, :, 1:] # spectrum without DC component
        power = torch.square(spectrum) # square ** 2

        # frequency, amplitude, and offset
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.num_frames
        offset = rfft.real[:, :, 0] / self.num_frames # DC component

        return freq, amp, offset

    def forward(self, x):
        # batch_size, num_frames, input_channels = x.shape

        # signal embedding
        x = x.transpose(1, 2)
        L = self.encoder(x) # (batch_size, phase_channels, num_frames)

        # fft
        F, A, B = self.FFT(L, dim=2)

        # phase
        Sxy = self.fc(L.transpose(0, 1)).transpose(0, 1) # (batch_size, phase_channels, 2)
        S = torch.atan2(Sxy[..., 1], Sxy[..., 0]) / self.two_pi # (batch_size, phase_channels)

        # parameters (batch_size, phase_channels, 1)
        F = F.unsqueeze(-1)
        A = A.unsqueeze(-1)
        B = B.unsqueeze(-1)
        S = S.unsqueeze(-1)
        params = [F, A, B, S]

        # latent reconstruction (= inverse transform)
        Lhat = A * torch.sin(self.two_pi * (F * self.args + S)) + B

        # signal reconstruction
        y = self.decoder(Lhat)
        y = y.transpose(1, 2) # (batch_size, num_frames, input_channels)

        return y, L, Lhat, params