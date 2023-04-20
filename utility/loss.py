import os
import torch
import torch.nn.functional as F

def kl_loss(mean, logvar):
    # mean: (B, D)
    # logvar: (B, D)
    loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    return torch.mean(loss)

def recon_loss(pred, gt):
    return F.l1_loss(pred, gt)

def smooth_loss(pred):
    # pred: (B, T, D)
    # smoothness loss
    loss = F.l1_loss(pred[:, 1:] - pred[:, :-1], torch.zeros_like(pred[:, 1:]))
    return loss