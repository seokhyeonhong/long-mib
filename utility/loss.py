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

def traj_loss(pred, gt):
    # pred: (B, T, 5)
    # gt: (B, T, 5)
    loss_xz = F.l1_loss(pred[..., :2], gt[..., :2])
    loss_fwd = F.l1_loss(1 - torch.sum(pred[..., 2:] * gt[..., 2:], dim=-1), torch.zeros_like(pred[..., 0]))
    return loss_xz + loss_fwd

def smooth_loss(pred):
    # pred: (B, T, D)
    # smoothness loss
    loss = F.l1_loss(pred[:, 1:] - pred[:, :-1], torch.zeros_like(pred[:, 1:]))
    return loss

def foot_loss(vel_foot, contact):
    return F.l1_loss(vel_foot * contact, torch.zeros_like(vel_foot))

def discriminator_loss(real, fake):
    # real: (B, T)
    # fake: (B, T)
    # discriminator loss
    loss_real = -torch.mean(torch.log(real + 1e-8))
    loss_fake = -torch.mean(torch.log(1 - fake + 1e-8))
    return loss_real + loss_fake

def generator_loss(fake):
    # fake: (B, T)
    # generator loss
    loss = -torch.mean(torch.log(fake + 1e-8))
    return loss