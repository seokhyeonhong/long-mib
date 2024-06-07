import torch
import torch.nn as nn
import torch.nn.functional as F

def rot_loss(pred, gt, context_frames):
    loss = F.l1_loss(pred[:, context_frames:-1], gt[:, context_frames:-1])
    return loss

def pos_loss(pred, gt, context_frames):
    loss = F.l1_loss(pred[:, context_frames:-1], gt[:, context_frames:-1])
    return loss

def smooth_loss(pred, context_frames):
    loss = F.l1_loss(pred[:, context_frames:] - pred[:, context_frames-1:-1],
                     torch.zeros_like(pred[:, context_frames:]))
    return loss

def contact_loss(pred, gt, context_frames):
    loss = F.l1_loss(pred[:, context_frames:-1], gt[:, context_frames:-1])
    return loss

def foot_loss(contact, vel, context_frames):
    loss = F.l1_loss(contact[:, context_frames:-1].detach() * vel[:, context_frames:-1],
                     torch.zeros_like(vel[:, context_frames:-1]))
    return loss

def phase_loss(pred_phase, gt_phase, context_frames):
    loss_phase = F.l1_loss(pred_phase[:, context_frames:-1], gt_phase[:, context_frames:-1])
    return loss_phase

def traj_loss(pred, gt, context_frames):
    pred_pos, pred_dir = torch.split(pred, [2, 2], dim=-1)
    gt_pos, gt_dir = torch.split(gt, [2, 2], dim=-1)
    loss_pos = F.l1_loss(pred_pos[:, context_frames:-1], gt_pos[:, context_frames:-1])
    loss_dir = F.l1_loss(1 - torch.sum(pred_dir[:, context_frames:-1] * gt_dir[:, context_frames:-1], dim=-1),
                            torch.zeros_like(pred_dir[:, context_frames:-1, 0]))
    return loss_pos + loss_dir

def score_loss(pred, gt, context_frames):
    loss = F.l1_loss(pred[:, context_frames:-1], gt[:, context_frames:-1])
    return loss


"""
Loss functions for RMI
"""
def disc_loss(real_score, fake_score):
    real = torch.mean((real_score - 1) ** 2)
    fake = torch.mean(fake_score ** 2)
    return 0.5 * (real + fake)

def gen_loss(fake_score):
    loss = torch.mean((fake_score - 1) ** 2)
    return 0.5 * loss