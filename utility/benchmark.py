import os
import torch

def L2P(norm_pred_global_p, norm_GT_global_p):
    # norm_pred_global_p: (B, T, J, 3)
    # norm_GT_global_p: (B, T, J, 3)
    B, T, J, _ = norm_pred_global_p.shape
    norm_pred_global_p = norm_pred_global_p.reshape(B, T, J*3)
    norm_GT_global_p = norm_GT_global_p.reshape(B, T, J*3)
    
    # L2P
    L2P = torch.norm(norm_pred_global_p - norm_GT_global_p, dim=-1)
    return L2P

def L2Q(norm_pred_global_Q, norm_GT_global_Q):
    # norm_pred_global_Q: (B, T, J, 4)
    # norm_GT_global_Q: (B, T, J, 4)
    B, T, J, _ = norm_pred_global_Q.shape
    w_positive = (norm_pred_global_Q[..., 0:1] > 0).float()
    norm_pred_global_Q = norm_pred_global_Q * w_positive + (1 - w_positive) * (-norm_pred_global_Q)
    
    w_positive = (norm_GT_global_Q[..., 0:1] > 0).float()
    norm_GT_global_Q = norm_GT_global_Q * w_positive + (1 - w_positive) * (-norm_GT_global_Q)

    norm_pred_global_Q = norm_pred_global_Q.reshape(B, T, J*4)
    norm_GT_global_Q = norm_GT_global_Q.reshape(B, T, J*4)
    
    # L2Q
    L2Q = torch.norm(norm_pred_global_Q - norm_GT_global_Q, dim=-1)
    return L2Q

def NPSS(pred, GT):
    # GT: (B, T, D)
    # pred: (B, T, D)

    # Fourier coefficients along the time dimension
    GT_fourier_coeffs = torch.real(torch.fft.fft(GT, dim=1))
    pred_fourier_coeffs = torch.real(torch.fft.fft(pred, dim=1))

    # square of the Fourier coefficients
    GT_power = torch.square(GT_fourier_coeffs)
    pred_power = torch.square(pred_fourier_coeffs)

    # sum of powers over time
    GT_power_sum = torch.sum(GT_power, dim=1)
    pred_power_sum = torch.sum(pred_power, dim=1)

    # normalize powers with total
    GT_power_norm = GT_power / GT_power_sum.unsqueeze(1)
    pred_power_norm = pred_power / pred_power_sum.unsqueeze(1)

    # cumulative sum over time
    GT_power_cumsum = torch.cumsum(GT_power_norm, dim=1)
    pred_power_cumsum = torch.cumsum(pred_power_norm, dim=1)

    # earth mover distance
    emd = torch.norm((pred_power_cumsum - GT_power_cumsum), p=1, dim=1)

    # weighted EMD
    power_weighted_emd = torch.sum(emd * GT_power_sum) / torch.sum(GT_power_sum)

    return power_weighted_emd