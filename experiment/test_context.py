import sys
sys.path.append(".")

import os
import torch
from torch.utils.data import DataLoader

import copy
from tqdm import tqdm

from pymovis.utils import util
from pymovis.ops import rotation, motionops

from utility import testutil, benchmark
from utility.config import Config
from utility.dataset import MotionDataset
from model.twostage import ContextTransformer

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/context.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=config)
    skeleton   = dataset.skeleton

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    motion_mean, motion_std = motion_mean[..., :-5], motion_std[..., :-5] # exclude trajectory
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # model
    print("Initializing model...")
    model = ContextTransformer(dataset.shape[-1] - 5, config).to(device) # exclude trajectory
    testutil.load_model(model, config)
    model.eval()

    # training loop
    l2p, l2q, npss = [], [], []
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            T = config.context_frames + config.max_transition + 1
            GT_motion = GT_motion[:, :T, :-5]
            B, T, D = GT_motion.shape

            # GT motion
            GT_motion = GT_motion.to(device)
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))
            GT_global_R, GT_global_p = motionops.R_fk(GT_local_R, GT_root_p, skeleton)
            GT_global_Q = rotation.R_to_Q(GT_global_R)

            # ContextTransformer
            batch = (GT_motion - motion_mean) / motion_std
            pred_motion, mask = model.forward(batch, ratio_constrained=0, prob_constrained=0)
            pred_motion = mask * batch + (1 - mask) * pred_motion
            pred_motion = pred_motion * motion_std + motion_mean

            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-3, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))
            pred_global_R, pred_global_p = motionops.R_fk(pred_local_R, pred_root_p, skeleton)
            pred_global_Q = rotation.R_to_Q(pred_global_R)

            # benchmark
            L2P = benchmark.L2P(pred_global_p[:, config.context_frames:], GT_global_p[:, config.context_frames:])
            L2Q = benchmark.L2Q(pred_global_Q[:, config.context_frames:], GT_global_Q[:, config.context_frames:])

            NPSS_pred = pred_global_Q.reshape(B, T, -1)
            NPSS_GT   = GT_global_Q.reshape(B, T, -1)
            NPSS      = benchmark.NPSS(NPSS_pred[:, config.context_frames:], NPSS_GT[:, config.context_frames:])
            
            l2p.append(L2P)
            l2q.append(L2Q)
            npss.append(NPSS)

    l2p  = torch.cat(l2p, dim=0)
    l2q  = torch.cat(l2q, dim=0)
    npss = torch.stack(npss, dim=0)

    print("=====================================")
    print("| L2P:  {:.6f}".format(l2p.mean()))
    print("| L2Q:  {:.6f}".format(l2q.mean()))
    print("| NPSS: {:.6f}".format(npss.mean()))
    print("=====================================")