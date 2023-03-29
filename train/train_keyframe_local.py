import sys
sys.path.append(".")

import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import time
from tqdm import tqdm

from pymovis.utils import util, torchconst
from pymovis.ops import motionops, rotation

from utility.dataset import KeyframeDataset
from utility.config import Config
from model.ours import KeyframeTransformerLocal
from utility import trainutil

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/keyframe_local.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = KeyframeDataset(train=True, config=config)
    skeleton   = dataset.skeleton
    v_forward  = torch.from_numpy(config.v_forward).to(device)

    kf_mean, kf_std = dataset.statistics(dim=(0, 1))
    kf_mean, kf_std = kf_mean.to(device), kf_std.to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = KeyframeTransformerLocal(dataset.shape[-1], config).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.d_model**-0.5, betas=(0.9, 0.98), eps=1e-9)
    scheduler = trainutil.get_noam_scheduler(config, optim)
    init_epoch, iter = trainutil.load_latest_ckpt(model, optim, config, scheduler)
    init_iter = iter

    # save and log
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    config.write(os.path.join(config.save_dir, "config.json"))
    writer = SummaryWriter(config.log_dir)

    # training loop
    loss_dict = {
        "total":  0,
        "frame":  0,
        "pose":   0,
    }
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_keyframe in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            B, T, D = GT_keyframe.shape

            # GT
            GT_keyframe = GT_keyframe.to(device)
            GT_local_R6, GT_root_p, GT_kf_prob, GT_traj = torch.split(GT_keyframe, [D-9, 3, 1, 5], dim=-1)
            GT_local_R6 = GT_local_R6.reshape(B, T, -1, 6)
            _, GT_global_p = motionops.R6_fk(GT_local_R6, GT_root_p, skeleton)

            # forward
            batch = (GT_keyframe - kf_mean) / kf_std
            pred_motion, _ = model.forward(batch)
            pred_motion = pred_motion * kf_std[..., :-5] + kf_mean[..., :-5] # exclude traj features

            pred_local_R6, pred_root_p, pred_kf_prob = torch.split(pred_motion, [D-9, 3, 1], dim=-1)
            pred_kf_prob = torch.sigmoid(pred_kf_prob)
            pred_local_R6 = pred_local_R6.reshape(B, T, -1, 6)
            _, pred_global_p = motionops.R6_fk(pred_local_R6, pred_root_p, skeleton)

            # weight by keyframe probability
            GT_local_R6   = GT_local_R6.reshape(B, T, -1) * GT_kf_prob
            GT_global_p   = GT_global_p.reshape(B, T, -1) * GT_kf_prob
            pred_local_R6 = pred_local_R6.reshape(B, T, -1) * GT_kf_prob
            pred_global_p = pred_global_p.reshape(B, T, -1) * GT_kf_prob

            # loss
            loss_keytime = config.weight_keytime * F.l1_loss(pred_kf_prob[:, config.context_frames:-1], GT_kf_prob[:, config.context_frames:-1])
            loss_rot     = config.weight_keypose * F.l1_loss(pred_local_R6[:, config.context_frames:-1], GT_local_R6[:, config.context_frames:-1])
            loss_pos     = config.weight_keypose * F.l1_loss(pred_global_p[:, config.context_frames:-1], GT_global_p[:, config.context_frames:-1])
            loss = loss_keytime + loss_rot + loss_pos

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            # log
            loss_dict["total"]  += loss.item()
            loss_dict["frame"]  += loss_keytime.item()
            loss_dict["pose"]   += loss_rot.item() + loss_pos.item()

            if iter % config.log_interval == 0:
                tqdm.write(f"Iter {iter} | Loss: {loss_dict['total'] / config.log_interval:.4f} | Frame: {loss_dict['frame'] / config.log_interval:.4f} | Pose: {loss_dict['pose'] / config.log_interval:.4f} | Elapsed: {(time.perf_counter() - start_time) / 60:.2f} min")
                writer.add_scalar("loss/total",  loss_dict["total"]  / config.log_interval, iter)
                writer.add_scalar("loss/frame",  loss_dict["frame"]  / config.log_interval, iter)
                writer.add_scalar("loss/pose",   loss_dict["pose"]   / config.log_interval, iter)
                loss_dict = {
                    "total":  0,
                    "frame": 0,
                    "pose": 0,
                }
            
            if iter % config.save_interval == 0:
                trainutil.save_ckpt(model, optim, epoch, iter, config, scheduler)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(model, optim, epoch, iter, config, scheduler)