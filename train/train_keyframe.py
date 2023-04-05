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
from pymovis.ops import motionops, rotation, mathops

from utility.dataset import KeyframeDataset
from utility.config import Config
from model.ours import KeyframeTransformer
from utility import trainutil

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/keyframe.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = KeyframeDataset(train=True, config=config)
    skeleton   = dataset.skeleton
    v_forward  = torch.from_numpy(config.v_forward).to(device)

    kf_mean, kf_std = dataset.statistics(dim=(0, 1))
    kf_mean, kf_std = kf_mean.to(device), kf_std.to(device)
    
    feet_ids = []
    for name in config.contact_joint_names:
        feet_ids.append(skeleton.idx_by_name[name])

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = KeyframeTransformer(dataset.shape[-1], config).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    init_epoch, iter = trainutil.load_latest_ckpt(model, optim, config)
    init_iter = iter

    # save and log
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    config.write(os.path.join(config.save_dir, "config.json"))
    writer = SummaryWriter(config.log_dir)

    # training loop
    loss_dict = {
        "total":   0,
        "frame":   0,
        "rot":     0,
        "pos":     0,
        "traj":    0,
        "contact": 0,
    }
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_keyframe in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            B, T, D = GT_keyframe.shape

            # GT
            GT_keyframe = GT_keyframe.to(device)
            GT_local_R6, GT_root_p, GT_kf_score, GT_traj = torch.split(GT_keyframe, [D-7, 3, 1, 3], dim=-1)
            GT_local_R6 = GT_local_R6.reshape(B, T, -1, 6)
            _, GT_global_p = motionops.R6_fk(GT_local_R6, GT_root_p, skeleton)

            GT_feet_v = GT_global_p[:, 1:, feet_ids] - GT_global_p[:, :-1, feet_ids]
            GT_feet_v = torch.sum(GT_feet_v**2, dim=-1) # squared norm
            GT_feet_v = torch.cat([GT_feet_v[:, 0:1], GT_feet_v], dim=1)
            GT_contact = (GT_feet_v < config.contact_vel_threshold).float()

            # forward
            batch = (GT_keyframe - kf_mean) / kf_std
            pred_motion = model.forward(batch)
            pred_motion = pred_motion * kf_std[..., :-3] + kf_mean[..., :-3] # exclude traj features

            # predicted keyframe features
            pred_local_R6, pred_root_p, pred_kf_prob = torch.split(pred_motion, [D-7, 3, 1], dim=-1)
            pred_local_R6 = pred_local_R6.reshape(B, T, -1, 6)
            pred_kf_score = torch.clip(pred_kf_prob, 0, 1)
            _, pred_global_p = motionops.R6_fk(pred_local_R6, pred_root_p, skeleton)

            pred_feet_v = pred_global_p[:, 1:, feet_ids] - pred_global_p[:, :-1, feet_ids]
            pred_feet_v = torch.sum(pred_feet_v**2, dim=-1) # squared norm
            pred_feet_v = torch.cat([pred_feet_v[:, 0:1], pred_feet_v], dim=1)

            # predicted trajectory
            pred_traj_xz = pred_root_p[..., (0, 2)]
            pred_root_R = rotation.R6_to_R(pred_local_R6[:, :, 0])
            pred_traj_forward = F.normalize(torch.matmul(pred_root_R, v_forward) * torchconst.XZ(device), dim=-1)
            global_forward = torchconst.FORWARD(device).expand(B, T, -1)
            pred_signed_angle = mathops.signed_angle(global_forward, pred_traj_forward)
            pred_traj = torch.cat([pred_traj_xz, pred_signed_angle.unsqueeze(-1)], dim=-1)

            # weight by keyframe probability
            GT_local_R6   = GT_local_R6.reshape(B, T, -1)
            GT_global_p   = GT_global_p.reshape(B, T, -1)
            GT_traj       = GT_traj.reshape(B, T, -1)

            pred_local_R6 = pred_local_R6.reshape(B, T, -1)
            pred_global_p = pred_global_p.reshape(B, T, -1)
            pred_traj     = pred_traj.reshape(B, T, -1)

            # loss
            loss_frame   = config.weight_frame * torch.mean(torch.abs(pred_kf_score - GT_kf_score))
            loss_rot     = config.weight_rot   * torch.mean(torch.abs(pred_local_R6 - GT_local_R6) * GT_kf_score)
            loss_pos     = config.weight_pos   * torch.mean(torch.abs(pred_global_p - GT_global_p) * GT_kf_score)
            loss_traj    = config.weight_traj  * torch.mean(torch.abs(pred_traj - GT_traj) * GT_kf_score)
            loss_contact = config.weight_contact * torch.mean(pred_feet_v * GT_contact)
            loss = loss_frame + loss_rot + loss_pos + loss_traj + loss_contact

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # log
            loss_dict["total"]   += loss.item()
            loss_dict["frame"]   += loss_frame.item()
            loss_dict["rot"]     += loss_rot.item()
            loss_dict["pos"]     += loss_pos.item()
            loss_dict["traj"]    += loss_traj.item()
            loss_dict["contact"] += loss_contact.item()

            if iter % config.log_interval == 0:
                tqdm.write(f"Iter {iter} | Loss: {loss_dict['total'] / config.log_interval:.4f} | Frame: {loss_dict['frame'] / config.log_interval:.4f} | Rot: {loss_dict['rot'] / config.log_interval:.4f} | Pos: {loss_dict['pos'] / config.log_interval:.4f} | Traj: {loss_dict['traj'] / config.log_interval:.4f} | Contact: {loss_dict['contact'] / config.log_interval:.4f} | Time: {(time.perf_counter() - start_time) / 60:.2f} min")
                writer.add_scalar("loss/total",   loss_dict["total"]  / config.log_interval, iter)
                writer.add_scalar("loss/frame",   loss_dict["frame"]  / config.log_interval, iter)
                writer.add_scalar("loss/rot",     loss_dict["rot"]    / config.log_interval, iter)
                writer.add_scalar("loss/pos",     loss_dict["pos"]    / config.log_interval, iter)
                writer.add_scalar("loss/traj",    loss_dict["traj"]   / config.log_interval, iter)
                writer.add_scalar("loss/contact", loss_dict["contact"]/ config.log_interval, iter)
                
                for k in loss_dict.keys():
                    loss_dict[k] = 0
            
            if iter % config.save_interval == 0:
                trainutil.save_ckpt(model, optim, epoch, iter, config)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(model, optim, epoch, iter, config)