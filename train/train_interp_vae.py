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

from utility.dataset import MotionDataset
from utility.config import Config
from model.ours import RefineVAE
from utility import trainutil

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/refine_vae.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=True, config=config)
    skeleton   = dataset.skeleton
    v_forward  = torch.from_numpy(skeleton.v_forward).to(device)

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    
    feet_ids = []
    for name in config.contact_joint_names:
        feet_ids.append(skeleton.idx_by_name[name])

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = RefineVAE(dataset.shape[-1], config).to(device)
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
        "rot":    0,
        "pos":    0,
        "vel":    0,
        "traj":   0,
        "kl":0,
    }
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_motion in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            # GT
            T = config.context_frames + config.max_transition + 1
            GT_motion = GT_motion[:, :T].to(device)
            B, T, D = GT_motion.shape

            GT_motion, GT_traj = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-6, 3], dim=-1)
            GT_local_R6 = GT_local_R6.reshape(B, T, -1, 6)
            GT_local_R  = rotation.R6_to_R(GT_local_R6)
            _, GT_global_p = motionops.R_fk(GT_local_R, GT_root_p, skeleton)

            GT_motion = torch.cat([GT_motion, GT_traj], dim=-1)

            # keyframes and interpolated motion
            keyframes = model.get_random_keyframes(T)
            interp_motion = model.get_interpolated_motion(GT_local_R, GT_root_p, keyframes)
            interp_motion = torch.cat([interp_motion, GT_traj], dim=-1)

            # forward
            GT_batch = (GT_motion - motion_mean) / motion_std
            interp_batch = (interp_motion - motion_mean) / motion_std
            pred_motion, pred_mu, pred_sigma = model.forward(GT_batch, interp_batch, keyframes)
            pred_motion = pred_motion * motion_std[:-3] + motion_mean[:-3] # exclude trajectory

            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-6, 3], dim=-1)
            pred_local_R6 = pred_local_R6.reshape(B, T, -1, 6)
            _, pred_global_p = motionops.R6_fk(pred_local_R6, pred_root_p, skeleton)

            # predicted trajectory
            pred_traj_xz = pred_root_p[..., (0, 2)]
            pred_root_R = rotation.R6_to_R(pred_local_R6[:, :, 0])
            pred_traj_forward = F.normalize(torch.matmul(pred_root_R, v_forward) * torchconst.XZ(device), dim=-1)
            global_forward = torchconst.FORWARD(device).expand(B, T, -1)
            pred_signed_angle = mathops.signed_angle(global_forward, pred_traj_forward)
            pred_traj = torch.cat([pred_traj_xz, pred_signed_angle.unsqueeze(-1)], dim=-1)
            
            # loss
            loss_rot  = config.weight_rot * F.l1_loss(pred_local_R6, GT_local_R6)
            loss_pos  = config.weight_pos * F.l1_loss(pred_global_p, GT_global_p)
            loss_vel  = config.weight_vel * F.l1_loss(pred_global_p[:, 1:] - pred_global_p[:, :-1], GT_global_p[:, 1:] - GT_global_p[:, :-1])
            loss_traj = config.weight_traj * F.l1_loss(pred_traj, GT_traj)
            loss_kl   = config.weight_kl * torch.mean(0.5 * torch.sum(torch.exp(pred_sigma) + pred_mu**2 - 1.0 - pred_sigma, dim=-1))
            loss      = loss_rot + loss_pos + loss_vel + loss_traj + loss_kl

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            # log
            loss_dict["total"]  += loss.item()
            loss_dict["rot"]    += loss_rot.item()
            loss_dict["pos"]    += loss_pos.item()
            loss_dict["vel"]    += loss_vel.item()
            loss_dict["traj"]   += loss_traj.item()
            loss_dict["kl"]     += loss_kl.item()
            

            if iter % config.log_interval == 0:
                tqdm.write(f"Iter {iter} | Loss: {loss_dict['total'] / config.log_interval:.4f} | Rot: {loss_dict['rot'] / config.log_interval:.4f} | Pos: {loss_dict['pos'] / config.log_interval:.4f} | Vel: {loss_dict['vel'] / config.log_interval:.4f} | Traj: {loss_dict['traj'] / config.log_interval:.4f} | KL: {loss_dict['kl'] / config.log_interval:.4f} | Time: {(time.perf_counter() - start_time) / 60:.2f} min")
                writer.add_scalar("loss/total",  loss_dict["total"]  / config.log_interval, iter)
                writer.add_scalar("loss/rot",    loss_dict["rot"]    / config.log_interval, iter)
                writer.add_scalar("loss/pos",    loss_dict["pos"]    / config.log_interval, iter)
                writer.add_scalar("loss/vel",    loss_dict["vel"]    / config.log_interval, iter)
                writer.add_scalar("loss/traj",   loss_dict["traj"]   / config.log_interval, iter)
                writer.add_scalar("loss/kl",     loss_dict["kl"]     / config.log_interval, iter)

                for k in loss_dict.keys():
                    loss_dict[k] = 0.0
            
            if iter % config.save_interval == 0:
                trainutil.save_ckpt(model, optim, epoch, iter, config, scheduler)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(model, optim, epoch, iter, config, scheduler)