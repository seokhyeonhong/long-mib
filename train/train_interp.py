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

from utility.dataset import MotionDataset
from utility.config import Config
from model.ours import InterpolationTransformer
from utility import trainutil

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/interp_complete.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=True, config=config)
    skeleton   = dataset.skeleton
    v_forward  = torch.from_numpy(skeleton.v_forward).to(device)

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    motion_mean, motion_std = motion_mean[:-5], motion_std[:-5] # exclude trajectory
    
    feet_ids = []
    for name in config.contact_joint_names:
        feet_ids.append(skeleton.idx_by_name[name])

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = InterpolationTransformer(dataset.shape[-1], config).to(device) # exclude trajectory
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
        "contact":0,
        "feet":   0,
    }
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_motion in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            # GT
            T = config.context_frames + config.max_transition + 1
            GT_motion = GT_motion[:, :T].to(device)
            B, T, D = GT_motion.shape

            GT_motion, GT_traj = torch.split(GT_motion, [D-5, 5], dim=-1)
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-8, 3], dim=-1)
            GT_local_R6 = GT_local_R6.reshape(B, T, -1, 6)
            GT_local_R  = rotation.R6_to_R(GT_local_R6)
            _, GT_global_p = motionops.R_fk(GT_local_R, GT_root_p, skeleton)

            GT_feet_v = GT_global_p[:, 1:, feet_ids] - GT_global_p[:, :-1, feet_ids]
            GT_feet_v = torch.sum(GT_feet_v**2, dim=-1) # squared norm
            GT_contact = (GT_feet_v < config.contact_vel_threshold).float()
            GT_contact = torch.cat([GT_contact[:, 0:1], GT_contact], dim=1) # pad

            # keyframes and interpolated motion
            keyframes = model.get_random_keyframes(T)
            interp_motion = model.get_interpolated_motion(GT_local_R, GT_root_p, keyframes)

            # forward
            batch = (interp_motion - motion_mean) / motion_std
            pred_motion, pred_contact = model.forward(batch, keyframes, GT_traj)
            pred_motion = pred_motion * motion_std + motion_mean # exclude trajectory

            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-8, 3], dim=-1)
            pred_local_R6 = pred_local_R6.reshape(B, T, -1, 6)
            _, pred_global_p = motionops.R6_fk(pred_local_R6, pred_root_p, skeleton)

            pred_root_xz = pred_root_p[..., (0, 2)]
            pred_root_R  = rotation.R6_to_R(pred_local_R6[:, :, 0])
            pred_root_fwd = F.normalize((torch.matmul(pred_root_R, v_forward) * torchconst.XZ(device)), dim=-1)
            pred_traj = torch.cat([pred_root_xz, pred_root_fwd], dim=-1)

            pred_feet_v = pred_global_p[:, 1:, feet_ids] - pred_global_p[:, :-1, feet_ids]
            pred_feet_v = torch.sum(pred_feet_v**2, dim=-1) # squared norm
            pred_feet_v = torch.cat([pred_feet_v[:, 0:1], pred_feet_v], dim=1) # pad
            
            # loss
            loss_rot  = config.weight_rot * F.l1_loss(pred_local_R6, GT_local_R6)
            loss_pos  = config.weight_pos * F.l1_loss(pred_global_p, GT_global_p)
            loss_vel  = config.weight_vel * F.l1_loss(pred_global_p[:, 1:] - pred_global_p[:, :-1], GT_global_p[:, 1:] - GT_global_p[:, :-1])
            loss_traj = config.weight_traj * F.l1_loss(pred_traj, GT_traj)
            loss_contact = config.weight_contact * F.binary_cross_entropy(pred_contact, GT_contact)
            loss_feet = config.weight_feet * F.l1_loss(pred_contact.detach() * pred_feet_v, torch.zeros_like(pred_feet_v))
            loss = loss_rot + loss_pos + loss_vel + loss_traj + loss_contact + loss_feet

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
            loss_dict["contact"]+= loss_contact.item()
            loss_dict["feet"]   += loss_feet.item()
            

            if iter % config.log_interval == 0:
                tqdm.write(f"Iter {iter} | Loss: {loss_dict['total'] / config.log_interval:.4f} | Rot: {loss_dict['rot'] / config.log_interval:.4f} | Pos: {loss_dict['pos'] / config.log_interval:.4f} | Vel: {loss_dict['vel'] / config.log_interval:.4f} | Traj: {loss_dict['traj'] / config.log_interval:.4f} | Contact: {loss_dict['contact'] / config.log_interval:.4f} | Feet: {loss_dict['feet'] / config.log_interval:.4f} | Time: {(time.perf_counter() - start_time) / 60:.2f} min")
                writer.add_scalar("loss/total",  loss_dict["total"]  / config.log_interval, iter)
                writer.add_scalar("loss/rot",    loss_dict["rot"]    / config.log_interval, iter)
                writer.add_scalar("loss/pos",    loss_dict["pos"]    / config.log_interval, iter)
                writer.add_scalar("loss/vel",    loss_dict["vel"]    / config.log_interval, iter)
                writer.add_scalar("loss/traj",   loss_dict["traj"]   / config.log_interval, iter)
                writer.add_scalar("loss/contact",loss_dict["contact"]/ config.log_interval, iter)
                writer.add_scalar("loss/feet",   loss_dict["feet"]   / config.log_interval, iter)
                loss_dict = {
                    "total":  0,
                    "rot":    0,
                    "pos":    0,
                    "vel":    0,
                    "traj":   0,
                    "contact":0,
                    "feet":   0,
                }
            
            if iter % config.save_interval == 0:
                trainutil.save_ckpt(model, optim, epoch, iter, config, scheduler)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(model, optim, epoch, iter, config, scheduler)