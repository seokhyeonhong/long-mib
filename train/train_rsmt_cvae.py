import sys
sys.path.append(".")

import os
import time
import random
from tqdm import tqdm
import argparse

from aPyOpenGL import transforms as trf

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import utils, loss, ops
from utils.dataset import MotionDataset
from model.cvae import MotionVAE


if __name__ =="__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default="rsmt_cvae.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = utils.load_config(f"config/{args.dataset}/{args.config}")
    utils.seed()

    # dataset
    dataset = MotionDataset(train=True, config=config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    skeleton = dataset.skeleton

    mean, std = dataset.motion_statistics(device)

    val_dataset = MotionDataset(train=False, config=config)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    contact_idx = []
    for joint in config.contact_joints:
        contact_idx.append(skeleton.idx_by_name[joint])

    # model, optimizer, scheduler
    model = MotionVAE(config, dataset.motion.shape[-1], dataset.phase.shape[-1]).to(device)
    optim = Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), amsgrad=True)

    init_epoch = utils.load_latest_ckpt(model, optim, config)

    # save and log
    os.makedirs(config.save_dir, exist_ok=True)
    utils.write_config(config)
    writer = SummaryWriter(config.save_dir)
    loss_dict = {
        "recon": 0.0,
        "kl": 0.0,
        "foot": 0.0,
        "total": 0.0,
    }

    # function for each iteration
    def train_iter(batch, train=True):
        model.train() if train else model.eval()

        # GT data
        GT_motion = batch["motion"].to(device)
        GT_phase = batch["phase"].to(device)

        B, T, M = GT_motion.shape
        GT_local_ortho6ds, GT_root_pos = torch.split(GT_motion, [M-3, 3], dim=-1)
        GT_local_ortho6ds = GT_local_ortho6ds.reshape(B, T, skeleton.num_joints, 6)
        _, GT_global_positions = trf.t_ortho6d.fk(GT_local_ortho6ds, GT_root_pos, skeleton)

        GT_global_vel = GT_global_positions[:, 1:] - GT_global_positions[:, :-1] # (B, t-1, J, 3)
        GT_global_vel = torch.cat([GT_global_vel[:, 0:1], GT_global_vel], dim=1) # (B, t, J, 3)
        GT_global_vel_hip = GT_global_vel[:, :, 0]
        GT_global_vel_mag = torch.sum(GT_global_vel ** 2, dim=-1) # (B, t-1, J)
        GT_contact = (GT_global_vel_mag[..., contact_idx] < config.contact_threshold).float() # (B, t, 4)

        # normalization
        GT_motion = (GT_motion - mean) / std

        # random frame index
        curr_idx = list(range(0, T-1))
        next_idx = list(range(1, T))

        # forward
        delta_motion, mu, logvar = model.forward(
            GT_motion[:, curr_idx], GT_motion[:, next_idx], GT_phase[:, next_idx], GT_global_vel_hip[:, next_idx]
        )

        # pred
        pred_motion = GT_motion[:, curr_idx].clone() + delta_motion
        pred_motion = pred_motion * std + mean

        pred_local_ortho6ds, pred_root_pos = torch.split(pred_motion, [M-3, 3], dim=-1)
        pred_local_ortho6ds = pred_local_ortho6ds.reshape(B, T-1, skeleton.num_joints, 6)
        _, pred_global_positions = trf.t_ortho6d.fk(pred_local_ortho6ds, pred_root_pos, skeleton)

        pred_global_vel = pred_global_positions - GT_global_positions[:, :-1] # (B, t-1, J, 3)
        pred_global_vel = torch.cat([pred_global_vel[:, 0:1], pred_global_vel], dim=1) # (B, t, J, 3)
        pred_global_vel_mag = torch.sum(pred_global_vel ** 2, dim=-1) # (B, t, J)
        pred_global_vel_mag_foot = pred_global_vel_mag[..., contact_idx]

        # loss
        loss_recon = F.mse_loss(delta_motion, GT_motion[:, next_idx] - GT_motion[:, curr_idx])
        loss_kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1))
        loss_foot = F.mse_loss(pred_global_vel_mag_foot * GT_contact, torch.zeros_like(pred_global_vel_mag_foot))
        loss = loss_recon + config.weight_kl * loss_kl + config.weight_foot * loss_foot

        # backward
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        # log
        loss_dict["recon"] += loss_recon.item()
        loss_dict["kl"] += loss_kl.item()
        loss_dict["foot"] += loss_foot.item()
        loss_dict["total"] += loss.item()

    # main loop
    start_time = time.perf_counter()
    for epoch in range(init_epoch+1, config.epochs+1):
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False)):
            train_iter(batch, train=True)   

        # log training
        elapsed = time.perf_counter() - start_time
        utils.write_log(writer, loss_dict, len(dataloader), epoch, elapsed=elapsed, train=True)
        utils.reset_log(loss_dict)

        # validation
        if epoch % config.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_dataloader, desc=f"Validation", leave=False)):
                    train_iter(batch, train=False)

                # log validation
                utils.write_log(writer, loss_dict, len(val_dataloader), epoch, train=False)
                utils.reset_log(loss_dict)

        # save checkpoint - every 10 epochs
        if epoch % config.save_interval == 0:
            utils.save_ckpt(model, optim, epoch, config)

    # save checkpoint - last epoch
    utils.save_ckpt(model, optim, epoch, config)
    print(f"Training finished in {(time.perf_counter() - start_time) / 60:.2f} min")