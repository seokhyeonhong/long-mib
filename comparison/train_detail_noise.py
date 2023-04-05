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

from pymovis.utils import util
from pymovis.ops import motionops

from utility.dataset import MotionDataset
from utility.config import Config
from model.twostage import ContextTransformer, DetailTransformer
from utility import trainutil, testutil

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/detail_noise.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=True, config=config)
    skeleton   = dataset.skeleton

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    motion_mean, motion_std = motion_mean[..., :-5], motion_std[..., :-5] # exclude trajectory
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    feet_ids = []
    for name in config.contact_joint_names:
        feet_ids.append(skeleton.idx_by_name[name])

    # model
    print("Initializing model...")
    ctx_model = ContextTransformer(dataset.shape[-1] - 5, Config.load("configs/context_noise.json")).to(device) # exclude trajectory
    testutil.load_model(ctx_model, Config.load("configs/context_noise.json"))
    ctx_model.eval()

    det_model = DetailTransformer(dataset.shape[-1] - 5, config).to(device) # exclude trajectory
    optim = torch.optim.Adam(det_model.parameters(), lr=config.d_model**-0.5, betas=(0.9, 0.98), eps=1e-9)
    scheduler = trainutil.get_noam_scheduler(config, optim)
    init_epoch, iter = trainutil.load_latest_ckpt(det_model, optim, config, scheduler)
    init_iter = iter

    # save and log
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    config.write(os.path.join(config.save_dir, "config.json"))
    writer = SummaryWriter(config.log_dir)

    # training loop
    loss_dict = {
        "total":   0,
        "rot":     0,
        "pos":     0,
        "contact": 0,
        "foot":    0,
    }
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_motion in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            transition_frames = random.randint(config.min_transition, config.max_transition)
            T = config.context_frames + transition_frames + 1
            GT_motion = GT_motion[:, :T, :-5] # exclude trajectory
            B, T, D = GT_motion.shape

            # GT
            GT_motion = GT_motion.to(device)
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R6 = GT_local_R6.reshape(B, T, -1, 6)
            _, GT_global_p = motionops.R6_fk(GT_local_R6, GT_root_p, skeleton)

            GT_feet_v = GT_global_p[:, 1:, feet_ids] - GT_global_p[:, :-1, feet_ids]
            GT_feet_v = torch.sum(GT_feet_v**2, dim=-1) # squared norm
            GT_feet_v = torch.cat([GT_feet_v[:, 0:1], GT_feet_v], dim=1)
            GT_contact = (GT_feet_v < config.contact_vel_threshold).float()

            # forward - ContextTransformer
            with torch.no_grad():
                batch = (GT_motion - motion_mean) / motion_std
                batch += torch.randn_like(batch) * 0.05
                ctx_motion, mask = ctx_model.forward(batch)
                ctx_motion = mask * batch + (1 - mask) * ctx_motion # restore GT

            # forward - DetailTransformer
            pred_motion, pred_contact = det_model.forward(ctx_motion, mask)
            pred_motion = pred_motion * motion_std + motion_mean

            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-3, 3], dim=-1)
            pred_local_R6 = pred_local_R6.reshape(B, T, -1, 6)
            _, pred_global_p = motionops.R6_fk(pred_local_R6, pred_root_p, skeleton)

            pred_feet_v = pred_global_p[:, 1:, feet_ids] - pred_global_p[:, :-1, feet_ids]
            pred_feet_v = torch.sum(pred_feet_v**2, dim=-1) # squared norm
            pred_feet_v = torch.cat([pred_feet_v[:, 0:1], pred_feet_v], dim=1)
            
            # loss
            loss_rot     = config.weight_rot     * F.l1_loss(pred_local_R6[:, config.context_frames:-1], GT_local_R6[:, config.context_frames:-1])
            loss_pos     = config.weight_pos     * F.l1_loss(pred_global_p[:, config.context_frames:-1], GT_global_p[:, config.context_frames:-1])
            loss_contact = config.weight_contact * F.l1_loss(pred_contact[:, config.context_frames:-1], GT_contact[:, config.context_frames:-1])
            loss_foot    = config.weight_foot    * F.l1_loss(pred_contact[:, config.context_frames:-1].detach() * pred_feet_v[:, config.context_frames:-1], torch.zeros_like(pred_feet_v[:, config.context_frames:-1]))
            loss = loss_rot + loss_pos + loss_contact + loss_foot

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            # log
            loss_dict["total"]   += loss.item()
            loss_dict["rot"]     += loss_rot.item()
            loss_dict["pos"]     += loss_pos.item()
            loss_dict["contact"] += loss_contact.item()
            loss_dict["foot"]    += loss_foot.item()

            if iter % config.log_interval == 0:
                tqdm.write(f"Iter {iter} | Loss: {loss_dict['total'] / config.log_interval:.4f} | Rot: {loss_dict['rot'] / config.log_interval:.4f} | Pos: {loss_dict['pos'] / config.log_interval:.4f} | Contact: {loss_dict['contact'] / config.log_interval:.4f} | Foot: {loss_dict['foot'] / config.log_interval:.4f} | Time: {(time.perf_counter() - start_time) / 60:.2f} min")
                writer.add_scalar("loss/total",   loss_dict["total"]   / config.log_interval, iter)
                writer.add_scalar("loss/rot",     loss_dict["rot"]     / config.log_interval, iter)
                writer.add_scalar("loss/pos",     loss_dict["pos"]     / config.log_interval, iter)
                writer.add_scalar("loss/contact", loss_dict["contact"] / config.log_interval, iter)
                writer.add_scalar("loss/foot",    loss_dict["foot"]    / config.log_interval, iter)

                loss_dict = {
                    "total":  0,
                    "rot":    0,
                    "pos":    0,
                    "contact": 0,
                    "foot":   0,
                }
            
            if iter % config.save_interval == 0:
                trainutil.save_ckpt(det_model, optim, epoch, iter, config, scheduler)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(det_model, optim, epoch, iter, config, scheduler)