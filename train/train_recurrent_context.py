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
from model.twostage import ContextTransformer
from utility import trainutil

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/recurrent_context.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=True, config=config)
    skeleton   = dataset.skeleton

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    v_forward = torch.from_numpy(config.v_forward).to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = ContextTransformer(dataset.shape[-1], config).to(device)
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
    sparse_frames = torch.arange(config.max_transition // config.fps) * config.fps
    sparse_frames += (config.context_frames-1) + config.fps
    sparse_frames = torch.cat([torch.arange(config.context_frames), sparse_frames])
    loss_dict = {
        "total":  0,
        "rot":    0,
        "pos":    0,
        "vel":    0,
    }
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_motion in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            B, T, D = GT_motion.shape

            T = config.context_frames + config.max_transition
            GT_motion = GT_motion[:, :T, :]
            GT_motion = GT_motion.to(device)

            # GT
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R6 = GT_local_R6.reshape(B, T, -1, 6)
            _, GT_global_p = motionops.R6_fk(GT_local_R6, GT_root_p, skeleton)

            # forward
            batch = (GT_motion - motion_mean) / motion_std

            refine_frames = torch.cat([torch.arange(config.context_frames), torch.arange(config.context_frames+config.fps-1, config.context_frames+config.fps)])
            loss = 0
            for i in range(config.max_transition // config.fps):
                # 1. denormalize and get motion features
                input_batch = batch[:, refine_frames[0]:refine_frames[-1]+1]
                input_batch = input_batch * motion_std + motion_mean
                input_local_R6, input_root_p = torch.split(input_batch, [D-3, 3], dim=-1)
                input_local_R6 = input_local_R6.reshape(*input_local_R6.shape[:2], -1, 6)

                # 2. calculate delta R and delta p to align the root
                input_root_R = rotation.R6_to_R(input_local_R6)[:, :, 0]
                forward = torch.matmul(input_root_R[:, config.context_frames-1], v_forward)
                forward = F.normalize(forward * torchconst.XZ(device), dim=-1)
                up = torchconst.Y(device).unsqueeze(0).repeat(B, 1)

                delta_R = torch.stack([torch.cross(up, forward), up, forward], dim=-2).unsqueeze(1)
                input_root_R = torch.matmul(delta_R, input_root_R)
                input_root_R6 = rotation.R_to_R6(input_root_R)

                delta_p = (input_root_p[:, config.context_frames-1] * torchconst.XZ(device)).unsqueeze(1)
                input_root_p = torch.matmul(delta_R, (input_root_p - delta_p).unsqueeze(-1)).squeeze(-1)

                # 3. align at the last context frame and normalize
                input_batch[:, :, :6] = input_root_R6
                input_batch[:, :, -3:] = input_root_p
                input_batch = (input_batch - motion_mean) / motion_std

                # 4. forward pass
                pred_motion, mask = model.forward(input_batch, ratio_constrained=0.0, prob_constrained=0.0)

                # 5. denormalize and get motion features
                pred_motion = pred_motion * motion_std + motion_mean
                pred_local_R6, pred_root_p = torch.split(pred_motion, [D-3, 3], dim=-1)
                pred_local_R6 = pred_local_R6.reshape(*pred_local_R6.shape[:2], -1, 6)
                _, pred_global_p = motionops.R6_fk(pred_local_R6, pred_root_p, skeleton)

                # 6. align GT motion features
                gt_local_R6 = GT_local_R6[:, refine_frames[0]:refine_frames[-1]+1].clone()
                gt_root_p = GT_root_p[:, refine_frames[0]:refine_frames[-1]+1].clone()

                gt_root_R = rotation.R6_to_R(gt_local_R6)[:, :, 0]
                gt_root_R = torch.matmul(delta_R, gt_root_R)
                gt_root_R6 = rotation.R_to_R6(gt_root_R)

                gt_root_p = torch.matmul(delta_R, (gt_root_p - delta_p).unsqueeze(-1)).squeeze(-1)

                gt_local_R6[:, :, 0] = gt_root_R6
                gt_local_R6 = gt_local_R6.reshape(*gt_local_R6.shape[:2], -1, 6)
                _, gt_global_p = motionops.R6_fk(gt_local_R6, gt_root_p, skeleton)

                # loss
                loss_rot = config.weight_rot * F.l1_loss(pred_local_R6, gt_local_R6)
                loss_pos = config.weight_pos * F.l1_loss(pred_global_p, gt_global_p)
                loss_vel = config.weight_vel * F.l1_loss(pred_global_p[:, 1:] - pred_global_p[:, :-1], gt_global_p[:, 1:] - gt_global_p[:, :-1])
                loss += loss_rot + loss_pos + loss_vel

                # log
                loss_dict["total"] += loss.item()
                loss_dict["rot"] += loss_rot.item()
                loss_dict["pos"] += loss_pos.item()
                loss_dict["vel"] += loss_vel.item()
                
                # normalize again with normalized
                pred_motion = (pred_motion - motion_mean) / motion_std
                batch[:, refine_frames[0]:refine_frames[-1]+1] = pred_motion.detach()
                refine_frames += config.fps
            
            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            
            if iter % config.log_interval == 0:
                tqdm.write(f"Iter {iter} | Loss: {loss_dict['total'] / config.log_interval:.4f} | Rot: {loss_dict['rot'] / config.log_interval:.4f} | Pos: {loss_dict['pos'] / config.log_interval:.4f} | Vel: {loss_dict['vel'] / config.log_interval:.4f} | Time: {(time.perf_counter() - start_time) / 60:.2f} min")
                writer.add_scalar("loss/total", loss_dict["total"] / config.log_interval, iter)
                writer.add_scalar("loss/rot",   loss_dict["rot"]   / config.log_interval, iter)
                writer.add_scalar("loss/pos",   loss_dict["pos"]   / config.log_interval, iter)
                writer.add_scalar("loss/vel",   loss_dict["vel"]   / config.log_interval, iter)
                loss_dict = {
                    "total":  0,
                    "rot":    0,
                    "pos":    0,
                    "vel":    0,
                }
            
            if iter % config.save_interval == 0:
                trainutil.save_ckpt(model, optim, epoch, iter, config, scheduler)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(model, optim, epoch, iter, config, scheduler)