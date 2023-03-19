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
from model.twostage import ContextTransformer, DetailTransformer
from model.ours import SparseTransformer
from utility import trainutil, testutil

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/finetune.json")
    sparse_config = Config.load("configs/sparse.json")
    context_config = Config.load("configs/context.json")
    detail_config = Config.load("configs/detail.json")
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
    sparse_model = SparseTransformer(dataset.shape[-1], sparse_config).to(device)
    sparse_optim = torch.optim.Adam(sparse_model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    sparse_scheduler = trainutil.get_noam_scheduler(config, sparse_optim)
    trainutil.load_latest_ckpt(sparse_model, sparse_optim, config, sparse_scheduler)
    
    context_model = ContextTransformer(dataset.shape[-1], context_config).to(device)
    context_optim = torch.optim.Adam(context_model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    context_scheduler = trainutil.get_noam_scheduler(config, context_optim)
    trainutil.load_latest_ckpt(context_model, context_optim, config, context_scheduler)

    detail_model = DetailTransformer(dataset.shape[-1], detail_config).to(device)
    detail_optim = torch.optim.Adam(detail_model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    detail_scheduler = trainutil.get_noam_scheduler(config, detail_optim)
    trainutil.load_latest_ckpt(detail_model, detail_optim, config, detail_scheduler)

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
    for epoch in range(1, config.epochs+1):
        for GT_motion in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            B, T, D = GT_motion.shape
            
            # GT
            GT_motion = GT_motion.to(device)
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R6 = GT_local_R6.reshape(B, T, -1, 6)
            _, GT_global_p = motionops.R6_fk(GT_local_R6, GT_root_p, skeleton)

            # SparseTransformer - Get keyframes first
            batch = (GT_motion - motion_mean) / motion_std

            keyframe_batch = batch[:, sparse_frames]
            pred_keyframe, mask = sparse_model.forward(keyframe_batch, sparse_frames)
            pred_keyframe = mask * keyframe_batch + (1-mask) * pred_keyframe
            batch[:, sparse_frames] = pred_keyframe

            """ Infill missing frames """
            pred_motion = batch.clone()
            refine_frames = torch.cat([torch.arange(config.context_frames), torch.tensor(config.context_frames+config.fps-1)])
            for i in range(config.max_transition // config.fps):
                # denormalize
                input_batch = pred_motion[:, refine_frames[0]:refine_frames[-1]+1]
                input_batch = input_batch * motion_std + motion_mean
                local_R6, root_p = torch.split(input_batch, [D-3, 3], dim=-1)

                # delta to align at the last context frame
                local_R = rotation.R6_to_R(local_R6.reshape(local_R6.shape[0], local_R6.shape[1], -1, 6))
                root_R = local_R[:, :, 0]
                forward = torch.matmul(root_R[:, config.context_frames-1], v_forward)
                forward = F.normalize(forward * torchconst.XZ(device), dim=-1)
                up = torchconst.Y(device).unsqueeze(0).repeat(B, 1)
                delta_R = torch.stack([torch.cross(up, forward), up, forward], dim=-2).unsqueeze(1)
                root_R = torch.matmul(delta_R, root_R)
                root_R6 = rotation.R_to_R6(root_R)

                delta_p = root_p[:, config.context_frames-1:config.context_frames] * torchconst.XZ(device)
                root_p = root_p - delta_p

                # align and normalize
                input_batch[:, :, :6] = root_R6
                input_batch[:, :, -3:] = root_p
                input_batch = (input_batch - motion_mean) / motion_std

                # ContextTransformer
                context_motion, mask = context_model.forward(input_batch, ratio_constrained=0.0, prob_constrained=0.0)
                context_motion = mask * input_batch + (1 - mask) * context_motion

                detail_motion, detail_contact = detail_model.forward(context_motion, mask)
                detail_motion = mask * input_batch + (1 - mask) * detail_motion

                # denormalize again
                detail_motion = detail_motion * motion_std + motion_mean
                local_R6, root_p = torch.split(detail_motion, [D-3, 3], dim=-1)

                # restore the original root
                local_R = rotation.R6_to_R(local_R6.reshape(local_R6.shape[0], local_R6.shape[1], -1, 6))
                root_R = local_R[:, :, 0]
                root_R = torch.matmul(delta_R.transpose(-1, -2), root_R)
                root_R6 = rotation.R_to_R6(root_R)
                root_p = root_p + delta_p
                detail_motion[:, :, :6] = root_R6
                detail_motion[:, :, -3:] = root_p

                # normalize
                detail_motion = (detail_motion - motion_mean) / motion_std
                pred_motion[:, refine_frames[0]:refine_frames[-1]+1] = detail_motion
                refine_frames += config.fps





            pred_motion = pred_motion * motion_std + motion_mean

            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-3, 3], dim=-1)
            pred_local_R6 = pred_local_R6.reshape(B, T, -1, 6)
            _, pred_global_p = motionops.R6_fk(pred_local_R6, pred_root_p, skeleton)
            
            # loss
            loss_rot = config.weight_rot * F.l1_loss(pred_local_R6, GT_local_R6)
            loss_pos = config.weight_pos * F.l1_loss(pred_global_p, GT_global_p)
            loss_vel = config.weight_vel * F.l1_loss(pred_global_p[:, 1:] - pred_global_p[:, :-1], GT_global_p[:, 1:] - GT_global_p[:, :-1])
            loss = loss_rot + loss_pos + loss_vel

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            # log
            loss_dict["total"] += loss.item()
            loss_dict["rot"]   += loss_rot.item()
            loss_dict["pos"]   += loss_pos.item()
            loss_dict["vel"]   += loss_vel.item()

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
                trainutil.save_ckpt(refine_model, optim, epoch, iter, config, scheduler)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(refine_model, optim, epoch, iter, config, scheduler)