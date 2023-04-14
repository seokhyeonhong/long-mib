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

from utility.dataset import KeyframePairDataset
from utility.config import Config
from model.ours import KeyframeContrastive
from utility import trainutil

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/keyframe_contrastive.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = KeyframePairDataset(train=True, config=config)
    val_dataset = KeyframePairDataset(train=False, config=config)
    skeleton   = dataset.skeleton
    v_forward  = torch.from_numpy(config.v_forward).to(device)

    kf_mean, kf_std = dataset.statistics(dim=(0, 1))
    kf_mean, kf_std = kf_mean.to(device), kf_std.to(device)
    
    # exclude score from statistics
    D = dataset.shape[-1]
    mean_motion, _, mean_traj = torch.split(kf_mean, [D-4, 1, 3], dim=-1)
    kf_mean = torch.cat([mean_motion, mean_traj], dim=-1)

    std_motion, _, std_traj = torch.split(kf_std, [D-4, 1, 3], dim=-1)
    kf_std = torch.cat([std_motion, std_traj], dim=-1)
    
    # dataloader
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # model
    print("Initializing model...")
    model = KeyframeContrastive(dataset.shape[-1] - 1, config).to(device) # exclude keyframe score
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
        "score":   0,
        "pose":    0,
        "traj":    0,
        "triplet": 0,
    }
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_positive, GT_negative in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            """ 1. GT data """
            B, T, D = GT_positive.shape
            GT_positive, GT_negative = GT_positive.to(device), GT_negative.to(device)

            GT_local_R6, GT_root_p, GT_kf_score, GT_traj = torch.split(GT_positive, [D-7, 3, 1, 3], dim=-1)
            _, GT_global_p = motionops.R6_fk(GT_local_R6.reshape(B, T, -1, 6), GT_root_p, skeleton)

            neg_local_R6, neg_root_p, neg_kf_score, neg_traj = torch.split(GT_negative, [D-7, 3, 1, 3], dim=-1)

            """ 2. Positive and Negative Pairs """
            x_positive = torch.cat([GT_local_R6, GT_root_p, GT_traj], dim=-1)
            x_negative = torch.cat([neg_local_R6, neg_root_p, neg_traj], dim=-1)

            """ 3. Forward """
            pred_motion, pred_kf_score, x_anchor, x_positive, x_negative = model.forward(x_positive, x_negative)
            pred_motion = pred_motion * kf_std[..., :-3] + kf_mean[..., :-3] # exclude traj features

            """ 4. Predicted Motion and Trajectory Features """
            # motion
            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
            _, pred_global_p = motionops.R6_fk(pred_local_R6.reshape(B, T, -1, 6), pred_root_p, skeleton)

            # trajectory
            pred_traj_xz = pred_root_p[..., (0, 2)]
            pred_root_R = rotation.R6_to_R(pred_local_R6[:, :, :6])
            pred_traj_forward = F.normalize(torch.matmul(pred_root_R, v_forward) * torchconst.XZ(device), dim=-1)
            global_forward = torchconst.FORWARD(device).expand(B, T, -1)
            pred_signed_angle = mathops.signed_angle(global_forward, pred_traj_forward)
            pred_traj = torch.cat([pred_traj_xz, pred_signed_angle.unsqueeze(-1)], dim=-1)

            """ 5. Loss """
            loss_score   = config.weight_score * F.l1_loss(pred_kf_score, GT_kf_score)
            loss_pose    = config.weight_pose * (
                torch.mean(torch.abs(pred_local_R6 - GT_local_R6) * GT_kf_score) +\
                torch.mean(torch.abs(pred_global_p - GT_global_p).reshape(B, T, -1) * GT_kf_score)
            )
            loss_traj    = config.weight_traj  * torch.mean(torch.abs(pred_traj - GT_traj) * GT_kf_score)
            loss_triplet = config.weight_triplet * torch.mean(
                torch.max(torch.zeros(B, T, device=device, dtype=x_anchor.dtype), torch.norm(x_anchor - x_positive, dim=-1)**2 - torch.norm(x_anchor - x_negative, dim=-1)**2 + config.margin),
            )
            loss = loss_score + loss_pose + loss_traj + loss_triplet

            """ 6. Backward """
            optim.zero_grad()
            loss.backward()
            optim.step()

            """ 7. Log """
            loss_dict["total"]   += loss.item()
            loss_dict["score"]   += loss_score.item()
            loss_dict["pose"]    += loss_pose.item()
            loss_dict["traj"]    += loss_traj.item()
            loss_dict["triplet"] += loss_triplet.item()

            if iter % config.log_interval == 0:
                tqdm.write(f"Iter {iter} | Loss: {loss_dict['total'] / config.log_interval:.4f} | Score: {loss_dict['score'] / config.log_interval:.4f} | Pose: {loss_dict['pose'] / config.log_interval:.4f} | Traj: {loss_dict['traj'] / config.log_interval:.4f} | Triplet: {loss_dict['triplet'] / config.log_interval:.4f} | Time: {(time.perf_counter() - start_time) / 60:.2f} min")
                writer.add_scalar("loss/total",   loss_dict["total"]  / config.log_interval, iter)
                writer.add_scalar("loss/score",   loss_dict["score"]  / config.log_interval, iter)
                writer.add_scalar("loss/pose",    loss_dict["pose"]   / config.log_interval, iter)
                writer.add_scalar("loss/traj",    loss_dict["traj"]   / config.log_interval, iter)
                writer.add_scalar("loss/triplet", loss_dict["triplet"]/ config.log_interval, iter)
                
                for k in loss_dict.keys():
                    loss_dict[k] = 0
            
            if iter % config.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_dict = {"total": 0, "score": 0, "pose": 0, "traj": 0, "triplet": 0}
                    for GT_positive, GT_negative in tqdm(val_dataloader, desc=f"Validation", leave=False):
                        B, T, D = GT_positive.shape
                        GT_positive, GT_negative = GT_positive.to(device), GT_negative.to(device)

                        GT_local_R6, GT_root_p, GT_kf_score, GT_traj = torch.split(GT_positive, [D-7, 3, 1, 3], dim=-1)
                        _, GT_global_p = motionops.R6_fk(GT_local_R6.reshape(B, T, -1, 6), GT_root_p, skeleton)

                        neg_local_R6, neg_root_p, neg_kf_score, neg_traj = torch.split(GT_negative, [D-7, 3, 1, 3], dim=-1)

                        x_positive = torch.cat([GT_local_R6, GT_root_p, GT_traj], dim=-1)
                        x_negative = torch.cat([neg_local_R6, neg_root_p, neg_traj], dim=-1)

                        pred_motion, pred_kf_score, x_anchor, x_positive, x_negative = model.forward(x_positive, x_negative)
                        pred_motion = pred_motion * kf_std[..., :-3] + kf_mean[..., :-3]

                        pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
                        _, pred_global_p = motionops.R6_fk(pred_local_R6.reshape(B, T, -1, 6), pred_root_p, skeleton)

                        pred_traj_xz = pred_root_p[..., (0, 2)]
                        pred_root_R = rotation.R6_to_R(pred_local_R6[:, :, :6])
                        pred_traj_forward = F.normalize(torch.matmul(pred_root_R, v_forward) * torchconst.XZ(device), dim=-1)
                        global_forward = torchconst.FORWARD(device).expand(B, T, -1)
                        pred_signed_angle = mathops.signed_angle(global_forward, pred_traj_forward)
                        pred_traj = torch.cat([pred_traj_xz, pred_signed_angle.unsqueeze(-1)], dim=-1)

                        loss_score   = config.weight_score * F.l1_loss(pred_kf_score, GT_kf_score)
                        loss_pose    = config.weight_pose * (
                            torch.mean(torch.abs(pred_local_R6 - GT_local_R6) * GT_kf_score) +\
                            torch.mean(torch.abs(pred_global_p - GT_global_p).reshape(B, T, -1) * GT_kf_score)
                        )
                        loss_traj    = config.weight_traj  * torch.mean(torch.abs(pred_traj - GT_traj) * GT_kf_score)
                        loss_triplet = config.weight_triplet * torch.mean(
                            torch.max(torch.zeros(B, T, device=device, dtype=x_anchor.dtype), torch.norm(x_anchor - x_positive, dim=-1)**2 - torch.norm(x_anchor - x_negative, dim=-1)**2 + config.margin),
                        )
                        loss = loss_score + loss_pose + loss_traj + loss_triplet
                        
                        val_loss_dict["total"]   += loss.item()
                        val_loss_dict["score"]   += loss_score.item()
                        val_loss_dict["pose"]    += loss_pose.item()
                        val_loss_dict["traj"]    += loss_traj.item()
                        val_loss_dict["triplet"] += loss_triplet.item()
                        
                    tqdm.write(f"Validation | Loss: {val_loss_dict['total'] / len(val_dataloader):.4f} | Score: {val_loss_dict['score'] / len(val_dataloader):.4f} | Pose: {val_loss_dict['pose'] / len(val_dataloader):.4f} | Traj: {val_loss_dict['traj'] / len(val_dataloader):.4f} | Triplet: {val_loss_dict['triplet'] / len(val_dataloader):.4f}")
                    writer.add_scalar("val_loss/total",   val_loss_dict["total"]  / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/score",   val_loss_dict["score"]  / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/pose",    val_loss_dict["pose"]   / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/traj",    val_loss_dict["traj"]   / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/triplet", val_loss_dict["triplet"]/ len(val_dataloader), iter)

                    for k in val_loss_dict.keys():
                        val_loss_dict[k] = 0

                model.train()

            if iter % config.save_interval == 0:
                trainutil.save_ckpt(model, optim, epoch, iter, config)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(model, optim, epoch, iter, config)