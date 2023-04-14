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
from model.ours import KeyframeGAN
from utility import trainutil

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/keyframe_gan.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = KeyframeDataset(train=True, config=config)
    val_dataset = KeyframeDataset(train=False, config=config)
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
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # model
    print("Initializing model...")
    model = KeyframeGAN(dataset.shape[-1] - 4, config).to(device) # exclude trajectory and keyframe score
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
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
        "gen":     0,
        "disc":    0,
    }
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_keyframe in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            B, T, D = GT_keyframe.shape

            """ 1. Prepare GT data """
            GT_keyframe = GT_keyframe.to(device)
            GT_local_R6, GT_root_p, GT_kf_score, GT_traj = torch.split(GT_keyframe, [D-7, 3, 1, 3], dim=-1)
            _, GT_global_p = motionops.R6_fk(GT_local_R6.reshape(B, T, -1, 6), GT_root_p, skeleton)

            GT_batch = torch.cat([GT_local_R6, GT_root_p, GT_traj], dim=-1) # exclude keyframe score
            GT_batch = (GT_batch - kf_mean) / kf_std

            """ 2. Train discriminator """
            fake_motion, _ = model.generate(GT_batch) if isinstance(model, torch.nn.Module) else model.module.generate(GT_batch)
            disc_real_short, disc_real_long = model.discriminate(GT_batch[..., :-3]) if isinstance(model, torch.nn.Module) else model.module.discriminate(GT_batch[..., :-3])
            disc_fake_short, disc_fake_long = model.discriminate(fake_motion.detach()) if isinstance(model, torch.nn.Module) else model.module.discriminate(fake_motion.detach())

            loss_disc_real = -(torch.mean(torch.log(disc_real_short + 1e-8)) + torch.mean(torch.log(disc_real_long + 1e-8)))
            loss_disc_fake = -(torch.mean(torch.log(1 - disc_fake_short + 1e-8)) + torch.mean(torch.log(1 - disc_fake_long + 1e-8)))
            loss_disc = config.weight_adv * (loss_disc_real + loss_disc_fake)

            optim.zero_grad()
            loss_disc.backward()
            optim.step()

            """ 3. Train generator """
            pred_motion, pred_kf_score = model.generate(GT_batch) if isinstance(model, torch.nn.Module) else model.module.generate(GT_batch)
            disc_fake_short, disc_fake_long = model.discriminate(pred_motion) if isinstance(model, torch.nn.Module) else model.module.discriminate(pred_motion)
            
            # GAN loss
            loss_gen = config.weight_adv * -(torch.mean(torch.log(disc_fake_short + 1e-8)) + torch.mean(torch.log(disc_fake_long + 1e-8)))

            # predicted keyframe features
            pred_motion = pred_motion * kf_std[..., :-3] + kf_mean[..., :-3] # exclude traj features

            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
            _, pred_global_p = motionops.R6_fk(pred_local_R6.reshape(B, T, -1, 6), pred_root_p, skeleton)

            # predicted trajectory
            pred_traj_xz = pred_root_p[..., (0, 2)]
            pred_root_R = rotation.R6_to_R(pred_local_R6[:, :, :6])
            pred_traj_forward = F.normalize(torch.matmul(pred_root_R, v_forward) * torchconst.XZ(device), dim=-1)
            global_forward = torchconst.FORWARD(device).expand(B, T, -1)
            pred_signed_angle = mathops.signed_angle(global_forward, pred_traj_forward)
            pred_traj = torch.cat([pred_traj_xz, pred_signed_angle.unsqueeze(-1)], dim=-1)

            # loss
            loss_score   = config.weight_score * torch.mean(torch.abs(pred_kf_score - GT_kf_score))
            loss_pose    = config.weight_pose  * (
                torch.mean(torch.abs(pred_local_R6 - GT_local_R6) * GT_kf_score) +\
                torch.mean(torch.abs(pred_global_p - GT_global_p).reshape(B, T, -1) * GT_kf_score)
            )
            loss_traj    = config.weight_traj  * torch.mean(torch.abs(pred_traj - GT_traj) * GT_kf_score)
            loss = loss_gen + loss_score + loss_pose + loss_traj

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # log
            loss_dict["total"]   += loss.item()
            loss_dict["score"]   += loss_score.item()
            loss_dict["pose"]    += loss_pose.item()
            loss_dict["traj"]    += loss_traj.item()
            loss_dict["gen"]     += loss_gen.item()
            loss_dict["disc"]    += loss_disc.item()

            if iter % config.log_interval == 0:
                tqdm.write(f"Iter {iter} | Loss: {loss_dict['total'] / config.log_interval:.4f} | Score: {loss_dict['score'] / config.log_interval:.4f} | Pose: {loss_dict['pose'] / config.log_interval:.4f} | Traj: {loss_dict['traj'] / config.log_interval:.4f} | Gen: {loss_dict['gen'] / config.log_interval:.4f} | Disc: {loss_dict['disc'] / config.log_interval:.4f} | Time: {(time.perf_counter() - start_time) / 60:.2f} min")
                writer.add_scalar("loss/total", loss_dict["total"]  / config.log_interval, iter)
                writer.add_scalar("loss/score", loss_dict["score"]  / config.log_interval, iter)
                writer.add_scalar("loss/pose",  loss_dict["pose"]   / config.log_interval, iter)
                writer.add_scalar("loss/traj",  loss_dict["traj"]   / config.log_interval, iter)
                writer.add_scalar("loss/gen",   loss_dict["gen"]    / config.log_interval, iter)
                writer.add_scalar("loss/disc",  loss_dict["disc"]   / config.log_interval, iter)
                
                for k in loss_dict.keys():
                    loss_dict[k] = 0
            
            if iter % config.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_dict = {"pose": 0, "traj": 0, "score": 0, "total": 0}
                    for GT_motion in tqdm(val_dataloader, desc="Validation"):
                        GT_motion = GT_motion.to(device)
                        GT_local_R6, GT_root_p, GT_kf_score, GT_traj = torch.split(GT_keyframe, [D-7, 3, 1, 3], dim=-1)
                        GT_batch = torch.cat([GT_local_R6, GT_root_p, GT_traj], dim=-1) # exclude keyframe score
                        GT_batch = (GT_batch - kf_mean) / kf_std

                        # forward
                        pred_motion, pred_kf_score = model.generate(GT_batch) if isinstance(model, torch.nn.Module) else model.module.generate(GT_batch)
                        pred_motion = pred_motion * kf_std[..., :-3] + kf_mean[..., :-3]
                        
                        # predicted motion
                        pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
                        _, pred_global_p = motionops.R6_fk(pred_local_R6.reshape(B, T, -1, 6), pred_root_p, skeleton)

                        # predicted trajectory
                        pred_traj_xz = pred_root_p[..., (0, 2)]
                        pred_root_R = rotation.R6_to_R(pred_local_R6[:, :, :6])
                        pred_traj_forward = F.normalize(torch.matmul(pred_root_R, v_forward) * torchconst.XZ(device), dim=-1)
                        global_forward = torchconst.FORWARD(device).expand(B, T, -1)
                        pred_signed_angle = mathops.signed_angle(global_forward, pred_traj_forward)
                        pred_traj = torch.cat([pred_traj_xz, pred_signed_angle.unsqueeze(-1)], dim=-1)

                        # loss
                        loss_score   = torch.mean(torch.abs(pred_kf_score - GT_kf_score))
                        loss_pose    = torch.mean(torch.abs(pred_local_R6 - GT_local_R6)) + torch.mean(torch.abs(pred_global_p - GT_global_p))
                        loss_traj    = torch.mean(torch.abs(pred_traj - GT_traj) * GT_kf_score)
                        loss = loss_score + loss_pose + loss_traj

                        # log
                        val_loss_dict["total"]   += loss.item()
                        val_loss_dict["score"]   += loss_score.item()
                        val_loss_dict["pose"]    += loss_pose.item()
                        val_loss_dict["traj"]    += loss_traj.item()

                    tqdm.write(f"Iter {iter} | Val Loss: {val_loss_dict['total'] / len(val_dataloader):.4f} | Val Score: {val_loss_dict['score'] / len(val_dataloader):.4f} | Val Pose: {val_loss_dict['pose'] / len(val_dataloader):.4f} | Val Traj: {val_loss_dict['traj'] / len(val_dataloader):.4f}")
                    writer.add_scalar("val_loss/total", val_loss_dict["total"]  / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/score", val_loss_dict["score"]  / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/pose",  val_loss_dict["pose"]   / len(val_dataloader), iter)
                    writer.add_scalar("val_loss/traj",  val_loss_dict["traj"]   / len(val_dataloader), iter)
                model.train()

            if iter % config.save_interval == 0:
                trainutil.save_ckpt(model, optim, epoch, iter, config)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(model, optim, epoch, iter, config)