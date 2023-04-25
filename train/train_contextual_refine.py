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
from model.contextual import ContextualTransformer
from utility import utils

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/contextual_refine.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset     = MotionDataset(train=True, config=config)
    val_dataset = MotionDataset(train=False, config=config)
    skeleton    = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    
    feet_ids = []
    for name in config.contact_joint_names:
        feet_ids.append(skeleton.idx_by_name[name])

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # model
    print("Initializing model...")
    ctx_config = Config.load("configs/contextual.json")
    ctx_model = ContextualTransformer(dataset.shape[-1], ctx_config, is_context=True).to(device)
    utils.load_model(ctx_model, ctx_config)
    ctx_model.eval()

    det_model = ContextualTransformer(dataset.shape[-1], config, is_context=False).to(device)
    optim = torch.optim.Adam(det_model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    init_epoch, iter = utils.load_latest_ckpt(det_model, optim, config)
    init_iter = iter

    # save and log
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    config.write(os.path.join(config.save_dir, "config.json"))
    writer = SummaryWriter(config.log_dir)

    # loss dict
    loss_dict = {
        "total":   0,
        "pose":    0,
        "traj":    0,
        "contact": 0,
        "foot":    0,
    }

    # training
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_motion in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            """ 1. Random transition length """
            transition = random.randint(config.min_transition, config.max_transition)
            T = config.context_frames + transition + 1
            GT_motion = GT_motion[:, :T, :].to(device)
            B, T, D = GT_motion.shape

            """ 2. GT motion data """
            GT_local_R6, GT_global_p, GT_traj = utils.get_motion_and_trajectory(GT_motion, skeleton, v_forward)
            GT_feet_v, GT_contact = utils.get_velocity_and_contact(GT_global_p, feet_ids, config.contact_vel_threshold)

            """ 3. Forward """
            # normalize - forward - denormalize
            GT_batch = (GT_motion - motion_mean) / motion_std

            with torch.no_grad():
                ctx_motion, mask = ctx_model.forward(GT_batch, GT_traj)
            
            det_motion, _ = det_model.forward(GT_batch, GT_traj, mask)
            det_motion, det_contact = torch.split(det_motion, [D, 4], dim=-1)
            det_motion = det_motion * motion_std + motion_mean

            # predicted motion features
            det_local_R6, det_global_p, det_traj = utils.get_motion_and_trajectory(det_motion, skeleton, v_forward)
            det_feet_v, _ = utils.get_velocity_and_contact(det_global_p, feet_ids, config.contact_vel_threshold)

            # loss
            loss_pose    = config.weight_pose * (utils.recon_loss(det_local_R6, GT_local_R6) + utils.recon_loss(det_global_p, GT_global_p))
            loss_traj    = config.weight_traj * utils.traj_loss(det_traj, GT_traj)
            loss_contact = config.weight_contact * utils.recon_loss(det_contact, GT_contact)
            loss_foot    = config.weight_foot * utils.foot_loss(det_feet_v, det_contact.detach())
            loss         = loss_pose + loss_traj + loss_contact + loss_foot

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # log
            loss_dict["total"]   += loss.item()
            loss_dict["pose"]    += loss_pose.item()
            loss_dict["traj"]    += loss_traj.item()
            loss_dict["contact"] += loss_contact.item()
            loss_dict["foot"]    += loss_foot.item()

            """ 5. Log """
            if iter % config.log_interval == 0:
                utils.write_log(writer, loss_dict, config.log_interval, iter, train=True)
                utils.reset_log(loss_dict)
            
            """ 6. Validation """
            if iter % config.val_interval == 0:
                det_model.eval()
                with torch.no_grad():
                    val_loss_dict = {
                        "total":   0,
                        "pose":    0,
                        "traj":    0,
                        "contact": 0,
                        "foot":    0,
                    }
                    for GT_motion in val_dataloader:
                        """ 6-1. Max transition length """
                        T = config.context_frames + config.max_transition + 1
                        GT_motion = GT_motion[:, :T, :].to(device)
                        B, T, D = GT_motion.shape

                        """ 6-2. GT motion data """
                        GT_local_R6, GT_global_p, GT_traj = utils.get_motion_and_trajectory(GT_motion, skeleton, v_forward)
                        GT_feet_v, GT_contact = utils.get_velocity_and_contact(GT_global_p, feet_ids, config.contact_vel_threshold)

                        """ 6-3. Forward """
                        # normalize - forward - denormalize
                        GT_batch = (GT_motion - motion_mean) / motion_std
                        ctx_motion, mask = ctx_model.forward(GT_batch, GT_traj)
                        det_motion, _ = det_model.forward(GT_batch, GT_traj, mask)
                        det_motion, det_contact = torch.split(det_motion, [D, 4], dim=-1)
                        det_motion = det_motion * motion_std + motion_mean

                        # predicted motion
                        det_local_R6, det_global_p, det_traj = utils.get_motion_and_trajectory(det_motion, skeleton, v_forward)
                        det_feet_v, _ = utils.get_velocity_and_contact(det_global_p, feet_ids, config.contact_vel_threshold)

                        """ 6-4. Loss """
                        loss_pose    = config.weight_pose * (utils.recon_loss(det_local_R6, GT_local_R6) + utils.recon_loss(det_global_p, GT_global_p))
                        loss_traj    = config.weight_traj * utils.traj_loss(det_traj, GT_traj)
                        loss_contact = config.weight_contact * utils.recon_loss(det_contact, GT_contact)
                        loss_foot    = config.weight_foot * utils.foot_loss(det_feet_v, det_contact.detach())
                        loss         = loss_pose + loss_traj + loss_contact + loss_foot

                        """ 7-5. Update loss dict """
                        val_loss_dict["total"]   += loss.item()
                        val_loss_dict["pose"]    += loss_pose.item()
                        val_loss_dict["traj"]    += loss_traj.item()
                        val_loss_dict["contact"] += loss_contact.item()
                        val_loss_dict["foot"]    += loss_foot.item()

                # write and print log
                utils.write_log(writer, val_loss_dict, len(val_dataloader), iter, train=False)
                utils.reset_log(val_loss_dict)

                # train mode
                det_model.train()

            """ 8. Save checkpoint """
            if iter % config.save_interval == 0:
                utils.save_ckpt(det_model, optim, epoch, iter, config)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            # update iter
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    utils.save_ckpt(det_model, optim, epoch, iter, config)