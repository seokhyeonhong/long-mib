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
from model.gan import TwoStageGAN
from utility import trainutil, loss


if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/twostagegan.json")
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
    model = TwoStageGAN(dataset.shape[-1], config).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    init_epoch, iter = trainutil.load_latest_ckpt(model, optim, config)
    init_iter = iter

    # save and log
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    config.write(os.path.join(config.save_dir, "config.json"))
    writer = SummaryWriter(config.log_dir)

    # loss dict
    loss_dict = {
        "total":       0,

        "ctx":         0,
        "ctx/pose":    0,
        "ctx/traj":    0,
        "ctx/smooth":  0,
        "ctx/gen":     0,
        "ctx/disc":    0,

        "det":         0,
        "det/pose":    0,
        "det/traj":    0,
        "det/contact": 0,
        "det/foot":    0,
        "det/gen":     0,
        "det/disc":    0,
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
            GT_local_R6, GT_global_p, GT_traj = get_motion_and_trajectory(GT_motion, skeleton, v_forward)
            GT_feet_v, GT_contact = get_velocity_and_contact(GT_global_p, feet_ids, config.contact_vel_threshold)

            """ 3. Train discriminator """
            # normalize and generate
            GT_batch = (GT_motion - motion_mean) / motion_std
            ctx_motion, det_motion, det_contact = model.generate(GT_batch, GT_traj)

            # discriminate
            real_ctx_short, real_ctx_long, real_det_short, real_det_long = model.discriminate(GT_batch, GT_batch)
            fake_ctx_short, fake_ctx_long, fake_det_short, fake_det_long = model.discriminate(ctx_motion.detach(), det_motion.detach())

            # loss
            loss_ctx = config.weight_adv * (loss.discriminator_loss(real_ctx_short, fake_ctx_short) + loss.discriminator_loss(real_ctx_long, fake_ctx_long))
            loss_det = config.weight_adv * (loss.discriminator_loss(real_det_short, fake_det_short) + loss.discriminator_loss(real_det_long, fake_det_long))
            loss_disc = loss_ctx + loss_det
            
            # backward
            optim.zero_grad()
            loss_disc.backward()
            optim.step()

            # log
            loss_dict["ctx"]      += loss_ctx.item()
            loss_dict["ctx/disc"] += loss_ctx.item()
            loss_dict["det"]      += loss_det.item()
            loss_dict["det/disc"] += loss_det.item()
            loss_dict["total"]    += loss_disc.item()

            """ 4. Train generator """
            # generate
            ctx_motion, det_motion, det_contact = model.generate(GT_batch, GT_traj)

            # discriminate
            fake_ctx_short, fake_ctx_long, fake_det_short, fake_det_long = model.discriminate(ctx_motion, det_motion)

            # adversarial loss
            loss_ctx_gen = config.weight_adv * (loss.generator_loss(fake_ctx_short) + loss.generator_loss(fake_ctx_long))
            loss_det_gen = config.weight_adv * (loss.generator_loss(fake_det_short) + loss.generator_loss(fake_det_long))

            # predicted motion features
            ctx_motion = ctx_motion * motion_std + motion_mean
            det_motion = det_motion * motion_std + motion_mean

            ctx_local_R6, ctx_global_p, ctx_traj = get_motion_and_trajectory(ctx_motion, skeleton, v_forward)
            det_local_R6, det_global_p, det_traj = get_motion_and_trajectory(det_motion, skeleton, v_forward)
            det_feet_v, _ = get_velocity_and_contact(det_global_p, feet_ids, config.contact_vel_threshold)

            # loss
            loss_ctx_pose = config.weight_ctx_pose * (loss.recon_loss(ctx_local_R6, GT_local_R6) + loss.recon_loss(ctx_global_p, GT_global_p))
            loss_ctx_traj = config.weight_ctx_traj * loss.traj_loss(ctx_traj, GT_traj)
            loss_ctx_smooth = config.weight_ctx_smooth * (loss.smooth_loss(ctx_local_R6) + loss.smooth_loss(ctx_global_p))
            loss_ctx = loss_ctx_gen + loss_ctx_pose + loss_ctx_traj + loss_ctx_smooth

            loss_det_pose = config.weight_det_pose * (loss.recon_loss(det_local_R6, GT_local_R6) + loss.recon_loss(det_global_p, GT_global_p))
            loss_det_traj = config.weight_det_traj * loss.traj_loss(det_traj, GT_traj)
            loss_det_contact = config.weight_det_contact * loss.recon_loss(det_contact, GT_contact)
            loss_det_foot = config.weight_det_foot * loss.foot_loss(det_feet_v, det_contact.detach())
            loss_det = loss_det_gen + loss_det_pose + loss_det_traj + loss_det_contact + loss_det_foot

            loss_total = loss_ctx + loss_det

            # backward
            optim.zero_grad()
            loss_total.backward()
            optim.step()

            # log
            loss_dict["total"]       += loss_total.item()

            loss_dict["ctx"]         += loss_ctx.item()
            loss_dict["ctx/gen"]     += loss_ctx_gen.item()
            loss_dict["ctx/pose"]    += loss_ctx_pose.item()
            loss_dict["ctx/traj"]    += loss_ctx_traj.item()
            loss_dict["ctx/smooth"]  += loss_ctx_smooth.item()

            loss_dict["det"]         += loss_det.item()
            loss_dict["det/gen"]     += loss_det_gen.item()
            loss_dict["det/pose"]    += loss_det_pose.item()
            loss_dict["det/traj"]    += loss_det_traj.item()
            loss_dict["det/contact"] += loss_det_contact.item()
            loss_dict["det/foot"]    += loss_det_foot.item()

            """ 5. Log """
            if iter % config.log_interval == 0:
                tqdm.write(f"Iter {iter} | Total: {(loss_dict['total'] / config.log_interval):.4f} | Context: {(loss_dict['ctx'] / config.log_interval):.4f} | Detail: {(loss_dict['det'] / config.log_interval):.4f}")
                trainutil.write_log(writer, loss_dict, config.log_interval, iter, train=True)

                # reset loss dict
                for k in loss_dict.keys():
                    loss_dict[k] = 0
            
            """ 6. Validation """
            if iter % config.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_dict = {
                        "total":       0,

                        "ctx":         0,
                        "ctx/pose":    0,
                        "ctx/traj":    0,
                        "ctx/smooth":  0,

                        "det":         0,
                        "det/pose":     0,
                        "det/traj":    0,
                        "det/contact": 0,
                        "det/foot":    0,
                    }
                    for GT_motion in val_dataloader:
                        """ 6-1. Max transition length """
                        T = config.context_frames + config.max_transition + 1
                        GT_motion = GT_motion[:, :T, :].to(device)
                        B, T, D = GT_motion.shape

                        """ 6-2. GT motion data """
                        GT_local_R6, GT_global_p, GT_traj = get_motion_and_trajectory(GT_motion, skeleton, v_forward)
                        GT_feet_v, GT_contact = get_velocity_and_contact(GT_global_p, feet_ids, config.contact_vel_threshold)

                        """ 6-3. Forward """
                        # generate motion
                        GT_batch = (GT_motion - motion_mean) / motion_std
                        ctx_motion, det_motion, det_contact = model.generate(GT_batch, GT_traj)

                        ctx_motion = ctx_motion * motion_std + motion_mean
                        det_motion = det_motion * motion_std + motion_mean

                        # predicted motion
                        ctx_local_R6, ctx_global_p, ctx_traj = get_motion_and_trajectory(ctx_motion, skeleton, v_forward)
                        det_local_R6, det_global_p, det_traj = get_motion_and_trajectory(det_motion, skeleton, v_forward)
                        det_feet_v, _ = get_velocity_and_contact(det_global_p, feet_ids, config.contact_vel_threshold)

                        """ 6-4. Loss """
                        # ContextNet
                        loss_ctx_pose   = config.weight_ctx_pose * (loss.recon_loss(ctx_local_R6, GT_local_R6) + loss.recon_loss(ctx_global_p, GT_global_p))
                        loss_ctx_traj   = config.weight_ctx_traj * loss.traj_loss(ctx_traj, GT_traj)
                        loss_ctx_smooth = config.weight_ctx_smooth * (loss.smooth_loss(ctx_local_R6) + loss.smooth_loss(ctx_global_p))
                        loss_ctx        = loss_ctx_pose + loss_ctx_traj + loss_ctx_smooth

                        # DetailNet
                        loss_det_pose    = config.weight_det_pose * (loss.recon_loss(det_local_R6, GT_local_R6) + loss.recon_loss(det_global_p, GT_global_p))
                        loss_det_traj    = config.weight_det_traj * loss.traj_loss(det_traj, GT_traj)
                        loss_det_contact = config.weight_det_contact * loss.recon_loss(det_contact, GT_contact)
                        loss_det_foot    = config.weight_det_foot * loss.foot_loss(det_feet_v, det_contact.detach())
                        loss_det         = loss_det_pose + loss_det_traj + loss_det_contact + loss_det_foot

                        # total loss
                        loss_total = loss_ctx + loss_det

                        """ 7-5. Update loss dict """
                        val_loss_dict["total"]       += loss_total.item()
                        
                        val_loss_dict["ctx"]         += loss_ctx.item()
                        val_loss_dict["ctx/pose"]    += loss_ctx_pose.item()
                        val_loss_dict["ctx/traj"]    += loss_ctx_traj.item()
                        val_loss_dict["ctx/smooth"]  += loss_ctx_smooth.item()

                        val_loss_dict["det"]         += loss_det.item()
                        val_loss_dict["det/pose"]    += loss_det_pose.item()
                        val_loss_dict["det/traj"]    += loss_det_traj.item()
                        val_loss_dict["det/contact"] += loss_det_contact.item()
                        val_loss_dict["det/foot"]    += loss_det_foot.item()

                # average
                for k in val_loss_dict.keys():
                    val_loss_dict[k] /= len(val_dataloader)
                
                # write and print log
                trainutil.write_log(writer, val_loss_dict, 1, iter, train=False)
                tqdm.write(f"Validation at Iter {iter} | Total: {val_loss_dict['total']:.4f} | Context: {val_loss_dict['ctx']:.4f} | Detail: {val_loss_dict['det']:.4f}")

                # train mode
                model.train()

            """ 8. Save checkpoint """
            if iter % config.save_interval == 0:
                trainutil.save_ckpt(model, optim, epoch, iter, config)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            # update iter
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(model, optim, epoch, iter, config)