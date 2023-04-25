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
from model.contextual import ContextualGAN
from utility import utils

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/contextual_gan.json")
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
    model = ContextualGAN(dataset.shape[-1], config, is_context=True).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    init_epoch, iter = utils.load_latest_ckpt(model, optim, config)
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
        "smooth":  0,
        "gen":     0,
        "disc":    0,
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

            """ 3. Train discriminator """
            # normalize - generate
            GT_batch = (GT_motion - motion_mean) / motion_std
            ctx_motion, _ = model.generate(GT_batch, GT_traj)

            # discriminator
            short_fake, long_fake = model.discriminate(ctx_motion.detach())
            short_real, long_real = model.discriminate(GT_batch)

            # loss
            loss_disc_short = config.weight_adv * utils.discriminator_loss(short_real, short_fake)
            loss_disc_long  = config.weight_adv * utils.discriminator_loss(long_real, long_fake)
            loss_disc       = loss_disc_short + loss_disc_long

            # backward
            optim.zero_grad()
            loss_disc.backward()
            optim.step()

            # log
            loss_dict["total"] += loss_disc.item()
            loss_dict["disc"]  += loss_disc.item()

            """ 4. Train generator """
            # generate
            ctx_motion, _ = model.generate(GT_batch, GT_traj)

            # adversarial loss
            short_fake, long_fake = model.discriminate(ctx_motion)
            loss_gen_short = config.weight_adv * utils.generator_loss(short_fake)
            loss_gen_long  = config.weight_adv * utils.generator_loss(long_fake)
            loss_gen       = loss_gen_short + loss_gen_long

            # predicted motion features
            ctx_motion = ctx_motion * motion_std + motion_mean
            ctx_local_R6, ctx_global_p, ctx_traj = utils.get_motion_and_trajectory(ctx_motion, skeleton, v_forward)

            # loss
            loss_pose   = config.weight_pose * (utils.recon_loss(ctx_local_R6, GT_local_R6) + utils.recon_loss(ctx_global_p, GT_global_p))
            loss_traj   = config.weight_traj * utils.traj_loss(ctx_traj, GT_traj)
            loss_smooth = config.weight_smooth * (utils.smooth_loss(ctx_local_R6) + utils.smooth_loss(ctx_global_p))
            loss        = loss_gen + loss_pose + loss_traj + loss_smooth

            # backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # log
            loss_dict["total"]  += loss.item()
            loss_dict["pose"]   += loss_pose.item()
            loss_dict["traj"]   += loss_traj.item()
            loss_dict["smooth"] += loss_smooth.item()
            loss_dict["gen"]    += loss_gen.item()

            """ 5. Log """
            if iter % config.log_interval == 0:
                utils.write_log(writer, loss_dict, config.log_interval, iter, train=True)
                utils.reset_log(loss_dict)
            
            """ 6. Validation """
            if iter % config.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_dict = {
                        "total":   0,
                        "pose":    0,
                        "traj":    0,
                        "smooth":  0,
                    }
                    for GT_motion in val_dataloader:
                        """ 6-1. Max transition length """
                        T = config.context_frames + config.max_transition + 1
                        GT_motion = GT_motion[:, :T, :].to(device)
                        B, T, D = GT_motion.shape

                        """ 6-2. GT motion data """
                        GT_local_R6, GT_global_p, GT_traj = utils.get_motion_and_trajectory(GT_motion, skeleton, v_forward)

                        """ 6-3. Forward """
                        # normalize - forward - denormalize
                        GT_batch = (GT_motion - motion_mean) / motion_std
                        ctx_motion, _ = model.generate(GT_batch, GT_traj)
                        ctx_motion = ctx_motion * motion_std + motion_mean

                        # predicted motion
                        ctx_local_R6, ctx_global_p, ctx_traj = utils.get_motion_and_trajectory(ctx_motion, skeleton, v_forward)

                        """ 6-4. Loss """
                        loss_pose   = config.weight_pose * (utils.recon_loss(ctx_local_R6, GT_local_R6) + utils.recon_loss(ctx_global_p, GT_global_p))
                        loss_traj   = config.weight_traj * utils.traj_loss(ctx_traj, GT_traj)
                        loss_smooth = config.weight_smooth * (utils.smooth_loss(ctx_local_R6) + utils.smooth_loss(ctx_global_p))
                        loss        = loss_pose + loss_traj + loss_smooth

                        """ 7-5. Update loss dict """
                        val_loss_dict["total"]  += loss.item()
                        val_loss_dict["pose"]   += loss_pose.item()
                        val_loss_dict["traj"]   += loss_traj.item()
                        val_loss_dict["smooth"] += loss_smooth.item()

                # write and print log
                utils.write_log(writer, val_loss_dict, len(val_dataloader), iter, train=False)
                utils.reset_log(val_loss_dict)

                # train mode
                model.train()

            """ 8. Save checkpoint """
            if iter % config.save_interval == 0:
                utils.save_ckpt(model, optim, epoch, iter, config)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            # update iter
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    utils.save_ckpt(model, optim, epoch, iter, config)