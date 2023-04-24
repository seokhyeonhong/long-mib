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
from utility import utils
from model.gan import KeyframeGAN

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/keyframe_gan.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset     = KeyframeDataset(train=True, config=config)
    val_dataset = KeyframeDataset(train=False, config=config)
    skeleton    = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)

    kf_mean, kf_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = kf_mean[..., :-1], kf_std[..., :-1]
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    
    feet_ids = []
    for name in config.contact_joint_names:
        feet_ids.append(skeleton.idx_by_name[name])

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # model
    print("Initializing model...")
    model = KeyframeGAN(dataset.shape[-1] - 1, config).to(device)
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
        "total": 0,
        "score": 0,
        "pose":  0,
        "traj":  0,
        "gen":   0,
        "disc":  0,
    }

    # training
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_keyframe in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            """ 1. Random transition length """
            T = config.context_frames + config.max_transition + 1
            GT_keyframe = GT_keyframe[:, :T, :].to(device)
            B, T, D = GT_keyframe.shape

            """ 2. GT motion data """
            GT_motion, GT_kf_score = torch.split(GT_keyframe, [D-1, 1], dim=-1)
            GT_local_R6, GT_global_p, GT_traj = utils.get_motion_and_trajectory(GT_motion, skeleton, v_forward)

            """ 3. Train discriminator """
            # normalize and generate
            GT_batch = (GT_motion - motion_mean) / motion_std
            pred_motion, pred_score, _ = model.generate(GT_batch, GT_traj)

            # discriminate
            real_short, real_long = model.discriminate(GT_batch)
            fake_short, fake_long = model.discriminate(pred_motion.detach())

            # loss
            loss_short = config.weight_adv * utils.discriminator_loss(real_short, fake_short)
            loss_long  = config.weight_adv * utils.discriminator_loss(real_long, fake_long)
            loss_disc  = loss_short + loss_long
            
            # backward
            optim.zero_grad()
            loss_disc.backward()
            optim.step()

            # log
            loss_dict["total"] += loss_disc.item()
            loss_dict["disc"]  += loss_disc.item()

            """ 4. Train generator """
            # generate
            pred_motion, pred_score, _ = model.generate(GT_batch, GT_traj)

            # discriminate
            fake_short, fake_long = model.discriminate(pred_motion)

            # loss
            loss_gen = config.weight_adv * (utils.generator_loss(fake_short) + utils.generator_loss(fake_long))

            # predicted motion features
            pred_motion = pred_motion * motion_std + motion_mean
            pred_local_R6, pred_global_p, pred_traj = utils.get_motion_and_trajectory(pred_motion, skeleton, v_forward)

            # loss
            loss_score = config.weight_score * utils.recon_loss(pred_score, GT_kf_score)
            loss_pose  = config.weight_pose * (utils.recon_loss(pred_local_R6, GT_local_R6) + utils.recon_loss(pred_global_p, GT_global_p))
            loss_traj  = config.weight_traj * utils.traj_loss(pred_traj, GT_traj)
            loss_total = loss_gen + loss_score + loss_pose + loss_traj

            # backward
            optim.zero_grad()
            loss_total.backward()
            optim.step()

            # log
            loss_dict["total"] += loss_total.item()
            loss_dict["score"] += loss_score.item()
            loss_dict["pose"]  += loss_pose.item()
            loss_dict["traj"]  += loss_traj.item()
            loss_dict["gen"]   += loss_gen.item()

            """ 5. Log """
            if iter % config.log_interval == 0:
                utils.write_log(writer, loss_dict, config.log_interval, iter, elapsed=time.perf_counter() - start_time, train=True)
                utils.reset_log(loss_dict)
            
            """ 6. Validation """
            if iter % config.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_dict = {
                        "total": 0,
                        "score": 0,
                        "pose":  0,
                        "traj":  0,
                    }
                    for GT_keyframe in val_dataloader:
                        """ 6-1. Max transition length """
                        T = config.context_frames + config.max_transition + 1
                        GT_keyframe = GT_keyframe[:, :T, :].to(device)
                        B, T, D = GT_keyframe.shape

                        """ 6-2. GT motion data """
                        GT_motion, GT_kf_score = torch.split(GT_keyframe, [D-1, 1], dim=-1)
                        GT_local_R6, GT_global_p, GT_traj = utils.get_motion_and_trajectory(GT_motion, skeleton, v_forward)

                        """ 6-3. Forward """
                        # generate motion
                        GT_batch = (GT_motion - motion_mean) / motion_std
                        pred_motion, pred_score, _ = model.generate(GT_batch, GT_traj)
                        pred_motion = pred_motion * motion_std + motion_mean

                        # predicted motion
                        pred_local_R6, pred_global_p, pred_traj = utils.get_motion_and_trajectory(pred_motion, skeleton, v_forward)

                        """ 6-4. Loss """
                        # loss
                        loss_score = config.weight_score * utils.recon_loss(pred_score, GT_kf_score)
                        loss_pose  = config.weight_pose * (utils.recon_loss(pred_local_R6, GT_local_R6) + utils.recon_loss(pred_global_p, GT_global_p))
                        loss_traj  = config.weight_traj * utils.traj_loss(pred_traj, GT_traj)
                        loss_total = loss_score + loss_pose + loss_traj

                        """ 6-5. Update loss dict """
                        val_loss_dict["total"] += loss_total.item()
                        val_loss_dict["score"] += loss_score.item()
                        val_loss_dict["pose"]  += loss_pose.item()
                        val_loss_dict["traj"]  += loss_traj.item()

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