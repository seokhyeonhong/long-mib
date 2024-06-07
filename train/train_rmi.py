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
from model.rmi import RmiGenerator, RmiDiscriminator

def get_transition(generator: RmiGenerator, motion, contact, context_frames):
    B, T, D = motion.shape

    local_rot, root_pos = torch.split(motion, [D-3, 3], dim=-1)
    root_vel = root_pos[:, 1:] - root_pos[:, :-1]
    root_vel = torch.cat([root_vel[:, 0:1], root_vel], dim=1)

    target = motion[:, -1]
    target_local_rot, target_root_pos = torch.split(target, [D-3, 3], dim=-1)
    
    generator.init_hidden(B, motion.device)
    pred_rot, pred_root_pos, pred_contact = [local_rot[:, 0]], [root_pos[:, 0]], [contact[:, 0]]
    for i in range(context_frames):
        tta = T - i - 1
        lr, rp, c = generator.forward(local_rot[:, i], root_pos[:, i], root_vel[:, i], contact[:, i], target_local_rot, target_root_pos, tta)
        pred_rot.append(lr)
        pred_root_pos.append(rp)
        pred_contact.append(c)
    for i in range(context_frames, T-1):
        tta = T - i - 1
        lr, rp, c = generator.forward(lr, rp, rp - pred_root_pos[-1], c, target_local_rot, target_root_pos, tta)
        pred_rot.append(lr)
        pred_root_pos.append(rp)
        pred_contact.append(c)
    
    # stack transition frames without context frames
    pred_rot = torch.stack(pred_rot, dim=1)
    pred_root_pos = torch.stack(pred_root_pos, dim=1)
    pred_contact = torch.stack(pred_contact, dim=1)

    pred_motion = torch.cat([pred_rot, pred_root_pos], dim=-1)

    return pred_motion, pred_contact
    

if __name__ =="__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default="rmi.yaml")
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
    model_g = RmiGenerator(config, skeleton.num_joints).to(device)
    model_d_short = RmiDiscriminator(config, skeleton.num_joints, window_size=2).to(device)
    model_d_long = RmiDiscriminator(config, skeleton.num_joints, window_size=10).to(device)

    optim_g = Adam(model_g.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), amsgrad=True)
    optim_d_short = Adam(model_d_short.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), amsgrad=True)
    optim_d_long = Adam(model_d_long.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), amsgrad=True)

    init_epoch = utils.load_latest_ckpt(model_g, optim_g, config)

    # save and log
    os.makedirs(config.save_dir, exist_ok=True)
    utils.write_config(config)
    writer = SummaryWriter(config.save_dir)
    loss_dict = {
        "rot": 0.0,
        "pos": 0.0,
        "root": 0.0,
        "contact": 0.0,
        "disc": 0.0,
        "gen": 0.0,
        "total": 0.0,
    }

    # function for each iteration
    def train_iter(batch, train=True):
        # transitiion length
        trans_len = random.randint(config.min_trans, config.max_trans) if train else config.max_trans
        target_idx = config.context_frames + trans_len

        # GT data
        GT_motion = batch["motion"].to(device)
        GT_motion = GT_motion[:, :target_idx+1].to(device)

        B, T, M = GT_motion.shape
        GT_local_ortho6ds, GT_root_pos = torch.split(GT_motion, [M-3, 3], dim=-1)
        GT_local_ortho6ds = GT_local_ortho6ds.reshape(B, T, skeleton.num_joints, 6)
        _, GT_global_positions = trf.t_ortho6d.fk(GT_local_ortho6ds, GT_root_pos, skeleton)

        GT_foot_vel = GT_global_positions[:, 1:, contact_idx] - GT_global_positions[:, :-1, contact_idx]
        GT_foot_vel = torch.sum(GT_foot_vel ** 2, dim=-1) # (B, t-1, 4)
        GT_foot_vel = torch.cat([GT_foot_vel[:, 0:1], GT_foot_vel], dim=1) # (B, t, 4)
        GT_contact  = (GT_foot_vel < config.contact_threshold).float() # (B, t, 4)

        # apply z-score normalization
        GT_motion = (GT_motion - mean) / std

        #############################
        # train discriminator
        #############################
        if train:
            model_d_long.train()
            model_d_short.train()
        else:
            model_d_long.eval()
            model_d_short.eval()
            
        model_g.eval()
        with torch.no_grad():
            fake_motion, _ = get_transition(model_g, GT_motion, GT_contact, config.context_frames)
            fake_motion = fake_motion.detach()

        # discriminator loss
        loss_d_long = loss.disc_loss(model_d_long(GT_motion), model_d_long(fake_motion))
        loss_d_short = loss.disc_loss(model_d_short(GT_motion), model_d_short(fake_motion))

        if train:
            optim_d_long.zero_grad()
            optim_d_short.zero_grad()

            loss_d_long.backward()
            loss_d_short.backward()

            optim_d_long.step()
            optim_d_short.step()

        #############################
        # train generator
        #############################
        if train:
            model_g.train()
        else:
            model_g.eval()
            
        model_d_long.eval()
        model_d_short.eval()

        fake_motion, pred_contact = get_transition(model_g, GT_motion, GT_contact, config.context_frames)

        # generator loss
        loss_g_long = loss.gen_loss(model_d_long(fake_motion))
        loss_g_short = loss.gen_loss(model_d_short(fake_motion))

        # denormalize
        fake_motion = fake_motion * std + mean

        # predicted motion data
        pred_local_ortho6ds, pred_root_pos = torch.split(fake_motion, [M-3, 3], dim=-1)
        pred_local_ortho6ds = pred_local_ortho6ds.reshape(B, T, skeleton.num_joints, 6)
        _, pred_global_positions = trf.t_ortho6d.fk(pred_local_ortho6ds, pred_root_pos, skeleton)

        # loss
        loss_rot     = F.l1_loss(pred_local_ortho6ds, GT_local_ortho6ds)
        loss_pos     = F.l1_loss(pred_global_positions, GT_global_positions)
        loss_root    = F.l1_loss(pred_root_pos, GT_root_pos)
        loss_contact = F.l1_loss(pred_contact, GT_contact)

        loss_total = config.weight_rot * loss_rot + \
                        config.weight_pos * loss_pos + \
                        config.weight_root * loss_root + \
                        config.weight_contact * loss_contact + \
                        config.weight_adv * loss_g_long + \
                        config.weight_adv * loss_g_short

        if train:
            optim_g.zero_grad()
            loss_total.backward()
            optim_g.step()

        loss_dict["rot"] += loss_rot.item()
        loss_dict["pos"] += loss_pos.item()
        loss_dict["root"] += loss_root.item()
        loss_dict["contact"] += loss_contact.item()
        loss_dict["disc"] += (loss_d_long.item() + loss_d_short.item()) / 2
        loss_dict["gen"] += (loss_g_long.item() + loss_g_short.item()) / 2
        loss_dict["total"] += loss_total.item()

    # main loop
    start_time = time.perf_counter()
    for epoch in range(init_epoch+1, config.epochs+1):
        # train
        model_g.train()
        model_d_long.train()
        model_d_short.train()
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False)):
            train_iter(batch, train=True)   

        # log training
        elapsed = time.perf_counter() - start_time
        utils.write_log(writer, loss_dict, len(dataloader), epoch, elapsed=elapsed, train=True)
        utils.reset_log(loss_dict)

        # validation
        if epoch % config.val_interval == 0:
            model_g.eval()
            model_d_long.eval()
            model_d_short.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_dataloader, desc=f"Validation", leave=False)):
                    train_iter(batch, train=False)

                # log validation
                utils.write_log(writer, loss_dict, len(val_dataloader), epoch, train=False)
                utils.reset_log(loss_dict)

        # save checkpoint - every 10 epochs
        if epoch % config.save_interval == 0:
            utils.save_ckpt(model_g, optim_g, epoch, config)

    # save checkpoint - last epoch
    utils.save_ckpt(model_g, optim_g, epoch, config)
    print(f"Training finished in {(time.perf_counter() - start_time) / 60:.2f} min")