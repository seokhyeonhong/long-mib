import sys
sys.path.append(".")

import os
import time
import random
from tqdm import tqdm
import argparse

from aPyOpenGL import transforms as trf

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import utils, loss, ops
from utils.dataset import MotionDataset
from model.twostage import ContextTransformer
from model.scheduler import NoamScheduler

if __name__ =="__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default="context.yaml")
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

    # model, optimizer, scheduler
    model = ContextTransformer(config, dataset).to(device)
    optimizer = Adam(model.parameters(), lr=0) # lr will be set by scheduler
    scheduler = NoamScheduler(optimizer, config.d_model, config.warmup_iters, lr=config.get("lr", None))
    init_epoch = utils.load_latest_ckpt(model, optimizer, config, scheduler=scheduler)

    # save and log
    os.makedirs(config.save_dir, exist_ok=True)
    utils.write_config(config)
    writer = SummaryWriter(config.save_dir)
    loss_dict = {
        "rot": 0.0,
        "pos": 0.0,
        "smooth": 0.0,
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

        # forward
        GT_motion = (GT_motion - mean) / std
        ctx_out, _ = model.forward(GT_motion, train=train)
        ctx_motion = ctx_out["motion"]

        # restore constrained frames
        pred_motion = GT_motion.clone().detach()
        pred_motion[:, config.context_frames:-1] = ctx_motion[:, config.context_frames:-1]

        # denormalize
        pred_motion = pred_motion * std + mean

        # predicted motion data
        pred_local_ortho6ds, pred_root_pos = torch.split(pred_motion, [M-3, 3], dim=-1)
        pred_local_ortho6ds = pred_local_ortho6ds.reshape(B, T, skeleton.num_joints, 6)
        _, pred_global_positions = trf.t_ortho6d.fk(pred_local_ortho6ds, pred_root_pos, skeleton)

        # loss
        loss_rot    = loss.rot_loss(pred_local_ortho6ds, GT_local_ortho6ds, config.context_frames)
        loss_pos    = loss.pos_loss(pred_global_positions, GT_global_positions, config.context_frames)
        loss_smooth = loss.smooth_loss(pred_global_positions, config.context_frames)
        loss_total  = config.weight_rot * loss_rot + \
                        config.weight_pos * loss_pos + \
                        config.weight_smooth * loss_smooth

        loss_dict["rot"] += loss_rot.item()
        loss_dict["pos"] += loss_pos.item()
        loss_dict["smooth"] += loss_smooth.item()
        loss_dict["total"] += loss_total.item()

        # backward
        if train:
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()
        
    # main loop
    start_time = time.perf_counter()
    for epoch in range(init_epoch+1, config.epochs+1):
        # train
        model.train()
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False)):
            train_iter(batch, train=True)   

        # log training
        elapsed = time.perf_counter() - start_time
        utils.write_log(writer, loss_dict, len(dataloader), epoch, elapsed=elapsed, train=True)
        utils.reset_log(loss_dict)

        # validation
        if epoch % config.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_dataloader, desc=f"Validation", leave=False)):
                    train_iter(batch, train=False)

                # log validation
                utils.write_log(writer, loss_dict, len(val_dataloader), epoch, train=False)
                utils.reset_log(loss_dict)

        # save checkpoint - every 10 epochs
        if epoch % config.save_interval == 0:
            utils.save_ckpt(model, optimizer, epoch, config, scheduler=scheduler)

    # save checkpoint - last epoch
    utils.save_ckpt(model, optimizer, epoch, config, scheduler=scheduler)
    print(f"Training finished in {(time.perf_counter() - start_time) / 60:.2f} min")