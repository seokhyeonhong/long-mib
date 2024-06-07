import sys; sys.path.append(".")

import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import utils
from utils.dataset import PAEDataset

from model.pae import PAE
from model.scheduler import CyclicLRWithRestarts

def forward_step(model, batch, mean, std):
    # normalize
    batch = (batch - mean) / std

    # forward
    pred_y, latent, signal, params = model.forward(batch)

    # loss
    loss = F.mse_loss(pred_y, batch)

    return {
        "loss": loss,
        "pred_y": pred_y,
        "latent": latent,
        "signal": signal,
        "params": params
    }

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = utils.load_config(f"config/{args.dataset}/pae.yaml")
    utils.seed()

    # dataset
    dataset = PAEDataset(train=True, config=config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    mean, std = dataset.motion_statistics(device)

    val_dataset = PAEDataset(train=False, config=config)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # model, optimizer, scheduler
    model = PAE(
        input_channels=dataset.motion_dim,
        phase_channels=config.phase_channels,
        num_frames=dataset.num_frames,
        time_duration=1.).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CyclicLRWithRestarts(optim, batch_size=config.batch_size, epoch_size=len(dataset), restart_period=config.restart_period, t_mult=config.restart_mult, policy="cosine", verbose=True)
    init_epochs = utils.load_latest_ckpt(model, optim, config, scheduler=scheduler)

    # save and log
    os.makedirs(config.save_dir, exist_ok=True)
    utils.write_config(config)
    writer = SummaryWriter(config.save_dir)
    loss_log = 0.0

    # train
    start_time = time.perf_counter()
    for epoch in range(init_epochs+1, config.epochs+1):
        model.train()
        scheduler.step()
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False)):
            train_log = forward_step(model, batch.to(device), mean, std)

            # backward
            optim.zero_grad()
            train_log["loss"].backward()
            optim.step()
            scheduler.batch_step()

            # log
            loss_log += train_log["loss"].item()

        # log - every epoch
        writer.add_scalar("train/loss", loss_log / len(dataloader), epoch)
        tqdm.write(f"Train at Epoch {epoch} | Loss: {loss_log / len(dataloader):.4f} | Time: {(time.perf_counter() - start_time) / 60:.2f} min")
        loss_log = 0.0

        # validation - every epoch
        model.eval()
        with torch.no_grad():
            loss_val_log = 0.0
            phases = {
                "freq": [],
                "amp": [],
                "bias": [],
                "phase": []
            }
            y_lims = {
                "freq": [0, 4],
                "amp": [0, 1],
                "bias": [-1, 1],
                "phase": [-0.5, 0.5]
            }
            for i, batch in enumerate(tqdm(val_dataloader, desc=f"Generating phase at Epoch {epoch} / {config.epochs}", leave=False)):
                # forward
                res = forward_step(model, batch.to(device), mean, std)
                loss_val_log += res["loss"].item()

                phases["freq"].append(res["params"][0].squeeze(-1).detach())
                phases["amp"].append(res["params"][1].squeeze(-1).detach())
                phases["bias"].append(res["params"][2].squeeze(-1).detach())
                phases["phase"].append(res["params"][3].squeeze(-1).detach())
            
            writer.add_scalar("val/loss", loss_val_log / len(val_dataloader), epoch)
            tqdm.write(f"Valid at Epoch {epoch} | Loss: {loss_val_log / len(val_dataloader):.4f} | Time: {(time.perf_counter() - start_time) / 60:.2f} min")
            loss_val_log = 0.0
            
            # save phase figure
            phases_idx = torch.randint(0, len(val_dataset) - 500, size=(1,)).item()
            for key in phases.keys():
                phases[key] = torch.cat(phases[key], dim=0).cpu().numpy()
                phases[key] = phases[key][phases_idx:phases_idx+500]

            for key in phases.keys():
                fig, axis = plt.subplots(config.phase_channels, sharex=True, figsize=(12, 2*config.phase_channels))
                for i in range(config.phase_channels):
                    axis[i].plot(phases[key][:, i])
                    axis[i].set_ylim(y_lims[key])
                plt.tight_layout()

                try:
                    dir_path = os.path.join(config.save_dir, "phase-figures")
                    os.makedirs(dir_path, exist_ok=True)
                    plt.savefig(os.path.join(dir_path, f"{epoch}_{key}.png"))
                except IOError as e:
                    print(e)
                
                plt.close()
            
            # save 2d phase manifold figure
            phase_x = phases["amp"] * np.cos(2 * np.pi * phases["phase"])
            phase_y = phases["amp"] * np.sin(2 * np.pi * phases["phase"])
            fig, axis = plt.subplots(config.phase_channels, figsize=(6, 2*config.phase_channels))
            for i in range(config.phase_channels):
                axis[i].plot(phase_x[:, i])
                axis[i].plot(phase_y[:, i])
                axis[i].set_ylim([-1, 1])
            plt.tight_layout()

            try:
                dir_path = os.path.join(config.save_dir, "phase-figures")
                os.makedirs(dir_path, exist_ok=True)
                plt.savefig(os.path.join(dir_path, f"{epoch}_2d.png"))
            except IOError as e:
                print(e)

            plt.close()

        # save - every epoch
        utils.save_ckpt(model, optim, epoch, config, scheduler=scheduler)
    
    # save
    print(f"Training finished in {(time.perf_counter() - start_time) / 60:.2f} min")
    utils.save_ckpt(model, optim, epoch, config, scheduler=scheduler)