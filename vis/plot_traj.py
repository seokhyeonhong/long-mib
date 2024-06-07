import sys
sys.path.append(".")
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import utils
from utils.dataset import MotionDataset

def main(train=True):
    lafan_cfg = utils.load_config(f"config/lafan1/default.yaml")
    human36m_cfg = utils.load_config(f"config/human36m/default.yaml")
    style_cfg = utils.load_config(f"config/100style/default.yaml")
    utils.seed()

    # dataset
    lafan_dataset = MotionDataset(train=train, config=lafan_cfg)
    human36m_dataset = MotionDataset(train=train, config=human36m_cfg)
    style_dataset = MotionDataset(train=train, config=style_cfg)

    lafan_dataloader = DataLoader(lafan_dataset, batch_size=lafan_cfg.batch_size, shuffle=train)
    human36m_dataloader = DataLoader(human36m_dataset, batch_size=human36m_cfg.batch_size, shuffle=train)
    style_dataloader = DataLoader(style_dataset, batch_size=style_cfg.batch_size, shuffle=train)

    # iterate and plot with same color
    fig, ax = plt.subplots()
    for batch in tqdm(lafan_dataloader):
        motion = batch["motion"]
        pos = motion[:, :, (-3, -1)].numpy()
        for i in range(pos.shape[0]):
            ax.plot(pos[i, :, 0], pos[i, :, 1], color="blue", alpha=0.1)

    for batch in tqdm(style_dataloader):
        motion = batch["motion"]
        pos = motion[:, :, (-3, -1)].numpy()
        for i in range(pos.shape[0]):
            ax.plot(pos[i, :, 0], pos[i, :, 1], color="green", alpha=0.1)
            
    for batch in tqdm(human36m_dataloader):
        motion = batch["motion"]
        pos = motion[:, :, (-3, -1)].numpy()
        for i in range(pos.shape[0]):
            ax.plot(pos[i, :, 0], pos[i, :, 1], color="red", alpha=0.1)


    # origin
    ax.plot([0], [0], color="yellow", marker="o")
    
    # save
    plt.savefig("traj.png")
    # plt.show

    

if __name__ == "__main__":
    main(True)