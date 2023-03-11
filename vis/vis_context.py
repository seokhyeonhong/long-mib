import sys
sys.path.append(".")

import os
import torch
from torch.utils.data import DataLoader

import copy
from tqdm import tqdm

from pymovis.utils import util
from pymovis.motion import Motion, FBX
from pymovis.vis import AppManager
from pymovis.ops import rotation, motionops

from utility import testutil
from utility.config import Config
from utility.dataset import MotionDataset
from vis.visapp import VisApp
from model.twostage import ContextTransformer

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/context.json", "2023-03-11-20-30-10")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=config)
    skeleton   = dataset.skeleton

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # model
    print("Initializing model...")
    model = ContextTransformer(dataset.shape[-1], config).to(device)
    testutil.load_model(model, config)
    model.eval()

    # character
    GT_character = FBX("dataset/GT_ybot.fbx")
    pred_character = FBX("dataset/pred_ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            B, T, D = GT_motion.shape

            T = config.context_frames + 30
            # T = config.context_frames + config.max_transition + 1
            GT_motion = GT_motion[:, :T, :]
            GT_motion = GT_motion.to(device)

            # GT motion
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            # CoarseNet
            batch = (GT_motion - motion_mean) / motion_std
            pred_motion, mask = model.forward(batch, ratio_constrained=0, prob_constrained=0)
            pred_motion = pred_motion * motion_std + motion_mean
            pred_motion = mask * GT_motion + (1 - mask) * pred_motion

            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-3, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))

            # animation
            for b in range(B):
                GT_motion = Motion.from_torch(skeleton, GT_local_R[b], GT_root_p[b])
                pred_motion = Motion.from_torch(skeleton, pred_local_R[b], pred_root_p[b])

                app_manager = AppManager()
                app = VisApp(GT_motion, pred_motion, GT_character.model(), pred_character.model())
                app_manager.run(app)