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
from pymovis.ops import rotation

from utility import testutil
from utility.config import Config
from utility.dataset import MotionDataset
from vis.visapp import ContextMotionApp
from model.twostage import ContextTransformer

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/context.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=config)
    skeleton   = dataset.skeleton

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    motion_mean, motion_std = motion_mean[..., :-3], motion_std[..., :-3] # exclude trajectory
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = ContextTransformer(dataset.shape[-1] - 3, config).to(device) # exclude trajectory
    testutil.load_model(model, config)
    model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            # T = config.context_frames + 121
            T = config.context_frames + config.max_transition + 1
            GT_motion = GT_motion[:, :T, :-3] # exclude trajectory
            B, T, D = GT_motion.shape

            GT_motion = GT_motion.to(device)

            # GT motion
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            # CoarseNet
            batch = (GT_motion - motion_mean) / motion_std
            pred_motion, _ = model.forward(batch, ratio_constrained=0, prob_constrained=0)
            pred_motion = pred_motion * motion_std + motion_mean

            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-3, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))

            # animation
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)
            pred_local_R = pred_local_R.reshape(B*T, -1, 3, 3)
            pred_root_p = pred_root_p.reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)

            app_manager = AppManager()
            app = ContextMotionApp(GT_motion, pred_motion, ybot.model(), T)
            app_manager.run(app)