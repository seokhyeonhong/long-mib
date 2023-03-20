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
from vis.visapp import DetailMotionApp
from model.twostage import ContextTransformer, DetailTransformer

def get_moiton(skeleton, local_R, root_p):
    B, T, *_ = local_R.shape
    local_R = local_R.reshape(B*T, -1, 3, 3)
    root_p = root_p.reshape(B*T, 3)
    motion = Motion.from_torch(skeleton, local_R, root_p)
    return motion

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/detail.json")
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
    ctx_model = ContextTransformer(dataset.shape[-1], Config.load("configs/context.json")).to(device)
    testutil.load_model(ctx_model, Config.load("configs/context.json"))
    ctx_model.eval()

    det_model = DetailTransformer(dataset.shape[-1], config).to(device)
    testutil.load_model(det_model, config)
    det_model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            B, T, D = GT_motion.shape

            T = config.context_frames + config.max_transition + 1
            GT_motion = GT_motion[:, :T, :]
            GT_motion = GT_motion.to(device)

            # GT motion
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            # ContextTransformer
            batch = (GT_motion - motion_mean) / motion_std
            context_motion, mask = ctx_model.forward(batch, ratio_constrained=0, prob_constrained=0)
            context_motion = mask * batch + (1 - mask) * context_motion

            # DetailTransformer
            detail_motion, _ = det_model.forward(context_motion, mask)
            detail_motion = mask * batch + (1 - mask) * detail_motion

            # denormalize
            context_motion = context_motion * motion_std + motion_mean
            detail_motion = detail_motion * motion_std + motion_mean

            # motion objects
            context_local_R6, context_root_p = torch.split(context_motion, [D-3, 3], dim=-1)
            context_local_R = rotation.R6_to_R(context_local_R6.reshape(B, T, -1, 6))

            detail_local_R6, detail_root_p = torch.split(detail_motion, [D-3, 3], dim=-1)
            detail_local_R = rotation.R6_to_R(detail_local_R6.reshape(B, T, -1, 6))

            # animation
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, 3)
            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)

            context_local_R = context_local_R.reshape(B*T, -1, 3, 3)
            context_root_p = context_root_p.reshape(B*T, 3)
            context_motion = Motion.from_torch(skeleton, context_local_R, context_root_p)

            detail_local_R = detail_local_R.reshape(B*T, -1, 3, 3)
            detail_root_p = detail_root_p.reshape(B*T, 3)
            detail_motion = Motion.from_torch(skeleton, detail_local_R, detail_root_p)

            app_manager = AppManager()
            app = DetailMotionApp(GT_motion, context_motion, detail_motion, ybot.model(), T)
            app_manager.run(app)