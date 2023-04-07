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
    motion_mean, motion_std = motion_mean[..., :-3], motion_std[..., :-3] # exclude trajectory
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    ctx_config = Config.load("configs/context.json")
    ctx_model = ContextTransformer(dataset.shape[-1] - 3, ctx_config).to(device)
    testutil.load_model(ctx_model, ctx_config)
    ctx_model.eval()

    det_model = DetailTransformer(dataset.shape[-1] - 3, config).to(device)
    testutil.load_model(det_model, config)
    det_model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            T = config.context_frames + config.max_transition + 1
            GT_motion = GT_motion[:, :T, :-3] # exclude trajectory
            B, T, D = GT_motion.shape

            # GT motion
            GT_motion = GT_motion.to(device)
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))
            GT_local_R6 = rotation.R_to_R6(GT_local_R).reshape(B, T, -1)
            GT_motion = torch.cat([GT_local_R6, GT_root_p], dim=-1)

            # ContextTransformer
            batch = (GT_motion - motion_mean) / motion_std
            ctx_motion, mask = ctx_model.forward(batch, ratio_constrained=0, prob_constrained=0)
            ctx_motion = mask * batch + (1 - mask) * ctx_motion

            # DetailTransformer
            det_motion, _ = det_model.forward(ctx_motion, mask)
            det_motion = mask * batch + (1 - mask) * det_motion

            # denormalize
            ctx_motion = ctx_motion * motion_std + motion_mean
            det_motion = det_motion * motion_std + motion_mean
            GT_motion  = batch * motion_std + motion_mean

            # motion objects
            ctx_local_R6, ctx_root_p = torch.split(ctx_motion, [D-3, 3], dim=-1)
            ctx_local_R = rotation.R6_to_R(ctx_local_R6.reshape(B, T, -1, 6))

            det_local_R6, det_root_p = torch.split(det_motion, [D-3, 3], dim=-1)
            det_local_R = rotation.R6_to_R(det_local_R6.reshape(B, T, -1, 6))

            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            # animation
            GT_motion  = get_moiton(skeleton, GT_local_R,  GT_root_p)
            ctx_motion = get_moiton(skeleton, ctx_local_R, ctx_root_p)
            det_motion = get_moiton(skeleton, det_local_R, det_root_p)

            app_manager = AppManager()
            app = DetailMotionApp(GT_motion, ctx_motion, det_motion, ybot.model(), T)
            app_manager.run(app)