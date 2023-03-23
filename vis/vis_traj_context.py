import sys
sys.path.append(".")

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
from tqdm import tqdm

from pymovis.utils import util, torchconst
from pymovis.motion import Motion, FBX
from pymovis.vis import AppManager
from pymovis.ops import rotation, motionops

from utility import testutil
from utility.config import Config
from utility.dataset import KeyframeDataset
from vis.visapp import ContextMotionApp
from model.twostage import TrajContextTransformer

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/traj_context.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = KeyframeDataset(train=True, config=config)
    skeleton   = dataset.skeleton
    v_forward  = torch.from_numpy(config.v_forward).to(device)

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = TrajContextTransformer(dataset.shape[-1], config).to(device)
    testutil.load_model(model, config)
    model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            B, T, D = GT_motion.shape

            # T = config.context_frames + 121
            T = config.context_frames + config.max_transition + 1
            GT_motion = GT_motion[:, :T, :]
            GT_motion = GT_motion.to(device)

            # GT motion
            GT_motion = GT_motion.to(device)
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R6 = GT_local_R6.reshape(B, T, -1, 6)
            
            GT_local_R = rotation.R6_to_R(GT_local_R6)
            GT_root_R = GT_local_R[:, :, 0]
            GT_forward = F.normalize(torch.matmul(GT_root_R, v_forward)  * torchconst.XZ(device), dim=-1)

            _, GT_global_p = motionops.R_fk(GT_local_R, GT_root_p, skeleton)
            GT_root_xz = GT_root_p[..., (0, 2)]

            # CoarseNet
            batch = (GT_motion - motion_mean) / motion_std
            batch = torch.cat([batch, GT_forward, GT_root_xz], dim=-1)
            pred_motion, _ = model.forward(batch, ratio_constrained=0, prob_constrained=0)
            # pred_motion = mask * batch + (1 - mask) * pred_motion
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