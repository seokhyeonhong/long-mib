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

from utility import utils
from utility.config import Config
from utility.dataset import MotionDataset
from vis.visapp import DetailMotionApp
from model.gan import TwoStageGAN

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/twostagegan.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset     = MotionDataset(train=False, config=config)
    skeleton    = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = TwoStageGAN(dataset.shape[-1], config).to(device)
    utils.load_model(model, config)
    model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            """ 1. Max transition length """
            T = config.context_frames + config.max_transition + 1
            GT_motion = GT_motion[:, :T, :].to(device)
            B, T, D = GT_motion.shape

            """ 2. GT motion data """
            GT_local_R6, GT_global_p, GT_traj = utils.get_motion_and_trajectory(GT_motion, skeleton, v_forward)

            """ 3. Modify trajectory """
            # GT_traj = utils.get_interpolated_trajectory(GT_traj, config.context_frames)

            """ 4. Generate """
            # forward
            GT_batch = (GT_motion - motion_mean) / motion_std
            ctx_motion, det_motion, det_contact = model.generate(GT_batch, GT_traj)

            # predicted motion features
            ctx_motion = ctx_motion * motion_std + motion_mean
            det_motion = det_motion * motion_std + motion_mean

            ctx_local_R6, ctx_global_p, ctx_traj = utils.get_motion_and_trajectory(ctx_motion, skeleton, v_forward)
            det_local_R6, det_global_p, det_traj = utils.get_motion_and_trajectory(det_motion, skeleton, v_forward)

            # animation
            GT_local_R = rotation.R6_to_R(GT_local_R6).reshape(B*T, -1, 3, 3)
            GT_root_p = GT_global_p[:, :, 0, :].reshape(B*T, -1)
            ctx_local_R = rotation.R6_to_R(ctx_local_R6).reshape(B*T, -1, 3, 3)
            ctx_root_p = ctx_global_p[:, :, 0, :].reshape(B*T, -1)
            det_local_R = rotation.R6_to_R(det_local_R6).reshape(B*T, -1, 3, 3)
            det_root_p = det_global_p[:, :, 0, :].reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            ctx_motion = Motion.from_torch(skeleton, ctx_local_R, ctx_root_p)
            det_motion = Motion.from_torch(skeleton, det_local_R, det_root_p)

            app_manager = AppManager()
            app = DetailMotionApp(GT_motion, ctx_motion, det_motion, ybot.model(), T)
            app_manager.run(app)