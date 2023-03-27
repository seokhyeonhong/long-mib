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
from pymovis.ops import rotation

from utility import testutil
from utility.config import Config
from utility.dataset import MotionDataset
from vis.visapp import ContextMotionApp
from model.twostage import ContextTransformer, DetailTransformer

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/context.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=config)
    skeleton   = dataset.skeleton
    v_forward  = torch.from_numpy(config.v_forward).to(device)

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    motion_mean, motion_std = motion_mean[..., :-5], motion_std[..., :-5] # exclude trajectory
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = ContextTransformer(dataset.shape[-1] - 5, config).to(device) # exclude trajectory
    testutil.load_model(model, config)
    model.eval()

    det_model = DetailTransformer(dataset.shape[-1] - 5, Config.load("configs/detail.json")).to(device) # exclude trajectory
    testutil.load_model(det_model, Config.load("configs/detail.json"))
    det_model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            # T = config.context_frames + 121
            # T = config.context_frames + config.max_transition + 1
            GT_motion = GT_motion[:, :, :-5] # exclude trajectory
            B, T, D = GT_motion.shape

            GT_motion = GT_motion.to(device)

            # GT motion
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))
            batch = (GT_motion - motion_mean) / motion_std

            # ContextTransformer
            keyframes = [40, 70, 100, 130, 160, 190]
            frame_from, frame_to = 0, keyframes[0]
            for f in range(len(keyframes)+1):
                # input batch
                input_batch = batch[:, frame_from:frame_to+1]
                input_batch = input_batch * motion_std + motion_mean
                local_R6, root_p = torch.split(input_batch, [D-3, 3], dim=-1)

                # delta rotation to fit at the last context frame
                local_R = rotation.R6_to_R(local_R6.reshape(local_R6.shape[0], local_R6.shape[1], -1, 6))
                root_R = local_R[:, :, 0]
                forward = torch.matmul(root_R[:, config.context_frames-1], v_forward)
                forward = F.normalize(forward * torchconst.XZ(device), dim=-1)
                up = torchconst.UP(device).unsqueeze(0).repeat(B, 1)
                delta_R = torch.stack([torch.cross(up, forward), up, forward], dim=-2).unsqueeze(1)
                root_R = torch.matmul(delta_R, root_R)
                root_R6 = rotation.R_to_R6(root_R)

                # delta position to fit at the last context frame
                delta_p = root_p[:, config.context_frames-1:config.context_frames] * torchconst.XZ(device)
                root_p = torch.matmul(delta_R, (root_p - delta_p).unsqueeze(-1)).squeeze(-1)

                # update
                input_batch[:, :, :6] = root_R6
                input_batch[:, :, -3:] = root_p
                input_batch = (input_batch - motion_mean) / motion_std

                # forward and denormalize
                ctx_pred, ctx_mask = model.forward(input_batch, ratio_constrained=0.0, prob_constrained=0.0)
                ctx_pred = ctx_mask * input_batch + (1 - ctx_mask) * ctx_pred
                # ctx_pred, _ = det_model.forward(ctx_pred, ctx_mask)
                # ctx_pred = ctx_mask * input_batch + (1 - ctx_mask) * ctx_pred
                ctx_pred = ctx_pred * motion_std + motion_mean

                # re-update
                local_R = rotation.R6_to_R(local_R6.reshape(local_R6.shape[0], local_R6.shape[1], -1, 6))
                root_R = local_R[:, :, 0]
                root_R = torch.matmul(delta_R.transpose(-1, -2), root_R)
                root_R6 = rotation.R_to_R6(root_R)
                root_p = torch.matmul(delta_R.transpose(-1, -2), root_p.unsqueeze(-1)).squeeze(-1) + delta_p
                ctx_pred[:, :, :6] = root_R6
                ctx_pred[:, :, -3:] = root_p

                # re-normalize
                ctx_pred = (ctx_pred - motion_mean) / motion_std
                batch[:, frame_from:frame_to+1] = ctx_pred

                frame_from = frame_to - config.context_frames + 1
                frame_to = keyframes[f+1] if f < len(keyframes)-1 else T

            # pred_motion, mask = model.forward(batch, ratio_constrained=0, prob_constrained=0)
            # pred_motion = mask * batch + (1 - mask) * pred_motion
            # pred_motion = pred_motion * motion_std + motion_mean

            pred_motion = batch * motion_std + motion_mean
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