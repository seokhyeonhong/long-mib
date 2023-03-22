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
from vis.visapp import SparseMotionApp
from model.twostage import ContextTransformer, DetailTransformer
from model.ours import SparseTransformer

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/sparse.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=config)
    skeleton   = dataset.skeleton

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    v_forward = torch.from_numpy(config.v_forward).to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    sparse_model = SparseTransformer(dataset.shape[-1], config).to(device)
    testutil.load_model(sparse_model, config)
    sparse_model.eval()

    context_model = ContextTransformer(dataset.shape[-1], Config.load("configs/context.json")).to(device)
    testutil.load_model(context_model, Config.load("configs/context.json"))
    context_model.eval()

    # detail_model = DetailTransformer(dataset.shape[-1], Config.load("configs/detail.json")).to(device)
    # testutil.load_model(detail_model, Config.load("configs/detail.json"))
    # detail_model.eval()

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    config.max_transition = 90
    sparse_frames = torch.arange(config.max_transition // config.fps) * config.fps
    sparse_frames += (config.context_frames-1) + config.fps
    sparse_frames = torch.cat([torch.arange(config.context_frames), sparse_frames])
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            B, T, D = GT_motion.shape

            T = config.context_frames + config.max_transition
            GT_motion = GT_motion[:, :T, :]
            GT_motion = GT_motion.to(device)

            # GT motion
            GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            # SparseTransformer
            batch = (GT_motion - motion_mean) / motion_std

            keyframe_batch = batch[:, sparse_frames]
            pred_keyframe, mask = sparse_model.forward(keyframe_batch, sparse_frames)
            pred_keyframe = mask * keyframe_batch + (1 - mask) * pred_keyframe
            batch[:, sparse_frames] = pred_keyframe
            
            # ContextTransformer
            pred_motion = batch.clone()
            refine_frames = torch.cat([torch.arange(config.context_frames), torch.arange(config.context_frames+config.fps-1, config.context_frames+config.fps)])
            for i in range(config.max_transition // config.fps):
                # 원본 모션 데이터로 denormalize
                input_batch = pred_motion[:, refine_frames[0]:refine_frames[-1]+1]
                input_batch = input_batch * motion_std + motion_mean
                local_R6, root_p = torch.split(input_batch, [D-3, 3], dim=-1)

                # context frame 마지막에 forward 맞출 delta
                local_R = rotation.R6_to_R(local_R6.reshape(local_R6.shape[0], local_R6.shape[1], -1, 6))
                root_R = local_R[:, :, 0]
                forward = torch.matmul(root_R[:, config.context_frames-1], v_forward)
                forward = F.normalize(forward * torchconst.XZ(device), dim=-1)
                up = torchconst.Y(device).unsqueeze(0).repeat(B, 1)
                delta_R = torch.stack([torch.cross(up, forward), up, forward], dim=-2).unsqueeze(1)
                root_R = torch.matmul(delta_R, root_R)
                root_R6 = rotation.R_to_R6(root_R)

                # context frame 마지막에 root position 맞출 delta
                delta_p = root_p[:, config.context_frames-1:config.context_frames] * torchconst.XZ(device)
                root_p = torch.matmul(delta_R, (root_p - delta_p).unsqueeze(-1)).squeeze(-1)

                # context frame 마지막에 forward, root position 맞춘 데이터로 다시 normalize
                input_batch[:, :, :6] = root_R6
                input_batch[:, :, -3:] = root_p
                input_batch = (input_batch - motion_mean) / motion_std

                # refine
                pred, mask = context_model.forward(input_batch, ratio_constrained=0.0, prob_constrained=0.0)
                pred = mask * input_batch + (1 - mask) * pred

                # pred, _ = detail_model.forward(pred, mask)
                # pred = mask * input_batch + (1 - mask) * pred

                # refine된 데이터로 다시 denormalize
                pred = pred * motion_std + motion_mean
                local_R6, root_p = torch.split(pred, [D-3, 3], dim=-1)

                # refine된 데이터로 forward, root position 맞춘 데이터로 다시 normalize
                local_R = rotation.R6_to_R(local_R6.reshape(local_R6.shape[0], local_R6.shape[1], -1, 6))
                root_R = local_R[:, :, 0]
                root_R = torch.matmul(delta_R.transpose(-1, -2), root_R)
                root_R6 = rotation.R_to_R6(root_R)
                root_p = torch.matmul(delta_R.transpose(-1, -2), root_p.unsqueeze(-1)).squeeze(-1) + delta_p
                pred[:, :, :6] = root_R6
                pred[:, :, -3:] = root_p

                # refine된 데이터로 다시 normalize
                pred = (pred - motion_mean) / motion_std
                pred_motion[:, refine_frames[0]:refine_frames[-1]+1] = pred
                refine_frames += config.fps

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
            app = SparseMotionApp(GT_motion, pred_motion, ybot.model(), T, sparse_frames)
            app_manager.run(app)