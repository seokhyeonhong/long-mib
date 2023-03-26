import sys
sys.path.append(".")

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
import glm
from tqdm import tqdm

from pymovis.utils import util, torchconst
from pymovis.motion import Motion, FBX
from pymovis.ops import rotation
from pymovis.vis import AppManager, MotionApp, Render, YBOT_FBX_DICT

from utility import testutil
from utility.config import Config
from utility.dataset import KeyframeDataset, MotionDataset
from vis.visapp import ContextMotionApp
from model.ours import KeyframeTransformer
from model.twostage import ContextTransformer

class KeyframeApp(MotionApp):
    def __init__(self, GT_motion, pred_motion, model, keyframes):
        super().__init__(GT_motion, model, YBOT_FBX_DICT)

        self.GT_motion = GT_motion
        self.pred_motion = pred_motion

        self.keyframes = keyframes

        self.GT_model = model
        self.GT_model.set_source_skeleton(self.GT_motion.skeleton, YBOT_FBX_DICT)
        self.pred_model = copy.deepcopy(model)
        self.pred_model.set_source_skeleton(self.pred_motion.skeleton, YBOT_FBX_DICT)
        self.pred_model.meshes[0].materials[0].albedo = glm.vec3(0.5, 0.5, 0.5)

        self.key_model = copy.deepcopy(model)
        self.key_model.set_source_skeleton(self.pred_motion.skeleton, YBOT_FBX_DICT)
        self.key_model.meshes[0].materials[0].albedo = glm.vec3(0.5, 0.5, 0.5)

    def render(self):
        super().render(render_model=False)

        self.GT_model.set_pose_by_source(self.GT_motion.poses[self.frame])
        Render.model(self.GT_model).draw()

        # self.pred_model.set_pose_by_source(self.pred_motion.poses[self.frame])
        # Render.model(self.pred_model).draw()

        # nearest but smaller keyframe
        keyframe = min([k for k in self.keyframes if k > self.frame], default=0)
        self.key_model.set_pose_by_source(self.pred_motion.poses[keyframe])
        Render.model(self.key_model).set_all_alphas(0.5).draw()

    def render_text(self):
        super().render_text()

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/keyframe.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = KeyframeDataset(train=False, config=config)
    skeleton   = dataset.skeleton
    v_forward  = torch.from_numpy(skeleton.v_forward).to(device)

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = KeyframeTransformer(dataset.shape[-1], config).to(device) # exclude trajectory
    testutil.load_model(model, config)
    model.eval()

    ctx = ContextTransformer(dataset.shape[-1] - 6, Config.load("configs/context.json")).to(device) # exclude trajectory and prob
    testutil.load_model(ctx, Config.load("configs/context.json"))
    ctx.eval()

    temp_dataset = MotionDataset(train=False, config=Config.load("configs/context.json"))
    temp_mean, temp_std = temp_dataset.statistics(dim=(0, 1))
    temp_mean, temp_std = temp_mean.to(device), temp_std.to(device)
    temp_mean, temp_std = temp_mean[..., :-5], temp_std[..., :-5] # exclude trajectory

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            B, T, D = GT_motion.shape

            # GT motion
            GT_motion = GT_motion.to(device)
            GT_local_R6, GT_root_p, GT_kf_prob, GT_traj = torch.split(GT_motion, [D-9, 3, 1, 5], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            # forward
            batch = (GT_motion - motion_mean) / motion_std
            pred_motion, _ = model.forward(batch)
            pred_motion = pred_motion * motion_std[..., :-5] + motion_mean[..., :-5] # exclude traj features

            pred_local_R6, pred_root_p, pred_kf_prob = torch.split(pred_motion, [D-9, 3, 1], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))

            # for each batch
            results = []
            keyframes = []
            for b in tqdm(range(B)):
                top_keyframes = torch.topk(pred_kf_prob[b:b+1, config.context_frames+1:-1], 10, dim=1).indices + config.context_frames + 1
                top_keyframes = top_keyframes.reshape(-1).sort().values
                for k in top_keyframes:
                    keyframes.append(k.item() + b * T)
                keyframes.append((b+1)*T-1)

                # copy pseudo-GT motion
                ctx_local_R6 = pred_local_R6[b:b+1].clone()
                ctx_root_p = pred_root_p[b:b+1].clone()
                # ctx_local_R6 = GT_local_R6[b:b+1].clone()
                # ctx_root_p = GT_root_p[b:b+1].clone()
                ctx_batch = torch.cat([ctx_local_R6, ctx_root_p], dim=-1)
                ctx_batch = (ctx_batch - temp_mean) / temp_std

                # recurrent prediction
                frame_from, frame_to = 0, top_keyframes[0]
                for f in range(11):
                    # input batch
                    input_batch = ctx_batch[:, frame_from:frame_to+1]
                    input_batch = input_batch * temp_std + temp_mean
                    local_R6, root_p = torch.split(input_batch, [D-9, 3], dim=-1)

                    # delta rotation to fit at the last context frame
                    local_R = rotation.R6_to_R(local_R6.reshape(local_R6.shape[0], local_R6.shape[1], -1, 6))
                    root_R = local_R[:, :, 0]
                    forward = torch.matmul(root_R[:, config.context_frames-1], v_forward)
                    forward = F.normalize(forward * torchconst.XZ(device), dim=-1)
                    up = torchconst.UP(device).unsqueeze(0)
                    delta_R = torch.stack([torch.cross(up, forward), up, forward], dim=-2).unsqueeze(1)
                    root_R = torch.matmul(delta_R, root_R)
                    root_R6 = rotation.R_to_R6(root_R)

                    # delta position to fit at the last context frame
                    delta_p = root_p[:, config.context_frames-1:config.context_frames] * torchconst.XZ(device)
                    root_p = torch.matmul(delta_R, (root_p - delta_p).unsqueeze(-1)).squeeze(-1)

                    # update
                    input_batch[:, :, :6] = root_R6
                    input_batch[:, :, -3:] = root_p
                    input_batch = (input_batch - temp_mean) / temp_std

                    # forward and denormalize
                    ctx_pred, ctx_mask = ctx.forward(input_batch, ratio_constrained=0.0, prob_constrained=0.0)
                    ctx_pred = ctx_mask * input_batch + (1 - ctx_mask) * ctx_pred
                    ctx_pred = ctx_pred * temp_std + temp_mean

                    # re-update
                    local_R = rotation.R6_to_R(local_R6.reshape(local_R6.shape[0], local_R6.shape[1], -1, 6))
                    root_R = local_R[:, :, 0]
                    root_R = torch.matmul(delta_R.transpose(-1, -2), root_R)
                    root_R6 = rotation.R_to_R6(root_R)
                    root_p = torch.matmul(delta_R.transpose(-1, -2), root_p.unsqueeze(-1)).squeeze(-1) + delta_p
                    ctx_pred[:, :, :6] = root_R6
                    ctx_pred[:, :, -3:] = root_p

                    # re-normalize
                    ctx_pred = (ctx_pred - temp_mean) / temp_std
                    ctx_batch[:, frame_from:frame_to+1] = ctx_pred

                    frame_from = frame_to - config.context_frames + 1
                    frame_to = top_keyframes[f+1] if f < 9 else T
                
                results.append(ctx_batch)
            
            # concatenate
            pred_motion = torch.cat(results, dim=0)
            pred_motion = pred_motion * temp_std + temp_mean
            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-9, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))

            # animation
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)
            pred_local_R = pred_local_R.reshape(B*T, -1, 3, 3)
            pred_root_p = pred_root_p.reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)

            app_manager = AppManager()
            GT_kf_prob = GT_kf_prob.reshape(B*T, -1)
            pred_kf_prob = pred_kf_prob.reshape(B*T, -1)
            app = KeyframeApp(GT_motion, pred_motion, ybot.model(), keyframes)
            app_manager.run(app)