import sys
sys.path.append(".")

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
import glm, glfw
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
from model.twostage import ContextTransformer, DetailTransformer

class KeyframeApp(MotionApp):
    def __init__(self, GT_motion, pred_motion, model, keyframes, time_per_motion, traj):
        super().__init__(GT_motion, model, YBOT_FBX_DICT)

        self.GT_motion = GT_motion
        self.pred_motion = pred_motion

        self.keyframes = keyframes
        self.time_per_motion = time_per_motion

        self.GT_model = model
        self.GT_model.set_source_skeleton(self.GT_motion.skeleton, YBOT_FBX_DICT)
        self.pred_model = copy.deepcopy(model)
        self.pred_model.set_source_skeleton(self.pred_motion.skeleton, YBOT_FBX_DICT)
        self.pred_model.meshes[0].materials[0].albedo = glm.vec3(0.5, 0.5, 0.5)

        self.key_model = copy.deepcopy(model)
        self.key_model.set_source_skeleton(self.pred_motion.skeleton, YBOT_FBX_DICT)
        self.key_model.meshes[0].materials[0].albedo = glm.vec3(0.5, 0.5, 0.5)

        self.show_GT = True
        self.show_pred = True

        self.traj = traj
        self.traj_sphere = Render.sphere(0.1).set_albedo(glm.vec3(1, 0, 0))

    def render(self):
        super().render(render_model=False)

        if self.show_GT:
            self.GT_model.set_pose_by_source(self.GT_motion.poses[self.frame])
            Render.model(self.GT_model).draw()

        # predicted pose
        if self.show_pred:
            self.pred_model.set_pose_by_source(self.pred_motion.poses[self.frame])
            Render.model(self.pred_model).draw()

        # target pose per motion
        ith_motion = self.frame // self.time_per_motion
        self.GT_model.set_pose_by_source(self.pred_motion.poses[(ith_motion+1) * self.time_per_motion - 1])
        Render.model(self.GT_model).set_all_alphas(0.5).draw()

        # nearest but smaller keyframe
        keyframe = min([k for k in self.keyframes if k > self.frame], default=0)
        self.key_model.set_pose_by_source(self.pred_motion.poses[keyframe])
        Render.model(self.key_model).set_all_alphas(0.5).draw()

        # trajectory
        for t in range(self.time_per_motion):
            self.traj_sphere.set_position(self.traj[self.time_per_motion*ith_motion + t, 0], 0, self.traj[self.time_per_motion*ith_motion + t, 1]).draw()

    def render_text(self):
        super().render_text()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if key == glfw.KEY_Q and action == glfw.PRESS:
            self.show_GT = not self.show_GT
        if key == glfw.KEY_W and action == glfw.PRESS:
            self.show_pred = not self.show_pred

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

    kf_mean, kf_std = dataset.statistics(dim=(0, 1))
    kf_mean, kf_std = kf_mean.to(device), kf_std.to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = KeyframeTransformer(dataset.shape[-1], config).to(device) # exclude trajectory
    testutil.load_model(model, config)
    model.eval()

    ctx = ContextTransformer(dataset.shape[-1] - 6, Config.load("configs/context_noise.json")).to(device) # exclude trajectory and prob
    testutil.load_model(ctx, Config.load("configs/context_noise.json"))
    ctx.eval()

    det = DetailTransformer(dataset.shape[-1] - 6, Config.load("configs/detail_noise.json")).to(device) # exclude trajectory and prob
    testutil.load_model(det, Config.load("configs/detail_noise.json"))
    det.eval()

    motion_mean, motion_std = MotionDataset(train=False, config=Config.load("configs/context_noise.json")).statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    motion_mean, motion_std = motion_mean[..., :-5], motion_std[..., :-5] # exclude trajectory

    # character
    ybot = FBX("dataset/ybot.fbx")

    # toe_jids = [17, 21]
    # # toe_jids = [9, 13, 17, 21]
    # toe_fids = []
    # for jid in toe_jids:
    #     for i in range(6):
    #         toe_fids.append(6*jid + i)

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            B, T, D = GT_motion.shape

            # GT motion
            GT_motion = GT_motion.to(device)

            # lerp for trajectory
            t = torch.linspace(0, 1, T-config.context_frames+1, device=device).unsqueeze(0).unsqueeze(-1)
            delta = (GT_motion[:, -1, -5:-3] - GT_motion[:, config.context_frames-1, -5:-3])[:, None]
            GT_motion[:, config.context_frames-1:, -5:-3] = delta * t + GT_motion[:, config.context_frames-1:config.context_frames, -5:-3]
# 
            direction = F.normalize(GT_motion[:, -1, -5:-3] - GT_motion[:, config.context_frames-1, -5:-3], dim=-1)
            direction = torch.stack([direction[..., 0], torch.zeros_like(direction[..., 0]), direction[..., 1]], dim=-1)
            GT_motion[:, config.context_frames-1:, -3:] = direction.unsqueeze(1)
# 
            # GT_motion[:, config.context_frames-1:, -5:] = -GT_motion[:, config.context_frames-1:, -5:]
            # GT_motion[:, -1, (D-9, D-7)] = -GT_motion[:, -1, (D-9, D-7)]

            GT_local_R6, GT_root_p, GT_kf_prob, GT_traj = torch.split(GT_motion, [D-9, 3, 1, 5], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            # forward
            batch = (GT_motion - kf_mean) / kf_std
            pred_motion, _ = model.forward(batch, ratio_constrained=0.0, prob_constrained=0.0)
            pred_motion = pred_motion * kf_std[..., :-5] + kf_mean[..., :-5] # exclude traj features
            pred_motion[:, :config.context_frames] = GT_motion[:, :config.context_frames, :-5] # restore context frames
            pred_motion[:, -1] = GT_motion[:, -1, :-5] # restore last frame
            # pred_motion[:, :, toe_fids] = GT_motion[:, :, toe_fids] # restore toe features

            pred_local_R6, pred_root_p, pred_kf_prob = torch.split(pred_motion, [D-9, 3, 1], dim=-1)
            pred_local_R6 = rotation.R_to_R6(rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))).reshape(B, T, -1)

            # for each batch
            results = []
            total_keyframes = []
            for b in tqdm(range(B)):
                # adaptive keyframe selection
                keyframes = []
                transition_start = config.context_frames
                while transition_start < T:
                    transition_end = min(transition_start + config.fps, T-1)
                    if transition_end == T-1:
                        keyframes.append(T-1)
                        break

                    top_keyframe = torch.topk(pred_kf_prob[b:b+1, transition_start+5:transition_end+1], 1, dim=1).indices + transition_start+5
                    top_keyframe = top_keyframe.item()
                    keyframes.append(top_keyframe)
                    transition_start = top_keyframe + 1
                
                for kf in keyframes:
                    total_keyframes.append(b*T + kf)

                # motion batch from keyframe prediction
                local_R6 = pred_local_R6[b:b+1].clone()
                root_p = pred_root_p[b:b+1].clone()
                # local_R6 = GT_local_R6[b:b+1].clone()
                # root_p = GT_root_p[b:b+1].clone()
                motion_batch = torch.cat([local_R6, root_p], dim=-1)
                motion_batch = (motion_batch - motion_mean) / motion_std

                # recurrent prediction
                frame_start, frame_end = 0, keyframes[0]
                for f in range(len(keyframes)+1):
                    # input batch
                    input_batch = motion_batch[:, frame_start:frame_end+1]
                    input_batch = input_batch * motion_std + motion_mean
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
                    input_batch = (input_batch - motion_mean) / motion_std

                    # forward and denormalize
                    pred_motion, mask = ctx.forward(input_batch, ratio_constrained=0.0, prob_constrained=0.0)
                    pred_motion = mask * input_batch + (1 - mask) * pred_motion
                    pred_motion, _ = det.forward(pred_motion, mask)
                    pred_motion = mask * input_batch + (1 - mask) * pred_motion
                    pred_motion = pred_motion * motion_std + motion_mean

                    # re-update
                    local_R = rotation.R6_to_R(local_R6.reshape(local_R6.shape[0], local_R6.shape[1], -1, 6))
                    root_R = local_R[:, :, 0]
                    root_R = torch.matmul(delta_R.transpose(-1, -2), root_R)
                    root_R6 = rotation.R_to_R6(root_R)
                    root_p = torch.matmul(delta_R.transpose(-1, -2), root_p.unsqueeze(-1)).squeeze(-1) + delta_p
                    pred_motion[:, :, :6] = root_R6
                    pred_motion[:, :, -3:] = root_p

                    # re-normalize
                    pred_motion = (pred_motion - motion_mean) / motion_std
                    motion_batch[:, frame_start:frame_end+1] = pred_motion

                    frame_start = frame_end - config.context_frames + 1
                    frame_end = keyframes[f+1] if f < len(keyframes)-1 else T
                
                results.append(motion_batch.clone())
            
            # concatenate
            pred_motion = torch.cat(results, dim=0)
            pred_motion = pred_motion * motion_std + motion_mean
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
            app = KeyframeApp(GT_motion, pred_motion, ybot.model(), total_keyframes, T, GT_traj.reshape(B*T, -1).cpu().numpy())
            app_manager.run(app)