import sys
sys.path.append(".")

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pickle
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
from model.ours import KeyframeGAN, KeyframeTransformer, InterpolationTransformerGlobal, InterpolationTransformerLocal, RefineGAN

class KeyframeApp(MotionApp):
    def __init__(self, GT_motion, pred_motion, interp_motion, model, keyframes, time_per_motion, traj):
        super().__init__(GT_motion, model, YBOT_FBX_DICT)

        self.GT_motion = GT_motion
        self.pred_motion = pred_motion
        self.interp_motion = interp_motion

        self.keyframes = keyframes
        self.time_per_motion = time_per_motion

        self.GT_model = model
        self.GT_model.set_source_skeleton(self.GT_motion.skeleton, YBOT_FBX_DICT)

        self.pred_model = copy.deepcopy(model)
        self.pred_model.set_source_skeleton(self.pred_motion.skeleton, YBOT_FBX_DICT)
        self.pred_model.meshes[0].materials[0].albedo = glm.vec3(0.5, 0.5, 0.5)

        self.interp_model = copy.deepcopy(model)
        self.interp_model.set_source_skeleton(self.interp_motion.skeleton, YBOT_FBX_DICT)
        self.interp_model.meshes[0].materials[0].albedo = glm.vec3(0.5, 0.5, 0.5)

        self.key_model = copy.deepcopy(model)
        self.key_model.set_source_skeleton(self.pred_motion.skeleton, YBOT_FBX_DICT)
        self.key_model.meshes[0].materials[0].albedo = glm.vec3(0.5, 0.5, 0.5)

        self.show_GT = True
        self.show_pred = True
        self.show_interp = True

        self.traj = traj
        self.traj_sphere = Render.sphere(0.1).set_albedo(glm.vec3(1, 0, 0))

    def render(self):
        super().render(render_model=False)

        if self.show_GT:
            self.GT_model.set_pose_by_source(self.GT_motion.poses[self.frame])
            Render.model(self.GT_model).set_all_alphas(1.0).draw()

        # predicted pose
        if self.show_pred:
            self.pred_model.set_pose_by_source(self.pred_motion.poses[self.frame])
            Render.model(self.pred_model).draw()
        
        # interpolated pose
        if self.show_interp:
            self.interp_model.set_pose_by_source(self.interp_motion.poses[self.frame])
            Render.model(self.interp_model).draw()

        # target pose per motion
        ith_motion = self.frame // self.time_per_motion
        self.GT_model.set_pose_by_source(self.pred_motion.poses[(ith_motion+1) * self.time_per_motion - 1])
        Render.model(self.GT_model).set_all_alphas(0.5).draw()
        self.GT_model.set_pose_by_source(self.pred_motion.poses[(ith_motion) * self.time_per_motion])
        Render.model(self.GT_model).set_all_alphas(0.5).draw()
        self.GT_model.set_pose_by_source(self.pred_motion.poses[(ith_motion) * self.time_per_motion + 5])
        Render.model(self.GT_model).set_all_alphas(0.5).draw()
        self.GT_model.set_pose_by_source(self.pred_motion.poses[(ith_motion) * self.time_per_motion + 9])
        Render.model(self.GT_model).set_all_alphas(0.5).draw()

        # nearest but smaller keyframe
        # keyframe = min([k for k in self.keyframes if k > self.frame], default=0)
        keyframes = [k for k in self.keyframes if ith_motion*self.time_per_motion < k < (ith_motion+1) * self.time_per_motion]
        for kf in keyframes:
            self.key_model.set_pose_by_source(self.pred_motion.poses[kf])
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
        if key == glfw.KEY_E and action == glfw.PRESS:
            self.show_interp = not self.show_interp

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/keyframe_gan.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = KeyframeDataset(train=False, config=config)
    skeleton   = dataset.skeleton
    v_forward  = torch.from_numpy(skeleton.v_forward).to(device)

    kf_mean, kf_std = dataset.statistics(dim=(0, 1))
    kf_mean, kf_std = kf_mean.to(device), kf_std.to(device)

    # exclude score from statistics
    D = dataset.shape[-1]
    mean_motion, _, mean_traj = torch.split(kf_mean, [D-4, 1, 3], dim=-1)
    kf_mean = torch.cat([mean_motion, mean_traj], dim=-1)

    std_motion, _, std_traj = torch.split(kf_std, [D-4, 1, 3], dim=-1)
    kf_std = torch.cat([std_motion, std_traj], dim=-1)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model
    print("Initializing model...")
    model = KeyframeGAN(dataset.shape[-1] - 4, config).to(device) # exclude trajectory and keyframe score
    testutil.load_model(model, config)
    model.eval()

    interp_config = Config.load("configs/interp_local.json")
    interp = InterpolationTransformerLocal(dataset.shape[-1] - 1, interp_config).to(device) # exclude prob
    testutil.load_model(interp, interp_config)
    interp.eval()

    motion_mean, motion_std = MotionDataset(train=False, config=interp_config).statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_motion in tqdm(dataloader):
            # GT_motion = GT_motion[:, :41]
            B, T, D = GT_motion.shape

            # GT motion
            GT_motion = GT_motion.to(device)

            GT_local_R6, GT_root_p, GT_kf_prob, GT_traj = torch.split(GT_motion, [D-7, 3, 1, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))
            
            # traj interpolation
            traj_from = GT_traj[:, config.context_frames-1, -3:].unsqueeze(1)
            traj_to   = GT_traj[:, -1, -3:].unsqueeze(1)
            t = torch.linspace(0, 1, T - config.context_frames + 1)[:, None].to(device)
            GT_traj[:, config.context_frames-1:, -3:] = traj_from + (traj_to - traj_from) * t
            GT_traj[:, config.context_frames:, -3] = torch.linspace(0, torch.pi, T - config.context_frames)[None, :].to(device)
            GT_traj[:, config.context_frames:, -2] = torch.sin(GT_traj[:, config.context_frames:, -3])
            GT_traj[:, config.context_frames:, -1] = GT_traj[:, config.context_frames-1, -1].unsqueeze(1).clone()

            GT_root_p[:, -1, (0, 2)] = GT_traj[:, -1, (-3, -2)]
            GT_local_R[:, -1, 0] = GT_local_R[:, config.context_frames-1, 0]

            # forward
            GT_local_R6 = rotation.R_to_R6(GT_local_R).reshape(B, T, -1)
            GT_motion = torch.cat([GT_local_R6, GT_root_p, GT_traj], dim=-1)
            batch = (GT_motion - kf_mean) / kf_std
            pred_motion, pred_kf_score = model.generate(batch)
            pred_motion = pred_motion * kf_std[..., :-3] + kf_mean[..., :-3] # exclude traj features
            pred_motion[:, :config.context_frames] = GT_motion[:, :config.context_frames, :-3] # restore context frames
            pred_motion[:, -1] = GT_motion[:, -1, :-3] # restore last frame

            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
            pred_local_R6 = rotation.R_to_R6(rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))).reshape(B, T, -1)

            # for each batch
            results = []
            interps = []
            total_keyframes = []
            for b in tqdm(range(B)):
                # adaptive keyframe selection
                keyframes = [config.context_frames-1]
                transition_start = config.context_frames
                while transition_start < T:
                    transition_end = min(transition_start + 30, T-1)
                    if transition_end == T-1:
                        keyframes.append(T-1)
                        break

                    top_keyframe = torch.topk(pred_kf_score[b:b+1, transition_start+5:transition_end+1], 1, dim=1).indices + transition_start+5
                    top_keyframe = top_keyframe.item()
                    keyframes.append(top_keyframe)
                    transition_start = top_keyframe + 1
                
                for kf in keyframes:
                    total_keyframes.append(b*T + kf)

                # motion batch from keyframe prediction
                # local_R6 = GT_local_R6[b:b+1].clone()
                # root_p = GT_root_p[b:b+1].clone()
                local_R6 = pred_local_R6[b:b+1].clone()
                root_p = pred_root_p[b:b+1].clone()
                local_R = rotation.R6_to_R(local_R6.reshape(local_R6.shape[0], local_R6.shape[1], -1, 6))
                motion_batch = torch.cat([local_R6, root_p], dim=-1)

                # interpolate
                motion_batch = interp.get_interpolated_motion(local_R, root_p, keyframes)
                motion_batch = torch.cat([motion_batch, GT_traj[b:b+1]], dim=-1)

                # refine
                motion_batch = (motion_batch - motion_mean) / motion_std
                pred = interp.forward(motion_batch, keyframes)
                # pred = interp.generate(motion_batch, keyframes)
                # pred = interp.sample(motion_batch, keyframes)
                
                results.append(pred.clone())
                interps.append(motion_batch[..., :-3].clone())
            
            # concatenate
            pred_motion = torch.cat(results, dim=0)
            pred_motion = pred_motion * motion_std[:-3] + motion_mean[:-3]
            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))

            interp_motion = torch.cat(interps, dim=0)
            interp_motion = interp_motion * motion_std[:-3] + motion_mean[:-3]
            interp_local_R6, interp_root_p = torch.split(interp_motion, [D-7, 3], dim=-1)
            interp_local_R = rotation.R6_to_R(interp_local_R6.reshape(B, T, -1, 6))

            # animation
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)
            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            
            pred_local_R = pred_local_R.reshape(B*T, -1, 3, 3)
            pred_root_p = pred_root_p.reshape(B*T, -1)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)

            interp_local_R = interp_local_R.reshape(B*T, -1, 3, 3)
            interp_root_p = interp_root_p.reshape(B*T, -1)
            interp_motion = Motion.from_torch(skeleton, interp_local_R, interp_root_p)

            app_manager = AppManager()
            GT_kf_prob = GT_kf_prob.reshape(B*T, -1)
            pred_kf_score = pred_kf_score.reshape(B*T, -1)

            app = KeyframeApp(GT_motion, pred_motion, interp_motion, ybot.model(), total_keyframes, T, GT_traj.reshape(B*T, -1).cpu().numpy())
            app_manager.run(app)