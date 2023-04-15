import sys
sys.path.append(".")

import os
import torch
from torch.utils.data import DataLoader

import copy
import glm
from tqdm import tqdm

from pymovis.utils import util
from pymovis.motion import Motion, FBX
from pymovis.ops import rotation
from pymovis.vis import AppManager, MotionApp, Render, YBOT_FBX_DICT

from utility import testutil
from utility.config import Config
from utility.dataset import KeyframeDataset
from vis.visapp import ContextMotionApp
from model.ours import KeyframeTransformer, KeyframeGAN

class KeyframeApp(MotionApp):
    def __init__(self, GT_motion, pred_motion, model, GT_prob, pred_prob, time_per_motion):
        super().__init__(GT_motion, model, YBOT_FBX_DICT)

        self.GT_motion = GT_motion
        self.pred_motion = pred_motion

        self.GT_prob = GT_prob
        self.pred_prob = pred_prob

        self.time_per_motion = time_per_motion

        self.GT_model = model
        self.GT_model.set_source_skeleton(self.GT_motion.skeleton, YBOT_FBX_DICT)
        self.pred_model = copy.deepcopy(model)
        self.pred_model.set_source_skeleton(self.pred_motion.skeleton, YBOT_FBX_DICT)
        self.pred_model.meshes[0].materials[0].albedo = glm.vec3(0.5, 0.5, 0.5)

    def render(self):
        super().render(render_model=False)

        self.GT_model.set_pose_by_source(self.GT_motion.poses[self.frame])
        Render.model(self.GT_model).draw()
        # Render.model(self.GT_model).set_all_alphas(self.GT_prob[self.frame]).draw()

        self.pred_model.set_pose_by_source(self.pred_motion.poses[self.frame])
        Render.model(self.pred_model).draw()

    def render_text(self):
        super().render_text()
        Render.text_on_screen()

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/keyframe_gan.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset    = KeyframeDataset(train=False, config=config)
    skeleton   = dataset.skeleton

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

    # character
    ybot = FBX("dataset/ybot.fbx")

    # training loop
    with torch.no_grad():
        for GT_keyframe in tqdm(dataloader):
            B, T, D = GT_keyframe.shape

            """ 1. Prepare GT data """
            GT_keyframe = GT_keyframe.to(device)
            GT_local_R6, GT_root_p, GT_kf_score, GT_traj = torch.split(GT_keyframe, [D-7, 3, 1, 3], dim=-1)
            GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

            GT_batch = torch.cat([GT_local_R6, GT_root_p, GT_traj], dim=-1) # exclude keyframe score
            GT_batch = (GT_batch - kf_mean) / kf_std

            """ 2. Generate """
            pred_motion, pred_kf_score = model.generate(GT_batch)
            pred_motion = pred_motion * kf_std[..., :-3] + kf_mean[..., :-3] # exclude traj features

            pred_local_R6, pred_root_p = torch.split(pred_motion, [D-7, 3], dim=-1)
            pred_local_R = rotation.R6_to_R(pred_local_R6.reshape(B, T, -1, 6))
            
            # animation
            GT_local_R = GT_local_R.reshape(B*T, -1, 3, 3)
            GT_root_p = GT_root_p.reshape(B*T, -1)
            pred_local_R = pred_local_R.reshape(B*T, -1, 3, 3)
            pred_root_p = pred_root_p.reshape(B*T, -1)

            GT_motion = Motion.from_torch(skeleton, GT_local_R, GT_root_p)
            pred_motion = Motion.from_torch(skeleton, pred_local_R, pred_root_p)

            app_manager = AppManager()
            GT_kf_score = GT_kf_score.reshape(B*T, -1)
            pred_kf_score = pred_kf_score.reshape(B*T, -1)
            app = KeyframeApp(GT_motion, pred_motion, ybot.model(), GT_kf_score, pred_kf_score, T)
            app_manager.run(app)