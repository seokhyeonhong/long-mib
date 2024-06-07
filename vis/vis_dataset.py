from __future__ import annotations
import sys
sys.path.append(".")

import argparse
import numpy as np
import glm
import glfw
from aPyOpenGL import agl, transforms as trf
from OpenGL.GL import *

import torch
from torch.utils.data import DataLoader

from utils import utils
from utils.dataset import MotionDataset
from vis.visapp import two_bone_ik
from vis.motionapp import MotionApp

class DatasetApp(agl.App):
    def __init__(
        self,
        dataset: str, # "lafan1" or "human36m"
        motion_features: np.ndarray, # (B, T, D)
        skeleton: agl.Skeleton,
        traj_features: np.ndarray = None # (B, T, 4)
    ):
        super().__init__()

        self.dataset = dataset

        # motion
        B, T, D = motion_features.shape
        self.local_quats = trf.n_quat.from_ortho6d(motion_features[:, :, :-3].reshape(B, T, skeleton.num_joints, 6))
        self.root_pos = motion_features[:, :, -3:]
        self.skeleton = skeleton

        # trajectory
        self.traj_pos, self.traj_dir = None, None
        if traj_features is not None:
            self.traj_pos = traj_features[:, :, :2]
            self.traj_dir = traj_features[:, :, 2:]

        # numbers
        self.num_motions = motion_features.shape[0]
        self.frames_per_motion = motion_features.shape[1]
        self.total_frames = self.num_motions * self.frames_per_motion

        # visibility
        self.show_traj = True
        self.show_joints = True
        self.show_xray = True
    
    def start(self):
        super().start()
        self.model = agl.FBX("dataset/fbx-models/ybot.fbx").model()
        self.arrow = agl.FBX("dataset/fbx-models/arrow.fbx").model()
        self.joint_sphere = agl.Render.sphere(0.025).instance_num(self.skeleton.num_joints).albedo([1, 0, 0]).color_mode(True)

        self.ui.add_menu("DatasetApp")
        self.ui.add_menu_item("DatasetApp", "Show Trajectory", lambda: setattr(self, "show_traj", not self.show_traj), glfw.KEY_T)
        self.ui.add_menu_item("DatasetApp", "Show Joints", lambda: setattr(self, "show_joints", not self.show_joints), glfw.KEY_J)
        self.ui.add_menu_item("DatasetApp", "Show X-Ray", lambda: setattr(self, "show_xray", not self.show_xray), glfw.KEY_X)

    def update(self):
        super().update()
        self.frame = self.frame % self.total_frames
        
        # motion index (midx) & frame index (fidx)
        self.midx = self.frame // self.frames_per_motion
        self.fidx = self.frame % self.frames_per_motion

        self.pose = agl.Pose(self.skeleton, self.local_quats[self.midx, self.fidx], self.root_pos[self.midx, self.fidx])
        _, gp = trf.n_quat.fk(self.local_quats[self.midx, self.fidx], self.root_pos[self.midx, self.fidx], self.skeleton)
        self.joint_sphere.position(gp)

        if self.model is not None:
            self.model.set_pose(self.pose)

    def render(self):
        super().render()
        if self.model is not None:
            agl.Render.model(self.model).draw()

        if self.show_traj and self.traj_pos is not None and self.traj_dir is not None:
            arrow_pos = np.array([self.traj_pos[self.midx, self.fidx, 0], 0, self.traj_pos[self.midx, self.fidx, 1]])
            arrow_dir = self.traj_dir[self.midx, self.fidx] # (sin and cos)
            arrow_ang = np.arctan2(arrow_dir[0], arrow_dir[1])
            arrow_ori = glm.angleAxis(arrow_ang, glm.vec3(0, 1, 0))
            agl.Render.model(self.arrow).position(arrow_pos).orientation(arrow_ori).draw()
    
    def render_xray(self):
        super().render_xray()
        if self.show_xray:
            agl.Render.skeleton(self.pose).draw()
            self.joint_sphere.draw()

    def render_text(self):
        super().render_text()
        agl.Render.text_on_screen(f"Motion {self.midx + 1} / {self.num_motions}").position([0, 0.1, 0]).scale(0.5).draw()
        agl.Render.text_on_screen(f"Frame {self.fidx + 1} / {self.frames_per_motion}").position([0, 0.05, 0]).scale(0.5).draw()
    
    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        if action != glfw.PRESS:
            return
        
        if glfw.KEY_0 <= key <=glfw.KEY_9:
            self.frame = 0.1 * (key - glfw.KEY_0) * self.total_frames
            glfw.set_time(self.frame / self.fps)

def main(train=True):
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lafan1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = utils.load_config(f"config/{args.dataset}/default.yaml")
    utils.seed()

    # dataset
    dataset = MotionDataset(train=train, config=config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=train)
    skeleton = dataset.skeleton

    contact_idx = []
    for joint in config.contact_joints:
        contact_idx.append(skeleton.idx_by_name[joint])

    # iterate
    for batch in dataloader:
        motion = batch["motion"].to(device)
        traj   = batch["traj"].to(device)

        B, T, M = motion.shape
        GT_local_ortho6ds, GT_root_pos = torch.split(motion, [M-3, 3], dim=-1)
        GT_local_ortho6ds = GT_local_ortho6ds.reshape(B, T, skeleton.num_joints, 6)
        _, GT_global_positions = trf.t_ortho6d.fk(GT_local_ortho6ds, GT_root_pos, skeleton)

        GT_foot_vel = GT_global_positions[:, 1:, contact_idx] - GT_global_positions[:, :-1, contact_idx]
        GT_foot_vel = torch.sum(GT_foot_vel ** 2, dim=-1) # (B, t-1, 4)
        GT_foot_vel = torch.cat([GT_foot_vel[:, 0:1], GT_foot_vel], dim=1) # (B, t, 4)
        GT_contact  = (GT_foot_vel < config.contact_threshold).float() # (B, t, 4)

        agl.AppManager.start(MotionApp(motion, "Dataset", skeleton, dataset=args.dataset, contacts=GT_contact, contact_idx=contact_idx))

if __name__ == "__main__":
    main(True)