from __future__ import annotations
from typing import Union

import copy
import numpy as np
import torch
import glfw
import glm

from aPyOpenGL import agl, transforms as trf

H36M_BVH2FBX = {
    "Hips": "mixamorig:Hips",
    "Spine": "mixamorig:Spine",
    "Spine1": "mixamorig:Spine1",
    "Neck": "mixamorig:Neck",
    "Head": "mixamorig:Head",
    "LeftShoulder": "mixamorig:LeftShoulder",
    "LeftUpArm": "mixamorig:LeftArm",
    "LeftForeArm": "mixamorig:LeftForeArm",
    "LeftHand": "mixamorig:LeftHand",
    "LeftHandThumb": "mixamorig:LeftHand",
    "L_Wrist_End": "mixamorig:LeftHand",
    "RightShoulder": "mixamorig:RightShoulder",
    "RightUpArm": "mixamorig:RightArm",
    "RightForeArm": "mixamorig:RightForeArm",
    "RightHand": "mixamorig:RightHand",
    "RightHandThumb": "mixamorig:RightHand",
    "R_Wrist_End": "mixamorig:RightHand",
    "LeftUpLeg": "mixamorig:LeftUpLeg",
    "LeftLowLeg": "mixamorig:LeftLeg",
    "LeftFoot": "mixamorig:LeftFoot",
    "LeftToeBase": "mixamorig:LeftToeBase",
    "RightUpLeg": "mixamorig:RightUpLeg",
    "RightLowLeg": "mixamorig:RightLeg",
    "RightFoot": "mixamorig:RightFoot",
    "RightToeBase": "mixamorig:RightToeBase",
}


FBX2FBX = {
    "mixamorig:Hips": "mixamorig:Hips",
    "mixamorig:Spine": "mixamorig:Spine",
    "mixamorig:Spine1": "mixamorig:Spine1",
    "mixamorig:Spine2": "mixamorig:Spine2",
    "mixamorig:Neck": "mixamorig:Neck",
    "mixamorig:Head": "mixamorig:Head",
    "mixamorig:LeftShoulder": "mixamorig:LeftShoulder",
    "mixamorig:LeftArm": "mixamorig:LeftArm",
    "mixamorig:LeftForeArm": "mixamorig:LeftForeArm",
    "mixamorig:LeftHand": "mixamorig:LeftHand",
    "mixamorig:RightShoulder": "mixamorig:RightShoulder",
    "mixamorig:RightArm": "mixamorig:RightArm",
    "mixamorig:RightForeArm": "mixamorig:RightForeArm",
    "mixamorig:RightHand": "mixamorig:RightHand",
    "mixamorig:LeftUpLeg": "mixamorig:LeftUpLeg",
    "mixamorig:LeftLeg": "mixamorig:LeftLeg",
    "mixamorig:LeftFoot": "mixamorig:LeftFoot",
    "mixamorig:LeftToeBase": "mixamorig:LeftToeBase",
    "mixamorig:RightUpLeg": "mixamorig:RightUpLeg",
    "mixamorig:RightLeg": "mixamorig:RightLeg",
    "mixamorig:RightFoot": "mixamorig:RightFoot",
    "mixamorig:RightToeBase": "mixamorig:RightToeBase",
}

def _reshape(x, traj=False):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if traj:
        x = x[..., :2]
        x = np.concatenate([x[..., 0:1], np.zeros_like(x[..., 0:1]), x[..., 1:2]], axis=-1)
    return x.reshape(-1, x.shape[-1])

class MotionStruct:
    def __init__(
        self,
        features: Union[torch.Tensor, np.ndarray],
        num_batches: int,
        skeleton: agl.Skeleton,
        tag: str,
        traj: Union[torch.Tensor, np.ndarray] = None,
        kf_indices=None,
        contact=None,
    ):
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if len(features.shape) != 2:
            raise ValueError(f"features must be a 2D tensor, got {features.shape}")
        
        # convert local rotations to quaternions
        local_rots, root_pos = features[:, :-3], features[:, -3:]
        local_quats = trf.n_quat.from_ortho6d(local_rots.reshape(-1, skeleton.num_joints, 6)) # [B*T, J, 4]
        root_pos = root_pos.reshape(-1, 3)

        # set attributes
        self.num_batches = num_batches
        self.frame_per_batch = features.shape[0] // num_batches
        self.skeleton = skeleton
        self.poses = [agl.Pose(skeleton, lq, rp) for lq, rp in zip(local_quats, root_pos)]
        self.tag = tag
        self.traj = traj
        self.kf_indices = kf_indices
        self.contact = contact
        self.offset = np.array([0, 0, 0], dtype=np.float32)
        self.visible = False

        # traj and contact rendering
        self.traj_spheres = agl.Render.sphere(0.05).albedo([0.8, 0.1, 0.1]).instance_num(min(features.shape[0] // num_batches, 100))
        self.curr_traj_sphere = agl.Render.sphere(0.05).albedo([0.1, 0.8, 0.1])
        self.contact_spheres = [ agl.Render.sphere(0.05).albedo([0.1, 0.8, 0.1]) for _ in range(4) ]
    
    def render(self, character, frame, alpha=1.0, traj=False):
        if not self.visible:
            return
        
        character.set_pose(agl.Pose(self.skeleton, self.poses[frame].local_quats, self.poses[frame].root_pos + self.offset))
        agl.Render.model(character).alpha(alpha).draw()

        if traj and self.traj is not None:
            traj_idx = np.arange(self.frame_per_batch) + (frame // self.frame_per_batch) * self.frame_per_batch
            curr_idx = (frame % self.frame_per_batch) + (frame // self.frame_per_batch) * self.frame_per_batch
            other_idx = np.setdiff1d(traj_idx, curr_idx)
            self.traj_spheres.position(self.traj[other_idx] + self.offset).draw()
            self.curr_traj_sphere.position(self.traj[curr_idx] + self.offset).draw()
        
    def render_tag(self, frame: int, offset=[0, 0.8, 0]):
        if not self.visible:
            return
        
        pos = self.poses[frame].root_pos + np.array(offset) + self.offset
        agl.Render.text(self.tag).position(pos).scale(0.5).draw()
    
    def render_xray(self, frame):
        if not self.visible:
            return
        
        pose = agl.Pose(self.skeleton, self.poses[frame].local_quats, self.poses[frame].root_pos + self.offset)
        agl.Render.skeleton(pose).draw()

    def render_contact(self, frame, contact=False, contact_idx=None):
        # contact
        if contact and contact_idx is not None and self.contact is not None:
            on_contact = self.contact[frame]
            _, gp = trf.n_quat.fk(self.poses[frame].local_quats, self.poses[frame].root_pos, self.skeleton)
            contact_pos = gp[contact_idx]
            for i, idx in enumerate(contact_idx):
                if on_contact[i]:
                    self.contact_spheres[i].position(contact_pos[i] + self.offset).draw()
    
    
    def switch_visible(self):
        self.visible = not self.visible

    def get_base(self, frame):
        pos = self.poses[frame].root_pos + self.offset
        pos = glm.vec3(pos[0], 0, pos[2])

        dir = trf.n_quat.mul_vec(self.poses[frame].local_quats[0], np.array([0, 0, 1]))
        dir = np.array([dir[0], 0, dir[2]])
        dir = dir / (np.linalg.norm(dir) + 1e-8)

        q = trf.n_quat.between_vecs(np.array([0, 0, 1]), dir)
        q = trf.n_quat.to_rotmat(q)
        r = glm.mat3(*q.transpose().flatten())

        return pos, r

class MotionApp(agl.App):
    def __init__(
        self,
        motions: Union[list[torch.Tensor], list[np.ndarray], torch.Tensor, np.ndarray], # list of (B, T, D) tensors
        tags: Union[list[str], str],
        skeleton: agl.Skeleton,
        dataset: str = "lafan1",
        trajs: Union[list[torch.Tensor], list[np.ndarray]] = None, # list of (B, T, D) tensors
        contacts: Union[list[torch.Tensor], list[np.ndarray]] = None, # list of (B, T, 4) tensors
        contact_idx: list[int] = None,
        kf_indices: list[list[int]] = None,
    ):
        super().__init__()

        self.motions = motions if isinstance(motions, list) else [motions]
        self.tags = tags if isinstance(tags, list) else [tags]
        self.skeleton = skeleton
        self.dataset = dataset

        if contacts is None:
            self.contacts = [None for _ in range(len(motions))]
        else:
            self.contacts = contacts if isinstance(contacts, list) else [contacts]

        if trajs is None:
            self.trajs = [None for _ in range(len(motions))]
        else:
            self.trajs = trajs if isinstance(trajs, list) else [trajs]

        if kf_indices is None:
            self.kf_indices = [None for _ in range(len(motions))]
        else:
            self.kf_indices = kf_indices if isinstance(kf_indices, list) else [kf_indices]

        # motion info
        self.num_batches = self.motions[0].shape[0]
        self.frame_per_batch = self.motions[0].shape[1]
        self.total_frames = self.motions[0].shape[0] * self.motions[0].shape[1]
        self.contact_idx = contact_idx

        # reshape
        self.motions = [_reshape(m) for m in self.motions]
        if self.trajs is not None:
            self.trajs = [_reshape(t, traj=True) for t in self.trajs]
        if self.contacts is not None:
            self.contacts = [_reshape(c) for c in self.contacts]

        # rendering settings
        self.move_character = False
        self._show = {
            "arrow": False,
            "target": True,
            "tag": True,
            "xray": False,
            "info": True,
            "traj": True,
            "every_ten": False,
            "keyframe": False,
            "contact": False,
            "transition": False,
            "alpha": 0.2,
        }
    
    def _switch_move_character(self):
        self.move_character = not self.move_character
        for idx, motion in enumerate(self.motions):
            motion.offset = np.array([(idx * 1.5 if self.move_character else 0), 0, 0], dtype=np.float32)
    
    def _switch_show(self, key):
        self._show[key] = not self._show[key]
    
    def _switch_alpha(self):
        self._show["alpha"] = 0.2 if self._show["alpha"] == 1.0 else 1.0

    def start(self):
        super().start()

        # character model
        if self.dataset == "lafan1":
            self.character1 = agl.FBX("dataset/fbx-models/ybot.fbx").model()
            self.character2 = agl.FBX("dataset/fbx-models/ybot.fbx").model()
            self.character2.meshes[0].materials[0].albedo = glm.vec3([0.5, 0.5, 0.5])
        elif self.dataset == "100style":
            self.character1 = agl.FBX("dataset/fbx-models/ybot-fingers.fbx").model()
            self.character2 = agl.FBX("dataset/fbx-models/ybot-fingers.fbx").model()
            self.character2.meshes[0].materials[0].albedo = glm.vec3([0.5, 0.5, 0.5])
            self.character1.set_joint_map(FBX2FBX)
            self.character2.set_joint_map(FBX2FBX)
        elif self.dataset in ["human36m", "mann"]:
            self.character1, self.character2 = None, None
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        
        # model
        self.arrow = agl.FBX("dataset/fbx-models/arrow.fbx").model()

        # convert motions to MotionStruct
        motions = []
        for i in range(len(self.motions)):
            motions.append(MotionStruct(self.motions[i], self.num_batches, self.skeleton, self.tags[i], traj=self.trajs[i], kf_indices=self.kf_indices[i], contact=self.contacts[i]))
        self.motions = motions

        # ui
        self.ui.add_menu("MotionApp")
        self.ui.add_menu_item("MotionApp", "Move Character", self._switch_move_character, glfw.KEY_M)
        self.ui.add_menu_item("MotionApp", "Show Target", lambda: self._switch_show("target"), glfw.KEY_T)
        self.ui.add_menu_item("MotionApp", "Show X-Ray", lambda: self._switch_show("xray"), glfw.KEY_X)
        self.ui.add_menu_item("MotionApp", "Show Tag", lambda: self._switch_show("tag"), glfw.KEY_Y)
        self.ui.add_menu_item("MotionApp", "Show Info", lambda: self._switch_show("info"), glfw.KEY_I)
        self.ui.add_menu_item("MotionApp", "Show Traj", lambda: self._switch_show("traj"), glfw.KEY_J)
        self.ui.add_menu_item("MotionApp", "Show Every Ten", lambda: self._switch_show("every_ten"), glfw.KEY_E)
        self.ui.add_menu_item("MotionApp", "Show Keyframe", lambda: self._switch_show("keyframe"), glfw.KEY_K)
        self.ui.add_menu_item("MotionApp", "Show Arrow", lambda: self._switch_show("arrow"), glfw.KEY_R)
        self.ui.add_menu_item("MotionApp", "Show Contact", lambda: self._switch_show("contact"), glfw.KEY_C)
        self.ui.add_menu_item("MotionApp", "Show Transition", lambda: self._switch_show("transition"), glfw.KEY_N)
        self.ui.add_menu_item("MotionApp", "Switch Alpha", self._switch_alpha, glfw.KEY_P)
        
        for motion in self.motions:
            self.ui.add_menu_item("MotionApp", f"Show {motion.tag}", motion.switch_visible)
        self.ui.add_menu_item("MotionApp", "Show All", lambda: [motion.switch_visible() for motion in self.motions])
    
    def update(self):
        super().update()
        self.frame = self.frame % self.total_frames
        self.bidx, self.fidx = self.frame // self.frame_per_batch, self.frame % self.frame_per_batch

    def render(self):
        super().render()
        if self._show["arrow"]:
            for idx, motion in enumerate(self.motions):
                pos, dir = motion.get_base(self.frame)
                agl.Render.model(self.arrow).position(pos).orientation(dir).draw()

        if self.character1 is None or self.character2 is None:
            return

        if self._show["transition"]:
            if self.fidx < 10 or self.fidx == self.frame_per_batch - 1:
                character = self.character1
            else:
                character = self.character2
        else:
            character = self.character1
        
        # current frame
        for idx, motion in enumerate(self.motions):
            motion.render(self.character1 if motion.tag in ["GT", "Dataset"] else character, self.frame, traj=self._show["traj"])

            # every ten frames
            if self._show["every_ten"]:
                motion.render(self.character1, self.bidx * self.frame_per_batch)
                motion.render(self.character1, (self.bidx + 1) * self.frame_per_batch - 1)
                for i in range(10, self.frame_per_batch - 10, 10):
                    idx = self.bidx * self.frame_per_batch + i
                    motion.render(self.character2, idx)
            
            # keyframe
            if self._show["keyframe"] and motion.kf_indices is not None:
                for kf in motion.kf_indices[self.bidx][1:-1]:
                    motion.render(self.character2, kf + self.bidx * self.frame_per_batch)#, alpha=0.5)
            
        # target frame
        if self._show["target"]:
            for motion in self.motions:
                motion.render(self.character1, (self.bidx + 1) * self.frame_per_batch - 1, alpha=self._show["alpha"])

    def render_text(self):
        super().render_text()

        # motion tags
        if self._show["tag"]:
            for motion in self.motions:
                motion.render_tag(self.frame)

        # motion info
        if self._show["info"]:
            agl.Render.text_on_screen(f"Motion {self.bidx + 1} / {self.num_batches}").position([0, 0.1, 0]).scale(0.5).draw()
            agl.Render.text_on_screen(f"Frame {self.fidx + 1} / {self.frame_per_batch}").position([0, 0.05, 0]).scale(0.5).draw()

    def render_xray(self):
        super().render_xray()
        if self._show["contact"]:
            for motion in self.motions:
                motion.render_contact(self.frame, contact=self._show["contact"], contact_idx=self.contact_idx)

        if not self._show["xray"]:
            return

        # current frame
        for motion in self.motions:
            motion.render_xray(self.frame)

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if action != glfw.PRESS:
            return
        
        if mods & glfw.MOD_ALT:
            if (glfw.KEY_1 <= key <= len(self.motions) + glfw.KEY_1):
                self.motions[key - glfw.KEY_1].switch_visible()
        elif (glfw.KEY_0 <= key <= glfw.KEY_9):
            self.frame = (self.total_frames * (key - glfw.KEY_0)) // 10
            glfw.set_time(self.frame / self.fps)