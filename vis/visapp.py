from __future__ import annotations
import glm
import glfw
import numpy as np
import copy
from aPyOpenGL import agl, transforms as trf

def ybot_copy(ybot_model, albedo=None):
    res = copy.deepcopy(ybot_model)
    if albedo is not None:
        res.meshes[0].materials[0].albedo = glm.vec3(albedo)
        res.meshes[1].materials[0].albedo = glm.vec3(albedo) * 0.5
    
    """
    Memo: Pretty albedo colors
        - [0.41, 0.30, 0.43]: purple
        - [0.65, 0.54, 0.27]: yellow
        - [0.44, 0.46, 0.50]: gray
        - [0.18, 0.39, 0.19]: green
    """
    return res

def two_bone_ik(pose: agl.Pose, target_pos: np.ndarray, effector_idx: int, eps: float=1e-8):
    mid_idx = pose.skeleton.parent_idx[effector_idx]
    base_idx = pose.skeleton.parent_idx[mid_idx]

    global_quat, global_pos = trf.n_quat.fk(pose.local_quats, pose.root_pos, pose.skeleton)
    global_rotmat = trf.n_quat.to_rotmat(global_quat)

    a = global_pos[base_idx]
    b = global_pos[mid_idx]
    c = global_pos[effector_idx]

    global_a_R = global_rotmat[base_idx]
    global_b_R = global_rotmat[mid_idx]

    lab = np.linalg.norm(b - a)
    lcb = np.linalg.norm(b - c)
    lat = np.clip(np.linalg.norm(target_pos - a), eps, lab + lcb - eps)

    def normalize(v):
        return v / (np.linalg.norm(v) + eps)

    ac_ab_0 = np.arccos(np.clip(np.dot(normalize(c - a), normalize(b - a)), -1, 1))
    ba_bc_0 = np.arccos(np.clip(np.dot(normalize(a - b), normalize(c - b)), -1, 1))
    ac_at_0 = np.arccos(np.clip(np.dot(normalize(c - a), normalize(target_pos - a)), -1, 1))

    ac_ab_1 = np.arccos(np.clip((lcb*lcb - lab*lab - lat*lat) / (-2*lab*lat), -1, 1))
    ba_bc_1 = np.arccos(np.clip((lat*lat - lab*lab - lcb*lcb) / (-2*lab*lcb), -1, 1))

    forward = trf.n_quat.mul_vec(pose.local_quats[0], np.array([0, 0, 1]))
    forward = normalize(forward * np.array([1, 0, 1]))
    axis_0 = normalize(np.cross(c - a, forward))
    axis_1 = normalize(np.cross(c - a, target_pos - a))

    r0 = trf.n_aaxis.to_quat((ac_ab_1 - ac_ab_0) * (trf.n_rotmat.inv(global_a_R) @ axis_0))
    r1 = trf.n_aaxis.to_quat((ba_bc_1 - ba_bc_0) * (trf.n_rotmat.inv(global_b_R) @ axis_0))
    r2 = trf.n_aaxis.to_quat(ac_at_0 * (trf.n_rotmat.inv(global_a_R) @ axis_1))

    local_quats = pose.local_quats.copy()
    local_quats[base_idx] = trf.n_quat.mul(trf.n_quat.mul(local_quats[base_idx], r0), r2)
    local_quats[mid_idx] = trf.n_quat.mul(local_quats[mid_idx], r1)

    return agl.Pose(pose.skeleton, local_quats, pose.root_pos.copy())

COLORS = [
    # [0.65, 0.54, 0.27], # yellow
    [0.41, 0.30, 0.43], # purple
    # [0.44, 0.46, 0.50], # gray
    # [0.18, 0.39, 0.19], # green
    # [0.50, 0.20, 0.20], # red
]

class MotionStruct:
    def __init__(self, skeleton, local_quats, root_pos, kf_indices, contact=None, text=None):
        self.poses = [agl.Pose(skeleton, lq, rp) for lq, rp in zip(local_quats, root_pos)]
        self.kf_indices = kf_indices
        self.contact = contact
        self.text = text

class InbetweenApp(agl.App):
    def __init__(
            self,
            GT_motion: MotionStruct,
            pred_motions: list[MotionStruct],
            skeleton: agl.Skeleton,
            frames_per_motion: int,
            trajs=None,
            ik=True,
        ):
        """
        Args:
            GT_motion: tuple(local_quats, root_pos)
            pred_motions: list of tuple(local_quats, root_pos)
            skeleton: agl.Skeleton
            frames_per_motion: int
            trajs: np.ndarray of shape (num_frames, 4) where each row is (pos_x, pos_z, dir_x, dir_z)
            keyframes: list of list of int where each list of int is a list of keyframe indices for each motion
        """
        super().__init__()

        # motion
        self.GT_motion = GT_motion
        self.pred_motions = pred_motions
        self.skeleton = skeleton

        self.render_motion = [True] * (len(self.pred_motions) + 1) # 0 for GT, 1: for pred

        # frames
        self.total_frames = len(self.GT_motion.poses)
        self.frames_per_motion = frames_per_motion

        # camera options
        self.focus_on_root = False
        
        # traj
        self.trajs = trajs
        
        # visibility options
        self.render_traj = False
        self.render_kfs = False
        self.render_target = False

        # IK
        self.ik = ik

        # additional options
        self.move_character = 0
        self.every_10_frames = False
        self.opaque = False

    def start(self):
        super().start()

        self.GT_model = agl.FBX("dataset/characters/ybot.fbx").model()
        self.pred_models: list[agl.Model] = []
        for i in range(len(self.pred_motions)):
            # random_albedo = np.random.randn(3) / 6 + 0.5 # between 0 and 1
            # random_albedo = np.random.rand(3)
            # print(f"Albedo at {i}th character: {random_albedo}")
            # self.pred_models.append(ybot_copy(self.GT_model, albedo=random_albedo))
            self.pred_models.append(ybot_copy(self.GT_model, albedo=COLORS[i] if i < len(COLORS) else COLORS[-1]))

        # ui
        self.ui.add_menu("InbetweenApp")
    
        # ui options - text
        self.text = agl.Render.text_on_screen()
        self.ui.add_menu_item("InbetweenApp", "Text", self.text.switch_visible, key=glfw.KEY_T)
        self.ui.add_menu_item("InbetweenApp", "Render Target", lambda: setattr(self, "render_target", not self.render_target), key=glfw.KEY_R)
        self.ui.add_menu_item("InbetweenApp", "Keyframe Visualization", lambda: setattr(self, "render_kfs", not self.render_kfs), key=glfw.KEY_K)
        self.ui.add_menu_item("InbetweenApp", "Move Character", lambda: setattr(self, "move_character", (self.move_character + 1) % 3), key=glfw.KEY_M)
        self.ui.add_menu_item("InbetweenApp", "Focus on Root", lambda: setattr(self, "focus_on_root", not self.focus_on_root), key=glfw.KEY_C)
        self.ui.add_menu_item("InbetweenApp", "Every 10 Frames", lambda: setattr(self, "every_10_frames", not self.every_10_frames), key=glfw.KEY_E)
        self.ui.add_menu_item("InbetweenApp", "Opaque", lambda: setattr(self, "opaque", not self.opaque), key=glfw.KEY_O)

        # ui options - additional visualizations
        if self.trajs is not None:
            self.render_traj = True
            self.traj_spheres = agl.Render.sphere(0.05).instance_num(self.frames_per_motion - 1).albedo([1, 0, 0]).color_mode(True)
            self.curr_traj_sphere = agl.Render.sphere(0.05).albedo([1, 0, 0]).color_mode(True)
            self.ui.add_menu_item("InbetweenApp", "Traj Visualization", lambda: setattr(self, "render_traj", not self.render_traj), key=glfw.KEY_J)
        
        # IK post processing
        if self.ik:
            left_foot_idx = self.skeleton.idx_by_name["mixamorig:LeftFoot"]
            right_foot_idx = self.skeleton.idx_by_name["mixamorig:RightFoot"]

            ik_threshold = 0.8
            inertial_left, inertial_right = 0, 0
            max_count = 10
            count_left, count_right = 0, 0
            for midx in range(len(self.pred_motions)):
                motion = self.pred_motions[midx]
                for frame in range(len(motion.poses)):
                    if frame % self.frames_per_motion < 10 or frame % self.frames_per_motion == self.frames_per_motion-1:
                        inertial_left = False
                        inertial_right = False
                        count_left = 0
                        count_right = 0
                        continue
                    
                    _, curr_global_p = trf.n_quat.fk(motion.poses[frame].local_quats, motion.poses[frame].root_pos, self.skeleton)
                    _, prev_global_p = trf.n_quat.fk(motion.poses[frame-1].local_quats, motion.poses[frame-1].root_pos, self.skeleton)

                    if motion.contact[frame, 0] > ik_threshold:
                        target_left = prev_global_p[left_foot_idx]
                        # breakpoint()
                        motion.poses[frame] = two_bone_ik(motion.poses[frame], target_left, left_foot_idx)
                        # breakpoint()
                        inertial_left = True
                        count_left = 0
                    elif inertial_left is True:
                        count_left += 1
                        disp = curr_global_p[left_foot_idx] - target_left
                        motion.poses[frame] = two_bone_ik(motion.poses[frame], target_left + disp * ((1 / max_count) * count_left), left_foot_idx)
                        if count_left >= max_count:
                            inertial_left = False
                            count_left = 0

                    if motion.contact[frame, 2] > ik_threshold:
                        target_right = prev_global_p[right_foot_idx]
                        motion.poses[frame] = two_bone_ik(motion.poses[frame], target_right, right_foot_idx)
                        inertial_right = True
                        count_right = 0
                    elif inertial_right is True:
                        count_right += 1
                        disp = curr_global_p[right_foot_idx] - target_right
                        motion.poses[frame] = two_bone_ik(motion.poses[frame], target_right + disp * ((1 / max_count) * count_right), right_foot_idx)
                        if count_right >= max_count:
                            inertial_right = False
                            count_right = 0

    def update(self):
        super().update()
        self.frame = self.frame % self.total_frames

        # update camera focus position
        root_pos_sum = self.GT_motion.poses[self.frame].root_pos.copy()
        for idx, motion in enumerate(self.pred_motions):
            d_pos = np.array([0, 0, 0])
            if self.move_character == 1:
                d_pos = np.array([(idx+1)*1.5, 0, 0])
            elif self.move_character == 2:
                d_pos = np.array([0, 0, (idx+1)*1.5])
            root_pos_sum += motion.poses[self.frame].root_pos + d_pos
        
        cam_focus = root_pos_sum / (len(self.pred_motions) + 1)
        if self.focus_on_root:
            d_pos = glm.vec3(-6, 2, 0) if self.move_character == 2 else glm.vec3(0, 2, 6)
            self.camera.set_position(cam_focus + d_pos)
            self.camera.set_focus_position(cam_focus)
            self.camera.set_up(glm.vec3(0, 1, 0))
    
    def _render_model(self, idx, frame, alpha=1.0):
        if not self.render_motion[idx]:
            return
        
        if idx == 0:
            model = self.GT_model
            pose = self.GT_motion.poses[frame]

            # ith_motion = frame // self.frames_per_motion
            # traj_pos = []
            # for p in self.GT_motion.poses[ith_motion*self.frames_per_motion:(ith_motion+1)*self.frames_per_motion]:
            #     root_pos = p.root_pos
            #     traj_pos.append([root_pos[0], root_pos[2]])
            # positions = [glm.vec3(pos[0], 0, pos[1]) for idx, pos in enumerate(traj_pos) if idx != frame % self.frames_per_motion]
            # self.traj_spheres.albedo([0, 1, 1]).position(positions).draw()
            # self.curr_traj_sphere.albedo([0, 1, 1]).position(glm.vec3(traj_pos[frame % self.frames_per_motion][0], 0, traj_pos[frame % self.frames_per_motion][1])).draw()
        else:
            model = self.pred_models[idx-1]
            pose = self.pred_motions[idx-1].poses[frame]
        
        if self.move_character == 1:
            d_pos = np.array([idx*1.5, 0, 0])
            pose = agl.Pose(self.skeleton, pose.local_quats, pose.root_pos + d_pos)
        elif self.move_character == 2:
            d_pos = np.array([0, 0, idx*1.5])
            pose = agl.Pose(self.skeleton, pose.local_quats, pose.root_pos + d_pos)
        
        model.set_pose(pose)
        agl.Render.model(model).alpha(alpha).draw()
    
    def _reder_motion_text(self, idx):
        if not self.render_motion[idx]:
            return
        
        if idx == 0:
            pose = self.GT_motion.poses[self.frame]
            text = self.GT_motion.text
        else:
            pose = self.pred_motions[idx-1].poses[self.frame]
            text = self.pred_motions[idx-1].text

        if self.move_character:
            d_pos = np.array([idx, 0, 0])
            pose = agl.Pose(self.skeleton, pose.local_quats, pose.root_pos + d_pos)
        
        # if text is not None:
        #     agl.Render.text(text).position(pose.root_pos + glm.vec3(0, 1.0, 0)).scale(0.5).draw()

    def render(self):
        super().render()
        ith_motion = self.frame // self.frames_per_motion
        ith_frame = self.frame % self.frames_per_motion
        
        # render trajectory
        if self.render_traj:
            traj_pos = self.trajs[ith_motion*self.frames_per_motion:(ith_motion+1)*self.frames_per_motion, 0:2]
            d_pos = np.array([0, 0, 0])
            if self.move_character == 1:
                d_pos = np.array([1.5, 0, 0])
            elif self.move_character == 2:
                d_pos = np.array([0, 0, 1.5])
            d_pos = glm.vec3(d_pos)
            positions = [glm.vec3(pos[0], 0, pos[1]) + d_pos for idx, pos in enumerate(traj_pos) if idx != ith_frame]
            self.traj_spheres.albedo([1, 0, 0]).position(positions).draw()
            self.curr_traj_sphere.albedo([1, 0, 0]).position(glm.vec3(traj_pos[ith_frame, 0], 0, traj_pos[ith_frame, 1]) + d_pos).draw()
        
        # current frame
        for i in range(len(self.pred_models) + 1):
            self._render_model(i, self.frame, alpha=1.0 if not self.opaque else 0.0)
        
        # target frame
        if self.render_target:
            target_frame = (ith_motion+1) * self.frames_per_motion - 1
            for i in range(len(self.pred_models) + 1):
                self._render_model(i, target_frame, alpha=0.3)

        # render every 10 frames
        if self.every_10_frames:
            # GT
            for frame in range(10, self.frames_per_motion-10, 10):
                for idx in range(len(self.pred_models) + 1):
                    self._render_model(idx, ith_motion * self.frames_per_motion + frame, alpha=1.0)
            
            # pred
            for frame in [0, self.frames_per_motion-1]:
                for idx in range(len(self.pred_models) + 1):
                    self._render_model(idx, ith_motion * self.frames_per_motion + frame, alpha=0.3)

    def render_text(self):
        super().render_text()

        ith_motion = self.frame // self.frames_per_motion
        ith_frame = self.frame % self.frames_per_motion
        num_motions = self.total_frames // self.frames_per_motion

        self.text.text(f"Motion {ith_motion+1} / {num_motions}\nFrame {ith_frame+1} / {self.frames_per_motion}").position([0, 0.2, 0]).scale(0.5).draw()
        agl.Render.text_on_screen(f"Frame {ith_frame+1:3d} / {self.frames_per_motion}").position([0.02, 0.9, 0]).scale(0.5).draw()


        # # render motion text
        # for i in range(len(self.pred_motions) + 1):
        #     self._reder_motion_text(i)

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        if action != glfw.PRESS:
            return
        
        # command
        left_alt_pressed = (glfw.get_key(window, glfw.KEY_LEFT_ALT) == glfw.PRESS)
        if left_alt_pressed:
            if glfw.KEY_0 <= key <= glfw.KEY_9 and (key - glfw.KEY_0) < len(self.render_motion):
                self.render_motion[key - glfw.KEY_0] = not self.render_motion[key - glfw.KEY_0]
        else:
            if glfw.KEY_0 <= key <=glfw.KEY_9:
                self.frame = int(0.1 * (key - glfw.KEY_0) * self.total_frames)
                glfw.set_time(self.frame / self.fps)