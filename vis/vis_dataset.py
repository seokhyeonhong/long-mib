import sys
sys.path.append(".")
sys.path.append("..")

import torch
from torch.utils.data import DataLoader

import glm
import glfw
import copy
from pymovis.motion import Motion, FBX
from pymovis.vis import AppManager, MotionApp, YBOT_FBX_DICT, Render
from pymovis.ops import rotation

from utility.config import Config
from utility.dataset import MotionDataset
from utility import utils
from vis.visapp import SingleMotionApp

class DatasetApp(MotionApp):
    def __init__(self, motion, model, traj):
        super().__init__(motion, model, YBOT_FBX_DICT)
        # self.prob = prob
        self.copy_model = copy.deepcopy(model)
        # self.prob_sorted_idx = torch.argsort(prob.squeeze(), descending=True)[11:]
        self.prob_idx = 0
        self.traj = traj

        # vis
        self.arrow = Render.arrow().set_albedo([1, 0, 0])
        self.sphere = Render.sphere(0.05, 4, 4).set_albedo([1, 0, 0])
    
    def render(self):
        super().render()
        xz, forward = self.traj[self.frame, :2], self.traj[self.frame, 2:]
        R = glm.angleAxis(glm.radians(90), glm.cross(glm.vec3(0, 1, 0), glm.vec3(forward[0], forward[1], forward[2])))
        p = glm.vec3(xz[0], 0, xz[1])
        self.arrow.set_position(p).set_orientation(R).draw()

        for i in range(self.prob_idx):
            self.copy_model.set_pose_by_source(self.motion.poses[self.prob_sorted_idx[i]])
            Render.model(self.copy_model).set_all_alphas(0.5).draw()
        
        # for frame in range(self.total_frames):
        #     position = self.motion.poses[frame].root_p
        #     self.sphere.set_position(position[0], 0, position[2]).draw()
    
    def render_text(self):
        super().render_text()
        Render.text_on_screen(f"PROB_IDX: {self.prob_idx}").draw()
    
    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if key == glfw.KEY_D and action == glfw.PRESS:
            self.prob_idx += 1
        elif key == glfw.KEY_S and action == glfw.PRESS:
            self.prob_idx -= 1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/context.json")

    dataset = MotionDataset(train=False, config=config)
    v_forward = torch.from_numpy(config.v_forward).to(device)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    skeleton = dataset.skeleton

    character = FBX("dataset/ybot.fbx")

    for GT_motion in dataloader:
        """ 1. GT motion data """
        B, T, D = GT_motion.shape
        GT_motion = GT_motion.to(device)
        # GT_local_R6, GT_global_p, GT_traj = utils.get_motion_and_trajectory(GT_motion, skeleton, v_forward)
        GT_local_R6, GT_root_p = torch.split(GT_motion, [D-3, 3], dim=-1)
        GT_local_R = rotation.R6_to_R(GT_local_R6.reshape(B, T, -1, 6))

        """ 2. Animation """
        motion = Motion.from_torch(skeleton, GT_local_R.reshape(B*T, -1, 3, 3), GT_root_p.reshape(B*T, 3))

        """ 3. Visualization """
        app_manager = AppManager()
        app = SingleMotionApp(motion, character.model(), T)
        app_manager.run(app)