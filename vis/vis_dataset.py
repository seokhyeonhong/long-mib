import sys
sys.path.append(".")
sys.path.append("..")

import torch
from torch.utils.data import DataLoader

import glfw
import copy
from pymovis.motion import Motion, FBX
from pymovis.vis import AppManager, MotionApp, YBOT_FBX_DICT, Render
from pymovis.ops import rotation

from utility.config import Config
from utility.dataset import MotionDataset, KeyframeDataset
from vis.visapp import SingleMotionApp

class KeyframeApp(MotionApp):
    def __init__(self, motion, model, dict, prob):
        super().__init__(motion, model, dict)
        self.prob = prob
        self.copy_model = copy.deepcopy(model)
        self.prob_sorted_idx = torch.argsort(prob.squeeze(), descending=True)[11:]
        self.prob_idx = 0
        self.sphere = Render.sphere(0.05, 4, 4).set_albedo([1, 0, 0])

        for p, idx in zip(prob.squeeze(), range(len(prob.squeeze()))):
            print(f"{idx}: {p}")
    
    def render(self):
        super().render()

        for i in range(self.prob_idx):
            self.copy_model.set_pose_by_source(self.motion.poses[self.prob_sorted_idx[i]])
            Render.model(self.copy_model).set_all_alphas(0.5).draw()
        
        for frame in range(self.total_frames):
            position = self.motion.poses[frame].root_p
            self.sphere.set_position(position[0], 0, position[2]).draw()
    
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
    config = Config.load("configs/sparse.json")
    character = FBX("dataset/ybot.fbx")
    dataset = KeyframeDataset(train=False, config=config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    skeleton = dataset.skeleton

    sparse_frames = torch.arange(config.max_transition // config.fps) * config.fps
    sparse_frames += (config.context_frames-1) + config.fps
    sparse_frames = torch.cat([torch.arange(config.context_frames), sparse_frames])

    for feature in dataloader:
        B, T, D = feature.shape

        local_R6, root_p, kf_prob = torch.split(feature, [D-4, 3, 1], dim=-1)
        local_R = rotation.R6_to_R(local_R6.reshape(B, T, D//6, 6))

        for b in range(B):
            motion = Motion.from_torch(skeleton, local_R[b], root_p[b])

            app_manager = AppManager()
            app = KeyframeApp(motion, character.model(), YBOT_FBX_DICT, kf_prob[b])
            app_manager.run(app)