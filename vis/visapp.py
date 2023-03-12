import copy
import numpy as np
import glfw

from pymovis.motion import FBX, Motion
from pymovis.vis import MotionApp, Render, YBOT_FBX_DICT

class MultiMotionApp(MotionApp):
    def __init__(self, GT_motion, pred_motion, GT_model, pred_model, frames_per_motion):
        super().__init__(GT_motion, GT_model, YBOT_FBX_DICT)
        self.frames_per_motion = frames_per_motion

        # visibility
        self.axis.set_visible(False)
        self.text.set_visible(False)
        self.show_GT = True
        self.show_pred = True
        self.show_skeleton = False

        # motion and model
        self.GT_motion     = GT_motion
        self.GT_model      = GT_model

        self.pred_motion   = pred_motion
        self.pred_model    = pred_model
        self.pred_model.set_source_skeleton(self.motion.skeleton, YBOT_FBX_DICT)

        self.target_model  = copy.deepcopy(self.GT_model)
    
    def render(self):
        ith_motion = self.frame // self.frames_per_motion

        if self.show_GT:
            self.motion = self.GT_motion
            self.model = self.GT_model
            super().render(not self.show_skeleton, self.show_skeleton)

        if self.show_pred:
            self.motion = self.pred_motion
            self.model = self.pred_model
            super().render(not self.show_skeleton, self.show_skeleton)

        # draw target
        self.target_model.set_pose_by_source(self.GT_motion.poses[(ith_motion+1)*self.frames_per_motion - 1])
        Render.model(self.target_model).set_all_color_modes(True).set_all_alphas(0.5).draw()
    
    def render_text(self):
        super().render_text()
        Render.text_on_screen(f"Motion {self.frame // self.frames_per_motion} - Frame {self.frame % self.frames_per_motion}").set_position(10, 10, 0).draw()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if key == glfw.KEY_Q and action == glfw.PRESS:
            self.show_GT = not self.show_GT
        elif key == glfw.KEY_W and action == glfw.PRESS:
            self.show_pred = not self.show_pred
        elif key == glfw.KEY_S and action == glfw.PRESS:
            self.show_skeleton = not self.show_skeleton