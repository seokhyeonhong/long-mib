import copy
import numpy as np
import glfw

from pymovis.motion import FBX, Motion
from pymovis.vis import MotionApp, Render, YBOT_FBX_DICT

class VisApp(MotionApp):
    def __init__(self, GT_motion, pred_motions, GT_model, pred_model):
        super().__init__(GT_motion, GT_model, YBOT_FBX_DICT)

        # visibility
        self.axis.set_visible(False)
        self.grid.set_visible(False)
        self.show_GT = True
        self.show_pred = True
        self.show_skeleton = False

        # motion and model
        self.GT_motion     = GT_motion
        self.GT_model      = GT_model

        if isinstance(pred_motions, Motion):
            pred_motions = [pred_motions]
        
        self.pred_motions  = pred_motions
        self.pred_models   = [copy.deepcopy(pred_model)] * len(pred_motions)
        for pred_model in self.pred_models:
            pred_model.set_source_skeleton(self.motion.skeleton, YBOT_FBX_DICT)

        self.target_model  = copy.deepcopy(self.GT_model)
        self.target_model.set_pose_by_source(self.GT_motion.poses[-1])
        
    def render(self):
        if self.show_GT:
            self.motion = self.GT_motion
            self.model = self.GT_model
            super().render(not self.show_skeleton, self.show_skeleton)

        if self.show_pred:
            for i, pred_motion in enumerate(self.pred_motions):
                self.motion = pred_motion
                self.model = self.pred_models[i]
                super().render(not self.show_skeleton, self.show_skeleton)

        # draw target
        Render.model(self.target_model).set_all_color_modes(True).set_all_alphas(0.5).draw()
        
    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if key == glfw.KEY_Q and action == glfw.PRESS:
            self.show_GT = not self.show_GT
        elif key == glfw.KEY_W and action == glfw.PRESS:
            self.show_pred = not self.show_pred
        elif key == glfw.KEY_S and action == glfw.PRESS:
            self.show_skeleton = not self.show_skeleton