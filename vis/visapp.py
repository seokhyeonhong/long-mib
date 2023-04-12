import copy
import numpy as np
import glfw
import glm

from pymovis.vis import MotionApp, Render, YBOT_FBX_DICT

class SingleMotionApp(MotionApp):
    def __init__(self, motion, ybot_model, frames_per_motion):
        super().__init__(motion, ybot_model, YBOT_FBX_DICT)
        self.frames_per_motion = frames_per_motion
        self.show_transition = False
        self.show_many = False
        self.copy_model = copy.deepcopy(ybot_model)
        self.sphere = Render.sphere(0.05).set_albedo([1, 0, 0])

    def render(self):
        super().render()

        if self.show_many:
            if self.show_transition:
                ith_motion = self.frame // self.frames_per_motion
                for i in range(0, self.frames_per_motion, 3):
                    if i < 10:
                        self.model.set_pose_by_source(self.motion.poses[ith_motion*self.frames_per_motion + i])
                        Render.model(self.model).draw()
                    else:
                        self.copy_model.set_pose_by_source(self.motion.poses[ith_motion*self.frames_per_motion + i])
                        Render.model(self.copy_model).set_albedo_of(glm.vec3(0.5), 0).draw()

                self.model.set_pose_by_source(self.motion.poses[(ith_motion+1)*self.frames_per_motion - 1])
                Render.model(self.model).draw()
            else:
                ith_motion = self.frame // self.frames_per_motion
                for i in range(0, 10, 3):
                    self.model.set_pose_by_source(self.motion.poses[ith_motion*self.frames_per_motion + i])
                    Render.model(self.model).draw()
                self.model.set_pose_by_source(self.motion.poses[(ith_motion+1)*self.frames_per_motion - 1])
                Render.model(self.model).draw()
        else:
            self.model.set_pose_by_source(self.motion.poses[self.frame])
            Render.model(self.model).draw()
        
        ith_motion = self.frame // self.frames_per_motion
        for i in range(ith_motion*self.frames_per_motion, (ith_motion+1)*self.frames_per_motion):
            self.sphere.set_position(self.motion.poses[i].root_p[0], 0, self.motion.poses[i].root_p[2]).draw()

    def render_text(self):
        super().render_text()
        # Render.text_on_screen(f"Motion {self.frame // self.frames_per_motion} - Frame {self.frame % self.frames_per_motion}").set_position(10, 10, 0).draw()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if key == glfw.KEY_Z and action == glfw.PRESS:
            self.show_transition = not self.show_transition
        elif key == glfw.KEY_X and action == glfw.PRESS:
            self.show_many = not self.show_many

class ContextMotionApp(MotionApp):
    def __init__(self, GT_motion, pred_motion, ybot_model, frames_per_motion):
        super().__init__(GT_motion, ybot_model, YBOT_FBX_DICT)
        self.frames_per_motion = frames_per_motion

        # visibility
        self.axis.set_visible(False)
        self.text.set_visible(False)
        self.show_GT = True
        self.show_pred = True
        self.show_skeleton = False

        # motion and model
        self.GT_motion     = GT_motion
        self.GT_model      = ybot_model

        self.pred_motion   = pred_motion
        self.pred_model    = copy.deepcopy(ybot_model)
        self.pred_model.set_source_skeleton(self.motion.skeleton, YBOT_FBX_DICT)
        self.pred_model.meshes[0].materials[0].albedo = glm.vec3(0.5, 0.5, 0.5)

        self.target_model  = copy.deepcopy(self.GT_model)
    
    def render(self):
        ith_motion = self.frame // self.frames_per_motion

        if self.show_GT:
            self.motion = self.GT_motion
            self.model = self.GT_model
            super().render(render_xray=self.show_skeleton)

        if self.show_pred:
            self.motion = self.pred_motion
            self.model = self.pred_model
            super().render(render_xray=self.show_skeleton)

        # draw target
        self.target_model.set_pose_by_source(self.GT_motion.poses[(ith_motion+1)*self.frames_per_motion - 1])
        Render.model(self.target_model).set_all_color_modes(False).set_all_alphas(0.5).draw()

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

class DetailMotionApp(MotionApp):
    def __init__(self, GT_motion, context_motion, detail_motion, ybot_model, frames_per_motion):
        super().__init__(GT_motion, ybot_model, YBOT_FBX_DICT)
        self.frames_per_motion = frames_per_motion

        # visibility
        self.axis.set_visible(False)
        self.text.set_visible(False)
        self.show_GT = True
        self.show_context = True
        self.show_detail = True
        self.show_skeleton = False

        # motion and model
        self.GT_motion     = GT_motion
        self.GT_model      = ybot_model

        self.context_motion = context_motion
        self.context_model = copy.deepcopy(ybot_model)
        self.context_model.set_source_skeleton(self.motion.skeleton, YBOT_FBX_DICT)
        self.context_model.meshes[0].materials[0].albedo = glm.vec3(1.0, 0.2, 0.2)

        self.detail_motion = detail_motion
        self.detail_model  = copy.deepcopy(ybot_model)
        self.detail_model.set_source_skeleton(self.motion.skeleton, YBOT_FBX_DICT)
        self.detail_model.meshes[0].materials[0].albedo = glm.vec3(0.5)

        self.target_model  = copy.deepcopy(self.GT_model)
    
    def render(self):
        ith_motion = self.frame // self.frames_per_motion
        ith_frame = self.frame % self.frames_per_motion

        # if self.show_GT:
        #     self.motion = self.GT_motion
        #     self.model = self.GT_model
        #     super().render(render_xray=self.show_skeleton)

        # if self.show_context:
        #     self.motion = self.context_motion
        #     self.model = self.context_model
        #     super().render(render_xray=self.show_skeleton)

        if self.show_detail:
            self.motion = self.detail_motion
            self.model = self.GT_model if ith_frame < 10 or ith_frame == self.frames_per_motion - 1 else self.detail_model
            super().render(render_xray=self.show_skeleton)

        # draw target
        self.target_model.set_pose_by_source(self.GT_motion.poses[(ith_motion+1)*self.frames_per_motion - 1])
        Render.model(self.target_model).set_all_color_modes(False).set_all_alphas(0.5).draw()

        print(f"Motion {self.frame // self.frames_per_motion} - Frame {self.frame % self.frames_per_motion}")
    
    # def render_text(self):
    #     super().render_text()
        # Render.text_on_screen(f"Motion {self.frame // self.frames_per_motion} - Frame {self.frame % self.frames_per_motion}").set_position(10, 10, 0).set_scale(0.5).draw()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)

        if key == glfw.KEY_Q and action == glfw.PRESS:
            self.show_GT = not self.show_GT
        elif key == glfw.KEY_W and action == glfw.PRESS:
            self.show_context = not self.show_context
        elif key == glfw.KEY_E and action == glfw.PRESS:
            self.show_detail = not self.show_detail
        elif key == glfw.KEY_S and action == glfw.PRESS:
            self.show_skeleton = not self.show_skeleton

class SparseMotionApp(MotionApp):
    def __init__(self, GT_motion, pred_motion, ybot_model, frames_per_motion, sparse_frames):
        super().__init__(GT_motion, ybot_model, YBOT_FBX_DICT)
        self.frames_per_motion = frames_per_motion
        self.sparse_frames = sparse_frames

        # visibility
        self.axis.set_visible(False)
        self.text.set_visible(False)
        self.show_GT = True
        self.show_pred = True
        self.show_skeleton = False

        # motion and model
        self.GT_motion     = GT_motion
        self.GT_model      = ybot_model

        self.pred_motion   = pred_motion
        self.pred_model    = copy.deepcopy(ybot_model)
        self.pred_model.set_source_skeleton(self.motion.skeleton, YBOT_FBX_DICT)
        self.pred_model.meshes[0].materials[0].albedo = glm.vec3(0.5, 0.5, 0.5)

        self.target_model  = copy.deepcopy(self.GT_model)
        self.sparse_target_model = copy.deepcopy(self.GT_model)
        self.sparse_target_model.meshes[0].materials[0].albedo = glm.vec3(0.5, 0.5, 0.5)
    
    def render(self):
        ith_motion = self.frame // self.frames_per_motion
        ith_frame = self.frame % self.frames_per_motion

        if self.show_GT:
            self.motion = self.GT_motion
            self.model = self.GT_model
            super().render(render_xray=self.show_skeleton)

        if self.show_pred:
            self.motion = self.pred_motion
            self.model = self.pred_model
            super().render(render_xray=self.show_skeleton)

        # draw target
        self.target_model.set_pose_by_source(self.GT_motion.poses[(ith_motion+1)*self.frames_per_motion - 1])
        Render.model(self.target_model).set_all_color_modes(False).set_all_alphas(0.5).draw()

        # draw sparse target
        cand = self.sparse_frames > ith_frame
        cand = self.sparse_frames[cand]
        if cand.numel() == 0:
            cand = self.sparse_frames.max()
        else:
            cand = cand.min()
        self.sparse_target_model.set_pose_by_source(self.pred_motion.poses[(ith_motion)*self.frames_per_motion + cand])
        Render.model(self.sparse_target_model).set_all_color_modes(False).set_all_alphas(0.5).draw()
    
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