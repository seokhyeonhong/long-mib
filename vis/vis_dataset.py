import sys
sys.path.append(".")
sys.path.append("..")

from pymovis.motion import Motion, FBX
from pymovis.vis import AppManager, MotionApp, YBOT_FBX_DICT
from pymovis.ops import rotation

from utility.config import Config
from utility.dataset import MotionDataset

if __name__ == "__main__":
    config = Config.load("configs/config.json")
    character = FBX("dataset/ybot.fbx")
    dataset = MotionDataset(train=True, config=config)

    for feature in dataset:
        local_R6, root_p = feature[:, :-3], feature[:, -3:]
        T, D = local_R6.shape
        local_R = rotation.R6_to_R(local_R6.reshape(T, D//6, 6))
        motion = Motion.from_torch(dataset.skeleton, local_R, root_p)

        app_manager = AppManager()
        app = MotionApp(motion, character.model(), YBOT_FBX_DICT)
        app_manager.run(app)