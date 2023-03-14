import sys
sys.path.append(".")
sys.path.append("..")

import torch
from torch.utils.data import DataLoader

from pymovis.motion import Motion, FBX
from pymovis.vis import AppManager, MotionApp, YBOT_FBX_DICT
from pymovis.ops import rotation

from utility.config import Config
from utility.dataset import MotionDataset
from vis.visapp import SingleMotionApp

if __name__ == "__main__":
    config = Config.load("configs/sparse.json")
    character = FBX("dataset/ybot.fbx")
    dataset = MotionDataset(train=True, config=config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    skeleton = dataset.skeleton

    sparse_frames = torch.arange(config.max_transition // config.fps) * config.fps
    sparse_frames += (config.context_frames-1) + config.fps
    sparse_frames = torch.cat([torch.arange(config.context_frames), sparse_frames])

    for feature in dataloader:
        feature = feature[:, sparse_frames]
        B, T, D = feature.shape

        local_R6, root_p = torch.split(feature, [D-3, 3], dim=-1)
        local_R = rotation.R6_to_R(local_R6.reshape(B, T, D//6, 6))

        local_R = local_R.reshape(B*T, -1, 3, 3)
        root_p = root_p.reshape(B*T, 3)
        motion = Motion.from_torch(skeleton, local_R, root_p)

        app_manager = AppManager()
        app = MotionApp(motion, character.model(), YBOT_FBX_DICT)
        app_manager.run(app)