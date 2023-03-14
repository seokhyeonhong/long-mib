import os
import json
from datetime import datetime
import numpy as np

from pymovis.vis.const import INCH_TO_METER

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, path):
        """
        Load config from json file and convert to Config object
        
        Args:
            path (str): path to json file
            postfix (str): postfix to add to log_dir and save_dir. Specified when test, inference, or continue training

        Returns:
            Config: config object
        """

        with open(path, "r") as f:
            config = json.loads(f.read())
        
        config = cls(config)

        # make list to numpy array
        config.v_forward = np.array(config.v_forward).astype(np.float32)
        config.v_up      = np.array(config.v_up).astype(np.float32)

        # train/test dataset
        config.trainset_dir = os.path.join(config.dataset_dir, "train")
        config.testset_dir  = os.path.join(config.dataset_dir, "test")
        config.trainset_npy = f"{config.trainset_dir}/length{config.window_length}_offset{config.window_offset}_fps{config.fps}.npy"
        config.testset_npy  = f"{config.testset_dir}/length{config.window_length}_offset{config.window_offset}_fps{config.fps}.npy"

        # directories to save and log training
        postfix = f"transition{config.min_transition}-{config.max_transition}"
        config.log_dir  = os.path.join(config.log_dir, postfix)
        config.save_dir = os.path.join(config.save_dir, postfix)

        return config

    def write(self, path):
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                self[key] = value.tolist()
                
        with open(path, "w") as f:
            f.write(json.dumps(self, indent=4))