import os
import pickle
import numpy as np
import random

import torch
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    """
    Motion dataset for training and testing
    Features:
        - motion features (number of joints * 6 + 3) for each frame, 6D orientations and a 3D translation vector
        - trajectory features (5) for each frame, a 3D forward vector and a 2D xz position vector
    """
    def __init__(self, train, config):
        self.train  = train
        self.config = config

        with open(os.path.join(self.config.dataset_dir, "skeleton.pkl"), "rb") as f:
            self.skeleton = pickle.load(f)

        self.features = torch.from_numpy(np.load(config.trainset_npy if train else config.testset_npy))
        self.shape = self.features.shape
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

    def statistics(self, dim=(0, 1)):
        print(f"Calculating MotionDataset mean and std, dim={dim}...")

        # calculate statistics from training set
        trainset = MotionDataset(True, self.config)

        # mean and std
        X = torch.stack([trainset[i] for i in range(len(trainset))], dim=0)
        mean = torch.mean(X, dim=dim)
        std = torch.std(X, dim=dim) + 1e-8

        return mean, std

class KeyframeDataset(Dataset):
    """
    Motion dataset for training and testing
    Features:
        - motion features (number of joints * 6 + 3) for each frame, 6D orientations and a 3D translation vector
        - keyframe probability (1) for each frame
        - trajectory features (5) for each frame, xz and forward vector
    """
    def __init__(self, train, config):
        self.train  = train
        self.config = config

        with open(os.path.join(self.config.dataset_dir, "skeleton.pkl"), "rb") as f:
            self.skeleton = pickle.load(f)

        self.features = torch.from_numpy(np.load(config.keyframe_trainset_npy if train else config.keyframe_testset_npy))
        self.shape = self.features.shape
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

    def statistics(self, dim=(0, 1)):
        print(f"Calculating KeyframeDataset mean and std, dim={dim}...")

        # calculate statistics from training set
        trainset = KeyframeDataset(True, self.config)

        # mean and std
        X = torch.stack([trainset[i] for i in range(len(trainset))], dim=0)
        mean = torch.mean(X, dim=dim)
        std = torch.std(X, dim=dim) + 1e-8

        return mean, std

class KeyframePairDataset(Dataset):
    """
    Motion dataset for training and testing
    Features:
        - motion features (number of joints * 6 + 3) for each frame, 6D orientations and a 3D translation vector
        - keyframe probability (1) for each frame
        - trajectory features (5) for each frame, xz and forward vector
    """
    def __init__(self, train, config):
        self.train  = train
        self.config = config

        with open(os.path.join(self.config.dataset_dir, "skeleton.pkl"), "rb") as f:
            self.skeleton = pickle.load(f)

        self.features = torch.from_numpy(np.load(config.keyframe_trainset_npy if train else config.keyframe_testset_npy))
        self.shape = self.features.shape
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item1 = self.features[idx]
        item2 = self.features[random.randint(0, len(self.features)-1)]
        return item1, item2

    def statistics(self, dim=(0, 1)):
        print(f"Calculating KeyframePairDataset mean and std, dim={dim}...")

        # calculate statistics from training set
        trainset = KeyframePairDataset(True, self.config)

        # mean and std
        X = torch.stack([trainset[i][0] for i in range(len(trainset))], dim=0)
        mean = torch.mean(X, dim=dim)
        std = torch.std(X, dim=dim) + 1e-8

        return mean, std