import os
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset

class MotionDataset(Dataset):
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
        # load and return mean and std if they exist
        mean_path = os.path.join(self.config.dataset_dir, f"mean_{dim}.pt")
        std_path  = os.path.join(self.config.dataset_dir, f"std_{dim}.pt")

        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean = torch.load(mean_path)
            std  = torch.load(std_path)
            return mean, std
        
        if not self.train:
            raise ValueError("Mean and std must be calculated and saved on training set first")

        # load motion features and calculate mean and std
        X = torch.stack([self[i] for i in range(len(self))], dim=0)
        mean = torch.mean(X, dim=dim)
        std = torch.std(X, dim=dim) + 1e-8

        # save mean and std
        torch.save(mean, mean_path)
        torch.save(std, std_path)

        return mean, std