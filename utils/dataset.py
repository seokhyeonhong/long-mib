import os
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from aPyOpenGL import agl, transforms as trf

class MotionDataset(Dataset):
    def __init__(self, train, config, verbose=True):
        self.train = train
        self.config = config

        # load features
        features = np.load(os.path.join(config.dataset_dir, "MIB", f"{'train' if train else 'test'}-{config.npz_path}"))

        self.motion = torch.from_numpy(features["motion"]).float() # (B, T, 6J+3)
        self.phase = torch.from_numpy(features["phase"]).float()   # (B, T, 2P) where P is the number of phase channels
        self.traj = torch.from_numpy(features["traj"]).float()     # (B, T, 4)
        self.score = torch.from_numpy(features["scores"]).float()  # (B, T, 1)

        # if "human36m" in self.config.dataset_dir:
        #     if self.config.window_offset != 1:
        #         raise ValueError("window_offset must be 1 for human36m dataset")
        #     self.motion = self.motion[::2]
        #     self.phase = self.phase[::2]
        #     self.traj = self.traj[::2]
        #     self.score = self.score[::2]

        if verbose:
            print("Shapes:")
            print(f"\t- motion.shape: {self.motion.shape}")
            print(f"\t- phase.shape: {self.phase.shape}")
            print(f"\t- traj.shape: {self.traj.shape}")
            print(f"\t- score.shape: {self.score.shape}")

        # dimensions
        self.num_frames = self.motion.shape[1]
        self.motion_dim = self.motion.shape[2]
        self.phase_dim = self.phase.shape[2]
        self.traj_dim = self.traj.shape[2]
        self.score_dim = self.score.shape[2]

        # load skeletons
        self.skeleton = self.load_skeleton(os.path.join(config.dataset_dir, "skeleton.pkl"))
    
    def load_skeleton(self, path) -> agl.Skeleton:
        if not os.path.exists(path):
            print(f"Cannot find skeleton from {path}")
            return None
        
        with open(path, "rb") as f:
            skeleton = pickle.load(f)
            
        return skeleton
    
    def __len__(self):
        return len(self.motion)
    
    def __getitem__(self, idx):
        res = {
            "motion": self.motion[idx],
            "phase": self.phase[idx],
            "traj": self.traj[idx],
            "score": self.score[idx],
        }
        return res
    
    def motion_statistics(self, device="cuda"):
        if os.path.exists(os.path.join(self.config.dataset_dir, "MIB", "motion_statistics.pth")):
            mean = torch.load(os.path.join(self.config.dataset_dir, "MIB", "motion_statistics.pth"))["mean"].to(device)
            std = torch.load(os.path.join(self.config.dataset_dir, "MIB", "motion_statistics.pth"))["std"].to(device)
        else:
            motion = MotionDataset(train=True, config=self.config, verbose=False).motion.to(device)
            mean = torch.mean(motion, dim=(0, 1))
            std = torch.std(motion, dim=(0, 1)) + 1e-8
            torch.save({"mean": mean.cpu(), "std": std.cpu()}, os.path.join(self.config.dataset_dir, "MIB", "motion_statistics.pth"))
        return mean, std

    def traj_statistics(self, device="cuda"):
        if os.path.exists(os.path.join(self.config.dataset_dir, "MIB", "traj_statistics.pth")):
            mean = torch.load(os.path.join(self.config.dataset_dir, "MIB", "traj_statistics.pth"))["mean"].to(device)
            std = torch.load(os.path.join(self.config.dataset_dir, "MIB", "traj_statistics.pth"))["std"].to(device)
        else:
            traj = MotionDataset(train=True, config=self.config, verbose=False).traj.to(device)
            mean = torch.mean(traj, dim=(0, 1))
            std = torch.std(traj, dim=(0, 1)) + 1e-8
            torch.save({"mean": mean.cpu(), "std": std.cpu()}, os.path.join(self.config.dataset_dir, "MIB", "traj_statistics.pth"))
        return mean, std
    
    def l2p_statistics(self, device="cuda"):
        if os.path.exists(os.path.join(self.config.dataset_dir, "MIB", "l2p_statistics.pth")):
            mean = torch.load(os.path.join(self.config.dataset_dir, "MIB", "l2p_statistics.pth"))["mean"].to(device)
            std = torch.load(os.path.join(self.config.dataset_dir, "MIB", "l2p_statistics.pth"))["std"].to(device)
        else:
            motion = MotionDataset(train=True, config=self.config, verbose=False).motion
            dloader = DataLoader(motion, batch_size=self.config.batch_size, shuffle=False)
            global_pos = []
            for batch in dloader:
                batch = batch.to(device)
                B, T, D = batch.shape
                local_ortho6ds, root_pos = torch.split(batch, [D-3, 3], dim=-1)
                local_ortho6ds = local_ortho6ds.reshape(B, T, -1, 6)
                _, gp = trf.t_ortho6d.fk(local_ortho6ds, root_pos, self.skeleton)
                global_pos.append(gp)

            global_pos = torch.cat(global_pos, dim=0) # (B, T, J, 3)
            mean = torch.mean(global_pos, dim=(0, 1))
            std = torch.std(global_pos, dim=(0, 1)) + 1e-8
            torch.save({"mean": mean.cpu(), "std": std.cpu()}, os.path.join(self.config.dataset_dir, "MIB", "l2p_statistics.pth"))
        return mean, std
    
class PAEDataset(Dataset):
    def __init__(self, train, config):
        self.train = train
        self.config = config

        # load features
        features = np.load(os.path.join(config.dataset_dir, "PAE", f"{'train' if train else 'test'}-{config.npz_path}"))
        self.features = torch.from_numpy(features["motion"]).float() # (B, T, 3J)

        # if "human36m" in self.config.dataset_dir:
        #     if self.config.window_offset != 1:
        #         raise ValueError("window_offset must be 1 for human36m dataset")
        #     self.features = self.features[::2]

        # dimensions
        self.num_frames = self.features.shape[1]
        self.motion_dim = self.features.shape[2]
        print(f"Shapes:")
        print(f"\t- features.shape: {self.features.shape}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    def motion_statistics(self, device="cuda"):
        feat = PAEDataset(train=True, config=self.config).features
        mean = torch.mean(feat, dim=(0, 1))
        std = torch.std(feat, dim=(0, 1)) + 1e-8
        return mean.to(device), std.to(device)