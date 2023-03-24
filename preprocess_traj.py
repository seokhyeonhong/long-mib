import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from pymovis.motion import BVH
from pymovis.ops import rotation
from pymovis.utils import util, torchconst

from utility.dataset import KeyframeDataset
from utility.config import Config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/context.json")
    
    # dataset
    print("Loading dataset...")
    dataset    = KeyframeDataset(train=False, config=config)
    v_forward  = torch.from_numpy(config.v_forward).to(device)

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    features = []
    for GT_feature in tqdm(dataloader):
        B, T, D = GT_feature.shape

        # get GT features
        GT_feature = GT_feature.to(device)
        local_R6, root_p, kf_prob = torch.split(GT_feature, [D-4, 3, 1], dim=-1)

        # get trajectory features
        local_R = rotation.R6_to_R(local_R6.reshape(B, T, -1, 6))
        forward = F.normalize(torch.matmul(local_R[:, :, 0], v_forward) * torchconst.XZ(device), dim=-1)
        xz      = root_p[..., (0, 2)]

        # make features
        feature = torch.cat([local_R6, root_p, kf_prob, xz, forward], dim=-1)
        features.append(feature)

    features = torch.cat(features, dim=0).cpu().numpy()
    np.save(config.traj_testset_npy, features)
        
if __name__ == "__main__":
    main()