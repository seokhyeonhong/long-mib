import os
import pickle
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pymovis.motion import BVH
from pymovis.ops import rotation, motionops
from pymovis.utils import util

from utility.config import Config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_path = "./dataset/salient/ybot_fallAndGetUp3_subject1.bvh"
    motion = BVH.load(file_path)
    motion.poses = motion.poses[200:250]

    import time
    start = time.perf_counter()
    skeleton = motion.skeleton
    local_R = np.stack([pose.local_R for pose in motion.poses], axis=0).reshape(len(motion), -1, 3, 3)
    root_p  = np.stack([pose.root_p for pose in motion.poses], axis=0).reshape(len(motion), -1)

    local_R = torch.from_numpy(local_R).to(device) # (T, J, 3, 3)
    root_p  = torch.from_numpy(root_p).to(device) # (T, 3)
    _, global_p = motionops.R_fk(local_R, root_p, skeleton)

    # error table
    Es = []
    e = torch.zeros(len(motion), len(motion), device=device)

    for i in tqdm(range(len(motion))):
        for j in range(i+1, len(motion)):
            # linear interpolation of root position between i-th keyframe and j-th keyframe
            p_dist = root_p[j] - root_p[i]
            t = torch.linspace(0, 1, j-i+1, device=device)
            p_approx = root_p[i] + p_dist * t[:, None]

            # rotation interpolation between i-th keyframe and j-th keyframe
            R_dist = torch.matmul(local_R[i].transpose(-2, -1), local_R[j])
            angle, axis = rotation.R_to_A(R_dist)
            angle_approx = angle * t[:, None]
            axis = axis.unsqueeze(0).repeat(abs(j-i+1), 1, 1)
            R_dist = rotation.A_to_R(angle_approx, axis) # (j-i+1, J, 3, 3)
            R_approx = torch.matmul(local_R[None, i], R_dist)

            # error between i-th keyframe and j-th keyframe
            p_approx = p_approx.reshape(-1, 3)
            R_approx = R_approx.reshape(-1, 3, 3)
            _, global_p_approx = motionops.R_fk(R_approx, p_approx, skeleton)
            
            error = torch.norm(global_p_approx - global_p[i:j+1], dim=-1)
            error = torch.sum(error, dim=-1)
            
            e[i, j] = torch.max(error)

    Es.append(e)

    # dynamic programming - minimize path cost increasing m
    num_keyframes = 30
    for m in range(num_keyframes):
        E = torch.empty(len(motion), len(motion), device=device)
        E[:] = float("inf")

        for i in range(len(motion)):
            for j in range(i+1, len(motion)):
                E[i, j] = torch.min(Es[-1][i, i:j] + e[i:j, j])
                
if __name__ == "__main__":
    main()