import os
import pickle
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from pymovis.motion import BVH
from pymovis.ops import rotation, motionops
from pymovis.utils import util

from utility.dataset import MotionDataset
from utility.config import Config

def get_interpolated_motion(feature_from, feature_to, num_frames):
    B, T, D = feature_from.shape
    local_R6_from, root_p_from = feature_from[..., :-3], feature_from[..., -3:]
    local_R6_to, root_p_to = feature_to[..., :-3], feature_to[..., -3:]
    t = torch.linspace(0, 1, num_frames, device=feature_from.device)[None, :, None]

    # linear interpolation of root position between i-th keyframe and j-th keyframe
    p_dist = root_p_to - root_p_from
    p_approx = root_p_from + p_dist * t
    
    # rotation interpolation between i-th keyframe and j-th keyframe
    local_R_from = rotation.R6_to_R(local_R6_from.reshape(B, T, -1, 6))
    local_R_to   = rotation.R6_to_R(local_R6_to.reshape(B, T, -1, 6))
    R_dist = torch.matmul(local_R_from.transpose(-1, -2), local_R_to)
    angle, axis = rotation.R_to_A(R_dist)
    angle = angle * t
    axis = axis.repeat(1, num_frames, 1, 1)
    R_approx = torch.matmul(local_R_from, rotation.A_to_R(angle, axis))
    R6_approx = rotation.R_to_R6(R_approx)

    # concatenate root position and rotation
    R6_approx = R6_approx.reshape(B, num_frames, -1)
    p_approx = p_approx.reshape(B, num_frames, -1)
    feature_approx = torch.cat([R6_approx, p_approx], dim=-1)
    return feature_approx

def main():
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/context.json")
    
    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=False, config=config)
    skeleton   = dataset.skeleton

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # initial cost matrix
    results = []
    for GT_feature in tqdm(dataloader):
        GT_feature = GT_feature.to(device)

        feature = GT_feature[:, config.context_frames-1:]
        B, T, D = feature.shape
        local_R6, root_p = feature[..., :-3], feature[..., -3:]
        local_R6 = local_R6.reshape(B, T, -1, 6)
        _, global_p = motionops.R6_fk(local_R6, root_p, skeleton)

        E_init = torch.zeros(B, T, T, device=device)
        E_init.fill_(float("inf"))
        for frame_gap in tqdm(range(1, T)):
            frame_from = torch.arange(0, T-frame_gap)
            frame_to   = frame_from + frame_gap

            feature_from = feature[:, frame_from]
            feature_to   = feature[:, frame_to]
            
            N = T-frame_gap
            feature_from = feature_from.reshape(B*N, 1, D)
            feature_to   = feature_to.reshape(B*N, 1, D)

            # interpolate motion between i-th keyframe and j-th keyframe
            feature_approx = get_interpolated_motion(feature_from, feature_to, frame_gap+1)
            feature_approx = feature_approx.reshape(B, N, frame_gap+1, D)
            local_R6_approx, root_p_approx = feature_approx[..., :-3], feature_approx[..., -3:]
            local_R_approx = rotation.R6_to_R(local_R6_approx.reshape(-1, 6)).reshape(B, N, frame_gap+1, -1, 3, 3)
            root_p_approx = root_p_approx.reshape(B, N, frame_gap+1, 3)
            _, global_p_approx = motionops.R_fk(local_R_approx, root_p_approx, skeleton)

            # compute error between interpolated motion and original motion
            for i in range(0, T-frame_gap):
                j = i + frame_gap
                error = torch.norm(global_p_approx[:, i] - global_p[:, i:j+1], dim=-1)
                # error = torch.abs(feature_approx[:, i] - feature[:, i:j+1])
                error = torch.mean(error, dim=-1)
                E_init[:, i, j] = torch.max(error, dim=-1).values

        # dynamic programming for minimum cost path
        for b in tqdm(range(B)):
            E = E_init[b].clone()
            i = 0
            j = T - 1
            ks = [i, j]

            for m in range(1, T-1):
                error = float("inf")
                min_k, max_k = ks[0], ks[1]
                k_optimal = -1
                sorted_ks = sorted(ks)
                for idx, min_k in enumerate(sorted_ks[:-1]):
                    max_k = sorted_ks[idx+1]
                    for k in range(min_k, max_k):
                        curr_cost = E[min_k, k] + E[k, max_k]
                        if curr_cost < error:
                            error = curr_cost
                            k_optimal = k

                ks.append(k_optimal)
                
            # keyframe probability
            keyframe_prob = torch.linspace(0, 1, T-1)
            keyframe_prob = keyframe_prob[1:]
            probs = []

            for idx, k in enumerate(ks[2:]):
                probs.append(keyframe_prob[k-1].item())
            probs = [1] * config.context_frames + probs + [1]
            probs = torch.tensor(probs, dtype=torch.float32, device=device)

            new_feature = torch.cat([GT_feature[b], probs[:, None]], dim=-1)
            results.append(new_feature)
        break
    results = torch.stack(results, dim=0).cpu().numpy()
    np.save(f"dataset/train/keyframe_length{config.window_length}_offset{config.window_offset}_fps{config.fps}.npy", results)

if __name__ == "__main__":
    main()