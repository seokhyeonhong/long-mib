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

def insert_keyframes(S, cost, key, kfs, error):
    S[key] = kfs
    cost[key[0], key[1]] = error

def get_salient_poses(b, T, E_init):
    S = {}
    cost = np.zeros((T, T))
    for e in range(2, T+1):
        if e == 2:
            insert_keyframes(S, cost, (2, e-1), np.array([0, e-1]), E_init[b, 0, e-1])

        else:
            insert_keyframes(S, cost, (2, e-1), np.array([0, e-1]), E_init[b, 0, e-1])
            insert_keyframes(S, cost, (e-1, e-1), np.arange(e-1), 0)

            for k in range(3, e):
                # compute argmin (S(k-1, j) + {e})
                min_cost = float("inf")
                jstar = -1

                js = np.arange(k-1, e-1)
                temp = []
                for j in js:
                    temp.append(S[(k-1, j)][-1])
                temp = np.array(temp)

                error = np.maximum(cost[(k-1, js)], E_init[b, temp, e-1])
                min_cost = np.min(error)
                jstar = js[np.argmin(error)]

                kfselection = np.append(S[(k-1, jstar)], np.array([e-1]))
                insert_keyframes(S, cost, (k, e-1), kfselection, min_cost)
    return S

def get_keyframes(config, train=True):
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # dataset
    print("Loading dataset...")
    dataset    = MotionDataset(train=train, config=config)
    skeleton   = dataset.skeleton

    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    # directory to save
    save_dir = config.keyframe_train_dir if train else config.keyframe_test_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # get salient keyframes
    for idx, GT_feature in tqdm(enumerate(dataloader)):
        GT_feature = GT_feature[:, :config.context_frames+config.max_transition+1]
        GT_feature = GT_feature.to(device)
        
        # split motion features to start from context frame
        feature = GT_feature[:, config.context_frames-1:, :-3] # except root trajectory
        B, T, D = feature.shape
        local_R6, root_p = torch.split(feature, [D-3, 3], dim=-1)
        local_R6 = local_R6.reshape(B, T, -1, 6)
        _, global_p = motionops.R6_fk(local_R6, root_p, skeleton)

        E_init = torch.zeros(B, T, T, device=device)
        E_init.fill_(float("inf"))
        for frame_gap in tqdm(range(1, T), leave=False):
            if frame_gap == 1:
                for i in range(T-1):
                    j = i + 1
                    E_init[:, i, j] = 0
                continue

            frame_from = torch.arange(0, T-frame_gap)
            frame_to   = frame_from + frame_gap

            feature_from = feature[:, frame_from]
            feature_to   = feature[:, frame_to]
            
            N = T-frame_gap
            feature_from = feature_from.reshape(B*N, 1, D)
            feature_to   = feature_to.reshape(B*N, 1, D)

            # piecewise interpolation motion between i-th keyframe and j-th keyframe
            feature_approx = get_interpolated_motion(feature_from, feature_to, frame_gap+1)
            feature_approx = feature_approx.reshape(B, N, frame_gap+1, D)
            local_R6_approx, root_p_approx = feature_approx[..., :-3], feature_approx[..., -3:]
            local_R_approx = rotation.R6_to_R(local_R6_approx.reshape(-1, 6)).reshape(B, N, frame_gap+1, -1, 3, 3)
            root_p_approx = root_p_approx.reshape(B, N, frame_gap+1, 3)
            _, global_p_approx = motionops.R_fk(local_R_approx, root_p_approx, skeleton)

            # GT motion features to compare
            global_ps = []
            for i in range(0, T-frame_gap):
                j = i + frame_gap
                global_ps.append(global_p[:, i:j+1])
            global_ps = torch.stack(global_ps, dim=1)

            # error between interpolated motion and GT motion
            error = torch.sum((global_p_approx - global_ps)**2, dim=-1)
            error = torch.sum(error, dim=-1)
            error = torch.max(error, dim=-1)[0]

            # store error
            for i in range(0, T-frame_gap):
                j = i + frame_gap
                E_init[:, i, j] = error[:, i]

        # dynamic programming for minimum cost path
        salient_poses = util.run_parallel_sync(get_salient_poses, range(B), T=T, E_init=E_init.cpu().numpy())
        save_data = []
        for sp in salient_poses:
            data = {}
            for k in range(3, T):
                data[k] = sp[(k, T-1)]
            save_data.append(data)
        with open(f"{save_dir}/{idx:08d}.pkl", "wb") as f:
            pickle.dump(save_data, f)

def generate_dataset(config, train=True):
    save_dir = config.keyframe_train_dir if train else config.keyframe_test_dir
    keyframe_data = []
    for file in sorted(os.listdir(save_dir)):
        if file.endswith(".pkl"):
            with open(f"{save_dir}/{file}", "rb") as f:
                keyframe_data.extend(pickle.load(f))

    # dataset
    dataset = MotionDataset(train=train, config=config)
    
    # save data
    save_features = []
    for idx, motion in tqdm(enumerate(dataset)):
        motion = motion[:config.context_frames+config.max_transition+1]
        T, D = motion.shape
        local_R6, root_p, traj = torch.split(motion, [D-6, 3, 3], dim=-1)

        keyframes = keyframe_data[idx]
        prob = torch.zeros(T - config.context_frames + 1)
        for k in range(3, config.max_transition+1):
            kfs = keyframes[k]
            prob[kfs] += 1
        
        # prob 값대로 순위를 매기기
        rank_idx = torch.argsort(prob, descending=True)
        tier = [-1] * len(rank_idx)
        for i in range(len(rank_idx)):
            val = i
            if i > 0 and prob[rank_idx[i]] == prob[rank_idx[i-1]]:
                val = tier[rank_idx[i-1]]
            tier[rank_idx[i]] = val
        
        tier = torch.tensor(tier)
        tier = (config.max_transition+1 - tier) / (config.max_transition+1)


        # prob = prob / (config.max_transition-2)
        # breakpoint()
        prob = torch.cat([torch.ones(config.context_frames-1), tier])
        prob = prob.unsqueeze(-1)

        feature = torch.cat([local_R6, root_p, prob, traj], dim=-1).cpu().numpy()
        save_features.append(feature)

    save_features = np.stack(save_features, axis=0)
    np.save(config.keyframe_trainset_npy if train else config.keyframe_testset_npy, save_features)
    print(f"save_features.shape: {save_features.shape} saved to {config.keyframe_trainset_npy if train else config.keyframe_testset_npy}")

def main():
    config = Config.load("configs/traj_context.json")

    # get_keyframes(config, train=True)
    generate_dataset(config, train=True)

    # get_keyframes(config, train=False)
    generate_dataset(config, train=False)

if __name__ == "__main__":
    main()