from __future__ import annotations

import sys
sys.path.append(".")

import os
import numpy as np
import argparse
from aPyOpenGL import agl, transforms as trf

from utils import utils

def get_features(motions: list[agl.Motion], skeleton: agl.Skeleton, window_length, window_offset, fps):
    # get root quaternions and global positions
    root_quats, global_pos = [], []
    for motion in motions:
        # local quaternions and root positions
        lqs, rp = [], []
        for pose in motion.poses:
            lqs.append(pose.local_quats)
            rp.append(pose.root_pos)
        lqs = np.stack(lqs, axis=0) # (T, J, 4)
        rp = np.stack(rp, axis=0)

        # solve fk
        _, gps = trf.n_quat.fk(lqs, rp, skeleton)

        # append
        root_quats.append(lqs[:, 0])
        global_pos.append(gps)

    # get basis quaternions
    basis_quats = []
    for rq in root_quats:
        fwd = trf.n_quat.mul_vec(rq, np.array([0, 0, 1], dtype=np.float32)) # (T, 3)
        fwd = fwd * np.array([1, 0, 1], dtype=np.float32)
        fwd = fwd / (np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-8)
        basis_quat = trf.n_quat.between_vecs(np.array([0, 0, 1], dtype=np.float32), fwd) # (T, 4)
        basis_quats.append(basis_quat[:, None, :])
    
    # transform global velocities to local velocities
    local_vels = []
    for i in range(len(global_pos)):
        gv = global_pos[i][1:] - global_pos[i][:-1] # (T-1, J, 3)
        gv = np.concatenate([np.zeros_like(gv[:1]), gv], axis=0) * fps # (T, J, 3)
        lv = trf.n_quat.mul_vec(trf.n_quat.inv(basis_quats[i]), gv) # (T, J, 3)
        local_vels.append(lv)
    
    # make windows
    windows = []
    for lv in local_vels:
        for i in range(0, lv.shape[0] - window_length + 1, window_offset):
            windows.append(lv[i:i+window_length])
    
    # stack
    windows = np.stack(windows, axis=0) # (B, window_length, J, 3)

    return windows

def preprocess(config, dataset, train=True):
    if dataset == "lafan1":
        # load motions
        fbx_path = os.path.join(config.dataset_dir, "train.fbx" if train else "test.fbx")
        fbx = agl.FBX(fbx_path)

        # get features
        motions = fbx.motions()
        skeleton = fbx.skeleton()
        fps = fbx.fps()

    elif dataset == "human36m":
        dataset_dir = config.dataset_dir
        motions = []
        fps = 30
        for file in os.listdir(os.path.join(dataset_dir, "train" if train else "test")):
            path = os.path.join(dataset_dir, "train" if train else "test", file)
            if file.endswith(".fbx"):
                fbx_file = agl.FBX(path)
                motions.extend(fbx_file.motions())
                fps = fbx_file.fps()
            elif file.endswith(".bvh"):
                bvh_file = agl.BVH(path, target_fps=fps, scale=0.1)
                motions.append(bvh_file.motion())
        skeleton = motions[0].skeleton
        
    elif dataset == "mann":
        # load motions
        dataset_dir = config.dataset_dir
        motions = []
        for file in os.listdir(dataset_dir):
            if file.endswith(".bvh"):
                if train and not "003.bvh" in file:
                    bvh_file = agl.BVH(os.path.join(dataset_dir, file), target_fps=30, scale=0.01)
                elif not train and "003.bvh" in file:
                    bvh_file = agl.BVH(os.path.join(dataset_dir, file), target_fps=30, scale=0.01)
                else:
                    continue

                motions.append(bvh_file.motion())
                fps = 30
        skeleton = motions[0].skeleton
    
    elif dataset == "100style":
        dataset_dir = os.path.join(config.dataset_dir, "train" if train else "test")
        motions = []
        for file in os.listdir(dataset_dir):
            if not file.endswith(".fbx"):
                continue

            # load motions
            fbx_path = os.path.join(dataset_dir, file)
            fbx = agl.FBX(fbx_path)

            # get features
            motions.extend(fbx.motions())
            skeleton = fbx.skeleton()
            fps = fbx.fps()

    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    # # get features
    # features = get_features(motions, skeleton, config.window_length, config.window_offset, fps=fps)
    # features = features.reshape(features.shape[0], features.shape[1], -1) # (B, window_length, J*3)

    # # save
    # save_dir = os.path.join(config.dataset_dir, "PAE")
    # os.makedirs(save_dir, exist_ok=True)
    # np.savez_compressed(os.path.join(save_dir, f"{'train' if train else 'test'}-{config.npz_path}"), motion=features)
    # print(f"Saved {'train' if train else 'test'} dataset (shape: {features.shape})")
    num_frames = 0
    for m in motions:
        num_frames += len(m.poses)
    print(f"Num frames: {num_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    config = utils.load_config(f"config/{args.dataset}/pae.yaml")
    preprocess(config, args.dataset, train=True)
    preprocess(config, args.dataset, train=False)