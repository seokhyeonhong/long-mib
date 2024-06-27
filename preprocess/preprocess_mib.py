from __future__ import annotations

import sys; sys.path.append(".")

import os
import numpy as np
import pickle
from tqdm import tqdm
import argparse
from aPyOpenGL import agl, transforms as trf

import torch
from torch.utils.data import DataLoader

from utils import utils
from utils.dataset import PAEDataset
from model.pae import PAE

global batch_size

def load_phase(train, dataset):
    phase_cfg = utils.load_config(f"config/{dataset}/pae.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset = PAEDataset(train=train, config=phase_cfg)
    dataloader = DataLoader(dataset, batch_size=phase_cfg.batch_size, shuffle=False)
    mean, std = dataset.motion_statistics()
    mean, std = mean.to(device), std.to(device)

    # model
    model = PAE(
        input_channels=dataset.motion_dim,
        phase_channels=phase_cfg.phase_channels,
        num_frames=dataset.num_frames,
        time_duration=1.).to(device)
    utils.load_model(model, phase_cfg)

    # forward
    phases = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting phase", leave=False):
            batch = batch.to(device)
            batch = (batch - mean) / std
            *_, params = model.forward(batch)

            amp, shift = params[1], params[3] # (B, phase_channels, 1) for each

            # phase
            phase_x = amp * torch.sin(2*torch.pi * shift)
            phase_y = amp * torch.cos(2*torch.pi * shift)
            phase = torch.cat([phase_x, phase_y], dim=-1) # (B, phase_channels, 2)

            phases.append(phase.cpu().numpy())
    
    phases = np.concatenate(phases, axis=0) # (B, phase_channels, 2)
    phases = phases.reshape(phases.shape[0], -1) # (B, phase_channels*2)

    return phases

def split_features(motions: list[agl.Motion], phase_latents, window_length, window_offset, fps=30):
    # split features
    local_quats, root_pos, phases = [], [], []
    phase_from = 0
    for motion in tqdm(motions, desc="Splitting motions", leave=False):
        # local quaternions and root position
        lqs, rp = [], []
        for pose in motion.poses[fps:-fps]:
            lqs.append(pose.local_quats)
            rp.append(pose.root_pos)

        lqs = np.stack(lqs, axis=0)
        rp = np.stack(rp, axis=0)

        # phase latents
        phase = phase_latents[phase_from:phase_from + motion.num_frames - fps*2]

        # split
        for i in range(0, motion.num_frames - fps*2 - window_length + 1, window_offset):
            local_quats.append(lqs[i:i+window_length])
            root_pos.append(rp[i:i+window_length])
            phases.append(phase[i:i+window_length])
        
        phase_from += motion.num_frames - fps*2
    
    if phase_from != phase_latents.shape[0]:
        raise ValueError(f"phase_from != phase_latents.shape[0] ({phase_from} != {phase_latents.shape[0]})")
    
    # stack
    local_quats = np.stack(local_quats, axis=0) # (B, T, J, 4)
    root_pos    = np.stack(root_pos, axis=0)   # (B, T, 3)
    phases      = np.stack(phases, axis=0) # (B, T, phase_channels*4)

    return local_quats, root_pos, phases

def align_features(local_quats, root_pos, context_frames):
    # align root local rotation matrix
    fwd = trf.n_quat.mul_vec(local_quats[:, context_frames-1:context_frames, 0], np.array([0, 0, 1])) # (B, 1, 3)
    fwd = fwd * np.array([1, 0, 1])
    fwd = fwd / (np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-8)

    delta_quat = trf.n_quat.between_vecs(fwd, np.array([0, 0, 1])) # (B, 1, 4)

    local_quats[:, :, 0] = trf.n_quat.mul(delta_quat, local_quats[:, :, 0]) # (B, T, 4)

    # align root position
    delta_pos = root_pos[:, context_frames-1:context_frames] * np.array([1, 0, 1]) # (B, 1, 3)
    root_pos = trf.n_quat.mul_vec(delta_quat, root_pos - delta_pos) # (B, T, 3)

    return local_quats, root_pos # (B, T, J, 4), (B, T, 3)

def compute_piecewise_error_table(local_quats, root_pos, skeleton, context_frames):
    local_quats = local_quats[:, context_frames-1:].copy() # (B, 1, J, 4)
    root_pos = root_pos[:, context_frames-1:].copy() # (B, 1, 3)
    B, T, J, _ = local_quats.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res = []
    
    for b in tqdm(range(0, B, batch_size), desc="Computing piecewise error table", leave=False):
        GT_lq = torch.from_numpy(local_quats[b:b+batch_size]).to(device) # (B, T, J, 4)
        GT_rp = torch.from_numpy(root_pos[b:b+batch_size]).to(device)
        _, GT_gp = trf.t_quat.fk(GT_lq, GT_rp, skeleton) # (B, T, J, 3)

        # table to store error
        error_table = torch.zeros((GT_lq.shape[0], T, T), device=device, dtype=torch.float32)

        # set column <= row indices to inf
        row_indices = torch.arange(T)[:, None] # (T, 1)
        col_indices = torch.arange(T)[None, :] # (1, T)
        mask = row_indices >= col_indices
        error_table[:, mask] = torch.inf

        # case 1: frame gap == 1
        i = torch.arange(T-1)
        j = i + 1
        error_table[:, i, j] = 0

        # case 2: frame gap > 1
        for frame_gap in range(2, T):
            N = frame_gap + 1

            # unfold the original data
            lq = GT_lq.unfold(1, N, 1) # (B, num_chunk, J, 4, N)
            rp = GT_rp.unfold(1, N, 1) # (B, num_chunk, 3, N)
            gp = GT_gp.unfold(1, N, 1) # (B, num_chunk, J, 3, N)

            lq = lq.permute(0, 1, 4, 2, 3) # (B, num_chunk, N, J, 4)
            rp = rp.permute(0, 1, 3, 2)    # (B, num_chunk, N, 3)
            gp = gp.permute(0, 1, 4, 2, 3) # (B, num_chunk, N, J, 3)

            # interpolation weights
            t = torch.linspace(0, 1, N, dtype=torch.float32, device=device)

            # local quaternions (B, num_chunk, num_frames, J, 4)
            lq_approx = trf.t_quat.interpolate(lq[:, :, 0], lq[:, :, -1], t) # (B, num_chunk, J, 4, N)
            lq_approx = lq_approx.permute(0, 1, 4, 2, 3) # (B, num_chunk, N, J, 4)

            # root position (B, num_chunk, num_frames, 3)
            rp_from = rp[:, :, 0:1, :] # (B, num_chunk, 1, 3)
            rp_to   = rp[:, :, -1:, :]
            rp_approx = rp_from + (rp_to - rp_from) * t[None, None, :, None] # (B, num_chunk, num_frames, 3)

            # global positions (B, num_chunk, num_frames, J, 3)
            _, gp_approx = trf.t_quat.fk(lq_approx, rp_approx, skeleton)

            # compute error
            error = torch.norm(gp_approx - gp, dim=-1) # (B, num_chunk, num_frames, J)
            error = torch.sum(error, dim=-1) # (B, num_chunk, num_frames)
            error = torch.mean(error, dim=-1) # (B, num_chunk)
            # error = torch.sum(error, dim=-1) # (B, num_chunk)

            # update error table
            num_chunk = error.shape[1]
            i_indices = torch.arange(num_chunk)
            j_indices = i_indices + frame_gap
            error_table[:, i_indices, j_indices] = error

        res.append(error_table)
    
    error_table = torch.cat(res, dim=0)
    return error_table

def compute_pairwise_dist(error_table):
    B, T, _ = error_table.shape # (batch, frame_from, frame_to)
    device = error_table.device

    
    dist = []
    for b_ in tqdm(range(0, B, batch_size), desc="Computing pairwise distance", leave=False):
        e_tab = error_table[b_:b_+batch_size] # (batch, frame_from, frame_to)
        b = e_tab.shape[0]

        # cost table
        # dimension: (batch, num_keyframes, last_frame_index)
        E = torch.zeros((b, T, T), device=device, dtype=torch.float32)
        row_indices = torch.arange(T)[:, None] # (T, 1)
        col_indices = torch.arange(T)[None, :] # (1, T)
        mask = row_indices > col_indices
        E[:, mask] = torch.inf

        # notation: E(m, i, j) and e(i, j)
        # i < k < j must hold
        # Here i is fixed as zero, so E(m, 0, j) = E(m, j)

        # base case: no keyframe, which means E(0, j) = e(0, j)
        E[:, 0] = e_tab[:, 0]

        # general case: keyframe exists
        # E(m, j) = min_k (E(m-1, k) + e(k, j)) where 0 < k < j
        for m in range(1, T):
            for j in range(m+1, T):
                min_val = torch.min(E[:, m-1, 1:j] + e_tab[:, 1:j, j], dim=-1).values
                E[:, m, j] = min_val
        
        dist.append(E)

    dist = torch.cat(dist, dim=0)
    return dist

def compute_keyframe_scores(salient_poses, error_table):
    B, T, _ = error_table.shape # (batch, frame_from, frame_to)
    device = error_table.device

    # find the optimal keyframe via iterative backtracking
    # j is fixed as T-1
    # keyframe_idx at m keyframes: argmin_k (E(m-1, k) + e(k, T-1))

    # Salient Poses: batchified backtracking
    
    kf_scores = []
    for b_ in tqdm(range(0, B, batch_size), desc="Computing salient pose and keyframe scores", leave=False):
        sp = salient_poses[b_:b_+batch_size] # (batch, num_keyframes, num_frames, J, 4)
        e_tab = error_table[b_:b_+batch_size] # (batch, frame_from, frame_to)
        b = e_tab.shape[0]
            
        kf_indices = {} # key: (num_keyframes, end_frame), value: tensor of keyframe indices (B, num_keyframes)
        for m in range(T-1):
            for j in range(m+1, T):
                if m == 0:
                    kf_indices[(m, j)] = torch.tensor([[0, j] for _ in range(b)], dtype=torch.long, device=device)
                else:
                    min_val, min_idx = torch.min(sp[:, m-1, m:j] + e_tab[:, m:j, j], dim=-1)
                    kf_before = torch.cat([kf_indices[(m-1, min_idx[batch].item() + m)][batch][None, :] for batch in range(b)], dim=0)
                    kf_added  = torch.tensor([[j] for _ in range(b)], dtype=torch.long, device=device)
                    kf_indices[(m, j)] = torch.cat([kf_before, kf_added], dim=-1)

        # find the optimal keyframe
        counts = torch.zeros((b, T), dtype=torch.int32, device=device)
        for num_kf in range(1, T-1):
            kf = kf_indices[(num_kf, T-1)]
            counts[torch.arange(b)[:, None], kf] += 1

        scores = counts / (T-2)
        kf_scores.append(scores)
    
    kf_scores = torch.cat(kf_scores, dim=0)
    kf_scores = torch.cat([torch.ones_like(kf_scores[:, :9]), kf_scores], dim=1) # (B, T)
    return kf_scores[..., None].cpu().numpy()


def get_keyframe_scores(local_quats, root_pos, skeleton, context_frames):
    error_table = compute_piecewise_error_table(local_quats, root_pos, skeleton, context_frames)
    dist = compute_pairwise_dist(error_table)
    scores = compute_keyframe_scores(dist, error_table)
    return scores

def get_motion_features(local_quats, root_pos):
    B, T, J, _ = local_quats.shape

    # motion features
    local_ortho6ds = trf.n_quat.to_ortho6d(local_quats) # (B, T, J, 6)
    local_ortho6ds = local_ortho6ds.reshape(B, T, -1) # (B, T, J*6)
    motion_features = np.concatenate([local_ortho6ds, root_pos], axis=-1) # (B, T, J*6+3)
    
    # basis position
    basis_pos = root_pos * np.array([1, 0, 1], dtype=np.float32) # (B, T, 3)
    basis_pos = basis_pos[..., (0, 2)] # xz only

    # basis direction
    root_fwd = trf.n_quat.mul_vec(local_quats[:, :, 0], np.array([0, 0, 1], dtype=np.float32)) # (B, T, 3)
    root_fwd = root_fwd * np.array([1, 0, 1], dtype=np.float32)
    root_fwd = root_fwd / (np.linalg.norm(root_fwd, axis=-1, keepdims=True) + 1e-8) # (B, T, 3)
    
    basis_dir = root_fwd[..., (0, 2)] # xz only

    # traj features
    traj_features = np.concatenate([basis_pos, basis_dir], axis=-1) # (B, T, 3)

    return motion_features, traj_features

def preprocess(config, dataset, train=True):
    # load phase variables
    phases = load_phase(train, dataset)

    # load motions
    if dataset == "lafan1":
        fbx_path = os.path.join(config.dataset_dir, "train.fbx" if train else "test.fbx")
        fbx = agl.FBX(fbx_path)
        motions = fbx.motions()
        skeleton = fbx.skeleton()

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
    
    # get features
    local_quats, root_pos, phases = split_features(motions, phases, config.window_length, config.window_offset, fps=fps)
    local_quats, root_pos = align_features(local_quats, root_pos, config.context_frames)
    # kf_scores = get_keyframe_scores(local_quats, root_pos, skeleton, config.context_frames)
    kf_scores = np.ones((local_quats.shape[0], local_quats.shape[1], 1), dtype=np.float32)
    motion_features, traj_features = get_motion_features(local_quats, root_pos)

    # save
    save_dir = os.path.join(config.dataset_dir, "MIB")
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, f"{'train' if train else 'test'}-{config.npz_path}"), motion=motion_features, phase=phases, traj=traj_features, scores=kf_scores)
    print(f"> Saved {'train' if train else 'test'} dataset (motion: {motion_features.shape}, phase: {phases.shape}, traj: {traj_features.shape}, scores: {kf_scores.shape})")

    # # save skeleton
    # if train:
    #     with open(os.path.join(config.dataset_dir, "skeleton.pkl"), "wb") as f:
    #         pickle.dump(skeleton, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    config = utils.load_config(f"config/{args.dataset}/default.yaml")
    batch_size = config.batch_size
    preprocess(config, args.dataset, train=True)
    preprocess(config, args.dataset, train=False)