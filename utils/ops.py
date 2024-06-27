import torch
import torch.nn.functional as F
import numpy as np
import random
from aPyOpenGL import transforms as trf

def get_signed_angle(v_from, v_to, axis):
    v_from_ = v_from / (np.linalg.norm(v_from, axis=-1, keepdims=True) + 1e-8) # (..., 3)
    v_to_   = v_to / (np.linalg.norm(v_to,   axis=-1, keepdims=True) + 1e-8)   # (..., 3)

    dot = np.sum(v_from_ * v_to_, axis=-1, keepdims=True) # (...,)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(dot)

    cross = np.cross(v_from_, v_to_)
    cross = np.sum(cross * axis, axis=-1, keepdims=True)
    angle = np.where(cross < 0, -angle, angle)

    return angle

def get_signed_angle_torch(v_from, v_to, axis):
    v_from_ = F.normalize(v_from, dim=-1, eps=1e-8)
    v_to_   = F.normalize(v_to,   dim=-1, eps=1e-8)

    dot = torch.sum(v_from_ * v_to_, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7) # prevent nan during backward
    angle = torch.acos(dot)

    cross = torch.cross(v_from_, v_to_, dim=-1)
    cross = torch.sum(cross * axis, dim=-1, keepdim=True)
    angle = torch.where(cross < 0, -angle, angle)

    return angle

def fk(features, skeleton, context_frames):
    B, T, D = features.shape

    # split features
    basis_vel, basis_ang_vel, local_ortho6ds, local_root_pos = torch.split(features, [2, 1, skeleton.num_joints*6, 3], dim=-1)
    local_ortho6ds = local_ortho6ds.reshape(B, T, -1, 6)
    local_rotmats = trf.t_rotmat.from_ortho6d(local_ortho6ds)

    # basis
    basis_pos = torch.cumsum(basis_vel, dim=1)
    basis_pos = basis_pos - basis_pos[:, context_frames-1:context_frames]
    basis_pos = torch.cat([basis_pos[..., 0:1], torch.zeros_like(basis_pos[..., 0:1]), basis_pos[..., 1:]], dim=-1)

    basis_rot = torch.cumsum(basis_ang_vel, dim=1)
    basis_rot = basis_rot - basis_rot[:, context_frames-1:context_frames]
    basis_rotmat = trf.t_rotmat.from_aaxis(basis_rot * torch.tensor([0, 1, 0], dtype=torch.float32, device=features.device))

    # local rotations
    root_pre_quat = torch.from_numpy(skeleton.joints[0].pre_quat).float().to(features.device)
    root_pre_rotmat = trf.t_quat.to_rotmat(root_pre_quat)
    r0 = torch.matmul(root_pre_rotmat.transpose(-1, -2), basis_rotmat)
    r1 = torch.matmul(root_pre_rotmat, local_rotmats[:, :, 0])
    local_rotmats[:, :, 0] = torch.matmul(r0, r1)

    # root position
    root_pos = torch.matmul(basis_rotmat, local_root_pos[..., None])[..., 0] + basis_pos

    # returned features
    _, global_pos = trf.t_rotmat.fk(local_rotmats, root_pos, skeleton)
    local_ortho6ds = trf.t_rotmat.to_ortho6d(local_rotmats)

    return local_ortho6ds, global_pos

# def decompose_phase(phase):
#     B, T, P = phase.shape
#     phase = phase.reshape(B, T, -1, 2)

#     amp = torch.norm(phase, dim=-1, keepdim=True) # (B, T, P, 1)
#     shift = torch.atan2(phase[..., :1], phase[..., 1:]) / (2*torch.pi) # (B, T, P, 1)
#     freq = shift[:, 1:] - shift[:, :-1] # (B, T, P-1, 1)
#     freq = torch.cat([freq[:, :1], freq], dim=1) # (B, T, P, 1)
#     freq = torch.fmod(freq + 0.5, 1.0) - 0.5 # (B, T, P, 1)

#     return torch.cat([amp, shift, freq], dim=-1) # (B, T, P, 3)

def motion_to_traj(motion):
    B, T, D = motion.shape
    device = motion.device
    local_ortho6ds, root_pos = torch.split(motion, [D-3, 3], dim=-1)

    # basis position
    basis_pos = root_pos * torch.tensor([1, 0, 1], dtype=torch.float32, device=device) # (B, T, 3)
    basis_pos = basis_pos[..., (0, 2)] # xz only

    # basis angles
    local_quats = trf.t_ortho6d.to_quat(local_ortho6ds.reshape(B, T, -1, 6)) # (B, T, J, 4)
    
    world_fwd = torch.tensor([0, 0, 1], dtype=torch.float32, device=device) # (3,)
    world_fwd = world_fwd.repeat(B, T, 1) # (B, T, 3)

    root_fwd = trf.t_quat.mul_vec(local_quats[:, :, 0], world_fwd) # (B, T, 3)
    root_fwd = root_fwd * torch.tensor([1, 0, 1], dtype=torch.float32, device=device) # (B, T, 3)
    root_fwd = F.normalize(root_fwd, dim=-1, eps=1e-8) # (B, T, 3)

    basis_dir = root_fwd[..., (0, 2)] # (B, T, 2)

    # traj features
    traj_features = torch.cat([basis_pos, basis_dir], dim=-1) # (B, T, 4)
    return traj_features

def get_random_keyframe(config, seq_len):
    keyframes = [config.context_frames - 1]

    trans_start = config.context_frames
    while trans_start + config.max_kf_dist < seq_len - 1:
        trans_end = min(trans_start + config.max_kf_dist, seq_len - 1)
        kf = random.randint(trans_start + config.min_trans, trans_end)
        keyframes.append(kf)
        trans_start = kf + 1
    
    if keyframes[-1] != seq_len - 1:
        keyframes.append(seq_len - 1)
    
    return keyframes

def get_random_keyframe_prob(config, seq_len, p=0.1):
    keyframes = [config.context_frames - 1]

    # transition
    rand_nums = np.random.rand(seq_len - config.context_frames - 1)
    kfs = np.where(rand_nums < p)[0] + config.context_frames
    keyframes += kfs.tolist()

    # last frame
    keyframes.append(seq_len - 1)

    return keyframes

def get_keyframes_by_score(config, score):
    B, T, _ = score.shape
    res = []
    for b in range(B):
        keyframes = [config.context_frames - 1]
        trans_start = config.context_frames
        while trans_start < T:
            trans_end = min(trans_start + config.max_kf_dist, T-1)
            if trans_end == T-1:
                keyframes.append(T-1)
                break

            # top-scored frame
            top_keyframe = torch.argmax(score[b:b+1, trans_start+config.min_kf_dist:trans_end+1], dim=1) + trans_start + config.min_kf_dist
            top_keyframe = top_keyframe.item()
            keyframes.append(top_keyframe)
            trans_start = top_keyframe + 1
        res.append(keyframes)
    return res

def get_keyframes_by_score_threshold(config, score, threshold=0.8):
    B, T, _ = score.shape
    res = []
    for b in range(B):
        keyframes = [config.context_frames - 1]
        large_score_mask = score[b:b+1, config.context_frames:-1] > threshold
        kf_idx = torch.nonzero(large_score_mask, as_tuple=False)[..., 1] + config.context_frames
        keyframes += kf_idx.tolist()
        keyframes.append(T-1)
        res.append(keyframes)
    return res

def get_keyframes_by_topk(config, score, top=5):
    B, T, _ = score.shape
    res = []
    for b in range(B):
        keyframes = [config.context_frames - 1]
        if top >= T - config.context_frames:
            keyframes += list(range(config.context_frames, T))
            res.append(keyframes)
            continue
        kf_idx = torch.topk(score[b:b+1, config.context_frames:-1], top, dim=1)[1].squeeze() + config.context_frames
        keyframes += sorted(kf_idx.tolist())
        keyframes.append(T-1)
        res.append(keyframes)
    return res

def get_keyframes_by_random(config, score, prob=0.1):
    B, T, _ = score.shape
    res = []
    for b in range(B):
        keyframes = [config.context_frames - 1]
        rand_nums = np.random.rand(T - config.context_frames - 1)
        kfs = np.where(rand_nums < prob)[0] + config.context_frames
        keyframes += kfs.tolist()
        keyframes.append(T-1)
        res.append(keyframes)
    return res

def get_keyframes_by_uniform(config, score, step=1):
    B, T, _ = score.shape
    res = []
    for b in range(B):
        keyframes = list(range(config.context_frames - 1, T, step))
        if keyframes[-1] != T-1:
            keyframes.append(T-1)
        res.append(keyframes)
    return res

def interpolate_motion_by_keyframes(motion, keyframes):
    B, T, D = motion.shape
    device = motion.device

    # split
    local_ortho6ds, root_pos = torch.split(motion, [D-3, 3], dim=-1)
    local_quats = trf.t_ortho6d.to_quat(local_ortho6ds.reshape(B, T, -1, 6)) # (B, T, J, 4)

    # interpolation
    q_res = [local_quats[:, :keyframes[0]]]
    p_res = [root_pos[:, :keyframes[0]]]
    for i in range(len(keyframes) - 1):
        kf1, kf2 = keyframes[i], keyframes[i+1]
        t = torch.arange(0, kf2-kf1, dtype=torch.float32, device=device) / (kf2-kf1)

        # local quats
        q1 = local_quats[:, kf1] # (B, J, 4)
        q2 = local_quats[:, kf2]
        q_interp = trf.t_quat.interpolate(q1, q2, t) # (B, J, 4, t)
        q_interp = q_interp.permute(0, 3, 1, 2) # (B, t, J, 4)

        # root pos
        t = t[..., None]
        p1 = root_pos[:, kf1:kf1+1] # (B, 1, 3)
        p2 = root_pos[:, kf2:kf2+1]
        p_interp = p1 * (1-t) + p2 * t

        # update
        q_res.append(q_interp)
        p_res.append(p_interp)

    # last keyframe
    q_res.append(local_quats[:, keyframes[-1]:])
    p_res.append(root_pos[:, keyframes[-1]:])

    q_res = torch.cat(q_res, dim=1)
    p_res = torch.cat(p_res, dim=1)
    
    # merge
    o6_res = trf.t_quat.to_ortho6d(q_res)
    o6_res = o6_res.reshape(B, T, -1)

    return torch.cat([o6_res, p_res], dim=-1)

def remove_quat_discontinuities(quats):
    B, T, J, D = quats.shape
    quats_inv = -quats

    for i in range(1, T):
        replace_mask = torch.sum(quats[:, i-1:i] * quats[:, i:i+1], dim=-1) < torch.sum(quats[:, i-1:i] * quats_inv[:, i:i+1], dim=-1)
        replace_mask = replace_mask[..., None].float()
        quats[:, i:i+1] = replace_mask * quats_inv[:, i:i+1] + (1.0 - replace_mask) * quats[:, i:i+1]

    return quats


def get_contact(motion, skeleton, joint_ids, threshold):
    B, T, M = motion.shape
    local_ortho6ds, root_pos = torch.split(motion, [M-3, 3], dim=-1)
    local_ortho6ds = local_ortho6ds.reshape(B, T, -1, 6)
    _, global_positions = trf.t_ortho6d.fk(local_ortho6ds, root_pos, skeleton)

    foot_vel = global_positions[:, 1:, joint_ids] - global_positions[:, :-1, joint_ids]
    foot_vel = torch.sum(foot_vel ** 2, dim=-1) # (B, t-1, 4)
    foot_vel = torch.cat([foot_vel[:, 0:1], foot_vel], dim=1) # (B, t, 4)
    contact  = (foot_vel < threshold).float() # (B, t, 4)

    return contact