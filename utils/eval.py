import torch
from torch.utils.data import DataLoader
from aPyOpenGL import transforms as trf

from model.twostage import ContextTransformer, DetailTransformer
from model.rmi import RmiGenerator
from utils.dataset import MotionDataset
from utils import ops, utils

class Evaluator:
    def __init__(self, args):
        # arguments
        self.args = args
        self.config = utils.load_config(f"config/{args.dataset}/{args.config}")
        self.ts_configs = [utils.load_config(f"config/{args.dataset}/{config}") for config in args.ts_configs]
        self.rmi_configs = [utils.load_config(f"config/{args.dataset}/{config}") for config in args.rmi_configs]
        self.ours_configs = [utils.load_config(f"config/{args.dataset}/{config}") for config in args.ours_configs]

        # dataset
        utils.seed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = MotionDataset(train=False, config=self.config)
        self.dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        self.skeleton = dataset.skeleton

        # statistics
        self.mean, self.std = dataset.motion_statistics(self.device)
        self.traj_mean, self.traj_std = dataset.traj_statistics(self.device)
        self.l2p_mean, self.l2p_std = dataset.l2p_statistics(self.device)

        self.contact_idx = []
        for joint in self.config.contact_joints:
            self.contact_idx.append(self.skeleton.idx_by_name[joint])

        # load trained models
        self.ctx_models, self.det_models = [], []
        for ctx_config, det_config in zip(self.ts_configs[::2], self.ts_configs[1::2]):
            ctx_model = ContextTransformer(ctx_config, dataset).to(self.device)
            det_model = DetailTransformer(det_config, dataset).to(self.device)
            utils.load_model(ctx_model, ctx_config)
            utils.load_model(det_model, det_config)
            ctx_model.eval()
            det_model.eval()
            self.ctx_models.append(ctx_model)
            self.det_models.append(det_model)
        
        self.rmi_models = []
        for rmi_config in self.rmi_configs:
            rmi_model = RmiGenerator(rmi_config, self.skeleton.num_joints).to(self.device)
            utils.load_model(rmi_model, rmi_config)
            rmi_model.eval()
            self.rmi_models.append(rmi_model)

        self.kf_models, self.ref_models = [], []
        self.ref_configs = []
        for kf_config, ref_config in zip(self.ours_configs[::2], self.ours_configs[1::2]):
            kf_model = ContextTransformer(kf_config, dataset).to(self.device)
            ref_model = DetailTransformer(ref_config, dataset).to(self.device)
            utils.load_model(kf_model, kf_config)
            utils.load_model(ref_model, ref_config)
            kf_model.eval()
            ref_model.eval()
            self.kf_models.append(kf_model)
            self.ref_models.append(ref_model)
            self.ref_configs.append(ref_config)

    @torch.no_grad()
    def eval(self, num_frames=None, kf_sampling=["score"], traj_option=None):
        for i, batch in enumerate(self.dataloader):
            res = {
                "motions": [],
                "tags": [],
                "trajs": [],
                "contacts": [],
                "keyframes": [],
                "skeleton": self.skeleton,
            }

            # GT data
            GT_motion  = batch["motion"].to(self.device)
            GT_phase   = batch["phase"].to(self.device)
            GT_traj    = batch["traj"].to(self.device)
            GT_score   = batch["score"].to(self.device)
            GT_contact = ops.get_contact(GT_motion, self.skeleton, self.contact_idx, self.config.contact_threshold)

            if num_frames is not None:
                GT_motion = GT_motion[:, :num_frames]
                GT_phase = GT_phase[:, :num_frames]
                GT_traj = GT_traj[:, :num_frames]
                GT_score = GT_score[:, :num_frames]
                GT_contact = GT_contact[:, :num_frames]

            res["motions"].append(GT_motion.clone())
            res["contacts"].append(GT_contact.clone())
            res["keyframes"].append(None)
            res["tags"].append("GT")
            res["trajs"].append(GT_traj.clone())

            # interpolate motion
            B, T, D = GT_motion.shape
            if self.args.interp:
                keyframes = [self.config.context_frames-1, T-1]
                interp_motion = ops.interpolate_motion_by_keyframes(GT_motion, keyframes)
                res["motions"].append(interp_motion)
                res["contacts"].append(GT_contact)
                res["keyframes"].append(None)
                res["tags"].append("Interp")
                res["trajs"].append(GT_traj)

            # forward rmi model
            for idx, rmi_model in enumerate(self.rmi_models):
                rmi = _rmi_transition(self.config, rmi_model, GT_motion, self.mean, self.std, GT_contact)
                res["motions"].append(rmi["motion"])
                res["contacts"].append(rmi["contact"])
                res["keyframes"].append(None)
                res["trajs"].append(GT_traj)
                if len(self.rmi_models) > 1:
                    res["tags"].append(f"ERD-QV-{idx}")
                else:
                    res["tags"].append("ERD-QV")

            # forward two-stage model
            for idx, (ctx_model, det_model) in enumerate(zip(self.ctx_models, self.det_models)):
                twostage = _twostage_transition(self.config, ctx_model, det_model, GT_motion, self.mean, self.std, GT_contact)

                res["motions"].append(twostage["motion"])
                res["contacts"].append(twostage["contact"])
                res["keyframes"].append(None)
                res["trajs"].append(GT_traj)
                if len(self.ctx_models) > 1:
                    res["tags"].append(f"TS-Trans-{idx}")
                else:
                    res["tags"].append("TS-Trans")

            # forward our model
            for idx, (kf_model, ref_model) in enumerate(zip(self.kf_models, self.ref_models)):
                ctx_frames = self.config.context_frames
                if traj_option == "interp":
                    t = torch.linspace(0, 1, T-ctx_frames+1).to(self.device)
                    traj_pos = GT_traj[..., 0:2]
                    traj_dir = GT_traj[..., 2:4] # (sin, cos)
                    traj_ang = torch.atan2(traj_dir[..., 0:1], traj_dir[..., 1:2])
                    
                    traj_pos_from = GT_traj[:, ctx_frames-1:ctx_frames, 0:2]
                    traj_pos_to = GT_traj[:, -1:, 0:2]

                    traj_ang_from = traj_ang[:, ctx_frames-1:ctx_frames]
                    traj_ang_to = traj_ang[:, -1:]

                    traj_pos = traj_pos_from + (traj_pos_to - traj_pos_from) * t[None, :, None]
                    traj_ang = traj_ang_from + (traj_ang_to - traj_ang_from) * t[None, :, None]
                    traj_dir = torch.cat([torch.sin(traj_ang), torch.cos(traj_ang)], dim=-1)

                    GT_traj[:, ctx_frames-1:, 0:2] = traj_pos
                    GT_traj[:, ctx_frames-1:, 2:4] = traj_dir

                elif traj_option == "scale":
                    traj_pos = GT_traj[..., 0:2]
                    traj_vel = traj_pos[:, 1:] - traj_pos[:, :-1]
                    traj_vel = torch.cat([torch.zeros_like(traj_vel[:, :1]), traj_vel], dim=1)
                    traj_vel *= 1.2 # velocity scale
                    traj_pos = torch.cumsum(traj_vel, dim=1)
                    traj_pos = traj_pos - traj_pos[:, ctx_frames-1:ctx_frames]
                    GT_traj[:, ctx_frames-1:, 0:2] = traj_pos[:, ctx_frames-1:]
                    GT_motion[:, ctx_frames-1:, (-3, -1)] = traj_pos[:, ctx_frames-1:]
                
                elif traj_option in ["replace", "random"]:
                    batch_idx = torch.arange(B)
                    shuffle_idx = torch.randperm(B)
                    GT_traj[batch_idx, ctx_frames:] = GT_traj[shuffle_idx, ctx_frames:]

                    fwd_from = torch.matmul(trf.t_ortho6d.to_rotmat(GT_motion[batch_idx, -1, 0:6]), torch.tensor([0, 0, 1.0]).to(self.device))
                    fwd_to   = torch.matmul(trf.t_ortho6d.to_rotmat(GT_motion[shuffle_idx, -1, 0:6]), torch.tensor([0, 0, 1.0]).to(self.device))

                    up_axis  = torch.tensor([0, 1.0, 0]).to(self.device)
                    signed_angles = ops.get_signed_angle_torch(fwd_from, fwd_to, up_axis)
                    delta_R = trf.t_rotmat.from_aaxis(up_axis * signed_angles)

                    GT_root_rotmat = trf.t_ortho6d.to_rotmat(GT_motion[:, -1, 0:6])
                    new_root_rotmat = torch.matmul(delta_R, GT_root_rotmat)
                    new_rot6d = trf.t_ortho6d.from_rotmat(new_root_rotmat)

                    GT_motion[batch_idx, -1, 0:6] = new_rot6d
                    GT_motion[:, -1, (-3, -1)] = GT_traj[:, -1, 0:2]

                elif traj_option is not None:
                    raise NotImplementedError(f"Invalid traj_option: {traj_option}")

                ours = _ours_transition(self.ref_configs[idx], kf_model, ref_model, GT_motion, self.mean, self.std, GT_contact, GT_phase, GT_traj, GT_score, self.traj_mean, self.traj_std, kf_sampling)
                res["motions"].append(ours["motion"])
                res["contacts"].append(ours["contact"])
                res["keyframes"].append(ours["keyframes"])
                res["trajs"].append(GT_traj)
                if len(self.kf_models) > 1:
                    res["tags"].append(f"Ours-{idx}")
                else:
                    res["tags"].append("Ours")

            yield res

@torch.no_grad()
def _twostage_transition(config, ctx_model: ContextTransformer, det_model: DetailTransformer, GT_motion, mean, std, GT_contact):
    motion = (GT_motion - mean) / std

    # forward ContextTransformer
    ctx_out, midway_targets = ctx_model.forward(motion, train=False)
    ctx_motion = ctx_out["motion"]

    # restore constrained frames
    pred_ctx_motion = motion.clone().detach()
    pred_ctx_motion[:, config.context_frames:-1] = ctx_motion[:, config.context_frames:-1]

    # forward DetailTransformer
    det_out = det_model.forward(pred_ctx_motion, midway_targets)
    det_motion = det_out["motion"]
    det_contact = det_out["contact"]

    # restore constrained frames
    pred_det_motion = motion.clone().detach()
    pred_det_motion[:, config.context_frames:-1] = det_motion[:, config.context_frames:-1]

    pred_det_contact = GT_contact.clone().detach()
    pred_det_contact[:, config.context_frames:-1] = det_contact[:, config.context_frames:-1]

    # denormalize
    # pred_det_motion = pred_det_motion * std + mean

    return {
        "motion": pred_det_motion * std + mean,
        "contact": pred_det_contact,
    }

@torch.no_grad()
def _rmi_transition(config, generator: RmiGenerator, GT_motion, mean, std, GT_contact):
    B, T, D = GT_motion.shape
    motion = (GT_motion - mean) / std

    local_rot, root_pos = torch.split(motion, [D-3, 3], dim=-1)
    root_vel = root_pos[:, 1:] - root_pos[:, :-1]
    root_vel = torch.cat([root_vel[:, 0:1], root_vel], dim=1)

    target = motion[:, -1]
    target_local_rot, target_root_pos = torch.split(target, [D-3, 3], dim=-1)
    
    generator.init_hidden(B, motion.device)
    pred_rot, pred_root_pos, pred_contact = [local_rot[:, 0]], [root_pos[:, 0]], [GT_contact[:, 0]]
    for i in range(config.context_frames):
        tta = T - i - 1
        lr, rp, c = generator.forward(local_rot[:, i], root_pos[:, i], root_vel[:, i], GT_contact[:, i], target_local_rot, target_root_pos, tta)
        pred_rot.append(lr)
        pred_root_pos.append(rp)
        pred_contact.append(c)
    for i in range(config.context_frames, T-1):
        tta = T - i - 1
        lr, rp, c = generator.forward(lr, rp, rp - pred_root_pos[-1], c, target_local_rot, target_root_pos, tta)
        pred_rot.append(lr)
        pred_root_pos.append(rp)
        pred_contact.append(c)
    
    # stack transition frames without context frames
    pred_rot = torch.stack(pred_rot, dim=1)
    pred_root_pos = torch.stack(pred_root_pos, dim=1)
    pred_contact = torch.stack(pred_contact, dim=1)

    motion = torch.cat([pred_rot, pred_root_pos], dim=-1)
    motion = motion * std + mean

    pred_motion = torch.cat([GT_motion[:, :config.context_frames], motion[:, config.context_frames:]], dim=1)
    pred_contact = torch.cat([GT_contact[:, :config.context_frames], pred_contact[:, config.context_frames:]], dim=1)

    return {
        "motion": pred_motion,
        "contact": pred_contact,
    }

@torch.no_grad()
def _ours_transition(config,
                     kf_model: ContextTransformer,
                     ref_model: DetailTransformer,
                     GT_motion,
                     mean,
                     std,
                     GT_contact,
                     GT_phase=None,
                     GT_traj=None,
                     GT_score=None,
                     traj_mean=None,
                     traj_std=None,
                     kf_sampling=["score"],):
    """
    config: config of RefineNet
    """
    motion = (GT_motion - mean) / std
    if config.use_traj:
        GT_traj = (GT_traj - traj_mean) / traj_std

    # forward ContextTransformer
    ctx_out, midway_targets = kf_model.forward(motion, phase=GT_phase, traj=GT_traj, train=False)
    ctx_motion = ctx_out["motion"]
    if config.use_phase:
        ctx_phase = ctx_out["phase"]
    if config.use_score:
        ctx_score = ctx_out["score"]

    # restore constrained frames
    pred_ctx_motion = motion.clone().detach()
    pred_ctx_motion[:, config.context_frames:-1] = ctx_motion[:, config.context_frames:-1]
    pred_ctx_motion[:, midway_targets] = motion[:, midway_targets]

    if config.use_phase:
        pred_ctx_phase = GT_phase.clone().detach()
        pred_ctx_phase[:, config.context_frames:-1] = ctx_phase[:, config.context_frames:-1]
        pred_ctx_phase[:, midway_targets] = GT_phase[:, midway_targets]
    else:
        pred_ctx_phase = None

    if config.use_score:
        pred_score = GT_score.clone().detach()
        pred_score[:, config.context_frames:-1] = ctx_score[:, config.context_frames:-1]
        
        pred_ctx_motion = pred_ctx_motion * std + mean
        if kf_sampling[0] == "score":
            keyframes = ops.get_keyframes_by_score(config, pred_score)
        elif kf_sampling[0] == "threshold":
            keyframes = ops.get_keyframes_by_score_threshold(config, pred_score, threshold=kf_sampling[1])
        elif kf_sampling[0] == "topk":
            keyframes = ops.get_keyframes_by_topk(config, pred_score, top=kf_sampling[1])
        elif kf_sampling[0] == "random":
            keyframes = ops.get_keyframes_by_random(config, pred_score, prob=kf_sampling[1])
        elif kf_sampling[0] == "uniform":
            keyframes = ops.get_keyframes_by_uniform(config, pred_score, step=kf_sampling[1])
        else:
            raise NotImplementedError(f"Invalid keyframe sampling method: {kf_sampling}")
        for b in range(motion.shape[0]):
            pred_ctx_motion[b:b+1] = ops.interpolate_motion_by_keyframes(pred_ctx_motion[b:b+1], keyframes[b])
        pred_ctx_motion = (pred_ctx_motion - mean) / std
    else:
        keyframes = None

    # forward DetailTransformer
    det_out = ref_model.forward(pred_ctx_motion, midway_targets, phase=pred_ctx_phase, traj=GT_traj)
    det_motion = det_out["motion"]
    det_contact = det_out["contact"]

    # restore constrained frames
    pred_det_motion = motion.clone().detach()
    pred_det_motion[:, config.context_frames:-1] = det_motion[:, config.context_frames:-1]

    pred_det_contact = GT_contact.clone().detach()
    pred_det_contact[:, config.context_frames:-1] = det_contact[:, config.context_frames:-1]

    return {
        "motion": pred_det_motion * std + mean,
        "contact": pred_det_contact,
        "keyframes": keyframes,
    }

#####################################
# benchmark metrics
#####################################
def l2p(GT_motion, pred_motion, skeleton, l2p_mean, l2p_std, ctx_frames=10):
    B, T, D = GT_motion.shape

    def convert(motion):
        rot, pos = torch.split(motion, [D-3, 3], dim=-1)
        rot = rot.reshape(B, T, skeleton.num_joints, 6)
        _, gp = trf.t_ortho6d.fk(rot, pos, skeleton)
        gp = gp[:, ctx_frames:-1]
        return (gp - l2p_mean) / l2p_std
    
    GT_gp = convert(GT_motion)
    pred_gp = convert(pred_motion)

    norm = torch.sqrt(torch.sum((GT_gp - pred_gp) ** 2, dim=(2, 3)))
    return torch.mean(norm).item()

def l2q(GT_motion, pred_motion, ctx_frames=10):
    B, T, D = GT_motion.shape

    def convert(motion):
        rot, pos = torch.split(motion, [D-3, 3], dim=-1)
        rot = trf.t_quat.from_ortho6d(rot.reshape(B, T, -1, 6))
        rot = ops.remove_quat_discontinuities(rot)
        rot = rot[:, ctx_frames:-1]
        return rot
    
    GT_rot = convert(GT_motion)
    pred_rot = convert(pred_motion)
    norm = torch.sqrt(torch.sum((GT_rot - pred_rot) ** 2, dim=(2, 3)))
    return torch.mean(norm).item()

def npss(GT_motion, pred_motion, ctx_frames=10):
    B, T, D = GT_motion.shape

    def convert(motion):
        rot, pos = torch.split(motion, [D-3, 3], dim=-1)
        rot = trf.t_quat.from_ortho6d(rot.reshape(B, T, -1, 6))
        rot = ops.remove_quat_discontinuities(rot).reshape(B, T, -1)
        rot = rot[:, ctx_frames:-1]

        # Fourier coefficients along the time dimension
        fourier_coeffs = torch.real(torch.fft.fft(rot, dim=1))

        # square of the Fourier coefficients
        power = torch.square(fourier_coeffs)

        # sum of powers over time
        total_power = torch.sum(power, dim=1)

        # normalize powers with total
        norm_power = power / (total_power[:, None] + 1e-8)

        # cumulative sum over time
        cdf_power = torch.cumsum(norm_power, dim=1)

        return cdf_power, total_power
    
    GT_cdf_power, GT_total_power = convert(GT_motion)
    pred_cdf_power, _ = convert(pred_motion)

    # earth mover distance
    emd = torch.norm((pred_cdf_power - GT_cdf_power), p=1, dim=1)

    # weighted EMD
    power_weighted_emd = torch.sum(emd * GT_total_power) / torch.sum(GT_total_power)

    return power_weighted_emd.item()

def foot_skate(pred_motion, pred_contact, skeleton, foot_ids, ctx_frames=10):
    B, T, D = pred_motion.shape
    
    rot, pos = torch.split(pred_motion, [D-3, 3], dim=-1)
    rot = rot.reshape(B, T, -1, 6)
    _, gp = trf.t_ortho6d.fk(rot, pos, skeleton)

    # foot velocity
    fv = gp[:, 1:, foot_ids] - gp[:, :-1, foot_ids]
    fv = torch.sum(fv ** 2, dim=-1) # (B, T-1, 4)
    fv = torch.cat([fv[:, 0:1], fv], dim=1) # (B, T, 4)

    # # foot position
    # fp = gp[:, :, foot_ids] # (B, T, 4, 3)

    # # weight
    # weight = torch.clamp(2.0 - 2.0 ** (fp[..., 1] / height_threshold), min=0, max=1) # (B, T, 4)

    # # # mask - if all weights are zero, skip this sample
    # # mask = torch.sum(weight.reshape(B, -1), dim=-1) > 0 # (B)
    # # fv = fv[mask]
    # # weight = weight[mask]

    metric = torch.sum(fv * pred_contact, dim=-1)
    metric = torch.mean(metric[:, ctx_frames:-1], dim=-1)

    # # metric
    # metric = torch.sum(fv * weight, dim=-1) # (B, T)
    # metric = torch.mean(metric[:, ctx_frames:-1], dim=-1) # (B)

    return torch.mean(metric).item() * 100

def traj_pos_error(GT_traj, pred_motion, ctx_frames=10):
    B, T, D = pred_motion.shape
    GT_pos = GT_traj[:, :, 0:2]
    _, pred_pos = torch.split(pred_motion, [D-3, 3], dim=-1)
    pred_pos = pred_pos[..., (0, 2)]

    norm = torch.sqrt(torch.sum((GT_pos - pred_pos) ** 2, dim=-1))
    return torch.mean(norm[:, ctx_frames:-1]).item() * 100