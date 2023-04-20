import sys
sys.path.append(".")

import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import time
from tqdm import tqdm

from pymovis.utils import util, torchconst
from pymovis.ops import motionops, rotation, mathops

from utility.dataset import MotionDataset
from utility.config import Config
from model.vae import TwoStageVAE
from utility import trainutil, loss

def get_motion_and_trajectory(motion, skeleton, v_forward):
    B, T, D = motion.shape

    # motion
    local_R6, root_p = torch.split(motion, [D-3, 3], dim=-1)
    _, global_p = motionops.R6_fk(local_R6.reshape(B, T, -1, 6), root_p, skeleton)

    # trajectory
    root_xz = root_p[..., (0, 2)]
    root_fwd = torch.matmul(rotation.R6_to_R(local_R6[..., :6]), v_forward)
    root_fwd = F.normalize(root_fwd * torchconst.XZ(motion.device), dim=-1)
    traj = torch.cat([root_xz, root_fwd], dim=-1)

    return local_R6.reshape(B, T, -1, 6), global_p.reshape(B, T, -1, 3), traj

def get_velocity_and_contact(global_p, joint_ids, threshold):
    feet_v = global_p[:, 1:, joint_ids] - global_p[:, :-1, joint_ids]
    feet_v = torch.sum(feet_v**2, dim=-1) # squared norm
    feet_v = torch.cat([feet_v[:, 0:1], feet_v], dim=1)
    contact = (feet_v < threshold).float()
    return feet_v, contact

if __name__ == "__main__":
    # initial settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config.load("configs/twostagevae.json")
    util.seed()

    # dataset
    print("Loading dataset...")
    dataset     = MotionDataset(train=True, config=config)
    val_dataset = MotionDataset(train=False, config=config)
    skeleton    = dataset.skeleton
    v_forward   = torch.from_numpy(config.v_forward).to(device)

    motion_mean, motion_std = dataset.statistics(dim=(0, 1))
    motion_mean, motion_std = motion_mean.to(device), motion_std.to(device)
    
    feet_ids = []
    for name in config.contact_joint_names:
        feet_ids.append(skeleton.idx_by_name[name])

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # model
    print("Initializing model...")
    model = TwoStageVAE(dataset.shape[-1], config).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    init_epoch, iter = trainutil.load_latest_ckpt(model, optim, config)
    init_iter = iter

    # save and log
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    config.write(os.path.join(config.save_dir, "config.json"))
    writer = SummaryWriter(config.log_dir)

    # training loop
    loss_dict = {
        "total":       0,

        "kl":          0,

        "ctx":         0,
        "ctx/rot":     0,
        "ctx/pos":     0,
        "ctx/traj":    0,
        "ctx/smooth":  0,

        "det":         0,
        "det/rot":     0,
        "det/pos":     0,
        "det/vel":     0,
        "det/traj":    0,
        "det/contact": 0,
        "det/foot":    0,
    }
    start_time = time.perf_counter()
    for epoch in range(init_epoch, config.epochs+1):
        for GT_motion in tqdm(dataloader, desc=f"Epoch {epoch} / {config.epochs}", leave=False):
            """ 1. Random transition length """
            transition = random.randint(config.min_transition, config.max_transition)
            T = config.context_frames + transition + 1
            GT_motion = GT_motion[:, :T, :].to(device)
            B, T, D = GT_motion.shape

            """ 2. GT motion data """
            GT_local_R6, GT_global_p, GT_traj = get_motion_and_trajectory(GT_motion, skeleton, v_forward)
            GT_feet_v, GT_contact = get_velocity_and_contact(GT_global_p, feet_ids, config.contact_vel_threshold)

            """ 3. Forward """
            # forward
            GT_batch = (GT_motion - motion_mean) / motion_std
            ctx_motion, det_motion, contact, vae_mean, vae_logvar = model.forward(GT_motion, GT_traj)

            ctx_motion = ctx_motion * motion_std + motion_mean
            det_motion = det_motion * motion_std + motion_mean

            # motion and trajectory
            ctx_local_R6, ctx_global_p, ctx_traj = get_motion_and_trajectory(ctx_motion, skeleton, v_forward)
            det_local_R6, det_global_p, det_traj = get_motion_and_trajectory(det_motion, skeleton, v_forward)

            # contact
            det_feet_v, det_contact = get_velocity_and_contact(det_global_p, feet_ids, config.contact_vel_threshold)

            """ 4. Loss """
            # KL loss
            loss_kl = config.weight_kl * loss.kl_loss(vae_mean, vae_logvar)

            # ContextNet loss
            loss_ctx_rot = config.weight_ctx_rot * loss.recon_loss(ctx_local_R6, GT_local_R6)
            loss_ctx_pos = config.weight_ctx_pos * loss.recon_loss(ctx_global_p, GT_global_p)
            loss_ctx_traj = config.weight_ctx_traj * loss.recon_loss(ctx_traj, GT_traj)
            loss_ctx_smooth = config.weight_ctx_smooth * (loss.smooth_loss(ctx_local_R6) + loss.smooth_loss(ctx_global_p))
            loss_ctx = loss_ctx_rot + loss_ctx_pos + loss_ctx_traj + loss_ctx_smooth

            # DetailNet loss
            loss_det_rot = config.weight_det_rot * loss.recon_loss(det_local_R6, GT_local_R6)
            loss_det_pos = config.weight_det_pos * loss.recon_loss(det_global_p, GT_global_p)
            loss_det_vel = config.weight_det_vel * loss.recon_loss(det_global_p[:, 1:] - det_global_p[:, :-1], GT_global_p[:, 1:] - GT_global_p[:, :-1])
            loss_det_traj = config.weight_det_traj * loss.recon_loss(det_traj, GT_traj)
            loss_det_contact = config.weight_det_contact * loss.recon_loss(det_contact, GT_contact)
            loss_det_foot = config.weight_det_foot * loss.smooth_loss(det_contact.detach() * det_feet_v)
            loss_det = loss_det_rot + loss_det_pos + loss_det_vel + loss_det_traj + loss_det_contact + loss_det_foot

            # total loss
            loss_total = loss_ctx + loss_det + loss_kl
            
            """ 5. Backpropagation """
            optim.zero_grad()
            loss_total.backward()
            optim.step()

            """ 6. Log """
            loss_dict["total"]       += loss_total.item()
            
            loss_dict["kl"]          += loss_kl.item()
            
            loss_dict["ctx"]         += loss_ctx.item()
            loss_dict["ctx/rot"]     += loss_ctx_rot.item()
            loss_dict["ctx/pos"]     += loss_ctx_pos.item()
            loss_dict["ctx/traj"]    += loss_ctx_traj.item()
            loss_dict["ctx/smooth"]  += loss_ctx_smooth.item()

            loss_dict["det"]         += loss_det.item()
            loss_dict["det/rot"]     += loss_det_rot.item()
            loss_dict["det/pos"]     += loss_det_pos.item()
            loss_dict["det/vel"]     += loss_det_vel.item()
            loss_dict["det/traj"]    += loss_det_traj.item()
            loss_dict["det/contact"] += loss_det_contact.item()
            loss_dict["det/foot"]    += loss_det_foot.item()

            if iter % config.log_interval == 0:
                tqdm.write(f"Iter {iter} | Total: {(loss_dict['total'] / config.log_interval):.4f} | Context: {(loss_dict['ctx'] / config.log_interval):.4f} | Detail: {(loss_dict['det'] / config.log_interval):.4f} | KL: {(loss_dict['kl'] / config.log_interval):.4f}")
                trainutil.write_log(writer, loss_dict, config.log_interval, iter, train=True)

                # reset loss dict
                for k in loss_dict.keys():
                    loss_dict[k] = 0
            
            """ 7. Validation """
            if iter % config.val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss_dict = {
                        "total":       0,

                        "ctx":         0,
                        "ctx/rot":     0,
                        "ctx/pos":     0,
                        "ctx/traj":    0,
                        "ctx/smooth":  0,

                        "det":         0,
                        "det/rot":     0,
                        "det/pos":     0,
                        "det/vel":     0,
                        "det/traj":    0,
                        "det/contact": 0,
                        "det/foot":    0,
                    }
                    for GT_motion in val_dataloader:
                        """ 7-1. Max transition length """
                        T = config.context_frames + config.max_transition + 1
                        GT_motion = GT_motion[:, :T, :].to(device)
                        B, T, D = GT_motion.shape

                        """ 7-2. GT motion data """
                        GT_local_R6, GT_global_p, GT_traj = get_motion_and_trajectory(GT_motion, skeleton, v_forward)
                        GT_feet_v, GT_contact = get_velocity_and_contact(GT_global_p, feet_ids, config.contact_vel_threshold)

                        """ 7-3. Forward """
                        # forward
                        GT_batch = (GT_motion - motion_mean) / motion_std
                        ctx_motion, det_motion, contact = model.sample(GT_motion, GT_traj)

                        ctx_motion = ctx_motion * motion_std + motion_mean
                        det_motion = det_motion * motion_std + motion_mean

                        # motion and trajectory
                        ctx_local_R6, ctx_global_p, ctx_traj = get_motion_and_trajectory(ctx_motion, skeleton, v_forward)
                        det_local_R6, det_global_p, det_traj = get_motion_and_trajectory(det_motion, skeleton, v_forward)

                        # contact
                        det_feet_v, det_contact = get_velocity_and_contact(det_global_p, feet_ids, config.contact_vel_threshold)

                        """ 7-4. Loss """
                        # ContextNet loss
                        loss_ctx_rot = config.weight_ctx_rot * loss.recon_loss(ctx_local_R6, GT_local_R6)
                        loss_ctx_pos = config.weight_ctx_pos * loss.recon_loss(ctx_global_p, GT_global_p)
                        loss_ctx_traj = config.weight_ctx_traj * loss.recon_loss(ctx_traj, GT_traj)
                        loss_ctx_smooth = config.weight_ctx_smooth * (loss.smooth_loss(ctx_local_R6) + loss.smooth_loss(ctx_global_p))
                        loss_ctx = loss_ctx_rot + loss_ctx_pos + loss_ctx_traj + loss_ctx_smooth

                        # DetailNet loss
                        loss_det_rot = config.weight_det_rot * loss.recon_loss(det_local_R6, GT_local_R6)
                        loss_det_pos = config.weight_det_pos * loss.recon_loss(det_global_p, GT_global_p)
                        loss_det_vel = config.weight_det_vel * loss.recon_loss(det_global_p[:, 1:] - det_global_p[:, :-1], GT_global_p[:, 1:] - GT_global_p[:, :-1])
                        loss_det_traj = config.weight_det_traj * loss.recon_loss(det_traj, GT_traj)
                        loss_det_contact = config.weight_det_contact * loss.recon_loss(det_contact, GT_contact)
                        loss_det_foot = config.weight_det_foot * loss.smooth_loss(det_contact.detach() * det_feet_v)
                        loss_det = loss_det_rot + loss_det_pos + loss_det_vel + loss_det_traj + loss_det_contact + loss_det_foot

                        # total loss
                        loss_total = loss_ctx + loss_det

                        """ 7-5. Update loss dict """
                        val_loss_dict["total"]       += loss_total.item()

                        val_loss_dict["ctx"]         += loss_ctx.item()
                        val_loss_dict["ctx/rot"]     += loss_ctx_rot.item()
                        val_loss_dict["ctx/pos"]     += loss_ctx_pos.item()
                        val_loss_dict["ctx/traj"]    += loss_ctx_traj.item()
                        val_loss_dict["ctx/smooth"]  += loss_ctx_smooth.item()

                        val_loss_dict["det"]         += loss_det.item()
                        val_loss_dict["det/rot"]     += loss_det_rot.item()
                        val_loss_dict["det/pos"]     += loss_det_pos.item()
                        val_loss_dict["det/vel"]     += loss_det_vel.item()
                        val_loss_dict["det/traj"]    += loss_det_traj.item()
                        val_loss_dict["det/contact"] += loss_det_contact.item()
                        val_loss_dict["det/foot"]    += loss_det_foot.item()

                # average
                for k in val_loss_dict.keys():
                    val_loss_dict[k] /= len(val_dataloader)
                
                # write and print log
                trainutil.write_log(writer, val_loss_dict, config.val_interval, iter, train=False)
                tqdm.write(f"Validation at Iter {iter} | Total: {val_loss_dict['total']:.4f} | Context: {val_loss_dict['ctx']:.4f} | Detail: {val_loss_dict['det']:.4f}")

                # train mode
                model.train()

            """ 8. Save checkpoint """
            if iter % config.save_interval == 0:
                trainutil.save_ckpt(model, optim, epoch, iter, config)
                tqdm.write(f"Saved checkpoint at iter {iter}")
            
            # update iter
            iter += 1
    
    print(f"Training finished in {time.perf_counter() - start_time:.2f} seconds")
    trainutil.save_ckpt(model, optim, epoch, iter, config)