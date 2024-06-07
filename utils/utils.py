import os
import torch
import numpy as np
import random
from tqdm import tqdm
from omegaconf import OmegaConf

def load_config(filepath):
    cfg = OmegaConf.load(filepath)
    if cfg.get("context_frames", None) is None:
        cfg.npz_path = f"length{cfg.window_length}-offset{cfg.window_offset}.npz"
    else:
        cfg.npz_path = f"length{cfg.window_length}-offset{cfg.window_offset}-context{cfg.context_frames}.npz"
    cfg.skeleton_path = os.path.join(cfg.dataset_dir, "skeleton.pkl")
    return cfg

def write_config(config):
    with open(os.path.join(config.save_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)

def seed(x=1234):
    torch.manual_seed(x)
    torch.cuda.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(x)
    random.seed(x)

def write_log(writer, loss_dict, interval, iter, elapsed=None, train=True):
    msg = f"{'Train' if train else 'Val'} at {iter}: "
    for key, value in loss_dict.items():
        writer.add_scalar(f"{'train' if train else 'val'}/{key}", value / interval, iter)
        msg += f"{key}: {value / interval:.4f} | "
    if elapsed is not None:
        msg += f"Time: {(elapsed / 60):.2f} min"
    tqdm.write(msg)

def reset_log(loss_dict):
    for key in loss_dict.keys():
        loss_dict[key] = 0

def load_model(model, config, epoch=None):
    ckpt_list = os.listdir(config.save_dir)
    if len(ckpt_list) > 0:
        ckpt_list = [ckpt for ckpt in ckpt_list if ckpt.endswith(".pth")]
        ckpt_list = sorted(ckpt_list)
        if epoch is None:
            ckpt_path = os.path.join(config.save_dir, ckpt_list[-1])
            ckpt = torch.load(ckpt_path, map_location="cuda:0")
            model.load_state_dict(ckpt["model"])
            print(f"> Loaded checkpoint: {ckpt_path}")
        else:
            ckpt_path = os.path.join(config.save_dir, f"ckpt_{epoch:04d}.pth")
            ckpt = torch.load(ckpt_path, map_location="cuda:0")
            model.load_state_dict(ckpt["model"])
            print(f"> Loaded checkpoint: {ckpt_path}")
    else:
        raise Exception("No checkpoint found.")
    
def load_latest_ckpt(model, optim, config, scheduler=None):
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
        
    ckpt_list = os.listdir(config.save_dir)
    ckpt_list = [f for f in ckpt_list if f.endswith(".pth")]
    ckpt_list = sorted(ckpt_list)
    if len(ckpt_list) > 0:
        ckpt_path = os.path.join(config.save_dir, ckpt_list[-1])
        ckpt = torch.load(ckpt_path, map_location="cuda:0")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        epoch = ckpt["epoch"]
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"> Checkpoint loaded: {ckpt_path}, epoch: {epoch}")
    else:
        epoch = 0
        print(f"> No checkpoint found from {config.save_dir}. Start training from scratch.")

    return epoch

def save_ckpt(model, optim, epoch, config, scheduler=None):
    ckpt_path = os.path.join(config.save_dir, f"ckpt_{epoch:04d}.pth")
    ckpt = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
    }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    
    torch.save(ckpt, ckpt_path)
    print(f"> Saved checkpoint at epoch {epoch}: {ckpt_path}")