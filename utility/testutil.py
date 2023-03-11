import os
import torch

def load_model(model, config, iter=None):
    ckpt_list = os.listdir(config.save_dir)
    if len(ckpt_list) > 0:
        if iter is None:
            ckpt_list = [ckpt for ckpt in ckpt_list if ckpt.endswith(".pth")]
            ckpt_list = sorted(ckpt_list)
            ckpt_path = os.path.join(config.save_dir, ckpt_list[-1])
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded checkpoint: {ckpt_path}")
        else:
            ckpt_path = os.path.join(config.save_dir, f"ckpt_{iter:08d}.pth")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded checkpoint: {ckpt_path}")
    else:
        raise Exception("No checkpoint found.")