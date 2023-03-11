import os
import torch

def save_ckpt(model, optim, epoch, iter, config, scheduler=None):
    ckpt_path = os.path.join(config.save_dir, f"ckpt_{iter:08d}.pth")
    ckpt = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "iter": iter,
    }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    
    torch.save(ckpt, ckpt_path)

def load_latest_ckpt(model, optim, config, scheduler=None):
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
        
    ckpt_list = os.listdir(config.save_dir)
    ckpt_list = [f for f in ckpt_list if f.endswith(".pth")]
    ckpt_list = sorted(ckpt_list)
    if len(ckpt_list) > 0:
        ckpt_path = os.path.join(config.save_dir, ckpt_list[-1])
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        epoch = ckpt["epoch"]
        iter = ckpt["iter"]
        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"Checkpoint loaded: {ckpt_path}, epoch: {epoch}, iter: {iter}")
    else:
        epoch = 1
        iter = 1
        print("No checkpoint found. Start training from scratch.")

    return epoch, iter

def get_noam_scheduler(config, optim):
    warmup_iters = config.warmup_iters

    def _lr_lambda(iter, warmup_iters=warmup_iters):
        if iter < warmup_iters:
            return iter * warmup_iters ** (-1.5)
        else:
            return (iter ** (-0.5))
    
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=_lr_lambda)