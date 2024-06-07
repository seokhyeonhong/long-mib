import sys
sys.path.append(".")

from tqdm import tqdm
import argparse

from aPyOpenGL import agl

import torch
from torch.utils.data import DataLoader

from utils import utils, ops, eval
from utils.dataset import MotionDataset
from model.twostage import ContextTransformer, DetailTransformer
from model.rmi import RmiGenerator
from vis.motionapp import MotionApp

if __name__ =="__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default="default.yaml")
    parser.add_argument("--kf_configs", type=str, nargs="+", default=["keyframe.yaml",
                                                                      "keyframe.yaml",
                                                                      "keyframe.yaml",])
    parser.add_argument("--ref_configs", type=str, nargs="+", default=["refine.yaml",
                                                                       "refine-w-kfemb.yaml",
                                                                       "refine-wo-mask.yaml",])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = utils.load_config(f"config/{args.dataset}/{args.config}")
    kf_configs = [utils.load_config(f"config/{args.dataset}/{config}") for config in args.kf_configs]
    ref_configs = [utils.load_config(f"config/{args.dataset}/{config}") for config in args.ref_configs]
    utils.seed()

    # dataset
    dataset = MotionDataset(train=False, config=config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    mean, std = dataset.motion_statistics(device)
    traj_mean, traj_std = dataset.traj_statistics(device)
    skeleton = dataset.skeleton

    contact_idx = []
    for joint in config.contact_joints:
        contact_idx.append(skeleton.idx_by_name[joint])

    # load trained models
    kf_models = [ContextTransformer(kf_config, dataset).to(device) for kf_config in kf_configs]
    ref_models = [DetailTransformer(ref_config, dataset).to(device) for ref_config in ref_configs]
    
    for kf_model, kf_config in zip(kf_models, kf_configs):
        utils.load_model(kf_model, kf_config)
        kf_model.eval()
    for ref_model, ref_config in zip(ref_models, ref_configs):
        utils.load_model(ref_model, ref_config)
        ref_model.eval()

    # function for each iteration
    def vis_iter(batch):
        # GT data
        GT_motion  = batch["motion"].to(device)
        GT_phase   = batch["phase"].to(device)
        GT_traj    = batch["traj"].to(device)
        GT_score   = batch["score"].to(device)
        GT_contact = ops.get_contact(GT_motion, skeleton, contact_idx, config.contact_threshold)

        # forward two-stage model
        motions, tags = [GT_motion], ["GT"]
        for i, ref_config in enumerate(ref_configs):
            motions.append(eval.ours_transition(ref_config, kf_model, ref_model, GT_motion, mean, std, GT_contact, GT_phase, GT_traj, GT_score, traj_mean, traj_std)["ref_motion"])
            tags.append(f"Ours-{i}")

        agl.AppManager.start(MotionApp(motions,
                                       tags,
                                       skeleton,
                                       dataset=args.dataset))
        
    # main loop
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, leave=False)):
            vis_iter(batch)