import sys
sys.path.append(".")

from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader

from utils import utils, ops, eval
from utils.dataset import MotionDataset
from model.twostage import ContextTransformer, DetailTransformer
from model.rmi import RmiGenerator

if __name__ =="__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str, default="default.yaml")
    parser.add_argument("--kf_configs", type=str, nargs="+", default=["keyframe.yaml",
                                                                      "keyframe-wo-phase.yaml",
                                                                      "keyframe-wo-score.yaml",
                                                                      "keyframe-wo-traj.yaml"])
    parser.add_argument("--ref_configs", type=str, nargs="+", default=["refine.yaml",
                                                                       "refine-wo-phase.yaml",
                                                                       "refine-wo-score.yaml",
                                                                       "refine-wo-traj.yaml"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = utils.load_config(f"config/{args.dataset}/{args.config}")
    kf_configs = [utils.load_config(f"config/{args.dataset}/{kf_config}") for kf_config in args.kf_configs]
    ref_configs = [utils.load_config(f"config/{args.dataset}/{ref_config}") for ref_config in args.ref_configs]
    utils.seed()

    # dataset
    dataset = MotionDataset(train=False, config=config)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    mean, std = dataset.motion_statistics(device)
    traj_mean, traj_std = dataset.traj_statistics(device)
    l2p_mean, l2p_std = dataset.l2p_statistics(device)
    skeleton = dataset.skeleton

    contact_idx = []
    for joint in config.contact_joints:
        contact_idx.append(skeleton.idx_by_name[joint])

    # load trained models
    kf_models = [ContextTransformer(kf_config, dataset).to(device) for kf_config in kf_configs]
    ref_models = [DetailTransformer(ref_config, dataset).to(device) for ref_config in ref_configs]

    for kf_model, kf_config, ref_model, ref_config in zip(kf_models, kf_configs, ref_models, ref_configs):
        utils.load_model(kf_model, kf_config)
        utils.load_model(ref_model, ref_config)

    # function for each iteration
    def eval_iter(batch, num_trans):
        T = config.context_frames + num_trans + 1
        # GT data
        GT_motion  = batch["motion"][:, :T].to(device)
        GT_phase   = batch["phase"][:, :T].to(device)
        GT_traj    = batch["traj"][:, :T].to(device)
        GT_score   = batch["score"][:, :T].to(device)
        GT_contact = ops.get_contact(GT_motion, skeleton, contact_idx, config.contact_threshold)

        # forward two-stage model
        ablations = []
        for idx, (kf_model, ref_model) in enumerate(zip(kf_models, ref_models)):
            phase = GT_phase if kf_configs[idx].use_phase else None
            score = GT_score if kf_configs[idx].use_score else None
            traj = GT_traj if kf_configs[idx].use_traj else None

            ablations.append(eval.ours_transition(ref_configs[idx], kf_model, ref_model, GT_motion, mean, std, GT_contact, phase, traj, score, traj_mean, traj_std)["ref_motion"])

        return GT_motion, ablations

    # main loop
    for kf_model, ref_model in zip(kf_models, ref_models):
        kf_model.eval()
        ref_model.eval()

    transitions = [ 15, 30, 60, 90 ]
    results = {
        "l2p": [],
        "l2q": [],
        "npss": [],
    }
    for trans in transitions:
        gt = []
        ablations = [[] for _ in range(len(kf_models))]
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, leave=False)):
                gt_motion, ablation_motion = eval_iter(batch, trans)
                gt.append(gt_motion)
                for idx, ablation in enumerate(ablation_motion):
                    ablations[idx].append(ablation)

        gt = torch.cat(gt, dim=0)
        for idx, ablation in enumerate(ablations):
            ablations[idx] = torch.cat(ablation, dim=0)

        print(f"Results on {trans} transition frames")

        # L2P
        l2p_ablations = []
        for ablation in ablations:
            l2p_ablations.append(eval.l2p(gt, ablation, skeleton, l2p_mean, l2p_std, config.context_frames))
            print(f"L2P (ablation{idx+1}): {l2p_ablations[-1]:.3f}")

        # L2Q
        l2q_ablations = []
        for ablation in ablations:
            l2q_ablations.append(eval.l2q(gt, ablation, config.context_frames))
            print(f"L2Q (ablation{idx+1}): {l2q_ablations[-1]:.3f}")

        # NPSS
        npss_ablations = []
        for ablation in ablations:
            npss_ablations.append(eval.npss(gt, ablation, config.context_frames))
            print(f"NPSS (ablation{idx+1}): {npss_ablations[-1]:.3f}")

        # save
        results["l2p"].append([l2p for l2p in l2p_ablations])
        results["l2q"].append([l2q for l2q in l2q_ablations])
        results["npss"].append([npss for npss in npss_ablations])

    # save in text for latex table
    def get_row(method, metric, idx):
        row = str(method)
        for i in range(len(transitions)):
            if metric in ["l2p", "l2q"]:
                row += f" & {results[metric][i][idx]:.2f}"
            else:
                row += f" & {results[metric][i][idx] * 10:.2f}"
        return row + " \\\\\n"
    
    # save in text for latex table
    with open(f"eval/ablation-{args.dataset}.txt", "w") as f:
        row = f"$t_\\mathrm{{trans}}$"
        for _ in range(3): # l2p, l2q, npss
            for t in transitions:
                row += f" & {t}"
        f.write(f"{row} \\\\ \\midrule\n")

        for i in range(len(kf_models)):
            # method name is "wo-something"
            method = args.kf_configs[i].split(".")[0].split("-")
            if len(method) == 1:
                method = "Ours"
            else:
                method = f"w/o {method[-1].capitalize()}"
            f.write(get_row(method, i))