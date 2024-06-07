import sys
sys.path.append(".")

from tqdm import tqdm
import argparse

import torch

from utils import eval

def arg_as_list(s):
    import ast
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument must be a list")
    return v

if __name__ =="__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--interp", type=lambda s: s.lower() in ['true', '1'])
    parser.add_argument("--config", type=str, default="default.yaml")
    parser.add_argument("--ts_configs", type=str, nargs="+", default=[]) # odd: context, even: detail
    parser.add_argument("--rmi_configs", type=str, nargs="+", default=[])
    parser.add_argument("--ours_configs", type=str, nargs="+", default=[]) # odd: keyframe, even: refine
    parser.add_argument("--traj_edit", type=str, default=None)
    args = parser.parse_args()

    # evaluator
    evaluator = eval.Evaluator(args)
    transitions = [15, 30, 60, 90]
    # transitions = [15, ]
    results = {
        "tags": None,
        "l2p": [],
        "l2q": [],
        "npss": [],
        "foot skate": [],
    }
    for trans in tqdm(transitions):
        num_frames = evaluator.config.context_frames + trans + 1
        motion_list, contact_list = [], []
        traj_list = []
        tags, skeleton = None, None
        for idx, res in enumerate(evaluator.eval(num_frames, traj_option=args.traj_edit)):
            motion_list.append(res["motions"])
            contact_list.append(res["contacts"])
            traj_list.append(res["trajs"])
            if idx == 0:
                skeleton = res["skeleton"]
            if results["tags"] is None:
                results["tags"] = res["tags"][1:]

        # concat (0: GT, 1~: others)
        motion_list = [torch.cat([motion[i] for motion in motion_list], dim=0) for i in range(len(motion_list[0]))]
        contact_list = [torch.cat([contact[i] for contact in contact_list], dim=0) for i in range(len(contact_list[0]))]
        traj_list = [torch.cat([traj[i] for traj in traj_list], dim=0) for i in range(len(traj_list[0]))]

        # L2P
        l2p_list = []
        for motion in motion_list[1:]:
            l2p = eval.l2p(motion_list[0], motion, skeleton, evaluator.l2p_mean, evaluator.l2p_std, evaluator.config.context_frames)
            l2p_list.append(l2p)

        # L2Q
        l2q_list = []
        for motion in motion_list[1:]:
            l2q = eval.l2q(motion_list[0], motion, evaluator.config.context_frames)
            l2q_list.append(l2q)

        # NPSS
        npss_list = []
        for motion in motion_list[1:]:
            npss = eval.npss(motion_list[0], motion, evaluator.config.context_frames)
            npss_list.append(npss)
        
        # Foot skate
        fs_list = []
        for motion, contact in zip(motion_list[1:], contact_list[1:]):
            fs = eval.foot_skate(motion, contact, skeleton, evaluator.contact_idx, ctx_frames=evaluator.config.context_frames)
            fs_list.append(fs)
        
        # Optional: traj position error
        if args.traj_edit is not None:
            pos_err_list = []
            for motion, traj in zip(motion_list[1:], traj_list[1:]):
                pos_err = eval.traj_pos_error(traj, motion, evaluator.config.context_frames)
                pos_err_list.append(pos_err)

        results["l2p"].append(l2p_list)
        results["l2q"].append(l2q_list)
        results["npss"].append(npss_list)
        results["foot skate"].append(fs_list)

        if args.traj_edit is not None:
            if "traj" not in results:
                results["traj"] = []
            results["traj"].append(pos_err_list)

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
    with open(f"eval/benchmark-{args.dataset}.txt", "w") as f:
        metric_list = ["l2p", "l2q", "npss", "foot skate"]
        if args.traj_edit is not None:
            metric_list.append("traj")
        for metric in metric_list:
            f.write(f"{metric.upper() if metric != 'foot skate' else 'Foot skate'}" + "\n")

            row = f"$t_\\mathrm{{trans}}$"
            for t in transitions:
                row += f" & {t}"
            f.write(f"{row} \\\\ \\midrule\n")

            for i in range(len(results["tags"])):
                f.write(get_row(results["tags"][i], metric, i))
            f.write("\n")