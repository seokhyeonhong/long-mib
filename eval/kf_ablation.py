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
    parser.add_argument("--config", type=str, default="default.yaml")
    parser.add_argument("--kf_config", type=str, default="keyframe.yaml")
    parser.add_argument("--ref_config", type=str, default="refine.yaml")
    args = parser.parse_args()

    args.interp = False
    args.ts_configs = []
    args.rmi_configs = []
    args.ours_configs = [args.kf_config, args.ref_config]

    # evaluator
    evaluator = eval.Evaluator(args)
    transitions = [60, 90]
    results = {
        "tags": None,
        "l2p": [],
        "l2q": [],
        "npss": [],
        "foot skate": [],
    }
    if args.dataset == "lafan1":
        threshold = [0.68, 0.68]
        topk = [3, 4]
        random = [0.05, 0.05]
    elif args.dataset == "human36m":
        threshold = [0.72, 0.71]
        topk = [3, 4]
        random = [0.05, 0.05]
    for tidx, trans in enumerate(tqdm(transitions)):
        num_frames = evaluator.config.context_frames + trans + 1
        motion_list, contact_list = [], []
        tags, skeleton = None, None
        for kf_sampling in [["topk", topk[tidx]]]:
        # for kf_sampling in [["score", None]]:
        # for kf_sampling in [["threshold", threshold[tidx]], ["topk", topk[tidx]], ["random", random[tidx]]]:
            n_batch, n_kf = 0, 0
            for idx, res in enumerate(evaluator.eval(num_frames, kf_sampling)):
                motion_list.append(res["motions"])
                contact_list.append(res["contacts"])
                if idx == 0:
                    skeleton = res["skeleton"]
                if results["tags"] is None:
                    results["tags"] = res["tags"][1:]
                
                keyframes = res["keyframes"][1:][0]
                n_batch += len(keyframes)
                for kf in keyframes:
                    n_kf += len(kf)
            print(f"Average number of keyframes: {n_kf / n_batch:.2f}")

        # concat (0: GT, 1~: others)
        motion_list = [torch.cat([motion[i] for motion in motion_list], dim=0) for i in range(len(motion_list[0]))]
        contact_list = [torch.cat([contact[i] for contact in contact_list], dim=0) for i in range(len(contact_list[0]))]

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

        results["l2p"].append(l2p_list)
        results["l2q"].append(l2q_list)
        results["npss"].append(npss_list)
        results["foot skate"].append(fs_list)

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
    with open(f"eval/ablation-kf-{args.dataset}.txt", "w") as f:
        for metric in ["l2p", "l2q", "npss", "foot skate"]:
            f.write(f"{metric.upper() if metric != 'foot skate' else 'Foot skate'}" + "\n")

            row = f"$t_\\mathrm{{trans}}$"
            for t in transitions:
                row += f" & {t}"
            f.write(f"{row} \\\\ \\midrule\n")

            for i in range(len(results["tags"])):
                f.write(get_row(results["tags"][i], metric, i))
            f.write("\n")