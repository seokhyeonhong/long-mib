import sys
sys.path.append(".")

import argparse
from aPyOpenGL import agl

from utils.eval import Evaluator
from vis.motionapp import MotionApp

if __name__ =="__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--interp", type=lambda s: s.lower() in ['true', '1'])
    parser.add_argument("--config", type=str, default="default.yaml")
    parser.add_argument("--traj_edit", type=str, default=None)
    parser.add_argument("--ts_configs", type=str, nargs="+", default=[]) # odd: context, even: detail
    parser.add_argument("--rmi_configs", type=str, nargs="+", default=[])
    parser.add_argument("--ours_configs", type=str, nargs="+", default=[]) # odd: keyframe, even: refine
    args = parser.parse_args()

    # evaluator
    evaluator = Evaluator(args)
    for res in evaluator.eval(traj_option=args.traj_edit):
        agl.AppManager.start(MotionApp(res["motions"], res["tags"], res["skeleton"],
                                        trajs=res["trajs"],
                                        kf_indices=res["keyframes"],
                                        dataset=args.dataset))