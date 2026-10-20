#!/usr/bin/env python
"""Field-of-view sweep: the same checkpoint on KITTI cropped to narrower views
(--crop-fw, focal/width after resize). If the scale and rotation biases come from
a focal-length prior learned on driving data seen only at focal/width >= 0.79,
depth scale (GT/pred) and yaw gain (pred/GT) should both move to 1.0 together."""
import argparse
import glob
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from analyze_chunked import load_run, pose_scores  # noqa: E402
from bias_correct import SEQS, fit, frame_data  # noqa: E402

SE = "/group/compact-3dmem/campaigns/cut3r_rel/scale_eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="scale_rel_continuous")
    ap.add_argument("--crops", nargs="+", default=["0.59", "0.68", "0.79", "0.95"])
    args = ap.parse_args()
    print("| focal/width | t_err % | t_err %, one scale per seq | r_err | ATE Sim(3) m | ATE SE(3) m | "
          "translation gain | yaw gain | depth scale GT/pred | depth abs. rel. (median-scaled) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in args.crops:
        d = f"{SE}/{args.arm}_cropfw{c.replace('.', '')}"
        runs = {s: load_run(f"{d}/kitti/{s}")[1:3] for s in SEQS}
        sc = [pose_scores(P, G, True) for P, G in runs.values()]
        C = fit([frame_data(P, G) for P, G in runs.values()])
        S, A = [], []
        for r in sorted(glob.glob(f"{d}/kitti_depthval/*/")):
            recs = load_run(r)[3]
            S.append(np.nanmedian(recs["self_shape_scale"]))
            A.append(np.nanmedian(recs["self_absrel"]))
        m = lambda k: np.mean([x[k] for x in sc])  # noqa: E731
        print(f"| {c} | {m('t_err'):.1f} | {m('t_err_scaled'):.1f} | {m('r_err'):.2f} | {m('ate_sim3'):.1f} | "
              f"{m('ate_se3'):.1f} | {C['k']:.3f} | {C['yaw_gain']:.3f} | {np.median(S):.3f} | {np.mean(A):.4f} |")


if __name__ == "__main__":
    main()
