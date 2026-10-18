#!/usr/bin/env python
"""Rotation-bias diagnosis on KITTI: signed per-frame rotation error (bias) and
its spread (noise) for a model under the default preprocessing, mirrored
video, and a crop centred on the principal point.

Reading: a bias produced by the images flips sign under mirroring; a constant
offset built into the model keeps its sign. A bias produced by the
principal-point mismatch between training crops and evaluation crops vanishes
with --center-pp. Mirroring negates GT yaw and roll, so errors are compared in
each run's own (mirrored) frame."""
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from analyze_chunked import kitti_odometry_error, load_run  # noqa: E402

PE = "/group/compact-3dmem/campaigns/cut3r_rel/pilot_eval"
SE = "/group/compact-3dmem/campaigns/cut3r_rel/scale_eval"
MODELS = {"pilot_rel": f"{PE}/pilot_rel_continuous", "pilot_rel_v2loss": f"{PE}/pilot_rel_v2loss_continuous",
          "scale_frame0": f"{SE}/scale_frame0_continuous", "scale_rel_ep5": f"{SE}/scale_rel_ep5_continuous",
          "scale_rel": f"{SE}/scale_rel_continuous"}
SEQS = ["03", "04", "06", "07", "10"]


def rates(P):
    R = np.linalg.inv(P[:-1, :3, :3]) @ P[1:, :3, :3]
    return np.degrees(np.stack([R[:, 0, 2] - R[:, 2, 0], R[:, 2, 1] - R[:, 1, 2], R[:, 1, 0] - R[:, 0, 1]], 1) / 2)


def main():
    print("| model | preprocessing | bias yaw | bias pitch | bias roll | noise yaw | noise roll | t_err % |")
    print("|---|---|---|---|---|---|---|---|")
    for model, base in MODELS.items():
        for cond, suffix in (("default", ""), ("mirrored", "_hflip"), ("principal-point crop", "_centerpp")):
            if not osp.isdir(f"{base}{suffix}"):
                continue  # condition never run for this model
            E, T = [], []
            for s in SEQS:
                d = f"{base}{suffix}/kitti/{s}"
                if not osp.exists(f"{d}/frames.npz"):
                    break
                _, P, G, _, _ = load_run(d)
                E.append(rates(P) - rates(G))
                T.append(kitti_odometry_error(P, G)[0])
            if len(T) != len(SEQS):
                print(f"| {model} | {cond} | (missing) | | | | | |")
                continue
            E = np.concatenate(E)
            b, n = np.median(E, 0), np.median(np.abs(E), 0)
            print(f"| {model} | {cond} | {b[0]:+.3f} | {b[1]:+.3f} | {b[2]:+.3f} | {n[0]:.3f} | {n[2]:.3f} | {np.mean(T):.1f} |")


if __name__ == "__main__":
    main()
