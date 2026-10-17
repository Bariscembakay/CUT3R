#!/usr/bin/env python
"""Per-sequence top-down KITTI trajectories: GT, released CUT3R, CUT3R-rel.
Unaligned, in frame-0 coordinates (every trajectory starts at the origin facing
+z). KITTI cameras are y-down, so the ground plane is x-z: plotted as x
(right) vs z (forward). Writes one PNG and one CSV per sequence."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze_chunked import kitti_odometry_error  # noqa: E402
from analyze_chunked import ate, load_run  # noqa: E402

RUNS = {"CUT3R (released)": ("/group/compact-3dmem/campaigns/frame_anchor/cut3r_pub_palindrome_s1", "#2a78d6"),
        "CUT3R-rel (ours)": ("/group/compact-3dmem/campaigns/cut3r_rel/pilot_eval/pilot_rel_continuous", "#1baf7a")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for s in ["%02d" % i for i in range(11)]:
        traj, G = {}, None
        for name, (root, _) in RUNS.items():
            _, P, G, _, _ = load_run(f"{root}/kitti/{s}")
            traj[name] = P
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(G[:, 0, 3], G[:, 2, 3], color="#52514e", lw=2, ls="--", label="GT")
        rows = [G[:, [0, 2], 3]]
        for name, (_, color) in RUNS.items():
            P = traj[name]
            t, _ = kitti_odometry_error(P, G)
            ax.plot(P[:, 0, 3], P[:, 2, 3], color=color, lw=1.6,
                    label=f"{name}: t_err {t:.0f}%, ATE Sim(3) {ate(P, G, True):.0f} m, SE(3) {ate(P, G, False):.0f} m")
            rows.append(P[:, [0, 2], 3])
        ax.plot(0, 0, "o", color="#0b0b0b", ms=6, label="start")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z, forward (m)")
        ax.set_title(f"KITTI {s}: top-down, unaligned, frame-0 coordinates", fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(f"{args.out}/kitti_{s}_xz.png", dpi=120)
        plt.close(fig)
        np.savetxt(f"{args.out}/kitti_{s}_xz.csv", np.concatenate(rows, 1), delimiter=",", fmt="%.4f",
                   header="gt_x,gt_z,cut3r_x,cut3r_z,cut3r_rel_x,cut3r_rel_z", comments="")
        print("wrote", s)


if __name__ == "__main__":
    main()
