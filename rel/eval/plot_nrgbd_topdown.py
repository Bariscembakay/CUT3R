#!/usr/bin/env python
"""Per-scene top-down NRGBD trajectories: GT, released CUT3R, a CUT3R-rel arm (--ours).
Unaligned, in frame-0 coordinates. The first camera can be tilted indoors, so
positions are projected onto the plane perpendicular to the mean GT camera
up-vector -- one fixed projection shared by all three trajectories, not a fit.
Writes one PNG and one CSV per scene."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze_chunked import ate, load_run, rpe  # noqa: E402

RELEASED = "/group/compact-3dmem/campaigns/frame_anchor/cut3r_pub_palindrome_s1"
SCENES = ["breakfast_room", "complete_kitchen", "green_room", "grey_white_room", "kitchen",
          "morning_apartment", "staircase", "thin_geometry", "whiteroom"]


def ground_basis(G):
    ok = np.isfinite(G).all(axis=(1, 2))
    up = -(G[ok, :3, 1]).mean(0)            # camera -y (OpenCV) is up
    up /= np.linalg.norm(up)
    fwd = G[ok][0, :3, 2] - up * (G[ok][0, :3, 2] @ up)   # first camera's heading, levelled
    e2 = fwd / np.linalg.norm(fwd)
    e1 = np.cross(e2, up)
    return e1, e2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ours", default="/group/compact-3dmem/campaigns/cut3r_rel/pilot_eval/pilot_rel_continuous",
                    help="CUT3R-rel eval arm dir")
    ap.add_argument("--label", default="CUT3R-rel (ours)")
    args = ap.parse_args()
    RUNS = {"CUT3R (released)": (RELEASED, "#2a78d6"), args.label: (args.ours, "#1baf7a")}
    os.makedirs(args.out, exist_ok=True)
    for s in SCENES:
        traj, G = {}, None
        for name, (root, _) in RUNS.items():
            _, P, G, _, _ = load_run(f"{root}/nrgbd/{s}")
            traj[name] = P
        e1, e2 = ground_basis(G)
        proj = lambda T: np.stack([T[:, :3, 3] @ e1, T[:, :3, 3] @ e2], 1)  # noqa: E731
        fig, ax = plt.subplots(figsize=(7, 7))
        g = proj(G)
        ax.plot(g[:, 0], g[:, 1], color="#52514e", lw=2, ls="--", label="GT")
        cols = [g]
        for name, (_, color) in RUNS.items():
            P = traj[name]
            p = proj(P)
            rot, tr = rpe(P, G)
            ax.plot(p[:, 0], p[:, 1], color=color, lw=1.4,
                    label=f"{name}: ATE Sim(3) {ate(P, G, True):.2f} m, SE(3) {ate(P, G, False):.2f} m, "
                          f"RPE@10 {rot:.2f} deg / {tr:.3f} m")
            cols.append(p)
        ax.plot(0, 0, "o", color="#0b0b0b", ms=6, label="start")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("right of first camera (m)")
        ax.set_ylabel("ahead of first camera (m)")
        ax.set_title(f"NRGBD {s}: top-down (gravity-levelled), unaligned, frame-0 coordinates", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(f"{args.out}/nrgbd_{s}_topdown.png", dpi=120)
        plt.close(fig)
        np.savetxt(f"{args.out}/nrgbd_{s}_topdown.csv", np.concatenate(cols, 1), delimiter=",", fmt="%.4f",
                   header="gt_right,gt_ahead,cut3r_right,cut3r_ahead,cut3r_rel_right,cut3r_rel_ahead", comments="")
        print("wrote", s)


if __name__ == "__main__":
    main()
