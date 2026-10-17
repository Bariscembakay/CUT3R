#!/usr/bin/env python
"""CUT3R-rel pilot scoring: released CUT3R vs the frame-0 fine-tune (arm A) vs
CUT3R-rel, each streamed continuously and with reset-every-50 chaining.
Released-model records come from the frame_anchor campaign (same dump code,
same GPU type, strict fp32); the pilot ones from cut3r_rel/pilot_eval."""
import argparse
import glob
import json
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze_chunked import load_run, pose_scores  # noqa: E402
from plot_trajectories import decompose, LAG  # noqa: E402

FA = "/group/compact-3dmem/campaigns/frame_anchor"
PE = "/group/compact-3dmem/campaigns/cut3r_rel/pilot_eval"
RUNS = {
    "released": f"{FA}/cut3r_pub_palindrome_s1",
    "A (frame-0 ft)": f"{PE}/pilot_frame0_continuous",
    "CUT3R-rel": f"{PE}/pilot_rel_continuous",
    "released, reset/50": f"{FA}/cut3r_pub_chunk50_ov1_s1",
    "A, reset/50": f"{PE}/pilot_frame0_chunk50_ov1",
    "CUT3R-rel, reset/50": f"{PE}/pilot_rel_chunk50_ov1",
}
COLORS = {"released": "#2a78d6", "A (frame-0 ft)": "#eb6834", "CUT3R-rel": "#1baf7a"}
POS_BINS = [0, 50, 100, 200, 400, 800, 1600, 1e9]
GEOM = ("world_shape_med", "self_shape_med", "ws_ratio", "self_absrel")


def scenes_of(bench):
    return sorted(osp.basename(d.rstrip("/")) for d in glob.glob(f"{RUNS['released']}/{bench}/*/"))


def complete(run, bench, scenes):
    return all(osp.exists(f"{RUNS[run]}/{bench}/{s}/frames.npz") for s in scenes)


def fmt(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/group/compact-3dmem/campaigns/cut3r_rel/pilot_eval/_analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    md, summary = ["# CUT3R-rel pilot: evaluation\n"], {}

    for bench in ("kitti", "nrgbd", "kitti_depthval"):
        scenes = scenes_of(bench)
        runs = [r for r in RUNS if complete(r, bench, scenes)]
        missing = [r for r in RUNS if r not in runs]
        md.append(f"\n## {bench} — all {len(scenes)} scenes, mean over scenes"
                  + (f"  (not yet available: {', '.join(missing)})" if missing else "") + "\n")
        pose, geom, traj = {}, {}, {}
        for r in runs:
            per, g = {}, []
            for s in scenes:
                m, P, G, recs, _ = load_run(f"{RUNS[r]}/{bench}/{s}")
                if G is not None:
                    per[s] = pose_scores(P, G, bench == "kitti")
                    traj[(r, s)] = (P, G)
                if np.isfinite(recs.get("world_shape_med", np.array([np.nan]))).any():
                    ws = recs["world_shape_med"] / recs["self_shape_med"]
                    g.append({k: float(np.nanmedian(ws if k == "ws_ratio" else recs[k])) for k in GEOM})
            if per:
                pose[r] = per
            if g:
                geom[r] = g
        summary[bench] = dict(pose=pose, geom=geom)
        if pose:
            keys = (["t_err", "r_err"] if bench == "kitti" else []) + \
                ["ate_sim3", "ate_se3", "rpe10_rot", "rpe10_trans", "fwd_speed", "max_dist"]
            keys = [k for k in keys if any(np.isfinite(v[k]) for p in pose.values() for v in p.values())]
            md.append("| run | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1))
            for r, per in pose.items():
                md.append(f"| {r} | " + " | ".join(fmt(np.mean([v[k] for v in per.values()])) for k in keys) + " |")
        if geom:
            md.append("\nGeometry (median per scene, mean over scenes); world = cross pointmap "
                      "(frame 0 for released/A, camera t-1 for CUT3R-rel)\n\n| run | "
                      + " | ".join(GEOM) + " |\n|" + "---|" * (len(GEOM) + 1))
            for r, g in geom.items():
                md.append(f"| {r} | " + " | ".join(fmt(np.mean([x[k] for x in g]), 4) for k in GEOM) + " |")
        if bench == "kitti" and pose:
            md.append("\nKITTI t_err (%) per sequence\n\n| seq | " + " | ".join(pose) + " |\n|" + "---|" * (len(pose) + 1))
            for s in scenes:
                md.append(f"| {s} | " + " | ".join(fmt(pose[r][s]["t_err"]) for r in pose) + " |")
            cont = [r for r in ("released", "A (frame-0 ft)", "CUT3R-rel") if r in pose]
            md.append("\nContinuous streams by stream position: forward speed ratio / local heading drift "
                      "(deg per 10 frames)\n\n| position | " + " | ".join(cont) + " |\n|" + "---|" * (len(cont) + 1))
            dec = {}
            for r in cont:
                S, H, Pp = [], [], []
                for s in scenes:
                    P, G = traj[(r, s)]
                    _, speed, _, _ = decompose(P, G)
                    yaw = lambda T: np.degrees(np.arctan2(T[:, 0, 2], T[:, 2, 2]))  # noqa: E731
                    dy = lambda T: (yaw(T)[LAG:] - yaw(T)[:-LAG] + 180) % 360 - 180  # noqa: E731
                    S.append(speed); H.append(np.abs(dy(P) - dy(G))); Pp.append(np.arange(len(speed)))
                dec[r] = tuple(map(np.concatenate, (S, H, Pp)))
            for lo, hi in zip(POS_BINS[:-1], POS_BINS[1:]):
                cells = []
                for r in cont:
                    S, H, Pp = dec[r]
                    sel = (Pp >= lo) & (Pp < hi) & np.isfinite(S)
                    cells.append(f"{np.median(S[sel]):.2f} / {np.median(H[sel]):.2f}")
                md.append(f"| [{lo:g}, {hi:g}) | " + " | ".join(cells) + " |")

            fig, axes = plt.subplots(3, 4, figsize=(17, 12.5))
            for ax, s in zip(axes.flat, scenes):
                G = traj[(cont[0], s)][1]
                ax.plot(G[:, 0, 3], G[:, 2, 3], color="#52514e", lw=1.6, ls="--", label="GT")
                for r in cont:
                    P = traj[(r, s)][0]
                    ax.plot(P[:, 0, 3], P[:, 2, 3], color=COLORS[r], lw=1.3, label=r)
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_title(f"KITTI {s}", fontsize=9)
                ax.grid(alpha=0.25, lw=0.5)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
            axes.flat[-1].axis("off")
            h, lab = axes.flat[0].get_legend_handles_labels()
            axes.flat[-1].legend(h, lab, frameon=False, fontsize=10, loc="center left",
                                 title="continuous streams\ntop-down (x, z), metres")
            fig.tight_layout()
            fig.savefig(f"{args.out}/kitti_trajectories_continuous.png", dpi=95)
            plt.close(fig)

    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1, default=float)
    open(f"{args.out}/summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
