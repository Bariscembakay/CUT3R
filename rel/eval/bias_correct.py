#!/usr/bin/env python
"""How much of a model's KITTI error is a constant per-frame calibration error?
Corrects every predicted relative pose with constants, re-chains the trajectory
and rescores it:
  scale        |t_pred| = k |t_gt| per frame (LS through the origin); translations / k
  rot offset   one rotation R_c left-multiplied onto every relative rotation
               (chordal L2 mean of R_gt R_pred^T)
  yaw gain     yaw_pred = g yaw_gt + b per frame (LS); yaw <- (yaw - b) / g
Gains are regressed pred-on-GT and inverted, so the correction is unbiased given
GT: it is the accumulated bias that drifts, while per-frame noise averages out.
"oracle" fits each sequence on itself (an upper bound); "LOO" fits on the other
ten, i.e. one calibration constant shipped with the model."""
import argparse
import os.path as osp
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from analyze_chunked import load_run, pose_scores  # noqa: E402
from analyze_pilot import RUNS  # noqa: E402

SEQS = [f"{i:02d}" for i in range(11)]
KEYS = ["t_err", "r_err", "ate_sim3", "ate_se3", "rpe10_rot", "rpe10_trans"]
VARIANTS = {"raw": (), "scale": ("scale",), "rot offset + scale": ("offset", "scale"),
            "yaw gain + scale": ("yaw", "scale")}


def rel(P):
    return np.linalg.inv(P[:-1]) @ P[1:]


def frame_data(P, G):
    Rp, Rg = rel(P), rel(G)
    return dict(Rp=Rp[:, :3, :3], Rg=Rg[:, :3, :3],
                wp=Rotation.from_matrix(Rp[:, :3, :3]).as_rotvec(), wg=Rotation.from_matrix(Rg[:, :3, :3]).as_rotvec(),
                tp=np.linalg.norm(Rp[:, :3, 3], axis=1), tg=np.linalg.norm(Rg[:, :3, 3], axis=1))


def fit(F):
    """Constants from pooled frames (a list of frame_data dicts)."""
    c = {k: np.concatenate([f[k] for f in F]) for k in F[0]}
    U, _, Vt = np.linalg.svd((c["Rg"] @ c["Rp"].transpose(0, 2, 1)).sum(0))
    g, b = np.polyfit(c["wg"][:, 1], c["wp"][:, 1], 1)
    return dict(R_c=U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt, yaw_gain=g, yaw_offset=b,
                k=(c["tp"] * c["tg"]).sum() / (c["tg"] ** 2).sum())


def correct(P, C, terms):
    T = rel(P)
    if "offset" in terms:
        T[:, :3, :3] = C["R_c"] @ T[:, :3, :3]
    if "yaw" in terms:
        w = Rotation.from_matrix(T[:, :3, :3]).as_rotvec()
        w[:, 1] = (w[:, 1] - C["yaw_offset"]) / C["yaw_gain"]
        T[:, :3, :3] = Rotation.from_rotvec(w).as_matrix()
    if "scale" in terms:
        T[:, :3, 3] /= C["k"]
    out = [P[0]]
    for Ti in T:
        out.append(out[-1] @ Ti)
    return np.stack(out)


def ypr(R):
    """Yaw (about camera y), pitch (x), roll (z) in degrees, small-angle."""
    return np.degrees([R[0, 2] - R[2, 0], R[2, 1] - R[1, 2], R[1, 0] - R[0, 1]]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="scale REL", help="row name in analyze_pilot.RUNS")
    args = ap.parse_args()
    data = {s: load_run(f"{RUNS[args.run]}/kitti/{s}")[1:3] for s in SEQS}
    F = {s: frame_data(*data[s]) for s in SEQS}

    print(f"## {args.run}: fitted constants (pred vs GT, per frame)\n")
    print("| fit | offset yaw deg/frame | offset pitch | offset roll | yaw gain | yaw offset deg/frame | translation gain |")
    print("|---|---|---|---|---|---|---|")
    for s in SEQS + ["all 11 pooled"]:
        C = fit([F[s]] if s in F else list(F.values()))
        y, p, r = -ypr(C["R_c"])  # R_c undoes the offset, so the offset is its negative
        print(f"| {s} | {y:+.4f} | {p:+.4f} | {r:+.4f} | {C['yaw_gain']:.3f} | "
              f"{np.degrees(C['yaw_offset']):+.4f} | {C['k']:.3f} |")

    c = {k: np.concatenate([f[k] for f in F.values()]) for k in ("wp", "wg", "tp", "tg")}
    print("\nMedian pred/GT ratio by GT magnitude, all frames\n\n| GT speed m/frame | frames | |t_pred| / |t_gt| |\n|---|---|---|")
    for lo, hi in ((0.05, 0.4), (0.4, 0.7), (0.7, 1.0), (1.0, 1.3), (1.3, 1.6), (1.6, 3)):
        m = (c["tg"] >= lo) & (c["tg"] < hi)
        print(f"| [{lo}, {hi}) | {m.sum()} | {np.median(c['tp'][m] / c['tg'][m]):.3f} |")
    print("\n| GT abs yaw deg/frame | frames | yaw_pred / yaw_gt |\n|---|---|---|")
    yg, yp = np.degrees(c["wg"][:, 1]), np.degrees(c["wp"][:, 1])
    for lo, hi in ((0.2, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 20)):
        m = (np.abs(yg) >= lo) & (np.abs(yg) < hi)
        print(f"| [{lo}, {hi}) | {m.sum()} | {np.median(yp[m] / yg[m]):.3f} |")

    per = {}
    for name, terms in VARIANTS.items():
        for mode in (["—"] if not terms else ["LOO", "oracle"]):
            per[(name, mode)] = {}
            for s in SEQS:
                P, G = data[s]
                if terms:
                    P = correct(P, fit([F[s]] if mode == "oracle" else [F[o] for o in SEQS if o != s]), terms)
                per[(name, mode)][s] = pose_scores(P, G, True)

    print(f"\n## {args.run}: KITTI 00-10 after correction (mean over all 11 sequences)\n")
    print("| correction | fit | " + " | ".join(KEYS) + " |\n|" + "---|" * (len(KEYS) + 2))
    for (name, mode), rows in per.items():
        print(f"| {name} | {mode} | " + " | ".join(f"{np.mean([r[k] for r in rows.values()]):.3g}" for k in KEYS) + " |")
    print("\nt_err % per sequence\n\n| seq | " + " | ".join(f"{n} ({m})" for n, m in per) + " |\n|" + "---|" * (len(per) + 1))
    for s in SEQS:
        print(f"| {s} | " + " | ".join(f"{rows[s]['t_err']:.1f}" for rows in per.values()) + " |")


if __name__ == "__main__":
    main()
