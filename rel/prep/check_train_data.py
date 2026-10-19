#!/usr/bin/env python
"""Audit what the scale-up training loaders actually feed the model, per dataset:
focal/width as seen after crop+resize, frame-to-frame motion (to compare with
KITTI's 10 Hz regime), and depth/pose consistency -- view i's depth lifted with
its pose and reprojected into view i+1 must reproduce view i+1's depth. A ratio
off 1.0 means pose translation and depth disagree in scale; a high outlier rate
means misregistered depth, wrong intrinsics or a wrong pose convention.
nuScenes: also whether its poses carry any pitch, roll or height change."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from dust3r.datasets import *  # noqa: F401,F403,E402  (dataset classes for eval())

RES = "[(512, 384), (512, 336), (512, 288), (512, 256), (512, 208), (512, 144), (384, 512), (336, 512), (288, 512), (256, 512)]"
COMMON = f"video_only=True, no_raymap=True, aug_crop=16, resolution={RES}, transform=SeqColorJitter, num_views=32, n_corres=0"
DATASETS = {
    "arkitscenes": f"VideoRGBD_Multi(split='train', ROOT='/data/arkitscenes-cut3r-video-full', name='arkitscenes_video', max_interval=4, {COMMON})",
    "tartanair_v1": f"TartanAirRaw_Multi(ROOT='/data/tartanair-v1', {COMMON})",
    "tartanair_v2": f"TartanAirV2Raw_Multi(ROOT='/data/tartanair-v2', {COMMON})",
    "scannet": f"ScanNetExtracted_Multi(split='train', ROOT='/data/scannet-v2-extracted', max_interval=30, {COMMON})",
    "nuscenes": f"VideoRGBD_Multi(split='train', ROOT='/data/nuscenes-cut3r-video', name='nuscenes_front', max_interval=12, {COMMON})",
    "tartanground": f"TartanGroundRaw_Multi(ROOT='/data/tartanground', {COMMON})",
}
HOLDOUT = {"arkitscenes": "rel/splits/arkitscenes_val.txt", "tartanair_v1": "rel/splits/tartanair_val_envs.txt",
           "tartanair_v2": "rel/splits/tartanair_val_envs.txt", "tartanground": "rel/splits/tartanair_val_envs.txt",
           "scannet": "rel/splits/scannet_val.txt", "nuscenes": "rel/splits/nuscenes_val.txt"}


def ypr(R):
    return np.degrees(np.stack([R[..., 0, 2] - R[..., 2, 0], R[..., 2, 1] - R[..., 1, 2], R[..., 1, 0] - R[..., 0, 1]], -1) / 2)


def reproject(vi, vj):
    """Median z_reproj / depth_j and the fraction within 5%, over pixels valid in both."""
    X = vi["pts3d"][vi["valid_mask"]]
    Xj = (np.linalg.inv(vj["camera_pose"]) @ np.c_[X, np.ones(len(X))].T).T[:, :3]
    z = Xj[:, 2]
    K = vj["camera_intrinsics"]
    keep = z > 0.1
    u = np.round(K[0, 0] * Xj[keep, 0] / z[keep] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * Xj[keep, 1] / z[keep] + K[1, 2]).astype(int)
    H, W = vj["depthmap"].shape
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    d = vj["depthmap"][v[ok], u[ok]]
    zz = z[keep][ok]
    m = d > 0
    return (zz[m] / d[m]) if m.sum() else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--only", nargs="*", default=list(DATASETS))
    ap.add_argument("--splits", action="store_true", help="only count train / val scenes under rel/splits")
    args = ap.parse_args()
    if args.splits:
        print("| dataset | train scenes | val scenes | train clip starts | val clip starts |\n|---|---|---|---|---|")
        for name in args.only:
            n = {m: eval(DATASETS[name][:-1] + f", holdout='{HOLDOUT[name]}', holdout_mode='{m}')")
                 for m in ("exclude", "only")}
            print(f"| {name} | {len(n['exclude'].scenes)} | {len(n['only'].scenes)} | {len(n['exclude'])} | {len(n['only'])} |",
                  flush=True)
        return
    rng = np.random.default_rng(0)
    print("| dataset | ordered clips | f/W p10/50/90 | |t| m/step p10/50/90 | rot deg/step p10/50/90 | median |yaw| / |pitch| / |roll| deg/step | reproj z/d median | within 5% |")
    print("|---|---|---|---|---|---|---|---|")
    for name in args.only:
        ds = eval(DATASETS[name])
        fw, t, rot, axes, ratios, ordered = [], [], [], [], [], []
        for _ in range(args.clips):
            views = ds[(int(rng.integers(len(ds))), int(rng.integers(10)), 32)]
            ordered.append(bool(views[0]["is_video"]))
            P = np.stack([v["camera_pose"] for v in views])
            rel = np.linalg.inv(P[:-1]) @ P[1:]
            t.append(np.linalg.norm(rel[:, :3, 3], axis=1))
            R = rel[:, :3, :3]
            rot.append(np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1))))
            axes.append(np.abs(ypr(R)))
            fw += [v["camera_intrinsics"][0, 0] / v["img"].shape[-1] for v in views]
            for i in range(0, 31, 4):
                ratios.append(reproject(views[i], views[i + 1]))
        t, rot, axes, ratios = np.concatenate(t), np.concatenate(rot), np.concatenate(axes), np.concatenate(ratios)
        q = lambda x: "/".join(f"{v:.3g}" for v in np.percentile(x, [10, 50, 90]))  # noqa: E731
        print(f"| {name} | {np.mean(ordered):.2f} | {q(fw)} | {q(t)} | {q(rot)} | "
              + " / ".join(f"{v:.3f}" for v in np.median(axes, 0)) +
              f" | {np.median(ratios):.4f} | {np.mean(np.abs(ratios - 1) < 0.05):.3f} |", flush=True)

    if "nuscenes" in args.only:
        root = "/data/nuscenes-cut3r-video/scans_train"
        scenes = sorted(os.listdir(root))
        dz, dp, dr = [], [], []
        for s in scenes[:: max(1, len(scenes) // 40)]:
            names = list(np.load(f"{root}/{s}/new_scene_metadata.npz")["images"])
            P = np.stack([np.load(f"{root}/{s}/cam/{n}.npz")["pose"] for n in names])
            rel = np.linalg.inv(P[:-1]) @ P[1:]
            a = ypr(rel[:, :3, :3])
            dz.append(np.ptp(P[:, 2, 3])); dp.append(np.abs(a[:, 1]).max()); dr.append(np.abs(a[:, 2]).max())
        print(f"\nnuScenes: {len(scenes)} scenes; over {len(dz)} sampled scenes, max per-scene world-z range "
              f"{max(dz):.4f} m, max |pitch| step {max(dp):.4f} deg, max |roll| step {max(dr):.4f} deg")


if __name__ == "__main__":
    main()
