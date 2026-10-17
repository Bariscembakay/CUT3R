#!/usr/bin/env python
"""Checks TartanAirRaw_Multi: (1) the NED -> OpenCV pose conversion by depth
reprojection between frames two apart, (2) the loader yields a collated batch.
Usage: python check_tartanair_raw.py <tartanair_v1_root>"""
import glob
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../../src"))
from accelerate import Accelerator  # noqa: E402
from dust3r.datasets import get_data_loader  # noqa: E402
from dust3r.datasets.tartanair_raw import TARTANAIR_K, ned_poses_to_opencv_c2w  # noqa: E402

root = sys.argv[1]
K = TARTANAIR_K.astype(float)
for seq in sorted(glob.glob(f"{root}/*/Easy/P000"))[:4]:
    P = ned_poses_to_opencv_c2w(np.loadtxt(f"{seq}/pose_left.txt")).astype(float)
    errs = []
    for i in range(0, min(len(P) - 2, 400), 40):
        Da = np.load(f"{seq}/depth_left/{i:06d}_left_depth.npy")
        Db = np.load(f"{seq}/depth_left/{i + 2:06d}_left_depth.npy")
        v, u = np.nonzero((Da > 0) & (Da < 100))
        z = Da[v, u]
        X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z, np.ones_like(z)], 1)
        Xb = (np.linalg.inv(P[i + 2]) @ P[i] @ X.T).T[:, :3]
        ub = np.round(K[0, 0] * Xb[:, 0] / Xb[:, 2] + K[0, 2]).astype(int)
        vb = np.round(K[1, 1] * Xb[:, 1] / Xb[:, 2] + K[1, 2]).astype(int)
        ok = (Xb[:, 2] > 0.1) & (ub >= 0) & (ub < 640) & (vb >= 0) & (vb < 480)
        zb = Db[vb[ok], ub[ok]]
        errs.append(np.median(np.abs(Xb[ok, 2] - zb) / zb))
    print(f"{seq.split('/')[-3]}: reprojection depth rel err median {np.median(errs):.4f}")

ds = (f"TartanAirRaw_Multi(ROOT='{root}', video_only=True, no_raymap=True, "
      f"resolution=[(512, 384)], num_views=8, seed=0)")
batch = next(iter(get_data_loader(ds, batch_size=2, num_workers=0, accelerator=Accelerator(cpu=True),
                                  shuffle=False, fixed_length=True)))
print("batch:", len(batch), "views; img", tuple(batch[0]["img"].shape), "; is_video",
      batch[0]["is_video"].tolist(), "; ray_mask any", any(bool(v["ray_mask"].any()) for v in batch))
