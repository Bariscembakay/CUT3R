#!/usr/bin/env python
"""Overlap tables for the overlap sampler (base dataset `overlap=`): for every
frame i of every scene, the co-visibility of i and i + g for each gap g in
OVERLAP_GAPS. Frames come from the loader's own _read_frame, so depth, intrinsics
and poses are exactly what training sees (native resolution, before crops).

overlap(i, j) = min(o(i->j), o(j->i)); o(a->b) is the share of a's valid depth
points (at most 1024, a fixed random subset) that land in front of b and inside
its image, and are not hidden behind b's own depth there (10 % margin). Where b
has no depth (sparse LiDAR), an in-frustum point counts as visible.

Writes <out>/<name>.npy (float16 [frames, gaps], -1 past a scene's end) and
<out>/<name>_index.json ({"gaps": [...], "scenes": {key: [offset, frames]}})."""
import argparse
import json
import os
import os.path as osp
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", "src"))
from dust3r.datasets import *  # noqa: F401,F403,E402  (dataset classes for eval())
from dust3r.datasets.base.base_multiview_dataset import OVERLAP_GAPS  # noqa: E402

MAX_PTS = 1024
DS = None


def frame_points(depth, K, pose, rng):
    v, u = np.nonzero(depth > 0)
    if len(u) > MAX_PTS:
        keep = rng.choice(len(u), MAX_PTS, replace=False)
        u, v = u[keep], v[keep]
    z = depth[v, u]
    X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    return X @ pose[:3, :3].T + pose[:3, 3]


def visible(Xw, depth, K, pose):
    if len(Xw) == 0:
        return 0.0
    w2c = np.linalg.inv(pose)
    Xc = Xw @ w2c[:3, :3].T + w2c[:3, 3]
    z = Xc[:, 2]
    front = z > 1e-3
    u = np.round(K[0, 0] * Xc[:, 0] / np.where(front, z, 1) + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * Xc[:, 1] / np.where(front, z, 1) + K[1, 2]).astype(int)
    H, W = depth.shape
    inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    d = np.zeros_like(z)
    d[inside] = depth[v[inside], u[inside]]
    occluded = inside & (d > 0) & (z > d * 1.1)
    return float((inside & ~occluded).sum() / len(Xw))


def scene_table(sid):
    ids = DS.scene_img_list[sid]
    rng = np.random.default_rng(sid)
    frames = []
    for gid in ids:
        _, depth, K, pose = DS._read_frame(gid, rgb=False)
        pose = np.asarray(pose, np.float64)
        frames.append((depth, np.asarray(K, np.float64), pose, frame_points(depth, K, pose, rng)))
    T = np.full((len(ids), len(OVERLAP_GAPS)), -1.0, np.float32)
    for i in range(len(ids)):
        for k, g in enumerate(OVERLAP_GAPS):
            j = i + g
            if j >= len(ids):
                break
            a, b = frames[i], frames[j]
            T[i, k] = min(visible(a[3], b[0], b[1], b[2]), visible(b[3], a[0], a[1], a[2]))
    return DS.scene_key(sid), T.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="loader expression, as in the training config")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-scenes", type=int, default=0)
    args = ap.parse_args()
    global DS
    DS = eval(args.dataset)  # workers inherit it by fork
    sids = list(range(len(DS.scene_img_list)))[: args.max_scenes or None]
    print(f"{args.name}: {len(sids)} scenes, {sum(len(DS.scene_img_list[s]) for s in sids)} frames", flush=True)
    os.makedirs(args.out, exist_ok=True)
    tables, done = {}, 0
    with Pool(args.workers) as pool:
        for key, T in pool.imap_unordered(scene_table, sids, chunksize=1):
            tables[key] = T
            done += 1
            if done % 50 == 0 or done == len(sids):
                print(f"{done}/{len(sids)}", flush=True)
    keys = sorted(tables)
    offsets = np.cumsum([0] + [len(tables[k]) for k in keys])
    np.save(osp.join(args.out, args.name + ".npy"), np.concatenate([tables[k] for k in keys]))
    json.dump({"gaps": list(OVERLAP_GAPS), "scenes": {k: [int(o), len(tables[k])] for k, o in zip(keys, offsets)}},
              open(osp.join(args.out, args.name + "_index.json"), "w"))
    allT = np.concatenate([tables[k] for k in keys]).astype(np.float32)
    med = [np.median(allT[:, k][allT[:, k] >= 0]) if (allT[:, k] >= 0).any() else np.nan
           for k in range(len(OVERLAP_GAPS))]
    print("median overlap by gap: " + ", ".join(f"{g}:{m:.2f}" for g, m in zip(OVERLAP_GAPS, med)))


if __name__ == "__main__":
    main()
