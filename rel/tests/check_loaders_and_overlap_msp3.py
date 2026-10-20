#!/usr/bin/env python
"""Real-data checks on msp3 (all training datasets local there).

dump <src> <out.npz>: views from each loader at fixed (idx, aspect, seed), using the
    code under <src> (run once per revision; compare with `compare`).
compare <a.npz> <b.npz>: every array bit-identical.
tables <overlap dir>: overlap tables for the first 3 scenes of each dataset.
sample <overlap dir>: per dataset, the overlap sampler on the scenes that have a
    table; consecutive pairs' overlap recomputed from the frames, and the gap
    distribution."""
import importlib
import json
import subprocess
import sys

import numpy as np

RES = "[(512, 384), (512, 336), (512, 288), (512, 256), (512, 208), (512, 144), (384, 512), (336, 512), (288, 512), (256, 512)]"
# ImgNorm, not SeqColorJitter: the jitter draws from an unseeded generator, so two
# runs of identical loader code would still differ in `img`
KW = f"video_only=True, no_raymap=True, aug_crop=16, resolution={RES}, transform=ImgNorm, num_views=32, n_corres=0"
DATASETS = {
    "arkitscenes": f"VideoRGBD_Multi(split='train', ROOT='/data/arkitscenes-cut3r-video-full', name='arkitscenes_video', max_interval=4, {KW}",
    "tartanair_v1": f"TartanAirRaw_Multi(ROOT='/data/tartanair-v1', {KW}",
    "tartanair_v2": f"TartanAirV2Raw_Multi(ROOT='/data/tartanair-v2', {KW}",
    "scannet": f"ScanNetExtracted_Multi(split='train', ROOT='/data/scannet-v2-extracted', max_interval=30, {KW}",
    "nuscenes_6cam": f"VideoRGBD_Multi(split='train', ROOT='/data/nuscenes-cut3r-video-6cam', name='nuscenes_6cam', max_interval=12, {KW}",
    "tartanground": f"TartanGroundRaw_Multi(ROOT='/data/tartanground', {KW}",
}


def load(src):
    sys.path.insert(0, src)
    return importlib.import_module("dust3r.datasets")


def dump(src, out):
    mod = load(src)
    arrays = {}
    for name, spec in DATASETS.items():
        ds = eval(spec + ", seed=123)", vars(mod))
        for i, idx in enumerate((0, len(ds) // 2, len(ds) - 1)):
            for v, view in enumerate(ds[(idx, i * 3 % 10, 32)]):
                for k in ("img", "depthmap", "camera_pose", "camera_intrinsics", "pts3d", "valid_mask", "ray_map"):
                    arrays[f"{name}/{i}/{v}/{k}"] = np.asarray(view[k])
        print(name, "dumped", flush=True)
    np.savez(out, **arrays)


def compare(a, b):
    A, B = np.load(a), np.load(b)
    assert set(A.files) == set(B.files), set(A.files) ^ set(B.files)
    bad = [k for k in A.files if not np.array_equal(A[k], B[k])]
    print(f"{len(A.files)} arrays, {len(bad)} differ" + (f": {bad[:5]}" if bad else " -- bit-identical"))
    assert not bad


def sample(odir):
    sys.path.insert(0, "src")
    sys.path.insert(0, "rel/prep")
    mod = load("src")
    from compute_overlap import frame_points, visible
    print("| dataset | clips | min / median recomputed overlap of consecutive frames | gap shares |")
    print("|---|---|---|---|")
    for name, spec in DATASETS.items():
        ds = eval(spec + f", overlap='{odir}/{name}', overlap_tau=0.3, seed=7)", vars(mod))
        have = set(json.load(open(f"{odir}/{name}_index.json"))["scenes"])
        starts = [s for s in ds.start_img_ids if ds.scene_key(ds.sceneids[s]) in have]
        rng = np.random.default_rng(0)
        ovs, gaps = [], []
        for _ in range(20):
            s = starts[int(rng.integers(len(starts)))]
            ids = ds.scene_img_list[ds.sceneids[s]]
            pos, _ = ds.get_seq_from_start_id(32, s, ids, rng)
            gaps += np.diff(pos).tolist()
            for p, q in list(zip(pos, pos[1:]))[::4]:
                fa, fb = (ds._read_frame(ids[x], rgb=False) for x in (p, q))
                pa = frame_points(fa[1], fa[2], np.asarray(fa[3], float), np.random.default_rng(p))
                pb = frame_points(fb[1], fb[2], np.asarray(fb[3], float), np.random.default_rng(q))
                ovs.append(min(visible(pa, fb[1], fb[2], np.asarray(fb[3], float)),
                               visible(pb, fa[1], fa[2], np.asarray(fa[3], float))))
        g = np.array(gaps)
        share = {int(x): round(float(np.mean(g == x)), 2) for x in np.unique(g)}
        print(f"| {name} | 20 | {min(ovs):.2f} / {np.median(ovs):.2f} | {share} |", flush=True)


def tables(odir, scenes="3"):
    for name, spec in DATASETS.items():
        subprocess.run([sys.executable, "rel/prep/compute_overlap.py", "--dataset", spec + ")", "--name", name,
                        "--out", odir, "--max-scenes", scenes, "--workers", scenes], check=True)


if __name__ == "__main__":
    {"dump": dump, "compare": compare, "tables": tables, "sample": sample}[sys.argv[1]](*sys.argv[2:])
