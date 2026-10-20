#!/usr/bin/env python
"""ScanNet++ iPhone captures -> per-frame video format read by `VideoRGBD_Multi`:

  <out>/scans_train/scene_<id>/color/<frame>.jpg   512x384 from rgb.mkv (1920x1440, 60 fps)
                               /depth/<frame>.png   uint16 mm, ARKit LiDAR depth (256x192),
                                                    nearest-resized to 512x384
                               /cam/<frame>.npz     pose (cam-to-world, OpenCV), intrinsics
                               /new_scene_metadata.npz

Every --stride-th video frame is kept (6 -> 10 fps). Poses and intrinsics are the
per-frame ARKit values in pose_intrinsic_imu.json; `aligned_pose` is registered to
the laser scan, so it is metric, and is already OpenCV (x right, y down, z forward):
its frame-to-frame motion matches the COLMAP poses to 0.14 deg, an ARKit/OpenGL
axis flip to 4.2 deg. (CUT3R's own preprocess_scannetpp.py renders mesh depth for
sparse DSLR/iPhone image pairs instead: no ordered video.)
"""
import argparse
import json
import os
import os.path as osp
import shutil
from multiprocessing import Pool

import cv2
import lz4.block
import numpy as np

W_OUT, H_OUT = 512, 384
DEPTH_W, DEPTH_H = 256, 192


def read_depth_frames(path, keep):
    """depth.bin: per frame a little-endian uint32 size, then an LZ4 block of uint16 mm."""
    out = {}
    with open(path, "rb") as f:
        i = 0
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            size = int.from_bytes(head, "little")
            if i in keep:
                raw = lz4.block.decompress(f.read(size), uncompressed_size=DEPTH_W * DEPTH_H * 2)
                out[i] = np.frombuffer(raw, np.uint16).reshape(DEPTH_H, DEPTH_W)
            else:
                f.seek(size, 1)
            i += 1
    return out


def convert(job):
    scene, root, out, stride = job
    final = osp.join(out, "scans_train", f"scene_{scene}")
    if osp.exists(osp.join(final, "new_scene_metadata.npz")):
        return scene, "done"
    od = final + ".tmp"
    shutil.rmtree(od, ignore_errors=True)
    try:
        d = osp.join(root, "data", scene, "iphone")
        meta = json.load(open(osp.join(d, "pose_intrinsic_imu.json")))
        names = sorted(meta)
        keep = set(range(0, len(names), stride))
        depth = read_depth_frames(osp.join(d, "depth.bin"), keep)
        for sub in ("color", "depth", "cam"):
            os.makedirs(osp.join(od, sub), exist_ok=True)
        cap = cv2.VideoCapture(osp.join(d, "rgb.mkv"))
        kept, i = [], 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i in keep and i in depth and i < len(names):
                m = meta[names[i]]
                c2w = np.array(m["aligned_pose"], float)
                if np.isfinite(c2w).all():
                    h0, w0 = img.shape[:2]
                    sx, sy = W_OUT / w0, H_OUT / h0
                    K = np.array(m["intrinsic"], float)
                    Ko = np.array([[K[0, 0] * sx, 0, (K[0, 2] + 0.5) * sx - 0.5],
                                   [0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5], [0, 0, 1]], np.float32)
                    base = names[i]
                    cv2.imwrite(osp.join(od, "color", base + ".jpg"),
                                cv2.resize(img, (W_OUT, H_OUT), interpolation=cv2.INTER_AREA),
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(osp.join(od, "depth", base + ".png"),
                                cv2.resize(depth[i], (W_OUT, H_OUT), interpolation=cv2.INTER_NEAREST_EXACT))
                    np.savez(osp.join(od, "cam", base + ".npz"), pose=c2w.astype(np.float32), intrinsics=Ko)
                    kept.append(base)
            i += 1
        if len(kept) < 64:
            shutil.rmtree(od, ignore_errors=True)
            return scene, f"only {len(kept)} frames"
        np.savez(osp.join(od, "new_scene_metadata.npz"), images=np.array(kept))
        os.rename(od, final)
        return scene, f"ok {len(kept)}"
    except Exception as e:  # one bad scene must not kill the batch
        shutil.rmtree(od, ignore_errors=True)
        return scene, f"error {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/ScanNetpp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["nvs_sem_train", "nvs_sem_val"])
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    scenes = sorted({l.strip() for s in args.splits for l in open(osp.join(args.root, "splits", s + ".txt")) if l.strip()})
    scenes = scenes[: args.max_scenes or None]
    print(f"{len(scenes)} scenes", flush=True)
    os.makedirs(args.out, exist_ok=True)
    log = {}
    with Pool(args.workers) as pool:
        for scene, status in pool.imap_unordered(convert, [(s, args.root, args.out, args.stride) for s in scenes]):
            log[scene] = status
            print(f"{len(log)}/{len(scenes)} {scene}: {status}", flush=True)
    json.dump(log, open(osp.join(args.out, "prep_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
