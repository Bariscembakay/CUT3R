#!/usr/bin/env python
"""ARKitScenes raw (FullTrainingSetWithWideAssets) -> per-frame video format read
by `VideoRGBD_Multi`:

  <out>/scans_train/<scene>/color/<ts>.jpg   512x384 RGB from `wide` (1920x1440)
                           /depth/<ts>.png   uint16 mm, `lowres_depth` matched by
                                             timestamp, nearest-resized to 512x384
                           /cam/<ts>.npz     pose (cam-to-world, OpenCV), intrinsics
                           /new_scene_metadata.npz   images = time-ordered basenames

Conventions follow CUT3R's preprocess_arkitscenes.py: the trajectory stores
world->camera as axis-angle + translation, poses are interpolated to the image
timestamps, and frames are rotated so the sky is up (most scenes are recorded
with the sky to the left). Unlike it, the rotated principal point is exact
((H-1)-cy) rather than a plain swap.
"""
import argparse
import json
import os
import os.path as osp
import shutil
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

W_OUT, H_OUT = 512, 384
MAX_DEPTH_DT = 0.02


def read_traj(path):
    ts, c2w = [], []
    for line in open(path):
        t = line.split()
        if len(t) != 7:
            continue
        w2c = np.eye(4)
        w2c[:3, :3] = cv2.Rodrigues(np.array(t[1:4], float))[0]
        w2c[:3, 3] = np.array(t[4:7], float)
        ts.append(float(t[0]))
        c2w.append(np.linalg.inv(w2c))
    return np.array(ts), np.array(c2w)


# rotation -> (cv2 code, R_new<-old for camera coordinates, K transform)
ROTATIONS = {
    "none": (None, np.eye(3), lambda fx, fy, cx, cy, W, H: (fx, fy, cx, cy, W, H)),
    "cw": (cv2.ROTATE_90_CLOCKWISE, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]),
           lambda fx, fy, cx, cy, W, H: (fy, fx, (H - 1) - cy, cx, H, W)),
    "ccw": (cv2.ROTATE_90_COUNTERCLOCKWISE, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]]),
            lambda fx, fy, cx, cy, W, H: (fy, fx, cy, (W - 1) - cx, H, W)),
    "180": (cv2.ROTATE_180, np.diag([-1, -1, 1.0]),
            lambda fx, fy, cx, cy, W, H: (fx, fy, (W - 1) - cx, (H - 1) - cy, W, H)),
}


def upright_rotation(c2w):
    """Which image rotation puts world up (+z) at the top of the image (-y)."""
    g = (np.transpose(c2w[:, :3, :3], (0, 2, 1)) @ np.array([0, 0, 1.0])).mean(0)
    axis = int(np.argmax(np.abs(g[:2])))
    if axis == 1:
        return "none" if g[1] < 0 else "180"
    return "cw" if g[0] < 0 else "ccw"  # sky on the left -> rotate clockwise


def frame_ts(name):
    return float(name.rsplit("_", 1)[1][: -len(".png")])


def convert(args):
    scene, root, out = args
    sd = osp.join(root, scene)
    final = osp.join(out, "scans_train", f"scene_{scene}")
    if osp.exists(osp.join(final, "new_scene_metadata.npz")):
        return scene, "done"
    # Written under a temp name and renamed at the end, so an interrupted or
    # rejected scene never leaves a half-written directory for the loader.
    od = final + ".tmp"
    shutil.rmtree(od, ignore_errors=True)
    try:
        ts, c2w = read_traj(osp.join(sd, "lowres_wide.traj"))
        if len(ts) < 10:
            return scene, "short traj"
        rot = upright_rotation(c2w)
        code, R_no, kfn = ROTATIONS[rot]
        R_on = np.eye(4)
        R_on[:3, :3] = R_no.T
        wide = sorted(os.listdir(osp.join(sd, "wide")), key=frame_ts)
        depth_names = sorted(os.listdir(osp.join(sd, "lowres_depth")), key=frame_ts)
        depth_ts = np.array([frame_ts(n) for n in depth_names])
        slerp = Slerp(ts, Rotation.from_matrix(c2w[:, :3, :3]))
        for sub in ("color", "depth", "cam"):
            os.makedirs(osp.join(od, sub), exist_ok=True)
        names = []
        for n in wide:
            t = frame_ts(n)
            if not ts[0] <= t <= ts[-1]:
                continue
            j = int(np.argmin(np.abs(depth_ts - t)))
            if abs(depth_ts[j] - t) > MAX_DEPTH_DT:
                continue
            pin = osp.join(sd, "wide_intrinsics", n[:-4] + ".pincam")
            if not osp.exists(pin):
                continue
            w, h, fx, fy, cx, cy = np.loadtxt(pin)
            img = cv2.imread(osp.join(sd, "wide", n), cv2.IMREAD_COLOR)
            dep = cv2.imread(osp.join(sd, "lowres_depth", depth_names[j]), cv2.IMREAD_UNCHANGED)
            if img is None or dep is None or img.shape[:2] != (int(h), int(w)):
                continue
            if code is not None:
                img, dep = cv2.rotate(img, code), cv2.rotate(dep, code)
            fx, fy, cx, cy, w, h = kfn(fx, fy, cx, cy, w, h)
            wo, ho = (W_OUT, H_OUT) if w >= h else (H_OUT, W_OUT)
            sx, sy = wo / w, ho / h
            K = np.array([[fx * sx, 0, (cx + 0.5) * sx - 0.5],
                          [0, fy * sy, (cy + 0.5) * sy - 0.5],
                          [0, 0, 1]], np.float32)
            pose = np.eye(4)
            pose[:3, :3] = slerp([t]).as_matrix()[0]
            pose[:3, 3] = [np.interp(t, ts, c2w[:, k, 3]) for k in range(3)]
            pose = pose @ R_on
            base = f"{t:.3f}"
            cv2.imwrite(osp.join(od, "color", base + ".jpg"),
                        cv2.resize(img, (wo, ho), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(osp.join(od, "depth", base + ".png"),
                        cv2.resize(dep, (wo, ho), interpolation=cv2.INTER_NEAREST_EXACT))
            np.savez(osp.join(od, "cam", base + ".npz"), pose=pose.astype(np.float32), intrinsics=K)
            names.append(base)
        if len(names) < 64:
            shutil.rmtree(od, ignore_errors=True)
            return scene, f"only {len(names)} frames"
        np.savez(osp.join(od, "new_scene_metadata.npz"), images=np.array(names))
        os.rename(od, final)
        return scene, f"ok {len(names)} rot={rot}"
    except Exception as e:  # one bad scene must not kill the batch
        shutil.rmtree(od, ignore_errors=True)
        return scene, f"error {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/ARKitScenes/FullTrainingSetWithWideAssets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-scenes", type=int, default=300, help="first N usable scenes, sorted by id")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    scenes = sorted(d for d in os.listdir(args.root) if osp.isdir(osp.join(args.root, d)))
    # Over-request: short or broken scenes are rejected inside convert().
    cand = scenes[: int(args.num_scenes * 1.3) + 5]
    os.makedirs(args.out, exist_ok=True)
    log = {}
    with Pool(args.workers) as pool:
        for scene, status in pool.imap_unordered(convert, [(s, args.root, args.out) for s in cand]):
            log[scene] = status
            ok = sum(v.startswith(("ok", "done")) for v in log.values())
            print(f"{len(log)}/{len(cand)} {scene}: {status}  (kept {ok})", flush=True)
            if ok >= args.num_scenes:
                pool.terminate()
                break
    json.dump(log, open(osp.join(args.out, "prep_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
