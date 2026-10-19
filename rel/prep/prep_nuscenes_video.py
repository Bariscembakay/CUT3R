#!/usr/bin/env python
"""nuScenes (trainval) cameras -> per-frame video format read by
`VideoRGBD_Multi`, with sparse metric depth from LiDAR (default: front camera
only; with several --channels, scene dirs are scene_<name>__<channel>):

  <out>/scans_train/scene_<name>/color/<ts>.jpg   512x288 from CAM_FRONT (1600x900, ~12 Hz)
                                /depth/<ts>.png   uint16 mm: the LIDAR_TOP sweep nearest in
                                                  time, moved into the camera through the
                                                  global ego poses at both timestamps
                                /cam/<ts>.npz     pose (cam-to-world, OpenCV), intrinsics
                                /new_scene_metadata.npz

Poses are global (map) frame minus the scene's first camera position, so the
numbers stay small in float32. Frames with no LiDAR sweep within 30 ms are dropped.
"""
import argparse
import json
import os
import os.path as osp
import shutil
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

W_OUT, H_OUT = 512, 288
MAX_LIDAR_DT_US = 30_000
MIN_DEPTH = 0.5


def se3(translation, rotation_wxyz):
    T = np.eye(4)
    w, x, y, z = rotation_wxyz
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = translation
    return T


def project_depth(pts_cam, K, w, h):
    z = pts_cam[:, 2]
    keep = z > MIN_DEPTH
    p, z = pts_cam[keep], z[keep]
    u = np.round(K[0, 0] * p[:, 0] / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * p[:, 1] / z + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[ok], v[ok], z[ok]
    depth = np.full((h, w), np.inf)
    np.minimum.at(depth, (v, u), z)  # nearest return per pixel
    depth[~np.isfinite(depth)] = 0
    return np.clip(np.round(depth * 1000), 0, 65535).astype(np.uint16)


def convert(job):
    name, cams, lidars, root, out = job
    final = osp.join(out, "scans_train", f"scene_{name}")
    if osp.exists(osp.join(final, "new_scene_metadata.npz")):
        return name, "done"
    od = final + ".tmp"
    shutil.rmtree(od, ignore_errors=True)
    try:
        for sub in ("color", "depth", "cam"):
            os.makedirs(osp.join(od, sub), exist_ok=True)
        lid_ts = np.array([l["timestamp"] for l in lidars])
        origin, names = None, []
        for c in cams:
            j = int(np.argmin(np.abs(lid_ts - c["timestamp"])))
            if abs(lid_ts[j] - c["timestamp"]) > MAX_LIDAR_DT_US:
                continue
            img = cv2.imread(osp.join(root, c["filename"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h0, w0 = img.shape[:2]
            c2w = c["ego"] @ c["calib"]
            pts = np.fromfile(osp.join(root, lidars[j]["filename"]), dtype=np.float32).reshape(-1, 5)[:, :3]
            lid2w = lidars[j]["ego"] @ lidars[j]["calib"]
            pts_cam = (np.linalg.inv(c2w) @ lid2w @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
            sx, sy = W_OUT / w0, H_OUT / h0
            K = c["K"]
            Ko = np.array([[K[0, 0] * sx, 0, (K[0, 2] + 0.5) * sx - 0.5],
                           [0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5], [0, 0, 1]], np.float32)
            depth = project_depth(pts_cam, Ko, W_OUT, H_OUT)
            if origin is None:
                origin = c2w[:3, 3].copy()
            pose = c2w.copy()
            pose[:3, 3] -= origin
            base = str(c["timestamp"])
            cv2.imwrite(osp.join(od, "color", base + ".jpg"),
                        cv2.resize(img, (W_OUT, H_OUT), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(osp.join(od, "depth", base + ".png"), depth)
            np.savez(osp.join(od, "cam", base + ".npz"), pose=pose.astype(np.float32), intrinsics=Ko)
            names.append(base)
        if len(names) < 64:
            shutil.rmtree(od, ignore_errors=True)
            return name, f"only {len(names)} frames"
        np.savez(osp.join(od, "new_scene_metadata.npz"), images=np.array(names))
        os.rename(od, final)
        return name, f"ok {len(names)}"
    except Exception as e:  # one bad scene must not kill the batch
        shutil.rmtree(od, ignore_errors=True)
        return name, f"error {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/nuscenes_full/trainval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--channels", nargs="+", default=["CAM_FRONT"])
    args = ap.parse_args()
    meta = osp.join(args.root, "v1.0-trainval")
    load = lambda f: json.load(open(osp.join(meta, f)))  # noqa: E731
    channel = {s["token"]: s["channel"] for s in load("sensor.json")}
    calib = {c["token"]: c for c in load("calibrated_sensor.json")}
    scene_of_sample = {s["token"]: s["scene_token"] for s in load("sample.json")}
    scene_name = {s["token"]: s["name"] for s in load("scene.json")}
    print("loading sample_data.json ...", flush=True)
    keep = {}
    for sd in load("sample_data.json"):
        ch = channel[calib[sd["calibrated_sensor_token"]]["sensor_token"]]
        if ch in args.channels or ch == "LIDAR_TOP":
            keep[sd["token"]] = (ch, sd)
    print(f"kept {len(keep)} {'/'.join(args.channels)}/LIDAR_TOP records; loading ego_pose.json ...", flush=True)
    ego = {e["token"]: se3(e["translation"], e["rotation"])
           for e in load("ego_pose.json") if e["token"] in keep}
    per_scene = defaultdict(lambda: defaultdict(list))
    for ch, sd in keep.values():
        cs = calib[sd["calibrated_sensor_token"]]
        rec = dict(timestamp=sd["timestamp"], filename=sd["filename"], ego=ego[sd["ego_pose_token"]],
                   calib=se3(cs["translation"], cs["rotation"]))
        if ch != "LIDAR_TOP":
            rec["K"] = np.array(cs["camera_intrinsic"], float)
        per_scene[scene_name[scene_of_sample[sd["sample_token"]]]][ch].append(rec)
    jobs = []
    by_time = lambda x: sorted(x, key=lambda r: r["timestamp"])  # noqa: E731
    for name in sorted(per_scene)[: args.max_scenes or None]:
        lidars = by_time(per_scene[name]["LIDAR_TOP"])
        for ch in args.channels:
            out_name = f"{name}__{ch}" if len(args.channels) > 1 else name
            jobs.append((out_name, by_time(per_scene[name][ch]), lidars, args.root, args.out))
    print(f"{len(jobs)} scene videos", flush=True)
    os.makedirs(args.out, exist_ok=True)
    log = {}
    with Pool(args.workers) as pool:
        for name, status in pool.imap_unordered(convert, jobs):
            log[name] = status
            print(f"{len(log)}/{len(jobs)} {name}: {status}", flush=True)
    json.dump(log, open(osp.join(args.out, "prep_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
