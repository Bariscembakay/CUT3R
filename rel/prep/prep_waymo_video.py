#!/usr/bin/env python
"""Waymo Open Dataset (perception v1.4.3 .tfrecord segments) -> per-frame video
format read by `VideoRGBD_Multi`, one video per segment and camera:

  <out>/scans_train/scene_<segment>__<CAMERA>/color/<frame>.jpg   long side 512 (1920x1280 / 1920x886)
                                             /depth/<frame>.png   uint16 mm, LiDAR (all 5 sensors)
                                             /cam/<frame>.npz     pose (cam-to-world, OpenCV), intrinsics
                                             /new_scene_metadata.npz

Follows CUT3R's datasets_preprocess/preprocess_waymo.py: LiDAR points land on
the pixels Waymo projects them to (these include lens distortion and rolling
shutter), depth is the point's z in the camera, the pose is the vehicle pose at
the image's capture time x camera-to-vehicle x Waymo->OpenCV axes, and the
intrinsics are the pinhole part of the calibration. Positions are relative to
the segment's first camera so they stay small in float32. 10 Hz, ~198 frames.
Needs tensorflow + waymo-open-dataset (a separate env: see prep_waymo_job.sh)."""
import argparse
import glob
import json
import os
import os.path as osp
import shutil
from multiprocessing import Pool

import cv2
import numpy as np

CAMERAS = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
# Waymo camera axes (x forward, y left, z up) -> OpenCV (x right, y down, z forward)
WAYMO_TO_OPENCV = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)
LONG_SIDE = 512
MIN_DEPTH = 0.5


def convert(job):
    path, out = job
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import frame_utils

    seg = osp.basename(path).removesuffix(".tfrecord")
    finals = {c: osp.join(out, "scans_train", f"scene_{seg}__{n}") for c, n in CAMERAS.items()}
    if all(osp.exists(osp.join(f, "new_scene_metadata.npz")) for f in finals.values()):
        return seg, "done"
    tmps = {c: f + ".tmp" for c, f in finals.items()}
    try:
        for t in tmps.values():
            shutil.rmtree(t, ignore_errors=True)
            for sub in ("color", "depth", "cam"):
                os.makedirs(osp.join(t, sub), exist_ok=True)
        names = {c: [] for c in CAMERAS}
        calib, origin = None, {}
        for data in tf.data.TFRecordDataset(path, compression_type=""):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(data.numpy())  # bytes: newer protobuf rejects bytearray
            if calib is None:
                calib = {c.name: c for c in frame.context.camera_calibrations}
            ri, cp, _, top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
            pts, cps = frame_utils.convert_range_image_to_point_cloud(frame, ri, cp, top_pose)
            pts, cps = np.concatenate(pts, 0), np.concatenate(cps, 0)  # vehicle frame; (cam, x, y) x2
            for image in frame.images:
                c = image.name
                cal = calib[c]
                cam_to_car = np.array(cal.extrinsic.transform, float).reshape(4, 4)
                car_to_world = np.array(image.pose.transform, float).reshape(4, 4)
                c2w = car_to_world @ cam_to_car @ np.linalg.inv(WAYMO_TO_OPENCV)
                if c not in origin:
                    origin[c] = c2w[:3, 3].copy()
                c2w[:3, 3] -= origin[c]
                # LiDAR returns Waymo projects into this camera (first or second return)
                sel = np.concatenate([np.where(cps[:, 0] == c)[0], np.where(cps[:, 3] == c)[0]])
                uv = np.concatenate([cps[cps[:, 0] == c][:, 1:3], cps[cps[:, 3] == c][:, 4:6]], 0).astype(float)
                Xc = (WAYMO_TO_OPENCV @ np.linalg.inv(cam_to_car) @ np.c_[pts[sel], np.ones(len(sel))].T).T[:, :3]
                img = cv2.imdecode(np.frombuffer(image.image, np.uint8), cv2.IMREAD_COLOR)
                h0, w0 = img.shape[:2]
                s = LONG_SIDE / max(w0, h0)
                w1, h1 = round(w0 * s), round(h0 * s)
                f1, f2, cx, cy = cal.intrinsic[:4]
                K = np.array([[f1 * s, 0, (cx + 0.5) * s - 0.5], [0, f2 * s, (cy + 0.5) * s - 0.5], [0, 0, 1]], np.float32)
                u = np.round((uv[:, 0] + 0.5) * s - 0.5).astype(int)
                v = np.round((uv[:, 1] + 0.5) * s - 0.5).astype(int)
                z = Xc[:, 2]
                ok = (z > MIN_DEPTH) & (u >= 0) & (u < w1) & (v >= 0) & (v < h1)
                depth = np.full((h1, w1), np.inf)
                np.minimum.at(depth, (v[ok], u[ok]), z[ok])  # nearest return per pixel
                depth[~np.isfinite(depth)] = 0
                base = f"{len(names[c]):05d}"
                t = tmps[c]
                cv2.imwrite(osp.join(t, "color", base + ".jpg"), cv2.resize(img, (w1, h1), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(osp.join(t, "depth", base + ".png"), np.clip(np.round(depth * 1000), 0, 65535).astype(np.uint16))
                np.savez(osp.join(t, "cam", base + ".npz"), pose=c2w.astype(np.float32), intrinsics=K,
                         timestamp=np.int64(image.pose_timestamp * 1e6))
                names[c].append(base)
        for c in CAMERAS:
            np.savez(osp.join(tmps[c], "new_scene_metadata.npz"), images=np.array(names[c]))
            shutil.rmtree(finals[c], ignore_errors=True)
            os.rename(tmps[c], finals[c])
        return seg, f"ok {min(len(n) for n in names.values())}"
    except Exception as e:  # one bad segment must not kill the batch
        for t in tmps.values():
            shutil.rmtree(t, ignore_errors=True)
        return seg, f"error {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"/scratch/{os.environ['USER']}/waymo_v1_4_3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["training", "validation"])
    ap.add_argument("--max-segments", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    files = sorted(f for s in args.splits for f in glob.glob(osp.join(args.root, s, "*.tfrecord")))
    files = files[: args.max_segments or None]
    print(f"{len(files)} segments", flush=True)
    os.makedirs(args.out, exist_ok=True)
    log = {}
    with Pool(args.workers) as pool:
        for seg, status in pool.imap_unordered(convert, [(f, args.out) for f in files]):
            log[seg] = status
            print(f"{len(log)}/{len(files)} {seg}: {status}", flush=True)
    json.dump(log, open(osp.join(args.out, "prep_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
