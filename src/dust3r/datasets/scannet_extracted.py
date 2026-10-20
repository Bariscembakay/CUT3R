import json
import os
import os.path as osp

import cv2
import numpy as np

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class ScanNetExtracted_Multi(BaseMultiViewDataset):
    """ScanNet v2 as extracted from .sens: ROOT/<scene>/{color/%06d.jpg (1296x968),
    depth/%06d.png (640x480, mm), poses.txt (one cam-to-world 4x4 per line),
    intrinsics.json}. Follows datasets_preprocess/preprocess_scannet.py: colour
    is resized to the depth grid and the depth intrinsics are used. Frames with
    a non-finite pose (ScanNet's tracking failures) are dropped."""

    def __init__(self, *args, ROOT, max_interval=30, video_prob=0.6, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = max_interval
        self.video_prob = video_prob
        super().__init__(*args, **kwargs)
        self._load_data()

    def _load_data(self):
        offset, scenes, sceneids, images, start_img_ids, scene_img_list = 0, [], [], [], [], []
        self.scene_poses, self.scene_K, self.scene_shift = [], [], []
        cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
        for scene in sorted(os.listdir(self.ROOT)):
            d = osp.join(self.ROOT, scene)
            if not osp.exists(osp.join(d, "poses.txt")) or not self.keep_scene(scene):
                continue
            meta = json.load(open(osp.join(d, "intrinsics.json")))
            poses = np.loadtxt(osp.join(d, "poses.txt")).reshape(-1, 4, 4)
            ok = np.where(np.isfinite(poses).all(axis=(1, 2)))[0]
            if len(ok) < cut_off:
                continue
            img_ids = list(np.arange(len(ok)) + offset)
            start_img_ids.extend(img_ids[: len(ok) - cut_off + 1])
            sceneids.extend([len(scenes)] * len(ok))
            images.extend(int(i) for i in ok)
            scene_img_list.append(img_ids)
            scenes.append(scene)
            self.scene_poses.append(poses.astype(np.float32))
            self.scene_K.append(np.array(meta["intrinsic_depth"], np.float32)[:3, :3])
            self.scene_shift.append(float(meta.get("depth_shift", 1000.0)))
            offset += len(ok)
        self.scenes, self.sceneids, self.images = scenes, sceneids, images
        self.start_img_ids, self.scene_img_list = start_img_ids, scene_img_list

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def scene_key(self, sid):
        return self.scenes[sid]

    def _read_frame(self, view_idx, rgb=True):
        """Native-resolution frame: (image or None, depth in m with 0 = invalid, K, cam-to-world)."""
        sid = self.sceneids[view_idx]
        d = osp.join(self.ROOT, self.scenes[sid])
        f = self.images[view_idx]
        depthmap = imread_cv2(osp.join(d, "depth", f"{f:06d}.png"), cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32) / self.scene_shift[sid]
        depthmap[~np.isfinite(depthmap)] = 0
        image = None
        if rgb:
            image = imread_cv2(osp.join(d, "color", f"{f:06d}.jpg"))
            image = cv2.resize(image, depthmap.shape[::-1], interpolation=cv2.INTER_LANCZOS4)
        return image, depthmap, self.scene_K[sid].copy(), self.scene_poses[sid][f]

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views, start_id, all_image_ids, rng, max_interval=self.max_interval,
            video_prob=self.video_prob, fix_interval_prob=0.6, block_shuffle=16,
        )
        views = []
        for v, view_idx in enumerate(np.array(all_image_ids)[pos]):
            sid = self.sceneids[view_idx]
            f = self.images[view_idx]
            rgb, depthmap, K, camera_pose = self._read_frame(view_idx)
            rgb, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb, depthmap, K, resolution, rng=rng, info=view_idx
            )
            img_mask, ray_mask = self.get_img_and_ray_masks(self.is_metric, v, rng, p=[0.75, 0.2, 0.05])
            views.append(dict(
                img=rgb, depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose, camera_intrinsics=intrinsics.astype(np.float32),
                dataset="ScanNet", label=f"{self.scenes[sid]}_{f:06d}", instance=f"{idx}_{view_idx}",
                is_metric=self.is_metric, is_video=ordered_video,
                quantile=np.array(0.98, dtype=np.float32), img_mask=img_mask, ray_mask=ray_mask,
                camera_only=False, depth_only=False, single_view=False, reset=False,
            ))
        assert len(views) == num_views
        return views
