import os
import os.path as osp

import cv2
import numpy as np

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class VideoRGBD_Multi(BaseMultiViewDataset):
    """Metric RGB-D video in ScanNet_Multi's per-frame layout:

    ROOT/scans_train/<scene>/{color/<id>.jpg, depth/<id>.png (uint16 mm),
    cam/<id>.npz (pose = cam-to-world OpenCV, intrinsics)}, plus
    new_scene_metadata.npz listing the time-ordered ids. Written by the
    rel/prep/prep_*_video.py converters. `max_interval` is in frames, so set it
    per dataset from its frame rate.
    """

    def __init__(self, *args, ROOT, name, max_interval=10, video_prob=0.6, **kwargs):
        self.ROOT = ROOT
        self.name = name
        self.video = True
        self.is_metric = True
        self.max_interval = max_interval
        self.video_prob = video_prob
        super().__init__(*args, **kwargs)
        self._load_data()

    def _load_data(self):
        self.scene_root = osp.join(self.ROOT, "scans_train" if self.split == "train" else "scans_test")
        offset, scenes, sceneids, images, start_img_ids, scene_img_list = 0, [], [], [], [], []
        cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
        for scene in sorted(os.listdir(self.scene_root)):
            meta = osp.join(self.scene_root, scene, "new_scene_metadata.npz")
            # hold-out lists name the source scene: "scene_<name>[__<camera>]" -> <name>
            if not osp.exists(meta) or not self.keep_scene(scene.removeprefix("scene_").split("__")[0]):
                continue
            basenames = list(np.load(meta, allow_pickle=True)["images"])
            if len(basenames) < cut_off:
                continue
            img_ids = list(np.arange(len(basenames)) + offset)
            start_img_ids.extend(img_ids[: len(basenames) - cut_off + 1])
            sceneids.extend([len(scenes)] * len(basenames))
            images.extend(basenames)
            scene_img_list.append(img_ids)
            scenes.append(scene)
            offset += len(basenames)
        self.scenes, self.sceneids, self.images = scenes, sceneids, images
        self.start_img_ids, self.scene_img_list = start_img_ids, scene_img_list

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views, start_id, all_image_ids, rng, max_interval=self.max_interval,
            video_prob=self.video_prob, fix_interval_prob=0.6, block_shuffle=16,
        )
        views = []
        for v, view_idx in enumerate(np.array(all_image_ids)[pos]):
            scene = self.scenes[self.sceneids[view_idx]]
            d = osp.join(self.scene_root, scene)
            basename = self.images[view_idx]
            rgb_image = imread_cv2(osp.join(d, "color", basename + ".jpg"))
            depthmap = imread_cv2(osp.join(d, "depth", basename + ".png"), cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0
            cam = np.load(osp.join(d, "cam", basename + ".npz"))
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, cam["intrinsics"], resolution, rng=rng, info=view_idx
            )
            img_mask, ray_mask = self.get_img_and_ray_masks(self.is_metric, v, rng, p=[0.75, 0.2, 0.05])
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=cam["pose"].astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.name,
                label=scene + "_" + basename,
                instance=f"{idx}_{view_idx}",
                is_metric=self.is_metric,
                is_video=ordered_video,
                quantile=np.array(0.98, dtype=np.float32),
                img_mask=img_mask,
                ray_mask=ray_mask,
                camera_only=False,
                depth_only=False,
                single_view=False,
                reset=False,
            ))
        assert len(views) == num_views
        return views
