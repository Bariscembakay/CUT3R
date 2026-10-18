import os
import os.path as osp

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dust3r.datasets.tartanair import TartanAir_Multi
from dust3r.utils.image import imread_cv2

TARTANAIR_K = np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]], np.float32)


def ned_poses_to_opencv_c2w(pose_left):
    """pose_left.txt rows (x y z qx qy qz qw, NED camera-to-world) -> OpenCV c2w,
    exactly as datasets_preprocess/preprocess_tartanair.py does it."""
    c2w = np.tile(np.eye(4), (len(pose_left), 1, 1))
    c2w[:, :3, :3] = Rotation.from_quat(pose_left[:, 3:7]).as_matrix()
    c2w[:, :3, 3] = pose_left[:, :3]
    w2c = np.linalg.inv(c2w)[:, [1, 2, 0, 3]]
    return np.linalg.inv(w2c).astype(np.float32)


class TartanAirRaw_Multi(TartanAir_Multi):
    """TartanAir_Multi on the raw download (<env>/<Easy|Hard>/<Pxxx>/{image_left,
    depth_left, pose_left.txt}) instead of preprocess_tartanair.py's per-frame
    copies: same views, same pose convention, same sky/depth handling."""

    MODES = ("Easy", "Hard")
    IMG = ("image_left", "_left.png")
    DEPTH = ("depth_left", "_left_depth.npy")
    POSE_FILE = "pose_left.txt"
    K = TARTANAIR_K

    def read_depth(self, path):
        return np.load(path)

    def _load_data(self):
        offset, j = 0, 0
        scenes, sceneids, images, scene_img_list, start_img_ids = [], [], [], [], []
        self.scene_poses = []
        cut_off = self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
        for env in sorted(d for d in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, d))):
            for mode in self.MODES:
                mode_dir = osp.join(self.ROOT, env, mode)
                if not osp.isdir(mode_dir):
                    continue
                for seq in sorted(os.listdir(mode_dir)):
                    seq_dir = osp.join(mode_dir, seq)
                    if not osp.exists(osp.join(seq_dir, self.POSE_FILE)):
                        continue
                    poses = ned_poses_to_opencv_c2w(np.loadtxt(osp.join(seq_dir, self.POSE_FILE)))
                    basenames = sorted(f[:6] for f in os.listdir(osp.join(seq_dir, self.IMG[0]))
                                       if f.endswith(self.IMG[1]))
                    assert len(basenames) == len(poses), seq_dir
                    if len(basenames) < cut_off:
                        continue
                    img_ids = list(np.arange(len(basenames)) + offset)
                    start_img_ids.extend(img_ids[: len(basenames) - cut_off + 1])
                    scenes.append(seq_dir)
                    self.scene_poses.append(poses)
                    scene_img_list.append(img_ids)
                    sceneids.extend([j] * len(basenames))
                    images.extend(basenames)
                    offset += len(basenames)
                    j += 1
        self.scenes, self.sceneids, self.images = scenes, sceneids, images
        self.scene_img_list, self.start_img_ids = scene_img_list, start_img_ids

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views, start_id, all_image_ids, rng, max_interval=self.max_interval,
            video_prob=0.8, fix_interval_prob=0.8, block_shuffle=16,
        )
        views = []
        for v, view_idx in enumerate(np.array(all_image_ids)[pos]):
            scene_id = self.sceneids[view_idx]
            scene_dir = self.scenes[scene_id]
            basename = self.images[view_idx]
            image = imread_cv2(osp.join(scene_dir, self.IMG[0], basename + self.IMG[1]))
            depthmap = self.read_depth(osp.join(scene_dir, self.DEPTH[0], basename + self.DEPTH[1]))
            camera_pose = self.scene_poses[scene_id][int(basename)]

            sky_mask = depthmap >= 1000
            depthmap[sky_mask] = -1.0
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
            threshold = np.percentile(depthmap[depthmap > 0], 98) if (depthmap > 0).any() else 0
            depthmap[depthmap > threshold] = 0.0

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, self.K.copy(), resolution, rng, info=(scene_dir, basename)
            )
            img_mask, ray_mask = self.get_img_and_ray_masks(self.is_metric, v, rng, p=[0.75, 0.2, 0.05])
            views.append(dict(
                img=image, depthmap=depthmap, camera_pose=camera_pose,
                camera_intrinsics=intrinsics, dataset="TartanAir", label=scene_dir,
                is_metric=self.is_metric, instance=f"{scene_dir}_{basename}",
                is_video=ordered_video, quantile=np.array(1.0, dtype=np.float32),
                img_mask=img_mask, ray_mask=ray_mask, camera_only=False,
                depth_only=False, single_view=False, reset=False,
            ))
        assert len(views) == num_views
        return views


class TartanAirV2Raw_Multi(TartanAirRaw_Multi):
    """TartanAir v2 (<env>/<Data_easy|Data_hard>/<Pxxx>/{image_lcam_front,
    depth_lcam_front, pose_lcam_front.txt}): 640x640 pinhole, 90 deg FoV; depth
    is a float32 stored in the four 8-bit channels of an RGBA png."""

    MODES = ("Data_easy", "Data_hard")
    IMG = ("image_lcam_front", "_lcam_front.png")
    DEPTH = ("depth_lcam_front", "_lcam_front_depth.png")
    POSE_FILE = "pose_lcam_front.txt"
    K = np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]], np.float32)

    def read_depth(self, path):
        # cv2 directly: imread_cv2 applies BGR->RGB to any 3-D array, which would
        # reorder (and drop) the bytes of the packed float
        rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise IOError(f"Could not load depth {path}")
        return rgba.view("<f4").squeeze(-1).copy()
