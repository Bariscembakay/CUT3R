#!/usr/bin/env python
"""Phase-0 check of Regr3DPoseRel on a real batch, no model involved.

Predictions are synthesised from GT, so a correct criterion must return ~0 for
the convention it implements and clearly > 0 for the other one:
  1. relative "perfect" predictions -> Regr3DPoseRel ~0, whole clip and a later
     TBPTT chunk whose 4-step compositions reach into the previous chunk;
  2. frame-0 "perfect" predictions  -> Regr3DPose ~0, Regr3DPoseRel >> 0;
  3. view 1 has identical targets in both arms (its previous view is view 0).
Usage: python test_rel_criterion.py <video_rgbd_root>
"""
import os.path as osp
import sys

import torch

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../../src"))
from accelerate import Accelerator  # noqa: E402
from dust3r.datasets import get_data_loader  # noqa: E402
from dust3r.losses import L21, Regr3DPose, Regr3DPoseRel  # noqa: E402
from dust3r.utils.camera import camera_to_pose_encoding  # noqa: E402
from dust3r.utils.geometry import geotrf, inv  # noqa: E402

TOL = 1e-4


def fake_preds(batch, relative):
    preds = []
    for i, v in enumerate(batch):
        c = v["camera_pose"]
        ref = c if i == 0 else (batch[i - 1]["camera_pose"] if relative else batch[0]["camera_pose"])
        conf = torch.full(v["pts3d"].shape[:-1], 2.0)
        preds.append(dict(
            pts3d_in_self_view=geotrf(inv(c), v["pts3d"]),
            pts3d_in_other_view=geotrf(inv(ref), v["pts3d"]),
            conf_self=conf, conf=conf,
            camera_pose=camera_to_pose_encoding(inv(ref) @ c),
        ))
    return preds


def run(crit, gts, preds, **kw):
    loss, details = crit(gts, preds, **kw)
    return float(loss), float(details["pose_loss"]), details.get("composed_pose_loss")


def main():
    root = sys.argv[1]
    ds = (f"VideoRGBD_Multi(split='train', ROOT='{root}', name='test', max_interval=4, "
          f"video_only=True, no_raymap=True, resolution=[(384, 512)], num_views=8, seed=0)")
    loader = get_data_loader(ds, batch_size=2, num_workers=0, accelerator=Accelerator(cpu=True),
                             shuffle=False, fixed_length=True)
    batch = next(iter(loader))
    rel = Regr3DPoseRel(L21, norm_mode="?avg_dis", sky_loss_value=0, window=4)
    a = Regr3DPose(L21, norm_mode="?avg_dis", sky_loss_value=0)
    P_rel, P_abs = fake_preds(batch, True), fake_preds(batch, False)
    det = [{k: v.detach() for k, v in p.items()} for p in P_rel]
    ok = True

    def check(name, val, want_zero):
        nonlocal ok
        good = (abs(val) < TOL) if want_zero else (val > 100 * TOL)
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name}: {val:.3e}")

    tot, pose, comp = run(rel, batch, P_rel)
    check("rel criterion, rel preds, whole clip: total", tot, True)
    check("  pose (1-step + composed)", pose, True)
    check("  composed only", comp, True)
    tot, pose, comp = run(rel, batch[4:], P_rel[4:], views_all=batch, chunk_start=4, preds_all=det)
    check("rel criterion, rel preds, chunk [4:8] reaching into [0:4]: total", tot, True)
    check("  composed only", comp, True)
    tot, pose, _ = run(a, batch, P_abs, camera1=batch[0]["camera_pose"])
    check("frame-0 criterion, frame-0 preds: total", tot, True)
    tot, pose, _ = run(rel, batch, P_abs)
    check("rel criterion, frame-0 preds (must be > 0): pose", pose, False)
    same = torch.allclose(Regr3DPoseRel._prev_pose(batch, 1), batch[0]["camera_pose"])
    check("view 1: previous view is view 0 (1 = yes)", float(same), False)

    # v2: angle rotation loss, window 16, composed pointmaps; needs a 20-view clip
    ds20 = ds.replace("num_views=8", "num_views=20").replace("no_raymap=True", "no_raymap=True, static_prob=0.0")
    b20 = next(iter(get_data_loader(ds20, batch_size=2, num_workers=0, accelerator=Accelerator(cpu=True),
                                    shuffle=False, fixed_length=True)))
    v2 = Regr3DPoseRel(L21, norm_mode="?avg_dis", sky_loss_value=0, window=16, rot_loss="angle",
                       rot_weight=10.0, composed_pts_ks=(2, 4, 8, 16))
    R20, A20 = fake_preds(b20, True), fake_preds(b20, False)
    d20 = [{k: v.detach() for k, v in p.items()} for p in R20]
    loss, det = v2(b20, R20)
    check("v2, rel preds, 20-view clip: total", float(loss), True)
    check("  pose (angle, composed to 16)", float(det["pose_loss"]), True)
    check("  composed pointmaps", det["composed_pts_loss"], True)
    loss, det = v2(b20[16:], R20[16:], views_all=b20, chunk_start=16, preds_all=d20)
    check("v2, chunk [16:20] composing back into detached [0:16]: composed pts", det["composed_pts_loss"], True)
    loss, det = v2(b20, A20)
    check("v2, frame-0 preds (must be > 0): composed pts", det["composed_pts_loss"], False)

    # static-frame augmentation: with probability 1 one frame must repeat
    dss = ds.replace("no_raymap=True", "no_raymap=True, static_prob=1.0")
    bs = next(iter(get_data_loader(dss, batch_size=2, num_workers=0, accelerator=Accelerator(cpu=True),
                                   shuffle=False, fixed_length=True)))
    rep = min(sum(bool(torch.allclose(bs[i]["camera_pose"][b], bs[i - 1]["camera_pose"][b]))
                  for i in range(1, len(bs))) for b in range(bs[0]["camera_pose"].shape[0]))
    check("static_prob=1.0: every sample has an adjacent repeated view (>= 1)", float(rep), False)
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
