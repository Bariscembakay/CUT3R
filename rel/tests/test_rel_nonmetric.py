#!/usr/bin/env python
"""CUT3R's handling of non-metric and camera-only data in the relative loss.

1. Metric data with depth: bit-identical to the committed loss (test_rel_raymap 1).
2. Non-metric samples: the loss does not change when their prediction (point maps
   and translations) is rescaled -- the prediction is normalised by its own scale.
   Metric samples do change, so the check has teeth.
3. Camera-only samples (no depth): the loss is finite, has no point-map term from
   them, and is invariant to rescaling their prediction when non-metric.
Usage: python rel/tests/test_rel_nonmetric.py [baseline git rev, default HEAD]"""
import os.path as osp
import sys

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from test_rel_raymap import B, KW, V, make_clip, run, test_image_only_unchanged  # noqa: E402
from dust3r.losses import ConfLoss, L21, Regr3DPoseRel  # noqa: E402


def scaled(preds, s, rows):
    """Copy of the predictions with sample rows' point maps and translations times s."""
    out = []
    for p in preds:
        q = {k: v.detach().clone() for k, v in p.items()}
        for k in ("pts3d_in_self_view", "pts3d_in_other_view"):
            q[k][rows] *= s
        q["camera_pose"][rows, :3] *= s
        out.append({k: v.requires_grad_() if k in p and p[k].requires_grad else v for k, v in q.items()})
    return out


def losses(views, preds):
    return torch.stack([loss for loss, _, _ in run(ConfLoss(Regr3DPoseRel(L21, **KW), alpha=0.2), views, preds)])


def test_non_metric(camera_only=False):
    views, preds = make_clip(5)
    nm = torch.tensor([True, False, True])  # samples 0 and 2 non-metric
    for v in views:
        v["is_metric"] = ~nm
        if camera_only:
            v["camera_only"] = torch.tensor([True, False, False])
            v["valid_mask"] = v["valid_mask"] & ~v["camera_only"][:, None, None]
    base = losses(views, preds)
    assert torch.isfinite(base).all(), base
    inv_ = losses(views, scaled(preds, 3.0, nm))
    assert torch.allclose(base, inv_, rtol=1e-4, atol=1e-5), (base, inv_)
    moved = losses(views, scaled(preds, 3.0, ~nm))
    assert not torch.allclose(base, moved, rtol=1e-3), "metric sample should feel the rescale"
    tag = "camera-only + non-metric" if camera_only else "non-metric"
    print(f"{'3' if camera_only else '2'}. {tag}: loss invariant to rescaling those samples' predictions "
          f"(max rel. diff {float(((base - inv_).abs() / base.abs()).max()):.1e}); metric samples are not")


if __name__ == "__main__":
    test_image_only_unchanged(sys.argv[1] if len(sys.argv) > 1 else "HEAD")
    test_non_metric()
    test_non_metric(camera_only=True)
    print("all passed")
