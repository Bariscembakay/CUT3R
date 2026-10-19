#!/usr/bin/env python
"""Ray-map-aware CUT3R-rel: the relative loss chains through image views only and
ray maps are expressed in the last image view's camera.

1. Image views only: the new Regr3DPoseRel gives bit-identical loss and details to
   the committed one (git show <rev>:src/dust3r/losses.py), so training without
   ray maps is unchanged.
2. With ray-only views: reference indices, pose targets and composed-pose terms
   match a per-sample brute force.
3. raymap_frame="last_image": each view's ray origin/directions are its camera in
   the last image view's frame; "first" still reproduces upstream.
Usage: python rel/tests/test_rel_raymap.py [baseline git rev, default HEAD]"""
import importlib.util
import os.path as osp
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
sys.path.insert(0, osp.join(REPO, "src"))
from dust3r.losses import ConfLoss, L21, Regr3DPoseRel  # noqa: E402
from dust3r.utils.camera import camera_to_pose_encoding  # noqa: E402
from dust3r.utils.geometry import geotrf, inv  # noqa: E402

B, V, H, W, CHUNK = 3, 12, 6, 8, 4
KW = dict(norm_mode="?avg_dis", window=6, rot_loss="angle", rot_weight=10.0, composed_pts_ks=(2, 4))


def baseline_losses(rev):
    src = subprocess.run(["git", "-C", REPO, "show", f"{rev}:src/dust3r/losses.py"],
                         check=True, capture_output=True, text=True).stdout
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    f.write(src)
    f.close()
    spec = importlib.util.spec_from_file_location("losses_baseline", f.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rand_se3(g, rot_scale, t_scale):
    w = torch.randn(3, generator=g) * rot_scale
    th = w.norm()
    Kx = torch.tensor([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = torch.eye(3) + torch.sin(th) / th * Kx + (1 - torch.cos(th)) / th ** 2 * Kx @ Kx
    T = torch.eye(4)
    T[:3, :3], T[:3, 3] = R, torch.randn(3, generator=g) * t_scale
    return T


def make_clip(seed, img_mask=None):
    g = torch.Generator().manual_seed(seed)
    poses = torch.zeros(V, B, 4, 4)
    for b in range(B):
        P = rand_se3(g, 0.3, 1.0)
        for v in range(V):
            poses[v, b] = P
            P = P @ rand_se3(g, 0.05, 0.5)
    views, preds = [], []
    for v in range(V):
        z = 1 + 4 * torch.rand(B, H, W, generator=g)
        uv = torch.randn(B, H, W, 2, generator=g)
        X_cam = torch.cat([uv * z[..., None], z[..., None]], -1)
        views.append(dict(
            camera_pose=poses[v], pts3d=geotrf(poses[v], X_cam),
            valid_mask=torch.rand(B, H, W, generator=g) > 0.1, sky_mask=torch.zeros(B, H, W, dtype=torch.bool),
            img_mask=torch.ones(B, dtype=torch.bool) if img_mask is None else img_mask[v],
            ray_mask=torch.zeros(B, dtype=torch.bool) if img_mask is None else ~img_mask[v],
            reset=torch.zeros(B, dtype=torch.bool), is_metric=torch.ones(B, dtype=torch.bool),
            is_metric_scale=torch.ones(B, dtype=torch.bool), camera_only=torch.zeros(B, dtype=torch.bool),
            depth_only=torch.zeros(B, dtype=torch.bool), single_view=torch.zeros(B, dtype=torch.bool),
            quantile=torch.full((B,), 0.98), img=torch.rand(B, 3, H, W, generator=g)))
        noisy = torch.stack([rand_se3(g, 0.05, 0.5) for _ in range(B)])
        preds.append(dict(
            pts3d_in_self_view=(X_cam + 0.1 * torch.randn(X_cam.shape, generator=g)).requires_grad_(),
            pts3d_in_other_view=torch.randn(B, H, W, 3, generator=g).requires_grad_(),
            conf_self=1 + torch.rand(B, H, W, generator=g), conf=1 + torch.rand(B, H, W, generator=g),
            camera_pose=camera_to_pose_encoding(noisy).requires_grad_()))
    return views, preds


def run(crit, views, preds):
    out = []
    for cs in range(0, V, CHUNK):
        prev = [{k: x.detach() for k, x in p.items()} for p in preds[:cs]]
        loss, details = crit(views[cs:cs + CHUNK], preds[cs:cs + CHUNK], views_all=views, chunk_start=cs,
                             preds_all=prev)
        grads = torch.autograd.grad(loss, [preds[cs]["camera_pose"], preds[cs]["pts3d_in_self_view"]])
        out.append((loss.detach(), {k: v for k, v in details.items() if isinstance(v, float)}, grads))
    return out


def test_image_only_unchanged(rev):
    old = baseline_losses(rev)
    views, preds = make_clip(0)
    a = run(old.ConfLoss(old.Regr3DPoseRel(old.L21, **KW), alpha=0.2), views, preds)
    b = run(ConfLoss(Regr3DPoseRel(L21, **KW), alpha=0.2), views, preds)
    rel_err = 0.0
    for (la, da, ga), (lb, db, gb) in zip(a, b):
        assert torch.equal(la, lb), (la, lb)
        assert da == db, {k: (da[k], db.get(k)) for k in da if da[k] != db.get(k)}
        for x, y in zip(ga, gb):
            rel_err = max(rel_err, float((x - y).abs().max() / x.abs().max()))
    # gathering steps from a stacked tensor changes only the order in which the
    # backward pass sums a step's uses: fp32 rounding, far below bf16 training noise
    assert rel_err < 1e-6, rel_err
    print(f"1. image views only: loss and {len(a[0][1])} detail terms bit-identical to {rev}; "
          f"gradients equal up to fp32 summation order (max rel. diff {rel_err:.1e})")


def brute_ref(img, g, b):
    for p in range(g - 1, -1, -1):
        if img[p, b]:
            return p
    return g


def test_ray_only_views():
    gen = torch.Generator().manual_seed(1)
    img = torch.rand(V, B, generator=gen) > 0.35
    img[0] = True
    views, preds = make_clip(2, img_mask=img)
    crit = Regr3DPoseRel(L21, **KW)
    for g in range(V):
        ref = crit._ref_index(views, g)
        assert ref.tolist() == [brute_ref(img, g, b) for b in range(B)], (g, ref, img[:, :])
        prev = crit._prev_pose(views, g)
        for b in range(B):
            assert torch.equal(prev[b], views[brute_ref(img, g, b)]["camera_pose"][b])

    # composed terms against a per-sample brute force
    from dust3r.utils.camera import pose_encoding_to_camera
    nf = torch.ones(B, 1)
    pose_masks = torch.ones(B, dtype=torch.bool)
    for cs in range(0, V, CHUNK):
        gts, prs = views[cs:cs + CHUNK], preds[cs:cs + CHUNK]
        prev = [{k: x.detach() for k, x in p.items()} for p in preds[:cs]]
        valids = [v["valid_mask"] for v in gts]
        got, got_pts = crit._composed_pose_loss(views, gts, prs, prev, cs, nf, pose_masks, valids)
        terms, pts_terms = [], []
        step = lambda m: pose_encoding_to_camera((prs[m - cs] if m >= cs else prev[m])["camera_pose"])  # noqa: E731
        for j in range(len(gts)):
            g = cs + j
            chains = {}
            for b in range(B):
                if not img[g, b]:
                    continue
                anc = [g]
                while len(anc) <= KW["window"] and brute_ref(img, anc[-1], b) != anc[-1]:
                    anc.append(brute_ref(img, anc[-1], b))
                chains[b] = anc
            for k in range(2, KW["window"] + 1):
                errs, perr = [], []
                for b, anc in chains.items():
                    if len(anc) <= k:
                        continue
                    C = step(g)[b]
                    for m in anc[1:k]:
                        C = step(m)[b] @ C
                    T = inv(views[anc[k]]["camera_pose"][b]) @ views[g]["camera_pose"][b]
                    errs.append(torch.norm(C[:3, 3] - T[:3, 3]) + crit._rot_err(C[None, :3, :3], T[None, :3, :3])[0])
                    if k in crit.composed_pts_ks:
                        pr = geotrf(C[None], prs[j]["pts3d_in_self_view"][b:b + 1])[0]
                        gt = geotrf(inv(views[anc[k]]["camera_pose"][b])[None], gts[j]["pts3d"][b:b + 1])[0]
                        perr.append(torch.norm(pr - gt, dim=-1)[valids[j][b]])
                if errs:
                    terms.append(torch.stack(errs).mean())
                if perr:
                    pts_terms.append(torch.cat(perr).mean())
        want = torch.stack(terms).mean() if terms else torch.zeros(())
        want_pts = torch.stack(pts_terms).mean() if pts_terms else torch.zeros(())
        assert torch.allclose(got, want, atol=1e-5), (cs, got, want)
        assert torch.allclose(got_pts, want_pts, atol=1e-5), (cs, got_pts, want_pts)
    # no pose loss on ray-only views
    crit.get_all_pts3d(views[:CHUNK], preds[:CHUNK], views_all=views, chunk_start=0, preds_all=[])
    assert torch.equal(crit._view_pose_masks, torch.stack([v["img_mask"] for v in views[:CHUNK]], 1))
    print(f"2. with ray-only views ({int((~img).sum())} of {V * B}): references, pose targets, "
          "composed terms and per-view pose masks match the brute force")


def test_raymap_frame():
    import PIL.Image
    from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset, get_ray_map

    g = torch.Generator().manual_seed(3)
    poses = [np.eye(4, dtype=np.float32)]
    for _ in range(V - 1):
        poses.append((poses[-1] @ rand_se3(g, 0.1, 0.5).numpy()).astype(np.float32))
    img_mask = [True] + [bool(x) for x in (torch.rand(V - 1, generator=g) > 0.4)]
    K = np.array([[20.0, 0, W / 2], [0, 20.0, H / 2], [0, 0, 1]], np.float32)

    class Fake(BaseMultiViewDataset):
        def __len__(self):
            return 1

        def _get_views(self, idx, resolution, rng, num_views):
            return [dict(img=PIL.Image.new("RGB", (W, H)), depthmap=np.ones((H, W), np.float32),
                         camera_pose=poses[v], camera_intrinsics=K, dataset="fake", label="x", instance=str(v),
                         is_metric=True, is_video=True, quantile=np.array(0.98, np.float32),
                         img_mask=img_mask[v], ray_mask=not img_mask[v], camera_only=False, depth_only=False,
                         single_view=False, reset=False) for v in range(num_views)]

    kw = dict(num_views=V, resolution=(W, H), seed=7)
    rel = Fake(raymap_frame="last_image", **kw)[(0, 0, V)]
    first = Fake(**kw)[(0, 0, V)]
    last = 0
    for v in range(V):
        ref = poses[v] if v == 0 else poses[last]
        assert np.allclose(rel[v]["ray_map"], get_ray_map(ref, poses[v], K, H, W), atol=1e-6)
        assert np.allclose(first[v]["ray_map"], get_ray_map(poses[0], poses[v], K, H, W), atol=1e-6)
        t = np.linalg.inv(ref) @ poses[v]
        assert np.allclose(rel[v]["ray_map"][0, 0, :3], t[:3, 3], atol=1e-5)
        if img_mask[v]:
            last = v
    print(f"3. ray maps: 'last_image' is relative to the last image view ({sum(not m for m in img_mask)} "
          "ray-only views in the clip), 'first' reproduces upstream")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    test_image_only_unchanged(sys.argv[1] if len(sys.argv) > 1 else "HEAD")
    test_ray_only_views()
    test_raymap_frame()
    print("all passed")
