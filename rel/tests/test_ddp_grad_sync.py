#!/usr/bin/env python
"""Does the TBPTT training step average gradients across GPUs?

loss_of_one_batch_tbptt runs the model through accelerator.unwrap_model(model), so
DDP's forward (which arms gradient all-reduce) is bypassed. Run with
`accelerate launch --num_processes 2`: each rank takes one TBPTT step on different
synthetic data from identical weights, then the ranks compare a checksum of every
trainable parameter. Synced: identical. Not synced: they differ.
`--single <dir>` (one process) takes the same step on both ranks' data as one batch
of 2; `--compare <dir>` checks that this update matches the 2-GPU one, which is
what correct gradient averaging must give (up to bf16 rounding)."""
import sys

import torch
import torch.distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs

sys.path.insert(0, "src")
from croco.utils.misc import NativeScalerWithGradNormCount  # noqa: E402
from dust3r.inference import loss_of_one_batch_tbptt  # noqa: E402
from dust3r.losses import ConfLoss, L21, Regr3DPoseRel, RGBLoss, MSE  # noqa: E402,F401
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.geometry import geotrf  # noqa: E402

V, B, H, W = 8, 1, 224, 224


def views(seed, device):
    g = torch.Generator().manual_seed(seed)
    out = []
    for v in range(V):
        pose = torch.eye(4).repeat(B, 1, 1)
        pose[:, 2, 3] = 0.3 * v
        z = 1 + 3 * torch.rand(B, H, W, generator=g)
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        X = torch.stack([(xx - W / 2) / 200 * z, (yy - H / 2) / 200 * z, z], -1)
        out.append({k: x.to(device) for k, x in dict(
            img=torch.rand(B, 3, H, W, generator=g) * 2 - 1, true_shape=torch.tensor([[H, W]] * B),
            camera_pose=pose, pts3d=geotrf(pose, X), valid_mask=torch.ones(B, H, W, dtype=torch.bool),
            sky_mask=torch.zeros(B, H, W, dtype=torch.bool), img_mask=torch.ones(B, dtype=torch.bool),
            ray_mask=torch.zeros(B, dtype=torch.bool), ray_map=torch.zeros(B, H, W, 6),
            reset=torch.zeros(B, dtype=torch.bool), update=torch.ones(B, dtype=torch.bool),
            is_metric=torch.ones(B, dtype=torch.bool), is_metric_scale=torch.ones(B, dtype=torch.bool),
            camera_only=torch.zeros(B, dtype=torch.bool), depth_only=torch.zeros(B, dtype=torch.bool),
            single_view=torch.zeros(B, dtype=torch.bool), quantile=torch.full((B,), 0.98),
            camera_intrinsics=torch.tensor([[200.0, 0, W / 2], [0, 200.0, H / 2], [0, 0, 1]]).repeat(B, 1, 1),
        ).items()})
    return out


def update_sample(model, before, n=1_000_000):
    """The weight update on a fixed random sample of trainable parameters."""
    flat = torch.cat([p.detach().float().flatten() for p in model.parameters() if p.requires_grad])
    idx = torch.randperm(len(flat), generator=torch.Generator().manual_seed(0))[:n].to(flat.device)
    return (flat[idx] - before[idx]).cpu()


def main():
    single = "--single" in sys.argv
    out = sys.argv[-1] if (single or "--save" in sys.argv) else None
    acc = Accelerator(mixed_precision="no" if "--fp32" in sys.argv else "bf16",  # bf16: as train.py
                      kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    torch.manual_seed(0)
    model = ARCroco3DStereo.from_pretrained("/group/compact-3dmem/checkpoints/CUT3R/cut3r_512_dpt_4_64.pth")
    model.set_freeze("encoder")
    params = [p for p in model.parameters() if p.requires_grad]
    # SGD makes the update the gradient itself; AdamW's first step is ~lr*sign(grad),
    # which turns rounding noise in near-zero gradients into full-size differences
    opt = torch.optim.SGD(params, lr=1e-2) if "--sgd" in sys.argv else torch.optim.AdamW(params, lr=1e-4)
    before = torch.cat([p.detach().float().flatten() for p in model.parameters() if p.requires_grad]).cuda()
    model, opt = acc.prepare(model, opt)
    crit = (ConfLoss(Regr3DPoseRel(L21, norm_mode="?avg_dis", window=4), alpha=0.2)).to(acc.device)
    scaler = NativeScalerWithGradNormCount(accelerator=acc)
    batch = views(seed=100 + acc.process_index, device=acc.device)  # different data per rank
    if single:  # both ranks' data as one batch of 2
        other = views(seed=101, device=acc.device)
        batch = [{k: torch.cat([a[k], b[k]]) for k in a} for a, b in zip(batch, other)]
    loss_of_one_batch_tbptt(batch, model, crit, chunk_size=4, loss_scaler=scaler, optimizer=opt,
                            accelerator=acc, symmetrize_batch=False, use_amp=False)
    if out is not None and acc.is_main_process:
        torch.save(update_sample(acc.unwrap_model(model), before), f"{out}/update_{'single' if single else 'ddp'}.pt")
    if single:
        return
    flat = torch.cat([p.detach().float().flatten() for p in acc.unwrap_model(model).parameters() if p.requires_grad])
    mine = torch.stack([flat.sum(), (flat * torch.arange(len(flat), device=flat.device) % 9973).sum()])
    allv = [torch.zeros_like(mine) for _ in range(acc.num_processes)]
    dist.all_gather(allv, mine)
    if acc.is_main_process:
        same = all(torch.equal(allv[0], x) for x in allv[1:])
        print("rank checksums:", [x.tolist() for x in allv])
        print("GRADIENTS SYNCED: weights identical across ranks" if same else
              "GRADIENTS NOT SYNCED: each rank trained its own copy")


def compare(d):
    a, b = torch.load(f"{d}/update_ddp.pt"), torch.load(f"{d}/update_single.pt")
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    rel = float((a - b).norm() / b.norm())
    print(f"2-GPU update vs 1-GPU update on the combined batch: cosine {cos:.5f}, relative L2 diff {rel:.4f}")


if __name__ == "__main__":
    compare(sys.argv[2]) if sys.argv[1:2] == ["--compare"] else main()
