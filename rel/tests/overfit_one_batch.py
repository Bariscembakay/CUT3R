#!/usr/bin/env python
"""Phase-0: one fixed batch through the real training path (config -> model with
released weights -> loss_of_one_batch_tbptt -> optimizer), for one arm.
Reports the loss trajectory (must fall), seconds per iteration and peak memory.
Usage: python overfit_one_batch.py <config_name> [iters] [batch_size] [lr]"""
import os.path as osp
import sys
import time

import torch
from math import inf  # noqa: F401  (the model config string uses inf)

SRC = osp.join(osp.dirname(osp.abspath(__file__)), "../../src")
sys.path.insert(0, SRC)
from accelerate import Accelerator  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

import croco.utils.misc as misc  # noqa: E402
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa: E402
from dust3r.datasets import get_data_loader  # noqa: E402
from dust3r.inference import loss_of_one_batch_tbptt  # noqa: E402
from dust3r.losses import *  # noqa: E402,F401,F403
from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig, strip_module  # noqa: E402,F401


def main():
    name = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
    with initialize_config_dir(config_dir=osp.abspath(osp.join(SRC, "../config")), version_base=None):
        cfg = compose(config_name=name)
    torch.manual_seed(0)
    acc = Accelerator(mixed_precision="bf16")
    model = eval(cfg.model)
    ckpt = torch.load(cfg.pretrained, map_location="cpu", weights_only=False)
    print(model.load_state_dict(strip_module(ckpt["model"]), strict=False))
    del ckpt
    model.to(acc.device)
    model.gradient_checkpointing_enable()
    model.fixed_input_length = False
    crit = eval(cfg.train_criterion).to(acc.device)
    opt = torch.optim.AdamW(misc.get_parameter_groups(model, cfg.weight_decay), lr=lr, betas=(0.9, 0.95))
    scaler = NativeScaler(accelerator=acc)
    opt, model = acc.prepare(opt, model)

    loader = get_data_loader(cfg.dataset1, batch_size=bs, num_workers=4, accelerator=acc,
                             shuffle=True, fixed_length=True)
    loader.dataset.set_epoch(0)
    if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(0)
    batch = next(iter(loader))
    batch = [{k: (v.to(acc.device) if torch.is_tensor(v) else v) for k, v in view.items()} for view in batch]
    print(f"[{name}] batch {bs} x {len(batch)} views, img {tuple(batch[0]['img'].shape[-2:])}, "
          f"trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f} M")
    torch.cuda.reset_peak_memory_stats()
    for it in range(iters):
        t0 = time.time()
        res = loss_of_one_batch_tbptt(batch, model, crit, chunk_size=4, loss_scaler=scaler,
                                      optimizer=opt, accelerator=acc)
        torch.cuda.synchronize()
        loss, det = res["loss"]
        pose = [v for k, v in det.items() if k.startswith("pose_loss")]
        comp = [v for k, v in det.items() if k.startswith("composed_pose_loss")]
        mean = lambda xs: sum(float(x) for x in xs) / max(len(xs), 1)  # noqa: E731
        print(f"[{name}] it {it:3d}  loss {loss:.4f}  pose {mean(pose):.4f}"
              + (f"  composed {mean(comp):.4f}" if comp else "")
              + f"  {time.time() - t0:.1f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
