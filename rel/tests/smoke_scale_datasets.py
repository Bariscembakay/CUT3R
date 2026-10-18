#!/usr/bin/env python
"""Builds every training dataset of a scale-up config exactly as train.py would
(hydra compose + eval) and loads one batch from each. CPU only.
Usage: python smoke_scale_datasets.py [config_name]"""
import os.path as osp
import sys
import time

SRC = osp.join(osp.dirname(osp.abspath(__file__)), "../../src")
sys.path.insert(0, SRC)
from accelerate import Accelerator  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from dust3r.datasets import get_data_loader  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "scale_rel"
with initialize_config_dir(config_dir=osp.abspath(osp.join(SRC, "../config")), version_base=None):
    cfg = compose(config_name=name)
print("criterion:", cfg.train_criterion)
for k in ("dataset1", "dataset2", "dataset3", "dataset4", "dataset5"):
    t0 = time.time()
    loader = get_data_loader(cfg[k].replace(f"num_views={cfg.num_views}", "num_views=8"), batch_size=2, num_workers=0,
                             accelerator=Accelerator(cpu=True), shuffle=True, fixed_length=True)
    loader.dataset.set_epoch(0)
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(0)
    batch = next(iter(loader))
    v = batch[0]
    print(f"{k}: {type(loader.dataset).__name__} len={len(loader.dataset)} img {tuple(v['img'].shape)} "
          f"valid {float(v['valid_mask'].float().mean()):.2f} metric {bool(v['is_metric'].all())} "
          f"video {bool(v['is_video'].all())} raymap {any(bool(x['ray_mask'].any()) for x in batch)} ({time.time() - t0:.0f}s)", flush=True)
print("SMOKE OK")
