#!/usr/bin/env python
"""Scenes, frames and clip starts of every training and validation set of a config,
built exactly as train.py builds them. Usage: count_config_datasets.py <config name>"""
import os.path as osp
import sys

from hydra import compose, initialize_config_dir

REPO = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
sys.path.insert(0, osp.join(REPO, "src"))
from dust3r.datasets import *  # noqa: F401,F403,E402

with initialize_config_dir(config_dir=osp.join(REPO, "config"), version_base=None):
    cfg = compose(config_name=sys.argv[1])
print("| set | loader | scenes | frames | clip starts | clips per epoch |\n|---|---|---|---|---|---|")
for role, spec in (("train", str(cfg.train_dataset)), ("val", str(cfg.test_dataset))):
    for part in spec.split(" + "):
        n, expr = part.split("@", 1)
        ds = eval(expr.strip())
        label = getattr(ds, "name", type(ds).__name__)
        print(f"| {role} | {label} | {len(ds.scenes)} | {len(ds.images)} | {len(ds)} | {n.strip()} |", flush=True)
