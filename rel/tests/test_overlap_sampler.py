#!/usr/bin/env python
"""Overlap sampler (base dataset `overlap=`) on a synthetic overlap table.

1. Off (overlap=None): positions identical to the upstream sampler, same rng draws.
2. On: every clip has num_views strictly increasing positions inside the scene;
   every consecutive pair overlaps at least overlap_tau (by the table); gaps cover
   the whole ladder; about overlap_const_frac of the clips keep one target, which
   on a stationary table shows as a single repeated gap.
3. A scene too short for the drawn gaps falls back to the upstream sampler."""
import json
import os.path as osp
import sys
import tempfile

import numpy as np

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", "src"))
from dust3r.datasets.base.base_multiview_dataset import OVERLAP_GAPS, BaseMultiViewDataset  # noqa: E402

N, VIEWS = 4000, 32
GAPS = np.array(OVERLAP_GAPS)


class Fake(BaseMultiViewDataset):
    def __init__(self, n=N, **kw):
        super().__init__(num_views=VIEWS, resolution=(64, 48), video_only=True, **kw)
        self.sceneids = [0] * n
        self.ids = list(range(n))

    def scene_key(self, sid):
        return "s0"


def table_files(n, decay=12.0):
    T = np.full((n, len(GAPS)), -1.0, np.float32)
    for k, g in enumerate(GAPS):
        T[: n - g, k] = np.exp(-g / decay)  # overlap falls with the gap, the same everywhere
    d = tempfile.mkdtemp()
    np.save(osp.join(d, "t.npy"), T.astype(np.float16))
    json.dump({"gaps": list(OVERLAP_GAPS), "scenes": {"s0": [0, n]}}, open(osp.join(d, "t_index.json"), "w"))
    return osp.join(d, "t"), T


def main():
    off = Fake()
    for seed in range(20):
        a = off.get_seq_from_start_id(VIEWS, 100, off.ids, np.random.default_rng(seed), max_interval=12)
        b = off._get_seq_from_start_id(VIEWS, 100, off.ids, np.random.default_rng(seed), max_interval=12)
        assert list(a[0]) == list(b[0]) and a[1] == b[1]
    print("1. overlap off: identical to the upstream sampler")

    path, T = table_files(N)
    on = Fake(overlap=path, overlap_tau=0.3, overlap_const_frac=0.3)
    rng = np.random.default_rng(0)
    gaps, const = [], 0
    for _ in range(2000):
        start = int(rng.integers(0, N - 64))
        pos, is_video = on.get_seq_from_start_id(VIEWS, start, on.ids, rng, max_interval=12)
        assert is_video and len(pos) == VIEWS and all(b > a for a, b in zip(pos, pos[1:])) and pos[-1] < N
        g = np.diff(pos)
        for p, step in zip(pos[:-1], g):
            k = int(np.searchsorted(GAPS, step))
            assert GAPS[k] == step and T[p, k] >= 0.3 - 1e-3, (p, step, T[p, k])
        gaps.append(g)
        const += len(set(g.tolist())) == 1
    gaps = np.concatenate(gaps)
    share = {int(x): round(float(np.mean(gaps == x)), 3) for x in np.unique(gaps)}
    assert set(share) <= set(OVERLAP_GAPS) and min(share) == 1 and max(share) >= 12, share
    assert 0.25 < const / 2000 < 0.42, const / 2000  # 0.3, plus varying clips that happen to repeat
    print(f"2. overlap on: all consecutive pairs overlap >= 0.3; gap shares {share}; "
          f"{const / 2000:.2f} of clips at one constant gap")

    short_path, _ = table_files(40)
    short = Fake(n=40, overlap=short_path)
    pos, _ = short.get_seq_from_start_id(VIEWS, 0, short.ids, np.random.default_rng(1), max_interval=12)
    assert len(pos) == VIEWS
    print("3. short scene: falls back to the upstream sampler")
    print("all passed")


if __name__ == "__main__":
    main()
