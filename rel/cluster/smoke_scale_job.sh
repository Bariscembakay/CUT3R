#!/usr/bin/env bash
# CPU-only, in the training zone: update the clone, build the env, then load one
# batch from every dataset of a scale-up config (rel/tests/smoke_scale_datasets.py).
# Usage: sbatch --constraint=zone-msp3 [...] smoke_scale_job.sh [config_name]
set -euo pipefail
REPO="${REPO:-$HOME/CUT3R}"
git -C "$REPO" fetch -q origin cut3r-rel && git -C "$REPO" merge -q --ff-only origin/cut3r-rel
git -C "$REPO" log --oneline -1
source "$REPO/rel/cluster/setup_env.sh"
for ds in arkitscenes-cut3r-video-full nuscenes-cut3r-video scannet-v2-extracted tartanair-v1 tartanair-v2; do
    dataset pull "$ds" >/dev/null
done
"${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python" -u "$REPO/rel/tests/smoke_scale_datasets.py" "${1:-scale_rel}"
