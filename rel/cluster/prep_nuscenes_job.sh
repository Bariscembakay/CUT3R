#!/usr/bin/env bash
# CPU-only: nuScenes trainval front camera + LiDAR depth -> video format (rel/prep/prep_nuscenes_video.py).
# Run in zone-sof1, where nuscenes_full is local. Usage: sbatch [...] prep_nuscenes_job.sh <out_dir>
set -euo pipefail
OUT="$1"
REPO="${REPO:-$HOME/CUT3R}"   # sbatch runs a copy of this script, so BASH_SOURCE cannot locate the repo
source "$REPO/rel/cluster/setup_env.sh"
PY="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"
dataset pull nuscenes_full >/dev/null
"$PY" -u "$REPO/rel/prep/prep_nuscenes_video.py" --out "$OUT" --workers "${SLURM_CPUS_PER_TASK:-16}"
