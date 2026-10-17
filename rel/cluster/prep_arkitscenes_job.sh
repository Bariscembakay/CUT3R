#!/usr/bin/env bash
# CPU-only: ARKitScenes raw -> per-frame video format (rel/prep/prep_arkitscenes_video.py).
# Run in zone-sof1: /data is read live from the zone's CephFS, cross-site from gcp-eu1.
# Usage: sbatch [...] prep_arkitscenes_job.sh <out_dir> <num_scenes>
set -euo pipefail
OUT="$1"; N="${2:-300}"
# sbatch runs a copy of this script from /var/lib/slurm, so BASH_SOURCE cannot locate the repo.
REPO="${REPO:-$HOME/CUT3R}"
source "$REPO/rel/cluster/setup_env.sh"
PY="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"
dataset pull ARKitScenes >/dev/null
"$PY" "$REPO/rel/prep/prep_arkitscenes_video.py" --out "$OUT" --num-scenes "$N" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"
