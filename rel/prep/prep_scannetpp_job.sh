#!/usr/bin/env bash
# ScanNet++ iPhone video conversion into the dataset registry, via node-local
# /scratch (the output is ~3.5M small files, over the /group file quota).
# Usage: sbatch [...] prep_scannetpp_job.sh <dataset name> [converter args...]
set -euo pipefail
NAME="$1"; shift
OUT="/scratch/$USER/prep_$NAME"
source ~/CUT3R/rel/cluster/setup_env.sh
PY="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"
"$PY" -m pip install -q lz4  # depth.bin is LZ4-compressed per frame
dataset pull ScanNetpp >/dev/null
echo "git $(git -C ~/CUT3R rev-parse --short HEAD) on $(hostname), $(nproc) cpus, scratch free $(df -h /scratch | tail -1 | awk '{print $4}')"
"$PY" ~/CUT3R/rel/prep/prep_scannetpp_video.py --out "$OUT" "$@"
cp "$OUT/prep_log.json" "/group/compact-3dmem/datasets/_download_logs/${NAME}_prep_log.json"
dataset create "$NAME" "$OUT"
dataset list "$NAME"
rm -rf "$OUT"
