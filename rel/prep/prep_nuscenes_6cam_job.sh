#!/usr/bin/env bash
# nuScenes 6-camera conversion into the dataset registry. The output is ~3.5M small
# files, more than /group/compact-3dmem's file quota allows, so it is written to
# node-local /scratch and registered from there; only the log lands on /group.
# Usage: sbatch [...] prep_nuscenes_6cam_job.sh <dataset name> [converter args...]
set -euo pipefail
NAME="$1"; shift
OUT="/scratch/$USER/prep_$NAME"
source ~/CUT3R/rel/cluster/setup_env.sh
dataset pull nuscenes_full >/dev/null
echo "git $(git -C ~/CUT3R rev-parse --short HEAD) on $(hostname), $(nproc) cpus, scratch free $(df -h /scratch | tail -1 | awk '{print $4}')"
"${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python" ~/CUT3R/rel/prep/prep_nuscenes_video.py \
    --out "$OUT" --channels CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT "$@"
cp "$OUT/prep_log.json" "/group/compact-3dmem/datasets/_download_logs/${NAME}_prep_log.json"
dataset create "$NAME" "$OUT"
dataset list "$NAME"
rm -rf "$OUT"
