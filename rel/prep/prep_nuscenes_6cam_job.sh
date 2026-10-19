#!/usr/bin/env bash
# nuScenes 6-camera conversion: $1 = output dir, remaining args go to the converter
set -euo pipefail
OUT="$1"; shift
source ~/CUT3R/rel/cluster/setup_env.sh
dataset pull nuscenes_full >/dev/null
echo "git $(git -C ~/CUT3R rev-parse --short HEAD) on $(hostname), $(nproc) cpus"
"${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python" ~/CUT3R/rel/prep/prep_nuscenes_video.py \
    --out "$OUT" --channels CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT "$@"
