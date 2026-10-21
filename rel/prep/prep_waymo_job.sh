#!/usr/bin/env bash
# Waymo v1.4.3 .tfrecords (downloaded by download_waymo_job.sh into this node's
# /scratch) -> per-camera video format on /scratch. The `waymo` env (python 3.10,
# tensorflow 2.12, waymo-open-dataset) is separate from `cut3r` (torch).
# Usage: sbatch --nodelist=<download node> [...] prep_waymo_job.sh <out dir> [converter args...]
set -euo pipefail
OUT="$1"; shift
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}"
export MAMBA_ROOT_PREFIX
WPY="$MAMBA_ROOT_PREFIX/envs/waymo/bin/python"
(
    exec 200>"$MAMBA_ROOT_PREFIX/.setup_waymo_env.lock"
    flock 200
    if [ ! -x "$WPY" ]; then
        micromamba create -q -y -n waymo -c conda-forge python=3.10
        "$WPY" -m pip install -q waymo-open-dataset-tf-2-12-0==1.6.4 opencv-python-headless
    fi
    # waymo's frame_utils parses bytearrays, which protobuf 4's upb backend rejects
    "$WPY" -c "import google.protobuf as p; assert p.__version__ == '3.20.3'" 2>/dev/null \
        || "$WPY" -m pip install -q protobuf==3.20.3
)
echo "git $(git -C ~/CUT3R rev-parse --short HEAD) on $(hostname), $(nproc) cpus, scratch free $(df -h /scratch | tail -1 | awk '{print $4}')"
"$WPY" ~/CUT3R/rel/prep/prep_waymo_video.py --out "$OUT" "$@"
