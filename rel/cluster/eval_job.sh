#!/usr/bin/env bash
# Frame-anchor suite on a checkpoint: stream scenes, ship per-frame records.
# Usage: sbatch [...] eval_job.sh <ckpt> <relative 0|1> <arm_name> <benchmark> <scene...>
#   ORDER=forward|palindrome|chunked (default forward)  EXTRA_ARGS="--chunk 50 ..."
set -euo pipefail
CKPT="$1"; REL="$2"; ARM="$3"; BENCH="$4"; shift 4
REPO="${REPO:-$HOME/CUT3R}"
CAMPAIGN=/group/compact-3dmem/campaigns/cut3r_rel/pilot_eval
source "$REPO/rel/cluster/setup_env.sh"
PY="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"
case "$BENCH" in
    nrgbd)          dataset pull NRGBD >/dev/null; ROOT=/data/NRGBD ;;
    kitti)          dataset pull kitti_odometry >/dev/null; ROOT=/data/kitti_odometry ;;
    kitti_depthval) ROOT=/group/compact-3dmem/datasets/kitti_depth ;;
    *) echo "unknown benchmark $BENCH" >&2; exit 2 ;;
esac
WORK="/scratch/$USER/${SLURM_JOB_ID:-local}/cut3r_rel_eval"
echo "[eval_job] $(hostname) ckpt=$CKPT relative=$REL arm=$ARM bench=$BENCH git=$(git -C "$REPO" rev-parse --short HEAD)"
for SCENE in "$@"; do
    OUT="$WORK/$ARM/$BENCH/$SCENE"
    # shellcheck disable=SC2086
    "$PY" "$REPO/rel/eval/cut3r_anchor_dump.py" --benchmark "$BENCH" --scene "$SCENE" --root "$ROOT" \
        --ckpt "$CKPT" --relative "$REL" --order "${ORDER:-forward}" --dump-stride 0 --out "$OUT" ${EXTRA_ARGS:-}
    mkdir -p "$CAMPAIGN/$ARM/$BENCH/$SCENE"
    rsync -a "$OUT/" "$CAMPAIGN/$ARM/$BENCH/$SCENE/"
done
