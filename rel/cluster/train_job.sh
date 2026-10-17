#!/usr/bin/env bash
# One CUT3R-rel pilot arm on one GPU. Output (checkpoints, tensorboard, log.txt)
# is written on /scratch and shipped to the campaign on exit, success or not.
# Usage: sbatch [...] train_job.sh <config_name> [hydra overrides...]
set -euo pipefail
CFG="$1"; shift
REPO="${REPO:-$HOME/CUT3R}"   # sbatch runs a copy of this script, so BASH_SOURCE cannot locate the repo
CAMPAIGN=/group/compact-3dmem/campaigns/cut3r_rel
source "$REPO/rel/cluster/setup_env.sh"
PY="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"

# Node-local copy from stage_data_job.sh if complete, else the registry mounts.
export DATA_ROOT="/scratch/$USER/data"
if [ ! -f "$DATA_ROOT/.staged_ok" ]; then
    for ds in arkitscenes-cut3r-video tartanair-v1; do dataset pull "$ds" >/dev/null; done
    export DATA_ROOT=/data
fi
RUN_DIR="/scratch/$USER/cut3r_rel_runs/$CFG"
DEST="$CAMPAIGN/pilot/$CFG"
mkdir -p "$RUN_DIR" "$DEST"
ship() { rsync -a "$RUN_DIR/" "$DEST/" || true; echo "[train_job] shipped $RUN_DIR -> $DEST"; }
trap ship EXIT

echo "[train_job] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train_job] git $(git -C "$REPO" rev-parse HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l)"
echo "[train_job] DATA_ROOT=$DATA_ROOT config=$CFG overrides=$*"
cd "$REPO/src"
"$PY" train.py --config-name "$CFG" "save_dir=/scratch/$USER/cut3r_rel_runs" "exp_name=$CFG" "$@"
