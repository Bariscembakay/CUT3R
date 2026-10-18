#!/usr/bin/env bash
# One CUT3R-rel scale-up arm on msp3, N GPUs via accelerate. Code arrives by git
# (the fork's cut3r-rel branch), data from msp3's registry, results shipped to
# sof1's /group on exit (success or not).
# Usage: sbatch --constraint=zone-msp3 --gres=gpu:h200:N [...] train_job_msp3.sh <config_name> [hydra overrides...]
set -euo pipefail
CFG="$1"; shift
REPO="${REPO:-$HOME/CUT3R}"
FORK="${FORK:-https://github.com/Bariscembakay/CUT3R.git}"
BRANCH=cut3r-rel
CKPT=/group/compact-3dmem/checkpoints/CUT3R/cut3r_512_dpt_4_64.pth
DEST="/group/compact-3dmem/campaigns/cut3r_rel/scale/$CFG"   # on sof1

if [ ! -d "$REPO/.git" ]; then
    git clone -q -b "$BRANCH" "$FORK" "$REPO"
else
    git -C "$REPO" fetch -q origin "$BRANCH"
    git -C "$REPO" checkout -q "$BRANCH"
    git -C "$REPO" merge -q --ff-only "origin/$BRANCH"
fi
source "$REPO/rel/cluster/setup_env.sh"
ENV="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r"

# msp3's /group is a different (300 G) filesystem: the weights must exist there too.
if [ ! -f "$CKPT" ]; then
    mkdir -p "$(dirname "$CKPT")"
    rsync -a "sof1:$CKPT" "$CKPT"
fi
for ds in arkitscenes-cut3r-video-full nuscenes-cut3r-video scannet-v2-extracted tartanair-v1 tartanair-v2; do
    dataset pull "$ds" >/dev/null
done
export DATA_ROOT=/data
# wandb credentials are per zone; without them, log offline and sync from sof1.
[ -f "$HOME/.netrc" ] || export WANDB_MODE=offline

RUN_DIR="/scratch/$USER/cut3r_rel_runs/$CFG"
mkdir -p "$RUN_DIR"
ship() { ssh sof1 "mkdir -p $DEST" && rsync -a "$RUN_DIR/" "sof1:$DEST/" || true; echo "[train_job_msp3] shipped $RUN_DIR -> sof1:$DEST"; }
trap ship EXIT

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "[train_job_msp3] $(hostname) ${NGPU}x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train_job_msp3] git $(git -C "$REPO" rev-parse HEAD) config=$CFG overrides=$*"
cd "$REPO/src"
"$ENV/bin/accelerate" launch --num_processes "$NGPU" --main_process_port $((29500 + RANDOM % 1000)) \
    train.py --config-name "$CFG" "save_dir=/scratch/$USER/cut3r_rel_runs" "exp_name=$CFG" "$@"
