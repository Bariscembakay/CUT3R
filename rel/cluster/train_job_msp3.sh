#!/usr/bin/env bash
# One CUT3R-rel scale-up arm on msp3, N GPUs via accelerate. Code arrives by git
# (the fork's cut3r-rel branch), data from msp3's registry, results shipped to
# sof1's /group on exit (success or not).
# Usage: sbatch --constraint=zone-msp3 --gres=gpu:h200:N --signal=B:USR1@1200 [...] train_job_msp3.sh <config_name> [hydra overrides...]
# Resubmitting the same config resumes from the last shipped epoch, on any node.
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
for ds in arkitscenes-cut3r-video-full nuscenes-cut3r-video nuscenes-cut3r-video-6cam scannet-v2-extracted tartanair-v1 tartanair-v2 tartanground; do
    dataset pull "$ds" >/dev/null
done
export DATA_ROOT=/data
# wandb credentials are per zone; without them, log offline and sync from sof1.
[ -f "$HOME/.netrc" ] || export WANDB_MODE=offline

RUN_DIR="/scratch/$USER/cut3r_rel_runs/$CFG"
mkdir -p "$RUN_DIR"
# /scratch is per node, so a resubmission that lands elsewhere resumes from sof1's copy.
if [ ! -f "$RUN_DIR/checkpoint-last.pth" ] && ssh sof1 test -f "$DEST/checkpoint-last.pth"; then
    rsync -a "sof1:$DEST/checkpoint-last.pth" "$RUN_DIR/"
    echo "[train_job_msp3] resuming from sof1:$DEST/checkpoint-last.pth"
fi
LOG=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'StdOut=\K\S+')
ship() {
    ssh sof1 "mkdir -p $DEST/_joblogs" && rsync -a "$RUN_DIR/" "sof1:$DEST/" && rsync -a "$LOG" "sof1:$DEST/_joblogs/" || true
    echo "[train_job_msp3] shipped $RUN_DIR -> sof1:$DEST"
}
trap ship EXIT
trap 'exit 143' TERM   # bash skips the EXIT trap on an untrapped SIGTERM
# A 6.8 GB checkpoint takes minutes to cross zones -- far longer than Slurm's
# kill grace -- so --signal=B:USR1@1200 ships while training keeps running.
trap ship USR1

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "[train_job_msp3] $(hostname) ${NGPU}x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train_job_msp3] git $(git -C "$REPO" rev-parse HEAD) config=$CFG overrides=$*"
cd "$REPO/src"
"$ENV/bin/accelerate" launch --num_processes "$NGPU" --main_process_port $((29500 + RANDOM % 1000)) \
    train.py --config-name "$CFG" "save_dir=/scratch/$USER/cut3r_rel_runs" "exp_name=$CFG" "$@" &
TRAIN=$!
# Backgrounded so USR1 interrupts `wait` (a foreground child would defer the trap
# until training ended); keep waiting until training itself exits.
rc=0
while :; do
    wait "$TRAIN" && { rc=0; break; } || rc=$?
    kill -0 "$TRAIN" 2>/dev/null || break
done
exit "$rc"
