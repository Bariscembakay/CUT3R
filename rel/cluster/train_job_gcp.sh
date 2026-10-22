#!/usr/bin/env bash
# One CUT3R-rel arm on a GCP GPU node, N GPUs via accelerate. GCP nodes share
# sof1's /home and /group, so the code is this working tree (it must be committed:
# the commit is logged) and results go straight to /group. Data must be staged
# first (stage_data_job.sh) into this node's /scratch.
# Usage: sbatch --constraint=zone-gcp-eu9 --nodelist=<node> --gres=gpu:h100:N --signal=B:USR1@1200 [...] train_job_gcp.sh <config_name> [hydra overrides...]
# Resubmitting the same config on the same node resumes from the last shipped epoch.
set -euo pipefail
CFG="$1"; shift
REPO="${REPO:-$HOME/CUT3R}"
DATASETS="arkitscenes-cut3r-video-full tartanair-v1 tartanair-v2 scannet-v2-extracted nuscenes-cut3r-video-6cam tartanground scannetpp-cut3r-video waymo-cut3r-video"
DATA=/scratch/$USER/data
DEST="/group/compact-3dmem/campaigns/cut3r_rel/scale2/$CFG"

for ds in $DATASETS; do
    [ -f "$DATA/$ds/.staged" ] || { echo "[train_job_gcp] $ds not staged on $(hostname)"; exit 3; }
done
if [ -n "$(git -C "$REPO" status --porcelain -- src config rel/cluster)" ]; then
    echo "[train_job_gcp] uncommitted changes in src/config: the logged commit would not be the code that ran"
    exit 3
fi
source "$REPO/rel/cluster/setup_env.sh"
ENV="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r"
export DATA_ROOT="$DATA"
# batch 8 of 64 views peaks at 79 of 80 GB reserved; this keeps fragmentation from adding to it
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="/scratch/$USER/cut3r_rel_runs/$CFG"
mkdir -p "$RUN_DIR" "$DEST/_joblogs"
if [ ! -f "$RUN_DIR/checkpoint-last.pth" ] && [ -f "$DEST/checkpoint-last.pth" ]; then
    cp "$DEST/checkpoint-last.pth" "$RUN_DIR/"
    echo "[train_job_gcp] resuming from $DEST/checkpoint-last.pth"
fi
LOG=$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'StdOut=\K\S+')
ship() {
    rsync -a "$RUN_DIR/" "$DEST/" && cp "$LOG" "$DEST/_joblogs/" || true
    echo "[train_job_gcp] shipped $RUN_DIR -> $DEST"
}
# GCP nodes are deleted when their lease ends, /scratch with them: ship every 2 h too
( while sleep 7200; do ship; done ) &
SHIPPER=$!
trap 'kill $SHIPPER 2>/dev/null; ship' EXIT
trap 'exit 143' TERM   # bash skips the EXIT trap on an untrapped SIGTERM
trap ship USR1         # --signal=B:USR1@1200: copy out before the walltime kill

NGPU="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
echo "[train_job_gcp] $(hostname) ${NGPU}x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[train_job_gcp] git $(git -C "$REPO" rev-parse HEAD) config=$CFG overrides=$*"
cd "$REPO/src"
"$ENV/bin/accelerate" launch --num_processes "$NGPU" --main_process_port $((29500 + RANDOM % 1000)) \
    train.py --config-name "$CFG" "save_dir=/scratch/$USER/cut3r_rel_runs" "exp_name=$CFG" "$@" &
TRAIN=$!
# backgrounded so USR1 interrupts `wait`; keep waiting until training itself exits
rc=0
while :; do
    wait "$TRAIN" && { rc=0; break; } || rc=$?
    kill -0 "$TRAIN" 2>/dev/null || break
done
exit "$rc"
