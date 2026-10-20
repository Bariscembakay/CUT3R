#!/usr/bin/env bash
# Waymo Open Dataset v1.4.3 perception, individual .tfrecord files (training +
# validation; archived_files would be the same segments again, tarred) into this
# node's /scratch. Needs `gcloud auth login` done once (credentials in ~/.config/gcloud).
# Usage: sbatch --nodelist=<node> [...] download_waymo_job.sh
set -euo pipefail
SDK=/scratch/$USER/google-cloud-sdk
if [ ! -x "$SDK/bin/gcloud" ]; then
    mkdir -p /scratch/$USER/gcloud_dl
    curl -sfL -o /scratch/$USER/gcloud_dl/cli.tar.gz \
        https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
    tar -xzf /scratch/$USER/gcloud_dl/cli.tar.gz -C /scratch/$USER && rm -rf /scratch/$USER/gcloud_dl
    "$SDK/install.sh" --quiet --usage-reporting=false --path-update=false --command-completion=false
fi
OUT=/scratch/$USER/waymo_v1_4_3
mkdir -p "$OUT"
chmod 700 "$OUT"  # licensed data: not readable by other users of the node
echo "$(hostname): $(df -h /scratch | tail -1 | awk '{print $4}') free; account $("$SDK/bin/gcloud" config get-value account 2>/dev/null)"
for split in training validation; do
    "$SDK/bin/gcloud" storage cp --recursive --no-clobber \
        "gs://waymo_open_dataset_v_1_4_3/individual_files/$split" "$OUT/"
    echo "$(date) $split: $(ls "$OUT/$split" | wc -l) files, $(du -sh "$OUT/$split" | cut -f1)"
done
