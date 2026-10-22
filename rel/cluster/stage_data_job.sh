#!/usr/bin/env bash
# Copy registry datasets into this node's /scratch/$USER/data, so a training job
# on a GCP node reads local disk instead of /data, which these nodes read across
# sites (slow enough to stall the loader into the idle-GPU reaper). CPU-only;
# resumable (rsync); a dataset is complete once <name>/.staged exists. One rsync
# stream reads /data at ~35 MB/s here and concurrent streams add up, so each
# dataset is copied as STAGE_STREAMS parallel rsyncs over its depth-2 entries.
# Usage: sbatch --nodelist=<node> [...] stage_data_job.sh <dataset> [<dataset> ...]
set -euo pipefail
DATA=/scratch/$USER/data
P=${STAGE_STREAMS:-8}
mkdir -p "$DATA"
for ds in "$@"; do
    [ -f "$DATA/$ds/.staged" ] && { echo "$ds: already staged"; continue; }
    dataset pull "$ds" >/dev/null
    t0=$(date +%s)
    mkdir -p "$DATA/$ds"
    (cd "/data/$ds" && find . -mindepth 1 -maxdepth 1 ! -type d && find . -mindepth 2 -maxdepth 2) |
        xargs -P "$P" -I{} rsync -aR --whole-file "/data/$ds/./{}" "$DATA/$ds/"
    touch "$DATA/$ds/.staged"
    echo "$(date) $ds: $(du -sh "$DATA/$ds" | cut -f1) in $(( ($(date +%s) - t0) / 60 )) min"
done
