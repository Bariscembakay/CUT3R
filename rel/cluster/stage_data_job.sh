#!/usr/bin/env bash
# Copy registry datasets into this node's /scratch/$USER/data, so a training job
# on a GCP node reads local disk instead of /data, which these nodes read across
# sites (slow enough to stall the loader into the idle-GPU reaper). CPU-only;
# resumable (rsync); a dataset is complete once <name>/.staged exists.
# Usage: sbatch --nodelist=<node> [...] stage_data_job.sh <dataset> [<dataset> ...]
set -euo pipefail
DATA=/scratch/$USER/data
mkdir -p "$DATA"
for ds in "$@"; do
    [ -f "$DATA/$ds/.staged" ] && { echo "$ds: already staged"; continue; }
    dataset pull "$ds" >/dev/null
    t0=$(date +%s)
    rsync -a --whole-file "/data/$ds/" "$DATA/$ds/"
    touch "$DATA/$ds/.staged"
    echo "$(date) $ds: $(du -sh "$DATA/$ds" | cut -f1) in $(( ($(date +%s) - t0) / 60 )) min"
done
