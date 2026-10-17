#!/usr/bin/env bash
# CPU-only: copy the pilot datasets onto this node's /scratch, so training does
# not read /data live (from gcp-eu1 that is sof1's CephFS, cross-site, ~60 MB/s
# for a single stream). Parallel rsync per sequence directory. Idempotent.
# Usage: sbatch --nodelist=<node> [...] stage_data_job.sh [dest]
set -euo pipefail
DEST="${1:-/scratch/$USER/data}"
mkdir -p "$DEST"
for ds in arkitscenes-cut3r-video tartanair-v1; do
    dataset pull "$ds" >/dev/null
done
copy() {  # <src_root> <dest_root> <depth of the parallel unit>
    local src="$1" dst="$2" depth="$3"
    (cd "$src" && find . -mindepth "$depth" -maxdepth "$depth" -type d) |
        xargs -P "${SLURM_CPUS_PER_TASK:-16}" -I{} bash -c \
            'mkdir -p "'"$dst"'/{}" && rsync -a "'"$src"'/{}/" "'"$dst"'/{}/"'
    rsync -a --exclude='*/' "$src/" "$dst/"   # top-level files
}
time copy /data/arkitscenes-cut3r-video "$DEST/arkitscenes-cut3r-video" 2
time copy /data/tartanair-v1 "$DEST/tartanair-v1" 3
du -sh "$DEST"/*
touch "$DEST/.staged_ok"
