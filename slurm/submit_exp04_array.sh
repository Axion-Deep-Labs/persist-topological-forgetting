#!/bin/bash
# Build a manifest of pending (seed, wd) pairs and submit them as a single
# SLURM array job. Replaces submit_exp04_full.sh, which used an external
# dripper loop that died when login-node sessions ended.
#
# Behavior:
#   - Reads seeds and weight_decay_sweep from configs/exp04_full.yaml
#   - Skips (seed, wd) pairs that already have a complete topology_metrics.json
#   - Writes pending pairs to slurm/manifests/exp04_<timestamp>.txt
#   - Submits sbatch --array=0-(N-1)%18 slurm/run_exp04_array.sh <manifest>
#   - SLURM handles the concurrent cap; no external process to die
#
# Usage:
#   bash slurm/submit_exp04_array.sh
#
# Overrides:
#   CONCURRENT=12 bash slurm/submit_exp04_array.sh   # cap concurrent at 12

set -euo pipefail

CONCURRENT=${CONCURRENT:-18}
CONFIG=${CONFIG:-configs/exp04_full.yaml}

mkdir -p slurm/manifests slurm/logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="slurm/manifests/exp04_${TIMESTAMP}.txt"

# Pull seeds + WDs from the config
SEEDS=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(' '.join(str(s) for s in c['seeds']))")
WDS=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(' '.join(str(w) for w in c['weight_decay_sweep']))")

is_complete() {
    local seed=$1 wd=$2
    local wd_tag
    wd_tag=$(printf "%.2f" "$wd")
    local f="results/exp04_full/wd_${wd_tag}/seed_${seed}/topology_metrics.json"
    [[ -f "$f" ]] || return 1
    local n
    n=$(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo 0)
    [[ "$n" -ge 81 ]]
}

skipped=0
pending=0
: > "$MANIFEST"
for WD in $WDS; do
    for SEED in $SEEDS; do
        if is_complete "$SEED" "$WD"; then
            skipped=$((skipped + 1))
            continue
        fi
        echo "$SEED $WD" >> "$MANIFEST"
        pending=$((pending + 1))
    done
done

echo "Manifest: $MANIFEST"
echo "Pending:  $pending"
echo "Skipped:  $skipped (already complete)"

if [[ "$pending" -eq 0 ]]; then
    echo "Nothing to submit. All trajectories complete."
    exit 0
fi

ARRAY_SPEC="0-$((pending - 1))%${CONCURRENT}"
echo "Submitting array: sbatch --array=${ARRAY_SPEC} slurm/run_exp04_array.sh $MANIFEST"

JOB=$(sbatch --parsable --array="$ARRAY_SPEC" slurm/run_exp04_array.sh "$MANIFEST")
echo "Submitted array job: $JOB"
echo "Monitor:  squeue -u \$USER -j $JOB"
echo "Logs:     slurm/logs/${JOB}_*_g4arr.{out,err}"
