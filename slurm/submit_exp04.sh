#!/bin/bash
# Build a manifest of pending EXP-04 (seed, wd) pairs and sbatch the dripper.
#
# This replaces the old submit_exp04_full.sh (which ran on the login node and
# died) and submit_exp04_array.sh (incompatible with QOSMaxSubmitJobPerUserLimit).
#
# Flow:
#   1. Read seeds + weight_decay_sweep from configs/exp04_full.yaml
#   2. Skip pairs whose topology_metrics.json already has >= 81 records
#   3. Write pending pairs to slurm/manifests/exp04_<timestamp>.txt
#   4. sbatch slurm/drip_exp04.sh <manifest>  — runs the dripper as a SLURM job
#
# The dripper is itself a queued job. It uses 1 of the ~20 submit slots,
# leaving 17 for the actual A100 work jobs, all running in parallel up to
# the concurrent cap.
#
# Usage:
#   bash slurm/submit_exp04.sh

set -euo pipefail

CONFIG=${CONFIG:-configs/exp04_full.yaml}

mkdir -p slurm/manifests slurm/logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="slurm/manifests/exp04_${TIMESTAMP}.txt"

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

echo ""
echo "Submitting dripper: sbatch slurm/drip_exp04.sh $MANIFEST"
JOB=$(sbatch --parsable slurm/drip_exp04.sh "$MANIFEST")
echo "Dripper job: $JOB"
echo ""
echo "Monitor dripper:  tail -f slurm/logs/${JOB}_drip.out"
echo "Monitor queue:    squeue -u \$USER"
echo "Cancel dripper:   scancel $JOB   (does NOT cancel already-submitted work jobs)"
