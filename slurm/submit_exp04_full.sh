#!/bin/bash
# Submit EXP-04 full study (30 seeds × 3 weight decays = 90 jobs) to SLURM,
# respecting NMSU's ~20-job submitted-at-once QOS limit.
#
# Behavior:
#   - Reads seeds and weight_decay_sweep from configs/exp04_full.yaml
#   - Submits up to MAX_QUEUED jobs at a time
#   - When the queue fills, sleeps SLEEP_SECS, then re-checks
#   - Skips (seed, wd) pairs whose output dir already has a complete topology JSON
#   - Logs every submission to slurm/logs/submit_exp04_full.log
#
# Usage:
#   bash slurm/submit_exp04_full.sh
#
# Override defaults:
#   MAX_QUEUED=15 SLEEP_SECS=300 bash slurm/submit_exp04_full.sh

set -euo pipefail

MAX_QUEUED=${MAX_QUEUED:-18}    # leave 2 slots for verification re-runs / urgent jobs
SLEEP_SECS=${SLEEP_SECS:-180}   # 3 min between queue checks
USER_ID=${USER_ID:-cag1145}
LOG_FILE="slurm/logs/submit_exp04_full.log"

mkdir -p slurm/logs
echo "" >> "$LOG_FILE"
echo "==== Submission run at $(date) ====" >> "$LOG_FILE"

# Pull seeds and WD sweep out of the config (proper YAML parse via python3)
SEEDS=$(python3 -c "import yaml; c=yaml.safe_load(open('configs/exp04_full.yaml')); print(' '.join(str(s) for s in c['seeds']))")
WDS=$(python3 -c "import yaml; c=yaml.safe_load(open('configs/exp04_full.yaml')); print(' '.join(str(w) for w in c['weight_decay_sweep']))")

echo "Seeds: $SEEDS"
echo "Weight decays: $WDS"
echo ""

queued_count() {
    # Returns count of currently queued+running jobs for this user.
    squeue -h -u "$USER_ID" -o "%i" 2>/dev/null | wc -l
}

is_complete() {
    # Returns 0 if (seed, wd) already has a finished topology_metrics.json with
    # at least 81 records (matches checkpoint count). Conservative: re-runs if
    # file is missing or short.
    local seed=$1 wd=$2
    local wd_tag=$(printf "%.2f" "$wd")
    local f="results/exp04_full/wd_${wd_tag}/seed_${seed}/topology_metrics.json"
    if [[ ! -f "$f" ]]; then
        return 1
    fi
    local n
    n=$(python3 -c "import json,sys; print(len(json.load(open('$f'))))" 2>/dev/null || echo 0)
    if [[ "$n" -ge 81 ]]; then
        return 0
    fi
    return 1
}

submitted=0
skipped=0
total=0

for WD in $WDS; do
    for SEED in $SEEDS; do
        total=$((total + 1))

        if is_complete "$SEED" "$WD"; then
            echo "[skip] seed=$SEED wd=$WD already complete"
            echo "[skip] seed=$SEED wd=$WD already complete" >> "$LOG_FILE"
            skipped=$((skipped + 1))
            continue
        fi

        # Wait until queue has space
        while true; do
            n=$(queued_count)
            if [[ "$n" -lt "$MAX_QUEUED" ]]; then
                break
            fi
            echo "[wait] $n jobs queued (cap $MAX_QUEUED), sleeping ${SLEEP_SECS}s..."
            sleep "$SLEEP_SECS"
        done

        # Submit
        WD_TAG=$(printf "%.2f" "$WD")
        JOB=$(sbatch --parsable \
                     --job-name="g4_${SEED}_${WD_TAG}" \
                     slurm/run_exp04_full.sh "$SEED" "$WD")
        echo "[sub]  seed=$SEED wd=$WD -> job $JOB"
        echo "[sub]  seed=$SEED wd=$WD -> job $JOB ($(date +%T))" >> "$LOG_FILE"
        submitted=$((submitted + 1))
        sleep 1  # tiny pause to avoid sbatch rate-limit
    done
done

echo ""
echo "==== Submission complete ===="
echo "  total considered:  $total"
echo "  submitted:         $submitted"
echo "  skipped (done):    $skipped"
echo "  log:               $LOG_FILE"
echo ""
echo "Monitor: squeue -u $USER_ID"
echo "Tail:    tail -f $LOG_FILE"

echo "==== Done at $(date): submitted=$submitted skipped=$skipped ====" >> "$LOG_FILE"
