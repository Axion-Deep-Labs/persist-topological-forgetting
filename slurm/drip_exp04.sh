#!/bin/bash
#SBATCH --job-name=g4drip
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/%j_drip.out
#SBATCH --error=slurm/logs/%j_drip.err

# EXP-04 dripper: runs as a SLURM CPU job (lives on a compute node so it
# survives login-node disconnects). Reads a manifest of pending (seed, wd)
# pairs and submits them one at a time, waiting for queue slots to open.
#
# Caps respected:
#   - MAX_SUBMIT: total submitted (queued + running) jobs for this user.
#     NMSU QOSMaxSubmitJobPerUserLimit is ~20, so default is 18: 1 for this
#     dripper, up to 17 for work jobs, 2 slot buffer for ad-hoc submissions.
#   - The work jobs sized to one A100 each via slurm/run_exp04_full.sh
#
# Usage (do NOT run on login node — sbatch it):
#   sbatch slurm/drip_exp04.sh <manifest_path>

set -uo pipefail   # NOT -e: tolerate transient squeue/sbatch failures

MANIFEST=${1:?"Usage: sbatch slurm/drip_exp04.sh <manifest>"}
MAX_SUBMIT=${MAX_SUBMIT:-18}
SLEEP_SECS=${SLEEP_SECS:-180}
USER_ID=${USER_ID:-cag1145}

cd /fs1/scratch/cag1145/axiondeep-research

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest $MANIFEST not found"
    exit 2
fi

echo "============================================"
echo "EXP-04 dripper"
echo "Manifest:    $MANIFEST"
echo "MAX_SUBMIT:  $MAX_SUBMIT (queued+running cap including this dripper)"
echo "SLEEP_SECS:  $SLEEP_SECS"
echo "Started:     $(date)"
echo "============================================"

queued_count() {
    squeue -h -u "$USER_ID" -o "%i" 2>/dev/null | wc -l
}

LINES=$(wc -l < "$MANIFEST")
echo "Pending entries: $LINES"
echo ""

idx=0
submitted=0
while [[ $idx -lt $LINES ]]; do
    LINE=$(sed -n "$((idx + 1))p" "$MANIFEST")
    SEED=$(echo "$LINE" | awk '{print $1}')
    WD=$(echo "$LINE" | awk '{print $2}')

    if [[ -z "$SEED" || -z "$WD" ]]; then
        idx=$((idx + 1))
        continue
    fi

    # Wait for room in the queue (this dripper itself counts toward queued_count)
    while true; do
        n=$(queued_count || echo 99)
        if [[ "$n" -lt "$MAX_SUBMIT" ]]; then
            break
        fi
        echo "[wait] $n jobs queued (cap $MAX_SUBMIT), sleeping ${SLEEP_SECS}s..."
        sleep "$SLEEP_SECS"
    done

    WD_TAG=$(printf "%.2f" "$WD")
    JOB=$(sbatch --parsable \
                 --job-name="g4_${SEED}_${WD_TAG}" \
                 slurm/run_exp04_full.sh "$SEED" "$WD" 2>&1)

    # sbatch may transiently fail with QOS error if our cap calc drifts
    if [[ "$JOB" =~ ^[0-9]+$ ]]; then
        echo "[sub] idx=$idx seed=$SEED wd=$WD -> job $JOB ($(date +%T))"
        submitted=$((submitted + 1))
        idx=$((idx + 1))
        sleep 2
    else
        echo "[err] idx=$idx seed=$SEED wd=$WD -> $JOB"
        echo "[err] sbatch failed; sleeping ${SLEEP_SECS}s and retrying"
        sleep "$SLEEP_SECS"
        # do NOT increment idx; retry this entry
    fi
done

echo ""
echo "============================================"
echo "Dripper complete. Submitted $submitted of $LINES manifest entries."
echo "Finished:    $(date)"
echo "============================================"
