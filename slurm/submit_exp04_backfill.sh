#!/bin/bash
# EXP-04 training backfill submitter (run on the Discovery LOGIN node, not via sbatch).
#
# The partition QOS caps the user at MaxSubmitPU=20 (jobs in the system) and
# MaxJobsPU=10 (running). A 120-element array can't be submitted at once, so this
# script tops the queue back up to the cap, submitting ONLY (WD,seed) runs that are:
#   - not yet complete  (no checkpoints/step_120000.pt), AND
#   - not already queued/running (by global array task id).
# Re-run it whenever the queue drains (or drive it from a watch/loop). It never
# double-submits a running seed, so it is safe to re-run anytime.
#
# Usage:
#   bash slurm/submit_exp04_backfill.sh --dry-run   # show what it WOULD submit
#   bash slurm/submit_exp04_backfill.sh             # actually submit
set -euo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

CAP=20                       # MaxSubmitPU (jobs in system: pending+running)
JOB_NAME="exp04_train"       # must match #SBATCH --job-name in run_exp04_train.sh
SBATCH_SCRIPT="slurm/run_exp04_train.sh"
WD_GRID=(0.01 0.03 0.10 0.30)
NUM_SEEDS=30
SEED_BASE=2000
RESULTS="results/exp04_full"
TOTAL=$(( ${#WD_GRID[@]} * NUM_SEEDS ))   # 120

# --- currently queued/running global task ids for our train jobs ---
# -r expands array elements to one line each; %K = array (global) task id, %j = name.
declare -A IS_ACTIVE
while read -r kid name; do
    [ "$name" = "$JOB_NAME" ] || continue
    [[ "$kid" =~ ^[0-9]+$ ]] || continue
    IS_ACTIVE[$kid]=1
done < <(squeue -h -r -u "$USER" -o "%K %j" 2>/dev/null || true)

QUEUE_DEPTH=$(squeue -h -r -u "$USER" -o "%j" 2>/dev/null | grep -c "^${JOB_NAME}$" || true)
SLOTS=$(( CAP - QUEUE_DEPTH ))
echo "[backfill] queue_depth=${QUEUE_DEPTH} cap=${CAP} free_slots=${SLOTS}"

# --- incomplete & not-queued task ids ---
TODO=()
DONE_CT=0
for TID in $(seq 0 $(( TOTAL - 1 )) ); do
    WD_IDX=$(( TID / NUM_SEEDS ))
    SEED_IDX=$(( TID % NUM_SEEDS ))
    WD="${WD_GRID[$WD_IDX]}"
    SEED=$(( SEED_BASE + SEED_IDX ))
    if [ -f "${RESULTS}/wd_${WD}/seed_${SEED}/checkpoints/step_120000.pt" ]; then
        DONE_CT=$(( DONE_CT + 1 ))
        continue
    fi
    [ -n "${IS_ACTIVE[$TID]:-}" ] && continue
    TODO+=("$TID")
done
echo "[backfill] complete=${DONE_CT}/${TOTAL}  incomplete&not-queued=${#TODO[@]}"

if [ "${#TODO[@]}" -eq 0 ]; then
    echo "[backfill] nothing to do — all runs complete or in flight."
    exit 0
fi
if [ "$SLOTS" -le 0 ]; then
    echo "[backfill] queue is full (${QUEUE_DEPTH}/${CAP}); re-run after it drains."
    exit 0
fi

# take up to SLOTS of them
N=$(( SLOTS < ${#TODO[@]} ? SLOTS : ${#TODO[@]} ))
SUBMIT=("${TODO[@]:0:$N}")
LIST=$(IFS=,; echo "${SUBMIT[*]}")
echo "[backfill] submitting ${N} task(s): ${LIST}"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[backfill] --dry-run: not submitting."
    exit 0
fi
sbatch --array="${LIST}" "${SBATCH_SCRIPT}"
