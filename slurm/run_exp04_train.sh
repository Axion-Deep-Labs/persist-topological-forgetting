#!/bin/bash
#SBATCH --job-name=exp04_train
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-119%20
#SBATCH --output=slurm/logs/%A_%a_exp04_train.out
#SBATCH --error=slurm/logs/%A_%a_exp04_train.err
#
# EXP-04 constrained study — TRAINING ONLY (decoupled from PH).
# Trains one (weight_decay, seed) per array task and saves the 151-checkpoint
# schedule defined in configs/exp04_full_study.yaml. No topology here.
#
# 120 tasks = 4 WD x 30 seeds. Array throttled to 20 concurrent (Discovery QOS cap=20).
#
# Run topology SEPARATELY after this array completes (slurm/run_exp04_topology.sh),
# so a PH failure never forces a retrain and vice versa.
#
# Usage on HPC:
#   cd /fs1/scratch/cag1145/axiondeep-research
#   sbatch slurm/run_exp04_train.sh
#   # then, after it finishes cleanly:
#   sbatch slurm/run_exp04_topology.sh

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/fs1/scratch/cag1145/axiondeep-research}"
cd "${REPO_ROOT}"
source /fs1/scratch/cag1145/persist-env/bin/activate

# --- Map array task id -> (weight_decay, seed) ---
# MUST match run_exp04_topology.sh exactly so dirs line up.
WD_GRID=(0.01 0.03 0.10 0.30)
NUM_SEEDS=30
SEED_BASE=2000

TID="${SLURM_ARRAY_TASK_ID}"
WD_IDX=$(( TID / NUM_SEEDS ))
SEED_IDX=$(( TID % NUM_SEEDS ))
WD="${WD_GRID[$WD_IDX]}"
SEED=$(( SEED_BASE + SEED_IDX ))
OUTDIR="results/exp04_full/wd_${WD}"

echo "[exp04_train] task=${TID} WD=${WD} seed=${SEED} outdir=${OUTDIR} host=$(hostname)"

python -u -m experiments.exp04_grokking_topology.run_pilot \
    --config configs/exp04_full_study.yaml \
    --skip-analysis \
    --seed "${SEED}" \
    --weight-decay "${WD}" \
    --output-dir "${OUTDIR}"

echo "[exp04_train] done task=${TID} WD=${WD} seed=${SEED}"
