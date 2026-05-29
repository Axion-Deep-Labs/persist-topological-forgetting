#!/bin/bash
#SBATCH --job-name=exp04_topo
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-119%20
#SBATCH --output=slurm/logs/%A_%a_exp04_topo.out
#SBATCH --error=slurm/logs/%A_%a_exp04_topo.err
#
# EXP-04 constrained study — PERSISTENT HOMOLOGY + BASELINES ONLY (decoupled).
# Reads the checkpoints written by run_exp04_train.sh and computes PH (151 ckpts
# x 5 slices) + comparator/baselines for one (weight_decay, seed) per array task.
# No training here. Resume-safe: run_analysis_pass skips already-computed steps
# and saves incrementally, so a re-run of a killed task picks up where it left off.
#
# 120 tasks = 4 WD x 30 seeds. Array throttled to 20 concurrent (Discovery QOS cap=20).
# --mem=32G + thread caps below mitigate the scipy/ripser RAM spike that OOM-killed
# the pilot re-analysis locally.
#
# Usage on HPC (AFTER run_exp04_train.sh has finished cleanly):
#   cd /fs1/scratch/cag1145/axiondeep-research
#   sbatch slurm/run_exp04_topology.sh
#
# To chain automatically instead of waiting manually, submit with a dependency:
#   TRAIN=$(sbatch --parsable slurm/run_exp04_train.sh)
#   sbatch --dependency=afterok:${TRAIN} slurm/run_exp04_topology.sh
# (afterok on an array waits for ALL train tasks; for per-task recovery prefer the
#  manual two-step submission so one failed train seed doesn't block all PH.)

set -euo pipefail

# Cap BLAS/OMP threads to suppress scipy/ripser peak RAM (pilot OOM mitigation).
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

REPO_ROOT="${SLURM_SUBMIT_DIR:-/fs1/scratch/cag1145/axiondeep-research}"
cd "${REPO_ROOT}"
source /fs1/scratch/cag1145/persist-env/bin/activate

# --- Map array task id -> (weight_decay, seed) --- MUST match run_exp04_train.sh.
WD_GRID=(0.01 0.03 0.10 0.30)
NUM_SEEDS=30
SEED_BASE=2000

TID="${SLURM_ARRAY_TASK_ID}"
WD_IDX=$(( TID / NUM_SEEDS ))
SEED_IDX=$(( TID % NUM_SEEDS ))
WD="${WD_GRID[$WD_IDX]}"
SEED=$(( SEED_BASE + SEED_IDX ))
OUTDIR="results/exp04_full/wd_${WD}"
CKPT_DIR="${OUTDIR}/seed_${SEED}/checkpoints"

echo "[exp04_topo] task=${TID} WD=${WD} seed=${SEED} outdir=${OUTDIR} host=$(hostname)"

# Guard: do not silently produce empty results if training output is missing.
if [ ! -d "${CKPT_DIR}" ]; then
    echo "[exp04_topo] ERROR: checkpoints missing at ${CKPT_DIR} — train this seed first." >&2
    exit 2
fi

python -u -m experiments.exp04_grokking_topology.run_pilot \
    --config configs/exp04_full_study.yaml \
    --skip-training \
    --seed "${SEED}" \
    --weight-decay "${WD}" \
    --output-dir "${OUTDIR}"

echo "[exp04_topo] done task=${TID} WD=${WD} seed=${SEED}"
