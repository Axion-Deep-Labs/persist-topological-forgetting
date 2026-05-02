#!/bin/bash
#SBATCH --job-name=grokF
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%j_%x.out
#SBATCH --error=slurm/logs/%j_%x.err

# EXP-04 Full Study runner: takes <seed> <weight_decay> as args.
# Each job runs one seed at one WD value, full pipeline (train + analysis).
#
# Usage:
#   sbatch slurm/run_exp04_full.sh <seed> <weight_decay>
#
# Output goes to: results/exp04_full/wd_<wd>/seed_<seed>/

SEED=${1:?"Usage: sbatch run_exp04_full.sh <seed> <weight_decay>"}
WD=${2:?"Usage: sbatch run_exp04_full.sh <seed> <weight_decay>"}

source /fs1/scratch/cag1145/persist-env/bin/activate
export CLEARML_OFF=1
export PYTHONUNBUFFERED=1

# Format WD for directory name (0.03 -> 0.03, 0.1 -> 0.10, 0.3 -> 0.30)
WD_TAG=$(printf "%.2f" "$WD")
OUT_DIR="results/exp04_full/wd_${WD_TAG}"

echo "============================================"
echo "Job: $SLURM_JOB_ID"
echo "EXP-04: Grokking Topology — FULL STUDY"
echo "Seed: $SEED"
echo "Weight decay: $WD (dir tag: $WD_TAG)"
echo "Output: $OUT_DIR"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "============================================"

cd /fs1/scratch/cag1145/axiondeep-research
mkdir -p "$OUT_DIR"

/fs1/scratch/cag1145/persist-env/bin/python -m experiments.exp04_grokking_topology.run_pilot \
    --config configs/exp04_full.yaml \
    --seed "$SEED" \
    --weight-decay "$WD" \
    --output-dir "$OUT_DIR"

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "Finished: $(date)"
exit $EXIT_CODE
