#!/bin/bash
#SBATCH --job-name=metric_diag
#SBATCH --partition=normal
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%j_metric_diagnostic.out
#SBATCH --error=slurm/logs/%j_metric_diagnostic.err
#
# Phase I-B metric verification diagnostic.
# Pure read-from-disk analysis: compares full-softmax vs restricted-softmax
# retention at step 10 across all 6 cross-dataset pairs * 19 architectures.
# CPU-only; no GPU needed.
#
# Writes:
#   results/exp04_metric_diagnostic/per_arch.json
#   results/exp04_metric_diagnostic/per_pair.json
#   results/exp04_metric_diagnostic/aggregate.json
#
# Three-tier verdict on the aggregate (locked in script):
#   ALIGNED
#   MOSTLY_ALIGNED_GEOMETRY_DEPENDENT
#   MISALIGNED
#
# Usage on HPC:
#   cd /fs1/scratch/cag1145/axiondeep-research
#   sbatch slurm/run_metric_diagnostic.sh

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/fs1/scratch/cag1145/axiondeep-research}"
echo "[metric_diagnostic] REPO_ROOT=${REPO_ROOT}"
cd "${REPO_ROOT}"
echo "[metric_diagnostic] PWD=${PWD}"

source /fs1/scratch/cag1145/persist-env/bin/activate
echo "[metric_diagnostic] python=$(which python)"

python scripts/exp04_metric_diagnostic.py
