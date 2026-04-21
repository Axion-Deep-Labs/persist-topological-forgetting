#!/bin/bash
#SBATCH --job-name=p3b_restricted
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%j_phase3b_restricted.out
#SBATCH --error=slurm/logs/%j_phase3b_restricted.err
#
# Restricted-softmax re-evaluation of all cross-dataset forgetting runs.
# Pure compute-from-disk. See drafts/phase_1b_g1_metric_audit_memo.md.
#
# Usage on HPC:
#   cd /fs1/scratch/cag1145/axiondeep-research
#   sbatch slurm/run_phase3b_restricted.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source /fs1/scratch/cag1145/persist-env/bin/activate

python -m experiments.exp01_topological_persistence.phase3b_restricted_softmax_eval \
    --results-dir ./results \
    --configs-dir ./configs \
    --eval-batch-size 128
