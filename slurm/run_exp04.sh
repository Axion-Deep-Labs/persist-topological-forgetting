#!/bin/bash
#SBATCH --job-name=grok
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%j_%x.out
#SBATCH --error=slurm/logs/%j_%x.err

SEED=${1:?"Usage: sbatch run_exp04.sh <seed> [--skip-training|--skip-analysis]"}
shift
EXTRA_ARGS="$@"

source /fs1/scratch/cag1145/persist-env/bin/activate
export CLEARML_OFF=1
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Job: $SLURM_JOB_ID"
echo "EXP-04: Grokking Topology Pilot"
echo "Seed: $SEED"
echo "Extra args: $EXTRA_ARGS"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "============================================"

cd /fs1/scratch/cag1145/axiondeep-research

/fs1/scratch/cag1145/persist-env/bin/python -m experiments.exp04_grokking_topology.run_pilot \
    --config configs/exp04_pilot.yaml \
    --seed $SEED \
    $EXTRA_ARGS

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "Finished: $(date)"
exit $EXIT_CODE
