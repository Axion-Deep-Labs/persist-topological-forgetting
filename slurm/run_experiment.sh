#!/bin/bash
#SBATCH --job-name=persist
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%j_%x.out
#SBATCH --error=slurm/logs/%j_%x.err

# Usage:
#   sbatch slurm/run_experiment.sh configs/exp01_vit_b_16_imagenet100.yaml phase1
#   sbatch slurm/run_experiment.sh configs/exp01_vit_b_16_imagenet100.yaml phase2
#   sbatch slurm/run_experiment.sh configs/exp01_vit_b_16_imagenet100.yaml phase3
#   sbatch slurm/run_experiment.sh configs/exp01_vit_b_16_imagenet100.yaml phase3 --ewc
#   sbatch slurm/run_experiment.sh configs/exp01_vit_b_16_imagenet100.yaml phase3 --si

CONFIG=${1:?"Usage: sbatch run_experiment.sh <config.yaml> <phase> [extra_args]"}
PHASE=${2:?"Usage: sbatch run_experiment.sh <config.yaml> <phase> [extra_args]"}
shift 2
EXTRA_ARGS="$@"

# Map phase name to module
case $PHASE in
    phase1) MODULE="phase1_train_task_a" ;;
    phase2) MODULE="phase2_landscape_topology" ;;
    phase2c) MODULE="phase2c_cubical_persistence" ;;
    phase2b) MODULE="phase2b_displacement_analysis" ;;
    phase3) MODULE="phase3_sequential_forgetting" ;;
    phase4) MODULE="phase4_correlation" ;;
    phase5) MODULE="phase5_predictive_model" ;;
    phase6) MODULE="phase6_pooled_interaction" ;;
    *) echo "Unknown phase: $PHASE"; exit 1 ;;
esac

# Environment setup for NMSU Discovery HPC
# Note: module loads fail on compute nodes but venv is self-contained (PyTorch+CUDA bundled)
source /fs1/scratch/cag1145/persist-env/bin/activate
export CLEARML_OFF=1
export PYTHONUNBUFFERED=1

# Auto-add --pretrained for ImageNet-100 Phase 1 runs
PRETRAINED_FLAG=""
if [[ "$CONFIG" == *"imagenet100"* ]] && [[ "$PHASE" == "phase1" ]]; then
    PRETRAINED_FLAG="--pretrained"
fi

echo "============================================"
echo "Job: $SLURM_JOB_ID"
echo "Config: $CONFIG"
echo "Phase: $PHASE (module: $MODULE)"
echo "Extra args: $EXTRA_ARGS $PRETRAINED_FLAG"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "============================================"

python -m experiments.exp01_topological_persistence.${MODULE} \
    --config ${CONFIG} \
    ${PRETRAINED_FLAG} \
    ${EXTRA_ARGS}

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "Finished: $(date)"
exit $EXIT_CODE
