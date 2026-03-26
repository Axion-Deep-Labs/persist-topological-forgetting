#!/bin/bash
# Submit all Phase I ImageNet-100 experiments to SLURM.
# Phase dependencies: phase1 -> phase2 (parallel) + phase3 variants (parallel)

set -e

CONFIGS=(
    configs/exp01_resnet101_imagenet100.yaml
    configs/exp01_convnext_small_imagenet100.yaml
    configs/exp01_convnext_base_imagenet100.yaml
    configs/exp01_convnext_large_imagenet100.yaml
    configs/exp01_efficientnet_b5_imagenet100.yaml
    configs/exp01_densenet201_imagenet100.yaml
    configs/exp01_vit_b_16_imagenet100.yaml
    configs/exp01_vit_l_16_imagenet100.yaml
    configs/exp01_vit_h_14_imagenet100.yaml
    configs/exp01_wrn4010_imagenet100.yaml
)

mkdir -p slurm/logs

echo "Submitting Phase I experiments to SLURM..."
echo ""

for CONFIG in "${CONFIGS[@]}"; do
    NAME=$(basename "$CONFIG" .yaml)

    # Phase 1: Training (no dependency)
    J1=$(sbatch --parsable --job-name="${NAME}_p1" \
         slurm/run_experiment.sh "$CONFIG" phase1)
    echo "[$NAME] Phase 1 (train):       Job $J1"

    # Phase 2: Topology (depends on Phase 1)
    J2=$(sbatch --parsable --dependency=afterok:$J1 --job-name="${NAME}_p2" \
         slurm/run_experiment.sh "$CONFIG" phase2)
    echo "[$NAME] Phase 2 (topology):    Job $J2 (after $J1)"

    # Phase 3 naive: Sequential forgetting (depends on Phase 1)
    J3=$(sbatch --parsable --dependency=afterok:$J1 --job-name="${NAME}_p3" \
         slurm/run_experiment.sh "$CONFIG" phase3)
    echo "[$NAME] Phase 3 (naive):       Job $J3 (after $J1)"

    # Phase 3 EWC (depends on Phase 1)
    J3E=$(sbatch --parsable --dependency=afterok:$J1 --job-name="${NAME}_p3e" \
          slurm/run_experiment.sh "$CONFIG" phase3 --ewc)
    echo "[$NAME] Phase 3 (EWC):         Job $J3E (after $J1)"

    # Phase 3 SI (depends on Phase 1)
    J3S=$(sbatch --parsable --dependency=afterok:$J1 --job-name="${NAME}_p3s" \
          slurm/run_experiment.sh "$CONFIG" phase3 --si)
    echo "[$NAME] Phase 3 (SI):          Job $J3S (after $J1)"

    echo "---"
done

echo ""
echo "All jobs submitted. Phase 2 and Phase 3 variants start after Phase 1 completes."
echo "Monitor with: squeue -u \$USER"
