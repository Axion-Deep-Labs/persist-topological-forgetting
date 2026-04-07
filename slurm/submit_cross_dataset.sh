#!/bin/bash
# Phase I-B: Cross-dataset forgetting experiments
# Submits naive + EWC forgetting for all 19 architectures across 6 cross-dataset pairs.
#
# Each pair: Task A model (existing checkpoint) -> Task B (different dataset)
# Topology and checkpoints are symlinked from original Task A result dirs.
#
# Usage: bash slurm/submit_cross_dataset.sh [--dry-run]
#
# Total: 6 pairs x 19 archs x 2 methods (naive + EWC) = 228 jobs

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
SLURM_LOG_DIR="${REPO_ROOT}/slurm/logs"
mkdir -p "$SLURM_LOG_DIR"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

# Architecture base names (no dataset suffix = cifar100)
# Maps: config_file -> base_result_dir_name
declare -A ARCH_CONFIGS
ARCH_CONFIGS=(
    ["exp01.yaml"]="exp01"
    ["exp01_resnet50.yaml"]="exp01_resnet50"
    ["exp01_vit.yaml"]="exp01_vit"
    ["exp01_vittiny.yaml"]="exp01_vittiny"
    ["exp01_densenet121.yaml"]="exp01_densenet121"
    ["exp01_efficientnet.yaml"]="exp01_efficientnet"
    ["exp01_mobilenetv3.yaml"]="exp01_mobilenetv3"
    ["exp01_shufflenet.yaml"]="exp01_shufflenet"
    ["exp01_regnet.yaml"]="exp01_regnet"
    ["exp01_convnext.yaml"]="exp01_convnext"
    ["exp01_vgg16bn.yaml"]="exp01_vgg16bn"
    ["exp01_mlpmixer.yaml"]="exp01_mlpmixer"
    ["exp01_wrn281.yaml"]="exp01_wrn281"
    ["exp01_wrn282.yaml"]="exp01_wrn282"
    ["exp01_wrn284.yaml"]="exp01_wrn284"
    ["exp01_wrn286.yaml"]="exp01_wrn286"
    ["exp01_wrn288.yaml"]="exp01_wrn288"
    ["exp01_wrn2810.yaml"]="exp01_wrn2810"
    ["exp01_resnet18wide.yaml"]="exp01_resnet18wide"
)

# Dataset suffix mapping (cifar100 has no suffix in dir names)
declare -A DATASET_SUFFIXES
DATASET_SUFFIXES=(
    ["cifar100"]=""
    ["cub200"]="_cub200"
    ["resisc45"]="_resisc45"
)

# Cross-dataset pairs: task_a_dataset -> task_b_dataset
TASK_A_DATASETS=("cifar100" "cifar100" "cub200" "cub200" "resisc45" "resisc45")
TASK_B_DATASETS=("cub200" "resisc45" "cifar100" "resisc45" "cifar100" "cub200")

submitted=0
skipped=0

for pair_idx in "${!TASK_A_DATASETS[@]}"; do
    task_a="${TASK_A_DATASETS[$pair_idx]}"
    task_b="${TASK_B_DATASETS[$pair_idx]}"
    task_a_suffix="${DATASET_SUFFIXES[$task_a]}"

    echo ""
    echo "=== Cross-dataset pair: ${task_a} -> ${task_b} ==="

    for config in "${!ARCH_CONFIGS[@]}"; do
        base_name="${ARCH_CONFIGS[$config]}"

        # Task A result directory (where checkpoint and topology live)
        task_a_dir="${RESULTS_DIR}/${base_name}${task_a_suffix}"

        # Cross-dataset output directory
        xd_dir="${RESULTS_DIR}/${base_name}${task_a_suffix}_xd_${task_b}"

        # Config file path (Task A config, with dataset suffix variant)
        if [[ "$task_a_suffix" == "" ]]; then
            config_path="configs/${config}"
        else
            # e.g., exp01_resnet50.yaml -> exp01_resnet50_cub200.yaml
            config_path="configs/${config%.yaml}${task_a_suffix}.yaml"
        fi

        # Verify Task A dir exists with checkpoint
        if [[ ! -f "${task_a_dir}/checkpoints/task_a_best.pt" ]]; then
            echo "  SKIP ${base_name}${task_a_suffix}: no Task A checkpoint"
            skipped=$((skipped + 1))
            continue
        fi

        # Verify config exists
        if [[ ! -f "${REPO_ROOT}/${config_path}" ]]; then
            echo "  SKIP ${base_name}${task_a_suffix}: config ${config_path} not found"
            skipped=$((skipped + 1))
            continue
        fi

        # Create output directory and symlinks
        mkdir -p "${xd_dir}"
        # Symlink topology and checkpoints from Task A dir
        if [[ ! -L "${xd_dir}/topology" ]] && [[ ! -d "${xd_dir}/topology" ]]; then
            ln -s "${task_a_dir}/topology" "${xd_dir}/topology"
        fi
        if [[ ! -L "${xd_dir}/checkpoints" ]] && [[ ! -d "${xd_dir}/checkpoints" ]]; then
            ln -s "${task_a_dir}/checkpoints" "${xd_dir}/checkpoints"
        fi

        # Skip if both naive and EWC already done
        naive_done=false
        ewc_done=false
        [[ -f "${xd_dir}/forgetting/forgetting_curve.json" ]] && naive_done=true
        [[ -f "${xd_dir}/forgetting_ewc/forgetting_curve.json" ]] && ewc_done=true

        if $naive_done && $ewc_done; then
            echo "  DONE ${base_name}${task_a_suffix} -> ${task_b}: both naive and EWC complete"
            skipped=$((skipped + 2))
            continue
        fi

        arch_short="${base_name#exp01_}"
        [[ "$arch_short" == "exp01" ]] && arch_short="resnet18"
        job_name="xd_${arch_short}_${task_a}_${task_b}"

        # Submit naive job
        if ! $naive_done; then
            naive_cmd="cd ${REPO_ROOT} && source /fs1/scratch/cag1145/persist-env/bin/activate && python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting --config ${config_path} --cross-dataset ${task_b} --task-a-dir ${task_a_dir} --output-dir-override ${xd_dir}"

            if $DRY_RUN; then
                echo "  [DRY] naive: $job_name"
            else
                sbatch \
                    --job-name="${job_name}_naive" \
                    --partition=normal \
                    --gres=gpu:a100:1 \
                    --mem=16G \
                    --time=02:00:00 \
                    --output="${SLURM_LOG_DIR}/%j_${job_name}_naive.out" \
                    --error="${SLURM_LOG_DIR}/%j_${job_name}_naive.err" \
                    --wrap="${naive_cmd}"
                echo "  Submitted naive: $job_name"
                submitted=$((submitted + 1))
            fi
        fi

        # Submit EWC job
        if ! $ewc_done; then
            ewc_cmd="cd ${REPO_ROOT} && source /fs1/scratch/cag1145/persist-env/bin/activate && python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting --config ${config_path} --cross-dataset ${task_b} --task-a-dir ${task_a_dir} --output-dir-override ${xd_dir} --ewc"

            if $DRY_RUN; then
                echo "  [DRY] ewc: $job_name"
            else
                sbatch \
                    --job-name="${job_name}_ewc" \
                    --partition=normal \
                    --gres=gpu:a100:1 \
                    --mem=16G \
                    --time=02:00:00 \
                    --output="${SLURM_LOG_DIR}/%j_${job_name}_ewc.out" \
                    --error="${SLURM_LOG_DIR}/%j_${job_name}_ewc.err" \
                    --wrap="${ewc_cmd}"
                echo "  Submitted ewc: $job_name"
                submitted=$((submitted + 1))
            fi
        fi
    done
done

echo ""
echo "================================"
echo "Submitted: ${submitted} jobs"
echo "Skipped:   ${skipped}"
echo "================================"
