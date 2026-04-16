#!/bin/bash
# One-off: submit the remaining resisc45 -> cub200 cross-dataset jobs that
# submit_cross_dataset.sh can't fill safely while its earlier jobs are still queued.
#
# Why this exists: submit_cross_dataset.sh decides "done" by looking for
# forgetting_curve.json on disk. A job that is queued but not yet complete has
# no curve file, so every re-invocation re-submits it — duplicating work and
# eating slots under the 20-job QOSMaxSubmitJobPerUserLimit cap on Discovery.
# This script is queue-aware: it skips configs that already have a matching
# job name in squeue.
#
# Usage: bash slurm/submit_missing_xd.sh
# Re-run after slots free up; safe to invoke repeatedly.

set -uo pipefail  # intentionally no -e: we want to keep iterating past QOS cap errors

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
SLURM_LOG_DIR="${REPO_ROOT}/slurm/logs"
mkdir -p "$SLURM_LOG_DIR"

# (base_config_stem : arch_short : method) for the 11 genuinely-missing jobs
# Direction is resisc45 -> cub200 for all of them.
JOBS=(
    "exp01_densenet121:densenet121:naive"
    "exp01_densenet121:densenet121:ewc"
    "exp01:resnet18:naive"
    "exp01:resnet18:ewc"
    "exp01_mobilenetv3:mobilenetv3:naive"
    "exp01_mobilenetv3:mobilenetv3:ewc"
    "exp01_wrn281:wrn281:naive"
    "exp01_wrn281:wrn281:ewc"
    "exp01_wrn284:wrn284:naive"
    "exp01_wrn284:wrn284:ewc"
    "exp01_mlpmixer:mlpmixer:ewc"
)

submitted=0
skipped=0
capped=false

for entry in "${JOBS[@]}"; do
    IFS=: read -r base arch_short method <<< "$entry"

    task_a_dir="${RESULTS_DIR}/${base}_resisc45"
    xd_dir="${RESULTS_DIR}/${base}_resisc45_xd_cub200"
    config_path="configs/${base}_resisc45.yaml"

    curve_subdir="forgetting"
    [[ "$method" == "ewc" ]] && curve_subdir="forgetting_ewc"
    job_name="xd_${arch_short}_resisc45_cub200_${method}"

    if [[ -f "${xd_dir}/${curve_subdir}/forgetting_curve.json" ]]; then
        echo "SKIP ${job_name}: forgetting_curve.json already present"
        skipped=$((skipped+1))
        continue
    fi

    if squeue -u "$USER" -h -o '%j' | grep -qx "$job_name"; then
        echo "SKIP ${job_name}: already in queue"
        skipped=$((skipped+1))
        continue
    fi

    if [[ ! -f "${task_a_dir}/checkpoints/task_a_best.pt" ]]; then
        echo "SKIP ${job_name}: Task A checkpoint missing at ${task_a_dir}/checkpoints/task_a_best.pt"
        skipped=$((skipped+1))
        continue
    fi

    if [[ ! -f "${REPO_ROOT}/${config_path}" ]]; then
        echo "SKIP ${job_name}: config ${config_path} not found"
        skipped=$((skipped+1))
        continue
    fi

    mkdir -p "${xd_dir}"
    [[ ! -e "${xd_dir}/topology" ]]    && ln -s "${task_a_dir}/topology"    "${xd_dir}/topology"
    [[ ! -e "${xd_dir}/checkpoints" ]] && ln -s "${task_a_dir}/checkpoints" "${xd_dir}/checkpoints"

    extra_flag=""
    [[ "$method" == "ewc" ]] && extra_flag="--ewc"

    cmd="cd ${REPO_ROOT} && source /fs1/scratch/cag1145/persist-env/bin/activate && python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting --config ${config_path} --cross-dataset cub200 --task-a-dir ${task_a_dir} --output-dir-override ${xd_dir} ${extra_flag}"

    out=$(sbatch \
        --job-name="${job_name}" \
        --partition=normal \
        --gres=gpu:a100:1 \
        --mem=16G \
        --time=02:00:00 \
        --output="${SLURM_LOG_DIR}/%j_${job_name}.out" \
        --error="${SLURM_LOG_DIR}/%j_${job_name}.err" \
        --wrap="${cmd}" 2>&1)

    if echo "$out" | grep -q "QOSMaxSubmitJobPerUserLimit"; then
        echo "HIT QOS cap at ${job_name} — stopping. Re-run after queue drains."
        capped=true
        break
    fi

    echo "${out} (${job_name})"
    submitted=$((submitted+1))
done

echo ""
echo "================================"
echo "Submitted this run: ${submitted}"
echo "Skipped:            ${skipped}"
if $capped; then
    echo "Stopped early at QOS cap. Re-run when squeue -u \$USER | wc -l drops."
fi
echo "================================"
