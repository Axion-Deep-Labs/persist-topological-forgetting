#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Multi-seed + Multi-slice batch runner for EXP-01
#
# This script runs the full pipeline (Phase 1→2→3) for:
#   - 3 training seeds (42, 123, 7) per architecture
#   - 5 landscape slices (--run-id 1..5) per seed's checkpoint
#
# Estimated time: ~20-30 min per architecture × 3 seeds = ~10-15 hours total
# Run overnight with: nohup bash scripts/run_multiseed_multislice.sh > run_log.txt 2>&1 &
# ═══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python"
SEEDS=(42 123 7)
SLICES=(1 2 3 4 5)

# All 14 CIFAR-100 configs
CONFIGS=(
    configs/exp01.yaml
    configs/exp01_resnet50.yaml
    configs/exp01_vit.yaml
    configs/exp01_wrn2810.yaml
    configs/exp01_mlpmixer.yaml
    configs/exp01_resnet18wide.yaml
    configs/exp01_densenet121.yaml
    configs/exp01_efficientnet.yaml
    configs/exp01_vgg16bn.yaml
    configs/exp01_convnext.yaml
    configs/exp01_mobilenetv3.yaml
    configs/exp01_vittiny.yaml
    configs/exp01_shufflenet.yaml
    configs/exp01_regnet.yaml
)

echo "═══════════════════════════════════════════════════"
echo "EXP-01 Multi-Seed + Multi-Slice Batch Run"
echo "  Seeds: ${SEEDS[*]}"
echo "  Slices per seed: ${#SLICES[@]}"
echo "  Architectures: ${#CONFIGS[@]}"
echo "  Total Phase 1 runs: $((${#CONFIGS[@]} * ${#SEEDS[@]}))"
echo "  Total Phase 2 runs: $((${#CONFIGS[@]} * ${#SEEDS[@]} * ${#SLICES[@]}))"
echo "  Total Phase 3 runs: $((${#CONFIGS[@]} * ${#SEEDS[@]}))"
echo "═══════════════════════════════════════════════════"
echo ""

for CFG in "${CONFIGS[@]}"; do
    ARCH=$(basename "$CFG" .yaml)
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Architecture: $ARCH"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "  ── Seed $SEED ──"

        # Phase 1: Train Task A
        echo "  [Phase 1] Training Task A (seed=$SEED)..."
        $PYTHON -m experiments.exp01_topological_persistence.phase1_train_task_a \
            --config "$CFG" --seed "$SEED"

        # Phase 2: Landscape topology (5 slices)
        for SLICE in "${SLICES[@]}"; do
            echo "  [Phase 2] Landscape topology (seed=$SEED, slice=$SLICE)..."
            $PYTHON -m experiments.exp01_topological_persistence.phase2_landscape_topology \
                --config "$CFG" --seed "$SEED" --run-id "$SLICE"
        done

        # Phase 3: Sequential forgetting
        echo "  [Phase 3] Sequential forgetting (seed=$SEED)..."
        $PYTHON -m experiments.exp01_topological_persistence.phase3_sequential_forgetting \
            --config "$CFG" --seed "$SEED"

        echo "  ✓ Seed $SEED complete for $ARCH"
    done

    echo "✓ $ARCH complete (all seeds)"
done

echo ""
echo "═══════════════════════════════════════════════════"
echo "All runs complete!"
echo "Next: Run Phase 4 correlation with multi-seed/multi-slice aggregation"
echo "═══════════════════════════════════════════════════"
