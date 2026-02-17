# Axion Deep Labs — Research Experiments

## Overview
Experimental codebase for three priority research experiments:
- **EXP-01** (PERSIST): Topological Signatures of Knowledge Persistence — ACTIVE
- **EXP-02** (PHI): Integrated Information Across Architectures — Planned
- **EXP-03** (GENESIS): Bekenstein Bound Analogs — Planned

## Structure
```
experiments/
  shared/              — Datasets, models, baseline metrics, utilities
  exp01_.../           — Phase 1-4 scripts for topological persistence
  exp02_.../           — Phi survey (planned)
  exp03_.../           — Bekenstein analog (planned)
configs/               — YAML configs per experiment (one per architecture)
results/               — Output (gitignored, large files)
dashboard/             — Flask web dashboard (localhost:5050)
```

## Running Experiments

### Dashboard (Recommended)
```bash
.venv/bin/python dashboard/app.py
# Open http://localhost:5050
```
Features: experiment queue, GPU/CPU/RAM monitor, live output, pause/resume/stop.

### Manual
```bash
python -m experiments.exp01_topological_persistence.phase1_train_task_a --config configs/exp01.yaml
```

## EXP-01 Current State (as of 2025-02-10)
- 3 architectures complete (ResNet-18, ResNet-50, ViT-Small)
- WRN-28-10 in progress (Phase 3)
- 4 more queued (MLP-Mixer, ResNet-18 Wide, DenseNet-121, EfficientNet-B0)
- Preliminary Spearman ρ = 0.866 (H0 persistence vs forgetting resistance)
- Need n≥5 for statistical significance
- See EXPERIMENT_LOG.md for full run history and results

## Rules
- NEVER commit data/ or results/ directories
- All experiments must be reproducible via seed in config
- Save all hyperparameters in config files, not hardcoded
- ClearML is disabled (`CLEARML_OFF=1`) — use dashboard instead
- Baseline metrics (Hessian, Fisher, sharpness, barrier) must be computed alongside topology in Phase 2
- Update EXPERIMENT_LOG.md after each completed run
