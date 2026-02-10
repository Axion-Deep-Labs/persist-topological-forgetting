# Axion Deep Labs — Research Experiments

## Overview
Experimental codebase for three priority research experiments:
- **EXP-01** (PERSIST): Topological Signatures of Knowledge Persistence
- **EXP-02** (PHI): Integrated Information Across Architectures
- **EXP-03** (GENESIS): Bekenstein Bound Analogs

## Structure
```
experiments/
  shared/          — Datasets, models, utilities
  exp01_.../       — Phase 1-4 scripts for topological persistence
  exp02_.../       — Phi survey (planned)
  exp03_.../       — Bekenstein analog (planned)
configs/           — YAML configs per experiment
results/           — Output (gitignored, large files)
```

## Running Experiments
All scripts run from project root:
```bash
python -m experiments.exp01_topological_persistence.phase1_train_task_a --config configs/exp01.yaml
```

## Rules
- NEVER commit data/ or results/ directories
- All experiments must be reproducible via seed in config
- Save all hyperparameters in config files, not hardcoded
