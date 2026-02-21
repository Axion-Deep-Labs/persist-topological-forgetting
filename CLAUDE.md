# Axion Deep Labs — Research Experiments

## Overview
Experimental codebase for three priority research experiments:
- **EXP-01** (PERSIST): Topological Signatures of Knowledge Persistence — ACTIVE
- **EXP-02** (PHI): Integrated Information Across Architectures — Planned
- **EXP-03** (GENESIS): Bekenstein Bound Analogs — Planned

## Structure
```
experiments/
  shared/              — Datasets (CIFAR-100, CUB-200, RESISC-45), models (19 archs), baseline metrics, utilities
  exp01_.../           — Phase 1-4 scripts for topological persistence
  exp02_.../           — Phi survey (planned)
  exp03_.../           — Bekenstein analog (planned)
configs/               — 57 YAML configs (19 architectures × 3 datasets)
results/               — Output (gitignored, large files)
dashboard/             — Flask web dashboard (localhost:5050), 3-dataset selector
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

## EXP-01 Current State (as of 2026-02-20)
- **3 datasets:** CIFAR-100, CUB-200-2011 (fine-grained birds), NWPU-RESISC45 (satellite scenes)
- **19 architectures:** 14 original + WRN-28-k width ladder (k=1,2,4,6,8,10)
- **57 configs total** (19 per dataset)
- CIFAR-100: 14 original architectures complete (Phases 1-3), 5 WRN width ladder pending
- CUB-200 and RESISC-45: all 19 architectures pending (new datasets)
- CIFAR-10 removed (floor effect, no statistical power)
- Phase 4 includes WRN Width Ladder Analysis (within-ladder Spearman + partial H1|params)
- Dashboard: 3-dataset selector, "Run All Datasets" button
- See EXPERIMENT_LOG.md for full run history and results

## Rules
- NEVER commit data/ or results/ directories
- All experiments must be reproducible via seed in config (landscape seed is randomized but logged in topology_summary.json)
- Save all hyperparameters in config files, not hardcoded
- ClearML is disabled (`CLEARML_OFF=1`) — use dashboard instead
- Baseline metrics (Hessian, Fisher, sharpness, barrier) computed alongside topology in Phase 2 (fail-safe: individual metric failures don't block results)
- Update EXPERIMENT_LOG.md after each completed run

## Vocab Lesson Plan
Joshua is studying for an advanced AI/ML engineering exam. Vocabulary lesson plan with 10 words/day is maintained in `~/CLAUDE.md`. Current day tracked there. When Joshua asks to study or continue vocab, FIRST teach the words with explanations and analogies, THEN quiz him by asking him to define each word in his own words. Correct misconceptions. Only advance to the next day when he says he's ready.
