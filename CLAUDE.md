# Axion Deep Labs — Research Experiments

## Overview
Experimental codebase for three priority research experiments:
- **EXP-01** (PERSIST): Topological Signatures of Knowledge Persistence — ACTIVE
- **EXP-02** (PHI): Integrated Information Across Architectures — Planned
- **EXP-03** (GENESIS): Bekenstein Bound Analogs — Planned

## Structure
```
experiments/
  shared/              — Datasets (CIFAR-100, CUB-200, RESISC-45), models (19 archs), EWC, baseline metrics, utilities
  exp01_.../           — Phase 1-5 scripts for topological persistence
  exp02_.../           — Phi survey (planned)
  exp03_.../           — Bekenstein analog (planned)
configs/               — 57 YAML configs (19 architectures x 3 datasets)
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

## EXP-01 Current State (as of 2026-02-21)
- **3 datasets:** CIFAR-100, CUB-200-2011 (fine-grained birds), NWPU-RESISC45 (satellite scenes)
- **19 architectures:** 14 original + WRN-28-k width ladder (k=1,2,4,6,8,10)
- **57 configs total** (19 per dataset)
- **Full pipeline:** Phase 1 (train) -> Phase 2 (5 Ripser slices) -> Phase 2c (cubical PH) -> Phase 3 (naive + EWC + cosine) -> Phase 4 (correlation + diagnostics) -> Phase 5 (predictive model + permutation test)
- CIFAR-100: 14 original architectures complete (Phases 1-3 naive), WRN width ladder partial
- CUB-200 and RESISC-45: all 19 architectures pending
- Phase 4: slice robustness diagnostics, cubical vs Ripser comparison, EWC benefit analysis, WRN ladder analysis
- Phase 5: LOAO CV predictive model with 5 models (A/A2/B/C/D), permutation test, matched-dimensionality control
- Dashboard: 3-dataset selector, "Run All Datasets" button, "Run Predictive" button
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
