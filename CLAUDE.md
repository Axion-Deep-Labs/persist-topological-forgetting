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

## EXP-01 Current State (as of 2026-02-20)
- All 14 architectures complete on CIFAR-100 (Phases 1, 2 ×5 slices, 3)
- CIFAR-10: 6/14 complete, 7 remaining (resume via dashboard)
- Phase 4 correlation (n=14, CIFAR-100): H1 ρ=0.61 (p=0.021, p_Bonf=0.21), params ρ=−0.74 (p=0.002, p_Bonf=0.02)
- Bonferroni correction + Kendall's tau added to Phase 4
- Landscape validation (NaN/Inf checks) added to Phase 2
- Task B learning validation added to Phase 3
- Phase 2b multi-slice fallback support added
- Dashboard: 5 landscape slices, Clean & Rebuild, Re-run All P3
- **Note:** Phase 2 seed bug fixed — existing multi-slice runs need re-run
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
