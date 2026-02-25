# Axion Deep Labs — Research Experiments

## Overview
Experimental codebase for three priority research experiments:
- **EXP-01** (PERSIST): Topological Signatures of Knowledge Persistence — Preliminary COMPLETE, Phase I (HPC scale validation) PLANNED
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

## EXP-01 Current State (as of 2026-02-25)

### Preliminary Proof-of-Concept (COMPLETE — the "petri dish")
- **3 datasets:** CIFAR-100, CUB-200-2011 (fine-grained birds), NWPU-RESISC45 (satellite scenes)
- **19 architectures:** 14 original + WRN-28-k width ladder (k=1,2,4,6,8,10), all under 45M params
- **57/57 configs complete** (19 per dataset, all Phases 1-6)
- **Full pipeline:** Phase 1 (train) -> Phase 2 (5 Ripser slices) -> Phase 2c (cubical PH) -> Phase 3 (naive + EWC + cosine) -> Phase 4 (correlation + diagnostics) -> Phase 5 (predictive model + permutation test) -> Phase 6 (pooled interaction + clustered bootstrap)
- CIFAR-100: 19/19 complete, params dominate (rho=-0.76), topology redundant
- CUB-200: 19/19 complete, topology rescues prediction (p=0.037 suggestive, does not survive Bonferroni)
- RESISC-45: 19/19 complete, topology does not help (p=0.566), but H0 predicts EWC benefit (rho=0.86, p=2.4e-6)
- Dashboard: 3-dataset selector, "Run All Datasets" button, "Run Predictive" button
- See EXPERIMENT_LOG.md for full run history and results

### Phase I: Scale Validation (PLANNED — requires supercomputer)
- **Goal:** Test whether topological signal survives at production scale
- **Models:** 100M-7B+ parameters (ViT-Large, foundation models, LLMs)
- **Datasets:** ImageNet, NLP tasks, medical imaging
- **Task sequences:** 10-100+ sequential tasks (vs current 2-task)
- **CL methods:** SI, PackNet, replay, adapters (vs current EWC only)
- **Architectures:** 50-100+ (vs current 19) for statistical power
- **Compute:** Requires NSF ACCESS or equivalent supercomputer allocation
- **Key risks:** Signal may vanish at scale; PH computation may be intractable; subsampling may lose fidelity

## Rules
- NEVER commit data/ or results/ directories
- All experiments must be reproducible via seed in config (landscape seed is randomized but logged in topology_summary.json)
- Save all hyperparameters in config files, not hardcoded
- ClearML is disabled (`CLEARML_OFF=1`) — use dashboard instead
- Baseline metrics (Hessian, Fisher, sharpness, barrier) computed alongside topology in Phase 2 (fail-safe: individual metric failures don't block results)
- Update EXPERIMENT_LOG.md after each completed run

## Vocab Lesson Plan
Joshua is studying for an advanced AI/ML engineering exam. Vocabulary lesson plan with 10 words/day is maintained in `~/CLAUDE.md`. Current day tracked there. When Joshua asks to study or continue vocab, FIRST teach the words with explanations and analogies, THEN quiz him by asking him to define each word in his own words. Correct misconceptions. Only advance to the next day when he says he's ready.
