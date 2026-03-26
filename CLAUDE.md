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

### Local Dashboard (Recommended for monitoring)
```bash
.venv/bin/python dashboard/app.py
# Open http://localhost:5050
```
Features: experiment queue, GPU/CPU/RAM monitor, live output, pause/resume/stop.

### Local Manual
```bash
python -m experiments.exp01_topological_persistence.phase1_train_task_a --config configs/exp01.yaml
```

### NMSU Discovery HPC (Phase I)

**Connection:**
1. Connect VPN: launch Cisco Secure Client → `vpn.nmsu.edu` → Crystal's NMSU credentials + Duo MFA
2. SSH: `ssh cag1145@discovery.nmsu.edu`

**Environment setup (once per session):**
```bash
source /fs1/scratch/cag1145/persist-env/bin/activate
```
Note: Module loads (`os/rhel_8`, `spack/2023a`, etc.) fail on compute nodes but the venv has all dependencies self-contained. PyTorch+CUDA work without module loads.

**Submit single experiment:**
```bash
sbatch slurm/run_experiment.sh configs/exp01_vit_b_16_imagenet100.yaml phase1
```

**Submit all Phase I ImageNet-100 experiments:**
```bash
bash slurm/submit_all.sh
```

**Monitor:**
```bash
squeue -u cag1145                              # Job status
cat slurm/logs/<jobid>_<name>.out              # Job output
sacct -j <jobid> --format=JobID,Elapsed,State  # Completed job info
```

**Cluster details:**
- GPUs: NVIDIA A100-PCIE-40GB (on discovery-g* nodes)
- Partitions: `normal` (7-day limit), `backfill` (14-day), `interactive` (1-day)
- Storage: 100GB home (`/fs1/home/cag1145`), 1TB scratch (`/fs1/scratch/cag1145`)
- No compute hour quota (unlimited under `nmsu` account)

**Lessons learned (2026-03-23):**
- VPN requires Cisco Secure Client (not GlobalProtect); `openconnect` fails due to SAML/SSO + Duo MFA
- Cisco Secure Client installer: `/home/joshua/Corporate/axiondeep-research/cisco-secure-client-linux64-5.1.10.233-core-vpn-webdeploy-k9.sh`
- Module load hierarchy: `os/rhel_8` → `spack/2023a` → `gcc/12.2.0` → then python/cuda. But compute nodes don't need this — venv is self-contained.
- SLURM partition is `normal` (not `gpu`) — GPU nodes are `discovery-g*` within the normal partition, requested via `--gres=gpu:1`

## EXP-01 Current State (as of 2026-03-23)

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

### Phase I: Scale Validation (IN PROGRESS — NMSU Discovery HPC)
- **Status:** HPC environment set up and GPU verified (2026-03-23)
- **Compute:** NMSU Discovery cluster (A100 40GB PCIe GPUs, unlimited hours via Crystal's NMSU affiliation)
- **Environment:** `/fs1/scratch/cag1145/persist-env` (PyTorch 2.5.1+cu121, Python 3.10.8)
- **Goal:** Test whether topological signal survives at production scale
- **Models:** 10 ImageNet-100 configs ready (ResNet-101, ConvNeXt-S/B/L, EfficientNet-B5, DenseNet-201, ViT-B/L/H, WRN-40-10)
- **CL methods:** EWC + SI (both implemented in phase3, `--ewc` and `--si` flags)
- **Pipeline:** `submit_all.sh` automates phase1 → phase2 + phase3 (naive/EWC/SI) with SLURM dependency chains
- **Key risks:** Signal may vanish at scale; PH computation may be intractable; subsampling may lose fidelity
- **Next:** Clone repo to cluster, adapt SLURM scripts for Discovery, download ImageNet-100, submit first batch

## Rules
- NEVER commit data/ or results/ directories
- All experiments must be reproducible via seed in config (landscape seed is randomized but logged in topology_summary.json)
- Save all hyperparameters in config files, not hardcoded
- ClearML is disabled (`CLEARML_OFF=1`) — use dashboard instead
- Baseline metrics (Hessian, Fisher, sharpness, barrier) computed alongside topology in Phase 2 (fail-safe: individual metric failures don't block results)
- Update EXPERIMENT_LOG.md after each completed run

## Vocab Lesson Plan
Joshua is studying for an advanced AI/ML engineering exam. Vocabulary lesson plan with 10 words/day is maintained in `~/CLAUDE.md`. Current day tracked there. When Joshua asks to study or continue vocab, FIRST teach the words with explanations and analogies, THEN quiz him by asking him to define each word in his own words. Correct misconceptions. Only advance to the next day when he says he's ready.
