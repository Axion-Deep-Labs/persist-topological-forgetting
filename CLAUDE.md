# Axion Deep Labs — Research Experiments

## Overview
Experimental codebase for research experiments:
- **EXP-01** (PERSIST): Topological Signatures of Knowledge Persistence -- Preliminary COMPLETE, Phase I (HPC scale validation) PLANNED
- **EXP-04** (Grokking Topology): Topological Dynamics of Grokking -- Pilot IN PROGRESS (3 bugs fixed 2026-03-26)
- **EXP-02** (PHI): Integrated Information Across Architectures -- Planned
- **EXP-03** (GENESIS): Bekenstein Bound Analogs -- Planned

## Structure
```
experiments/
  shared/              -- Datasets (CIFAR-100, CUB-200, RESISC-45), models (19 archs), EWC, SI, baseline metrics, utilities
  exp01_.../           -- Phase 1-7 scripts for topological persistence
  exp04_.../           -- Grokking topology (model, dataset, train, topology, baselines, pilot runner, calibration)
  exp02_.../           -- Phi survey (planned)
  exp03_.../           -- Bekenstein analog (planned)
configs/               -- 57 EXP-01 configs + 10 ImageNet-100 configs (8 valid) + exp04_pilot.yaml
results/               -- Output (gitignored, large files)
dashboard/             -- Flask web dashboard (localhost:5050), EXP-01 3-dataset selector
dashboard_exp04/       -- Flask dashboard (localhost:5051), EXP-04 per-seed and cross-seed views
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
cat slurm/logs/<jobid>_<name>.out              # Job output (local repo)
# Full HPC path: /fs1/scratch/cag1145/axiondeep-research/slurm/logs/<jobid>_<name>.{out,err}
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

## EXP-04 Current State (as of 2026-03-26)

### Grokking Topology Pilot

- **Question:** Does PH of loss landscape slices provide an early-warning signal for grokking?
- **Task:** Modular addition (mod 97), 1-layer transformer decoder, d_model=128, 4 heads
- **Training:** AdamW, full-batch, lr=1e-3, weight_decay=0.03, 100K steps
- **Calibration:** WD=0.03 gives ~70K-step grokking delay (optimal for analysis)
- **Dashboard:** `.venv/bin/python dashboard_exp04/app.py` -> http://localhost:5051

### Pilot Status

| Seed | Training | Topology | Baselines | Notes |
|------|----------|----------|-----------|-------|
| 42   | Done     | Done*    | Needs rerun | Onset: step 40K |
| 137  | Done     | Done*    | Needs rerun | Onset: step 42K |
| 256  | Done     | Done*    | Needs rerun | Onset: step 80.5K |
| 1024 | Done     | Pending  | Pending     | |
| 7777 | Not started | --   | --          | |

*Topology data is forward-compatible (new fields added) but should be rerun for consistency.

### Bugs Fixed (2026-03-26) -- All Seeds Need Re-Analysis

1. **H0 feature count structurally constant (topology.py):** On a 50x50 grid, H0 count = n-1 = 2499 always. Added `h0_significant_count` (persistence > median) and `h0_median_persistence`. Primary endpoint changed to `h0_total_persistence`.
2. **Commutator defect always zero (baselines.py):** Two bugs: (a) full-batch training meant only 1 batch, function returned 0. Fixed: synthetic random 50/50 splits of training data. (b) Hessian-vector product lacked `.detach()` on vector argument, making `d(ga.gb)/d(theta)` symmetric -- `Hab == Hba` by construction. Fixed: detach vector arg. After fix: commutator defect = 5414 at step 20K (was 0), shows dynamics peaking pre-grokking.
3. **H1 essentially dead:** 50x50 grid too coarse for 1-cycles. Max H1 count = 1.2. Demoted to exploratory.

### Re-Analysis Command
```bash
# Re-run analysis on existing checkpoints (no retraining needed)
.venv/bin/python -m experiments.exp04_grokking_topology.run_pilot --config configs/exp04_pilot.yaml --skip-training
```

### Pilot Gate (after re-analysis)
Criterion: at least 1 PH stat shows consistent directional behavior in >= 3/5 seeds before grokking onset. If met, proceed to full study (30 seeds x 3 WD values = 90 runs).

---

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
- **Status:** 8/8 valid configs complete through Phases 1-3 (2026-04-01). Phase 4-6 analysis ready to submit.
- **Compute:** NMSU Discovery cluster (A100 40GB PCIe GPUs, unlimited hours via Crystal's NMSU affiliation)
- **Environment:** `/fs1/scratch/cag1145/persist-env` (PyTorch 2.5.1+cu121, Python 3.10.8)
- **Goal:** Test whether topological signal survives at production scale
- **Models (8 valid):** ResNet-101, ConvNeXt-S/B/L, EfficientNet-B5, DenseNet-201, ViT-B/16, ViT-L/16
- **Dropped:** ViT-H/14 (SWAG weights require 518x518 input, incompatible with 224x224 pipeline), WRN-40-10 (CIFAR-32 architecture OOMs at 224x224 on A100 40GB)
- **CL methods:** EWC + SI (both implemented in phase3, `--ewc` and `--si` flags)
- **Pipeline:** `submit_all.sh` automates phase1 → phase2 + phase3 (naive/EWC/SI) with SLURM dependency chains
- **Bugs fixed (2026-04-01):** Phase 4 ARCH_CLASSES missing ImageNet-100 entries + `_imagenet100` suffix strip. Phase 5 dataset detection defaulting to CIFAR-100. Phase 6 generalized from 3 hardcoded datasets to N dynamic datasets with unbalanced architecture support.
- **Next:** Submit Phase 4-6 analysis on HPC, then Phase I-B pre-registered replication if signal holds

## Rules
- NEVER commit data/ or results/ directories
- All experiments must be reproducible via seed in config (landscape seed is randomized but logged in topology_summary.json)
- Save all hyperparameters in config files, not hardcoded
- ClearML is disabled (`CLEARML_OFF=1`) — use dashboard instead
- Baseline metrics (Hessian, Fisher, sharpness, barrier) computed alongside topology in Phase 2 (fail-safe: individual metric failures don't block results)
- Update EXPERIMENT_LOG.md after each completed run
- **ALWAYS build resume support into any long-running pipeline.** Save results incrementally after each step, load existing results on startup, skip completed steps. Call `gc.collect()` and `torch.cuda.empty_cache()` after each step. Use atomic writes (write to .tmp then os.replace). A killed process should lose at most one step of work, never the entire run.

## Vocab Lesson Plan
Joshua is studying for an advanced AI/ML engineering exam. Vocabulary lesson plan with 10 words/day is maintained in `~/CLAUDE.md`. Current day tracked there. When Joshua asks to study or continue vocab, FIRST teach the words with explanations and analogies, THEN quiz him by asking him to define each word in his own words. Correct misconceptions. Only advance to the next day when he says he's ready.
