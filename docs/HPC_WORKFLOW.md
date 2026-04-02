# PERSIST Phase I-A: NMSU Discovery HPC Workflow

> Last updated: 2026-04-01

## Overview

8 valid ImageNet-100 experiments across architectures from 20M to 304M parameters.
Each config runs 5 SLURM jobs (phase1 training, phase2 topology, phase3 naive/EWC/SI).

**Status (2026-04-01):** All 8 configs complete through Phases 1-3. Phase 4-6 analysis ready to submit.

**Dropped configs:**
- ViT-H/14 (632M): SWAG weights require 518x518 input, incompatible with 224x224 pipeline
- WRN-40-10 (56M): CIFAR-32 architecture, OOMs at 224x224 on A100 40GB

---

## Prerequisites

Complete these before touching the cluster.

### 1. ImageNet Access (Crystal)

Register at https://image-net.org/download-images.php with NMSU credentials.
Request access to ILSVRC2012 (ImageNet-1K). Academic approval is typically 1-2 business days.
Once approved, you get download links for:
- `ILSVRC2012_img_train.tar` (~138GB)
- `ILSVRC2012_img_val.tar` (~6.3GB)
- `ILSVRC2012_devkit_t12.tar.gz` (not needed)

### 2. VPN

Install Cisco Secure Client. Endpoint: `vpn.nmsu.edu`.
Auth: Crystal's NMSU credentials + Duo MFA.
Installer on local machine: `~/Corporate/axiondeep-research/cisco-secure-client-linux64-5.1.10.233-core-vpn-webdeploy-k9.sh`

### 3. SSH Access

```
ssh cag1145@discovery.nmsu.edu
```

---

## One-Time Cluster Setup

Run these once on the cluster login node.

### Clone the repo

```bash
cd /fs1/scratch/cag1145
git clone https://github.com/Axion-Deep-Labs/persist-topological-forgetting.git axiondeep-research
cd axiondeep-research
mkdir -p slurm/logs data/imagenet
```

### Verify the environment

```bash
source /fs1/scratch/cag1145/persist-env/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.5.1 True
python -c "import ripser, gudhi; print('topology OK')"
ls configs/exp01_*_imagenet100.yaml | wc -l
# Expected: 10
```

### Check storage

```bash
df -h /fs1/scratch/cag1145
```

Current quota is 1TB. ImageNet-1K uses ~155GB. Results for 10 configs use ~5-10GB. Plenty of room.
The quota increase Amy mentioned is only needed later for ImageNet-21K (1.2TB).

---

## Data: Getting ImageNet onto the Cluster

Two options. Option A is faster if Crystal has download links.

### Option A: Download directly on the cluster (recommended)

University network is fast. Download on the login node, not a compute node.

```bash
cd /fs1/scratch/cag1145/axiondeep-research/data/imagenet

# Download (replace URLs with actual links from image-net.org)
wget <ILSVRC2012_img_train_URL> -O ILSVRC2012_img_train.tar
wget <ILSVRC2012_img_val_URL> -O ILSVRC2012_img_val.tar

# Extract training set (creates train/ with 1000 class folders)
mkdir -p train && tar -xf ILSVRC2012_img_train.tar -C train/

# Each class folder is itself a tar. Extract them all.
cd train
for f in *.tar; do
    dir="${f%.tar}"
    mkdir -p "$dir"
    tar -xf "$f" -C "$dir"
    rm "$f"
done
cd ..

# Extract validation set
mkdir -p val && tar -xf ILSVRC2012_img_val.tar -C val/

# Validation images need to be organized into class folders.
# Use the standard reorganization script:
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
cd val && bash ../valprep.sh && cd ..

# Clean up tars (optional, saves ~144GB)
rm -f ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar
```

### Option B: Transfer from local machine

If you download to your local machine first:

```bash
# From your local machine (not the cluster):
rsync -avP /path/to/imagenet/train/ cag1145@discovery.nmsu.edu:/fs1/scratch/cag1145/axiondeep-research/data/imagenet/train/
rsync -avP /path/to/imagenet/val/ cag1145@discovery.nmsu.edu:/fs1/scratch/cag1145/axiondeep-research/data/imagenet/val/
```

This is slower (limited by home upload speed). Use Option A if possible.

### Verify data

```bash
ls /fs1/scratch/cag1145/axiondeep-research/data/imagenet/train/ | wc -l
# Expected: 1000

ls /fs1/scratch/cag1145/axiondeep-research/data/imagenet/val/ | wc -l
# Expected: 1000

ls /fs1/scratch/cag1145/axiondeep-research/data/imagenet/train/n01440764/ | head -5
# Should show .JPEG files
```

---

## Pre-Flight Check

Run the verification script before submitting jobs:

```bash
cd /fs1/scratch/cag1145/axiondeep-research
source /fs1/scratch/cag1145/persist-env/bin/activate
python scripts/verify_hpc_setup.py
```

This checks: Python environment, GPU access, all 10 configs present, ImageNet directory structure, scratch space.

---

## Submit All Experiments

One command launches all 50 jobs with dependency chains:

```bash
cd /fs1/scratch/cag1145/axiondeep-research
source /fs1/scratch/cag1145/persist-env/bin/activate
export CLEARML_OFF=1
export PYTHONUNBUFFERED=1
bash slurm/submit_all.sh
```

### What happens

For each of the 10 configs, `submit_all.sh` submits:

| Job | Phase | Depends On | Duration | What It Does |
|-----|-------|-----------|----------|--------------|
| J1 | phase1 | nothing | 8-24h | Fine-tune pretrained model on Task A (50 classes) |
| J2 | phase2 | J1 | 4-8h | Loss landscape topology (5 random 2D slices, persistent homology) |
| J3 | phase3 | J1 | 8-12h | Sequential forgetting, naive (no regularization) |
| J4 | phase3 --ewc | J1 | 8-12h | Sequential forgetting with EWC |
| J5 | phase3 --si | J1 | 8-12h | Sequential forgetting with SI |

J2-J5 all start in parallel once J1 finishes. SLURM handles the scheduling.

### Submit a single config (for testing)

```bash
sbatch slurm/run_experiment.sh configs/exp01_resnet101_imagenet100.yaml phase1
```

---

## Monitoring

```bash
# Active jobs
squeue -u cag1145

# Detailed job list
squeue -u cag1145 -l

# Live output from a running job
tail -f slurm/logs/<JOBID>_<NAME>.out

# Job history (completed/failed)
sacct -u cag1145 --format=JobID,JobName%30,Elapsed,State,ExitCode -S 2026-03-27

# GPU usage (from within a running job's node)
srun --jobid=<JOBID> nvidia-smi
```

---

## Troubleshooting

### Job stuck in PENDING

```bash
squeue -u cag1145 -l  # check NODELIST(REASON) column
```

- `(Priority)` or `(Resources)`: Normal queue wait. Discovery GPU nodes are shared.
- `(Dependency)`: Waiting for predecessor job. Check if J1 is still running.
- `(DependencyNeverSatisfied)`: Predecessor failed. Check its logs, fix, resubmit.

### Job fails immediately

```bash
cat slurm/logs/<JOBID>_<NAME>.err
```

Common causes:
- `FileNotFoundError: ImageNet train dir not found` -- data not in the right place
- `CUDA out of memory` -- model too large for 40GB A100. Try reducing batch_size in the config.
- `ModuleNotFoundError` -- venv not activated. The `run_experiment.sh` activates it automatically, but verify path.

### Dependency chain broken

If phase1 fails, all dependent jobs get `DependencyNeverSatisfied`. Fix the issue, then:

```bash
# Cancel the stuck jobs
scancel <J2> <J3> <J4> <J5>

# Resubmit just that config manually
CONFIG=configs/exp01_resnet101_imagenet100.yaml
J1=$(sbatch --parsable slurm/run_experiment.sh $CONFIG phase1)
sbatch --dependency=afterok:$J1 slurm/run_experiment.sh $CONFIG phase2
sbatch --dependency=afterok:$J1 slurm/run_experiment.sh $CONFIG phase3
sbatch --dependency=afterok:$J1 slurm/run_experiment.sh $CONFIG phase3 --ewc
sbatch --dependency=afterok:$J1 slurm/run_experiment.sh $CONFIG phase3 --si
```

### Cancel everything

```bash
scancel -u cag1145
```

---

## SLURM Resource Allocation

From `slurm/run_experiment.sh`:

| Resource | Value | Notes |
|----------|-------|-------|
| Partition | normal | 7-day max wall time |
| GPUs | 1x A100-PCIE-40GB | via `--gres=gpu:1` |
| CPUs | 8 | Data loading workers |
| Memory | 64GB | Sufficient for all 10 architectures |
| Wall time | 24h | Per job. Phases 2-3 typically finish in 4-12h. |

ViT-H (632M params) may need more memory or gradient accumulation. If it OOMs, increase `grad_accum_steps` in the config or reduce `batch_size` from 64 to 32.

---

## The 10 Architectures

| Config | Architecture | Params | Notes |
|--------|-------------|--------|-------|
| exp01_resnet101_imagenet100 | ResNet-101 | 44M | Standard deep residual |
| exp01_convnext_small_imagenet100 | ConvNeXt-Small | 50M | Modern CNN |
| exp01_convnext_base_imagenet100 | ConvNeXt-Base | 89M | |
| exp01_convnext_large_imagenet100 | ConvNeXt-Large | 198M | |
| exp01_efficientnet_b5_imagenet100 | EfficientNet-B5 | 30M | Compound scaling |
| exp01_densenet201_imagenet100 | DenseNet-201 | 20M | Dense connections |
| exp01_vit_b_16_imagenet100 | ViT-Base/16 | 86M | Vision transformer |
| exp01_vit_l_16_imagenet100 | ViT-Large/16 | 304M | |
| exp01_vit_h_14_imagenet100 | ViT-Huge/14 | 632M | Largest; monitor for OOM |
| exp01_wrn4010_imagenet100 | WRN-40-10 | 56M | Wide residual |

All use pretrained ImageNet-1K weights (auto-downloaded by torchvision on first run).

---

## Results Structure

After completion, each config produces:

```
results/exp01_<arch>_imagenet100/
  checkpoints/task_a_best.pt, task_a_final.pt
  topology/slice_00..04/  (loss grids, persistence diagrams, topology_summary.json)
  phase3_naive/forgetting_curve.json
  phase3_ewc/forgetting_curve.json, fisher_info.pt
  phase3_si/forgetting_curve.json, si_importance.pt
```

### Pulling results back to local machine

```bash
# From your local machine:
rsync -avP cag1145@discovery.nmsu.edu:/fs1/scratch/cag1145/axiondeep-research/results/ ~/Corporate/axiondeep-research/results/
```

### Analysis (after all 10 complete)

Phases 4-6 run locally or on the cluster:

```bash
python -m experiments.exp01_topological_persistence.phase4_correlation --dataset imagenet100
python -m experiments.exp01_topological_persistence.phase5_predictive_model --dataset imagenet100
python -m experiments.exp01_topological_persistence.phase6_pooled_interaction --dataset imagenet100
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Connect | `ssh cag1145@discovery.nmsu.edu` |
| Activate env | `source /fs1/scratch/cag1145/persist-env/bin/activate` |
| Submit all | `bash slurm/submit_all.sh` |
| Check jobs | `squeue -u cag1145` |
| Job output | `cat slurm/logs/<JOBID>_<NAME>.out` |
| Cancel all | `scancel -u cag1145` |
| Disk usage | `du -sh data/ results/` |
| Pull results | `rsync -avP cag1145@discovery.nmsu.edu:.../results/ ./results/` |
