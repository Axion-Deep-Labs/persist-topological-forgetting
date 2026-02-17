# Axion Deep Labs — Research Experiments

Experimental codebase investigating the relationship between loss landscape geometry and catastrophic forgetting in neural networks.

## EXP-01: Topological Signatures of Knowledge Persistence (PERSIST)

**Research Question:** Does the topological structure of a model's loss landscape predict how resistant it is to catastrophic forgetting?

**Method:**
1. **Phase 1** — Train 8 architectures on Task A (CIFAR-100 classes 0–49)
2. **Phase 2** — Sample the 2D loss landscape around convergence, compute persistent homology (H0, H1) via sublevel set filtration, plus baseline geometry metrics (Hessian trace, max eigenvalue, Fisher trace, loss barrier)
3. **Phase 3** — Train sequentially on Task B (classes 50–99), measure Task A retention at intervals
4. **Phase 4** — Spearman rank correlation of each metric against forgetting resistance across architectures

**Architectures:**

| ID | Architecture | Params | Status |
|----|-------------|--------|--------|
| exp01 | ResNet-18 | ~11M | Complete |
| exp01_resnet50 | ResNet-50 | ~23.6M | Complete |
| exp01_vit | ViT-Small | ~3M | Complete |
| exp01_wrn2810 | WRN-28-10 | ~36.5M | In Progress |
| exp01_mlpmixer | MLP-Mixer | ~2.3M | Pending |
| exp01_resnet18wide | ResNet-18 Wide | ~44.7M | Pending |
| exp01_densenet121 | DenseNet-121 | ~7M | Pending |
| exp01_efficientnet | EfficientNet-B0 | ~4.1M | Pending |

**Key Finding (n=3, preliminary):**
H0 persistence (total lifetime of connected components in the loss surface) correlates with forgetting resistance (Spearman ρ = 0.866). ViT-Small shows 2× the H0 persistence of ResNets and is the only architecture to retain any Task A accuracy after Task B training.

## Project Structure

```
axiondeep-research/
├── configs/                 # YAML experiment configs (one per architecture)
├── dashboard/               # Local Flask dashboard (localhost:5050)
│   ├── app.py               # Backend: experiment runner, system monitor, API
│   └── templates/index.html # Frontend: cards, results table, live logs
├── experiments/
│   ├── shared/              # Shared code across experiments
│   │   ├── datasets.py      # Split-CIFAR-100 data loader
│   │   ├── models.py        # 8 architecture definitions
│   │   ├── baseline_metrics.py  # Hessian, Fisher, sharpness, loss barrier
│   │   └── utils.py         # Seed, config loading, checkpointing
│   ├── exp01_topological_persistence/
│   │   ├── phase1_train_task_a.py
│   │   ├── phase2_landscape_topology.py
│   │   ├── phase3_sequential_forgetting.py
│   │   ├── phase4_correlation.py
│   │   └── Articles/        # Paper drafts and references
│   ├── exp02_phi_survey/     # (Planned) Integrated Information
│   └── exp03_bekenstein/     # (Planned) Bekenstein Bound Analogs
├── results/                 # Output data (gitignored)
├── data/                    # Datasets (gitignored, auto-downloaded)
├── notebooks/               # Jupyter analysis notebooks
└── pyproject.toml           # Dependencies
```

## Running Experiments

### Via Dashboard (Recommended)
```bash
cd ~/projects/axiondeep-research
.venv/bin/python dashboard/app.py
# Open http://localhost:5050
```

### Manual
```bash
# Phase 1: Train on Task A
python -m experiments.exp01_topological_persistence.phase1_train_task_a --config configs/exp01.yaml

# Phase 2: Loss landscape topology + baseline metrics
python -m experiments.exp01_topological_persistence.phase2_landscape_topology --config configs/exp01.yaml

# Phase 3: Sequential forgetting measurement
python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting --config configs/exp01.yaml

# Phase 4: Cross-architecture correlation (needs 3+ complete experiments)
python -m experiments.exp01_topological_persistence.phase4_correlation \
    --results-dirs results/exp01 results/exp01_resnet50 results/exp01_vit
```

## Hardware

- **GPU:** NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **RAM:** 64 GB DDR5
- **CPU:** 24 cores

## Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires Python >= 3.10, PyTorch >= 2.2, CUDA.

## Key References

- Li et al. (2018) — Filter-normalized loss landscape visualization
- Rieck et al. (2019) — Topological methods for neural network analysis
- French (1999) — Catastrophic forgetting survey
- Mirzadeh et al. (2020) — Loss landscape geometry and forgetting
