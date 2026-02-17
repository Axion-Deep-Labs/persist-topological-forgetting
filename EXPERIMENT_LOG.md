# Experiment Log

## EXP-01: Topological Persistence

### Completed Runs

#### ResNet-18 (`exp01`)
- **Date:** 2025-02-07
- **Config:** `configs/exp01.yaml`
- **Phase 1:** Task A accuracy = 82.0%, 100 epochs
- **Phase 2:** H0 = 2151.5, H1 = 0.0, grid = 25×25, range = [-1, 1]
- **Phase 3:** Retention @ step 100 = 0.2%, @ step 500 = 0.0% (instant forgetting)
- **Baseline metrics:** Not yet computed (pre-baseline update)
- **Notes:** First architecture tested. Forgets almost instantly.

#### ResNet-50 (`exp01_resnet50`)
- **Date:** 2025-02-08
- **Config:** `configs/exp01_resnet50.yaml`
- **Phase 1:** Task A accuracy = 83.6%, 100 epochs
- **Phase 2:** H0 = 1639.0, H1 = 0.0, grid = 25×25
- **Phase 3:** Retention @ step 100 = 0.1%, @ step 500 = 0.0% (fastest forgetting)
- **Baseline metrics:** Not yet computed (pre-baseline update)
- **Notes:** Lowest H0 despite most parameters. Topology ≠ model size proxy. Confirmed.

#### ViT-Small (`exp01_vit`)
- **Date:** 2025-02-08
- **Config:** `configs/exp01_vit.yaml`
- **Phase 1:** Task A accuracy = 62.2%, 100 epochs
- **Phase 2:** H0 = 4254.2, H1 = 0.0, grid = 25×25
- **Phase 3:** Retention @ step 100 = 6.0%, gradual decay over thousands of steps
- **Baseline metrics:** Not yet computed (pre-baseline update)
- **Notes:** Highest H0 (2× ResNets), only architecture with measurable retention. Supports hypothesis.

### In Progress

#### WRN-28-10 (`exp01_wrn2810`)
- **Date:** 2025-02-10
- **Config:** `configs/exp01_wrn2810.yaml`
- **Phase 1:** Complete
- **Phase 2:** Complete
- **Phase 3:** Running (started ~23:20)
- **Notes:** First architecture with baseline metrics (Hessian, Fisher, sharpness, loss barrier).

### Pending

| Architecture | Config | Status |
|-------------|--------|--------|
| MLP-Mixer | `exp01_mlpmixer.yaml` | Not started |
| ResNet-18 Wide | `exp01_resnet18wide.yaml` | Not started |
| DenseNet-121 | `exp01_densenet121.yaml` | Not started |
| EfficientNet-B0 | `exp01_efficientnet.yaml` | Not started |

### Cross-Architecture Correlation (n=3, Preliminary)

| Architecture | H0 Persistence | Retention @10k | Ranking |
|-------------|---------------|----------------|---------|
| ViT-Small | 4,254.2 | 1.35% | Most resistant |
| ResNet-18 | 2,151.5 | 0.0% | Middle |
| ResNet-50 | 1,639.0 | 0.0% | Least resistant |

- **Spearman ρ = 0.866** (H0 vs retention)
- **p = 0.333** (not significant at n=3, need n≥5)
- Ranking: perfect match with hypothesis

### Known Issues
- H1 = 0 for all architectures (likely discretization artifact of 25×25 grid)
- First 3 architectures lack baseline metrics (ran before `baseline_metrics.py` existed)
- Need to re-run Phase 2 for exp01, exp01_resnet50, exp01_vit to get baseline metrics

---

## Methodology Notes

### Loss Landscape Sampling
- 2D slice via filter-normalized random directions (Li et al., 2018)
- 25×25 grid, range [-1, 1] around converged weights
- Evaluated on test set (deterministic, smaller)

### Persistent Homology
- Sublevel set filtration on 8-connected grid
- Sparse distance matrix (lower-star: edge value = max of endpoints)
- Computed via Ripser (sparse mode)
- H0 = connected components, H1 = loops (all zero so far)

### Baseline Metrics (added 2025-02-10)
- Hessian trace: Hutchinson estimator, 30 Rademacher samples
- Max eigenvalue: Power iteration, 50 iterations
- Fisher Information trace: 10 batches of squared gradients
- Loss barrier: Max loss increase along 10 filter-normalized random directions

### Forgetting Measurement
- Train Task B (CIFAR-100 classes 50–99) for 10,000 steps
- Evaluate Task A accuracy at steps: 0, 100, 500, 1000, 2000, 5000, 10000
- Retention ratio = Task A acc / initial Task A acc
