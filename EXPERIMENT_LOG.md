# Experiment Log

## EXP-01: Topological Persistence

### Completed Runs (8/8 Architectures)

#### ResNet-18 (`exp01`)
- **Date:** 2025-02-07
- **Config:** `configs/exp01.yaml`
- **Phase 1:** Task A accuracy = 82.0%, 100 epochs
- **Phase 2:** H0 = 2151.5, H1 = 0.0, grid = 25x25, range = [-1, 1]
- **Phase 3:** Ret@100 = 0.2%, Ret@10k = 0.0% (instant forgetting)
- **Baseline metrics:** Not computed (pre-baseline update)
- **Notes:** First architecture tested. Forgets almost instantly.

#### ResNet-50 (`exp01_resnet50`)
- **Date:** 2025-02-08
- **Config:** `configs/exp01_resnet50.yaml`
- **Phase 1:** Task A accuracy = 83.6%, 100 epochs
- **Phase 2:** H0 = 1639.0, H1 = 0.0, grid = 25x25
- **Phase 3:** Ret@100 = 0.1%, Ret@10k = 0.0% (fastest forgetting)
- **Baseline metrics:** Not computed (pre-baseline update)
- **Notes:** Lowest H0 despite most parameters. Topology != model size proxy.

#### ViT-Small (`exp01_vit`)
- **Date:** 2025-02-08
- **Config:** `configs/exp01_vit.yaml`
- **Phase 1:** Task A accuracy = 62.2%, 100 epochs
- **Phase 2:** H0 = 4254.2, H1 = 0.0, grid = 25x25
- **Phase 3:** Ret@100 = 9.6%, Ret@1k = 6.7%, Ret@10k = 1.4% (gradual decay)
- **Baseline metrics:** Not computed (pre-baseline update)
- **Notes:** Highest H0 (2x ResNets), ONLY architecture with measurable retention at 10k. Strongest hypothesis support.

#### WRN-28-10 (`exp01_wrn2810`)
- **Date:** 2025-02-10
- **Config:** `configs/exp01_wrn2810.yaml`
- **Phase 1:** Task A accuracy = 84.0%, epoch 97
- **Phase 2:** H0 = 2272.6, H1 = 0.0, Hessian trace = -1781.1, Max eig = -2642.0, Fisher = 199776.9, Barrier = 1.2e30
- **Phase 3:** Ret@100 = 0.3%, Ret@10k = 0.0%
- **Notes:** First with baseline metrics. Negative Hessian trace/eigenvalue suggest numerical instability at this scale. Barrier is astronomically high (filter normalization blowup on 36.5M params).

#### MLP-Mixer (`exp01_mlpmixer`)
- **Date:** 2025-02-11
- **Config:** `configs/exp01_mlpmixer.yaml`
- **Phase 1:** Task A accuracy = 61.5%, epoch 79
- **Phase 2:** H0 = 3758.8, H1 = 0.0, Hessian trace = 1322.9, Max eig = 199.8, Fisher = 150.6, Barrier = 32.3
- **Phase 3:** Ret@100 = 0.0%, Ret@10k = 0.0% (instant forgetting)
- **Notes:** 2nd highest H0 but zero retention. CHALLENGE to hypothesis — high topological complexity doesn't guarantee persistence for token-mixing architectures. Clean baseline metrics.

#### ResNet-18 Wide (`exp01_resnet18wide`)
- **Date:** 2025-02-17
- **Config:** `configs/exp01_resnet18wide.yaml`
- **Phase 1:** Task A accuracy = 83.1%, epoch 89
- **Phase 2:** H0 = 1559.4, H1 = 0.0, Hessian trace = 19515.6, Max eig = 6297.0, Fisher = 6397.2, Barrier = 3.7e23
- **Phase 3:** Ret@100 = 0.0%, Ret@10k = 0.0% (instant forgetting)
- **Notes:** Lowest H0 of all architectures. Widening a ResNet reduces topological complexity. Very sharp minimum (highest Hessian trace + eigenvalue).

#### DenseNet-121 (`exp01_densenet121`)
- **Date:** 2025-02-17
- **Config:** `configs/exp01_densenet121.yaml`
- **Phase 1:** Task A accuracy = 84.5%, epoch 92 (highest accuracy)
- **Phase 2:** H0 = 2070.5, H1 = 0.0, Hessian trace = 3342.5, Max eig = 1769.0, Fisher = 2083.8, Barrier = 1.3e34
- **Phase 3:** Ret@100 = 0.0%, Ret@10k = 0.0% (instant forgetting)
- **Notes:** Best accuracy but low H0 and zero retention. Barrier metric blown up.

#### EfficientNet-B0 (`exp01_efficientnet`)
- **Date:** 2025-02-17
- **Config:** `configs/exp01_efficientnet.yaml`
- **Phase 1:** Task A accuracy = 76.6%, epoch 99
- **Phase 2:** H0 = 3579.9, H1 = 1.74 (16 features — FIRST non-zero H1!), Hessian trace = 782.7, Max eig = 177.5, Fisher = 59.2, Barrier = overflow (Infinity)
- **Phase 3:** Ret@100 = 7.1%, Ret@1k = 0.1%, Ret@10k = 0.0%
- **Notes:** 3rd highest H0, and ONLY architecture with non-zero H1 features (16 loops detected). Some retention at 100 steps but decays to zero by 10k. Barrier computation overflowed to Infinity (fixed in baseline_metrics.py).

### Cross-Architecture Summary (n=8, All Complete)

| Architecture | Params | Acc | H0 | H1 | Ret@100 | Ret@10k |
|---|---|---|---|---|---|---|
| ViT-Small | ~3M | 62.2% | 4254.2 | 0.0 | 9.6% | 1.35% |
| MLP-Mixer | ~2.3M | 61.5% | 3758.8 | 0.0 | 0.0% | 0.0% |
| EfficientNet-B0 | ~4.1M | 76.6% | 3579.9 | 1.74 | 7.1% | 0.0% |
| WRN-28-10 | ~36.5M | 84.0% | 2272.6 | 0.0 | 0.3% | 0.0% |
| ResNet-18 | ~11M | 82.0% | 2151.5 | 0.0 | 0.2% | 0.0% |
| DenseNet-121 | ~7M | 84.5% | 2070.5 | 0.0 | 0.0% | 0.0% |
| ResNet-50 | ~23.6M | 83.6% | 1639.0 | 0.0 | 0.1% | 0.0% |
| ResNet-18 Wide | ~44.7M | 83.1% | 1559.4 | 0.0 | 0.0% | 0.0% |

*Sorted by H0 persistence (descending)*

### Preliminary Observations (Pre-Phase 4)

1. **ViT-Small remains the only architecture with retention at 10k steps** (1.35%). It also has the highest H0 by a wide margin.
2. **MLP-Mixer breaks the simple H0-retention mapping** — 2nd highest H0 (3758.8) but zero retention. Suggests H0 alone is insufficient; architecture type matters.
3. **EfficientNet is the only model with non-zero H1** (16 loop features). It also shows early retention (7.1% at step 100) that decays to zero.
4. **Topology != model size**: ResNet-18 Wide (44.7M params, lowest H0) vs ViT-Small (3M params, highest H0). Confirmed across all 8 architectures.
5. **Baseline metrics are unreliable at large scale**: WRN-28-10, ResNet-18 Wide, DenseNet-121 all have numerically unstable barrier estimates (1e23–1e34). The barrier metric needs better normalization.
6. **Forgetting is catastrophic for all CNN-like architectures**: Only ViT (attention-based) retains any knowledge past 1k steps.

### Phase 4 Correlation Results (n=8, ret@10k — SUPERSEDED)

Initial correlation run with all 8 architectures using ret@10k:
- **Spearman rho = 0.5774** (H0 persistence vs ret@10k, n=8)
- Not statistically significant — 7/8 architectures have ret@10k = 0.0%, giving almost no variance
- Only H0 was computable for full n=8; baseline metrics returned N/A for first 3 architectures (ResNet-18, ResNet-50, ViT-Small) which lack Hessian/Fisher/barrier data

**This result is superseded by the parameter updates below.**

### Parameter Updates (2026-02-17)

#### 1. Grid Resolution: 25x25 → 50x50
- **Rationale:** H1 = 0 for 7/8 architectures at 25x25 (625 points). Only EfficientNet showed non-zero H1 (16 features). The coarse grid likely under-resolves loop structures in the loss landscape.
- **Change:** All 8 config files updated: `steps_per_direction: 50`, `num_landscape_samples: 2500`
- **Impact:** 4x compute cost (~3.3 hours total for all 8 Phase 2 re-runs). Should reveal H1 features for more architectures.
- **Requires:** Phase 2 re-run for all 8 architectures (also adds missing baseline metrics for first 3)

#### 2. Retention Metric: ret@10k → ret@100
- **Rationale:** ret@10k has near-zero variance (7/8 architectures = 0.0%, only ViT = 1.35%). ret@100 provides much better spread:
  - ViT-Small: 9.6%, EfficientNet: 7.1%, WRN-28-10: 0.26%, ResNet-18: 0.24%
  - ResNet-50: 0.14%, DenseNet-121: 0.05%, MLP-Mixer: 0.03%, ResNet-18 Wide: 0.0%
- **Change:** `phase4_correlation.py` now uses `compute_retention_at_step(forget, 100)` instead of 10000
- **Impact:** More variance → more meaningful Spearman correlation. AURC metric unchanged (already uses full curve).

#### 3. Dashboard Re-run Capability
- Added `/api/rerun` endpoint to dashboard backend
- Per-phase re-run buttons on each experiment card
- "Re-run All P2" button in header for batch Phase 2 re-runs
- Existing results backed up to `.bak` before re-running

#### 4. Randomized Landscape Seeds
- **Rationale:** Fixed seed = same 2D slice every run. Randomized seeds enable stability analysis across multiple slices without code changes.
- **Change:** Phase 2 now generates a random seed for landscape directions at runtime. Seed is logged in `topology_summary.json` as `landscape_seed` for reproducibility.
- **Impact:** Each "Re-run P2" produces a different 2D slice. Run multiple times per architecture to assess topological feature consistency.

#### 5. Phase 2 Performance Optimizations (2026-02-17)
- **Mixed precision (AMP):** Forward passes wrapped in `torch.amp.autocast("cuda")`. ~2x throughput on RTX 4090 tensor cores. Loss accumulated in fp32.
- **GPU-resident test set:** Entire Task A test set (~5000 images, ~20MB) pre-loaded to GPU. Eliminates CPU→GPU transfer on every grid point (previously 2500 transfers per run).
- **Row-wise incremental perturbation:** Alpha component set once per row. Beta incremented by `beta_step * dir2` per column instead of recomputing `base + alpha * dir1 + beta * dir2` from scratch. Cuts parameter write operations ~50%.
- **Estimated speedup:** ~2-2.5x faster (e.g. ResNet-18 from ~33 min to ~15 min at 50x50).

#### 6. Baseline Metrics Robustness (2026-02-17)
- **Problem:** ViT-Small Phase 2 crashed during Hessian computation (likely OOM from `create_graph=True` with large batch).
- **Fixes:**
  - Each baseline metric runs independently — one failure doesn't block the others
  - Hessian batch capped at 64 samples (down from 256) to reduce memory for second-order gradients
  - Landscape tensors (`base_params`, `dir1`, `dir2`, GPU dataset) freed before baseline metrics start
  - Phase 2 wrapped in try/except: baseline failure saves topology results (the important part) regardless

### Next Steps
- Re-run Phase 2 for all 8 architectures with 50x50 grid (adds baseline metrics for first 3)
- Re-run Phase 4 correlation with ret@100 and complete baseline metric coverage
- Evaluate whether 50x50 grid reveals H1 features for more architectures
- If H1 correlates: run stability analysis (multiple random seeds per architecture)

### Known Issues
- H1 = 0 for 7/8 architectures at 25x25 (likely discretization artifact; EfficientNet is the exception). 50x50 re-run should resolve.
- Baseline metrics numerically unstable for large models: WRN-28-10 (negative Hessian), DenseNet-121/ResNet-18 Wide (barrier 1e23–1e34). Individual metric failures now handled gracefully.
- EfficientNet barrier was `Infinity` in JSON — replaced with `null`, added `sanitize_for_json()` to dashboard

---

## Methodology Notes

### Loss Landscape Sampling
- 2D slice via filter-normalized random directions (Li et al., 2018)
- 50x50 grid (upgraded from 25x25), range [-1, 1] around converged weights
- Randomized landscape seed per run (logged in summary for reproducibility)
- Evaluated on test set (pre-loaded to GPU, mixed precision forward passes)
- Incremental row-wise perturbation for efficiency

### Persistent Homology
- Sublevel set filtration on 8-connected grid
- Sparse distance matrix (lower-star: edge value = max of endpoints)
- Computed via Ripser (sparse mode)
- H0 = connected components, H1 = loops

### Baseline Metrics (added 2025-02-10, hardened 2026-02-17)
- Hessian trace: Hutchinson estimator, 30 Rademacher samples, batch capped at 64
- Max eigenvalue: Power iteration, 50 iterations, batch capped at 64
- Fisher Information trace: 10 batches of squared gradients
- Loss barrier: Max loss increase along 10 filter-normalized random directions
- Each metric runs independently (failure in one does not block others)

### Forgetting Measurement
- Train Task B (CIFAR-100 classes 50-99) for 25,000 steps
- Evaluate Task A accuracy at steps: 0, 100, 500, 1000, 5000, 10000, 25000
- Primary retention metric: ret@100 (switched from ret@10k for better variance)
- Secondary metric: AURC (area under retention curve, uses full forgetting trajectory)
