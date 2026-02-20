# Experiment Log

## EXP-01: Topological Persistence

### Current State (14/14 Architectures, CIFAR-100 Complete)

All 14 architectures complete on CIFAR-100 (Phases 1, 2 ×5 slices, and 3). CIFAR-10 in progress (6/14 complete).

| Architecture | Params | Task A Acc | ret@100 | H1 Pers | Type |
|---|---|---|---|---|---|
| ViT-Tiny | 0.8M | 52.7% | 22.5% | 0.18 | Transformer |
| ShuffleNet-V2 | 1.3M | 76.8% | 17.3% | 0.69 | CNN |
| ViT-Small | 3.0M | 62.2% | 9.6% | 0.32 | Transformer |
| MobileNet-V3-S | 1.5M | 68.6% | 7.6% | 1.90 | CNN+SE |
| EfficientNet-B0 | 4.1M | 76.6% | 7.1% | 2.12 | CNN+SE |
| RegNet-Y-400MF | 4.3M | 72.2% | 2.0% | 0.02 | CNN+SE |
| VGG-16-BN | 15.0M | 78.4% | 0.8% | 0.00 | CNN |
| WRN-28-10 | 36.5M | 84.0% | 0.3% | 0.08 | CNN |
| ResNet-18 | 11.2M | 82.0% | 0.2% | 0.00 | CNN |
| ResNet-50 | 23.6M | 83.6% | 0.1% | 0.00 | CNN |
| DenseNet-121 | 7.0M | 84.5% | 0.05% | 0.26 | CNN |
| MLP-Mixer | 2.3M | 61.5% | 0.03% | 0.00 | MLP |
| ConvNeXt-Tiny | 28.0M | 56.7% | 0.0%* | 0.11 | Modern CNN |
| ResNet-18 Wide | 44.7M | 83.1% | 0.0% | 0.00 | CNN |

*ConvNeXt shows non-monotonic recovery (0% → 3% → 0%).

#### Phase 4 Correlation (n=14, CIFAR-100, current)
- **H1 persistence:** ρ = 0.61, p = 0.021 (nominal), p_Bonf = 0.21 (not significant after correction)
- **Parameter count:** ρ = −0.74, p = 0.002, p_Bonf = 0.02 (survives Bonferroni)
- **Partial H1|params:** ρ = 0.35, p = 0.24 (non-significant)
- **VIF(H1, params):** 1.45
- **Rank regression R²:** 0.61 (only params significant)
- LOO: 14/14 folds significant for H1; permutation p ≈ 0.02

---

### Historical Runs (25x25 grid, n=8 — SUPERSEDED by 50x50 re-runs)

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

### Parameter Updates (2026-02-20)

#### 7. Multi-slice Seed Fix
- **Problem:** `set_seed(cfg["seed"])` always produced identical landscape_seed (478163327) for all multi-slice runs — all 5 "slices" were actually the same slice.
- **Fix:** `set_seed(cfg["seed"] + run_offset)` where `run_offset = int(args.run_id)`. Now produces 5 genuinely unique seeds.
- **Impact:** All existing multi-slice runs invalid — must re-run Phase 2 for all architectures.

#### 8. Early Eval Steps
- **Problem:** Old eval_steps `[100, 500, 1000, 5000, 10000, 25000]` missed the critical early-forgetting window. 9/14 CIFAR-10 architectures at 0% by step 100.
- **Change:** All 28 configs updated to `[10, 25, 50, 100, 250, 500, 1000, 5000]`. Max steps reduced from 25000 to 5000.
- **Impact:** All Phase 3 runs need re-run with new eval steps.

#### 9. Bonferroni Correction + Kendall's Tau
- **Problem:** 10 metrics × p<0.05 = 43% family-wise error rate without correction.
- **Change:** Phase 4 now reports Bonferroni-corrected p-values (p × 10) and Kendall's tau alongside Spearman ρ.
- **Impact:** H1 (p=0.021) does NOT survive Bonferroni (p_Bonf=0.21). Only params survives.

#### 10. Landscape Validation
- **Change:** Phase 2 now checks for NaN, Inf, and degenerate (near-zero variance) loss grids after computation. NaN replaced with max finite value; Inf clamped.

#### 11. Task B Learning Check
- **Change:** Phase 3 now warns if final Task B accuracy < 2× chance level, indicating retention metric may be unreliable.

#### 12. Phase 2b Multi-slice Awareness
- **Change:** Phase 2b now falls back to multi-slice files (`*_run*.pt/npz/json`) when default files don't exist, instead of erroring out.

#### 13. Dashboard Updates
- Clean & Rebuild button (one-click fix for invalid runs)
- Re-run All P3 button
- Multi-slice P2 progress indicator (e.g., "P2 3/5")
- Removed multiplier mechanism (replaced by explicit PHASES entries for 5 slices)

### Next Steps
- **Complete CIFAR-10 runs** (7/14 remaining): resume via dashboard "Run Both Datasets"
- **Re-run all Phase 2** with seed fix (Clean & Rebuild)
- **Re-run all Phase 3** with early eval steps
- Multi-seed analysis for 4 extreme archs: ViT-Small, ShuffleNet, ResNet-50, MLP-Mixer
- Update paper with Bonferroni-corrected statistics, multi-slice error bars
- CIFAR-10 cross-dataset replication analysis

### Known Issues
- **Param count confound:** ρ(params,ret) = −0.74 dominates ρ(H1,ret) = 0.61; H1 non-significant after partialing out params (p = 0.24). Scale dominates topology at n=14.
- **H1 does not survive Bonferroni:** p_Bonf = 0.21 with 10 tests. Nominally significant only.
- **Floor effect in CIFAR-10:** 9/14 architectures at 0% retention by step 100 (old eval steps). New early checkpoints should help.
- Barrier metric overflows for large models (clamped at 1e6)
- Hessian trace goes negative for WRN-28-10 (saddle point)
- MLP-Mixer challenges topology-retention hypothesis (high H0, zero retention)

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
- 5 independent random slices per architecture (mean ± std in Phase 4)

### Baseline Metrics (added 2025-02-10, hardened 2026-02-17)
- Hessian trace: Hutchinson estimator, 30 Rademacher samples, batch capped at 64
- Max eigenvalue: Power iteration, 50 iterations, batch capped at 64
- Fisher Information trace: 10 batches of squared gradients
- Loss barrier: Max loss increase along 10 filter-normalized random directions
- Each metric runs independently (failure in one does not block others)

### Statistical Analysis (upgraded 2026-02-20)
- Spearman rank correlation + Kendall's tau (robust to ties)
- Bonferroni correction: 10 metrics, adjusted α = 0.005
- Partial correlation controlling for parameter count
- Symmetric partial correlations + rank regression + VIF
- Permutation test: 10,000 shuffles, two-tailed
- Leave-one-out cross-validation: min/mean/max p across folds

### Forgetting Measurement
- Train Task B (CIFAR-100 classes 50-99) for 5,000 steps
- Evaluate Task A accuracy at steps: 0, 10, 25, 50, 100, 250, 500, 1000, 5000
- Early checkpoints (10, 25, 50) added 2026-02-20 to capture fast-forgetting architectures
- Primary retention metric: ret@100 (switched from ret@10k for better variance)
- Secondary metric: AURC (area under retention curve, uses full forgetting trajectory)
- Note: max_steps reduced from 25,000 to 5,000 — all architectures reach terminal retention by step 1,000
