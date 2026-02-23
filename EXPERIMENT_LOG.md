# Experiment Log

## EXP-01: Topological Persistence

### Current State (38/57 Configurations Complete)

**Two datasets fully complete (19 architectures each, all 7 phases). RESISC-45 pending.**

- **CIFAR-100:** 19/19 architectures, Phases 1-5 complete
- **CUB-200-2011:** 19/19 architectures, Phases 1-5 complete
- **RESISC-45:** 0/19 architectures (pending)

---

### CIFAR-100 Results (n=19, Easy Benchmark)

| Architecture | Params | Task A Acc | ret@100 | ret@10 | H1 Pers | Type |
|---|---|---|---|---|---|---|
| ViT-Tiny | 0.3M | 52.7% | 22.5% | 95.9% | 0.01 | Transformer |
| ShuffleNet-V2 | 1.3M | 76.8% | 17.3% | 84.7% | 0.79 | CNN |
| ViT-Small | 2.2M | 62.2% | 9.6% | 94.7% | 0.24 | Transformer |
| MobileNet-V3-S | 1.1M | 68.6% | 7.6% | 75.0% | 1.89 | CNN |
| EfficientNet-B0 | 4.1M | 76.6% | 7.1% | 78.6% | 1.91 | CNN |
| WRN-28-1 | 0.4M | 71.7% | 6.6% | 51.0% | 0.00 | WRN-ladder |
| RegNet-Y-400MF | 4.0M | 72.2% | 2.0% | 54.1% | 0.05 | CNN |
| WRN-28-2 | 1.5M | 78.6% | 1.1% | 22.8% | 0.00 | WRN-ladder |
| VGG-16-BN | 14.8M | 78.4% | 0.8% | 88.0% | 0.00 | CNN |
| WRN-28-8 | 23.4M | 82.9% | 0.7% | 4.4% | 0.01 | WRN-ladder |
| WRN-28-4 | 5.9M | 81.8% | 0.3% | 8.5% | 0.02 | WRN-ladder |
| WRN-28-10 | 36.5M | 84.0% | 0.3% | 5.3% | 0.07 | WRN-ladder |
| ResNet-18 | 11.2M | 82.0% | 0.2% | 46.7% | 0.00 | CNN |
| WRN-28-6 | 13.2M | 82.8% | 0.1% | 4.5% | 0.02 | WRN-ladder |
| ResNet-50 | 23.7M | 83.6% | 0.1% | 56.0% | 0.00 | CNN |
| DenseNet-121 | 7.1M | 84.5% | 0.05% | 25.7% | 0.01 | CNN |
| MLP-Mixer | 2.3M | 61.5% | 0.03% | 0.03% | 0.12 | MLP |
| ConvNeXt-Tiny | 27.9M | 56.7% | 0.0% | 45.0% | 0.00 | CNN |
| ResNet-18 Wide | 44.7M | 83.1% | 0.0% | 29.7% | 0.00 | CNN |

#### Phase 4 Correlation (n=19, CIFAR-100)
- **Parameter count:** rho = -0.76, p = 0.0002, p_Bonf = 0.002 (survives Bonferroni)
- **H1 persistence:** rho = 0.47, p = 0.042, p_Bonf = 0.50 (does NOT survive)
- **Partial H1|params:** rho = 0.33, p = 0.19 (not significant)
- **H0 persistence:** rho = 0.37, p = 0.12
- **Conclusion:** On this easy task, parameter count dominates. Topology is redundant.

#### Phase 4 WRN Width Ladder (n=6, CIFAR-100)
- H0 perfectly monotonic with width (rho = -1.0 vs params)
- H0 vs retention: rho = 0.71, p = 0.11 (suggestive but n too small)
- Kruskal-Wallis across slices: p = 4.0e-12 (H0 strongly distinguishes architectures)
- Cohen's d between adjacent widths: 2.1 to 11.4 (massive effect sizes)
- Pairwise ordering probability: 80-100% across all adjacent pairs

#### Phase 4 EWC Benefit (n=19, CIFAR-100)
- H0 vs EWC benefit: rho = 0.76, p = 0.0002
- Params vs EWC benefit: rho = -0.74, p = 0.0003
- H1 vs EWC benefit: rho = 0.24, p = 0.33 (not significant)

#### Phase 4 Cubical vs Ripser Agreement (CIFAR-100)
- H1: rho = 1.0 (perfect agreement)
- H0: rho = 0.15 (different representations, expected)

#### Phase 5 Predictive Model (n=19, CIFAR-100)

| Outcome | Params-only rho | Params+Topo rho | Perm. p | Verdict |
|---|---|---|---|---|
| ret@100 | 0.43 | 0.30 | 0.295 | Not significant |
| ret@10 | -0.08 | 0.14 | 0.095 | Ambiguous |
| Early AURC | 0.14 | 0.37 | 0.162 | Ambiguous |

**CIFAR-100 summary:** Topology does not add significant predictive value beyond parameter count on this easy task.

---

### CUB-200-2011 Results (n=19, Hard Fine-Grained Classification)

| Architecture | Params | Task A Acc | ret@100 | ret@10 | H1 Pers | Type |
|---|---|---|---|---|---|---|
| ViT-Tiny | 0.3M | — | 31.1% | — | — | Transformer |
| ViT-Small | 2.2M | — | 23.4% | — | — | Transformer |
| WRN-28-10 | 36.5M | — | 8.1% | — | — | WRN-ladder |
| WRN-28-8 | 23.4M | — | 5.0% | — | — | WRN-ladder |
| EfficientNet-B0 | 4.1M | — | 3.5% | — | — | CNN |
| WRN-28-4 | 5.9M | — | 2.7% | — | — | WRN-ladder |
| DenseNet-121 | 7.1M | — | 2.1% | — | — | CNN |
| MobileNet-V3-S | 1.1M | — | 1.6% | — | — | CNN |
| WRN-28-2 | 1.5M | — | 1.6% | — | — | WRN-ladder |
| WRN-28-6 | 13.2M | — | 1.3% | — | — | WRN-ladder |
| WRN-28-1 | 0.4M | — | 1.3% | — | — | WRN-ladder |
| ResNet-18 | 11.2M | — | 0.6% | — | — | CNN |
| ShuffleNet-V2 | 1.3M | — | 0.3% | — | — | CNN |
| ResNet-18 Wide | 44.7M | — | 0.2% | — | — | CNN |
| RegNet-Y-400MF | 4.0M | — | 0.2% | — | — | CNN |
| ResNet-50 | 23.7M | — | 0.1% | — | — | CNN |
| VGG-16-BN | 14.8M | — | 0.0% | — | — | CNN |
| ConvNeXt-Tiny | 27.9M | — | 0.0% | — | — | CNN |
| MLP-Mixer | 2.3M | — | 0.0% | — | — | MLP |

#### Phase 4 Correlation (n=19, CUB-200)
- **Parameter count:** rho = -0.27, p = 0.27 (NOT significant, fails on hard task)
- **H0 persistence:** rho = -0.35, p = 0.15
- **H1 persistence:** (see Phase 5 for predictive value)
- **Conclusion:** On this hard fine-grained task, parameter count is NOT a reliable predictor. The rankings shuffle compared to CIFAR-100.

#### Phase 4 WRN Width Ladder (n=6, CUB-200)
- H0 monotonic with width (rho = -1.0 vs params, same as CIFAR-100)
- H0 vs retention: rho = -0.83, p = 0.04 (significant but OPPOSITE direction to CIFAR-100)

#### Phase 5 Predictive Model (n=19, CUB-200) — KEY RESULT

| Outcome | Params-only rho | Params+Topo rho | Perm. p | Verdict |
|---|---|---|---|---|
| **ret@10** | **-0.92** | **0.34** | **0.037** | **SIGNIFICANT** |
| ret@100 | -0.90 | -0.12 | 0.375 | Not significant |
| Early AURC | -0.98 | -0.55 | 0.926 | Not significant |

**CUB-200 ret@10 finding:**
- Params alone predict the WRONG direction (rho = -0.92)
- Adding topology RESCUES prediction (rho flips to +0.34)
- 17.5% MAE reduction (0.186 to 0.154)
- Permutation test: p = 0.037
- Matched-dimensionality control: exceeds 95th percentile of random features (p = 0.0)
- Topology alone (Model D): rho = 0.33, MAE = 0.147 (outperforms params-only)

---

### Cross-Dataset Insight

**The central finding:** Topology's predictive value depends on task difficulty.

- **Easy tasks (CIFAR-100):** Parameter count is all you need. Bigger models retain better. Topology is redundant because scale already explains the variance.
- **Hard tasks (CUB-200):** Parameter count FAILS as a predictor. The retention rankings shuffle. Topology captures early knowledge fragility that nothing else does.

This is exactly the commercially relevant regime. Real-world continual learning tasks (medical imaging, rare fraud patterns, edge-case driving scenarios) are hard and fine-grained, like CUB-200.

---

### Architecture Details (Historical)

#### ResNet-18 (`exp01`)
- **Phase 1:** Task A accuracy = 82.0%, 100 epochs
- **Phase 2:** H0 = 8458 (5-slice mean), H1 = 0.0
- **Phase 3:** ret@100 = 0.2%, ret@10 = 46.7%

#### ResNet-50 (`exp01_resnet50`)
- **Phase 1:** Task A accuracy = 83.6%, 100 epochs
- **Phase 2:** H0 = 6333, H1 = 0.0
- **Phase 3:** ret@100 = 0.1%, ret@10 = 56.0%

#### ViT-Small (`exp01_vit`)
- **Phase 1:** Task A accuracy = 62.2%, 100 epochs
- **Phase 2:** H0 = 16254, H1 = 0.24
- **Phase 3:** ret@100 = 9.6%, ret@10 = 94.7%

#### WRN-28-10 (`exp01_wrn2810`)
- **Phase 1:** Task A accuracy = 84.0%
- **Phase 2:** H0 = 8835, H1 = 0.07
- **Phase 3:** ret@100 = 0.3%, ret@10 = 5.3%

#### EfficientNet-B0 (`exp01_efficientnet`)
- **Phase 1:** Task A accuracy = 76.6%
- **Phase 2:** H0 = 14335, H1 = 1.91 (highest H1)
- **Phase 3:** ret@100 = 7.1%, ret@10 = 78.6%

#### MobileNet-V3-S (`exp01_mobilenetv3`)
- **Phase 2:** H1 = 1.89 (2nd highest H1)
- **Phase 3:** ret@100 = 7.6%, ret@10 = 75.0%

#### ShuffleNet-V2 (`exp01_shufflenet`)
- **Phase 2:** H1 = 0.79 (3rd highest H1)
- **Phase 3:** ret@100 = 17.3%, ret@10 = 84.7%

#### ViT-Tiny (`exp01_vittiny`)
- **Phase 2:** H1 = 0.01
- **Phase 3:** ret@100 = 22.5% (highest retention), ret@10 = 95.9%

#### MLP-Mixer (`exp01_mlpmixer`)
- **Phase 2:** H0 = 15390, H1 = 0.12
- **Phase 3:** ret@100 = 0.03%, ret@10 = 0.03% (near-instant forgetting)
- Challenges topology-retention hypothesis (moderate H1, zero retention)

#### ConvNeXt-Tiny (`exp01_convnext`)
- **Phase 2:** H0 = 31210 (highest H0), H1 = 0.0
- **Phase 3:** ret@100 = 0.0%, ret@10 = 45.0%
- Shows non-monotonic recovery pattern

---

### Parameter Updates (2026-02-17)

#### 1. Grid Resolution: 25x25 to 50x50
- 4x compute cost, but reveals H1 features missed at coarser resolution

#### 2. Retention Metric: ret@10k to ret@100
- Old metric had near-zero variance (7/8 architectures at 0.0%)
- ret@100 provides much better spread for correlation analysis

#### 3. Dashboard Re-run Capability
- Per-phase re-run buttons, "Re-run All P2/P3" in header

#### 4. Randomized Landscape Seeds
- Phase 2 generates random seed per run (logged in topology_summary.json)
- 5 independent slices per architecture

#### 5. Phase 2 Performance Optimizations
- Mixed precision (AMP), GPU-resident test set, row-wise incremental perturbation
- Roughly 2-2.5x faster

#### 6. Baseline Metrics Robustness
- Each metric runs independently (one failure does not block others)
- Hessian batch capped at 64 samples

### Parameter Updates (2026-02-20)

#### 7. Multi-slice Seed Fix
- `set_seed(cfg["seed"] + run_offset)` produces 5 genuinely unique seeds

#### 8. Early Eval Steps
- Updated to [10, 25, 50, 100, 250, 500, 1000, 5000]

#### 9. Bonferroni Correction + Kendall's Tau
- 12 metrics, Bonferroni-corrected p-values reported

#### 10-13. Various fixes
- Landscape NaN/Inf validation
- Task B learning check
- Phase 2b multi-slice fallback
- Dashboard clean/rebuild, re-run buttons

### Parameter Updates (2026-02-20, session 2)

#### 14. CIFAR-10 Removed
- Floor effect, no statistical variance

#### 15. CUB-200-2011 Added
- 200 bird species, fine-grained, auto-download from Caltech

#### 16. NWPU-RESISC45 Added
- 45 satellite scene classes, cross-domain validation

#### 17. WRN-28-k Width Ladder (k=1,2,4,6,8)
- Same architecture, same depth, varying only width (0.4M to 36.5M params)

#### 18. Dashboard 3-Dataset Support
- 3-dataset selector, 19 experiments per dataset

#### 19. Phase 4 WRN Ladder Analysis
- Within-ladder Spearman, partial correlations, slice robustness diagnostics

### Parameter Updates (2026-02-21)

#### 20. Phase 2c: Cubical Persistent Homology
- GUDHI CubicalComplex on existing loss grids
- H1 agreement with Ripser: rho = 1.0

#### 21. Phase 3 EWC and Cosine LR Variants
- EWC with diagonal Fisher, cosine LR schedule

#### 22. Phase 5: Predictive Model with LOAO CV
- 5 models (A/A2/B/C/D), permutation test, matched-dimensionality control
- Fixed alpha selection: alpha chosen once on real data, reused for permutations

#### 23. Phase 4 Enhancements
- Cubical metrics, early_aurc, ret@10, slice robustness, EWC benefit analysis

#### 24. Dashboard Updates
- Phase 2c, 3 EWC, 3 cosine added
- Run Predictive button

---

### Next Steps
- **Run all 19 architectures on RESISC-45** (3rd domain, cross-domain generalization)
- **Multi-seed runs** for confidence intervals on the CUB-200 ret@10 finding
- **Scale to 30+ architectures** for more statistical power (target: p < 0.01)
- **Characterize task-difficulty boundary** (when does topology start mattering?)
- **Prototype forgetting risk API**
- **ArXiv publication + NeurIPS/ICML submission**

### Known Issues
- **Param count confound on CIFAR-100:** rho = -0.76 dominates everything. Topology redundant on easy tasks.
- **CUB-200 finding is ret@10 only:** ret@100 and early_aurc not significant. The signal is in early forgetting.
- **H1 does not survive Bonferroni on CIFAR-100:** p_Bonf = 0.50 with 12 tests
- Barrier metric overflows for large models (clamped at 1e6)
- Hessian trace goes negative for some models (saddle point)
- MLP-Mixer challenges topology-retention hypothesis (moderate H1, zero retention)

---

## Methodology Notes

### Loss Landscape Sampling
- 2D slice via filter-normalized random directions (Li et al., 2018)
- 50x50 grid, range [-1, 1] around converged weights
- Randomized landscape seed per run (logged in summary for reproducibility)
- Evaluated on test set (pre-loaded to GPU, mixed precision forward passes)
- 5 independent random slices per architecture

### Persistent Homology
- **Ripser (graph-based):** Sublevel set filtration, 8-connected grid, sparse distance matrix
- **GUDHI (cubical):** CubicalComplex on raw loss grids, sublevel set filtration
- H0 = connected components, H1 = loops
- Cross-method H1 agreement: rho = 1.0

### Forgetting Measurement
- Train Task B for 5,000 steps
- Evaluate Task A accuracy at steps: 0, 10, 25, 50, 100, 250, 500, 1000, 5000
- Three conditions: naive, EWC (lambda=1000), cosine LR
- Primary metrics: ret@100, ret@10, early_aurc (0-500), full AURC

### Statistical Analysis
- Spearman rank correlation + Kendall's tau
- Bonferroni correction (12 tests, adjusted alpha = 0.004)
- Partial correlation controlling for parameter count
- Symmetric partial correlations + rank regression + VIF
- Permutation test: 1,000 shuffles (topology columns only)
- Leave-one-architecture-out CV with nested alpha selection
- Matched-dimensionality control (1,000 random feature draws)

### Predictive Model (Phase 5)
- **Model A:** retention ~ params (baseline)
- **Model B:** retention ~ params + H0_rip + H1_rip (Ripser topology)
- **Model C:** retention ~ params + H0_cub + H1_cub (cubical topology)
- **Model D:** retention ~ H0_rip + H1_rip (topology alone)
- Ridge regression with LOAO CV (all slices from held-out arch in test fold)
- Alpha selected once via nested LOO, reused for permutation iterations
