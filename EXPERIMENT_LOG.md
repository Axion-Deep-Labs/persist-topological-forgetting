# Experiment Log

## Submission Artifacts

| Venue | Date | Commit SHA | Tag | Artifact |
|-------|------|------------|-----|----------|
| CoLLAs 2026 | 2026-04-15 | `7d88806` | `collas-2026-submitted` | `paper/collas/main.pdf` |

---

## EXP-04: Topological Dynamics of Grokking

### Current State (2026-03-26)

**Pilot in progress. 3 critical bugs fixed. All seeds need re-analysis.**

---

### Calibration Sweep (2026-03-25)

Tested weight decay [0.01, 0.03, 0.1, 0.3] on mod-97 addition, 1-layer transformer decoder (302K params), 100K steps.

| WD | Memorization | Generalization | Delay | Accepted |
|----|-------------|----------------|-------|----------|
| 0.01 | Step 500 | Never | -- | NO (pure memorization) |
| **0.03** | **Step 500** | **Step 70,500** | **70K** | **YES (optimal)** |
| 0.1 | Step 500 | Step 13,000 | 12.5K | YES (too fast) |
| 0.3 | Step 500 | Step 4,000 | 3.5K | YES (way too fast) |

Selected WD=0.03 for pilot (longest delay = most pre-grokking data).

---

### Pilot Runs (2026-03-25 to 2026-03-26)

| Seed | Training | Topology | Baselines | Grokking Onset |
|------|----------|----------|-----------|----------------|
| 42 | Done | Done* | Needs rerun | Step 40,000 |
| 137 | Done | Done* | Needs rerun | Step 42,000 |
| 256 | Done | Done* | Needs rerun | Step 80,500 |
| 1024 | Done | Pending | Pending | (test_acc=1.0) |
| 7777 | Not started | -- | -- | -- |

*Topology data forward-compatible (new fields added) but baselines must be recomputed.

---

### Bugs Found and Fixed (2026-03-26)

**Bug 1: H0 feature count structurally constant (topology.py)**
- On a 50x50 grid (n=2500), lower-star filtration always produces exactly n-1 = 2499 finite H0 bars. This is a topological invariant of the grid graph, independent of function values.
- **Fix:** Added `h0_significant_count` (H0 features with persistence > median) and `h0_median_persistence`. Changed primary endpoint from `h0_feature_count` to `h0_total_persistence`.
- **Impact:** Primary endpoint was dead by construction. New endpoint shows 700x dynamic range across training (seed 256: 14K to 10.2M).

**Bug 2: Commutator defect always zero (baselines.py)**
- Two independent bugs:
  - (a) Full-batch training gave 1 batch to the dataloader. The function needed >= 2 batches and returned 0. **Fix:** Collect all data, create synthetic 50/50 random splits for sub-batch pairs.
  - (b) Hessian-vector product lacked `.detach()` on vector argument. `dot_ab = sum(ga * gb)` is symmetric since both ga and gb have `create_graph=True`. Derivative flows through both terms, giving `d(ga.gb)/d(theta) = H_A*g_B + H_B*g_A` on both sides, making `Hab == Hba` identically. **Fix:** Detach the vector argument: `v_b = [gb.detach() for gb in grad_b]`.
  - **Note:** Bug (b) was present in the original code and would have produced zeros even with multiple batches.
- **After fix:** Commutator defect shows clear dynamics: 2.68 (step 0) -> 25,942 (step 30K, pre-grokking peak) -> 327 (step 40K, post-onset) -> 0 (step 50K) -> 1,262 (step 80K, late instability).

**Bug 3: H1 essentially dead**
- 50x50 grid is too coarse for meaningful 1-cycle detection. Max H1 feature count across all seeds: 1.2 (averaged across slices). Only ~10/81 checkpoints had any H1 features.
- **Fix:** Demoted H1 endpoints to exploratory in config and DESIGN.md. Future option: increase grid to 100x100 (4x compute per checkpoint).

---

### Preliminary Signal Assessment (Pre-Fix Data, Informative for Topology Only)

**H0 total persistence (new primary):**
- Seed 42 (onset 40K): 14K -> 37K (slow rise, memorization) -> 26K (grokking) -> 2.3M (post-grokking explosion)
- Seed 137 (onset 42K): 14K -> 39K (memorization) -> 33K (grokking) -> 4.3M (post-grokking)
- Seed 256 (onset 80.5K): 14K -> 39K -> 10.2M (massive pre-grokking explosion at 46K) -> 1M (onset) -> 355K (post)
- Signal: H0 persistence shows massive dynamic range (700x) but timing relative to onset varies. Seed 256's explosion precedes grokking. Seeds 42/137 explode after.

**H0 persistence entropy:**
- Consistent slow decrease across all seeds: 7.816 -> ~7.65-7.72
- Represents increasing dominance of a few large persistence bars

**Sharpness (Hessian trace):**
- Strongly negative during memorization (saddle-dominated landscape)
- Transitions positive during pre-grokking period
- Returns to near-zero post-grokking
- Shows clear pre-onset dynamics

**Commutator defect (after fix):**
- Spikes during pre-grokking transition, drops post-onset
- Need full re-analysis to confirm cross-seed consistency

---

### Next Steps
1. Re-analyze all seeds: `.venv/bin/python -m experiments.exp04_grokking_topology.run_pilot --config configs/exp04_pilot.yaml --skip-training`
2. Run seed 7777 (full pipeline)
3. Evaluate pilot gate: consistent directional PH behavior in >= 3/5 seeds before onset
4. If gate passes: full study (30 seeds x 3 WD = 90 runs, ~12 days on RTX 4090)

---

## EXP-01: Topological Persistence

### Current State

**Preliminary (57/57 complete). Phase I scale validation in progress on NMSU Discovery HPC.**

- **CIFAR-100:** 19/19 architectures, Phases 1-6 complete (preliminary)
- **CUB-200-2011:** 19/19 architectures, Phases 1-6 complete (preliminary)
- **RESISC-45:** 19/19 architectures, Phases 1-6 complete (preliminary)
- **ImageNet-100:** 8/8 valid configs complete through ALL phases (2026-04-02). Phase 4-6 analysis complete.

---

### Phase I-A: ImageNet-100 Results (2026-03-28 to 2026-04-01)

**8 of 10 configs completed all training and forgetting phases on NMSU Discovery HPC.**

| Architecture | Params | Phase 1 | Phase 2 | Phase 3 (Naive) | Phase 3 (EWC) | Phase 3 (SI) |
|---|---|---|---|---|---|---|
| ResNet-101 | ~44M | Complete | Complete | Complete | Complete | Complete |
| ConvNeXt-Small | ~50M | Complete | Complete | Complete | Complete | Complete |
| ConvNeXt-Base | ~89M | Complete | Complete | Complete | Complete | Complete |
| ConvNeXt-Large | ~198M | Complete | Complete | Complete | Complete | Complete |
| EfficientNet-B5 | ~30M | Complete | Complete | Complete | Complete | Complete |
| DenseNet-201 | ~20M | Complete | Complete | Complete | Complete | Complete |
| ViT-B/16 | ~86M | Complete | Complete | Complete | Complete | Complete |
| ViT-L/16 | ~304M | Complete | Complete | Complete | Complete | Complete |

**Dropped configs:**
- **ViT-H/14 (632M params):** SWAG pretrained weights (`vit_h_14_swag-80465313.pth`) require 518x518 input. Pipeline uses 224x224 for all ImageNet-100 configs. Error: `AssertionError: Wrong image height! Expected 518 but got 224!`. Also far outside the experiment's parameter range.
- **WRN-40-10 (~56M params):** Architecture designed for 32x32 CIFAR. Config had `img_size: 32` but ImageNet-100 data loader hardcodes 224x224 via `get_224x224_transforms()`. At 224x224, feature maps are 49x larger per layer. OOM on A100 40GB: 38.5/39.5 GB used at layer2 forward pass.

**HPC issues resolved:**
- Missing tqdm in persist-env (2026-03-28): `pip install tqdm`
- GPU targeting (2026-03-28): `--gres=gpu:1` landed on P100/V100/T4 which OOM on EWC/SI. Fixed: `--gres=gpu:a100:1`
- DependencyNeverSatisfied cascade (2026-04-01): ViT-H and WRN failures cascaded to 8 stuck downstream jobs. Cancelled and diagnosed via log files.

**Analysis pipeline bugs fixed (2026-04-01):**
1. Phase 4: `ARCH_CLASSES` dict missing all ImageNet-100 architecture entries. Added 8 new entries.
2. Phase 4: `_imagenet100` suffix not in strip list for architecture class lookup. Added.
3. Phase 5: Dataset detection fell through to "CIFAR-100" for ImageNet-100 runs. Added `_imagenet100` check.
4. Phase 6: Entire script hardcoded for exactly 3 datasets with 19 matching architectures. Generalized to N datasets with variable architecture counts, dynamic design matrix, clustered bootstrap, and permutation tests.
5. Phase 6 reduced model: Off-by-one index bug in partial effects (`beta1r[2+K+i]` should be `beta1r[3+K+i]`). Fixed.

**Phase 4 results (ImageNet-100, n=8, 2026-04-02):**
- H0 NOT significant: rho=-0.4048, p=0.3199
- **H1 SIGNIFICANT (Bonferroni): rho=0.9341, p=0.0007, p_bonf=0.0081**
- Best predictor of forgetting: H1 Persistence (|rho| = 0.9341)
- Parameter count alone does NOT predict retention (rho=0.5238, p=0.1827)
- Bug: Phase 4 output file named `correlation_results_cifar100.json` (dataset detection missing `_imagenet100`). Renamed manually on HPC. Code fix committed.

**Phase 5 results (ImageNet-100, n=8, 2026-04-02):**
- Topology does NOT improve prediction beyond params (LOAO Ridge, permutation p=0.978)
- EWC benefit: H0 rho=-0.4762 (p=0.233), H1 rho=0.5749 (p=0.136) -- not significant
- Low power caveat: only 8 architectures for LOAO cross-validation

**Phase 6 results (2026-04-02):**

*3-dataset replication (n=57, CIFAR-100 + CUB-200 + RESISC-45):*
| Outcome | dR2 | p(full) | p(interaction) |
|---------|-----|---------|----------------|
| **EWC Benefit (AURC)** | 0.085 | **0.046** | **0.046** |
| **Retention @ 100** | 0.127 | **0.035** | **0.035** |
| Retention @ 10 | 0.075 | 0.196 | 0.196 |
| EWC Benefit (ret@10) | 0.002 | 0.984 | 0.984 |

Exact replication of original Phase 6 results.

*4-dataset exploratory (n=65, + ImageNet-100):*
| Outcome | dR2 | p(full) | p(interaction) |
|---------|-----|---------|----------------|
| EWC Benefit (AURC) | 0.081 | 0.063 | 0.063 |
| Retention @ 100 | 0.005 | 0.051 | 0.051 |
| Retention @ 10 | 0.047 | 0.219 | 0.219 |
| SI Benefit (AURC) | 0.192 | 1.000 | 1.000 |
| SI Benefit (ret@10) | 0.098 | 1.000 | 1.000 |

Adding ImageNet-100 dilutes signal slightly (EWC benefit p: 0.046 -> 0.063). SI shows zero topology moderation. Reduced model EWC benefit still significant (p=0.030) with 4 datasets.

**Interpretation:**
1. Original 3-dataset signal replicates perfectly
2. ImageNet-100 Phase 4 reveals H1 (not H0) as dominant predictor at larger scale -- novel finding
3. 4-dataset pooling shows expected attenuation, not contradiction
4. SI finding is EWC-specific (SI shows no topology moderation)
5. Low power (n=8) limits ImageNet-100-specific conclusions

**Additional bug fixed (2026-04-02):**
6. Phase 4: Dataset detection for output filename missing `_imagenet100` (same pattern as Phase 5). Fixed.

---

### Phase I: HPC Setup (2026-03-23)

**NMSU Discovery cluster access established.**

- **Account:** cag1145 (Crystal Gutierrez, NMSU affiliation), approved 2026-03-16, ID 4505765
- **GPU verified:** NVIDIA A100-PCIE-40GB (SLURM job 515171 on discovery-g13)
- **Environment:** `/fs1/scratch/cag1145/persist-env` — PyTorch 2.5.1+cu121, Python 3.10.8, ripser, gudhi, scikit-learn
- **Storage:** 100GB home + 1TB scratch, no compute hour quota
- **VPN:** Cisco Secure Client required (`vpn.nmsu.edu`, SAML/SSO + Duo MFA). `openconnect` does NOT work.
- **SLURM:** Partition `normal` (not `gpu`), GPU nodes are `discovery-g*`, request via `--gres=gpu:1`
- **Module loads:** `os/rhel_8 → spack/2023a → gcc/12.2.0 → python/cuda` on login node only. Compute nodes don't need module loads — venv is self-contained.
- **Configs adapted:** `run_experiment.sh` updated for Discovery (partition=normal, time=24h, venv path)
- **Next steps:** Clone repo to `/fs1/scratch/cag1145/`, download ImageNet-100, submit first batch via `submit_all.sh`

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

### RESISC-45 Results (n=19, Hard Satellite Scene Classification)

#### Phase 4 Correlation (n=19, RESISC-45)
- **Parameter count:** rho = -0.29, p = 0.22 (NOT significant)
- **H0 persistence:** rho = 0.44, p = 0.059 (marginal)
- **H1 persistence:** rho = 0.17, p = 0.48 (not significant)
- **Conclusion:** Like CUB-200, parameter count fails on this hard task. Unlike CUB-200, topology also provides no direct predictive signal for retention.

#### Phase 4 WRN Width Ladder (n=6, RESISC-45)
- H0 monotonic with width (rho = -1.0 vs params, consistent across all 3 datasets)
- H0 vs retention: rho = 0.32, p = 0.54 (not significant)

#### Phase 4 EWC Benefit (n=19, RESISC-45) -- STRONGEST SIGNAL
- H0 vs EWC benefit: rho = 0.86, p = 2.4e-6 (highly significant)
- Params vs EWC benefit: rho = -0.54, p = 0.018
- H1 vs EWC benefit: rho = 0.05, p = 0.84 (not significant)
- **Cross-dataset replication:** Matches CIFAR-100 (rho = 0.76, p = 0.0002)

#### Phase 4 Cubical vs Ripser Agreement (RESISC-45)
- H1: rho = 1.0 (perfect agreement, consistent across all 3 datasets)

#### Phase 5 Predictive Model (n=19, RESISC-45)

| Outcome | Params-only rho | Params+Topo rho | Perm. p | Verdict |
|---|---|---|---|---|
| ret@100 | -0.32 | -0.33 | 0.566 | Not significant |
| ret@10 | -0.89 | -0.95 | 0.628 | Not significant |
| Early AURC | -0.32 | -0.44 | 0.743 | Not significant |

**RESISC-45 summary:** Topology does not add significant predictive value for retention. However, H0 strongly predicts EWC benefit (rho = 0.86), replicating the CIFAR-100 finding.

---

### Cross-Dataset Summary (57/57 Complete)

#### Predictive Model Comparison

| Dataset | Task Difficulty | Params-only rho | +Topology rho | Perm. p | Verdict |
|---|---|---|---|---|---|
| CIFAR-100 | Easy | 0.43 | 0.30 | 0.295 | Topology redundant |
| **CUB-200** | **Hard (fine-grained)** | **-0.92** | **0.34** | **0.037** | **Topology rescues** |
| RESISC-45 | Hard (satellite) | -0.89 | -0.95 | 0.628 | Topology ineffective |

**Note:** CUB-200 p=0.037 does NOT survive Bonferroni correction across 3 datasets (adjusted alpha = 0.0167).

#### EWC Benefit: Most Stable Cross-Dataset Signal

| Dataset | H0 vs EWC benefit rho | p-value | Params vs EWC benefit rho | p-value |
|---|---|---|---|---|
| CIFAR-100 | 0.76 | 0.0002 | -0.74 | 0.0003 |
| **RESISC-45** | **0.86** | **2.4e-6** | -0.54 | 0.018 |
| CUB-200 | 0.31 | 0.19 | -0.40 | 0.09 |

H0 topology predicts how much EWC regularization helps on 2 of 3 datasets. This suggests topology is a **mitigation sensitivity marker**: it indicates which architectures will benefit most from continual learning interventions, rather than predicting raw forgetting directly.

#### Revised Interpretation

1. **Topology is not a universal forgetting predictor.** It rescues prediction on CUB-200 (fine-grained) but not RESISC-45 (satellite scenes), despite both being hard tasks.
2. **The CUB-200 finding is task-specific, not difficulty-general.** Fine-grained discrimination (distinguishing 200 bird species) creates a forgetting regime where loss landscape geometry matters. Satellite scene classification does not.
3. **H0 predicts mitigation benefit, not raw forgetting.** The H0-EWC benefit correlation replicates across CIFAR-100 (rho=0.76) and RESISC-45 (rho=0.86). Loss landscape connectivity predicts how much regularization helps.
4. **WRN H0 monotonicity is universal.** H0 decreases perfectly with width (rho=-1.0 vs params) across all 3 datasets. The topological measurement is reliable; its relationship to forgetting is task-dependent.

#### Phase 6: Pooled Interaction Analysis (n=57, OLS + Clustered Bootstrap)

Formal test of dataset moderation via pooled regression with interaction terms.
CIFAR-100 as reference. H0 z-scored within dataset. Clustered bootstrap (5,000 iterations, 19 architecture blocks). Permutation tests (1,000 iterations, H0 shuffled within dataset, two-tailed).

**Models:**
- M0: Y ~ log_params + dataset + log_params x dataset
- M1: Y ~ log_params + H0z + dataset + log_params x dataset + H0z x dataset

##### A. Forgetting Prediction Moderation

| Outcome | M0 R2 | M1 R2 | dR2 | Full block p | Interaction p |
|---|---|---|---|---|---|
| **ret@10 (primary)** | 0.179 | 0.254 | 0.075 | 0.196 | 0.196 |
| ret@100 (robustness) | 0.297 | 0.423 | **0.127** | **0.035** | **0.035** |
| Early AURC (robustness) | 0.228 | 0.313 | 0.085 | 0.138 | 0.138 |

Per-dataset H0z partial effects on ret@10 (95% clustered bootstrap CIs):
- CIFAR-100: -0.001 [-0.486, +0.073] (includes zero, no effect)
- **CUB-200: -0.123 [-0.183, -0.046]** (excludes zero, topology matters)
- RESISC-45: -0.021 [-0.264, +0.083] (includes zero, no effect)

**Interpretation:** The primary outcome (ret@10) block test lacks power at n=57, but the CUB-200 partial effect CI clearly excludes zero while CIFAR and RESISC do not. The robustness check on ret@100 is significant (p=0.035). Dataset moderates H0's effect on forgetting.

##### B. EWC Benefit Moderation

| Outcome | M0 R2 | M1 R2 | dR2 | Full block p | Interaction p |
|---|---|---|---|---|---|
| **EWC benefit (early AURC)** | 0.502 | **0.587** | **0.085** | **0.046** | **0.046** |
| EWC benefit (ret@10) | 0.083 | 0.085 | 0.002 | 0.984 | 0.984 |

Per-dataset H0z partial effects on EWC benefit AURC (95% clustered bootstrap CIs):
- **CIFAR-100: +0.016 [+0.005, +0.062]** (excludes zero, positive)
- CUB-200: +0.002 [-0.008, +0.013] (includes zero, no effect)
- **RESISC-45: +0.007 [+0.004, +0.012]** (excludes zero, positive)

**Interpretation:** Dataset significantly moderates the H0-EWC benefit relationship (p=0.046). H0 predicts EWC benefit on CIFAR-100 and RESISC-45 (CIs exclude zero) but not CUB-200. This formalizes the "2 of 3 replicate" observation.

##### VIF Check
All VIFs below 3.5 for z-scored model. Raw H0 sensitivity shows VIF up to 25 (expected due to scale differences across datasets) but identical inference (same p-values).

##### Robustness: Reduced Model (no log_params x dataset)
- ret@100: p=0.031 (consistent)
- EWC benefit AURC: p=0.031 (consistent)
- H0 interaction conclusions unchanged

##### Formal Statement
"Dataset significantly moderates the topology-EWC benefit relationship (permutation p=0.046), with H0 predicting EWC benefit on CIFAR-100 and RESISC-45 (CIs excluding zero) but not CUB-200. For forgetting prediction, H0's effect is concentrated on CUB-200 (CI excludes zero) with the ret@100 block test reaching significance (p=0.035)."

---

### Manuscript Elements

#### 1. Analysis Timeline and Decision Gates

The study was designed as a three-dataset cross-architecture survey from the outset (57 configurations: 19 architectures x 3 datasets). The original hypothesis was that persistent homology features of loss landscapes (H0 and H1) predict resistance to catastrophic forgetting. Retention at step 10 (ret@10) was pre-specified as the primary outcome for forgetting prediction, with ret@100 and early AURC as robustness checks.

CIFAR-100 was run first and showed that parameter count dominates (rho = -0.76, survives Bonferroni) with topology adding no significant predictive value (Phase 5 permutation p = 0.295). This was the expected null for easy tasks. CUB-200 was run second and showed that parameter count fails (rho = -0.27, not significant) while topology rescues prediction (permutation p = 0.037). These two results were consistent with the pre-specified hypothesis that topology matters when scale alone is insufficient. RESISC-45 was run third and returned a null result for topology (p = 0.566), despite also being a hard task. This falsified the simpler "topology helps on hard tasks" framing.

The EWC benefit analysis (H0 predicting how much EWC regularization helps) was computed as part of Phase 4 diagnostics, not as the original target hypothesis. The shift from "topology predicts forgetting" to "topology predicts mitigation benefit" emerged from the data after observing the RESISC-45 null. The Phase 6 pooled interaction model was designed post hoc to formalize the cross-dataset moderation pattern. We report this analysis path transparently: the EWC moderation finding (p = 0.046) should be interpreted as a data-driven discovery requiring pre-registered replication, not as a confirmatory result.

#### 2. Proposed Mechanism: Basin Fragmentation and Regularization Sensitivity

H0 in persistent homology counts connected components in the sublevel set filtration of the loss landscape. A high H0 count indicates a fragmented landscape with many disconnected basins at low loss values, while a low H0 count indicates a smooth landscape with few basins.

We propose the **basin fragmentation hypothesis**: H0 measures the degree of loss landscape fragmentation, which determines how much curvature-based regularization can help by preventing inter-basin drift during sequential training.

When a landscape has many disconnected basins (high H0):
- Naive sequential training is likely to push parameters out of the current basin into a different one, causing catastrophic forgetting
- EWC penalizes movement away from the current optimum, weighted by Fisher information (local curvature), keeping the model within its basin
- The benefit of this penalty is large because without it, the model would drift between basins

When a landscape has few basins (low H0, smooth landscape):
- There is effectively one broad basin; naive training perturbs weights but does not cross basin boundaries
- EWC provides little additional benefit because the model stays in the same basin regardless
- The penalty is wasted on a problem that does not exist

This is consistent with our data: H0 predicts EWC benefit on CIFAR-100 (rho = 0.76) and RESISC-45 (rho = 0.86), both datasets where EWC produces measurable variance in retention. The CUB-200 null for EWC benefit (rho = 0.31, p = 0.19) may indicate that fine-grained discrimination creates a forgetting regime driven by feature-level interference rather than parameter-level basin drift, a mechanism that EWC's Fisher-weighted penalty cannot address.

The WRN width ladder provides supporting evidence: H0 decreases perfectly with width (rho = -1.0 vs params) across all three datasets, consistent with wider networks having smoother landscapes with fewer disconnected basins. This measurement is robust (validated by cubical persistent homology, rho = 1.0 agreement with Ripser across all datasets).

This mechanism is tentative. We do not claim to have proven that basin fragmentation causes the observed relationship. A causal test would require intervening on landscape topology (e.g., via landscape-aware regularization) and measuring the effect on EWC benefit.

#### 3. External Validity and Limitations

**What we claim:**
- Dataset significantly moderates the relationship between loss landscape topology (H0) and EWC benefit (Phase 6 permutation p = 0.046)
- H0 partial effects on EWC benefit exclude zero on CIFAR-100 and RESISC-45 but not CUB-200
- On CUB-200 specifically, topology provides the only predictive signal for early forgetting (ret@10 CI excludes zero), but this finding does not survive cross-dataset multiplicity correction

**What we do not claim:**
- That topology universally predicts forgetting. It does not (RESISC-45 null).
- That the EWC moderation finding is confirmatory. It emerged from exploratory analysis and requires pre-registered replication.
- That the basin fragmentation mechanism is established. It is a plausible interpretation consistent with the data, not a tested causal hypothesis.

**Scope limitations:**
- **Sample size:** 19 architectures provide moderate statistical power. The WRN width ladder (6 points) controls for architecture family but has limited degrees of freedom for within-ladder inference.
- **One mitigation method:** Only EWC was tested. If H0 does not predict benefit under alternative mitigation strategies (Synaptic Intelligence, PackNet, progressive neural networks), the finding is EWC-specific rather than topology-general. Testing at least one additional mitigation method is the critical next experiment.
- **Three datasets:** Good domain diversity (natural images, fine-grained classification, satellite remote sensing) but not exhaustive. The CUB-200 forgetting prediction signal and the CUB-200 EWC benefit null could both be artifacts of this specific dataset's structure.
- **2D landscape projections:** Persistent homology is computed on 2D cross-sections of a high-dimensional loss landscape. These projections are inherently stochastic (5 independent slices per architecture mitigate but do not eliminate sampling variance). Topological features of the full landscape may differ.
- **p-value thresholds:** EWC moderation p = 0.046 and forgetting ret@100 p = 0.035 are borderline. With different random seeds or bootstrap samples, these could shift above 0.05.

**Falsification targets:**
1. If Synaptic Intelligence benefit shows no H0 correlation on CIFAR-100 or RESISC-45, the mechanism is EWC-specific
2. If adding 10+ architectures eliminates the CUB-200 ret@10 signal (CI crosses zero), the forgetting prediction claim fails
3. If a landscape intervention (e.g., sharpness-aware minimization) changes H0 without changing EWC benefit, the causal link is broken
4. If cubical persistence (full grid, not subsampled) disagrees with Ripser-based H0 on the moderation result, the topological measurement is method-dependent

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

### Preliminary Complete — Phase I Scale Validation Roadmap

The preliminary proof-of-concept ("petri dish") established that the topological signal exists on small-to-medium models (0.3M-44.7M parameters). Phase I addresses the fundamental open research questions that require supercomputer resources.

**Phase I Research Questions (requires supercomputer allocation):**
- **Scale survival:** Does the topological signal persist on 100M-7B+ parameter models, or is it a small-model artifact? This is an empirically open question — nobody knows.
- **PH tractability at scale:** Ripser complexity is O(n^3) in simplex count. Computing PH on large parameter spaces may require novel distributed algorithms (itself a research contribution).
- **Subsampling fidelity:** Do 5 random 2D slices capture relevant topology when parameter dimensionality is 10^8-10^10? There may be a phase transition where subsampling destroys the signal.
- **Higher-dimensional homology:** H2, H3 may carry critical information but are exponentially more expensive to compute.
- **Long task sequences:** 10-100+ sequential tasks (vs current 2-task). Does topology at task boundary N predict forgetting at task boundary N+47?
- **Multiple CL methods:** SI, PackNet, replay, adapter-based methods. Does topology predict which method works best for a given landscape?
- **Large-scale datasets:** ImageNet (1.4M images), NLP tasks, medical imaging. Current datasets are all small-image.
- **Foundation model fine-tuning:** Predict catastrophic forgetting when fine-tuning LLMs on sequential downstream tasks. Commercial killer app requiring thousands of GPU-hours per run.
- **50-100+ architectures:** Statistical power for robust claims that survive strict multiple-comparison correction.

**Genuine failure modes:**
- Topological signal vanishes at scale (small-model artifact)
- PH computation scales worse than training itself (impractical tool)
- Long task sequences show chaotic topology evolution (defeats prediction)
- Signal is EWC-specific and does not generalize to other CL methods
- Subsampling loses fidelity in high-dimensional parameter spaces

**CL methods to test in Phase I (priority order):**
- **Synaptic Intelligence (SI):** Regularization-based like EWC but tracks weight importance during training instead of after. First method to test because if SI also shows the H0 correlation, the finding generalizes to regularization-based methods broadly, not just EWC's Fisher information.
- **PackNet:** Prunes and freezes weights after Task A, gives Task B only leftover capacity. Hard partitioning, no penalty. If H0 predicts PackNet benefit, the finding is deeper than regularization.
- **Replay:** Saves a buffer of Task A examples and mixes them into Task B training. Mechanically unrelated to landscape geometry. If H0 still predicts replay benefit, basin fragmentation may be the wrong explanation even if the correlation is real.
- **Adapters:** Freezes pretrained weights entirely, adds small trainable modules per task. Different question: does topology predict adapter efficiency?

**Publication:**
- ArXiv paper drafted (preliminary results), pending review before upload
- Phase I results would target NeurIPS/ICML main conference

### Phase II Vision (Contingent on Phase I Success)

**Goal:** Build a practical tool that prevents catastrophic forgetting using topology as guidance.

Phase I answers "is this real at scale?" If yes, Phase II turns the diagnostic into an intervention.

**Core idea:** Given a pretrained model and a sequence of tasks, run the topology diagnostic on the loss landscape, and use the result to automatically select and configure the right mitigation strategy. A topology-guided continual learning system that prevents forgetting rather than just predicting it.

**Possible directions:**
- **Topology-aware regularization:** Use H0 to set EWC/SI penalty strength automatically. Fragmented landscape gets stronger penalty.
- **Method selection:** Topology profile recommends which CL method to apply (regularization vs replay vs architectural) based on landscape shape.
- **Landscape-aware training:** Modify the training process itself to reshape the landscape before sequential learning, reducing fragmentation proactively.
- **Automated CL pipeline:** End-to-end system where practitioners provide a model and task sequence, and the tool handles mitigation selection, tuning, and monitoring.

**Open question for Dr. Cao:** Should Phase II prioritize building the tool (engineering) or formalizing the theory first (proving basin fragmentation mathematically)?

### Known Issues
- **CUB-200 p=0.037 does not survive Bonferroni** across 3 datasets (adjusted alpha = 0.0167)
- **Param count confound on CIFAR-100:** rho = -0.76 dominates everything. Topology redundant on easy tasks.
- **CUB-200 finding is ret@10 only:** ret@100 and early_aurc not significant. The signal is in early forgetting.
- **RESISC-45 topology signal absent:** Despite being a hard task, topology does not predict forgetting here.
- **CUB-200 EWC benefit signal weak:** H0 vs EWC benefit rho = 0.31, p = 0.19 (does not replicate CIFAR/RESISC)
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
