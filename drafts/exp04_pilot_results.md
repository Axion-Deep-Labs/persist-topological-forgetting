# EXP-04: Topological Dynamics of Grokking -- Pilot Results

**Authors:** Joshua R. Gutierrez, Crystal A. Gutierrez
**Date:** 2026-04-04
**Status:** Pilot complete. Full study not recommended. Results may be folded into PERSIST paper as boundary-condition analysis.

---

## 1. Research Question

Does persistent homology of loss landscape slices provide a reliable early-warning signal for grokking? Specifically, does topology offer predictive value beyond simpler baseline metrics (weight norm, sharpness, commutator defect)?

## 2. Experimental Setup

- **Task:** Modular addition (mod 97), 50% train split
- **Model:** 1-layer transformer decoder, d_model=128, 4 heads, 302K parameters
- **Training:** AdamW, full-batch, lr=1e-3, weight_decay=0.03, 100K steps
- **Seeds:** 42, 137, 256, 1024, 7777
- **Topology:** 5 random 2D slices of the loss landscape per checkpoint, 50x50 grid, sublevel set filtration via Ripser, filter-normalized directions (Li et al., 2018)
- **Checkpoints:** Every 2,000 steps (81 checkpoints per seed, steps 0-100K with additional granularity near expected grokking)
- **Baselines:** Sharpness (Hessian trace), commutator defect, spectral concentration, weight norm (L2), generalization gap, training loss curvature, validation loss slope
- **Compute:** NMSU Discovery HPC, A100-PCIE-40GB, ~3 hours per seed

## 3. Training Dynamics

All 5 seeds memorize by step 2,000 (train_acc = 1.0). Grokking onset varies dramatically, splitting into two regimes.

| Seed | Train >= 0.99 | Test >= 0.50 | Test >= 0.90 (onset) | Test >= 0.99 | Transition width (0.10 to 0.90) |
|------|---------------|--------------|----------------------|--------------|-------------------------------|
| 42   | 2,000 | 62,000 | **68,000** | 76,000 | 26,000 steps |
| 137  | 2,000 | 40,000 | **42,000** | 46,000 | 12,000 steps |
| 256  | 2,000 | 40,000 | **44,000** | 48,000 | 10,000 steps |
| 1024 | 2,000 | 42,000 | **44,000** | 46,000 | 14,000 steps |
| 7777 | 2,000 | 68,000 | **74,000** | 81,500 | 14,000 steps |

**Two distinct groups:**
- **Early grokkers:** Seeds 137, 256, 1024 (onset 42,000-44,000)
- **Late grokkers:** Seeds 42, 7777 (onset 68,000-74,000)

**Seed 42 anomaly:** A catastrophic loss spike at step 64,000 resets train_acc to 0.014. The model re-memorizes by step 66,000 and groks at 68,000. At step 64K: sharpness drops from 150,873 to 1,522, commutator drops from 72,967 to 24, weight norm drops from 235 to 220. The spike may function as a phase transition that accelerates grokking. After achieving test_acc = 1.0 at step 80,000, seed 42 partially de-groks at step 90,000 (test_acc drops to 0.982, test_loss jumps to 0.137), never fully recovering.

## 4. Topology Results

### 4.1 h0_total_persistence (Primary Endpoint)

All seeds start at h0_tp ~ 14,350-14,600 (random initialization). After memorization, h0_tp settles at 31,000-39,000.

| Seed | h0_tp at 20K | h0_tp peak | Peak step | Peak/init ratio | h0_tp at grok onset |
|------|-------------|------------|-----------|-----------------|---------------------|
| 42   | 37,194 | 4,975,076 | 40,000 | **346x** | 708,086 |
| 137  | 38,518 | 362,461 | 100,000 | 25x | 29,120 |
| 256  | 38,391 | 47,867 | 24,000 | 3.3x | 27,312 |
| 1024 | 39,637 | 39,637 | 20,000 | 2.7x | 25,698 |
| 7777 | 38,819 | 4,987,181 | 40,000 | **342x** | 589,664 |

**Key findings:**

1. Late grokkers (seeds 42, 7777) undergo an explosive 345x growth in h0_tp between steps 22,000 and 40,000. The landscape becomes massively fragmented. Early grokkers never experience this -- they peak at 2.7-3.3x the initial value.

2. h0_tp peaks BEFORE grokking onset in 4/5 seeds (lead times: 20,000-34,000 steps). Exception: seed 137 peaks at step 100,000 due to anomalous post-grokking landscape restructuring.

3. h0_tp is declining at the moment of grokking onset in all 5 seeds. The landscape simplifies as generalization emerges.

### 4.2 Three-Phase Structure (Late Grokkers Only)

Late grokkers exhibit three distinct phases:

1. **Phase A -- Random to Memorization (steps 0-2,000):** h0_tp roughly doubles (14K to 31K) as the model memorizes the training set. Sharpness goes negative.

2. **Phase B -- Exponential Fragmentation (steps 2,000-40,000):** h0_tp grows slowly at first (1-4% per 2,000 steps), then exponentially from step 22,000 onwards (50-120% per 2,000 steps). Weight decay fights the memorized solution, fragmenting the landscape into disconnected basins.

3. **Phase C -- Simplification Toward Grokking (steps 40,000-onset):** h0_tp reverses and declines monotonically. The landscape smooths as the model transitions from memorization to a generalizing solution.

Early grokkers skip Phase B almost entirely. They grok before the landscape has time to become maximally rough.

### 4.3 Other Topology Metrics

- **h0_persistence_entropy:** Extremely stable (7.78-7.82) during memorization plateau. Drops modestly (to 7.64-7.73) during h0_tp explosion. Perfectly collinear with h0_total_persistence -- carries no independent information.
- **h0_max_persistence, h0_median_persistence:** Track h0_total_persistence exactly. No independent signal.
- **h0_significant_count:** Structurally constant at 1,249 across all seeds and steps. This is a grid artifact (50x50 grid has n-1 = 2,499 H0 features; exactly half exceed the median by construction). Uninformative.
- **H1 (1-cycles):** Essentially zero across all seeds. Maximum observed: 1.56. The 50x50 grid is too coarse to resolve loops. Confirmed dead.

### 4.4 Slice Variance

Slice variance in h0_tp peaks at the same time as h0_tp itself for late grokkers:

| Seed | Slice var peak step | Slice var peak value | Before onset? | Lead |
|------|--------------------|-----------------------|---------------|------|
| 42   | 42,000 | 8.88e10 | Yes | 26,000 |
| 137  | 99,500 | 6.68e8 | No | -- |
| 256  | 24,000 | 4.73e6 | Yes | 20,000 |
| 1024 | 20,000 | 4.35e6 | Yes | 24,000 |
| 7777 | 40,000 | 1.74e11 | Yes | 34,000 |

Late grokkers show 10,000-100,000x higher slice variance than early grokkers. The landscape becomes not just rougher but asymmetric -- different 2D slices see qualitatively different topology.

## 5. Baseline Results

### 5.1 Commutator Defect

Peaks before grokking onset in all 5 seeds (5/5 consistent):

| Seed | Peak value | Peak step | Lead time |
|------|-----------|-----------|-----------|
| 42   | 252,451 | 58,000 | 10,000 |
| 137  | 66,806 | 38,000 | 4,000 |
| 256  | 103,408 | 32,000 | 12,000 |
| 1024 | 51,746 | 42,000 | 2,000 |
| 7777 | 3,278,376 | 48,000 | 26,000 |

### 5.2 Sharpness

All seeds show negative sharpness from step 2,000 through ~20,000 (the memorized minimum is locally concave). Sharpness flips to positive at step 22,000-24,000 for all seeds. This sign flip fires at the same time regardless of when grokking occurs, representing the earliest detectable change in any metric.

### 5.3 Weight Norm (L2)

Late grokkers show massive growth (42 to 427-439, peaking at step 42,000). Early grokkers stay modest (peak 65-68).

## 6. The Decisive Comparison: Topology vs. Weight Norm

### 6.1 Head-to-Head Correlation

| Predictor | Spearman rho vs. grok onset | p-value |
|-----------|---------------------------|---------|
| h0_tp peak | 0.667 | 0.219 |
| Weight norm peak | 0.667 | 0.219 |
| Weight norm at step 40K | 0.564 | 0.322 |

h0_tp peak and weight norm peak produce **identical seed rankings** (Spearman rho = 1.000 between them). Neither achieves statistical significance at n=5.

### 6.2 Pre-Grokking Trajectory Correlation

At step 32,000 (before the earliest grokking at 42,000), both h0_tp and weight norm achieve rho = 0.97 (p = 0.005) with grokking onset. From step 34,000 onward through the grokking window, weight norm is the more stable predictor (holds rho ~ 0.97 while h0_tp drops to 0.67-0.82).

### 6.3 Within-Seed Correlation

| Seed | h0_tp vs. weight_norm (pre-onset) |
|------|----------------------------------|
| 42   | rho = 0.978 |
| 137  | rho = -0.021 |
| 256  | rho = 0.828 |
| 1024 | rho = 0.220 |
| 7777 | rho = 0.992 |

For the late grokkers where topology shows its strongest signal (seeds 42, 7777), h0_tp is nearly perfectly correlated with weight norm (rho = 0.978-0.992). Weight norm takes microseconds to compute; topology takes 56 seconds per checkpoint.

For early grokkers where h0_tp is independent of weight norm, h0_tp is also flat -- there is no topology signal to exploit.

### 6.4 Nuanced Features

All secondary topology features were tested for independent signal:

- **h0_persistence_entropy vs. grok onset:** rho = -0.97 at step 32K, but perfectly collinear with h0_tp.
- **Slice variance of h0_tp:** rho = 1.0 with weight norm peak. Completely redundant.
- **h0_tp / weight_norm ratio:** rho = 0.82 vs. grok onset, but rho = 0.9 with weight norm peak. Not independent.
- **Rate of change (d(h0_tp)/dt vs. d(WN)/dt):** Both spike at the same steps. No temporal lead for topology.
- **H1:** Noise at this grid resolution.

No topology feature provides information that weight norm does not already capture.

## 7. The Bimodal Problem

The data does not form a gradient. It splits into two discrete regimes:

| Regime | Seeds | h0_tp peak | Weight norm peak | Grok onset |
|--------|-------|-----------|-----------------|------------|
| Late grokkers | 42, 7777 | ~5,000,000 | 420-440 | 68,000-74,000 |
| Early grokkers | 137, 256, 1024 | 40,000-360,000 | 55-95 | 42,000-44,000 |

Any feature that captures this binary split automatically achieves high correlation. Neither metric discriminates meaningfully within groups. Within the early group, seed 137 has 10x the h0_tp of seeds 256/1024 but groks first -- inverting the expected relationship.

With only n=5, all correlations are driven by the two-group structure. This could represent two distinct basins of attraction with intrinsic grokking dynamics unrelated to topology.

## 8. Summary of Findings

### What is real:
1. Late grokkers undergo a transient landscape fragmentation phase (345x explosion in h0_tp) that early grokkers bypass entirely.
2. h0_tp peaks before grokking onset in 4/5 seeds.
3. The three-phase structure (memorize, fragment, simplify) is a coherent descriptive narrative for delayed grokking.
4. Slice variance reveals landscape asymmetry preceding grokking.

### What is not established:
1. Topology does not provide information beyond weight norm for predicting grokking onset. Rankings are identical (rho = 1.0 between h0_tp peak and WN peak).
2. Within-seed correlation between h0_tp and weight norm is 0.978-0.992 for the seeds where topology shows signal.
3. Baselines (commutator defect 5/5, sharpness sign flip 5/5) are more consistent across seeds than any topology metric.
4. The sharpness sign flip at step 22,000-24,000 is the earliest warning signal and does not require topology.
5. Early grokkers (3/5 seeds) show essentially no usable topology signal.

## 9. Verdict

**The pilot gate is technically met** (h0_max_persistence shows consistent increase in 4/5 seeds before grokking onset), **but the topology signal is redundant with weight norm dynamics.** At the only pre-grokking step where topology achieves strong predictive correlation (step 32K, rho = 0.97), weight norm achieves identical performance.

The honest state: we have a real descriptive pattern, but not a topology-specific contribution. The 56-second-per-checkpoint PH computation is an expensive confirmation of a microsecond scalar.

## 10. Recommendation

**Do not proceed to the full 90-run study.** The weight norm redundancy means topology cannot be the headline finding.

**Recommended use of these results:**

Fold the grokking analysis into the PERSIST (CoLLAs 2026) paper as a boundary-condition section. This strengthens PERSIST by:
- Demonstrating intellectual honesty (topology helps in continual learning but not grokking)
- Showing the authors tested generalizability and reported where the method fails
- Providing a natural contrast: topology adds unique signal in CL (where it predicts EWC benefit beyond what simpler metrics capture) but not in grokking (where weight norm suffices)

**Proposed section title:** "Boundary Condition: Topology Does Not Add Value in Grokking Dynamics"

**If a standalone publication is pursued later:** Requires dissolving the binary split (30+ seeds, multiple weight decay values) and demonstrating that topology captures landscape *structure* that scalar weight norm misses -- perhaps via weight-space persistent homology instead of loss landscape slices. This is a new experiment, not a continuation of this pilot.

## Appendix A: Bugs Fixed During Pilot

1. **H0 feature count structurally constant (2026-03-26):** On a 50x50 grid, H0 count = n-1 = 2,499 always. Replaced with h0_total_persistence as primary endpoint. Added h0_significant_count (persistence > median), which turned out to also be constant at 1,249.

2. **Commutator defect always zero (2026-03-26):** Two bugs: (a) full-batch training produced only 1 batch, so the function returned 0. Fixed: synthetic random 50/50 splits. (b) Hessian-vector product lacked .detach() on the vector argument, making the dot product symmetric (Hab == Hba by construction). Fixed: detach vector arg.

3. **H1 dead at 50x50 resolution (2026-03-26):** Maximum H1 count = 1.2 across all seeds. Grid too coarse for 1-cycles. Demoted to exploratory.

4. **OOM at step 72-73/81 on HPC (2026-04-04):** All 5 seeds killed by OOM (16GB SLURM allocation). Memory accumulated between topology slices inside compute_topology_at_checkpoint. Fixed: added gc.collect() and del between slices, bumped SLURM to 24GB. Resubmitted; all 5 seeds completed successfully.

## Appendix B: Data Locations

- **Training metrics:** results/exp04_pilot/seed_{N}/training_metrics.json
- **Topology metrics:** results/exp04_pilot/seed_{N}/topology_metrics.json
- **Baseline metrics:** results/exp04_pilot/seed_{N}/baseline_metrics.json
- **Checkpoints (HPC only):** /fs1/scratch/cag1145/axiondeep-research/results/exp04_pilot/seed_{N}/checkpoints/
- **Config:** configs/exp04_pilot.yaml
- **Code:** experiments/exp04_grokking_topology/
- **SLURM script:** slurm/run_exp04.sh
- **SLURM logs:** /fs1/scratch/cag1145/axiondeep-research/slurm/logs/519881-519885_grok.{out,err}
