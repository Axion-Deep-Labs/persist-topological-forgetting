# EXP-04 Pilot Correctness Report

**Date:** 2026-04-24
**Scope:** Seeds 42, 137, 256, 1024 (7777 quarantined, checkpoints missing)
**Artifacts:** `results/exp04_pilot/seed_*/` — all JSONs timestamped 2026-04-04 16:16
**Method:** Step alignment, field presence, checkpoint counts, onset recomputation, log-space discontinuity scan

## Decision

**REPAIR LOCALLY BEFORE SCALING.** Two items block HPC design; one item blocks pre-registration.

## Summary table

| Check | seed 42 | seed 137 | seed 256 | seed 1024 |
|---|---|---|---|---|
| Train/topo/base step alignment | ✅ | ✅ | ✅ | ✅ |
| Records (train/topo/base) | 81/81/81 | 81/81/81 | 81/81/81 | 81/81/81 |
| Checkpoints on disk | 81 | 81 | 81 | 81 |
| NaNs in any signal | 0 | 0 | 0 | 0 |
| Bugfix fields present (`h0_total_persistence`, etc.) | ✅ | ✅ | ✅ | ✅ |
| Commutator defect non-zero (bugfix #2 live) | ✅ | ✅ | ✅ | ✅ |
| Clean grokking trajectory | 🔴 | ✅ | ✅ | ✅ |
| Onset (test≥0.9 & train≥0.99) | 68K | 42K | 44K | 44K |

## Blocking issues

### 🔴 B1. `h0_significant_count` is structurally constant

Defined in the 2026-03-26 bugfix as "H0 features with persistence > median." Unique values across **all 324 records** (4 seeds × 81 steps): `{1249.0}`. This is mathematically inevitable — "count above the median of N values" is always ≈N/2 by construction (2498/2 = 1249).

The 2026-03-26 bugfix replaced one structural constant (`h0_feature_count = 2499`) with another (`h0_significant_count = 1249`). **Only `h0_total_persistence` and `h0_median_persistence` carry real signal on H0.**

**Impact on pre-registration:** primary endpoint must be selected from `h0_total_persistence` or `h0_median_persistence`. `h0_significant_count` cannot be a primary, secondary, or even meaningful exploratory variable as currently defined.

**Fix:** redefine with an absolute threshold (e.g. persistence > calibrated floor from calibration sweep) or delete the field. Code: `experiments/exp04_grokking_topology/topology.py`.

### 🔴 B2. Seed 42 has a mid-training collapse at step 64K

| step | train_acc | test_acc | h0_total_persistence | commutator_defect | sharpness |
|---|---|---|---|---|---|
| 60K | 1.00 | 0.386 | 1.28e6 | 1.35e5 | 2.22e5 |
| 62K | 1.00 | 0.632 | 1.20e6 | 7.30e4 | 1.51e5 |
| **64K** | **0.014** | **0.007** | **7.40e5** | **24.1** | **1.52e3** |
| 66K | 1.00 | 0.899 | 7.39e5 | 593 | 1.07e4 |
| 68K | 1.00 | 0.909 | 7.08e5 | 1.23e3 | 1.33e4 |

Every signal drops 1–4 orders of magnitude in a single step, then partially recovers. Pre-64K the model was approaching grokking (test_acc = 0.63 at 62K); post-collapse it re-groks at 66K onto a different landscape (h0_tp halved, sharpness 15× lower). This is a loss explosion or LR-scale event, not a natural grokking trajectory.

**Impact:** seed 42's trajectory is not comparable to 137/256/1024 and cannot be included in a cross-seed PH consistency analysis without either (a) re-running, or (b) explicit acknowledgement as a distinct dynamical regime.

**Other seeds confirmed clean of post-saturation train_acc drops** (verification query returned `[]` for 137/256/1024).

### 🟡 B3. CLAUDE.md onset values are wrong

| Seed | CLAUDE.md | Recomputed (test≥0.9 & train≥0.99) | Recomputed (test≥0.80) |
|---|---|---|---|
| 42 | 40K | 68K | 66K |
| 137 | 42K | 42K ✅ | 40K |
| 256 | 80.5K | 44K | 42K |
| 1024 | — | 44K | 44K |

CLAUDE.md's 80.5K for seed 256 corresponds to no threshold I can reconstruct (its test_acc is 0.9998 by step 46K). CLAUDE.md's 40K for seed 42 predates the 64K instability that shifted the real onset to 68K — probably copied from an earlier run.

**Impact:** doc drift. Any analysis anchored to these stale values would mis-locate the pre-grokking window. Not a data bug, but a state integrity bug.

**Fix:** update both `CLAUDE.md` (root + axiondeep-research/) after onset values are locked.

## Non-blocking observations

- **H1 remains dead**, as flagged by the 2026-03-26 bugfix. Feature-count means across seeds: 0.05, 0.09, 0.23, 0.08. Grid-resolution bump to 100×100 remains unvalidated — must be tested before relying on H1 in any scaled design.
- **Commutator defect + sharpness show log-space spikes** mostly adjacent to grokking transitions (seed 42 at 62K→64K, seed 137 at 54K→56K and 58K→60K, seed 256 at 32K→34K and 74K→76K, seed 1024 at 42K→44K and 62K→64K). These likely reflect real dynamical events at the transition but will interact with any changepoint/Pelt analysis — note for full-study design, not a correctness issue.
- **Seed 1024 grokking is sub-cadence.** All three test-accuracy sensitivity thresholds (0.80, 0.90, 0.95) fire at step 44K simultaneously — the transition is sharper than the 2K coarse cadence can resolve. Fine-cadence window currently starts at 80K (from config), but real onsets are 42K–68K, so the fine window never fires before grokking. **Fix for full study:** either trigger fine cadence by adaptive threshold (once test_acc > 0.3, drop to 500-step) or move fine window start earlier in config.
- **Sharpness goes negative** (min = –24K on seed 256). Expected mid-training at saddle points; not a sign bug per inspection of baselines.py semantics, but worth flagging in any report that consumes this field.

## Quarantined: seed 7777

- `checkpoints/` directory: empty (0 files vs. 81 for every other seed)
- `topology_metrics.json`, `baseline_metrics.json`, `training_metrics.json`: all present, 81 records each, same Apr 4 16:16 timestamp as the other seeds
- **Only possible provenances:** (a) checkpoints generated, analyzed, then deleted; (b) JSONs copied from another seed; (c) stale residue from an earlier pipeline version
- Excluded from this verification pass. Investigate after B1–B3 are resolved, or drop entirely and replace with a fresh seed for the HPC run.

## Repair plan (local, before HPC)

1. **Fix `h0_significant_count`** in `experiments/exp04_grokking_topology/topology.py` — replace the median-threshold with an absolute persistence floor calibrated from the calibration sweep data in `results/exp04_calibration/`. Re-run the 4 clean seeds with `--skip-training` to regenerate topology JSONs.
2. **Decide seed 42 disposition** — either exclude from the pilot (reducing to 3 clean seeds, still meets pilot gate of ≥3/5 by the letter) or re-train with the same seed and new optimizer stability tweaks. Flag: re-training a single seed to exclude an instability event is scientifically questionable (selection bias). Exclusion is cleaner.
3. **Update CLAUDE.md** — onset table + primary endpoint list (drop `h0_significant_count`, lock choice between `h0_total_persistence` / `h0_median_persistence` with a one-line justification).
4. **Re-run discontinuity scan** post-fix to confirm B1 repair.

Gate to HPC design: all four items above complete, verification re-run shows no structural constants in pre-reg primary set, cross-seed H0 signal visibly diverges across seeds (sanity plot).

## Pre-registration implications

| Endpoint | Pre-B1 fix | Post-B1 fix |
|---|---|---|
| `h0_total_persistence` | candidate primary | **candidate primary** — range varies 1–2 oom across seeds, no NaN |
| `h0_significant_count` | candidate primary | **disqualified** — structural constant |
| `h0_median_persistence` | candidate secondary | candidate primary — range varies 1–2 oom across seeds |
| `commutator_defect` | comparator | **comparator** — range varies 8–14 oom across seeds, live & non-zero |
| H1 anything | exploratory | exploratory (pending grid-res validation) |

Recommendation once B1/B2 are repaired: **primary = `h0_total_persistence`**, comparator = `commutator_defect`, all else secondary/exploratory. `h0_median_persistence` as pre-registered secondary gives one fallback without multiplying comparisons.
