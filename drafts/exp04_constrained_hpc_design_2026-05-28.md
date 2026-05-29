# EXP-04 Constrained HPC Study — Design

**Drafted:** 2026-05-28
**Status:** Draft — pre-submission checklist NOT yet cleared (queue/GPU confirmation pending)
**Config:** `configs/exp04_full_study.yaml`
**SLURM:** `slurm/run_exp04_train.sh`, `slurm/run_exp04_topology.sh`

---

## 1. Why this study exists (honest pilot framing)

The pilot outcome was **MIXED — it did NOT establish a positive result.**

- **PRIMARY** endpoint `h0_total_persistence`: **NEGATIVE** (consistent in only 2/3 in-scope seeds).
- **SECONDARY** endpoints: **POSITIVE** — `h0_effective_feature_count` and `h0_persistence_entropy` (3/3 decrease pre-onset; entropy also holds at the 20% window).
- **COMPARATOR** `commutator_defect`: **POSITIVE** (3/3 decrease).
- **Interpretation: inconclusive.** The pilot's 10% pre-onset window held only **n=3 checkpoints** (n=2 at 5%, uncomputable). That is a cadence/resolution flaw, not a real test of a temporal signal.

> The pilot did not support the pre-registered primary endpoint, but did support multiple pre-registered secondary endpoints and the comparator. It therefore **motivates a larger study rather than establishing a positive result.**

This study replicates with proper power and resolution. **What changes is the DESIGN, not the hypothesis.**

---

## 2. Frozen vs. upgraded

**FROZEN (do not touch — changing these post-pilot-failure is hypothesis drift / p-hacking):**
- Endpoint hierarchy: primary `h0_total_persistence`; secondary = effective-feature-count, persistence-entropy, median-persistence; comparator `commutator_defect`.
- Task: modular addition mod 97, 50/50 split.
- Model: 1-layer transformer decoder, d_model=128, 4 heads, 302K params.
- Onset rule: test_acc ≥ 0.90 AND train_acc ≥ 0.99.
- PH method: 50×50 grid, 5 filter-normalized slices, per-slice-then-average.

**UPGRADED (the design fixes the pilot's weaknesses):**
| Dimension | Pilot | This study |
|-----------|-------|------------|
| Seeds / condition | 5 | **30** |
| Weight-decay grid | 1 (0.03) | **4: 0.01, 0.03, 0.10, 0.30** |
| Total steps | 100K | **120K** (capture late-grokking tail at high WD) |
| Checkpoints | 81 (fine window at 80K — too late) | **151, fixed dense schedule** |
| Pre-onset window density | n=3 | dense 500-step grid through 30K–90K |

WD=1.0+ (instant/no-delayed grokking) is **deliberately excluded** — would only be added later as an explicit negative control.

---

## 3. Checkpoint schedule (FIXED, no leakage)

Three contiguous segments in **global step-space** — uses no onset information, so there is no look-ahead leakage into the pre-onset window:

```
coarse  every 2,000  over [0,      30,000]   ->  16 checkpoints
dense   every   500  over [30,000, 90,000]   -> 121 checkpoints
coarse  every 2,000  over [90,000, 120,000]  ->  16 checkpoints
                                       total  -> 151 unique / run
```

Verified: clean seams at 30K (2000→500) and 90K (500→2000); dense region uniformly 500 steps. Covers the pilot's ~42–44K onsets (WD=0.03) and the later onsets expected at higher WD, all at 500-step resolution. **Event-triggered checkpointing was explicitly rejected** (2026-05-28) — it risks leakage and complicates analysis. Implemented via the new `checkpoints.segments` form in `get_checkpoint_steps` (backward-compatible with the pilot's two-phase config).

---

## 4. Decoupled train → PH architecture

Training and topology are **separate SLURM array jobs** (confirmed feasible: `run_pilot.py` already supports `--skip-analysis`, `--skip-training`, `--seed`, `--weight-decay`, `--output-dir`; `train_seed` writes checkpoints, `run_analysis_pass` only reads them).

- **`run_exp04_train.sh`** — array 0–119, training only. Each task = one (WD, seed); writes 151 checkpoints to `results/exp04_full/wd_<WD>/seed_<SEED>/checkpoints/`.
- **`run_exp04_topology.sh`** — array 0–119, PH + baselines only. Reads those checkpoints; resume-safe (skips computed steps, incremental atomic saves). Guards against missing checkpoints (exit 2). Thread caps (OMP/BLAS=4) + `--mem=32G` mitigate the scipy/ripser RAM spike that OOM-killed the pilot re-analysis locally.

Both scripts share an identical `task_id → (WD, seed)` map (`WD_IDX = tid//30`, `SEED_IDX = tid%30`, seeds 2000–2029). Verified: 120 tasks = 4 WD × 30 seeds, all unique.

**Recovery benefit:** a failed PH task re-runs in ~1.7 h without retraining; a failed train task doesn't block PH on other seeds. Submit train first, confirm clean, then submit PH (preferred over an `afterok` dependency so one bad train seed doesn't gate all PH).

---

## 5. Resource budget (from pilot timing)

- PH ≈ **40 s/checkpoint** (5 slices) → 151 × 40 s ≈ **1.7 h/run** + baselines.
- 120 runs → ~**205 GPU-h of PH**; at 20 concurrent (QOS cap) ≈ 6 waves ≈ **~10–12 h wall**.
- Training: full-batch tiny model, 120K steps — bounded by walltime header (8 h, conservative).
- Storage: 151 ckpts × ~1.2 MB ≈ 180 MB/run × 120 ≈ **~21.6 GB** (1 TB scratch — fine).

---

## 6. Analysis plan (locked)

- **Stratify by WD first, pool second.** Per-WD directional test is primary; pooled analysis is secondary and must treat WD as a stratum (no silent cross-regime pooling). With n=30/stratum the pre-onset directional test has real power.
- **Primary test:** directional consistency of `h0_total_persistence` in the pre-onset window, **per WD**.
- **Regime / phase-structure discovery is EXPLORATORY** unless it repeats across WD values.
- **No-onset runs** (likely some/all of WD=0.01 within 120K): report as a separate slow/no-grokking regime. Do **not** impute an onset or silently drop them.
- Acknowledge secondary + WD-strata multiple-testing exposure; report effect sizes with bootstrap CIs.

---

## 7. Pre-submission checklist

| # | Check | Status |
|---|-------|--------|
| 1 | **Queue availability** on `normal`/A100 | ⬜ PENDING — confirm on VPN |
| 2 | **EXP-01 still consuming the GPU slot?** (Phase I-B re-eval was SLURM 527670, submitted 2026-04-21 — likely long done, but verify) | ⬜ PENDING — confirm on VPN |
| 3 | **PH separable from training?** | ✅ CONFIRMED from code — decoupled into two array jobs (§4) |

Cannot reach the cluster non-interactively from the dev box (needs Cisco VPN + Duo). **Run these two once on the VPN:**

```bash
# 1 + 2: my running/queued jobs (is anything from EXP-01 still active?) and partition headroom
squeue -u cag1145 -o "%.10i %.20j %.8T %.10M %.6D %R"
sinfo -p normal -o "%.12P %.6a %.10l %.6D %.6t %C"   # %C = Allocated/Idle/Other/Total CPUs; check idle A100 nodes
```

If EXP-01 jobs are still running and near the QOS cap (20), either wait or lower this study's `%20` throttle so the two don't contend.

---

## 8. Submission sequence (on HPC, after checklist clears)

```bash
cd /fs1/scratch/cag1145/axiondeep-research
git pull                                          # bring config + slurm scripts + train.py change

# 1) Train all 120 runs (4 WD x 30 seeds)
sbatch slurm/run_exp04_train.sh
squeue -u cag1145                                 # monitor

# 2) After the train array finishes cleanly, run PH
sbatch slurm/run_exp04_topology.sh
```

Spot-check after training: confirm `results/exp04_full/wd_*/seed_*/checkpoints/` each hold 151 `step_*.pt` before launching PH.

---

## 9. Open risks

- **WD=0.01 may never grok within 120K** → no onset for those runs. Handled by §6 (separate regime), but it means the 0.01 stratum may contribute to regime contrast, not the pre-onset directional test.
- **Primary may fail again.** That is an acceptable, pre-committed outcome: a primary failure at n=30 with dense checkpoints is a **real negative result**, not a prompt to swap endpoints.
- **QOS cap=20** shared with any concurrent EXP-01 work (checklist #2).
