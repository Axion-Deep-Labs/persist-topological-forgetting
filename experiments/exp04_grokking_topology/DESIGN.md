# EXP-04: Topological Dynamics of Grokking

## 1. Question

Does persistent homology of loss landscape slices provide a reliable early-warning signal for grokking that outperforms existing geometric predictors?

## 2. Hypothesis

H0 feature count (primary endpoint) exhibits a consistent pre-transition shift that predicts grokking onset with measurable lead time across seeds.

Secondary exploratory endpoints (H0 total persistence, H1 total persistence, H0 persistence entropy) are tracked but not part of the primary hypothesis.

## 3. Null Hypothesis

No topological feature provides earlier or more reliable prediction of grokking onset than existing geometric baselines.

## 4. Grokking Onset Definition

First step t where test accuracy exceeds 90% and training accuracy is already above 99%. This separates the generalization transition from initial learning. Sensitivity analysis at thresholds 80% and 95% to confirm results are not threshold-dependent.

## 5. Evaluation Metrics

Compare all predictors using:
- **Lead time:** Mean number of steps between predictor changepoint and grokking onset. Changepoint detection via PELT with a fixed penalty parameter across all predictors, not tuned per method. Results also reported using a simple threshold-based heuristic (2 standard deviations from pre-memorization mean) as a robustness check.
- **AUROC:** Binary classification of "within pre-grokking window" vs. "not." Pre-grokking window defined as last 10% of steps before onset. Sensitivity analysis at 5% and 20%.
- **Reliability:** Fraction of seeds where the predictor signals before onset. A predictor is considered reliable if it signals before onset in at least 70% of seeds.

**Success criterion:** A PH feature matches or exceeds the commutator defect in lead time and AUROC in at least one weight decay condition, with reliability >= 70%.

**Primary comparison:** Does H0 feature count match or exceed the commutator defect on lead time, AUROC, and reliability?

## 6. Pilot Study (5 runs, before full commitment)

Before committing full compute, run 5 seeds at weight decay = 1.0 (standard grokking regime) through the complete pipeline:
- Train to 100K steps (full-batch, 1 step = 1 epoch) with checkpoints every 2,000 steps.
- Compute all PH features and baselines at each checkpoint.
- Visual inspection: do any topological features show a visible shift before grokking onset?
**Pilot success condition:** Proceed to full experiment only if at least one PH statistic shows consistent directional behavior across >= 3/5 runs prior to grokking onset. "Consistent directional" means the same direction of change (increase or decrease) before test accuracy rises. If PH curves are flat, chaotic, or inconsistent across seeds, stop and reassess before committing compute.

This is not a statistical test. It is a sanity check to verify: (a) PH curves are smooth enough to analyze, (b) different seeds show vaguely similar dynamics, (c) slice averaging stabilizes the signal, and (d) signals do not spike randomly.

## 7. Full Experiment

- **Task:** Modular addition (a + b mod 97). Standard grokking benchmark (Power et al., 2022).
- **Model:** 1-layer transformer decoder, d_model=128, 4 attention heads. Single architecture family.
- **Data split:** 50% train / 50% test.
- **Training:** AdamW, full-batch (1 step = 1 epoch, matching Power et al. 2022). A preliminary hyperparameter calibration sweep will identify a weight decay regime that produces delayed generalization within the study budget for the chosen architecture. Weight decay sweep for full study informed by calibration results. 100K steps.
- **Seeds:** 30 independent runs per weight decay setting (90 runs total).
- **Checkpoint cadence:** Every 2,000 steps (50 uniform checkpoints per run). For the last 20% of training steps globally (steps 80,000-100,000), checkpoint frequency increases to every 500 steps. This is defined by training step count, not relative to onset, to avoid information leakage.
- **PH computation:** 2D loss landscape slices using filter-normalized random directions (Li et al. 2018). 50x50 grid. 5 slices per checkpoint, averaged. This is a standardized slice-based topology proxy, not a claim about the full loss landscape topology. We measure topology summaries extracted from standardized 2D projections and test whether they contain predictive signal.
- **Tracked per checkpoint:**
  - Training metrics: train loss, test loss, train acc, test acc
  - Topological (computed per slice, then averaged across slices):
    - *Primary endpoint:* H0 feature count
    - *Secondary endpoints:* H0 total persistence, H1 total persistence, H0 persistence entropy
    - *Exploratory only (logged, not used in primary analysis):* H1 feature count, H1 max persistence, H1 persistence entropy
  - Local geometry baselines: commutator defect, sharpness (trace of Hessian approximation)
  - Global baselines: spectral concentration (top eigenvalue ratio of weight matrix SVD)
  - Simple controls: weight norm (L2), training loss curvature (second derivative), generalization gap (train acc - test acc), validation loss slope

## 8. Baselines

**Local geometry:**
- Commutator defect (Dohmatob et al., 2026)
- Sharpness / trace of Hessian approximation

**Global / spectral:**
- Spectral concentration (top eigenvalue ratio)

**Simple controls:**
- Weight norm (L2)
- Training loss curvature (second derivative)
- Generalization gap (train acc - test acc)
- Validation loss slope

## 9. Risks

**PH instability (primary risk):** Persistent homology on 2D slices can be noisy and sensitive to projection direction. Any apparent pre-grokking signal could be a projection artifact rather than a real landscape property.
- *Mitigation 1:* 5 independent random slices per checkpoint, averaged. Report slice-to-slice variance. If inter-slice variance exceeds inter-seed variance, the method is too noisy and the result is null regardless of mean signal.
- *Mitigation 2 — Slice stability control:* On a subset of 10 runs (spread across weight decay conditions), compute PH with both 5 and 10 slices at the same checkpoints. Compare predictor rankings between the two slice counts. If the ranking of topological features relative to baselines changes with slice count, the result is projection-fragile and cannot be reported as robust.

**Grokking variability:** Grokking timing varies across seeds, which is expected but complicates alignment.
- *Mitigation:* Align all runs to onset step (t=0 at grokking) for visualization and analysis. Report raw and aligned results.

**Changepoint detection sensitivity:** PELT algorithm introduces algorithmic degrees of freedom.
- *Mitigation:* Fixed penalty parameter across all predictors. Robustness check with threshold-based heuristic.

**Checkpoint resolution:** Grokking transitions can be sharp. 2,000-step intervals may miss the exact transition.
- *Mitigation:* Adaptive checkpoint frequency (every 500 steps in last 20% before onset).

## 10. Falsification Criteria

- If no PH feature shows a consistent changepoint before grokking onset across >= 20/30 seeds in any weight decay condition, the topological signal does not exist.
- If PH features signal but strictly after the commutator defect in all conditions, topology adds nothing beyond existing methods.
- If inter-slice PH variance exceeds inter-seed variance, the measurement is too noisy to be useful.

## 11. Scope Boundaries

- This is a correlational study, not a mechanistic one. A precursor signal is not a causal explanation.
- No claims about unifying grokking with catastrophic forgetting. If results warrant, that connection is future work.
- "Underexplored" is the accurate framing of the literature gap. We do not claim this is the first application of TDA to grokking without a thorough literature review confirming that.

## 12. Compute Estimate

| Component | GPU-hours |
|-----------|-----------|
| Pilot (5 runs, full pipeline) | ~15 |
| Training (90 runs x ~10 min) | ~15 |
| PH landscape (22,500 evals x ~30s) | ~190 |
| Adaptive checkpoints (~20% more evals) | ~38 |
| Slice stability control (10 runs x 5 extra slices) | ~20 |
| Baselines (sharpness, commutator defect) | ~40 |
| **Total** | **~318** |

Pilot: ~1 day. Full study: ~12 days wall time on a single RTX 4090.

## 13. References

- Power et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.
- Dohmatob et al. (2026). Early-Warning Signals of Grokking via Loss-Landscape Geometry. arXiv:2602.16967.
- Lau et al. (2026). Grokking as a Phase Transition between Competing Basins. arXiv:2603.01192.
- Thilak et al. (2023). Predicting Grokking Long Before it Happens. arXiv:2306.13253.
- DeMoss et al. (2025). The Complexity Dynamics of Grokking. Physica D.
- Li et al. (2018). Visualizing the Loss Landscape of Neural Nets.
