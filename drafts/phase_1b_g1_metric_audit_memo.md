# Phase I-B G1 Metric Audit — Decision Memo

**For:** Alien meeting, 2026-04-21
**From:** Joshua Gutierrez (PI), research-assistant audit
**Date drafted:** 2026-04-16
**Status:** DRAFT — Joshua to review / edit before the meeting
**Scope:** Pre-analysis metric audit. Phase 4 analysis on Phase I-B is frozen until this memo produces a decision.

---

## TL;DR

The cross-dataset retention metric used in the completed 114/114 sweep is a **full-softmax argmax over the expanded classifier head** (Task A classes 0..K_A−1 plus Task B classes K_A..K_A+K_B−1). Because Task B training only ever updates labels in the Task B slice, those logits are systematically inflated during training. As a result, "Task A retention" can decay even if the feature backbone is unchanged, because Task A test images lose the argmax to a Task B class. This is well-known multi-head CL recency bias. It breaks direct comparability to Phase I-A within-dataset retention, where Task A and Task B are drawn from the same image distribution and the recency-bias contribution is expected to be smaller.

**Good news for feasibility:** `save_checkpoints: true` is set in all 57 per-arch configs, and `phase3_sequential_forgetting.py:384-388` saves a post-Task-B checkpoint at every eval step. Assuming those files are on disk (verify as Action 1), restricted-softmax re-evaluation can be done **post-hoc without retraining**.

**Recommendation:** Do not replace the existing metric. Compute restricted-softmax retention as a parallel second metric from existing checkpoints, compare full vs restricted across all 114 records, and let the comparison itself decide what the Phase I-B story is:
- If full and restricted agree (high rank correlation, small mean |Δ|) → existing metric stands, restricted reported as robustness check, continuity with Phase I-A preserved.
- If they diverge substantially → restricted becomes primary, full reported in supplement, Phase I-B explicitly labeled as not directly comparable to Phase I-A. That divergence is itself the main finding.
- If checkpoints are NOT actually on disk → freeze Phase I-B as exploratory, bring to the meeting, decide whether to rerun for a clean metric or ship the existing sweep as an exploratory sensitivity study with the G1 caveat disclosed.

---

## 1. The evaluation rule used in the completed sweep (mathematically)

Let θ* denote the Task A-converged parameters. Let W ∈ ℝ^{(K_A+K_B) × d}, b ∈ ℝ^{K_A+K_B} denote the expanded classifier head, constructed by copying the Task A head into rows [0, K_A) and initializing rows [K_A, K_A+K_B) with the PyTorch default (kaiming-uniform for Linear, not zero). Training on Task B uses shifted labels y' = y + K_A ∈ [K_A, K_A+K_B), with objective:

    L(x, y'; θ) = CrossEntropy(Wφ(x;θ) + b, y')   (for Task B batches)

At each eval step k ∈ {0, 10, 25, 50, 100, 250, 500, 1000, 5000}, Task A accuracy is measured on the Task A test loader (N_A samples, true labels y ∈ [0, K_A)):

    acc_A(k) = (1/N_A) Σ_i 𝟙[argmax_{j ∈ [0, K_A+K_B)} (W_j φ(x_i; θ_k) + b_j) = y_i]

The argmax is taken over **all K_A+K_B classes**. Retention is then

    ret_A(k) = acc_A(k) / acc_A(0)

Code references: `phase3_sequential_forgetting.py:314, 372` (calls `evaluate`); `experiments/shared/utils.py:46-56` defines `evaluate`, where `_, predicted = outputs.max(1)` takes the argmax over the full logit vector.

EWC benefit (the Phase I-A headline outcome) is the area-under-curve improvement:

    EWC_benefit = AURC_EWC − AURC_naive        (trapezoidal, steps 0..500)

where AURC is computed from the same full-softmax retention values.

## 2. The bias G1 introduces, and why it breaks Phase I-A comparability

The full argmax mixes two signals:

**(a) Feature-backbone drift** — what we intend to measure. As θ moves away from θ* during Task B training, the features φ(x_i; θ) change, and the Task A-class logits degrade on Task A inputs.

**(b) Classifier-head recency bias** — what we do not intend to measure. Only the Task B rows of W and b receive positive gradient updates from the Task B cross-entropy objective; the Task A rows are updated only indirectly through shared features and decay from weight-decay regularization. The Task B-row magnitudes grow systematically over Task B training, so for any φ(x), the Task B logits are pushed up relative to the Task A logits. Even with a completely unchanged backbone, a Task A test image can switch from argmax=y ∈ [0, K_A) to argmax=j ∈ [K_A, K_A+K_B) simply because the Task B logits now exceed the Task A logit that was previously the max.

Why this breaks comparability to Phase I-A:

- **Within-dataset (Phase I-A):** Task A and Task B are drawn from the same image distribution (e.g., CIFAR-100 classes 0–49 vs 50–99). Features that discriminate Task A classes also partially discriminate Task B classes, so the Task A-class logits on Task A inputs don't lag as badly behind the inflating Task B logits. Recency bias is a smaller component of measured forgetting.
- **Cross-dataset (Phase I-B):** Task A and Task B are drawn from visually distinct distributions (natural images ↔ fine-grained birds ↔ satellite scenes). The features learned on Task B data may respond very differently to Task A images, and the inflated Task B logits can dominate the argmax for Task A inputs even when the backbone still encodes Task A structure reasonably well. Recency bias can become a dominant, not marginal, component of "retention."

The consequence: the Phase I-B "retention" number as currently defined is a function of both feature drift (Phase I-A's intent) **and** a classifier-head artifact that has no within-dataset analog at the same magnitude. Any topology → retention coefficient estimated on this metric commingles the two. Comparing the Phase I-B coefficient to the Phase I-A coefficient is not apples-to-apples.

A secondary concern worth noting: at step 0, acc_A(0) is measured with the **already-expanded** classifier (classifier expansion happens at `phase3_sequential_forgetting.py:216-238`, before the step-0 evaluate at line 314). Because the new Task B rows are default-initialized (kaiming-uniform, not zero), there is already a small amount of recency-bias contamination at the normalization point. This is minor compared to the post-training effect but should be disclosed if we report ret_A(k).

## 3. Available artifacts

| Artifact | Available? | Notes |
|---|---|---|
| Task A checkpoints (`checkpoints/task_a_best.pt`) | **Yes** | Saved by phase1, used as the starting point for every cross-dataset run. One per (arch, starting_dataset) = 57 files. Symlinked into each xd result dir. |
| Per-step post-Task-B checkpoints (`forgetting_*/step_{k}.pt`) | **Very likely yes — verify Action 1** | `save_checkpoints: true` in every per-arch config (verified across all 57 configs). `phase3_sequential_forgetting.py:384-388` saves at each eval step when that flag is truthy. Expected 8 files per condition × 2 conditions (naive, ewc) × 114 runs = 1,824 files, one per (arch, pair, method, step). |
| Forgetting curve (`forgetting_*/forgetting_curve.json`) | **Yes** | 114 × 2 = 228 files. Contains acc_A(k), acc_B(k), forgetting(k) under the current full-softmax rule. |
| Task A test dataloader / class indices | **Reproducible** | Determined by config + seed. `get_split_dataset(cfg)` returns Task A with labels [0, K_A). Reproducible deterministically from each config + seed without re-running training. |
| Task B test dataloader (cross-dataset) | **Reproducible** | `get_cross_dataset_task_b(dataset_name, data_dir, batch_size, seed)` returns Task B with original labels in [0, K_B). Deterministic given the same seed. |
| Saved logits (per-example, per-step) | **No** | Not saved. Only aggregate accuracy is stored in forgetting_curve.json. |
| Per-example predictions | **No** | Same — only aggregate accuracy stored. |
| Class-index mappings (Task A = [0, K_A), Task B shifted to [K_A, K_A+K_B)) | **Yes, in config** | K_A stored as `num_classes_a`, K_B as `num_classes_b`. Label shift is applied at `phase3_sequential_forgetting.py:335`: `labels = (labels + num_classes_a).to(device)`. |
| SLURM stdout logs (for the "Task B barely learned" warning) | **Yes, on HPC** | Need to rsync and grep; relevant to G2 exclusion list, not G1 directly. |

**Critical action to confirm feasibility:** Verify that `forgetting_*/step_*.pt` files actually exist on disk. If `save_checkpoints: true` was respected by every run (and no checkpoints were deleted for disk-space reasons), restricted-softmax re-evaluation is a pure compute-from-disk job with no retraining. If they are missing, we are back to the v2-plan Option (b) rerun.

## 4. Feasible corrections and what each one answers

Four corrections are technically feasible; only the first two are in scope for the Alien meeting decision.

### 4.1 Restricted-softmax re-evaluation — primary candidate

For each saved `step_{k}.pt`, reload the model, then re-evaluate the Task A test set under:

    acc_A^restricted(k) = (1/N_A) Σ_i 𝟙[argmax_{j ∈ [0, K_A)} (W_j φ(x_i; θ_k) + b_j) = y_i]

Argmax is restricted to Task A positions only; Task B logits are masked out before argmax.

- **Question answered:** "Is the feature backbone still encoding Task A class structure, in the sense that the correct Task A class has the highest logit among Task A classes?" This isolates backbone drift from classifier-head recency bias.
- **Question NOT answered:** "Does the model as a whole still classify Task A images correctly when both tasks' classes are live?" The full-softmax metric answers that question, and it is a legitimate deployment-relevant metric — just not the right basis for a feature-geometry claim.
- **What this metric does and does not capture.** Restricted-softmax measures rank preservation within Task A classes: whether the correct Task A logit is still the largest among the K_A Task A logits. It does NOT capture absolute logit degradation, calibration drift, or margin collapse. Two models can have identical restricted accuracy but very different confidence structure, which may matter for EWC interpretation and for any topology relationship that is sensitive to margin rather than argmax. Restricted-softmax is therefore framed as a rank-based probe of backbone drift, not as a full measure of retention.
- **Feasibility:** Compute-from-disk only, conditional on Action 1 confirming step-level checkpoints exist. Estimate: 2–4 hours HPC time for 114 × 2 × 8 = 1,824 evaluations on small models.
- **Assumptions required for checkpoint-based re-evaluation to be valid.** The post-hoc approach is only sound if (a) each `step_{k}.pt` contains the **full** classifier head state (W and b for all K_A + K_B rows), not just backbone weights; (b) the evaluation pipeline is reproducible bit-for-bit — same preprocessing transforms, same normalization statistics, same dataloader ordering (or verifiably IID), dropout disabled, BatchNorm in eval mode with stored running statistics; and (c) the label mapping is perfectly reconstructible from config — no off-by-K_A errors, no dataset-specific remapping bugs. If any of these fails, the restricted metric measures a different θ_k state than the training process actually visited, and we introduce a second-order artifact while trying to remove the first. These conditions must be verified before trusting the output. The step-0 sanity check in §6 Action 4 is the main operational check for (b) and (c).

### 4.2 Calibrated / BiC-style classifier adjustment

Post-hoc linear rescaling of the Task B logits (subtract a per-class bias computed on a small validation set) before full-softmax argmax. Common in class-incremental CL literature (Wu et al. 2019, "Large Scale Incremental Learning").

- **Question answered:** "If we correct for recency bias at the classifier level (without touching the backbone), does the measured retention change?" This is a more lenient correction than full restriction; it admits some Task B logit contribution but debiased.
- **When useful:** As a sensitivity check layered on top of 4.1. If restricted and BiC-calibrated give similar answers, the finding is robust to the specific debiasing choice.
- **Feasibility:** Same compute profile as 4.1. Requires a held-out Task A calibration split (can subsample from Task A test or use a fresh Task A train fold).

### 4.3 Feature-space probe — frozen Task-A-only head trained post hoc

Train a fresh linear classifier on the **Task B-trained backbone's penultimate-layer features** using a small Task A calibration set, then evaluate on the Task A test set. This answers "do the learned features still linearly separate Task A classes?"

- **Question answered:** "Is Task A class information preserved in the feature space, regardless of what the classifier head says?" This is the cleanest possible isolation of feature-backbone drift, but it changes the experimental design (requires a probe-training step and a calibration set).
- **Feasibility:** More work than 4.1 or 4.2; introduces a new hyperparameter (probe training settings). Out of scope for the Alien meeting decision but worth noting as a future option if restricted softmax proves inconclusive.

### 4.4 Rerun with task-specific heads (two-head architecture)

Modify phase3 to maintain separate Task A and Task B classifier heads, then replay all 114 runs. This eliminates recency bias structurally.

- **Question answered:** The original scientific question, cleanly — but under a different experimental design than Phase I-A used. Creates its own continuity question: the Phase I-A comparison becomes "different metric, different head structure" rather than "different metric, same head structure."
- **Feasibility:** Multi-day HPC rerun of 228 jobs. Requires code changes in phase3 and possibly phase4. Last resort if post-hoc corrections are not possible or not credible.

## 5. Recommended path

Subject to Action 1 confirming step-level checkpoints exist on disk:

1. **Do not replace the existing metric.** Keep the completed 114/114 sweep as-is.
2. **Implement 4.1 (restricted-softmax re-evaluation) as a parallel second metric.** Output `ret_A^full(k)` and `ret_A^restricted(k)` side by side in a new `forgetting_curve_restricted.json` per run directory.
3. **Compare the two across all 114 records** — rank correlation, mean |Δ|, per-pair breakdown.
4. **Let the comparison decide the Phase I-B story:**
   - **High agreement** (rank corr > 0.9, mean |Δ| < 0.05): existing metric stands. Restricted reported in supplement as a robustness check. Phase I-B continues to be presented alongside Phase I-A.
   - **Moderate divergence** (0.7 < rank corr < 0.9, or 0.05 < mean |Δ| < 0.15): restricted becomes primary; full reported in supplement; continuity to Phase I-A partial; note the shift prominently.
   - **High divergence** (rank corr < 0.7 or mean |Δ| > 0.15): restricted is primary and the divergence itself is a finding. Phase I-B cannot be directly compared to Phase I-A; the cross-dataset section of any writeup reports the divergence as evidence that within-dataset and cross-dataset retention measure different things.

   These thresholds are heuristic and used for internal decision-making only, not as inferential cutoffs. The underlying full distributions (Spearman ρ over all 114 records, per-pair breakdowns of Δ, bootstrap confidence intervals for the rank correlation) are reported alongside the point estimates so that any downstream reader can judge agreement/divergence on their own criteria rather than being bound to the 0.9 / 0.7 cuts.

   **Divergence is not automatically a finding.** If the two metrics diverge substantially, we must first rule out that one of them is broken — specifically the new one. The step-0 sanity check in §6 Action 4 is the minimum guardrail; more diagnostics (per-arch distributions of restricted ret_A across architectures, spot-checks against recomputed full ret_A from the same checkpoints) may be needed depending on what the aggregate numbers look like. Only after those pass is divergence interpretable as a substantive result about recency bias.
5. **Freeze Phase 4 analysis** until the comparison is complete and the primary metric is chosen.
6. **If Action 1 fails** (step-level checkpoints not on disk): do not silently bypass this. Freeze Phase I-B as exploratory, bring the feasibility gap to the meeting, and decide between (a) the v2-plan Option (b) rerun, (b) treating Phase I-B as a disclosed sensitivity study with the G1 caveat prominently in the writeup, or (c) shelving Phase I-B until a cleaner rerun is feasible.

The parallel-metric approach is explicitly preferable to a silent swap for three reasons. First, it preserves the audit trail: a future reviewer can see both numbers and judge for themselves. Second, if restricted materially moves the results, the move itself is evidence that the original metric was contaminated — which is a stronger scientific claim than just reporting the corrected number. Third, it avoids pinning the project on a single debiasing choice; if 4.2 or 4.3 are added later, they can be layered on without backtracking.

## 6. Actions before the 2026-04-21 meeting

1. **[Today or tomorrow] Confirm step-level checkpoints exist on disk.** On HPC:
   ```
   find /fs1/scratch/cag1145/axiondeep-research/results -path "*_xd_*/forgetting*" -name "step_*.pt" | wc -l
   ```
   Expected: 1,824 (114 runs × 2 conditions × 8 eval steps). Any shortfall should be spot-checked against individual run dirs.
2. **[Today or tomorrow] G2 sanity parse.** Count SLURM stdout files containing "Task B barely learned":
   ```
   grep -l "Task B barely learned" /fs1/scratch/cag1145/axiondeep-research/slurm/logs/*xd*.out 2>/dev/null | wc -l
   ```
   Log affected (arch, pair) pairs as a G2 exclusion list — relevant to any analysis, not just G1.
3. **[Before meeting] Decide whether to have me draft the restricted-softmax re-eval script** so it is ready to run immediately after the meeting if the decision goes that way. The script is small (~150 LOC) and follows `phase3_sequential_forgetting.py`'s evaluation code path.
4. **[Pipeline sanity check, first thing after re-eval runs] Verify `ret_A^restricted(0) ≈ acc_A(0)` per the Task A checkpoint.** Before computing any full-vs-restricted comparisons or interpreting any divergence as a finding, confirm that at step 0 — when the model has not yet been trained on Task B, so there is no recency bias to remove — the restricted metric reproduces the Task A checkpoint's baseline accuracy (`task_a_acc` from phase1) within rounding. Concretely, for each of the 57 (arch, starting_dataset) combinations, compute restricted ret_A at step 0 and diff against `initial_task_a_acc` from the existing `forgetting_curve.json`. If these don't match to within ≤0.5% absolute, the re-evaluation pipeline has a reproducibility bug (preprocessing, dataloader, BN state, or label mapping) and we fix that before trusting any later-step numbers. This is the main operational guardrail against the "second-order artifact" failure mode in §4.1.
5. **[At meeting] Present this memo and discuss:**
   - Confirmation of Action 1 result (do the checkpoints exist? do they contain full classifier head state?)
   - Confirmation that the step-0 sanity check (Action 4) is agreed as a gate before any comparison is interpreted
   - Agreement on the parallel-metric approach vs a silent swap
   - Agreement on the three-tier decision rule (treated as heuristic, not inferential)
   - Any preference among 4.1, 4.2, 4.3, 4.4 or ordering of 4.1 + 4.2

## 7. What this memo does not resolve

- **Which CL methods Phase I-B includes.** Only naive + EWC were run. SI was not. The Phase I-A SI null result cannot be replicated cross-dataset without 114 additional SI jobs. Independent of G1; decide separately.
- **D1 pseudo-replication (114 ≠ 114 independent observations).** Will be handled at the Phase 4 analysis layer via mixed effects / clustered bootstrap, per the v2 plan.
- **Preregistration framing.** The v2 plan labels Phase I-B as exploratory. This memo does not alter that.
- **The final Phase I-B outcome category (A–E from the v2 plan).** Cannot be determined until the metric is decided and the analysis runs.
