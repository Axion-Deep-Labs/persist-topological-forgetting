# PERSIST Phase I-B: Cross-Dataset Forgetting — Execution Plan (v2)

**Author:** Claude (research assistant) for Joshua Gutierrez (PI)
**Date drafted:** 2026-04-10
**Revision:** v2 (incorporates Joshua's scientific critique of v1)
**Status:** DRAFT — for review
**Context:** CoLLAs 2026 paper will be submitted on Phase I-A as-is. Phase I-B is post-submission work, no longer on the April 16 critical path. Goal is to land a defensible cross-dataset analysis suitable for arXiv v2 / CoLLAs camera-ready, treating any standalone publication ambition as conditional on what the data actually shows.

---

## Revision History

**v1 → v2 changes** (in response to scientific critique, 2026-04-10):

1. **Metric definition gate added (new §1).** The cross-dataset retention metric is not directly comparable to Phase I-A's within-dataset retention because the expanded shared softmax conflates feature drift with classifier-head recency bias. v2 makes metric validation the **first gate**, blocking all downstream analysis until we know whether the existing forgetting curves measure what we think they measure.
2. **Pair-distance ordinal encoding demoted from primary moderator** (your critique #1, #4). v1 used `pair_distance ∈ {near, far}` as the headline moderator, but with 2 near pairs and 4 far pairs all involving RESISC, "distance" was confounded with "RESISC involvement." v2 uses **pair fixed effects** as the primary design and treats any continuous distance metric as a secondary analysis on pair-level residuals.
3. **Pseudo-replication addressed via mixed effects** (critique #2). v1's pooled Spearman across N=114 treated repeated architecture rows as independent. v2 uses architecture-clustered inference (mixed model with arch as random effect, or clustered bootstrap with arch_name as cluster) and reports **effective N** explicitly.
4. **All "pre-registered" language removed** (critique #3). v1 mixed exploratory and confirmatory framing. v2 labels the entire Phase I-B analysis as **exploratory**, with a frozen internal analysis plan, and includes a separate optional path for actual preregistration on a held-out replication set.
5. **H1 primacy dropped** (critique #6). v1 baked in H1 as the primary topology feature based on Phase I-A's ImageNet-100 finding, but Phase I-B is back at the 19 small-scale architectures where Phase I-A's small-scale runs showed H0 dominance. v2 treats **H0 and H1 as co-primary**, with both reported in every test, and is honest that we don't know in advance which should dominate at this scale.
6. **EWC interpretation reframed as hypothesis, not expectation** (critique #5). v1 wrote "regularization should matter more under interference" as if it were near-certain. v2 flags this as a hypothesis that could go either direction and gives credit to multiple competing explanations.
7. **Outcome D ("sign flip") downgraded from "low downside"** (critique #7). v1's outcome table called all four outcomes publishable. v2 separates **clean outcomes (A, B, C)** from **messy outcomes (D, E)** that may force a methodological detour with no guaranteed payoff.
8. **Publication positioning toned down** (critique #8). "NeurIPS-grade" removed. CoLLAs and NeurIPS deadlines clarified with timezone precision. The standalone-paper path is reframed as conditional on either a much larger replication or a focused boundary-condition story, not as a default outcome.
9. **New §0 (Threats to validity) added** that captures every known concern up front, so reviewers (or future-Joshua) can see the limits before reading the design.

---

## §0. Threats to Validity (Read This First)

Before any execution, the following concerns must be acknowledged and either resolved or explicitly accepted as caveats in the writeup. Some are gating (block execution); some are limitations (disclose in the paper).

### 0.1 Gating concerns (must resolve before §3 analysis can begin)

**G1. Cross-dataset retention metric may be dominated by classifier-head recency bias, not feature drift.**
The Phase 3 cross-dataset code (`phase3_sequential_forgetting.py:216-238, :328-373`) expands the classifier head to `num_classes_a + num_classes_b` outputs and uses a single shared softmax. Task B training only ever updates labels in positions `[num_classes_a, num_classes_a + num_classes_b)`, which systematically pushes those logits up. As a result, "Task A accuracy" measured by full-softmax argmax can decay even if the feature backbone is unchanged, purely because Task A test images lose the argmax to a Task B class. This is well-known multi-head CL recency bias. Within-dataset Phase I-A is partially shielded because Task A and Task B classes look similar (same distribution, same low-level features), so there's less spurious logit imbalance. Cross-dataset, especially natural→satellite, the recency-bias contribution to "forgetting" may dominate the actual representational drift. **If true, the cross-dataset retention metric measures something qualitatively different from Phase I-A and any apparent topology→retention relationship may be a consequence of (or be obscured by) recency bias rather than feature geometry.**

Resolution: §1 metric validation gate. Re-evaluate every Task A test set under a **restricted softmax** (argmax over positions `[0, num_classes_a)` only) and compare against the existing full-softmax retention. If the two metrics agree (rank correlation > 0.9 across the 114 records), use the existing data and disclose the design choice. If they diverge, the restricted-softmax retention becomes the primary outcome variable AND the analysis loses direct continuity with Phase I-A.

**G2. Some Phase 3 cross-dataset runs may have failed the existing Task-B-learning sanity check.**
Lines 400-402 of `phase3_sequential_forgetting.py` already warn: *"WARNING: Task B barely learned (final acc = X%, chance = Y%). Retention metric may not reflect true forgetting resistance."* This warning fires when final Task B accuracy < 2× chance level. We need to count how many of the 228 runs triggered this warning. Any (arch, pair) combination that did is unusable for the topology→forgetting analysis: if the model never learned Task B, the "forgetting" we measure is just noise around the loaded checkpoint.

Resolution: §1 includes a parse of `slurm/logs/*xd*out` for the warning string. Affected runs are excluded from the analysis with a clear footnote.

### 0.2 Design concerns (must inform analysis, can disclose as limitations)

**D1. Pseudo-replication: 114 records ≠ 114 independent observations.**
Each of 19 architectures appears in 6 pairs. The Task A topology is a single fixed value per architecture per starting dataset, reused across multiple Task B outcomes. A naive pooled Spearman correlation overstates power. Effective N is closer to 19 (architectures, treating pair structure as repeated measures) or 57 (arch × starting-dataset combinations), not 114. All inferential statistics must respect this via mixed effects or architecture-clustered inference.

**D2. Pair structure is not a clean continuum of "distance."**
Six pairs partition 3×3 minus diagonal: CIFAR↔CUB, CIFAR↔RESISC, CUB↔RESISC, in both directions. v1 collapsed these to "near" (CIFAR↔CUB only) vs "far" (anything with RESISC), but that encoding makes "distance" perfectly colinear with "RESISC involvement." Direction also matters: CIFAR→CUB and CUB→CIFAR are distinct task transitions, not interchangeable. v2's primary design uses **pair fixed effects** (six pair dummies, no continuous distance) and treats any continuous distance metric as a secondary post-hoc decomposition of pair-level residuals.

**D3. H0 vs H1 primacy is unsettled at this scale.**
Phase I-A established H1 dominance at ImageNet-100 scale (rho=0.93, p=0.0007). Phase I-A's small-scale runs (3 datasets × 19 architectures) showed H0 dominance instead. Phase I-B uses the small-scale 19 architectures, so we cannot assume H1 will dominate. Both H0 and H1 are reported as co-primary in every test. We do not have a confirmatory hypothesis about which should win.

**D4. SI cross-dataset jobs were not submitted.**
Phase I-A tested three CL methods: naive, EWC, SI. Phase I-B's 228 jobs only cover naive + EWC. The "SI null result" from Phase I-A (zero topology moderation) cannot be replicated cross-dataset without submitting additional 114 SI jobs. Decision needed: submit SI now and add a few days to the timeline, or accept naive+EWC only and disclose the asymmetry.

**D5. Phase I-B uses the small-scale architecture pool, not ImageNet-100 architectures.**
The 19 small-scale architectures top out at 44.7M parameters. Cross-dataset effects at this scale may not generalize to ImageNet-100-class models. If we want to claim "cross-dataset findings hold at production scale," we'd need to submit a separate cross-dataset run on the 8 ImageNet-100 architectures, which requires Task A checkpoints for those architectures on each starting dataset (currently we only have ImageNet-100 starting checkpoints, not CUB or RESISC).

**D6. EWC benefit interpretation under cross-dataset shift is genuinely uncertain.**
v1 framed "regularization matters more under interference" as the expected direction. In reality, EWC benefit could go in either direction:
- *More benefit under shift:* more inter-task interference → more value in protecting important Task A weights
- *Less benefit under shift:* the Fisher information identifying "important Task A weights" is computed on Task A data, which may not capture the parameters most threatened by Task B's very different gradient distribution. Importance estimates might be miscalibrated for shifted tasks
- *Benefit independent of shift:* EWC's effect is a function of the regularization strength and feature backbone capacity, not the task pair

We don't know which dominates. The analysis tests for moderation but does not predict its direction.

### 0.3 Limitations (disclose in writeup, don't try to fix)

**L1. 19 architectures is small for cross-architecture inference.** Phase I-A acknowledges this. Phase I-B inherits the limit. Adding more architectures is a multi-week effort and is out of scope here.

**L2. 6 pairs is small for pair-level inference.** Six dataset pairs is enough to detect large pair effects but not enough to model continuous pair properties (CKA distance, label-space overlap, etc.) with any precision. The continuous-distance secondary analysis is exploratory.

**L3. Two-task sequences only.** Real continual learning is many tasks. Cross-dataset two-task analysis is a step toward realism, not a complete answer.

**L4. EWC and SI are both regularization-based methods.** Replay-based and architectural CL methods are not tested. Findings may not generalize to those families.

**L5. The 19 architectures span many design families but unevenly.** 12 CNNs, 2 transformers, 1 MLP, 6 of which are the WRN width ladder. Within-class N is very small for transformers and MLPs.

---

## §1. Current State of Phase I-B (as of 2026-04-10)

### 1.1 What exists

| Layer | Status | Evidence |
|---|---|---|
| Job submission script | ✓ Built | `slurm/submit_cross_dataset.sh` (178 lines, committed in 7d88806 on 2026-04-07) |
| Submission run | ✓ Submitted | 228 jobs submitted Apr 7: 6 pairs × 19 architectures × 2 methods (naive + EWC) |
| Phase 3 cross-dataset support | ✓ Built | `phase3_sequential_forgetting.py:68-74` adds `--cross-dataset`, `--task-a-dir`, `--output-dir-override` flags |
| Local symlink scaffolding | ✓ Created | 114 `exp01_*_xd_*` directories in `results/`, each with `topology` and `checkpoints` symlinked back to Task A dir |
| HPC results | ⚠ **Unconfirmed** | Submit script reports "DONE" for some pairs, but actual completion across all 6 pairs is unverified. |
| Local results sync | ✗ **Not done** | `find results -name "forgetting_curve.json" -path "*xd_*"` returns 0 files locally |
| Metric validation | ✗ **Not done** | Restricted-softmax retention has not been computed for any run |
| Sanity check parse (Task B learned?) | ✗ **Not done** | The existing script logs warnings to stdout; we haven't aggregated them |
| Analysis pipeline | ✗ **Does not exist** | Phase 4/5/6 scripts have no concept of `(task_a → task_b)` pairs |

### 1.2 Cross-dataset pairs (6 total, no ordinal encoding)

| # | Task A | Task B |
|---|---|---|
| 1 | CIFAR-100 | CUB-200 |
| 2 | CIFAR-100 | RESISC-45 |
| 3 | CUB-200 | CIFAR-100 |
| 4 | CUB-200 | RESISC-45 |
| 5 | RESISC-45 | CIFAR-100 |
| 6 | RESISC-45 | CUB-200 |

**v2 note:** v1 grouped these as "near" (#1, #3) and "far" (#2, #4, #5, #6). v2 treats all six as distinct conditions with no a priori similarity ordering. Continuous distance metrics, if computed, are derived from the data (CKA on probe sets) rather than imposed by hand.

### 1.3 Architectures

Same 19 from Phase I-A small-scale runs (12 CNNs + 6 WRN width ladder + 2 transformers + 1 MLP = 21 entries; the WRN ladder counts within CNNs for design-family analysis but is reported separately for the width-scaling analysis).

### 1.4 Effective N for inference

| Naive count | What it represents | Use |
|---|---|---|
| 228 | Total Phase 3 runs (naive + EWC) | Raw run count |
| 114 | Unique (arch, pair) combinations after collapsing methods into derived columns | Record count |
| 19 | Unique architectures | Effective N for arch-clustered inference |
| 57 | Unique (arch, starting_dataset) — Task A topology values | Effective N for "topology → retention" correlation, treating each topology measurement as one observation |

The headline analyses use **57 as the effective N for topology effects** and **114 with arch-clustered inference for pair-level effects**. We do not pool 114 records as if they were independent.

---

## §2. Scientific Framing — What Phase I-B Will Claim

### 2.1 Core question

> **Does the topology→forgetting relationship observed in Phase I-A within-dataset settings hold when Task A and Task B come from different image distributions, after accounting for classifier-head recency bias?**

This is an **exploratory** question. We do not have a strong directional hypothesis, and we are not preregistering any specific test as confirmatory. The findings will be presented as exploratory cross-dataset extensions of Phase I-A, not as a confirmatory replication.

### 2.2 What this analysis cannot establish (and we will not claim)

- That topology→forgetting is "universal" across CL settings (we have one cross-dataset experiment, not a survey)
- That any specific moderator (task similarity, label-space overlap, feature alignment) explains pair-level heterogeneity in a generalizable way
- That EWC benefit is "stronger under interference" (the moderation could go either direction, and we don't have enough pairs to estimate it precisely)
- That the small-scale findings extend to ImageNet-100-scale cross-dataset (we don't have those checkpoints)

### 2.3 Three exploratory questions, not "claims"

In v1 these were framed as "Claims 1-3" with "pre-registered alpha." In v2 they are unmoderated exploratory questions with no preregistered alpha. We report effect sizes, confidence intervals (clustered bootstrap), and uncorrected p-values, and we explicitly note multiple-testing exposure.

**Q1: Does H0 or H1 (Task A topology) predict cross-dataset retention at all, after accounting for pair effects and architecture clustering?**

Test: Mixed-effects model
```
ret_10 ~ H0z + H1z + log_params + (1 | arch_name) + (1 | pair_id)
```
(or equivalent OLS with clustered SEs by `arch_name`, depending on convergence). H0 and H1 are entered together to test which dominates conditional on the other.

Effect of interest: coefficients on H0z and H1z, with 95% bootstrap CIs (5000 reps, clusters = arch_name).

Decision rule: report the coefficient and CI. Do not declare significance based on a single threshold. Note that this is one of three exploratory tests; multiple-testing exposure is acknowledged.

**Q2: Do the H0 / H1 effects vary across the six task pairs?**

Test: Add interaction terms
```
ret_10 ~ (H0z + H1z) × pair_id + log_params + (1 | arch_name)
```
Joint test of all `(H0z, H1z) × pair_id` interaction terms (likelihood-ratio test against the no-interaction model from Q1).

Effect of interest: whether the joint test rejects (suggests pair-level heterogeneity in topology effects) and which specific pairs drive it. Visualize as a forest plot of per-pair H0 and H1 coefficients with CIs.

Decision rule: If the joint test rejects, look at the per-pair coefficient pattern. If pairs cluster sensibly (e.g., all pairs starting from the same Task A behave similarly), that's a substantive finding. If the pattern is incoherent (random across pairs), that's likely noise from the small N per pair (≈19 records each).

**Q3: Does any continuous, data-derived measure of pair similarity explain the pair-level heterogeneity from Q2?**

Only run if Q2 rejects. This is a secondary, post-hoc decomposition.

Test: For each pair, compute a similarity metric (e.g., CKA between Task A and Task B penultimate-layer features on a common probe set; or label-space overlap if using a vision-language alignment). Then ask whether the per-pair H0/H1 coefficients from Q2 correlate with the similarity metric across the 6 pairs.

Effect of interest: rank correlation between similarity and per-pair H0/H1 coefficient. With only 6 pairs, this is descriptive, not inferential. Report it as "pair-level pattern" not as a hypothesis test.

**Q4 (separate, EWC-specific): Is the EWC benefit moderated by topology in cross-dataset settings, replicating Phase I-A's finding?**

Test: Mixed-effects model with EWC benefit as outcome
```
ewc_benefit ~ (H0z + H1z) + log_params + (1 | arch_name) + (1 | pair_id)
```
where `ewc_benefit = ewc_ret_10 - naive_ret_10`.

Decision rule: Report coefficients with bootstrap CIs. Compare directly to Phase I-A's finding (EWC benefit p=0.046, pooled across 3 within-dataset analyses). Note: the direction of the topology × EWC-benefit interaction is genuinely uncertain in cross-dataset settings (see D6 in §0.2).

### 2.4 Outcome categories — honest version

| Outcome | What it looks like | What we can publish |
|---|---|---|
| **A. Topology effect persists, no pair-level heterogeneity** (Q1 finds nonzero H0 or H1 coefficient, Q2 rejects no interaction) | The Phase I-A within-dataset finding extends cleanly to cross-dataset | Strong arXiv v2 section. Does not require new datasets to publish. |
| **B. Topology effect persists with pair-level heterogeneity** (Q1 finds nonzero coefficient, Q2 rejects, Q3 finds an interpretable continuous moderator) | The relationship exists but its strength depends on task similarity | Strongest possible result. Could be the seed of a focused follow-up paper. |
| **C. Topology effect persists with incoherent pair-level heterogeneity** (Q1 nonzero, Q2 rejects, Q3 finds nothing interpretable) | Real effect but we can't explain pair-to-pair variation with the data we have | Honest arXiv v2 section. Frame as "topology→forgetting holds cross-dataset but its strength varies by pair in ways we cannot yet explain." |
| **D. Topology effect vanishes under cross-dataset** (Q1 finds zero or near-zero coefficient for both H0 and H1) | Boundary condition: PERSIST applies within-distribution only | Publishable as a clear negative result, IF we can rule out metric artifacts (G1 resolved). If G1 is not cleanly resolved, this is **not publishable** as a clean negative because the null could be a metric artifact rather than a true boundary condition. |
| **E. Sign flip** (Q1 finds significant coefficient with opposite sign from Phase I-A) | High-topology architectures retain BETTER under cross-dataset shift, opposite to within-dataset | **Not publishable as a clean finding without investigation.** Most likely indicates either a metric artifact (recency bias dominating differently across configs) or a confound we have not identified. Would force a multi-week methodological detour with no guarantee of resolution. **This is not "low downside."** |

**v2 honesty note:** A and B are clean wins. C is publishable but requires careful framing. D is publishable conditional on G1 resolution. E is a real risk and could waste significant analysis time without delivering a publishable result.

---

## §3. Execution Plan — Five Gates (was four in v1)

The new gate ordering is: Verify HPC → Sync → **Metric Validation (NEW)** → Build Analysis Pipeline → Run Inference & Write Up.

### Gate 0: Verify HPC job state (today, ~10 minutes)

Same as v1 §3 Gate 0. Joshua runs three commands on HPC and pastes the output:

```bash
# (1) Total count of forgetting result files
find /fs1/scratch/cag1145/axiondeep-research/results -name "forgetting_curve.json" -path "*xd_*" | wc -l

# (2) Per-pair breakdown
for ds in cifar100 cub200 resisc45; do
  for tb in cifar100 cub200 resisc45; do
    [ "$ds" = "$tb" ] && continue
    n=$(find /fs1/scratch/cag1145/axiondeep-research/results -name "forgetting_curve.json" -path "*${ds}*xd_${tb}*" 2>/dev/null | wc -l)
    echo "$ds -> $tb: $n / 38"
  done
done

# (3) Anything still queued or running?
squeue -u cag1145 -o "%.10i %.20j %.8T %.10M" | head -40

# (4) NEW for v2: count Task-B-not-learned warnings
grep -l "Task B barely learned" /fs1/scratch/cag1145/axiondeep-research/slurm/logs/*xd*out 2>/dev/null | wc -l
```

**Decision tree:**
- If 228/228 present and 0 sanity warnings: proceed to Gate 1.
- If <228 present: identify gaps, check `slurm/logs/*xd*err`, resubmit subset.
- If sanity warnings present: log which (arch, pair) combinations are affected. They will be excluded in Gate 2.

### Gate 1: Sync results from HPC to laptop (~30 minutes)

Same as v1 §3 Gate 1. Add `--include='*.out'` to the rsync include list so we also pull the SLURM stdout files for the warning parse.

```bash
rsync -av \
  --include='*/' \
  --include='forgetting_curve.json' \
  --include='final_metrics.json' \
  --include='metadata.json' \
  --include='ewc_history.json' \
  --include='si_history.json' \
  --exclude='*' \
  cag1145@discovery.nmsu.edu:/fs1/scratch/cag1145/axiondeep-research/results/ \
  /home/joshua/Corporate/axiondeep-research/results/

# Also sync SLURM logs for warning parse
rsync -av \
  --include='*xd*.out' \
  --include='*xd*.err' \
  --exclude='*' \
  cag1145@discovery.nmsu.edu:/fs1/scratch/cag1145/axiondeep-research/slurm/logs/ \
  /home/joshua/Corporate/axiondeep-research/slurm/logs_xd/
```

### Gate 2 (NEW for v2): Metric Validation (~1 day)

**This is the gating step. Nothing in §3 Gates 3-4 runs until this gate produces a clean answer.**

#### 2.1 — Build the restricted-softmax retention re-evaluator (~3 hours)

Write a new script `experiments/exp01_topological_persistence/phase3b_restricted_softmax_eval.py`. For each of the 114 cross-dataset run directories, this script:

1. Loads the Task A model checkpoint (`xd_dir/checkpoints/task_a_best.pt`, which is symlinked to the original Task A checkpoint).
2. Loads the post-Task-B model state. **Problem:** the current Phase 3 code does not save the post-Task-B checkpoint by default — it only saves the forgetting curve. We need either:
   - **Option (a):** Re-run Phase 3 cross-dataset with `forgetting.save_checkpoints: true` in the config so we have the post-Task-B model state to evaluate. Adds ~228 reruns. Several days of HPC time. **Probably necessary.**
   - **Option (b):** Modify Phase 3 to save the final post-Task-B model regardless of the save_checkpoints flag, then rerun. Same time cost as (a) but cleaner.
   - **Option (c):** Skip restricted-softmax and accept the metric ambiguity. Disclose in §0 G1 as an unresolved limitation. NOT recommended.

   **Recommendation:** Option (b). It's the cleanest and the rerun is straightforward — we already have all the configs and the SLURM scripts.

3. For each saved Task B step in the forgetting curve, re-evaluate Task A test set using a restricted softmax over positions `[0, num_classes_a)` only (mask out Task B logits before argmax).
4. Save the restricted-softmax forgetting curve as `xd_dir/forgetting/forgetting_curve_restricted.json`.

#### 2.2 — Compare full vs restricted softmax retention (~30 min)

For each of the 114 (arch, pair) records:
- Compute `ret_10_full` (existing) and `ret_10_restricted` (new)
- Compute `delta = ret_10_full - ret_10_restricted` (positive means restricted softmax shows MORE forgetting, i.e., less recency bias contribution)

Across all 114 records:
- Spearman correlation between `ret_10_full` and `ret_10_restricted`
- Mean and std of `delta`
- Per-pair breakdown of `delta`

#### 2.3 — Decision rule

| Condition | Decision |
|---|---|
| `corr(full, restricted) > 0.9` AND `mean(|delta|) < 0.05` | Recency bias is small. Use existing full-softmax retention as primary outcome. Disclose this validation in §0. |
| `0.7 < corr < 0.9` OR `0.05 < mean(|delta|) < 0.15` | Recency bias is moderate. Use restricted-softmax retention as primary AND report full-softmax in supplementary. Note continuity with Phase I-A is partial. |
| `corr < 0.7` OR `mean(|delta|) > 0.15` | Recency bias is large. Use restricted-softmax exclusively. The cross-dataset analysis is no longer directly comparable to Phase I-A's within-dataset retention. Disclose prominently. |
| `corr < 0` (rare) | Something is wrong. Investigate before proceeding. |

#### 2.4 — Sanity check warning parse (~30 min)

```bash
grep -l "Task B barely learned" /home/joshua/Corporate/axiondeep-research/slurm/logs_xd/*.out
```

For each affected run, identify the (arch, pair) and add to an exclusion list. Document the count and which configs in §0 G2.

### Gate 3: Build the cross-dataset analysis pipeline (~1-2 days)

#### 3.1 — Phase 4 cross-dataset adapter

Same as v1 §3 Gate 2 §2.1. Add a `--cross-dataset-pair task_a:task_b` flag to `phase4_correlation.py` and write `correlation_results_xd_{task_a}_to_{task_b}.json`. **Adds two changes from v1:**
- Outputs both `ret_10_full` and `ret_10_restricted` columns
- Includes a `task_b_learned` boolean flag (from the warning parse) so downstream filtering is trivial

Estimated effort: 4-5 hours (slightly more than v1 estimate due to two-metric reporting).

#### 3.2 — Mixed-effects analysis script (replaces v1's Phase 6 extension and Phase 7 regression)

Write a new script `experiments/exp01_topological_persistence/phase8_cross_dataset_mixed_effects.py`. v1 split this into "Phase 6 extension" + "Phase 7 focused regression"; v2 combines them because the design now uses pair fixed effects (or random effects) throughout, so there's no division of labor.

The script uses `statsmodels` (already in venv) for the mixed model:

```python
import statsmodels.formula.api as smf

# Q1: H0/H1 main effects, controlling for arch and pair
md_q1 = smf.mixedlm(
    "ret_10_primary ~ H0z + H1z + log_params",
    data=df,
    groups="arch_name",
    re_formula="1",
).fit()

# Q2: Add pair × topology interactions, test against Q1
md_q2 = smf.mixedlm(
    "ret_10_primary ~ (H0z + H1z) * C(pair_id) + log_params",
    data=df,
    groups="arch_name",
    re_formula="1",
).fit()

# Likelihood-ratio test for pair-level heterogeneity
lr_stat = 2 * (md_q2.llf - md_q1.llf)
df_diff = md_q2.df_modelwc - md_q1.df_modelwc
p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)

# Q4: EWC benefit, same structure
md_q4 = smf.mixedlm(
    "ewc_benefit ~ H0z + H1z + log_params",
    data=df,
    groups="arch_name",
    re_formula="1",
).fit()

# Clustered bootstrap for CIs (5000 reps, cluster = arch_name)
def bootstrap_clustered(df, n_reps=5000):
    archs = df['arch_name'].unique()
    coefs = []
    for _ in range(n_reps):
        sampled_archs = np.random.choice(archs, size=len(archs), replace=True)
        boot_df = pd.concat([df[df.arch_name == a] for a in sampled_archs])
        # ... refit and store coefficients
    return np.percentile(coefs, [2.5, 97.5])
```

**Pre-flight checks the script must run:**
- Effective N report: print N records, N unique archs, N unique pairs
- Convergence diagnostics: print mixed-model convergence message
- VIF check: ensure no severe multicollinearity (> 10) between H0z, H1z, log_params
- Sensitivity to outliers: leave-one-out by architecture, report coefficient stability

Estimated effort: 5-6 hours including testing.

#### 3.3 — Optional: continuous distance secondary analysis (Q3)

Only run if Q2 rejects (i.e., we find evidence of pair-level heterogeneity). Compute pairwise CKA distance between architectures' penultimate-layer features on a common probe set (CIFAR-10 test set, downloaded if not present). Then for each architecture, compute its per-pair H0 / H1 coefficient from Q2 and ask whether those coefficients vary with CKA distance.

With only 6 pairs this is descriptive at best. Frame as "the pattern across pairs is consistent with / inconsistent with a CKA-based similarity story" rather than a hypothesis test.

Estimated effort: 3-4 hours.

### Gate 4: Write up and decide where it lands (~1-3 days)

Same options as v1 §3 Gate 3, with revised positioning:

- **Default: arXiv v2 of the existing PERSIST paper.** Add a "Cross-Dataset Generalization (Exploratory)" section. ~3-4 pages. Push after CoLLAs deadline. Update abstract to mention "exploratory cross-dataset extension on 19 architectures × 6 dataset pairs."
- **Conditional: CoLLAs camera-ready addition.** Only if accepted; only if results are clean (Outcomes A or B from §2.4); only if camera-ready timeline allows.
- **Conditional: standalone follow-up paper.** Only if (a) Outcome B with strong CKA-based moderation pattern, AND (b) we add either more architectures or more dataset pairs to address L1/L2. Without expansion, 19 × 6 is too thin for a standalone paper at NeurIPS or ICML.

**v2 honesty note:** v1 used "NeurIPS-grade" language for Outcome A. v2 removes this. The realistic best case is a strong arXiv v2 section that informs a future, larger Phase I-C study.

### Calendar references

- **CoLLAs 2026 main submission:** April 15, 2026 23:59 AoE per the CoLLAs website (equivalent to April 16 ~12:00 UTC). The OpenReview portal may show "April 16, 2026 UTC" for the same deadline due to timezone conversion display. Both refer to the same instant.
- **NeurIPS 2026 main paper deadline:** May 6, 2026 AoE (per NeurIPS 2026 call for papers).
- **Implication for follow-up:** If Phase I-B analysis completes by ~April 25 and Outcome B holds, a NeurIPS submission is possible but extremely tight (~10 days for write-up after analysis). More realistic target for a standalone follow-up: ICLR 2027 or NeurIPS 2026 workshop tracks.

---

## §4. Risks and Decision Points (revised from v1)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **G1 — Recency bias dominates retention metric** | Medium-high | High (would invalidate continuity with Phase I-A) | Gate 2 metric validation, restricted-softmax re-evaluation |
| **G2 — Some Phase 3 runs failed Task-B-learning sanity check** | Medium | Medium (excludes some records) | Parse SLURM logs, exclude affected runs, document |
| HPC jobs not actually complete | Medium | High | Gate 0 verifies, resubmit gappy subset |
| Failures from missing config variants | Low-medium | Medium | Check `_cub200`, `_resisc45` config existence per arch |
| **D1 — Pseudo-replication** | Certain (it's a structural feature) | Medium (must be addressed in inference) | Mixed effects with arch random effect, clustered bootstrap |
| **D2 — Pair structure not a clean continuum** | Certain | Medium | Pair fixed effects primary; continuous distance secondary only |
| **D3 — H0 vs H1 primacy unsettled** | Certain | Medium | Both reported as co-primary; no hidden choice |
| **D6 — EWC interpretation direction uncertain** | Certain | Low (it's just a hypothesis) | Frame as exploratory, report direction observed |
| **Outcome E — sign flip** | Low-medium | High (forces investigation, may not be publishable) | If observed, halt analysis pipeline and investigate metric definition before any writeup |
| Mixed-effects model convergence failures | Low-medium | Medium | Have OLS-with-clustered-SE as fallback |
| 6 pairs too few for meaningful pair × topology interaction tests | Medium | Medium | Pre-declare that Q2 is exploratory and underpowered |
| Continuous distance metric (CKA) doesn't vary enough across 6 pairs | Medium | Low (Q3 is secondary anyway) | Try multiple distance metrics, report all |

---

## §5. What This Plan Does NOT Cover (Out of Scope)

(Same as v1, plus the following clarifications)

- **Genuine preregistration of Phase I-B.** v1 mistakenly used pre-registered language. If we want to do this properly, the path is: (a) freeze this document as the analysis plan, (b) submit to OSF or aspredicted.org **before** running Gate 3, (c) hold out a fraction of the architectures or pairs for confirmatory testing. This adds calendar time but is the only way to get confirmatory claims out of cross-dataset work. **Default in v2 is exploratory framing without preregistration.**
- **Submitting SI cross-dataset jobs.** Decision item in §6 Q4.
- **Cross-dataset on ImageNet-100 architectures.** Would require new Task A checkpoints on CUB and RESISC for the 8 ImageNet-100 architectures. Multi-day HPC effort. Out of scope for this plan but should be tracked as Phase I-C.

---

## §6. Concrete Punch List (revised from v1)

| # | Action | Owner | Estimate | Blocker? |
|---|---|---|---|---|
| 1 | Run Gate 0 verification commands on HPC shell | Joshua | 10 min | None — do now |
| 2 | Paste Gate 0 output to Claude | Joshua | 1 min | Depends on #1 |
| 3 | If gaps: identify failures, fix, resubmit | Joshua + Claude | 30 min – 4 hr | Depends on #2 |
| 4 | Decide on G1 mitigation: Option (a), (b), or (c) from §3 Gate 2.1 | Joshua | 5 min | Conceptual blocker |
| 5 | If Option (a) or (b): modify Phase 3 to save post-Task-B checkpoint, resubmit 228 jobs | Joshua + Claude | 4-6 hours code + 1-3 days HPC | Depends on #4 |
| 6 | Sync results + slurm logs from HPC to laptop | Joshua | 30 min | Depends on #3 (and #5 if applicable) |
| 7 | Implement restricted-softmax re-evaluator (`phase3b_restricted_softmax_eval.py`) | Claude | 3 hours | Depends on #5 (if Option a/b) |
| 8 | Run restricted-softmax eval on all 114 records | Joshua | 1-2 hours HPC | Depends on #7 |
| 9 | Compute full-vs-restricted comparison stats; apply Gate 2 decision rule | Claude | 30 min | Depends on #8 |
| 10 | Parse SLURM stdout for "Task B barely learned" warnings; build exclusion list | Claude | 30 min | Depends on #6 |
| 11 | **METRIC GATE DECISION:** based on #9 + #10, decide whether to proceed and which retention metric is primary | Joshua | 30 min | Depends on #9, #10 |
| 12 | Implement Phase 4 cross-dataset adapter with both retention metrics | Claude | 4-5 hours | Depends on #11 |
| 13 | Run Phase 4 for all 6 pairs | Joshua + Claude | 1 hr | Depends on #12 |
| 14 | Implement mixed-effects analysis script (`phase8_cross_dataset_mixed_effects.py`) | Claude | 5-6 hours | Depends on #13 |
| 15 | Run Q1, Q2, Q4; report effect sizes + CIs + LR test for Q2 | Joshua + Claude | 1 hr | Depends on #14 |
| 16 | If Q2 rejects: implement Q3 secondary CKA analysis | Claude | 3-4 hours | Depends on #15 |
| 17 | Categorize outcome (A/B/C/D/E from §2.4) | Joshua | 30 min | Depends on #15, #16 |
| 18 | If outcome E (sign flip): halt and investigate | Joshua + Claude | open-ended | Depends on #17 |
| 19 | Write arXiv v2 cross-dataset section based on outcome | Joshua + Claude | 6-10 hours | Depends on #17 |
| 20 | Update EXPERIMENT_LOG.md with Phase I-B results | Claude | 30 min | Depends on #17 |
| 21 | Update TODO.md to mark Phase I-B status | Claude | 5 min | Depends on #17 |
| 22 | Push arXiv v2 (after CoLLAs deadline of April 15 AoE) | Joshua | 30 min | Depends on #19, post-Apr 15 |

**Total estimated effort:** 30-45 hours of focused work, spread across **5-10 calendar days** depending on whether Option (a)/(b) requires the 228-job rerun.

**v2 honesty note on timeline:** v1 estimated 16-25 hours over 3-5 days. v2 is much higher because Gate 2 (metric validation) is real work, may require an HPC rerun, and is non-negotiable. If we skip Gate 2 we save several days but risk publishing analysis on a metric we don't understand.

---

## §7. Open Questions for Joshua

1. **G1 mitigation choice:** Option (a) [rerun with save_checkpoints], (b) [modify Phase 3 to always save final, then rerun], or (c) [skip restricted-softmax, accept ambiguity]? **Strong recommendation: (b).** It's only marginally more work than (a) and produces cleaner code for future work.
2. **SI cross-dataset jobs:** Submit alongside this work to give cross-dataset SI null replication, or accept naive+EWC asymmetry with Phase I-A's three-method design? Recommend submitting now if we're already going to rerun for Option (b).
3. **Preregistration:** Do you want to genuinely preregister (freeze this doc, push to OSF, add a held-out replication set) or proceed exploratory? Default v2 is exploratory.
4. **Task-A-only restricted softmax — alternative formulations:** Restricted softmax is one way to address recency bias. Alternatives: (a) calibrated softmax with per-class bias correction (BiC-style), (b) cosine similarity in feature space instead of classifier output, (c) probe accuracy with a frozen Task-A-only linear head trained post hoc. We're using restricted softmax because it's the simplest and most directly comparable to existing data, but let me know if you'd prefer one of the alternatives or want to report multiple.
5. **Outcome E protocol:** If we observe a sign flip in Q1, what's the maximum time we're willing to spend investigating before either publishing as a clean negative or shelving Phase I-B? Recommend setting an upfront cap of ~1 week.
6. **CKA probe set for Q3:** CIFAR-10 test set is the cleanest common reference because all 19 architectures can ingest it. Confirm or suggest alternative.
7. **Continuous distance metric for Q3:** CKA is the default. Alternatives: Wasserstein distance between class embeddings, label-space overlap (zero for these dataset pairs since classes are disjoint), pretrained-CLIP feature alignment between training images. Default to CKA unless you have a preference.

---

## §8. Document Status

This is v2, drafted in response to the scientific critique of v1. Joshua to review, edit, approve. Once approved, the action items in §6 become the working plan. EXPERIMENT_LOG.md and TODO.md will be updated to reference this document.

**Next step pending Joshua's review of v2:** Decide on G1 mitigation (Q1 in §7), then execute Gate 0 (verify HPC state).
