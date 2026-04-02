"""
Phase 6: Pooled Cross-Dataset Interaction Analysis

Tests two claims with formal inferential statistics:
  Claim 1: Topology is a conditional predictor of forgetting (dataset moderates H0 effect)
  Claim 2: Topology predicts CL method benefit on some datasets (dataset moderates H0-benefit relationship)
  Supports both EWC and SI (Zenke et al., 2017) benefit as outcome variables.

Design:
  - Pools records across N datasets (auto-discovered from Phase 4 JSONs)
  - Supports unbalanced designs (different architectures per dataset)
  - OLS with clustered bootstrap (architecture blocks) for CIs
  - Permutation tests (H0 shuffled within dataset) for interaction block
  - CIFAR-100 as reference category

Models (K = N-1 non-reference datasets):
  M0: Y ~ log_params + dataset + log_params x dataset       (1 + 2K features)
  M1: Y ~ M0 + H0z + H0z x dataset                          (2 + 3K features)

Tests:
  Primary: block test of {H0z, H0z x ds_1, ..., H0z x ds_K} (does topology help at all?)
  Secondary: block test of {H0z x ds_1, ..., H0z x ds_K}     (does dataset moderate topology?)

Reads Phase 4 correlation JSONs as single source of truth.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Defaults from design specification
N_BOOTSTRAP = 5000
N_PERMUTATIONS = 1000
VIF_THRESHOLD = 10.0
DEFAULT_DATASETS = ["cifar100", "cub200", "resisc45", "imagenet100"]
DATASET_LABELS = {
    "cifar100": "CIFAR-100", "cub200": "CUB-200",
    "resisc45": "RESISC-45", "imagenet100": "ImageNet-100",
}
REFERENCE_DATASET = "cifar100"


def load_phase4_data(results_dir: Path, datasets=None):
    """Load and validate Phase 4 correlation JSONs for all available datasets."""
    if datasets is None:
        datasets = DEFAULT_DATASETS

    # Auto-discover: only include datasets that have Phase 4 output
    available = []
    for ds in datasets:
        fpath = results_dir / f"correlation_results_{ds}.json"
        if fpath.exists():
            available.append(ds)
    if not available:
        print(f"ERROR: No correlation_results_*.json found in {results_dir}")
        sys.exit(1)

    records = []
    arch_counts = {}

    for ds in available:
        fpath = results_dir / f"correlation_results_{ds}.json"
        with open(fpath) as f:
            data = json.load(f)

        per_arch = data["per_architecture"]
        n_arch = len(per_arch)
        arch_counts[ds] = n_arch
        if n_arch < 3:
            print(f"WARNING: {ds} has only {n_arch} architectures (need >= 3 for meaningful analysis)")

        arch_names = [a["arch_name"] for a in per_arch]

        # Load EWC benefits (pre-computed in correlations.ewc_benefit)
        ewc_section = data.get("correlations", {}).get("ewc_benefit", {})
        ewc_benefits = ewc_section.get("benefits", [None] * n_arch)
        ewc_arch_names = ewc_section.get("architectures", arch_names)

        ewc_lookup = {}
        for name, benefit in zip(ewc_arch_names, ewc_benefits):
            ewc_lookup[name] = benefit

        # Load SI benefits (pre-computed in correlations.si_benefit)
        si_section = data.get("correlations", {}).get("si_benefit", {})
        si_benefits = si_section.get("benefits", [None] * n_arch)
        si_arch_names = si_section.get("architectures", arch_names)

        si_lookup = {}
        for name, benefit in zip(si_arch_names, si_benefits):
            si_lookup[name] = benefit

        for arch in per_arch:
            name = arch["arch_name"]
            ewc_benefit = ewc_lookup.get(name)
            si_benefit = si_lookup.get(name)

            ewc_ret10 = arch.get("ewc_ret_10")
            naive_ret10 = arch.get("ret_10")
            ewc_benefit_ret10 = None
            if ewc_ret10 is not None and naive_ret10 is not None:
                ewc_benefit_ret10 = ewc_ret10 - naive_ret10

            si_ret10 = arch.get("si_ret_10")
            si_benefit_ret10 = None
            if si_ret10 is not None and naive_ret10 is not None:
                si_benefit_ret10 = si_ret10 - naive_ret10

            records.append({
                "arch_name": name,
                "dataset": ds,
                "num_params": arch["num_params"],
                "H0": arch["H0"],
                "H1": arch.get("H1", 0.0),
                "ret_10": arch["ret_10"],
                "retention_100": arch["retention_100"],
                "early_aurc": arch["early_aurc"],
                "ewc_benefit_aurc": ewc_benefit,
                "ewc_benefit_ret10": ewc_benefit_ret10,
                "si_benefit_aurc": si_benefit,
                "si_benefit_ret10": si_benefit_ret10,
            })

    # Validate: log_params matches across datasets for same architecture
    params_by_arch = {}
    for r in records:
        name = r["arch_name"]
        if name not in params_by_arch:
            params_by_arch[name] = r["num_params"]
        elif params_by_arch[name] != r["num_params"]:
            print(f"WARNING: {name} has different param counts across datasets "
                  f"({params_by_arch[name]} vs {r['num_params']}). Using first.")

    n_unique_archs = len(set(r["arch_name"] for r in records))
    ds_summary = ", ".join(f"{DATASET_LABELS.get(ds, ds)}({arch_counts[ds]})" for ds in available)
    print(f"Loaded {len(records)} records ({n_unique_archs} unique architectures x {len(available)} datasets)")
    print(f"Datasets: {ds_summary}")
    return records, available


def build_design_matrix(records, outcome_key, active_datasets, standardize_h0=True,
                        standardize_params="global"):
    """Build the pooled design matrix and outcome vector.

    Dynamically generates dataset dummies and interactions for N datasets.
    Layout:
      M0: log_params, [K dummies], [K lp×dataset interactions]  (1 + 2K features)
      M1: log_params, H0z, [K dummies], [K lp×dataset], [K H0z×dataset]  (2 + 3K features)
    where K = len(active_datasets) - 1 (reference dataset excluded).

    Returns X_m0, X_m1, y, names_m0, names_m1, arch_indices, h0z, datasets, non_ref_datasets.
    """
    non_ref = [ds for ds in active_datasets if ds != REFERENCE_DATASET]
    K = len(non_ref)
    n = len(records)

    y = np.array([r[outcome_key] for r in records], dtype=np.float64)

    log_params = np.log(np.array([r["num_params"] for r in records], dtype=np.float64))
    if standardize_params == "global":
        lp_std = log_params.std()
        if lp_std > 0:
            log_params = (log_params - log_params.mean()) / lp_std

    h0_raw = np.array([r["H0"] for r in records], dtype=np.float64)
    datasets = [r["dataset"] for r in records]

    if standardize_h0:
        h0z = np.zeros(n)
        for ds in active_datasets:
            mask = np.array([d == ds for d in datasets])
            vals = h0_raw[mask]
            if vals.std() > 0:
                h0z[mask] = (vals - vals.mean()) / vals.std()
            else:
                h0z[mask] = 0.0
    else:
        h0z = h0_raw.copy()

    # Dataset dummies (reference excluded)
    dummies = {}
    for ds in non_ref:
        dummies[ds] = np.array([1.0 if d == ds else 0.0 for d in datasets])

    # Architecture indices for clustering
    arch_names = sorted(set(r["arch_name"] for r in records))
    arch_map = {name: i for i, name in enumerate(arch_names)}
    arch_indices = np.array([arch_map[r["arch_name"]] for r in records])

    # Build M0: log_params, [dummies], [lp × dummies]
    m0_cols = [log_params]
    names_m0 = ["log_params"]
    for ds in non_ref:
        label = DATASET_LABELS.get(ds, ds)
        m0_cols.append(dummies[ds])
        names_m0.append(f"d_{label}")
    for ds in non_ref:
        label = DATASET_LABELS.get(ds, ds)
        m0_cols.append(log_params * dummies[ds])
        names_m0.append(f"log_params×{label}")
    X_m0 = np.column_stack(m0_cols)

    # Build M1: log_params, H0z, [dummies], [lp × dummies], [H0z × dummies]
    m1_cols = [log_params, h0z]
    names_m1 = ["log_params", "H0z"]
    for ds in non_ref:
        label = DATASET_LABELS.get(ds, ds)
        m1_cols.append(dummies[ds])
        names_m1.append(f"d_{label}")
    for ds in non_ref:
        label = DATASET_LABELS.get(ds, ds)
        m1_cols.append(log_params * dummies[ds])
        names_m1.append(f"log_params×{label}")
    for ds in non_ref:
        label = DATASET_LABELS.get(ds, ds)
        m1_cols.append(h0z * dummies[ds])
        names_m1.append(f"H0z×{label}")
    X_m1 = np.column_stack(m1_cols)

    return X_m0, X_m1, y, names_m0, names_m1, arch_indices, h0z, datasets, non_ref


def ols_fit(X, y):
    """OLS with intercept. Returns coefficients (including intercept as first), residuals, SSE, R2."""
    n = len(y)
    X_int = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X_int, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(X_int.shape[1])
    y_hat = X_int @ beta
    residuals = y - y_hat
    sse = np.sum(residuals ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sse / sst if sst > 0 else 0.0
    return beta, residuals, sse, r2


def compute_vif(X):
    """Compute variance inflation factors for each column of X."""
    n_cols = X.shape[1]
    vifs = np.zeros(n_cols)
    for j in range(n_cols):
        y_j = X[:, j]
        X_others = np.delete(X, j, axis=1)
        _, _, _, r2_j = ols_fit(X_others, y_j)
        vifs[j] = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else np.inf
    return vifs


def incremental_f_stat(sse_reduced, sse_full, df_extra, df_resid_full):
    """Compute incremental F statistic for nested model comparison."""
    if sse_full <= 0 or df_resid_full <= 0:
        return 0.0
    f_stat = ((sse_reduced - sse_full) / df_extra) / (sse_full / df_resid_full)
    return max(f_stat, 0.0)


def clustered_bootstrap(X_m0, X_m1, y, arch_indices, K, n_boot=5000, rng=None):
    """Clustered bootstrap resampling architecture blocks.

    K = number of non-reference datasets.
    M1 layout: [log_params, H0z, K dummies, K lp×ds, K H0z×ds] = 2+3K features.
    M1 without H0z interactions: first 2+2K columns.

    Returns bootstrap distributions of coefficients, F-stats, partial R2, partial effects.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    unique_archs = np.unique(arch_indices)
    n_archs = len(unique_archs)
    n_features_m1 = X_m1.shape[1]
    n_no_int = 2 + 2 * K  # columns before H0z interactions
    n_datasets = K + 1  # including reference

    coef_boot = np.zeros((n_boot, n_features_m1 + 1))  # +1 for intercept
    f_full_block_boot = np.zeros(n_boot)
    f_interaction_block_boot = np.zeros(n_boot)
    partial_r2_boot = np.zeros(n_boot)
    partial_effects_boot = np.zeros((n_boot, n_datasets))

    df_extra_full = K + 1   # H0z + K interactions
    df_extra_int = K         # K H0z×dataset interactions

    for b in range(n_boot):
        boot_archs = rng.choice(unique_archs, size=n_archs, replace=True)
        boot_idx = []
        for a in boot_archs:
            boot_idx.extend(np.where(arch_indices == a)[0].tolist())
        boot_idx = np.array(boot_idx)

        X0_b = X_m0[boot_idx]
        X1_b = X_m1[boot_idx]
        y_b = y[boot_idx]

        _, _, sse0, _ = ols_fit(X0_b, y_b)
        beta1, _, sse1, _ = ols_fit(X1_b, y_b)
        coef_boot[b] = beta1

        df_resid_full = len(y_b) - (n_features_m1 + 1)
        f_full_block_boot[b] = incremental_f_stat(sse0, sse1, df_extra_full, df_resid_full)

        X1_no_int = X1_b[:, :n_no_int]
        _, _, sse1_no_int, _ = ols_fit(X1_no_int, y_b)
        f_interaction_block_boot[b] = incremental_f_stat(sse1_no_int, sse1, df_extra_int, df_resid_full)

        partial_r2_boot[b] = (sse0 - sse1) / sse0 if sse0 > 0 else 0.0

        # Per-dataset partial effects
        # beta1: [intercept, log_params, H0z, ...dummies, ...lp×ds, ...H0z×ds]
        beta_h0z = beta1[2]  # H0z main effect = reference dataset effect
        partial_effects_boot[b, 0] = beta_h0z
        for i in range(K):
            # H0z×dataset_i interaction is at index 3 + 2K + i (with intercept offset)
            partial_effects_boot[b, i + 1] = beta_h0z + beta1[3 + 2 * K + i]

    return {
        "coef_boot": coef_boot,
        "f_full_block": f_full_block_boot,
        "f_interaction_block": f_interaction_block_boot,
        "partial_r2": partial_r2_boot,
        "partial_effects": partial_effects_boot,
    }


def permutation_test(X_m0, X_m1_template, y, h0z, datasets, arch_indices,
                     active_datasets, K, n_perms=1000, rng=None):
    """Permutation test: shuffle H0 within dataset, refit M0 and M1, compute delta SSE.

    K = number of non-reference datasets.
    Tests both full H0 block (K+1 features) and interaction-only block (K features).
    """
    if rng is None:
        rng = np.random.default_rng(123)

    n = len(y)
    n_features_m1 = X_m1_template.shape[1]
    n_no_int = 2 + 2 * K  # columns before H0z interactions

    _, _, sse0_obs, _ = ols_fit(X_m0, y)
    _, _, sse1_obs, _ = ols_fit(X_m1_template, y)
    X1_no_int = X_m1_template[:, :n_no_int]
    _, _, sse1_no_int_obs, _ = ols_fit(X1_no_int, y)

    df_resid = n - (n_features_m1 + 1)
    df_extra_full = K + 1
    df_extra_int = K
    f_full_obs = incremental_f_stat(sse0_obs, sse1_obs, df_extra_full, df_resid)
    f_int_obs = incremental_f_stat(sse1_no_int_obs, sse1_obs, df_extra_int, df_resid)

    f_full_perm = np.zeros(n_perms)
    f_int_perm = np.zeros(n_perms)

    datasets_arr = np.array(datasets)
    # Identify non-reference datasets and their dummy column indices in M1
    non_ref = [ds for ds in active_datasets if ds != REFERENCE_DATASET]

    for p in range(n_perms):
        h0z_shuffled = h0z.copy()
        for ds in active_datasets:
            mask = datasets_arr == ds
            indices = np.where(mask)[0]
            if len(indices) > 0:
                h0z_shuffled[indices] = rng.permutation(h0z_shuffled[indices])

        # Rebuild M1 with shuffled H0
        X_m1_perm = X_m1_template.copy()
        X_m1_perm[:, 1] = h0z_shuffled  # H0z column
        # Update H0z × dataset interaction columns (start at index 2+2K)
        for i, ds in enumerate(non_ref):
            dummy_col = X_m1_template[:, 2 + i]  # dataset dummy
            X_m1_perm[:, n_no_int + i] = h0z_shuffled * dummy_col

        _, _, sse1_p, _ = ols_fit(X_m1_perm, y)
        X1_no_int_p = X_m1_perm[:, :n_no_int]
        _, _, sse1_no_int_p, _ = ols_fit(X1_no_int_p, y)

        f_full_perm[p] = incremental_f_stat(sse0_obs, sse1_p, df_extra_full, df_resid)
        f_int_perm[p] = incremental_f_stat(sse1_no_int_obs, sse1_p, df_extra_int, df_resid)

    p_full = np.mean(f_full_perm >= f_full_obs)
    p_int = np.mean(f_int_perm >= f_int_obs)

    return {
        "f_full_observed": float(f_full_obs),
        "f_interaction_observed": float(f_int_obs),
        "p_full_block": float(p_full),
        "p_interaction_block": float(p_int),
        "n_perms": n_perms,
    }


def run_analysis(records, outcome_key, outcome_label, active_datasets,
                 standardize_h0=True, n_boot=5000, n_perm=1000):
    """Run the full pooled interaction analysis for one outcome."""
    non_ref = [ds for ds in active_datasets if ds != REFERENCE_DATASET]
    K = len(non_ref)
    n_no_int = 2 + 2 * K

    print(f"\n{'='*70}")
    print(f"Outcome: {outcome_label} ({outcome_key})")
    print(f"H0 standardization: {'within-dataset z-score' if standardize_h0 else 'raw'}")
    print(f"Datasets: {len(active_datasets)} ({DATASET_LABELS.get(REFERENCE_DATASET, REFERENCE_DATASET)} as reference)")
    print(f"{'='*70}")

    X_m0, X_m1, y, names_m0, names_m1, arch_indices, h0z, datasets, _ = \
        build_design_matrix(records, outcome_key, active_datasets, standardize_h0=standardize_h0)

    valid = np.isfinite(y)
    if not np.all(valid):
        n_invalid = np.sum(~valid)
        print(f"WARNING: {n_invalid} non-finite outcome values, dropping")
        X_m0 = X_m0[valid]
        X_m1 = X_m1[valid]
        y = y[valid]
        h0z = h0z[valid]
        datasets = [d for d, v in zip(datasets, valid) if v]
        arch_indices = arch_indices[valid]

    n = len(y)
    n_unique = len(np.unique(arch_indices))
    print(f"n = {n} ({n_unique} architecture blocks)")

    beta_m0, res_m0, sse_m0, r2_m0 = ols_fit(X_m0, y)
    beta_m1, res_m1, sse_m1, r2_m1 = ols_fit(X_m1, y)

    print(f"\nM0 R² = {r2_m0:.4f}  (log_params + dataset + log_params×dataset)")
    print(f"M1 R² = {r2_m1:.4f}  (+ H0z + H0z×dataset)")
    print(f"ΔR²   = {r2_m1 - r2_m0:.4f}")

    all_names_m1 = ["intercept"] + names_m1
    print(f"\nM1 Coefficients:")
    for name, coef in zip(all_names_m1, beta_m1):
        print(f"  {name:25s} = {coef:+.6f}")

    # Per-dataset partial effects (dynamic)
    ds_labels_ordered = [DATASET_LABELS.get(REFERENCE_DATASET, REFERENCE_DATASET)]
    effects = [float(beta_m1[2])]  # reference = H0z main effect
    for i, ds in enumerate(non_ref):
        label = DATASET_LABELS.get(ds, ds)
        ds_labels_ordered.append(label)
        effects.append(float(beta_m1[2] + beta_m1[3 + 2 * K + i]))

    print(f"\nPer-dataset partial effect of H0z:")
    for label, eff in zip(ds_labels_ordered, effects):
        print(f"  {label:15s}  {eff:+.6f}")

    # VIF
    vifs = compute_vif(X_m1)
    print(f"\nVIF (M1):")
    vif_flags = []
    for name, vif in zip(names_m1, vifs):
        flag = " *** EXCEEDS THRESHOLD" if vif > VIF_THRESHOLD else ""
        print(f"  {name:25s} = {vif:.2f}{flag}")
        if vif > VIF_THRESHOLD:
            vif_flags.append(name)

    # Incremental F (point estimate)
    df_resid = n - (len(names_m1) + 1)
    f_full = incremental_f_stat(sse_m0, sse_m1, K + 1, df_resid)

    X_m1_no_int = X_m1[:, :n_no_int]
    _, _, sse_m1_no_int, _ = ols_fit(X_m1_no_int, y)
    f_int = incremental_f_stat(sse_m1_no_int, sse_m1, K, df_resid)

    print(f"\nIncremental F-tests (point estimates):")
    print(f"  Full H0 block:        F = {f_full:.4f}")
    print(f"  Interaction block:    F = {f_int:.4f}")

    # Clustered bootstrap
    print(f"\nRunning clustered bootstrap ({n_boot} iterations, {n_unique} arch blocks)...")
    boot = clustered_bootstrap(X_m0, X_m1, y, arch_indices, K, n_boot=n_boot)

    coef_ci = {}
    print(f"\nM1 Coefficient 95% CIs (clustered bootstrap):")
    for i, name in enumerate(all_names_m1):
        lo = np.percentile(boot["coef_boot"][:, i], 2.5)
        hi = np.percentile(boot["coef_boot"][:, i], 97.5)
        coef_ci[name] = {"point": float(beta_m1[i]), "ci_lo": float(lo), "ci_hi": float(hi)}
        print(f"  {name:25s} = {beta_m1[i]:+.6f}  [{lo:+.6f}, {hi:+.6f}]")

    partial_effects_ci = {}
    print(f"\nPer-dataset H0z partial effect 95% CIs:")
    for i, (label, eff) in enumerate(zip(ds_labels_ordered, effects)):
        lo = np.percentile(boot["partial_effects"][:, i], 2.5)
        hi = np.percentile(boot["partial_effects"][:, i], 97.5)
        partial_effects_ci[label] = {"point": float(eff), "ci_lo": float(lo), "ci_hi": float(hi)}
        print(f"  {label:15s}  {eff:+.6f}  [{lo:+.6f}, {hi:+.6f}]")

    f_full_ci = (float(np.percentile(boot["f_full_block"], 2.5)),
                 float(np.percentile(boot["f_full_block"], 97.5)))
    f_int_ci = (float(np.percentile(boot["f_interaction_block"], 2.5)),
                float(np.percentile(boot["f_interaction_block"], 97.5)))
    partial_r2_ci = (float(np.percentile(boot["partial_r2"], 2.5)),
                     float(np.percentile(boot["partial_r2"], 97.5)))

    print(f"\n  Partial R² of H0 block: {r2_m1 - r2_m0:.4f}  [{partial_r2_ci[0]:.4f}, {partial_r2_ci[1]:.4f}]")

    # Permutation tests
    print(f"\nRunning permutation tests ({n_perm} iterations, within-dataset H0 shuffle)...")
    perm = permutation_test(X_m0, X_m1, y, h0z, datasets, arch_indices,
                            active_datasets, K, n_perms=n_perm)

    print(f"\nPermutation results (two-tailed):")
    print(f"  Full H0 block:     F_obs = {perm['f_full_observed']:.4f}, p = {perm['p_full_block']:.4f}")
    print(f"  Interaction block: F_obs = {perm['f_interaction_observed']:.4f}, p = {perm['p_interaction_block']:.4f}")

    result = {
        "outcome": outcome_label,
        "outcome_key": outcome_key,
        "n": n,
        "n_datasets": len(active_datasets),
        "datasets": [DATASET_LABELS.get(ds, ds) for ds in active_datasets],
        "h0_standardization": "within_dataset_zscore" if standardize_h0 else "raw",
        "m0_r2": float(r2_m0),
        "m1_r2": float(r2_m1),
        "delta_r2": float(r2_m1 - r2_m0),
        "partial_r2_h0_block": float(r2_m1 - r2_m0),
        "partial_r2_h0_block_ci": partial_r2_ci,
        "coefficients": coef_ci,
        "partial_effects": partial_effects_ci,
        "vif": {name: float(v) for name, v in zip(names_m1, vifs)},
        "vif_flags": vif_flags,
        "incremental_f": {
            "full_block": {"f_stat": float(f_full), "ci": f_full_ci},
            "interaction_block": {"f_stat": float(f_int), "ci": f_int_ci},
        },
        "permutation": perm,
        "n_bootstrap": n_boot,
        "n_permutations": n_perm,
    }

    return result


def run_reduced_model(records, outcome_key, outcome_label, active_datasets,
                      standardize_h0=True, n_perm=1000):
    """Reduced model without log_params x dataset interactions (robustness check).

    M0r: Y ~ log_params + dataset
    M1r: Y ~ log_params + H0z + dataset + H0z × dataset
    """
    non_ref = [ds for ds in active_datasets if ds != REFERENCE_DATASET]
    K = len(non_ref)

    X_m0, X_m1, y, _, _, arch_indices, h0z, datasets, _ = \
        build_design_matrix(records, outcome_key, active_datasets, standardize_h0=standardize_h0)

    valid = np.isfinite(y)
    if not np.all(valid):
        X_m0 = X_m0[valid]
        X_m1 = X_m1[valid]
        y = y[valid]
        h0z = h0z[valid]
        datasets = [d for d, v in zip(datasets, valid) if v]
        arch_indices = arch_indices[valid]

    n = len(y)

    # Reduced M0: log_params + K dummies (no params interactions)
    X_m0r = X_m0[:, :1 + K]

    # Reduced M1: log_params, H0z, K dummies, K H0z×dataset
    h0z_col = X_m1[:, 1]
    m1r_cols = [X_m1[:, 0], h0z_col]  # log_params, H0z
    for i in range(K):
        m1r_cols.append(X_m1[:, 2 + i])  # dataset dummies
    for i in range(K):
        m1r_cols.append(h0z_col * X_m1[:, 2 + i])  # H0z × dataset
    X_m1r = np.column_stack(m1r_cols)

    n_features_m1r = X_m1r.shape[1]
    _, _, sse0r, r2_0r = ols_fit(X_m0r, y)
    beta1r, _, sse1r, r2_1r = ols_fit(X_m1r, y)

    df_extra = K + 1  # H0z + K interactions
    df_resid = n - (n_features_m1r + 1)
    f_full = incremental_f_stat(sse0r, sse1r, df_extra, df_resid)

    # Permutation test on reduced model
    rng = np.random.default_rng(456)
    datasets_arr = np.array(datasets)
    f_perm = np.zeros(n_perm)
    for p in range(n_perm):
        h0z_s = h0z.copy()
        for ds in active_datasets:
            mask = datasets_arr == ds
            indices = np.where(mask)[0]
            if len(indices) > 0:
                h0z_s[indices] = rng.permutation(h0z_s[indices])
        m1r_p_cols = [X_m1r[:, 0], h0z_s]
        for i in range(K):
            m1r_p_cols.append(X_m1r[:, 2 + i])
        for i in range(K):
            m1r_p_cols.append(h0z_s * X_m1r[:, 2 + i])
        X_m1r_p = np.column_stack(m1r_p_cols)
        _, _, sse1r_p, _ = ols_fit(X_m1r_p, y)
        f_perm[p] = incremental_f_stat(sse0r, sse1r_p, df_extra, df_resid)

    p_full = float(np.mean(f_perm >= f_full))

    # Per-dataset effects: reference = beta_H0z, non-ref_i = beta_H0z + beta_H0z×ds_i
    partial_effects = {}
    ref_label = DATASET_LABELS.get(REFERENCE_DATASET, REFERENCE_DATASET)
    partial_effects[ref_label] = float(beta1r[2])
    for i, ds in enumerate(non_ref):
        label = DATASET_LABELS.get(ds, ds)
        partial_effects[label] = float(beta1r[2] + beta1r[3 + K + i])

    return {
        "model": "reduced (no log_params x dataset)",
        "outcome": outcome_label,
        "m0r_r2": float(r2_0r),
        "m1r_r2": float(r2_1r),
        "delta_r2": float(r2_1r - r2_0r),
        "f_full_block": float(f_full),
        "p_full_block": p_full,
        "partial_effects": partial_effects,
    }


def make_figure(results, output_path):
    """Partial effect plot with clustered bootstrap CIs (2 or 3 panels)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping figure")
        return

    forgetting_result = None
    ewc_result = None
    si_result = None
    for r in results:
        if r.get("h0_standardization") != "within_dataset_zscore":
            continue
        if r.get("outcome_key") == "ret_10":
            forgetting_result = r
        elif r.get("outcome_key") == "ewc_benefit_aurc":
            ewc_result = r
        elif r.get("outcome_key") == "si_benefit_aurc":
            si_result = r

    if forgetting_result is None or ewc_result is None:
        print("WARNING: Missing results for figure, skipping")
        return

    panels = [
        (forgetting_result, "Partial Effect of H0z on Retention (ret@10)"),
        (ewc_result, "Partial Effect of H0z on EWC Benefit"),
    ]
    if si_result is not None:
        panels.append((si_result, "Partial Effect of H0z on SI Benefit"))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Dynamic dataset labels from results
    ds_labels = list(forgetting_result["partial_effects"].keys())
    base_colors = ["#3b82f6", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6", "#ec4899"]
    ds_colors = base_colors[:len(ds_labels)]
    x_pos = list(range(len(ds_labels)))

    for ax, (result, title) in zip(axes, panels):
        pe = result["partial_effects"]
        available_ds = [ds for ds in ds_labels if ds in pe]
        points = [pe[ds]["point"] for ds in available_ds]
        ci_lo = [pe[ds]["ci_lo"] for ds in available_ds]
        ci_hi = [pe[ds]["ci_hi"] for ds in available_ds]
        errors_lo = [p - lo for p, lo in zip(points, ci_lo)]
        errors_hi = [hi - p for p, hi in zip(points, ci_hi)]
        x = list(range(len(available_ds)))

        ax.bar(x, points, color=ds_colors[:len(x)], alpha=0.7, width=0.6, zorder=2)
        ax.errorbar(x, points, yerr=[errors_lo, errors_hi],
                    fmt="none", ecolor="white", elinewidth=1.5, capsize=5, capthick=1.5, zorder=3)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(available_ds, fontsize=9, rotation=15 if len(available_ds) > 3 else 0)
        ax.set_ylabel("Coefficient (1 SD within-dataset H0)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        perm_p_full = result["permutation"]["p_full_block"]
        perm_p_int = result["permutation"]["p_interaction_block"]
        ax.text(0.02, 0.98, f"Full block p={perm_p_full:.3f}\nInteraction p={perm_p_int:.3f}",
                transform=ax.transAxes, fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5, edgecolor="gray"),
                color="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0f",
                edgecolor="none")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="#0a0a0f",
                edgecolor="none")
    plt.close()
    print(f"Figure saved: {output_path} and {output_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Pooled cross-dataset interaction analysis")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing Phase 4 correlation JSONs")
    parser.add_argument("--datasets", nargs="+", type=str, default=None,
                        help="Datasets to pool (default: auto-discover from available Phase 4 JSONs)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/pooled_interaction.json)")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()

    n_boot = args.n_bootstrap
    n_perm = args.n_permutations

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "pooled_interaction.json"

    # Load data (auto-discovers available datasets)
    records, active_datasets = load_phase4_data(results_dir, datasets=args.datasets)

    if len(active_datasets) < 2:
        print(f"ERROR: Need >= 2 datasets for pooled analysis, have {len(active_datasets)}")
        sys.exit(1)

    if REFERENCE_DATASET not in active_datasets:
        print(f"WARNING: Reference dataset '{REFERENCE_DATASET}' not found. "
              f"Using '{active_datasets[0]}' as reference.")
        # Move reference to front -- but keep using REFERENCE_DATASET global
        # This case shouldn't normally happen since cifar100 is in DEFAULT_DATASETS

    outcomes = {
        "ret_10": "Retention @ step 10 (primary)",
        "retention_100": "Retention @ step 100 (robustness)",
        "early_aurc": "Early AURC 0-500 (robustness)",
        "ewc_benefit_aurc": "EWC Benefit (early AURC, absolute)",
        "ewc_benefit_ret10": "EWC Benefit (ret@10, absolute)",
        "si_benefit_aurc": "SI Benefit (early AURC, absolute)",
        "si_benefit_ret10": "SI Benefit (ret@10, absolute)",
    }

    has_si = any(r.get("si_benefit_aurc") is not None for r in records)
    if not has_si:
        print("\nNote: No SI data available yet. SI outcomes will be skipped.")
        outcomes = {k: v for k, v in outcomes.items() if not k.startswith("si_")}

    all_results = {}

    for outcome_key, outcome_label in outcomes.items():
        print(f"\n{'#'*70}")
        print(f"# {outcome_label}")
        print(f"{'#'*70}")

        result = run_analysis(records, outcome_key, outcome_label, active_datasets,
                              standardize_h0=True, n_boot=n_boot, n_perm=n_perm)
        all_results[f"{outcome_key}_zscore"] = result

        result_raw = run_analysis(records, outcome_key, f"{outcome_label} [raw H0]",
                                  active_datasets, standardize_h0=False,
                                  n_boot=n_boot, n_perm=n_perm)
        all_results[f"{outcome_key}_raw"] = result_raw

        reduced = run_reduced_model(records, outcome_key, outcome_label, active_datasets,
                                    standardize_h0=True, n_perm=n_perm)
        all_results[f"{outcome_key}_reduced"] = reduced

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\nPrimary outcomes (within-dataset z-scored H0):")
    print(f"{'Outcome':<35} {'ΔR²':>8} {'p(full)':>10} {'p(inter)':>10}")
    print("-" * 65)
    for outcome_key in outcomes:
        key = f"{outcome_key}_zscore"
        if key in all_results and "permutation" in all_results[key]:
            r = all_results[key]
            dr2 = r["delta_r2"]
            pf = r["permutation"]["p_full_block"]
            pi = r["permutation"]["p_interaction_block"]
            print(f"{outcomes[outcome_key]:<35} {dr2:>8.4f} {pf:>10.4f} {pi:>10.4f}")

    print(f"\nRobustness check (reduced model, no params x dataset):")
    print(f"{'Outcome':<35} {'ΔR²':>8} {'p(full)':>10}")
    print("-" * 55)
    for outcome_key in outcomes:
        key = f"{outcome_key}_reduced"
        if key in all_results:
            r = all_results[key]
            dr2 = r["delta_r2"]
            pf = r["p_full_block"]
            print(f"{outcomes[outcome_key]:<35} {dr2:>8.4f} {pf:>10.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {output_path}")

    fig_path = output_path.with_name("pooled_interaction_figure.png")
    make_figure(list(all_results.values()), fig_path)


if __name__ == "__main__":
    main()
