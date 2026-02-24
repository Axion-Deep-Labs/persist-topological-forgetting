"""
Phase 6: Pooled Cross-Dataset Interaction Analysis

Tests two claims with formal inferential statistics:
  Claim 1: Topology is a conditional predictor of forgetting (dataset moderates H0 effect)
  Claim 2: Topology predicts EWC benefit on some datasets (dataset moderates H0-EWC relationship)

Design:
  - Pooled n=57 (19 architectures x 3 datasets)
  - OLS with clustered bootstrap (19 architecture blocks) for CIs
  - Permutation tests (H0 shuffled within dataset) for interaction block
  - CIFAR-100 as reference category

Models:
  M0: Y ~ log_params + dataset + log_params x dataset
  M1: Y ~ log_params + H0z + dataset + log_params x dataset + H0z x dataset

Tests:
  Primary: block test of {H0z, H0z x CUB, H0z x RESISC} (does topology help at all?)
  Secondary: block test of {H0z x CUB, H0z x RESISC} (does dataset moderate topology?)

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
DATASETS = ["cifar100", "cub200", "resisc45"]
DATASET_LABELS = {"cifar100": "CIFAR-100", "cub200": "CUB-200", "resisc45": "RESISC-45"}
REFERENCE_DATASET = "cifar100"


def load_phase4_data(results_dir: Path):
    """Load and validate Phase 4 correlation JSONs for all 3 datasets."""
    records = []
    arch_names_by_dataset = {}

    for ds in DATASETS:
        fpath = results_dir / f"correlation_results_{ds}.json"
        if not fpath.exists():
            print(f"ERROR: Missing {fpath}")
            sys.exit(1)

        with open(fpath) as f:
            data = json.load(f)

        per_arch = data["per_architecture"]
        if len(per_arch) != 19:
            print(f"ERROR: {ds} has {len(per_arch)} architectures, expected 19")
            sys.exit(1)

        arch_names = [a["arch_name"] for a in per_arch]
        arch_names_by_dataset[ds] = arch_names

        # Load EWC benefits (pre-computed in correlations.ewc_benefit)
        ewc_section = data.get("correlations", {}).get("ewc_benefit", {})
        ewc_benefits = ewc_section.get("benefits", [None] * 19)
        ewc_arch_names = ewc_section.get("architectures", arch_names)

        # Build EWC benefit lookup by architecture name
        ewc_lookup = {}
        for name, benefit in zip(ewc_arch_names, ewc_benefits):
            ewc_lookup[name] = benefit

        for arch in per_arch:
            name = arch["arch_name"]
            ewc_benefit = ewc_lookup.get(name)

            # Compute EWC benefit for ret@10 as absolute difference
            ewc_ret10 = arch.get("ewc_ret_10")
            naive_ret10 = arch.get("ret_10")
            ewc_benefit_ret10 = None
            if ewc_ret10 is not None and naive_ret10 is not None:
                ewc_benefit_ret10 = ewc_ret10 - naive_ret10

            records.append({
                "arch_name": name,
                "dataset": ds,
                "num_params": arch["num_params"],
                "H0": arch["H0"],
                "H1": arch.get("H1", 0.0),
                "ret_10": arch["ret_10"],
                "retention_100": arch["retention_100"],
                "early_aurc": arch["early_aurc"],
                "ewc_benefit_aurc": ewc_benefit,  # ewc_early_aurc - early_aurc
                "ewc_benefit_ret10": ewc_benefit_ret10,
            })

    # Validate: same 19 architecture names across all datasets
    ref_names = sorted(arch_names_by_dataset[REFERENCE_DATASET])
    for ds in DATASETS:
        ds_names = sorted(arch_names_by_dataset[ds])
        if ds_names != ref_names:
            print(f"ERROR: Architecture mismatch between {REFERENCE_DATASET} and {ds}")
            print(f"  {REFERENCE_DATASET}: {ref_names}")
            print(f"  {ds}: {ds_names}")
            sys.exit(1)

    # Validate: log_params matches across datasets for same architecture
    params_by_arch = {}
    for r in records:
        name = r["arch_name"]
        if name not in params_by_arch:
            params_by_arch[name] = r["num_params"]
        elif params_by_arch[name] != r["num_params"]:
            print(f"WARNING: {name} has different param counts across datasets "
                  f"({params_by_arch[name]} vs {r['num_params']}). Using first.")

    print(f"Loaded {len(records)} records (19 architectures x 3 datasets)")
    print(f"Architecture names: {ref_names}")
    return records


def build_design_matrix(records, outcome_key, standardize_h0=True, standardize_params="global"):
    """Build the pooled design matrix and outcome vector.

    Returns X_m0 (M0 features), X_m1 (M1 features), y, feature_names_m0, feature_names_m1,
    arch_indices (for clustering), and dataset labels.
    """
    n = len(records)

    # Outcome
    y = np.array([r[outcome_key] for r in records], dtype=np.float64)

    # Log params
    log_params = np.log(np.array([r["num_params"] for r in records], dtype=np.float64))
    if standardize_params == "global":
        log_params = (log_params - log_params.mean()) / log_params.std()

    # H0
    h0_raw = np.array([r["H0"] for r in records], dtype=np.float64)
    datasets = [r["dataset"] for r in records]

    if standardize_h0:
        # Within-dataset z-score
        h0z = np.zeros(n)
        for ds in DATASETS:
            mask = np.array([d == ds for d in datasets])
            vals = h0_raw[mask]
            if vals.std() > 0:
                h0z[mask] = (vals - vals.mean()) / vals.std()
            else:
                h0z[mask] = 0.0
    else:
        h0z = h0_raw.copy()

    # Dataset dummies (CIFAR as reference)
    d_cub = np.array([1.0 if d == "cub200" else 0.0 for d in datasets])
    d_resisc = np.array([1.0 if d == "resisc45" else 0.0 for d in datasets])

    # Interactions
    lp_x_cub = log_params * d_cub
    lp_x_resisc = log_params * d_resisc
    h0_x_cub = h0z * d_cub
    h0_x_resisc = h0z * d_resisc

    # Architecture indices for clustering
    arch_names = sorted(set(r["arch_name"] for r in records))
    arch_map = {name: i for i, name in enumerate(arch_names)}
    arch_indices = np.array([arch_map[r["arch_name"]] for r in records])

    # M0: log_params, d_cub, d_resisc, lp_x_cub, lp_x_resisc
    X_m0 = np.column_stack([log_params, d_cub, d_resisc, lp_x_cub, lp_x_resisc])
    names_m0 = ["log_params", "d_CUB", "d_RESISC", "log_params×CUB", "log_params×RESISC"]

    # M1: M0 + h0z, h0_x_cub, h0_x_resisc
    X_m1 = np.column_stack([log_params, h0z, d_cub, d_resisc, lp_x_cub, lp_x_resisc,
                            h0_x_cub, h0_x_resisc])
    names_m1 = ["log_params", "H0z", "d_CUB", "d_RESISC", "log_params×CUB", "log_params×RESISC",
                "H0z×CUB", "H0z×RESISC"]

    return X_m0, X_m1, y, names_m0, names_m1, arch_indices, h0z, datasets


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


def clustered_bootstrap(X_m0, X_m1, y, arch_indices, n_boot=5000, rng=None):
    """Clustered bootstrap resampling 19 architecture blocks.

    Returns bootstrap distributions of:
    - M1 coefficients
    - Incremental F (M0 vs M1, full block)
    - Incremental F (M1-no-interaction vs M1, interaction block only)
    - Partial R2 of H0 block
    - Per-dataset partial effects of H0z
    """
    if rng is None:
        rng = np.random.default_rng(42)

    unique_archs = np.unique(arch_indices)
    n_archs = len(unique_archs)
    n_features_m1 = X_m1.shape[1]

    # Storage
    coef_boot = np.zeros((n_boot, n_features_m1 + 1))  # +1 for intercept
    f_full_block_boot = np.zeros(n_boot)
    f_interaction_block_boot = np.zeros(n_boot)
    partial_r2_boot = np.zeros(n_boot)
    # Per-dataset partial effects: H0z effect on each dataset
    # CIFAR = beta_H0z, CUB = beta_H0z + beta_H0z×CUB, RESISC = beta_H0z + beta_H0z×RESISC
    partial_effects_boot = np.zeros((n_boot, 3))

    for b in range(n_boot):
        # Resample architecture blocks
        boot_archs = rng.choice(unique_archs, size=n_archs, replace=True)
        boot_idx = []
        for a in boot_archs:
            boot_idx.extend(np.where(arch_indices == a)[0].tolist())
        boot_idx = np.array(boot_idx)

        X0_b = X_m0[boot_idx]
        X1_b = X_m1[boot_idx]
        y_b = y[boot_idx]

        # M0 fit
        _, _, sse0, r2_0 = ols_fit(X0_b, y_b)

        # M1 fit
        beta1, _, sse1, r2_1 = ols_fit(X1_b, y_b)
        coef_boot[b] = beta1

        # Full block incremental F: {H0z, H0z×CUB, H0z×RESISC}
        df_extra_full = 3  # H0z + 2 interactions
        df_resid_full = len(y_b) - (n_features_m1 + 1)
        f_full_block_boot[b] = incremental_f_stat(sse0, sse1, df_extra_full, df_resid_full)

        # Interaction block only: test {H0z×CUB, H0z×RESISC} given H0z main effect
        # M1-no-interaction: log_params, H0z, d_CUB, d_RESISC, lp×CUB, lp×RESISC (no H0 interactions)
        X1_no_int = X1_b[:, :6]  # First 6 columns
        _, _, sse1_no_int, _ = ols_fit(X1_no_int, y_b)
        df_extra_int = 2  # H0z×CUB, H0z×RESISC
        f_interaction_block_boot[b] = incremental_f_stat(sse1_no_int, sse1, df_extra_int, df_resid_full)

        # Partial R2 of H0 block
        partial_r2_boot[b] = (sse0 - sse1) / sse0 if sse0 > 0 else 0.0

        # Per-dataset partial effects
        # beta1 layout: [intercept, log_params, H0z, d_CUB, d_RESISC, lp×CUB, lp×RESISC, H0z×CUB, H0z×RESISC]
        beta_h0z = beta1[2]            # H0z main effect (= CIFAR effect)
        beta_h0z_cub = beta1[7]        # H0z × CUB interaction
        beta_h0z_resisc = beta1[8]     # H0z × RESISC interaction
        partial_effects_boot[b, 0] = beta_h0z                      # CIFAR
        partial_effects_boot[b, 1] = beta_h0z + beta_h0z_cub       # CUB
        partial_effects_boot[b, 2] = beta_h0z + beta_h0z_resisc    # RESISC

    return {
        "coef_boot": coef_boot,
        "f_full_block": f_full_block_boot,
        "f_interaction_block": f_interaction_block_boot,
        "partial_r2": partial_r2_boot,
        "partial_effects": partial_effects_boot,
    }


def permutation_test(X_m0, X_m1_template, y, h0z, datasets, arch_indices,
                     n_perms=1000, rng=None):
    """Permutation test: shuffle H0 within dataset, refit M0 and M1, compute delta SSE.

    Tests both:
    - Full block: {H0z, H0z×CUB, H0z×RESISC}
    - Interaction only: {H0z×CUB, H0z×RESISC}

    Returns observed F-stats and permutation p-values (two-tailed).
    """
    if rng is None:
        rng = np.random.default_rng(123)

    n = len(y)
    n_features_m1 = X_m1_template.shape[1]

    # Observed statistics
    _, _, sse0_obs, _ = ols_fit(X_m0, y)
    _, _, sse1_obs, _ = ols_fit(X_m1_template, y)
    X1_no_int = X_m1_template[:, :6]
    _, _, sse1_no_int_obs, _ = ols_fit(X1_no_int, y)

    df_resid = n - (n_features_m1 + 1)
    f_full_obs = incremental_f_stat(sse0_obs, sse1_obs, 3, df_resid)
    f_int_obs = incremental_f_stat(sse1_no_int_obs, sse1_obs, 2, df_resid)

    f_full_perm = np.zeros(n_perms)
    f_int_perm = np.zeros(n_perms)

    datasets_arr = np.array(datasets)

    for p in range(n_perms):
        # Shuffle H0 within each dataset
        h0z_shuffled = h0z.copy()
        for ds in DATASETS:
            mask = datasets_arr == ds
            indices = np.where(mask)[0]
            h0z_shuffled[indices] = rng.permutation(h0z_shuffled[indices])

        # Rebuild M1 with shuffled H0
        X_m1_perm = X_m1_template.copy()
        X_m1_perm[:, 1] = h0z_shuffled  # H0z column
        d_cub = X_m1_template[:, 2]
        d_resisc = X_m1_template[:, 3]
        X_m1_perm[:, 6] = h0z_shuffled * d_cub      # H0z × CUB
        X_m1_perm[:, 7] = h0z_shuffled * d_resisc    # H0z × RESISC

        _, _, sse1_p, _ = ols_fit(X_m1_perm, y)
        X1_no_int_p = X_m1_perm[:, :6]
        _, _, sse1_no_int_p, _ = ols_fit(X1_no_int_p, y)

        f_full_perm[p] = incremental_f_stat(sse0_obs, sse1_p, 3, df_resid)
        f_int_perm[p] = incremental_f_stat(sse1_no_int_obs, sse1_p, 2, df_resid)

    # Two-tailed p-values
    p_full = np.mean(f_full_perm >= f_full_obs)
    p_int = np.mean(f_int_perm >= f_int_obs)

    return {
        "f_full_observed": float(f_full_obs),
        "f_interaction_observed": float(f_int_obs),
        "p_full_block": float(p_full),
        "p_interaction_block": float(p_int),
        "n_perms": n_perms,
    }


def run_analysis(records, outcome_key, outcome_label, standardize_h0=True,
                 n_boot=5000, n_perm=1000):
    """Run the full pooled interaction analysis for one outcome."""
    print(f"\n{'='*70}")
    print(f"Outcome: {outcome_label} ({outcome_key})")
    print(f"H0 standardization: {'within-dataset z-score' if standardize_h0 else 'raw'}")
    print(f"{'='*70}")

    X_m0, X_m1, y, names_m0, names_m1, arch_indices, h0z, datasets = \
        build_design_matrix(records, outcome_key, standardize_h0=standardize_h0)

    # Check for NaN/Inf
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
    print(f"n = {n}")

    # --- Point estimates ---
    beta_m0, res_m0, sse_m0, r2_m0 = ols_fit(X_m0, y)
    beta_m1, res_m1, sse_m1, r2_m1 = ols_fit(X_m1, y)

    print(f"\nM0 R² = {r2_m0:.4f}  (log_params + dataset + log_params×dataset)")
    print(f"M1 R² = {r2_m1:.4f}  (+ H0z + H0z×dataset)")
    print(f"ΔR²   = {r2_m1 - r2_m0:.4f}")

    # Coefficients
    all_names_m1 = ["intercept"] + names_m1
    print(f"\nM1 Coefficients:")
    for name, coef in zip(all_names_m1, beta_m1):
        print(f"  {name:20s} = {coef:+.6f}")

    # Per-dataset partial effects
    beta_h0z = beta_m1[2]
    beta_h0z_cub = beta_m1[7]
    beta_h0z_resisc = beta_m1[8]
    effect_cifar = beta_h0z
    effect_cub = beta_h0z + beta_h0z_cub
    effect_resisc = beta_h0z + beta_h0z_resisc

    print(f"\nPer-dataset partial effect of H0z:")
    print(f"  CIFAR-100:  {effect_cifar:+.6f}")
    print(f"  CUB-200:    {effect_cub:+.6f}")
    print(f"  RESISC-45:  {effect_resisc:+.6f}")

    # --- VIF ---
    vifs = compute_vif(X_m1)
    print(f"\nVIF (M1):")
    vif_flags = []
    for name, vif in zip(names_m1, vifs):
        flag = " *** EXCEEDS THRESHOLD" if vif > VIF_THRESHOLD else ""
        print(f"  {name:20s} = {vif:.2f}{flag}")
        if vif > VIF_THRESHOLD:
            vif_flags.append(name)

    # --- Incremental F (point estimate) ---
    df_resid = n - (len(names_m1) + 1)
    f_full = incremental_f_stat(sse_m0, sse_m1, 3, df_resid)

    X_m1_no_int = X_m1[:, :6]
    _, _, sse_m1_no_int, r2_m1_no_int = ols_fit(X_m1_no_int, y)
    f_int = incremental_f_stat(sse_m1_no_int, sse_m1, 2, df_resid)

    print(f"\nIncremental F-tests (point estimates):")
    print(f"  Full H0 block:        F = {f_full:.4f}")
    print(f"  Interaction block:    F = {f_int:.4f}")

    # --- Clustered bootstrap ---
    print(f"\nRunning clustered bootstrap ({n_boot} iterations, 19 arch blocks)...")
    boot = clustered_bootstrap(X_m0, X_m1, y, arch_indices, n_boot=n_boot)

    # Coefficient CIs (95%)
    coef_ci = {}
    print(f"\nM1 Coefficient 95% CIs (clustered bootstrap):")
    for i, name in enumerate(all_names_m1):
        lo = np.percentile(boot["coef_boot"][:, i], 2.5)
        hi = np.percentile(boot["coef_boot"][:, i], 97.5)
        coef_ci[name] = {"point": float(beta_m1[i]), "ci_lo": float(lo), "ci_hi": float(hi)}
        print(f"  {name:20s} = {beta_m1[i]:+.6f}  [{lo:+.6f}, {hi:+.6f}]")

    # Partial effects CIs
    partial_effects_ci = {}
    print(f"\nPer-dataset H0z partial effect 95% CIs:")
    for i, ds_label in enumerate(["CIFAR-100", "CUB-200", "RESISC-45"]):
        pe = [effect_cifar, effect_cub, effect_resisc][i]
        lo = np.percentile(boot["partial_effects"][:, i], 2.5)
        hi = np.percentile(boot["partial_effects"][:, i], 97.5)
        partial_effects_ci[ds_label] = {"point": float(pe), "ci_lo": float(lo), "ci_hi": float(hi)}
        print(f"  {ds_label:12s}  {pe:+.6f}  [{lo:+.6f}, {hi:+.6f}]")

    # Bootstrap F CIs
    f_full_ci = (float(np.percentile(boot["f_full_block"], 2.5)),
                 float(np.percentile(boot["f_full_block"], 97.5)))
    f_int_ci = (float(np.percentile(boot["f_interaction_block"], 2.5)),
                float(np.percentile(boot["f_interaction_block"], 97.5)))
    partial_r2_ci = (float(np.percentile(boot["partial_r2"], 2.5)),
                     float(np.percentile(boot["partial_r2"], 97.5)))

    print(f"\n  Partial R² of H0 block: {r2_m1 - r2_m0:.4f}  [{partial_r2_ci[0]:.4f}, {partial_r2_ci[1]:.4f}]")

    # --- Permutation tests ---
    print(f"\nRunning permutation tests ({n_perm} iterations, within-dataset H0 shuffle)...")
    perm = permutation_test(X_m0, X_m1, y, h0z, datasets, arch_indices, n_perms=n_perm)

    print(f"\nPermutation results (two-tailed):")
    print(f"  Full H0 block:     F_obs = {perm['f_full_observed']:.4f}, p = {perm['p_full_block']:.4f}")
    print(f"  Interaction block: F_obs = {perm['f_interaction_observed']:.4f}, p = {perm['p_interaction_block']:.4f}")

    # --- Assemble results ---
    result = {
        "outcome": outcome_label,
        "outcome_key": outcome_key,
        "n": n,
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


def run_reduced_model(records, outcome_key, outcome_label, standardize_h0=True, n_perm=1000):
    """Reduced model without log_params x dataset interactions (robustness check).

    M0r: Y ~ log_params + dataset
    M1r: Y ~ log_params + H0z + dataset + H0z × dataset
    """
    X_m0, X_m1, y, _, _, arch_indices, h0z, datasets = \
        build_design_matrix(records, outcome_key, standardize_h0=standardize_h0)

    valid = np.isfinite(y)
    if not np.all(valid):
        X_m0 = X_m0[valid]
        X_m1 = X_m1[valid]
        y = y[valid]
        h0z = h0z[valid]
        datasets = [d for d, v in zip(datasets, valid) if v]
        arch_indices = arch_indices[valid]

    n = len(y)

    # Reduced M0: log_params, d_CUB, d_RESISC (no params interactions)
    X_m0r = X_m0[:, :3]

    # Reduced M1: log_params, H0z, d_CUB, d_RESISC, H0z×CUB, H0z×RESISC
    d_cub = X_m1[:, 2]
    d_resisc = X_m1[:, 3]
    h0z_col = X_m1[:, 1]
    X_m1r = np.column_stack([X_m1[:, 0], h0z_col, d_cub, d_resisc,
                             h0z_col * d_cub, h0z_col * d_resisc])

    _, _, sse0r, r2_0r = ols_fit(X_m0r, y)
    beta1r, _, sse1r, r2_1r = ols_fit(X_m1r, y)

    df_resid = n - 7  # 6 features + intercept
    f_full = incremental_f_stat(sse0r, sse1r, 3, df_resid)

    # Permutation test on reduced model
    rng = np.random.default_rng(456)
    datasets_arr = np.array(datasets)
    f_perm = np.zeros(n_perm)
    for p in range(n_perm):
        h0z_s = h0z.copy()
        for ds in DATASETS:
            mask = datasets_arr == ds
            indices = np.where(mask)[0]
            h0z_s[indices] = rng.permutation(h0z_s[indices])
        X_m1r_p = np.column_stack([X_m1r[:, 0], h0z_s, d_cub, d_resisc,
                                   h0z_s * d_cub, h0z_s * d_resisc])
        _, _, sse1r_p, _ = ols_fit(X_m1r_p, y)
        f_perm[p] = incremental_f_stat(sse0r, sse1r_p, 3, df_resid)

    p_full = float(np.mean(f_perm >= f_full))

    # Per-dataset effects from reduced model
    # beta1r: [intercept, log_params, H0z, d_CUB, d_RESISC, H0z×CUB, H0z×RESISC]
    effect_cifar = float(beta1r[2])
    effect_cub = float(beta1r[2] + beta1r[5])
    effect_resisc = float(beta1r[2] + beta1r[6])

    return {
        "model": "reduced (no log_params × dataset)",
        "outcome": outcome_label,
        "m0r_r2": float(r2_0r),
        "m1r_r2": float(r2_1r),
        "delta_r2": float(r2_1r - r2_0r),
        "f_full_block": float(f_full),
        "p_full_block": p_full,
        "partial_effects": {
            "CIFAR-100": effect_cifar,
            "CUB-200": effect_cub,
            "RESISC-45": effect_resisc,
        },
    }


def make_figure(results, output_path):
    """Two-panel partial effect plot with clustered bootstrap CIs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping figure")
        return

    # Find forgetting and EWC results
    forgetting_result = None
    ewc_result = None
    for r in results:
        if r.get("outcome_key") == "ret_10" and r.get("h0_standardization") == "within_dataset_zscore":
            forgetting_result = r
        elif r.get("outcome_key") == "ewc_benefit_aurc" and r.get("h0_standardization") == "within_dataset_zscore":
            ewc_result = r

    if forgetting_result is None or ewc_result is None:
        print("WARNING: Missing results for figure, skipping")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ds_labels = ["CIFAR-100", "CUB-200", "RESISC-45"]
    ds_colors = ["#3b82f6", "#f59e0b", "#ef4444"]
    x_pos = [0, 1, 2]

    for ax, result, title in [
        (ax1, forgetting_result, "Partial Effect of H0z on Retention (ret@10)"),
        (ax2, ewc_result, "Partial Effect of H0z on EWC Benefit"),
    ]:
        pe = result["partial_effects"]
        points = [pe[ds]["point"] for ds in ds_labels]
        ci_lo = [pe[ds]["ci_lo"] for ds in ds_labels]
        ci_hi = [pe[ds]["ci_hi"] for ds in ds_labels]
        errors_lo = [p - lo for p, lo in zip(points, ci_lo)]
        errors_hi = [hi - p for p, hi in zip(points, ci_hi)]

        ax.bar(x_pos, points, color=ds_colors, alpha=0.7, width=0.6, zorder=2)
        ax.errorbar(x_pos, points, yerr=[errors_lo, errors_hi],
                    fmt="none", ecolor="white", elinewidth=1.5, capsize=5, capthick=1.5, zorder=3)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ds_labels, fontsize=10)
        ax.set_ylabel("Coefficient (1 SD within-dataset H0)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add p-values from permutation
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
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/pooled_interaction.json)")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()

    n_boot = args.n_bootstrap
    n_perm = args.n_permutations

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "pooled_interaction.json"

    # Load data
    records = load_phase4_data(results_dir)

    # Define outcomes
    outcomes = {
        "ret_10": "Retention @ step 10 (primary)",
        "retention_100": "Retention @ step 100 (robustness)",
        "early_aurc": "Early AURC 0-500 (robustness)",
        "ewc_benefit_aurc": "EWC Benefit (early AURC, absolute)",
        "ewc_benefit_ret10": "EWC Benefit (ret@10, absolute)",
    }

    all_results = {}

    for outcome_key, outcome_label in outcomes.items():
        print(f"\n{'#'*70}")
        print(f"# {outcome_label}")
        print(f"{'#'*70}")

        # Primary: within-dataset z-scored H0
        result = run_analysis(records, outcome_key, outcome_label, standardize_h0=True,
                              n_boot=n_boot, n_perm=n_perm)
        all_results[f"{outcome_key}_zscore"] = result

        # Sensitivity: raw H0
        result_raw = run_analysis(records, outcome_key, f"{outcome_label} [raw H0]",
                                  standardize_h0=False, n_boot=n_boot, n_perm=n_perm)
        all_results[f"{outcome_key}_raw"] = result_raw

        # Robustness: reduced model (no params × dataset interactions)
        reduced = run_reduced_model(records, outcome_key, outcome_label, standardize_h0=True,
                                    n_perm=n_perm)
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

    print(f"\nRobustness check (reduced model, no params × dataset):")
    print(f"{'Outcome':<35} {'ΔR²':>8} {'p(full)':>10}")
    print("-" * 55)
    for outcome_key in outcomes:
        key = f"{outcome_key}_reduced"
        if key in all_results:
            r = all_results[key]
            dr2 = r["delta_r2"]
            pf = r["p_full_block"]
            print(f"{outcomes[outcome_key]:<35} {dr2:>8.4f} {pf:>10.4f}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {output_path}")

    # Figure
    fig_path = output_path.with_name("pooled_interaction_figure.png")
    make_figure(list(all_results.values()), fig_path)


if __name__ == "__main__":
    main()
