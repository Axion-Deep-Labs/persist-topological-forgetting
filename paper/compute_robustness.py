#!/usr/bin/env python3
"""
Robustness analyses for ArXiv paper.
Computes: bootstrap CIs for Spearman, Kendall tau, leave-one-out influence,
LOAO fold variance, Cook's distance for pooled OLS.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATASETS = ["cifar100", "cub200", "resisc45"]
DS_LABELS = {"cifar100": "CIFAR-100", "cub200": "CUB-200", "resisc45": "RESISC-45"}
N_BOOT = 10000
RNG = np.random.default_rng(42)


def load_phase4():
    """Load per-architecture data from Phase 4 correlation results."""
    all_records = {}
    for ds in DATASETS:
        path = RESULTS_DIR / f"correlation_results_{ds}.json"
        with open(path) as f:
            data = json.load(f)
        records = []
        for arch in data["per_architecture"]:
            ewc_aurc = arch.get("ewc_early_aurc")
            naive_aurc = arch.get("early_aurc")
            ewc_benefit = (ewc_aurc - naive_aurc) if (ewc_aurc is not None and naive_aurc is not None) else None
            records.append({
                "arch_name": arch["arch_name"],
                "num_params": arch["num_params"],
                "H0": arch["H0"],
                "H1": arch["H1"],
                "ret_10": arch["ret_10"],
                "retention_100": arch["retention_100"],
                "early_aurc": naive_aurc,
                "ewc_early_aurc": ewc_aurc,
                "ewc_benefit_aurc": ewc_benefit,
            })
        all_records[ds] = records
    return all_records


def bootstrap_spearman_ci(x, y, n_boot=N_BOOT, alpha=0.05):
    """Bootstrap 95% CI for Spearman rho."""
    n = len(x)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.choice(n, size=n, replace=True)
        rhos[i] = stats.spearmanr(x[idx], y[idx]).statistic
    lo = np.percentile(rhos, 100 * alpha / 2)
    hi = np.percentile(rhos, 100 * (1 - alpha / 2))
    return lo, hi


def leave_one_out_influence(x, y):
    """Drop each observation, recompute Spearman rho. Return array of rhos."""
    n = len(x)
    rhos = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rhos[i] = stats.spearmanr(x[mask], y[mask]).statistic
    return rhos


def compute_correlation_robustness(all_records):
    """For key correlations: Spearman CI, Kendall tau, LOO influence."""
    results = {}

    # Key correlations to test
    tests = [
        ("H0", "ewc_benefit_aurc", "H0 vs EWC benefit (AURC)"),
        ("H0", "ret_10", "H0 vs ret@10"),
        ("num_params", "ret_10", "params vs ret@10"),
        ("num_params", "retention_100", "params vs ret@100"),
    ]

    for ds in DATASETS:
        records = all_records[ds]
        ds_results = {}

        for x_key, y_key, label in tests:
            x = np.array([r[x_key] for r in records if r[x_key] is not None and r[y_key] is not None])
            y = np.array([r[y_key] for r in records if r[x_key] is not None and r[y_key] is not None])
            arch_names = [r["arch_name"] for r in records if r[x_key] is not None and r[y_key] is not None]

            if len(x) < 5:
                continue

            # Spearman
            sp_rho, sp_p = stats.spearmanr(x, y)
            sp_ci = bootstrap_spearman_ci(x, y)

            # Kendall tau
            kt_tau, kt_p = stats.kendalltau(x, y)

            # LOO influence
            loo_rhos = leave_one_out_influence(x, y)
            loo_range = float(loo_rhos.max() - loo_rhos.min())
            most_influential_idx = int(np.argmax(np.abs(loo_rhos - sp_rho)))
            most_influential_arch = arch_names[most_influential_idx]
            loo_without = float(loo_rhos[most_influential_idx])

            ds_results[f"{x_key}_vs_{y_key}"] = {
                "label": label,
                "n": len(x),
                "spearman_rho": round(sp_rho, 4),
                "spearman_p": round(sp_p, 6),
                "spearman_ci_95": [round(sp_ci[0], 4), round(sp_ci[1], 4)],
                "kendall_tau": round(kt_tau, 4),
                "kendall_p": round(kt_p, 6),
                "loo_rho_range": round(loo_range, 4),
                "loo_rho_min": round(float(loo_rhos.min()), 4),
                "loo_rho_max": round(float(loo_rhos.max()), 4),
                "loo_most_influential_arch": most_influential_arch,
                "loo_rho_without_most_influential": round(loo_without, 4),
                "loo_all_rhos": {arch_names[i]: round(float(loo_rhos[i]), 4) for i in range(len(arch_names))},
            }

        results[DS_LABELS[ds]] = ds_results

    return results


def compute_pooled_ols_diagnostics(all_records):
    """Cook's distance, leverage, and influence for pooled OLS interaction model."""
    # Build pooled dataset (57 observations)
    records = []
    for ds in DATASETS:
        for r in all_records[ds]:
            if r["ewc_benefit_aurc"] is not None:
                records.append({**r, "dataset": ds})

    n = len(records)
    y = np.array([r["ewc_benefit_aurc"] for r in records])
    log_params = np.array([np.log(r["num_params"]) for r in records])
    h0 = np.array([r["H0"] for r in records])

    # Standardize log_params globally
    lp_mean, lp_std = log_params.mean(), log_params.std()
    lp_z = (log_params - lp_mean) / lp_std

    # Z-score H0 within dataset
    h0z = np.zeros_like(h0)
    for ds in DATASETS:
        mask = np.array([r["dataset"] == ds for r in records])
        h0z[mask] = (h0[mask] - h0[mask].mean()) / h0[mask].std()

    # Dataset dummies
    d_cub = np.array([1.0 if r["dataset"] == "cub200" else 0.0 for r in records])
    d_resisc = np.array([1.0 if r["dataset"] == "resisc45" else 0.0 for r in records])

    # M1 design matrix with intercept
    X = np.column_stack([
        np.ones(n),
        lp_z, d_cub, d_resisc,
        lp_z * d_cub, lp_z * d_resisc,
        h0z, h0z * d_cub, h0z * d_resisc,
    ])

    # OLS fit
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    residuals = y - y_hat
    k = X.shape[1]
    mse = np.sum(residuals ** 2) / (n - k)

    # Hat matrix and leverage
    XtX_inv = np.linalg.inv(X.T @ X)
    H = X @ XtX_inv @ X.T
    leverage = np.diag(H)

    # Cook's distance
    cooks_d = (residuals ** 2 / (k * mse)) * (leverage / (1 - leverage) ** 2)

    # Studentized residuals
    s_i = np.sqrt(mse * (1 - leverage))
    studentized = residuals / s_i

    # DFBETAS for H0z coefficient (index 6)
    dfbetas_h0z = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta_minus_i, _, _, _ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        dfbetas_h0z[i] = (beta[6] - beta_minus_i[6])

    # Flag high-influence points
    cooks_threshold = 4.0 / n
    high_cooks = [(records[i]["arch_name"], records[i]["dataset"],
                    round(float(cooks_d[i]), 4), round(float(leverage[i]), 4))
                   for i in range(n) if cooks_d[i] > cooks_threshold]

    return {
        "n_observations": n,
        "n_predictors": k,
        "r_squared": round(float(1 - np.sum(residuals ** 2) / np.sum((y - y.mean()) ** 2)), 4),
        "cooks_d_threshold": round(cooks_threshold, 4),
        "cooks_d_max": round(float(cooks_d.max()), 4),
        "cooks_d_mean": round(float(cooks_d.mean()), 4),
        "n_high_influence": len(high_cooks),
        "high_influence_points": high_cooks,
        "leverage_max": round(float(leverage.max()), 4),
        "leverage_mean": round(float(leverage.mean()), 4),
        "studentized_residuals_max_abs": round(float(np.abs(studentized).max()), 4),
        "dfbetas_h0z_max_abs": round(float(np.abs(dfbetas_h0z).max()), 6),
        "dfbetas_h0z_by_obs": [
            {"arch": records[i]["arch_name"], "dataset": records[i]["dataset"],
             "dfbeta": round(float(dfbetas_h0z[i]), 6),
             "cooks_d": round(float(cooks_d[i]), 4),
             "leverage": round(float(leverage[i]), 4)}
            for i in np.argsort(-np.abs(dfbetas_h0z))[:10]  # top 10
        ],
    }


def compute_loao_fold_variance(all_records):
    """Compute per-fold prediction error variance from LOAO Ridge."""
    from sklearn.linear_model import RidgeCV

    results = {}
    for ds in DATASETS:
        records = all_records[ds]
        # Use params + topology as features
        X_params = np.array([r["num_params"] for r in records]).reshape(-1, 1)
        X_topo = np.column_stack([
            [r["H0"] for r in records],
            [r["H1"] for r in records],
        ])
        X_combined = np.column_stack([X_params, X_topo])

        for outcome_key, outcome_label in [("ret_10", "ret@10"), ("retention_100", "ret@100")]:
            y = np.array([r[outcome_key] for r in records])
            n = len(y)

            # LOAO per fold errors
            fold_errors_params = []
            fold_errors_topo = []
            fold_errors_combined = []
            alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False

                # Params only
                ridge_p = RidgeCV(alphas=alphas, cv=min(n - 1, 5))
                ridge_p.fit(X_params[mask], y[mask])
                pred_p = ridge_p.predict(X_params[~mask])[0]
                fold_errors_params.append(float(pred_p - y[i]))

                # Topology only
                ridge_t = RidgeCV(alphas=alphas, cv=min(n - 1, 5))
                ridge_t.fit(X_topo[mask], y[mask])
                pred_t = ridge_t.predict(X_topo[~mask])[0]
                fold_errors_topo.append(float(pred_t - y[i]))

                # Combined
                ridge_c = RidgeCV(alphas=alphas, cv=min(n - 1, 5))
                ridge_c.fit(X_combined[mask], y[mask])
                pred_c = ridge_c.predict(X_combined[~mask])[0]
                fold_errors_combined.append(float(pred_c - y[i]))

            results[f"{DS_LABELS[ds]}_{outcome_key}"] = {
                "outcome": outcome_label,
                "dataset": DS_LABELS[ds],
                "n_folds": n,
                "params_only": {
                    "mae": round(float(np.mean(np.abs(fold_errors_params))), 4),
                    "std_abs_error": round(float(np.std(np.abs(fold_errors_params))), 4),
                    "max_abs_error": round(float(np.max(np.abs(fold_errors_params))), 4),
                },
                "topo_only": {
                    "mae": round(float(np.mean(np.abs(fold_errors_topo))), 4),
                    "std_abs_error": round(float(np.std(np.abs(fold_errors_topo))), 4),
                    "max_abs_error": round(float(np.max(np.abs(fold_errors_topo))), 4),
                },
                "combined": {
                    "mae": round(float(np.mean(np.abs(fold_errors_combined))), 4),
                    "std_abs_error": round(float(np.std(np.abs(fold_errors_combined))), 4),
                    "max_abs_error": round(float(np.max(np.abs(fold_errors_combined))), 4),
                },
            }

    return results


def main():
    print("Loading Phase 4 data...")
    all_records = load_phase4()
    for ds in DATASETS:
        print(f"  {DS_LABELS[ds]}: {len(all_records[ds])} architectures")

    print("\n1. Computing correlation robustness (Spearman CI, Kendall tau, LOO influence)...")
    corr_results = compute_correlation_robustness(all_records)

    print("\n2. Computing pooled OLS diagnostics (Cook's distance, leverage, DFBETAS)...")
    ols_results = compute_pooled_ols_diagnostics(all_records)

    print("\n3. Computing LOAO fold variance...")
    loao_results = compute_loao_fold_variance(all_records)

    # Combine and save
    output = {
        "correlation_robustness": corr_results,
        "pooled_ols_diagnostics": ols_results,
        "loao_fold_variance": loao_results,
    }

    out_path = RESULTS_DIR / "robustness_analyses.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n--- Correlation Robustness ---")
    for ds_label, ds_results in corr_results.items():
        print(f"\n  {ds_label}:")
        for key, val in ds_results.items():
            print(f"    {val['label']}:")
            print(f"      Spearman: rho={val['spearman_rho']}, p={val['spearman_p']}, "
                  f"95% CI=[{val['spearman_ci_95'][0]}, {val['spearman_ci_95'][1]}]")
            print(f"      Kendall:  tau={val['kendall_tau']}, p={val['kendall_p']}")
            print(f"      LOO: range={val['loo_rho_range']}, most influential={val['loo_most_influential_arch']} "
                  f"(rho without={val['loo_rho_without_most_influential']})")

    print("\n--- Pooled OLS Diagnostics ---")
    print(f"  R^2: {ols_results['r_squared']}")
    print(f"  Cook's D: max={ols_results['cooks_d_max']}, threshold={ols_results['cooks_d_threshold']}")
    print(f"  High influence points: {ols_results['n_high_influence']}")
    for pt in ols_results["high_influence_points"]:
        print(f"    {pt[0]} ({pt[1]}): Cook's D={pt[2]}, leverage={pt[3]}")
    print(f"  Max |studentized residual|: {ols_results['studentized_residuals_max_abs']}")
    print(f"  Max |DFBETAS(H0z)|: {ols_results['dfbetas_h0z_max_abs']}")
    print("\n  Top 5 influential on H0z coefficient:")
    for pt in ols_results["dfbetas_h0z_by_obs"][:5]:
        print(f"    {pt['arch']} ({pt['dataset']}): DFBETA={pt['dfbeta']:.6f}, Cook's D={pt['cooks_d']}")

    print("\n--- LOAO Fold Variance ---")
    for key, val in loao_results.items():
        print(f"  {val['dataset']} {val['outcome']}:")
        print(f"    Params: MAE={val['params_only']['mae']} +/- {val['params_only']['std_abs_error']}")
        print(f"    Topo:   MAE={val['topo_only']['mae']} +/- {val['topo_only']['std_abs_error']}")
        print(f"    Combo:  MAE={val['combined']['mae']} +/- {val['combined']['std_abs_error']}")


if __name__ == "__main__":
    main()
