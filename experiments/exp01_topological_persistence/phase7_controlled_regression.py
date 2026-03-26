"""
Phase 7: Controlled Multiple Regression — Does H0 predict EWC benefit
after controlling for parameter count and depth?

Per-dataset and pooled regressions:
  Model 1: EWC_benefit ~ log_params                    (baseline)
  Model 2: EWC_benefit ~ H0                            (topology alone)
  Model 3: EWC_benefit ~ log_params + H0               (key test)
  Model 4: EWC_benefit ~ log_params + H0 + depth       (full control)

If H0 remains significant in Models 3-4, it is not just a proxy for
model size. This is the single table that answers the reviewer question.
"""

import json
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.shared.models import get_model

DATASETS = ["cifar100", "cub200", "resisc45"]
DATASET_CLASSES = {"cifar100": 50, "cub200": 100, "resisc45": 22}


def count_weight_layers(model):
    """Count Conv2d + Linear layers (weight-bearing depth)."""
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            count += 1
    return count


def get_arch_depth(arch_key, num_classes=50):
    """Instantiate model and count weight-bearing layers."""
    try:
        model = get_model(arch_key, num_classes=num_classes)
        return count_weight_layers(model)
    except Exception as e:
        print(f"  Warning: could not compute depth for {arch_key}: {e}")
        return None


# Map from arch_name (display name in correlation JSON) to model key
ARCH_NAME_TO_KEY = {
    "ResNet-18": "resnet18",
    "ResNet-50": "resnet50",
    "VGG-16-BN": "vgg16_bn",
    "WRN-28-10": "wrn2810",
    "WRN-28-1": "wrn281",
    "WRN-28-2": "wrn282",
    "WRN-28-4": "wrn284",
    "WRN-28-6": "wrn286",
    "WRN-28-8": "wrn288",
    "DenseNet-121": "densenet121",
    "EfficientNet-B0": "efficientnet_b0",
    "ConvNeXt-Tiny": "convnext_tiny",
    "MobileNet-V3-S": "mobilenet_v3_small",
    "MobileNet-V3-Small": "mobilenet_v3_small",
    "ViT-Small": "vit_small",
    "ViT-Tiny": "vit_tiny",
    "MLP-Mixer": "mlp_mixer",
    "ShuffleNet-V2": "shufflenet_v2",
    "RegNet-Y-400MF": "regnet_y400mf",
    "ResNet-18-Wide": "resnet18_wide",
}


# ── OLS via numpy + scipy (no statsmodels dependency) ──────────────

def ols_fit(y, X):
    """
    OLS regression via normal equations.
    Returns: betas, se, t_vals, p_vals, R2, R2_adj, residuals
    """
    n, k = X.shape
    # beta = (X'X)^-1 X'y
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        betas = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        betas = np.linalg.lstsq(X, y, rcond=None)[0]

    y_hat = X @ betas
    residuals = y - y_hat

    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    R2_adj = 1.0 - (1.0 - R2) * (n - 1) / (n - k) if n > k else 0.0

    # Standard errors
    dof = n - k
    mse = ss_res / dof if dof > 0 else 0.0
    try:
        cov = mse * np.linalg.inv(XtX)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)

    # t-values and p-values (two-tailed)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_vals = np.where(se > 0, betas / se, 0.0)
    p_vals = np.array([2.0 * stats.t.sf(abs(t), dof) for t in t_vals]) if dof > 0 else np.ones(k)

    # F-statistic (all predictors excluding intercept)
    # Compare to intercept-only model
    ss_tot_mean = ss_tot  # already centered
    ss_reg = ss_tot_mean - ss_res
    df_reg = k - 1  # exclude intercept
    if df_reg > 0 and dof > 0 and ss_res > 0:
        F_stat = (ss_reg / df_reg) / (ss_res / dof)
        F_p = stats.f.sf(F_stat, df_reg, dof)
    else:
        F_stat = None
        F_p = None

    return betas, se, t_vals, p_vals, R2, R2_adj, residuals, F_stat, F_p


def run_regression(y, X_dict, model_name):
    """Run OLS regression and return results dict."""
    X_raw = np.column_stack(list(X_dict.values()))
    # Add intercept
    n = len(y)
    X = np.column_stack([np.ones(n), X_raw])
    predictor_names = ["const"] + list(X_dict.keys())

    k = X.shape[1]

    if n <= k:
        return {"model": model_name, "error": f"n={n} <= k={k}, cannot fit"}

    try:
        betas, se, t_vals, p_vals, R2, R2_adj, residuals, F_stat, F_p = ols_fit(y, X)
    except Exception as e:
        return {"model": model_name, "error": str(e)}

    coefficients = {}
    for i, name in enumerate(predictor_names):
        coefficients[name] = {
            "beta": float(betas[i]),
            "se": float(se[i]),
            "t": float(t_vals[i]),
            "p": float(p_vals[i]),
        }

    # AIC / BIC
    ss_res = np.sum(residuals ** 2)
    ll = -n / 2.0 * (np.log(2 * np.pi * ss_res / n) + 1)
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    return {
        "model": model_name,
        "n": n,
        "k": k,
        "R2": float(R2),
        "R2_adj": float(R2_adj),
        "F": float(F_stat) if F_stat is not None else None,
        "F_p": float(F_p) if F_p is not None else None,
        "AIC": float(aic),
        "BIC": float(bic),
        "coefficients": coefficients,
    }


def load_dataset(dataset_name):
    """Load per-architecture data from Phase 4 correlation results."""
    fpath = RESULTS_DIR / f"correlation_results_{dataset_name}.json"
    if not fpath.exists():
        print(f"  Missing: {fpath}")
        return None

    with open(fpath) as f:
        data = json.load(f)

    per_arch = data["per_architecture"]

    # EWC benefits
    ewc_section = data.get("correlations", {}).get("ewc_benefit", {})
    ewc_benefits = ewc_section.get("benefits", [])
    ewc_archs = ewc_section.get("architectures", [])

    # Build benefit lookup by arch name
    benefit_lookup = {}
    if ewc_benefits and ewc_archs:
        for name, benefit in zip(ewc_archs, ewc_benefits):
            benefit_lookup[name] = benefit

    # SI benefits (if available)
    si_section = data.get("correlations", {}).get("si_benefit", {})
    si_benefits = si_section.get("benefits", [])
    si_archs = si_section.get("architectures", [])

    si_benefit_lookup = {}
    if si_benefits and si_archs:
        for name, benefit in zip(si_archs, si_benefits):
            si_benefit_lookup[name] = benefit

    records = []
    for arch in per_arch:
        name = arch["arch_name"]
        benefit = benefit_lookup.get(name)
        if benefit is None:
            continue

        record = {
            "arch_name": name,
            "dataset": dataset_name,
            "num_params": arch["num_params"],
            "log_params": np.log(arch["num_params"]),
            "H0": arch["H0"],
            "ewc_benefit": benefit,
        }

        si_ben = si_benefit_lookup.get(name)
        if si_ben is not None:
            record["si_benefit"] = si_ben

        records.append(record)

    return records


def run_all_models(records, label):
    """Run the 4-model regression battery on a set of records."""
    if not records or len(records) < 5:
        return {"label": label, "error": f"Too few records: {len(records) if records else 0}"}

    y = np.array([r["ewc_benefit"] for r in records])
    log_params = np.array([r["log_params"] for r in records])
    H0 = np.array([r["H0"] for r in records])
    depth = np.array([r.get("depth") for r in records], dtype=float)

    has_depth = not np.any(np.isnan(depth))

    results = {"label": label, "n": len(records)}

    # Model 1: baseline (params only)
    results["model1_params_only"] = run_regression(
        y, {"log_params": log_params}, "EWC_benefit ~ log_params"
    )

    # Model 2: topology alone
    results["model2_H0_only"] = run_regression(
        y, {"H0": H0}, "EWC_benefit ~ H0"
    )

    # Model 3: key test
    results["model3_params_H0"] = run_regression(
        y, {"log_params": log_params, "H0": H0}, "EWC_benefit ~ log_params + H0"
    )

    # Model 4: full control (if depth available)
    if has_depth:
        results["model4_full"] = run_regression(
            y, {"log_params": log_params, "H0": H0, "depth": depth},
            "EWC_benefit ~ log_params + H0 + depth"
        )
    else:
        results["model4_full"] = {"model": "EWC_benefit ~ log_params + H0 + depth",
                                   "error": "depth not available for all architectures"}

    # Spearman correlations for reference
    rho_H0, p_H0 = stats.spearmanr(H0, y)
    rho_params, p_params = stats.spearmanr(log_params, y)
    results["spearman"] = {
        "H0_vs_benefit": {"rho": float(rho_H0), "p": float(p_H0)},
        "log_params_vs_benefit": {"rho": float(rho_params), "p": float(p_params)},
    }

    # Partial correlation: H0 vs benefit, controlling for log_params
    # Residualize both H0 and benefit on log_params, then correlate residuals
    ones = np.ones(len(log_params))
    X_params = np.column_stack([ones, log_params])
    _, _, _, _, _, _, resid_H0, _, _ = ols_fit(H0, X_params)
    _, _, _, _, _, _, resid_y, _, _ = ols_fit(y, X_params)
    rho_partial, p_partial = stats.spearmanr(resid_H0, resid_y)
    results["partial_correlation_H0_controlling_params"] = {
        "rho": float(rho_partial), "p": float(p_partial)
    }

    return results


def run_all_models_target(records, label, target_key):
    """Run the 4-model regression battery on a set of records with arbitrary target."""
    if not records or len(records) < 5:
        return {"label": label, "error": f"Too few records: {len(records) if records else 0}"}

    y = np.array([r[target_key] for r in records])
    log_params = np.array([r["log_params"] for r in records])
    H0 = np.array([r["H0"] for r in records])
    depth = np.array([r.get("depth") for r in records], dtype=float)

    has_depth = not np.any(np.isnan(depth))
    target_label = target_key.replace("_", " ").title()

    results = {"label": label, "n": len(records)}

    results["model1_params_only"] = run_regression(
        y, {"log_params": log_params}, f"{target_label} ~ log_params"
    )

    results["model2_H0_only"] = run_regression(
        y, {"H0": H0}, f"{target_label} ~ H0"
    )

    results["model3_params_H0"] = run_regression(
        y, {"log_params": log_params, "H0": H0}, f"{target_label} ~ log_params + H0"
    )

    if has_depth:
        results["model4_full"] = run_regression(
            y, {"log_params": log_params, "H0": H0, "depth": depth},
            f"{target_label} ~ log_params + H0 + depth"
        )
    else:
        results["model4_full"] = {"model": f"{target_label} ~ log_params + H0 + depth",
                                   "error": "depth not available for all architectures"}

    rho_H0, p_H0 = stats.spearmanr(H0, y)
    rho_params, p_params = stats.spearmanr(log_params, y)
    results["spearman"] = {
        "H0_vs_benefit": {"rho": float(rho_H0), "p": float(p_H0)},
        "log_params_vs_benefit": {"rho": float(rho_params), "p": float(p_params)},
    }

    ones = np.ones(len(log_params))
    X_params = np.column_stack([ones, log_params])
    _, _, _, _, _, _, resid_H0, _, _ = ols_fit(H0, X_params)
    _, _, _, _, _, _, resid_y, _, _ = ols_fit(y, X_params)
    rho_partial, p_partial = stats.spearmanr(resid_H0, resid_y)
    results["partial_correlation_H0_controlling_params"] = {
        "rho": float(rho_partial), "p": float(p_partial)
    }

    return results


def print_table(all_results, target_name="EWC benefit"):
    """Print a clean summary table."""
    print("\n" + "=" * 90)
    print(f"CONTROLLED REGRESSION: Does H0 predict {target_name} after controlling for capacity?")
    print("=" * 90)

    for res in all_results:
        label = res["label"]
        n = res.get("n", "?")
        print(f"\n{'-' * 90}")
        print(f"  {label} (n={n})")
        print(f"{'-' * 90}")

        if "error" in res:
            print(f"  ERROR: {res['error']}")
            continue

        # Spearman correlations
        sp = res.get("spearman", {})
        h0_sp = sp.get("H0_vs_benefit", {})
        par_sp = sp.get("log_params_vs_benefit", {})
        partial = res.get("partial_correlation_H0_controlling_params", {})
        print(f"  Spearman H0 vs benefit:         rho={h0_sp.get('rho', '?'):+.3f}  p={h0_sp.get('p', '?'):.4f}")
        print(f"  Spearman params vs benefit:      rho={par_sp.get('rho', '?'):+.3f}  p={par_sp.get('p', '?'):.4f}")
        print(f"  Partial corr (H0 | params):      rho={partial.get('rho', '?'):+.3f}  p={partial.get('p', '?'):.4f}")

        # Regression table
        for model_key in ["model1_params_only", "model2_H0_only", "model3_params_H0", "model4_full"]:
            m = res.get(model_key, {})
            if "error" in m:
                print(f"\n  {m.get('model', model_key)}: {m['error']}")
                continue

            model_name = m.get("model", model_key)
            r2 = m.get("R2", 0)
            r2_adj = m.get("R2_adj", 0)
            print(f"\n  {model_name}")
            print(f"  R2={r2:.3f}  R2_adj={r2_adj:.3f}  AIC={m.get('AIC', 0):.1f}  BIC={m.get('BIC', 0):.1f}")
            print(f"  {'Predictor':<15} {'beta':>10} {'SE':>10} {'t':>8} {'p':>10}  {'sig':>5}")
            print(f"  {'-' * 65}")
            for name, coef in m.get("coefficients", {}).items():
                if name == "const":
                    continue
                sig = ""
                p = coef["p"]
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                elif p < 0.10:
                    sig = "."
                beta_str = f"{coef['beta']:>10.4f}" if abs(coef['beta']) >= 0.00005 else f"{coef['beta']:>10.2e}"
                se_str = f"{coef['se']:>10.4f}" if abs(coef['se']) >= 0.00005 else f"{coef['se']:>10.2e}"
                print(f"  {name:<15} {beta_str} {se_str} {coef['t']:>8.3f} {p:>10.4f}  {sig:>5}")

    print(f"\n{'=' * 90}")
    print(f"Key question: Is H0 significant (p < 0.05) in Model 3 or 4 for {target_name}?")
    print("If yes: H0 is not a proxy for model size.")
    print("If no: H0 and param count are confounded at this sample size.")
    print("=" * 90)


def main():
    print("Phase 7: Controlled Multiple Regression")
    print("Loading data...\n")

    # Compute depth for each architecture
    print("Computing architecture depths...")
    depth_map = {}
    for display_name, model_key in ARCH_NAME_TO_KEY.items():
        d = get_arch_depth(model_key, num_classes=50)
        depth_map[display_name] = d
        if d is not None:
            print(f"  {display_name}: {d} weight layers")

    # Load per-dataset
    all_records = {}
    pooled = []
    for ds in DATASETS:
        print(f"\nLoading {ds}...")
        records = load_dataset(ds)
        if records is None:
            continue

        # Attach depth
        for r in records:
            r["depth"] = depth_map.get(r["arch_name"])

        all_records[ds] = records
        pooled.extend(records)
        print(f"  Loaded {len(records)} architectures")

    # Run per-dataset EWC regressions
    all_results = []
    for ds in DATASETS:
        if ds in all_records:
            result = run_all_models(all_records[ds], ds.upper())
            all_results.append(result)

    # Run pooled EWC regression (all 57)
    if pooled:
        pooled_result = run_all_models(pooled, "POOLED (all datasets)")
        all_results.append(pooled_result)

    # Print EWC results
    print_table(all_results)

    # ─── SI Benefit Regressions ───
    # Check if any records have si_benefit
    si_records = {ds: [r for r in recs if r.get("si_benefit") is not None]
                  for ds, recs in all_records.items()}
    si_pooled = [r for r in pooled if r.get("si_benefit") is not None]
    has_si = any(len(recs) >= 5 for recs in si_records.values()) or len(si_pooled) >= 5

    si_results = []
    if has_si:
        print("\n\n" + "#" * 90)
        print("  SI BENEFIT REGRESSIONS")
        print("#" * 90)

        for ds in DATASETS:
            recs = si_records.get(ds, [])
            if len(recs) >= 5:
                result = run_all_models_target(recs, f"{ds.upper()} [SI]", "si_benefit")
                si_results.append(result)

        if len(si_pooled) >= 5:
            pooled_si_result = run_all_models_target(si_pooled, "POOLED [SI]", "si_benefit")
            si_results.append(pooled_si_result)

        if si_results:
            print_table(si_results, target_name="SI benefit")
            all_results.extend(si_results)

    # Save JSON
    output_path = RESULTS_DIR / "controlled_regression.json"

    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=make_serializable)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
