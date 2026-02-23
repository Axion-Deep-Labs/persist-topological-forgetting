"""
EXP-01 Phase 5: Architecture-grouped predictive model with permutation testing.

Tests whether topology provides incremental predictive value beyond parameter
count for predicting forgetting resistance. Uses leave-one-architecture-out
(LOAO) cross-validation with strict fold isolation.

Models:
  A:  retention ~ params                     (baseline)
  A2: retention ~ params + z1 + z2           (matched-dimensionality null)
  B:  retention ~ params + H0_rip + H1_rip   (Ripser topology)
  C:  retention ~ params + H0_cub + H1_cub   (cubical topology)
  D:  retention ~ H0_rip + H1_rip            (topology alone)

All preprocessing (standardization, alpha selection) occurs inside each
training fold to prevent leakage. Permutation test shuffles topology features
across architectures to test incremental value.

Usage:
    python -m experiments.exp01_topological_persistence.phase5_predictive_model \
        --results-dirs results/exp01 results/exp01_vit ...
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# ─── Data Loading ───

def load_architecture_data(result_dir):
    """Load topology (aggregated across slices) and forgetting for one architecture.

    Returns dict with: params, H0_rip, H1_rip, H0_cub, H1_cub, early_aurc,
    ret_10, ret_100, arch_name, or None if data missing.
    """
    topo_dir = os.path.join(result_dir, "topology")
    forget_dir = os.path.join(result_dir, "forgetting")

    # Load Ripser topology (aggregated mean across slices)
    ripser_data = _load_aggregated_topology(topo_dir, "topology_summary")
    if ripser_data is None:
        return None

    # Load cubical topology (aggregated mean across slices)
    cubical_data = _load_aggregated_topology(topo_dir, "cubical_summary")

    # Load forgetting curve
    forget_path = os.path.join(forget_dir, "forgetting_curve.json")
    if not os.path.exists(forget_path):
        return None
    with open(forget_path) as f:
        forget = json.load(f)

    initial_acc = forget["initial_task_a_acc"]
    if initial_acc == 0:
        return None

    # Compute forgetting metrics
    curve = forget["curve"]
    ret_10 = _retention_at_step(curve, 10, initial_acc)
    ret_100 = _retention_at_step(curve, 100, initial_acc)
    early_aurc = _compute_early_aurc(curve, initial_acc, max_step=500)

    # Count params from checkpoint
    num_params = _count_params(result_dir)

    # Load EWC forgetting if available
    ewc_path = os.path.join(result_dir, "forgetting_ewc", "forgetting_curve.json")
    ewc_early_aurc = None
    if os.path.exists(ewc_path):
        with open(ewc_path) as f:
            ewc_forget = json.load(f)
        ewc_initial = ewc_forget["initial_task_a_acc"]
        if ewc_initial > 0:
            ewc_early_aurc = _compute_early_aurc(ewc_forget["curve"], ewc_initial, max_step=500)

    return {
        "label": os.path.basename(result_dir),
        "num_params": num_params,
        "H0_rip": ripser_data.get("H0"),
        "H1_rip": ripser_data.get("H1"),
        "H0_cub": cubical_data.get("H0") if cubical_data else None,
        "H1_cub": cubical_data.get("H1") if cubical_data else None,
        "early_aurc": early_aurc,
        "ret_10": ret_10,
        "ret_100": ret_100,
        "ewc_early_aurc": ewc_early_aurc,
    }


def _load_aggregated_topology(topo_dir, prefix):
    """Load and aggregate topology summaries (mean across slices)."""
    run_files = sorted(glob.glob(os.path.join(topo_dir, f"{prefix}_run*.json")))
    default_path = os.path.join(topo_dir, f"{prefix}.json")

    all_runs = []
    if os.path.exists(default_path):
        with open(default_path) as f:
            all_runs.append(json.load(f))
    for rf in run_files:
        with open(rf) as f:
            all_runs.append(json.load(f))

    if not all_runs:
        return None

    result = {}
    for key in ["H0", "H1", "H0_count", "H1_count"]:
        vals = [r.get(key) for r in all_runs if r.get(key) is not None]
        result[key] = float(np.mean(vals)) if vals else None

    return result


def _retention_at_step(curve, step, initial_acc):
    """Get retention ratio at a specific step."""
    for point in curve:
        if point["step"] == step:
            return point["task_a_acc"] / initial_acc
    return None


def _compute_early_aurc(curve, initial_acc, max_step=500):
    """Area under retention curve from step 0 to max_step, normalized."""
    points = [(p["step"], p["task_a_acc"] / initial_acc) for p in curve
              if p["step"] <= max_step]
    if len(points) < 2:
        return None
    points.sort()
    auc = 0.0
    for i in range(1, len(points)):
        dt = points[i][0] - points[i-1][0]
        avg_ret = (points[i][1] + points[i-1][1]) / 2
        auc += avg_ret * dt
    return auc / max_step if max_step > 0 else 0.0


def _count_params(result_dir):
    """Count model parameters from checkpoint."""
    import torch
    ckpt_path = os.path.join(result_dir, "checkpoints", "task_a_best.pt")
    if not os.path.exists(ckpt_path):
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        return sum(v.numel() for v in state_dict.values())
    except Exception:
        return None


# ─── Model Evaluation ───

def loao_evaluate(X, y, alpha_candidates=None, fixed_alphas=None):
    """Leave-one-architecture-out evaluation with nested alpha selection.

    All preprocessing happens inside each fold.

    Args:
        X: feature matrix (n_archs, n_features)
        y: outcome vector (n_archs,)
        alpha_candidates: ridge alpha values to try
        fixed_alphas: if provided, skip nested CV and use these per-fold alphas

    Returns:
        predictions: array of LOAO predictions
        mae: mean absolute error
        rho: Spearman correlation of predictions vs actual
        fold_alphas: array of selected alphas per fold
    """
    if alpha_candidates is None:
        alpha_candidates = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    n = len(y)
    predictions = np.zeros(n)
    fold_alphas = np.zeros(n)

    for i in range(n):
        # Split: train on all except i
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]

        # Standardize inside fold (training stats only)
        scaler_X = StandardScaler().fit(X_train)
        scaler_y_mean = y_train.mean()
        scaler_y_std = y_train.std()
        if scaler_y_std < 1e-10:
            scaler_y_std = 1.0

        X_train_s = scaler_X.transform(X_train)
        y_train_s = (y_train - scaler_y_mean) / scaler_y_std
        X_test_s = scaler_X.transform(X_test)

        if fixed_alphas is not None:
            best_alpha = fixed_alphas[i]
        else:
            # Nested LOO for alpha selection
            best_alpha = alpha_candidates[0]
            best_nested_mae = float("inf")

            for alpha in alpha_candidates:
                nested_preds = np.zeros(len(y_train))
                for j in range(len(y_train)):
                    X_nest_train = np.delete(X_train_s, j, axis=0)
                    y_nest_train = np.delete(y_train_s, j)
                    X_nest_test = X_train_s[j:j+1]

                    model = Ridge(alpha=alpha, fit_intercept=True)
                    model.fit(X_nest_train, y_nest_train)
                    nested_preds[j] = model.predict(X_nest_test)[0]

                nested_mae = np.mean(np.abs(nested_preds - y_train_s))
                if nested_mae < best_nested_mae:
                    best_nested_mae = nested_mae
                    best_alpha = alpha

        fold_alphas[i] = best_alpha

        # Fit final model with best alpha
        model = Ridge(alpha=best_alpha, fit_intercept=True)
        model.fit(X_train_s, y_train_s)

        # Predict held-out architecture
        pred_s = model.predict(X_test_s)[0]
        predictions[i] = pred_s * scaler_y_std + scaler_y_mean

    mae = np.mean(np.abs(predictions - y))
    rho, p = stats.spearmanr(predictions, y)
    return predictions, mae, rho, fold_alphas


def permutation_test_incremental(X_base, X_topo, y, n_perms=1000, rng_seed=42):
    """Permutation test for incremental predictive value of topology features.

    Shuffles topology columns across architectures (keeping base features and
    outcome paired) and measures how often random topology achieves the same
    error reduction. Alpha is selected once on real data via nested CV, then
    reused for all permutations (testing topology signal, not alpha selection).

    Args:
        X_base: baseline features (n, n_base) e.g. [params]
        X_topo: topology features (n, n_topo) e.g. [H0, H1]
        y: outcome (n,)
        n_perms: number of permutations
        rng_seed: random seed for reproducibility

    Returns:
        delta_observed: MAE_base - MAE_combined (positive = topology helps)
        perm_deltas: array of permutation deltas
        p_value: fraction of perm deltas >= observed
    """
    rng = np.random.RandomState(rng_seed)

    # Observed: baseline MAE (full nested CV)
    _, mae_base, _, _ = loao_evaluate(X_base, y)

    # Observed: combined MAE (full nested CV, capture alphas)
    X_combined = np.hstack([X_base, X_topo])
    _, mae_combined, _, combined_alphas = loao_evaluate(X_combined, y)
    delta_observed = mae_base - mae_combined

    # Permutation null (reuse alphas from real data)
    perm_deltas = np.zeros(n_perms)
    for p in range(n_perms):
        # Shuffle topology rows independently of base features
        perm_idx = rng.permutation(len(y))
        X_topo_shuffled = X_topo[perm_idx]
        X_perm = np.hstack([X_base, X_topo_shuffled])
        _, mae_perm, _, _ = loao_evaluate(X_perm, y, fixed_alphas=combined_alphas)
        perm_deltas[p] = mae_base - mae_perm

    p_value = np.mean(perm_deltas >= delta_observed)
    return delta_observed, perm_deltas, p_value


def matched_dimensionality_null(X_base, n_extra, y, n_draws=1000, rng_seed=42):
    """Null distribution of error reduction from adding n_extra random features.

    Tests whether improvement from adding topology features is simply due to
    having more predictors (overfitting with extra degrees of freedom). Alpha
    is selected once on the first random draw via nested CV, then reused.

    Args:
        X_base: baseline features (n, n_base)
        n_extra: number of random features to add (match topology feature count)
        y: outcome (n,)
        n_draws: number of random draws
        rng_seed: random seed

    Returns:
        null_deltas: array of error reductions from random features
        null_95: 95th percentile of null distribution
    """
    rng = np.random.RandomState(rng_seed)
    _, mae_base, _, _ = loao_evaluate(X_base, y)

    # Select alpha on first random draw, reuse for rest
    Z_first = rng.randn(len(y), n_extra)
    X_first = np.hstack([X_base, Z_first])
    _, mae_first, _, ref_alphas = loao_evaluate(X_first, y)

    null_deltas = np.zeros(n_draws)
    null_deltas[0] = mae_base - mae_first

    for d in range(1, n_draws):
        Z = rng.randn(len(y), n_extra)
        X_augmented = np.hstack([X_base, Z])
        _, mae_aug, _, _ = loao_evaluate(X_augmented, y, fixed_alphas=ref_alphas)
        null_deltas[d] = mae_base - mae_aug

    null_95 = np.percentile(null_deltas, 95)
    return null_deltas, null_95


# ─── Main Analysis ───

def run_analysis(all_data, outcome_key, outcome_name, dataset_label):
    """Run full predictive analysis for one outcome metric on one dataset."""
    print(f"\n{'=' * 70}")
    print(f"PREDICTIVE MODEL — {dataset_label} (n={len(all_data)} architectures)")
    print(f"Outcome: {outcome_name}")
    print(f"{'=' * 70}")

    # Extract features and outcome
    y = np.array([d[outcome_key] for d in all_data])
    params = np.array([d["num_params"] for d in all_data]).reshape(-1, 1)
    h0_rip = np.array([d.get("H0_rip", 0) or 0 for d in all_data])
    h1_rip = np.array([d.get("H1_rip", 0) or 0 for d in all_data])
    topo_rip = np.column_stack([h0_rip, h1_rip])

    has_cubical = all(d.get("H0_cub") is not None for d in all_data)
    if has_cubical:
        h0_cub = np.array([d["H0_cub"] for d in all_data])
        h1_cub = np.array([d["H1_cub"] for d in all_data])
        topo_cub = np.column_stack([h0_cub, h1_cub])

    # ── Model A: params only ──
    _, mae_a, rho_a, _ = loao_evaluate(params, y)
    print(f"\n  Model A (params only):           MAE = {mae_a:.4f}, rho = {rho_a:.4f}")

    # ── Model B: params + Ripser topology ──
    X_b = np.hstack([params, topo_rip])
    _, mae_b, rho_b, _ = loao_evaluate(X_b, y)
    reduction_b = (mae_a - mae_b) / mae_a * 100 if mae_a > 0 else 0
    print(f"  Model B (params + ripser):       MAE = {mae_b:.4f}, rho = {rho_b:.4f}")

    # ── Model C: params + cubical topology ──
    mae_c, rho_c, reduction_c = None, None, None
    if has_cubical:
        X_c = np.hstack([params, topo_cub])
        _, mae_c, rho_c, _ = loao_evaluate(X_c, y)
        reduction_c = (mae_a - mae_c) / mae_a * 100 if mae_a > 0 else 0
        print(f"  Model C (params + cubical):      MAE = {mae_c:.4f}, rho = {rho_c:.4f}")

    # ── Model D: topology alone ──
    _, mae_d, rho_d, _ = loao_evaluate(topo_rip, y)
    print(f"  Model D (topology only):         MAE = {mae_d:.4f}, rho = {rho_d:.4f}")

    # ── Error reduction ──
    print(f"\n  Error reduction B vs A:          {reduction_b:.1f}%")
    if reduction_c is not None:
        print(f"  Error reduction C vs A:          {reduction_c:.1f}%")

    # ── Permutation test for Ripser topology ──
    print(f"\n  Permutation test (B vs A, 1000 permutations)...")
    delta_obs, perm_deltas, p_perm = permutation_test_incremental(
        params, topo_rip, y, n_perms=1000
    )
    print(f"    Observed delta:                {delta_obs:.4f}")
    print(f"    Null 95th percentile:          {np.percentile(perm_deltas, 95):.4f}")
    print(f"    Permutation p-value:           {p_perm:.4f}")

    # ── Matched-dimensionality control (Model A2) ──
    print(f"\n  Matched-dimensionality null (A2 vs A, 1000 draws)...")
    null_deltas, null_95 = matched_dimensionality_null(params, 2, y, n_draws=1000)
    exceeds_a2 = delta_obs > null_95
    print(f"    A2 null median:                {np.median(null_deltas):.4f}")
    print(f"    A2 null 95th percentile:       {null_95:.4f}")
    print(f"    Topology exceeds A2 95th:      {'YES' if exceeds_a2 else 'NO'}")
    a2_p = np.mean(null_deltas >= delta_obs)
    print(f"    A2 p-value:                    {a2_p:.4f}")

    # ── Permutation test for cubical topology ──
    perm_c_results = None
    if has_cubical:
        print(f"\n  Permutation test (C vs A, 1000 permutations)...")
        delta_c, perm_c, p_c = permutation_test_incremental(
            params, topo_cub, y, n_perms=1000, rng_seed=43
        )
        print(f"    Observed delta:                {delta_c:.4f}")
        print(f"    Permutation p-value:           {p_c:.4f}")
        perm_c_results = {"delta": float(delta_c), "p_value": float(p_c)}

    # ── Verdict ──
    print(f"\n  --- VERDICT ---")
    sig_perm = p_perm < 0.05
    sig_a2 = exceeds_a2
    if sig_perm and sig_a2:
        print(f"  Topology provides SIGNIFICANT incremental value beyond params.")
        print(f"  Gain is NOT from added dimensionality (exceeds A2 null).")
        verdict = "significant"
    elif sig_perm and not sig_a2:
        print(f"  Permutation test significant but does NOT exceed matched-dim null.")
        print(f"  Gain may be from extra degrees of freedom, not topology content.")
        verdict = "ambiguous_dimensionality"
    elif not sig_perm and sig_a2:
        print(f"  Exceeds matched-dim null but permutation test not significant.")
        print(f"  Unusual. Check for instability in LOAO with small n.")
        verdict = "ambiguous_permutation"
    else:
        print(f"  Topology does NOT improve prediction beyond params.")
        verdict = "not_significant"

    # ── Results dict ──
    results = {
        "dataset": dataset_label,
        "n_architectures": len(all_data),
        "outcome": outcome_name,
        "architectures": [d["label"] for d in all_data],
        "model_a": {"mae": float(mae_a), "rho": float(rho_a)},
        "model_b": {
            "mae": float(mae_b), "rho": float(rho_b),
            "reduction_pct": float(reduction_b),
        },
        "model_d": {"mae": float(mae_d), "rho": float(rho_d)},
        "permutation_test_b": {
            "delta_observed": float(delta_obs),
            "null_95th": float(np.percentile(perm_deltas, 95)),
            "p_value": float(p_perm),
            "n_perms": 1000,
        },
        "matched_dim_control": {
            "null_median": float(np.median(null_deltas)),
            "null_95th": float(null_95),
            "exceeds_95th": bool(exceeds_a2),
            "p_value": float(a2_p),
            "n_draws": 1000,
        },
        "verdict": verdict,
    }
    if mae_c is not None:
        results["model_c"] = {
            "mae": float(mae_c), "rho": float(rho_c),
            "reduction_pct": float(reduction_c),
        }
    if perm_c_results:
        results["permutation_test_c"] = perm_c_results

    return results


def main():
    parser = argparse.ArgumentParser(description="EXP-01 Phase 5: Predictive Model")
    parser.add_argument("--results-dirs", nargs="+", type=str, required=True,
                        help="Result directories for architectures")
    args = parser.parse_args()

    print("EXP-01 Phase 5: Architecture-Grouped Predictive Model")
    print("  Unit of analysis: architecture (slices aggregated to mean)")
    print("  CV method: leave-one-architecture-out (LOAO)")
    print("  All preprocessing inside folds (no leakage)")
    print()

    # Load data for all architectures
    all_data = []
    for rdir in args.results_dirs:
        data = load_architecture_data(rdir)
        if data is None:
            print(f"  Skipping {rdir}: incomplete data")
            continue
        if data["num_params"] is None:
            print(f"  Skipping {rdir}: no checkpoint for param count")
            continue
        all_data.append(data)

    if len(all_data) < 5:
        print(f"\nNeed >= 5 architectures, have {len(all_data)}. Exiting.")
        return

    print(f"\n  Loaded {len(all_data)} architectures")

    # Detect dataset from directory names
    first_label = all_data[0]["label"]
    if first_label.endswith("_cub200"):
        dataset_label = "CUB-200"
    elif first_label.endswith("_resisc45"):
        dataset_label = "RESISC-45"
    else:
        dataset_label = "CIFAR-100"

    # Run analysis for each outcome metric
    all_results = {}
    for outcome_key, outcome_name in [
        ("early_aurc", "Early AURC (0-500)"),
        ("ret_10", "Retention @ step 10"),
        ("ret_100", "Retention @ step 100"),
    ]:
        # Filter to architectures with this outcome
        valid = [d for d in all_data if d.get(outcome_key) is not None]
        if len(valid) < 5:
            print(f"\n  Skipping {outcome_name}: only {len(valid)} architectures have data")
            continue
        results = run_analysis(valid, outcome_key, outcome_name, dataset_label)
        all_results[outcome_key] = results

    # EWC benefit analysis (if available)
    ewc_data = [d for d in all_data
                if d.get("ewc_early_aurc") is not None and d.get("early_aurc") is not None]
    if len(ewc_data) >= 5:
        print(f"\n{'=' * 70}")
        print(f"EWC BENEFIT PREDICTION (n={len(ewc_data)})")
        print(f"{'=' * 70}")
        ewc_benefit = np.array([d["ewc_early_aurc"] - d["early_aurc"] for d in ewc_data])
        params = np.array([d["num_params"] for d in ewc_data]).reshape(-1, 1)
        h0 = np.array([d.get("H0_rip", 0) or 0 for d in ewc_data])
        h1 = np.array([d.get("H1_rip", 0) or 0 for d in ewc_data])

        # Spearman: topology vs EWC benefit
        rho_h0, p_h0 = stats.spearmanr(h0, ewc_benefit)
        rho_h1, p_h1 = stats.spearmanr(h1, ewc_benefit)
        rho_params, p_params = stats.spearmanr(params.flatten(), ewc_benefit)
        print(f"  H0 vs EWC benefit:  rho={rho_h0:.4f} (p={p_h0:.4f})")
        print(f"  H1 vs EWC benefit:  rho={rho_h1:.4f} (p={p_h1:.4f})")
        print(f"  Params vs EWC benefit: rho={rho_params:.4f} (p={p_params:.4f})")

        all_results["ewc_benefit"] = {
            "n": len(ewc_data),
            "h0_rho": float(rho_h0), "h0_p": float(p_h0),
            "h1_rho": float(rho_h1), "h1_p": float(p_h1),
            "params_rho": float(rho_params), "params_p": float(p_params),
        }

    # Save results
    out_dir = os.path.dirname(args.results_dirs[0])
    out_path = os.path.join(out_dir, f"predictive_model_{dataset_label.lower().replace('-', '')}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
