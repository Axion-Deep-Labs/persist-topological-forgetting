"""
EXP-01 Phase 4: Correlate topological features with forgetting resistance.

Compares H0 persistence against baseline metrics (Hessian trace, max eigenvalue,
Fisher trace, loss barrier) to determine whether topology captures something
that simpler geometry metrics do not.

Supports:
  - Multi-slice aggregation: when multiple topology_summary_run*.json files exist,
    computes mean ± std across slices for robust topology estimates
  - Displacement metrics: includes curvature_ratio and basin_fraction from Phase 2.5
  - Normalized barrier: cross-architecture comparable barrier height
  - Partial correlation: controls for model size (num_params) as confound
  - Architecture class analysis: within-class (CNN/Transformer/MLP) correlations
  - Baseline comparison: does num_params alone predict retention?

Usage:
    python -m experiments.exp01_topological_persistence.phase4_correlation \
        --config configs/exp01.yaml

    # Cross-architecture (after running multiple configs):
    python -m experiments.exp01_topological_persistence.phase4_correlation \
        --results-dirs results/exp01 results/exp01_vit results/exp01_resnet50 ...
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.shared.utils import load_config


# All metrics to correlate against retention
METRICS = [
    ("H0", "H0 Persistence"),
    ("H1", "H1 Persistence"),
    ("hessian_trace_mean", "Hessian Trace"),
    ("max_eigenvalue", "Max Eigenvalue (Sharpness)"),
    ("fisher_trace", "Fisher Information Trace"),
    ("max_barrier", "Loss Barrier Height"),
    ("max_barrier_normalized", "Barrier (Normalized)"),
    ("curvature_ratio", "Displacement Curvature"),
    ("basin_fraction", "Basin Fraction"),
    ("num_params", "Parameter Count"),
]

# Architecture class tags for within-class analysis
ARCH_CLASSES = {
    "exp01": ("ResNet-18", "CNN"),
    "exp01_resnet50": ("ResNet-50", "CNN"),
    "exp01_resnet18wide": ("ResNet-18-Wide", "CNN"),
    "exp01_wrn2810": ("WRN-28-10", "CNN"),
    "exp01_densenet121": ("DenseNet-121", "CNN"),
    "exp01_efficientnet": ("EfficientNet-B0", "CNN"),
    "exp01_vgg16bn": ("VGG-16-BN", "CNN"),
    "exp01_convnext": ("ConvNeXt-Tiny", "CNN"),
    "exp01_mobilenetv3": ("MobileNet-V3-S", "CNN"),
    "exp01_shufflenet": ("ShuffleNet-V2", "CNN"),
    "exp01_regnet": ("RegNet-Y-400MF", "CNN"),
    "exp01_vit": ("ViT-Small", "Transformer"),
    "exp01_vittiny": ("ViT-Tiny", "Transformer"),
    "exp01_mlpmixer": ("MLP-Mixer", "MLP"),
}


def load_topology_aggregated(result_dir):
    """Load topology data, aggregating across multiple slices if available.

    Looks for topology_summary_run*.json files first (multi-slice).
    Falls back to topology_summary.json (single slice).

    Returns dict with metric means, stds, and n_slices.
    """
    topo_dir = os.path.join(result_dir, "topology")

    # Check for multi-slice runs
    run_files = sorted(glob.glob(os.path.join(topo_dir, "topology_summary_run*.json")))

    if run_files:
        # Multi-slice: aggregate across runs
        all_runs = []
        for rf in run_files:
            with open(rf) as f:
                all_runs.append(json.load(f))

        # Compute mean/std for each numeric metric
        aggregated = {"n_slices": len(all_runs)}
        metric_keys = ["H0", "H1", "H0_count", "H1_count", "H0_max_lifetime", "H1_max_lifetime",
                       "hessian_trace_mean", "max_eigenvalue", "fisher_trace",
                       "max_barrier", "max_barrier_normalized", "loss_min", "loss_max"]

        for key in metric_keys:
            vals = [r.get(key) for r in all_runs if r.get(key) is not None]
            if vals:
                aggregated[key] = float(np.mean(vals))
                aggregated[f"{key}_std"] = float(np.std(vals))
            else:
                aggregated[key] = None

        # Copy non-numeric fields from first run
        aggregated["checkpoint_accuracy"] = all_runs[0].get("checkpoint_accuracy")
        aggregated["landscape_seeds"] = [r.get("landscape_seed") for r in all_runs]

        return aggregated

    # Single slice fallback
    topo_path = os.path.join(topo_dir, "topology_summary.json")
    if not os.path.exists(topo_path):
        return None

    with open(topo_path) as f:
        data = json.load(f)
    data["n_slices"] = 1
    return data


def load_displacement_metrics(result_dir):
    """Load displacement analysis metrics if available."""
    disp_path = os.path.join(result_dir, "displacement", "displacement_summary.json")
    if not os.path.exists(disp_path):
        return {}

    with open(disp_path) as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    return {
        "curvature_ratio": metrics.get("curvature_ratio"),
        "basin_fraction": metrics.get("basin_fraction"),
        "out_of_plane_ratio": metrics.get("out_of_plane_ratio"),
        "total_displacement": metrics.get("total_displacement"),
    }


def count_model_params(result_dir):
    """Count model parameters from checkpoint file."""
    ckpt_path = os.path.join(result_dir, "checkpoints", "task_a_best.pt")
    if not os.path.exists(ckpt_path):
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        return sum(v.numel() for v in state_dict.values())
    except Exception:
        return None


def load_experiment_data(result_dir):
    """Load topology summary and forgetting curve from a single experiment run."""
    topo = load_topology_aggregated(result_dir)
    if topo is None:
        raise FileNotFoundError(f"No topology summary in {result_dir}")

    forget_path = os.path.join(result_dir, "forgetting", "forgetting_curve.json")
    with open(forget_path) as f:
        forget = json.load(f)

    # Merge displacement metrics
    disp = load_displacement_metrics(result_dir)
    topo.update(disp)

    return topo, forget


def compute_retention_at_step(forget_data, step):
    """Get Task A accuracy at a specific training step."""
    for point in forget_data["curve"]:
        if point["step"] == step:
            return point["task_a_acc"]
    return None


def compute_forgetting_auc(forget_data):
    """Compute area under the retention curve (normalized).

    Higher AUC = more retention across the full forgetting trajectory.
    Normalized by max_step so AUC is in [0, 1] range.
    """
    initial_acc = forget_data["initial_task_a_acc"]
    if initial_acc == 0:
        return 0.0

    curve = forget_data["curve"]
    auc = 0.0
    prev_step = 0
    max_step = 0
    for point in curve:
        if point["step"] > 0:
            step_retention = point["task_a_acc"] / initial_acc
            auc += step_retention * (point["step"] - prev_step)
            prev_step = point["step"]
            max_step = point["step"]

    return auc / max_step if max_step > 0 else 0.0


def partial_correlation(x, y, z):
    """Compute partial Spearman correlation between x and y, controlling for z.

    Uses the standard formula:
    r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))
    where r values are Spearman rank correlations.
    """
    r_xy, _ = stats.spearmanr(x, y)
    r_xz, _ = stats.spearmanr(x, z)
    r_yz, _ = stats.spearmanr(y, z)

    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-10:
        return float('nan'), float('nan')

    r_partial = (r_xy - r_xz * r_yz) / denom

    # Approximate p-value using t-distribution
    n = len(x)
    df = n - 3  # degrees of freedom for partial correlation
    if df <= 0:
        return float(r_partial), float('nan')

    t_stat = r_partial * np.sqrt(df / (1 - r_partial**2 + 1e-10))
    p_value = 2 * stats.t.sf(abs(t_stat), df)

    return float(r_partial), float(p_value)


def single_run_analysis(result_dir):
    """Analyze a single experiment run."""
    topo, forget = load_experiment_data(result_dir)

    print("=" * 60)
    print("SINGLE RUN ANALYSIS")
    print("=" * 60)

    n_slices = topo.get("n_slices", 1)
    if n_slices > 1:
        print(f"\n  Aggregated across {n_slices} landscape slices")

    print(f"\nTopological Features (Task A converged model):")
    for dim in range(3):
        key = f"H{dim}"
        if key in topo and topo[key] is not None:
            std_key = f"{key}_std"
            std_str = f" ± {topo[std_key]:.4f}" if std_key in topo else ""
            print(f"  {key} total persistence:   {topo[key]:.4f}{std_str}")
            print(f"  {key} feature count:       {topo.get(f'{key}_count', 0)}")
            if f"{key}_max_lifetime" in topo:
                print(f"  {key} max lifetime:        {topo[f'{key}_max_lifetime']:.4f}")

    # Baseline metrics
    baseline_keys = ["hessian_trace_mean", "max_eigenvalue", "fisher_trace", "max_barrier", "max_barrier_normalized"]
    has_baseline = any(k in topo and topo[k] is not None for k in baseline_keys)
    if has_baseline:
        print(f"\nBaseline Geometry Metrics:")
        for key in baseline_keys:
            val = topo.get(key)
            if val is not None:
                print(f"  {key}: {val:.4f}")

    # Displacement metrics
    disp_keys = ["curvature_ratio", "basin_fraction", "out_of_plane_ratio", "total_displacement"]
    has_disp = any(k in topo and topo[k] is not None for k in disp_keys)
    if has_disp:
        print(f"\nDisplacement Metrics:")
        for key in disp_keys:
            val = topo.get(key)
            if val is not None:
                print(f"  {key}: {val:.4f}")

    print(f"\nForgetting Curve:")
    print(f"  Initial Task A accuracy: {forget['initial_task_a_acc']:.1%}")
    for point in forget["curve"]:
        if point["step"] > 0:
            retention = point["task_a_acc"] / forget["initial_task_a_acc"]
            print(f"  Step {point['step']:>6}: Task A = {point['task_a_acc']:.1%}, "
                  f"Retention = {retention:.1%}, "
                  f"Task B = {point.get('task_b_acc', 0):.1%}")

    auc = compute_forgetting_auc(forget)
    print(f"\n  Forgetting AUC (normalized): {auc:.4f}")

    ret_100 = compute_retention_at_step(forget, 100)
    if ret_100 is not None:
        retention_ratio = ret_100 / forget["initial_task_a_acc"]
        print(f"  Retention ratio at step 100: {retention_ratio:.1%}")
        print(f"  H0 total persistence: {topo.get('H0', 0):.4f}")


def cross_architecture_analysis(result_dirs):
    """Correlate ALL metrics with forgetting across architectures."""
    print("=" * 70)
    print("CROSS-ARCHITECTURE CORRELATION ANALYSIS")
    print("=" * 70)

    # Collect data from all architectures
    all_data = []
    for rdir in result_dirs:
        try:
            topo, forget = load_experiment_data(rdir)
        except FileNotFoundError as e:
            print(f"  Skipping {rdir}: {e}")
            continue

        ret_100 = compute_retention_at_step(forget, 100)
        if ret_100 is None:
            last_point = forget["curve"][-1]
            ret_100 = last_point["task_a_acc"]

        retention = ret_100 / forget["initial_task_a_acc"]
        auc = compute_forgetting_auc(forget)

        # Count params
        num_params = count_model_params(rdir)

        # Get architecture class
        label = os.path.basename(rdir)
        arch_name, arch_class = ARCH_CLASSES.get(label, (label, "Unknown"))

        entry = {
            "label": label,
            "arch_name": arch_name,
            "arch_class": arch_class,
            "retention_100": retention,
            "forgetting_auc": auc,
            "accuracy": forget["initial_task_a_acc"],
            "n_slices": topo.get("n_slices", 1),
            "num_params": num_params,
        }
        # Collect all metric values (and their stds for error bars)
        for metric_key, _ in METRICS:
            if metric_key == "num_params":
                continue  # already set
            entry[metric_key] = topo.get(metric_key, None)
            std_key = f"{metric_key}_std"
            if std_key in topo:
                entry[std_key] = topo[std_key]

        all_data.append(entry)

    if len(all_data) < 3:
        print(f"\n  Need >= 3 architectures for correlation. Have {len(all_data)}.")
        return

    # Print summary table
    print(f"\n  n = {len(all_data)} architectures")
    has_multi_slice = any(d.get("n_slices", 1) > 1 for d in all_data)
    if has_multi_slice:
        print(f"  Multi-slice aggregation active (topology values are means across slices)")

    print(f"\n{'Architecture':>20} | {'Class':>5} | {'Params':>8} | {'Acc':>5} | {'H0':>7} | {'H1':>6} | {'AUC':>5} | {'Ret@100':>7}")
    print("-" * 90)
    for d in all_data:
        def fmt(key, w=7, dec=1):
            v = d.get(key)
            if v is None:
                return f"{'N/A':>{w}}"
            return f"{v:>{w}.{dec}f}"

        params_str = f"{d['num_params']/1e6:.1f}M" if d['num_params'] else "N/A"
        print(f"{d['arch_name']:>20} | {d['arch_class']:>5} | {params_str:>8} | {d['accuracy']:4.1%} | {fmt('H0', 7)} | {fmt('H1', 6)} | {d['forgetting_auc']:5.3f} | {d['retention_100']:6.1%}")

    # ─── Standard Spearman Correlations ───
    retention_vals = [d["retention_100"] for d in all_data]
    auc_vals = [d["forgetting_auc"] for d in all_data]

    print(f"\n{'=' * 70}")
    print(f"SPEARMAN RANK CORRELATION (n={len(all_data)} architectures)")
    print(f"{'=' * 70}")
    print(f"\n{'Metric':>30} | {'ρ (ret@100)':>11} | {'p-value':>9} | {'ρ (AUC)':>10} | {'p-value':>9} | {'Sig?':>4} | {'Avail':>5}")
    print("-" * 100)

    best_metric = None
    best_rho = -1
    all_results = {}

    for metric_key, metric_name in METRICS:
        vals = [d[metric_key] for d in all_data]
        non_none = [(v, r, a) for v, r, a in zip(vals, retention_vals, auc_vals) if v is not None]

        if len(non_none) < 3:
            print(f"{metric_name:>30} | {'N/A':>11} | {'N/A':>9} | {'N/A':>10} | {'N/A':>9} | {'N/A':>4} | {len(non_none):>3}/{len(all_data)}")
            continue

        m_vals, m_ret, m_auc = zip(*non_none)

        # Check for constant input
        if len(set(m_vals)) <= 1:
            print(f"{metric_name:>30} | {'constant':>11} | {'N/A':>9} | {'constant':>10} | {'N/A':>9} | {'N/A':>4} | {len(non_none):>3}/{len(all_data)}")
            continue

        rho_ret, p_ret = stats.spearmanr(m_vals, m_ret)
        rho_auc, p_auc = stats.spearmanr(m_vals, m_auc)
        sig = "YES" if p_ret < 0.05 else "no"

        print(f"{metric_name:>30} | {rho_ret:>11.4f} | {p_ret:>9.4f} | {rho_auc:>10.4f} | {p_auc:>9.4f} | {sig:>4} | {len(non_none):>3}/{len(all_data)}")

        all_results[metric_key] = {
            "metric_name": metric_name,
            "values": list(m_vals),
            "n_available": len(non_none),
            "rho_retention": float(rho_ret) if not np.isnan(rho_ret) else None,
            "p_retention": float(p_ret) if not np.isnan(p_ret) else None,
            "rho_auc": float(rho_auc) if not np.isnan(rho_auc) else None,
            "p_auc": float(p_auc) if not np.isnan(p_auc) else None,
        }

        if not np.isnan(rho_ret) and abs(rho_ret) > best_rho:
            best_rho = abs(rho_ret)
            best_metric = metric_name

    # ─── Partial Correlations (controlling for num_params) ───
    params_vals = [d["num_params"] for d in all_data]
    has_params = all(p is not None for p in params_vals)

    if has_params:
        print(f"\n{'=' * 70}")
        print(f"PARTIAL CORRELATION — controlling for num_params (n={len(all_data)})")
        print(f"{'=' * 70}")
        print(f"\n{'Metric':>30} | {'ρ_partial':>10} | {'p-value':>9} | {'Sig?':>4}")
        print("-" * 60)

        for metric_key, metric_name in METRICS:
            if metric_key == "num_params":
                continue
            vals = [d[metric_key] for d in all_data]
            if any(v is None for v in vals):
                # Filter to matching indices
                valid = [(v, r, p) for v, r, p in zip(vals, retention_vals, params_vals) if v is not None]
                if len(valid) < 5:
                    print(f"{metric_name:>30} | {'N/A':>10} | {'N/A':>9} | {'N/A':>4}")
                    continue
                m_vals, m_ret, m_params = zip(*valid)
            else:
                m_vals, m_ret, m_params = vals, retention_vals, params_vals

            if len(set(m_vals)) <= 1:
                continue

            rho_p, p_p = partial_correlation(m_vals, m_ret, m_params)
            sig = "YES" if not np.isnan(p_p) and p_p < 0.05 else "no"
            rho_str = f"{rho_p:.4f}" if not np.isnan(rho_p) else "N/A"
            p_str = f"{p_p:.4f}" if not np.isnan(p_p) else "N/A"
            print(f"{metric_name:>30} | {rho_str:>10} | {p_str:>9} | {sig:>4}")

            # Store in results
            if metric_key in all_results:
                all_results[metric_key]["rho_partial"] = float(rho_p) if not np.isnan(rho_p) else None
                all_results[metric_key]["p_partial"] = float(p_p) if not np.isnan(p_p) else None

    # ─── Symmetric Partial Correlations + Rank Regression ───
    if has_params:
        h1_vals_full = [d.get("H1") for d in all_data]
        if all(v is not None for v in h1_vals_full):
            h1_arr = np.array(h1_vals_full)
            params_arr = np.array(params_vals)
            ret_arr = np.array(retention_vals)

            print(f"\n{'=' * 70}")
            print(f"SYMMETRIC PARTIAL CORRELATIONS (n={len(all_data)})")
            print(f"{'=' * 70}")

            # H1 vs params collinearity
            r_h1_p, p_h1_p = stats.spearmanr(h1_arr, params_arr)
            print(f"\n  H1 vs params: rho={r_h1_p:.4f} (p={p_h1_p:.4f})")

            # Partial: params vs ret | H1
            rho_p1, pp1 = partial_correlation(params_arr, ret_arr, h1_arr)
            print(f"  Params vs ret | H1:  rho_partial={rho_p1:.4f} (p={pp1:.4f})")

            # Partial: H1 vs ret | params
            rho_p2, pp2 = partial_correlation(h1_arr, ret_arr, params_arr)
            print(f"  H1 vs ret | params:  rho_partial={rho_p2:.4f} (p={pp2:.4f})")

            # Rank regression: rank(ret) ~ rank(params) + rank(H1)
            rank_ret = stats.rankdata(ret_arr)
            rank_params = stats.rankdata(params_arr)
            rank_h1 = stats.rankdata(h1_arr)

            X = np.column_stack([np.ones(len(all_data)), rank_params, rank_h1])
            beta = np.linalg.lstsq(X, rank_ret, rcond=None)[0]
            y_hat = X @ beta
            ss_res = np.sum((rank_ret - y_hat)**2)
            ss_tot = np.sum((rank_ret - rank_ret.mean())**2)
            r_sq = 1 - ss_res / ss_tot
            mse = ss_res / (len(all_data) - 3)
            cov = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            t_stats_reg = beta / se
            p_vals_reg = 2 * stats.t.sf(np.abs(t_stats_reg), len(all_data) - 3)

            print(f"\n  Rank regression: rank(ret) ~ rank(params) + rank(H1)")
            print(f"    R-squared: {r_sq:.4f}")
            print(f"    Params:  beta={beta[1]:.3f}, se={se[1]:.3f}, p={p_vals_reg[1]:.4f}")
            print(f"    H1:      beta={beta[2]:.3f}, se={se[2]:.3f}, p={p_vals_reg[2]:.4f}")

            # Store
            all_results["symmetric_partials"] = {
                "h1_params_collinearity": float(r_h1_p),
                "params_vs_ret_controlling_h1": {"rho": float(rho_p1), "p": float(pp1)},
                "h1_vs_ret_controlling_params": {"rho": float(rho_p2), "p": float(pp2)},
                "rank_regression_r_squared": float(r_sq),
                "rank_regression_params_p": float(p_vals_reg[1]),
                "rank_regression_h1_p": float(p_vals_reg[2]),
            }

    # ─── Architecture Class Analysis ───
    classes = set(d["arch_class"] for d in all_data)
    classes_with_enough = [c for c in classes if sum(1 for d in all_data if d["arch_class"] == c) >= 3]

    if classes_with_enough:
        print(f"\n{'=' * 70}")
        print(f"WITHIN-CLASS CORRELATIONS")
        print(f"{'=' * 70}")

        for cls in sorted(classes_with_enough):
            cls_data = [d for d in all_data if d["arch_class"] == cls]
            cls_ret = [d["retention_100"] for d in cls_data]
            n_cls = len(cls_data)

            print(f"\n  {cls} (n={n_cls}): {', '.join(d['arch_name'] for d in cls_data)}")

            for metric_key, metric_name in [("H0", "H0"), ("H1", "H1"), ("num_params", "Params")]:
                cls_vals = [d[metric_key] for d in cls_data]
                valid = [(v, r) for v, r in zip(cls_vals, cls_ret) if v is not None]
                if len(valid) < 3 or len(set(v for v, _ in valid)) <= 1:
                    continue
                m_vals, m_ret = zip(*valid)
                rho, p = stats.spearmanr(m_vals, m_ret)
                sig = "*" if p < 0.05 else ""
                print(f"    {metric_name:>10}: ρ = {rho:.4f} (p={p:.4f}) {sig}")

    # ─── Permutation Test ───
    print(f"\n{'=' * 70}")
    print(f"PERMUTATION TEST (10,000 shuffles)")
    print(f"{'=' * 70}")
    print(f"\n{'Metric':>30} | {'Obs ρ':>8} | {'Perm p':>8} | {'Tail %':>7}")
    print("-" * 65)

    n_perms = 10000
    rng = np.random.RandomState(42)
    for metric_key, metric_name in METRICS:
        vals = [d[metric_key] for d in all_data]
        non_none = [(v, r) for v, r in zip(vals, retention_vals) if v is not None]
        if len(non_none) < 3:
            continue
        m_vals, m_ret = zip(*non_none)
        m_vals = np.array(m_vals)
        m_ret = np.array(m_ret)

        if len(set(m_vals)) <= 1:
            continue

        obs_rho, _ = stats.spearmanr(m_vals, m_ret)
        # Shuffle retention labels and recompute rho
        perm_rhos = np.empty(n_perms)
        for pi in range(n_perms):
            shuffled = rng.permutation(m_ret)
            perm_rhos[pi], _ = stats.spearmanr(m_vals, shuffled)

        # Two-tailed: fraction of permutations with |rho| >= |observed|
        perm_p = np.mean(np.abs(perm_rhos) >= abs(obs_rho))
        tail_pct = perm_p * 100

        print(f"{metric_name:>30} | {obs_rho:>8.4f} | {perm_p:>8.4f} | {tail_pct:>6.1f}%")

        # Store in results
        if metric_key in all_results:
            all_results[metric_key]["perm_p_retention"] = float(perm_p)

    # Also test AURC
    print(f"\n  Against AURC:")
    print(f"  {'Metric':>28} | {'Obs ρ':>8} | {'Perm p':>8} | {'Tail %':>7}")
    print(f"  {'-' * 61}")
    for metric_key, metric_name in METRICS:
        vals = [d[metric_key] for d in all_data]
        non_none = [(v, a) for v, a in zip(vals, auc_vals) if v is not None]
        if len(non_none) < 3:
            continue
        m_vals, m_auc = zip(*non_none)
        m_vals = np.array(m_vals)
        m_auc = np.array(m_auc)

        if len(set(m_vals)) <= 1:
            continue

        obs_rho, _ = stats.spearmanr(m_vals, m_auc)
        perm_rhos = np.empty(n_perms)
        for pi in range(n_perms):
            shuffled = rng.permutation(m_auc)
            perm_rhos[pi], _ = stats.spearmanr(m_vals, shuffled)

        perm_p = np.mean(np.abs(perm_rhos) >= abs(obs_rho))
        tail_pct = perm_p * 100
        print(f"  {metric_name:>28} | {obs_rho:>8.4f} | {perm_p:>8.4f} | {tail_pct:>6.1f}%")

        if metric_key in all_results:
            all_results[metric_key]["perm_p_auc"] = float(perm_p)

    # ─── Summary ───
    print(f"\n{'─' * 70}")
    if best_metric:
        print(f"  Best predictor of forgetting: {best_metric} (|ρ| = {best_rho:.4f})")

        n = len(all_data)
        print(f"  Critical |ρ| for p<0.05 at n={n}: see p-values above")

        # Check H0 vs best
        h0_result = all_results.get("H0")
        h1_result = all_results.get("H1")

        # Report H0
        if h0_result and h0_result["rho_retention"] is not None:
            h0_rho = abs(h0_result["rho_retention"])
            h0_p = h0_result["p_retention"]
            if h0_p < 0.05:
                print(f"\n  ★ H0 SIGNIFICANT: ρ={h0_result['rho_retention']:.4f}, p={h0_p:.4f}")
            else:
                print(f"\n  ○ H0 NOT SIGNIFICANT: ρ={h0_result['rho_retention']:.4f}, p={h0_p:.4f}")

        # Report H1
        if h1_result and h1_result["rho_retention"] is not None:
            h1_rho = abs(h1_result["rho_retention"])
            h1_p = h1_result["p_retention"]
            if h1_p < 0.05:
                print(f"  ★ H1 SIGNIFICANT: ρ={h1_result['rho_retention']:.4f}, p={h1_p:.4f}")
            else:
                print(f"  ○ H1 NOT SIGNIFICANT: ρ={h1_result['rho_retention']:.4f}, p={h1_p:.4f}")

        # Report params baseline
        params_result = all_results.get("num_params")
        if params_result and params_result["rho_retention"] is not None:
            pr = params_result["rho_retention"]
            pp = params_result["p_retention"]
            print(f"\n  BASELINE (num_params only): ρ={pr:.4f}, p={pp:.4f}")
            if pp < 0.05:
                print(f"  ⚠ WARNING: Parameter count alone predicts retention. Topology must beat this.")
            else:
                print(f"  ✓ Parameter count alone does NOT predict retention. Topology adds value.")

    # Save full results
    results = {
        "n_architectures": len(all_data),
        "architectures": [d["label"] for d in all_data],
        "arch_classes": {d["label"]: d["arch_class"] for d in all_data},
        "retention_ratios_100": retention_vals,
        "forgetting_auc_values": auc_vals,
        "per_architecture": all_data,
        "correlations": all_results,
        "best_metric": best_metric,
        "best_rho": float(best_rho) if best_rho > 0 else None,
    }
    out_path = os.path.join(os.path.dirname(result_dirs[0]), "correlation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="EXP-01 Phase 4: Correlation Analysis")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--results-dirs", nargs="+", type=str, default=None,
                        help="Multiple result directories for cross-architecture analysis")
    args = parser.parse_args()

    if args.results_dirs and len(args.results_dirs) > 1:
        cross_architecture_analysis(args.results_dirs)
    elif args.config:
        cfg = load_config(args.config)
        single_run_analysis(cfg["output_dir"])
    elif args.results_dirs and len(args.results_dirs) == 1:
        single_run_analysis(args.results_dirs[0])
    else:
        parser.error("Provide --config or --results-dirs")


# Need torch for loading checkpoints
try:
    import torch
except ImportError:
    pass


if __name__ == "__main__":
    main()
