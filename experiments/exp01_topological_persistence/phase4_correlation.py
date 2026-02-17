"""
EXP-01 Phase 4: Correlate topological features with forgetting resistance.

Compares H0 persistence against baseline metrics (Hessian trace, max eigenvalue,
Fisher trace, loss barrier) to determine whether topology captures something
that simpler geometry metrics do not.

Usage:
    python -m experiments.exp01_topological_persistence.phase4_correlation \
        --config configs/exp01.yaml

    # Cross-architecture (after running multiple configs):
    python -m experiments.exp01_topological_persistence.phase4_correlation \
        --results-dirs results/exp01 results/exp01_vit results/exp01_resnet50 ...
"""

import argparse
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
    ("hessian_trace_mean", "Hessian Trace"),
    ("max_eigenvalue", "Max Eigenvalue (Sharpness)"),
    ("fisher_trace", "Fisher Information Trace"),
    ("max_barrier", "Loss Barrier Height"),
]


def load_experiment_data(result_dir):
    """Load topology summary and forgetting curve from a single experiment run."""
    topo_path = os.path.join(result_dir, "topology", "topology_summary.json")
    forget_path = os.path.join(result_dir, "forgetting", "forgetting_curve.json")

    with open(topo_path) as f:
        topo = json.load(f)
    with open(forget_path) as f:
        forget = json.load(f)

    return topo, forget


def compute_retention_at_step(forget_data, step):
    """Get Task A accuracy at a specific training step."""
    for point in forget_data["curve"]:
        if point["step"] == step:
            return point["task_a_acc"]
    return None


def single_run_analysis(result_dir):
    """Analyze a single experiment run."""
    topo, forget = load_experiment_data(result_dir)

    print("=" * 60)
    print("SINGLE RUN ANALYSIS")
    print("=" * 60)

    print(f"\nTopological Features (Task A converged model):")
    for dim in range(3):
        key = f"H{dim}"
        if key in topo:
            print(f"  {key} total persistence:   {topo[key]:.4f}")
            print(f"  {key} feature count:       {topo.get(f'{key}_count', 0)}")
            if f"{key}_max_lifetime" in topo:
                print(f"  {key} max lifetime:        {topo[f'{key}_max_lifetime']:.4f}")

    # Baseline metrics (if available)
    baseline_keys = ["hessian_trace_mean", "max_eigenvalue", "fisher_trace", "max_barrier"]
    has_baseline = any(k in topo for k in baseline_keys)
    if has_baseline:
        print(f"\nBaseline Geometry Metrics:")
        for key in baseline_keys:
            if key in topo:
                print(f"  {key}: {topo[key]:.4f}")

    print(f"\nForgetting Curve:")
    print(f"  Initial Task A accuracy: {forget['initial_task_a_acc']:.1%}")
    for point in forget["curve"]:
        if point["step"] > 0:
            retention = point["task_a_acc"] / forget["initial_task_a_acc"]
            print(f"  Step {point['step']:>6}: Task A = {point['task_a_acc']:.1%}, "
                  f"Retention = {retention:.1%}, "
                  f"Task B = {point.get('task_b_acc', 0):.1%}")

    ret_10k = compute_retention_at_step(forget, 10000)
    if ret_10k is not None:
        retention_ratio = ret_10k / forget["initial_task_a_acc"]
        print(f"\n  Retention ratio at step 10,000: {retention_ratio:.1%}")
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

        ret_10k = compute_retention_at_step(forget, 10000)
        if ret_10k is None:
            last_point = forget["curve"][-1]
            ret_10k = last_point["task_a_acc"]

        retention = ret_10k / forget["initial_task_a_acc"]

        # Also compute area-under-retention-curve (more robust than single step)
        aurc = 0.0
        prev_step = 0
        for point in forget["curve"]:
            if point["step"] > 0:
                step_retention = point["task_a_acc"] / forget["initial_task_a_acc"]
                aurc += step_retention * (point["step"] - prev_step)
                prev_step = point["step"]

        entry = {
            "label": os.path.basename(rdir),
            "retention_10k": retention,
            "aurc": aurc,
            "accuracy": forget["initial_task_a_acc"],
        }
        # Collect all metric values
        for metric_key, _ in METRICS:
            entry[metric_key] = topo.get(metric_key, None)

        all_data.append(entry)

    if len(all_data) < 3:
        print(f"\n  Need >= 3 architectures for correlation. Have {len(all_data)}.")
        return

    # Print summary table
    print(f"\n{'Architecture':>20} | {'Accuracy':>8} | {'H0 Pers':>8} | {'Hess Tr':>9} | {'Max Eig':>9} | {'Fisher':>9} | {'Barrier':>8} | {'Ret@10k':>7} | {'AURC':>8}")
    print("-" * 120)
    for d in all_data:
        h0 = f"{d['H0']:.1f}" if d['H0'] is not None else "N/A"
        ht = f"{d['hessian_trace_mean']:.1f}" if d.get('hessian_trace_mean') is not None else "N/A"
        me = f"{d['max_eigenvalue']:.1f}" if d.get('max_eigenvalue') is not None else "N/A"
        ft = f"{d['fisher_trace']:.1f}" if d.get('fisher_trace') is not None else "N/A"
        mb = f"{d['max_barrier']:.2f}" if d.get('max_barrier') is not None else "N/A"
        print(f"{d['label']:>20} | {d['accuracy']:7.1%} | {h0:>8} | {ht:>9} | {me:>9} | {ft:>9} | {mb:>8} | {d['retention_10k']:6.1%} | {d['aurc']:8.1f}")

    # Correlate each metric against retention
    retention_vals = [d["retention_10k"] for d in all_data]
    aurc_vals = [d["aurc"] for d in all_data]

    print(f"\n{'=' * 70}")
    print(f"SPEARMAN RANK CORRELATION (n={len(all_data)} architectures)")
    print(f"{'=' * 70}")
    print(f"\n{'Metric':>30} | {'ρ (ret@10k)':>11} | {'p-value':>9} | {'ρ (AURC)':>10} | {'p-value':>9} | {'Sig?':>4}")
    print("-" * 90)

    best_metric = None
    best_rho = -1
    all_results = {}

    for metric_key, metric_name in METRICS:
        vals = [d[metric_key] for d in all_data]
        if any(v is None for v in vals):
            print(f"{metric_name:>30} | {'N/A':>11} | {'N/A':>9} | {'N/A':>10} | {'N/A':>9} | {'N/A':>4}")
            continue

        # Check for constant input
        if len(set(vals)) <= 1:
            print(f"{metric_name:>30} | {'constant':>11} | {'N/A':>9} | {'constant':>10} | {'N/A':>9} | {'N/A':>4}")
            continue

        rho_ret, p_ret = stats.spearmanr(vals, retention_vals)
        rho_aurc, p_aurc = stats.spearmanr(vals, aurc_vals)
        sig = "YES" if p_ret < 0.05 else "no"

        print(f"{metric_name:>30} | {rho_ret:>11.4f} | {p_ret:>9.4f} | {rho_aurc:>10.4f} | {p_aurc:>9.4f} | {sig:>4}")

        all_results[metric_key] = {
            "metric_name": metric_name,
            "values": vals,
            "rho_retention": float(rho_ret) if not np.isnan(rho_ret) else None,
            "p_retention": float(p_ret) if not np.isnan(p_ret) else None,
            "rho_aurc": float(rho_aurc) if not np.isnan(rho_aurc) else None,
            "p_aurc": float(p_aurc) if not np.isnan(p_aurc) else None,
        }

        if not np.isnan(rho_ret) and abs(rho_ret) > best_rho:
            best_rho = abs(rho_ret)
            best_metric = metric_name

    # Summary
    print(f"\n{'─' * 70}")
    if best_metric:
        print(f"  Best predictor of forgetting: {best_metric} (|ρ| = {best_rho:.4f})")

        h0_result = all_results.get("H0")
        if h0_result and h0_result["rho_retention"] is not None:
            h0_rho = abs(h0_result["rho_retention"])
            if h0_rho >= best_rho - 0.01:
                print(f"\n  ★ TOPOLOGY WINS (or ties): H0 persistence is the best/co-best predictor.")
                print(f"    This supports the claim that topology captures unique geometric information.")
            elif h0_rho > 0.5:
                print(f"\n  ◎ TOPOLOGY IS COMPETITIVE: H0 persistence correlates (ρ={h0_rho:.3f})")
                print(f"    but {best_metric} correlates more strongly (ρ={best_rho:.3f}).")
                print(f"    Investigate whether they capture different aspects of geometry.")
            else:
                print(f"\n  ○ TOPOLOGY UNDERPERFORMS: H0 persistence (ρ={h0_rho:.3f}) is weaker than")
                print(f"    {best_metric} (ρ={best_rho:.3f}). The topological framing may not add value")
                print(f"    beyond simpler geometry metrics.")

    # Save full results
    results = {
        "architectures": [d["label"] for d in all_data],
        "retention_ratios": retention_vals,
        "aurc_values": aurc_vals,
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


if __name__ == "__main__":
    main()
