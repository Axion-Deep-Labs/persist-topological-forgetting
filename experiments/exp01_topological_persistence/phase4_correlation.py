"""
EXP-01 Phase 4: Correlate topological features with forgetting resistance.

Loads topology summary (Phase 2) and forgetting curve (Phase 3),
computes correlation between topological persistence and retention rate.

For a single architecture this produces a baseline measurement.
For cross-architecture comparison, run Phases 1-3 with different architectures
and this script aggregates the results.

Usage:
    python -m experiments.exp01_topological_persistence.phase4_correlation \
        --config configs/exp01.yaml

    # Cross-architecture (after running multiple configs):
    python -m experiments.exp01_topological_persistence.phase4_correlation \
        --results-dirs results/exp01_resnet18 results/exp01_vit results/exp01_lstm
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.shared.utils import load_config


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
    print(f"  H0 total persistence:   {topo.get('H0', 'N/A'):.4f}")
    print(f"  H0 feature count:       {topo.get('H0_count', 'N/A')}")
    print(f"  H1 total persistence:   {topo.get('H1', 'N/A'):.4f}")
    print(f"  H1 feature count:       {topo.get('H1_count', 'N/A')}")
    print(f"  H1 max lifetime:        {topo.get('H1_max_lifetime', 'N/A'):.4f}")
    print(f"  H2 total persistence:   {topo.get('H2', 'N/A'):.4f}")
    print(f"  H2 feature count:       {topo.get('H2_count', 'N/A')}")

    print(f"\nForgetting Curve:")
    print(f"  Initial Task A accuracy: {forget['initial_task_a_acc']:.1%}")
    for point in forget["curve"]:
        if point["step"] > 0:
            retention = point["task_a_acc"] / forget["initial_task_a_acc"]
            print(f"  Step {point['step']:>6}: Task A = {point['task_a_acc']:.1%}, "
                  f"Retention = {retention:.1%}, "
                  f"Task B = {point.get('task_b_acc', 0):.1%}")

    # Key metric: retention at step 10000
    ret_10k = compute_retention_at_step(forget, 10000)
    if ret_10k is not None:
        retention_ratio = ret_10k / forget["initial_task_a_acc"]
        print(f"\n  Retention ratio at step 10,000: {retention_ratio:.1%}")
        print(f"  H1 total persistence: {topo.get('H1', 0):.4f}")
        print(f"\n  → This is one data point. Run with multiple architectures")
        print(f"    for Spearman correlation analysis.")


def cross_architecture_analysis(result_dirs):
    """Correlate topology with forgetting across multiple architectures."""
    print("=" * 60)
    print("CROSS-ARCHITECTURE CORRELATION ANALYSIS")
    print("=" * 60)

    h1_persistence = []
    retention_ratios = []
    labels = []

    for rdir in result_dirs:
        try:
            topo, forget = load_experiment_data(rdir)
        except FileNotFoundError as e:
            print(f"  Skipping {rdir}: {e}")
            continue

        h1 = topo.get("H1", 0)
        ret_10k = compute_retention_at_step(forget, 10000)
        if ret_10k is None:
            # Use last available step
            last_point = forget["curve"][-1]
            ret_10k = last_point["task_a_acc"]

        retention = ret_10k / forget["initial_task_a_acc"]

        h1_persistence.append(h1)
        retention_ratios.append(retention)
        labels.append(os.path.basename(rdir))

        print(f"\n  {os.path.basename(rdir)}:")
        print(f"    H1 persistence: {h1:.4f}")
        print(f"    Retention ratio: {retention:.1%}")

    if len(h1_persistence) >= 3:
        rho, p_value = stats.spearmanr(h1_persistence, retention_ratios)
        print(f"\n{'─' * 40}")
        print(f"  Spearman ρ(H1 persistence, retention): {rho:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant (p < 0.05): {'YES' if p_value < 0.05 else 'NO'}")

        if rho > 0.5 and p_value < 0.05:
            print(f"\n  ★ POSITIVE RESULT: Topological depth predicts forgetting resistance.")
            print(f"    Proceed to Phase 5 (topological regularizer).")
        elif rho < -0.5 and p_value < 0.05:
            print(f"\n  ✦ INVERSE RESULT: Deeper topology correlates with MORE forgetting.")
            print(f"    Investigate mechanism — possible over-specialization of landscape.")
        else:
            print(f"\n  ○ No significant correlation at current sample size.")
            print(f"    Consider: more architectures, different homology dimensions,")
            print(f"    or alternative topology metrics (Wasserstein, Betti curves).")

        # Save results
        results = {
            "architectures": labels,
            "h1_persistence": h1_persistence,
            "retention_ratios": retention_ratios,
            "spearman_rho": rho,
            "p_value": p_value,
        }
        out_path = os.path.join(os.path.dirname(result_dirs[0]), "correlation_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {out_path}")
    else:
        print(f"\n  Need >= 3 architectures for correlation. Have {len(h1_persistence)}.")
        print(f"  Run experiments with different architectures first.")


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
