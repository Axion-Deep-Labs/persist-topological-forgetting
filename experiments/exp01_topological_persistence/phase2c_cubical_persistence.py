"""
EXP-01 Phase 2c: Cubical persistent homology on existing loss landscapes.

Computes cubical PH using GUDHI on the raw 50x50 loss grid. This is the
mathematically natural topology for a scalar field on a regular grid, and
serves as a validation baseline for the Ripser graph-based approach in Phase 2.

Reads loss_landscape.npz (and _run*.npz variants) from Phase 2 output.
No GPU required.

Usage:
    # Single experiment
    python -m experiments.exp01_topological_persistence.phase2c_cubical_persistence \
        --results-dir results/exp01

    # All experiments with landscape data
    python -m experiments.exp01_topological_persistence.phase2c_cubical_persistence \
        --all
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def compute_cubical_persistence(loss_grid, maxdim=1):
    """Compute persistent homology using cubical complexes on the loss grid.

    Cubical complexes are the natural representation for scalar fields on
    regular grids. Uses sublevel set filtration: features appear as the
    threshold rises through the loss surface.

    Args:
        loss_grid: 2D numpy array of loss values
        maxdim: maximum homology dimension (1 = H0 + H1)

    Returns:
        dict with H0, H1 persistence statistics
    """
    import gudhi

    cc = gudhi.CubicalComplex(
        dimensions=list(loss_grid.shape),
        top_dimensional_cells=loss_grid.flatten().tolist(),
    )
    cc.persistence()

    result = {}
    for dim in range(maxdim + 1):
        key = f"H{dim}"
        pairs = cc.persistence_intervals_in_dimension(dim)

        if len(pairs) == 0:
            result[key] = 0.0
            result[f"{key}_count"] = 0
            result[f"{key}_max_lifetime"] = 0.0
            continue

        # Filter to finite lifetimes
        finite_mask = np.isfinite(pairs[:, 1])
        finite_pairs = pairs[finite_mask]

        if len(finite_pairs) == 0:
            result[key] = 0.0
            result[f"{key}_count"] = 0
            result[f"{key}_max_lifetime"] = 0.0
            continue

        lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
        result[key] = float(np.sum(lifetimes))
        result[f"{key}_count"] = int(len(finite_pairs))
        result[f"{key}_max_lifetime"] = float(np.max(lifetimes))

    return result


def process_landscape(npz_path, output_path):
    """Process a single loss landscape .npz file."""
    data = np.load(npz_path)
    loss_grid = data["loss_grid"]

    t0 = time.time()
    stats = compute_cubical_persistence(loss_grid, maxdim=1)
    compute_time = time.time() - t0

    summary = {
        "source_landscape": npz_path,
        "grid_shape": list(loss_grid.shape),
        "loss_min": float(loss_grid.min()),
        "loss_max": float(loss_grid.max()),
        "loss_mean": float(loss_grid.mean()),
        "cubical_compute_time_s": compute_time,
        **stats,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def process_experiment(result_dir):
    """Process all landscape files for a single experiment."""
    topo_dir = os.path.join(result_dir, "topology")
    if not os.path.isdir(topo_dir):
        return 0

    processed = 0

    # Default landscape
    default_npz = os.path.join(topo_dir, "loss_landscape.npz")
    if os.path.exists(default_npz):
        out_path = os.path.join(topo_dir, "cubical_summary.json")
        if not os.path.exists(out_path):
            summary = process_landscape(default_npz, out_path)
            print(f"  {os.path.basename(result_dir)}: H0={summary['H0']:.1f}, "
                  f"H1={summary['H1']:.4f} ({summary['cubical_compute_time_s']:.2f}s)")
            processed += 1

    # Multi-slice landscapes
    run_files = sorted(glob.glob(os.path.join(topo_dir, "loss_landscape_run*.npz")))
    for npz_path in run_files:
        # Extract run ID: loss_landscape_run1.npz -> run1
        basename = os.path.basename(npz_path)
        run_id = basename.replace("loss_landscape_", "").replace(".npz", "")
        out_path = os.path.join(topo_dir, f"cubical_summary_{run_id}.json")

        if not os.path.exists(out_path):
            summary = process_landscape(npz_path, out_path)
            print(f"  {os.path.basename(result_dir)} [{run_id}]: H0={summary['H0']:.1f}, "
                  f"H1={summary['H1']:.4f} ({summary['cubical_compute_time_s']:.2f}s)")
            processed += 1

    return processed


def main():
    parser = argparse.ArgumentParser(description="EXP-01 Phase 2c: Cubical Persistence")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Single experiment results directory")
    parser.add_argument("--all", action="store_true",
                        help="Process all experiment directories in results/")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cubical_summary.json exists")
    args = parser.parse_args()

    print("EXP-01 Phase 2c: Cubical Persistent Homology")
    print("  Method: GUDHI CubicalComplex (sublevel set filtration)")
    print()

    # Verify gudhi is available
    try:
        import gudhi  # noqa: F401
    except ImportError:
        print("ERROR: gudhi not installed. Run: pip install gudhi")
        sys.exit(1)

    if args.all:
        results_root = os.path.join(os.path.dirname(__file__), "../../results")
        results_root = os.path.abspath(results_root)
        if not os.path.isdir(results_root):
            print(f"Results directory not found: {results_root}")
            sys.exit(1)

        total = 0
        for entry in sorted(os.listdir(results_root)):
            result_dir = os.path.join(results_root, entry)
            if os.path.isdir(result_dir) and not entry.startswith("."):
                total += process_experiment(result_dir)

        print(f"\nPhase 2c complete. Processed {total} landscape files.")

    elif args.results_dir:
        total = process_experiment(args.results_dir)
        print(f"\nPhase 2c complete. Processed {total} landscape files.")

    else:
        parser.error("Provide --results-dir or --all")


if __name__ == "__main__":
    main()
