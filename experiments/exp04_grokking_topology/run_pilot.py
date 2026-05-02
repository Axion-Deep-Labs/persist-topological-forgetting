"""
EXP-04: Pilot runner.

Runs 5 seeds through the full pipeline:
  1. Train to 100K steps with checkpoint saving
  2. At each checkpoint: compute PH (5 slices) + all baselines
  3. Save complete results per seed

Usage:
    python -m experiments.exp04_grokking_topology.run_pilot \
        --config configs/exp04_pilot.yaml

Output:
    results/exp04_pilot/seed_<N>/
        training_metrics.json   — train/test loss/acc at each checkpoint
        topology_metrics.json   — PH stats (averaged + per-slice) at each checkpoint
        baseline_metrics.json   — all baseline metrics at each checkpoint
        checkpoints/            — model state dicts
"""

import argparse
import gc
import os
import sys
import json
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.shared.utils import set_seed, load_config
from experiments.exp04_grokking_topology.dataset import get_dataloaders
from experiments.exp04_grokking_topology.model import build_model
from experiments.exp04_grokking_topology.train import train_seed, get_checkpoint_steps
from experiments.exp04_grokking_topology.topology import compute_topology_at_checkpoint
from experiments.exp04_grokking_topology.baselines import compute_all_baselines


def _load_existing_results(path):
    """Load existing results JSON and return (list, set of completed steps)."""
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
        completed = {r["step"] for r in results}
        return results, completed
    return [], set()


def _save_json(path, data):
    """Atomically save JSON (write tmp then rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def run_analysis_pass(cfg, seed, output_dir, device):
    """Run PH and baseline computation on saved checkpoints for one seed.

    Supports resume: loads existing results and skips already-computed steps.
    Saves incrementally after each checkpoint to avoid losing progress on OOM/kill.
    """
    run_dir = os.path.join(output_dir, f"seed_{seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    # Load training metrics (needed for loss curvature, gen gap, val slope)
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    with open(metrics_path) as f:
        training_data = json.load(f)
    metrics_log = training_data["metrics"]

    # Rebuild dataloaders (fixed split, seed doesn't matter for data)
    set_seed(seed)
    _, test_loader = get_dataloaders(cfg)

    # Build model shell (weights loaded per checkpoint)
    model = build_model(cfg, device)

    # Get checkpoint steps
    ckpt_steps = get_checkpoint_steps(cfg)

    # Resume support: load existing results and skip completed steps
    topo_path = os.path.join(run_dir, "topology_metrics.json")
    bl_path = os.path.join(run_dir, "baseline_metrics.json")
    topology_results, topo_done = _load_existing_results(topo_path)
    baseline_results, bl_done = _load_existing_results(bl_path)
    # A step is fully done only if both topology and baselines are computed
    done_steps = topo_done & bl_done
    if done_steps:
        print(f"  Resuming: {len(done_steps)}/{len(ckpt_steps)} steps already complete")

    for idx, step in enumerate(ckpt_steps):
        if step in done_steps:
            continue

        ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Skipping step {step} (no checkpoint)")
            continue

        print(f"\n  === Step {step} ({idx + 1}/{len(ckpt_steps)}) ===")

        # Find matching metrics entry
        metrics_idx = None
        for mi, m in enumerate(metrics_log):
            if m["step"] == step:
                metrics_idx = mi
                break

        # --- Topology ---
        if step not in topo_done:
            print(f"  Computing topology (5 slices, 50x50 grid)...")
            set_seed(seed * 100000 + step)
            t0 = time.time()
            topo = compute_topology_at_checkpoint(model, ckpt_path, test_loader, cfg, device)
            topo_time = time.time() - t0
            topo_entry = {
                "step": step,
                "time_seconds": round(topo_time, 1),
                **topo["averaged"],
                "slice_variance": topo["slice_variance"],
            }
            topology_results.append(topo_entry)
            print(f"    H0 pers={topo['averaged']['h0_total_persistence']:.1f} "
                  f"H0 eff_count={topo['averaged']['h0_effective_feature_count']:.2f} "
                  f"H0 ent={topo['averaged']['h0_persistence_entropy']:.4f} "
                  f"({topo_time:.1f}s)")
            del topo
        else:
            print(f"  Topology already computed, skipping")

        # --- Baselines ---
        if step not in bl_done:
            print(f"  Computing baselines...")
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()
            t0 = time.time()
            baselines = compute_all_baselines(model, test_loader, device, metrics_log, metrics_idx)
            bl_time = time.time() - t0
            bl_entry = {"step": step, "time_seconds": round(bl_time, 1), **baselines}
            baseline_results.append(bl_entry)
            sharp = baselines.get('sharpness')
            comm = baselines.get('commutator_defect')
            sharp_str = f"{sharp:.4f}" if sharp is not None else "FAILED"
            comm_str = f"{comm:.6f}" if comm is not None else "FAILED"
            print(f"    sharpness={sharp_str} commutator={comm_str} ({bl_time:.1f}s)")
            del baselines
        else:
            print(f"  Baselines already computed, skipping")

        # Save incrementally after each checkpoint
        _save_json(topo_path, topology_results)
        _save_json(bl_path, baseline_results)

        # Free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  Saved topology metrics to {topo_path}")
    print(f"  Saved baseline metrics to {bl_path}")


def main():
    parser = argparse.ArgumentParser(description="EXP-04 Pilot: Grokking Topology")
    parser.add_argument("--config", type=str, default="configs/exp04_pilot.yaml")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only run analysis on existing checkpoints")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip analysis, only run training")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run a single seed (for HPC parallel jobs)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Override config weight_decay (for WD sweep in full study)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override default results/exp04_pilot output (e.g. results/exp04_full/wd_0.10)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.weight_decay is not None:
        cfg["training"]["weight_decay"] = float(args.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    output_dir = args.output_dir if args.output_dir else os.path.join("results", "exp04_pilot")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("EXP-04: Topological Dynamics of Grokking")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Seeds: {seeds}")
    print(f"  Task: mod {cfg['task']['modulus']} addition")
    print(f"  Weight decay: {cfg['training']['weight_decay']}")
    print(f"  Output: {output_dir}")
    print()

    total_start = time.time()

    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 60}")
        print(f"  SEED {seed} ({i + 1}/{len(seeds)})")
        print(f"{'=' * 60}")

        # Phase 1: Training
        if not args.skip_training:
            print(f"\n  --- Training ---")
            train_seed(cfg, seed, output_dir, device)

        # Phase 2: Analysis (PH + baselines)
        if not args.skip_analysis:
            print(f"\n  --- Analysis ---")
            run_analysis_pass(cfg, seed, output_dir, device)

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  PILOT COMPLETE ({total_time / 3600:.1f} hours)")
    print(f"  Results in: {output_dir}")
    print(f"{'=' * 60}")

    # Summary: check pilot gate
    print(f"\n  PILOT GATE CHECK:")
    print(f"  Criterion: at least one PH stat shows consistent directional")
    print(f"  behavior across >= {cfg['pilot_gate']['min_consistent_seeds']}/{len(seeds)} seeds")
    print(f"  before grokking onset.")
    print(f"\n  Review results manually before proceeding to full study.")


if __name__ == "__main__":
    main()
