"""
EXP-01 Phase 2.5: Displacement Analysis — Track model trajectory through loss landscape.

After Task B training displaces the model from its Task A minimum, this script
projects the displaced weights back into Phase 2's 2D landscape coordinate system
to measure where the model landed.

Testable prediction:
  - Models with H1 > 0 (topological loops) should show curved trajectories
    that stay in low-loss basins
  - Models with H1 = 0 should show straight-line escapes to high-loss regions

Requires:
  - Phase 2 completed (landscape_directions.pt, loss_landscape.npz)
  - Phase 3 completed with save_checkpoints: true (forgetting/step_*.pt)

Usage:
    python -m experiments.exp01_topological_persistence.phase2b_displacement_analysis \
        --config configs/exp01.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.shared.datasets import SplitCIFAR100
from experiments.shared.models import get_model
from experiments.shared.utils import set_seed, load_config, load_checkpoint


def flatten_params(param_list):
    """Flatten a list of parameter tensors into a single 1D vector."""
    return torch.cat([p.reshape(-1) for p in param_list])


def project_onto_landscape(displaced_params, base_params, dir1, dir2):
    """Project displaced model parameters into the 2D landscape coordinate system.

    Given:
      displaced_params = base_params + alpha * dir1 + beta * dir2 + residual
    Solve for alpha, beta via dot products:
      alpha = <delta, dir1_flat> / <dir1_flat, dir1_flat>
      beta  = <delta, dir2_flat> / <dir2_flat, dir2_flat>

    Returns:
      alpha, beta: coordinates in the 2D landscape plane
      residual_norm: magnitude of out-of-plane displacement
      displacement_norm: total displacement magnitude
    """
    delta = flatten_params(displaced_params) - flatten_params(base_params)
    dir1_flat = flatten_params(dir1)
    dir2_flat = flatten_params(dir2)

    alpha = torch.dot(delta, dir1_flat) / torch.dot(dir1_flat, dir1_flat)
    beta = torch.dot(delta, dir2_flat) / torch.dot(dir2_flat, dir2_flat)

    # Residual: component of delta not captured by dir1, dir2
    projection = alpha * dir1_flat + beta * dir2_flat
    residual = delta - projection

    return (
        float(alpha),
        float(beta),
        float(residual.norm()),
        float(delta.norm()),
    )


@torch.no_grad()
def evaluate_loss(model, images, labels):
    """Compute cross-entropy loss on a batch."""
    criterion = nn.CrossEntropyLoss()
    outputs = model(images)
    return criterion(outputs, labels).item()


def main():
    parser = argparse.ArgumentParser(description="EXP-01 Phase 2.5: Displacement Analysis")
    parser.add_argument("--config", type=str, default="configs/exp01.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    output_dir = cfg["output_dir"]
    topo_dir = os.path.join(output_dir, "topology")
    forget_dir = os.path.join(output_dir, "forgetting")
    disp_dir = os.path.join(output_dir, "displacement")
    os.makedirs(disp_dir, exist_ok=True)

    print("EXP-01 Phase 2.5: Displacement Analysis")
    print(f"  Architecture: {cfg['architecture']}")
    print(f"  Device: {device}")
    print()

    # ── Load landscape directions from Phase 2 ──
    directions_path = os.path.join(topo_dir, "landscape_directions.pt")
    if not os.path.exists(directions_path):
        print(f"ERROR: {directions_path} not found.")
        print("  Phase 2 must be re-run with the updated script that saves directions.")
        sys.exit(1)

    print("Loading landscape directions from Phase 2...")
    directions_data = torch.load(directions_path, map_location="cpu", weights_only=False)
    dir1 = directions_data["dir1"]
    dir2 = directions_data["dir2"]
    base_params = directions_data["base_params"]
    print(f"  Loaded {len(dir1)} parameter groups")

    # ── Load loss landscape for overlay ──
    landscape_path = os.path.join(topo_dir, "loss_landscape.npz")
    if os.path.exists(landscape_path):
        landscape_data = np.load(landscape_path)
        loss_grid = landscape_data["loss_grid"]
        alphas = landscape_data["alphas"]
        betas = landscape_data["betas"]
        print(f"  Landscape grid: {loss_grid.shape[0]}x{loss_grid.shape[1]}")
        print(f"  Alpha range: [{alphas[0]:.2f}, {alphas[-1]:.2f}]")
        print(f"  Beta range:  [{betas[0]:.2f}, {betas[-1]:.2f}]")
    else:
        loss_grid = None
        print("  WARNING: loss_landscape.npz not found, skipping overlay")

    # ── Load topology summary for H0/H1 context ──
    topo_summary_path = os.path.join(topo_dir, "topology_summary.json")
    h0_persistence = None
    h1_persistence = None
    if os.path.exists(topo_summary_path):
        with open(topo_summary_path) as f:
            topo_summary = json.load(f)
        h0_persistence = topo_summary.get("H0")
        h1_persistence = topo_summary.get("H1")
        print(f"  H0 persistence: {h0_persistence}")
        print(f"  H1 persistence: {h1_persistence}")

    # ── Load model and Task A test data ──
    model = get_model(cfg["architecture"], num_classes=cfg["num_classes_a"]).to(device)
    data = SplitCIFAR100(cfg["data_dir"], split_at=cfg["task_a_classes"][1])
    _, test_loader = data.get_task_a(batch_size=256)

    # Pre-load test set to GPU for fast evaluation
    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
    test_images = torch.cat(all_images).to(device)
    test_labels = torch.cat(all_labels).to(device)
    print(f"  Test set: {test_images.shape[0]} samples")

    # ── Find Phase 3 checkpoints ──
    eval_steps = cfg["forgetting"]["eval_steps"]
    checkpoint_paths = {}
    for step in eval_steps:
        ckpt_path = os.path.join(forget_dir, f"step_{step}.pt")
        if os.path.exists(ckpt_path):
            checkpoint_paths[step] = ckpt_path

    if not checkpoint_paths:
        print(f"\nERROR: No Phase 3 checkpoints found in {forget_dir}/")
        print("  Phase 3 must be run with save_checkpoints: true")
        sys.exit(1)

    print(f"\n  Found {len(checkpoint_paths)} checkpoints: steps {sorted(checkpoint_paths.keys())}")

    # ── Compute displacement trajectory ──
    print("\nComputing displacement trajectory...")
    trajectory = []

    # Step 0: base model (origin of landscape)
    for param, base in zip(model.parameters(), base_params):
        param.data.copy_(base.to(device))
    model.eval()
    base_loss = evaluate_loss(model, test_images, test_labels)

    trajectory.append({
        "step": 0,
        "alpha": 0.0,
        "beta": 0.0,
        "residual_norm": 0.0,
        "displacement_norm": 0.0,
        "loss_at_position": base_loss,
        "in_basin": True,
    })
    print(f"  Step     0: (0.000, 0.000) loss={base_loss:.4f} [origin]")

    # Each checkpoint
    for step in sorted(checkpoint_paths.keys()):
        ckpt = torch.load(checkpoint_paths[step], map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]

        # Extract parameter values in same order as base_params
        displaced_params = []
        for name, param in model.named_parameters():
            displaced_params.append(state_dict[name])

        # Project into landscape coordinates
        alpha, beta, residual_norm, displacement_norm = project_onto_landscape(
            displaced_params, base_params, dir1, dir2
        )

        # Load displaced model and evaluate Task A loss at this position
        model.load_state_dict(state_dict)
        model.eval()
        loss_at_position = evaluate_loss(model, test_images, test_labels)

        # Check if still in low-loss basin (within 2x base loss)
        in_basin = loss_at_position < base_loss * 2.0

        # Task A accuracy at this step (from checkpoint)
        task_a_acc = ckpt.get("accuracy", None)

        point = {
            "step": step,
            "alpha": alpha,
            "beta": beta,
            "residual_norm": residual_norm,
            "displacement_norm": displacement_norm,
            "loss_at_position": loss_at_position,
            "in_basin": in_basin,
            "task_a_acc": task_a_acc,
        }
        trajectory.append(point)

        basin_str = "IN basin" if in_basin else "ESCAPED"
        print(f"  Step {step:>5}: ({alpha:+.4f}, {beta:+.4f}) loss={loss_at_position:.4f} "
              f"disp={displacement_norm:.4f} resid={residual_norm:.4f} [{basin_str}]")

    # ── Compute trajectory metrics ──
    final = trajectory[-1]
    total_displacement = final["displacement_norm"]

    # Trajectory curvature: ratio of path length to straight-line distance
    path_length = 0.0
    for i in range(1, len(trajectory)):
        da = trajectory[i]["alpha"] - trajectory[i-1]["alpha"]
        db = trajectory[i]["beta"] - trajectory[i-1]["beta"]
        path_length += np.sqrt(da**2 + db**2)

    straight_line = np.sqrt(final["alpha"]**2 + final["beta"]**2)
    curvature_ratio = path_length / straight_line if straight_line > 1e-8 else float("inf")

    # Fraction of trajectory steps in basin
    basin_fraction = sum(1 for p in trajectory if p["in_basin"]) / len(trajectory)

    # Max loss encountered along trajectory
    max_loss = max(p["loss_at_position"] for p in trajectory)

    # Out-of-plane ratio: how much displacement is NOT in the 2D plane
    out_of_plane_ratio = final["residual_norm"] / final["displacement_norm"] if final["displacement_norm"] > 1e-8 else 0.0

    print(f"\n  Trajectory Summary:")
    print(f"    Total displacement:     {total_displacement:.4f}")
    print(f"    In-plane (2D) distance: {straight_line:.4f}")
    print(f"    Path length (2D):       {path_length:.4f}")
    print(f"    Curvature ratio:        {curvature_ratio:.3f} (1.0 = straight line)")
    print(f"    Out-of-plane ratio:     {out_of_plane_ratio:.3f}")
    print(f"    Basin fraction:         {basin_fraction:.1%}")
    print(f"    Max loss on path:       {max_loss:.4f}")

    # ── Save results ──
    summary = {
        "architecture": cfg["architecture"],
        "h0_persistence": h0_persistence,
        "h1_persistence": h1_persistence,
        "trajectory": trajectory,
        "metrics": {
            "total_displacement": total_displacement,
            "in_plane_distance": straight_line,
            "path_length_2d": path_length,
            "curvature_ratio": curvature_ratio,
            "out_of_plane_ratio": out_of_plane_ratio,
            "basin_fraction": basin_fraction,
            "max_loss_on_path": max_loss,
            "base_loss": base_loss,
            "final_loss": final["loss_at_position"],
        },
    }

    with open(os.path.join(disp_dir, "displacement_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save trajectory as numpy for easy plotting
    traj_array = np.array([(p["alpha"], p["beta"], p["loss_at_position"], p["step"])
                           for p in trajectory])
    np.save(os.path.join(disp_dir, "trajectory.npy"), traj_array)

    # ── Generate trajectory plot overlaid on landscape ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        if loss_grid is not None:
            # Plot loss landscape as contour
            A, B = np.meshgrid(alphas, betas, indexing="ij")
            levels = np.linspace(loss_grid.min(), min(loss_grid.max(), loss_grid.min() + 5), 30)
            cf = ax.contourf(A, B, loss_grid, levels=levels, cmap="viridis")
            plt.colorbar(cf, ax=ax, label="Loss")
            ax.contour(A, B, loss_grid, levels=levels, colors="white", linewidths=0.3, alpha=0.5)

        # Plot trajectory
        traj_alphas = [p["alpha"] for p in trajectory]
        traj_betas = [p["beta"] for p in trajectory]
        ax.plot(traj_alphas, traj_betas, "r-o", linewidth=2, markersize=5,
                label="Displacement trajectory", zorder=5)

        # Mark start and end
        ax.plot(0, 0, "w*", markersize=15, markeredgecolor="black", zorder=6, label="Task A minimum")
        ax.plot(traj_alphas[-1], traj_betas[-1], "rx", markersize=12, markeredgewidth=3,
                zorder=6, label=f"After {trajectory[-1]['step']} steps")

        # Annotate steps
        for p in trajectory[1:]:
            ax.annotate(f"{p['step']}", (p["alpha"], p["beta"]),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=7, color="white")

        ax.set_xlabel("Direction 1 (α)")
        ax.set_ylabel("Direction 2 (β)")
        ax.set_title(f"{cfg['architecture']} — Displacement Trajectory\n"
                     f"H0={h0_persistence:.1f}, H1={h1_persistence:.1f}, "
                     f"curvature={curvature_ratio:.2f}, basin_frac={basin_fraction:.0%}")
        ax.legend(loc="upper right")

        plt.tight_layout()
        plot_path = os.path.join(disp_dir, "trajectory_overlay.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\n  Trajectory plot saved to: {plot_path}")
    except ImportError:
        print("\n  matplotlib not available, skipping plot")

    print(f"\nPhase 2.5 complete. Results saved to: {disp_dir}/")


if __name__ == "__main__":
    main()
