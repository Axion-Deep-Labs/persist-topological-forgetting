"""
EXP-01 Phase 2: Sample loss landscape & compute persistent homology.

1. Load converged Task A model
2. Generate two random filter-normalized directions (Li et al., 2018)
3. Evaluate loss on a 2D grid of perturbations
4. Compute persistent homology (H0, H1, H2) on the loss surface
5. Save persistence diagrams and Betti curves

Usage:
    python -m experiments.exp01_topological_persistence.phase2_landscape_topology \
        --config configs/exp01.yaml
"""

import argparse
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from clearml import Task as ClearMLTask

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.shared.datasets import SplitCIFAR100
from experiments.shared.models import get_model
from experiments.shared.utils import set_seed, load_config, load_checkpoint


def get_random_direction(model):
    """Generate a random direction in parameter space with filter normalization.

    Filter normalization (Li et al., 2018): for each filter/neuron,
    normalize the random direction to have the same norm as the corresponding
    model parameters. This ensures perturbations are proportional to weight scale.
    """
    direction = []
    for param in model.parameters():
        d = torch.randn_like(param)
        if param.dim() >= 2:
            # Filter normalize: scale each filter to match param norm
            for i in range(param.shape[0]):
                d_filter = d[i]
                p_filter = param[i]
                p_norm = p_filter.norm()
                d_norm = d_filter.norm()
                if d_norm > 0:
                    d[i] = d_filter * (p_norm / d_norm)
        else:
            # For 1D params (biases, BN), just match norm
            p_norm = param.norm()
            d_norm = d.norm()
            if d_norm > 0:
                d = d * (p_norm / d_norm)
        direction.append(d)
    return direction


def set_perturbed_params(model, base_params, dir1, dir2, alpha, beta):
    """Set model parameters to: base + alpha * dir1 + beta * dir2."""
    for param, base, d1, d2 in zip(model.parameters(), base_params, dir1, dir2):
        param.data.copy_(base + alpha * d1 + beta * d2)


@torch.no_grad()
def compute_loss_on_grid(model, dataloader, base_params, dir1, dir2, grid_range, steps, device):
    """Evaluate loss across a 2D grid of perturbations."""
    criterion = nn.CrossEntropyLoss()
    alphas = np.linspace(grid_range[0], grid_range[1], steps)
    betas = np.linspace(grid_range[0], grid_range[1], steps)

    loss_grid = np.zeros((steps, steps))

    total_points = steps * steps
    pbar = tqdm(total=total_points, desc="Sampling loss landscape")

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            set_perturbed_params(model, base_params, dir1, dir2, alpha, beta)

            total_loss = 0.0
            total_samples = 0
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

            loss_grid[i, j] = total_loss / total_samples
            pbar.update(1)

    pbar.close()

    # Restore original parameters
    for param, base in zip(model.parameters(), base_params):
        param.data.copy_(base)

    return loss_grid, alphas, betas


def compute_persistent_homology(loss_grid, maxdim=1):
    """Compute persistent homology on the loss surface using sublevel set filtration.

    Treats each grid point as a vertex with filtration value = loss.
    Uses a sparse distance matrix for memory efficiency.
    """
    import ripser
    from scipy.sparse import lil_matrix

    steps = loss_grid.shape[0]
    n = steps * steps
    loss_flat = loss_grid.flatten()

    # Build sparse adjacency (8-connected grid) — much more memory efficient
    dist_matrix = lil_matrix((n, n))

    for idx in range(n):
        i, j = idx // steps, idx % steps
        # 8-neighbors (4 cardinal + 4 diagonal)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < steps and 0 <= nj < steps:
                nidx = ni * steps + nj
                # Lower-star: edge appears at max of its two vertex filtration values
                dist_matrix[idx, nidx] = max(loss_flat[idx], loss_flat[nidx])

    dist_matrix = dist_matrix.tocsr()

    print(f"Computing persistent homology (H0-H{maxdim}, {n} points, sparse)...")
    result = ripser.ripser(dist_matrix, maxdim=maxdim, distance_matrix=True)

    diagrams = result["dgms"]
    return diagrams


def compute_total_persistence(diagrams):
    """Sum of lifetimes for each homology dimension."""
    totals = {}
    for dim, dgm in enumerate(diagrams):
        finite_mask = np.isfinite(dgm[:, 1])
        lifetimes = dgm[finite_mask, 1] - dgm[finite_mask, 0]
        totals[f"H{dim}"] = float(np.sum(lifetimes))
        totals[f"H{dim}_count"] = int(np.sum(finite_mask))
        totals[f"H{dim}_max_lifetime"] = float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0
    return totals


def main():
    parser = argparse.ArgumentParser(description="EXP-01 Phase 2: Landscape Topology")
    parser.add_argument("--config", type=str, default="configs/exp01.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path (default: task_a_best.pt)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    landscape_cfg = cfg["landscape"]
    topo_cfg = cfg["topology"]
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    output_dir = cfg["output_dir"]

    # ClearML experiment tracking
    task = ClearMLTask.init(
        project_name="EXP-01 Topological Persistence",
        task_name="Phase 2 — Landscape Topology",
        task_type=ClearMLTask.TaskTypes.data_processing,
    )
    task.connect(cfg, name="config")
    logger = task.get_logger()

    print(f"EXP-01 Phase 2: Loss Landscape Topology")
    print(f"  Device: {device}")
    print(f"  Grid: {landscape_cfg['steps_per_direction']}x{landscape_cfg['steps_per_direction']}")
    print(f"  Range: {landscape_cfg['range']}")
    print(f"  Max homology dim: {topo_cfg['max_dimension']}")
    print()

    set_seed(cfg["seed"])

    # Load data (use test set for landscape evaluation — smaller, deterministic)
    data = SplitCIFAR100(cfg["data_dir"], split_at=cfg["task_a_classes"][1])
    _, test_loader = data.get_task_a(batch_size=256)

    # Load model
    model = get_model(cfg["architecture"], num_classes=cfg["num_classes_a"]).to(device)
    ckpt_path = args.checkpoint or os.path.join(output_dir, "checkpoints", "task_a_best.pt")
    epoch, accuracy = load_checkpoint(ckpt_path, model)
    print(f"  Loaded checkpoint: {ckpt_path}")
    print(f"  Epoch: {epoch}, Accuracy: {accuracy:.1%}")

    # Save base parameters
    base_params = [p.data.clone() for p in model.parameters()]

    # Generate random directions (filter-normalized)
    print("\nGenerating filter-normalized random directions...")
    dir1 = get_random_direction(model)
    dir2 = get_random_direction(model)

    # Compute loss landscape
    print(f"\nSampling loss landscape ({landscape_cfg['steps_per_direction']}x{landscape_cfg['steps_per_direction']} grid)...")
    t0 = time.time()
    loss_grid, alphas, betas = compute_loss_on_grid(
        model, test_loader, base_params, dir1, dir2,
        grid_range=landscape_cfg["range"],
        steps=landscape_cfg["steps_per_direction"],
        device=device,
    )
    landscape_time = time.time() - t0
    print(f"  Landscape computed in {landscape_time:.1f}s")
    print(f"  Loss range: [{loss_grid.min():.4f}, {loss_grid.max():.4f}]")

    # Save landscape data
    topo_dir = os.path.join(output_dir, "topology")
    os.makedirs(topo_dir, exist_ok=True)
    np.savez(
        os.path.join(topo_dir, "loss_landscape.npz"),
        loss_grid=loss_grid,
        alphas=alphas,
        betas=betas,
    )

    # Compute persistent homology
    t0 = time.time()
    diagrams = compute_persistent_homology(loss_grid, maxdim=topo_cfg["max_dimension"])
    topo_time = time.time() - t0
    print(f"  Persistent homology computed in {topo_time:.1f}s")

    # Compute summary statistics
    persistence_stats = compute_total_persistence(diagrams)
    print(f"\n  Topological Summary:")
    for key, val in persistence_stats.items():
        print(f"    {key}: {val}")
        logger.report_single_value(name=key, value=val)

    # Log loss landscape as heatmap
    logger.report_surface(
        title="Loss Landscape",
        series="2D slice",
        matrix=loss_grid,
        iteration=0,
        xlabels=[f"{a:.2f}" for a in alphas[::10]],
        ylabels=[f"{b:.2f}" for b in betas[::10]],
    )

    # Save persistence diagrams
    for dim, dgm in enumerate(diagrams):
        np.save(os.path.join(topo_dir, f"persistence_diagram_H{dim}.npy"), dgm)

    # Save summary
    summary = {
        "checkpoint": ckpt_path,
        "checkpoint_epoch": epoch,
        "checkpoint_accuracy": accuracy,
        "grid_steps": landscape_cfg["steps_per_direction"],
        "grid_range": landscape_cfg["range"],
        "filter_normalized": landscape_cfg["filter_normalize"],
        "loss_min": float(loss_grid.min()),
        "loss_max": float(loss_grid.max()),
        "loss_mean": float(loss_grid.mean()),
        "landscape_compute_time_s": landscape_time,
        "topology_compute_time_s": topo_time,
        **persistence_stats,
    }
    with open(os.path.join(topo_dir, "topology_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPhase 2 complete. Results saved to: {topo_dir}/")
    print(f"\nNext: Run phase3_sequential_forgetting.py to train on Task B and measure retention.")


if __name__ == "__main__":
    main()
