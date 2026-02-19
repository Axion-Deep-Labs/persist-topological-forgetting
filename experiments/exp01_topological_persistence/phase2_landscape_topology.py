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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.shared.datasets import get_split_dataset
from experiments.shared.models import get_model
from experiments.shared.utils import set_seed, load_config, load_checkpoint
from experiments.shared.baseline_metrics import compute_all_baseline_metrics


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


def preload_dataset_to_gpu(dataloader, device):
    """Load entire dataset into GPU memory as a single tensor pair.

    CIFAR-100 test set (Task A) is ~5000 images × 3 × 32 × 32 ≈ 20 MB.
    Eliminates CPU→GPU transfer overhead on every grid point evaluation.
    """
    all_images, all_labels = [], []
    for images, labels in dataloader:
        all_images.append(images)
        all_labels.append(labels)
    return torch.cat(all_images).to(device), torch.cat(all_labels).to(device)


@torch.no_grad()
def compute_loss_on_grid(model, images_gpu, labels_gpu, base_params, dir1, dir2, grid_range, steps, device):
    """Evaluate loss across a 2D grid of perturbations.

    Optimizations vs naive approach:
    - Pre-loaded GPU data: no CPU→GPU transfer per grid point
    - Mixed precision (AMP): ~2x throughput on tensor cores
    - Row-wise perturbation: set alpha once per row, only update beta per column
    """
    criterion = nn.CrossEntropyLoss()
    alphas = np.linspace(grid_range[0], grid_range[1], steps)
    betas = np.linspace(grid_range[0], grid_range[1], steps)
    beta_step = betas[1] - betas[0] if steps > 1 else 0.0

    loss_grid = np.zeros((steps, steps))
    total_samples = images_gpu.shape[0]

    total_points = steps * steps
    pbar = tqdm(total=total_points, desc="Sampling loss landscape")

    # Process in batches for AMP (full dataset at once if it fits)
    batch_size = 512

    for i, alpha in enumerate(alphas):
        # Set base + alpha * dir1 + betas[0] * dir2 at start of each row
        for param, base, d1, d2 in zip(model.parameters(), base_params, dir1, dir2):
            param.data.copy_(base + alpha * d1 + betas[0] * d2)

        for j in range(steps):
            if j > 0:
                # Incremental: only add beta_step * dir2 (instead of full recompute)
                for param, d2 in zip(model.parameters(), dir2):
                    param.data.add_(beta_step * d2)

            # Evaluate loss with mixed precision
            total_loss = 0.0
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                with torch.amp.autocast("cuda"):
                    outputs = model(images_gpu[start:end])
                    loss = criterion(outputs, labels_gpu[start:end])
                total_loss += loss.item() * (end - start)

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
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run identifier for multi-slice stability analysis. "
                             "Results saved as topology_summary_run{ID}.json")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed (for multi-seed runs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    landscape_cfg = cfg["landscape"]
    topo_cfg = cfg["topology"]
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    output_dir = cfg["output_dir"]

    # Multi-seed support: override seed and redirect output
    if args.seed is not None:
        cfg["seed"] = args.seed
        output_dir = os.path.join(output_dir, f"seed{args.seed}")
        cfg["output_dir"] = output_dir

    print(f"EXP-01 Phase 2: Loss Landscape Topology")
    print(f"  Device: {device}")
    print(f"  Seed: {cfg['seed']}")
    print(f"  Grid: {landscape_cfg['steps_per_direction']}x{landscape_cfg['steps_per_direction']}")
    print(f"  Range: {landscape_cfg['range']}")
    print(f"  Max homology dim: {topo_cfg['max_dimension']}")
    print()

    set_seed(cfg["seed"])

    # Load data (use test set for landscape evaluation — smaller, deterministic)
    data = get_split_dataset(cfg)
    _, test_loader = data.get_task_a(batch_size=256)

    # Load model
    model = get_model(cfg["architecture"], num_classes=cfg["num_classes_a"]).to(device)
    ckpt_path = args.checkpoint or os.path.join(output_dir, "checkpoints", "task_a_best.pt")
    epoch, accuracy = load_checkpoint(ckpt_path, model)
    print(f"  Loaded checkpoint: {ckpt_path}")
    print(f"  Epoch: {epoch}, Accuracy: {accuracy:.1%}")

    # Pre-load test set into GPU memory (eliminates CPU→GPU transfer per grid point)
    print("  Pre-loading test set to GPU...")
    images_gpu, labels_gpu = preload_dataset_to_gpu(test_loader, device)
    print(f"  Loaded {images_gpu.shape[0]} samples ({images_gpu.element_size() * images_gpu.nelement() / 1024**2:.1f} MB on GPU)")

    # Save base parameters
    base_params = [p.data.clone() for p in model.parameters()]

    # Reseed with a random seed for landscape directions (different slice each run)
    import random as _random
    landscape_seed = _random.randint(0, 2**31 - 1)
    set_seed(landscape_seed)
    print(f"\n  Landscape seed: {landscape_seed}")

    # Generate random directions (filter-normalized)
    print("Generating filter-normalized random directions...")
    dir1 = get_random_direction(model)
    dir2 = get_random_direction(model)

    # Compute loss landscape
    print(f"\nSampling loss landscape ({landscape_cfg['steps_per_direction']}x{landscape_cfg['steps_per_direction']} grid)...")
    t0 = time.time()
    loss_grid, alphas, betas = compute_loss_on_grid(
        model, images_gpu, labels_gpu, base_params, dir1, dir2,
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

    # Determine suffix for multi-slice runs
    run_suffix = f"_run{args.run_id}" if args.run_id else ""

    np.savez(
        os.path.join(topo_dir, f"loss_landscape{run_suffix}.npz"),
        loss_grid=loss_grid,
        alphas=alphas,
        betas=betas,
    )

    # Save random directions and base params for displacement analysis (Phase 2.5)
    torch.save(
        {"dir1": dir1, "dir2": dir2, "base_params": base_params},
        os.path.join(topo_dir, f"landscape_directions{run_suffix}.pt"),
    )
    print(f"  Saved landscape directions to landscape_directions{run_suffix}.pt")

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

    # Save persistence diagrams
    for dim, dgm in enumerate(diagrams):
        np.save(os.path.join(topo_dir, f"persistence_diagram_H{dim}{run_suffix}.npy"), dgm)

    # Restore model to original params for baseline metrics
    for param, base in zip(model.parameters(), base_params):
        param.data.copy_(base)

    # Free landscape data from GPU before heavy baseline computations
    del base_params, dir1, dir2, images_gpu, labels_gpu
    torch.cuda.empty_cache()

    # Compute baseline geometry metrics (Hessian, Fisher, sharpness)
    # Wrapped in try/except: second-order gradients can OOM on some architectures
    print("\n" + "=" * 50)
    baseline_metrics = {}
    try:
        baseline_metrics = compute_all_baseline_metrics(model, test_loader, device)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"  WARNING: Baseline metrics failed ({e})")
        print("  Topology results are still valid. Continuing...")
        torch.cuda.empty_cache()

    # Save summary (topology + baseline metrics)
    summary = {
        "checkpoint": ckpt_path,
        "checkpoint_epoch": epoch,
        "checkpoint_accuracy": accuracy,
        "landscape_seed": landscape_seed,
        "grid_steps": landscape_cfg["steps_per_direction"],
        "grid_range": landscape_cfg["range"],
        "filter_normalized": landscape_cfg["filter_normalize"],
        "loss_min": float(loss_grid.min()),
        "loss_max": float(loss_grid.max()),
        "loss_mean": float(loss_grid.mean()),
        "landscape_compute_time_s": landscape_time,
        "topology_compute_time_s": topo_time,
        **persistence_stats,
        **baseline_metrics,
    }
    summary_filename = f"topology_summary{run_suffix}.json"
    with open(os.path.join(topo_dir, summary_filename), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPhase 2 complete. Results saved to: {topo_dir}/{summary_filename}")
    if run_suffix:
        print(f"  (Multi-slice run ID: {args.run_id})")
    print(f"\nNext: Run phase3_sequential_forgetting.py to train on Task B and measure retention.")


if __name__ == "__main__":
    main()
