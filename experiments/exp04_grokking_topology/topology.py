"""
EXP-04: Loss landscape topology computation.

Computes persistent homology on 2D loss landscape slices at each checkpoint.
Uses the same method as PERSIST (EXP-01): filter-normalized random directions,
sublevel set filtration via Ripser.

This is a standardized slice-based topology proxy. We measure topology summaries
extracted from standardized 2D projections and test whether they contain
predictive signal. This is not a claim about the full loss landscape topology.

Computation: per slice first, then average across slices.
"""

import gc
import os
import json
import numpy as np
import torch
import torch.nn as nn
import ripser
from scipy.sparse import lil_matrix

from .dataset import get_dataloaders
from .model import build_model


def get_random_direction(model):
    """Generate a filter-normalized random direction in parameter space.

    Li et al. (2018): for each filter/neuron, normalize the random direction
    to have the same norm as the corresponding model parameters.
    """
    direction = []
    for param in model.parameters():
        d = torch.randn_like(param)
        if param.dim() >= 2:
            for i in range(param.shape[0]):
                d_filter = d[i]
                p_filter = param[i]
                p_norm = p_filter.norm()
                d_norm = d_filter.norm()
                if d_norm > 0:
                    d[i] = d_filter * (p_norm / d_norm)
        else:
            p_norm = param.norm()
            d_norm = d.norm()
            if d_norm > 0:
                d = d * (p_norm / d_norm)
        direction.append(d)
    return direction


@torch.no_grad()
def compute_loss_grid(model, dataloader, base_params, dir1, dir2, grid_size, device):
    """Evaluate loss on a 2D grid of perturbations.

    Grid range fixed at [-1, 1] in both directions (filter normalization
    ensures this is scale-appropriate).
    """
    criterion = nn.CrossEntropyLoss()
    grid_range = (-1.0, 1.0)
    alphas = np.linspace(grid_range[0], grid_range[1], grid_size)
    betas = np.linspace(grid_range[0], grid_range[1], grid_size)

    # Preload dataset to device (small for modular arithmetic)
    all_x, all_y = [], []
    for x, y in dataloader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x).to(device)
    all_y = torch.cat(all_y).to(device)

    loss_grid = np.zeros((grid_size, grid_size))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Perturb parameters
            for param, base, d1, d2 in zip(model.parameters(), base_params, dir1, dir2):
                param.data.copy_(base + alpha * d1 + beta * d2)

            logits = model(all_x)
            loss_grid[i, j] = criterion(logits, all_y).item()

    # Restore base parameters
    for param, base in zip(model.parameters(), base_params):
        param.data.copy_(base)

    return loss_grid


def compute_persistent_homology(loss_grid, maxdim=1):
    """Compute PH on loss surface via sublevel set filtration (Ripser).

    8-connected grid, lower-star filtration.
    """
    steps = loss_grid.shape[0]
    n = steps * steps
    loss_flat = loss_grid.flatten()

    dist_matrix = lil_matrix((n, n))
    for idx in range(n):
        i, j = idx // steps, idx % steps
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < steps and 0 <= nj < steps:
                nidx = ni * steps + nj
                dist_matrix[idx, nidx] = max(loss_flat[idx], loss_flat[nidx])

    result = ripser.ripser(dist_matrix.tocsr(), maxdim=maxdim, distance_matrix=True)
    return result["dgms"]


def extract_stats(diagrams):
    """Extract topology statistics from persistence diagrams.

    h{dim}_effective_feature_count uses the inverse participation ratio
    (sum lifetimes)^2 / sum(lifetimes^2). Uniform persistences -> n; single
    dominant persistence -> 1. Replaces an earlier h0_significant_count that
    was structurally pinned near n/2 on grid filtrations.
    """
    stats = {}
    for dim, dgm in enumerate(diagrams):
        finite_mask = np.isfinite(dgm[:, 1])
        lifetimes = dgm[finite_mask, 1] - dgm[finite_mask, 0]

        prefix = f"h{dim}"
        stats[f"{prefix}_feature_count"] = int(np.sum(finite_mask))
        stats[f"{prefix}_total_persistence"] = float(np.sum(lifetimes))
        stats[f"{prefix}_max_persistence"] = float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0

        if len(lifetimes) > 0:
            total = float(np.sum(lifetimes))
            sq = float(np.sum(lifetimes ** 2))
            stats[f"{prefix}_effective_feature_count"] = (total * total) / sq if sq > 0 else 0.0
            stats[f"{prefix}_median_persistence"] = float(np.median(lifetimes))
        else:
            stats[f"{prefix}_effective_feature_count"] = 0.0
            stats[f"{prefix}_median_persistence"] = 0.0

        # Persistence entropy
        if len(lifetimes) > 0 and np.sum(lifetimes) > 0:
            probs = lifetimes / np.sum(lifetimes)
            probs = probs[probs > 0]
            stats[f"{prefix}_persistence_entropy"] = float(-np.sum(probs * np.log(probs)))
        else:
            stats[f"{prefix}_persistence_entropy"] = 0.0

    return stats


def compute_topology_at_checkpoint(model, ckpt_path, dataloader, cfg, device):
    """Compute PH for a single checkpoint across multiple slices.

    Returns averaged stats and per-slice stats (for variance reporting).
    """
    topo_cfg = cfg["topology"]
    n_slices = topo_cfg["n_slices"]
    grid_size = topo_cfg["grid_size"]

    # Load checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    base_params = [p.data.clone() for p in model.parameters()]

    per_slice_stats = []
    for s in range(n_slices):
        # Different random directions for each slice
        dir1 = get_random_direction(model)
        dir2 = get_random_direction(model)

        loss_grid = compute_loss_grid(model, dataloader, base_params, dir1, dir2, grid_size, device)
        diagrams = compute_persistent_homology(loss_grid)
        stats = extract_stats(diagrams)
        per_slice_stats.append(stats)

        # Free memory between slices to prevent OOM on long runs
        del dir1, dir2, loss_grid, diagrams
        gc.collect()

    # Average across slices
    all_keys = per_slice_stats[0].keys()
    averaged = {}
    slice_variance = {}
    for key in all_keys:
        values = [s[key] for s in per_slice_stats]
        averaged[key] = float(np.mean(values))
        slice_variance[f"{key}_slice_var"] = float(np.var(values))

    # Restore base params
    for param, base in zip(model.parameters(), base_params):
        param.data.copy_(base)

    return {
        "averaged": averaged,
        "slice_variance": slice_variance,
        "per_slice": per_slice_stats,
    }
