"""
Baseline geometry metrics for comparison with topological persistence.

These measure loss landscape properties using standard (non-topological) methods.
If topology predicts forgetting no better than these, TDA adds nothing new.
If topology outperforms them, we've found something genuinely novel.

Metrics:
    1. Hessian trace (Hutchinson estimator) — average curvature
    2. Max Hessian eigenvalue (power iteration) — sharpness
    3. Fisher Information trace — parameter importance
    4. Loss barrier height — max loss along random directions
"""

import torch
import torch.nn as nn
import numpy as np


@torch.no_grad()
def loss_barrier_height(model, dataloader, device, n_directions=10, step_size=0.1, n_steps=20):
    """Measure max loss increase along random directions from converged weights.

    This is the simplest baseline: how high does the loss get when you
    perturb the weights? If this predicts forgetting as well as topology,
    then H0 persistence is just a fancy loss-range proxy.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Get base loss
    base_loss = 0.0
    n_batches = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        base_loss += criterion(model(images), labels).item()
        n_batches += 1
    base_loss /= n_batches

    # Save original parameters
    orig_params = {n: p.clone() for n, p in model.named_parameters()}

    max_barrier = 0.0
    for _ in range(n_directions):
        # Random direction (filter-normalized like Phase 2)
        direction = {}
        for n, p in model.named_parameters():
            d = torch.randn_like(p)
            # Filter normalize: scale direction to match parameter norm
            if p.dim() >= 2:
                norm_p = p.norm()
                norm_d = d.norm()
                if norm_d > 0:
                    d = d * (norm_p / norm_d)
            direction[n] = d

        for step in range(1, n_steps + 1):
            # Perturb
            alpha = step * step_size
            for n, p in model.named_parameters():
                p.data = orig_params[n] + alpha * direction[n]

            # Evaluate
            loss = 0.0
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                loss += criterion(model(images), labels).item()
                break  # Single batch for speed
            barrier = loss - base_loss
            max_barrier = max(max_barrier, barrier)

        # Restore
        for n, p in model.named_parameters():
            p.data = orig_params[n].clone()

    return {"base_loss": base_loss, "max_barrier": max_barrier}


def hessian_trace_hutchinson(model, dataloader, device, criterion=None, n_samples=30):
    """Estimate Hessian trace via Hutchinson's stochastic estimator.

    Tr(H) = E[v^T H v] where v ~ Rademacher.
    Higher trace = higher average curvature = sharper minimum.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()

    # Get a batch for Hessian computation
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    traces = []
    for _ in range(n_samples):
        model.zero_grad()
        loss = criterion(model(images), labels)

        # First-order gradients
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # Rademacher vector
        v = [torch.randint_like(p, 0, 2) * 2.0 - 1.0 for p in model.parameters()]

        # Hessian-vector product: Hv = d/dp (grad^T v)
        grad_v = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(grad_v, model.parameters())

        # v^T H v
        trace_est = sum((vi * hvi).sum().item() for vi, hvi in zip(v, Hv))
        traces.append(trace_est)

    return {
        "hessian_trace_mean": float(np.mean(traces)),
        "hessian_trace_std": float(np.std(traces)),
    }


def max_hessian_eigenvalue(model, dataloader, device, criterion=None, n_iters=50):
    """Estimate largest Hessian eigenvalue via power iteration.

    This is the standard 'sharpness' metric (Keskar et al., 2017).
    Sharper minima (higher eigenvalue) are believed to generalize worse
    and may also forget faster.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()

    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    # Initialize random eigenvector
    v = [torch.randn_like(p) for p in model.parameters()]
    v_norm = sum((vi ** 2).sum() for vi in v).sqrt()
    v = [vi / v_norm for vi in v]

    eigenvalue = 0.0
    for _ in range(n_iters):
        model.zero_grad()
        loss = criterion(model(images), labels)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # Hv
        grad_v = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(grad_v, model.parameters())

        # Eigenvalue estimate: v^T H v
        eigenvalue = sum((vi * hvi).sum().item() for vi, hvi in zip(v, Hv))

        # Update v = Hv / ||Hv||
        v = [hvi.detach() for hvi in Hv]
        v_norm = sum((vi ** 2).sum() for vi in v).sqrt()
        if v_norm > 0:
            v = [vi / v_norm for vi in v]

    return {"max_eigenvalue": eigenvalue}


@torch.no_grad()
def fisher_information_trace(model, dataloader, device, criterion=None, n_batches=10):
    """Estimate Fisher Information trace (sum of squared gradients).

    F_ii = E[(d log p / d theta_i)^2]
    Higher Fisher trace = parameters are more "important" to the current task.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    fisher_diag = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    count = 0
    for images, labels in dataloader:
        if count >= n_batches:
            break
        images, labels = images.to(device), labels.to(device)

        # Need gradients for Fisher
        with torch.enable_grad():
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher_diag[n] += p.grad.data ** 2
        count += 1

    # Average
    fisher_trace = sum(fd.sum().item() for fd in fisher_diag.values()) / max(count, 1)

    return {"fisher_trace": fisher_trace}


def compute_all_baseline_metrics(model, dataloader, device):
    """Compute all baseline geometry metrics. Returns dict."""
    print("  Computing baseline metrics...")

    print("    Hessian trace (Hutchinson, 30 samples)...")
    ht = hessian_trace_hutchinson(model, dataloader, device, n_samples=30)

    print("    Max Hessian eigenvalue (power iteration, 50 iters)...")
    me = max_hessian_eigenvalue(model, dataloader, device, n_iters=50)

    print("    Fisher Information trace (10 batches)...")
    fi = fisher_information_trace(model, dataloader, device, n_batches=10)

    print("    Loss barrier height (10 directions)...")
    lb = loss_barrier_height(model, dataloader, device, n_directions=10)

    metrics = {**ht, **me, **fi, **lb}

    print(f"\n  Baseline Metrics Summary:")
    print(f"    Hessian trace:        {metrics['hessian_trace_mean']:.4f} +/- {metrics['hessian_trace_std']:.4f}")
    print(f"    Max eigenvalue:       {metrics['max_eigenvalue']:.4f}")
    print(f"    Fisher trace:         {metrics['fisher_trace']:.4f}")
    print(f"    Loss barrier height:  {metrics['max_barrier']:.4f}")

    return metrics
