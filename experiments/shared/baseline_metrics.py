"""
Baseline geometry metrics for comparison with topological persistence.

These measure loss landscape properties using standard (non-topological) methods.
If topology predicts forgetting no better than these, TDA adds nothing new.
If topology outperforms them, we've found something genuinely novel.

Metrics:
    1. Hessian trace (Hutchinson estimator) — average curvature
    2. Max Hessian eigenvalue (power iteration) — sharpness
    3. Fisher Information trace — parameter importance
    4. Loss barrier height — max loss along random directions (normalized by sqrt(num_params))
"""

import torch
import torch.nn as nn
import numpy as np


def _count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def loss_barrier_height(model, dataloader, device, n_directions=10, step_size=0.1, n_steps=20):
    """Measure max loss increase along random directions from converged weights.

    Returns both raw and normalized barrier. The normalized barrier divides by
    sqrt(num_params) to make barriers comparable across architectures of
    different sizes.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    num_params = _count_parameters(model)

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

            # Evaluate (single batch for speed)
            loss = 0.0
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                loss += criterion(model(images), labels).item()
                break
            barrier = loss - base_loss
            # Clamp to prevent overflow — loss can explode for large models
            if np.isfinite(barrier) and barrier < 1e6:
                max_barrier = max(max_barrier, barrier)

        # Restore
        for n, p in model.named_parameters():
            p.data = orig_params[n].clone()

    # Normalize by sqrt(num_params) for cross-architecture comparability
    norm_factor = np.sqrt(num_params)
    normalized_barrier = max_barrier / norm_factor if norm_factor > 0 else max_barrier

    return {
        "base_loss": base_loss,
        "max_barrier": max_barrier,
        "max_barrier_normalized": normalized_barrier,
    }


def hessian_trace_hutchinson(model, dataloader, device, criterion=None, n_samples=10):
    """Estimate Hessian trace via Hutchinson's stochastic estimator.

    Tr(H) = E[v^T H v] where v ~ Rademacher.
    Higher trace = higher average curvature = sharper minimum.

    Uses fp32 accumulation and reduced samples (10) for stability on large models.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()

    # Get a batch for Hessian computation (cap at 64 to limit memory for create_graph)
    images, labels = next(iter(dataloader))
    images, labels = images[:64].to(device), labels[:64].to(device)

    traces = []
    for i in range(n_samples):
        model.zero_grad()
        loss = criterion(model(images), labels)

        # First-order gradients
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # Rademacher vector
        v = [torch.randint_like(p, 0, 2).float() * 2.0 - 1.0 for p in model.parameters()]

        # Hessian-vector product: Hv = d/dp (grad^T v)
        grad_v = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(grad_v, model.parameters())

        # v^T H v — accumulate in fp64 for numerical stability
        trace_est = sum((vi.double() * hvi.double()).sum().item() for vi, hvi in zip(v, Hv))
        traces.append(trace_est)

        # Clear computation graph each iteration to prevent memory buildup
        del grads, grad_v, Hv
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "hessian_trace_mean": float(np.mean(traces)),
        "hessian_trace_std": float(np.std(traces)),
    }


def max_hessian_eigenvalue(model, dataloader, device, criterion=None, n_iters=30):
    """Estimate largest Hessian eigenvalue via power iteration.

    This is the standard 'sharpness' metric (Keskar et al., 2017).
    Reduced to 30 iterations (from 50) — power iteration converges fast.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()

    # Cap at 64 samples to limit memory for create_graph
    images, labels = next(iter(dataloader))
    images, labels = images[:64].to(device), labels[:64].to(device)

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

        # Eigenvalue estimate: v^T H v (fp64 accumulation)
        eigenvalue = sum((vi.double() * hvi.double()).sum().item() for vi, hvi in zip(v, Hv))

        # Update v = Hv / ||Hv||
        v = [hvi.detach() for hvi in Hv]
        v_norm = sum((vi ** 2).sum() for vi in v).sqrt()
        if v_norm > 0:
            v = [vi / v_norm for vi in v]

        del grads, grad_v, Hv
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
    """Compute all baseline geometry metrics. Returns dict.

    Computes each metric independently so a failure in one doesn't block the rest.
    """
    print("  Computing baseline metrics...")
    metrics = {}

    # Clear GPU cache before heavy second-order computations
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for name, fn, kwargs in [
        ("Hessian trace (Hutchinson, 10 samples)", hessian_trace_hutchinson, {"n_samples": 10}),
        ("Max Hessian eigenvalue (power iteration, 30 iters)", max_hessian_eigenvalue, {"n_iters": 30}),
        ("Fisher Information trace (10 batches)", fisher_information_trace, {"n_batches": 10}),
        ("Loss barrier height (10 directions)", loss_barrier_height, {"n_directions": 10}),
    ]:
        print(f"    {name}...")
        try:
            result = fn(model, dataloader, device, **kwargs)
            metrics.update(result)
        except (RuntimeError, Exception) as e:
            print(f"    WARNING: {name} failed: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\n  Baseline Metrics Summary:")
    for key in ["hessian_trace_mean", "max_eigenvalue", "fisher_trace", "max_barrier", "max_barrier_normalized"]:
        val = metrics.get(key)
        if val is not None:
            print(f"    {key}: {val:.6f}")
        else:
            print(f"    {key}: FAILED")

    return metrics
