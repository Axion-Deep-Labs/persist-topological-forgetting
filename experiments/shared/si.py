"""
Synaptic Intelligence (SI) — Zenke et al. (2017).

Two pure functions:
- compute_si_importance: accumulate per-parameter importance via online path integral
- si_penalty: quadratic penalty anchoring weights to Task A solution (same form as EWC)
"""

import torch
import torch.nn as nn


def compute_si_importance(model, dataloader, optimizer_cls, lr, device,
                          n_epochs=3, xi=0.1, optimizer_kwargs=None):
    """Compute SI importance scores by re-training on Task A from checkpoint.

    Tracks omega_k += |grad_k * delta_theta_k| per step, then consolidates:
        Omega_k = omega_k / (delta_theta_k^2 + xi)

    Since Phase 1 already ran without SI tracking, we approximate by re-training
    for n_epochs from the converged checkpoint. Uses the full training LR and
    multiple epochs to generate meaningful parameter displacement and accumulate
    sufficient path-integral signal.

    xi=0.1 is calibrated for post-hoc approximation. A large xi prevents
    extreme amplification from near-stationary parameters (common when
    re-training a converged model). Importance discrimination comes from
    omega_accum variation, not from the consolidation denominator.

    Args:
        model: trained Task A model (restored to original state after computation)
        dataloader: Task A training data
        optimizer_cls: optimizer class (e.g., torch.optim.SGD)
        lr: learning rate for re-training (use full training LR, not reduced)
        device: torch device
        n_epochs: epochs of Task A re-training to accumulate importance (default: 3)
        xi: damping constant to prevent division by zero (default: 0.1)
        optimizer_kwargs: extra kwargs for optimizer (e.g., momentum, weight_decay)

    Returns:
        omega: dict mapping param name -> importance tensor (same shape as param)
        theta_star: dict mapping param name -> Task A parameter values (detached clone)
    """
    # Save theta_star (Task A solution) before any re-training
    theta_star = {n: p.data.clone() for n, p in model.named_parameters()}

    # Save BatchNorm running statistics (not in named_parameters)
    saved_buffers = {n: b.clone() for n, b in model.named_buffers()}

    # Accumulator for path integral
    omega_accum = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    # Previous params for delta computation
    prev_params = {n: p.data.clone() for n, p in model.named_parameters()}

    criterion = nn.CrossEntropyLoss()
    opt_kwargs = {"lr": lr}
    if optimizer_kwargs:
        opt_kwargs.update(optimizer_kwargs)
    optimizer = optimizer_cls(model.parameters(), **opt_kwargs)

    model.train()
    for epoch in range(n_epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate: omega += |grad * delta_theta|
            for n, p in model.named_parameters():
                if p.grad is not None:
                    delta = p.data - prev_params[n]
                    omega_accum[n] += (p.grad.data * delta).abs()
                prev_params[n] = p.data.clone()

    # Consolidate: Omega_k = omega_k / (delta_theta_k^2 + xi)
    omega = {}
    for n, p in model.named_parameters():
        total_delta = p.data - theta_star[n]
        omega[n] = omega_accum[n] / (total_delta.pow(2) + xi)

    # Clip at 99th percentile to remove extreme outliers.
    # Even with xi=0.1, a few parameters may have disproportionate omega.
    # Use numpy for quantile on large tensors (torch.quantile has size limits).
    import numpy as np
    all_omega_np = torch.cat([v.flatten() for v in omega.values()]).cpu().numpy()
    clip_val = float(np.percentile(all_omega_np, 99))
    if clip_val > 0:
        for n in omega:
            omega[n] = omega[n].clamp(max=clip_val)

    # Restore model to theta_star (undo re-training perturbation)
    for n, p in model.named_parameters():
        p.data.copy_(theta_star[n])

    # Restore BatchNorm running statistics
    for n, b in model.named_buffers():
        if n in saved_buffers:
            b.copy_(saved_buffers[n])

    return omega, theta_star


def si_penalty(model, omega, theta_star, lambd):
    """Compute SI penalty: lambda/2 * sum(Omega_k * (theta_k - theta_star_k)^2).

    Structurally identical to EWC penalty. Uses path-integral importance
    instead of Fisher diagonal. The 0.5 factor matches EWC convention so
    that the same lambda value produces comparable penalty magnitude.

    Args:
        model: current model being trained on Task B
        omega: importance scores from compute_si_importance() (mean-normalized)
        theta_star: Task A parameter values from compute_si_importance()
        lambd: penalty weight (typical: 1000, comparable to EWC lambda)

    Returns:
        penalty: scalar tensor (add to Task B cross-entropy loss)
    """
    penalty = 0.0
    for n, p in model.named_parameters():
        if n in omega:
            penalty += (omega[n] * (p - theta_star[n]).pow(2)).sum()
    return 0.5 * lambd * penalty
