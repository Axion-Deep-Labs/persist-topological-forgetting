"""
Elastic Weight Consolidation (EWC) — Kirkpatrick et al. (2017).

Two pure functions:
- compute_fisher: diagonal Fisher information from Task A data
- ewc_penalty: quadratic penalty anchoring weights to Task A solution
"""

import torch
import torch.nn.functional as F


def compute_fisher(model, dataloader, device, n_samples=1000):
    """Compute diagonal Fisher information matrix on Task A data.

    The Fisher diagonal F_i = E[(d log p(y|x) / d theta_i)^2] measures
    how important each parameter is for Task A predictions.

    Args:
        model: trained Task A model (will be set to eval mode)
        dataloader: Task A training data
        device: torch device
        n_samples: max samples to use (caps compute cost)

    Returns:
        fisher: dict mapping param name -> Fisher diagonal tensor (same shape as param)
        theta_star: dict mapping param name -> Task A parameter values (detached clone)
    """
    model.eval()

    # Store Task A parameters
    theta_star = {n: p.data.clone() for n, p in model.named_parameters()}

    # Accumulate squared gradients
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    total = 0

    for images, labels in dataloader:
        if total >= n_samples:
            break
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        model.zero_grad()
        outputs = model(images)
        # Use log-softmax + NLL for proper log-likelihood gradients
        log_probs = F.log_softmax(outputs, dim=1)
        # Sample from model's own predictions (true Fisher, not empirical)
        targets = torch.multinomial(torch.exp(log_probs), 1).squeeze(1)
        loss = F.nll_loss(log_probs, targets)
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2) * batch_size

        total += batch_size

    # Normalize by number of samples
    for n in fisher:
        fisher[n] /= total

    model.train()
    return fisher, theta_star


def ewc_penalty(model, fisher, theta_star, lambd):
    """Compute EWC penalty: lambda/2 * sum(F_i * (theta_i - theta_star_i)^2).

    Args:
        model: current model being trained on Task B
        fisher: diagonal Fisher from compute_fisher()
        theta_star: Task A parameter values from compute_fisher()
        lambd: penalty weight (typical: 100-10000)

    Returns:
        penalty: scalar tensor (add to Task B cross-entropy loss)
    """
    penalty = 0.0
    for n, p in model.named_parameters():
        if n in fisher:
            penalty += (fisher[n] * (p - theta_star[n]).pow(2)).sum()
    return 0.5 * lambd * penalty
