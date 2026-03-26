"""
EXP-04: Baseline metrics for comparison with topological predictors.

Local geometry:
  - Commutator defect (Dohmatob et al., 2026)
  - Sharpness (trace of Hessian, Hutchinson estimator)

Global / spectral:
  - Spectral concentration (top eigenvalue ratio of weight SVD)

Simple controls:
  - Weight norm (L2)
  - Training loss curvature (second derivative of loss curve)
  - Generalization gap (train_acc - test_acc)
  - Validation loss slope
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend


# Context manager to disable efficient attention (needs create_graph support)
_MATH_ONLY = [SDPBackend.MATH]


def hessian_trace(model, dataloader, device, n_samples=10):
    """Estimate Hessian trace via Hutchinson's stochastic estimator.

    Tr(H) = E[v^T H v] where v ~ Rademacher.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    x_batch, y_batch = next(iter(dataloader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    traces = []
    for _ in range(n_samples):
        model.zero_grad()
        with sdpa_kernel(_MATH_ONLY):
            loss = criterion(model(x_batch), y_batch)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        v = [torch.randint_like(p, 0, 2).float() * 2.0 - 1.0 for p in model.parameters()]
        grad_v = sum((g * vi).sum() for g, vi in zip(grads, v))
        Hv = torch.autograd.grad(grad_v, model.parameters())

        trace_est = sum((vi.double() * hvi.double()).sum().item() for vi, hvi in zip(v, Hv))
        traces.append(trace_est)

        del grads, grad_v, Hv
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"sharpness": float(np.mean(traces))}


def commutator_defect(model, dataloader, device, n_pairs=5):
    """Estimate commutator defect: ||grad_A grad_B - grad_B grad_A||.

    Measures non-commutativity of gradient updates from different mini-batches.
    Higher defect = more curved loss surface = potential pre-grokking signal.

    Per Dohmatob et al. (2026).
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    batches = []
    for x, y in dataloader:
        batches.append((x.to(device), y.to(device)))
        if len(batches) >= 2 * n_pairs:
            break

    if len(batches) < 2:
        return {"commutator_defect": 0.0}

    defects = []
    for i in range(0, min(len(batches) - 1, 2 * n_pairs), 2):
        x_a, y_a = batches[i]
        x_b, y_b = batches[i + 1]

        # Grad from batch A
        model.zero_grad()
        with sdpa_kernel(_MATH_ONLY):
            loss_a = criterion(model(x_a), y_a)
        grad_a = torch.autograd.grad(loss_a, model.parameters(), create_graph=True)

        # Hessian-vector product: H * grad_b (approximates grad_a grad_b)
        model.zero_grad()
        with sdpa_kernel(_MATH_ONLY):
            loss_b = criterion(model(x_b), y_b)
        grad_b = torch.autograd.grad(loss_b, model.parameters(), create_graph=True)

        # grad_a^T H_b direction
        dot_ab = sum((ga * gb).sum() for ga, gb in zip(grad_a, grad_b))
        Hab = torch.autograd.grad(dot_ab, model.parameters(), retain_graph=True)

        # grad_b^T H_a direction
        dot_ba = sum((gb * ga).sum() for gb, ga in zip(grad_b, grad_a))
        Hba = torch.autograd.grad(dot_ba, model.parameters())

        # Commutator defect: ||Hab - Hba||
        defect = sum(((hab - hba) ** 2).sum().item() for hab, hba in zip(Hab, Hba))
        defects.append(np.sqrt(defect))

        del grad_a, grad_b, Hab, Hba
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"commutator_defect": float(np.mean(defects)) if defects else 0.0}


@torch.no_grad()
def spectral_concentration(model):
    """Top eigenvalue ratio of concatenated weight matrices.

    Ratio of largest singular value to sum of all singular values.
    Higher = more concentrated spectrum = lower effective rank.
    """
    all_svs = []
    for param in model.parameters():
        if param.dim() >= 2:
            # SVD of weight matrix
            svs = torch.linalg.svdvals(param.reshape(param.shape[0], -1).float())
            all_svs.append(svs.cpu().numpy())

    if not all_svs:
        return {"spectral_concentration": 0.0}

    all_svs = np.concatenate(all_svs)
    total = np.sum(all_svs)
    if total == 0:
        return {"spectral_concentration": 0.0}

    return {"spectral_concentration": float(np.max(all_svs) / total)}


@torch.no_grad()
def weight_norm_l2(model):
    """L2 norm of all parameters."""
    total = sum((p ** 2).sum().item() for p in model.parameters())
    return {"weight_norm_l2": float(np.sqrt(total))}


def training_loss_curvature(metrics_log, current_idx):
    """Second derivative of training loss curve at current checkpoint.

    Uses finite differences over the metrics log.
    Needs at least 3 points.
    """
    if current_idx < 2 or current_idx >= len(metrics_log):
        return {"training_loss_curvature": 0.0}

    l_prev = metrics_log[current_idx - 2]["train_loss"]
    l_curr = metrics_log[current_idx - 1]["train_loss"]
    l_next = metrics_log[current_idx]["train_loss"]

    # Second derivative (finite difference, uniform spacing assumed)
    curvature = l_next - 2 * l_curr + l_prev
    return {"training_loss_curvature": float(curvature)}


def generalization_gap(train_acc, test_acc):
    """Train accuracy minus test accuracy."""
    return {"generalization_gap": float(train_acc - test_acc)}


def validation_loss_slope(metrics_log, current_idx, window=3):
    """Slope of test loss over recent checkpoints.

    Linear regression over last `window` checkpoints.
    """
    if current_idx < window - 1:
        return {"validation_loss_slope": 0.0}

    start = current_idx - window + 1
    losses = [metrics_log[i]["test_loss"] for i in range(start, current_idx + 1)]
    steps = [metrics_log[i]["step"] for i in range(start, current_idx + 1)]

    # Simple linear regression slope
    x = np.array(steps, dtype=np.float64)
    y = np.array(losses, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return {"validation_loss_slope": 0.0}

    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    return {"validation_loss_slope": float(slope)}


def compute_all_baselines(model, dataloader, device, metrics_log, current_idx):
    """Compute all baseline metrics for one checkpoint.

    Each metric is computed independently so a failure in one doesn't block others.
    """
    results = {}

    for name, fn in [
        ("sharpness", lambda: hessian_trace(model, dataloader, device)),
        ("commutator_defect", lambda: commutator_defect(model, dataloader, device)),
        ("spectral_concentration", lambda: spectral_concentration(model)),
        ("weight_norm_l2", lambda: weight_norm_l2(model)),
        ("training_loss_curvature", lambda: training_loss_curvature(metrics_log, current_idx)),
        ("generalization_gap", lambda: generalization_gap(
            metrics_log[current_idx]["train_acc"],
            metrics_log[current_idx]["test_acc"])),
        ("validation_loss_slope", lambda: validation_loss_slope(metrics_log, current_idx)),
    ]:
        try:
            results.update(fn())
        except Exception as e:
            print(f"    WARNING: {name} failed: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results
