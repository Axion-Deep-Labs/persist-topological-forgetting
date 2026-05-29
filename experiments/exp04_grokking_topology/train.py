"""
EXP-04: Training loop with checkpoint saving at config-defined cadence.

Trains on modular addition, saves checkpoints at:
- Every standard_cadence steps (0 to fine_cadence_start)
- Every fine_cadence steps (fine_cadence_start to total_steps)

Logs train/test loss and accuracy at every checkpoint.
"""

import os
import json
import torch
import torch.nn as nn

from .dataset import get_dataloaders
from .model import build_model


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Compute loss and accuracy on a dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def get_checkpoint_steps(cfg):
    """Build sorted list of steps at which to save checkpoints.

    Two schedule forms are supported:

    1. Segmented (preferred, full study): ckpt_cfg["segments"] is a list of
       {start, end, cadence}. Each segment adds steps start, start+cadence, ...
       up to and including end. Segments should be contiguous (end of one =
       start of next) and their boundaries divisible by the neighbouring
       cadences so no gap opens at a seam. This is a FIXED schedule defined in
       global step-space — it does NOT use onset information, so there is no
       look-ahead leakage into the pre-onset window.

    2. Two-phase (legacy, pilot): standard_cadence until fine_cadence_start,
       then fine_cadence to total_steps.
    """
    ckpt_cfg = cfg["checkpoints"]
    total_steps = cfg["training"]["total_steps"]

    steps = set()
    if "segments" in ckpt_cfg:
        for seg in ckpt_cfg["segments"]:
            s = seg["start"]
            end = min(seg["end"], total_steps)
            cadence = seg["cadence"]
            while s <= end:
                steps.add(s)
                s += cadence
    else:
        standard = ckpt_cfg["standard_cadence"]
        fine = ckpt_cfg["fine_cadence"]
        fine_start = ckpt_cfg["fine_cadence_start"]
        s = 0
        while s <= min(fine_start, total_steps):
            steps.add(s)
            s += standard
        s = fine_start
        while s <= total_steps:
            steps.add(s)
            s += fine

    steps.add(0)
    steps.add(total_steps)
    return sorted(steps)


def train_seed(cfg, seed, output_dir, device):
    """Run one full training run for a given seed.

    Returns path to the metrics JSON file.
    """
    from experiments.shared.utils import set_seed
    set_seed(seed)

    run_dir = os.path.join(output_dir, f"seed_{seed}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data (split is fixed across seeds via internal generator)
    train_loader, test_loader = get_dataloaders(cfg)
    print(f"  Train: {len(train_loader.dataset)} pairs, Test: {len(test_loader.dataset)} pairs")

    # Model
    model = build_model(cfg, device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    tr_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["learning_rate"],
        weight_decay=tr_cfg["weight_decay"],
    )

    # Checkpoint schedule
    ckpt_steps = get_checkpoint_steps(cfg)
    ckpt_set = set(ckpt_steps)
    total_steps = tr_cfg["total_steps"]

    print(f"  Total steps: {total_steps}")
    print(f"  Checkpoints: {len(ckpt_steps)} ({ckpt_steps[0]} to {ckpt_steps[-1]})")

    # Training
    metrics_log = []
    step = 0
    model.train()

    # Evaluate at step 0
    train_loss, train_acc = evaluate(model, train_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    entry = {
        "step": 0,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    metrics_log.append(entry)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "step_000000.pt"))
    print(f"  [Step 0] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
          f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    while step < total_steps:
        for x, y in train_loader:
            if step >= total_steps:
                break
            x, y = x.to(device), y.to(device)
            model.train()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            step += 1

            if step in ckpt_set:
                train_loss, train_acc = evaluate(model, train_loader, criterion, device)
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                entry = {
                    "step": step,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
                metrics_log.append(entry)
                ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
                torch.save(model.state_dict(), ckpt_path)

                # Log progress
                grok_marker = " <-- GROKKING?" if (test_acc > 0.5 and train_acc > 0.99) else ""
                print(f"  [Step {step}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                      f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}{grok_marker}")

    # Save metrics
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"seed": seed, "config": cfg, "metrics": metrics_log}, f, indent=2)

    print(f"  Saved {len(metrics_log)} checkpoints to {ckpt_dir}")
    print(f"  Saved metrics to {metrics_path}")
    return metrics_path
