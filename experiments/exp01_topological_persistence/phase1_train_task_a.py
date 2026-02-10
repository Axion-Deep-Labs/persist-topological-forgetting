"""
EXP-01 Phase 1: Train ResNet-18 on Task A (Split-CIFAR-100, classes 0-49).

Trains to convergence, saves checkpoints for landscape analysis.

Usage:
    python -m experiments.exp01_topological_persistence.phase1_train_task_a \
        --config configs/exp01.yaml
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.shared.datasets import SplitCIFAR100
from experiments.shared.models import get_model
from experiments.shared.utils import set_seed, load_config, save_checkpoint, evaluate


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="EXP-01 Phase 1: Train on Task A")
    parser.add_argument("--config", type=str, default="configs/exp01.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    output_dir = cfg["output_dir"]

    print(f"EXP-01 Phase 1: Train on Task A")
    print(f"  Device: {device}")
    print(f"  Architecture: {cfg['architecture']}")
    print(f"  Output: {output_dir}")
    print()

    set_seed(cfg["seed"])

    # Data
    data = SplitCIFAR100(cfg["data_dir"], split_at=cfg["task_a_classes"][1])
    train_loader, test_loader = data.get_task_a(batch_size=train_cfg["batch_size"])
    print(f"  Task A: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")

    # Model
    model = get_model(cfg["architecture"], num_classes=cfg["num_classes_a"]).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Optimizer + Scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_cfg["lr"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=train_cfg["warmup_epochs"])
    cosine = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"] - train_cfg["warmup_epochs"])
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[train_cfg["warmup_epochs"]])

    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0.0
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'LR':>8} | {'Time':>6}")
    print("-" * 65)

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.1%} | {test_acc:7.1%} | {lr:8.6f} | {elapsed:5.1f}s")

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                model, optimizer, epoch, test_acc,
                os.path.join(output_dir, "checkpoints", "task_a_best.pt"),
            )

        # Save periodic checkpoints for topology analysis
        if epoch in [25, 50, 75, train_cfg["epochs"]]:
            save_checkpoint(
                model, optimizer, epoch, test_acc,
                os.path.join(output_dir, "checkpoints", f"task_a_epoch_{epoch}.pt"),
            )

    # Save final
    save_checkpoint(
        model, optimizer, train_cfg["epochs"], test_acc,
        os.path.join(output_dir, "checkpoints", "task_a_final.pt"),
    )

    print(f"\nPhase 1 complete. Best test accuracy: {best_acc:.1%}")
    print(f"Checkpoints saved to: {output_dir}/checkpoints/")
    print(f"\nNext: Run phase2_landscape_topology.py to compute persistent homology.")


if __name__ == "__main__":
    main()
