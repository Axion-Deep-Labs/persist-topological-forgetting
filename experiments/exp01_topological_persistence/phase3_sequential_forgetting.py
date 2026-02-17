"""
EXP-01 Phase 3: Train on Task B, measure Task A forgetting at intervals.

Loads the converged Task A model, trains sequentially on Task B (classes 50-99),
and evaluates Task A test accuracy at configured intervals.

Usage:
    python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting \
        --config configs/exp01.yaml
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.shared.datasets import SplitCIFAR100
from experiments.shared.models import get_model
from experiments.shared.utils import set_seed, load_config, load_checkpoint, save_checkpoint, evaluate


def main():
    parser = argparse.ArgumentParser(description="EXP-01 Phase 3: Sequential Forgetting")
    parser.add_argument("--config", type=str, default="configs/exp01.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    forget_cfg = cfg["forgetting"]
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    output_dir = cfg["output_dir"]

    print("EXP-01 Phase 3: Sequential Forgetting Measurement")
    print(f"  Device: {device}")
    print(f"  Eval steps: {forget_cfg['eval_steps']}")
    print()

    set_seed(cfg["seed"])

    # Data
    data = SplitCIFAR100(cfg["data_dir"], split_at=cfg["task_a_classes"][1])
    _, task_a_test = data.get_task_a(batch_size=256)
    task_b_train, task_b_test = data.get_task_b(batch_size=train_cfg["batch_size"])

    print(f"  Task A test: {len(task_a_test.dataset)} samples")
    print(f"  Task B train: {len(task_b_train.dataset)}, test: {len(task_b_test.dataset)} samples")

    # Load Task A model
    # We need a model with num_classes = total classes (100) for sequential learning
    # Strategy: expand the final FC layer, keeping Task A weights frozen in the features
    model = get_model(cfg["architecture"], num_classes=cfg["num_classes_a"]).to(device)
    ckpt_path = args.checkpoint or os.path.join(output_dir, "checkpoints", "task_a_best.pt")
    _, task_a_acc = load_checkpoint(ckpt_path, model)
    print(f"  Task A model accuracy: {task_a_acc:.1%}")

    # Expand classifier for Task B classes (handle both .fc and .head)
    fc_attr = "fc" if hasattr(model, "fc") else "head"
    old_fc = getattr(model, fc_attr)
    new_fc = nn.Linear(old_fc.in_features, cfg["num_classes_a"] + cfg["num_classes_b"]).to(device)
    with torch.no_grad():
        new_fc.weight[:cfg["num_classes_a"]] = old_fc.weight
        new_fc.bias[:cfg["num_classes_a"]] = old_fc.bias
    setattr(model, fc_attr, new_fc)

    # Optimizer for Task B (train all parameters — this is the naive sequential baseline)
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_cfg["lr"] * 0.1,  # Lower LR for fine-tuning
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    # Remap Task B labels to [50, 100) range for the expanded classifier
    # We need to override the dataloader to shift labels
    eval_steps = set(forget_cfg["eval_steps"])
    max_steps = max(forget_cfg["eval_steps"])

    # Forgetting curve
    forgetting_curve = []
    step = 0

    # Evaluate initial Task A accuracy (should match loaded checkpoint)
    initial_a_acc = evaluate(model, task_a_test, device)
    forgetting_curve.append({"step": 0, "task_a_acc": initial_a_acc, "task_b_acc": 0.0})
    print(f"\n  Step 0: Task A acc = {initial_a_acc:.1%}")

    # Training loop on Task B
    forget_dir = os.path.join(output_dir, "forgetting")
    os.makedirs(forget_dir, exist_ok=True)

    print(f"\n{'Step':>6} | {'Task A Acc':>9} | {'Task B Acc':>9} | {'Forgetting':>10}")
    print("-" * 48)

    model.train()
    epoch = 0
    while step < max_steps:
        epoch += 1
        for images, labels in task_b_train:
            images = images.to(device)
            # Shift labels to [50, 100) range
            labels = (labels + cfg["num_classes_a"]).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            step += 1

            if step in eval_steps:
                task_a_acc = evaluate(model, task_a_test, device)
                task_b_acc = evaluate_shifted(model, task_b_test, device, cfg["num_classes_a"])
                forgetting = initial_a_acc - task_a_acc

                forgetting_curve.append({
                    "step": step,
                    "task_a_acc": task_a_acc,
                    "task_b_acc": task_b_acc,
                    "forgetting": forgetting,
                })
                print(f"{step:6d} | {task_a_acc:8.1%} | {task_b_acc:8.1%} | {forgetting:9.1%}")

                if forget_cfg.get("save_checkpoints"):
                    save_checkpoint(
                        model, optimizer, step, task_a_acc,
                        os.path.join(forget_dir, f"step_{step}.pt"),
                    )

                model.train()

            if step >= max_steps:
                break

    # Save forgetting curve
    with open(os.path.join(forget_dir, "forgetting_curve.json"), "w") as f:
        json.dump({
            "initial_task_a_acc": initial_a_acc,
            "checkpoint": ckpt_path,
            "curve": forgetting_curve,
        }, f, indent=2)

    print(f"\nPhase 3 complete. Forgetting curve saved to: {forget_dir}/")
    print(f"\nNext: Run phase4_correlation.py to correlate topology with retention.")


@torch.no_grad()
def evaluate_shifted(model, dataloader, device, offset):
    """Evaluate with labels shifted by offset."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = (labels + offset).to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return correct / total


if __name__ == "__main__":
    main()
