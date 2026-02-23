"""
EXP-01 Phase 3: Train on Task B, measure Task A forgetting at intervals.

Loads the converged Task A model, trains sequentially on Task B (classes 50-99),
and evaluates Task A test accuracy at configured intervals.

Supports:
  - Naive sequential training (default)
  - EWC regularization (--ewc): anchors weights to Task A solution
  - Cosine LR schedule (--lr-schedule cosine): decays LR during Task B

Usage:
    # Naive baseline
    python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting \
        --config configs/exp01.yaml

    # With EWC
    python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting \
        --config configs/exp01.yaml --ewc --ewc-lambda 1000

    # With cosine LR
    python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting \
        --config configs/exp01.yaml --lr-schedule cosine
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

from experiments.shared.datasets import get_split_dataset
from experiments.shared.models import get_model
from experiments.shared.utils import set_seed, load_config, load_checkpoint, save_checkpoint, evaluate


def main():
    parser = argparse.ArgumentParser(description="EXP-01 Phase 3: Sequential Forgetting")
    parser.add_argument("--config", type=str, default="configs/exp01.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed (for multi-seed runs)")
    parser.add_argument("--ewc", action="store_true",
                        help="Enable EWC regularization during Task B training")
    parser.add_argument("--ewc-lambda", type=float, default=1000.0,
                        help="EWC penalty weight (default: 1000)")
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                        help="LR schedule for Task B training (default: constant)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    forget_cfg = cfg["forgetting"]
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    output_dir = cfg["output_dir"]

    # Multi-seed support: override seed and redirect output
    if args.seed is not None:
        cfg["seed"] = args.seed
        output_dir = os.path.join(output_dir, f"seed{args.seed}")
        cfg["output_dir"] = output_dir

    # Determine variant label for output directory
    variant_parts = []
    if args.ewc:
        variant_parts.append("ewc")
    if args.lr_schedule == "cosine":
        variant_parts.append("cosine")
    variant_label = "_".join(variant_parts) if variant_parts else None

    print("EXP-01 Phase 3: Sequential Forgetting Measurement")
    print(f"  Device: {device}")
    print(f"  Seed: {cfg['seed']}")
    print(f"  Eval steps: {forget_cfg['eval_steps']}")
    if args.ewc:
        print(f"  EWC: enabled (lambda={args.ewc_lambda})")
    if args.lr_schedule != "constant":
        print(f"  LR schedule: {args.lr_schedule}")
    if variant_label:
        print(f"  Variant: {variant_label}")
    print()

    set_seed(cfg["seed"])

    # Data
    data = get_split_dataset(cfg)
    task_a_train, task_a_test = data.get_task_a(batch_size=256)
    task_b_train, task_b_test = data.get_task_b(batch_size=train_cfg["batch_size"])

    print(f"  Task A test: {len(task_a_test.dataset)} samples")
    print(f"  Task B train: {len(task_b_train.dataset)}, test: {len(task_b_test.dataset)} samples")

    # Load Task A model
    model = get_model(cfg["architecture"], num_classes=cfg["num_classes_a"]).to(device)
    ckpt_path = args.checkpoint or os.path.join(output_dir, "checkpoints", "task_a_best.pt")
    _, task_a_acc = load_checkpoint(ckpt_path, model)
    print(f"  Task A model accuracy: {task_a_acc:.1%}")

    # Compute Fisher information for EWC before expanding the classifier
    fisher = None
    theta_star = None
    if args.ewc:
        from experiments.shared.ewc import compute_fisher
        print("  Computing Fisher information on Task A data...")
        fisher, theta_star = compute_fisher(model, task_a_train, device, n_samples=1000)
        print(f"  Fisher computed ({len(fisher)} parameter groups)")

    # Expand classifier for Task B classes (handle both .fc and .head)
    # Also handle Sequential classifiers (e.g., MobileNet-V3-Small)
    fc_attr = "fc" if hasattr(model, "fc") else "head"
    old_fc = getattr(model, fc_attr)

    if isinstance(old_fc, nn.Sequential):
        # Find the last Linear layer in the Sequential
        last_linear_idx = None
        for idx, layer in enumerate(old_fc):
            if isinstance(layer, nn.Linear):
                last_linear_idx = idx
        old_linear = old_fc[last_linear_idx]
        new_linear = nn.Linear(old_linear.in_features, cfg["num_classes_a"] + cfg["num_classes_b"]).to(device)
        with torch.no_grad():
            new_linear.weight[:cfg["num_classes_a"]] = old_linear.weight
            new_linear.bias[:cfg["num_classes_a"]] = old_linear.bias
        old_fc[last_linear_idx] = new_linear
    else:
        new_fc = nn.Linear(old_fc.in_features, cfg["num_classes_a"] + cfg["num_classes_b"]).to(device)
        with torch.no_grad():
            new_fc.weight[:cfg["num_classes_a"]] = old_fc.weight
            new_fc.bias[:cfg["num_classes_a"]] = old_fc.bias
        setattr(model, fc_attr, new_fc)

    # Update Fisher and theta_star for expanded classifier if EWC is active
    # New classifier params have zero Fisher (unknown to Task A), so EWC
    # won't penalize them, which is the correct behavior.
    if args.ewc:
        from experiments.shared.ewc import ewc_penalty
        # Update Fisher and theta_star for expanded classifier:
        # - New params (not in fisher): add zero entries
        # - Resized params (shape mismatch): pad Fisher with zeros, pad theta_star
        #   with current values. EWC only penalizes the original Task A portion.
        for n, p in model.named_parameters():
            if n not in fisher:
                fisher[n] = torch.zeros_like(p)
                theta_star[n] = p.data.clone()
            elif fisher[n].shape != p.shape:
                old_fisher = fisher[n]
                old_theta = theta_star[n]
                new_fisher = torch.zeros_like(p)
                new_theta = p.data.clone()
                # Copy old values into the leading slice (Task A classes)
                slices = tuple(slice(0, s) for s in old_fisher.shape)
                new_fisher[slices] = old_fisher
                new_theta[slices] = old_theta
                fisher[n] = new_fisher
                theta_star[n] = new_theta

    # Optimizer for Task B (train all parameters)
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_cfg["lr"] * 0.1,  # Lower LR for fine-tuning
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    eval_steps = set(forget_cfg["eval_steps"])
    max_steps = max(forget_cfg["eval_steps"])

    # LR scheduler
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

    # Forgetting curve
    forgetting_curve = []
    step = 0

    # Evaluate initial Task A accuracy (should match loaded checkpoint)
    initial_a_acc = evaluate(model, task_a_test, device)
    forgetting_curve.append({"step": 0, "task_a_acc": initial_a_acc, "task_b_acc": 0.0})
    print(f"\n  Step 0: Task A acc = {initial_a_acc:.1%}")

    # Output directory (variant-specific)
    if variant_label:
        forget_dir = os.path.join(output_dir, f"forgetting_{variant_label}")
    else:
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

            # Add EWC penalty
            if args.ewc:
                loss = loss + ewc_penalty(model, fisher, theta_star, args.ewc_lambda)

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

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

    # Check Task B learning quality
    final_point = forgetting_curve[-1]
    final_b_acc = final_point.get("task_b_acc", 0.0)
    num_b_classes = cfg["num_classes_b"]
    chance_level = 1.0 / num_b_classes
    if final_b_acc < chance_level * 2:
        print(f"\n  WARNING: Task B barely learned (final acc = {final_b_acc:.1%}, chance = {chance_level:.1%})")
        print(f"  Retention metric may not reflect true forgetting resistance.")

    # Save forgetting curve
    metadata = {
        "initial_task_a_acc": initial_a_acc,
        "checkpoint": ckpt_path,
        "final_task_b_acc": final_b_acc,
        "curve": forgetting_curve,
    }
    if args.ewc:
        metadata["ewc"] = True
        metadata["ewc_lambda"] = args.ewc_lambda
    if args.lr_schedule != "constant":
        metadata["lr_schedule"] = args.lr_schedule

    with open(os.path.join(forget_dir, "forgetting_curve.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nPhase 3 complete. Forgetting curve saved to: {forget_dir}/")
    print(f"  Final Task B accuracy: {final_b_acc:.1%}")
    if variant_label:
        print(f"  Variant: {variant_label}")
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
