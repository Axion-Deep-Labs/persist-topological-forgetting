"""
EXP-01 Phase 3: Train on Task B, measure Task A forgetting at intervals.

Loads the converged Task A model, trains sequentially on Task B (classes 50-99),
and evaluates Task A test accuracy at configured intervals.

Supports:
  - Naive sequential training (default)
  - EWC regularization (--ewc): anchors weights to Task A solution
  - SI regularization (--si): path-integral importance (Zenke et al., 2017)
  - Cosine LR schedule (--lr-schedule cosine): decays LR during Task B

Note: --ewc and --si are mutually exclusive.

Usage:
    # Naive baseline
    python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting \
        --config configs/exp01.yaml

    # With EWC
    python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting \
        --config configs/exp01.yaml --ewc --ewc-lambda 1000

    # With SI
    python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting \
        --config configs/exp01.yaml --si --si-lambda 1000

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

from experiments.shared.datasets import get_split_dataset, get_cross_dataset_task_b
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
    parser.add_argument("--si", action="store_true",
                        help="Enable SI regularization during Task B training")
    parser.add_argument("--si-lambda", type=float, default=1000.0,
                        help="SI penalty weight (default: 1000, comparable to EWC lambda)")
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                        help="LR schedule for Task B training (default: constant)")
    parser.add_argument("--si-batch-size", type=int, default=256,
                        help="Batch size for SI/EWC importance computation (reduce for large models, default: 256)")
    parser.add_argument("--cross-dataset", type=str, default=None,
                        choices=["cifar100", "cub200", "resisc45"],
                        help="Use a different dataset for Task B (cross-dataset forgetting)")
    parser.add_argument("--task-a-dir", type=str, default=None,
                        help="Path to original Task A result dir with checkpoints/ (required with --cross-dataset)")
    parser.add_argument("--output-dir-override", type=str, default=None,
                        help="Override output directory (for cross-dataset result dirs)")
    args = parser.parse_args()

    if args.ewc and args.si:
        parser.error("--ewc and --si are mutually exclusive; choose one regularization method")
    if args.cross_dataset and not args.task_a_dir:
        parser.error("--task-a-dir is required with --cross-dataset")

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    forget_cfg = cfg["forgetting"]
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    output_dir = cfg["output_dir"]

    # Output directory override (for cross-dataset)
    if args.output_dir_override:
        output_dir = args.output_dir_override
        cfg["output_dir"] = output_dir

    # Multi-seed support: override seed and redirect output
    if args.seed is not None:
        cfg["seed"] = args.seed
        output_dir = os.path.join(output_dir, f"seed{args.seed}")
        cfg["output_dir"] = output_dir

    # Determine variant label for output directory
    variant_parts = []
    if args.ewc:
        variant_parts.append("ewc")
    if args.si:
        variant_parts.append("si")
    if args.lr_schedule == "cosine":
        variant_parts.append("cosine")
    variant_label = "_".join(variant_parts) if variant_parts else None

    print("EXP-01 Phase 3: Sequential Forgetting Measurement")
    print(f"  Device: {device}")
    print(f"  Seed: {cfg['seed']}")
    print(f"  Eval steps: {forget_cfg['eval_steps']}")
    if args.ewc:
        print(f"  EWC: enabled (lambda={args.ewc_lambda})")
    if args.si:
        print(f"  SI: enabled (lambda={args.si_lambda})")
    if args.lr_schedule != "constant":
        print(f"  LR schedule: {args.lr_schedule}")
    if variant_label:
        print(f"  Variant: {variant_label}")
    print()

    set_seed(cfg["seed"])

    # Data
    if args.cross_dataset:
        # Cross-dataset: Task A from original config, Task B from different dataset
        task_a_dataset = cfg.get("dataset", "cifar100")
        print(f"  Cross-dataset mode: Task A = {task_a_dataset}, Task B = {args.cross_dataset}")

        # Task A data (for evaluation and Fisher/SI computation)
        data = get_split_dataset(cfg)
        task_a_train, task_a_test = data.get_task_a(batch_size=args.si_batch_size)

        # Task B data (full cross-dataset, all classes)
        task_b_train, task_b_test, xd_num_classes = get_cross_dataset_task_b(
            args.cross_dataset, cfg["data_dir"],
            batch_size=train_cfg["batch_size"],
            seed=cfg.get("seed", 42),
        )
        cfg["num_classes_b"] = xd_num_classes
        print(f"  Task B ({args.cross_dataset}): {xd_num_classes} classes")
    else:
        # Standard within-dataset split
        data = get_split_dataset(cfg)
        task_a_train, task_a_test = data.get_task_a(batch_size=args.si_batch_size)
        task_b_train, task_b_test = data.get_task_b(batch_size=train_cfg["batch_size"])

    print(f"  Task A test: {len(task_a_test.dataset)} samples")
    print(f"  Task B train: {len(task_b_train.dataset)}, test: {len(task_b_test.dataset)} samples")

    # Load Task A model
    model = get_model(cfg["architecture"], num_classes=cfg["num_classes_a"]).to(device)
    if args.cross_dataset:
        ckpt_path = args.checkpoint or os.path.join(args.task_a_dir, "checkpoints", "task_a_best.pt")
    else:
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

        # Diagnostic: log Fisher scale for lambda calibration comparison
        all_fisher = torch.cat([v.flatten() for v in fisher.values()])
        print(f"  Fisher stats: mean={all_fisher.mean().item():.4e}, "
              f"median={all_fisher.median().item():.4e}, "
              f"max={all_fisher.max().item():.4e}")

    # Compute SI importance before expanding the classifier
    si_omega = None
    si_theta_star = None
    if args.si:
        from experiments.shared.si import compute_si_importance
        print("  Computing SI importance on Task A data (3 epoch re-train)...")
        si_opt_name = train_cfg.get("optimizer", "sgd")
        si_opt_cls = optim.AdamW if si_opt_name == "adamw" else optim.SGD
        si_opt_kwargs = {"weight_decay": train_cfg["weight_decay"]}
        if si_opt_name != "adamw":
            si_opt_kwargs["momentum"] = train_cfg["momentum"]
        si_omega, si_theta_star = compute_si_importance(
            model, task_a_train, si_opt_cls,
            lr=train_cfg["lr"], device=device, n_epochs=3,
            optimizer_kwargs={
                **si_opt_kwargs,
            },
        )
        print(f"  SI importance computed ({len(si_omega)} parameter groups)")

        # Diagnostic: log omega scale to calibrate lambda
        import numpy as np_diag
        all_omega_np = torch.cat([v.flatten() for v in si_omega.values()]).cpu().numpy()
        nonzero_pct = (all_omega_np > 0).mean() * 100
        print(f"  Omega stats: mean={all_omega_np.mean():.4e}, "
              f"median={float(np_diag.median(all_omega_np)):.4e}, "
              f"max={all_omega_np.max():.4e}, "
              f"nonzero={nonzero_pct:.1f}%")
        nz = all_omega_np[all_omega_np > 0]
        if len(nz) > 0:
            print(f"  Omega (nonzero only): mean={nz.mean():.4e}, "
                  f"median={float(np_diag.median(nz)):.4e}, "
                  f"p25={float(np_diag.percentile(nz, 25)):.4e}, "
                  f"p75={float(np_diag.percentile(nz, 75)):.4e}")

    # Reset RNG after EWC/SI computation so classifier init and data order
    # match the naive baseline (importance computation consumes randomness)
    if args.ewc or args.si:
        set_seed(cfg["seed"])

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

    # Update SI omega and theta_star for expanded classifier if SI is active
    if args.si:
        from experiments.shared.si import si_penalty
        for n, p in model.named_parameters():
            if n not in si_omega:
                si_omega[n] = torch.zeros_like(p)
                si_theta_star[n] = p.data.clone()
            elif si_omega[n].shape != p.shape:
                old_omega = si_omega[n]
                old_theta = si_theta_star[n]
                new_omega = torch.zeros_like(p)
                new_theta = p.data.clone()
                slices = tuple(slice(0, s) for s in old_omega.shape)
                new_omega[slices] = old_omega
                new_theta[slices] = old_theta
                si_omega[n] = new_omega
                si_theta_star[n] = new_theta

    # Optimizer for Task B (train all parameters)
    task_b_lr = train_cfg["lr"] * 0.1
    opt_name = train_cfg.get("optimizer", "sgd")
    if opt_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=task_b_lr,
            weight_decay=train_cfg["weight_decay"],
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=task_b_lr,
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
            ce_loss = criterion(outputs, labels)
            loss = ce_loss

            # Add EWC penalty
            reg_pen_val = 0.0
            if args.ewc:
                ewc_pen = ewc_penalty(model, fisher, theta_star, args.ewc_lambda)
                reg_pen_val = ewc_pen.item()
                loss = loss + ewc_pen

            # Add SI penalty
            if args.si:
                si_pen = si_penalty(model, si_omega, si_theta_star, args.si_lambda)
                reg_pen_val = si_pen.item()
                loss = loss + si_pen

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            step += 1

            if step in eval_steps:
                # Diagnostic: check for NaN and loss values
                has_nan = any(torch.isnan(p).any().item() for p in model.parameters())
                if has_nan:
                    print(f"  [DEBUG] NaN in parameters at step {step}!")
                if args.si or args.ewc:
                    print(f"  [DEBUG] step {step}: CE={ce_loss.item():.4f}, "
                          f"penalty={reg_pen_val:.6f}, total={loss.item():.4f}")

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
    if args.cross_dataset:
        metadata["cross_dataset"] = True
        metadata["task_a_dataset"] = cfg.get("dataset", "cifar100")
        metadata["task_b_dataset"] = args.cross_dataset
        metadata["task_a_dir"] = args.task_a_dir
    if args.ewc:
        metadata["ewc"] = True
        metadata["ewc_lambda"] = args.ewc_lambda
    if args.si:
        metadata["si"] = True
        metadata["si_lambda"] = args.si_lambda
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
