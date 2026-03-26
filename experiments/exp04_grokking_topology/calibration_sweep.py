"""
EXP-04: Weight decay calibration sweep.

Runs a single seed at 4 weight decay values to find the grokking regime.
No topology or baselines — just training curves.

Acceptance criterion:
  - Train accuracy > 99% early (memorization)
  - Test accuracy materially below train for a sustained window
  - Test accuracy crosses 90% before 100K steps

Usage:
    python -m experiments.exp04_grokking_topology.calibration_sweep \
        --config configs/exp04_pilot.yaml
"""

import os
import sys
import json
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.shared.utils import set_seed, load_config
from experiments.exp04_grokking_topology.dataset import get_dataloaders
from experiments.exp04_grokking_topology.model import build_model


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
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


def run_sweep_condition(cfg, weight_decay, seed, total_steps, device):
    """Train one condition, log every 500 steps."""
    set_seed(seed)

    cfg_copy = {**cfg, "training": {**cfg["training"], "weight_decay": weight_decay}}
    train_loader, test_loader = get_dataloaders(cfg_copy)
    model = build_model(cfg_copy, device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=weight_decay,
    )

    log_cadence = 500
    metrics = []
    step = 0

    # Step 0
    train_loss, train_acc = evaluate(model, train_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    metrics.append({"step": 0, "train_acc": train_acc, "test_acc": test_acc,
                     "train_loss": train_loss, "test_loss": test_loss})

    while step < total_steps:
        for x, y in train_loader:
            if step >= total_steps:
                break
            x, y = x.to(device), y.to(device)
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            step += 1

            if step % log_cadence == 0:
                train_loss, train_acc = evaluate(model, train_loader, criterion, device)
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                metrics.append({"step": step, "train_acc": train_acc, "test_acc": test_acc,
                                 "train_loss": train_loss, "test_loss": test_loss})

                gap = train_acc - test_acc
                grok_marker = ""
                if train_acc > 0.99 and test_acc < 0.5:
                    grok_marker = " [MEMORIZED, not generalized]"
                elif train_acc > 0.99 and test_acc > 0.9:
                    grok_marker = " [GROKKED]"

                print(f"    Step {step:>6}: train={train_acc:.4f} test={test_acc:.4f} "
                      f"gap={gap:.4f}{grok_marker}")

    return metrics


def analyze_condition(metrics, weight_decay):
    """Check acceptance criteria."""
    # When did train acc first exceed 99%?
    memorization_step = None
    for m in metrics:
        if m["train_acc"] > 0.99:
            memorization_step = m["step"]
            break

    # When did test acc first exceed 90%?
    generalization_step = None
    for m in metrics:
        if m["test_acc"] > 0.90:
            generalization_step = m["step"]
            break

    # Sustained gap: how many checkpoints with train > 99% and test < 50%?
    gap_checkpoints = sum(1 for m in metrics if m["train_acc"] > 0.99 and m["test_acc"] < 0.50)

    # Max gap
    max_gap = max((m["train_acc"] - m["test_acc"]) for m in metrics)

    result = {
        "weight_decay": weight_decay,
        "memorization_step": memorization_step,
        "generalization_step": generalization_step,
        "grokking_delay": (generalization_step - memorization_step) if (memorization_step and generalization_step) else None,
        "gap_checkpoints": gap_checkpoints,
        "max_gap": round(max_gap, 4),
        "final_train_acc": metrics[-1]["train_acc"],
        "final_test_acc": metrics[-1]["test_acc"],
    }

    # Acceptance: memorizes early, sustained gap, generalizes within budget
    accepted = (
        memorization_step is not None
        and memorization_step < 10000
        and generalization_step is not None
        and gap_checkpoints >= 5
    )
    result["accepted"] = accepted
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EXP-04 Weight Decay Calibration Sweep")
    parser.add_argument("--config", type=str, default="configs/exp04_pilot.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_decays = [0.01, 0.03, 0.1, 0.3]

    print("=" * 60)
    print("EXP-04: Weight Decay Calibration Sweep")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")
    print(f"  Steps: {args.steps}")
    print(f"  Weight decays: {weight_decays}")
    print()

    output_dir = os.path.join("results", "exp04_calibration")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for wd in weight_decays:
        print(f"\n{'=' * 60}")
        print(f"  Weight Decay = {wd}")
        print(f"{'=' * 60}")

        t0 = time.time()
        metrics = run_sweep_condition(cfg, wd, args.seed, args.steps, device)
        elapsed = time.time() - t0

        analysis = analyze_condition(metrics, wd)
        analysis["elapsed_seconds"] = round(elapsed, 1)
        all_results.append(analysis)

        # Save per-condition metrics
        condition_path = os.path.join(output_dir, f"wd_{wd}.json")
        with open(condition_path, "w") as f:
            json.dump({"weight_decay": wd, "seed": args.seed, "metrics": metrics}, f, indent=2)

        print(f"\n  Summary (wd={wd}):")
        print(f"    Memorization step: {analysis['memorization_step']}")
        print(f"    Generalization step: {analysis['generalization_step']}")
        print(f"    Grokking delay: {analysis['grokking_delay']}")
        print(f"    Sustained gap checkpoints: {analysis['gap_checkpoints']}")
        print(f"    Max gap: {analysis['max_gap']}")
        print(f"    Final: train={analysis['final_train_acc']:.4f} test={analysis['final_test_acc']:.4f}")
        print(f"    Accepted: {'YES' if analysis['accepted'] else 'NO'}")
        print(f"    Time: {elapsed:.0f}s")

    # Summary
    print(f"\n\n{'=' * 60}")
    print("CALIBRATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'WD':>6} | {'Mem Step':>9} | {'Gen Step':>9} | {'Delay':>7} | {'Gap Ckpts':>9} | {'Max Gap':>8} | {'Accept':>6}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['weight_decay']:>6} | {str(r['memorization_step']):>9} | "
              f"{str(r['generalization_step']):>9} | {str(r['grokking_delay']):>7} | "
              f"{r['gap_checkpoints']:>9} | {r['max_gap']:>8.4f} | "
              f"{'YES' if r['accepted'] else 'NO':>6}")

    accepted = [r for r in all_results if r["accepted"]]
    if accepted:
        # Pick the one with the longest delay (most data in the pre-grokking window)
        best = max(accepted, key=lambda r: r["grokking_delay"])
        print(f"\n  RECOMMENDED: weight_decay = {best['weight_decay']}")
        print(f"  Reason: longest grokking delay ({best['grokking_delay']} steps)")
    else:
        print(f"\n  NO WEIGHT DECAY PRODUCED CLEAR GROKKING.")
        print(f"  Escalation: match Power et al. architecture (2-layer, learned embeddings).")

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
