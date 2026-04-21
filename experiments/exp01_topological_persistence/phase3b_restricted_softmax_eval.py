"""
EXP-01 Phase 3b: Restricted-softmax re-evaluation of cross-dataset forgetting.

Post-hoc, compute-from-disk. Reads every step_{k}.pt checkpoint produced by
phase3_sequential_forgetting.py in cross-dataset mode and re-evaluates
Task A test accuracy with argmax restricted to the Task A logit rows
[0, num_classes_a). Writes forgetting_curve_restricted.json alongside the
existing forgetting_curve.json so both metrics can be compared.

Rationale and full audit in drafts/phase_1b_g1_metric_audit_memo.md.

Usage:
    # Process every *_xd_* run dir under results/
    python -m experiments.exp01_topological_persistence.phase3b_restricted_softmax_eval \
        --results-dir ./results

    # Process a single run dir
    python -m experiments.exp01_topological_persistence.phase3b_restricted_softmax_eval \
        --run-dir ./results/exp01_resnet50_xd_cub200

    # Recompute even if forgetting_curve_restricted.json already exists
    python -m experiments.exp01_topological_persistence.phase3b_restricted_softmax_eval \
        --results-dir ./results --force
"""

import argparse
import gc
import glob
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.shared.datasets import get_split_dataset
from experiments.shared.models import get_model
from experiments.shared.utils import load_config, set_seed


# Dataset name -> number of classes (must match get_cross_dataset_task_b).
XD_NUM_CLASSES = {
    "cifar100": 100,
    "cub200": 200,
    "resisc45": 45,
}

DATASET_SUFFIX = {
    "cifar100": "",
    "cub200": "_cub200",
    "resisc45": "_resisc45",
}

# Base arch keys used in submit_cross_dataset.sh, minus the "exp01" prefix.
KNOWN_ARCH_KEYS = {
    "resnet50", "vit", "vittiny", "densenet121", "efficientnet",
    "mobilenetv3", "shufflenet", "regnet", "convnext", "vgg16bn",
    "mlpmixer", "wrn281", "wrn282", "wrn284", "wrn286", "wrn288",
    "wrn2810", "resnet18wide",
}


def parse_run_dir_name(run_dir: str) -> Tuple[str, str, str]:
    """Parse a cross-dataset run dir name into (arch_key, task_a_ds, task_b_ds).

    Examples:
        exp01_xd_cub200                  -> (exp01, cifar100, cub200)
        exp01_resnet50_xd_cub200         -> (exp01_resnet50, cifar100, cub200)
        exp01_vittiny_resisc45_xd_cub200 -> (exp01_vittiny_resisc45, resisc45, cub200)
    """
    name = os.path.basename(run_dir.rstrip("/"))
    m = re.match(r"^(exp01(?:_[^_]+)?(?:_(cifar100|cub200|resisc45))?)_xd_(cifar100|cub200|resisc45)$", name)
    if m is None:
        raise ValueError(f"Could not parse run dir name: {name}")
    # Try the "has task_a dataset suffix" form first, then fall back.
    prefix = m.group(1)
    task_a_suffix_hit = m.group(2)
    task_b = m.group(3)
    if task_a_suffix_hit is not None:
        task_a = task_a_suffix_hit
        arch_prefix = prefix  # e.g. exp01_vittiny_resisc45 -- strip task_a suffix below
        # When prefix ends with _<task_a>, strip it to get the pure arch prefix
        if arch_prefix.endswith("_" + task_a):
            arch_prefix = arch_prefix[: -(len(task_a) + 1)]
    else:
        task_a = "cifar100"
        arch_prefix = prefix
    return arch_prefix, task_a, task_b


def config_path_for(arch_prefix: str, task_a_dataset: str, configs_dir: str) -> str:
    """Map (arch_prefix, task_a_dataset) -> config file path.

    Mirrors the logic in slurm/submit_cross_dataset.sh.
    """
    suffix = DATASET_SUFFIX[task_a_dataset]
    if suffix == "":
        return os.path.join(configs_dir, f"{arch_prefix}.yaml")
    return os.path.join(configs_dir, f"{arch_prefix}{suffix}.yaml")


def load_checkpoint_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt["model_state_dict"]


def infer_k_total_from_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    """Find the final classifier out_features by scanning state_dict keys."""
    candidates = []
    for key, tensor in sd.items():
        if tensor.ndim != 2:
            continue
        if not key.endswith(".weight"):
            continue
        # Prefer keys named like fc.weight, head.weight, classifier.*.weight.
        if any(tok in key for tok in ("fc.weight", "head.weight", "classifier", "heads.head")):
            candidates.append((key, tensor.shape[0]))
    if not candidates:
        raise RuntimeError("Could not locate classifier layer in state_dict")
    # If multiple, prefer the one with largest out_features (final head).
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    return int(candidates[0][1])


@torch.no_grad()
def evaluate_restricted(model: nn.Module, dataloader, device: torch.device,
                        num_classes_a: int) -> Tuple[float, float]:
    """Return (restricted_acc, full_acc) on Task A test set.

    - restricted: argmax over logit columns [0, num_classes_a)
    - full:       argmax over all logit columns (reproduces the original metric)
    """
    model.eval()
    correct_restricted = 0
    correct_full = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # Full softmax argmax (for sanity: should match forgetting_curve.json)
        _, pred_full = outputs.max(1)
        correct_full += pred_full.eq(labels).sum().item()
        # Restricted softmax argmax
        _, pred_restricted = outputs[:, :num_classes_a].max(1)
        correct_restricted += pred_restricted.eq(labels).sum().item()
        total += labels.size(0)
    return correct_restricted / total, correct_full / total


def find_step_checkpoints(forgetting_subdir: str) -> List[Tuple[int, str]]:
    """Return sorted [(step, path), ...] for all step_*.pt files in dir."""
    out = []
    for path in glob.glob(os.path.join(forgetting_subdir, "step_*.pt")):
        m = re.match(r"step_(\d+)\.pt$", os.path.basename(path))
        if m:
            out.append((int(m.group(1)), path))
    out.sort(key=lambda t: t[0])
    return out


def atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def process_condition(forgetting_subdir: str, cfg: dict, task_a_test,
                      arch: str, num_classes_a: int,
                      k_total: int, device: torch.device,
                      force: bool) -> Optional[dict]:
    """Re-evaluate one forgetting_* subdir (naive or ewc). Returns summary dict.
    Returns None if already done and not --force."""
    out_json = os.path.join(forgetting_subdir, "forgetting_curve_restricted.json")
    if os.path.exists(out_json) and not force:
        return None

    curve_json = os.path.join(forgetting_subdir, "forgetting_curve.json")
    if not os.path.exists(curve_json):
        print(f"    SKIP {forgetting_subdir}: no forgetting_curve.json")
        return None
    with open(curve_json) as f:
        existing = json.load(f)
    initial_full_acc = existing.get("initial_task_a_acc")
    existing_curve = {pt["step"]: pt for pt in existing.get("curve", [])}

    steps = find_step_checkpoints(forgetting_subdir)
    if not steps:
        print(f"    SKIP {forgetting_subdir}: no step_*.pt files found")
        return None

    # Build model once, reload state dict per step.
    model = get_model(arch, num_classes=k_total).to(device)

    new_curve = []
    # Step 0 is implicit (no step_0.pt saved). We record initial_task_a_acc
    # from the existing metadata as the step 0 point for the full metric.
    # The step-0 sanity check (memo Action 4) is performed separately in
    # run_step0_sanity() using the phase1 task_a_best.pt checkpoint.

    for step, ckpt_path in steps:
        sd = load_checkpoint_state_dict(ckpt_path)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"    WARN step {step}: unexpected keys {unexpected[:3]}...")
        if missing:
            print(f"    WARN step {step}: missing keys {missing[:3]}...")
        restricted_acc, full_acc = evaluate_restricted(
            model, task_a_test, device, num_classes_a
        )
        prior_full = existing_curve.get(step, {}).get("task_a_acc")
        new_curve.append({
            "step": step,
            "task_a_acc_full": full_acc,
            "task_a_acc_restricted": restricted_acc,
            "task_a_acc_full_original": prior_full,
            "full_vs_recomputed_diff": (
                None if prior_full is None else full_acc - prior_full
            ),
        })

    summary = {
        "initial_task_a_acc": initial_full_acc,
        "num_classes_a": num_classes_a,
        "k_total": k_total,
        "source_curve": os.path.basename(curve_json),
        "metric_notes": (
            "task_a_acc_full: argmax over all K_A+K_B logits (reproduction check). "
            "task_a_acc_restricted: argmax over [0, K_A) logits only, isolates "
            "backbone drift from classifier-head recency bias. See "
            "drafts/phase_1b_g1_metric_audit_memo.md."
        ),
        "curve": new_curve,
    }
    atomic_write_json(out_json, summary)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def run_step0_sanity(task_a_dir: str, cfg: dict, task_a_test,
                     arch: str, num_classes_a: int,
                     device: torch.device) -> Optional[dict]:
    """Load task_a_best.pt (phase1), evaluate Task A acc, return sanity record.

    The phase1 checkpoint has the un-expanded K_A-dim head. For restricted
    eval at step 0, restricted_acc reduces to plain Task A accuracy -- there
    are no Task B logits to mask. This should match initial_task_a_acc from
    each run's forgetting_curve.json within rounding.
    """
    ckpt_path = os.path.join(task_a_dir, "checkpoints", "task_a_best.pt")
    if not os.path.exists(ckpt_path):
        return None
    model = get_model(arch, num_classes=num_classes_a).to(device)
    sd = load_checkpoint_state_dict(ckpt_path)
    model.load_state_dict(sd, strict=True)
    restricted_acc, full_acc = evaluate_restricted(
        model, task_a_test, device, num_classes_a
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # For an un-expanded head, full == restricted by construction.
    return {
        "task_a_best_acc_reeval": restricted_acc,
        "task_a_best_full_acc_reeval": full_acc,
    }


def process_run_dir(run_dir: str, configs_dir: str, device: torch.device,
                    force: bool, eval_batch_size: int = 128) -> Dict:
    arch_prefix, task_a_ds, task_b_ds = parse_run_dir_name(run_dir)

    cfg_path = config_path_for(arch_prefix, task_a_ds, configs_dir)
    if not os.path.exists(cfg_path):
        print(f"SKIP {run_dir}: config not found at {cfg_path}")
        return {"run_dir": run_dir, "status": "no_config"}
    cfg = load_config(cfg_path)
    arch = cfg["architecture"]
    num_classes_a = cfg["num_classes_a"]
    xd_k_b = XD_NUM_CLASSES[task_b_ds]
    k_total = num_classes_a + xd_k_b

    print(f"\n=== {run_dir}")
    print(f"    config={os.path.basename(cfg_path)} arch={arch} "
          f"K_A={num_classes_a} K_B(xd)={xd_k_b} K_total={k_total}")

    # Seed before dataset construction so RESISC split matches training.
    set_seed(cfg["seed"])
    data = get_split_dataset(cfg)
    _, task_a_test = data.get_task_a(batch_size=eval_batch_size)
    print(f"    Task A test: {len(task_a_test.dataset)} samples")

    # Locate Task A dir (for step-0 sanity). The run dir has a `checkpoints`
    # symlink pointing at the Task A result dir; follow it.
    ckpt_link = os.path.join(run_dir, "checkpoints")
    task_a_dir = None
    if os.path.islink(ckpt_link):
        task_a_dir = os.path.dirname(os.path.realpath(ckpt_link))
    elif os.path.isdir(ckpt_link):
        task_a_dir = run_dir  # fallback (unlikely)

    sanity = None
    if task_a_dir is not None:
        sanity = run_step0_sanity(
            task_a_dir, cfg, task_a_test, arch, num_classes_a, device
        )
        if sanity is not None:
            print(f"    step-0 sanity: re-evaluated task_a_best_acc = "
                  f"{sanity['task_a_best_acc_reeval']:.4f}")

    # Process both naive and EWC subdirs if present.
    results = {"run_dir": run_dir, "conditions": {}}
    if sanity is not None:
        results["step0_sanity"] = sanity

    for cond_name, subdir in [("naive", "forgetting"), ("ewc", "forgetting_ewc")]:
        full_subdir = os.path.join(run_dir, subdir)
        if not os.path.isdir(full_subdir):
            continue
        print(f"  [{cond_name}] {full_subdir}")
        t0 = time.time()
        summary = process_condition(
            full_subdir, cfg, task_a_test, arch, num_classes_a,
            k_total, device, force
        )
        dt = time.time() - t0
        if summary is None:
            print(f"    already done (or empty)")
            results["conditions"][cond_name] = "skipped"
            continue
        diffs = [p["full_vs_recomputed_diff"] for p in summary["curve"]
                 if p["full_vs_recomputed_diff"] is not None]
        max_abs = max((abs(d) for d in diffs), default=0.0)
        print(f"    wrote forgetting_curve_restricted.json ({dt:.1f}s, "
              f"{len(summary['curve'])} steps, max |full-reeval diff| = {max_abs:.4f})")
        results["conditions"][cond_name] = {
            "steps": len(summary["curve"]),
            "max_full_reeval_diff": max_abs,
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Restricted-softmax re-eval of cross-dataset forgetting runs"
    )
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Parent dir containing *_xd_* run dirs")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Process a single run dir instead of scanning")
    parser.add_argument("--configs-dir", type=str, default="./configs")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if forgetting_curve_restricted.json exists")
    parser.add_argument("--eval-batch-size", type=int, default=128,
                        help="Batch size for Task A test loader (default 128)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (debug)")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    if args.run_dir is not None:
        run_dirs = [args.run_dir]
    else:
        run_dirs = sorted(
            d for d in glob.glob(os.path.join(args.results_dir, "*_xd_*"))
            if os.path.isdir(d)
        )
    print(f"Found {len(run_dirs)} candidate run dir(s)")

    summary_log = []
    for rd in run_dirs:
        try:
            result = process_run_dir(rd, args.configs_dir, device, args.force,
                                     eval_batch_size=args.eval_batch_size)
            summary_log.append(result)
        except Exception as exc:
            print(f"ERROR {rd}: {exc}")
            summary_log.append({"run_dir": rd, "status": "error", "error": str(exc)})

    # Write top-level summary for the meeting readout.
    summary_path = os.path.join(args.results_dir, "phase3b_restricted_summary.json")
    atomic_write_json(summary_path, summary_log)
    print(f"\nTop-level summary written: {summary_path}")


if __name__ == "__main__":
    main()
