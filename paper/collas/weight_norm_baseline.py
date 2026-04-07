#!/usr/bin/env python3
"""
Add weight norm as trivial baseline to the baseline comparison.
Loads task_a_best.pt for each config, computes L2 weight norm,
then runs the same partial correlation analysis.
"""

import torch
import math
import json
import os
import numpy as np
from scipy import stats

RESULTS_DIR = "/home/joshua/Corporate/axiondeep-research/results"

ARCHS = [
    "exp01", "exp01_resnet50", "exp01_vit", "exp01_vittiny",
    "exp01_densenet121", "exp01_efficientnet", "exp01_mobilenetv3",
    "exp01_shufflenet", "exp01_regnet", "exp01_convnext",
    "exp01_vgg16bn", "exp01_mlpmixer",
    "exp01_wrn281", "exp01_wrn282", "exp01_wrn284",
    "exp01_wrn286", "exp01_wrn288", "exp01_wrn2810",
    "exp01_resnet18wide",
]

DATASET_SUFFIXES = {"cifar100": "", "cub200": "_cub200", "resisc45": "_resisc45"}

def get_weight_norm(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    else:
        sd = ckpt
    params = [p for p in sd.values() if isinstance(p, torch.Tensor) and p.is_floating_point()]
    return math.sqrt(sum(p.float().pow(2).sum().item() for p in params))

def compute_aurc(curve, base_acc, max_step=500):
    if base_acc is None or base_acc == 0:
        return None
    points = [(0, 1.0)]
    for item in curve:
        step = item["step"]
        acc = item["task_a_acc"]
        if step > 0 and step <= max_step:
            points.append((step, acc / base_acc))
    if len(points) < 2:
        return None
    points.sort()
    return float(np.trapezoid([r for _, r in points], [s for s, _ in points]))

def main():
    print("=" * 70)
    print("WEIGHT NORM BASELINE: The trivial metric test")
    print("=" * 70)

    for dataset in ["cifar100", "cub200", "resisc45"]:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*70}")

        h0_vals, wn_vals, ewc_vals, names = [], [], [], []

        for arch in ARCHS:
            suffix = DATASET_SUFFIXES[dataset]
            dirname = f"{arch}{suffix}" if suffix else arch
            result_dir = os.path.join(RESULTS_DIR, dirname)
            if not os.path.exists(result_dir):
                continue

            # Topology
            topo_path = os.path.join(result_dir, "topology", "topology_summary.json")
            if not os.path.exists(topo_path):
                continue
            with open(topo_path) as f:
                topo = json.load(f)

            # Weight norm
            ckpt_path = os.path.join(result_dir, "checkpoints", "task_a_best.pt")
            if not os.path.exists(ckpt_path):
                continue
            wn = get_weight_norm(ckpt_path)

            # EWC benefit
            naive_path = os.path.join(result_dir, "forgetting", "forgetting_curve.json")
            ewc_path = os.path.join(result_dir, "forgetting_ewc", "forgetting_curve.json")
            if not os.path.exists(naive_path) or not os.path.exists(ewc_path):
                continue
            with open(naive_path) as f:
                naive = json.load(f)
            with open(ewc_path) as f:
                ewc = json.load(f)

            base_acc = naive.get("initial_task_a_acc")
            aurc_naive = compute_aurc(naive["curve"], base_acc)
            aurc_ewc = compute_aurc(ewc["curve"], base_acc)
            if aurc_naive is None or aurc_ewc is None:
                continue

            h0_vals.append(topo["H0"])
            wn_vals.append(wn)
            ewc_vals.append(aurc_ewc - aurc_naive)
            names.append(dirname)

        n = len(h0_vals)
        h0 = np.array(h0_vals)
        wn = np.array(wn_vals)
        ewc = np.array(ewc_vals)

        print(f"  n = {n}")

        # Raw correlations
        rho_h0, p_h0 = stats.spearmanr(h0, ewc)
        rho_wn, p_wn = stats.spearmanr(wn, ewc)
        print(f"\n  Raw Spearman with EWC benefit:")
        print(f"    H0:          rho = {rho_h0:.3f}, p = {p_h0:.6f}")
        print(f"    Weight norm: rho = {rho_wn:.3f}, p = {p_wn:.6f}")

        # H0 controlling for weight norm
        resid_h0 = h0 - np.polyval(np.polyfit(wn, h0, 1), wn)
        resid_ewc_wn = ewc - np.polyval(np.polyfit(wn, ewc, 1), wn)
        rho_h0_ctrl, p_h0_ctrl = stats.spearmanr(resid_h0, resid_ewc_wn)

        # Weight norm controlling for H0
        resid_wn = wn - np.polyval(np.polyfit(h0, wn, 1), h0)
        resid_ewc_h0 = ewc - np.polyval(np.polyfit(h0, ewc, 1), h0)
        rho_wn_ctrl, p_wn_ctrl = stats.spearmanr(resid_wn, resid_ewc_h0)

        print(f"\n  Partial correlations:")
        print(f"    H0 | weight norm:  rho = {rho_h0_ctrl:.3f}, p = {p_h0_ctrl:.6f} {'*' if p_h0_ctrl < 0.05 else ''}")
        print(f"    Weight norm | H0:  rho = {rho_wn_ctrl:.3f}, p = {p_wn_ctrl:.6f} {'*' if p_wn_ctrl < 0.05 else ''}")

        # Correlation between H0 and weight norm
        rho_h0_wn, _ = stats.spearmanr(h0, wn)
        print(f"\n  H0 vs weight norm:   rho = {rho_h0_wn:.3f}")

    print(f"\n{'='*70}")
    print("DONE")
    print("=" * 70)

if __name__ == "__main__":
    main()
