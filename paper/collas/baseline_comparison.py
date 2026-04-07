#!/usr/bin/env python3
"""
Baseline comparison for CoLLAs paper.
Tests whether H0 topology adds signal beyond cheaper baseline metrics
(sharpness, Fisher trace, barrier) for predicting EWC benefit.
"""

import json
import os
import numpy as np
from scipy import stats

RESULTS_DIR = "/home/joshua/Corporate/axiondeep-research/results"

# Architecture base names (CIFAR-100 dirs have no dataset suffix)
ARCHS = [
    "exp01",              # resnet18
    "exp01_resnet50",
    "exp01_vit",          # vit-small
    "exp01_vittiny",
    "exp01_densenet121",
    "exp01_efficientnet",  # efficientnet-b0
    "exp01_mobilenetv3",
    "exp01_shufflenet",
    "exp01_regnet",
    "exp01_convnext",
    "exp01_vgg16bn",
    "exp01_mlpmixer",
    "exp01_wrn281",
    "exp01_wrn282",
    "exp01_wrn284",
    "exp01_wrn286",
    "exp01_wrn288",
    "exp01_wrn2810",
    "exp01_resnet18wide",
]

DATASET_SUFFIXES = {
    "cifar100": "",        # no suffix for cifar100 (default dataset)
    "cub200": "_cub200",
    "resisc45": "_resisc45",
}

def get_dir_name(arch_base, dataset):
    suffix = DATASET_SUFFIXES[dataset]
    if suffix:
        return f"{arch_base}{suffix}"
    return arch_base

def compute_aurc(curve, base_acc, max_step=500):
    """Compute area under retention curve up to max_step."""
    if base_acc is None or base_acc == 0:
        return None

    points = [(0, 1.0)]  # step 0, retention = 1.0
    for item in curve:
        step = item["step"]
        acc = item["task_a_acc"]
        if step > 0 and step <= max_step:
            points.append((step, acc / base_acc))

    if len(points) < 2:
        return None

    points.sort()
    steps = [s for s, _ in points]
    rets = [r for _, r in points]
    return float(np.trapz(rets, steps))

def main():
    print("=" * 70)
    print("BASELINE COMPARISON: Does H0 add signal beyond cheaper metrics?")
    print("=" * 70)

    for dataset in ["cifar100", "cub200", "resisc45"]:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset.upper()} (n = 19)")
        print(f"{'='*70}")

        data = {
            "h0": [], "hessian": [], "eigenvalue": [],
            "fisher": [], "barrier": [], "ewc_benefit": [],
            "arch": [],
        }

        for arch in ARCHS:
            dirname = get_dir_name(arch, dataset)
            result_dir = os.path.join(RESULTS_DIR, dirname)

            if not os.path.exists(result_dir):
                continue

            # Load topology + baselines
            topo_path = os.path.join(result_dir, "topology", "topology_summary.json")
            if not os.path.exists(topo_path):
                continue
            with open(topo_path) as f:
                topo = json.load(f)

            # Load naive forgetting
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

            ewc_benefit = aurc_ewc - aurc_naive

            data["h0"].append(topo["H0"])
            data["hessian"].append(topo.get("hessian_trace_mean", np.nan))
            data["eigenvalue"].append(topo.get("max_eigenvalue", np.nan))
            data["fisher"].append(topo.get("fisher_trace", np.nan))
            data["barrier"].append(topo.get("max_barrier_normalized", np.nan))
            data["ewc_benefit"].append(ewc_benefit)
            data["arch"].append(dirname)

        n = len(data["h0"])
        if n < 5:
            print(f"  Only {n} configs with complete data, skipping")
            continue

        print(f"  {n} architectures with complete data")

        # Convert
        h0 = np.array(data["h0"])
        ewc_ben = np.array(data["ewc_benefit"])
        baselines = {
            "H0 (topology)": h0,
            "Hessian trace": np.array(data["hessian"]),
            "Max eigenvalue": np.array(data["eigenvalue"]),
            "Fisher trace": np.array(data["fisher"]),
            "Barrier (norm)": np.array(data["barrier"]),
        }

        # 1. Raw Spearman correlations with EWC benefit
        print()
        print("  --- Raw Spearman correlations with EWC benefit ---")
        print(f"  {'Metric':<20} {'rho':>8} {'p':>12} {'sig':>5}")
        print(f"  {'-'*47}")
        for name, vals in baselines.items():
            mask = ~np.isnan(vals)
            rho, p = stats.spearmanr(vals[mask], ewc_ben[mask])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {name:<20} {rho:>8.3f} {p:>12.6f} {sig:>5}")

        # 2. Key test: Does H0 add signal beyond each baseline?
        #    Partial correlation of H0 with EWC benefit, controlling for baseline
        print()
        print("  --- H0 partial correlation (controlling for each baseline) ---")
        print(f"  {'Controlled for':<20} {'H0 rho':>8} {'p':>12} {'sig':>5}")
        print(f"  {'-'*47}")
        for name, vals in baselines.items():
            if name == "H0 (topology)":
                continue
            mask = ~np.isnan(vals)
            if mask.sum() < 5:
                continue
            # Residualize H0 on baseline
            resid_h0 = h0[mask] - np.polyval(np.polyfit(vals[mask], h0[mask], 1), vals[mask])
            # Residualize EWC benefit on baseline
            resid_ewc = ewc_ben[mask] - np.polyval(np.polyfit(vals[mask], ewc_ben[mask], 1), vals[mask])
            rho, p = stats.spearmanr(resid_h0, resid_ewc)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {name:<20} {rho:>8.3f} {p:>12.6f} {sig:>5}")

        # 3. Reverse: Does each baseline add signal beyond H0?
        print()
        print("  --- Baseline partial correlation (controlling for H0) ---")
        print(f"  {'Baseline':<20} {'rho':>8} {'p':>12} {'sig':>5}")
        print(f"  {'-'*47}")
        for name, vals in baselines.items():
            if name == "H0 (topology)":
                continue
            mask = ~np.isnan(vals)
            if mask.sum() < 5:
                continue
            resid_bl = vals[mask] - np.polyval(np.polyfit(h0[mask], vals[mask], 1), h0[mask])
            resid_ewc = ewc_ben[mask] - np.polyval(np.polyfit(h0[mask], ewc_ben[mask], 1), h0[mask])
            rho, p = stats.spearmanr(resid_bl, resid_ewc)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {name:<20} {rho:>8.3f} {p:>12.6f} {sig:>5}")

    print()
    print("=" * 70)
    print("INTERPRETATION:")
    print("  If H0 partial rho remains significant after controlling for baselines,")
    print("  topology adds incremental signal that cheaper metrics miss.")
    print("  If baselines show NO partial signal after controlling for H0,")
    print("  then H0 subsumes whatever those baselines capture.")
    print("=" * 70)

if __name__ == "__main__":
    main()
