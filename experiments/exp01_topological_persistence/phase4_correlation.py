"""
EXP-01 Phase 4: Correlate topological features with forgetting resistance.

Compares H0 persistence against baseline metrics (Hessian trace, max eigenvalue,
Fisher trace, loss barrier) to determine whether topology captures something
that simpler geometry metrics do not.

Supports:
  - Multi-slice aggregation: when multiple topology_summary_run*.json files exist,
    computes mean ± std across slices for robust topology estimates
  - Displacement metrics: includes curvature_ratio and basin_fraction from Phase 2.5
  - Normalized barrier: cross-architecture comparable barrier height
  - Partial correlation: controls for model size (num_params) as confound
  - Architecture class analysis: within-class (CNN/Transformer/MLP) correlations
  - Baseline comparison: does num_params alone predict retention?

Usage:
    python -m experiments.exp01_topological_persistence.phase4_correlation \
        --config configs/exp01.yaml

    # Cross-architecture (after running multiple configs):
    python -m experiments.exp01_topological_persistence.phase4_correlation \
        --results-dirs results/exp01 results/exp01_vit results/exp01_resnet50 ...
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.shared.utils import load_config


# All metrics to correlate against retention
METRICS = [
    ("H0", "H0 Persistence"),
    ("H1", "H1 Persistence"),
    ("H0_cubical", "H0 Cubical"),
    ("H1_cubical", "H1 Cubical"),
    ("hessian_trace_mean", "Hessian Trace"),
    ("max_eigenvalue", "Max Eigenvalue (Sharpness)"),
    ("fisher_trace", "Fisher Information Trace"),
    ("max_barrier", "Loss Barrier Height"),
    ("max_barrier_normalized", "Barrier (Normalized)"),
    ("curvature_ratio", "Displacement Curvature"),
    ("basin_fraction", "Basin Fraction"),
    ("num_params", "Parameter Count"),
]

# Architecture class tags for within-class analysis
ARCH_CLASSES = {
    "exp01": ("ResNet-18", "CNN"),
    "exp01_resnet50": ("ResNet-50", "CNN"),
    "exp01_resnet18wide": ("ResNet-18-Wide", "CNN"),
    "exp01_wrn281": ("WRN-28-1", "WRN-ladder"),
    "exp01_wrn282": ("WRN-28-2", "WRN-ladder"),
    "exp01_wrn284": ("WRN-28-4", "WRN-ladder"),
    "exp01_wrn286": ("WRN-28-6", "WRN-ladder"),
    "exp01_wrn288": ("WRN-28-8", "WRN-ladder"),
    "exp01_wrn2810": ("WRN-28-10", "WRN-ladder"),
    "exp01_densenet121": ("DenseNet-121", "CNN"),
    "exp01_efficientnet": ("EfficientNet-B0", "CNN"),
    "exp01_vgg16bn": ("VGG-16-BN", "CNN"),
    "exp01_convnext": ("ConvNeXt-Tiny", "CNN"),
    "exp01_mobilenetv3": ("MobileNet-V3-S", "CNN"),
    "exp01_shufflenet": ("ShuffleNet-V2", "CNN"),
    "exp01_regnet": ("RegNet-Y-400MF", "CNN"),
    "exp01_vit": ("ViT-Small", "Transformer"),
    "exp01_vittiny": ("ViT-Tiny", "Transformer"),
    "exp01_mlpmixer": ("MLP-Mixer", "MLP"),
}


def load_topology_aggregated(result_dir):
    """Load topology data, aggregating across multiple slices if available.

    Looks for topology_summary_run*.json files first (multi-slice).
    Falls back to topology_summary.json (single slice).

    Returns dict with metric means, stds, and n_slices.
    """
    topo_dir = os.path.join(result_dir, "topology")

    # Check for multi-slice runs (run1..run4 files)
    run_files = sorted(glob.glob(os.path.join(topo_dir, "topology_summary_run*.json")))

    if run_files:
        # Multi-slice: aggregate across ALL slices including the default (no suffix)
        all_runs = []
        default_path = os.path.join(topo_dir, "topology_summary.json")
        if os.path.exists(default_path):
            with open(default_path) as f:
                all_runs.append(json.load(f))
        for rf in run_files:
            with open(rf) as f:
                all_runs.append(json.load(f))

        # Compute mean/std for each numeric metric
        aggregated = {"n_slices": len(all_runs)}
        metric_keys = ["H0", "H1", "H0_count", "H1_count", "H0_max_lifetime", "H1_max_lifetime",
                       "hessian_trace_mean", "max_eigenvalue", "fisher_trace",
                       "max_barrier", "max_barrier_normalized", "loss_min", "loss_max"]

        for key in metric_keys:
            vals = [r.get(key) for r in all_runs if r.get(key) is not None]
            if vals:
                aggregated[key] = float(np.mean(vals))
                aggregated[f"{key}_std"] = float(np.std(vals))
            else:
                aggregated[key] = None

        # Copy non-numeric fields from first run
        aggregated["checkpoint_accuracy"] = all_runs[0].get("checkpoint_accuracy")
        aggregated["landscape_seeds"] = [r.get("landscape_seed") for r in all_runs]

        # Validate seed uniqueness (detect old seed bug)
        seeds = [s for s in aggregated["landscape_seeds"] if s is not None]
        if len(seeds) > 1 and len(set(seeds)) < len(seeds):
            unique_pct = len(set(seeds)) / len(seeds) * 100
            print(f"  WARNING: {result_dir} has duplicate landscape seeds "
                  f"({len(set(seeds))}/{len(seeds)} unique = {unique_pct:.0f}%). "
                  f"Slices may be identical (seed bug). Re-run Phase 2.")
            aggregated["seed_bug_detected"] = True
        else:
            aggregated["seed_bug_detected"] = False

        return aggregated

    # Single slice fallback
    topo_path = os.path.join(topo_dir, "topology_summary.json")
    if not os.path.exists(topo_path):
        return None

    with open(topo_path) as f:
        data = json.load(f)
    data["n_slices"] = 1
    return data


def load_topology_per_slice(result_dir, prefix="topology_summary"):
    """Load per-slice topology data (not aggregated).

    Returns list of dicts, one per slice. Used for slice robustness diagnostics.
    """
    topo_dir = os.path.join(result_dir, "topology")
    slices = []

    # Default file (slice 0)
    default_path = os.path.join(topo_dir, f"{prefix}.json")
    if os.path.exists(default_path):
        with open(default_path) as f:
            slices.append(json.load(f))

    # Run files (slices 1-4)
    run_files = sorted(glob.glob(os.path.join(topo_dir, f"{prefix}_run*.json")))
    for rf in run_files:
        with open(rf) as f:
            slices.append(json.load(f))

    return slices


def load_cubical_aggregated(result_dir):
    """Load cubical PH data, aggregating across slices if available."""
    topo_dir = os.path.join(result_dir, "topology")
    run_files = sorted(glob.glob(os.path.join(topo_dir, "cubical_summary_run*.json")))
    default_path = os.path.join(topo_dir, "cubical_summary.json")

    all_runs = []
    if os.path.exists(default_path):
        with open(default_path) as f:
            all_runs.append(json.load(f))
    for rf in run_files:
        with open(rf) as f:
            all_runs.append(json.load(f))

    if not all_runs:
        return {}

    result = {}
    for key in ["H0", "H1", "H0_count", "H1_count"]:
        vals = [r.get(key) for r in all_runs if r.get(key) is not None]
        if vals:
            result[f"{key}_cubical"] = float(np.mean(vals))
            result[f"{key}_cubical_std"] = float(np.std(vals))
    return result


def compute_early_aurc(forget_data, max_step=500):
    """Area under retention curve from step 0 to max_step (trapezoidal, normalized)."""
    initial_acc = forget_data["initial_task_a_acc"]
    if initial_acc == 0:
        return 0.0

    points = [(p["step"], p["task_a_acc"] / initial_acc) for p in forget_data["curve"]
              if p["step"] <= max_step]
    if len(points) < 2:
        return None
    points.sort()
    auc = 0.0
    for i in range(1, len(points)):
        dt = points[i][0] - points[i - 1][0]
        avg_ret = (points[i][1] + points[i - 1][1]) / 2
        auc += avg_ret * dt
    return auc / max_step if max_step > 0 else 0.0


def compute_retention_ratio(forget_data, step):
    """Get retention ratio (acc/initial) at a specific step."""
    initial_acc = forget_data["initial_task_a_acc"]
    if initial_acc == 0:
        return None
    for point in forget_data["curve"]:
        if point["step"] == step:
            return point["task_a_acc"] / initial_acc
    return None


def slice_robustness_diagnostics(result_dirs, all_data):
    """Run slice robustness diagnostics when multi-slice data is available.

    Includes: Kruskal-Wallis, per-slice Spearman (WRN), pairwise ordering, Cohen's d.
    """
    # Check if any architecture has multiple slices
    has_multi = any(d.get("n_slices", 1) > 1 for d in all_data)
    if not has_multi:
        return {}

    print(f"\n{'=' * 70}")
    print(f"SLICE ROBUSTNESS DIAGNOSTICS")
    print(f"{'=' * 70}")

    # Load per-slice H0 for each architecture
    per_arch_h0 = {}
    for rdir, d in zip(result_dirs, all_data):
        slices = load_topology_per_slice(rdir)
        if len(slices) > 1:
            h0_vals = [s.get("H0", 0) for s in slices if s.get("H0") is not None]
            if h0_vals:
                per_arch_h0[d["arch_name"]] = h0_vals

    if len(per_arch_h0) < 2:
        print("  Not enough multi-slice architectures for diagnostics.")
        return {}

    # Kruskal-Wallis: H0 differs across architectures?
    groups = list(per_arch_h0.values())
    if all(len(g) > 1 for g in groups):
        h_stat, kw_p = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis (H0 across architectures): H={h_stat:.2f}, p={kw_p:.6f}")
        if kw_p < 0.05:
            print(f"  Between-architecture H0 differences exceed within-slice noise.")
        else:
            print(f"  WARNING: Within-slice noise comparable to between-architecture differences.")
    else:
        h_stat, kw_p = None, None

    # Per-slice Spearman for WRN ladder
    wrn_archs = [d for d in all_data if d["arch_class"] == "WRN-ladder"]
    wrn_archs_sorted = sorted(wrn_archs, key=lambda d: d.get("num_params", 0) or 0)
    wrn_names = [d["arch_name"] for d in wrn_archs_sorted]
    wrn_per_slice = [per_arch_h0.get(name) for name in wrn_names]

    slice_rhos = []
    pairwise_ordering = {}
    cohens_d_results = {}

    if all(s is not None for s in wrn_per_slice):
        n_slices = min(len(s) for s in wrn_per_slice)
        width_ranks = np.arange(len(wrn_names))  # increasing width

        print(f"\n  Per-slice Spearman (WRN ladder, {n_slices} slices):")
        for s in range(n_slices):
            h0_this_slice = [wrn_per_slice[w][s] for w in range(len(wrn_names))]
            rho_s, _ = stats.spearmanr(width_ranks, h0_this_slice)
            slice_rhos.append(rho_s)
            print(f"    Slice {s}: rho(width, H0) = {rho_s:.4f}")

        if slice_rhos:
            rhos = np.array(slice_rhos)
            print(f"    Mean rho: {rhos.mean():.4f} +/- {rhos.std():.4f}")
            print(f"    Min: {rhos.min():.4f}, Max: {rhos.max():.4f}")
            n_negative = np.sum(rhos < 0)
            print(f"    Negative (expected): {n_negative}/{len(rhos)}")

        # Pairwise ordering probability for adjacent widths
        print(f"\n  Pairwise ordering probability (adjacent WRN widths):")
        for i in range(len(wrn_names) - 1):
            a_vals = np.array(wrn_per_slice[i][:n_slices])
            b_vals = np.array(wrn_per_slice[i + 1][:n_slices])
            prob = np.mean(a_vals > b_vals)
            pair_key = f"{wrn_names[i]} > {wrn_names[i+1]}"
            pairwise_ordering[pair_key] = float(prob)
            print(f"    P({pair_key}): {prob:.2f}")

        # Cohen's d for adjacent widths
        print(f"\n  Cohen's d (adjacent WRN widths):")
        for i in range(len(wrn_names) - 1):
            a_vals = np.array(wrn_per_slice[i][:n_slices])
            b_vals = np.array(wrn_per_slice[i + 1][:n_slices])
            pooled_std = np.sqrt((a_vals.var() + b_vals.var()) / 2)
            d_val = (a_vals.mean() - b_vals.mean()) / pooled_std if pooled_std > 0 else float('inf')
            pair_key = f"{wrn_names[i]} vs {wrn_names[i+1]}"
            cohens_d_results[pair_key] = float(d_val)
            size = "large" if abs(d_val) > 0.8 else ("medium" if abs(d_val) > 0.5 else "small")
            print(f"    {pair_key}: d = {d_val:.2f} ({size})")

    results = {
        "kruskal_wallis_h": float(h_stat) if h_stat is not None else None,
        "kruskal_wallis_p": float(kw_p) if kw_p is not None else None,
        "per_slice_rhos": [float(r) for r in slice_rhos] if slice_rhos else None,
        "pairwise_ordering": pairwise_ordering,
        "cohens_d": cohens_d_results,
    }
    return results


def load_displacement_metrics(result_dir):
    """Load displacement analysis metrics if available."""
    disp_path = os.path.join(result_dir, "displacement", "displacement_summary.json")
    if not os.path.exists(disp_path):
        return {}

    with open(disp_path) as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    return {
        "curvature_ratio": metrics.get("curvature_ratio"),
        "basin_fraction": metrics.get("basin_fraction"),
        "out_of_plane_ratio": metrics.get("out_of_plane_ratio"),
        "total_displacement": metrics.get("total_displacement"),
    }


def count_model_params(result_dir):
    """Count model parameters from checkpoint file."""
    ckpt_path = os.path.join(result_dir, "checkpoints", "task_a_best.pt")
    if not os.path.exists(ckpt_path):
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        return sum(v.numel() for v in state_dict.values())
    except Exception:
        return None


def load_experiment_data(result_dir):
    """Load topology summary and forgetting curve from a single experiment run."""
    topo = load_topology_aggregated(result_dir)
    if topo is None:
        raise FileNotFoundError(f"No topology summary in {result_dir}")

    forget_path = os.path.join(result_dir, "forgetting", "forgetting_curve.json")
    with open(forget_path) as f:
        forget = json.load(f)

    # Merge displacement metrics
    disp = load_displacement_metrics(result_dir)
    topo.update(disp)

    return topo, forget


def compute_retention_at_step(forget_data, step):
    """Get Task A accuracy at a specific training step."""
    for point in forget_data["curve"]:
        if point["step"] == step:
            return point["task_a_acc"]
    return None


def compute_forgetting_auc(forget_data):
    """Compute area under the retention curve (normalized).

    Higher AUC = more retention across the full forgetting trajectory.
    Normalized by max_step so AUC is in [0, 1] range.
    """
    initial_acc = forget_data["initial_task_a_acc"]
    if initial_acc == 0:
        return 0.0

    curve = forget_data["curve"]
    auc = 0.0
    prev_step = 0
    max_step = 0
    for point in curve:
        if point["step"] > 0:
            step_retention = point["task_a_acc"] / initial_acc
            auc += step_retention * (point["step"] - prev_step)
            prev_step = point["step"]
            max_step = point["step"]

    return auc / max_step if max_step > 0 else 0.0


def partial_correlation(x, y, z):
    """Compute partial Spearman correlation between x and y, controlling for z.

    Uses the standard formula:
    r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))
    where r values are Spearman rank correlations.
    """
    r_xy, _ = stats.spearmanr(x, y)
    r_xz, _ = stats.spearmanr(x, z)
    r_yz, _ = stats.spearmanr(y, z)

    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-10:
        return float('nan'), float('nan')

    r_partial = (r_xy - r_xz * r_yz) / denom

    # Approximate p-value using t-distribution
    n = len(x)
    df = n - 3  # degrees of freedom for partial correlation
    if df <= 0:
        return float(r_partial), float('nan')

    t_stat = r_partial * np.sqrt(df / (1 - r_partial**2 + 1e-10))
    p_value = 2 * stats.t.sf(abs(t_stat), df)

    return float(r_partial), float(p_value)


def single_run_analysis(result_dir):
    """Analyze a single experiment run."""
    topo, forget = load_experiment_data(result_dir)

    print("=" * 60)
    print("SINGLE RUN ANALYSIS")
    print("=" * 60)

    n_slices = topo.get("n_slices", 1)
    if n_slices > 1:
        print(f"\n  Aggregated across {n_slices} landscape slices")

    print(f"\nTopological Features (Task A converged model):")
    for dim in range(3):
        key = f"H{dim}"
        if key in topo and topo[key] is not None:
            std_key = f"{key}_std"
            std_str = f" ± {topo[std_key]:.4f}" if std_key in topo else ""
            print(f"  {key} total persistence:   {topo[key]:.4f}{std_str}")
            print(f"  {key} feature count:       {topo.get(f'{key}_count', 0)}")
            if f"{key}_max_lifetime" in topo:
                print(f"  {key} max lifetime:        {topo[f'{key}_max_lifetime']:.4f}")

    # Baseline metrics
    baseline_keys = ["hessian_trace_mean", "max_eigenvalue", "fisher_trace", "max_barrier", "max_barrier_normalized"]
    has_baseline = any(k in topo and topo[k] is not None for k in baseline_keys)
    if has_baseline:
        print(f"\nBaseline Geometry Metrics:")
        for key in baseline_keys:
            val = topo.get(key)
            if val is not None:
                print(f"  {key}: {val:.4f}")

    # Displacement metrics
    disp_keys = ["curvature_ratio", "basin_fraction", "out_of_plane_ratio", "total_displacement"]
    has_disp = any(k in topo and topo[k] is not None for k in disp_keys)
    if has_disp:
        print(f"\nDisplacement Metrics:")
        for key in disp_keys:
            val = topo.get(key)
            if val is not None:
                print(f"  {key}: {val:.4f}")

    print(f"\nForgetting Curve:")
    print(f"  Initial Task A accuracy: {forget['initial_task_a_acc']:.1%}")
    for point in forget["curve"]:
        if point["step"] > 0:
            retention = point["task_a_acc"] / forget["initial_task_a_acc"]
            print(f"  Step {point['step']:>6}: Task A = {point['task_a_acc']:.1%}, "
                  f"Retention = {retention:.1%}, "
                  f"Task B = {point.get('task_b_acc', 0):.1%}")

    auc = compute_forgetting_auc(forget)
    print(f"\n  Forgetting AUC (normalized): {auc:.4f}")

    ret_100 = compute_retention_at_step(forget, 100)
    if ret_100 is not None:
        retention_ratio = ret_100 / forget["initial_task_a_acc"]
        print(f"  Retention ratio at step 100: {retention_ratio:.1%}")
        print(f"  H0 total persistence: {topo.get('H0', 0):.4f}")


def cross_architecture_analysis(result_dirs):
    """Correlate ALL metrics with forgetting across architectures."""
    print("=" * 70)
    print("CROSS-ARCHITECTURE CORRELATION ANALYSIS")
    print("=" * 70)

    # Collect data from all architectures
    all_data = []
    for rdir in result_dirs:
        try:
            topo, forget = load_experiment_data(rdir)
        except FileNotFoundError as e:
            print(f"  Skipping {rdir}: {e}")
            continue

        ret_100 = compute_retention_at_step(forget, 100)
        if ret_100 is None:
            last_point = forget["curve"][-1]
            ret_100 = last_point["task_a_acc"]
            print(f"  WARNING: step 100 not found for {rdir}, using step {last_point['step']} instead")

        retention = ret_100 / forget["initial_task_a_acc"]
        auc = compute_forgetting_auc(forget)

        # Count params
        num_params = count_model_params(rdir)

        # Get architecture class (strip dataset suffix for lookup)
        label = os.path.basename(rdir)
        lookup_key = label
        for suffix in ("_cub200", "_resisc45", "_cifar10", "_cifar100"):
            lookup_key = lookup_key.replace(suffix, "")
        arch_name, arch_class = ARCH_CLASSES.get(lookup_key, (label, "Unknown"))

        # Load cubical PH data
        cubical = load_cubical_aggregated(rdir)

        # Compute new forgetting metrics
        early_aurc = compute_early_aurc(forget, max_step=500)
        ret_10 = compute_retention_ratio(forget, step=10)

        entry = {
            "label": label,
            "arch_name": arch_name,
            "arch_class": arch_class,
            "retention_100": retention,
            "forgetting_auc": auc,
            "early_aurc": early_aurc,
            "ret_10": ret_10,
            "accuracy": forget["initial_task_a_acc"],
            "n_slices": topo.get("n_slices", 1),
            "num_params": num_params,
        }
        # Collect all metric values (and their stds for error bars)
        for metric_key, _ in METRICS:
            if metric_key == "num_params":
                continue  # already set
            # Cubical metrics come from cubical dict
            if metric_key.endswith("_cubical"):
                entry[metric_key] = cubical.get(metric_key, None)
                std_key = f"{metric_key}_std"
                if std_key in cubical:
                    entry[std_key] = cubical[std_key]
            else:
                entry[metric_key] = topo.get(metric_key, None)
                std_key = f"{metric_key}_std"
                if std_key in topo:
                    entry[std_key] = topo[std_key]

        # Load EWC forgetting data if available
        ewc_forget_path = os.path.join(rdir, "forgetting_ewc", "forgetting_curve.json")
        if os.path.exists(ewc_forget_path):
            with open(ewc_forget_path) as f:
                ewc_forget = json.load(f)
            entry["ewc_early_aurc"] = compute_early_aurc(ewc_forget, max_step=500)
            entry["ewc_ret_10"] = compute_retention_ratio(ewc_forget, step=10)
            entry["ewc_auc"] = compute_forgetting_auc(ewc_forget)
        else:
            entry["ewc_early_aurc"] = None
            entry["ewc_ret_10"] = None
            entry["ewc_auc"] = None

        # Load SI forgetting data if available
        si_forget_path = os.path.join(rdir, "forgetting_si", "forgetting_curve.json")
        if os.path.exists(si_forget_path):
            with open(si_forget_path) as f:
                si_forget = json.load(f)
            entry["si_early_aurc"] = compute_early_aurc(si_forget, max_step=500)
            entry["si_ret_10"] = compute_retention_ratio(si_forget, step=10)
            entry["si_auc"] = compute_forgetting_auc(si_forget)
        else:
            entry["si_early_aurc"] = None
            entry["si_ret_10"] = None
            entry["si_auc"] = None

        all_data.append(entry)

    if len(all_data) < 3:
        print(f"\n  Need >= 3 architectures for correlation. Have {len(all_data)}.")
        return

    # Print summary table
    print(f"\n  n = {len(all_data)} architectures")
    has_multi_slice = any(d.get("n_slices", 1) > 1 for d in all_data)
    if has_multi_slice:
        print(f"  Multi-slice aggregation active (topology values are means across slices)")

    print(f"\n{'Architecture':>20} | {'Class':>5} | {'Params':>8} | {'Acc':>5} | {'H0':>7} | {'H1':>6} | {'AUC':>5} | {'eAURC':>6} | {'R@10':>5} | {'R@100':>6}")
    print("-" * 105)
    for d in all_data:
        def fmt(key, w=7, dec=1):
            v = d.get(key)
            if v is None:
                return f"{'N/A':>{w}}"
            return f"{v:>{w}.{dec}f}"

        params_str = f"{d['num_params']/1e6:.1f}M" if d['num_params'] else "N/A"
        eaurc = f"{d['early_aurc']:.3f}" if d.get('early_aurc') is not None else "N/A"
        r10 = f"{d['ret_10']:.1%}" if d.get('ret_10') is not None else "N/A"
        print(f"{d['arch_name']:>20} | {d['arch_class']:>5} | {params_str:>8} | {d['accuracy']:4.1%} | {fmt('H0', 7)} | {fmt('H1', 6)} | {d['forgetting_auc']:5.3f} | {eaurc:>6} | {r10:>5} | {d['retention_100']:5.1%}")

    # ─── Standard Spearman Correlations ───
    retention_vals = [d["retention_100"] for d in all_data]
    auc_vals = [d["forgetting_auc"] for d in all_data]

    # Multiple testing correction (Bonferroni)
    num_tests = len(METRICS)
    bonf_alpha = 0.05 / num_tests

    print(f"\n{'=' * 70}")
    print(f"SPEARMAN RANK CORRELATION (n={len(all_data)} architectures)")
    print(f"  Bonferroni correction: {num_tests} tests, α_adj = {bonf_alpha:.4f}")
    print(f"  Significance: ** = survives Bonferroni, * = nominal p<0.05 only")
    print(f"{'=' * 70}")
    print(f"\n{'Metric':>30} | {'ρ (ret)':>8} | {'p':>8} | {'p_bonf':>8} | {'τ (ret)':>8} | {'ρ (AUC)':>8} | {'Sig':>3} | {'n':>3}")
    print("-" * 95)

    best_metric = None
    best_rho = -1
    all_results = {}

    for metric_key, metric_name in METRICS:
        vals = [d[metric_key] for d in all_data]
        non_none = [(v, r, a) for v, r, a in zip(vals, retention_vals, auc_vals) if v is not None]

        if len(non_none) < 3:
            print(f"{metric_name:>30} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'':>3} | {len(non_none):>3}")
            continue

        m_vals, m_ret, m_auc = zip(*non_none)

        # Check for constant input
        if len(set(m_vals)) <= 1:
            print(f"{metric_name:>30} | {'const':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'const':>8} | {'':>3} | {len(non_none):>3}")
            continue

        rho_ret, p_ret = stats.spearmanr(m_vals, m_ret)
        rho_auc, p_auc = stats.spearmanr(m_vals, m_auc)
        tau_ret, tau_p_ret = stats.kendalltau(m_vals, m_ret)
        tau_auc, tau_p_auc = stats.kendalltau(m_vals, m_auc)
        p_bonf = min(p_ret * num_tests, 1.0)
        sig = "**" if p_bonf < 0.05 else ("*" if p_ret < 0.05 else "")

        print(f"{metric_name:>30} | {rho_ret:>8.4f} | {p_ret:>8.4f} | {p_bonf:>8.4f} | {tau_ret:>8.4f} | {rho_auc:>8.4f} | {sig:>3} | {len(non_none):>3}")

        all_results[metric_key] = {
            "metric_name": metric_name,
            "values": list(m_vals),
            "n_available": len(non_none),
            "rho_retention": float(rho_ret) if not np.isnan(rho_ret) else None,
            "p_retention": float(p_ret) if not np.isnan(p_ret) else None,
            "p_bonferroni_retention": float(p_bonf),
            "rho_auc": float(rho_auc) if not np.isnan(rho_auc) else None,
            "p_auc": float(p_auc) if not np.isnan(p_auc) else None,
            "kendall_tau_retention": float(tau_ret) if not np.isnan(tau_ret) else None,
            "kendall_p_retention": float(tau_p_ret) if not np.isnan(tau_p_ret) else None,
            "kendall_tau_auc": float(tau_auc) if not np.isnan(tau_auc) else None,
            "kendall_p_auc": float(tau_p_auc) if not np.isnan(tau_p_auc) else None,
        }

        if not np.isnan(rho_ret) and abs(rho_ret) > best_rho:
            best_rho = abs(rho_ret)
            best_metric = metric_name

    # ─── Partial Correlations (controlling for num_params) ───
    params_vals = [d["num_params"] for d in all_data]
    has_params = all(p is not None for p in params_vals)

    if has_params:
        print(f"\n{'=' * 70}")
        print(f"PARTIAL CORRELATION — controlling for num_params (n={len(all_data)})")
        print(f"{'=' * 70}")
        print(f"\n{'Metric':>30} | {'ρ_partial':>10} | {'p-value':>9} | {'Sig?':>4}")
        print("-" * 60)

        for metric_key, metric_name in METRICS:
            if metric_key == "num_params":
                continue
            vals = [d[metric_key] for d in all_data]
            if any(v is None for v in vals):
                # Filter to matching indices
                valid = [(v, r, p) for v, r, p in zip(vals, retention_vals, params_vals) if v is not None]
                if len(valid) < 5:
                    print(f"{metric_name:>30} | {'N/A':>10} | {'N/A':>9} | {'N/A':>4}")
                    continue
                m_vals, m_ret, m_params = zip(*valid)
            else:
                m_vals, m_ret, m_params = vals, retention_vals, params_vals

            if len(set(m_vals)) <= 1:
                continue

            rho_p, p_p = partial_correlation(m_vals, m_ret, m_params)
            sig = "YES" if not np.isnan(p_p) and p_p < 0.05 else "no"
            rho_str = f"{rho_p:.4f}" if not np.isnan(rho_p) else "N/A"
            p_str = f"{p_p:.4f}" if not np.isnan(p_p) else "N/A"
            print(f"{metric_name:>30} | {rho_str:>10} | {p_str:>9} | {sig:>4}")

            # Store in results
            if metric_key in all_results:
                all_results[metric_key]["rho_partial"] = float(rho_p) if not np.isnan(rho_p) else None
                all_results[metric_key]["p_partial"] = float(p_p) if not np.isnan(p_p) else None

    # ─── Symmetric Partial Correlations + Rank Regression ───
    if has_params:
        h1_vals_full = [d.get("H1") for d in all_data]
        if all(v is not None for v in h1_vals_full):
            h1_arr = np.array(h1_vals_full)
            params_arr = np.array(params_vals)
            ret_arr = np.array(retention_vals)

            print(f"\n{'=' * 70}")
            print(f"SYMMETRIC PARTIAL CORRELATIONS (n={len(all_data)})")
            print(f"{'=' * 70}")

            # H1 vs params collinearity
            r_h1_p, p_h1_p = stats.spearmanr(h1_arr, params_arr)
            print(f"\n  H1 vs params: rho={r_h1_p:.4f} (p={p_h1_p:.4f})")

            # Partial: params vs ret | H1
            rho_p1, pp1 = partial_correlation(params_arr, ret_arr, h1_arr)
            print(f"  Params vs ret | H1:  rho_partial={rho_p1:.4f} (p={pp1:.4f})")

            # Partial: H1 vs ret | params
            rho_p2, pp2 = partial_correlation(h1_arr, ret_arr, params_arr)
            print(f"  H1 vs ret | params:  rho_partial={rho_p2:.4f} (p={pp2:.4f})")

            # Rank regression: rank(ret) ~ rank(params) + rank(H1)
            rank_ret = stats.rankdata(ret_arr)
            rank_params = stats.rankdata(params_arr)
            rank_h1 = stats.rankdata(h1_arr)

            X = np.column_stack([np.ones(len(all_data)), rank_params, rank_h1])
            beta = np.linalg.lstsq(X, rank_ret, rcond=None)[0]
            y_hat = X @ beta
            ss_res = np.sum((rank_ret - y_hat)**2)
            ss_tot = np.sum((rank_ret - rank_ret.mean())**2)
            r_sq = 1 - ss_res / ss_tot
            mse = ss_res / (len(all_data) - 3)
            cov = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            t_stats_reg = beta / se
            p_vals_reg = 2 * stats.t.sf(np.abs(t_stats_reg), len(all_data) - 3)

            print(f"\n  Rank regression: rank(ret) ~ rank(params) + rank(H1)")
            print(f"    R-squared: {r_sq:.4f}")
            print(f"    Params:  beta={beta[1]:.3f}, se={se[1]:.3f}, p={p_vals_reg[1]:.4f}")
            print(f"    H1:      beta={beta[2]:.3f}, se={se[2]:.3f}, p={p_vals_reg[2]:.4f}")

            # Store
            all_results["symmetric_partials"] = {
                "h1_params_collinearity": float(r_h1_p),
                "params_vs_ret_controlling_h1": {"rho": float(rho_p1), "p": float(pp1)},
                "h1_vs_ret_controlling_params": {"rho": float(rho_p2), "p": float(pp2)},
                "rank_regression_r_squared": float(r_sq),
                "rank_regression_params_p": float(p_vals_reg[1]),
                "rank_regression_h1_p": float(p_vals_reg[2]),
            }

    # ─── Architecture Class Analysis ───
    classes = set(d["arch_class"] for d in all_data)
    classes_with_enough = [c for c in classes if sum(1 for d in all_data if d["arch_class"] == c) >= 3]

    if classes_with_enough:
        print(f"\n{'=' * 70}")
        print(f"WITHIN-CLASS CORRELATIONS")
        print(f"{'=' * 70}")

        for cls in sorted(classes_with_enough):
            cls_data = [d for d in all_data if d["arch_class"] == cls]
            cls_ret = [d["retention_100"] for d in cls_data]
            n_cls = len(cls_data)

            print(f"\n  {cls} (n={n_cls}): {', '.join(d['arch_name'] for d in cls_data)}")

            for metric_key, metric_name in [("H0", "H0"), ("H1", "H1"), ("num_params", "Params")]:
                cls_vals = [d[metric_key] for d in cls_data]
                valid = [(v, r) for v, r in zip(cls_vals, cls_ret) if v is not None]
                if len(valid) < 3 or len(set(v for v, _ in valid)) <= 1:
                    continue
                m_vals, m_ret = zip(*valid)
                rho, p = stats.spearmanr(m_vals, m_ret)
                sig = "*" if p < 0.05 else ""
                print(f"    {metric_name:>10}: ρ = {rho:.4f} (p={p:.4f}) {sig}")

    # ─── WRN Width Ladder Analysis ───
    wrn_data = [d for d in all_data if d["arch_class"] == "WRN-ladder"]
    if len(wrn_data) >= 4:
        print(f"\n{'=' * 70}")
        print(f"WRN WIDTH LADDER ANALYSIS (n={len(wrn_data)})")
        print(f"  The decisive test: same architecture, varying only width.")
        print(f"{'=' * 70}")

        # Sort by param count for display
        wrn_sorted = sorted(wrn_data, key=lambda d: d.get("num_params", 0) or 0)

        # Build display table with multi-slice error bars
        has_h0_std = any(d.get("H0_std") is not None for d in wrn_sorted)
        has_h1_std = any(d.get("H1_std") is not None for d in wrn_sorted)
        n_slices_list = [d.get("n_slices", 1) for d in wrn_sorted]
        max_slices = max(n_slices_list)
        if max_slices > 1:
            print(f"  Multi-slice aggregation: up to {max_slices} slices per model")

        header = f"  {'Model':>12} | {'Params':>8} | {'H0':>10}"
        if has_h0_std:
            header += f" | {'H0 std':>8}"
        header += f" | {'H1':>8}"
        if has_h1_std:
            header += f" | {'H1 std':>8}"
        header += f" | {'Ret@100':>7} | {'Slices':>6}"
        print(f"\n{header}")
        print(f"  {'-' * len(header)}")
        for d in wrn_sorted:
            params_str = f"{d['num_params']/1e6:.1f}M" if d.get('num_params') else "N/A"
            h0_str = f"{d.get('H0', 0):.1f}"
            h1_str = f"{d.get('H1', 0):.4f}"
            line = f"  {d['arch_name']:>12} | {params_str:>8} | {h0_str:>10}"
            if has_h0_std:
                h0_std = d.get("H0_std")
                line += f" | {h0_std:>8.1f}" if h0_std is not None else f" | {'N/A':>8}"
            line += f" | {h1_str:>8}"
            if has_h1_std:
                h1_std = d.get("H1_std")
                line += f" | {h1_std:>8.4f}" if h1_std is not None else f" | {'N/A':>8}"
            line += f" | {d['retention_100']:6.1%} | {d.get('n_slices', 1):>6}"
            print(line)

        wrn_h0 = [d.get("H0", 0) or 0 for d in wrn_sorted]
        wrn_h1 = [d.get("H1", 0) or 0 for d in wrn_sorted]
        wrn_ret = [d["retention_100"] for d in wrn_sorted]
        wrn_params = [d.get("num_params", 0) or 0 for d in wrn_sorted]

        # ── H0 Analysis (the key question: is H0 monotonicity robust?) ──
        print(f"\n  --- H0 Analysis ---")

        # Spearman: H0 vs retention within ladder
        if len(set(wrn_h0)) > 1:
            rho_h0, p_h0 = stats.spearmanr(wrn_h0, wrn_ret)
            print(f"    H0 vs retention:     rho={rho_h0:.4f} (p={p_h0:.4f})")
        else:
            rho_h0, p_h0 = float('nan'), float('nan')
            print(f"    H0 vs retention:     constant H0, cannot compute")

        # H0 vs params (expected to be strong and negative)
        rho_h0_params, p_h0_params = stats.spearmanr(wrn_h0, wrn_params)
        print(f"    H0 vs params:        rho={rho_h0_params:.4f} (p={p_h0_params:.4f})")

        # Monotonicity check on mean H0 (sorted by increasing width/params)
        h0_diffs = [wrn_h0[i+1] - wrn_h0[i] for i in range(len(wrn_h0)-1)]
        n_decreasing = sum(1 for d in h0_diffs if d < 0)
        is_monotone = n_decreasing == len(h0_diffs)
        print(f"    H0 monotonicity:     {n_decreasing}/{len(h0_diffs)} consecutive decreases (sorted by width)")
        if is_monotone:
            print(f"    RESULT: H0 monotonically decreases with width in the mean.")
        else:
            n_increasing = sum(1 for d in h0_diffs if d > 0)
            print(f"    RESULT: H0 trend is NOT strictly monotonic ({n_increasing} increases detected).")

        # Error bar overlap check (if multi-slice data available)
        overlaps = None  # track for verdict
        if has_h0_std and max_slices > 1:
            print(f"\n    Multi-slice stability:")
            overlaps = 0
            pairs = 0
            for i in range(len(wrn_sorted)):
                for j in range(i+1, len(wrn_sorted)):
                    h0_i, std_i = wrn_sorted[i].get("H0", 0), wrn_sorted[i].get("H0_std", 0) or 0
                    h0_j, std_j = wrn_sorted[j].get("H0", 0), wrn_sorted[j].get("H0_std", 0) or 0
                    # Check 1-sigma overlap
                    sep = abs(h0_i - h0_j)
                    combined_std = std_i + std_j
                    if combined_std > 0 and sep < combined_std:
                        overlaps += 1
                        print(f"      WARNING: {wrn_sorted[i]['arch_name']} and {wrn_sorted[j]['arch_name']} "
                              f"H0 error bars overlap (sep={sep:.1f}, combined_std={combined_std:.1f})")
                    pairs += 1
            if overlaps == 0:
                print(f"      All {pairs} pairwise H0 values are separated beyond 1-sigma. Ordering is robust.")
            else:
                print(f"      {overlaps}/{pairs} pairs have overlapping H0 error bars. Ordering may not be robust.")

        # Partial H0 | params within ladder
        if len(wrn_data) >= 5 and len(set(wrn_params)) > 1 and len(set(wrn_h0)) > 1:
            rho_h0_part, p_h0_part = partial_correlation(wrn_h0, wrn_ret, wrn_params)
            print(f"    H0 vs ret | params:  rho_partial={rho_h0_part:.4f} (p={p_h0_part:.4f})")
        else:
            rho_h0_part, p_h0_part = float('nan'), float('nan')

        # ── H1 Analysis ──
        print(f"\n  --- H1 Analysis ---")
        if len(set(wrn_h1)) > 1:
            rho_h1, p_h1 = stats.spearmanr(wrn_h1, wrn_ret)
            rho_params, p_params = stats.spearmanr(wrn_params, wrn_ret)
            print(f"    H1 vs retention:     rho={rho_h1:.4f} (p={p_h1:.4f})")
            print(f"    Params vs retention:  rho={rho_params:.4f} (p={p_params:.4f})")

            if len(wrn_data) >= 5 and len(set(wrn_params)) > 1:
                rho_part, p_part = partial_correlation(wrn_h1, wrn_ret, wrn_params)
                print(f"    H1 vs ret | params:  rho_partial={rho_part:.4f} (p={p_part:.4f})")
        else:
            rho_h1, p_h1 = float('nan'), float('nan')
            rho_params, p_params = stats.spearmanr(wrn_params, wrn_ret)
            rho_part, p_part = float('nan'), float('nan')
            print(f"    H1 is zero/constant across the ladder.")
            print(f"    This suggests H1 is architecture-family specific, not scale-driven.")
            print(f"    Params vs retention:  rho={rho_params:.4f} (p={p_params:.4f})")

        # ── Verdict ──
        print(f"\n  --- VERDICT ---")
        # H0 verdict
        if not np.isnan(rho_h0) and not np.isnan(p_h0):
            if is_monotone and overlaps is not None and overlaps == 0:
                print(f"  H0: Monotonic decrease with width is ROBUST across {max_slices} slices.")
                print(f"       But H0 tracks params perfectly (rho={rho_h0_params:.4f}), so cannot separate the two.")
            elif is_monotone:
                print(f"  H0: Monotonic decrease with width, but SINGLE-SLICE only.")
                print(f"       Cannot confirm robustness until multi-slice runs complete.")
            else:
                print(f"  H0: Trend is not monotonic. H0 ordering is not stable across widths.")
        # H1 verdict
        h1_all_zero = all(h == 0 for h in wrn_h1)
        if h1_all_zero:
            print(f"  H1: Collapsed to zero across the entire WRN ladder.")
            print(f"       H1 topology is architecture-motif dependent, not scale-driven.")
        elif len(set(wrn_h1)) <= 1:
            print(f"  H1: Constant across the ladder (no discriminative power).")
        else:
            h1_sig = not np.isnan(p_part) and p_part < 0.05
            if h1_sig:
                print(f"  H1: Carries independent signal beyond scale within WRN ladder.")
            else:
                print(f"  H1: Does NOT carry independent signal within WRN ladder.")

        # Store WRN ladder results
        all_results["wrn_ladder"] = {
            "n": len(wrn_data),
            "architectures": [d["arch_name"] for d in wrn_sorted],
            "h0_values": wrn_h0,
            "h0_stds": [d.get("H0_std") for d in wrn_sorted],
            "h1_values": wrn_h1,
            "h1_stds": [d.get("H1_std") for d in wrn_sorted],
            "retention_values": wrn_ret,
            "param_counts": wrn_params,
            "n_slices": n_slices_list,
            "h0_monotonic": is_monotone,
            "h0_vs_retention_rho": float(rho_h0) if not np.isnan(rho_h0) else None,
            "h0_vs_retention_p": float(p_h0) if not np.isnan(p_h0) else None,
            "h0_vs_params_rho": float(rho_h0_params),
            "h0_partial_rho": float(rho_h0_part) if not np.isnan(rho_h0_part) else None,
            "h0_partial_p": float(p_h0_part) if not np.isnan(p_h0_part) else None,
            "h1_all_zero": h1_all_zero,
        }

    # ─── Slice Robustness Diagnostics ───
    robustness = slice_robustness_diagnostics(result_dirs, all_data)
    if robustness:
        all_results["slice_robustness"] = robustness

    # ─── Cubical vs Ripser Comparison ───
    ripser_h0 = [d.get("H0") for d in all_data]
    cubical_h0 = [d.get("H0_cubical") for d in all_data]
    ripser_h1 = [d.get("H1") for d in all_data]
    cubical_h1 = [d.get("H1_cubical") for d in all_data]

    has_cubical = any(v is not None for v in cubical_h0)
    if has_cubical:
        print(f"\n{'=' * 70}")
        print(f"CUBICAL vs RIPSER COMPARISON")
        print(f"{'=' * 70}")

        # H0 agreement
        valid_h0 = [(r, c) for r, c in zip(ripser_h0, cubical_h0)
                     if r is not None and c is not None]
        if len(valid_h0) >= 3:
            r_vals, c_vals = zip(*valid_h0)
            rho_h0_rc, p_h0_rc = stats.spearmanr(r_vals, c_vals)
            print(f"\n  H0 Ripser vs H0 Cubical: rho={rho_h0_rc:.4f} (p={p_h0_rc:.4f}, n={len(valid_h0)})")
            all_results["cubical_vs_ripser_H0_rho"] = float(rho_h0_rc)
            all_results["cubical_vs_ripser_H0_p"] = float(p_h0_rc)

        # H1 agreement
        valid_h1 = [(r, c) for r, c in zip(ripser_h1, cubical_h1)
                     if r is not None and c is not None]
        if len(valid_h1) >= 3:
            r_vals, c_vals = zip(*valid_h1)
            if len(set(r_vals)) > 1 and len(set(c_vals)) > 1:
                rho_h1_rc, p_h1_rc = stats.spearmanr(r_vals, c_vals)
                print(f"  H1 Ripser vs H1 Cubical: rho={rho_h1_rc:.4f} (p={p_h1_rc:.4f}, n={len(valid_h1)})")
                all_results["cubical_vs_ripser_H1_rho"] = float(rho_h1_rc)
                all_results["cubical_vs_ripser_H1_p"] = float(p_h1_rc)
            else:
                print(f"  H1 Ripser vs H1 Cubical: one or both constant, cannot correlate")

        # Cubical metrics vs retention
        print(f"\n  Cubical metrics vs forgetting:")
        for ckey, cname in [("H0_cubical", "H0 Cubical"), ("H1_cubical", "H1 Cubical")]:
            c_vals_all = [d.get(ckey) for d in all_data]
            valid = [(v, r) for v, r in zip(c_vals_all, retention_vals) if v is not None]
            if len(valid) >= 3:
                cv, cr = zip(*valid)
                if len(set(cv)) > 1:
                    rho_c, p_c = stats.spearmanr(cv, cr)
                    sig = "*" if p_c < 0.05 else ""
                    print(f"    {cname} vs ret@100: rho={rho_c:.4f} (p={p_c:.4f}) {sig}")

    # ─── EWC Benefit Analysis ───
    ewc_data = [(d, d.get("ewc_early_aurc"), d.get("early_aurc"))
                for d in all_data
                if d.get("ewc_early_aurc") is not None and d.get("early_aurc") is not None]
    if len(ewc_data) >= 3:
        print(f"\n{'=' * 70}")
        print(f"EWC BENEFIT ANALYSIS (n={len(ewc_data)})")
        print(f"{'=' * 70}")

        # EWC benefit = ewc_aurc - naive_aurc (positive = EWC helped)
        ewc_benefit = [e[1] - e[2] for e in ewc_data]
        print(f"\n  {'Architecture':>20} | {'Naive AURC':>10} | {'EWC AURC':>10} | {'Benefit':>10}")
        print(f"  {'-' * 60}")
        for d, ewc_aurc, naive_aurc in ewc_data:
            benefit = ewc_aurc - naive_aurc
            print(f"  {d['arch_name']:>20} | {naive_aurc:>10.4f} | {ewc_aurc:>10.4f} | {benefit:>+10.4f}")

        # Correlate topology with EWC benefit
        print(f"\n  Topology vs EWC benefit:")
        for tkey, tname in [("H0", "H0"), ("H1", "H1"), ("H0_cubical", "H0 Cubical"), ("num_params", "Params")]:
            t_vals = [d.get(tkey) for d, _, _ in ewc_data]
            valid = [(t, b) for t, b in zip(t_vals, ewc_benefit) if t is not None]
            if len(valid) >= 3:
                tv, bv = zip(*valid)
                if len(set(tv)) > 1:
                    rho_eb, p_eb = stats.spearmanr(tv, bv)
                    sig = "*" if p_eb < 0.05 else ""
                    print(f"    {tname:>15} vs EWC benefit: rho={rho_eb:.4f} (p={p_eb:.4f}) {sig}")

        all_results["ewc_benefit"] = {
            "n": len(ewc_data),
            "architectures": [d["arch_name"] for d, _, _ in ewc_data],
            "benefits": ewc_benefit,
        }

    # ─── SI Benefit Analysis ───
    si_data = [(d, d.get("si_early_aurc"), d.get("early_aurc"))
               for d in all_data
               if d.get("si_early_aurc") is not None and d.get("early_aurc") is not None]
    if len(si_data) >= 3:
        print(f"\n{'=' * 70}")
        print(f"SI BENEFIT ANALYSIS (n={len(si_data)})")
        print(f"{'=' * 70}")

        si_benefit = [s[1] - s[2] for s in si_data]
        print(f"\n  {'Architecture':>20} | {'Naive AURC':>10} | {'SI AURC':>10} | {'Benefit':>10}")
        print(f"  {'-' * 60}")
        for d, si_aurc, naive_aurc in si_data:
            benefit = si_aurc - naive_aurc
            print(f"  {d['arch_name']:>20} | {naive_aurc:>10.4f} | {si_aurc:>10.4f} | {benefit:>+10.4f}")

        # Correlate topology with SI benefit
        print(f"\n  Topology vs SI benefit:")
        for tkey, tname in [("H0", "H0"), ("H1", "H1"), ("H0_cubical", "H0 Cubical"), ("num_params", "Params")]:
            t_vals = [d.get(tkey) for d, _, _ in si_data]
            valid = [(t, b) for t, b in zip(t_vals, si_benefit) if t is not None]
            if len(valid) >= 3:
                tv, bv = zip(*valid)
                if len(set(tv)) > 1:
                    rho_sb, p_sb = stats.spearmanr(tv, bv)
                    sig = "*" if p_sb < 0.05 else ""
                    print(f"    {tname:>15} vs SI benefit: rho={rho_sb:.4f} (p={p_sb:.4f}) {sig}")

        all_results["si_benefit"] = {
            "n": len(si_data),
            "architectures": [d["arch_name"] for d, _, _ in si_data],
            "benefits": si_benefit,
        }

    # ─── Permutation Test ───
    print(f"\n{'=' * 70}")
    print(f"PERMUTATION TEST (10,000 shuffles)")
    print(f"{'=' * 70}")
    print(f"\n{'Metric':>30} | {'Obs ρ':>8} | {'Perm p':>8} | {'Tail %':>7}")
    print("-" * 65)

    n_perms = 10000
    rng = np.random.RandomState(42)
    for metric_key, metric_name in METRICS:
        vals = [d[metric_key] for d in all_data]
        non_none = [(v, r) for v, r in zip(vals, retention_vals) if v is not None]
        if len(non_none) < 3:
            continue
        m_vals, m_ret = zip(*non_none)
        m_vals = np.array(m_vals)
        m_ret = np.array(m_ret)

        if len(set(m_vals)) <= 1:
            continue

        obs_rho, _ = stats.spearmanr(m_vals, m_ret)
        # Shuffle retention labels and recompute rho
        perm_rhos = np.empty(n_perms)
        for pi in range(n_perms):
            shuffled = rng.permutation(m_ret)
            perm_rhos[pi], _ = stats.spearmanr(m_vals, shuffled)

        # Two-tailed: fraction of permutations with |rho| >= |observed|
        perm_p = np.mean(np.abs(perm_rhos) >= abs(obs_rho))
        tail_pct = perm_p * 100

        print(f"{metric_name:>30} | {obs_rho:>8.4f} | {perm_p:>8.4f} | {tail_pct:>6.1f}%")

        # Store in results
        if metric_key in all_results:
            all_results[metric_key]["perm_p_retention"] = float(perm_p)

    # Also test AURC
    print(f"\n  Against AURC:")
    print(f"  {'Metric':>28} | {'Obs ρ':>8} | {'Perm p':>8} | {'Tail %':>7}")
    print(f"  {'-' * 61}")
    for metric_key, metric_name in METRICS:
        vals = [d[metric_key] for d in all_data]
        non_none = [(v, a) for v, a in zip(vals, auc_vals) if v is not None]
        if len(non_none) < 3:
            continue
        m_vals, m_auc = zip(*non_none)
        m_vals = np.array(m_vals)
        m_auc = np.array(m_auc)

        if len(set(m_vals)) <= 1:
            continue

        obs_rho, _ = stats.spearmanr(m_vals, m_auc)
        perm_rhos = np.empty(n_perms)
        for pi in range(n_perms):
            shuffled = rng.permutation(m_auc)
            perm_rhos[pi], _ = stats.spearmanr(m_vals, shuffled)

        perm_p = np.mean(np.abs(perm_rhos) >= abs(obs_rho))
        tail_pct = perm_p * 100
        print(f"  {metric_name:>28} | {obs_rho:>8.4f} | {perm_p:>8.4f} | {tail_pct:>6.1f}%")

        if metric_key in all_results:
            all_results[metric_key]["perm_p_auc"] = float(perm_p)

    # ─── Summary ───
    print(f"\n{'─' * 70}")
    if best_metric:
        print(f"  Best predictor of forgetting: {best_metric} (|ρ| = {best_rho:.4f})")

        n = len(all_data)
        print(f"  Critical |ρ| for p<0.05 at n={n}: see p-values above")

        # Check H0 vs best
        h0_result = all_results.get("H0")
        h1_result = all_results.get("H1")

        # Report H0
        if h0_result and h0_result["rho_retention"] is not None:
            h0_p = h0_result["p_retention"]
            h0_pb = h0_result.get("p_bonferroni_retention", h0_p * num_tests)
            h0_tau = h0_result.get("kendall_tau_retention")
            tau_str = f", τ={h0_tau:.4f}" if h0_tau is not None else ""
            if h0_pb < 0.05:
                print(f"\n  ★ H0 SIGNIFICANT (Bonferroni): ρ={h0_result['rho_retention']:.4f}, p={h0_p:.4f}, p_bonf={h0_pb:.4f}{tau_str}")
            elif h0_p < 0.05:
                print(f"\n  ◐ H0 nominal p<0.05 but NOT Bonferroni: ρ={h0_result['rho_retention']:.4f}, p={h0_p:.4f}, p_bonf={h0_pb:.4f}{tau_str}")
            else:
                print(f"\n  ○ H0 NOT SIGNIFICANT: ρ={h0_result['rho_retention']:.4f}, p={h0_p:.4f}{tau_str}")

        # Report H1
        if h1_result and h1_result["rho_retention"] is not None:
            h1_p = h1_result["p_retention"]
            h1_pb = h1_result.get("p_bonferroni_retention", h1_p * num_tests)
            h1_tau = h1_result.get("kendall_tau_retention")
            tau_str = f", τ={h1_tau:.4f}" if h1_tau is not None else ""
            if h1_pb < 0.05:
                print(f"  ★ H1 SIGNIFICANT (Bonferroni): ρ={h1_result['rho_retention']:.4f}, p={h1_p:.4f}, p_bonf={h1_pb:.4f}{tau_str}")
            elif h1_p < 0.05:
                print(f"  ◐ H1 nominal p<0.05 but NOT Bonferroni: ρ={h1_result['rho_retention']:.4f}, p={h1_p:.4f}, p_bonf={h1_pb:.4f}{tau_str}")
            else:
                print(f"  ○ H1 NOT SIGNIFICANT: ρ={h1_result['rho_retention']:.4f}, p={h1_p:.4f}{tau_str}")

        # Report params baseline
        params_result = all_results.get("num_params")
        if params_result and params_result["rho_retention"] is not None:
            pr = params_result["rho_retention"]
            pp = params_result["p_retention"]
            print(f"\n  BASELINE (num_params only): ρ={pr:.4f}, p={pp:.4f}")
            if pp < 0.05:
                print(f"  ⚠ WARNING: Parameter count alone predicts retention. Topology must beat this.")
            else:
                print(f"  ✓ Parameter count alone does NOT predict retention. Topology adds value.")

    # Save full results
    early_aurc_vals = [d.get("early_aurc") for d in all_data]
    ret_10_vals = [d.get("ret_10") for d in all_data]

    results = {
        "n_architectures": len(all_data),
        "architectures": [d["label"] for d in all_data],
        "arch_classes": {d["label"]: d["arch_class"] for d in all_data},
        "retention_ratios_100": retention_vals,
        "forgetting_auc_values": auc_vals,
        "early_aurc_values": early_aurc_vals,
        "ret_10_values": ret_10_vals,
        "per_architecture": all_data,
        "correlations": all_results,
        "best_metric": best_metric,
        "best_rho": float(best_rho) if best_rho > 0 else None,
        "multiple_testing": {
            "method": "Bonferroni",
            "num_tests": num_tests,
            "bonferroni_alpha": float(bonf_alpha),
        },
    }
    # Detect dataset from directory names
    first_label = os.path.basename(result_dirs[0])
    if first_label.endswith("_cub200"):
        dataset_tag = "_cub200"
    elif first_label.endswith("_resisc45"):
        dataset_tag = "_resisc45"
    else:
        dataset_tag = "_cifar100"
    out_path = os.path.join(os.path.dirname(result_dirs[0]), f"correlation_results{dataset_tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="EXP-01 Phase 4: Correlation Analysis")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--results-dirs", nargs="+", type=str, default=None,
                        help="Multiple result directories for cross-architecture analysis")
    args = parser.parse_args()

    if args.results_dirs and len(args.results_dirs) > 1:
        cross_architecture_analysis(args.results_dirs)
    elif args.config:
        cfg = load_config(args.config)
        single_run_analysis(cfg["output_dir"])
    elif args.results_dirs and len(args.results_dirs) == 1:
        single_run_analysis(args.results_dirs[0])
    else:
        parser.error("Provide --config or --results-dirs")


# Need torch for loading checkpoints
try:
    import torch
except ImportError:
    pass


if __name__ == "__main__":
    main()
