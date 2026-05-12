"""EXP-04 metric verification diagnostic (multi-step).

Compares full-softmax vs restricted-softmax retention across multiple
evaluation steps and all six cross-dataset pairs, to characterize whether
and where the two metrics diverge in actual training trajectories.

Reads from the restricted-softmax re-evaluation file, which contains BOTH
metrics in each curve entry:
    task_a_acc_full          - re-computed full-softmax (matches original
                               forgetting_curve.json, verified by
                               full_vs_recomputed_diff = 0)
    task_a_acc_restricted    - argmax over [0, K_A) logits only,
                               isolates backbone drift

Steps evaluated: 10, 100, 500, 1000
  step 10  = paper's primary retention metric (ret@10). The headline
             "does the existing Phase 4/8 analysis hold under restricted
             softmax?" question pivots on this step.
  later   = characterize divergence onset and magnitude. Empirically
            grounds the paper's claim that restricted-softmax isolates
            backbone drift while full-softmax conflates it with
            classifier-head re-routing.

Three-tier verdict, applied per step and globally (using step 10 as
paper's primary):

  ALIGNED
    rankings preserved AND magnitudes preserved (max diff < 0.02 AND
    pooled rho > 0.95 AND no per-pair rho below 0.90)

  MOSTLY_ALIGNED_GEOMETRY_DEPENDENT
    rankings largely preserved (pooled rho > 0.85), but magnitudes
    diverge systematically; either some per-pair rho < 0.90 or the
    range across per-pair rhos exceeds 0.10

  MISALIGNED
    rankings unstable (pooled rho <= 0.85), or pooled magnitudes diverge
    severely

Outputs (written under results/exp04_metric_diagnostic/):
  per_arch.json   - per (pair, arch) row with full/restricted/diff at
                    each step
  per_pair.json   - per-pair per-step n, max_diff, mean_diff, rho,
                    verdict_local
  aggregate.json  - pooled per-step stats, divergence profile across
                    steps, global verdict on paper's primary metric

Usage (on HPC):
    cd /fs1/scratch/cag1145/axiondeep-research
    python3 scripts/exp04_metric_diagnostic.py            # all 6 pairs (default)
    python3 scripts/exp04_metric_diagnostic.py PAIR       # single pair, no JSON
"""
from __future__ import annotations

import json
import sys
import glob
import datetime as _dt
from pathlib import Path

ALL_PAIRS = [
    "cifar100_to_cub200",
    "cifar100_to_resisc45",
    "cub200_to_cifar100",
    "cub200_to_resisc45",
    "resisc45_to_cifar100",
    "resisc45_to_cub200",
]

EVAL_STEPS = [10, 100, 500, 1000]
PRIMARY_STEP = 10  # paper's primary retention metric


def ret_at_step_from_restricted_file(curve_data: dict, step: int,
                                     metric_key: str) -> float | None:
    """Compute ret@step using either 'task_a_acc_full' or
    'task_a_acc_restricted' from the restricted-softmax curve file."""
    initial = curve_data.get("initial_task_a_acc")
    if not initial:
        return None
    for pt in curve_data.get("curve", []):
        if pt.get("step") == step:
            val = pt.get(metric_key)
            if val is None:
                return None
            return val / initial
    return None


def rank(xs: list[float]) -> list[float]:
    indexed = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = rank(x), rank(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) ** 0.5) * (sum((b - my) ** 2 for b in ry) ** 0.5)
    return num / den if den else None


def collect_pair(pair: str) -> tuple[list[dict], dict]:
    """Return (per-arch records for this pair, discovery diagnostics)."""
    task_a, task_b = pair.split("_to_")
    pair_dirs = sorted(glob.glob(f"results/exp01_*_{task_a}_xd_{task_b}"))
    rows: list[dict] = []
    n_total = len(pair_dirs)
    n_no_restricted = 0
    n_no_steps = 0
    for d in pair_dirs:
        arch = Path(d).name.replace("exp01_", "").replace(f"_{task_a}_xd_{task_b}", "")
        rest_path = Path(d) / "forgetting" / "forgetting_curve_restricted.json"
        if not rest_path.exists():
            n_no_restricted += 1
            continue
        rest_data = json.load(open(rest_path))
        row = {"pair": pair, "arch": arch}
        had_any_step = False
        for step in EVAL_STEPS:
            full_v = ret_at_step_from_restricted_file(rest_data, step, "task_a_acc_full")
            rest_v = ret_at_step_from_restricted_file(rest_data, step, "task_a_acc_restricted")
            if full_v is None or rest_v is None:
                row[f"full_ret_{step}"] = None
                row[f"rest_ret_{step}"] = None
                row[f"abs_diff_{step}"] = None
                continue
            had_any_step = True
            row[f"full_ret_{step}"] = full_v
            row[f"rest_ret_{step}"] = rest_v
            row[f"abs_diff_{step}"] = abs(full_v - rest_v)
        if not had_any_step:
            n_no_steps += 1
            continue
        rows.append(row)
    discovery = {
        "dirs_found": n_total,
        "dirs_missing_restricted_json": n_no_restricted,
        "dirs_with_restricted_but_no_steps": n_no_steps,
        "dirs_usable": len(rows),
    }
    return rows, discovery


def per_pair_verdict(max_diff: float, rho: float) -> str:
    if max_diff < 0.02 and rho > 0.95:
        return "ALIGNED"
    if rho > 0.85:
        return "MOSTLY_ALIGNED"
    return "MISALIGNED"


def global_verdict(pooled_rho: float, pooled_max_diff: float,
                   per_pair_rhos: list[float]) -> tuple[str, dict]:
    rho_min = min(per_pair_rhos) if per_pair_rhos else 0.0
    rho_max = max(per_pair_rhos) if per_pair_rhos else 0.0
    rho_range = rho_max - rho_min
    any_below_90 = any(r < 0.90 for r in per_pair_rhos)
    geometry_dependence_systematic = any_below_90 or rho_range > 0.10
    diagnostics = {
        "per_pair_rho_min": rho_min,
        "per_pair_rho_max": rho_max,
        "per_pair_rho_range": rho_range,
        "any_pair_below_rho_0.90": any_below_90,
        "is_geometry_dependence_systematic": geometry_dependence_systematic,
    }
    if pooled_rho > 0.95 and pooled_max_diff < 0.02 and not any_below_90:
        verdict = "ALIGNED"
    elif pooled_rho > 0.85:
        verdict = "MOSTLY_ALIGNED_GEOMETRY_DEPENDENT" if geometry_dependence_systematic else "MOSTLY_ALIGNED"
    else:
        verdict = "MISALIGNED"
    return verdict, diagnostics


def pair_step_stats(rows: list[dict], step: int) -> dict | None:
    full_vals = [r[f"full_ret_{step}"] for r in rows if r.get(f"full_ret_{step}") is not None]
    rest_vals = [r[f"rest_ret_{step}"] for r in rows if r.get(f"rest_ret_{step}") is not None]
    diffs = [r[f"abs_diff_{step}"] for r in rows if r.get(f"abs_diff_{step}") is not None]
    if not diffs or len(full_vals) != len(rest_vals) or len(full_vals) < 3:
        return None
    rho = spearman(full_vals, rest_vals)
    if rho is None:
        rho = 0.0
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    return {
        "n": len(diffs),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "spearman_rho": rho,
        "verdict_local": per_pair_verdict(max_diff, rho),
    }


def main() -> int:
    # Single-pair legacy mode (stdout only, no JSON)
    if len(sys.argv) > 1 and sys.argv[1] in ALL_PAIRS:
        pair = sys.argv[1]
        rows, disc = collect_pair(pair)
        print(f"Pair: {pair}")
        print(f"Discovery: {disc}")
        print()
        for step in EVAL_STEPS:
            s = pair_step_stats(rows, step)
            if s is None:
                print(f"  step {step}: insufficient data")
                continue
            print(f"  step {step:>5}: n={s['n']:>2}  rho={s['spearman_rho']:>+.4f}  "
                  f"max_diff={s['max_abs_diff']:.4f}  mean_diff={s['mean_abs_diff']:.4f}  "
                  f"verdict={s['verdict_local']}")
        return 0

    # Default: sweep all 6 pairs and write JSON outputs.
    out_dir = Path("results/exp04_metric_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    per_pair_summary: dict[str, dict] = {}
    discovery_summary: dict[str, dict] = {}

    for pair in ALL_PAIRS:
        rows, disc = collect_pair(pair)
        discovery_summary[pair] = disc
        per_pair_summary[pair] = {
            "n_archs_with_any_step": len(rows),
            "by_step": {},
        }
        for step in EVAL_STEPS:
            s = pair_step_stats(rows, step)
            per_pair_summary[pair]["by_step"][str(step)] = s
        all_records.extend(rows)

    # Aggregate per-step pooled stats
    aggregate_by_step: dict[str, dict] = {}
    for step in EVAL_STEPS:
        pooled_full: list[float] = []
        pooled_rest: list[float] = []
        pooled_diffs: list[float] = []
        per_pair_rhos: list[float] = []
        for pair in ALL_PAIRS:
            s = per_pair_summary[pair]["by_step"].get(str(step))
            if s is None:
                continue
            per_pair_rhos.append(s["spearman_rho"])
            for r in all_records:
                if r["pair"] != pair:
                    continue
                full_v = r.get(f"full_ret_{step}")
                rest_v = r.get(f"rest_ret_{step}")
                if full_v is None or rest_v is None:
                    continue
                pooled_full.append(full_v)
                pooled_rest.append(rest_v)
                pooled_diffs.append(r[f"abs_diff_{step}"])
        if not pooled_diffs:
            aggregate_by_step[str(step)] = None
            continue
        pooled_rho = spearman(pooled_full, pooled_rest) or 0.0
        pooled_max_diff = max(pooled_diffs)
        pooled_mean_diff = sum(pooled_diffs) / len(pooled_diffs)
        verdict_step, geometry = global_verdict(pooled_rho, pooled_max_diff, per_pair_rhos)
        aggregate_by_step[str(step)] = {
            "n_trajectories": len(pooled_diffs),
            "pooled_max_abs_diff": pooled_max_diff,
            "pooled_mean_abs_diff": pooled_mean_diff,
            "pooled_spearman_rho": pooled_rho,
            "geometry_dependence": geometry,
            "verdict_step": verdict_step,
        }

    primary = aggregate_by_step.get(str(PRIMARY_STEP))
    verdict_global = primary["verdict_step"] if primary else "NO_DATA"

    # Divergence profile: at which step does max_diff first exceed thresholds?
    divergence_profile = {}
    for thresh in (0.02, 0.05, 0.10, 0.20):
        onset_step = None
        for step in EVAL_STEPS:
            s = aggregate_by_step.get(str(step))
            if s and s["pooled_max_abs_diff"] >= thresh:
                onset_step = step
                break
        divergence_profile[f"first_step_max_diff_geq_{thresh}"] = onset_step

    now = _dt.datetime.now(_dt.timezone.utc).isoformat()

    (out_dir / "per_arch.json").write_text(json.dumps({
        "generated_at": now,
        "eval_steps": EVAL_STEPS,
        "data": all_records,
    }, indent=2))
    (out_dir / "per_pair.json").write_text(json.dumps({
        "generated_at": now,
        "eval_steps": EVAL_STEPS,
        "data": per_pair_summary,
        "discovery": discovery_summary,
    }, indent=2))
    (out_dir / "aggregate.json").write_text(json.dumps({
        "generated_at": now,
        "eval_steps": EVAL_STEPS,
        "primary_step": PRIMARY_STEP,
        "by_step": aggregate_by_step,
        "divergence_profile": divergence_profile,
        "verdict_global": verdict_global,
    }, indent=2))

    # Console summary
    print("=" * 88)
    print(f"EXP-04 multi-step metric diagnostic - {now}")
    print("=" * 88)
    print("Discovery (per pair):")
    for pair, disc in discovery_summary.items():
        print(f"  {pair:<24} dirs={disc['dirs_found']:>2}  "
              f"missing_restricted={disc['dirs_missing_restricted_json']:>2}  "
              f"usable={disc['dirs_usable']:>2}")
    print()
    print("Per-step pooled summary:")
    print(f"{'step':>5}  {'n':>4}  {'rho':>+9}  {'max_diff':>9}  {'mean_diff':>10}  "
          f"{'rho_range':>10}  verdict")
    print("-" * 88)
    for step in EVAL_STEPS:
        s = aggregate_by_step.get(str(step))
        if s is None:
            print(f"{step:>5}  (no data)")
            continue
        print(f"{step:>5}  {s['n_trajectories']:>4}  {s['pooled_spearman_rho']:>+9.4f}  "
              f"{s['pooled_max_abs_diff']:>9.4f}  {s['pooled_mean_abs_diff']:>10.4f}  "
              f"{s['geometry_dependence']['per_pair_rho_range']:>10.4f}  "
              f"{s['verdict_step']}")
    print()
    print(f"Divergence profile (first step where pooled max_diff >= threshold):")
    for k, v in divergence_profile.items():
        print(f"  {k}: {v}")
    print()
    print(f"PRIMARY METRIC (ret_{PRIMARY_STEP}) GLOBAL VERDICT: {verdict_global}")
    print()
    print(f"Wrote: {out_dir}/per_arch.json")
    print(f"Wrote: {out_dir}/per_pair.json")
    print(f"Wrote: {out_dir}/aggregate.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
