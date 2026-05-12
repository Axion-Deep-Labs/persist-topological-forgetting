"""EXP-04 metric verification diagnostic.

Compares full-softmax vs restricted-softmax retention at step 10 across all
architectures of all six cross-dataset pairs, to determine whether the two
metrics diverge materially in actual training trajectories (vs being equal
only at step 0 as the Phase 3b sanity check shows).

Three-tier decision gate:

  ALIGNED
    rankings preserved AND magnitudes preserved (max diff < 0.02 AND
    pooled rho > 0.95 AND no per-pair rho below 0.90)
    Paper implication: metric distinction stays secondary; main story
    is the pooled conditional topology signal.

  MOSTLY_ALIGNED_GEOMETRY_DEPENDENT
    rankings largely preserved (pooled rho > 0.85), but magnitudes
    diverge systematically; either some per-pair rho < 0.90 or the
    range across per-pair rhos exceeds 0.10.
    Paper implication: full vs restricted distinction becomes part of
    the conceptual contribution; classifier-rerouting vs backbone-drift
    framing becomes central; heterogeneity is part of the finding.

  MISALIGNED
    rankings unstable (pooled rho <= 0.85), or pooled magnitudes diverge
    severely.
    Paper implication: claims narrow substantially; topology signal is
    metric-contingent rather than robust.

Designed to run on HPC where the per-run forgetting_curve_restricted.json
files exist. The aggregate Phase 3b summary alone does not contain
step-by-step trajectory data, only step-0 sanity checks.

Outputs (written under results/exp04_metric_diagnostic/):
  per_arch.json   - flat list of (pair, arch, full_ret_10, rest_ret_10, diff)
  per_pair.json   - per-pair n, max_diff, mean_diff, spearman_rho, verdict_local
  aggregate.json  - pooled stats, geometry-dependence diagnostics, global verdict

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


def ret_at_step(curve_data: dict, step: int) -> float | None:
    initial = curve_data.get("initial_task_a_acc")
    if not initial:
        return None
    for pt in curve_data.get("curve", []):
        if pt.get("step") == step:
            return pt.get("task_a_acc", 0) / initial
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


def collect_pair(pair: str) -> list[tuple[str, float, float, float]]:
    task_a, task_b = pair.split("_to_")
    pair_dirs = sorted(glob.glob(f"results/exp01_*_{task_a}_xd_{task_b}"))
    rows: list[tuple[str, float, float, float]] = []
    for d in pair_dirs:
        arch = Path(d).name.replace("exp01_", "").replace(f"_{task_a}_xd_{task_b}", "")
        full_path = Path(d) / "forgetting" / "forgetting_curve.json"
        rest_path = Path(d) / "forgetting" / "forgetting_curve_restricted.json"
        if not full_path.exists() or not rest_path.exists():
            continue
        full = json.load(open(full_path))
        rest = json.load(open(rest_path))
        full_r10 = ret_at_step(full, 10)
        rest_r10 = ret_at_step(rest, 10)
        if full_r10 is None or rest_r10 is None:
            continue
        rows.append((arch, full_r10, rest_r10, abs(full_r10 - rest_r10)))
    return rows


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


def main() -> int:
    # Single-pair legacy mode (stdout only, no JSON)
    if len(sys.argv) > 1 and sys.argv[1] in ALL_PAIRS:
        pair = sys.argv[1]
        rows = collect_pair(pair)
        rows.sort(key=lambda r: r[3], reverse=True)
        print(f"Pair: {pair}")
        print(f"N: {len(rows)}\n")
        print(f"{'arch':<28} {'full_ret_10':>11} {'rest_ret_10':>11} {'|diff|':>8}")
        print("-" * 62)
        for arch, full, rest, diff in rows:
            print(f"{arch:<28} {full:>11.4f} {rest:>11.4f} {diff:>8.4f}")
        if not rows:
            return 1
        diffs = [r[3] for r in rows]
        rho = spearman([r[1] for r in rows], [r[2] for r in rows]) or 0.0
        v = per_pair_verdict(max(diffs), rho)
        print(f"\nMax |diff|: {max(diffs):.4f}  Mean |diff|: {sum(diffs)/len(diffs):.4f}  rho: {rho:.4f}")
        print(f"Per-pair verdict: {v}")
        return 0

    # Default: sweep all 6 pairs and write JSON outputs.
    out_dir = Path("results/exp04_metric_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_arch_records: list[dict] = []
    per_pair_summary: dict[str, dict] = {}
    per_pair_rhos_for_global: list[float] = []
    pooled_full: list[float] = []
    pooled_rest: list[float] = []
    pooled_diffs: list[float] = []

    for pair in ALL_PAIRS:
        rows = collect_pair(pair)
        if not rows:
            per_pair_summary[pair] = {"n": 0, "verdict_local": "NO_DATA"}
            continue
        full_vals = [r[1] for r in rows]
        rest_vals = [r[2] for r in rows]
        diffs = [r[3] for r in rows]
        rho = spearman(full_vals, rest_vals) or 0.0
        max_diff = max(diffs)
        mean_diff = sum(diffs) / len(diffs)
        verdict_local = per_pair_verdict(max_diff, rho)
        per_pair_summary[pair] = {
            "n": len(rows),
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "spearman_rho": rho,
            "verdict_local": verdict_local,
        }
        per_pair_rhos_for_global.append(rho)
        pooled_full.extend(full_vals)
        pooled_rest.extend(rest_vals)
        pooled_diffs.extend(diffs)
        for arch, full, rest, diff in rows:
            per_arch_records.append({
                "pair": pair,
                "arch": arch,
                "full_ret_10": full,
                "rest_ret_10": rest,
                "abs_diff": diff,
            })

    pooled_rho = spearman(pooled_full, pooled_rest) or 0.0
    pooled_max_diff = max(pooled_diffs) if pooled_diffs else 0.0
    pooled_mean_diff = sum(pooled_diffs) / len(pooled_diffs) if pooled_diffs else 0.0
    verdict_global, geometry_diagnostics = global_verdict(
        pooled_rho, pooled_max_diff, per_pair_rhos_for_global
    )

    now = _dt.datetime.now(_dt.timezone.utc).isoformat()

    (out_dir / "per_arch.json").write_text(json.dumps({
        "generated_at": now,
        "data": per_arch_records,
    }, indent=2))
    (out_dir / "per_pair.json").write_text(json.dumps({
        "generated_at": now,
        "data": per_pair_summary,
    }, indent=2))
    (out_dir / "aggregate.json").write_text(json.dumps({
        "generated_at": now,
        "n_pairs_with_data": sum(1 for v in per_pair_summary.values() if v["n"] > 0),
        "n_total_trajectories": len(per_arch_records),
        "pooled_max_abs_diff": pooled_max_diff,
        "pooled_mean_abs_diff": pooled_mean_diff,
        "pooled_spearman_rho": pooled_rho,
        "geometry_dependence": geometry_diagnostics,
        "verdict_global": verdict_global,
    }, indent=2))

    # Console summary
    print("=" * 78)
    print(f"EXP-04 metric diagnostic - {now}")
    print("=" * 78)
    print(f"Pairs swept:    {len(ALL_PAIRS)}")
    print(f"Trajectories:   {len(per_arch_records)}")
    print()
    print(f"{'pair':<28} {'n':>4} {'rho':>8} {'max_diff':>10} {'mean_diff':>11} {'verdict_local':>30}")
    print("-" * 95)
    for pair in ALL_PAIRS:
        s = per_pair_summary[pair]
        if s["n"] == 0:
            print(f"{pair:<28} {0:>4}  (no data)")
            continue
        print(
            f"{pair:<28} {s['n']:>4} {s['spearman_rho']:>8.4f} "
            f"{s['max_abs_diff']:>10.4f} {s['mean_abs_diff']:>11.4f} "
            f"{s['verdict_local']:>30}"
        )
    print("-" * 95)
    print(f"POOLED                       {len(per_arch_records):>4} "
          f"{pooled_rho:>8.4f} {pooled_max_diff:>10.4f} {pooled_mean_diff:>11.4f}")
    print()
    print(f"Geometry-dependence diagnostics:")
    for k, v in geometry_diagnostics.items():
        print(f"  {k}: {v}")
    print()
    print(f"GLOBAL VERDICT: {verdict_global}")
    print()
    print(f"Wrote: {out_dir}/per_arch.json")
    print(f"Wrote: {out_dir}/per_pair.json")
    print(f"Wrote: {out_dir}/aggregate.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
