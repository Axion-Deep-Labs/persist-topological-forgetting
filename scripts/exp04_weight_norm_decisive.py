"""EXP-04 decisive kill-switch: weight_norm_l2 vs h0_total_persistence.

Tests the question Joshua flagged 2026-04-04: does h0_tp add anything beyond
weight_norm_l2 as a grokking-onset predictor?

Inputs (existing pilot data):
  results/exp04_pilot/seed_{42,137,256,1024}/baseline_metrics.json
  results/exp04_pilot/seed_{42,137,256,1024}/topology_metrics.json
  results/exp04_pilot/seed_{42,137,256,1024}/training_metrics.json

Outputs:
  - Per-seed correlations: weight_norm_l2[t] vs h0_total_persistence[t]
  - Per-seed predictor values at fixed comparison step (40K)
  - Cross-seed: how well does each predict grokking onset across seeds?
  - Verdict: weight norm wins / topology wins / inconclusive

Usage:
    .venv/bin/python scripts/exp04_weight_norm_decisive.py
"""
import json
from pathlib import Path
from statistics import mean

BASE = Path("results/exp04_pilot")

# Use 4 in-scope seeds. seed 1024 has incomplete topology (34/81 records); we
# use only the steps where both signals exist.
SEEDS = [42, 137, 256, 1024]
COMPARISON_STEP = 40_000  # the "predictor at step 40K" anchor from the memory


def load_seed(seed):
    root = BASE / f"seed_{seed}"
    tr = json.load(open(root / "training_metrics.json"))["metrics"]
    to = json.load(open(root / "topology_metrics.json"))
    ba = json.load(open(root / "baseline_metrics.json"))
    return tr, to, ba


def compute_onset(tr_metrics, test_thr=0.9, train_thr=0.99):
    for r in tr_metrics:
        if r.get("test_acc", 0) >= test_thr and r.get("train_acc", 0) >= train_thr:
            return r["step"]
    return None


def spearman(x, y):
    """Hand-rolled Spearman to avoid scipy dep."""
    if len(x) < 3 or len(x) != len(y):
        return None
    rx = rankdata(x)
    ry = rankdata(y)
    n = len(x)
    mean_rx = mean(rx); mean_ry = mean(ry)
    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    den_x = sum((a - mean_rx) ** 2 for a in rx) ** 0.5
    den_y = sum((b - mean_ry) ** 2 for b in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def rankdata(arr):
    """Rank with average for ties."""
    indexed = sorted(enumerate(arr), key=lambda p: p[1])
    ranks = [0.0] * len(arr)
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


def value_at_step(records, step, key):
    """Return the value of key at the closest <= step (None if none)."""
    best = None
    best_step = -1
    for r in records:
        s = r.get("step")
        if s is None or s > step:
            continue
        if s > best_step:
            best_step = s
            best = r.get(key)
    return best, best_step


def peak_value(records, key):
    """(peak value, peak step) of key across records."""
    best = None
    best_step = None
    for r in records:
        v = r.get(key)
        if v is None:
            continue
        if best is None or v > best:
            best = v
            best_step = r.get("step")
    return best, best_step


def main():
    print("=" * 78)
    print("EXP-04 DECISIVE: weight_norm_l2 vs h0_total_persistence as grokking predictors")
    print("=" * 78)

    per_seed = {}

    for seed in SEEDS:
        try:
            tr, to, ba = load_seed(seed)
        except FileNotFoundError as e:
            print(f"\n[seed {seed}] missing data: {e.filename}")
            continue

        # Build aligned series across steps where both signals exist
        topo_by_step = {r["step"]: r.get("h0_total_persistence") for r in to}
        wn_by_step = {r["step"]: r.get("weight_norm_l2") for r in ba}
        common = sorted(set(topo_by_step) & set(wn_by_step))
        common = [s for s in common
                  if topo_by_step[s] is not None and wn_by_step[s] is not None]

        h0_series = [topo_by_step[s] for s in common]
        wn_series = [wn_by_step[s] for s in common]

        # Within-trajectory correlation (the redundancy claim)
        rho_within = spearman(h0_series, wn_series)

        # Onset
        onset = compute_onset(tr)
        onset_080 = compute_onset(tr, 0.8, 0.99)

        # Predictors at 40K
        h0_at_40k, h0_step_used = value_at_step(to, COMPARISON_STEP, "h0_total_persistence")
        wn_at_40k, wn_step_used = value_at_step(ba, COMPARISON_STEP, "weight_norm_l2")
        h0_peak, h0_peak_step = peak_value(to, "h0_total_persistence")
        wn_peak, wn_peak_step = peak_value(ba, "weight_norm_l2")

        per_seed[seed] = {
            "n_aligned": len(common),
            "rho_within": rho_within,
            "onset": onset,
            "onset_080": onset_080,
            "h0_at_40k": h0_at_40k,
            "wn_at_40k": wn_at_40k,
            "h0_peak": h0_peak,
            "h0_peak_step": h0_peak_step,
            "wn_peak": wn_peak,
            "wn_peak_step": wn_peak_step,
        }

        print(f"\n[seed {seed}]")
        print(f"  aligned steps: {len(common)}  trajectory rho(h0_tp, wn_l2) = {rho_within:.4f}" if rho_within is not None else f"  aligned steps: {len(common)}  rho: insufficient data")
        print(f"  grokking onset: {onset} (loose: {onset_080})")
        print(f"  h0_total_persistence @ step ~40K: {h0_at_40k:.3g}  (used step {h0_step_used})" if h0_at_40k is not None else "  h0_at_40k: unavailable")
        print(f"  weight_norm_l2       @ step ~40K: {wn_at_40k:.4f}  (used step {wn_step_used})" if wn_at_40k is not None else "  wn_at_40k:  unavailable")
        print(f"  h0_total_persistence peak: {h0_peak:.3g} @ step {h0_peak_step}")
        print(f"  weight_norm_l2 peak:       {wn_peak:.4f} @ step {wn_peak_step}")

    # Cross-seed predictor comparison
    print("\n" + "=" * 78)
    print("CROSS-SEED: which predictor better correlates with grokking onset?")
    print("=" * 78)
    valid = {s: d for s, d in per_seed.items() if d["onset"] is not None}
    if len(valid) < 3:
        print(f"  Only {len(valid)} seeds with valid onset — n too small for honest comparison.")
        print("  This is expected with the 2-3 truly-clean seeds; full study needed for n=90.")

    if len(valid) >= 2:
        seeds_used = sorted(valid)
        onsets = [valid[s]["onset"] for s in seeds_used]
        h0_at_40 = [valid[s]["h0_at_40k"] for s in seeds_used if valid[s]["h0_at_40k"] is not None]
        wn_at_40 = [valid[s]["wn_at_40k"] for s in seeds_used if valid[s]["wn_at_40k"] is not None]
        h0_pks = [valid[s]["h0_peak"] for s in seeds_used]
        wn_pks = [valid[s]["wn_peak"] for s in seeds_used]

        print(f"\n  Seeds in scope: {seeds_used}")
        print(f"  Onsets: {onsets}")
        print(f"  Spearman rho with onset:")
        if len(h0_at_40) == len(onsets):
            print(f"    h0_total_persistence @ ~40K  vs onset: {spearman(h0_at_40, onsets)}")
        if len(wn_at_40) == len(onsets):
            print(f"    weight_norm_l2       @ ~40K  vs onset: {spearman(wn_at_40, onsets)}")
        print(f"    h0_total_persistence  peak    vs onset: {spearman(h0_pks, onsets)}")
        print(f"    weight_norm_l2        peak    vs onset: {spearman(wn_pks, onsets)}")

    # Within-trajectory redundancy summary
    print("\n" + "=" * 78)
    print("WITHIN-TRAJECTORY REDUNDANCY (the 04-04 memory claim)")
    print("=" * 78)
    print("  Memory: h0_tp vs wn_l2 should be ρ ~= 0.978 (seed 42), 0.992 (seed 7777)")
    for s, d in per_seed.items():
        rho = d["rho_within"]
        flag = ""
        if rho is not None:
            if abs(rho) > 0.95:
                flag = "  <- HIGH redundancy (topology kill-switch tripped on this seed)"
            elif abs(rho) > 0.80:
                flag = "  <- substantial overlap"
        print(f"  seed {s}: rho = {rho}{flag}" if rho is not None else f"  seed {s}: insufficient")

    # Verdict
    print("\n" + "=" * 78)
    print("VERDICT (n is small; this is an indicator, not the final answer)")
    print("=" * 78)
    high_redundancy = sum(1 for d in per_seed.values()
                          if d["rho_within"] is not None and abs(d["rho_within"]) > 0.95)
    print(f"  Seeds with rho > 0.95 between h0_tp and wn_l2: {high_redundancy}/{len(per_seed)}")
    if high_redundancy >= len(per_seed) - 1 and len(per_seed) >= 3:
        print("  >> KILL SWITCH TRIPPED on this pilot data: weight norm dominates topology.")
        print("     Recommendation: cancel HPC scaling jobs and reframe to Future Work.")
    elif high_redundancy >= 1:
        print("  >> SOME REDUNDANCY but not universal. Full study (n=90) needed.")
        print("     Recommendation: let HPC run; analyze partial-rho post-batch.")
    else:
        print("  >> Within-trajectory redundancy is weaker than the 04-04 memory suggested.")
        print("     Recommendation: full study justified; topology may add residual signal.")
    print()


if __name__ == "__main__":
    main()
