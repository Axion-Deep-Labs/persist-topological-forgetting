"""EXP-04 comprehensive WD-wave analysis.

Pre-registered partial-Spearman test already lives in
scripts/exp04_partial_spearman_decisive.py. This script extends that test by
explicitly addressing six review questions:

  1. Effect sizes (not just p-values): marginal and partial Spearman with
     bootstrap confidence intervals.
  2. PH vs commutator-defect head-to-head: same partial-correlation test on
     commutator_defect; whether h0_total_persistence beats or merely matches.
  3. Weight-norm shadow: marginal and partial correlations to quantify how
     much h0_tp signal vanishes once weight_norm_l2 is controlled.
  4. Seed reliability: leave-one-out jackknife on the primary partial rho;
     report range, min, max.
  5. Lead-time distribution: mean, median, IQR, 5/95 percentiles of grokking
     onset; not just the headline mean.
  6. Late-grokker dominance: split seeds at an onset threshold, recompute
     correlations with and without the late-grokker subpopulation.

Usage:
    cd /fs1/scratch/cag1145/axiondeep-research
    python scripts/exp04_comprehensive_analysis.py
    python scripts/exp04_comprehensive_analysis.py --wd 0.10
    python scripts/exp04_comprehensive_analysis.py --late-onset-threshold 60000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, median

# -------- defaults --------
ANCHOR_STEP = 40_000
LATE_ONSET_DEFAULT = 60_000
BOOTSTRAP_ITERS = 5000
BOOTSTRAP_SEED = 20260512


# -------- ranks + correlations --------
def rankdata(arr):
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


def pearson(x, y):
    n = len(x)
    if n < 3 or len(x) != len(y):
        return None
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(x, y):
    if len(x) < 3:
        return None
    return pearson(rankdata(x), rankdata(y))


def partial_spearman(x, y, z):
    if len(x) < 4 or not (len(x) == len(y) == len(z)):
        return None
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def residuals(target, covariate):
        mt, mc = mean(target), mean(covariate)
        num = sum((a - mc) * (b - mt) for a, b in zip(covariate, target))
        den = sum((a - mc) ** 2 for a in covariate)
        if den == 0:
            return [t - mt for t in target]
        slope = num / den
        intercept = mt - slope * mc
        return [t - (intercept + slope * c) for t, c in zip(target, covariate)]

    return pearson(residuals(rx, rz), residuals(ry, rz))


def bootstrap_ci(fn, args_tuple, iters=BOOTSTRAP_ITERS, alpha=0.05):
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(args_tuple[0])
    vals = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        boot_args = tuple([[a[i] for i in idx] for a in args_tuple])
        r = fn(*boot_args)
        if r is not None:
            vals.append(r)
    if not vals:
        return None, None, None
    vals.sort()
    lo = vals[int(alpha / 2 * len(vals))]
    hi = vals[int((1 - alpha / 2) * len(vals))]
    return mean(vals), lo, hi


# -------- data loading --------
def find_onset(training_records, test_thr=0.9, train_thr=0.99):
    for r in training_records:
        if r.get("test_acc", 0) >= test_thr and r.get("train_acc", 0) >= train_thr:
            return r["step"]
    return None


def value_at_or_near_step(records, key, step):
    """Return record value at step closest to `step` (most useful: closest BEFORE-or-at, but
    we use simple absolute-difference since grids are uniform)."""
    best, best_d = None, float("inf")
    for r in records:
        d = abs(r["step"] - step)
        if d < best_d:
            best_d = d
            best = r.get(key)
    return best


def peak_value(records, key):
    vals = [r.get(key) for r in records if r.get(key) is not None]
    return max(vals) if vals else None


def load_seed(root: Path, seed: int) -> dict | None:
    sd = root / f"seed_{seed}"
    if not sd.is_dir():
        return None
    try:
        topo = json.load(open(sd / "topology_metrics.json"))
        baseline = json.load(open(sd / "baseline_metrics.json"))
        training = json.load(open(sd / "training_metrics.json"))
    except FileNotFoundError:
        return None
    # topology_metrics may be {"records": [...]} or a flat list
    topo_recs = topo.get("records") if isinstance(topo, dict) else topo
    tr_recs = training.get("metrics") if isinstance(training, dict) else training
    if not topo_recs or not baseline or not tr_recs:
        return None
    onset = find_onset(tr_recs)
    if onset is None:
        return None
    return {
        "seed": seed,
        "onset": onset,
        "h0_peak": peak_value(topo_recs, "h0_total_persistence"),
        "h0_at_anchor": value_at_or_near_step(topo_recs, "h0_total_persistence", ANCHOR_STEP),
        "commutator_peak": peak_value(baseline, "commutator_defect"),
        "commutator_at_anchor": value_at_or_near_step(baseline, "commutator_defect", ANCHOR_STEP),
        "wn_at_anchor": value_at_or_near_step(baseline, "weight_norm_l2", ANCHOR_STEP),
        "sharpness_at_anchor": value_at_or_near_step(baseline, "sharpness", ANCHOR_STEP),
    }


def load_wave(root: Path) -> list[dict]:
    results = []
    for seed_dir in sorted(root.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        rec = load_seed(root, seed)
        if rec is not None and all(rec[k] is not None for k in
                                   ("h0_peak", "h0_at_anchor", "commutator_peak",
                                    "wn_at_anchor")):
            results.append(rec)
    return results


# -------- analysis sections --------
def section_effect_sizes(rows):
    onset = [r["onset"] for r in rows]
    h0p = [r["h0_peak"] for r in rows]
    h0a = [r["h0_at_anchor"] for r in rows]
    cdp = [r["commutator_peak"] for r in rows]
    cda = [r["commutator_at_anchor"] for r in rows]
    wna = [r["wn_at_anchor"] for r in rows]
    sha = [r["sharpness_at_anchor"] for r in rows]
    print()
    print("------------------------------------------------------------------------------")
    print("1. MARGINAL EFFECT SIZES (Spearman rho with grokking onset)")
    print("------------------------------------------------------------------------------")
    pairs = [
        ("h0_total_persistence (peak)",    h0p),
        ("h0_total_persistence (@anchor)", h0a),
        ("commutator_defect   (peak)",     cdp),
        ("commutator_defect   (@anchor)",  cda),
        ("weight_norm_l2      (@anchor)",  wna),
        ("sharpness           (@anchor)",  sha),
    ]
    print(f"  {'predictor':<38} {'rho':>+9}  {'bootstrap 95% CI':>26}")
    out = {}
    for name, vals in pairs:
        r = spearman(vals, onset)
        m, lo, hi = bootstrap_ci(spearman, (vals, onset))
        ci_str = f"[{lo:+.4f}, {hi:+.4f}]" if lo is not None else "(unavailable)"
        print(f"  {name:<38} {r:>+9.4f}  {ci_str:>26}")
        out[name] = {"rho": r, "boot_mean": m, "ci_low": lo, "ci_high": hi}
    return out


def section_ph_vs_commutator(rows):
    onset = [r["onset"] for r in rows]
    h0p = [r["h0_peak"] for r in rows]
    cdp = [r["commutator_peak"] for r in rows]
    wna = [r["wn_at_anchor"] for r in rows]
    print()
    print("------------------------------------------------------------------------------")
    print("2. PH vs COMMUTATOR-DEFECT head-to-head")
    print("------------------------------------------------------------------------------")
    print(f"  partial rho ( h0_peak       | wn@anchor ) = {partial_spearman(h0p, onset, wna):+.4f}")
    print(f"  partial rho ( commutator_pk | wn@anchor ) = {partial_spearman(cdp, onset, wna):+.4f}")
    print(f"  partial rho ( h0_peak       | commutator_pk ) = {partial_spearman(h0p, onset, cdp):+.4f}")
    print(f"  partial rho ( commutator_pk | h0_peak       ) = {partial_spearman(cdp, onset, h0p):+.4f}")
    return {
        "h0p_partial_wn": partial_spearman(h0p, onset, wna),
        "cdp_partial_wn": partial_spearman(cdp, onset, wna),
        "h0p_partial_cdp": partial_spearman(h0p, onset, cdp),
        "cdp_partial_h0p": partial_spearman(cdp, onset, h0p),
    }


def section_weight_norm_shadow(rows):
    onset = [r["onset"] for r in rows]
    h0p = [r["h0_peak"] for r in rows]
    h0a = [r["h0_at_anchor"] for r in rows]
    wna = [r["wn_at_anchor"] for r in rows]
    print()
    print("------------------------------------------------------------------------------")
    print("3. WEIGHT-NORM SHADOW: how much signal vanishes when wn is partialed")
    print("------------------------------------------------------------------------------")
    print(f"  marginal h0_peak    vs onset: rho = {spearman(h0p, onset):+.4f}")
    print(f"  partial  h0_peak    | wn:     rho = {partial_spearman(h0p, onset, wna):+.4f}")
    print(f"  marginal h0_anchor  vs onset: rho = {spearman(h0a, onset):+.4f}")
    print(f"  partial  h0_anchor  | wn:     rho = {partial_spearman(h0a, onset, wna):+.4f}")
    print(f"  marginal wn@anchor  vs onset: rho = {spearman(wna, onset):+.4f}")
    print(f"  partial  wn         | h0_peak:rho = {partial_spearman(wna, onset, h0p):+.4f}")


def section_jackknife(rows):
    onset = [r["onset"] for r in rows]
    h0p = [r["h0_peak"] for r in rows]
    wna = [r["wn_at_anchor"] for r in rows]
    print()
    print("------------------------------------------------------------------------------")
    print("4. JACKKNIFE on primary partial-rho ( h0_peak | wn@anchor ): drop each seed")
    print("------------------------------------------------------------------------------")
    vals = []
    drops = []
    for i in range(len(rows)):
        oi = onset[:i] + onset[i + 1:]
        hi = h0p[:i] + h0p[i + 1:]
        wi = wna[:i] + wna[i + 1:]
        r = partial_spearman(hi, oi, wi)
        if r is not None:
            vals.append(r)
            drops.append((rows[i]["seed"], rows[i]["onset"], r))
    if not vals:
        return None
    vals.sort()
    drops.sort(key=lambda t: t[2])
    print(f"  n leave-one-out fits:  {len(vals)}")
    print(f"  full-sample partial rho: {partial_spearman(h0p, onset, wna):+.4f}")
    print(f"  jackknife min / median / max: {vals[0]:+.4f}  /  {vals[len(vals)//2]:+.4f}  /  {vals[-1]:+.4f}")
    print(f"  most-influential seeds (drop -> lowest partial rho):")
    for s, o, r in drops[:3]:
        print(f"    drop seed={s:>4} (onset={o:>6}) -> partial rho = {r:+.4f}")
    print(f"  least-influential seeds (drop -> highest partial rho):")
    for s, o, r in drops[-3:]:
        print(f"    drop seed={s:>4} (onset={o:>6}) -> partial rho = {r:+.4f}")
    return {"jackknife_min": vals[0], "jackknife_max": vals[-1], "jackknife_median": vals[len(vals) // 2]}


def section_lead_time(rows):
    onset = sorted([r["onset"] for r in rows])
    n = len(onset)
    p = lambda q: onset[max(0, min(n - 1, int(round(q * (n - 1)))))]
    print()
    print("------------------------------------------------------------------------------")
    print("5. LEAD-TIME / ONSET DISTRIBUTION")
    print("------------------------------------------------------------------------------")
    print(f"  n: {n}")
    print(f"  min   p05   p25  median  p75   p95   max")
    print(f"  {p(0):>5} {p(0.05):>5} {p(0.25):>5} {p(0.50):>6} {p(0.75):>4} {p(0.95):>5} {p(1.0):>5}")
    # Crude histogram (10K-step buckets)
    buckets = {}
    for o in onset:
        b = (o // 10000) * 10000
        buckets[b] = buckets.get(b, 0) + 1
    print("  Histogram (10K-step buckets):")
    for b in sorted(buckets):
        bar = "#" * buckets[b]
        print(f"    {b:>6}-{b+9999:<6}  ({buckets[b]:>2})  {bar}")
    return {"onset_min": p(0), "onset_p25": p(0.25), "onset_median": p(0.5),
            "onset_p75": p(0.75), "onset_max": p(1.0)}


def section_late_grokker(rows, threshold):
    onset = [r["onset"] for r in rows]
    h0p = [r["h0_peak"] for r in rows]
    wna = [r["wn_at_anchor"] for r in rows]
    cdp = [r["commutator_peak"] for r in rows]

    late_idx = [i for i, o in enumerate(onset) if o >= threshold]
    normal_idx = [i for i, o in enumerate(onset) if o < threshold]
    print()
    print("------------------------------------------------------------------------------")
    print(f"6. LATE-GROKKER DOMINANCE (split at onset >= {threshold})")
    print("------------------------------------------------------------------------------")
    print(f"  Late grokkers (n={len(late_idx)}): seeds = "
          + ", ".join(str(rows[i]["seed"]) for i in late_idx))
    print(f"  Normal      (n={len(normal_idx)}):")

    def sub(idxs):
        return ([h0p[i] for i in idxs],
                [onset[i] for i in idxs],
                [wna[i] for i in idxs],
                [cdp[i] for i in idxs])

    def block(label, idxs):
        if len(idxs) < 4:
            print(f"  {label}: n={len(idxs)} too small for partial correlation")
            return None
        h, o, w, c = sub(idxs)
        m_h0 = spearman(h, o)
        p_h0_wn = partial_spearman(h, o, w)
        m_cd = spearman(c, o)
        p_cd_wn = partial_spearman(c, o, w)
        print(f"  {label}: n={len(idxs)}")
        print(f"    marginal h0_peak  vs onset:    {m_h0:+.4f}")
        print(f"    marginal commut.  vs onset:    {m_cd:+.4f}")
        print(f"    partial  h0_peak  | wn@anchor: {p_h0_wn:+.4f}")
        print(f"    partial  commut.  | wn@anchor: {p_cd_wn:+.4f}")
        return {"n": len(idxs), "h0_marginal": m_h0, "cd_marginal": m_cd,
                "h0_partial_wn": p_h0_wn, "cd_partial_wn": p_cd_wn}

    full = block("FULL SAMPLE", list(range(len(rows))))
    normal = block("NORMAL only (late grokkers excluded)", normal_idx)
    late = block("LATE only", late_idx)
    return {"full": full, "normal_only": normal, "late_only": late,
            "n_normal": len(normal_idx), "n_late": len(late_idx),
            "threshold": threshold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wd", default="0.03", help="weight-decay wave tag (default 0.03)")
    ap.add_argument("--late-onset-threshold", type=int, default=LATE_ONSET_DEFAULT)
    ap.add_argument("--out", default=None,
                    help="optional path to write JSON summary (default: stdout only)")
    args = ap.parse_args()

    root = Path(f"results/exp04_full/wd_{args.wd}")
    if not root.is_dir():
        raise SystemExit(f"no such wave directory: {root}")
    rows = load_wave(root)
    if not rows:
        raise SystemExit(f"no usable seeds in {root}")
    print("=" * 78)
    print(f"EXP-04 COMPREHENSIVE ANALYSIS  (WD = {args.wd}, n = {len(rows)})")
    print("=" * 78)

    sec1 = section_effect_sizes(rows)
    sec2 = section_ph_vs_commutator(rows)
    section_weight_norm_shadow(rows)
    sec4 = section_jackknife(rows)
    sec5 = section_lead_time(rows)
    sec6 = section_late_grokker(rows, args.late_onset_threshold)

    summary = {
        "wd": args.wd,
        "n": len(rows),
        "anchor_step": ANCHOR_STEP,
        "late_threshold": args.late_onset_threshold,
        "effect_sizes_marginal": sec1,
        "ph_vs_commutator": sec2,
        "jackknife": sec4,
        "onset_distribution": sec5,
        "late_grokker_split": sec6,
    }

    out_path = Path(args.out) if args.out else Path(
        f"results/exp04_comprehensive_analysis_wd_{args.wd}.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
