"""EXP-04 pre-registered decisive test, early-look on WD=0.03 wave 1 (n=17).

Pre-registration (configs/exp04_full.yaml):
    Hypothesis:  h0_total_persistence at peak provides predictive signal for
                 grokking onset BEYOND what weight_norm_l2 at step 40K provides.
    Decision:    partial Spearman rho(h0_tp_peak, onset | wn_l2_at_40k) > 0.30,
                 bootstrap 95% CI excludes zero.

Reads trajectories from results/exp04_full/wd_<WD_TAG>/seed_<N>/.
Computes:
  - per-seed grokking onset (config rule: test>=0.9 AND train>=0.99)
  - per-seed peak h0_total_persistence and weight_norm_l2 at step ~40K
  - partial Spearman correlation across seeds, with weight_norm_l2 as covariate
  - bootstrap 95% CI on partial rho

Outputs three verdicts:
  PASS         partial rho > 0.30 AND CI excludes 0       (topology wins)
  FAIL         partial rho <= 0.30 OR CI straddles 0      (topology redundant)
  BORDERLINE   partial rho in (0.30, 0.45) but CI loose   (need full n=90)

Usage:
    .venv/bin/python scripts/exp04_partial_spearman_decisive.py
    .venv/bin/python scripts/exp04_partial_spearman_decisive.py --wd 0.10
"""
import argparse
import json
import random
from pathlib import Path
from statistics import mean

ANCHOR_STEP = 40_000
PARTIAL_RHO_MIN = 0.30
BOOTSTRAP_ITERS = 5000
BOOTSTRAP_SEED = 20260510


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
    """Spearman correlation of x and y, controlling for z.

    Rank-transform all three, regress rx on rz and ry on rz via OLS, then
    Pearson-correlate the residuals.
    """
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


def bootstrap_ci(x, y, z, iters, alpha=0.05):
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(x)
    rhos = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [x[i] for i in idx]
        by = [y[i] for i in idx]
        bz = [z[i] for i in idx]
        try:
            r = partial_spearman(bx, by, bz)
            if r is not None:
                rhos.append(r)
        except Exception:
            continue
    if not rhos:
        return None, None, None
    rhos.sort()
    lo = rhos[int((alpha / 2) * len(rhos))]
    hi = rhos[int((1 - alpha / 2) * len(rhos))]
    point = mean(rhos)
    return point, lo, hi


def value_at_step(records, step, key):
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


def peak_value(records, key, max_step=None):
    best = None
    best_step = None
    for r in records:
        v = r.get(key)
        s = r.get("step")
        if v is None:
            continue
        if max_step is not None and s is not None and s > max_step:
            continue
        if best is None or v > best:
            best = v
            best_step = s
    return best, best_step


def compute_onset(tr_metrics, test_thr=0.9, train_thr=0.99):
    for r in tr_metrics:
        if r.get("test_acc", 0) >= test_thr and r.get("train_acc", 0) >= train_thr:
            return r["step"]
    return None


def load_seed(wd_tag, seed):
    root = Path(f"results/exp04_full/wd_{wd_tag}/seed_{seed}")
    tr = json.load(open(root / "training_metrics.json"))["metrics"]
    to = json.load(open(root / "topology_metrics.json"))
    ba = json.load(open(root / "baseline_metrics.json"))
    return tr, to, ba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wd", default="0.03", help="weight decay tag to analyze (default 0.03)")
    args = parser.parse_args()

    wd_tag = f"{float(args.wd):.2f}"
    base = Path(f"results/exp04_full/wd_{wd_tag}")
    if not base.exists():
        print(f"ERROR: {base} does not exist. Did you rsync from HPC?")
        return

    seed_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    seeds = [int(p.name.split("_")[1]) for p in seed_dirs]

    print("=" * 78)
    print(f"EXP-04 PARTIAL-SPEARMAN DECISIVE TEST  (WD = {wd_tag}, n = {len(seeds)})")
    print(f"Pre-reg threshold: partial rho > {PARTIAL_RHO_MIN}, bootstrap 95% CI excludes 0")
    print("=" * 78)

    rows = []
    for seed in seeds:
        try:
            tr, to, ba = load_seed(wd_tag, seed)
        except FileNotFoundError as e:
            print(f"[seed {seed}] missing data: {e.filename}")
            continue

        onset = compute_onset(tr)
        if onset is None:
            print(f"[seed {seed}] never grokked within window, dropping")
            continue

        h0_peak, h0_peak_step = peak_value(to, "h0_total_persistence")
        h0_at_40k, _ = value_at_step(to, ANCHOR_STEP, "h0_total_persistence")
        wn_at_40k, _ = value_at_step(ba, ANCHOR_STEP, "weight_norm_l2")

        if h0_peak is None or wn_at_40k is None:
            print(f"[seed {seed}] missing peak h0 or wn@40K, dropping")
            continue

        rows.append({
            "seed": seed,
            "onset": onset,
            "h0_peak": h0_peak,
            "h0_peak_step": h0_peak_step,
            "h0_at_40k": h0_at_40k,
            "wn_at_40k": wn_at_40k,
        })

    if len(rows) < 4:
        print(f"\nERROR: only {len(rows)} usable seeds; partial Spearman needs >= 4.")
        return

    print(f"\nUsable seeds: {len(rows)} of {len(seeds)}")
    print(f"\n{'seed':>5} {'onset':>8} {'h0_peak':>12} {'h0@40K':>12} {'wn@40K':>10}")
    for r in rows:
        print(f"  {r['seed']:>3} {r['onset']:>8} {r['h0_peak']:>12.3g} "
              f"{r['h0_at_40k']:>12.3g} {r['wn_at_40k']:>10.4f}")

    onsets = [r["onset"] for r in rows]
    h0_peaks = [r["h0_peak"] for r in rows]
    h0_at_40 = [r["h0_at_40k"] for r in rows]
    wn_at_40 = [r["wn_at_40k"] for r in rows]

    print("\n" + "-" * 78)
    print("Marginal Spearman correlations with grokking onset")
    print("-" * 78)
    print(f"  h0_total_persistence (peak)      vs onset: {spearman(h0_peaks, onsets):+.4f}")
    print(f"  h0_total_persistence (at ~40K)   vs onset: {spearman(h0_at_40, onsets):+.4f}")
    print(f"  weight_norm_l2       (at ~40K)   vs onset: {spearman(wn_at_40, onsets):+.4f}")

    print("\n" + "-" * 78)
    print("PRE-REG TEST: partial Spearman, controlling for weight_norm_l2 @ 40K")
    print("-" * 78)
    pr_primary = partial_spearman(h0_peaks, onsets, wn_at_40)
    pr_alt = partial_spearman(h0_at_40, onsets, wn_at_40)
    print(f"  primary (h0_peak | wn@40K)   ->  partial rho = {pr_primary:+.4f}")
    print(f"  alt     (h0@40K  | wn@40K)   ->  partial rho = {pr_alt:+.4f}")

    print("\n" + "-" * 78)
    print(f"Bootstrap 95% CI on primary partial rho ({BOOTSTRAP_ITERS} iters)")
    print("-" * 78)
    point, lo, hi = bootstrap_ci(h0_peaks, onsets, wn_at_40, BOOTSTRAP_ITERS)
    print(f"  point estimate (mean of bootstrap): {point:+.4f}")
    print(f"  95% CI: [{lo:+.4f}, {hi:+.4f}]")

    ci_excludes_zero = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    threshold_met = abs(pr_primary) > PARTIAL_RHO_MIN

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  partial rho threshold (>{PARTIAL_RHO_MIN}):   {'MET' if threshold_met else 'NOT MET'}")
    print(f"  bootstrap CI excludes zero:    {'YES' if ci_excludes_zero else 'NO'}")
    if threshold_met and ci_excludes_zero:
        print("\n  >> PASS: topology adds predictive signal beyond weight_norm_l2.")
        print("     Recommendation: launch WD=0.10 + WD=0.30 waves for full-n=90 paper.")
    elif not threshold_met and abs(pr_primary) < 0.15:
        print("\n  >> FAIL (clear): topology is redundant with weight_norm_l2 at WD=0.03.")
        print("     Recommendation: skip remaining waves; write up as clean negative result.")
    else:
        print("\n  >> BORDERLINE: signal is non-trivial but does not clear pre-reg bar at n=17.")
        print("     Recommendation: launch remaining waves; re-run on full n=90.")
    print()


if __name__ == "__main__":
    main()
