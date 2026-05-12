"""EXP-04 late-grokker diagnostic: feature or bug?

Question:
  Are the 5 late grokkers in the WD=0.03 wave (seeds 13, 29, 61, 83, 131,
  onsets 74000-81500) a legitimate slow-generalization regime, or are they
  optimization pathology (training instability, weight-norm runaway,
  loss collapses, etc.)?

  This determines what the paper can honestly say. If they are a real
  regime, the bimodality is the finding. If they are artifacts, the
  topology claim collapses without them.

What we look for:
  - Training loss spikes or collapses
  - Test-accuracy non-monotonicity
  - Weight-norm runaway vs bounded growth
  - Sharpness divergence
  - H0 persistence smoothness
  - Internal consistency: do the 5 late grokkers look similar to each
    other (real regime) or each weird in a different way (artifact cluster)?

Comparison: a matched sample of 5 normal grokkers (onset near 44K median).

Usage:
    cd /fs1/scratch/cag1145/axiondeep-research
    python scripts/exp04_late_grokker_diagnostic.py
    python scripts/exp04_late_grokker_diagnostic.py --wd 0.03
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median, stdev

LATE_SEEDS = [13, 29, 61, 83, 131]      # WD=0.03 onsets 81500, 74000, 76000, 78000, 78000
NORMAL_SEEDS = [11, 89, 101, 127, 139]  # WD=0.03 onsets near 44K median


def load_seed(root: Path, seed: int) -> dict:
    sd = root / f"seed_{seed}"
    if not sd.is_dir():
        return None
    topo = json.load(open(sd / "topology_metrics.json"))
    baseline = json.load(open(sd / "baseline_metrics.json"))
    training = json.load(open(sd / "training_metrics.json"))
    topo_recs = topo.get("records") if isinstance(topo, dict) else topo
    tr_recs = training.get("metrics") if isinstance(training, dict) else training
    if not topo_recs or not baseline or not tr_recs:
        return None
    return {"seed": seed, "training": tr_recs, "baseline": baseline, "topo": topo_recs}


def onset(tr, test_thr=0.9, train_thr=0.99):
    for r in tr:
        if r.get("test_acc", 0) >= test_thr and r.get("train_acc", 0) >= train_thr:
            return r["step"]
    return None


def grab(records, key):
    return [r[key] for r in records if key in r and r[key] is not None]


def grab_step(records, key, max_step=None):
    """Return list of (step, value) tuples."""
    out = []
    for r in records:
        if key in r and r[key] is not None:
            if max_step is None or r["step"] <= max_step:
                out.append((r["step"], r[key]))
    return out


def pathology_checks(data):
    """Returns dict of pathology flags + scalar diagnostics."""
    tr = data["training"]
    bl = data["baseline"]
    to = data["topo"]
    o = onset(tr)

    loss_curve = grab_step(tr, "train_loss")
    test_curve = grab_step(tr, "test_acc")
    wn_curve = grab_step(bl, "weight_norm_l2")
    sh_curve = grab_step(bl, "sharpness")
    h0_curve = grab_step(to, "h0_total_persistence")

    losses = [v for _, v in loss_curve]
    test_acc = [v for _, v in test_curve]
    wn = [v for _, v in wn_curve]
    sharp = [v for _, v in sh_curve]
    h0 = [v for _, v in h0_curve]

    # Pathology flags
    flags = {}

    # 1. Train_loss spike: any step > 10x its 5-step rolling median (post-step-1000 only)
    spike_count = 0
    if len(losses) > 50:
        post_warmup = [(s, v) for s, v in loss_curve if s > 1000]
        for i, (s, v) in enumerate(post_warmup[10:], start=10):
            window = sorted([post_warmup[i - j][1] for j in range(1, 11)])
            med = window[5]
            if med > 0 and v > 10 * med:
                spike_count += 1
    flags["loss_spike_count_post_1k"] = spike_count

    # 2. Test_acc non-monotonicity: max drawdown after first time it exceeds 0.5
    above_half_idx = next((i for i, v in enumerate(test_acc) if v > 0.5), None)
    if above_half_idx is not None and above_half_idx < len(test_acc) - 5:
        post = test_acc[above_half_idx:]
        running_max = post[0]
        max_dd = 0
        for v in post:
            if v > running_max:
                running_max = v
            dd = running_max - v
            if dd > max_dd:
                max_dd = dd
        flags["test_acc_max_drawdown_post_0.5"] = max_dd
    else:
        flags["test_acc_max_drawdown_post_0.5"] = None

    # 3. Weight-norm growth ratio: max / initial
    flags["wn_growth_ratio_max_over_initial"] = (max(wn) / wn[0]) if wn and wn[0] > 0 else None

    # 4. Sharpness divergence: max / median
    if sharp:
        sharp_med = median(sharp)
        flags["sharpness_max_over_median"] = (max(sharp) / sharp_med) if sharp_med > 0 else None
    else:
        flags["sharpness_max_over_median"] = None

    # 5. H0 trajectory smoothness: 95th percentile step-to-step jump / median value
    if len(h0) > 5:
        jumps = [abs(h0[i] - h0[i - 1]) for i in range(1, len(h0))]
        jumps.sort()
        p95 = jumps[int(0.95 * len(jumps))]
        med_val = median([v for v in h0 if v > 0])
        flags["h0_relative_jump_p95"] = p95 / med_val if med_val > 0 else None
    else:
        flags["h0_relative_jump_p95"] = None

    # 6. Pre-grok test plateau length (steps where test_acc stays < 0.2 before final climb)
    if o is not None:
        plateau_steps = [r["step"] for r in tr if r["step"] < o and r.get("test_acc", 0) < 0.2]
        flags["pre_grok_plateau_steps"] = len(plateau_steps)
    else:
        flags["pre_grok_plateau_steps"] = None

    # 7. Final state sanity
    if tr:
        last = tr[-1]
        flags["final_train_acc"] = last.get("train_acc")
        flags["final_test_acc"] = last.get("test_acc")
        flags["final_step"] = last["step"]

    return {"seed": data["seed"], "onset": o, **flags}


def consistency_table(records, label):
    """Print mean / stdev / cv for each metric across a group."""
    print(f"\n  {label} (n={len(records)}) cross-seed dispersion:")
    keys = ["loss_spike_count_post_1k", "test_acc_max_drawdown_post_0.5",
            "wn_growth_ratio_max_over_initial", "sharpness_max_over_median",
            "h0_relative_jump_p95", "pre_grok_plateau_steps",
            "final_train_acc", "final_test_acc"]
    print(f"    {'metric':<38} {'mean':>10} {'stdev':>10} {'CV%':>8}")
    rows_out = []
    for k in keys:
        vals = [r[k] for r in records if r.get(k) is not None]
        if len(vals) < 2:
            print(f"    {k:<38} {'(insuff data)':>10}")
            continue
        m = mean(vals)
        sd = stdev(vals)
        cv = (sd / abs(m) * 100) if m != 0 else float("inf")
        print(f"    {k:<38} {m:>10.4g} {sd:>10.4g} {cv:>7.1f}%")
        rows_out.append({"metric": k, "mean": m, "stdev": sd, "cv_pct": cv})
    return rows_out


def per_seed_table(records, label):
    print(f"\n  {label} per-seed:")
    print(f"    {'seed':>4} {'onset':>6} {'spike':>6} {'tDD':>6} "
          f"{'wn↑':>6} {'sh/med':>7} {'h0Δp95':>9} "
          f"{'plat':>5} {'tr_acc':>7} {'te_acc':>7}")
    for r in records:
        seed = r["seed"]; o = r["onset"]
        spike = r["loss_spike_count_post_1k"]
        tdd = r["test_acc_max_drawdown_post_0.5"]
        wn = r["wn_growth_ratio_max_over_initial"]
        sh = r["sharpness_max_over_median"]
        h0j = r["h0_relative_jump_p95"]
        plat = r["pre_grok_plateau_steps"]
        ta = r["final_train_acc"]; te = r["final_test_acc"]

        def f(v, d=2):
            return f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"

        print(f"    {seed:>4} {o or '—':>6} {spike:>6d} "
              f"{f(tdd,2):>6} {f(wn,1):>6} {f(sh,1):>7} "
              f"{f(h0j,1):>9} {plat or '—':>5} "
              f"{f(ta,3):>7} {f(te,3):>7}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wd", default="0.03")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(f"results/exp04_full/wd_{args.wd}")
    if not root.is_dir():
        raise SystemExit(f"no such wave dir: {root}")

    late = [pathology_checks(load_seed(root, s)) for s in LATE_SEEDS
            if load_seed(root, s) is not None]
    normal = [pathology_checks(load_seed(root, s)) for s in NORMAL_SEEDS
              if load_seed(root, s) is not None]

    print("=" * 80)
    print(f"EXP-04 LATE-GROKKER DIAGNOSTIC (WD = {args.wd})")
    print(f"  Late seeds:   {LATE_SEEDS}")
    print(f"  Normal seeds: {NORMAL_SEEDS}")
    print("=" * 80)

    print("\nLEGEND")
    print("  spike   = train_loss spikes >10x rolling median (post step 1000)")
    print("  tDD     = max test_acc drawdown after first time test_acc > 0.5")
    print("  wn↑     = max(wn) / wn(step 0); growth ratio")
    print("  sh/med  = max(sharpness) / median(sharpness)")
    print("  h0Δp95  = 95th percentile of |h0[t] - h0[t-1]| / median(h0)")
    print("  plat    = pre-grok plateau steps (test_acc < 0.2 before grokking)")

    per_seed_table(late, "LATE GROKKERS")
    per_seed_table(normal, "NORMAL GROKKERS")
    late_cv = consistency_table(late, "LATE")
    norm_cv = consistency_table(normal, "NORMAL")

    # Verdict synthesis
    print("\n" + "=" * 80)
    print("VERDICT SYNTHESIS")
    print("=" * 80)
    spikes_late = [r["loss_spike_count_post_1k"] for r in late]
    spikes_norm = [r["loss_spike_count_post_1k"] for r in normal]
    tdd_late = [r["test_acc_max_drawdown_post_0.5"] for r in late if r["test_acc_max_drawdown_post_0.5"] is not None]
    tdd_norm = [r["test_acc_max_drawdown_post_0.5"] for r in normal if r["test_acc_max_drawdown_post_0.5"] is not None]
    wn_late = [r["wn_growth_ratio_max_over_initial"] for r in late if r["wn_growth_ratio_max_over_initial"] is not None]
    wn_norm = [r["wn_growth_ratio_max_over_initial"] for r in normal if r["wn_growth_ratio_max_over_initial"] is not None]

    def summary(label, vals):
        if not vals: return f"  {label}: (no data)"
        return f"  {label:<35} mean={mean(vals):>10.3f}  range=[{min(vals):.3f}, {max(vals):.3f}]"

    print()
    print("Pathology indicators (LATE vs NORMAL):")
    print(summary("LATE   loss_spike_count_post_1k:", spikes_late))
    print(summary("NORMAL loss_spike_count_post_1k:", spikes_norm))
    print(summary("LATE   test_acc_max_drawdown:",   tdd_late))
    print(summary("NORMAL test_acc_max_drawdown:",   tdd_norm))
    print(summary("LATE   wn_growth_ratio:",         wn_late))
    print(summary("NORMAL wn_growth_ratio:",         wn_norm))

    # Heuristic verdict
    late_pathology = (mean(spikes_late) > 0.5 if spikes_late else False) \
                  or (mean(tdd_late) > 0.10 if tdd_late else False) \
                  or (mean(wn_late) / (mean(wn_norm) if wn_norm else 1) > 3)
    late_consistent = all(
        (max(vals) - min(vals)) / abs(mean(vals)) < 1.0 if mean(vals) != 0 else True
        for vals in [spikes_late, tdd_late, wn_late] if vals and len(vals) >= 2
    )
    print()
    if late_pathology and not late_consistent:
        verdict = "ARTIFACT_CLUSTER  -- late grokkers show pathology and each looks weird in a different way"
    elif late_pathology and late_consistent:
        verdict = "CONSISTENT_PATHOLOGY  -- late grokkers all share the same pathology pattern (interesting regime, but optimization-induced)"
    elif (not late_pathology) and late_consistent:
        verdict = "REAL_REGIME  -- late grokkers are internally consistent and not pathological; legitimate slow-generalization regime"
    else:
        verdict = "AMBIGUOUS  -- no clear pathology but late grokkers are heterogeneous; need more diagnostics"
    print(f"HEURISTIC VERDICT: {verdict}")
    print()
    print("  NOTE: This is a heuristic synthesis. Trust the per-seed table and CV")
    print("  numbers over the heuristic. The most diagnostic question is whether")
    print("  the 5 late grokkers look like each other (REAL_REGIME) or each like")
    print("  a different kind of weird (ARTIFACT_CLUSTER).")

    out_path = Path(args.out or f"results/exp04_late_grokker_diagnostic_wd_{args.wd}.json")
    out_path.write_text(json.dumps({
        "wd": args.wd,
        "late_seeds": LATE_SEEDS,
        "normal_seeds": NORMAL_SEEDS,
        "late_per_seed": late,
        "normal_per_seed": normal,
        "late_consistency": late_cv,
        "normal_consistency": norm_cv,
        "heuristic_verdict": verdict,
    }, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
