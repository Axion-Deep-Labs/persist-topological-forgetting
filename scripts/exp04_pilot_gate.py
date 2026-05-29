"""
EXP-04 PILOT GATE evaluation.

Question (config-defined): does at least ONE persistent-homology statistic show
*consistent directional behavior* across the in-scope seeds BEFORE grokking onset?

This is a within-seed temporal question (does the metric trend monotonically in the
run-up to grokking, in the same direction across seeds?) — distinct from the WD-wave
cross-run correlation in exp04_comprehensive_analysis.py.

Definitions (all from configs/exp04_pilot.yaml):
  - onset:  first step with test_acc >= 0.90 AND train_acc >= 0.99
  - pre-grokking window: last `frac` of the [0, onset] span, frac in {0.05, 0.10, 0.20}
    (0.10 primary; 0.05/0.20 sensitivity)
  - directional behavior in a seed: sign of Spearman rho(metric, step) within the window;
    |rho| reports monotonicity strength. We report raw rho per seed (no hidden threshold).
  - gate: a stat PASSES if its trend sign agrees across >= min_consistent_seeds in-scope seeds.

In-scope seeds: 137, 256, 1024 (42 excluded for training collapse, 7777 quarantined).

Usage:
    .venv/bin/python scripts/exp04_pilot_gate.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.exp04_comprehensive_analysis import find_onset, spearman  # noqa: E402

ROOT = Path("results/exp04_pilot")
IN_SCOPE = [137, 256, 1024]
EXCLUDED = {42: "training collapse @64K", 7777: "quarantined (no checkpoints)"}

PRIMARY = "h0_total_persistence"
SECONDARY = ["h0_effective_feature_count", "h0_persistence_entropy", "h0_median_persistence"]
COMPARATOR = "commutator_defect"   # baseline_metrics.json, not topology
WINDOW_FRACS = [0.05, 0.10, 0.20]
PRIMARY_FRAC = 0.10
MIN_CONSISTENT = 3
MONOTONICITY_FLOOR = 0.5   # |rho| below this = weak/ambiguous trend (reported, flagged)


def load(seed):
    sd = ROOT / f"seed_{seed}"
    topo = json.load(open(sd / "topology_metrics.json"))
    base = json.load(open(sd / "baseline_metrics.json"))
    train = json.load(open(sd / "training_metrics.json"))
    topo = topo.get("records") if isinstance(topo, dict) else topo
    train = train.get("metrics") if isinstance(train, dict) else train
    onset = find_onset(train)
    return topo, base, onset


def window_trend(records, key, onset, frac):
    """Spearman rho(metric, step) within the last `frac` of the [0, onset] span."""
    lo = onset * (1.0 - frac)
    pts = [(r["step"], r.get(key)) for r in records
           if lo <= r["step"] <= onset and r.get(key) is not None]
    if len(pts) < 3:
        return None, len(pts)
    steps = [p[0] for p in pts]
    vals = [p[1] for p in pts]
    return spearman(vals, steps), len(pts)


def main():
    print("=" * 78)
    print("EXP-04 PILOT GATE — pre-onset directional behavior")
    print("=" * 78)
    print(f"  In-scope seeds: {IN_SCOPE}")
    for s, why in EXCLUDED.items():
        print(f"  Excluded seed {s}: {why}")
    print(f"  Onset rule: test_acc>=0.90 & train_acc>=0.99")
    print(f"  Window: last {WINDOW_FRACS} of [0, onset]; primary={PRIMARY_FRAC}")
    print(f"  Gate: trend sign agrees across >= {MIN_CONSISTENT}/{len(IN_SCOPE)} seeds")
    print(f"  Monotonicity floor (|rho| flag): {MONOTONICITY_FLOOR}")
    print()

    data = {}
    for s in IN_SCOPE:
        topo, base, onset = load(s)
        data[s] = {"topo": topo, "base": base, "onset": onset}
        print(f"  seed {s}: onset={onset}")
    print()

    stats = [(PRIMARY, "topo", "PRIMARY")] + \
            [(k, "topo", "secondary") for k in SECONDARY] + \
            [(COMPARATOR, "base", "comparator")]

    gate_pass_any = False
    results = []
    for key, src, tier in stats:
        print(f"--- {key}  [{tier}] ---")
        per_frac = {}
        for frac in WINDOW_FRACS:
            rhos, ns = {}, {}
            for s in IN_SCOPE:
                recs = data[s][src]
                rho, n = window_trend(recs, key, data[s]["onset"], frac)
                rhos[s] = rho
                ns[s] = n
            valid = {s: r for s, r in rhos.items() if r is not None}
            signs = [(1 if r > 0 else -1) for r in valid.values()]
            pos = sum(1 for x in signs if x > 0)
            neg = sum(1 for x in signs if x < 0)
            consistent_n = max(pos, neg)
            direction = "increase" if pos >= neg else "decrease"
            strong = sum(1 for r in valid.values() if abs(r) >= MONOTONICITY_FLOOR)
            per_frac[frac] = {
                "rhos": rhos, "ns": ns, "consistent_n": consistent_n,
                "direction": direction, "strong": strong, "n_valid": len(valid),
            }
            rho_str = "  ".join(
                f"s{s}={('%+.3f' % rhos[s]) if rhos[s] is not None else 'n<3'}(n={ns[s]})"
                for s in IN_SCOPE)
            flag = "PASS" if consistent_n >= MIN_CONSISTENT and len(valid) >= MIN_CONSISTENT else "fail"
            mark = "" if frac != PRIMARY_FRAC else "  <-- primary"
            print(f"  frac={frac:<4}: {rho_str}  -> {consistent_n}/{len(valid)} {direction}, "
                  f"{strong} strong  [{flag}]{mark}")
        # gate decision uses the primary window fraction
        p = per_frac[PRIMARY_FRAC]
        passed = p["consistent_n"] >= MIN_CONSISTENT and p["n_valid"] >= MIN_CONSISTENT
        results.append((key, tier, passed, p))
        if passed and tier in ("PRIMARY", "secondary"):
            gate_pass_any = True
        print()

    print("=" * 78)
    print("GATE SUMMARY (primary window frac = %.2f)" % PRIMARY_FRAC)
    print("=" * 78)
    for key, tier, passed, p in results:
        strongnote = f", {p['strong']}/{p['n_valid']} strong (|rho|>={MONOTONICITY_FLOOR})"
        print(f"  {key:<32} [{tier:<10}] "
              f"{p['consistent_n']}/{p['n_valid']} {p['direction']}{strongnote}  "
              f"-> {'PASS' if passed else 'fail'}")
    print()
    ph_pass = [r for r in results if r[2] and r[1] in ("PRIMARY", "secondary")]
    primary_passed = next(r[2] for r in results if r[1] == "PRIMARY")
    sec_pass = [r[0] for r in results if r[2] and r[1] == "secondary"]
    comp_pass = next((r[2] for r in results if r[1] == "comparator"), False)

    # This is a go/no-go screen, NOT a success test. Report the pre-registered
    # primary outcome on its own line — a secondary-carried pass is NOT a positive
    # result for the hypothesis. (Decision posture set by Joshua, 2026-05-28.)
    print("PILOT OUTCOME: MIXED — does NOT establish a positive result.")
    print(f"  PRIMARY  (h0_total_persistence): {'POSITIVE' if primary_passed else 'NEGATIVE'}")
    print(f"  SECONDARY: {'POSITIVE — ' + ', '.join(sec_pass) if sec_pass else 'negative'}")
    print(f"  COMPARATOR (commutator_defect): {'POSITIVE' if comp_pass else 'negative'}")
    print(f"  INTERPRETATION: inconclusive — pre-onset windows are sparse (n=3 checkpoints");
    print(f"    at the 10% window; n=2 at 5%, uncomputable). This is a cadence/resolution")
    print(f"    limitation, not a clean test of the temporal signal.")
    print()
    if not primary_passed and (sec_pass or comp_pass):
        print("  DECISION: the pre-registered PRIMARY endpoint was NOT supported, but")
        print("  multiple pre-registered SECONDARY endpoints and the comparator were.")
        print("  This MOTIVATES a larger, properly powered replication; it does not count")
        print("  as a pilot success. Proceed to constrained HPC study with (a) dense")
        print("  pre-onset checkpointing, (b) n=30 seeds, (c) wider WD coverage.")
        print("  DO NOT revise the endpoint hierarchy — primary stays h0_total_persistence.")
        print("  If the primary fails AGAIN at n=30 with dense checkpoints, that is a real")
        print("  negative result.")
    elif primary_passed:
        print("  DECISION: primary supported — proceed, but confirm at scale before claiming.")
    else:
        print("  DECISION: no endpoint supported even with sparse windows. Per")
        print("  pre-registration, STOP and reconsider before scaling.")
    print()
    print("  NOTE: n=3 in-scope seeds. Go/no-go screen, not an effect-size estimate.")
    print("  Directional sign is the endpoint; |rho| reported for transparency only.")


if __name__ == "__main__":
    main()
