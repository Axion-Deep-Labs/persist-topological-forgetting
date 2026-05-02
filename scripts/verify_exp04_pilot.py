"""EXP-04 pilot verification script.

Run post-re-analysis. Validates:
  1. Step alignment across training/topology/baseline JSONs
  2. Artifact timestamps + bugfix field presence (h0_effective_feature_count)
  3. Checkpoint counts per seed
  4. Grokking onset recomputation from training logs
  5. h0_effective_feature_count is NOT structurally constant (B1 primary check)
  6. Log-space discontinuity scan for PH + baseline signals
  7. Seed 7777 quarantine acknowledgement

Usage:
    .venv/bin/python scripts/verify_exp04_pilot.py
"""
import json, math, os, statistics
from datetime import datetime
from pathlib import Path

IN_SCOPE = [137, 256, 1024]
EXCLUDED_WITH_DATA = [42]
QUARANTINED = [7777]
BASE = Path("results/exp04_pilot")


def load(seed):
    root = BASE / f"seed_{seed}"
    return (
        json.load(open(root / "training_metrics.json")),
        json.load(open(root / "topology_metrics.json")),
        json.load(open(root / "baseline_metrics.json")),
    )


def ckpt_count(seed):
    d = BASE / f"seed_{seed}" / "checkpoints"
    if not d.is_dir():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix in {".pt", ".pth"})


def mtime(p):
    return datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")


def compute_onset(metrics, test_thr=0.9, train_thr=0.99):
    for r in metrics:
        if r.get("test_acc", 0) >= test_thr and r.get("train_acc", 0) >= train_thr:
            return r["step"]
    return None


def find_instabilities(metrics, train_thr=0.9):
    seen_perfect = False
    out = []
    for r in metrics:
        if r.get("train_acc") == 1.0:
            seen_perfect = True
        elif seen_perfect and r.get("train_acc", 1) < train_thr:
            out.append((r["step"], r.get("train_acc"), r.get("test_acc")))
    return out


def log_jumps(series, steps, thresh=2.0, eps=1e-12):
    out = []
    n = min(len(series), len(steps))
    for i in range(1, n):
        a, b = series[i - 1], series[i]
        if a is None or b is None:
            continue
        la = math.log10(max(abs(a), eps))
        lb = math.log10(max(abs(b), eps))
        if abs(lb - la) > thresh:
            out.append((steps[i - 1], steps[i], round(lb - la, 2)))
    return out


def check_seed(seed, *, expect_clean_trajectory):
    tr, to, ba = load(seed)
    tr_metrics = tr["metrics"]
    tr_steps = [r["step"] for r in tr_metrics]
    to_steps = [r.get("step") for r in to]
    ba_steps = [r.get("step") for r in ba]

    root = BASE / f"seed_{seed}"
    ts = {
        "train": mtime(root / "training_metrics.json"),
        "topo": mtime(root / "topology_metrics.json"),
        "base": mtime(root / "baseline_metrics.json"),
    }

    aligned = tr_steps == to_steps == ba_steps

    # B1 primary check: h0_effective_feature_count present AND non-constant
    eff_counts = [r.get("h0_effective_feature_count") for r in to]
    eff_present = all(x is not None for x in eff_counts)
    eff_unique = len({round(x, 6) for x in eff_counts if x is not None})
    eff_stats = (min(eff_counts), max(eff_counts)) if eff_present and eff_counts else (None, None)

    # Legacy field should NOT appear in refreshed JSONs (optional assertion)
    legacy_present = "h0_significant_count" in (to[0] if to else {})

    h0_tp = [r["h0_total_persistence"] for r in to]
    cd = [r["commutator_defect"] for r in ba]
    sh = [r["sharpness"] for r in ba]

    onset = compute_onset(tr_metrics, 0.9, 0.99)
    onset_080 = compute_onset(tr_metrics, 0.8, 0.99)
    onset_095 = compute_onset(tr_metrics, 0.95, 0.99)
    instabilities = find_instabilities(tr_metrics, 0.9)

    print(f"\n--- seed {seed} ---")
    print(f"  artifact timestamps: train={ts['train']}  topo={ts['topo']}  base={ts['base']}")
    print(f"  records:   train={len(tr_metrics)}  topo={len(to)}  base={len(ba)}  checkpoints={ckpt_count(seed)}")
    print(f"  step alignment: {aligned}")
    print(f"  grokking onset (test>=0.9): {onset}   (0.80->{onset_080}, 0.95->{onset_095})")
    print(f"  post-saturation instabilities: {instabilities if instabilities else 'none'}")
    print(f"  h0_effective_feature_count: present={eff_present}  unique_values={eff_unique}  range={eff_stats}")
    print(f"  h0_significant_count legacy field present in new JSON? {legacy_present}")
    print(f"  h0_total_persistence range: [{min(h0_tp):.3g}, {max(h0_tp):.3g}]")
    print(f"  commutator_defect range:    [{min(cd):.3g}, {max(cd):.3g}]")
    sh_vals = [abs(x) for x in sh if x is not None]
    if sh_vals:
        print(f"  |sharpness| range:          [{min(sh_vals):.3g}, {max(sh_vals):.3g}]")

    topo_steps = [r["step"] for r in to]
    base_steps = [r.get("step") for r in ba]
    print(f"  log-space jumps (>2 oom):  "
          f"h0_tp={len(log_jumps(h0_tp, topo_steps))}  "
          f"cd={len(log_jumps(cd, base_steps))}  "
          f"|sh|={len(log_jumps([abs(x) if x is not None else None for x in sh], base_steps))}")

    problems = []
    if not aligned:
        problems.append("step misalignment")
    if not eff_present:
        problems.append("h0_effective_feature_count missing (re-analysis did not run or B1 fix not picked up)")
    if eff_unique < 10:
        problems.append(f"h0_effective_feature_count has only {eff_unique} unique values across {len(eff_counts)} records (expected variation)")
    if legacy_present:
        problems.append("h0_significant_count still present in new JSON (unexpected — check run_pilot did not cache old stats)")
    if expect_clean_trajectory and instabilities:
        problems.append(f"unexpected training instability at {instabilities}")
    return problems


def main():
    print("=" * 80)
    print(f"EXP-04 PILOT VERIFICATION — {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 80)

    all_problems = {}
    for seed in IN_SCOPE:
        all_problems[seed] = check_seed(seed, expect_clean_trajectory=True)
    for seed in EXCLUDED_WITH_DATA:
        all_problems[seed] = check_seed(seed, expect_clean_trajectory=False)

    print("\n" + "=" * 80)
    print("QUARANTINED")
    print("=" * 80)
    for seed in QUARANTINED:
        ck = ckpt_count(seed)
        has_json = (BASE / f"seed_{seed}" / "topology_metrics.json").exists()
        print(f"  seed {seed}: checkpoints={ck}  topology_json_present={has_json}  -> EXCLUDED from verification")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    any_fail = False
    for seed, probs in all_problems.items():
        status = "PASS" if not probs else "FAIL"
        if probs:
            any_fail = True
        tag = "in-scope" if seed in IN_SCOPE else "excluded-but-checked"
        print(f"  seed {seed} ({tag}): {status}")
        for p in probs:
            print(f"    - {p}")

    print("\nB1 primary check (h0_effective_feature_count non-constant across all in-scope seeds):")
    in_scope_unique = []
    for seed in IN_SCOPE:
        _, to, _ = load(seed)
        vals = [r.get("h0_effective_feature_count") for r in to]
        in_scope_unique.append(len({round(v, 6) for v in vals if v is not None}))
    print(f"  unique value counts per seed: {dict(zip(IN_SCOPE, in_scope_unique))}")
    b1_pass = all(u >= 10 for u in in_scope_unique)
    print(f"  B1 repair verified: {b1_pass}")

    print("\nDECISION:")
    if any_fail or not b1_pass:
        print("  Do not scale. Investigate FAIL items above before HPC design.")
    else:
        print("  All in-scope seeds clean, B1 repair verified. OK to proceed to constrained HPC design draft.")


if __name__ == "__main__":
    main()
