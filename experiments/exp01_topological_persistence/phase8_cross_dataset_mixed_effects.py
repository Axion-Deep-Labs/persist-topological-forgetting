"""
EXP-01 Phase 8: Cross-dataset mixed-effects analysis.

Pools per-pair Phase 4 outputs across all available cross-dataset pairs and runs
mixed-effects models with architecture as a random effect, addressing the
pseudo-replication problem identified in v2 of the Phase I-B plan (each
architecture appears in multiple pairs).

Headline questions:

  Q1: Does Task A topology (H0/H1) predict cross-dataset retention,
      controlling for log_params and architecture clustering?

  Q4: Does Task A topology predict EWC benefit (the Phase I-A signature),
      replicated cross-dataset?

H0 and H1 are reported as co-primary per v2 D3 (no hidden choice of metric).

Usage:
    .venv/bin/python -m experiments.exp01_topological_persistence.phase8_cross_dataset_mixed_effects
    .venv/bin/python -m experiments.exp01_topological_persistence.phase8_cross_dataset_mixed_effects \
        --results-dir results --output results/phase8_xd_mixed_effects.json
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_records(results_dir):
    """Load all correlation_results_xd_*.json files into a long-format dataframe.

    Each row is one (architecture, pair) observation. The same arch will appear
    multiple times across different pairs; that repetition is what the random
    effect on `arch_name` is meant to absorb.
    """
    pattern = os.path.join(results_dir, "correlation_results_xd_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No cross-dataset correlation files found in {results_dir}. "
            f"Run phase4_correlation.py with --cross-dataset-pair first."
        )

    records = []
    for f in files:
        # Filename pattern: correlation_results_xd_<task_a>_to_<task_b>.json
        basename = os.path.basename(f)
        tag = basename.replace("correlation_results_xd_", "").replace(".json", "")
        try:
            task_a, task_b = tag.split("_to_")
        except ValueError:
            print(f"  Skipping unrecognized filename: {basename}")
            continue
        pair_id = f"{task_a}_to_{task_b}"

        with open(f) as fp:
            data = json.load(fp)

        for arch in data.get("per_architecture", []):
            h0 = arch.get("H0")
            num_params = arch.get("num_params")
            ret_10 = arch.get("ret_10")
            if h0 is None or num_params is None or ret_10 is None:
                continue

            records.append({
                "pair_id": pair_id,
                "task_a": task_a,
                "task_b": task_b,
                "arch_name": arch.get("arch_name", "Unknown"),
                "arch_class": arch.get("arch_class", "Unknown"),
                "H0": float(h0),
                "H1": float(arch["H1"]) if arch.get("H1") is not None else np.nan,
                "num_params": float(num_params),
                "log_params": float(np.log10(num_params)),
                "ret_10": float(ret_10),
                "ewc_ret_10": float(arch["ewc_ret_10"]) if arch.get("ewc_ret_10") is not None else np.nan,
            })

    if not records:
        raise ValueError("No usable records found across all cross-dataset files.")

    df = pd.DataFrame(records)
    df["ewc_benefit"] = df["ewc_ret_10"] - df["ret_10"]

    # Pooled Z-scores across the full dataset (so coefficients are comparable)
    df["H0z"] = (df["H0"] - df["H0"].mean()) / df["H0"].std(ddof=0)
    if df["H1"].notna().any() and df["H1"].std(ddof=0) > 0:
        df["H1z"] = (df["H1"] - df["H1"].mean()) / df["H1"].std(ddof=0)
    else:
        df["H1z"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------


def fit_mixed(formula, data, groups_col="arch_name"):
    """Fit a mixed-effects model with random intercept on `groups_col`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, data=data, groups=data[groups_col]).fit()
    return model


def summarize_model(model, params_of_interest, label):
    """Print a focused coefficient table for the parameters we care about."""
    print(f"\n  {label}")
    print(f"  {'param':>12} | {'coef':>10} | {'se':>8} | {'95% CI':>22} | {'p':>8}")
    print("  " + "-" * 70)
    ci = model.conf_int()
    for p in params_of_interest:
        if p not in model.params.index:
            print(f"  {p:>12} | NOT IN MODEL")
            continue
        coef = float(model.params[p])
        se = float(model.bse[p])
        pval = float(model.pvalues[p])
        lo = float(ci.loc[p, 0])
        hi = float(ci.loc[p, 1])
        ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
        sig = " *" if pval < 0.05 else ""
        print(f"  {p:>12} | {coef:>+10.4f} | {se:>8.4f} | {ci_str:>22} | {pval:>8.4f}{sig}")
    print(f"\n  log-likelihood: {model.llf:.2f}    n: {int(model.nobs)}")


def model_to_dict(model, params_of_interest):
    ci = model.conf_int()
    params_out = {}
    for p in model.params.index:
        params_out[p] = {
            "coef": float(model.params[p]),
            "se": float(model.bse[p]),
            "p": float(model.pvalues[p]),
            "ci_low": float(ci.loc[p, 0]),
            "ci_high": float(ci.loc[p, 1]),
            "of_interest": p in params_of_interest,
        }
    return {
        "n": int(model.nobs),
        "log_likelihood": float(model.llf),
        "params": params_out,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="EXP-01 Phase 8: Cross-dataset mixed-effects analysis"
    )
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing correlation_results_xd_*.json")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <results-dir>/phase8_xd_mixed_effects.json)")
    args = parser.parse_args()

    print("=" * 72)
    print("PHASE 8: CROSS-DATASET MIXED-EFFECTS ANALYSIS")
    print("=" * 72)

    df = load_records(args.results_dir)
    print(f"\nLoaded {len(df)} records from {df['pair_id'].nunique()} pair(s)")
    print(f"Unique architectures: {df['arch_name'].nunique()}")
    print(f"Pairs:")
    for p in sorted(df["pair_id"].unique()):
        n_p = (df["pair_id"] == p).sum()
        print(f"  {p}: n={n_p}")

    # ---- Q1: retention ----
    print("\n" + "=" * 72)
    print("Q1: H0/H1 -> cross-dataset retention")
    print("=" * 72)
    q1_data = df.dropna(subset=["ret_10", "H0z", "H1z", "log_params"])
    print(f"  formula: ret_10 ~ H0z + H1z + log_params  +  (1 | arch_name)")
    print(f"  n records: {len(q1_data)}")
    print(f"  n unique archs: {q1_data['arch_name'].nunique()}")
    q1_model = fit_mixed("ret_10 ~ H0z + H1z + log_params", q1_data)
    summarize_model(q1_model, ["H0z", "H1z", "log_params"], "Q1 coefficients")

    # ---- Q4: EWC benefit ----
    print("\n" + "=" * 72)
    print("Q4: H0/H1 -> EWC benefit (Phase I-A signature replication)")
    print("=" * 72)
    q4_data = df.dropna(subset=["ewc_benefit", "H0z", "H1z", "log_params"])
    print(f"  formula: ewc_benefit ~ H0z + H1z + log_params  +  (1 | arch_name)")
    print(f"  n records: {len(q4_data)}")
    print(f"  n unique archs: {q4_data['arch_name'].nunique()}")
    if len(q4_data) < 5:
        print(f"\n  WARNING: only {len(q4_data)} EWC records, mixed model may fail to converge")
    q4_model = fit_mixed("ewc_benefit ~ H0z + H1z + log_params", q4_data)
    summarize_model(q4_model, ["H0z", "H1z", "log_params"], "Q4 coefficients")

    # ---- Save ----
    output_path = args.output or os.path.join(
        args.results_dir, "phase8_xd_mixed_effects.json"
    )
    out = {
        "n_records": int(len(df)),
        "n_pairs": int(df["pair_id"].nunique()),
        "n_architectures": int(df["arch_name"].nunique()),
        "pairs": sorted(df["pair_id"].unique().tolist()),
        "Q1_retention": {
            "formula": "ret_10 ~ H0z + H1z + log_params + (1 | arch_name)",
            **model_to_dict(q1_model, ["H0z", "H1z", "log_params"]),
        },
        "Q4_ewc_benefit": {
            "formula": "ewc_benefit ~ H0z + H1z + log_params + (1 | arch_name)",
            **model_to_dict(q4_model, ["H0z", "H1z", "log_params"]),
        },
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
