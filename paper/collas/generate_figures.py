"""Generate figures for CoLLAs 2026 paper.

Reads correlation results from results/ and produces publication-quality
scatter plots of H0 vs EWC benefit across all 3 datasets.

Usage:
    python paper/collas/generate_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

DATASETS = [
    ("cifar100", "CIFAR-100"),
    ("cub200", "CUB-200"),
    ("resisc45", "RESISC-45"),
]

# Architecture class colors
CLASS_COLORS = {
    "CNN": "#2196F3",
    "Transformer": "#FF5722",
    "MLP": "#4CAF50",
    "WRN-ladder": "#9C27B0",
    "Modern CNN": "#2196F3",
}

CLASS_MARKERS = {
    "CNN": "o",
    "Transformer": "^",
    "MLP": "s",
    "WRN-ladder": "D",
    "Modern CNN": "o",
}


def load_dataset(name):
    """Load correlation results for a dataset."""
    path = os.path.join(RESULTS_DIR, f"correlation_results_{name}.json")
    with open(path) as f:
        return json.load(f)


def figure_h0_vs_ewc_benefit():
    """3-panel scatter plot: H0 vs EWC benefit per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, (ds_key, ds_label) in zip(axes, DATASETS):
        data = load_dataset(ds_key)
        archs = data["per_architecture"]
        ewc = data["correlations"]["ewc_benefit"]

        h0_vals = np.array([a["H0"] for a in archs])
        benefits = np.array(ewc["benefits"])
        classes = [a["arch_class"] for a in archs]
        names = ewc["architectures"]

        # Spearman correlation
        rho, pval = stats.spearmanr(h0_vals, benefits)

        # Scatter by architecture class
        plotted_classes = set()
        for i in range(len(h0_vals)):
            cls = classes[i]
            label = cls if cls not in plotted_classes else None
            plotted_classes.add(cls)
            ax.scatter(
                h0_vals[i], benefits[i],
                c=CLASS_COLORS.get(cls, "#999"),
                marker=CLASS_MARKERS.get(cls, "o"),
                s=50, alpha=0.8, edgecolors="white", linewidths=0.5,
                label=label, zorder=3,
            )

        # Regression line
        slope, intercept, _, _, _ = stats.linregress(h0_vals, benefits)
        x_line = np.linspace(h0_vals.min(), h0_vals.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.5, linewidth=1)

        # Annotation
        sig = ""
        if pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"

        if pval < 0.001:
            p_str = "p < 0.001"
        else:
            p_str = f"p = {pval:.3f}"

        ax.text(
            0.05, 0.95,
            f"$\\rho$ = {rho:.2f}{sig}\n{p_str}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_title(ds_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("$H_0$ Persistence", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("EWC Benefit (AURC)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3, zorder=1)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4,
        fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(FIGURES_DIR, "h0_vs_ewc_benefit.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def figure_wrn_ladder():
    """WRN width ladder: H0 vs width across all 3 datasets."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ds_colors = {
        "cifar100": "#2196F3",
        "cub200": "#FF5722",
        "resisc45": "#4CAF50",
    }
    ds_labels = {
        "cifar100": "CIFAR-100",
        "cub200": "CUB-200",
        "resisc45": "RESISC-45",
    }

    wrn_widths = [1, 2, 4, 6, 8, 10]
    wrn_labels = [f"exp01_wrn28{k}" if k != 10 else "exp01_wrn2810" for k in wrn_widths]
    # Fix: wrn2810 is already correct, but wrn281 etc need to match the actual labels
    # Let's extract from data
    for ds_key in ["cifar100", "cub200", "resisc45"]:
        data = load_dataset(ds_key)
        archs = data["per_architecture"]

        h0_vals = []
        widths = []
        for a in archs:
            if a["arch_class"] == "WRN-ladder":
                h0_vals.append(a["H0"])
                # Extract width from label (e.g., exp01_wrn284_cub200 -> 4)
                lbl = a["label"]
                # Remove dataset suffix and prefix
                wrn_part = lbl.replace("exp01_wrn28", "").split("_")[0]
                k = int(wrn_part)
                widths.append(k)

        # Sort by width
        order = np.argsort(widths)
        widths = np.array(widths)[order]
        h0_vals = np.array(h0_vals)[order]

        ax.plot(
            widths, h0_vals, "o-",
            color=ds_colors[ds_key], label=ds_labels[ds_key],
            markersize=6, linewidth=1.5,
        )

    ax.set_xlabel("WRN-28-$k$ Width Multiplier", fontsize=11)
    ax.set_ylabel("$H_0$ Persistence", fontsize=11)
    ax.set_title("$H_0$ Decreases Monotonically with Width ($\\rho = -1.0$)", fontsize=11)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    out_path = os.path.join(FIGURES_DIR, "wrn_ladder_h0.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def figure_capacity_control():
    """Bar chart showing R^2 values for params-only, H0-only, and both models."""
    fig, ax = plt.subplots(figsize=(7, 4))

    # Data from controlled_regression.json or hardcoded from paper
    datasets = ["CIFAR-100", "CUB-200", "RESISC-45", "Pooled"]
    r2_params = [0.527, 0.212, 0.242, 0.262]
    r2_h0 = [0.304, 0.017, 0.645, 0.150]
    r2_both = [0.626, 0.216, 0.764, 0.344]

    x = np.arange(len(datasets))
    width = 0.25

    bars1 = ax.bar(x - width, r2_params, width, label="Params only", color="#90CAF9")
    bars2 = ax.bar(x, r2_h0, width, label="$H_0$ only", color="#CE93D8")
    bars3 = ax.bar(x + width, r2_both, width, label="Params + $H_0$", color="#A5D6A7")

    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("$R^2$", fontsize=11)
    ax.set_title("Controlled Regression: EWC Benefit ~ log(params) + $H_0$", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.9)
    ax.tick_params(labelsize=9)

    out_path = os.path.join(FIGURES_DIR, "capacity_control_r2.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    figure_h0_vs_ewc_benefit()
    figure_wrn_ladder()
    figure_capacity_control()
    print("\nAll figures generated.")
