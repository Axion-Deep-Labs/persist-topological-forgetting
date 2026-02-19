#!/usr/bin/env python3
"""Generate publication figures for the PERSIST paper.

Reads results/correlation_results.json and produces PDF figures in paper/figures/.
Usage:
    python paper/generate_figures.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# ── Paths ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "results", "correlation_results.json")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ──
with open(DATA_PATH) as f:
    data = json.load(f)

# ── Label mapping ──
LABEL_MAP = {
    "exp01": "ResNet-18",
    "exp01_resnet50": "ResNet-50",
    "exp01_vit": "ViT-Small",
    "exp01_wrn2810": "WRN-28-10",
    "exp01_mlpmixer": "MLP-Mixer",
    "exp01_resnet18wide": "ResNet-18-W",
    "exp01_densenet121": "DenseNet-121",
    "exp01_efficientnet": "EfficientNet-B0",
    "exp01_vgg16bn": "VGG-16-BN",
    "exp01_convnext": "ConvNeXt-T",
    "exp01_mobilenetv3": "MobileNet-V3",
    "exp01_vittiny": "ViT-Tiny",
    "exp01_shufflenet": "ShuffleNet-V2",
    "exp01_regnet": "RegNet-Y-400",
}

# Architecture type → color
TYPE_MAP = {
    "ResNet-18": "CNN",
    "ResNet-50": "CNN",
    "ResNet-18-W": "CNN",
    "WRN-28-10": "CNN",
    "DenseNet-121": "CNN",
    "VGG-16-BN": "CNN",
    "ShuffleNet-V2": "CNN",
    "EfficientNet-B0": "CNN+SE",
    "MobileNet-V3": "CNN+SE",
    "RegNet-Y-400": "CNN+SE",
    "ConvNeXt-T": "Modern CNN",
    "ViT-Small": "Transformer",
    "ViT-Tiny": "Transformer",
    "MLP-Mixer": "MLP",
}

COLORS = {
    "CNN": "#2176AE",
    "CNN+SE": "#57B8FF",
    "Modern CNN": "#B66D0D",
    "Transformer": "#D72638",
    "MLP": "#6B8F71",
}

MARKERS = {
    "CNN": "o",
    "CNN+SE": "s",
    "Modern CNN": "D",
    "Transformer": "^",
    "MLP": "P",
}

# ── Extract per-architecture arrays ──
archs = []
for entry in data["per_architecture"]:
    label = LABEL_MAP[entry["label"]]
    archs.append({
        "name": label,
        "type": TYPE_MAP[label],
        "H0": entry["H0"],
        "H1": entry["H1"],
        "ret100": entry["retention_100"] * 100,  # percent
        "aurc": entry["aurc"],
        "accuracy": entry["accuracy"] * 100,
        "fisher": entry["fisher_trace"],
        "hessian": entry.get("hessian_trace_mean"),
    })

# ── Style defaults ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def add_annotation(ax, archs_data, x_key, y_key, offset_map=None):
    """Add architecture labels next to each point."""
    for a in archs_data:
        x_val = a[x_key]
        y_val = a[y_key]
        if x_val is None or y_val is None:
            continue
        dx, dy = 5, 5
        if offset_map and a["name"] in offset_map:
            dx, dy = offset_map[a["name"]]
        ax.annotate(
            a["name"], (x_val, y_val),
            textcoords="offset points", xytext=(dx, dy),
            fontsize=6, color="0.3",
        )


def scatter_with_fit(ax, archs_data, x_key, y_key, xlabel, ylabel, title,
                     annotate=True, offset_map=None, log_x=False, log_y=False):
    """Scatter plot colored by architecture type with Spearman annotation."""
    xs, ys = [], []
    for a in archs_data:
        x_val = a[x_key]
        y_val = a[y_key]
        if x_val is None or y_val is None:
            continue
        color = COLORS[a["type"]]
        marker = MARKERS[a["type"]]
        ax.scatter(x_val, y_val, c=color, marker=marker, s=50, zorder=3,
                   edgecolors="white", linewidths=0.5)
        xs.append(x_val)
        ys.append(y_val)

    xs, ys = np.array(xs), np.array(ys)

    # Spearman correlation
    rho, p = stats.spearmanr(xs, ys)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.text(0.03, 0.97, f"$\\rho = {rho:.2f}${sig}\n$p = {p:.3f}$\n$n = {len(xs)}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.7"))

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    if annotate:
        add_annotation(ax, archs_data, x_key, y_key, offset_map)


def make_legend(ax):
    """Add a shared legend for architecture types."""
    handles = []
    for atype in ["CNN", "CNN+SE", "Modern CNN", "Transformer", "MLP"]:
        handles.append(Line2D([0], [0], marker=MARKERS[atype], color="w",
                              markerfacecolor=COLORS[atype], markersize=7,
                              markeredgecolor="white", markeredgewidth=0.5,
                              label=atype))
    ax.legend(handles=handles, loc="lower right", framealpha=0.9,
              edgecolor="0.7", fancybox=False)


# ═══════════════════════════════════════════
# Figure 1: H1 Persistence vs Retention@100
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4.5, 3.5))
offsets = {
    "ViT-Tiny": (5, -10),
    "ShuffleNet-V2": (5, -10),
    "ViT-Small": (5, 5),
    "MobileNet-V3": (-70, -10),
    "EfficientNet-B0": (-80, 5),
    "MLP-Mixer": (5, -10),
    "ResNet-18": (-60, -10),
    "ResNet-50": (-55, 5),
    "ResNet-18-W": (-68, 5),
    "DenseNet-121": (5, -10),
}
scatter_with_fit(ax, archs, "H1", "ret100",
                 "$H_1$ Total Persistence", "Retention@100 (%)",
                 "$H_1$ Persistence vs. Knowledge Retention",
                 offset_map=offsets)
make_legend(ax)
fig.savefig(os.path.join(FIG_DIR, "fig1_h1_vs_retention.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig1_h1_vs_retention.png"))
plt.close(fig)
print("  fig1_h1_vs_retention.pdf")

# ═══════════════════════════════════════════
# Figure 2: Fisher Trace vs AURC
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4.5, 3.5))
offsets_fisher = {
    "WRN-28-10": (5, -10),
    "ResNet-18-W": (-68, 5),
    "ViT-Small": (5, -10),
    "ViT-Tiny": (5, 5),
    "MLP-Mixer": (5, -10),
}
scatter_with_fit(ax, archs, "fisher", "aurc",
                 "Fisher Information Trace", "AURC",
                 "Fisher Information vs. Forgetting Resistance",
                 offset_map=offsets_fisher, log_x=True, log_y=True)
make_legend(ax)
fig.savefig(os.path.join(FIG_DIR, "fig2_fisher_vs_aurc.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig2_fisher_vs_aurc.png"))
plt.close(fig)
print("  fig2_fisher_vs_aurc.pdf")

# ═══════════════════════════════════════════
# Figure 3: H0 Persistence vs AURC
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4.5, 3.5))
scatter_with_fit(ax, archs, "H0", "aurc",
                 "$H_0$ Total Persistence", "AURC",
                 "$H_0$ Persistence vs. Area Under Retention Curve")
make_legend(ax)
fig.savefig(os.path.join(FIG_DIR, "fig3_h0_vs_aurc.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig3_h0_vs_aurc.png"))
plt.close(fig)
print("  fig3_h0_vs_aurc.pdf")

# ═══════════════════════════════════════════
# Figure 4: Retention@100 bar chart (sorted)
# ═══════════════════════════════════════════
sorted_archs = sorted(archs, key=lambda a: a["ret100"], reverse=True)
names = [a["name"] for a in sorted_archs]
ret_vals = [a["ret100"] for a in sorted_archs]
bar_colors = [COLORS[a["type"]] for a in sorted_archs]

fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.barh(range(len(names)), ret_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Retention@100 (%)")
ax.set_title("Task A Retention After 100 Steps of Task B Training")
ax.set_xlim(right=max(ret_vals) * 1.25)
ax.grid(True, axis="x", alpha=0.2)

# Add value labels on bars
for i, v in enumerate(ret_vals):
    if v > 0.5:
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7)
    else:
        ax.text(v + 0.1, i, f"{v:.2f}%", va="center", fontsize=7)

make_legend(ax)
fig.savefig(os.path.join(FIG_DIR, "fig4_retention_bar.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig4_retention_bar.png"))
plt.close(fig)
print("  fig4_retention_bar.pdf")

# ═══════════════════════════════════════════
# Figure 5: Correlation heatmap
# ═══════════════════════════════════════════
metric_keys = ["H0", "H1", "fisher_trace", "max_barrier", "hessian_trace_mean", "max_eigenvalue", "max_barrier_normalized"]
metric_labels = ["$H_0$ Pers", "$H_1$ Pers", "Fisher Tr", "Barrier", "Hessian Tr", "$\\lambda_{max}$", "Barrier (N)"]
retention_keys = ["rho_retention", "rho_aurc"]
retention_labels = ["ret@100", "AURC"]

# Build matrix
rho_matrix = np.zeros((len(metric_keys), 2))
p_matrix = np.zeros((len(metric_keys), 2))

for i, mk in enumerate(metric_keys):
    if mk in data["correlations"]:
        corr = data["correlations"][mk]
        rho_matrix[i, 0] = corr["rho_retention"]
        rho_matrix[i, 1] = corr["rho_aurc"]
        p_matrix[i, 0] = corr["p_retention"]
        p_matrix[i, 1] = corr["p_aurc"]

fig, ax = plt.subplots(figsize=(3.5, 4))
im = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

ax.set_xticks(range(2))
ax.set_xticklabels(retention_labels, fontsize=9)
ax.set_yticks(range(len(metric_labels)))
ax.set_yticklabels(metric_labels, fontsize=9)

# Annotate cells with rho and significance stars
for i in range(len(metric_keys)):
    for j in range(2):
        rho = rho_matrix[i, j]
        p = p_matrix[i, j]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        text_color = "white" if abs(rho) > 0.55 else "black"
        ax.text(j, i, f"{rho:.2f}{stars}", ha="center", va="center",
                fontsize=8, color=text_color, fontweight="bold" if stars else "normal")

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Spearman $\\rho$", fontsize=9)
ax.set_title("Metric–Retention Correlations", fontsize=10)
fig.savefig(os.path.join(FIG_DIR, "fig5_correlation_heatmap.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig5_correlation_heatmap.png"))
plt.close(fig)
print("  fig5_correlation_heatmap.pdf")

# ═══════════════════════════════════════════
# Figure 6: 2x2 panel — combined key plots
# ═══════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))

# (a) H1 vs ret@100
scatter_with_fit(axes[0, 0], archs, "H1", "ret100",
                 "$H_1$ Total Persistence", "Retention@100 (%)",
                 "(a) $H_1$ vs. Retention", annotate=False)

# (b) H0 vs AURC
scatter_with_fit(axes[0, 1], archs, "H0", "aurc",
                 "$H_0$ Total Persistence", "AURC",
                 "(b) $H_0$ vs. AURC", annotate=False)

# (c) Fisher vs AURC
scatter_with_fit(axes[1, 0], archs, "fisher", "aurc",
                 "Fisher Information Trace", "AURC",
                 "(c) Fisher vs. AURC", annotate=False, log_x=True, log_y=True)

# (d) Accuracy vs Retention
scatter_with_fit(axes[1, 1], archs, "accuracy", "ret100",
                 "Task A Accuracy (%)", "Retention@100 (%)",
                 "(d) Accuracy vs. Retention", annotate=False)

# Shared legend
handles = []
for atype in ["CNN", "CNN+SE", "Modern CNN", "Transformer", "MLP"]:
    handles.append(Line2D([0], [0], marker=MARKERS[atype], color="w",
                          markerfacecolor=COLORS[atype], markersize=7,
                          markeredgecolor="white", markeredgewidth=0.5,
                          label=atype))
fig.legend(handles=handles, loc="lower center", ncol=5, framealpha=0.9,
           edgecolor="0.7", fancybox=False, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(os.path.join(FIG_DIR, "fig6_panel.pdf"))
fig.savefig(os.path.join(FIG_DIR, "fig6_panel.png"))
plt.close(fig)
print("  fig6_panel.pdf")

print(f"\nAll figures saved to {FIG_DIR}/")
