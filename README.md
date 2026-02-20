# PERSIST: Topological Signatures of Knowledge Persistence

**Axion Deep Labs** | [axiondeep.com](https://www.axiondeep.com)

Catastrophic forgetting remains one of the central obstacles in continual learning. Most mitigation methods focus on regularization or replay, treating the symptom rather than examining the structure. We ask a different question: is a model's susceptibility to forgetting encoded in the geometry of its loss landscape before any sequential training occurs?

Standard curvature metrics (Hessian trace, sharpness) measure how steep the basin is. Topology measures something orthogonal: the *connectivity structure* of the loss surface — whether the basin is a smooth bowl, a fragmented archipelago, or a landscape threaded with ridges that form loops. We compute persistent homology on 2D cross-sections of loss landscapes across 14 architectures, then correlate topological features with knowledge retention under naive sequential training.

```
  Smooth basin (H1 = 0)          Basin with loop (H1 > 0)

       ___________                    ___________
      /           \                  /     __    \
     /             \                /     /  \    \
    /               \              /     / ridge   \
   /    minimum      \            /    minimum \    \
  /                   \          /         \___|    \
 /                     \        /                    \
```

## Scientific Question

Does topological structure of the loss landscape predict forgetting resistance? And if so, does topology capture something that model size alone does not?

## Main Findings

### The tension: topology correlates, but scale dominates

Across 14 architectures on CIFAR-100, H1 persistence (loop structure in the loss landscape) correlates with knowledge retention at ρ = 0.61 (p = 0.021, explaining 37% of rank variance). But parameter count correlates more strongly (ρ = −0.74, p = 0.002), and after partialing out model size, H1 drops to non-significance (partial ρ = 0.35, p = 0.24).

Smaller models retain more. Smaller models also tend to have more topological complexity. The question is whether topology carries independent signal or merely tracks scale.

### The most interesting result: within-class signal

When we restrict analysis to CNNs only (n=11), removing the confound of architecture-family differences, H1 re-emerges as a significant predictor of retention (ρ = 0.66, p = 0.026). This suggests that *within an architecture family*, where parameter count varies less dramatically, topological structure does carry information about forgetting resistance that scale alone does not explain.

This is the core scientific tension of the project. If topology is merely a geometric manifestation of scale, then width-controlled experiments (e.g., a WRN width ladder at fixed depth) should eliminate the H1 signal. If not, topology carries independent structural information. That experiment will determine whether PERSIST yields a correlational curiosity or a geometric insight.

### Where we stand honestly

| Finding | Status |
|---------|--------|
| H1 correlates with retention (ρ = 0.61) | Nominally significant, does not survive Bonferroni |
| Parameter count dominates (ρ = −0.74) | Survives Bonferroni (p_Bonf = 0.02) |
| H1 independent of scale? | Not yet — partial ρ = 0.35, p = 0.24 |
| Within-CNN H1 signal (ρ = 0.66) | Significant, needs width-controlled validation |
| CIFAR-10 replication | H1 non-significant (ρ ~ 0.13, p ~ 0.65); floor effect limits power |

## Caveats

- **Multiple comparisons:** 10 metrics tested; Bonferroni-adjusted α = 0.005. Only parameter count survives correction. H1 is nominally significant (p = 0.021, p_Bonf = 0.21).
- **Parameter count confound:** H1 and params are collinear (ρ = −0.55, VIF = 1.45). Rank regression R² = 0.61 with only params significant.
- **CIFAR-10 floor effect:** Retention collapses to near-zero for most architectures by step 100, limiting statistical power on the second dataset.
- **Multi-slice status:** Infrastructure supports 5 independent random 2D slices per architecture. A seed bug was recently fixed; true independent slices are pending re-run. Results shown are from the initial runs.
- **Sample size:** n = 14 architectures. Small-sample rank correlation should be interpreted with caution.

## Architectures (n=14, CIFAR-100)

Sorted by retention (descending).

| Architecture | Params | Task A Acc | ret@100 | H1 Pers | Type |
|---|---|---|---|---|---|
| ViT-Tiny | 0.8M | 52.7% | 22.5% | 0.18 | Transformer |
| ShuffleNet-V2 | 1.3M | 76.8% | 17.3% | 0.69 | CNN |
| ViT-Small | 3.0M | 62.2% | 9.6% | 0.32 | Transformer |
| MobileNet-V3-S | 1.5M | 68.6% | 7.6% | 1.90 | CNN+SE |
| EfficientNet-B0 | 4.1M | 76.6% | 7.1% | 2.12 | CNN+SE |
| RegNet-Y-400MF | 4.3M | 72.2% | 2.0% | 0.02 | CNN+SE |
| VGG-16-BN | 15.0M | 78.4% | 0.8% | 0.00 | CNN |
| WRN-28-10 | 36.5M | 84.0% | 0.3% | 0.08 | CNN |
| ResNet-18 | 11.2M | 82.0% | 0.2% | 0.00 | CNN |
| ResNet-50 | 23.6M | 83.6% | 0.1% | 0.00 | CNN |
| DenseNet-121 | 7.0M | 84.5% | 0.05% | 0.26 | CNN |
| MLP-Mixer | 2.3M | 61.5% | 0.03% | 0.00 | MLP |
| ConvNeXt-Tiny | 28.0M | 56.7% | 0.0%* | 0.11 | Modern CNN |
| ResNet-18 Wide | 44.7M | 83.1% | 0.0% | 0.00 | CNN |

*ConvNeXt shows non-monotonic recovery (0% -> 3% -> 0%).

## Experimental Design

### Pipeline

```
Phase 1:   Train on Task A (classes 0-49)  ->  converged checkpoint
Phase 2:   Sample 50x50 loss landscape     ->  persistent homology (H0, H1)
         + Baseline metrics                ->  Hessian, Fisher, sharpness, barrier
         + Landscape validation            ->  NaN/Inf/degeneracy checks
Phase 2b:  Displacement analysis           ->  trajectory through landscape after forgetting
Phase 3:   Train on Task B (classes 50-99) ->  forgetting curve at 8 intervals
         + Task B learning validation      ->  warn if Task B fails to converge
Phase 4:   Spearman + Kendall correlation  ->  Bonferroni, LOO, permutation, partials
```

### Topology construction

1. **Landscape sampling:** Two filter-normalized random directions (Li et al., 2018). Loss evaluated on a 50x50 grid over [-1, 1]^2. Grid validated for NaN/Inf/degeneracy.
2. **Weighted graph:** 8-connected grid with lower-star edge weights: w(u,v) = max(f(u), f(v)).
3. **Persistent homology:** Sparse distance matrix -> Ripser -> H0 (components) and H1 (loops) persistence diagrams.
4. **Multi-slice:** Infrastructure supports 5 independent random 2D slices per architecture (currently re-running after seed fix); Phase 4 aggregates (mean +/- std).

### Statistical methods

- **Spearman rank correlation** + **Kendall's tau** (more robust to ties at small n)
- **Bonferroni correction** for multiple testing (10 metrics, adjusted alpha = 0.005)
- **Partial correlation** controlling for parameter count
- **Symmetric partials** + rank regression + VIF to disentangle H1 vs params
- **Permutation test** (10,000 shuffles) for non-parametric significance
- **Leave-one-out** cross-validation (min/mean/max p-values across folds)

**Full correlation table (CIFAR-100, n=14):**

| Metric | rho (ret@100) | p | p_Bonf | tau (Kendall) |
|--------|------------|---|--------|-------------|
| **H1 persistence** (loops) | **0.61** | 0.021 | 0.210 | 0.47 |
| H0 persistence (components) | 0.32 | 0.263 | 1.000 | 0.24 |
| Fisher trace | -0.50 | 0.072 | 0.720 | -0.38 |
| **Parameter count** | **-0.74** | 0.002 | **0.020** | -0.60 |

LOO: 14/14 folds significant for H1. Permutation p ~ 0.02.

## Reproducibility

```bash
# Setup
python -m venv .venv && source .venv/bin/activate && pip install -e .

# Run via dashboard (recommended)
.venv/bin/python dashboard/app.py    # http://localhost:5050

# Or run manually
python -m experiments.exp01_topological_persistence.phase1_train_task_a --config configs/exp01.yaml
python -m experiments.exp01_topological_persistence.phase2_landscape_topology --config configs/exp01.yaml
python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting --config configs/exp01.yaml

# Multi-slice (run-id 1-4, plus default = 5 slices)
python -m experiments.exp01_topological_persistence.phase2_landscape_topology \
    --config configs/exp01.yaml --run-id 1

# Multi-seed
python -m experiments.exp01_topological_persistence.phase1_train_task_a \
    --config configs/exp01.yaml --seed 123

# Cross-architecture correlation (after all architectures complete)
python -m experiments.exp01_topological_persistence.phase4_correlation \
    --results-dirs results/exp01 results/exp01_resnet50 results/exp01_vit \
    results/exp01_wrn2810 results/exp01_mlpmixer results/exp01_resnet18wide \
    results/exp01_densenet121 results/exp01_efficientnet results/exp01_vgg16bn \
    results/exp01_convnext results/exp01_mobilenetv3 results/exp01_vittiny \
    results/exp01_shufflenet results/exp01_regnet
```

Phase 4 automatically aggregates across slices (mean +/- std) when multiple `topology_summary_run*.json` files exist.

## Engineering Details

### Project structure

```
configs/                  # YAML configs (one per architecture x dataset)
dashboard/                # Flask dashboard (localhost:5050)
experiments/
  shared/                 # Datasets, models (14 archs), baseline metrics, utils
  exp01_.../              # Phase 1-4 scripts + Phase 2b displacement analysis
paper/                    # LaTeX draft + figures
results/                  # Output (gitignored)
data/                     # Datasets (gitignored, auto-downloaded)
```

### Dashboard

The Flask dashboard (port 5050) provides: experiment queue with auto-chaining, per-phase re-run, multi-slice progress tracking, Clean & Rebuild for invalidated runs, and live GPU/CPU monitoring.

## References

- Li et al. (2018) — Visualizing the Loss Landscape of Neural Nets
- Bauer (2021) — Ripser: efficient computation of Vietoris-Rips persistence barcodes
- Kirkpatrick et al. (2017) — Overcoming catastrophic forgetting in neural networks
- Keskar et al. (2017) — On large-batch training for deep learning

## License

MIT

## Citation

Paper in preparation. Check back soon.
