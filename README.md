# PERSIST: Topological Signatures of Knowledge Persistence

**Axion Deep Labs** | [axiondeep.com](https://www.axiondeep.com)

Catastrophic forgetting remains one of the central obstacles in continual learning. Most mitigation methods focus on regularization or replay, treating the symptom rather than examining the structure. We ask a different question: is a model's susceptibility to forgetting encoded in the geometry of its loss landscape before any sequential training occurs?

Standard curvature metrics (Hessian trace, sharpness) measure how steep the basin is. Topology measures something orthogonal: the *connectivity structure* of the loss surface. We compute persistent homology on 2D cross-sections of loss landscapes across 19 architectures and 3 datasets, then correlate topological features with knowledge retention under naive sequential training.

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

## The Decisive Experiment: WRN Width Ladder

Previous results (14 architectures on CIFAR-100) showed H1 persistence correlating with retention at rho = 0.61, but parameter count dominated (rho = -0.74). After partialing out model size, H1 dropped to non-significance (partial rho = 0.35, p = 0.24). The central confound: topology and scale are entangled when comparing architectures with different depths, widths, and inductive biases.

To resolve this, we introduce a **WRN-28-k width ladder** (k = 1, 2, 4, 6, 8, 10). Same architecture, same depth, varying only width. This isolates scale from topology. If H1 still correlates with retention within this ladder, topology carries independent signal. If not, the honest conclusion is that scale dominates.

## Datasets (3)

| Dataset | Classes | Task Split | Domain | Purpose |
|---------|---------|------------|--------|---------|
| **CIFAR-100** | 100 | 50 / 50 | Natural images | Standard benchmark |
| **CUB-200-2011** | 200 | 100 / 100 | Fine-grained birds | Gold standard CL benchmark |
| **NWPU-RESISC45** | 45 | 23 / 22 | Satellite scenes | Cross-domain validation |

All datasets resized to 32x32 for model consistency across architectures.

## Architectures (19)

14 diverse architectures spanning CNNs, Transformers, and MLPs, plus a 6-point WRN width ladder.

| Architecture | Params | Type |
|---|---|---|
| ViT-Tiny | ~0.3M | Transformer |
| WRN-28-1 | ~0.4M | WRN-ladder |
| MobileNet-V3-S | ~1.1M | CNN+SE |
| ShuffleNet-V2 | ~1.3M | CNN |
| WRN-28-2 | ~1.5M | WRN-ladder |
| MLP-Mixer | ~2.3M | MLP |
| ViT-Small | ~3.0M | Transformer |
| RegNet-Y-400MF | ~3.9M | CNN+SE |
| EfficientNet-B0 | ~4.1M | CNN+SE |
| WRN-28-4 | ~5.9M | WRN-ladder |
| DenseNet-121 | ~7.0M | CNN |
| ResNet-18 | ~11.2M | CNN |
| WRN-28-6 | ~13.0M | WRN-ladder |
| VGG-16-BN | ~14.7M | CNN |
| WRN-28-8 | ~23.4M | WRN-ladder |
| ResNet-50 | ~23.6M | CNN |
| ConvNeXt-Tiny | ~27.9M | Modern CNN |
| WRN-28-10 | ~36.5M | WRN-ladder |
| ResNet-18 Wide | ~44.7M | CNN |

## Preliminary Findings (CIFAR-100, 14 original architectures)

### The tension: topology correlates, but scale dominates

H1 persistence (loop structure in the loss landscape) correlates with knowledge retention at rho = 0.61 (p = 0.021, explaining 37% of rank variance). But parameter count correlates more strongly (rho = -0.74, p = 0.002), and after partialing out model size, H1 drops to non-significance (partial rho = 0.35, p = 0.24).

### Within-class signal is promising

When restricted to CNNs only (n=11), removing architecture-family confounds, H1 re-emerges as a significant predictor (rho = 0.66, p = 0.026). This suggests topology carries information within an architecture family that scale alone does not explain.

### Where we stand honestly

| Finding | Status |
|---------|--------|
| H1 correlates with retention (rho = 0.61) | Nominally significant, does not survive Bonferroni |
| Parameter count dominates (rho = -0.74) | Survives Bonferroni (p_Bonf = 0.02) |
| H1 independent of scale? | Not yet. Partial rho = 0.35, p = 0.24 |
| Within-CNN H1 signal (rho = 0.66) | Significant, needs width-controlled validation |
| WRN width ladder | Pending (the decisive experiment) |
| CUB-200 replication | Pending |
| RESISC-45 cross-domain | Pending |

## Caveats

- **Multiple comparisons:** 10 metrics tested; Bonferroni-adjusted alpha = 0.005. Only parameter count survives correction.
- **Parameter count confound:** H1 and params are collinear (rho = -0.55, VIF = 1.45). Rank regression R-squared = 0.61 with only params significant.
- **Sample size:** n = 14 architectures (expanding to 19 with width ladder). Small-sample rank correlation should be interpreted with caution.
- **Multi-slice status:** 5 independent random 2D slices per architecture. Seed bug fixed; re-running with proper independent seeds.

## Experimental Design

### Pipeline

```
Phase 1:   Train on Task A               ->  converged checkpoint
Phase 2:   Sample 50x50 loss landscape    ->  persistent homology (H0, H1)
         + Baseline metrics               ->  Hessian, Fisher, sharpness, barrier
         + Landscape validation           ->  NaN/Inf/degeneracy checks
Phase 2b:  Displacement analysis          ->  trajectory through landscape after forgetting
Phase 3:   Train on Task B               ->  forgetting curve at 8 intervals
         + Task B learning validation     ->  warn if Task B fails to converge
Phase 4:   Spearman + Kendall correlation ->  Bonferroni, LOO, permutation, partials
         + WRN width ladder analysis      ->  within-ladder H1 vs retention
```

### Topology construction

1. **Landscape sampling:** Two filter-normalized random directions (Li et al., 2018). Loss evaluated on a 50x50 grid over [-1, 1]^2.
2. **Weighted graph:** 8-connected grid with lower-star edge weights: w(u,v) = max(f(u), f(v)).
3. **Persistent homology:** Sparse distance matrix -> Ripser -> H0 (components) and H1 (loops) persistence diagrams.
4. **Multi-slice:** 5 independent random 2D slices per architecture; Phase 4 aggregates (mean +/- std).

### Statistical methods

- **Spearman rank correlation** + **Kendall's tau** (more robust to ties at small n)
- **Bonferroni correction** for multiple testing (10 metrics, adjusted alpha = 0.005)
- **Partial correlation** controlling for parameter count
- **Symmetric partials** + rank regression + VIF to disentangle H1 vs params
- **Permutation test** (10,000 shuffles) for non-parametric significance
- **WRN width ladder analysis:** within-ladder Spearman + partial H1|params (the decisive test)

## Reproducibility

```bash
# Setup
python -m venv .venv && source .venv/bin/activate && pip install -e .
pip install torchgeo  # required for RESISC-45 download

# Run via dashboard (recommended)
.venv/bin/python dashboard/app.py    # http://localhost:5050

# Or run manually
python -m experiments.exp01_topological_persistence.phase1_train_task_a --config configs/exp01.yaml
python -m experiments.exp01_topological_persistence.phase2_landscape_topology --config configs/exp01.yaml
python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting --config configs/exp01.yaml

# Multi-slice (run-id 1-4, plus default = 5 slices)
python -m experiments.exp01_topological_persistence.phase2_landscape_topology \
    --config configs/exp01.yaml --run-id 1

# Cross-architecture correlation
python -m experiments.exp01_topological_persistence.phase4_correlation \
    --results-dirs results/exp01 results/exp01_resnet50 results/exp01_vit \
    results/exp01_wrn2810 results/exp01_wrn281 results/exp01_wrn282 \
    results/exp01_wrn284 results/exp01_wrn286 results/exp01_wrn288 \
    results/exp01_mlpmixer results/exp01_resnet18wide results/exp01_densenet121 \
    results/exp01_efficientnet results/exp01_vgg16bn results/exp01_convnext \
    results/exp01_mobilenetv3 results/exp01_vittiny results/exp01_shufflenet \
    results/exp01_regnet
```

### Config structure

57 configs total: 19 per dataset (14 architectures + 5 WRN width variants). Each YAML specifies dataset, class split, architecture, training hyperparameters, landscape grid, topology, and forgetting evaluation points.

## Project Structure

```
configs/                  # 57 YAML configs (19 per dataset × 3 datasets)
dashboard/                # Flask dashboard (localhost:5050)
  templates/              # Dashboard UI (3-dataset selector, system monitor)
experiments/
  shared/                 # Datasets (CIFAR-100, CUB-200, RESISC-45),
                          # Models (19 archs), baseline metrics, utils
  exp01_.../              # Phase 1-4 scripts + Phase 2b displacement
paper/                    # LaTeX draft + figures
results/                  # Output (gitignored)
data/                     # Datasets (gitignored, auto-downloaded)
```

### Dashboard

The Flask dashboard (port 5050) provides: 3-dataset selector (CIFAR-100, CUB-200, RESISC-45), experiment queue with auto-chaining, per-phase re-run, multi-slice progress tracking, "Run All Datasets" mode, Clean & Rebuild for invalidated runs, WRN width ladder correlation, and live GPU/CPU/RAM monitoring.

## References

- Li et al. (2018) -- Visualizing the Loss Landscape of Neural Nets
- Bauer (2021) -- Ripser: efficient computation of Vietoris-Rips persistence barcodes
- Kirkpatrick et al. (2017) -- Overcoming catastrophic forgetting in neural networks
- Keskar et al. (2017) -- On large-batch training for deep learning
- Wah et al. (2011) -- The Caltech-UCSD Birds-200-2011 Dataset
- Cheng et al. (2017) -- Remote sensing image scene classification (NWPU-RESISC45)

## License

MIT

## Citation

Paper in preparation. Check back soon.
