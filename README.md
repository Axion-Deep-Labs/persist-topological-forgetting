# PERSIST: Topological Signatures of Knowledge Persistence

**Axion Deep Labs** | [axiondeep.com](https://www.axiondeep.com)

Does the topological structure of a neural network's loss landscape correlate with resistance to catastrophic forgetting?

We compute persistent homology on 2D cross-sections of loss landscapes across 14 architectures trained on Split-CIFAR-100 (and Split-CIFAR-10), then measure how topological features relate to knowledge retention under naive sequential training.

## Key Findings (n=14, CIFAR-100)

| Metric | ρ (ret@100) | p | p_Bonf | τ (Kendall) |
|--------|------------|---|--------|-------------|
| **H1 persistence** (loops) | **0.61** | 0.021 | 0.210 | 0.47 |
| H0 persistence (components) | 0.32 | 0.263 | 1.000 | 0.24 |
| Fisher trace | −0.50 | 0.072 | 0.720 | −0.38 |
| **Parameter count** | **−0.74** | 0.002 | **0.020** | −0.60 |

**Statistical rigor:** 10 metrics tested; Bonferroni-adjusted α = 0.005. Only parameter count survives multiple-comparison correction. H1 is nominally significant (p = 0.021) but does not survive Bonferroni (p_Bonf = 0.21). Kendall's tau, which is more robust to ties at small n, confirms the ranking.

**Important caveat:** Parameter count is the strongest single correlate of retention. After partialing out model size, H1 drops to non-significance (partial ρ = 0.35, p = 0.24, VIF = 1.45). Multi-slice stability analysis and CIFAR-10 replication are in progress to determine whether topology carries independent signal beyond capacity.

All correlations confirmed via leave-one-out cross-validation (14/14 folds significant) and permutation testing (10,000 shuffles).

## Architectures

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

*ConvNeXt shows non-monotonic recovery (0% → 3% → 0%).

## Experimental Pipeline

```
Phase 1:   Train on Task A (classes 0-49)  →  converged checkpoint
Phase 2:   Sample 50×50 loss landscape     →  persistent homology (H0, H1)
         + Baseline metrics                →  Hessian, Fisher, sharpness, barrier
         + Landscape validation            →  NaN/Inf/degeneracy checks
Phase 2b:  Displacement analysis           →  trajectory through landscape after forgetting
Phase 3:   Train on Task B (classes 50-99) →  forgetting curve at 8 intervals
         + Task B learning validation      →  warn if Task B fails to converge
Phase 4:   Spearman + Kendall correlation  →  Bonferroni, LOO, permutation, partials
```

### Multi-slice & Multi-seed

Each architecture gets 5 random 2D landscape slices for stability analysis:

```bash
# Multiple random 2D cross-sections per architecture (run-id 1-4, plus default)
python -m experiments.exp01_topological_persistence.phase2_landscape_topology \
    --config configs/exp01.yaml --run-id 1

# Multiple training seeds
python -m experiments.exp01_topological_persistence.phase1_train_task_a \
    --config configs/exp01.yaml --seed 123
```

Phase 4 automatically aggregates across slices (mean ± std) when multiple `topology_summary_run*.json` files exist.

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run via dashboard (recommended)
.venv/bin/python dashboard/app.py
# Open http://localhost:5050

# Or run manually
python -m experiments.exp01_topological_persistence.phase1_train_task_a --config configs/exp01.yaml
python -m experiments.exp01_topological_persistence.phase2_landscape_topology --config configs/exp01.yaml
python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting --config configs/exp01.yaml

# Cross-architecture correlation (after multiple architectures complete)
python -m experiments.exp01_topological_persistence.phase4_correlation \
    --results-dirs results/exp01 results/exp01_resnet50 results/exp01_vit \
    results/exp01_wrn2810 results/exp01_mlpmixer results/exp01_resnet18wide \
    results/exp01_densenet121 results/exp01_efficientnet results/exp01_vgg16bn \
    results/exp01_convnext results/exp01_mobilenetv3 results/exp01_vittiny \
    results/exp01_shufflenet results/exp01_regnet
```

## Project Structure

```
├── configs/                  # YAML configs (one per architecture × dataset)
├── dashboard/                # Flask dashboard (localhost:5050)
├── experiments/
│   ├── shared/               # Datasets, models (14 archs), baseline metrics, utils
│   └── exp01_.../            # Phase 1-4 scripts + Phase 2b displacement analysis
├── paper/                    # LaTeX draft + figures
├── scripts/                  # Batch runners
├── results/                  # Output (gitignored)
└── data/                     # Datasets (gitignored, auto-downloaded)
```

## Topology Construction

1. **Landscape sampling:** Generate two filter-normalized random directions (Li et al., 2018). Evaluate loss on a 50×50 grid over [-1, 1]². Validated for NaN/Inf/degeneracy.
2. **Weighted graph:** 8-connected grid graph with lower-star edge weights: w(u,v) = max(f(u), f(v)).
3. **Persistent homology:** Sparse distance matrix → Ripser (Vietoris-Rips complex) → H0 and H1 persistence diagrams.
4. **Multi-slice stability:** 5 independent random slices per architecture; Phase 4 aggregates (mean ± std).

## Statistical Analysis

- **Spearman rank correlation** + **Kendall's tau** (robust to ties at small n)
- **Bonferroni correction** for multiple testing (10 metrics, adjusted α = 0.005)
- **Partial correlation** controlling for parameter count (confound analysis)
- **Symmetric partials** + rank regression to disentangle H1 vs params
- **Permutation test** (10,000 shuffles) for non-parametric significance
- **Leave-one-out** cross-validation (min/mean/max p-values across folds)

## Hardware

- GPU: NVIDIA GeForce RTX 4090 (24 GB VRAM)
- CPU: AMD Ryzen 9 9900X
- RAM: 64 GB DDR5
- ~20 min per architecture for full pipeline (Phase 1-3)
- ~20 hours for all 14 architectures × 2 datasets × 5 slices

## References

- Li et al. (2018) — Visualizing the Loss Landscape of Neural Nets
- Bauer (2021) — Ripser: efficient computation of Vietoris-Rips persistence barcodes
- Kirkpatrick et al. (2017) — Overcoming catastrophic forgetting in neural networks
- Keskar et al. (2017) — On large-batch training for deep learning
- Whitley et al. — Fitness landscape structure, local optima networks

## License

MIT

## Citation

Paper in preparation. Check back soon.
