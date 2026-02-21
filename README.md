# PERSIST: Topological Signatures of Knowledge Persistence

**Axion Deep Labs** | [axiondeep.com](https://www.axiondeep.com)

Can the topology of a loss landscape predict how well a model resists catastrophic forgetting?

We compute persistent homology on 2D cross-sections of loss landscapes across 19 architectures and 3 datasets, then test whether topological features (H0, H1) correlate with knowledge retention under sequential training. A WRN width ladder isolates topology from scale.

## Key Results (CIFAR-100, n=14)

| Metric | Spearman rho | p-value | Survives Bonferroni? |
|--------|-------------|---------|---------------------|
| H1 persistence vs retention | 0.61 | 0.021 | No |
| Parameter count vs retention | -0.74 | 0.002 | Yes |
| Partial H1 (controlling params) | 0.35 | 0.24 | -- |
| Within-CNN H1 (n=11) | 0.66 | 0.026 | -- |

H1 correlates with retention, but parameter count correlates more strongly. After partialing out model size, H1 drops to non-significance. Within CNNs only, H1 re-emerges as a predictor, suggesting topology carries signal within architecture families that scale alone does not explain.

The WRN width ladder (same architecture, same depth, varying only width) is in progress to resolve this.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install torchgeo gudhi
```

## Usage

**Dashboard (recommended):**
```bash
python dashboard/app.py    # http://localhost:5050
```

**Manual (single architecture):**
```bash
# Phase 1: Train on Task A
python -m experiments.exp01_topological_persistence.phase1_train_task_a --config configs/exp01.yaml

# Phase 2: Loss landscape topology (5 slices + cubical)
python -m experiments.exp01_topological_persistence.phase2_landscape_topology --config configs/exp01.yaml
python -m experiments.exp01_topological_persistence.phase2c_cubical_persistence --results-dir results/exp01

# Phase 3: Sequential forgetting (naive, EWC, cosine LR)
python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting --config configs/exp01.yaml
python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting --config configs/exp01.yaml --ewc
python -m experiments.exp01_topological_persistence.phase3_sequential_forgetting --config configs/exp01.yaml --lr-schedule cosine

# Phase 4: Cross-architecture correlation
python -m experiments.exp01_topological_persistence.phase4_correlation \
    --results-dirs results/exp01 results/exp01_resnet50 ...

# Phase 5: Predictive model (LOAO cross-validation)
python -m experiments.exp01_topological_persistence.phase5_predictive_model \
    --results-dirs results/exp01 results/exp01_resnet50 ...
```

## Datasets

| Dataset | Classes | Split | Domain |
|---------|---------|-------|--------|
| CIFAR-100 | 100 | 50/50 | Natural images |
| CUB-200-2011 | 200 | 100/100 | Fine-grained birds |
| NWPU-RESISC45 | 45 | 23/22 | Satellite scenes |

All resized to 32x32 for cross-architecture consistency.

## Architectures (19)

14 diverse architectures (ViT-Tiny, ViT-Small, MLP-Mixer, MobileNet-V3-S, ShuffleNet-V2, RegNet-Y-400MF, EfficientNet-B0, DenseNet-121, ResNet-18, ResNet-50, ResNet-18 Wide, VGG-16-BN, ConvNeXt-Tiny, WRN-28-10) plus a WRN-28-k width ladder (k=1, 2, 4, 6, 8, 10) ranging from 0.3M to 44.7M parameters.

## Project Structure

```
configs/          57 YAML configs (19 architectures x 3 datasets)
dashboard/        Flask dashboard with experiment queue and system monitor
experiments/
  shared/         Datasets, models, baseline metrics, EWC, utilities
  exp01_.../      Phase 1-5 scripts
results/          Output (gitignored)
data/             Datasets (gitignored, auto-downloaded)
```

## Methods

**Topology:** 50x50 loss landscape grid along filter-normalized random directions (Li et al., 2018). 5 independent 2D slices per architecture. Persistent homology via Ripser (graph-based) and GUDHI (cubical complexes).

**Forgetting:** Naive sequential training, EWC (Kirkpatrick et al., 2017), and cosine LR decay. Retention measured at 8 intervals over 10k steps.

**Statistics:** Spearman and Kendall correlation with Bonferroni correction, partial correlations controlling for parameter count, permutation tests (10k shuffles), WRN within-ladder analysis, slice robustness diagnostics, and leave-one-architecture-out Ridge regression.

## References

- Li et al. (2018). Visualizing the Loss Landscape of Neural Nets. *NeurIPS*.
- Bauer (2021). Ripser: efficient computation of Vietoris-Rips persistence barcodes. *JOSS*.
- Maria et al. (2014). The GUDHI Library. *INRIA*.
- Kirkpatrick et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.
- Wah et al. (2011). The Caltech-UCSD Birds-200-2011 Dataset.
- Cheng et al. (2017). Remote sensing image scene classification. *IEEE*.

## License

MIT
