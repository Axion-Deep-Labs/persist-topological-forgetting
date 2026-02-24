# PERSIST: Topological Signatures of Knowledge Persistence

**Axion Deep Labs** | [axiondeep.com](https://www.axiondeep.com)

Can the topology of a loss landscape predict how well a model resists catastrophic forgetting?

We compute persistent homology on 2D cross-sections of loss landscapes across 19 architectures and 3 datasets (57 total configurations), then test whether topological features predict knowledge retention under sequential training. A WRN width ladder isolates topology from scale.

## Key Findings

**1. Topology rescues forgetting prediction on fine-grained tasks.**
On CUB-200 (200 bird species), parameter count alone predicts retention in the *wrong direction* (rho = -0.92). Adding topological features rescues the prediction (permutation p = 0.037, MAE reduction 17.5%). This finding does not survive Bonferroni correction across 3 datasets (adjusted alpha = 0.0167), so we report it as suggestive.

**2. Topology is not a universal forgetting predictor.**
On RESISC-45 (satellite scenes, also hard), topology adds no predictive value (p = 0.566). The CUB-200 signal appears specific to fine-grained discrimination, not hard tasks in general.

**3. Topology predicts mitigation benefit.**
The most stable cross-dataset signal: H0 (connected components) predicts how much EWC regularization helps, replicating across CIFAR-100 (rho = 0.76, p = 0.0002) and RESISC-45 (rho = 0.86, p = 2.4e-6). Loss landscape connectivity is a mitigation sensitivity marker.

## Results

### Cross-Dataset Predictive Model (Phase 5, LOAO Ridge Regression)

| Dataset | Outcome | Params-only rho | +Topology rho | Perm. p | Verdict |
|---------|---------|-----------------|---------------|---------|---------|
| CIFAR-100 (n=19) | ret@100 | 0.43 | 0.30 | 0.295 | Not significant |
| **CUB-200 (n=19)** | **ret@10** | **-0.92** | **0.34** | **0.037** | **Suggestive** |
| RESISC-45 (n=19) | ret@100 | -0.32 | -0.33 | 0.566 | Not significant |

CUB-200 p=0.037 does not survive Bonferroni across 3 datasets (adjusted alpha = 0.0167).

### EWC Benefit: Cross-Dataset Replication

| Dataset | H0 vs EWC benefit rho | p-value |
|---------|----------------------|---------|
| CIFAR-100 | 0.76 | 0.0002 |
| RESISC-45 | 0.86 | 2.4e-6 |
| CUB-200 | 0.31 | 0.19 |

H0 (connected components) predicts how much EWC regularization helps on 2 of 3 datasets. Architectures with more connected components in their loss landscape benefit more from regularization-based mitigation.

### CUB-200 ret@10 Detail

- Params alone: rho = -0.92 (wrong direction)
- Params + topology: rho = 0.34 (rescued)
- Topology alone: rho = 0.33, MAE = 0.147 (outperforms params-only)
- Permutation test: p = 0.037 (1,000 shuffles)
- Matched-dimensionality control: exceeds 95th percentile of random features
- MAE reduction: 17.5% (0.186 to 0.154)

## Status

- **CIFAR-100:** 19/19 architectures, Phases 1-5 complete
- **CUB-200-2011:** 19/19 architectures, Phases 1-5 complete
- **RESISC-45:** 19/19 architectures, Phases 1-5 complete
- **57 of 57 total configurations complete**

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

**Topology:** 50x50 loss landscape grid along filter-normalized random directions (Li et al., 2018). 5 independent 2D slices per architecture. Persistent homology via Ripser (graph-based) and GUDHI (cubical complexes). Cross-method H1 agreement: rho = 1.0.

**Forgetting:** Naive sequential training, EWC (Kirkpatrick et al., 2017), and cosine LR decay. Retention measured at 8 intervals (steps 10 through 5,000).

**Statistics:** Spearman and Kendall correlation with Bonferroni correction (12 tests), partial correlations controlling for parameter count, WRN within-ladder analysis, slice robustness diagnostics. Leave-one-architecture-out Ridge regression with nested alpha selection, permutation tests (1,000 shuffles), and matched-dimensionality null controls.

## References

- Li et al. (2018). Visualizing the Loss Landscape of Neural Nets. *NeurIPS*.
- Bauer (2021). Ripser: efficient computation of Vietoris-Rips persistence barcodes. *JOSS*.
- Maria et al. (2014). The GUDHI Library. *INRIA*.
- Kirkpatrick et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.
- Adams et al. (2017). Persistence Images. *JMLR*.
- Keskar et al. (2017). On Large-Batch Training for Deep Learning. *ICLR*.
- Wah et al. (2011). The Caltech-UCSD Birds-200-2011 Dataset.
- Cheng et al. (2017). Remote sensing image scene classification. *IEEE*.

## License

MIT
