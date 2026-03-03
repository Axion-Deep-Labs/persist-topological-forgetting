# PERSIST: Topological Signatures of Knowledge Persistence

**Axion Deep Labs** | [axiondeep.com](https://www.axiondeep.com)

Can the topology of a loss landscape predict how well a model resists catastrophic forgetting?

## Project Status: Preliminary Proof-of-Concept Complete — Phase I Requires Supercomputer

**Preliminary work** (complete) established proof-of-concept on small-to-medium models (0.3M-44.7M parameters) across 3 small-image datasets. This is the petri dish: we computed persistent homology on 2D cross-sections of loss landscapes across 19 architectures and 57 total configurations, demonstrating that topological features correlate with forgetting dynamics and mitigation benefit at this scale.

**Phase I** (planned, requires HPC/supercomputer) will test whether these signals survive at production scale: 100M-7B+ parameter models, large-scale datasets (ImageNet, NLP), long task sequences (10-100+ tasks), and multiple continual learning methods. This is where the real research begins — the preliminary results may not generalize, and computing persistent homology on large weight spaces introduces fundamental computational barriers (superpolynomial complexity) that require novel distributed algorithms.

## Preliminary Findings (Small-Scale Proof-of-Concept)

**1. Topology rescues forgetting prediction on fine-grained tasks.**
On CUB-200 (200 bird species), parameter count alone predicts retention in the *wrong direction* (rho = -0.92). Adding topological features rescues the prediction (permutation p = 0.037, MAE reduction 17.5%). This finding does not survive Bonferroni correction across 3 datasets (adjusted alpha = 0.0167), so we report it as suggestive.

**2. Topology is not a universal forgetting predictor.**
On RESISC-45 (satellite scenes, also hard), topology adds no predictive value (p = 0.566). The CUB-200 signal appears specific to fine-grained discrimination, not hard tasks in general.

**3. Topology predicts mitigation benefit.**
The most stable cross-dataset signal: H0 (connected components) predicts how much EWC regularization helps, replicating across CIFAR-100 (rho = 0.76, p = 0.0002) and RESISC-45 (rho = 0.86, p = 2.4e-6). Loss landscape connectivity is a mitigation sensitivity marker.

## Preliminary Results

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

### Phase 6: Pooled Interaction Analysis (n=57)

Formal test of dataset moderation via OLS with interaction terms, clustered bootstrap (5,000 iterations, 19 architecture blocks), and permutation tests (1,000 iterations). CIFAR-100 as reference, H0 z-scored within dataset.

| Claim | Outcome | dR2 | Permutation p |
|-------|---------|-----|---------------|
| **EWC benefit moderation** | EWC benefit (AURC) | 0.085 | **0.046** |
| Forgetting moderation | ret@10 (primary) | 0.075 | 0.196 |
| Forgetting moderation | ret@100 (robustness) | 0.127 | **0.035** |

Per-dataset partial effects (95% clustered bootstrap CIs):

| Dataset | H0 effect on ret@10 | CI | H0 effect on EWC benefit | CI |
|---------|--------------------|----|-------------------------|----|
| CIFAR-100 | -0.001 | [-0.49, +0.07] | **+0.016** | **[+0.005, +0.062]** |
| CUB-200 | **-0.123** | **[-0.18, -0.05]** | +0.002 | [-0.008, +0.013] |
| RESISC-45 | -0.021 | [-0.26, +0.08] | **+0.007** | **[+0.004, +0.012]** |

**Bottom line:** Dataset significantly moderates the topology-EWC benefit relationship (p=0.046). H0 predicts mitigation benefit on CIFAR-100 and RESISC-45 (CIs exclude zero) but not CUB-200. For forgetting, H0's effect is concentrated on CUB-200 (CI excludes zero).

## Preliminary Status (Complete)

- **CIFAR-100:** 19/19 architectures, Phases 1-6 complete
- **CUB-200-2011:** 19/19 architectures, Phases 1-6 complete
- **RESISC-45:** 19/19 architectures, Phases 1-6 complete
- **57 of 57 total configurations complete**

## Phase I Roadmap (Planned — Requires Supercomputer)

The preliminary work demonstrated topological signal on small models. Phase I addresses the fundamental open questions that require HPC resources:

| Challenge | Preliminary (Done) | Phase I Target | Why It Matters |
|-----------|-------------------|----------------|----------------|
| Model scale | 0.3M-44.7M params | 100M-7B+ params | Signal may vanish at production scale |
| PH computation | 5 random 2D slices | Dense sampling, distributed PH | Subsampling may lose fidelity in high dimensions |
| Higher homology | H0, H1 only | H0, H1, H2, H3 | Higher-dim features exponentially expensive |
| Task sequences | 2-task (A then B) | 10-100+ sequential tasks | Real CL involves long curricula |
| CL methods | EWC only | SI, PackNet, replay, adapters | Current finding may be EWC-specific |
| Datasets | 3 small-image (32x32) | ImageNet, NLP, medical | Domain generalization unknown |
| Architectures | 19 (n too small for Bonferroni) | 50-100+ | Statistical power for robust claims |
| Foundation models | None | LLM/ViT-L fine-tuning | Commercial killer app |

**Compute requirements:** Phase I requires HPC/supercomputer allocation (NSF ACCESS or equivalent). A single topology extraction on a 7B-parameter model requires thousands of GPU-hours. Scaling Ripser beyond ~50M sampled points is an open computational problem (O(n^3) in simplex count). Novel distributed PH algorithms may be required as a Phase I research contribution.

**Genuine failure modes:**
- The topological signal may vanish at scale (small-model artifact)
- PH subsampling may lose fidelity in high-dimensional parameter spaces
- Long task sequences may show chaotic topology evolution defeating prediction
- Computational cost of PH may scale worse than training itself, making the tool impractical

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

**Statistics:** Spearman and Kendall correlation with Bonferroni correction (12 tests), partial correlations controlling for parameter count, WRN within-ladder analysis, slice robustness diagnostics. Leave-one-architecture-out Ridge regression with nested alpha selection, permutation tests (1,000 shuffles), and matched-dimensionality null controls. Pooled interaction model (Phase 6) with OLS, clustered bootstrap (5,000 iterations, 19 architecture blocks), and within-dataset permutation tests.

## Proposed Mechanism

We propose the **basin fragmentation hypothesis**: H0 counts connected components in the loss landscape's sublevel set filtration. A high H0 indicates many disconnected basins. EWC penalizes parameter drift using Fisher-weighted curvature, which is most effective when naive training would otherwise push parameters across basin boundaries. On fragmented landscapes (high H0), EWC prevents inter-basin drift; on smooth landscapes (low H0), there is only one broad basin and EWC provides little additional benefit.

This is consistent with H0 predicting EWC benefit on CIFAR-100 and RESISC-45 (where EWC produces measurable variance) but not CUB-200 (where fine-grained discrimination may create forgetting through feature-level interference rather than parameter-level basin drift). The WRN width ladder supports this: H0 decreases perfectly with width (rho = -1.0) across all three datasets, consistent with wider networks having smoother, less fragmented landscapes.

This mechanism is tentative and requires causal testing (e.g., landscape-aware regularization intervention).

## Limitations (Preliminary Work)

- **Small-scale models only:** All architectures are under 45M parameters. Production models are 100M-7B+. Whether topology is informative at that scale is genuinely unknown.
- **19 architectures:** Moderate sample size. The WRN width ladder controls for architecture family but has limited degrees of freedom. 50-100+ architectures needed for robust claims.
- **One mitigation method:** Only EWC tested. If the H0-benefit signal does not generalize to Synaptic Intelligence or PackNet, the finding is EWC-specific.
- **2-task sequences only:** Real continual learning involves 10-100+ tasks. Topology-forgetting dynamics over long sequences are entirely unexplored.
- **2D projections:** Topology computed on 2D landscape cross-sections, not the full high-dimensional landscape. 5 slices mitigate but do not eliminate sampling variance. At production scale, slice fidelity is an open question.
- **Small-image datasets only:** All images resized to 32x32. Whether the signal holds on ImageNet, NLP tasks, or medical imaging is unknown.
- **Borderline p-values:** EWC moderation p = 0.046, forgetting ret@100 p = 0.035. CUB-200 ret@10 p = 0.037 does not survive Bonferroni.
- **EWC benefit finding is exploratory:** The shift from "topology predicts forgetting" to "topology predicts mitigation benefit" emerged from the data. Phase 6 should be interpreted as discovery, not confirmation.
- **PH computational scaling unknown:** Ripser complexity is O(n^3) in simplex count. Whether PH extraction remains tractable on 100M+ parameter landscapes, or whether novel algorithms are required, is an open computational research question.

## Analysis Path Transparency

The original hypothesis targeted topology as a direct predictor of forgetting. CIFAR-100 was run first (params dominate, topology null). CUB-200 was run second (topology rescues prediction, p = 0.037). RESISC-45 was run third and returned a null for topology (p = 0.566), falsifying the simpler "hard tasks" framing. The EWC benefit analysis was computed as part of Phase 4 diagnostics, not the original target. The Phase 6 pooled interaction model was designed post hoc to formalize cross-dataset moderation. We report this path transparently: the EWC moderation finding requires pre-registered replication.

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

## Related Projects

- [SDI](https://github.com/Axion-Deep-Labs/structural-divergence-index) — Structural Divergence Index for model governance
- [DRIFT](https://github.com/Axion-Deep-Labs/drift-quantum-degradation) — Quantum circuit stability under iteration
- [PHI](https://github.com/Axion-Deep-Labs/phi-integrated-information) — Integrated information across architectures
- [GENESIS](https://github.com/Axion-Deep-Labs/genesis-capacity-scaling) — Information capacity scaling laws
