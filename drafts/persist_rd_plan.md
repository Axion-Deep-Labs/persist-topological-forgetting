# PERSIST Phase I R&D Plan
## Topological Diagnostic for Catastrophic Forgetting in Production ML Systems

**Axion Deep Labs** | PI: Joshua Gutierrez
**Target Solicitation:** NSF SBIR Phase I

---

## Commercial Hypothesis

Enterprise ML teams retrain and fine-tune models continuously. When models learn new tasks, they catastrophically forget previous capabilities. The current response is heuristic: teams guess which mitigation strategy to apply, tune it by trial and error, and discover failures after deployment. No diagnostic exists that predicts, before committing compute, which strategy will work and how much it will help.

**The cost of the status quo:** A single mitigation attempt on a 300M-parameter model requires 4-8 GPUs for 8-24 hours of fine-tuning, plus evaluation and debugging. At current cloud H100 pricing ($3-10/GPU-hour, per published rates from Lambda Labs, RunPod, and major cloud providers), one failed attempt costs $96-1,920 in compute alone. Teams typically iterate through 3-5 mitigation strategies before finding one that works, spending $300-9,600 in GPU cost per search. The larger expense is engineering time: a senior ML engineer ($75-100/hour) spending 2-4 days evaluating each failed attempt adds $1,200-3,200 per cycle. A typical mitigation search costs $4,000-15,000 in combined compute and engineering time. For billion-parameter models, these costs scale by an order of magnitude.

**If topology predicts mitigation benefit at production scale (rho > 0.4 across 100M+ parameter models and multiple CL methods), then:** a single topology scan (estimated cost: $50-500 depending on model size) replaces the trial-and-error cycle, saving $4,000-15,000+ per model update. This enables a SaaS diagnostic or SDK integration with immediate ROI for any team managing sequential model updates.

**Target market segments:**
- LLM fine-tuning pipelines (every company customizing foundation models)
- Medical AI under regulatory update requirements (FDA mandates continuous learning documentation)
- Autonomous vehicle perception systems (sequential domain adaptation)
- Recommendation systems with evolving product catalogs

**Addressable market:** The MLOps tooling market exceeds $1B annually. Continual learning failures are a known cost center with no existing diagnostic solution.

---

## Primary Phase I Objective

**Determine whether persistent homology predicts continual learning mitigation benefit at production scale (models > 100M parameters) across multiple mitigation methods.**

**Central hypothesis:** Basin fragmentation, measured by the H0 persistent homology of loss landscapes, is a structural property that determines the effectiveness of regularization-based continual learning methods, and this relationship persists at production scale.

All Phase I activities test components of this hypothesis. If scale validation and method generalization both succeed, PERSIST has a commercial product. If either fails, we know precisely why and whether a pivot is viable.

---

## Competitive Landscape

**Current industry practice:** Mitigation method selection is heuristic or grid-search based. Teams choose EWC, replay, or architectural methods based on literature or trial and error. No tool predicts which method will work best for a given model before training begins.

**Existing tools:** CL frameworks (Avalanche, Continuum) implement methods but provide no diagnostic guidance. Model monitoring tools (WhyLabs, Arize) detect drift after deployment but do not predict forgetting before it occurs. Loss landscape visualization (Weights & Biases) shows surfaces but extracts no predictive features.

**PERSIST's differentiation:** First predictive diagnostic for continual learning. Topology extracts a structural feature (basin fragmentation) that correlates with mitigation benefit before training begins. Prediction vs. detection.

**Technical novelty:** No prior work has demonstrated a statistically predictive relationship between persistent homology of loss landscapes and mitigation-specific continual learning benefit. Existing loss landscape studies (Li et al., 2018; Keskar et al., 2017) are descriptive: they visualize landscape geometry but do not connect it to actionable outcomes. PERSIST moves from visualization to diagnostic, linking a computable topological invariant (H0) to a decision that practitioners actually face (which mitigation strategy to deploy).

---

## Preliminary Results (Complete)

57 configurations (19 architectures x 3 datasets), all phases complete on models 0.3M-44.7M parameters.

- **H0 predicts EWC benefit:** CIFAR-100 rho = 0.76 (p = 0.0002), RESISC-45 rho = 0.86 (p = 2.4e-6). Replicates across 2 of 3 datasets.
- **Pooled interaction model:** Dataset moderates the topology-benefit relationship (p = 0.046)
- **Conditional forgetting prediction:** On CUB-200, topology rescues prediction where parameter count fails (p = 0.037, does not survive Bonferroni)
- **Basin fragmentation hypothesis:** H0 measures landscape fragmentation; regularization benefit scales with fragmentation degree

**What this establishes:** The signal exists at small scale under controlled conditions. **What it does not establish:** Whether the signal, the computation, or the commercial value survive at production scale.

---

## Phase I Technical Plan

### Risk 1 (Primary): Scale Survival

**Question:** Does the topological signal persist on 100M+ parameter models, or is it a small-model artifact?

All current models are under 45M parameters. Wide networks converge to flatter minima at scale, which could eliminate the H0 variance driving the signal. This is the single question that determines commercial viability.

**Activities:**
1. **Scale ladder** (Months 1-4): Extend WRN width ladder to 100M+, add ViT-Base (86M), ViT-Large (307M). Minimum 5 models above 100M. Compute PH, measure EWC benefit, test H0-benefit correlation at each scale.
2. **Signal decay curve** (Months 2-5): Plot correlation strength vs. model scale. Identify gradual decay, phase transition, or persistence.

**Success criteria:** Go if rho > 0.4 (p < 0.05) at 100M+. Pivot if rho 0.2-0.4. No-go if rho < 0.2.

### Risk 2: Mitigation Method Generalization

**Question:** Does topology predict benefit across mechanistically different CL methods?

A commercial diagnostic must work for the method the customer uses. EWC-specific findings have a narrow market.

**Activities (ordered by cost and information value):**
1. **Synaptic Intelligence (SI) replication** (Months 1-3): Run SI on all 57 configs. Cheapest generalization test. SI tracks importance online vs. EWC's offline Fisher computation. Positive result generalizes to regularization-based methods broadly.
2. **PackNet test** (Months 3-5): Hard weight pruning, mechanistically unrelated to EWC. Positive result means basin fragmentation runs deeper than regularization.
3. **Experience replay test** (Months 5-7): Buffer-based, unrelated to landscape geometry. Expected null (supports mechanism if confirmed).

**Success criteria:** Go if H0 predicts SI + one non-regularization method. Narrow if SI only. No-go if SI null on all datasets.

**Key decision point (Month 3):** SI results determine whether to invest in scale experiments.

### Risk 3: Task Sequence Length

**Question:** Does topology remain predictive over 10-100+ task sequences?

Production CL involves long curricula. The landscape changes after each task; topology measured once may be irrelevant by task 10.

**Activities:**
1. **5-task and 10-task sequences** (Months 3-7): Split CIFAR-100 into sequential tasks. Compute PH after each boundary. Test prediction horizon.
2. **Dynamic topology tracking** (Months 6-8): Test delta-H0 between tasks as forgetting predictor.

**Success criteria:** Go if predictive through 5+ tasks. Pivot if 2-3 tasks (still marketable for fine-tuning). No-go if no prediction beyond next task.

### Risk 4: Computational Tractability

**Question:** Can PH extraction cost less than the retraining it prevents?

Ripser scales O(n^3) in simplex count. If extraction at scale exceeds training cost, the tool is impractical regardless of signal quality.

**Activities:**
1. **Scaling benchmark** (Months 1-3): Measure PH time across grid sizes (50x50 to 500x500) and model sizes (1M to 1B). Identify cost crossover.
2. **Subsampling fidelity** (Months 2-5): Test whether 5 slices remain sufficient at scale.
3. **Distributed PH pipeline** (Months 4-8, if needed): GPU-accelerated Ripser, landmark approximation, distributed cubical complexes.

**Success criteria:** Go if < 10% of training cost at 100M. Viable if 10-50%. No-go if exceeds training cost with no viable optimization path.

### Risk 5: Statistical Power and Replication

**Question:** Do borderline findings (p = 0.037, p = 0.046) survive rigorous replication?

**Activities:**
1. **Architecture expansion** (Months 1-6): Add 30+ architectures per dataset. Target 50+ per dataset, 150+ total.
2. **Pre-registered replication** (Months 6-8): Register on OSF before expanded analysis. Primary: H0 predicts benefit on 2+ datasets.

**Success criteria:** Go if p < 0.01 pre-registered. Marginal if p < 0.05. No-go if p > 0.05.

---

## Customer Discovery (Phase I)

**Activities:**
1. **20 structured interviews** (Months 1-6): Enterprise ML teams managing sequential model updates. Protocol: current mitigation workflow, pain points, cost of failed updates, willingness to adopt diagnostic tooling. Targets: LLM fine-tuning teams, medical AI companies, AV perception teams, recommendation system engineers.
2. **2 letters of interest** (Months 6-9): Contingent on Phase I validation. "If PERSIST demonstrates X at scale, we would evaluate integration."
3. **Willingness-to-pay threshold**: Hypothesis: $500-5,000 per model evaluation or $2,000-10,000/month SaaS.
4. **Integration requirements**: Formats (ONNX, PyTorch, TensorFlow), environments (cloud, on-prem, air-gapped), acceptable diagnostic latency.

**Deliverable:** Customer discovery report with interview summaries, market sizing validation, integration spec, and 2 letters of interest.

---

## Compute Infrastructure

### Primary: NMSU Discovery HPC Cluster (No Cost to Project)

Phase I compute is provided through an institutional partnership with New Mexico State University (NMSU). Senior Personnel Crystal Gutierrez holds dual affiliation as an adjunct professor and graduate student at NMSU, providing direct access to the Discovery high-performance computing cluster at no cost to the project.

**Discovery cluster resources:**

| Resource | Specification |
|----------|---------------|
| GPU nodes (primary) | 2x NVIDIA A100 80GB nodes (512GB system RAM each) |
| GPU nodes (secondary) | 5x NVIDIA V100 32GB nodes |
| Job scheduler | SLURM (batch scheduling with dependency chains) |
| Storage | Shared high-performance scratch + persistent home directories |
| Networking | InfiniBand interconnect between nodes |
| Cost to project | **$0** (institutional access via NMSU affiliation) |

**Why Discovery is sufficient for Phase I objectives:** The A100 80GB GPUs provide the memory capacity required for training and topology extraction on models up to 307M parameters (ViT-Large). For the 1B+ parameter experiments (Risk 1, Priority 7), A100 nodes support gradient checkpointing and mixed-precision training with gradient offloading to the 512GB system RAM. The V100 nodes handle the high-throughput smaller experiments (SI replication, architecture expansion) in parallel, maximizing cluster utilization.

**SLURM integration:** All experiment scripts are SLURM-ready with dependency chain management. Phase 1 (training) jobs automatically trigger Phase 2 (topology) and Phase 3 (forgetting) jobs upon completion, enabling hands-off execution of the full pipeline across all configurations. Job scripts, configs, and submission tools are already prepared and tested (`slurm/run_experiment.sh`, `slurm/submit_all.sh`).

**Scheduling and availability:** Discovery operates as a shared institutional resource with fair-share scheduling. To mitigate queue contention during peak academic periods, experiments are prioritized by information value (see Compute Prioritization below), and the pipeline is designed for fault-tolerant batch submission. SLURM checkpoint/restart capabilities handle preemption without data loss.

### Contingency: Cloud Compute Budget ($8,000)

A cloud compute contingency of $8,000 is budgeted for two scenarios:

1. **Queue bottlenecks at go/no-go deadlines:** If Discovery queue times exceed 48 hours during critical decision points (Month 3 SI gate, Month 5 scale gate), short-burst cloud GPU instances ensure milestones are met on schedule.
2. **Overflow for large-scale experiments:** If the 1B+ parameter experiments (Priority 7) require sustained multi-day GPU access that exceeds Discovery's per-job time limits, cloud instances provide uninterrupted training runs.

At published A100 cloud pricing ($1.50-3.00/GPU-hour, per Lambda Labs, RunPod, and Jarvislabs 2026 pricing), $8,000 provides 2,600-5,300 GPU-hours of contingency capacity. This budget is a safety margin, not the primary compute plan.

### Local Development: Pre-owned RTX 4090 Workstation (No Phase I Cost)

A pre-owned RTX 4090 workstation (not a Phase I expense) handles code development, debugging, small-model prototyping, and data preprocessing locally. All production experiments run on Discovery.

### Compute Prioritization

All experiments run on NMSU Discovery HPC at zero compute cost to the project. Cloud contingency reserved for scheduling bottlenecks at go/no-go gates. Prioritization reflects experimental logic:

| Priority | Experiment | Est. GPU-hours | Decision it enables |
|----------|-----------|----------------|---------------------|
| 1 | SI replication (57 configs) | ~200 | Method generalization go/no-go (Month 3) |
| 2 | Scale ladder to 100M (5 models) | ~500 | Signal at production scale |
| 3 | Architecture expansion (+30) | ~600 | Statistical power for replication |
| 4 | 5-task sequences (top 10 archs) | ~300 | Task sequence viability |
| 5 | PH scaling benchmark | ~100 | Tractability assessment |
| 6 | PackNet + replay (57 configs each) | ~400 | Cross-method generalization |
| 7 | ViT-Large (307M) + 1B model | ~1,500 | Deep scale validation |
| 8 | 10-task sequences + dynamic topology | ~500 | Long-sequence prediction |
| 9 | Pre-registered replication (150+ configs) | ~900 | Confirmatory analysis |
| **Total** | | **~5,000** | Full Phase I experiment set |

Priorities 1-5 (minimum viable set, ~1,700 hours) answer the core questions. Priorities 6-9 deepen and confirm. Discovery's multi-node GPU cluster supports parallel job execution, enabling simultaneous runs across V100 nodes (Priorities 1, 3-5) while A100 nodes handle scale-dependent experiments (Priorities 2, 7). The full 5,000-hour experiment set completes within 9 months given standard Discovery utilization patterns.

**If a go/no-go gate fails early (e.g., SI null at Month 3), remaining Phase I effort redirects** to investigating the failure mechanism (publishable, advancing the field) and documenting negative results for the community.

---

## Timeline and Milestones

| Month | Activity | Deliverable | Gate |
|-------|----------|-------------|------|
| 1-2 | SI replication + PH benchmark + interviews begin | SI data, cost model | |
| 3 | **SI go/no-go** | SI results report | **No-go if null on all datasets** |
| 2-4 | Scale ladder + architecture expansion | Signal at scale, expanded data | |
| 5 | **Scale go/no-go** | Signal decay curve | **No-go if rho < 0.2 at 100M** |
| 3-5 | PackNet + 5-task sequences | Method + sequence data | |
| 6 | Pre-register on OSF | Registration | |
| 6-8 | Pre-registered analysis + ViT-Large + interviews complete | Confirmatory results | |
| 8 | **Replication go/no-go** | Pre-registered analysis | **No-go if p > 0.05** |
| 7-9 | Dynamic topology + distributed PH + LOIs | Prototype pipeline | |
| 9 | Final analysis + Phase I report | All deliverables | Phase II decision |

---

## Budget Overview

| Category | Est. Cost | % of Phase I |
|----------|-----------|-------------|
| Personnel (PI full-time + Senior Personnel part-time) | $163,500 | 60% |
| Cloud compute contingency | $8,000 | 3% |
| NSF I-Corps (commercialization training) | $25,000 | 9% |
| TABA (Technical and Business Assistance) | $6,500 | 2% |
| Customer discovery (travel, interviews) | $20,000 | 7% |
| Materials, supplies, indirect | $52,000 | 19% |
| **Total** | **~$275,000** | 100% |

**Personnel ($163,500):** The PI (Joshua Gutierrez) is funded full-time for the 9-month performance period. Senior Personnel (Crystal Gutierrez) is funded at substantial part-time, reflecting her role in experiment execution on NMSU Discovery, SLURM pipeline management, and CL method implementation (SI, PackNet, replay). Crystal's NMSU affiliation provides both HPC access and proximity to the research computing support staff.

**Cloud compute contingency ($8,000):** Safety margin for queue bottlenecks at go/no-go deadlines and overflow for large-scale experiments. Primary compute runs on NMSU Discovery at zero cost (see Compute Infrastructure section). At published A100 cloud pricing ($1.50-3.00/GPU-hour), this provides 2,600-5,300 GPU-hours of contingency capacity.

**I-Corps ($25,000):** Dedicated commercialization training for the PI, including customer discovery methodology, market validation frameworks, and mentor engagement through the NSF I-Corps program. The PI commits to completing the NSF I-Corps customer discovery curriculum during Phase I. Directly supports the 20-interview customer discovery plan and go-to-market readiness.

**TABA ($6,500):** Third-party technical and business advisory services for IP strategy assessment, market sizing validation, and Phase II commercialization planning. Supports the open-core IP model and pricing strategy development.

**Customer discovery ($20,000):** Travel for 20 structured interviews with enterprise ML teams, conference attendance for industry networking, and LOI partner engagement.

**Materials, supplies, indirect ($52,000):** Software licenses, data storage, publication fees, indirect costs.

## Personnel and Team Capacity

| Role | Effort | Responsibility |
|------|--------|---------------|
| PI (Joshua Gutierrez) | Full-time | Experimental design, PH computation, statistical analysis, distributed algorithms |
| Senior Personnel (Crystal Gutierrez) | Substantial part-time | ML pipeline development, CL method implementation, HPC experiment execution, data analysis |

The PI has already executed all 57 configurations (19 architectures x 3 datasets, 7 phases each) end-to-end without external support, including custom dataset pipelines, PH extraction, statistical analysis, and the pooled interaction model. Senior Personnel independently built and evaluated ML models on industrial datasets at Bayer through the Purdue Data Mine and currently teaches data analysis methods as an adjunct professor at NMSU. Crystal's NMSU affiliation provides institutional access to the Discovery HPC cluster, enabling all Phase I GPU-accelerated experiments at zero compute cost. This team has demonstrated capacity to manage large experimental pipelines with minimal overhead.

---

## Intellectual Property Strategy

**Open core model:**
- **Open:** Research methodology, preliminary benchmark data, and core PH extraction code. Builds scientific credibility and drives adoption.
- **Proprietary:** Scalable distributed PH implementation (Phase I R&D output), production-scale benchmark dataset during commercialization window, and method-selection algorithms.
- **Patent potential:** Novel distributed PH extraction methods developed under Risk 4, if applicable. Provisional patent filings during Phase I.

This resolves the open-science vs. commercial-defensibility tension: the science is reproducible, but the production engineering is proprietary.

---

## Phase I Deliverables

| Deliverable | Description |
|-------------|-------------|
| **Open benchmark dataset** | Topology metrics, forgetting curves, and mitigation benefit scores across 150+ configurations |
| **Scalable PH prototype** | Python package for topology extraction, tested on 300M+ parameter models |
| **Pre-registered replication** | OSF-registered confirmatory analysis (or documented null) |
| **Peer-reviewed submission** | NeurIPS/ICML target reporting Phase I results |
| **ArXiv preprint** | Preliminary results (in progress, pending endorsement) |
| **Customer discovery report** | 20 interviews, integration spec, WTP data, market sizing |
| **2 letters of interest** | Enterprise ML teams, contingent on validation |
| **Phase II plan** | Product development and go-to-market (if Phase I succeeds) |

---

## Broader Impacts: Trustworthy AI

Catastrophic forgetting is a reliability and safety problem:

- **Defense systems** learning sequentially must not forget previous threats when adapting to new ones. A predictive diagnostic prevents silent capability degradation.
- **Medical AI** under FDA requirements must demonstrate that updates do not compromise existing accuracy. Topology provides a pre-deployment safety check.
- **Critical infrastructure AI** under continuous distribution shift needs guaranteed retention.

Phase II will explore regulatory documentation support in partnership with domain experts. No clinical or defense deployment is proposed during Phase I.

---

## Phase II Vision (Contingent on Phase I)

- **Product:** SaaS diagnostic and SDK. Input: pretrained model + task description. Output: recommended mitigation strategy, expected benefit, confidence score.
- **Integration:** Plugins for CL frameworks (Avalanche, Continuum), MLOps platforms (MLflow, W&B), cloud pipelines (SageMaker, Vertex AI).
- **Revenue:** Per-evaluation ($500-5,000) or monthly SaaS ($2,000-10,000/month).
- **Infrastructure continuity:** NMSU Discovery HPC access continues through Crystal's NMSU affiliation for Phase II R&D. Product deployment infrastructure (cloud-based SaaS) is budgeted separately in the Phase II proposal.
- **Activities:** Product engineering, beta with LOI partners, regulatory documentation framework, go-to-market.

---

## Falsification Commitment

| Decision point | Timeline | Consequence |
|----------------|----------|-------------|
| SI null on all datasets | Month 3 | Finding is EWC-specific. Commercialization path will not proceed to Phase II absent positive validation. |
| Signal vanishes at 100M | Month 5 | Small-model artifact. Commercialization path will not proceed to Phase II absent positive validation. |
| Pre-registered replication fails | Month 8 | Preliminary results not confirmed. Commercialization path will not proceed to Phase II absent positive validation. |
| PH intractable, no workaround | Month 9 | Pivot to approximate topology methods or commercialization path will not proceed to Phase II. |

Commercialization path will not proceed to Phase II absent positive validation. Phase I funds understanding why the signal fails (publishable, advancing the field) but product development halts without confirmed technical outcomes.
