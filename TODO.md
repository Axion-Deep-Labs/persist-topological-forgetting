# Axion Deep Labs Research -- TODO

## Active

### PERSIST (EXP-01)

- [ ] **Phase I-B restricted-softmax re-eval (in progress):** SLURM job `527670` submitted 2026-04-21. Produces `results/phase3b_restricted_summary.json` + per-run `forgetting_curve_restricted.json`. When complete:
  - Verify step-0 sanity gate (per-config restricted ret_A(0) matches `initial_task_a_acc` to within 0.5%)
  - Compute full-vs-restricted rank correlation, mean |Δ|, per-pair breakdown
  - Apply three-tier decision rule from `drafts/phase_1b_g1_metric_audit_memo.md`
  - If primary metric chosen, unfreeze Phase 4 analysis
- [ ] **Phase I-B SI cross-dataset submission (orthogonal to G1):** 114 additional jobs. Phase I-A showed SI null; Phase I-B cross-dataset hasn't been run for SI.
- [ ] **Phase I-B preregistration framing for writeup:** exploratory vs confirmatory language; per `scientific_design_standards.md` memory.
- [ ] **ArXiv endorsement:** Email Son Tran (tson@cs.nmsu.edu) for cs.LG endorsement. Code: M93UIA. Reference conversations with Dr. Huiping Cao. If cs.LG endorsement not feasible, try cs.AI + cross-list, or find endorser from cited papers.
- [x] **Phase I-A training:** 8/8 valid ImageNet-100 configs complete through Phases 1-3 (2026-04-01). ViT-H/14 and WRN-40-10 dropped.
- [x] **Phase I-A analysis:** Phase 4-6 complete on HPC (2026-04-02). H1 dominant at scale, 3-dataset replicates, SI null.

### EXP-04 (Grokking Topology)

- [ ] **Re-analysis:** Rerun analysis on all 5 seeds with fixed topology.py + baselines.py. Command: `.venv/bin/python -m experiments.exp04_grokking_topology.run_pilot --config configs/exp04_pilot.yaml --skip-training`
- [ ] **Seed 7777:** Needs full run (training + analysis).
- [ ] **Pilot gate evaluation:** After re-analysis, check if any PH stat shows consistent directional behavior in ≥ 3/5 seeds before grokking onset. If yes, proceed to full study (30 seeds × 3 WD = 90 runs).
- [x] **Calibration sweep:** WD=0.03 is optimal (70K-step grokking delay). Config updated.
- [x] **Pilot bugs:** Fixed H0 count (constant), commutator defect (zero), H1 (dead). See CLAUDE.md.

### Publications & External

- [ ] **CoLLAs 2026 submission route:** Awaiting program chairs' response to late-registration email. When response arrives, choose among main track / Work-In-Progress (open until Jun 30) / arXiv-only. See memory `collas_2026_submission_state.md`.
- [x] **NSF SBIR kickoff (AlienTT, 2026-04-21):** Phase 1 deliverable reframed from "diagnostic validation" to "diagnostic + mitigation recommendation before retraining." Medical imaging locked as primary vertical (FDA PCCP hook). Market memo v1 filed at `~/Corporate/AxionDeep/Grants/NSF-SBIR/persist/market_memo_2026-04-21.md`. Corporate repo tag `nsf-sbir-framing-2026-04-21`.
- [ ] **NSF SBIR proposal artifacts revision:** PERSIST_Pitch_Deck.md + R&D plan + Phase 1 milestones need rewrite around mitigation discovery + medical-imaging validation. Market memo is the source doc. Old ("scale validation") framing must not go out externally unchanged.
- [ ] **NSF SBIR customer discovery:** 2-3 conversations with medical-imaging AI vendors (Aidoc / Rad AI / Paige / RapidAI / Viz.ai / HeartFlow). Script should probe PCCP workflow pain points, non-regression methodology, pricing anchors for regulatory-alignment tooling.

## Pending / Future

- [ ] 50-100+ architecture expansion for Phase I-B replication
- [ ] Longer task sequences (10-100+ tasks) for Phase I-B replication
- [ ] Additional CL methods beyond EWC/SI (e.g., LwF, MAS, PackNet)
- [ ] HPC dataset staging verified: ImageNet-1K extracted on Discovery (155GB, 2026-03-27)
