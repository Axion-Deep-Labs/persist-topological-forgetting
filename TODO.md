# Axion Deep Labs -- TODO

## Urgent / Blocking

- [ ] **ArXiv endorsement:** Email Son Tran (tson@cs.nmsu.edu) for cs.LG endorsement. Code: M93UIA. Reference conversations with Dr. Huiping Cao. Deadline: wait for response or Thursday 2026-03-27, whichever comes first. If he can't endorse for cs.LG, check if cs.AI + cross-list works, or find endorser from cited papers.
- [x] **HPC datasets:** ImageNet-1K downloaded and extracted on Discovery (155GB, 2026-03-27).
- [x] **EXP-04 calibration sweep:** WD=0.03 is optimal (70K-step grokking delay). Config updated.
- [x] **EXP-04 pilot bugs:** Fixed H0 count (constant), commutator defect (zero), H1 (dead). See CLAUDE.md for details.

## This Week

- [ ] **EXP-04 re-analysis:** Rerun analysis on all 5 seeds with fixed topology.py + baselines.py. Command: `.venv/bin/python -m experiments.exp04_grokking_topology.run_pilot --config configs/exp04_pilot.yaml --skip-training`
- [ ] **EXP-04 seed 7777:** Needs full run (training + analysis).
- [ ] **EXP-04 pilot gate:** After re-analysis, check if any PH stat shows consistent directional behavior in >= 3/5 seeds before grokking onset. If yes, proceed to full study.
- [x] **PERSIST Phase I-A training:** 8/8 valid ImageNet-100 configs complete through Phases 1-3 (2026-04-01). ViT-H/14 and WRN-40-10 dropped.
- [ ] **PERSIST Phase I-A analysis:** Submit Phase 4 (correlation), Phase 5 (predictive model), Phase 6 (pooled interaction) on HPC. Phase 6 runs twice: 3-dataset (n=57) and 4-dataset (n=65).
- [ ] **Demo sites:** Host Pastaggio's, UMO BBQ, BobBea's demos on demo.axiondeepdigital.com. Reach out to owners.
- [ ] **Reddit:** Day 3 (Mar 26) -- r/SEO, 3 helpful comments.
- [ ] **Quora:** 1 answer per day on SEO/website/audit questions (no links for first 3-5 answers).

## Pending

- [ ] Vesper restructuring (merge folders, purge creds, keep only DeepSeek)
- [ ] Blog syndication to Dev.to, Hashnode, HackerNoon
- [ ] Directory submissions for Axion Deep Digital
- [ ] Backlink critical path: EXP-01 -> arXiv preprint -> Papers With Code / The Gradient / TDS
- [ ] "61 Websites" article -> blogPosts.ts on axiondeepdigital
- [ ] CoLLAs paper: Crystal needs OpenReview account (deadline: Apr 16)
