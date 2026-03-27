# Axion Deep Labs -- TODO

## Urgent / Blocking

- [ ] **ArXiv endorsement:** Email Son Tran (tson@cs.nmsu.edu) for cs.LG endorsement. Code: M93UIA. Reference conversations with Dr. Huiping Cao. Deadline: wait for response or Thursday 2026-03-27, whichever comes first. If he can't endorse for cs.LG, check if cs.AI + cross-list works, or find endorser from cited papers.
- [ ] **HPC datasets:** Waiting on Nicholas (NMSU Discovery support) re: ImageNet-1K (155GB) and ImageNet-21K (1.2TB) availability on cluster. If not available, download to /fs1/scratch/cag1145/.
- [x] **EXP-04 calibration sweep:** WD=0.03 is optimal (70K-step grokking delay). Config updated.
- [x] **EXP-04 pilot bugs:** Fixed H0 count (constant), commutator defect (zero), H1 (dead). See CLAUDE.md for details.

## This Week

- [ ] **EXP-04 re-analysis:** Rerun analysis on all 5 seeds with fixed topology.py + baselines.py. Command: `.venv/bin/python -m experiments.exp04_grokking_topology.run_pilot --config configs/exp04_pilot.yaml --skip-training`
- [ ] **EXP-04 seed 7777:** Needs full run (training + analysis).
- [ ] **EXP-04 pilot gate:** After re-analysis, check if any PH stat shows consistent directional behavior in >= 3/5 seeds before grokking onset. If yes, proceed to full study.
- [ ] **PERSIST Phase I:** Once datasets are on cluster, clone repo + submit first batch via submit_all.sh.
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
