# EXP-04 Grokking — Next Steps (2026-04-24 snapshot)

**State:** Pilot verification flagged 2 new issues, both repaired in code. Local re-run queued but not yet executed. HPC move blocked pending re-run + post-run verification.

**Authoritative context:** `drafts/exp04_pilot_correctness_report_2026-04-24.md` (full verification report, before repairs).

---

## What's done

- **Code fix (B1):** `h0_significant_count` was structurally constant at n/2 = 1249. Replaced with `h0_effective_feature_count` (inverse participation ratio: `(Σℓ)²/Σℓ²`). Files: `experiments/exp04_grokking_topology/topology.py`, `configs/exp04_pilot.yaml`, `experiments/exp04_grokking_topology/run_pilot.py`.
- **Seed 42 excluded (B2):** 64K training collapse makes trajectory non-comparable. Not re-trained (selection bias). Data still in `results/exp04_pilot/seed_42/` for reference.
- **Seed 7777 quarantined:** empty `checkpoints/` but JSONs present with other-seeds' timestamps. Provenance unknown.
- **CLAUDE.md onsets corrected (B3):** old values (42=40K, 256=80.5K) replaced with config-rule values (42=68K, 137=42K, 256=44K, 1024=44K) in both `~/Corporate/CLAUDE.md` and `~/Corporate/axiondeep-research/CLAUDE.md`.
- **Pre-reg endpoints locked:** primary = `h0_total_persistence`, comparator = `commutator_defect`, secondary = `h0_effective_feature_count` / `h0_median_persistence` / `h0_persistence_entropy`. H1 exploratory only.
- **Old topology JSONs backed up** as `results/exp04_pilot/seed_{42,137,256,1024}/topology_metrics.pre-b1fix-2026-04-24.json`. Baseline JSONs untouched (the B1 fix doesn't affect baselines).
- **Verification script ready:** `scripts/verify_exp04_pilot.py`.

## What you're about to do (or may already have kicked off)

**Local re-run on RTX 4090, ~5 hours.** Topology-only; baselines resume-skip.

```bash
cd ~/Corporate/axiondeep-research
.venv/bin/python -m experiments.exp04_grokking_topology.run_pilot \
    --config configs/exp04_pilot.yaml --skip-training \
    2>&1 | tee results/exp04_pilot/rerun-2026-04-24.log
```

Run under `tmux` or `nohup` since it's 5h. Incremental saves after every checkpoint (line 153 of `run_pilot.py`) — kill/OOM costs at most 1 minute.

## What to do when re-run finishes

1. **Verify:**
   ```bash
   cd ~/Corporate/axiondeep-research
   .venv/bin/python scripts/verify_exp04_pilot.py
   ```
2. **Pass criteria:**
   - `B1 repair verified: True`
   - Seeds 137, 256, 1024 all `PASS`
   - `h0_significant_count` no longer present in new JSONs
   - `h0_effective_feature_count` has many unique values (not stuck at a constant)
3. **If pass:** draft the constrained HPC SLURM plan. Scope:
   - 30 seeds × 3–5 weight-decay values × 1 task (mod-add) × 1 model (302K-param transformer)
   - Pre-registered primary/comparator already locked above
   - WD values bracketed by calibration sweep (not assumed)
   - Grid resolution increase (to 100×100) tested on small subset BEFORE committing across all runs
4. **If fail:** repair locally again. No HPC scaling on failing pipelines.

## Open question to answer before HPC

- **H1 grid resolution.** Currently 50×50, where H1 is dead. Test 100×100 on 5–10 checkpoints from one seed to see if H1 becomes usable. If yes, promote to secondary. If no, keep H1 exploratory-only for the scaled study. Don't blanket-bump grid resolution without validating — compute cost scales quadratically.

## Deferred / NOT doing yet

- Full factorial (seeds × WD × tasks × models × grid-res). Joshua locked this as premature until pilot demonstrates topology adds signal beyond baselines.
- Task expansion (mod-mul, permutation groups). Defer until mod-add signal is real.
- Seed 7777 investigation. Defer until the 3 in-scope seeds clear verification.
- Re-registering seed 42 as a separate "instability dynamics" case. Possible future study.

## Decision rule (from Joshua, 2026-04-24)

> Does topology add anything beyond existing baselines? If no, we stop. If maybe, we expand.

Constrained HPC batch answers this. Full factorial does not — it dilutes the study before the core question is settled.
