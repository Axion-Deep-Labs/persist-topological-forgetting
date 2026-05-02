# EXP-04 Full Study — HPC Handoff (2026-05-02)

**Goal:** Run constrained 30-seed × 3-WD scaling on NMSU Discovery while PERSIST is paused (Cao C&P pending, AlienTT Tuesday). Settle the topology-vs-weight-norm question.

**Decisive kill-switch result on pilot data:** topology is NOT obviously redundant. Full study justified. Run details below.

---

## Step 1 — Local: commit + push (on the laptop)

```bash
cd ~/Corporate/axiondeep-research

# Stage what's ready for HPC
git add scripts/verify_exp04_pilot.py \
        scripts/exp04_weight_norm_decisive.py \
        experiments/exp04_grokking_topology/run_pilot.py \
        configs/exp04_pilot.yaml \
        configs/exp04_full.yaml \
        slurm/run_exp04_full.sh \
        slurm/submit_exp04_full.sh \
        drafts/exp04_full_study_handoff_2026-05-02.md \
        drafts/exp04_next_steps_2026-04-24.md \
        drafts/exp04_pilot_correctness_report_2026-04-24.md

git commit -m "EXP-04 full study: 30-seed × 3-WD scaling, decisive kill-switch, verify-script bug fix"
git push origin master
```

> ⚠️  Stage **only** the EXP-04 files above. Skip `topology.py` and `slurm/run_phase3b_restricted.sh` if those are unrelated WIP.

---

## Step 2 — SSH into Discovery

```bash
# 1. Cisco Secure Client → vpn.nmsu.edu (SAML/SSO + Duo)
# 2. SSH
ssh cag1145@discovery.nmsu.edu
```

---

## Step 3 — HPC: sync the repo

```bash
cd /fs1/scratch/cag1145/axiondeep-research
git pull origin master

# Confirm new files arrived
ls -la slurm/run_exp04_full.sh slurm/submit_exp04_full.sh configs/exp04_full.yaml scripts/verify_exp04_pilot.py
```

---

## Step 4 — HPC: re-run seed 1024 (closes verification gate)

```bash
mkdir -p slurm/logs
sbatch slurm/run_exp04.sh 1024 --skip-training
squeue -u cag1145
```

Expected wall-clock: ~2 hours. Topology-only (training already done; baselines have 81 records, only the 47 missing topology checkpoints will be computed thanks to incremental-save resume logic).

---

## Step 5 — HPC: launch full study (90 jobs, batched)

```bash
# Background it so the loop continues even if you disconnect
nohup bash slurm/submit_exp04_full.sh > slurm/logs/submit_exp04_full.nohup 2>&1 &
echo "Submitter PID: $!"

# Or, foreground in a screen/tmux session
# tmux new -s exp04_submit
# bash slurm/submit_exp04_full.sh
# (Ctrl-B D to detach, `tmux attach -t exp04_submit` to return)
```

The submitter:
- Reads 30 seeds × 3 WD = 90 jobs from `configs/exp04_full.yaml`
- Submits up to 18 at a time (NMSU QOS cap is ~20; leaves 2 slots free)
- Sleeps 3 min between queue-full checks
- Skips any (seed, wd) whose topology JSON already has 81 records (resume-friendly)
- Logs every submission to `slurm/logs/submit_exp04_full.log`

Override knobs: `MAX_QUEUED=15 SLEEP_SECS=300 bash slurm/submit_exp04_full.sh`

---

## Step 6 — Monitor

```bash
# Live queue
watch -n 10 'squeue -u cag1145'

# Submitter log
tail -f slurm/logs/submit_exp04_full.log

# Latest job stdout
ls -t slurm/logs/*.out | head -3 | xargs tail -n 30

# Completed-job summaries
sacct -u cag1145 --format=JobID,JobName,Elapsed,State,ExitCode -S "$(date -d 'today' +%Y-%m-%d)" | head -40
```

---

## Step 7 — When seed 1024 finishes (local, ~2h after Step 4)

```bash
# On the laptop, sync results back
rsync -av cag1145@discovery.nmsu.edu:/fs1/scratch/cag1145/axiondeep-research/results/exp04_pilot/seed_1024/ \
          ~/Corporate/axiondeep-research/results/exp04_pilot/seed_1024/

# Re-verify
cd ~/Corporate/axiondeep-research
.venv/bin/python scripts/verify_exp04_pilot.py
# Expect all 3 in-scope seeds = PASS
```

---

## Step 8 — When full study has 30+ jobs done (later this weekend or Monday)

```bash
# On HPC, sync results
rsync -av cag1145@discovery.nmsu.edu:/fs1/scratch/cag1145/axiondeep-research/results/exp04_full/ \
          ~/Corporate/axiondeep-research/results/exp04_full/

# Re-run decisive script with full data — it will need a small extension
# to read from results/exp04_full/wd_*/seed_*/ — TODO when data arrives.
```

---

## Compute budget

- **Per job:** ~3-5 hours wall-clock (full pipeline: train 100K steps + analysis on 81 checkpoints)
- **Concurrent:** 18 jobs (QOS cap minus headroom)
- **Total wall-clock for 90 jobs:** ~5 waves of 18 = ~15-25 hours wall-clock
- **Total GPU-hours:** ~270-450 (free under NMSU institutional account)
- **Realistically done by:** Sunday night (5/4) if started Friday afternoon. Tuesday morning at the latest.

## What this answers

After 90 trajectories complete, we can run a partial-Spearman analysis of `h0_total_persistence` (peak or @40K) on `grokking_onset_step`, controlling for `weight_norm_l2` at the same step. That's the actual decisive test from the 04-21 reframe.

- If partial ρ > 0.30 with bootstrap CI excluding zero: topology adds residual signal → reframed grokking study justified
- If partial ρ ~= 0 or CI includes zero: topology angle dies → write up as Future Work, fold into PERSIST narrative

## Deferred (do not bother with these now)

- Full factorial (seeds × WD × tasks × models). Locked as premature until partial-rho test settles.
- Seed 7777 investigation. Quarantined.
- 100×100 grid resolution. Test on subset first (per next-steps 04-24 doc) before any blanket bump.
- Re-registering seed 42 as separate "instability dynamics" case. Possible future study.
