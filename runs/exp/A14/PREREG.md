# A14 Pre-Registration — minimal online self-play (REINFORCE + Huber(V) + forward-KL)

Plan: `.omo/plans/fork-roadmap-steps-1-10.md` **todo 26** · Roadmap `docs/FORK_ROADMAP.md` step 9 (A14, the minimal version).
Written: **2026-09-19 20:54:08 +0800**, before any training of this experiment (no optimizer step of A14 has run at
the time of writing; the only prior self-play CPU/GPU work is todos 24/25's orchestration smokes, which this
document does not use as data). Repo HEAD `6986196`; vendor tree = pin `0cff2b5` + patches
`0002/0003/0004/0005/0012/0006/0007/0010/0008/0009/0013` (11 files dirty, `+557/−63`; patch 0013 is the
online tail-event padding fix). `configs/config.toml` sha256 `c12bc5f3…` (unmodified by this todo);
`runs/exp/SELFPLAY/pool.json` sha256 `96f54472…` after the cycle-1 sample (§3).

Status markers: this file is frozen before launch. Any post-launch deviation is recorded as a dated **addendum**
at the bottom, never by editing the sections above.

---

## 0. Pool-B substitution (binding, decided in T23/decisions.md C-1 — not re-litigated here)

- **Pool B `[50000,53000)` was consumed by todo 22** as the single confirmatory shot of the architecture verdict
  (E3a−E2 paired Δpt −2.5200, CI [−4.3427,−0.6446] ⇒ CLOSE the river-transformer line). It cannot serve A14.
- `60000` was consumed by the akochan ruler (T16). `10000/20000/30000` are pool A / variance pools.
- **A14 therefore runs: trend on pool A (`seed_start=10000`, `[10000,13000)`), confirm on a fresh, never-used
  window `seed_start=70000` (`[70000,73000)`),** same `seed_key = 0x5dfaa4cef265cd7`, same 12,000-game protocol.
  The confirm window is unseen-by-selection data, so the plan's intent (a held-out confirmatory measurement) is
  preserved even though the literal label "pool B" no longer applies. **`50000` is not touched again.**
- The previous arms' reference numbers on the same key: pinned at-175200 scored avg_rank 2.4768 [2.4599,2.4938],
  pt +1.8600 [+0.5663,+3.2026] on pool A; the five architecture arms landed 2.537–2.602 (A8 3.607).

## 1. Generation definition, games budget, wall-clock budget

- **A generation = one full pool cycle = every active pool member faced exactly once.** Cycle 1 has
  `ceil(4/2) = 2` waves (§3). Per wave, K=2 worker processes each play **one session of G = 2,000 games**
  against its assigned frozen member (3 opponent seats) with the trainee on 1 seat.
- **Generation size = 4 members × 2,000 games = 8,000 games** (2 waves × 2 workers × 2,000).
- **Number of generations to run: exactly 1.** The plan/roadmap require "≥ 1 full generation"; this satisfies it.
  **No extension after seeing results** (the budget is an upper bound, not a target to fill).
- **Wall-clock budget** (all declared before launch):
  - training hard cap **130 min** from the first worker start; per-wave soft cap **75 min** (on a cap, the
    current wave's workers are stopped at their next submission and the shortfall is recorded);
  - the whole experiment (setup + training + two 12k evaluations + reports) is targeted at ≤ ~4 h.
  - Basis: measured self-play throughput in todo 24/25 smokes ≈ 1.0–1.15 s/game/worker (≈ 0.9–1.0 games/s per
    worker; sessions of 100 games took 96–116 s), so 8,000 games are expected to take ~80–95 min wall clock.
- **Optimizer-step budget (expected, not a hard rule):** the online loader yields ≈ 0.125 optimizer steps per
  game (measured: 200-game drain → 25 steps at batch 1024), so the generation is expected to deliver
  ≈ 1,000 steps on top of the starting step 148,800. Because that is far below the repo's `save_every=2000`,
  A14 uses **`save_every=100`** in its trainer config (declared deviation; every ~800 games) so the run has
  usable checkpoints at all. The evaluation checkpoint is the **last periodic save**; the gap between it and the
  final training step is ≤ 100 steps and is recorded in VERDICT.

## 2. Trainee initialization, architecture, RL objective

- **Initialization: an exact byte copy of `checkpoints/mortal.pth.at-148800`** (steps=148800, sha256
  `73ec8108…`) placed at
  `runs/exp/A14/selfplay/cycle_1/wave_0/trainer/state.pth` (src/dst sha256 asserted equal before launch).
  Rationale: the evaluation champion **is** at-148800, so "Δ vs the anchor" is a meaningful change-from-start
  measurement; and the plan's A14 recipe names 148.8k as the frozen offline anchor / old champion.
- **KL anchor = the same file** (`[rl].anchor_state_file = checkpoints/mortal.pth.at-148800`): it is read-only
  in training (loaded into a no-grad model) and is never written. Declared: the KL center and the performance
  anchor are the same checkpoint.
- **Architecture: plain CNN** — every A14 config (trainer + workers) is post-processed to
  `[model] use_event_branch = false` (the repo config currently has `true` for the owner's live run; A14
  overrides it in its own generated configs only). The architecture line is CLOSED (todo 22) and A14 is a
  separate lever (RL + self-play); the CNN is the architecture to train. Expected trainer log:
  `mortal params: 10,787,456` (c=192/40 CNN; an event-branch model would print 11,818,752). This is asserted
  from the log as a launch-health check.
- **RL objective (patch 0008):** `[rl] use_pg = true`, `lam_v = 1.0`, `lam_kl = 0.05`. `use_pg` is the master
  switch; the plan/patch consult `lam_kl`/`lam_v` **only** when it is true — an inert RL run is the single most
  likely way to produce a meaningless A14, so the run must show `[rl] use_pg=True` and the anchor load line in
  the trainer log, and non-NaN `loss/pg_loss`, `loss/value_huber`, `loss/kl_anchor` scalars in TensorBoard.
  Loss = `dqn + cql(online: 0) + 0.2·next_rank + pg + 1.0·huber(v,G) + 0.05·KL(π_anchor‖π)`;
  `G = gamma^steps_to_done · kyoku_rewards` (gamma=1).
- **Resume semantics (declared expected behaviour, not an incident):** the initial state's own config has
  `[control].online = false` while the A14 run is `online = true`, so `train.py`'s condition
  `if not online or state['config']['control']['online']` **does not load the optimizer state** (fresh AdamW;
  lr is the constant 1e-4 from `[optim.scheduler]`). `scheduler`, `scaler`, `best_perf`, `steps` are loaded,
  and `steps` continues from 148,800.
- **Other trainer settings:** `save_every=100`, `test_every=100000` (no test play), `submit_every=400`,
  `batch_size=1024`, `opt_step_every=1`, `enable_compile=false` (declared; dynamic online batch shapes would
  recompile — same choice as E2/E3a/A7–A9), `freeze_bn.mortal=true`, `[dataset].num_workers=3`,
  `[online.server].sample_reuse_rate=0`.

## 3. Pool cycle used

`python3 scripts/selfplay_launch.py sample --cycle 1 --k 2 --write` (sampler seed 20260925, weighted without
replacement) produced **cycle 1** (written 2026-09-19T20:53:10+08:00; `pool.json` now carries
`assign.cycle_1`; `pool_source_sha256` is asserted by `launch`):

| wave | worker 0 member | worker 1 member |
|---|---|---|
| 0 | `at-175200` (`runs/exp/PINNED/mortal.pth.at-175200`, sha256 `f0a611b9…`, steps 175200) | `best-220000` (`checkpoints/best.pth`, sha256 `7fe084d9…`, steps 220000) |
| 1 | `at-148800` (`checkpoints/mortal.pth.at-148800`, sha256 `73ec8108…`, steps 148800) | `baseline` (`checkpoints/baseline.pth`, sha256 `83fd3929…`, random-baseline floor) |

Rotation semantics (todo 24): a worker process loads exactly one member for its whole life; changing the
opponent = restarting the worker with a new config. A14 runs **one server + one trainer** and rotates the two
worker processes between waves, so training is continuous across the whole generation.

## 4. Training orchestration and stop rules

- **K=2** (measured VRAM ceiling: 12,945/16,303 MiB peak with a ~4 GiB WSL baseline; MemAvailable low 4.5 GB).
  One GPU lane only; `/tmp/opencode/train.lock` is held for the whole A14 sequence (training + both evals).
- Worker stop: as soon as a worker's log shows `logs have been submitted` for its session, its process group is
  `SIGKILL`ed (no `SIGCONT`, per the T24 rmtree lesson). Its games are already in the server buffer at that
  point; if the kill races the next session's `shutil.rmtree`, only local evidence is affected, never data.
- **Catch-up rule (run is not stopped until the data is in training):** server `buffer/` empty **and**
  Σ `file list size:` over trainer epochs ≥ submitted games **and** a `param has been submitted` line after the
  final drain, plus a 60 s dwell.
- Stop conditions (none depends on training results): the declared caps; a trainer crash (§7).
- Reboot/interruption: resume from the last periodic save (≤100 steps) with a dated addendum; games already
  drained but not yet trained are lost and the shortfall recorded.

## 5. Evaluation protocol (declared before launch)

Both evaluations use the same protocol as every previous verdict in this plan:

- 12,000 games = 3,000 sets, `seed_key = 0x5dfaa4cef265cd7`, `iters=1`, `games_per_iter=12000`;
- **challenger** = `runs/exp/A14/state_final.pth` (snapshot of the trainer's last periodic save; the exact
  global step and sha256 are recorded in VERDICT); **champion** = `checkpoints/mortal.pth.at-148800`;
- `enable_rule_based_agari_guard=true`, `enable_amp=true`, `enable_compile=false`;
- **trend**: `seed_start=10000` (pool A), log_dir `runs/exp/A14/eval_poolA/games/`, report
  `runs/exp/A14/report_poolA.csv`;
- **confirm (primary)**: `seed_start=70000` (fresh window), log_dir `runs/exp/A14/eval_confirm/games/`, report
  `runs/exp/A14/report.csv`.
- Report tool: `scripts/eval_report.py` **unmodified** (set-level bootstrap B=2000, `numpy` rng seed 0;
  fields `avg_rank`, `avg_rank_ci_lo/hi`, `pt`, `pt_ci_lo/hi`, `rank_dist`, `sets`, `games`,
  `incomplete_sets`, `ignored_files`). Both reports must show `sets=3000 games=12000 incomplete_sets=0`.
- **Paired Δ (trainee − anchor):** in this 1v3 protocol the games are zero-sum in rank and points, so the
  challenger's per-set metrics *are* differentials against the 3 anchor seats. `scripts/exp_a14_delta.py`
  additionally recomputes the per-set vectors and the explicit paired differences
  `d_rank = r − (10−r)/3`, `d_pt = p − (−p/3)` (anchor-seat means), with the same B=2000/rng-seed-0 bootstrap
  procedure reused from `eval_report.py` / `exp_arch_b_delta.py`; recorded for both windows.
- **Resolution context (binding):** at 12k games the 80 %-power minimum detectable difference is
  ≈ **0.0239 avg_rank / 1.86 pt** (`runs/variance/README.md`; σ_set = 0.4678 / 36.37). Effects below the MDD
  are **not** evidence of no effect; the confirm CI width is quoted in VERDICT.
- **Invariants per run:** `ls runs/1v3/*.json.gz | wc -l` == **2004**; live checkpoints untouched
  (`checkpoints/mortal.pth` mtime 2026-09-17 02:03:09, `best.pth` 2026-09-16 02:59:45); no writes outside
  `runs/exp/A14/**` and the pool file.

## 6. Decision rule (pre-registered; no post-hoc changes)

On the **confirm** window (70000), with `pt` = the trainee's per-game point differential vs the anchor seats
and `rank` = the trainee's average placement:

- **Primary PASS** ⟺ the `pt` 95 % CI lies entirely above 0 (`pt_ci_lo > 0`) — the trainee is detectably
  better than the anchor.
- **Primary FAIL (worse)** ⟺ `pt_ci_hi < 0`. **Primary NULL** ⟺ the CI contains 0.
- **Auxiliary PASS** ⟺ `avg_rank_ci_hi < 2.500`.
- **A14 PASSES** (self-play line stays open; A14b remains a *conditional* next step and is **not** started in
  this todo) ⟺ primary PASS **and** auxiliary PASS.
- **Otherwise** (null, worse, or auxiliary fail) → **A14 does not pass: close the self-play line under this
  recipe and budget**, do not add mixed replay / league (roadmap stop-loss: "A14 不显著 → 不要加混合回放"),
  and record the negative/null result as the deliverable.
- Honest expectations fixed in advance: the trainee starts as an exact copy of the anchor, so an unchanged
  model scores pt ≈ 0 (CI containing 0) and avg_rank ≈ 2.5 with CI upper > 2.500 — i.e. **the auxiliary bar is
  demanding and a null will fail both criteria by construction**. A null is a legitimate outcome; the
  information delivered is the protocol + the CI width at the measured resolution, not a p-value hunt.

## 7. Known risks and incident protocol

- **Patch 0013 (online tail event-batch padding) has never been exercised by a live GPU online run** — it was
  verified by a deterministic replay of the real run2 drain set. If the trainer raises a `torch.cat` /
  `Expected size N but got size M` error: **stop, preserve the traceback, the drain dir and the buffer, and
  report it as a fresh finding**; if a vendor fix is required, emit a **new numbered patch** and say so loudly.
  No silent re-patching.
- `player.py:137-138` rmtrees the worker `log_dir` at every session start: every A14 worker log_dir is a fresh
  directory under `runs/exp/A14/selfplay/cycle_1/wave_<w>/worker_<k>/`; the launcher asserts the prefix and
  freshness and refuses otherwise. `runs/1v3/`, `runs/variance/`, `runs/akochan_eval/`,
  `runs/exp/SELFPLAY/**` evidence dirs and other arms' directories are never used as log dirs.
- Live checkpoints (`checkpoints/mortal.pth`, `best.pth`) are never a `state_file` target; every scratch path
  is under `runs/exp/A14/` and the live mtimes are re-checked after each phase.
- A worker crash stalls its wave; the orchestrator leaves the other processes alive and the cap/health checks
  decide. A trainer non-zero exit is an incident unless it matches the documented `sys.exit(0)` test-play
  restart (unreachable here: `test_every=100000`, steps ≈149k).

## 8. Artefacts (paths fixed before launch)

- `runs/exp/A14/{PREREG.md,train.log,report.csv,report_poolA.csv,VERDICT.md,runner.log,run_summary.json,
  state_final.pth,delta_confirm.json,delta_trend.json,eval_poolA.toml,eval_confirm.toml}`
- run tree: `runs/exp/A14/selfplay/cycle_1/wave_{0,1}/**` (configs, worker dirs, server buffer/drain,
  `trainer/{state.pth,tb}`), worker game logs under each `wave_*/worker_*/`
- eval game logs: `runs/exp/A14/eval_poolA/games/`, `runs/exp/A14/eval_confirm/games/`
- scripts: `scripts/selfplay_launch.py` (adds `--root`), `scripts/exp_a14_run.py` (orchestrator),
  `scripts/exp_a14_delta.py` (paired Δ); evidence `.omo/start-work/evidence/T26.txt`.

## 9. Reproducible command summary

```bash
source scripts/activate_env.sh
python3 scripts/selfplay_launch.py sample --cycle 1 --k 2 --write
python3 scripts/exp_a14_run.py prepare      # cycle 1 configs + post-processing + init-state copy
python3 scripts/exp_a14_run.py train        # requires /tmp/opencode/train.lock; detached in practice
python3 scripts/exp_a14_run.py eval --arm trend
python3 scripts/exp_a14_run.py eval --arm confirm
```

---

## Addendum log

_(empty before launch; append dated entries only)_
