# A14 VERDICT — minimal online self-play (todo 26, roadmap step 9)

Pre-registration: `runs/exp/A14/PREREG.md` (written 2026-09-19 20:54:08 +0800, before the first
training launch 21:04:23; sha256 `4f3032d7…`, recorded in `prepare_summary.json`).
Run: one full pool cycle = 8,000 games (4 pool members × 2,000), K=2, `online=true`,
**`use_pg=true`** (proof in §2), from `checkpoints/mortal.pth.at-148800`, 27.8 min wall clock.

## 0. Pool labelling — pool-B substitution (binding; decided in T23 C-1)

**Pool B `[50000,53000)` was consumed by todo 22** as the single confirmatory shot of the
architecture verdict (E3a−E2 Δpt −2.5200 [−4.3427,−0.6446]), and `60000` was consumed by the
akochan ruler (T16). A14 therefore runs, as declared in PREREG §0: **trend on pool A
(`seed_start=10000`, `[10000,13000)`)** and **confirm on the fresh, never-used window
`seed_start=70000` (`[70000,73000)`)** — same `seed_key=0x5dfaa4cef265cd7`, same 12,000-game
protocol. The confirm window is unseen-by-selection data, so the plan's intent (a held-out
confirmatory measurement) is preserved even though the label "pool B" no longer applies.
`50000` was not touched by this task.

## 1. Headline result (confirm window 70000; primary)

| item | value |
|---|---|
| source dir / games dir | `runs/exp/A14` / `runs/exp/A14/eval_confirm/games` (own log dir) |
| sets / games / incomplete_sets / ignored_files | **3000 / 12000 / 0 / 0** |
| rank dist (1/2/3/4) | `[2649 2487 2664 4200]` |
| avg_rank | **2.701250** 95 % CI **[2.682165, 2.719833]** |
| pt | **−18.056250** 95 % CI **[−19.500188, −16.589906]** |
| **paired Δ (trainee − anchor seats)** avg_rank | **+0.268333** 95 % CI **[+0.242886, +0.293111]** |
| **paired Δ (trainee − anchor seats)** pt | **−24.075000** 95 % CI **[−26.000250, −22.119875]** |

Report CSV: `runs/exp/A14/report.csv` (canonical copy; original
`runs/exp/A14/eval_confirm/report.csv`), produced by the **unmodified** `scripts/eval_report.py`
(B=2000 set-level bootstrap, rng seed 0). Paired Δ JSON: `runs/exp/A14/delta_confirm.json`
(`scripts/exp_a14_delta.py`; per-set zero-sum differential vs the 3 anchor seats, same bootstrap).
CI widths: 2.91 pt / 0.0377 rank. Measured resolution at 12k: 80 %-power MDD ≈ 0.0239 rank /
1.86 pt — **the observed effect is ≈10× the pt MDD and ≈11× the rank MDD.**

## 2. Pre-registered criteria and verdict

Decision rule (PREREG §6, frozen before launch): **primary PASS ⟺ `pt_ci_lo > 0`** (detectably
better than the anchor); FAIL-worse if `pt_ci_hi < 0`; NULL if the CI contains 0.
**Auxiliary PASS ⟺ raw `avg_rank_ci_hi < 2.500`.** A14 passes iff primary PASS **and**
auxiliary PASS.

- **Primary: FAIL (detectably worse).** The confirm Δpt 95 % CI **does not cross 0 — it lies
  entirely below 0**: the trainee is significantly *worse* than the anchor it was initialised
  from (lower bound −19.50 pt, i.e. ≈10 MDD below parity). The literal phrase "CI does not cross
  0" is satisfied only in the degenerate negative sense; under the pre-registered mapping this is
  a **fail**, not a pass.
- **Auxiliary: FAIL.** Confirm `avg_rank_ci_hi = 2.7198 > 2.500` (a copy of the anchor would sit
  at ≈2.50 with CI upper > 2.5, so this bar is demanding by construction — here it fails by a
  wide margin, 0.22 rank beyond the bar).
- **A14 does not pass.**

**Decision: CLOSE the self-play line under the declared recipe and budget.** Per the roadmap
stop-loss ("A14 不显著 → 不要加混合回放"), **A14b (mixed replay + generational league) is NOT
started**. The keep/close statement required by the plan: **close**; the line may only be
re-opened by a new dated pre-registration that controls the identified confounds (§4) on an
unconsumed seed pool.

## 3. Trend window (pool A) — direction consistent

| window | source games dir | sets/games | rank dist | avg_rank [95 % CI] | pt [95 % CI] |
|---|---|---|---|---|---|
| trend, pool A `[10000,13000)` | `runs/exp/A14/eval_poolA/games` | 3000/12000 | `[2659 2498 2722 4121]` | 2.692083 [2.672667, 2.711254] | −17.051250 [−18.528750, −15.573750] |
| confirm, fresh `[70000,73000)` | `runs/exp/A14/eval_confirm/games` | 3000/12000 | `[2649 2487 2664 4200]` | 2.701250 [2.682165, 2.719833] | −18.056250 [−19.500188, −16.589906] |

Both windows use their own `log_dir`; the trend's paired Δ (`runs/exp/A14/delta_trend.json`):
Δrank +0.256111 [+0.230222,+0.281672], Δpt −22.735000 [−24.705000,−20.765000]. Trend report:
`runs/exp/A14/report_poolA.csv`. Trend and confirm agree in sign and magnitude (both far outside
the noise floor) — this is not a resolution-limited null; the trainee **degraded**.

Reference on the same pool A / same champion (non-paired, different budget and weights): pinned
at-175200 scored 2.4768 [+2.46,+2.49] / +1.8600 pt; the five architecture arms 2.537–2.602. The
A14 trainee (2.692 / −17.05 pt) is far below all of them.

## 4. Why (mechanisms, honest scope)

The trainee started as an **exact byte copy** of `checkpoints/mortal.pth.at-148800`
(sha256 asserted equal before launch) and was the *same architecture* (plain CNN; the trainer
log prints `mortal params: 10,787,456`; the saved config carries `model.use_event_branch=false`
and no `event_branch.*` keys). The in-training session snapshots already show the drift:
wave 0 ≈2.47–2.50 (≈parity vs at-175200 / best-220000), wave 1 vs at-148800 2.5495 (−5.0 pt),
final eval 2.69–2.70 (−17 to −18 pt). Candidate mechanisms, all **properties of the recipe as
run** (declared in PREREG §2 where relevant):

1. **Upstream online-resume semantics do not load the optimizer state**: the init checkpoint's
   own config has `online=false` while the run is `online=true`, so `train.py`'s
   `if not online or state['config']['control']['online']` guard skips the AdamW state — the
   1,300 updates ran with **fresh Adam moments at the constant lr 1e-4** (first update ≈
   `lr·sign(grad)` per parameter). Declared in PREREG §2 as expected behaviour, not an incident.
2. **Online mode disables CQL** (`train.py:378`), removing the 5×CQL conservative Q regulariser
   the offline recipe relies on; the policy (softmax over Q) can then drift freely.
3. The only anchors are `KL(π_anchor‖π)` with λ=0.05 and `next_rank`·0.2. The KL was falling
   (0.053→0.040 in TB units, i.e. ×20 = KL ≈ 1.06→0.8) but clearly far too weak at this update
   magnitude/budget.
4. The budget is small (8,000 games / 1,300 steps); the loss terms were all finite and were
   optimised (dqn 0.236→0.073, huber 0.302→0.073, next_rank 0.786→0.119), so this is **objective
   mis-alignment / too-large steps**, not a diverged run.

**Scope of the conclusion**: *this* online RL recipe (as configured, at this budget) **damages** a
converged checkpoint. The run does not show "self-play RL cannot help" — a future pre-registered
attempt should first control mechanism 1 (proper optimizer/scheduler warm-start or lower lr) and
mechanism 2 (keep CQL online), and can only then test self-play as such.

## 5. Run-integrity record (all checks passed)

- **`use_pg=true` proof**: trainer log `[rl] use_pg=True lam_v=1.0 lam_kl=0.05 anchor=…` +
  `[rl] anchor loaded: … (version=4, steps=148800)`; TB scalars all finite over 13 save points
  (steps 148,900…150,100): `loss/pg_loss` 0.2398→0.3739, `loss/value_huber` 0.3024→0.0734,
  `loss/kl_anchor` 0.0533→0.0400 (plus dqn/next_rank). An inert (`use_pg=false`) run is excluded.
  TB dir: `runs/exp/A14/selfplay/cycle_1/wave_0/trainer/tb`.
- **Generation accounting**: 4 drains × `file list size: 2,000` = 8,000 games; worker logs:
  wave 0 = 2×2,000 (at-175200, best-220000), wave 1 = 2×2,000 (at-148800, baseline);
  no cap hit, `warnings: []`. Cycle 1 assign sampled fresh (`sample --cycle 1 --write`,
  2026-09-19T20:53:10): `[["at-175200","best-220000"],["at-148800","baseline"]]`.
- **Evaluation checkpoint**: `runs/exp/A14/state_final.pth`, a copy of the trainer's last periodic
  save (`steps=150,100`, i.e. +1,300 local steps), 130,732,154 B, sha256
  `ac95620f8f5d3d284e3741d7e2e48f053e9118ad9cba35172c5d734fc200bd87`. The final epoch ended 12 s
  after that save (≤ ~20 further steps / ≤ ~160 games not in the checkpoint; declared save gap
  ≤100 steps).
- **`save_every=100`** declared in PREREG §1 (the repo default 2,000 would have produced no
  checkpoint at this step count); `enable_compile=false`, `test_every=100000` (no test play).
- **First live exercise of patch 0013** (online tail event-batch padding): the run executed 4
  live drains with `num_workers=3` and batch 1024 and produced **no `torch.cat` error** —
  the tail path held on real GPU online data.
- **Frozen corpus**: `runs/1v3/*.json.gz = 2004` at every phase boundary. **Live checkpoints
  untouched**: `checkpoints/mortal.pth` mtime `2026-09-17 02:03:09`, `best.pth`
  `2026-09-16 02:59:45`. No writes outside `runs/exp/A14/**` + the pool file.
- **Wall clock** (budget ≤ ~4 h): training 27.8 min (cap 130 min), trend eval 37.0 min, confirm
  eval 36.6 min; A14 span 20:53→23:06 ≈ 2 h 13 min.

## 6. Operational findings (for A14b / future online work)

- Uncontended self-play throughput is **≈3.2 games/s per worker** (2,000 games in ~10 min);
  under concurrent training it was ≈1.0–1.1 games/s (measured in todos 24/25). The CNN trainer
  with `enable_compile=false` does ~1.6 optimizer steps/s ≈ 0.16 steps/game, so the arena, not
  the trainer, is the binding resource at these scales. 8,000 games are ~28 min end-to-end.
- The two-wave worker rotation inside one server/trainer launch worked as designed; worker
  `log_dir` rmtree only cost local evidence (games are submitted before the kill), never data.
- `save_every=100` is required for short runs; the last-save gap must be quoted with any verdict.

## 7. Artefacts and sources

- PREREG `runs/exp/A14/PREREG.md` · train log `runs/exp/A14/train.log` (copy of
  `runs/exp/A14/selfplay/cycle_1/wave_0/logs/trainer.log`) · runner `runner.log` +
  `eval_runner_{trend,confirm}.log` · `run_summary.json`, `prepare_summary.json`
- Reports: `report.csv` (confirm, primary), `report_poolA.csv` (trend), each also in its eval dir;
  deltas `delta_confirm.json`, `delta_trend.json`; eval configs `eval_trend.toml`
  (declared as `eval_poolA.toml` in PREREG §8 — same file, named by arm), `eval_confirm.toml`
- Game logs: `runs/exp/A14/eval_poolA/games/`, `runs/exp/A14/eval_confirm/games/`
  (each `sets=3000 games=12000 incomplete_sets=0 ignored_files=0`; source dirs recorded in the
  reports as `runs/exp/A14`; the per-arm `log_dir`s are pairwise distinct)
- Evidence: `.omo/start-work/evidence/T26.txt`. Scripts: `scripts/exp_a14_run.py`,
  `scripts/exp_a14_delta.py`, `scripts/selfplay_launch.py` (`--root`).
