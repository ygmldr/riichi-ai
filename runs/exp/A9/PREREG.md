# A9 Pre-Registration — training-time shuffled history (fixed permutation)

Plan: `.omo/plans/fork-roadmap-steps-1-10.md` todo 20 · Roadmap `docs/FORK_ROADMAP.md` §8
("A9 打乱历史 (训练时)"). Status: **written before training and before the arm's evaluation.**

Repo state at pre-registration: HEAD is the todo-19/todo-17 commit series; vendor tree =
pin `0cff2b5` + patches 0002/0003/0004/0005/0006/0007/0010/0012 (unmodified by this arm —
A9 is launcher-level, see §2). Live checkpoints are read-only for this arm.

---

## 1. Question and relation to A1

A1 (todo 12) permuted river window item slots **at inference** on the pinned challenger with a
registered seed and preserved slot multisets: `shuffle − original` Δavg_rank **+0.045833**
CI [+0.024000, +0.068917] (≈1.9× the 12k MDD) ⇒ discard ORDER is load-bearing at inference for
a model trained on ordered history. A1 cannot distinguish "the encoding/architecture does not
extract order" from "order is not useful for learnable strength".

A9 is the training-time analogue: train a from-scratch model while the history-dependent inputs
are permanently reordered by **one fixed permutation**. Paired reading (roadmap §5 table,
decision D5):

| | A9 degraded | A9 unchanged |
|---|---|---|
| A1 worse | order is used AND useful (supports the sequence-model line) | contradiction ⇒ check protocol/resolution |

"Unchanged" is defined by the CI (roadmap wording: 95 % CI of the difference inside ±0.05
avg_rank), not by a non-significant p-value read as "no effect". A9 has no verdict authority:
its pool-A `report.csv` only feeds the five-arm ranking of todo 22 alongside E2/E3a/A7/A8.
**A null result is a valid deliverable.**

## 2. Arm definition

**Architecture = E3a exactly** (`runs/exp/E3a/train.toml` conditions): CNN c=192/b=40 plus the
4-layer d=128 causal event Transformer (`[model] use_event_branch=true`), residual zero-init
fusion. Training config `runs/exp/A9/train.toml` is generated from
`runs/exp/E3a/train.toml` by replacing the arm path prefix only (`diff` after prefix
normalisation = empty), with `use_event_branch=true`.

**Intervention** — applied inside `collate_batch` (runs in the DataLoader workers) to every
training and holdout batch, via a launcher-level monkey-patch of
`vendor/mortal/mortal/dataloader.py`'s `collate_batch` (`scripts/exp_a9_shuffle.py` +
`scripts/exp_a9_train.py`); **no vendor file is modified**:

- **River window blocks** — the same eight item-encoded blocks A1 permuted, offsets/strides read
  from `runs/analysis/obs_layout.json` at import: `(35,6,4) (59,18,4) (132,6,8) (180,18,8)
  (327,6,8) (375,18,8) (522,6,8) (570,18,8)`; `kawa_overview_*` (order-insensitive tile-set
  summaries) and the recency planes are **not** permuted. Permutation seed = A1's registered
  `0xA15EED`; the draw is `numpy.random.default_rng([0xA15EED, 0, 0])`, one
  `rng.permutation(items)` per block in layout order — i.e. exactly A1's (call 0, obs 0) RNG
  stream, drawn once at startup.
- **Event token order** — one fixed position permutation `pi` of `[0, 384)`
  (`numpy.random.default_rng([0xA15EED, 1])`, normalised `pi[0] > pi[1]` so that every
  length-class restriction `pi[:L]` (L ≥ 2) is non-identity). A stream of length L has its
  L real tokens reordered by `argsort(pi[:L])` (the length-consistent extension of a global
  position permutation); padding stays zero, `lengths`/`mask`/`causal` are untouched.
- **Why both pathways**: E3a receives history through the CNN river windows *and* the event
  Transformer; permuting only one would leave the other ordered and make a null near-tautological.
  The plan's wording ("牌河平面块 / 事件 token 顺序") allows this choice; it is declared here.
- **Declaration difference from A1**: A1 re-drew a permutation per `(react_batch call, obs)`; a
  training-time intervention must be fixed across samples, so A1's construction is used with one
  draw reused forever. Registered permutation fingerprint (SHA-256 over the 8 block perms +
  `pi`): `bf16f51cbea1b8a52f02351d29f3843814090ff97b2df72ee03caf60c5a88027`.

## 3. Input self-check (asserted inside the data pipeline, every batch)

`scripts/exp_a9_shuffle.py`:

- per permuted river block: the item-**slot multiset is preserved exactly** (bit-packed
  lexicographic sort compare; all-zero None slots as their own class) **and** per-element value
  counts (sum over slots) are preserved; uniform (all-empty) blocks are asserted unchanged; the
  permuted **batch is asserted to DIFFER** from the unpermuted one; differing channels are
  asserted to lie inside the window region (blast radius).
- event tokens: per-sample row multiset preserved, padding region unchanged, `lengths`/`mask`
  unchanged, and every non-uniform stream must change; the batch must contain ≥ 1 changed stream.
- synthetic `--selftest` includes negative controls (deliberately corrupted block must be
  caught); `scripts/exp_a9_pipecheck.py` repeats the A/B check on 64 **real** gz-log entries
  (original vs patched collate on identical inputs) plus DataLoader integration at
  `num_workers=0` and `num_workers=2`. Evidence: `runs/exp/A9/PVALID.txt` (+ `selftest`,
  `pipecheck.json`).

## 4. Training protocol (declared BEFORE launch)

- **Step budget: 20,000 optimizer steps** (batch 1024, `opt_step_every=1`, 20.48 M samples),
  **from scratch**, matching E3a/E2/A7/A8. `train.py` has no internal step cap: an external
  watchdog (`scripts/exp_a9_watchdog.py`) SIGTERMs after `state.pth` `steps == 20000` (the same
  pattern E3a/E2 used).
- Config `runs/exp/A9/train.toml`: identical to `runs/exp/E3a/train.toml` except arm paths
  (`state_file`, `best_state_file`, `tensorboard_dir`, `test_play`, `train_play`, `1v3.log_dir`)
  and the pre-existing `use_event_branch=true`. Same `batch_size=1024`, `save_every=2000`,
  `test_every=100000` (no test play), `num_workers=3`, `enable_compile=false`,
  `enable_amp=true`, dataset globs 2009–2022 + 2023 holdout, read-only
  `checkpoints/file_index.pth`, same optimizer/scheduler.
- Intervention applies to the holdout loss too (same collate); the holdout number is a
  diagnostic of the same distribution, not a clean generalization metric.
- **Safety**: generator + launcher assert that none of `state_file`/`best_state_file`/
  `tensorboard_dir`/`test_play`/`train_play` points at a live path; live mtimes
  (`checkpoints/mortal.pth` 2026-09-17 02:03:09, `best.pth` 2026-09-16 02:59:45) are recorded
  before/after. `runs/1v3/*.json.gz` stays at exactly 2,004 files.
- **GPU lock**: one training at a time via atomic `mkdir /tmp/opencode/train.lock`; this arm
  waits for the sibling A7/A8 training. Lock acquisition/takeover is recorded in
  `runs/exp/A9/watchdog.log` and the evidence file.

## 5. Arm evaluation protocol (pool A, declared BEFORE launch)

- 12,000 games = 3,000 sets, `seed_key = 0x5dfaa4cef265cd7`, **`seed_start = 10000`**
  (pool A; pool B `50000` is reserved for todo 22 and is NOT consumed here).
- challenger = `runs/exp/A9/state.pth` @20k (engine name `a9`); champion =
  `checkpoints/mortal.pth.at-148800` (the same champion as every other pool-A arm).
- own `log_dir` = `runs/exp/A9/games/` (no other arm shares it).
- `MORTAL_EVT_DEBUG=/tmp/opencode/a9_evt_debug.jsonl` for the whole eval; the per-arm event-feed
  gate `runs/exp/A9/EVAL_FEED_A9.json` is produced by `scripts/exp_e3a_check_feed.py`
  (branch-enabled Brain raises on `events=None`, so a branch-less run cannot complete).
- **The evaluation applies no permutation** — the intervention is a *training-time*
  intervention. This is deliberate: a matched-permutation eval would add A1's known
  inference-shuffle effect (+0.0458 avg_rank) to A9's training effect and destroy the paired
  reading in §1. Limitation recorded: the checkpoint is evaluated under natural history order.
- Report: `scripts/eval_report.py --games-dir runs/exp/A9/games --seed-key
  0x5dfaa4cef265cd7 --seed-range 10000-13000 --checkpoint state.pth --source-dir runs/exp/A9
  --out runs/exp/A9/report.csv`. Expected `sets=3000 games=12000 incomplete_sets=0`.
- **Resolution context**: at 12 k games the 80 %-power MDD is ≈ **0.0239 avg_rank / 1.86 pt**
  (`runs/variance/README.md`). Pool-A reference numbers (same key/window, champion at-148800):
  E3a 2.5506 / −4.4138 pt; E2 2.5631 / −4.0163 pt; differences at this scale are below the
  resolution — interpretation belongs to todo 22 (pool B, single shot).

## 6. Artefacts

`runs/exp/A9/{PREREG.md, train.toml, eval.toml, train.log, state.pth, report.csv, PVALID.txt,
pipecheck.json, selftest.json, EVAL_FEED_A9.json, TRAIN_SUMMARY.md, watchdog.log, games/}`;
scripts `scripts/exp_a9_{shuffle,train,run.sh,watchdog,make_configs,pipecheck}.py`;
evidence `.omo/start-work/evidence/T20.txt`.

---

## 7. Addendum — interruptions and restart (2026-09-19)

- **Run 1** launched 2026-09-19 **11:22:06** (main pid 6108, watchdog pid 6140, 3 DataLoader
  workers) after the lock was taken at 11:16:48. It was killed by a **WSL restart at ~11:37**
  at ~step 600 — **before the first `save_every=2000` checkpoint, so no `state.pth` exists**
  and the arm restarts **from scratch**. The declared 20,000-step budget and all PREREG
  conditions are unchanged; run 2 is a fresh start, not a resume.
- Run-1 fragments kept on disk (not staged): `train.log.killed-wslrestart-20260919_1136`
  (42,455 B; 12 in-pipeline self-check lines, every one reporting
  `blocks=… changed=… uniform=0` and `ev_checked==ev_changed`), `tb.killed-wslrestart-20260919_1136/`,
  `watchdog.log.killed-wslrestart-20260919_1136`.
- The 2026-09-19 morning machine reboot (10:44) also destroyed a previously completed 100-step
  smoke run held under `/tmp/opencode/a9smoke/` (it had verified the launcher/monkey-patch path:
  perms printed, `dataloader.collate_batch` patched, `mortal params: 11,818,752`,
  `event tokens: (256,148,5) nonzero=412831`, `holdout_loss: 8.913287 / 9.064880` finite, saves at
  steps 50/100). It also exposed a pre-existing vendor limitation (unreachable in the bounded
  20k run): at the very end of a tiny epoch, `cat_event_batches` fails when tail batches have
  different event lengths; with the full 1.97 M-file corpus the epoch tail is never reached
  before the watchdog stops at 20,000.
- WSL restart reduced host RAM to ~15 GB: one GPU job at a time is enforced by holding the single
  `/tmp/opencode/train.lock` from training through the pool-A evaluation.
