# RiichiFormer VNext — Status

**Updated**: 2026-11-22
**Branch / worktree**: `rf/vnext-core` (`/home/ygsldr/riichi-ai/worktrees/vnext-core`)
**Base**: `8a915d1` / tag `rf3.3-w2`
**Contract hash**: `6ce138249a06173e02804aea7e81e6c4b95cdaa22b57a03984c172eab0883e58`
**Model params**: 6,538,716

## What exists now

| Area | Code | Evidence |
|---|---|---|
| Contracts | `riichiformer/vnext/contracts.py` | contract tests |
| Event + 34-tile state | `observable.py` | model/stream suite |
| Model / cross-attn / heads | `model.py` | model suite, CUDA profile |
| Runtime / native bridge | `runtime.py`, `eval/m5d_arena.py` | native 4-game smoke |
| GRP24 | `grp.py` | unit tests, synthetic CUDA loss |
| Critic | `critic.py` | gradient-isolation tests |
| BC adapter/trainer | `data.py`, `train.py` | synthetic CUDA 100-step smoke |
| RL contracts/toy | `rl.py` | unit tests, synthetic CUDA objective |
| V3.3 P1 | `m5d_arena.py`, `b0.py`, `rf_train_b0.py` | synthetic tests |
| Native submodule | `vendor/mortal` @ `0cff2b5` | local build + smoke |

## 2009 full-year run (launched 07:01)

* shard complete: **6,897 games / 4,568,777 decisions / 4,545,919 supervised**
* shard SHA `f411c9e2...`; source `tenhou_old`; policy `f0d472ee...`
* Census: SELF_TURN 3,635,926; CALL_RESPONSE 927,493; KAN_SELECTION 5,358;
  prefix P50/P90/P99/max 549/1,096/1,524/4,861
* 460 decisions above the 4,096 positional bound are skipped and counted
  (`--max-prefix-tokens 4096`), not silently truncated
* full BC: 71,387 steps, batch 64, warmup+cosine, 5-hour cap, output
  `runs/vnext_bc_2009`
* validation at 5,000: 50-batch BC CE 1.4891 / rank 1.2881; 500-batch re-eval
  **1.5019 / 1.1787**. At 10,000: 50-batch 1.4666 / 1.2870; 500-batch
  1.4895 / 1.1472. At 15,000: 50-batch **1.4539 / 1.2557**; 500-batch
  **1.4720 / 1.1363**; checkpoints
  `step_005000.pt` / `step_010000.pt`.

## Current measured evidence

* VNext-only CPU: **79 passed** in 1.81 s
  (`logs/VNEXT_REAL_TRAIN_SUITE.log`), including the real-trainer CLI tests.
* Combined VNext + B0 + arena/bridge + anchor CPU: **149 passed, 6 skipped,
  13 subtests** (`logs/VNEXT_COMBINED_REAL.log`).
* M5D anchor tests alone: **9 passed, 1 skipped, 7 subtests**; the skip is the
  documented frozen-registry driver/build drift below.
* Native VNext smoke: **4 games, 579 decisions, 0 illegal, 0 fallback,
  21.009 s**. The logs contain one `ankan` and one `kakan`, but
  `kan_selections=0`: the second `KAN_SELECTION` decision was not exercised.
  `logs/VNEXT_NATIVE_SMOKE.log`, `runs/vnext_native_smoke/smoke_report.json`.
* Bounded CUDA profile (128 × 256, FP32, 20 iterations):
  **median 12,433.7 forward decisions/s**, **median 1,635.6 train decisions/s**,
  **144.6 / 331.9 MiB** peak VRAM. Same-shape CPU: 1,638.0 / 413.4.
  `logs/VNEXT_GPU_PROFILE_FINAL.json`, `logs/VNEXT_GPU_PROFILE_FINAL2.json`,
  `logs/VNEXT_CPU_PROFILE_128x256.json`.
* Synthetic CUDA BC/GRP/RL: 100 BC steps finite, checkpoint state round-trip
  unchanged, GRP 50 steps finite, behavior/AWR/PPO/PBRS/GAE finite.
  `logs/VNEXT_GPU_SMOKE_AFTER_SYNC_FIX.log`.
* **Native trained-checkpoint smoke**: 4 games / 765 decisions and a
  16-game lifecycle rerun / 2,792 decisions and 8-game / 16-game 10.5k-checkpoint runs /
  1,296 / 2,893 decisions, 0 illegal/fallback; the 16-game run exposed and fixed a
  missing per-game backend reset.
  `logs/VNEXT_NATIVE_TRAINED_SMOKE.log`, `logs/VNEXT_NATIVE_TRAINED_16G.log`;
  two repeated 8-game runs produced identical action-trace hashes
  (`logs/VNEXT_NATIVE_DETERMINISM.log`).
* **Real-data pilot**: 1,000 games / 661,627 decisions train; 500 games /
  326,798 decisions held-out val.  1,000 steps: val BC CE 1.5983 → 1.5285.
  5,000 steps: val BC CE **1.5893 → 1.4719**, rank CE **1.3682 → 1.2956**,
  657.5 s; a separate 500-batch validation gives 1.4934 / 1.1744.  See [`PR12_REAL_PILOT.md`](PR12_REAL_PILOT.md).  Loss only; no
  strength claim.

The first GPU profile was 323.8 forward / 274.0 train decisions/s because every
forward re-ran the scalar-heavy `VNextModelBatch` contract.  `VNextPolicy.forward`
now fails closed on a device mismatch instead of re-validating/copying.  That is
a measured performance fix, not an architecture change.  Three after-fix runs
varied with WSL/GPU clocks: 9,133–17,934 forward and 1,174–2,491 train
decisions/s; the table uses the median.

## Native-build caveat

**ANCHOR_GATE: ENV_MISMATCH_REQUIRES_REBASE** — the smoke is not a frozen-M5D
reproduction.

The local build is `171bd45a...` with patch-manifest
`fb2d57f19b8...`, while the committed frozen M5D registry pins
`582e3673...` and the pre-VNext driver hashes.  The registry was **not**
rewritten.  A local smoke-only registry was generated under
`runs/vnext_native_smoke/`.  Therefore:

* the native smoke proves ABI/runtime integration, not reproduction of the
  original frozen Arena;
* `test_committed_yaml_matches_recomputation` skips when the pinned driver
  sources differ;
* reissuing the M5D anchor identity is a separate, explicit anchor-rebase
  decision, not something this branch did silently.

## What is NOT done

* Real 2.51M-match one epoch: measured ~47 days at the current CPU-bound data
  rate; only a full 2009-year run is feasible in this window (build in progress).
* Full B0 epoch, broad GRP fit, AWR/PPO league, Trend/Select/Confirm.
* End-to-end multi-kan (`KAN_SELECTION`) native test; this smoke had only
  single-candidate ankan/kakan.
* Real EMA checkpoint Arena comparison.
* Any strength/dan/Tenhou/Mortal claim.

## Review

[`REVIEW.md`](REVIEW.md) records the first independent review (FAIL at
`138a77b`), the second review (`PASS_WITH_P1` at `63ffd3b`), and the third
review of the native/GPU changes (`PASS_WITH_P1`, no P0), and the fourth review
of the real-data trainer (`PASS_WITH_P1`, no P0).  The Lead applied the review
fixes; they were not re-reviewed.  The remaining multi-kan native
case is explicitly `NOT_RUN`, so PR-05 is not called complete for
`KAN_SELECTION`.

## Gate

`G0` baseline/contracts: reviewable; `G1` model/runtime: synthetic + native
smoke; `G2` GRP/Critic: unit/synthetic only; `G3` BC: real 1,000-game pilot;
`G4` RL: toy/synthetic only; `G5` prereg: GPU resource authorised, data gate
still closed; `G6`: not started.

## Rights

11 source rows, **0 `ALLOWED`**; `SOURCE_RIGHTS_GATE.json` is
`BLOCKED_FOR_NEW_TRAINING`.  No restricted dataset, checkpoint, shard or game
log was read for training or uploaded.  No new seed window or Confirm pool was
consumed.  GPU availability is not a data licence.

## Next READY tickets

1. Resolve or explicitly re-scope the source-rights gate for a Train subset.
2. Run the real match-level Census only on that cleared subset.
3. If the gate opens, run a bounded real BC profile (not a full epoch) and
   record throughput/VRAM; keep Confirm untouched.
4. Separately decide whether to reissue the M5D anchor identity for the modified
   driver/native build.
