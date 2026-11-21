# RiichiFormer VNext — Status

**Updated**: 2026-11-22
**HEAD**: `rf/vnext-core` tip; review fixes in `63ffd3b`
**Lead branch**: `rf/vnext-core` (worktree `/home/ygsldr/riichi-ai/worktrees/vnext-core`)
**Base**: `8a915d1` / tag `rf3.3-w2`
**Contract hash**: `6ce138249a06173e02804aea7e81e6c4b95cdaa22b57a03984c172eab0883e58`
**Model params**: 6,538,716

## What exists now

| Area | Code | CPU tests |
|---|---|---|
| Contracts | `riichiformer/vnext/contracts.py` | 12 contract tests |
| Event + 34-tile state | `observable.py` | included in model suite |
| Model / cross-attn / heads | `model.py` | 18 model tests |
| Runtime | `runtime.py` | 4 tests |
| GRP24 | `grp.py` | 9 tests |
| Critic | `critic.py` | 4 tests |
| BC adapter/trainer | `data.py`, `train.py` | 8 tests |
| RL contracts/toy | `rl.py` | 15 tests |
| V3.3 P1 | `m5d_arena.py`, `b0.py`, `rf_train_b0.py` | 10 synthetic tests + 43 B0 tests |

Last clean VNext-only CPU run: **72 passed** in 1.96 s
(`docs/riichiformer/vnext/logs/VNEXT_CPU_SUITE.log`), with CUDA hidden.  The
combined VNext + B0 + arena-synthetic run is **125 passed, 6 deselected, 6
subtests** (`docs/riichiformer/vnext/logs/VNEXT_COMBINED_CPU.log`); it does not substitute for a full
native Arena.

## What is measured

* CPU forward: 2,141 decisions/s, 273,996 event-tokens/s (32 decisions × 128
  tokens, 4 threads).
* CPU train step: 340 decisions/s, 43,513 event-tokens/s; peak RSS 806.0 MiB.
* This is a synthetic contract-valid microbench, not a full-epoch estimate.
* V3.3 inherited baseline: 841 passed / 6 skipped (historical log), fresh
  integration run recorded in `logs/vnext/baseline_v33_integration.log`.

## What is NOT done

* Full V3.3 P1 native Arena smoke (no VNext checkpoint; `vendor/mortal` is
  uninitialized in this worktree).
* Real 2.51M-match Census and real-data BC training.
* GPU profile, full B0 epoch, GRP fit, Arena Trend/Select/Confirm.
* RL native league/rollout and any AWR/PPO comparison.
* Any strength/dan/Tenhou/Mortal claim.

## Review

[`REVIEW.md`](REVIEW.md) records the first independent review (FAIL at
`138a77b`), the second review (`PASS_WITH_P1` at `63ffd3b`), every P0/P1
disposition, and the post-fix evidence.  P1s 3/7 (native multi-kan/EMA arena)
remain `NOT_RUN`; no strength claim is made.

## Gate

`G0` baseline/contracts: reviewable; `G1` model/runtime synthetic: reviewable;
`G2` GRP/Critic: unit-level only; `G3` BC: synthetic only; `G4` RL: toy only;
`G5` prereg: draft, awaiting explicit bounded-profile authorization; `G6`: not
started and blocked by policy.

## Rights

11 source rows, 0 `ALLOWED`; `SOURCE_RIGHTS_GATE.json` is
`BLOCKED_FOR_NEW_TRAINING`.  No checkpoint, shard or log was published or
uploaded.  No new seed window or Confirm pool was consumed.

## Next READY tickets

1. Initialize/verify the native submodule in the VNext worktree, then run the
   V3.3 baseline and a VNext zero-step native adapter smoke.
2. Run a real Train/Val Census on an explicitly authorized, rights-compatible
   subset (no full corpus transformation by default).
3. After user authorizes a bounded GPU profile, measure BF16 throughput/VRAM
   and only then decide whether a short BC pilot is worthwhile.
