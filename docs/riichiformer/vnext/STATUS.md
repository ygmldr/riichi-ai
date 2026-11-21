# RiichiFormer VNext — Status

**Updated**: 2026-11-22  
**Lead branch**: `rf/vnext-core` (worktree `/home/ygsldr/riichi-ai/worktrees/vnext-core`)  
**Base**: `8a915d1` / tag `rf3.3-w2`  
**Contract hash**: `af7d9bb2c94228e4fda262f553a18b3c05da2e7fba1c5d51229231cd6399667d`  
**Model params**: 6,462,940

## What exists now

| Area | Code | CPU tests |
|---|---|---|
| Contracts | `riichiformer/vnext/contracts.py` | 12 contract tests |
| Event + 34-tile state | `observable.py` | included in model suite |
| Model / cross-attn / heads | `model.py` | 14 model tests |
| Runtime | `runtime.py` | 4 tests |
| GRP24 | `grp.py` | 7 tests |
| Critic | `critic.py` | 4 tests |
| BC adapter/trainer | `data.py`, `train.py` | 5 tests |
| RL contracts/toy | `rl.py` | 14 tests |
| V3.3 P1 | `m5d_arena.py`, `b0.py`, `rf_train_b0.py` | 8 synthetic tests + 43 B0 tests |

Last clean VNext-only CPU run: **62 passed** in 2.02 s
(`logs/vnext/vnext_cpu_only.log`), with CUDA hidden.  The VNext + P1 + B0
combined run is recorded separately; it does not substitute for a full native
Arena.

## What is measured

* CPU forward: 1,567 decisions/s, 200,616 event-tokens/s (32 decisions × 128
  tokens, 4 threads).
* CPU train step: 307 decisions/s, 39,339 event-tokens/s; peak RSS 804.6 MiB.
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
