# VNext Task Board (single Lead)

**Updated**: 2026-11-22  
**Integration branch**: `rf/vnext-core` (base `8a915d1`)  
**Rule**: a PR is `INTEGRATED` only after the Lead has inspected the diff, run
its tests (or recorded `NOT_RUN`), and received an independent review verdict.

| PR | Owner | Status | Base | Allowed paths | Evidence / next action |
|---|---|---|---|---|---|
| PR-00 Baseline audit | Lead | `REVIEW` | `8a915d1` | `docs/riichiformer/vnext/BASELINE_AUDIT.md`, `ACTION_MAP_LOCK.json`, `SOURCE_RIGHTS_GATE.json` | Fresh new-worktree suite is `ENV_BLOCKED` on uninitialized `vendor/mortal`; inherited V3.3 log is 841/6. Rights gate 0 ALLOWED. |
| PR-01 V3.3 P1 fixes | Lead/Runtime | `REVIEW` | `8a915d1` | `riichiformer/eval/m5d_arena.py`, `riichiformer/train/b0.py`, `scripts/rf_train_b0.py`, tests | EMA shadow test passes; multi-kan now requires KAN_SELECTION; validation reductions vectorized; resume restores prior candidate registry. GPU sync-count profile `NOT_RUN`. |
| PR-02 Contracts | Lead | `REVIEW` | `8a915d1` | `riichiformer/vnext/contracts.py`, `docs/.../CONTRACT_LOCK.md` | Manifest hash `af7d9bb2...`; action/schema hashes locked; target separation tested. |
| PR-03 Event + tile state encoder | Lead/Model | `REVIEW` | PR-02 | `riichiformer/vnext/observable.py`, tests | BOS/current-kyoku/p=0/same-prefix/34-tile/26-channel tests pass; hidden-mutation and long-corpus fixture `NOT_RUN`. |
| PR-04 Factorized/cross-attn/tile head | Lead/Model | `REVIEW` | PR-03 | `riichiformer/vnext/model.py`, tests | 6.46M-param default; mask/prefix/future-token/batch-order/ablation/backward tests pass; GPU benchmark `NOT_RUN`. |
| PR-05 Runtime/native parity | Lead/Runtime | `REVIEW` | PR-04 | `riichiformer/vnext/runtime.py`, tests | Cacheless reference adapter, kyoku reset, KAN_SELECTION second call; native Arena smoke `NOT_RUN` (no VNext checkpoint). |
| PR-06 GRP24 | Lead/Value | `REVIEW` | PR-02 | `riichiformer/vnext/grp.py`, tests | 24 permutations, marginals, zero-sum Phi, loss/calibration/state hash tests pass; real large fit `NOT_RUN`. |
| PR-07 Decision Critic | Lead/Value | `REVIEW` | PR-02 | `riichiformer/vnext/critic.py`, tests | Detach/joint gradient tests pass; calibration on real games `NOT_RUN`. |
| PR-08 BC census/trainer | Lead/Model | `REVIEW` | PR-03/04/01 | `riichiformer/vnext/data.py`, `riichiformer/vnext/train.py`, tests | Synthetic shard adapter, two-step BC, all-censored no-op, checkpoint roundtrip pass; real Census and throughput `NOT_RUN`. |
| PR-09 RL contracts | Lead/RL | `REVIEW` | PR-02/06/07 | `riichiformer/vnext/rl.py`, tests | Behavior μ, forced/guard, reward telescoping, two-timescale GAE, AWR/PPO toy tests pass; Native rollout `NOT_RUN`. |
| PR-10 League/toy | Lead/RL | `REVIEW` | PR-09 | `riichiformer/vnext/rl.py`, tests | League manifest and toy objectives only; native 1v3 smoke `NOT_RUN`. |
| PR-11 Preregistration/lock | Lead | `BLOCKED` | PR-08/09/10 | `docs/riichiformer/vnext/PREREG.md` | Needs real Census/throughput and explicit GPU authorization. |
| PR-12 Formal BC/RL/Arena | User-gated | `BLOCKED` | PR-11 | — | No long GPU job authorized; do not start. |

## Current blockers

1. `vendor/mortal` is uninitialized in `vnext-core`, so native tests that
   hard-code the worktree-local `.so` cannot run there.  The verified build
   exists in `worktrees/v33-integration`; either initialize the submodule or run
   native tests there.
2. No VNext checkpoint exists yet, so a real VNext native Arena smoke is
   `NOT_RUN`.
3. The real 2.51M-match Census is not executed; only a synthetic adapter test
   exists.  This is intentional: no long data transformation without measured
   need.
4. No GPU training/arena budget has been authorized.

## Non-negotiable stop line

No PR-12 work, no full B0 epoch, no broad GRP fit, no GPU arena league, and no
Confirm-pool consumption until the user explicitly authorizes resources after
PR-11.
