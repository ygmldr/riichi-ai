# VNext Task Board (single Lead)

**Updated**: 2026-11-22
**Integration branch**: `rf/vnext-core` (base `8a915d1`)
**Rule**: a PR is `INTEGRATED` only after the Lead inspected the diff, ran its
tests (or recorded `NOT_RUN`), and received an independent review verdict.
`INTEGRATED` describes the code path, not real-data training or strength.

| PR | Owner | Status | Evidence / remaining |
|---|---|---|---|
| PR-00 Baseline audit | Lead | `INTEGRATED` | Actual `rf3.3-w2` base inspected; rights gate 0 ALLOWED. |
| PR-01 V3.3 P1 fixes | Lead/Runtime | `INTEGRATED` | EMA shadow, KAN_SELECTION helpers, vectorised validation, resume registry; native 4-arg bridge with 3-arg fallback test. Multi-kan end-to-end still `NOT_RUN`. |
| PR-02 Contracts | Lead | `INTEGRATED` | Contract hash `6ce13824...`; action/schema/normalizer hashes locked. |
| PR-03 Event + tile state | Lead/Model | `INTEGRATED` | BOS/current-kyoku/p=0/34-tile/26-channel tests pass. |
| PR-04 Model / cross-attn / heads | Lead/Model | `INTEGRATED` | Mask/prefix/future/order/ablation/backward tests; CUDA profile 9,132 forward / 1,636 train decisions/s. |
| PR-05 Runtime/native | Lead/Runtime | `PARTIAL` | Cacheless adapter, kyoku reset, native 4-game smoke 579 decisions, 0 illegal/fallback. `kan_selections=0`, so the end-to-end multi-kan `KAN_SELECTION` path remains `NOT_RUN`. |
| PR-06 GRP24 | Lead/Value | `INTEGRATED` | 24 permutations, marginals, zero-sum Phi, independent checkpoint; synthetic CUDA 50-step loss finite. |
| PR-07 Decision Critic | Lead/Value | `INTEGRATED` | Detach/joint gradient tests; real-game calibration `NOT_RUN`. |
| PR-08 BC census/trainer | Lead/Model | `INTEGRATED` (real pilot) | Real `rf_train_vnext.py`, authenticated shards, 1,000-game/661k-decision pilot, held-out val CE 1.5983→1.5285. Full 2.51M epoch `NOT_RUN` (throughput). |
| PR-09 RL contracts | Lead/RL | `INTEGRATED` (toy) | Behavior μ/guard, PBRS/Mortal delta, GAE, AWR/PPO tests; synthetic CUDA finite; native rollout `NOT_RUN`. |
| PR-10 League | Lead/RL | `INTEGRATED` (toy) | League manifest/toy only; native 1v3 league `NOT_RUN`. |
| PR-11 Preregistration | Lead | `IN_PROGRESS` | Operator-authorized local pilot complete; real Census limited to the 2009 pilot/val shards; full-corpus Census not run. |
| PR-12 Formal BC/RL/Arena | Lead | `IN_PROGRESS` | Real 1,000-step BC pilot passed; full 2009-year shard build in progress. No Arena/strength claim. |

## Current blockers

1. **Rights**: external registry still 0 `ALLOWED`; the operator explicitly
   authorized local research on 2026-11-22.  The local policy is not an upstream
   licence and must not be described as one.
2. **Census**: the 2.51M reported matches are not an eligible decision count;
   the real match-level audit is `NOT_RUN`.
3. **Multi-kan native**: ankan/kakan occurred in the smoke, but each had one
   candidate (`kan_selections=0`); the second `KAN_SELECTION` path is still
   `NOT_RUN`.
4. **Frozen M5D registry**: the VNext driver edits and local native build do not
   match the committed registry. It was not silently rewritten; reissuing that
   identity is a separate anchor-rebase decision.
5. **Real Arena strength**: not run; no paired evidence, no claim.

## Non-negotiable stop line

No real-data training, no full B0 epoch, no broad GRP fit, no GPU arena league,
and no Confirm-pool consumption until the rights gate and an audited Train
subset exist.  Synthetic profiles may continue without a data licence, but they
must stay labelled `SYNTHETIC_ONLY`.
