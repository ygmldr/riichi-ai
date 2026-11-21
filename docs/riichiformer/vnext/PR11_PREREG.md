# PR-11 — VNext BC Preregistration (DRAFT, RIGHTS-BLOCKED)

**Status**: `DRAFT_RIGHTS_BLOCKED`
**Experiment ID**: `VNEXT-BC-PILOT-1`
**Owner**: single Lead
**Base**: `rf/vnext-core` at `8a915d1` + VNext PRs
**Frozen contract hash**: `6ce138249a06173e02804aea7e81e6c4b95cdaa22b57a03984c172eab0883e58`
**Model architecture hash**: `012f037139346913433c5129f6720f70e7708618a31af75fcd206f05d719f5df`
**Model parameters**: 6,538,716

This is a **plan**.  The user authorised GPU use on 2026-11-22, and a bounded
synthetic CUDA profile has run.  The rights gate remains
`BLOCKED_FOR_NEW_TRAINING`; no restricted corpus has been read and no real BC
epoch has started.

## 1. Fixed model / objective

| Item | Value |
|---|---|
| event backbone | 6 layers, d256, 4 heads, RoPE, pre-RMSNorm, SwiGLU |
| context | current kyoku + BOS; no cross-kyoku Actor attention |
| state | 34 × 26 visible tile features + 38 global features |
| cross-attention | 1 Decision-Query layer, strict prefix K/V |
| policy | 37 tile candidates + 9 native non-tile actions, mask 46 |
| critic | 4-class final rank, low-weight BC, detach option |
| GRP | independent 24-way, not part of first BC loss |
| loss | masked BC; optional rank CE `0.1`; gradient accumulation full-window denominator |
| optimizer | AdamW, warmup + cosine, BF16 autocast on GPU intended for real data; FP32 mask/softmax/loss |
| seed | fixed per run; exact value recorded in the run manifest |

Controls kept available: `cross_attn_layers=0`, `head_type=linear46`,
`tile_embedding=identity`, `model_mode=state_only`, GRP MLP-vs-GRU.  No control
is declared a winner before matched-budget results.

## 2. Data gate (blocking)

* Reported historical corpus: 2,512,424 matches; this is **not** an eligible
  decision count.
* Required before training: match-level Census, valid POV decisions,
  token-length P50/P90/P99/max, phase/action distribution, source/date/rule
  counts, duplicate count, and Match-ID Train/Val/Test overlap = 0.
* Train/Validation split must be by complete match, never by POV.
* Diagnostic validation shards may not be fed to the trainer.
* Rights: 11 registry rows, **0 `ALLOWED`**, gate
  `BLOCKED_FOR_NEW_TRAINING`; private research is not a licence.
* No new Confirm seed window may be used for model selection.

## 3. Measured resource basis

### CPU (synthetic, 128 decisions × 256 tokens, 4 threads)

* forward: **1,638.0 decisions/s**, 419,328 tokens/s;
* train step: **413.4 decisions/s**, 105,825 tokens/s;
* peak RSS: 987.7 MiB.

### GPU (synthetic, same shape, FP32, RTX 5070 Ti)

| Metric | Before sync fix | After sync fix |
|---|---:|---:|
| forward decisions/s | 323.8 | **12,433.7** (median of 3) |
| forward tokens/s | 82,894 | **3,183,023** (median) |
| train decisions/s | 274.0 | **1,635.6** (median of 3) |
| train tokens/s | 70,135 | **418,713** (median) |
| inference peak VRAM | 144.6 MiB | **144.6 MiB** |
| train peak VRAM | 331.9 MiB | **331.9 MiB** |

The initial GPU profile re-ran the full `VNextModelBatch` contract on every
forward; the profiler showed host synchronisation dominating.  The fixed code
fails closed on a device mismatch instead of re-validating/copying.  Three
runs varied with WSL/GPU clocks (9,133–17,934 forward, 1,174–2,491 train
decisions/s); the table uses medians.  These are synthetic microbenchmarks,
**not** an epoch projection.

Synthetic CUDA BC/GRP/RL smoke: 100 BC steps finite, 7.23 steps/s, checkpoint
state round-trip unchanged; 50 GRP steps finite; behavior/AWR/PPO/PBRS/GAE
finite.  No convergence or strength claim.

## 4. Resource request / authorization

| Field | Value |
|---|---|
| GPU use | **AUTHORISED** by user on 2026-11-22 |
| synthetic profile | **RUN** |
| real-data BC | **NOT AUTHORISED** while rights gate is blocked |
| full B0 epoch | `NOT_RUN` |
| GPU arena league | `NOT_RUN` |
| Confirm pool | **NOT TOUCHED** |

After a rights-cleared, audited Train subset exists, the next request is:

1. a bounded real BC profile (200–500 optimizer steps), hard wall-clock cap,
   no checkpoint promotion;
2. measured VRAM/throughput and validation overhead;
3. written stop/abort rule; no Arena/Confirm run in that step.

A full epoch, broad GRP fit, GPU arena league, or Confirm use requires another
explicit approval.

## 5. Checkpoint / selection protocol

* Each candidate stores model config, contract hash, schema/action/normalizer
  hashes, source/split/provenance, counters, optimizer/scheduler/EMA/RNG.
* Selection uses only independent held-out validation CE/calibration; no
  training loss is a strength metric.
* A candidate checkpoint is rejected if its contract hash/model config or
  validation isolation differs.
* Resume must restore the candidate registry and produce the same global
  selection as a continuous run (PR-01 regression).
* No strength/dan/Tenhou/Mortal claim may be written until a paired,
  four-seat-rotated arena with a block bootstrap is actually run.

## 6. Explicit NOT_RUN

`full_census`, `real_train_epoch`, `real_bc_profile`, `real_grp_fit`,
`arena_smoke_strength`, `trend_select_confirm`, `strength_vs_m5d`,
`rl_native_league`, `confirm_pool_consumption`, `publication_rights`,
`multi_kan_native_end_to_end`.

## 7. Approval record

| Field | Value |
|---|---|
| user GPU authorization | **PROVIDED 2026-11-22** |
| GPU job started | synthetic profile/smoke only |
| restricted data read | **NO** |
| seed pool consumed | **NO** |
| checkpoint promoted | **NO** |
| real checkpoint produced | **NO** |
| next action | resolve rights/Census, then request a bounded real BC profile |
