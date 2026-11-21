# PR-11 — VNext BC Preregistration (DRAFT, NOT AUTHORIZED)

**Status**: `DRAFT_NO_GPU_AUTHORIZATION`  
**Experiment ID**: `VNEXT-BC-PILOT-1`  
**Owner**: single Lead  
**Base**: `rf/vnext-core` at `8a915d1` + VNext PRs  
**Frozen contract hash**: `3c2d398c9870ede0dc6ffc8557e588dad42c9823508a62759f5b6a3fd1ad5325`  
**Model architecture hash**: `a0ca7a882ee6aa25b9c4afbb5efba246b6ea9a4661fa113666f1d7c3d85cab9d`  
**Model parameters**: 6,462,940

This is a **plan**, not a passed gate.  No GPU training has been started.  The
full 2.51M-match Census and the real Train/Validation split audit are still
`NOT_RUN`; the numbers below distinguish measured CPU facts from the reported
corpus size.

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
| optimizer | AdamW, warmup + cosine, BF16 autocast on GPU, FP32 mask/softmax/loss |
| seed | fixed per run; exact value recorded in the run manifest |

Controls kept available: `cross_attn_layers=0`, `head_type=linear46`,
`tile_embedding=identity`, `model_mode=state_only`, GRP MLP-vs-GRU.  No control
is declared a winner before matched-budget results.

## 2. Data gate (currently blocking)

* Reported historical corpus: 2,512,424 matches; this is **not** an eligible
  decision count.
* Required before training: match-level Census, valid POV decisions, token-length
  P50/P90/P99/max, phase/action distribution, source/date/rule counts, duplicate
  count, and Match-ID Train/Val/Test overlap = 0.
* Train/Validation split must be by complete match, never by POV.
* Diagnostic validation shards may not be fed to the trainer.
* Rights: 11 registry rows, **0 `ALLOWED`**, gate
  `BLOCKED_FOR_NEW_TRAINING`; private research is not a licence.
* No new Confirm seed window may be used for model selection.

## 3. Measured CPU basis (not a GPU projection)

From `scripts/rf_bench_vnext.py`, 32 decisions × 128 tokens, 4 CPU threads:

| Mode | Decisions/s | Event tokens/s |
|---|---:|---:|
| forward-only | 1,567 | 200,616 |
| one training step | 307 | 39,339 |

Peak RSS: 804.6 MiB.  Raw: `logs/vnext/profile_cpu.json`.

These numbers **must not** be extrapolated into an 8-hour guarantee or a 2.51M
epoch promise.  They only show that the CPU reference path runs and that the
real GPU profile is the next measurement, not a guessed training budget.

## 4. Requested resource authorization (not granted)

I request authorization only for a **bounded profiling/pilot**, not a full epoch:

1. a GPU short profile of 200–500 optimizer steps on a small, pre-audited Train
   subset, with a hard wall-clock cap (e.g. 1 hour) and no checkpoint promoted;
2. a measured VRAM/throughput record (decisions/s, tokens/s, peak VRAM,
   validation overhead);
3. a written stop/abort rule; no Arena/Confirm run in this step.

A full B0-BC epoch, a broad GRP fit, a GPU arena league, or any Confirm use
requires a **separate explicit approval** after the short profile is reported.

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

`full_census`, `real_train_epoch`, `gpu_profile`, `arena_smoke`,
`trend_select_confirm`, `strength_vs_m5d`, `rl_training`, `grp_fit`,
`confirm_pool_consumption`, and `publication_rights`.

## 7. Approval record

| Field | Value |
|---|---|
| user authorization | **NOT PROVIDED** |
| GPU job started | **NO** |
| seed pool consumed | **NO** |
| checkpoint promoted | **NO** |
| next action | wait for explicit bounded-profile approval |
