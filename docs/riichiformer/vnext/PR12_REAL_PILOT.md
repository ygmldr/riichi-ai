# PR-12 Real-Data Pilot (Operator-Authorized Local Research)

**Status**: `PILOT_PASS`; full 2009-year build in progress
**Updated**: 2026-11-22 03:55 Asia/Shanghai
**Branch**: `rf/vnext-core`
**Data policy**: operator-authorized local research on the existing Tenhou-derived
copy.  This is **not** a claim of upstream permission, and the external
`SOURCE_RIGHTS_GATE` facts were not rewritten.

## 1. What was built

| Artifact | Value |
|---|---|
| canonical sample | 4,000 games (2,000 from 2009, 2,000 from 2023) |
| split manifest | `rf-split/2`, protocol L, 2,000 train / 2,000 val |
| train shard | 1,000 games, **661,627 decisions**, 917 MB, `55ac5e03...` |
| val shard | 500 games, **326,798 decisions**, 453 MB, `ac5c68b8...` |
| local policy hash | `f0d472ee...` (`operator_private_research`) |

The val shard is a val-only filtered manifest, so it contains no train rows.
The trainer checks the shard policy hash, purpose, and split before yielding a
batch.

## 2. Trainer

`scripts/rf_train_vnext.py` is the missing real-data entry point:

- opens authenticated `rf-shard/3` train/validation shards;
- groups decisions with `VNextBatchAdapter`;
- runs masked BC + Critic rank CE (`rank_base=1` for Tenhou ranks 1..4);
- validates on a held-out shard;
- saves VNext checkpoints with source/split/canonical/policy provenance;
- supports resume, wall-clock cap, bounded steps and dry-run.

Two loader fixes were required by the pilot:

1. `ShardReader.iter_decisions` now decodes each POV event stream **once per
   game** instead of once per decision.  The old path was O(decisions × events)
   and dominated CPU time.
2. `VNextModelBatch.to(device)` no longer re-runs `__post_init__`; moving tensors
   cannot change the already-validated values.  The old path caused host
   synchronisation on every batch.

## 2b. Pilot Census (read-only, SQL)

| Metric | train 1,000 | val 500 |
|---|---:|---:|
| games | 1,000 | 500 |
| decisions | 661,627 | 326,798 |
| supervised | 658,379 | 325,291 |
| SELF_TURN / CALL_RESPONSE / KAN_SELECTION | 526,595 / 134,263 / 769 | 259,851 / 66,511 / 436 |
| rank 1/2/3/4 | 166,431 / 165,613 / 164,883 / 164,700 | 82,132 / 81,558 / 81,573 / 81,535 |
| prefix-token P50/P90/P99/max | 548 / 1,091 / 1,547 / 2,491 | 545 / 1,086 / 1,497 / 2,180 |
| policy hash | `f0d472ee...` | `f0d472ee...` |

Raw: `logs/VNEXT_CENSUS_TRAIN1000.json`, `logs/VNEXT_CENSUS_VAL500.json`.
The max prefix (2,491) is below the model's 4,096 max positions; no truncation
is needed for this pilot.  The 2,532 non-supervised/None action rows are kept
as censored targets, not invented labels.

## 2c. Native smoke with the trained checkpoint

The 5,000-step checkpoint (`9149b84d...`) was loaded through
`load_vnext_checkpoint` and played four native `OneVsThree` games against the
local M5D anchor:

- 4 games, 765 decisions, 0 illegal, 0 fallback;
- `kan_selections=0` (multi-kan end-to-end still `NOT_RUN`);
- synthetic fixture: false; strength claim: **NONE**.

This proves the trained checkpoint can drive the native bridge; it is not a
win-rate/dan/strength measurement.  A 4-set/16-game extension exposed a real
lifecycle bug: the native driver did not always forward `start_game`, so a
second set reused the prior POV adapter when the new kyoku id matched.  The
engine now resets the optional backend exactly once per game from `_react` as
well; the 16-game rerun passed with 2,792 decisions, 0 illegal, 0 fallback.

## 3. Pilot result

Command shape: 1,000 optimizer steps, batch 64, 1000-game train shard,
500-game held-out val shard, CUDA FP32.

| Step | val BC CE | val rank CE | val total |
|---:|---:|---:|---:|
| 200 | 1.5983 | 1.3769 | 1.7360 |
| 400 | 1.5765 | 1.3778 | 1.7143 |
| 600 | 1.5465 | 1.3675 | 1.6832 |
| 800 | 1.5428 | 1.3613 | 1.6789 |
| 1000 | **1.5285** | **1.3454** | **1.6631** |

- training wall: 158.9 s for 1,000 steps (64,000 decisions);
- observed ~**403 decisions/s** including data loading, validation excluded;
- train loss first/last: 2.2397 / 1.9160;
- checkpoint: `runs/vnext_real_pilot1000/candidates/step_001000.pt`;
- no strength, dan, or Arena claim.

This is a **loss** pilot, not a quality result.  The held-out CE decrease proves
the data/trainer/checkpoint loop is learning something, not that the policy is
strong.

A longer 5,000-step run on the same 1,000/500 shards completed in 657.5 s:

- val BC CE **1.5893 → 1.4719**;
- val rank CE **1.3682 → 1.2956**;
- val total **1.7261 → 1.6015**;
- train loss first/last 2.2397 / 1.5313;
- checkpoint `runs/vnext_real_pilot1000_long/candidates/step_005000.pt`.

A separate 500-batch / 32,000-decision validation of that checkpoint (not just
the 50-batch training-time sample) gives **BC CE 1.4934**, **rank CE 1.1744**,
total 1.6108.  The 50-batch result (1.4719 / 1.2956) was a noisier sample.

A full-shard run (10,500 steps, target above the shard's partial-batch count)
completed in 1,310.9 s with `exhausted=true` and a restart of the shard; the
last validation at step 10,000 was BC CE **1.45696**, rank CE **1.27958**,
total 1.58492.  Train loss first/last was 2.2397 / 1.5085.  Checkpoints:
`step_005000.pt`, `step_010000.pt`, `step_010500.pt`.  This is still a loss
result, not a strength claim.

Still a loss/optimisation result only; no strength claim.

## 4. Full-corpus feasibility (measured, not projected)

The local corpus is 2,512,424 matches.  The pilot shard has ~662 decisions per
match (4 POVs).  A full one-epoch pass is therefore ~1.66B decisions.

At the measured ~403 decisions/s (data loader + model), one full epoch would
take ~47 days.  Building a full shard at the observed ~0.7 games/s would take
~41 days.  **A 2.51M-match full epoch does not fit the review window.**  This is
a data-pipeline throughput limit, not a GPU limit: the synthetic GPU profile is
~1,600 train decisions/s, but real event/decision decoding is CPU-bound.

The feasible full-year run is 2009:

- 6,897 matches, ~4.55M decisions;
- shard build ~2.5–3 h;
- one BC epoch ~3 h at the measured rate.

That build is running now.  The 2.51M corpus needs parallel shard building or a
pre-encoded feature cache before it can be trained as one epoch; neither is
claimed here.

## 5. Evidence

- `runs/vnext_real_pilot1000/metrics.json`
- `runs/vnext_real_pilot1000/validation.jsonl`
- `runs/vnext_real_pilot1000/candidates/step_001000.pt(.json)`
- `runs/vnext_pilot_train_shard.log`, `runs/vnext_pilot_val_shard.log`
- `docs/riichiformer/vnext/logs/VNEXT_REAL_TRAIN_SUITE.log`

## 6. NOT_RUN / caveats

- Full 2.51M-match one epoch: infeasible in this review window at measured
  throughput; not attempted silently.
- Confirm pool: untouched.
- Arena / strength: no paired run, no claim.
- The external source-rights registry remains `RESTRICTED/UNKNOWN`; the local
  policy is an operator routing choice, not a licence.
