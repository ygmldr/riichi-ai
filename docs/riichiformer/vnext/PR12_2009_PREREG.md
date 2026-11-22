# VNEXT-BC-2009-1 — Full-Year 2009 BC Run (operator-authorized)

**Status**: `COMPLETED` (71,387 steps in 9,121.0 s; final checkpoint 71,387)
**Created**: 2026-11-22
**Owner**: single Lead
**Data**: local Tenhou-derived 2009, operator-authorized local research
**External rights**: unchanged `RESTRICTED/UNKNOWN`; this is not an upstream licence

## Purpose

A bounded, reproducible **full-year** BC run after the 1,000-game pilot.  This is
the largest complete year that fits the measured CPU-bound data pipeline before
the 23:00 review.  It is not the 2.51M-match corpus; that one-epoch estimate is
~47 days at the measured rate (`PR12_REAL_PILOT.md`).

## Data

| Item | Value |
|---|---|
| train | all 6,897 matches in `datasets/extracted/2009` |
| train shard | `artifacts/riichiformer/h02/year2009_shards/train.rfshard` |
| train decisions | **4,568,777** (4,545,919 supervised) |
| validation | 500 matches from 2023, val-only filtered manifest |
| validation shard | `artifacts/riichiformer/h02/pilot_shards/val.rfshard` |
| split | `rf-split/2`, protocol L |
| policy | `configs/riichiformer/local_research_policy.yaml`, hash `f0d472ee...` |

Train and validation match IDs are disjoint by year.  No Confirm pool is used.

## Training

| Field | Value |
|---|---|
| model | frozen VNext default, 6,538,716 params |
| loss | masked BC + rank CE 0.1, rank_base 1 |
| optimizer | AdamW, lr 1e-4, warmup 1,000 + cosine to 10%, grad clip 1.0 |
| batch | 64 decisions; prefixes > 4,096 tokens are skipped and counted |
| steps | 71,387 (`floor(4,568,777/64)`) |
| validation | every 5,000 steps, 50 batches |
| checkpoints | every 5,000 steps |
| wall cap | 5 hours |
| long-prefix skip | 460 decisions (0.01%) over 4,096; reported, never silently truncated |
| seed | 20261122 |
| device | CUDA FP32 |

## Acceptance / reporting

* finite losses, no NaN/Inf; wall-cap stops are reported as `STOPPED`, not
  `COMPLETED`;
* validation CE recorded at the fixed schedule; 500-batch re-eval: 5,000-step
  1.5019 / 1.1787; 10,000-step 1.4895 / 1.1472; 15,000-step 1.4720 /
  1.1363; 20,000-step 1.4551 / 1.1400; 25,000-step 1.4426 / 1.1363; 30,000-step 1.4311 / 1.1434; 35,000-step 1.4182 / 1.1360; 40,000-step 1.4089 / 1.1231; 45,000-step 1.3912 / 1.1420; 50,000-step 1.3766 / 1.1345; 55,000-step 1.3675 / 1.1295; 60,000-step 1.3609 / 1.1244; 65,000-step 50-batch 1.3515 / 1.2600; 70,000-step 1.3462 / 1.2503;
* checkpoint provenance carries source/split/canonical/policy hashes;
* no strength, dan, Arena or Confirm claim;
* `max_prefix_tokens=4096`; decisions above the positional bound are skipped
  and reported, not truncated or silently encoded;
* if the wall cap triggers, resume from the last checkpoint; resume skips the
  exact `batches_seen`, requires the same train/val shard hashes and strict
  optimizer/RNG state, and never replays the prefix silently.

## Stop rule

Stop on non-finite loss, shard/policy hash mismatch, or wall cap.  Do not switch
to the Confirm pool and do not promote a checkpoint based on training loss.

## Result (completed 09:37)

- steps: **71,387**
- wall: **9,121.0 s** (2 h 32 min)
- `exhausted=true`, `stopped_reason=target_reached`
- `long_prefix_skipped=460` / max prefix 4,096
- because those 460 decisions are absent from batches, the shard exhausted at
  about 71,379 batches; the trainer restarted the adapter and replayed ~450
  decisions (0.01%) to reach the 71,387-step target (`exhausted=true`)
- train loss first/last: 2.4687 / 2.5181; mean 1.5537 (last batch is noisy)
- 50-batch last validation (70k): BC CE 1.3462, rank CE 1.2503
- **final 500-batch validation (32,000 decisions): BC CE 1.3500, rank CE
  1.1276, total 1.4628**
- final checkpoint: `candidates/step_071387.pt`, SHA
  `6d16c4bc6dd97f744feab05d8cfa2d77e4646dde3a15358b4ff7cf85e6ac391c`
- final native smoke: 16 games, 2,827 decisions, 0 illegal, 0 fallback,
  `kan_selections=0`, `strength_claim=NONE`
- logs: `logs/VNEXT_VALIDATE_2009_FINAL_500BATCH.log`,
  `logs/VNEXT_NATIVE_2009_FINAL_16G.log`

This is a full-year BC loss result, not a strength/dan/Arena result.
