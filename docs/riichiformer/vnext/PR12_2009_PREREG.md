# VNEXT-BC-2009-1 — Full-Year 2009 BC Run (operator-authorized)

**Status**: `TRAINING` (shard complete 07:00; 6,897 games / 4,568,777 decisions; 71,387 steps)
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
  1.1363; 20,000-step 1.4551 / 1.1400; 25,000-step 1.4426 / 1.1363; 30,000-step 1.4311 / 1.1434; 35,000-step 1.4182 / 1.1360; 40,000-step 1.4089 / 1.1231; 45,000-step 1.3912 / 1.1420; 50,000-step 1.3766 / 1.1345; 55,000-step 1.3675 / 1.1295; 60,000-step 50-batch 1.3567 / 1.2551;
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
