# VNEXT-BC-2009-1 — Full-Year 2009 BC Run (operator-authorized)

**Status**: `BUILDING_SHARD` (980/6,897 at 04:11; watcher will launch automatically)
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
| train decisions | expected ~4.55M (measured pilot mean 661.6/match) |
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
| batch | 64 decisions |
| steps | one epoch, `floor(decisions/64)` (about 71k) |
| validation | every 5,000 steps, 50 batches |
| checkpoints | every 5,000 steps |
| wall cap | 5 hours |
| seed | 20261122 |
| device | CUDA FP32 |

## Acceptance / reporting

* finite losses, no NaN/Inf; wall-cap stops are reported as `STOPPED`, not
  `COMPLETED`;
* validation CE recorded at the fixed schedule;
* checkpoint provenance carries source/split/canonical/policy hashes;
* no strength, dan, Arena or Confirm claim;
* if the wall cap triggers, resume from the last checkpoint; resume skips the
  exact `batches_seen`, requires the same train/val shard hashes and strict
  optimizer/RNG state, and never replays the prefix silently.

## Stop rule

Stop on non-finite loss, shard/policy hash mismatch, or wall cap.  Do not switch
to the Confirm pool and do not promote a checkpoint based on training loss.
