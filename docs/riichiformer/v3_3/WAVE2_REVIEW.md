# RiichiFormer V3.3 Wave 2 — Lead review (draft)

Status: **IN PROGRESS**. This file is filled only after the four lane commits are
merged and an independent reviewer has run. A blank/`NOT_RUN` field is not a pass.

## Base

- Actual frozen W2 base: `e760226` (`rf3.2-w2`, tag object `c96b0eb`); `00e6576`
  is an ancestor. Full suite at base: 708 passed, 2 skipped, 281 subtests, 182.23 s.
- M5D-R0: `checkpoints/mortal.pth`, sha `0a44613c…`, 396k, pure CNN, deterministic
  argmax, CPU smoke only.
- No GPU job was started. GPU status before/after: 2,848 MiB baseline, 0 compute apps.

## Lane commits

| Lane | PR | Commit | Tests | Smoke | Verdict |
|---|---|---|---|---|---|
| A | PR-46A/B | TBD | TBD | TBD | TBD |
| B | PR-45/47 | TBD | TBD | TBD | TBD |
| C | PR-48 | TBD | TBD | TBD | TBD |
| D | PR-40R | TBD | TBD | TBD | TBD |

## Gate matrix

| Gate | Evidence | Status |
|---|---|---|
| G-V33-W2 | tag/base suite/reviewer P1 | PASS |
| G-V33-A0 | anchor identity, 4-seat, self-control | TBD |
| G-V33-D0 | validation disjointness, E2E | TBD |
| G-V33-B0 | protocol/tests only | TBD |
| G-V33-C0 | no confirm, no strength | TBD |

## Independent reviewer

Verdict: **TBD**. P0/P1/P2: TBD. Reviewer must not edit code.

## Actual CPU smoke measurements

| Run | Sets | Games | Wall | Result |
|---|---:|---:|---:|---|
| Lead scratch, Mortal self-control | 1 | 4 | 25.97 s | rankings `[1,1,1,1]`, ΔPT 0 |
| Lead scratch, 2k RF pilot vs M5D-R0 | 4 | 16 | 58.37 s | rankings `[0,1,2,13]`, weak pilot; not strength |
| Lane A self-control | TBD | TBD | TBD | TBD |
| Lane B validation corpus | TBD | TBD | TBD | TBD |

## Not run

- Full B0 epoch (6,191 batches × 2 seeds): **NOT_RUN** (no GPU authorization).
- Trend/Select/Confirm arena: **NOT_RUN**; no confirm pool created or consumed.
- Strength/dan/Tenhou-rank claim: **NOT_RUN**.
- Third-party licence: unchanged, 0 ALLOWED.

## PR-49+ recommendations

TBD after lane review. Candidate order: data quality/time weighting; candidate head;
risk/belief with zero-leak proof; rank/PT auxiliary head; memory; teacher distillation.
