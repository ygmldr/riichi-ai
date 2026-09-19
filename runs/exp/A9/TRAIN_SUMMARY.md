# A9 training summary (small-text extract of the raw log)

Raw log kept on disk (not staged): `runs/exp/A9/train.log` (run 2, complete);
run-1 fragment from before the 2026-09-19 WSL restart:
`runs/exp/A9/train.log.killed-wslrestart-20260919_1136`. Checkpoint
`runs/exp/A9/state.pth` (137 MB, not staged).

## Declared budget and outcome

- Declared in PREREG §4 before launch: **20,000 optimizer steps**, batch 1024
  (`opt_step_every=1`, 20.48 M samples), from scratch, matching E2/E3a/A7/A8.
- Reached: `state.pth` `steps: 20000` (`saved_at` 2026-09-19 14:32:24);
  watchdog confirmed and sent SIGTERM at 14:32:38, process dead by 14:32:43.
- Wall clock (run 2): step 0 logged 12:22:38, step 20,000 at 14:32:24 → **7,786 s
  ≈ 2.57 steps/s** (the same bounded loop as E3a/E2, with the A9 collate overhead).

## Trajectory (holdout loss at each 2 k save)

| steps | time | holdout_loss |
|---|---|---|
| 2,000 | 12:34:08 | 4.198254 |
| 4,000 | 12:44:45 | 3.823162 |
| 6,000 | 12:56:34 | 3.699666 |
| 8,000 | 13:10:38 | 3.500787 |
| 10,000 | 13:24:26 | 3.535427 |
| 12,000 | 13:38:03 | 3.568966 |
| 14,000 | 13:51:36 | 3.305532 |
| 16,000 | 14:05:09 | 3.334394 |
| 18,000 | 14:18:48 | 3.349849 |
| **20,000** | **14:32:24** | **3.319372** |

Finite throughout; no NaN/Inf and no fallback. Reference: E3a's holdout at 20 k was
3.224427 under ordered inputs (not a paired comparison — A9 trains on the permuted
distribution).

## In-pipeline input-count self-check (asserted on every collated batch)

Counters accumulated in the workers over the whole run (3 workers, final per-worker
values):

- collate batches: **19,950**; obs samples: **20,428,800**
- river blocks checked: **159,600** (8 per batch), changed: **159,600**, uniform: **0**
  ⇒ every block permutation measurably changed the batch; the arm cannot be a no-op
  duplicate of E3a, and each block's slot multiset + per-element counts were asserted
  equal before/after.
- event samples checked: **20,428,800**, changed: **20,428,800** ⇒ every checked
  token stream was reordered while its per-sample row multiset was asserted equal.

## Checkpoint facts

`steps=20000`, `config.model.use_event_branch=True`, `resnet.conv_channels=192`,
55 `event_branch.*` tensors, branch `|W|` sum 117,186.375 (branch active, not the
zero-init identity). Permutation fingerprint (from the launch banner, matches PVALID):
`bf16f51cbea1b8a52f02351d29f3843814090ff97b2df72ee03caf60c5a88027`.

## Conditions (identity with E3a/E2)

`runs/exp/A9/train.toml` is `runs/exp/E3a/train.toml` with only the arm path prefix
changed (`diff` after prefix normalisation is empty): batch 1024, `opt_step_every=1`,
`save_every=2000`, `test_every=100000`, `num_workers=3`, `enable_compile=false`,
`enable_amp=true`, globs 2009–2022 + 2023 holdout, `checkpoints/file_index.pth`
read-only, same optimizer/scheduler.

## Safety / invariants

- All control/log paths under `runs/exp/A9/` (asserted by the generator and again by the
  launcher before `train.main()`); live checkpoints never written: `mortal.pth`
  2026-09-17 02:03:09, `best.pth` 2026-09-16 02:59:45 (unchanged after the run).
- `runs/1v3/*.json.gz` = **2004** before and after (game logs go to `runs/exp/A9/games/`).
