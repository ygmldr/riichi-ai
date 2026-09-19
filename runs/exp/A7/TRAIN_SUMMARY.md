# A7 training summary (small-text extract of the raw logs)

Raw logs (NOT staged, kept on disk): `runs/exp/A7/train.log` (first segment, step 0–16,000)
and `runs/exp/A7/train.log.resume` (recovered segment, step 16,000–20,000). Checkpoint
`runs/exp/A7/state.pth` (130.7 MB, not staged).

## Declared budget and outcome

- Declared in PREREG before launch: **20,000 optimizer steps**, batch 1024
  (`opt_step_every=1`, 20.48 M frames), from scratch (config-only arm, `conv_channels=192`,
  `use_event_branch=false`).
- Reached: `state.pth` `steps: 20000`; final marker `total steps: 20,000` at 2026-09-19
  11:12:33, save mtime 11:12:33; watchdog stopped the process group at 11:13:37 (declared
  budget met).

## Trajectory

| event | time | notes |
|---|---|---|
| launch (pid 30171) | 2026-09-18 23:19:35 | `mortal params: 10,787,456`; first save 2,000 @ 23:28:07 |
| 4,000 / 6,000 / 8,000 / 10,000 | 23:35:58 / 23:44:17 / 23:52 / 00:02:20 | |
| 12,000 / 14,000 | 00:10 / 00:20:15 | |
| 16,000 | 00:28:24 | holdout 3.474910; **host reboot at ~00:37**, step-18,000 boundary not reached |
| resume (pid 955) | 2026-09-19 10:51:06 | `loaded: 2026-09-19 00:28:24` (own state.pth; optimizer/scheduler/scaler from checkpoint) |
| 18,000 | 11:01:54 | holdout 3.442542 |
| **20,000** | **11:12:33** | holdout 3.450485; watchdog kill 11:13:37 |

Holdout loss (steps 2k → 20k): 4.106447, 3.895275, 3.763070, 3.621482, 3.612935, 3.556684,
3.458721, 3.474910, 3.442542, 3.450485.

Wall/throughput: segment 1 = 4,129 s for 16,000 steps (**3.87 steps/s**); resume = 1,287 s for
4,000 steps (**3.11 steps/s**, including ~10 s model/checkpoint load); total training wall
~90 min across the reboot. The resume PREREG addendum (§7) records the interruption and the
non-continued DataLoader shuffle order (same caveat as E2 §6 / E3a §8).

Final checkpoint: `steps=20000`, `config.resnet.conv_channels=192`,
`config.model.use_event_branch=false`, 0 `event_branch.*` tensors (branch never instantiated),
optimizer/scheduler/scaler present (resume-compatible).

## Pool-A evaluation (12,000 games)

- Launch 2026-09-19 11:40:05 (pid 1150), arena `dumping game logs` 12:12:52, final rankings
  12:16 → **wall ≈ 36 min**; 12,000 new files, all `(seed ∈ [10000,13000), key
  0x5dfaa4cef265cd7)`, none in `runs/1v3/`.
- `challenger rankings: [2928 2930 2910 3232]` → avg_rank **2.5372**, pt **−3.4125**.
- `report.csv`: sets=3000 games=12000 incomplete_sets=0 ignored_files=0;
  avg_rank 95 % CI [2.5191, 2.5552]; pt 95 % CI [−4.7369, −2.0474].
- Earlier attempt (11:15:43) was killed by the WSL restart (~11:28) before the arena dumped
  anything; its log is kept as `eval.log.killed-wsl-restart`. The relaunch is a full replicate
  (arena determinism: identical seed window ⇒ byte-identical logs).
