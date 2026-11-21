# E3a training summary (small-text extract of the raw logs)

Raw logs (NOT staged, kept on disk): `runs/exp/E3a/train.log` (first segment, step 0–12,000)
and `runs/exp/E3a/train.log.resume` (resumed segment, step 12,000–20,000). Checkpoint
`runs/exp/E3a/state.pth` (137 MB, not staged).

## Declared budget and outcome

- Declared in PREREG before launch: **20,000 optimizer steps**, batch 1024
  (`opt_step_every=1`, 20.48M samples), from scratch; E2/A7/A8/A9 must reuse 20,000.
- Reached: `state.pth` `steps: 20000`; final marker `total steps: 20,000` at 2026-09-18 02:00:06,
  save mtime 02:00:07; watchdog stopped the process at 02:01:34 (declared budget met).

## Trajectory

| event | time | notes |
|---|---|---|
| launch (pid 91707) | 2026-09-17 23:22:35 | first save 2,000 @ 23:35:21 |
| 4,000 / 6,000 / 8,000 / 10,000 | 23:50 / — / — / 01:00:24 | external kill during agent-session interruption at ~step 13,027 |
| last save before kill | 01:09:31 | step 12,000 |
| resume (pid 1965) | 2026-09-18 01:18:15 | loaded checkpoint timestamp 01:09:31 (own state; PREREG §8 addendum) |
| 14,000 / 16,000 / 18,000 | 01:28:51 / 01:39:19 / 01:49:25 | |
| **20,000** | **02:00:06** | watchdog kill 02:01:34 |

Holdout loss (steps 2k → 20k): 4.004824, 3.793341, 3.618526, 3.643839, 3.458495, 3.465353,
3.354935, 3.374656, 3.308043, 3.224427.

Wall/throughput: segment 1 = 6,416 s for 12,000 steps (1.87 steps/s avg; early cycles
1.8–2.2, later 3.1–3.8); resume = 2,599 s for 8,000 steps (3.08 steps/s); total ~2h39m
excluding the 9-min relaunch gap.

Resources: training RSS 5.0 GB; MemAvailable ≥ 6 GB throughout (incl. the concurrent todo-17
E2 training from ~20:34 on 2026-09-18); GPU total used 15.2–15.5 / 16.3 GB on the shared box
of which 3.7–6 GB is a pre-existing non-process context (arm delta ≈ 9 GB). No OOM, no
fallback: the synthetic step benchmark (same code, batch 1024, T=143) measured CNN 133 ms vs
E3a 220 ms, i.e. the branch costs 1.65× raw (< the 2× fallback trigger). `enable_compile=false`
was declared in PREREG (the event length T is dynamic and torch.compile would recompile per
length); E2 must match.

Final checkpoint: `steps=20000`, `config.model.use_event_branch=true`, 55 `event_branch.*`
tensors, branch `|W|` sum 115,890.88, `proj` `|W|` max 0.09306,
sha256 `034ace1d94a59db4a3492674765eb6640786368545f7445ed5a2438e5f857056`.
