# A8 training summary (small-text extract of the raw logs)

Raw logs (NOT staged, kept on disk): `runs/exp/A8/train.log` (single clean segment, 0–20,000);
`runs/exp/A8/train.log.killed-lock-race` keeps the 32-step aborted launch of 2026-09-19 12:22
(the lock-protocol race described in evidence T21 [F]). Checkpoint `runs/exp/A8/state.pth`
(~150 MB, not staged).

## Declared budget and outcome

- Declared in PREREG before launch: **20,000 optimizer steps**, batch 1024
  (`opt_step_every=1`, 20.48 M frames), from scratch, E3a conditions + the `phi`-zero hook.
- Reached: `state.pth` `steps: 20000`; final marker `total steps: 20,000` at 2026-09-19
  16:49:24, save mtime 16:49:24; watchdog stopped the process group at 16:50:27.

## Trajectory

| event | time | notes |
|---|---|---|
| launch (pid 21397, pgid 21397) | 2026-09-19 15:31:15 | `mortal params: 11,818,752` (= E3a); `event tokens: (1024, 147, 5) nonzero=1742164` |
| hook armed | 15:31:23 | `A8_PHI_ZERO_ASSERT first_call: zeroed_shape=(1024, 1024) max|phi_before_zero|=1.21777 -> returned tensor all-zero` |
| 2,000 … 18,000 | 15:39 … 16:39 | periodic `A8_PHI_ZERO_ASSERT calls=…` every 512 forwards |
| **20,000** | **16:49:24** | holdout 6.853789; watchdog kill 16:50:27 |

Wall/throughput: 79 min for 20,000 steps (**≈4.2 steps/s**, the same code path as E3a; slightly
faster than E3a's 3.19 steps/s because the run was solo on the GPU). The zeroing hook
(`torch.zeros_like` + all-zero assert per step) added no measurable slowdown (step time ≈
238 ms).

Holdout loss (steps 2k → 20k): 7.491114, 7.201467, 7.066765, 6.953716, 6.844241, 6.866154,
6.745021, 6.761817, 6.790801, 6.853789.

**Observation (recorded, not hidden):** A8's holdout loss drops 7.49 → 6.84 over the first
10,000 steps and then **plateaus (6.75–6.87) over the last 10,000**, with the CQL component
stuck at 1.18–1.21 and next-rank at 1.37–1.38 (E3a at 20 k: total 3.224, cql ≈0.57,
next_rank ≈0.84; A7: 3.450). The arm learns (the first 10 k steps move the loss, and the
event branch receives gradient), but the event-only function is far weaker on the holdout
objective than the CNN or the fusion arms. This is the pre-declared QA scenario
(`PREREG` §3: "if A8's loss is completely flat it is recorded as 'A8 untrainable'") in its
milder form: not completely flat, but capacity-limited — an empirical result to be read
together with the pool-A placement.

## Final checkpoint

`steps=20000`, `config.model.use_event_branch=true` (4 layers, d=128, 4 heads, FFN=512),
55 `event_branch.*` tensors, `event_branch.proj.weight` `|W|` sum 2115.5 / max 0.0956
(never zero after training: the branch is what learned). `optimizer`/`scheduler`/`scaler`
present (resume-compatible).

## Zeroing proof during *this* training

`runs/exp/A8/phi_zero_proof_train.json` (periodic snapshot, last written at call 19,968 =
39 × 512; the watchdog's SIGTERM means no atexit write, by design the hook rewrites the proof
every 512 forwards):
`hooked_brains=1, calls=19968, assert_all_zero_after_zeroing=true,
max_abs_phi_before_zero=2.3027, total_zeroed_elements=20,937,965,568, armed=true`.
The discarded CNN embedding was always non-zero (max magnitude 2.30), so the zeroing was
material, never vacuous; and because a branch-enabled Brain raises on `events=None`, the run
could not have fallen back to the branch-less path.
