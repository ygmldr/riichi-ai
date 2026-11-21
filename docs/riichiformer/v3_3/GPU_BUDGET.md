# V3.3 next-stage GPU budget (request, not authorization)

No GPU job was started in this wave. Numbers below are **estimates derived from the
actual 2,000-step B0 pilot** and must be replaced by measurements from the approved
smoke before a real run.

## Measured baseline (pilot, 2026-11-21)

| Item | Value |
|---|---|
| Shard | 500 games, 333,224 decisions, 5,437,312 shared tokens |
| Pilot steps | 2,000 per seed, batch 64 |
| Decisions/tokens per seed | 107,026 / 1,766,242 |
| Wall per seed | 299.81 s / 306.01 s |
| Full dry-run batches | 6,191 |
| Peak observed VRAM | 4,649 MiB (includes idle baseline; isolated delta not measured) |

## Estimated full B0 (two seeds)

- 6,191 / 2,000 × ~303 s ≈ **938 s (~15.6 min)** per seed of pure training.
- Two seeds ≈ **31 min**.
- Add validation every 250 steps (24 points), checkpoint I/O and EMA: budget
  **45–60 min per seed**, **1.5–2 GPU-hours for two seeds**.
- Hard stop: any NaN/Inf, illegal action, OOM, or thermal/power limit; checkpoint
  retention as preregistered.

## Arena budget

- CPU 4–20-set smoke is the only arena authorized now; its measured wall time will
  determine whether a 100-set Trend is CPU-feasible.
- A future GPU Trend (100–250 sets, 400–1,000 games) should be requested separately;
  do not assume the old 12k-game 33-minute figure applies to the RF candidate.
- Confirm (2,000–3,000 sets) is a **separate** request after a frozen winner; no confirm
  pool has been created or consumed.

## Requested next authorization (not granted)

1. One bounded GPU job: B0 two-seed validation/selection, max 2 GPU-hours, hard stop
   conditions, no arena in the same job.
2. After code/CI review: one bounded CPU or GPU Trend run on a new select pool, with
   measured games/s and a hard set cap.
3. No Confirm run until a single winner and analysis plan are frozen.
