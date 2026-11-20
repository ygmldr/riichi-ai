# Player quality audit (PR-06)

- generated: 2026-11-20T18:43:23Z
- manifest: `/home/ygsldr/riichi-ai/artifacts/riichiformer/split/split_manifest_L.parquet` (sha256 `5e291ad09aa35ab6a392c24c25a4e5b1be97289aad4c6c938253215755f3f27b`)
- split filter: `train`; sample: 100000 games (seed 20261121)
- player-linked: **True**
- prior mean rank=2.5, k=100, min_games=50

## Coverage

- games parsed: 100000 / 100000 (errors 0: {})
- seat rows: 400000; unique players (HMAC): 10511
- insufficient history (<50 prior games): 227578 (0.5689 of rows)
- top25 seat rows: 43105 (threshold q=2.4483)

## q distribution (eligible rows only)

| p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|
| 2.4142 | 2.4483 | 2.4845 | 2.5225 | 2.5571 |

## Reliability (players' last appearance: prior q vs that game's rank)

| bucket | players | mean rank | win rate |
|---|---:|---:|---:|
| Q1_lowest_q | 540 | 2.6278 | 0.2296 |
| Q2 | 541 | 2.7209 | 0.1996 |
| Q3 | 541 | 2.6433 | 0.2181 |
| Q4_highest_q | 541 | 2.6895 | 0.22 |

## Verdict (100,000-game train sample, seed 20261121)

- 400,000 seat rows across 10,511 HMAC players; 227,578 rows are `insufficient_history`
  (56.89%) — this is an artefact of sampling only 100k of 2.5M games, not proof that players
  have short real histories.
- 2,163 players have ≥50 prior games.  Using each player's last appearance (q from prior
  games, rank from that game) as a time-forward check:

| bucket | n | mean rank | 95% CI | win rate |
|---|---:|---:|---|---:|
| Q1 (lowest q) | 540 | 2.6278 | [2.5315, 2.7240] | 22.96% |
| Q2 | 541 | 2.7209 | [2.6257, 2.8161] | 19.96% |
| Q3 | 541 | 2.6433 | [2.5481, 2.7384] | 21.81% |
| Q4 (highest q) | 541 | 2.6895 | [2.5915, 2.7874] | 22.00% |

- Q1 − Q4 mean-rank difference is only **−0.0617**, CIs overlap heavily, and the ordering is
  non-monotonic (Q2 is worst). **Verdict: `INCONCLUSIVE` — this sample does not demonstrate
  that q predicts future rank.**
- `top25_seat` is therefore a **candidate reweighting flag only**. PR-18 must pre-register a
  paired experiment; no report may claim “TOP25 players are stronger” from these numbers.
  `PASS_CORRECTNESS` for this tooling does **not** mean the quality score is validated.
- Eligible q distribution: p10 2.4142 / p25 2.4483 / p50 2.4845 / p75 2.5225 / p90 2.5571;
  top-25% threshold = **2.448276**.

## Privacy / metadata

- Names are read only in memory; artefacts contain **HMAC-SHA256** identifiers only.
- `RF_PLAYER_HMAC_KEY` is supplied via environment and is never written or logged.
- No local annual player database / dan / R / rating was found; those fields are `null`/unknown.
- No AI-account inference is made; absence of evidence is `unknown`, not `human`.

## Limitations

- sample-based: 100000 train games, not the full 2.5 M corpus;
  history is limited to games present in this sample;
- q is a shrunk mean rank from prior **train** games only; it is a candidate feature,
  not a certified player strength; `target_rank_for_eval` is never used to compute q;
- labels are excluded from history by time (`temporal=true`) or marked `temporal=false`;
- top25 is a candidate reweighting flag; the other three seats in a game are not marked.
