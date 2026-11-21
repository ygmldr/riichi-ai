# PR-41 / PR-42 — local self-play rollout, provenance, opponent registry, duplicate arena

Date: 2026-11-21 (Asia/Shanghai). Task: **PR-41 (own rollout) + PR-42 (OpponentRegistry /
DuplicateArena)**. Worktree `worktrees/v32-selfplay`, branch `rf/v32-selfplay`,
`BASE_SHA=cbf166a356f83c739b31f0b5a0b241b954ef9e0a`.

This lane adds local, offline, CPU-only self-play data generation and the evaluation
plumbing needed to compare local opponents. It does **not** claim Tenhou parity, an
official dan/rank, a Mortal comparison, or any strength result. No bot was run on an
online platform, no external log was downloaded, and no third-party checkpoint was
loaded.

## 1. Rulesets (never conflated)

| `ruleset_id` | Engine | Meaning | `tenhou_equivalent` |
|---|---|---|---|
| `libriichi_tenhou_approx/1` | `libriichi.arena.OneVsThree.py_vs_py` (pinned Mortal submodule `0cff2b5`) | Real 4-player hanchan with calls, kan, wins and Tenhou-like scoring. Still *Tenhou-approximate*: PR-40 deviations R02/R25/R26/R29 are tracked, R29 timing is unverified. | `false` |
| `synthetic_abstract_hanchan/1` | none (procedural) | Seeded draw/discard loop, tenpai/noten payments and riichi sticks only. No calls, no wins, no hand scoring, no aka, no wall model. Used only when the engine is unavailable or for plumbing tests. | `false` |

Both summaries and every per-game ledger row carry `ruleset_id`, `rules_engine`,
`tenhou_equivalent: false`, `strength_claim: "none"` and `parity_status`.

## 2. What was built

```
riichiformer/selfplay/
  tiles.py       tile text/kind/aka helpers over the frozen 34-kind contract
  shanten.py     pure-python standard/chiitoi/kokushi shanten, win check, waits,
                 exact ukeire (3n+1 and 3n+2 forms), open-hand aware
  view.py        SeatView: POV-safe observation (own hand + public events only)
  bots.py        LegalRandomBot, SimpleShantenUkeireBot, DefenseBot (+ factories)
  provenance.py  ruleset labels, generator content hash, source ledger, output guards
  seeds.py       disjoint train/select/confirm seed pools with JSON ledger
  rollout.py     libriichi bridge, procedural fallback, per-game ledger, audit
riichiformer/eval/
  opponents.py   OpponentRegistry: whitelisted, labelled opponents
  arena.py       DuplicateArena (4-seat rotation, seed-block bootstrap), GameAudit
scripts/rf_selfplay.py, scripts/rf_arena.py
tests/rf_test_selfplay.py, tests/rf_test_opponents.py
```

## 3. Bots (no weights, no checkpoint, no network)

| Bot | Behaviour |
|---|---|
| `legal-random` | Uniform over the native 46-action mask; calls, kan, riichi, kyuushu and wins when the mask allows. Weak control opponent and a broad legality exercise. |
| `shanten-ukeire` | Closed-hand shanten reduction over every legal discard (exact shanten of the resulting 13-tile hand). Ties at tenpai use the exact number of remaining winning tiles; ties above tenpai use a documented shape heuristic. Declares riichi when the mask allows and the discard keeps tenpai. |
| `defense` | `shanten-ukeire` plus a threat check: against a declared riichi it pushes only when already tenpai, otherwise plays the least dangerous legal tile (genbutsu first, then honours/terminals, suji-reduced middle tiles). Does not declare riichi while a threat is live. |

Every bot returns a 46-action index and the base class rejects any index outside the
view's mask (`IllegalBotAction`), plus the phase check from the frozen schema. The
rollout bridge additionally re-validates every translated MJAI reaction with the
engine's own `PlayerState.validate_reaction`; illegal/rejected/fallback counts are
recorded per POV and fail the run when non-zero.

## 4. POV safety

`SeatView` contains only: the seat's own concealed hand/akas/melds, the public rivers,
calls, riichi flags, dora indicators, scores, rounds and the frozen 46-mask. The
libriichi bridge parses `ctx.log` through an explicit whitelist
(`start_kyoku.scores/dora_marker`, `dahai`, `chi/pon/daiminkan/ankan/kakan`,
`reach`, `dora`, `tsumo`, `hora.actor/target`, `ryukyoku`) and never reads
`start_kyoku.tehais` or `hora.ura_markers`; a dedicated test feeds a log containing
both and asserts the view never sees opponent haipai or ura dora. Each bot instance
receives exactly one seat view; the bridge never hands a bot another seat's state.

Raw engine game logs (`<out>/games/*.json.gz`) are **full-information** (all four
haipai and ura markers) and are labelled as such in `summary.json`
(`outputs.game_log_note`); they are provenance artifacts, not POV-safe training rows.

## 5. Real local rules engine

`libriichi.arena.OneVsThree.py_vs_py` is driven with `engine_type="mjai-log"` seat
engines (the pinned submodule's public Python bridge). Each wall seed `(seed, key)` is
played **four times with rotated seats**: rotation `k` puts lineup position `p` at
absolute seat `(k + p) % 4`, so each lineup position occupies every seat once and the
challenger slot walks through seats 0..3 exactly like the pinned 1v3 arena.

The 46-action mask comes from `PlayerState.encode_obs(1, False)[1]` — the engine's own
mask, which is the only source that also encodes riichi-only discards and kuikae
(swap-call) restrictions. An independent derivation from `ActionCandidate` + hand +
akas is computed on every decision as a cross-check: it must explain **every** native
action (observed 0 undercounts). Actions are translated with the frozen
`runtime.mjai_adapter.action_to_mjai` and re-validated against the engine state; the
only payload patch is the three-tile `consumed` list the engine's `kakan`
deserializer requires (the frozen adapter omits it).

Observed verification counters from the acceptance runs: `mask_checks` = every
decision, `verify_shanten_checks` on call responses (py shanten == native shanten),
`verify_agari_checks` on `can_tsumo_agari` (py win check == native flag), all with 0
mismatches.

## 6. Synthetic fallback

`_run_seed_synthetic` deals a seeded 136-tile wall (14 dead wall, 52 haipai, 70 live),
lets the four bots draw/discard eight kyoku, applies tenpai/noten payments (3000 split)
and riichi sticks, and awards the remaining sticks to the top score, so the terminal
scores always sum to 100000. It is explicitly **not** mahjong-equivalent
(`parity_status=not_mahjong_equivalent`, no calls/wins/aka) and is used only for tests
and for environments where the pinned engine cannot be imported.

## 7. Provenance and seed pools

`provenance.generator_fingerprint()` hashes every `riichiformer/selfplay/*.py` source
plus the frozen schema version and action-map hash; the digest is embedded in every
per-game row, so a ledger is tied to the exact generator code. `provenance.json` adds
the ruleset metadata, source ledger (`selfplay_own`, `external_license: none`,
`third_party_logs/weights: false`, `network_access: false`), the libriichi module path
and sha256, Python/platform and the exact command.

`seeds.SeedLedger` keeps three disjoint pools (`train`, `select`, `confirm`) in one
JSON file: a wall seed can be reserved in exactly one pool and consumed once; any
reuse (same pool, other pool, or consumed seed) raises `SeedConflict`. `--games` must
be a multiple of 4 and `seed_range(start, games)` returns `games/4` unique walls. When
`--seed-ledger` is omitted the ledger lives at `<out>/seed_ledger.json`, so re-running
into the same output directory with overlapping seeds is refused.

`provenance.guard_out` refuses the repository root and any output path whose resolved
components include `datasets`, `runs`, `checkpoints` or `vendor`; `guard_aux_path`
applies the same rule to the seed ledger file.

## 8. OpponentRegistry and DuplicateArena

`OpponentRegistry` is closed by default:

* unknown names are refused (`UnknownOpponent`);
* a local heuristic cannot carry a checkpoint hash, be marked external, or use a
  reserved token (`mortal`, `akochan`, `tenhou`, `天凤`, `majsoul`, `雀魂`, `九段`) in
  its name/display name (`ImpersonationError`);
* a third-party engine entry requires `external=True`, a content hash and a licence
  caveat, and is refused unless the registry was created with `allow_external=True`;
* every entry names the ruleset it is eligible for.

`default_registry()` contains only the three weight-free local bots; no Mortal,
akochan or external checkpoint is registered anywhere in this repository.

`DuplicateArena` runs the four rotated games per wall seed, re-checks each game
(`GameAudit`: score sum 100000, ranks 1..4, seat permutation, four POVs, zero
illegal/rejected/fallbacks), and reports per-opponent rank/score statistics with
**seed-block** percentile CIs: each resample draws whole seeds with replacement and
concatenates their four games, preserving the within-wall correlation the duplicate
design creates. `bootstrap_independent_games` exists only as a documented negative
control (the four rotated games are not independent). Reports always carry
`official_rank: false` and `strength_claim: "none"`.

## 9. Commands

### Tests

```bash
cd worktrees/v32-selfplay
export PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release
.venv/bin/python -m pytest -q tests/rf_test_selfplay.py tests/rf_test_opponents.py
.venv/bin/python -m pytest -q $(git ls-files 'tests/rf_test_*.py' | tr '\n' ' ')
```

### Self-play (real local engine)

```bash
.venv/bin/python scripts/rf_selfplay.py \
  --out /tmp/rf41-selfplay --seed-start 1000 --games 40 \
  --ruleset libriichi_tenhou_approx/1 --seed-pool train \
  --seed-ledger /tmp/rf41-seeds.json
```

### Self-play (labelled procedural fallback)

```bash
.venv/bin/python scripts/rf_selfplay.py \
  --out /tmp/rf41-synth --seed-start 2000 --games 8 \
  --ruleset synthetic_abstract_hanchan/1 --seed-pool select \
  --seed-ledger /tmp/rf41-synth-seeds.json
```

### Duplicate arena (1v3, confirm pool)

```bash
.venv/bin/python scripts/rf_arena.py \
  --out /tmp/rf42-arena --seed-start 5000 --games 80 \
  --ruleset libriichi_tenhou_approx/1 \
  --opponents shanten-ukeire,defense --seed-pool confirm \
  --seed-ledger /tmp/rf42-seeds.json
```

Exit codes: `0` = all invariants hold; `1` = audit/legality violation; `2` = contract
error (unknown ruleset/opponent, frozen output path, seed reuse, missing libriichi).

## 10. Observed evidence (this machine, CPU only)

All runs below were executed in this worktree with
`PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release`; generator
`code_sha256=ca77438e481ec99e18155c969a82f21b4fa7a6c51f0030733de28bf33fe7a22b`
(8 modules). CPU only, no GPU, no network.

### Tests

| Command | Result | Exit |
|---|---|---|
| `pytest -q tests/rf_test_selfplay.py tests/rf_test_opponents.py` | 53 passed, 12 subtests passed, 7.10 s | 0 |
| `pytest -q $(git ls-files 'tests/rf_test_*.py' \| tr '\n' ' ')` | **547 passed, 2 skipped, 254 subtests passed, 165.31 s** | 0 |

The two skips are the pre-existing explicit skips in `rf_test_ruleset_parity.py`.
The suite run overlapped with three other lanes' suites on the same machine, so the
wall time is not a clean benchmark.

### Self-play, real local rules engine (`libriichi_tenhou_approx/1`)

`--seed-start 1000 --games 40 --seed-pool train` → `exit 0`

| Field | Value |
|---|---|
| games / walls | 40 / 10 |
| decisions | 31,924 |
| illegal / rejected / fallbacks | 0 / 0 / 0 |
| score conservation | 40/40 games sum to 100000 |
| kyoku / hora / reach / ryukyoku / calls | 404 / 187 / 362 / 218 / 587 |
| R02 multi-ron settled by arena | 1 |
| R26 four-riichi | 0 |
| R25 end-of-game kyotaku-carry proxy | 11 |
| R29 ippatsu proxy (UNVERIFIED) | 40 |
| tenhou_equivalent / strength_claim | `false` / `none` |

Per-game ledger rows carry seed, key, rotation, four POVs (decision counts + action
histograms), seats, scores, ranks, generator sha and the gzip game-log sha256.

### Self-play, procedural fallback (`synthetic_abstract_hanchan/1`)

`--seed-start 2000 --games 8 --seed-pool select` → `exit 0`: 8 games / 2 walls,
4,522 decisions, 0 illegal, 40/40-style score conservation (8/8), `tenhou_equivalent:
false`, `parity_status: not_mahjong_equivalent`.

### Duplicate arena (`libriichi_tenhou_approx/1`, 1v3, confirm pool)

`--opponents shanten-ukeire,defense --seed-start 5000 --games 24 --resamples 2000`
→ `exit 0`, 24 games / 6 walls, `GameAudit` clean (score sums, seat permutations,
4 POVs, 0 illegal), 230 kyoku, 124 hora, 207 reach, 106 ryukyoku.
Mean-rank 95% CIs are **seed-block** percentile intervals over 6 walls (e.g.
challenger `shanten-ukeire` 2.458 [2.083, 2.792] over 24 seats; champion `defense`
2.514 [2.403, 2.639] over 72 seats). These are internal seat statistics, not a
strength ranking: `official_rank=false`, `strength_claim=none`.

### Negative controls (all refused, `exit 2`)

| Attempt | Error |
|---|---|
| re-run wall 1000 in pool `train` | `SeedConflict: seed reuse refused` |
| arena `--opponents shanten-ukeire,mortal-1v3` | `UnknownOpponent` |
| `--out datasets/rf41` | `ProvenanceError: … 'datasets' is a frozen location` |
| `--games 5` | `SeedError: --games must be a multiple of 4` |

## 11. Findings

* **P0** — none observed: no hidden-information leak to a bot (whitelist test), no
  score-conservation failure, no illegal action, no network/external-weight use.
* **P1** — R25/R29 deviation counters are *proxies* (kyotaku still on the table at
  game end; wins within a short window after riichi acceptance), not a proof of the
  Tenhou timing rule; the ruleset stays `tenhou_approximate`.
* **P1** — raw engine logs are full-information and are marked as such; they must not
  be fed to a POV-only training path without projection.
* **P2** — the seed ledger has no file lock (serial use assumed); a crashed run keeps
  its seeds reserved (safe but wasteful).
* **P2** — the arena's per-opponent stats count one observation per seat, so a 1v3
  champion contributes three seats per game and its mean rank is not symmetric with
  the challenger's; the block CI is the comparison unit.
* **P2** — `SeatView.open_melds` is derived from the public log and is only
  informational; bots do not use it (own `meld_count` comes from the native state).

## 12. Not run / not claimed

* No online Tenhou/雀魂 bot play, no network access, no external log download.
* No third-party checkpoint, no Mortal/akochan arena run.
* No official dan/rank estimate, no 九段/天鳳位 claim, no Mortal comparison.
* No strength evaluation of the heuristic bots (they are baselines, not champions).
* No GPU use.

## 13. Next gate

PR-43 integration: run the accepted base's `rf_selfplay` with
`libriichi_tenhou_approx/1` into a private output directory, point the shard builder
at the resulting per-game ledger once the POV projection exists, and report
`SELFPLAY_PRIVATE_TRAIN_PASS` only if the shard round-trips through
`ShardReader`/`rf_train.py`. Arena comparisons stay internal (`official_rank: false`)
until a licensed strong-opponent anchor is available.
