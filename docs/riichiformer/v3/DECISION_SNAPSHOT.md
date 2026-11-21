# DECISION_SNAPSHOT — PR-10 native decision opportunities, POV masks, censoring

**Status**: PR-10 draft (Decision Snapshot owner). Not frozen until Lead accepts PR-10.
- **Task**: PR-10 — Decision Snapshot / censor / public-tile ledger
- **Worktree/branch**: `/home/ygsldr/riichi-ai/worktrees/decision` @ `rf/pr10-decision`
- **Base**: `8c7cbe087f8d946c41b8de4138739448d1930fa0` (includes PR-08 schema v3/2 and PR-09 `ProjectedStreamHeader`/`deltas_order="relative"`)
- **Schema**: `rf-schema-v3/2` (`schema_hash 12503d36e2571449d4e4bdd4c8780e00f285d3a2ec81f9e26af184d24c8ef43c`, action `6cf1efc5…`, normalizer `37df861b…`)
- **POV**: `rf-pov-v3/2`
- **Owned files**: `riichiformer/data/decision.py`, `riichiformer/data/tile_counter.py`, `tests/rf_test_decision.py`, `tests/rf_test_tile_counter.py`, this doc, `docs/riichiformer/v3/logs/PR10_pytest.log`.
- **Tests**: 20 passed (`14` decision + `6` tile-counter), exit 0.

## 1. API

```python
from riichiformer.data.decision import (
    DecisionBuildConfig, DecisionSample, build_decisions,
)

config = DecisionBuildConfig(
    ruleset_id=RulesetId.TENHOU_4P_RANKED,   # or MORTAL_ARENA_PINNED
    source_kind=SourceKind.HUMAN_LOG,        # or ARENA_SELFPLAY
    ruleset_deviation=(),                    # arena: subset of R02/R25/R26
)
samples: list[DecisionSample] = build_decisions(raw_mjai_events, config)
record = samples[0].as_record()
```

`DecisionSample` is an immutable pair:
- `input: DecisionInput` — forward-safe visible prefix only;
- `target: DecisionTarget` — expert action/censor/rank, kept out of `input`;
- `header: ProjectedStreamHeader` — POV attribution (below);
- `decision_index` — per-seat native decision ordinal.

`as_record()` returns exactly:

```json
{"header": {...}, "decision_index": 0,
 "input": {...DecisionInput.to_forward_dict()...},
 "target": {...DecisionTarget.to_dict()...}}
```

The header is stored **on every sample record** (including shard records), so a four-POV
or raw/projected mix-up is detectable. No schema field was added for it.

## 2. Native authority and exact alignment

- `libriichi.dataset.GameplayLoader(4, oracle=config.oracle,
  always_include_kan_select=True)` supplies the native actions and 46-wide masks.
- A separate `libriichi.state.PlayerState` replay reproduces the loader's
  `events.windows(4)` loop exactly:
  - `next = wnd[2]` when `wnd[1]` is `reach_accepted`/`dora`, else `wnd[1]`;
  - `cans = state.update(cur)`; decisions exist only when `cans.can_act()`;
  - label selection is a line-for-line port of `gameplay.rs`
    (`Dahai`/`Reach`/`Chi`/`Pon`/`Daiminkan`/`Kakan`/`Ankan`/`Ryukyoku`/Hora/Pass).
- Every generated supervised opportunity is cross-checked against `Gameplay`:
  label, `at_kyoku`, `at_turns`, and the exact native mask must match. Any mismatch
  raises `DecisionBuildError`; no silent repair.
- The POV projection is `POVProjector(pov)`; the visible prefix is
  `projector.events[:before_event_idx]`.

### Offset semantics

`before_event_idx` is the raw/projected index of the event being decided. `DecisionInput`
contains only events with `seq_idx < before_event_idx`; the action event itself is never in
the input. For a pass (45) the next event after the response window is at that index. For a
kan declaration and its tile selection, both entries share the same `before_event_idx`
(the kan event), because neither can see the kan event yet.

## 3. Phase table

| phase | detection (after native `cur`) | legal action family | examples |
|---|---|---|---|
| `SELF_TURN` | `cans.can_discard` true; `44` is kyuushu before a draw | `0..37`, `42`, `43`, `44` | discard, riichi declaration, tsumo, ankan/kakan decide, kyuushu |
| `CALL_RESPONSE` | `cans.can_discard` false | `38..43`, `45` | chi/pon/daiminkan, ron, pass |
| `KAN_SELECTION` | second native entry immediately after a `42` declare entry; `encode_obs(at_kan_select=True)` mask | `0..36` | choose the kan tile; multiple candidates stay in the mask |

`0..36` are discards in `SELF_TURN` and kan-tile choices in `KAN_SELECTION`; the phase
disambiguates the 46 action mapping. `42`/`43` legitimately overlap `SELF_TURN`/
`CALL_RESPONSE` with different semantics. `DecisionInput` runs
`schema.action_v3.validate_mask_for_phase`; an out-of-phase native mask is a hard failure.

Aka vs normal five: `dahai` uses native `Tile::as_usize()` (0..33 normal, `34..36` aka
5m/5p/5s); kan choices use the deaka kind. Native masks are retained verbatim.

## 4. Censor semantics (target only)

| situation | `expert_action` | `supervision_valid` | `censor_reason` |
|---|---|---|---|
| native label (0..45), in mask | label | true | null |
| native label 45, but another seat's `chi`/`pon`/`daiminkan` took the discard | 45 | **false** | `preempted_by_call` |
| native loader emits no label because another seat's hora preempted this seat | null | false | `preempted_by_ron` |
| native loader emits no label because another seat's call preempted this seat | null | false | `preempted_by_call` |
| native loader emits no label for an unlabelled chi-only window (its known limitation) | null | false | `unlabeled_opportunity` |
| native label not actually in the mask | label | false | `action_not_in_mask` |
| supervised action is outside the mask | — | — | hard `DecisionBuildError` |

A **genuine** pass is a 45 that the native loader emits when no other call/ron preempted
the window (e.g. a chi-only opportunity followed by the POV's own tsumo, or a pon/ron
opportunity followed by a normal discard/tsumo). It stays supervised. A 45 emitted while
another seat's call is the next event is marked censored, because the pass had no effect.

Censored samples still carry a full `DecisionInput` and a `rank_final`; they are excluded
from behavior cloning, not from coverage/audit.

## 5. State and physical-tile ledger

`SnapshotTracker` only consumes POV-projected events (relative seats, relative deltas):

- `start_kyoku`: scores, kyotaku, honba, bakaze, oya, own haipai, initial dora;
- `tsumo`: own draw added to own hand; draw count tracked;
- `dahai`: own tile removed; river appended; `TileCounter` river;
- `chi`/`pon`/`daiminkan`: called tile removed from target river and moved river→meld
  **once**; known consumed tiles removed from own hand and added to meld;
- `kakan`: pon meld becomes a 4-tile kakan; own added tile removed;
- `ankan`: own tiles removed/added; opponent ankan stays UNKNOWN (no physical count);
- `reach`/`reach_accepted`: declared/accepted; accepted deducts 1000 and adds kyotaku;
- `dora`: indicator appended;
- `hora`/`ryukyoku`: relative deltas applied; hora clears kyotaku;
- `end_kyoku`/`end_game`: boundary only.

`TileCounter` zones: own hand, river, meld, dora indicator, and `other_public` (folded
into the schema meld zone in `VisibleTileState`; currently empty). It enforces:
- per-kind physical conservation `own + public <= 4`;
- aka per colour `own + public <= 1`, with own/public five bounds;
- a called tile leaves the river and enters the meld exactly once;
- indicator kinds/counts exactly match the visible dora markers;
- public zone counts sum to `public_counts`.

`VisibleTileState` is constructed from the ledger; the schema independently re-checks
overflow, zone sums, aka bounds, and indicator exactness. `DecisionInput` re-checks
`sum(scores) + 1000*kyotaku == 100000`, winds/oya, mask/phase, and event cutoff.

## 6. Header attribution

`DecisionSample.header` is `pov.ProjectedStreamHeader`:

| field | meaning |
|---|---|
| `pov` | 0=self, 1=shimo, 2=toimen, 3=kami |
| `projection_version` | `rf-pov-v3/2` (schema v3/2 delta-order update) |
| `schema_version` | `rf-schema-v3/2` |
| `ruleset_id` | `tenhou_4p_ranked` or `mortal_arena_pinned` |
| `source_kind` | `human_log` / `arena_selfplay` / `synthetic` |
| `reveal_opponent_ankan` | policy flag (default false) |

`DecisionBuildConfig` validates ruleset/source/deviation consistency and any supplied
`ProjectionPolicy`. H-04 R02/R25/R26 are carried in `GlobalPublicState.ruleset_deviation`;
PR-10 does not infer them from events.

## 7. Test evidence

### Synthetic/real regression (14 decision + 6 tile tests)

- minimal aka discard: action 34, `SELF_TURN`, header, no target/rank in input;
- normal 5m: action 4;
- genuine pass vs preempted call: 45 valid vs 45 `preempted_by_call` invalid;
- preempted ron: `expert_action=None`, `preempted_by_ron`;
- double ron: two supervised `43` at the same boundary;
- multiple kan candidates: `KAN_SELECTION` mask has two tiles, target 1m;
- riichi declaration (37) then later discard; score 24000 / kyotaku 1 after acceptance;
- chi/pon response actions; config validation;
- tile ledger: aka river→meld, conservation, overflow, zone/indicator exactness,
  call-without-river, UNKNOWN rejection;
- real-kan wall regression: `tiles_left == 70 - tsumos in the current kyoku` on
  reviewer logs with kan (previously off by +1 after every kan).

### Real-log stratified sample

Command (read-only datasets; `PYTHONPATH` required):

```bash
cd /home/ygsldr/riichi-ai/worktrees/decision
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q -s \
  tests/rf_test_decision.py tests/rf_test_tile_counter.py
```

Result: **20 passed, exit 0** (`PR10_pytest.log`). Exact 10k sample:

- **16 games**, **10,000 decisions**, **0 build errors**;
- phases: `SELF_TURN=8000`, `CALL_RESPONSE=1990`, `KAN_SELECTION=10` (stratified);
- supervised **9,950**, censored **50** (29 `preempted_by_call`, 21 `preempted_by_ron`);
- **illegal labels outside mask = 0**;
- double-ron games in the sample: **1**; riichi-then-later-discard POVs: **57**;
- actions include all classes: 0..36 (aka 34/35/36), 37, 38/39/40, 41, 42, 43, 44, 45
  (`45=1537`, `42=10`, `43=140`, `44=4`, `37=123`, `41=185`, `38=45`, `39=44`, `40=51`);
- rank_final is present for all 10,000 targets; header attribution present for all;
- per-sample physical conservation/aka/zone sums pass.

## 8. Known gaps / follow-ups

- No 100k sample (time-boxed to 10k); no training.
- `unlabeled_opportunity` is the native loader's own limitation (chi on a discard not
  followed by a tsumo); it is censored, not guessed.
- `tiles_left` is `70 - draws + kans` (schema range only; not wall RNG).
- `other_public` is currently folded into the schema meld zone; no separate schema zone.
- R02/R25/R26 are caller metadata only; PR-10 does not classify server provenance.
- Synthetic games are minimal; full invalid-log behavior is fail-closed via
  `DecisionBuildError`/skip, not repaired.
- `DecisionSample` includes `target` for audit/shard storage; training consumers must
  read only `input` for the forward pass.
