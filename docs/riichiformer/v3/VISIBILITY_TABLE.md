# VISIBILITY_TABLE — PR-09 POV projection contract

**Status**: PR-09 implementation (Schema Owner), draft until Lead accepts PR-08 + PR-09.
- **Task**: PR-09 (per-POV observable-event projection), RiichiFormer V3.1
- **Worktree / branch**: `/home/ygsldr/riichi-ai/worktrees/pov` @ `rf/pr09-pov`
- **Base**: `885b14f` (PR-08 merged to integration; schema remains a freeze candidate)
- **Schema**: `rf-schema-v3/2` (`configs/riichiformer/schema_v3.json`); `schema_hash 12503d36e2571449d4e4bdd4c8780e00f285d3a2ec81f9e26af184d24c8ef43c`, `action_map_hash 6cf1efc5…`, `normalizer_hash 37df861b…`
- **Module**: `riichiformer/data/pov.py` (`POV_PROJECTION_VERSION = rf-pov-v3/2`)
- **Tests**: `tests/rf_test_pov.py` (28 passed; optional real-log test skipped when absent)
- **Log**: `docs/riichiformer/v3/logs/PR09_pytest.log`

## 1. API

```python
from riichiformer.data.pov import (
    ProjectionContext, ProjectionPolicy, POVProjector,
    project_event, project_stream,
)

event = project_event(raw_mjai_dict, pov, ProjectionContext(seq_idx, kyoku_idx))
projector = POVProjector(pov, policy=ProjectionPolicy(...))
event = projector.project(raw_mjai_dict)       # assigns seq/kyoku, appends
events = projector.extend(raw_events)          # atomic; rolls back on error
visible = projector.visible_events(before_event_idx)
header = projector.header                      # ProjectedStreamHeader
stream = projector.stream()                    # ProjectedStream(header, events)
blob = projector.encode_visible(before_event_idx)   # header + visible events
```

- `project_event` is the explicit PR-08 allowlist (`event_v3.event_from_mjai`) plus PR-09
  policy and relative-seat score deltas.  The raw dict is read field-by-field and never
  deep-copied or stored.
- `POVProjector` is one-game, one-POV state: it increments `kyoku_idx` on `start_kyoku`,
  seals after `end_game`, and exposes only the prefix with
  `seq_idx < before_event_idx`.  `reset()` starts a new game.
- Unknown event type raises `UnknownEventType`; events before `start_kyoku` raise
  `ProjectionOrderError`; projecting after `end_game` raises `ProjectionClosed`.
- `encode_visible`/`ProjectedStream.encode` are deterministic visibility witnesses
  (canonical JSON, sorted keys) containing the **header** plus visible events; they are
  not the PR-11 model encoder.
- `ProjectedStreamHeader` is the P1-2 attribution contract: `pov`, `projection_version`,
  `schema_version`, `ruleset_id`, `source_kind`, `reveal_opponent_ankan`.  It is static for
  one projector and is included in encoded payloads, so a four-POV or raw/projected mix-up
  is detectable; it is not a schema event and is not inserted into `DecisionInput`.
- `extend()` is atomic: on a projection error it restores the previous events/kyoku/closed
  state and re-raises.  `project()` is atomic per event.

## 2. Per-event field visibility / timing

Common fields for every event: `type` (event kind), `seq_idx`, `kyoku_idx`, `timing`.
`actor_rel`/`target_rel` are relative to the POV seat (`0=self, 1=shimo, 2=toimen,
3=kami`); absolute seats are not stored.  `kyoku_idx` is 0-based; `-1` for `start_game`.

| event | timing | public fields | POV-private / terminal fields | PR-09 projection rule |
|---|---|---|---|---|
| `start_game` | BOUNDARY | `aka_flag`, `kyoku_first` | — | POV-independent; `names`, account ids, `seed`, `id`, `log` are **dropped** (never read into the event) |
| `start_kyoku` | BOUNDARY | `bakaze`, `dora_marker`, `kyoku`, `honba`, `kyotaku`, `oya_rel`, `scores_rel` | `own_haipai` (self only) | only `tehais[pov]` is read; the other three hands are never stored; `scores_rel` rotated to self/shimo/toimen/kami |
| `tsumo` | TURN | `actor_rel` | `tile` (self only) | own draw retains tile/aka; opponent draw is `UNKNOWN (34, None)` — regression for old patch 0006 |
| `dahai` | TURN | `actor_rel`, `tile`, `tsumogiri` | — | public tile identity/aka and tsumogiri retained |
| `chi` | RESPONSE | `actor_rel`, `target_rel`, `tile`, `consumed` | — | public; `target_rel` must be kamicha (PR-08 validates) |
| `pon` | RESPONSE | `actor_rel`, `target_rel`, `tile`, `consumed` | — | public |
| `daiminkan` | RESPONSE | `actor_rel`, `target_rel`, `tile`, `consumed` | — | public |
| `kakan` | TURN | `actor_rel`, `tile`, `consumed` | — | public (the added tile is public) |
| `ankan` | TURN | `actor_rel` | `consumed` (rule-dependent) | opponent tiles are four `UNKNOWN` slots by default; opt-in only with `ProjectionPolicy(reveal_opponent_ankan=True, ankan_evidence=...)`; self ankan keeps its tiles in full-info logs |
| `reach` | TURN | `actor_rel` | — | public declaration; no score effect yet |
| `reach_accepted` | TURN | `actor_rel` | — | public; only this event deducts 1000 in H-01 settlement |
| `dora` | TURN | `dora_marker` | — | public; projected at the raw event position (after the kan/ankan that revealed it); no reordering |
| `hora` | SETTLEMENT | `actor_rel`, `target_rel`, `deltas_order` | `deltas`, `ura_markers` (after this event) | projected only at its position; deltas rotated to relative seats and `deltas_order="relative"`; `ura_markers` exist only in/after this event |
| `ryukyoku` | SETTLEMENT | `deltas_order` | `deltas` (after this event) | projected only at its position; deltas rotated to relative seats and `deltas_order="relative"` |
| `end_kyoku` | BOUNDARY | — | — | terminal for the kyoku; visible only at its position |
| `end_game` | BOUNDARY | — | — | terminal for the game; projector seals here |

The raw `start_kyoku.scores` and `hora`/`ryukyoku` `deltas` are **absolute seat order** in
MJAI.  PR-09 projects them into relative seat order (self, shimo, toimen, kami) and sets
`deltas_order="relative"` so they match `scores_rel` and `actor_rel`; raw MJAI events
default to `deltas_order="absolute"`; use
`pov.rotate_deltas_to_absolute(projected_deltas, pov)` to recover the raw order (e.g., for
H-01 `replay_scores`, which consumes raw logs, not projected events).

## 3. Timing rules

| timing | events | when it becomes visible |
|---|---|---|
| `BOUNDARY` | `start_game`, `start_kyoku`, `end_kyoku`, `end_game` | at its own event position; `end_game` seals the projector |
| `TURN` | `tsumo`, `dahai`, `reach`, `reach_accepted`, `dora`, `kakan`, `ankan` | at its own event position |
| `RESPONSE` | `chi`, `pon`, `daiminkan` | at its own event position (call window) |
| `SETTLEMENT` | `hora`, `ryukyoku` | at its own event position; their `deltas`/`ura_markers` are not available before it |

`visible_events(before_event_idx)` returns events with `seq_idx < before_event_idx`;
therefore a decision taken **before** a `hora` cannot see its `ura_markers` or deltas.
`encode_visible(before_event_idx)` is the deterministic byte witness of that prefix.
`end_kyoku`/`end_game` are not available before their positions either.

## 4. Reveal policy for opponent ankan

Default: four `UNKNOWN` consumed slots.  The only opt-in is

```python
ProjectionPolicy(
    ruleset_id=RulesetId.MORTAL_ARENA_PINNED,
    source_kind=SourceKind.ARENA_SELFPLAY,
    reveal_opponent_ankan=True,
    ankan_evidence="<which source/rule makes ankan tiles public>",
)
```

A non-empty `ankan_evidence` is required; without it the policy raises
`ProjectionPolicyError`.  The policy also rejects `human_log` + arena ruleset (and vice
versa).  The schema deliberately does not assume the omniscient `consumed` field.

H-04 evidence note: R05 says kan-dora timing matches Tenhou (ankan immediate, daiminkan/
kakan after discard/next rinshan), but the **tile reveal** of an opponent's ankan is
rule-dependent and is not established as universally public by H-04; hence default hidden.

## 5. H-04 caveats carried by the projection

The projector does not rewrite event order or settle rules; the caller must carry
`ruleset_id`/`ruleset_deviation`/`source_kind` in `DecisionInput`/shard metadata
(PR-08 `GlobalPublicState`).

| code | caveat | PR-09 behavior |
|---|---|---|
| **R02** | arena settles triple ron; Tenhou aborts | all three `hora` events are projected at their positions; no abort is injected. Samples from the arena must be tagged `ruleset_deviation=["R02"]` (PR-08 metadata) and are not Tenhou labels. |
| **R25** | arena last-dealer stop includes the just-paid kyotaku; Tenhou excludes it | not observable in the event stream; score/continuation metadata only. Do not infer the final hand/rank from the projected events alone. |
| **R26** | arena checks four-riichi abort after one extra draw; Tenhou aborts at the fourth acceptance | the extra `tsumo`/`dora` (if any) before the abortive `ryukyoku` is preserved in raw order; no normalization. `test_r26_four_riichi_extra_draw_order_is_preserved`. |

## 6. Old patch 0006 opponent-tsumo leak regression

Patch `0006-event-stream-tokens.patch` serialized `Event::Tsumo { pai }` directly into the
event-token stream, so an opponent's drawn tile could enter the feature stream.  PR-09
keeps only `actor_rel`/`seq` for an opponent draw and sets `tile=(34, None)`; the
regression test also verifies that the raw `pai` string is absent from the serialized
projected event and that the input dict is not mutated.

## 7. Property tests (28 passed)

| test | property |
|---|---|
| `test_four_pov_symmetry` | four POVs project the same public stream; relative actors/targets invert to raw absolute seats; `scores_rel`, own hands and deltas match per POV |
| `test_hidden_opponent_perturbation_is_byte_identical` | changing only opponents' hands/tsumo pa/ankan tiles leaves `encode_visible()` byte-identical |
| `test_hidden_ankan_perturbation_is_byte_identical` | opponent ankan consumed changes do not affect output |
| `test_own_hand_perturbation_is_visible` | changing only the own haipai changes `encode_visible()` |
| `test_future_truncation_does_not_change_past` | prefix output equals full-stream output at `before_event_idx`, before/after extending |
| `test_aka_migrates_from_river_to_meld_without_double_count` | called aka 5m leaves the river and enters the meld; total aka count 1 |
| `test_projected_deltas_are_marked_relative` | projected `hora`/`ryukyoku` deltas are relative and marked; raw events default absolute; inverse rotation recovers raw order |
| `test_projected_stream_header_attribution` | header carries pov/version/schema/ruleset/source; POV mix-ups detectable; encoded payload includes header |
| `test_malformed_opponent_ankan_is_masked_before_parsing` | hidden opponent ankan with absent/malformed consumed projects as UNKNOWN; own ankan still requires consumed |
| `test_extend_rolls_back_on_error` | failed `extend()` restores events and encoded payload |
| `test_kan_and_ura_timing` | initial dora vs kan dora order; `ura_markers` absent before `hora`, present after; terminal events gated |
| `test_r26_four_riichi_extra_draw_order_is_preserved` | R26 order preserved |
| `test_ruleset_event_order_is_not_rewritten` | R02 three ron projected without injected abort |
| `test_unknown_event_fails_closed_and_keeps_state` | unknown type raises; projector state unchanged |
| `test_0006_opponent_tsumo_pai_leak_regression` | opponent tile/aka UNKNOWN/None; no `pai` key; raw unchanged |
| `test_no_forbidden_keys_in_projected_stream` | no names/id/seed/log/tehais/pai/target fields in serialized output |
| `RealLogProjectionTests` | optional read-only 2025 logs × four POVs; every event projects and own haipai matches the raw POV hand |

Test command:

```bash
cd /home/ygsldr/riichi-ai/worktrees/pov
/home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q tests/rf_test_pov.py
# 28 passed (or 27 passed + 1 skipped if the local dataset is absent)
```

## 8. Scope limits / follow-ups

- PR-10 owns the production public-tile ledger (`tile_counter.py`); the aka migration test
  uses a local test ledger only.
- PR-11 owns the model encoder; `encode_visible` is a canonical-JSON visibility witness,
  not a feature tensor.
- **Delta order is explicit** (schema v3/2): raw MJAI events default to
  `deltas_order="absolute"`; `POVProjector` sets `"relative"`.  Downstream must use the
  PR-09 projection (or `rotate_deltas_to_relative`) when mixing deltas with `scores_rel`,
  and must use raw logs for H-01 `replay_scores`.  Hora/ryukyoku deltas are required by
  the schema.
- The P1-2 header is PR-09-only metadata; `DecisionInput`/schema fields are unchanged.
  PR-10 should retain `ProjectedStreamHeader` alongside each `DecisionInput` or in shard
  metadata so POV/raw provenance is auditable.
- Ankan reveal remains UNVERIFIED beyond the explicit evidence policy; R29 ippatsu is not
  encoded.  No Tenhou-equivalence claim is made.
