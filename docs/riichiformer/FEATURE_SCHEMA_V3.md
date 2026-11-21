# FEATURE_SCHEMA_V3 — PR-08 observable-event / action contract

**Status**: PR-08 implementation (Schema Owner), **draft until Lead ACCEPTED**.
PR-09 is not started; downstream code must not treat this schema as frozen until the Lead
says PR-08 is accepted.  The hash values below are the freeze candidates.

- **Task**: PR-08 (V3 schema and action contract), RiichiFormer V3.1
- **Worktree / branch**: `/home/ygsldr/riichi-ai/worktrees/schema` @ `rf/pr08-schema`
- **Base**: `59691f57bc7158dcdb2ba98eecd9d102857171f8` (tag `rf3-base`)
- **Version string**: `rf-schema-v3/2`
- **Schema file**: `configs/riichiformer/schema_v3.json` (canonical JSON)
- **Implementation**: `riichiformer/schemas/{event_v3,decision_v3,target_v3,action_v3,schema_v3}.py`
- **Tests**: `tests/rf_test_schema.py` (66 passed; see §13)

## 0. Frozen hashes (candidate)

| hash | value |
|---|---|
| `schema_hash` | `12503d36e2571449d4e4bdd4c8780e00f285d3a2ec81f9e26af184d24c8ef43c` |
| `action_map_hash` | `6cf1efc5ba99533ec4844f38184ef51a7941174164f9588f21d33ca70a873a2e` |
| `normalizer_hash` | `37df861b08b3958c3030c53dcbfa27762892c41332b383060caf46788f9ad444` |

Definition (also stored in `schema_v3.json.hash_definition`):

```text
canonical_json(x) = json.dumps(x, ensure_ascii=False, sort_keys=True,
                               separators=(',', ':'))
schema_hash      = sha256(canonical_json(schema without "schema_hash"))
action_map_hash  = sha256(canonical_json(schema["action_map"]))
normalizer_hash  = sha256(canonical_json(schema["normalizer"]))
```

Hashes are over semantic content, not raw bytes: reformatting the file does not change a
hash; changing one field does.  The loader (`riichiformer.schemas.schema_v3.load_schema`)
recomputes all three, optionally checks hashes recorded by a shard/checkpoint
(`expected_hashes=`), and then checks that the JSON matches the code
(`verify_code_consistency`).  Any mismatch raises `SchemaHashMismatch`; there is no
auto-migration.

---

## 1. Identifiers and encodings

### 1.1 Tiles

| value | meaning |
|---|---|
| `kind` 0..33 | `1m..9m, 1p..9p, 1s..9s, E, S, W, N, P, F, C` |
| `kind` 34 | `UNKNOWN` (own hidden draw / opponent hidden tile) |
| `aka = None` | normal tile |
| `aka = 0/1/2` | red five of 5m/5p/5s; only valid on `kind` 4/13/22 with matching colour |

Red fives are **not** separate kinds.  `TileRef(4, None)` (normal 5m) and `TileRef(4, 0)`
(aka 5m) are distinct; `TileRef(34, 0)` is invalid.  The MJAI strings `5mr/5pr/5sr/?`
round-trip through `tile_from_text` / `tile_text`.

State (`VisibleTileState`) keeps `own_aka[3]`/`public_aka[3]` as per-colour counts, and
`own_counts`/`public_counts` count a five in its normal kind; the invariants in §5.2 keep
the two consistent.

### 1.2 Relative seats

| `actor_rel` | meaning |
|---:|---|
| 0 | self (POV seat) |
| 1 | shimo (next player) |
| 2 | toimen (across) |
| 3 | kami (previous player) |
| 4 | none / no actor (only where the field allows it) |

Absolute seat numbers are **not** part of the canonical event/state.  `start_kyoku.scores`
is rotated to `scores_rel` with index 0 = self, and `oya_rel = (oya_abs - pov) % 4`.

### 1.3 Phases

`SELF_TURN`, `CALL_RESPONSE`, `KAN_SELECTION`.  An action index is never interpreted without
its phase.  `KAN_SELECTION` is the second native decision entry (`at_kan_select=True`, label
0..36); `SELF_TURN`/`CALL_RESPONSE` are the ordinary decision entries.

---

## 2. ObservableEventV3 allowlist

Every event is a frozen dataclass (`event_v3.ObservableEventV3` subclass).  A raw MJAI dict
is read through an explicit per-type allowlist; unknown `type` raises `UnknownEventType`;
there is no raw-dict fallthrough.  Canonical serialization uses
`{"type", "seq_idx", "kyoku_idx", "timing"}` plus the event fields below.

### 2.1 Common fields

| field | dtype | visibility | notes |
|---|---|---|---|
| `type` | `event_kind` | public | one of §2.2; `event.kind` is the enum |
| `seq_idx` | `u32` | public | global 0-based event index in the game |
| `kyoku_idx` | `i32` | public | 0-based kyoku-segment index (native `at_kyoku` convention); `-1` before the first `start_kyoku` |
| `timing` | `event_timing` | public | `BOUNDARY` / `TURN` / `RESPONSE` / `SETTLEMENT` |

### 2.2 Event fields (field-by-field visibility)

| event | type_id | timing | field | dtype | visibility |
|---|---:|---|---|---|---|
| `start_game` | 0 | BOUNDARY | `aka_flag` | `bool` | public |
| `start_game` | 0 | BOUNDARY | `kyoku_first` | `u8` | public |
| `start_kyoku` | 1 | BOUNDARY | `bakaze` | `tile` | public |
| `start_kyoku` | 1 | BOUNDARY | `dora_marker` | `tile` | public |
| `start_kyoku` | 1 | BOUNDARY | `kyoku` | `u8` | public |
| `start_kyoku` | 1 | BOUNDARY | `honba` | `u8` | public |
| `start_kyoku` | 1 | BOUNDARY | `kyotaku` | `u8` | public |
| `start_kyoku` | 1 | BOUNDARY | `oya_rel` | `actor_rel` | public |
| `start_kyoku` | 1 | BOUNDARY | `scores_rel` | `i32x4` | public |
| `start_kyoku` | 1 | BOUNDARY | `own_haipai` | `tile13` | **self_only** |
| `tsumo` | 2 | TURN | `actor_rel` | `actor_rel` | public |
| `tsumo` | 2 | TURN | `tile` | `tile` | **self_only** (opponent = UNKNOWN) |
| `dahai` | 3 | TURN | `actor_rel` | `actor_rel` | public |
| `dahai` | 3 | TURN | `tile` | `tile` | public |
| `dahai` | 3 | TURN | `tsumogiri` | `bool` | public |
| `chi` | 4 | RESPONSE | `actor_rel` | `actor_rel` | public |
| `chi` | 4 | RESPONSE | `target_rel` | `actor_rel` | public |
| `chi` | 4 | RESPONSE | `tile` | `tile` | public |
| `chi` | 4 | RESPONSE | `consumed` | `tile4` | public |
| `pon` | 5 | RESPONSE | `actor_rel` | `actor_rel` | public |
| `pon` | 5 | RESPONSE | `target_rel` | `actor_rel` | public |
| `pon` | 5 | RESPONSE | `tile` | `tile` | public |
| `pon` | 5 | RESPONSE | `consumed` | `tile4` | public |
| `daiminkan` | 6 | RESPONSE | `actor_rel` | `actor_rel` | public |
| `daiminkan` | 6 | RESPONSE | `target_rel` | `actor_rel` | public |
| `daiminkan` | 6 | RESPONSE | `tile` | `tile` | public |
| `daiminkan` | 6 | RESPONSE | `consumed` | `tile4` | public |
| `kakan` | 7 | TURN | `actor_rel` | `actor_rel` | public |
| `kakan` | 7 | TURN | `tile` | `tile` | public |
| `kakan` | 7 | TURN | `consumed` | `tile4` | public |
| `ankan` | 8 | TURN | `actor_rel` | `actor_rel` | public |
| `ankan` | 8 | TURN | `consumed` | `tile4` | **rule_dependent** (opponent hidden by default) |
| `reach` | 9 | TURN | `actor_rel` | `actor_rel` | public |
| `reach_accepted` | 10 | TURN | `actor_rel` | `actor_rel` | public |
| `dora` | 11 | TURN | `dora_marker` | `tile` | public |
| `hora` | 12 | SETTLEMENT | `actor_rel` | `actor_rel` | public |
| `hora` | 12 | SETTLEMENT | `target_rel` | `actor_rel` | public |
| `hora` | 12 | SETTLEMENT | `deltas` | `i32x4` | **public_after_event** |
| `hora` | 12 | SETTLEMENT | `ura_markers` | `tiles` | **public_after_event** |
| `hora` | 12 | SETTLEMENT | `deltas_order` | `deltas_order` | public |
| `ryukyoku` | 13 | SETTLEMENT | `deltas` | `i32x4` | **public_after_event** |
| `ryukyoku` | 13 | SETTLEMENT | `deltas_order` | `deltas_order` | public |
| `end_kyoku` | 14 | BOUNDARY | — | — | no fields |
| `end_game` | 15 | BOUNDARY | — | — | no fields |

### 2.3 Hard POV rules

1. **`start_kyoku` stores only `own_haipai`** (13 tiles of the POV seat).  Other `tehais`
   entries are never read into the object; the canonical dict has no `tehais` key.
2. **Opponent `tsumo`** has `tile = UNKNOWN (34, None)`; only the own draw keeps its tile.
3. **Opponent `ankan`** has four `UNKNOWN` consumed slots by default.  The exact public
   reveal is rule-dependent (H-04 R26 log-order caveat; not independently verified); a caller may pass
   `reveal_opponent_ankan=True`, and the event then records the four tiles.  The schema
   never silently assumes omniscient `consumed`.
4. **`start_game`** keeps only `aka_flag`/`kyoku_first`.  `names`, account ids, wall seeds,
   `id` and log URLs are ignored and cannot appear in the serialized event.
5. **`hora.ura_markers`** belong to the settlement event; PR-09 may expose them only at or
   after that event (never before).  `hora.deltas` are likewise post-event information.
6. **No future events.**  A `DecisionInput.events` tuple must contain only events with
   `seq_idx < before_event_idx`; violation raises `DecisionSchemaError`.
7. Unknown event type raises `UnknownEventType`; `from_dict` rejects extra/missing fields
   (`SchemaFieldError`) and invalid values (`SchemaValueError`).
8. **Delta order is explicit.** `hora`/`ryukyoku` carry `deltas_order`:
   `"absolute"` (default, raw MJAI seat order) or `"relative"` (self/shimo/toimen/kami).
   The **sole relative producer** is `event_v3.rotate_deltas_to_relative(deltas, pov)`;
   `event_v3.rotate_deltas_to_absolute(deltas_rel, pov)` is its inverse.  PR-09
   `project_event` uses the helper and sets `"relative"`; raw factory events remain
   `"absolute"`.  PR-10/PR-11 must consume only the PR-09 projection.  Raw
   `hora`/`ryukyoku` deltas are required (H-01: they are the score settlement
   protocol); a source that omits them fails closed.

`consumed` is always a 4-tuple; unused slots are `UNKNOWN` and validated as padding
(chi/pon: 2 known; daiminkan/kakan: 3 known; ankan: all known **or** all unknown).
`start_kyoku.scores_rel` + `1000 * kyotaku == 100000` is enforced (H-01).

---

## 3. DecisionInput (forward input)

`decision_v3.DecisionInput` has exactly these fields; `to_forward_dict()` whitelists them:

| field | type | notes |
|---|---|---|
| `schema_version` | `str` | must be `rf-schema-v3/2` |
| `events` | tuple of `ObservableEventV3` | strictly increasing `seq_idx`, all `< before_event_idx`; non-empty; no kyoku regression |
| `tile_state` | `VisibleTileState` | physical-tile ledger (§5) |
| `players_public` | 4 × `PlayerPublicState` | `seat_rel` 0..3 in order; exactly one dealer matching `oya_rel` |
| `global_public` | `GlobalPublicState` | round/kyoku/honba/kyotaku/tiles-left/winds + ruleset metadata |
| `legal_mask` | tuple[bool] × 46 | validated against `phase_legal_actions`; at least one legal |
| `phase` | `ActionPhase` | `SELF_TURN` / `CALL_RESPONSE` / `KAN_SELECTION` |
| `before_event_idx` | `int >= 1` | index of the event being decided (not yet visible) |

Additional invariants:

- `sum(player.score) + 1000 * kyotaku == 100000`;
- `players_public[i].seat_rel == i`, exactly one `is_oya`, and it equals `oya_rel`;
- `tile_state.ruleset_id == global_public.ruleset_id`;
- no `expert_action`/`supervision_valid`/`censor_reason`/`rank_final`/`oracle_labels`
  field exists, and `to_forward_dict()` recursively rejects any forbidden key
  (`FORBIDDEN_FORWARD_KEYS`).

### 3.1 VisibleTileState

| field | type | invariant |
|---|---|---|
| `own_counts[34]` | tuple[int] | 0..4; own hand only |
| `own_aka[3]` | tuple[int] | 0/1 per colour; `own_aka[c] <= own_counts[5 of c]` |
| `public_counts[34]` | tuple[int] | river + melds + revealed dora indicators, each physical tile once |
| `public_aka[3]` | tuple[int] | includes visible aka indicators; per-colour total own+public ≤ 1 |
| `public_counts_by_zone` | 3 × 34 | `river`, `meld`, `dora_indicator`; each kind sums to `public_counts`; indicator zone contains exactly the visible indicators |
| `dora_indicators` | tuple[TileRef] | ≤ 5, known tiles, counted once in the `dora_indicator` zone |
| `ruleset_id` | `RulesetId` | `tenhou_4p_ranked` / `mortal_arena_pinned` |

`own_counts[t] + public_counts[t] <= 4` for every kind; a tile consumed into a meld is not
also counted in the river (PR-10 owns the migration ledger and must satisfy this).

### 3.2 PlayerPublicState / PublicMeld / GlobalPublicState

`PlayerPublicState`: `seat_rel`, `score`, `wind` (E/S/W/N), `is_oya`, `riichi`
(`none/declared/accepted`), `river_count`, `melds` (≤ 4 `PublicMeld`).  `wind` must equal
`bakaze + (seat_rel - oya_rel)` mod 4 and `players_public[0].wind == self_wind`; no opponent
hand is representable.

`PublicMeld`: `meld_kind` (`chi/pon/daiminkan/kakan/ankan`), `tiles` (3 for chi/pon,
4 for kan; ankan may be four `UNKNOWN`), `from_rel` (0..4).  `is_open` is false for ankan.

`GlobalPublicState`: `bakaze`, `kyoku` 1..4, `honba`, `kyotaku`, `tiles_left` 0..70,
`self_wind`, `oya_rel`, `ruleset_id`, `ruleset_deviation`, `source_kind`.
`self_wind == bakaze` iff `oya_rel == 0`.  See §6 for ruleset metadata.

---

## 4. DecisionTarget (label side, separate module)

`target_v3.DecisionTarget` is **not imported** by `decision_v3` or by the runtime input path.
It lives in its own module; the import-isolation test runs a subprocess that imports
`decision_v3` and asserts `riichiformer.schemas.target_v3` is not in `sys.modules`.

| field | type | notes |
|---|---|---|
| `schema_version` | `str` | must be `rf-schema-v3/2` |
| `expert_action` | `int|None` | 0..45 when present |
| `supervision_valid` | `bool` | if true: action present, `censor_reason` absent |
| `censor_reason` | `str|None` | required when `supervision_valid=False` |
| `rank_final` | `int|None` | 1..4; **never** a forward input |
| `oracle_labels` | mapping|None | expensive labels; **never** a forward input |

`validate_against_input(input)` checks `input.legal_mask[expert_action]` and the phase
(`decode_action`); an action outside the mask raises `TargetNotInMask`.  The trainer must
read `batch.inputs` and `batch.targets` separately; online inference imports only inputs.

---

## 5. 46-action mapping (frozen)

`ACTION_SPACE = 46`.  Indices 0..36 have **phase-dependent** meaning; 37..45 are phase
specific.  This is the native mapping (`consts.rs`, `state/obs_repr.rs`,
`dataset/gameplay.rs`).

| idx | name | kind | tile/aka | SELF_TURN | CALL_RESPONSE | KAN_SELECTION |
|---:|---|---|---|---|---|---|
| 0 | `discard_1m` | discard | 0 | discard | — | kan_tile |
| 1 | `discard_2m` | discard | 1 | discard | — | kan_tile |
| 2 | `discard_3m` | discard | 2 | discard | — | kan_tile |
| 3 | `discard_4m` | discard | 3 | discard | — | kan_tile |
| 4 | `discard_5m` | discard | 4 | discard | — | kan_tile |
| 5 | `discard_6m` | discard | 5 | discard | — | kan_tile |
| 6 | `discard_7m` | discard | 6 | discard | — | kan_tile |
| 7 | `discard_8m` | discard | 7 | discard | — | kan_tile |
| 8 | `discard_9m` | discard | 8 | discard | — | kan_tile |
| 9 | `discard_1p` | discard | 9 | discard | — | kan_tile |
| 10 | `discard_2p` | discard | 10 | discard | — | kan_tile |
| 11 | `discard_3p` | discard | 11 | discard | — | kan_tile |
| 12 | `discard_4p` | discard | 12 | discard | — | kan_tile |
| 13 | `discard_5p` | discard | 13 | discard | — | kan_tile |
| 14 | `discard_6p` | discard | 14 | discard | — | kan_tile |
| 15 | `discard_7p` | discard | 15 | discard | — | kan_tile |
| 16 | `discard_8p` | discard | 16 | discard | — | kan_tile |
| 17 | `discard_9p` | discard | 17 | discard | — | kan_tile |
| 18 | `discard_1s` | discard | 18 | discard | — | kan_tile |
| 19 | `discard_2s` | discard | 19 | discard | — | kan_tile |
| 20 | `discard_3s` | discard | 20 | discard | — | kan_tile |
| 21 | `discard_4s` | discard | 21 | discard | — | kan_tile |
| 22 | `discard_5s` | discard | 22 | discard | — | kan_tile |
| 23 | `discard_6s` | discard | 23 | discard | — | kan_tile |
| 24 | `discard_7s` | discard | 24 | discard | — | kan_tile |
| 25 | `discard_8s` | discard | 25 | discard | — | kan_tile |
| 26 | `discard_9s` | discard | 26 | discard | — | kan_tile |
| 27 | `discard_E` | discard | 27 | discard | — | kan_tile |
| 28 | `discard_S` | discard | 28 | discard | — | kan_tile |
| 29 | `discard_W` | discard | 29 | discard | — | kan_tile |
| 30 | `discard_N` | discard | 30 | discard | — | kan_tile |
| 31 | `discard_P` | discard | 31 | discard | — | kan_tile |
| 32 | `discard_F` | discard | 32 | discard | — | kan_tile |
| 33 | `discard_C` | discard | 33 | discard | — | kan_tile |
| 34 | `discard_5mr` | discard_aka | 4 aka 0 | discard_aka | — | kan_tile_aka |
| 35 | `discard_5pr` | discard_aka | 13 aka 1 | discard_aka | — | kan_tile_aka |
| 36 | `discard_5sr` | discard_aka | 22 aka 2 | discard_aka | — | kan_tile_aka |
| 37 | `riichi` | riichi | — | riichi | — | — |
| 38 | `chi_low` | chi | — | — | chi_low | — |
| 39 | `chi_mid` | chi | — | — | chi_mid | — |
| 40 | `chi_high` | chi | — | — | chi_high | — |
| 41 | `pon` | pon | — | — | pon | — |
| 42 | `kan` | kan | — | kan_decide | kan_decide | — |
| 43 | `agari` | agari | — | tsumo_agari | ron_agari | — |
| 44 | `ryukyoku` | ryukyoku | — | ryukyoku | — | — |
| 45 | `pass` | pass | — | — | pass | — |

Legal index sets per phase (enforced by `validate_mask_for_phase`):

| phase | legal indices |
|---|---|
| `SELF_TURN` | 0..37, 42, 43, 44 |
| `CALL_RESPONSE` | 38, 39, 40, 41, 42, 43, 45 |
| `KAN_SELECTION` | 0..36 |

Notes:

- In `KAN_SELECTION`, 0..33 mean “kan this tile kind”, 34..36 “kan the aka five”.  The
  pinned native engine currently emits kan candidates as 0..33 (aka is deaka'd for the
  candidate), but the schema allows 34..36 for sources that expose the aka kan choice; the
  action is always decoded with the phase.
- In `SELF_TURN`, 42 means ankan/kakan decide; in `CALL_RESPONSE`, 42 means daiminkan decide.
- 43 is tsumo agari in `SELF_TURN` and ron agari in `CALL_RESPONSE`.
- 45 (pass) is only legal in `CALL_RESPONSE`.

---

## 6. Ruleset provenance (H-04 R02/R25/R26)

`RulesetId`: `tenhou_4p_ranked` (human Tenhou ranked logs; **no deviations allowed**) and
`mortal_arena_pinned` (pinned arena self-play; H-04 deviations must be tagged).

`RulesetDeviation` (stored as `ruleset_deviation` in `GlobalPublicState` and shard metadata):

| code | schema description | H-04 status | consequence |
|---|---|---|---|
| `R02` | `triple_ron_settled_instead_of_tenhou_abortive_draw` | FIX_NEEDED | arena pays all three ron; Tenhou aborts (no score, honba+1, dealer renchan, kyotaku carried). Samples containing it are **not** Tenhou labels; PR-12/PR-16 must exclude or downweight by default. |
| `R25` | `last_dealer_stop_uses_kyotaku_inclusive_scores` | FIX_NEEDED | arena's end-of-hanchan test includes the just-paid kyotaku; Tenhou excludes it. Do not infer Tenhou final ranking/continuation from such samples. |
| `R26` | `four_riichi_abort_after_one_extra_draw` | KNOWN_DIFFERENCE | arena draws/broadcasts one extra tile before aborting; score unchanged. POV/event sequences must allow this order; do not assume a fixed Tenhou event order. |

`source_kind`: `human_log`, `arena_selfplay`, `synthetic`.  Any `DecisionInput`/shard records
`ruleset_id`, `ruleset_deviation`, `source_kind`; an unknown ruleset or deviation raises.
`tenhou_4p_ranked` with a non-empty deviation tuple is rejected.

---

## 7. H-01 score protocol note

H-01 proved on the pinned native source and 614 real logs / 6,352 kyoku transitions:

- `hora.deltas` in Tenhou-derived MJAI logs **already include** `1000 × kyotaku` for the
  first winner of a settlement; a second multi-ron winner carries none;
- `ryukyoku.deltas` **exclude** kyotaku (the sticks stay on the table);
- leftover sticks at `end_game` go to the top scorer (lowest seat index on ties).

The schema therefore:

- stores `hora.deltas`/`ryukyoku.deltas` **verbatim** and never re-settles scores; use
  `riichiformer.data.replay.replay_scores` for settlement;
- enforces `sum(start_kyoku.scores_rel) + 1000 * kyotaku == 100000`;
- records `score_protocol.id = "tenhou_mjai_deltas_v1"`.  A source whose `hora` carries
  absolute `scores` instead of deltas is **not** covered by this schema version; the H-01
  replay layer rejects it (fail closed).  Do not reinterpret `hora.deltas` as
  “hand value only”.
- `schema_v3.json.score_protocol` carries these flags for PR-10/PR-12 consumers.

---

## 8. Hash/version rules

1. Any change to event fields, dtypes, visibility, action map, decision/target field lists,
   forbidden fields, or normalizer spec requires:
   - a new `schema_version` string (this document is `rf-schema-v3/2`),
   - regenerating `schema_v3.json`,
   - recomputing all three hashes,
   - rebuilding affected shards/checkpoints; a hash mismatch must raise, never migrate
     silently.
2. A shard/checkpoint stores the three hashes; loading calls
   `load_schema(expected_hashes=...)`.
3. `load_schema(verify_code=True)` (default) additionally verifies that the JSON matches
   `event_v3.event_type_specs()`, `action_v3.canonical_action_map()`, and the decision field
   lists.
4. `candidate_schema_version = "rf-candidate-v3/0"` and
   `belief_label_schema_version = "rf-belief-v3/0"` are reserved.  No expensive candidate
   or belief labels are produced under PR-08.
5. PR-09 is **not** started; the schema remains a draft until Lead acceptance.  Downstream
   agents may read the interface preview but must not consume it as a frozen training
   contract before acceptance.

---

## 9. Normalizer spec (hash-frozen, implementation in PR-11)

`normalizer_hash` covers `schema_v3.json.normalizer`:

| key | spec |
|---|---|
| `version` | `rf-normalizer-v3/1` |
| `score` | scalar, clip `[-100000, 100000]`, scale `/100000`, f32 |
| `score_rel` | ordinal, center 25000, scale 25000, clip 3σ |
| `counts` | count 0..4, scale `/4` |
| `aka` | count 0..1 |
| `tiles_left` | count 0..70 |
| `honba`, `kyotaku` | count 0..255 |
| `wind` | one-hot 4 |
| `phase` | one-hot 3 |
| `event_type` | embedding 16 |
| `tile_kind` | embedding 35 (34 kinds + UNKNOWN) |
| `aka_flag` | embedding 3 (`None`/0/1/2) |

PR-11 must implement exactly this and assert `normalizer_hash` against the schema.

---

## 10. Schema JSON layout

`configs/riichiformer/schema_v3.json` is canonical JSON with:

| key | content |
|---|---|
| `schema_version` | `rf-schema-v3/2` |
| `schema_hash`, `action_map_hash`, `normalizer_hash` | §0 |
| `tile_kinds`, `aka`, `actor_rel` | §1 |
| `phases`, `phase_legal_actions` | §1.3/§5 |
| `event_timing`, `common_event_fields`, `event_types`, `dtypes` | §2 |
| `decision_input_fields`, `decision_target_fields`, `forward_input_forbidden_fields` | §3/§4 |
| `action_map` | §5 (46 entries with phase semantics) |
| `normalizer` | §9 |
| `rulesets`, `ruleset_deviations`, `source_kinds` | §6 |
| `candidate_schema_version`, `belief_label_schema_version` | §8 |
| `score_protocol`, `pov_rules`, `decision_input_rules`, `versioning`, `hash_definition` | metadata |

---

## 11. Module dependency direction

```text
action_v3.py        (stdlib only; tile names, phases, 46 map, hash)
    ↑
event_v3.py         (tiles, explicit event dataclasses, POV factory)
    ↑
decision_v3.py      (VisibleTileState, PlayerPublicState, GlobalPublicState, DecisionInput)
    ↑                         ✗  no import
target_v3.py        (DecisionTarget; imports action_v3 only)
schema_v3.py        (JSON loader/hash verifier; imports all three for code consistency)
```

`decision_v3` does not import `target_v3`; the subprocess isolation test enforces this.
There is no model/runtime file in PR-08, so no `forward()` implementation is frozen here.

---

## 12. Evidence / limits

- 61 unit tests in `tests/rf_test_schema.py`, deterministic, no datasets/network:
  event roundtrip (all 16 types), unknown/extra/missing fields, POV hiding, aka identity,
  0..36 dual semantics, chi 38..40, target isolation, hash tamper, mask/phase validation,
  ruleset metadata, import isolation.
- The 46 mapping is transcribed from the pinned native source and matches the mask/label
  code; PR-10 must still prove per-decision mask equality on real logs against
  `GameplayLoader`/`PlayerState` (not in PR-08 scope).
- R02/R25/R26 are metadata only; this PR does not fix vendor rules or claim Tenhou
  equivalence.
- The schema does not settle scores; H-01 remains the authority (§7).
- H-04 R29 (ippatsu timing) and exact ankan public reveal are **UNVERIFIED** rule details;
  the schema keeps ankan `rule_dependent` and does not encode an ippatsu field.

## 13. Test command and counts

```bash
cd /home/ygsldr/riichi-ai/worktrees/schema
PYTHONPATH= /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q tests/rf_test_schema.py
# 66 passed
```

A full stdout/exit-code log is written to
`docs/riichiformer/v3/logs/PR08_pytest.log` before the commit.
