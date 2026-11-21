# ENCODER_PARITY — PR-11 single offline/online encoder

**Status**: PR-11 draft (Encoder owner). Not frozen until Lead accepts PR-10/PR-11.
- **Task**: PR-11 — one pure encoder + causal parity
- **Worktree/branch**: `/home/ygsldr/riichi-ai/worktrees/encode` @ `rf/pr11-encoder`
- **Base**: `924e9f654e328556d14df788b4de390ded98a1ea` (includes PR-08 v3/2, PR-09 POV header, PR-10 DecisionSnapshot, delta rotation helper)
- **Schema**: `rf-schema-v3/2` (`schema_hash 12503d36e2571449d4e4bdd4c8780e00f285d3a2ec81f9e26af184d24c8ef43c`, action `6cf1efc5…`, normalizer `37df861b…`)
- **POV/projection**: `rf-pov-v3/2`
- **Owned files**: `riichiformer/data/encode.py`, `riichiformer/runtime/encoder.py`, `tests/rf_test_encoder_parity.py`, `tests/rf_test_causal.py`, this doc, `docs/riichiformer/v3/logs/PR11_pytest.log`
- **Tests**: 16 passed (`7` parity + `9` causal), exit 0.

## 1. API

```python
from riichiformer.data.encode import StreamMeta, encode_decision, encode_batch

meta = StreamMeta.from_header(sample.header)       # PR-09 header + frozen hash
row  = encode_decision(sample.input, meta)          # EncodedDecision, no padding
rows = encode_batch([a.input, b.input], [meta_a, meta_b])
batch = rows.to_model_batch()                       # PR-13 ModelBatch
```

`riichiformer.runtime.encoder.OnlineEncoder` is a thin adapter:

```python
online = OnlineEncoder()
row = online.encode(sample.input, sample.header)    # delegates to encode_decision
```

There is exactly one mapping implementation. `runtime/encoder.py` contains no token/state
logic and imports no target/label/rank module.

`EncodedDecision` / `EncodedBatch` expose only forward arrays plus `StreamMeta`; they have
no target, label, rank, oracle, censor or future field. `encoded_hash()` is SHA-256 over
canonical JSON of all arrays + metadata, used as the offline/online byte witness.

## 2. Exact row/field mapping

One row is one `DecisionInput` (one POV opportunity). `L = len(input.events)`; the action
event is not part of `DecisionInput`, so the **decision token is the last visible event**
`L-1`. `decision_time_index = L-1`; `state_features` is injected at that token.

| ModelBatch field | shape | mapping |
|---|---|---|
| `event_type_id` | `[B,L]` | `EVENT_TYPE_IDS[event.EVENT_TYPE]` (0..15); padding `16` |
| `event_actor_rel` | `[B,L]` | `event.actor_rel` if present, else `4` (none); padding `5` |
| `event_tile_kind` | `[B,L]` | tile mapping below (0..34); padding `34`/UNKNOWN |
| `event_aka` | `[B,L]` | `0` none/normal; `1..3` = red 5m/5p/5s; padding `0` |
| `event_tsumogiri` | `[B,L]` | `dahai.tsumogiri` (0/1); 0 on every other event; padding `0` |
| `event_phase` | `[B,L]` | decision `ActionPhase` index at the decision token; `0` on other tokens; padding `3` |
| `event_kyoku_pos` | `[B,L]` | 0-based event ordinal within its kyoku; resets to 0 at `start_kyoku`; `>=512` raises; padding `0` |
| `padding_mask` | `[B,L]` | left-aligned True prefix per row; `encode_decision` row is all True |
| `state_features` | `[N,172]` | PR-13 `StateSnapshot` + `state_features_from_snapshot` (same layout/scale) |
| `decision_batch_index` | `[N]` | `0..B-1` (one decision per row) |
| `decision_time_index` | `[N]` | `L-1` per row |
| `legal_mask` | `[N,46]` | exact `DecisionInput.legal_mask`; phase validated by schema |
| `decision_phase` | `[N]` | `SELF_TURN=0`, `CALL_RESPONSE=1`, `KAN_SELECTION=2` |

### Per-event tile mapping

| event | `event_tile_kind` / `event_aka` |
|---|---|
| `start_game` | UNKNOWN / 0 |
| `start_kyoku` | public initial `dora_marker` / its aka |
| `tsumo` | own draw tile; opponent draw is already UNKNOWN from PR-09 |
| `dahai` | discarded tile / its aka; `tsumogiri` from event |
| `chi` / `pon` / `daiminkan` | called public tile / its aka |
| `kakan` | added public tile / its aka |
| `ankan` | own known kan uses `consumed[0]`; opponent ankan is UNKNOWN / 0 |
| `dora` | revealed indicator / its aka |
| `reach` / `reach_accepted` / `hora` / `ryukyoku` / `end_kyoku` / `end_game` | UNKNOWN / 0 |

`UNKNOWN = 34`, `aka 1/2/3` are exactly 5m/5p/5s (schema kinds 4/13/22). Opponent
`tsumo`/`ankan` hidden tiles therefore can never become a known tile in the tensor row.

### State features (172, PR-13 layout)

`own_counts[34]`, `public_counts[34]`, `own_aka[3]`, `public_aka[3]`,
`players[4×11]`, `global[16]`, `dora[35]`, `phase[3]`.
The encoder builds a `StateSnapshot` from `DecisionInput.tile_state` /
`players_public` / `global_public`; it does **not** reimplement the feature scales or
layout. `StateSnapshot` enforces visible invariants (physical counts, aka bounds, one
dealer/winds, Σscores+1000·kyotaku=100000, ≤5 indicators, own hand ≤14).

## 3. Provenance and fail-closed rules

`StreamMeta` carries `pov`, `projection_version`, `schema_version`, `schema_hash`,
`ruleset_id`, `source_kind`.

`encode_decision` rejects:
- non-`DecisionInput` input;
- `DecisionInput.schema_version != rf-schema-v3/2`;
- `before_event_idx != len(events)` (exact visible prefix required; future events fail);
- non-contiguous `seq_idx` (must be `0..L-1`);
- unregistered event class;
- `L > max_events` (default `4096`) — never silently truncates;
- `L-1 >= 512` kyoku position (raise, never clamp);
- illegal/invalid legal mask or phase;
- `StreamMeta` wrong schema version / schema hash / projection version / POV /
  ruleset / source kind;
- missing `stream_meta` keys.

`StreamMeta.from_mapping` accepts a PR-09 header mapping or a mapping with the five
required fields plus optional `schema_hash` (defaults to the frozen hash; a supplied
mismatch raises). Batch padding values are the PR-13 PAD/UNKNOWN ids; padding slots are
zeroed by the model embedding and cannot affect valid logits.

## 4. Offline/online byte contract

- Offline: `encode_decision(input, meta)` / `encode_batch`.
- Online: `OnlineEncoder.encode(input, header)` / `.encode_batch` delegates to the same
  functions.
- Parity: every tensor field is compared with `torch.equal`; `encoded_hash` must be
  identical. Representative synthetic row:
  `063a0da4f9fc19c09c0f4f6df1bbd11caa8339c1bdcff4fbb346f715a095dd31`
  (3-event prefix, `state_features [172]`, phase 0, decision token 2).
- Real-log parity: **12 logs × 4 POVs × 3 decisions = 144 comparisons**, all four POVs,
  all byte-equal. See `PR11_pytest.log`.

## 5. Tests

### Parity (`tests/rf_test_encoder_parity.py`, 7 passed)

- synthetic offline vs online byte equality (all tensors + canonical hash);
- batch padding vs single-row prefix equality, decision indices, ModelBatch shape;
- 172 state layout, 46 mask, no target/label/rank/oracle fields;
- wrong schema hash/version/projection version fail closed;
- no silent truncation;
- deterministic hash;
- 10+ real logs (12) all four POVs, 144 comparisons.

### Causal (`tests/rf_test_causal.py`, 9 passed)

- appending a future event to a longer stream leaves the earlier encoded row byte-equal;
- batched row-0 logits equal single-row logits (float tolerance 1e-5 due to GEMM
  round-off; encoded arrays are exact);
- changing a longer row's visible events does not move row 0;
- arbitrary padded-slot ids do not change valid logits; illegal probabilities exactly 0;
- hidden opponent hand perturbation leaves the encoded row unchanged;
- own hand change is visible;
- truncation raises;
- PR-10 sample → `ModelBatch` → PR-13 `Policy.forward` (no training); finite logits,
  zero illegal probability, no target fields;
- fixed-kan regression: `DecisionInput.tiles_left == 70 - tsumos in current kyoku`
  and encoded state slot 125 equals that value/70 on 2 real kan games (1238 decisions).

Commands:

```bash
cd /home/ygsldr/riichi-ai/worktrees/encode
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q -s \
  tests/rf_test_encoder_parity.py tests/rf_test_causal.py
```

## 6. Known gaps / follow-ups

- One decision per row; PR-12 may group multiple decisions per longer game row (the
  PR-13 `ModelBatch` already supports multiple decision indices per row; this encoder
  exposes one row per `DecisionInput`).
- `event_phase` is the decision phase at the decision token and `0` (SELF_TURN
  convention) on non-decision tokens; PR-13 fixtures treat phase as a categorical token
  feature, not a decision mask.
- Max default sequence `4096`; per-kyoku position is capped at 512 by the PR-13 vocab.
  A future PR-12 bucket must provide a larger `max_events`/model `max_positions` rather
  than truncate.
- No training, no dataset/run/checkpoint writes. No target/label/rank/oracle is accepted
  or emitted by the encoder.
- Real-log parity uses read-only 2025 logs and skips when the dataset is absent.
- `tiles_left`/state values come from PR-10's `DecisionInput`; the fixed wall value is
  `70 - tsumo events in the current kyoku` (kans do not add a wall tile). No wall RNG/seed
  is encoded.
