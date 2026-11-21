# PR-03/04 — VNext observable encoder and model

**Code**:

* `riichiformer/vnext/observable.py` — current-kyoku BOS tokenization,
  34×26 visible tile features, 38 global features, 37 candidate descriptors,
  `VNextEventStream` / `VNextBatchBuilder`.
* `riichiformer/vnext/model.py` — factorized tile embedding, structured tile
  encoder, decision-query cross-attention, tile-scored 37 + other 9 head,
  optional linear-46, optional state-only and critic head.

## Encoder findings

The existing `DecisionInput` already has the required POV separation.  Its
`events` sequence is the full visible match (V3.3 used it for the full-game
stream); VNext intentionally **filters to the current `kyoku_idx`** and prepends
`[BOS]`.  The first synthetic decision has `start_game` (kyoku −1),
`start_kyoku` (kyoku 0), and an own `tsumo`; VNext emits exactly
`[BOS, start_kyoku, tsumo]` and `prefix_token_offset=2`.

Categorical event features: event type, relative actor, tile kind, aka, public
tsumogiri, intra-kyoku position.  Opponent `tsumo`/`ankan` tiles remain UNKNOWN
and cannot carry aka.  Chi/Pon/Daiminkan/Kakan/Akan consumed tiles remain in the
`DecisionInput` objects.  VNext additionally carries three public consumed-tile
slots per meld token (`event_meld_tile`/`event_meld_aka`, UNKNOWN padded), so a
chi's two consumed tiles and their red-five flags are visible to the event
encoder rather than collapsed to the claimed tile.

Tile state is derived from visible data only: own counts/aka, public counts/aka,
each opponent's river and post-riichi river (from public `dahai`), each
opponent's public melds, visible dora indicators.  `genbutsu` is exactly the
opponent-river indicator; it is not an inference from a hidden hand.  No wall,
opponent hand, ura, final rank or future event enters `x_tile`/`x_global`.

## Model math

* `E(k) = E_suit + E_rank + α E_identity` for suits; honors use a separate
  honor table; aka is a separate residual.  Normal five and aka five are
  distinct candidates (34/35/36).
* `Q = query_mlp([tile_summary, global_vec, phase_embed])`.
* Cross-attention keys/values are the event states of the decision's own
  sequence, masked to `token <= prefix_token_offset` and the valid prefix.
  There is no cross-sequence KV.
* `z = fuse([Q, context, H[offset], global_vec])`.
* Tile head scores all 37 candidates with a phase-conditioned dot product and a
  small residual; other head scores native actions 37..45.  The merge is the
  native 0..36 + 37..45 order, then FP32 `-inf` masking and log-softmax.
* Head/backbone controls are real constructor switches, not branches in the
  data: `cross_attn_layers=0`, `head_type=linear46`, `tile_embedding=identity`,
  `model_mode=state_only`.

## Measured properties

| Item | Value |
|---|---|
| default trainable parameters | **6,538,716** |
| default layers/width/heads | 6 / 256 / 4 |
| CPU forward, 32 decisions × 128 tokens, 4 threads | **2,141 decisions/s**, 273,996 event-tokens/s |
| CPU train step, same synthetic batch | **340 decisions/s**, 43,513 event-tokens/s |
| peak RSS in this microbench | 806.0 MiB |
| device | CPU only; no CUDA in this measured run |

These are synthetic contract-valid batches, not a full-epoch projection and not
a strength result.  Script: `scripts/rf_bench_vnext.py`; raw:
`docs/riichiformer/vnext/logs/VNEXT_CPU_PROFILE.json`.

## Tests (all passing, CPU)

`tests/rf_test_vnext_model.py`:

* BOS/current-kyoku-only stream and p=0-relative offset;
* same prefix with SELF_TURN + KAN_SELECTION (distinct phase/mask);
* 34/26/38 shapes, own aka channel, candidate descriptors;
* two-stream padded batch;
* forward mask/log-softmax;
* future token mutation does not change a past decision;
* shared batch vs single-decision batch;
* decision-order permutation invariance;
* cross-attn=0, linear46, state-only controls;
* factorized vs identity and normal-vs-aka distinction;
* backward reaches actor parameters;
* CPU BF16 autocast keeps the policy output in FP32 and still validates.

`tests/rf_test_vnext_contracts.py` and `tests/rf_test_vnext_runtime.py` cover the
target separation, PT/GRP schemas and runtime reset/act paths.

## NOT_RUN

* Hidden-mutation bit-exact corpus test on the real 2.51M logs.
* Long-kyoku overflow on a real 4,800-event POV.
* GPU/BF16 forward parity and peak VRAM.
* A real VNext native Arena game (no VNext checkpoint yet).
* A full training epoch.
