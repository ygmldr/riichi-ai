# PR-38 / PR-39 implementation record

Task: RiichiFormer V3.2 Wave 1, Dev-COMPUTE.
Base SHA: `cbf166a356f83c739b31f0b5a0b241b954ef9e0a`
Branch: `rf/v32-compute` (worktree `worktrees/v32-compute`).

Scope of this record: the multi-decision shared forward (PR-38), the numerical /
config / recovery fixes (PR-39), the CPU-only benchmark, and every limitation that
was **not** measured. No GPU run, no `scripts/rf_train.py` change, no
`data/loader.py` / `data/shard.py` change, no vendor change.

Files:

| file | change |
|---|---|
| `riichiformer/data/multi_decision.py` (new) | `DecisionMarker`, `MultiDecisionStream`, `streams_from_samples`, profiler |
| `riichiformer/data/encode.py` | shared event mapping, `encode_multi_decision`, `encode_multi_decision_batch`, payload/hash |
| `riichiformer/models/batch.py` | `MultiDecisionBatch`, shared event/visibility/mask validators |
| `riichiformer/models/local_transformer.py` | optional explicit `attn_mask` + `position_ids` (legacy path unchanged) |
| `riichiformer/models/policy.py` | `Policy.forward_multi`, `multi_decision_attention_mask`, `PolicyOutput` metadata |
| `riichiformer/train/bc.py` | exact window-normalized gradient accumulation |
| `riichiformer/train/model_config.py` (new) | `resolve_mvp_config` (MVP-S default, explicit tiny smoke) |
| `riichiformer/train/checkpoint.py` | config/model-config/provenance checks, `strict_resume`, `init_weights_only` |
| `tests/rf_test_multi_decision.py` (new) | encoding/parity/extreme tests + CPU benchmark print |
| `tests/rf_test_training.py` | accumulation/config/resume/init-weights tests |
| `tests/rf_test_encoder_parity.py` | offline/online multi-decision mapping parity |

---

## 1. PR-38 — multi-decision shared forward

### 1.1 Problem and measured recomputation

`encode_batch` turns one `(game, POV)` stream into one **prefix row per decision**;
`Policy.forward` then recomputes the shared history for every row. A real 2025 log
(H-02 POV 1) has 158 decisions on a 1,000-event stream. The legacy path therefore
runs 77,171 event tokens through the transformer; a single full-stream pass needs
1,000 events + 158 markers = 1,158 tokens (66.6x fewer). The profiler
(`prefix_recompute_profile`) reports this per stream and is asserted in
`test_profiler_counts_recomputation`.

Real H-02 log `2025010100gm-00a9-0000-029e0f35.mjson`, profiler output (all four
POVs of the game):

| POV | events | decisions | prefix tokens | shared tokens | ratio | duplicate cutoffs |
|---|---|---|---|---|---|---|
| 0 | 1,001 | 157 | 80,370 | 1,158 | 69.4x | 0 |
| 1 | 1,000 | 158 | 77,171 | 1,158 | 66.6x | 0 |
| 2 | 994 | 150 | 74,011 | 1,144 | 64.7x | 0 |
| 3 | 996 | 154 | 82,137 | 1,150 | 71.4x | **1** (kan selection) |
| **game** | **3,991** | **619** | **313,689** | **4,610** | **68.0x** | 1 |

The POV 3 duplicate cutoff is a real kan-selection opportunity: the explicit
marker design (one slot per decision) is required, a cursor-only layout would
have to drop or merge one of the two decisions.

### 1.2 Types and the DECISION marker

* `MultiDecisionStream(events, decisions, meta)` — one complete causal POV event
  main sequence plus N decisions. A decision is either a PR-10 `DecisionInput`
  (its exact visible prefix must end at its cutoff) or a normalized
  `DecisionMarker(cutoff_idx, phase, legal_mask, state_features)`. The stream
  validates event contiguity, exact event classes, non-decreasing cutoffs and that
  each decision's prefix is the stream prefix (`verify="exact"` in
  `streams_from_samples`).
* `MultiDecisionBatch` (models/batch.py) — `[B, L]` event tensors (exact
  `ModelBatch` layout) plus per-decision `state_features [N,172]`,
  `legal_mask [N,46]`, `decision_phase [N]`, `decision_batch_index`,
  `decision_marker_index`, `decision_cutoff_idx`. `event_phase` is **0 on every
  real event token**: a decision's phase belongs to its marker, so the shared
  stream stays identical for all decisions. Marker slots are contiguous per row
  and cutoffs are non-decreasing; cutoffs must point at non-padded tokens.
* `Policy.forward_multi(batch)` — one `LocalTransformer` pass over
  `[B, L + M]` where `M = max marker index + 1`. Each marker

  1. reuses the event embedding of token `c - 1` with **its own phase** substituted
     (the legacy forward put the decision phase on that same last token),
  2. adds that decision's state encoding (legacy added it to the same token),
  3. sits at RoPE position `c - 1`, and
  4. attends only to events `0 .. c - 2` plus itself.

  Event queries attend causally among events and **never** to marker keys; marker
  queries never attend to another marker. This is the "provably equivalent cursor
  token" of the roadmap: for every decision the marker hidden state is exactly the
  legacy prefix hidden state, while the event computation is shared. It also
  handles **duplicate cutoffs** (a `KAN_SELECTION` opportunity shares its cutoff
  with the action it re-labels), which a cursor-only design cannot express.
* `multi_decision_attention_mask(batch)` exposes the block mask for review/tests.

No target, label, rank, oracle or censor field exists in any forward input:
`MultiDecisionBatch` field names and the `encode_multi_decision` signature are
asserted against the same forbidden-word list as `ModelBatch`.

### 1.3 Changes to existing modules (additive only)

* `LocalTransformer.forward(x, padding_mask, *, attn_mask=None, position_ids=None)`:
  an explicit boolean mask (True = may attend) and per-token RoPE positions. The
  legacy call path (`attn_mask=None`, no positions) is byte-for-byte the previous
  code; all 51 existing model tests still pass.
* `PolicyOutput` gained optional `selected_marker_offsets` / `decision_cutoff_idx`
  (None on the legacy path).
* `Policy.forward` signature is unchanged (`["self", "batch"]`, asserted).

### 1.4 Parity evidence

Same model, dropout 0, eval (and deterministic train), CPU float32; the reference
is the **existing** `Policy.forward` on `encode_decision(...).to_model_batch()`:

| check | measured | tolerance |
|---|---|---|
| joint vs per-prefix logits (9 decisions) | max abs **7.15e-07** | atol 1e-5, rtol 1e-5 |
| joint vs per-prefix gradients (weighted loss) | max abs **1.19e-07**, max rel **7.77e-07** | atol 2e-6, rtol 2e-5 |
| kan duplicate-cutoff logits / gradients | **7.15e-07 / 2.09e-07** | same |
| single-decision joint vs legacy | allclose 1e-5 | same |

The residual is SDPA reduction order (the legacy path uses the fast causal kernel,
the joint path an explicit mask); it is not a semantic difference. Bitwise equality
is deliberately **not** claimed. With `dropout > 0` the two paths consume different
RNG draws and are not comparable; MVP-S freezes `dropout = 0.0`.

### 1.5 Causal / leakage / extreme tests

* future-event mutation after a decision's cutoff leaves that decision's logits
  bitwise unchanged (`torch.equal`); later decisions on the same row may change;
* changing one decision's `state_features` changes **only** that decision
  (bitwise check across all rows);
* opponent concealed hand variants produce identical stream hashes for POVs that
  do not own the changed hand; revealing an opponent `tsumo` tile / an aka on
  UNKNOWN is refused by `MultiDecisionBatch.validate_visibility`;
* fixtures: pon/Pass (`CALL_RESPONSE`, action 45), call-preempted censored
  decision, kan selection with duplicate cutoff, two-kyoku streams (kyoku_pos
  resets), all-padded/first-event cutoffs, a 1-event/1-decision stream and a
  400-event/3-decision stream in one batch, joined length > `max_positions`
  refused (never truncated);
* offline/online mapping: `encode_multi_decision` (single) and
  `encode_multi_decision_batch` produce the same hash; each decision's prefix
  tensors equal `encode_decision`'s row.

---

## 2. PR-39 — accumulation, model config, recovery

### 2.1 Exact gradient accumulation (`train/bc.py`)

The old `training_step` backwarded each microbatch's **own mean** loss, so
`accumulation_count > 1` averaged means instead of normalizing by the global
weight. The fix uses the additive numerator:

```
numerator_k = sum_{i in microbatch k} w_i * -log p_i
window_total = sum_{k in window} sum_i w_i        # from target metadata, before forward
backward(numerator_k / window_total)              # or numerator_k * loss_scale
```

* `window_weight_total(targets_per_step)` / `valid_weight_sum(targets)` expose the
  denominator; `training_step(..., accumulation_weight_total=...)` uses it;
  `loss_scale` is the explicit alternative (mutually exclusive).
* `accumulation_count > 1` without either fails closed (`TrainingError`), so the
  bug cannot silently return.
* An all-censored microbatch performs no backward but still triggers the window's
  optimizer/scheduler step when earlier microbatches produced gradients; a fully
  censored window never steps and never touches the scheduler.
* `BCLossResult` now carries the graph-connected `numerator` and `total_weight`.

Measured on synthetic shards (6 games), 3 microbatches with unequal weights
(2.0/0.5/1.0/1.5), one forced action and one all-censored microbatch, compared
with one concatenated large batch:

| check | measured | tolerance |
|---|---|---|
| loss `sum(numerator)/window_total` | equal to 6 dp | 1e-6 |
| gradient sum vs big batch | max abs **5.96e-08** | 1e-6 |
| parameter delta after one AdamW step | max abs **2.20e-06** on a 1e-3 update (0.22%) | 1e-5 |

The parameter-delta residual is AdamW's per-coordinate normalization (`g/(sqrt(v)+eps)`)
amplifying sub-1e-7 gradient differences at near-zero coordinates; the gradient-sum
test is the tight equivalence assertion. A separate `loss_scale = 1/window_total`
path is tested to the same tolerance.

### 2.2 Model config (`train/model_config.py`)

* `resolve_mvp_config()` loads `configs/riichiformer/mvp_s.yaml` and verifies it
  still builds the frozen six-layer MVP-S: layers 6, **5,217,582 parameters**
  (measured by instantiating the model). A drifted default is an error.
* `resolve_mvp_config(tiny_smoke=True)` returns the frozen one-layer smoke config
  (502,510 parameters) and records `tiny_smoke=True`. It is never implicit and
  cannot be combined with an explicit config path. The default path is asserted to
  return >4M parameters.
* An explicit `model_config_path` is allowed for intentional variants and records
  its own parameter count/hash.
* `ResolvedModelConfig.as_provenance()` yields
  `{model_config_name, model_config_hash, model_parameters, model_tiny_smoke}`
  for checkpoint `extra`/provenance.
* `scripts/rf_train.py` was **not** changed by this lane; the Lead wired the library
  API in `c426ac7` (see §6). The default is now six-layer MVP-S, not `--layers 1`.

### 2.3 Checkpoint (`train/checkpoint.py`)

* `save_checkpoint(..., model_config=..., provenance=...)` stores
  `model_config` + `model_config_hash`, an optional validated provenance block
  (source id, split manifest, canonical manifest/index, research policy/hash,
  rights status) and keeps the legacy format string and three-hash contract.
* `load_checkpoint(..., expected_config, expected_config_hash,
  expected_model_config, expected_model_config_hash, expected_provenance,
  strict_resume=True)` verifies the stored training config against its own hash,
  the model config, and provenance (new block first, legacy hashes fallback).
  Strict resume additionally requires optimizer/scheduler/EMA/RNG/counters to be
  present instead of silently skipping them.
* `init_weights_only(path, model=..., ...)` is a separate function: model
  `state_dict(strict=True)` only — no optimizer, scheduler, scaler, EMA or RNG
  restore, no counter advance. Tests assert the RNG state is untouched, the
  optimizer state stays empty, and a mismatched architecture/config is refused.
* Old checkpoints (no model_config/provenance) still load; asking for a model
  config from one fails closed.

---

## 3. Benchmark (CPU only; no GPU was run)

Environment: same machine as the Lead acceptance, CPU (WSL2), Python 3.12,
torch 2.11.0+cu128, `PYTHONPATH=vendor/mortal/target/release`, full MVP-S
5,217,582 parameters, dropout 0, `torch.no_grad()` for forward-only numbers.
Real log: first suitable 2025 H-02 file, POV 1, 1,000 events, 158 decisions.
Single-run CPU numbers; expect roughly ±20% process-to-process noise.

### 3.1 Online-style forward (one decision per forward vs one shared pass)

Reproduce: `pytest -q tests/rf_test_multi_decision.py -k cpu -s` prints
`PR38_BENCH_JSON`.

| metric | legacy per-prefix | shared `forward_multi` | ratio |
|---|---|---|---|
| event tokens through the transformer | 77,171 | 1,158 | 66.6x fewer |
| forward time | 4.48 s | 0.067 s | 66.6x faster |
| tokens/s | 17,230 | 17,230 | same per-token cost |
| decisions/s | 35.3 | 2,350.9 | 66.6x |

(The token rate being equal is the sanity check: the saving is exactly the
recomputation, not a cheaper token.)

### 3.2 Training step (forward + backward), batched legacy vs shared

Legacy = `encode_batch(decisions)` → `[N, Lmax]` `ModelBatch` → `Policy.forward`
→ backward. Shared = `encode_multi_decision_batch([stream])` →
`forward_multi` → backward. Loss is a synthetic `logits.sum()/N` (timing only).
Manual run, same log, first 20 / 40 decisions of the POV:

| decisions | legacy tokens | shared tokens | legacy train | shared train | speedup |
|---|---|---|---|---|---|
| 20 | 1,135 | 131 (8.66x fewer) | 0.310 s | 0.027 s | 11.5x |
| 40 | 4,574 | 274 (16.7x fewer) | 1.443 s | 0.053 s | 27.0x |

Decisions/s: 64.6 → 740.2 (20) and 27.7 → 749.6 (40). Token rate 3,170–3,665 →
4,848–5,135 tokens/s (single-row batches are inefficient for the legacy path,
which is why the speedup exceeds the token ratio at 20 decisions).

### 3.3 CPU memory (peak RSS delta, separate processes, 40 decisions)

| path | sequence length | tokens | peak RSS delta |
|---|---|---|---|
| legacy batched backward | 234 | 4,574 | +1,615.9 MB |
| shared backward | 274 | 274 | +97.5 MB |

Base RSS (~730 MB) includes torch/libriichi/dataset imports. **VRAM was not
measured** (no GPU run); the GPU token/VRAM comparison is Lead-owned next.

---

## 4. API diff (new public surface)

```python
# data/multi_decision.py
DecisionMarker(cutoff_idx: int, phase: ActionPhase, legal_mask: tuple[bool, ...],
               state_features: Tensor)
MultiDecisionStream(events, decisions, meta)
MultiDecisionStream.from_decision_inputs(events, decisions, meta)
MultiDecisionStream.from_samples(samples) -> MultiDecisionStream
streams_from_samples(samples) -> list[MultiDecisionStream]
prefix_recompute_profile(streams) -> dict
check_event_sequence(events); check_prefix(prefix, events, cutoff, verify=...)

# data/encode.py
decision_marker_from_input(decision) -> DecisionMarker
encode_multi_decision(game_events, decisions, stream_meta, *, max_events=4096,
                      verify_prefix="structural") -> MultiDecisionBatch
encode_multi_decision_batch(streams, *, max_events=4096,
                            verify_prefix="structural") -> MultiDecisionBatch
multi_decision_payload(batch) -> dict
multi_decision_hash(batch) -> str

# models/batch.py
MultiDecisionBatch(...)  # see class docstring; validate()/validate_visibility()/to()/select_rows()
validate_event_block, validate_event_visibility, validate_phase_legal_masks

# models/local_transformer.py (additive keyword-only)
LocalTransformer.forward(x, padding_mask, *, attn_mask=None, position_ids=None)

# models/policy.py
Policy.forward_multi(batch: MultiDecisionBatch) -> PolicyOutput
multi_decision_attention_mask(batch) -> Tensor
PolicyOutput.selected_marker_offsets / PolicyOutput.decision_cutoff_idx

# train/bc.py
training_step(..., accumulation_step=0, accumulation_count=1,
              accumulation_weight_total=None, loss_scale=None)
valid_weight_sum(targets) -> float
window_weight_total(targets_per_step) -> float
BCLossResult.numerator

# train/model_config.py
resolve_mvp_config(model_config_path=None, *, tiny_smoke=False, repo_root=None,
                   verify_parameters=True) -> ResolvedModelConfig

# train/checkpoint.py
save_checkpoint(..., model_config=None, provenance=None)
load_checkpoint(..., expected_config=None, expected_config_hash=None,
                expected_model_config=None, expected_model_config_hash=None,
                strict_resume=False)
init_weights_only(path, *, model, expected_hashes=None,
                  expected_model_config=None, expected_model_config_hash=None)
config_hash(config) / canonical_config_dict(config)
```

No existing function signature or checkpoint format string was removed or renamed.

---

## 5. Not run / limitations (explicit)

* **No GPU run, no VRAM measurement, no GPU throughput claim.** Lead owns that.
  Note for the GPU parity check: CUDA SDPA may select a non-deterministic
  flash-attention backward kernel (torch warns about it under
  `use_deterministic_algorithms(..., warn_only=True)`); the CPU parity numbers here
  are not a promise of bitwise GPU parity. Either disable flash SDP for the parity
  run or use the documented atol/rtol.
* `scripts/rf_train.py` was not changed by this lane; the Lead wired
  `--model-config` / `--tiny-smoke` / `--init-weights-only` in `c426ac7` (see §6).
  The CLI gate is closed for the default/explicit paths; CDUA parity remains NOT_RUN.
* No trainer loop consumes `forward_multi` yet: `data/loader.py` / `data/shard.py`
  are PR-37/Lead-owned and the multi-decision data contract needs their freeze
  (roadmap Wave 2). The PR-38/39 tests exercise the model/encoder API directly.
* Parity is exact only at `dropout = 0`; MVP-S freezes dropout 0. A future
  dropout variant must document that joint and prefix forwards consume different
  RNG and are only equal in expectation.
* `forward_multi` requires `L + markers <= max_positions`; it refuses, never
  truncates. No sliding-window/chunked variant is provided.
* CPU benchmark is forward/backward timing with a synthetic loss; no real training
  loss/optimizer convergence claim.
* No strength claim of any kind (BC loss and throughput are not playing strength).

## 6. Test commands and exit codes

```bash
export PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release
.venv/bin/python -m pytest -q tests/rf_test_multi_decision.py        # exit 0
.venv/bin/python -m pytest -q tests/rf_test_training.py              # exit 0
.venv/bin/python -m pytest -q tests/rf_test_encoder_parity.py        # exit 0
.venv/bin/python -m pytest -q tests/rf_test_models.py                # exit 0
.venv/bin/python -m pytest -q \
  $(git ls-files 'tests/rf_test_*.py' | tr '\n' ' ')                 # exit 0
```

Final full-suite result (this worktree, after all edits):

```
533 passed, 2 skipped, 242 subtests passed in 198.76s, exit 0
```

The two skips are the pre-existing explicit skips in `rf_test_ruleset_parity.py`.
The suite must be re-run by the Lead after merging PR-37, whose `rf_train.py`
default model is the remaining P1.

## 6. Lead integration update (2026-11-21, `c426ac7`)

The lane report above correctly says the PR-14 CLI was not changed by Dev-COMPUTE. The
Lead subsequently wired the library APIs into `scripts/rf_train.py`:

- default model is now `configs/riichiformer/mvp_s.yaml` (six layers, **5,217,582**
  parameters), verified at startup and recorded as `model_config_hash`;
- `--tiny-smoke` is the only way to select the 502,510-parameter smoke model;
- `--model-config` / architecture overrides are mutually exclusive and explicit;
- `--init-weights-only` is separate from `--resume` and never restores
  optimizer/scheduler/RNG;
- strict resume checks the model-config hash and the source/split/policy/canonical
  provenance, and a split mismatch exits 1 without writing a checkpoint.

This does **not** change the remaining limitation: `MultiDecisionBatch` is still not
consumed by `data/loader.py` or the trainer loop, and CUDA parity/VRAM remains unmeasured.
