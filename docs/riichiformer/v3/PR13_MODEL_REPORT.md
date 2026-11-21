# PR-13 — MVP-S Local Transformer (synthetic-only forward/backward deliverable)

```yaml
task_id: PR-13
worktree: /home/ygsldr/riichi-ai/worktrees/model
branch: rf/pr13-model
base_sha: 885b14f   # pr08: freeze schema v3 events, 46-action contract and target isolation
code_commit: b18ae4fc19877e65a1441e31edea173c42036f6d
followup_commit: 8f58ac99f271b36145bd637d2a7a8abbba44425b   # PR-13 review P1-1/P1-2 fix
docs_commit: HEAD of rf/pr13-model (this report)
schema_on_base: rf-schema-v3/1 (schema_hash 0159d99b6193ea7a27c88e28be28c428bca2d2635ba7217f2afbc13f46b6bd2f,
                action_map_hash 6cf1efc5ba99533ec4844f38184ef51a7941174164f9588f21d33ca70a873a2e)
schema_after_merge: rf-schema-v3/2 verified — schema_hash
                    12503d36e2571449d4e4bdd4c8780e00f285d3a2ec81f9e26af184d24c8ef43c,
                    action_map_hash 6cf1efc5ba99533ec4844f38184ef51a7941174164f9588f21d33ca70a873a2e
verdict: PASS (synthetic only; reviewer P1-1 and P1-2 addressed)
non_claims: >-
  This is a synthetic forward/backward deliverable. No dataset was trained on, no
  checkpoint was written, and no model-strength / arena / Tenhou-equivalent claim is
  made. The overfit curve proves the training chain can update the policy; it says
  nothing about playing strength.
```

## 1. Scope

Implemented the complete MVP-S network as self-contained synthetic code, without reading
datasets and without any trainer (PR-14 owns BC training):

| file | role |
|---|---|
| `riichiformer/models/event_embedding.py` | 7 categorical event embeddings summed to `d=256`, padding mask |
| `riichiformer/models/state_encoder.py` | documented 172-float visible-state layout + `172→512→256` MLP |
| `riichiformer/models/local_transformer.py` | 6 Pre-RMSNorm causal blocks, 4 heads, RoPE, SwiGLU-704, SDPA |
| `riichiformer/models/policy.py` | full model, `Linear(256,46)` head, `-inf` legal masking, `PolicyOutput` |
| `riichiformer/models/batch.py` | `ModelBatch` — replaceable synthetic batch contract (PR-13-owned) |
| `configs/riichiformer/mvp_s.yaml` | frozen MVP-S architecture/trainer-runtime settings |
| `tests/rf_test_models.py` | 43 synthetic tests, CPU required, optional GPU BF16 smoke |
| `docs/riichiformer/v3/logs/PR13_pytest.log` | full command/output log with exit codes |

No schema, `data/**`, `replay.py`, canonical/dedup/split, `vendor/mortal`, or `.gitignore`
file was modified. `git diff --stat` for the code commit: 9 files, +2,853 lines (all new).

## 2. Architecture

### 2.1 Event embedding (sum to one `d=256` vector per event token)

| feature | vocabulary | notes |
|---|---:|---|
| `event_type_id` | 17 | schema type ids 0..15 + PAD 16 |
| `event_actor_rel` | 6 | self/shimo/toimen/kami/none = 0..4 + PAD 5 |
| `event_tile_kind` | 35 | 0..33 + UNKNOWN 34 (schema `TILE_KIND_COUNT + 1`) |
| `event_aka` | 4 | 0 = normal/none, 1..3 = red 5m/5p/5s |
| `event_tsumogiri` | 2 | 0/1 |
| `event_phase` | 4 | `SELF_TURN / CALL_RESPONSE / KAN_SELECTION` + PAD 3 |
| `event_kyoku_pos` | 512 | 0-based event ordinal **inside its kyoku**; overflow raises, never clamps |

The tables are summed (not concatenated), LayerNormed, optionally dropout-0, then
multiplied by the bool key-padding mask so padded slots are **exact zero**. Out-of-vocab
ids raise `EventEmbeddingError`; the model path skips the device-synchronising bound
check because `ModelBatch` has already validated every id.

### 2.2 State encoder and decision injection

`StateSnapshot` (synthetic builder input) enforces the visible-state invariants: physical
tile overflow, aka ≤ five count, aka ≤ 1 per colour, one dealer matching `oya_rel`, wind
consistency, `Σ scores + 1000·kyotaku = 100000`, 1..5 dora indicators, at most 14 own tiles.
It exposes **no** label/rank/oracle/opponent-hidden field.

Flat feature vector `F_state = 172`:

| segment | dim | normalization |
|---|---:|---|
| own tile counts | 34 | count / 4 |
| public tile counts | 34 | count / 4 |
| own aka | 3 | 0/1 |
| public aka | 3 | 0/1 |
| 4 players | 44 | score_rel (clip ±3σ), is_oya, riichi one-hot 3, river/70, melds/4, wind one-hot 4 |
| global | 16 | bakaze 4, kyoku/4, honba/255, kyotaku/255, tiles_left/70, self-wind 4, oya_rel 4 |
| dora | 35 | indicator counts/4 + num/5 |
| phase | 3 | one-hot |

`StateMLP: 172→512→256`, GELU, output LayerNorm. Its `[N,256]` output is added only at
each decision's `(batch, token)` position via a flattened `index_add`; every other token
stays untouched. Because attention is causal, a decision's state can influence later
tokens of the same POV but never earlier ones (tested).

### 2.3 Local transformer

* 6 blocks × Pre-RMSNorm residuals: `x = x + attn(norm(x)); x = x + ffn(norm(x))`.
* 4 heads, head dim 64, RoPE (`base=10000`, non-persistent cos/sin cache).
* SwiGLU `w2(silu(w1 x) * w3 x)`, hidden 704.
* `F.scaled_dot_product_attention`; dropout is 0.0 in config, so eval/train parity is exact.
* `[B,L,d]` in/out. Full or left-aligned masks use SDPA's causal fast path; arbitrary/holey
  key masks get an explicit causal ∧ padding boolean mask (pad keys get zero weight).
* `L > max_positions (4096)` raises `TransformerShapeError`; there is no truncation.
* No memory/cross-kyoku module (`memory: false`).

### 2.4 Policy head

`Linear(256,46)` → `masked_logits = logits.masked_fill(~legal_mask, -inf)` →
`logprobs = log_softmax(masked_logits)` → `PolicyOutput(logits, masked_logits, logprobs,
selected_decision_offsets, legal_mask)`. `selected_decision_offsets` are flat `B*L`
offsets. Illegal actions have exact zero probability; each row's legal log-probs normalize
to 1 (`PolicyOutput.validate()`). `forward(batch)` accepts only `ModelBatch`; labels,
targets, rank and oracle tensors have no path into the network.

`ModelBatch` fields: seven event-id tensors `[B,L]`, `padding_mask [B,L]`, `state_features
[N,F]`, `decision_batch_index/decision_time_index [N]`, `decision_phase [N]` (0..2),
`legal_mask [N,46]`, and a non-tensor `schema_version` metadata string defaulting to
`action_v3.SCHEMA_VERSION`. Padding must be a left-aligned True prefix. Legal masks are
validated against the frozen `PHASE_LEGAL_ACTIONS`. It imports no PR-11 `data/encode.py`
code and can be replaced when PR-11 lands.

### 2.5 POV visibility validation (review P1-1)

`ModelBatch.validate_visibility()` is executed by `__post_init__` **and** again by
`Policy.forward` (tensors are mutable, so an in-place mutation must not bypass it). It is
token-local, because the model consumes exactly one projected `(tile_kind, aka)` per
event token:

* opponent `tsumo` (`actor_rel != 0`) must be `tile_kind=UNKNOWN(34)` and `aka=0`;
* opponent `ankan` is concealed for the same reason (`UNKNOWN`, `aka=0`); a
  `daiminkan`/`kakan` keeps its public called/added tile and own `ankan` may be known;
* `UNKNOWN` cannot carry an aka, and aka 1/2/3 must be exactly 5m/5p/5s (schema kinds
  4/13/22) with the matching colour; `aka=0` means no red five;
* padding slots are ignored (they are not observable events);
* `ModelBatch` has no label/target/oracle field, and any future such field is rejected.

`ModelBatch` deliberately carries **no** `consumed` tensors: consumed/anka aggregation is
PR-09/PR-11's responsibility, and the model never sees it. If a future encoder adds consumed
ids they must obey the same UNKNOWN rules. Regression tests construct opponent-known
tsumo/ankan, UNKNOWN-with-aka, mismatched-aka and padded-slot batches and expect
`ModelBatchError`, while a hand-built PR-09-projected-like fixture (public calls, hidden
opponent draw/ankan, own aka draw, dora marker) forwards successfully.

## 3. Parameter count (measured, `Policy(MVPConfig())`)

| module | parameters |
|---|---:|
| event_embedding (7 tables 580×256 + LN) | 148,992 |
| state_encoder (172→512→256 + LN) | 220,416 |
| transformer (6 × 806,016 + final RMSNorm 256) | 4,836,352 |
| policy_head (256×46 + 46) | 11,822 |
| **total** | **5,217,582** |

Per transformer block: qkv 197,376 + out 65,792 + SwiGLU 542,336 + 2 RMSNorm 512.
Frozen in the test as `EXPECTED_MVP_S_PARAMETERS = 5_217_582`, with a 4.0M–6.5M budget
guardrail in `policy.py`. The test suite also asserts the breakdown sums to the total.

## 4. Overfit evidence (fixed synthetic batch)

`Policy(MVPConfig())`, fixed batch `B=6, L=64, N=12`, AdamW lr 3e-3, 200 steps, CPU,
labels are a deterministic legal function of the state features. Exact command in §6.
Raw curve from `PR13_pytest.log`:

| step | loss | argmax accuracy |
|---:|---:|---:|
| 0 | 1.140916 | 0.833 |
| 20 | 1.6153e-05 | 1.000 |
| 40 | 7.7684e-06 | 1.000 |
| 80 | 4.4107e-06 | 1.000 |
| 120 | 3.3180e-06 | 1.000 |
| 160 | 2.6524e-06 | 1.000 |
| 199 | 2.2550e-06 | 1.000 |

Initial loss 1.1409 → final 2.2550e-06; all labels memorized. This demonstrates the
forward/backward chain learns; it is not a strength result.

## 5. Throughput / memory (single short synthetic run)

Benchmark batch `B=32, L=256, N=128` (8,192 event tokens, 128 decisions). CPU = 14 logical
CPUs, default torch threads; GPU = RTX 5070 Ti 15.9 GiB, FP32 forward / BF16 autocast train
step. Numbers are the measured log values; a short unscaled smoke, not a throughput claim.

| device | mode | seconds | tokens/s | decisions/s |
|---|---|---:|---:|---:|
| CPU | forward (no grad) | 0.34806 | 23,536.1 | 367.8 |
| CPU | train step (fwd+bwd+AdamW) | 0.97393 | 8,411.3 | 131.4 |
| CUDA BF16 | forward (FP32 weights) | 0.00882 | 928,924.0 | 14,514.4 |
| CUDA BF16 | train step (fwd+bwd+AdamW) | 0.08173 | 100,237.4 | 1,566.2 |

Peak GPU memory for the benchmark: **863.5 MiB allocated / 1,032.0 MiB reserved**
(`torch.cuda.max_memory_allocated/reserved` after the B=32 bench). The optional one-step
BF16 test smoke (B=2, L=64, 2 layers) peaks at **33.0 MiB**. Small-batch forward timings
varied materially across short runs (CUDA launch/DVFS noise); train-step throughput was
stable, so both are reported without extrapolation. No long GPU run was performed.

## 6. Exact commands

```bash
cd /home/ygsldr/riichi-ai/worktrees/model

# tests (CPU required; GPU smoke runs when CUDA is available, otherwise skipped)
PYTHONPATH= /home/ygsldr/riichi-ai/.venv/bin/python -m compileall -q riichiformer/models tests/rf_test_models.py
PYTHONPATH= /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q -s tests/rf_test_models.py

# CPU bench + overfit curve
PYTHONPATH= /home/ygsldr/riichi-ai/.venv/bin/python tests/rf_test_models.py bench \
  --device cpu --batch-size 32 --seq-len 256 --decisions-per-row 4 --iters 5

# short BF16 GPU bench (no overfit, a few seconds)
PYTHONPATH= /home/ygsldr/riichi-ai/.venv/bin/python tests/rf_test_models.py bench \
  --device cuda --batch-size 32 --seq-len 256 --decisions-per-row 4 --iters 5 --skip-overfit

# regression sweep
PYTHONPATH= /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q tests/rf_test_*.py

# merged schema v3/2 verification (rf/v3 = cc699f1, PR-13 files copied in)
rm -rf /tmp/pr13-v32 && mkdir -p /tmp/pr13-v32
git archive cc699f1 | tar -x -C /tmp/pr13-v32
cp -a riichiformer/models /tmp/pr13-v32/riichiformer/
cp -a tests/rf_test_models.py /tmp/pr13-v32/tests/
cp -a configs/riichiformer/mvp_s.yaml /tmp/pr13-v32/configs/riichiformer/
cd /tmp/pr13-v32
PYTHONPATH= /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q tests/rf_test_models.py
```

Full output with exit codes: `docs/riichiformer/v3/logs/PR13_pytest.log`.

## 7. Test results

`tests/rf_test_models.py`: **51 passed, 15 subtests passed, 0 failed, 0 skipped** (26.34 s);
the GPU BF16 smoke executed (CUDA available) and printed `peak allocated VRAM: 33.0 MiB`.
Compileall exit 0. The same 51 tests pass on the merged `rf/v3` schema-v3/2 tree
(`cc699f1` + these files): **51 passed, 15 subtests, 26.74 s**, exit 0, with
`schema_version=rf-schema-v3/2`, `schema_hash=12503d36e257…`, action map unchanged.

Coverage against the PR-13 acceptance list:

| requirement | test(s) |
|---|---|
| batch / `[B,L,d]` shapes | `TestPolicyForward::test_shapes_and_output_contract`, `TestLocalTransformer::test_shape_preserved` |
| illegal action probability = 0 | `test_illegal_actions_have_zero_probability`, `test_single_legal_action_probability_one`, `PolicyOutput.validate()` |
| causal prefix invariance | `test_causal_prefix_invariance` (full L=80 vs prefix L=60; logits numerically equal within atol 1e-5, not a bitwise promise), `TestLocalTransformer::test_causal_future_independence` |
| POV sequences do not share state | `test_different_pov_sequences_do_not_share_state` (batched vs single + one-POV mutation), `test_select_rows_keeps_only_that_pov` |
| decision-local state injection | `test_state_injection_is_decision_local_and_causal` |
| all gradients finite | `test_cpu_forward_backward_all_gradients_finite` (full MVP-S, every parameter, grad norm) |
| strict `state_dict` roundtrip | `test_state_dict_strict_roundtrip` (missing/extra key raises) |
| parameter count | `test_parameter_count_matches_budget` (exact 5,217,582 + breakdown) |
| synthetic overfit | `test_synthetic_overfit` (loss ≤ 25 % of initial, argmax accuracy 1.0) |
| CPU forward/backward | `test_cpu_forward_backward_all_gradients_finite`, `test_cpu_optimizer_step_changes_weights` |
| optional BF16 GPU step | `TestGPUSmoke::test_one_step_bf16_gpu` (skipped without CUDA) |
| fail-closed contracts | prefix padding, phase masks, event ranges, NaN state, duplicate decisions, overlong sequence, out-of-vocab ids |
| POV visibility | `test_visibility_rejects_opponent_known_tsumo`, `test_visibility_rejects_unknown_or_mismatched_aka`, `test_visibility_rejects_opponent_known_ankan`, `test_visibility_allows_public_tiles_and_own_private_tiles`, `test_visibility_ignores_padding_slots`, `test_pr09_like_pov_batch_forwards`, `test_forward_rechecks_visibility_after_in_place_mutation` |
| no target/oracle input | `test_no_target_or_oracle_fields` (fields + forward signature + parameter names), enforced by `validate_visibility()` |
| frozen schema compatibility | `test_yaml_matches_dataclass_and_frozen_hashes` (runtime `load_schema()` + phase sets), merged v3/2 run |

### Regression sweep

`pytest -q tests/rf_test_*.py`: **311 passed, 21 skipped, 25 failed** (38.42 s). All 25
failures are `tests/rf_test_ruleset_parity.py::SourceContractTests`, which read
`vendor/mortal/libriichi/src/arena/board.rs`; that submodule directory is uninitialised in
this worktree, so the file is absent at BASE as well. PR-13 adds only new model/config/test
files and touches no `vendor/` or schema file, so this is an environment limitation, not a
PR-13 regression.

## 8. Schema-version note (review follow-up, verified on merged v3/2)

This worktree is on BASE `885b14f` with `rf-schema-v3/1`. The merged mainline schema is
`rf-schema-v3/2` (`rf/v3` = `cc699f1`) with
`schema_hash = 12503d36e2571449d4e4bdd4c8780e00f285d3a2ec81f9e26af184d24c8ef43c` and an
**unchanged** action map (`6cf1efc5…`). The 51-model-test suite was re-run on that merged
tree with this branch's files copied in and passed (see §7/§8 log section 5). PR-13 does not
hardcode either value:

* `ModelBatch.schema_version` defaults to `riichiformer.schemas.action_v3.SCHEMA_VERSION`;
* `configs/riichiformer/mvp_s.yaml` has no literal schema version/hash — it points at the
  code constant and `schema_v3.load_schema()`, which verifies all three hashes;
* the test loads `schema_v3.load_schema()` and asserts it agrees with `action_v3`, and the
  tamper test uses a deliberately invalid suffix rather than a version literal.

No architecture or action-space change was made. A future schema bump needs no PR-13 edit;
only PR-11/PR-14 consumers of the hashes need to carry the new values.

## 9. P0 / P1 / blockers

**P0:** none.

**P1 (post-review):**
1. **Addressed** — P1-1 POV visibility: `ModelBatch.validate_visibility()` now rejects
   opponent-known tsumo/ankan, UNKNOWN-with-aka, and mismatched aka kinds, re-runs in
   `Policy.forward`, and is covered by 8 regression tests. `ModelBatch` intentionally has
   no `consumed` ids; if PR-11 adds them they must follow the same UNKNOWN rules.
2. **Addressed** — P1-2 wording: prefix invariance is documented and commented as
   numerically equal within `atol=1e-5`, not bitwise.
3. **Addressed** — merged schema v3/2 verified locally via `rf/v3` archive (`cc699f1`) with
   this branch's files: 51 passed, 15 subtests, 26.74 s.
4. Throughput is a single short synthetic measurement with default torch threads; it is not
   a training-budget forecast. The state/event encoders are not yet wired to PR-11.
5. `ModelBatch.validate()` runs range/finiteness/visibility checks at construction for
   fail-closed behaviour; PR-14 may want a fast-path toggle once dataloaders are trusted.

**Blockers:** none for PR-13. Full-suite green requires initialising the `vendor/mortal`
submodule (pre-existing environment issue). PR-14 needs the real PR-11 encoder and the
source-rights gate; PR-16 needs H-04 ruleset clearance before any strength claim.

## 10. Verdict and next gates

PR-13 acceptance: **PASS** on its stated scope — forward/backward is stable on CPU, the
optional 1-step BF16 GPU smoke passes, shapes/masks/causality/no-leak/visibility invariants
are tested, the parameter count is 5,217,582, and the synthetic overfit proves the training
chain can learn. Reviewer P1-1 (visibility) and P1-2 (wording) are addressed; the suite was
re-run on the merged schema v3/2 tree. No model-strength or arena statement is made.

Allowed next PRs: PR-14 (BC trainer + checkpoint protocol) with this model config and the
`ModelBatch` contract, after the merged schema v3/2 verification; PR-11 can replace
`ModelBatch`/`StateSnapshot` construction without touching the network.
