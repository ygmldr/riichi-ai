# PR-01 — V3.3 P1 fixes

**Owner**: Lead / Runtime lane
**Base**: `8a915d1`
**Tests**: `tests/rf_test_vnext_pr01.py`, `tests/rf_test_m5d_arena.py -k "NativeContextTests or OutputGuardTests or PovSafetyTests"`

## 1. EMA Arena must load `ema.shadow`

**Before**: `RFPolicyBackend.from_checkpoint(..., use_ema=True)` selected
`payload["ema"]` directly.  A `ModelEMA.state_dict()` is
`{"decay", "updates", "shadow": {...}}`, so loading it as a model state dict
could only fail or, worse, be silently bypassed by a Raw fallback.

**After**:

```python
if use_ema:
    state = _ema_shadow(payload)
else:
    state = payload.get("model")
```

`_ema_shadow` requires a mapping `payload["ema"]["shadow"]` and raises
`BridgeError` when it is absent or empty.  There is no Raw fallback.

**Evidence**: `rf_test_vnext_pr01.EMAReadTests` constructs a real checkpoint
with a deliberately different raw/shadow weight, asserts that the EMA backend
reads exactly the shadow tensor, and asserts that a raw-only checkpoint fails
closed when EMA is requested.

## 2. Multi-kan must be a real second policy decision

**Before**: `decode_native_action(42, ...)` replaced a multi-candidate context
with `kan_options[0]`, silently choosing the first ankan/kakan.

**After**:

* action 42 with more than one concrete candidate raises
  `KanSelectionRequired` carrying the options;
* the arena then builds a `DecisionInput` in `ActionPhase.KAN_SELECTION` with a
  tile-only 46-mask, calls the policy a second time, and decodes the chosen
  tile in the same visible prefix;
* `context_from_native` accepts `KAN_SELECTION` as an explicit second phase
  without pretending the native phase changed;
* single-candidate kan still decodes directly.

**Evidence**: `rf_test_m5d_arena.NativeContextTests.test_multi_kan_requires_explicit_second_decision`
checks the raise, the option list, the mask helper and the second-stage
`KAN_SELECTION` decode; `stats["kan_selections"]` counts the second decision.
The full `RFMjaiLogEngine._react` path with a native multi-kan game is still
`NOT_RUN`: the 4-game native smoke had one ankan and one kakan, each with a
single candidate, and recorded `kan_selections=0`.  The native bridge itself is
exercised (579 decisions, zero illegal/fallback).

## 3. Validation reductions must not sync per decision

**Before**: `train/b0.py::evaluate_multi_batches` looped over every decision and
called `bool(legal_mask[i, action])`, `bool(finite_row[i])`,
`float(-logprob)` and `int(top3[i, 0])`.

**After**: per batch it builds target/weight/action tensors, then performs
`legal`, `finite`, CE numerator, top-1 and top-3 reductions with tensor
operations.  Only one scalar conversion per batch is used for the accumulated
counters; no per-decision `.item()`/`bool()`/`float()` path remains.

**Evidence**: 43 B0 protocol/CLI tests pass after the change, including
validation metrics and strict resume.  A VNext GPU throughput profile now runs
(`VNEXT_GPU_PROFILE_FINAL.json`), but a dedicated B0 GPU
synchronization-count profile is still `NOT_RUN`.

## 4. Resume must restore the candidate Registry

**Before**: `run_seed` started `candidates = []`.  A run stopped after step 6
and resumed at step 10 selected only the newly emitted step-10 candidate; the
step-6 candidate was absent from `selection.json`.

**After**: on `--resume`, `_load_prior_candidates` reads the existing
`candidates/step_*.pt` sidecars, verifies the experiment id, seed and protocol
fingerprint, restores raw/EMA validation metrics and counters, and feeds them
into the global selection list.  Foreign candidates are refused.

**Evidence**: `rf_test_b0_cli.test_strict_resume_equals_uninterrupted` now also
asserts identical `candidate_id` and `considered` lists, and that the resumed
selection considers two candidates (steps 6 and 10).

## 5. Limit / NOT_RUN

* The native submodule is now initialized and built in this worktree; a
  4-game VNext smoke ran with 579 decisions, zero illegal/fallback.  It used a
  random small VNext model and a local smoke-only registry, so it is not a
  frozen-M5D reproduction.
* Multi-kan `KAN_SELECTION` remains `NOT_RUN` (`kan_selections=0`).
* A VNext GPU profile and synthetic CUDA BC/GRP/RL smoke ran; a dedicated B0
  sync-count profile and all real-data training remain `NOT_RUN`.
* No strength claim is made from these correctness fixes.
