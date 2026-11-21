# PR-05 — Runtime adapter and parity status

**Code**: `riichiformer/vnext/runtime.py`  
**Tests**: `tests/rf_test_vnext_runtime.py` (4 passing, CPU)

## Implemented

* `VNextPolicyAdapter` owns one POV's `VNextEventStream`.
* A new `kyoku_idx` resets the event history; global score/phase state remains in
  the `DecisionInput` state snapshot.
* `act(decision)` observes the decision, builds a batch from the current-kyoku
  prefix, runs a deterministic argmax and validates the action against the
  native 46-mask.
* Two decisions sharing one prefix are two rows with distinct phase/decision
  IDs; K/V are shared conceptually, but the cacheless reference path recomputes
  the same prefix.
* `select_kan` requires `ActionPhase.KAN_SELECTION`, a non-empty tile-only mask
  (0..36), and returns a second, independently validated policy action.
* `load_vnext_checkpoint` rejects an MVP-S/unknown payload, checks the sidecar
  SHA, model config hash and optional VNext contract hash.

## Native integration

`riichiformer/eval/m5d_arena.py` now raises `KanSelectionRequired` for a bare
action 42 with multiple candidates and then performs a real second
`KAN_SELECTION` policy call, rather than `kan_options[0]`.  The VNext adapter
itself is tested synthetically; a real VNext native Arena game is `NOT_RUN`
because no VNext checkpoint has been trained and no GPU/Arena budget is
authorized.

## Parity policy

The current adapter is deliberately **cacheless**.  It rebuilds the visible
prefix and is the reference for a future KV cache.  A streaming cache must match
this path within the pre-registered FP32 tolerance; no cache is claimed to exist
yet.  The model's `forward_prefix` and shared `forward` are the same code path,
so the tested single/shared parity is real, not a second implementation.

## NOT_RUN

* 4–20 set native Arena smoke with a VNext checkpoint.
* Streaming KV vs cacheless latency/throughput.
* Real event-log reset under an abrupt `end_kyoku`/`start_kyoku` transition from
  the native driver (the synthetic reset test covers the adapter contract).
