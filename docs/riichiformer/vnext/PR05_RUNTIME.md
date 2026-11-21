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
`KAN_SELECTION` policy call, rather than `kan_options[0]`.  A real native
`OneVsThree` set ran with a randomly initialised VNext checkpoint: 4 games,
579 decisions, 0 illegal, 0 fallback, 20.9 s.  The native logs contained one
ankan and one kakan, but each had one candidate (`kan_selections=0`), so the
multi-kan branch is still `NOT_RUN`.  This smoke used a local smoke-only
registry because the worktree native build/patch identity differs from the
frozen M5D registry.  A second 4-game smoke loaded the 5,000-step BC checkpoint
(765 decisions, 0 illegal/fallback); it is an integration result, not a
strength measurement, and still had `kan_selections=0`.  A 4-set/16-game run
exposed that the native driver does not always forward `start_game`; the engine
now resets the backend once per game from `_react`, and the 16-game rerun passed
with 2,792 decisions, 0 illegal/fallback.  Two repeated 8-game runs had
identical action-trace hashes and scores (`VNEXT_NATIVE_DETERMINISM.log`).

## Parity policy

The current adapter is deliberately **cacheless**.  It rebuilds the visible
prefix and is the reference for a future KV cache.  A streaming cache must match
this path within the pre-registered FP32 tolerance; no cache is claimed to exist
yet.  The model's `forward_prefix` and shared `forward` are the same code path,
so the tested single/shared parity is real, not a second implementation.

## NOT_RUN

* A trained VNext checkpoint native Arena smoke (the current smoke is random
  weights and 4 games only).
* End-to-end multi-kan `KAN_SELECTION` (`kan_selections=0` in the smoke).
* Streaming KV vs cacheless latency/throughput.
* Real event-log reset under an abrupt `end_kyoku`/`start_kyoku` transition from
  the native driver (the synthetic reset test covers the adapter contract).
