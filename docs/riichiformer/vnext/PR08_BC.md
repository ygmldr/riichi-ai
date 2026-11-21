# PR-08 — VNext BC data adapter and trainer

**Code**: `riichiformer/vnext/data.py`, `riichiformer/vnext/train.py`  
**Tests**: `tests/rf_test_vnext_train.py` (5 passing, CPU)

## Data adapter

`VNextBatchAdapter` consumes the already-authenticated
`ShardReader.iter_decisions()` stream.  It groups decisions by
`(game_id, POV, kyoku)` and builds one current-kyoku `VNextEventStream` per
group.  Multiple decisions in a kyoku share the event row; targets remain a
separate tuple/dict and never enter `VNextModelBatch`.  This is an additive
adapter; the old shard schema and `rf-shard/3` rights checks are untouched.

Target tensors (`targets_from_dicts`) are trainer-only and validate action/rank
ranges, supervision, weights and forced flags.

## Trainer

* masked BC: `-sum(w_i log pi(a_i)) / sum(w_i)`;
* optional low-weight rank CE when the Critic is enabled;
* AdamW with no weight decay on biases/norms/embeddings;
* gradient accumulation requires the whole-window weight denominator and
  performs a no-op for an all-censored batch instead of stepping on zero;
* `train_step` reports additive numerator, valid/total counts, weight and
  whether the optimizer stepped;
* `save_vnext_checkpoint` stores the VNext model config, the three frozen
  schema hashes, `vnext_contract_hash`, counters and provenance;
* `load_vnext_checkpoint` verifies sidecar SHA, model config hash and contract
  hash; an old MVP-S state dict cannot be strict-loaded.

## Tests

* synthetic two-decision batch groups correctly and keeps target isolation;
* target tensor contract;
* two training steps produce finite loss and an optimizer step;
* an all-censored batch is a no-op with no gradients;
* checkpoint save/load restores identical weights and refuses a wrong contract
  hash.

## Bounded real diagnostic (read-only)

A 100-game `diagnostic` shard from the V3.3 validation lane
(`/tmp/v33_b_smoke/val100.rfshard`, `source_id=selfplay_own`, `UNKNOWN`,
`train_allowed=false`) was read through `ShardReader(purpose="audit")` and
`VNextBatchAdapter` for the first 104 decisions.  This is a **read-only adapter
smoke**, not training and not a Census:

```text
games=1 povs=1 decisions=104 supervised=103
phases={'SELF_TURN': 79, 'CALL_RESPONSE': 25}
ranks={'1': 104}
elapsed=0.3s batches=13
```

The one-game sample is not representative and no shard was modified.  Its
rights gate remains `UNKNOWN`/`train_allowed=false`.

## NOT_RUN

* Real 2.51M-match Census, per-phase decision counts and P50/P90/P99 token
  lengths.
* Real train/validation epoch or even a 100-step real-data run.
* GPU BF16 parity, VRAM, DataLoader IO and validation wall-time.
* Pre-encoded memmap/cache; only to be considered after profiling.

The measured CPU microbench (32 decisions, 128 tokens) is in
`PR03_04_MODEL.md`/`profile_cpu.json`; it is not a full-epoch budget.
