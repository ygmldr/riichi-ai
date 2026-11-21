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

## NOT_RUN

* Real 2.51M-match Census, per-phase decision counts and P50/P90/P99 token
  lengths.
* Real train/validation epoch or even a 100-step real-data run.
* GPU BF16 parity, VRAM, DataLoader IO and validation wall-time.
* Pre-encoded memmap/cache; only to be considered after profiling.

The measured CPU microbench (32 decisions, 128 tokens) is in
`PR03_04_MODEL.md`/`profile_cpu.json`; it is not a full-epoch budget.
