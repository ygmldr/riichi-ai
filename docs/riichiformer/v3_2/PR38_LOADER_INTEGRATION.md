# PR-38 loader / trainer integration

Task: RiichiFormer V3.2 Wave 2, Dev-COMPUTE follow-up (PR-38 data/training wiring).
Worktree: `/home/ygsldr/riichi-ai/worktrees/v32-b0` (`rf/v32-b0`).
Base SHA: `5242e5e843d2f250786e6b48d7d2ab4b86c3cb27` (`rf/v3.2-local` /
`rf/v32-integration` HEAD).
Status: implementation + CPU tests complete; **no GPU run, no push/merge**.

This record closes the PR-43 P1 "PR-38 multi-decision forward is not yet wired
into `data/loader.py`/`rf_train.py`". The legacy per-decision prefix path remains
the default and its API/semantics are unchanged.

## 1. Files

| file | change |
|---|---|
| `riichiformer/data/loader.py` | new `MultiShardBatch`, `iter_multi_decision_batches`, exact-prefix/observable-state checks, adaptive chunking (`max_events`/`max_positions`), optional deep snapshot replay; existing `iter_decisions`/`iter_batches` behaviour preserved |
| `riichiformer/train/bc.py` | new `training_step_multi` sharing the PR-39 accumulation body with `training_step` |
| `scripts/rf_train.py` | `--multi-decision` (default off), `--multi-max-events`, `--multi-max-positions`, `--multi-deep-verify`; multi dry-run/train output; `training_path` checkpoint provenance and cross-path resume guard |
| `tests/rf_test_multi_loader.py` (new) | 20 tests: grouping/accounting, chunking, parity, training step, tamper/rights/target negatives, CLI dry-run/train/resume |
| `docs/riichiformer/v3_2/PR38_LOADER_INTEGRATION.md` (new) | this record |

Not changed: `data/shard.py`, `data/encode.py` semantics, `models/**` semantics,
`policy/**`, `vendor/mortal`, other worktrees.

## 2. API diff

```python
# riichiformer/data/loader.py
class MultiShardBatch:
    batch: MultiDecisionBatch
    targets: tuple[dict, ...]                 # N (one per decision)
    metas: tuple[StreamMeta, ...]             # N
    lengths: tuple[int, ...]                  # N: event tokens of the decision's stream
    game_ids: tuple[str, ...]                 # N
    povs: tuple[int, ...]                     # N
    decision_indices: tuple[int, ...]         # N: stream-local PR-10 index
    decision_stream_index: tuple[int, ...]    # N: batch row (stream) of each decision
    # properties: stream_game_ids, stream_povs, stream_lengths, stream_metas,
    #             event_tokens, marker_tokens, tokens

ShardReader.iter_multi_decision_batches(
    batch_size: int = 8,
    max_decisions: int | None = None,
    *,
    max_events: int = MAX_EVENTS_DEFAULT,   # 4096
    max_positions: int | None = None,
    verify_state_snapshots: bool = False,
) -> Iterator[MultiShardBatch]

# riichiformer/train/bc.py
training_step_multi(model: Policy, batch: MultiDecisionBatch,
                    targets: Sequence[Mapping], *, optimizer,
                    scaler=None, scheduler=None, max_grad_norm=1.0,
                    bf16=False, accumulation_step=0, accumulation_count=1,
                    accumulation_weight_total=None,
                    loss_scale=None) -> BCLossResult
```

`__all__` of both modules was extended; no existing signature/class was removed.
`training_step` still requires a `ModelBatch`; `training_step_multi` requires a
`MultiDecisionBatch` (wrong types fail closed).

## 3. Loader contract

### 3.1 Grouping and batch boundaries

* Inputs are grouped by `(game_id, pov)` and ordered by `decision_index`.
  `decision_index` must be contiguous `0..N-1`; cutoffs must be non-decreasing
  (duplicate cutoffs, e.g. kan selection, are accepted). A POV with no decision
  rows is skipped; every stored POV stream is still subject to the shard's
  `games()`/`_assert_train_game` split/source/rights gate.
* Per game, `len(inputs) == games.decision_count` and the number of joined
  targets equals the number of inputs, otherwise `ShardLoadError`.
* A batch never mixes games and never contains two rows of the same POV. POVs of
  one game are padded to one `L`, so a multi-POV batch exercises the
  `MultiDecisionBatch` marker/visibility validation on each row.
* **Never silent truncation.** A POV chunk is split on decision boundaries only.
  The event prefix of a chunk is `stream[:last_cutoff_in_chunk]`, i.e. exactly
  the shared full-stream prefix the PR-38 encoder expects. Extra events after the
  last decision are not encoded because no decision can read them.
* Chunking limits:
  - `batch_size`: maximum decisions (markers) in one yielded batch.
  - `max_decisions`: optional lower per-POV chunk cap (clamped to `batch_size`).
  - `max_events`: maximum event prefix of one row (default 4096, matching the
    frozen encoder limit). A long POV is split at the last decision before the
    next cutoff would exceed it.
  - `max_positions`: optional joined `events + markers` cap per row; when the
    caller knows the model, B0 passes the model `max_positions`.
  - A **single decision** whose own prefix exceeds `max_events` (or whose
    `prefix + 1 marker` exceeds `max_positions`) raises `ShardLoadError`; it is
    never dropped, truncated or silently skipped.
* `MultiShardBatch.__post_init__` checks `batch.n_decisions == len(targets)`,
  every per-decision tuple length, `decision_stream_index ==
  batch.decision_batch_index`, row `(game, pov)` identity, per-row strictly
  increasing decision indices, and per-decision stream length equality.

### 3.2 Prefix / cutoff fail-closed checks

Every decision is built from `pov_streams[pov]`, not from a stored prefix. The
loader enforces, per decision:

1. cutoff is in `[1, len(stream)]` and `stream[cutoff-1].seq_idx == cutoff-1`;
2. the stored prefix is the O(1) exact slice `stream[:cutoff]` (both endpoints are
   the same event objects) and the stored `kyoku_idx` equals the cutoff event's
   kyoku;
3. an incremental public-state replay (wall count from tsumo count, kyoku/bakaze/
   honba/kyotaku/oya, scores, river/riichi/meld counts, dora indicators) equals
   the persisted snapshot. A tampered `before_event_idx` changes this state and
   is refused (`ShardLoadError`).

`verify_state_snapshots=True` (CLI `--multi-deep-verify`) additionally runs the
element-wise `check_prefix(..., verify="exact")` and the full
`_SnapshotTracker` tile-ledger replay, comparing the complete stored snapshot.
It is opt-in because the tracker costs tens of microseconds per event on long
streams (`>4096` events per POV); the cheap checks stay enabled by default.

`MultiDecisionStream`/`encode_multi_decision_batch` still receive pre-normalized
`DecisionMarker`s (the same state vector `decision_marker_from_input` builds), so
the encoder does not re-run the O(N×L) `DecisionInput` prefix validator. This is
the reason a long POV is practical: `DecisionInput` construction/validation was
the dominant loader cost (100-game probe dry-run went from an unusable profile to
~43 s; a single game's first 2 batches from ~16 s to ~1.7 s).

### 3.3 Token accounting

`MultiShardBatch.event_tokens` = padded shared event tokens
(`batch.padding_mask.sum()`), `marker_tokens` = real marker slots,
`tokens = event_tokens + marker_tokens`. This is the exact joined transformer
sequence length, summed once per row (not per decision). The prefix path keeps
its old counter (`padding_mask.sum()` over prefix rows).

## 4. Trainer contract

`training_step_multi` is the PR-38 counterpart of `training_step` and shares the
same internal body (`_training_step_core`) after a single `Policy.forward_multi`
call:

* `masked_bc_loss` weight/legality/censor validation is identical;
* gradient accumulation uses the same window denominator
  (`accumulation_weight_total` = `window_weight_total(...)`) or explicit
  `loss_scale`; `accumulation_count > 1` without either still fails closed;
* optimizer/scheduler/clip/scaler step boundary, all-censored microbatch and
  `model.train()` semantics are identical;
* targets are matched to `forward_multi` output rows in marker order
  (stream-major, then decision order); `masked_bc_loss` raises if the lengths or
  legal masks disagree.

No model/forward semantic changed. The measured multi-vs-prefix parity on a
loaded 9-decision/4-POV batch, same model, dropout 0, CPU float32:

```
PR38_LOADER_PARITY decisions=9 logit_max_abs=5.960e-07 loss_abs=0.000e+00
loss=2.141519
```

(within the PR-38 test tolerance atol/rtol 1e-5; the residual is SDPA reduction
order, not a semantic difference.)

## 5. CLI

`--multi-decision` is **off by default**, so old `--train`/`--dry-run` behaviour
and the prefix data path are unchanged.

* multi dry-run:
  `PR14_MULTI_DRY_RUN batches=<n> decisions=<n> tokens=<n> shard=... max_events=... max_positions=...`
* multi train prints `PR14_MODEL`, `PR14_MULTI_MODEL ... training_path=multi_decisions
  max_events=... max_positions=...`, one `PR14_MULTI_STEP ... tokens=... decisions=...`
  per optimizer step, `PR14_MULTI_DONE ...`, and the existing `PR14_DONE`.
* `--multi-max-events` (0 = auto: 4096 dry-run / model `max_positions` train) and
  `--multi-max-positions` (0 = auto: model `max_positions` train, unset dry-run)
  control the adaptive chunking. A requested cap above the model's
  `max_positions` is refused (`exit 2`) instead of producing a batch the model
  cannot forward.
* `--multi-deep-verify` selects the full snapshot audit; it is not needed for
  normal training.
* `--tiny-smoke`, `--model-config`, `--init-weights-only`, strict resume,
  provenance and policy authentication are unchanged. The model config is the
  same object on both paths.

Checkpoint `extra` now records:

```json
{"training_path": "multi_decisions" | "prefixes",
 "token_accounting": "shared_events_plus_markers" | "prefix_decision_tokens"}
```

Strict resume additionally checks a stored `training_path` against the requested
path and exits `1` (`CheckpointMismatch`) on a cross-path resume; legacy
checkpoints without `training_path` keep loading on either path. The three frozen
hashes, model-config hash, source/split/policy/canonical provenance and
optimizer/scheduler/RNG restore are unchanged.

## 6. Tests

New file `tests/rf_test_multi_loader.py` (20 tests, CPU only):

| area | assertions |
|---|---|
| grouping/accounting | 2 games × 4 POVs, 18 decisions; per-decision tuple lengths; `n_decisions == len(targets)`; row identity; no mixed game/POV; legacy target equality; cutoff equality; token accounting |
| chunking | decision-boundary chunks, same full-stream prefix, ordered cutoffs/indices, no truncation; `max_decisions` cap; `max_events`/`max_positions` caps and hard errors |
| no cross-stream mixing | per-decision state features / legal mask / phase and event tensors equal `encode_decision` for that POV |
| parity | `forward_multi` logits and `masked_bc_loss` vs per-prefix path within 1e-5; measured 5.96e-07 / 0.0 |
| trainer | finite loss, gradients exist/finite, accumulation>1 without a denominator refused; old `training_step` prefix path still works |
| negatives | `before_event_idx` ±1 tamper (cheap and deep), missing target, missing input/decision_count mismatch, val_holdout split, foreign source, bad batch args → fail closed |
| CLI | multi dry-run line, default path still `PR14_DRY_RUN`, tiny multi train + same-path resume, cross-path resume exit 1, over-cap `--multi-max-events` exit 2 |

Commands (CPU, `PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release`):

```bash
.venv/bin/python -m pytest -q tests/rf_test_multi_loader.py                 # 20 passed, exit 0
.venv/bin/python -m pytest -q tests/rf_test_multi_loader.py \
  tests/rf_test_shards.py tests/rf_test_training.py \
  tests/rf_test_multi_decision.py tests/rf_test_encoder_parity.py           # 133 passed, exit 0
```

Real-data probe (Lead-supplied 100-game H-02 shard, private profile, 65,486
decisions; CPU dry-run only):

```bash
scripts/rf_train.py --shard /tmp/b0_probe100.rfshard --dry-run \
  --multi-decision --batch-size 64 --multi-max-events 8192 \
  --expected-policy-hash 72ac...
# PR14_MULTI_DRY_RUN batches=1220 decisions=65486 tokens=1122619
```

Long-POV CPU train smoke (1-layer 8k-position test config, not a model choice):

```bash
scripts/rf_train.py --shard /tmp/b0_probe100.rfshard --train \
  --out /tmp/b0_multi_smoke12.pt --multi-decision \
  --model-config /tmp/b0_tiny8k.yaml --multi-max-events 8192 \
  --batch-size 64 --max-steps 12 --expected-policy-hash 72ac...
# step 12 hits the 4,858-event POV: tokens=4884 decisions=26; exit 0
```

Full rf suite on this clean-pin worktree (29 `tests/rf_test_*.py`, py3.12,
`PYTHONPATH=.../vendor/mortal/target/release`):

```
83 failed, 690 passed, 2 skipped, 1 warning, 211 subtests passed in 216.84s  (exit 1)
```

All 83 failures are **vendor-source anchor tests**, not code regressions in this
lane: `rf_test_ruleset_parity.py` (70 `SUBFAILED` + 10 `FAILED`) and
`rf_test_rule_fixes.py` (3 `FAILED`) read the pinned `vendor/mortal` Rust source
and expect the local `patches/0014-rule-r02-r25-r26.patch` to be applied. This
worktree keeps the clean vendor pin (0cff2b5) by instruction. Excluding exactly
those two environment-dependent files (Lead runs the final full suite on the
patched integration worktree):

```
642 passed, 1 skipped, 1 warning, 57 subtests passed in 194.91s  (exit 0)
```

The Lead's 500-game B0 shard run of `--multi-decision --batch-size 64
--max-steps 50` completed 50 steps / 2,585 decisions / 38,719 tokens in 32.76 s
on the GPU host (Lead-owned measurement; this lane did not run GPU).

## 7. Not run / limitations

* **No GPU run, no CUDA parity/VRAM/throughput claim** in this lane. The Lead
  measured a 2-step/8-batch GPU smoke (tokens 58/107, 1.85 GB) and the 500-game
  50-step run; those are Lead-owned and not repeated here.
* The clean-pin full suite is 690 passed with 83 vendor-source-anchor failures
  (`rf_test_ruleset_parity` / `rf_test_rule_fixes`, patches/0014 absent here).
  The same suite excluding those two environment-dependent files is 642 passed,
  exit 0. The Lead owns the final full-suite run on the patched integration
  worktree. No push/merge.
* The cheap prefix check does not rebuild the tile ledger. It detects cutoff
  tampering through the observable public state; a tamper that moves the cutoff
  only across a settlement event that changes no tracked public field is caught
  only by `--multi-deep-verify` (documented, not hidden). The deep replay is
  opt-in for performance on long POVs.
* The multi path has no CLI gradient-accumulation window (`accumulation_count`);
  `training_step_multi` supports it and tests its fail-closed contract, but
  `rf_train.py` keeps one microbatch per optimizer step as before.
* Loader time on a full 500-game epoch was not separately benchmarked; the
  100-game probe dry-run is ~43 s and the Lead's 50-step train is 32.76 s. The
  loader is single-threaded and deterministic; no shuffle/streaming prefetch was
  added.
* `max_events` above 4096 requires a model config whose `max_positions` covers
  the joined sequence; the CLI refuses an inconsistent request rather than
  failing inside `forward_multi`.
* Default batches never mix games, so `batch_size` is an upper bound; when a
  game has fewer decisions the yielded batch is smaller.
* No strength, convergence or candidate-quality claim is made from BC loss or
  token counts.

## 8. P0 / P1 / P2

* **P0:** none found in the targeted paths.
* **P1 (fixed in this lane):** long POV (`prefix > 4096`) failed at
  `encode_multi_decision_batch` because chunking was decision-only. Adaptive
  `max_events`/`max_positions` chunking + CLI caps reproduces the Lead's 500-game
  run without the failure; single over-cap decisions still fail closed.
* **P2:** the observable-only default verification (settlement-only tamper edge
  case) and the lack of a CLI accumulation window are documented above.
* **P2:** the loader imports the existing private `_SnapshotTracker` and
  `_state_snapshot` helpers for audit mode/marker state; no private module was
  modified. If those helpers move, the imports need updating.

## 9. Next gate

1. Lead: run the full rf suite on the integrated branch after merge.
2. Lead: rerun the 500-game B0 smoke with `--multi-decision` and the final B0
   model config, recording tokens/decisions and (if GPU) VRAM; keep strict resume
   and cross-path resume checks.
3. Do not start PR-44 B0 until the Lead's GPU parity/long-context contract and
   the full-suite result are recorded. No remote/vendor change was made.
