# SHARD_FORMAT — PR-12 sequence shard / loader

**Status**: PR-12 draft (Shard owner). PARTIAL until H-02 full 2.51M canonical/split lands; the 100k fixture is sufficient for implementation. **P1 follow-up applied**: write-root guard, representative selection, per-game train gate, chunked hashing/streaming samples, disk preflight. Format bumped to `rf-shard/2` (old `rf-shard/1` shards are rejected).
- **Task**: PR-12 — replayable, resumable Sequence Shard + DataLoader
- **Worktree/branch**: `/home/ygsldr/riichi-ai/worktrees/shard` @ `rf/pr12-shards`
- **Base**: `2afb1ace3152a90bd1754a363aeed44acca0f394` (fixed PR-10 `tiles_left`, fixed PR-11 encoder state test, PR-08/09/11/13 contracts)
- **Schema**: `rf-schema-v3/2` (`schema_hash 12503d36…`), action `6cf1efc5…`, normalizer `37df861b…`
- **POV/projection**: `rf-pov-v3/2`
- **Owned files**: `riichiformer/data/shard.py`, `riichiformer/data/loader.py`, `scripts/rf_build_shards.py`, `tests/rf_test_shards.py`, this doc, `docs/riichiformer/v3/logs/PR12_pytest.log`
- **Tests**: 23 passed, exit 0 (P1 follow-up).

## 1. On-disk format (rf-shard/2)

A shard is one SQLite DB plus `<path>.manifest.json`. Payloads are canonical
JSON, zlib-compressed and base64-encoded with a `z:` prefix. Writes go to `<path>.tmp`;
`finalize()` computes the file hash, writes the manifest atomically, then `os.replace`s
the DB. The final path never contains a partial game.

### Tables

| table | columns / purpose |
|---|---|
| `meta` | `key,value`; metadata JSON + counts |
| `games` | **one row per game/component**: `game_id` PK, `component_id`, `component_sha256`, `split`, `source_id`, per-game `train_allowed`, `rights_status`, compressed `pov_streams` (exactly four projected event streams), compressed `kyoku_offsets`, `decision_count` |
| `inputs` | **one row per decision**: `decision_id` PK, `game_id`, `pov`, `decision_index`, `before_event_idx`, `kyoku_idx`, `phase`, `legal_mask`, compressed `tile_state`/`players_public`/`global_public`, `input_sha256` |
| `targets` | **separate target partition**: `decision_id` PK/FK, `expert_action`, `supervision_valid`, `censor_reason`, `rank_final`, `target_sha256` |

There is no full-prefix copy per decision: the four POV streams are stored once, and each
input points at `(pov, before_event_idx)`. `inputs` has no target/expert/rank column;
`targets` has no event/state/mask column. `GameRecord` refuses `quarantine=True` at
construction, and the writer refuses it again.  `rf-shard/2` adds per-game
`train_allowed`/`rights_status` so a training shard can be checked row-by-row, not just
from shard metadata.

`kyoku_offsets` is per POV: `[{"kyoku_idx":0,"start":1,"end":...}, ...]`; decision rows
carry `kyoku_idx` and `before_event_idx` (the action event index, not included in the
visible prefix). No raw full-info game is stored; only PR-09-projected POV streams.

## 2. Metadata / hashes

`ShardMetadata` requires **explicit** `schema_version`, `schema_hash`,
`action_map_hash`, and `normalizer_hash` — there is no silent default to the frozen
hash. It also carries:

`purpose` (`train|audit|diagnostic`), `source_id`, `rights_status`, `train_allowed`,
`canonical_manifest_sha256`, `canonical_index_sha256`, `split_manifest_sha256`,
`split_version`, `ruleset_id`, `ruleset_deviations`, `source_kind`,
`projection_version`, `created_utc`.

Per-game rows additionally carry `component_id`, `component_sha256`, `split`,
`source_id`, `train_allowed`, and `rights_status`.  `purpose=train` requires all of:
shard metadata `train_allowed=True`/`rights_status=ALLOWED`, per-game `split=="train"`,
per-game `train_allowed=True`, per-game `rights_status=ALLOWED`, and a source covered by
the metadata source (`mixed:a+b` accepted as a set).  The writer enforces this on
`add_game`; the training `ShardReader` enforces it again for every game row before
yielding decisions.  Blocked sources can only be written as `audit`/`diagnostic` and are
loudly labelled.  The loader checks the manifest file hash (chunked), schema/action/
normalizer hashes, and can pin canonical-manifest/canonical-index/split-manifest hashes
plus per-game component SHAs via `expected_hashes`/`expected_component_sha256`.

## 3. Resume / idempotence

`ShardWriter(path, metadata, resume=True)` copies an existing final/temp DB, verifies
metadata (only `created_utc` may differ), and re-adds games by `game_id`: targets are
deleted before inputs and the game row, then re-inserted. Re-adding the same game does
not duplicate decisions. `finalize()` writes the DB first (fsynced, atomic rename), then the manifest
(fsynced, atomic rename).  The manifest carries both `shard_sha256` (file integrity,
includes `created_utc`) and `content_sha256` (deterministic metadata minus `created_utc`
plus all table rows; reproducible across rebuilds).  Tests cover resume across a first
5-game shard plus 15 more games, idempotent re-add, and content-SHA reproducibility.

## 4. Loader

`ShardReader(path, purpose, expected_hashes, expected_source_id, expected_component_sha256)`:
- training purpose refuses audit/diagnostic shards, blocked metadata, and any per-game
  row whose `split!="train"`, `train_allowed` is false, rights are not ALLOWED, or whose
  source is outside the metadata source set;
- audit/diagnostic can read blocked sources (same metadata);
- `iter_decisions()` reconstructs `DecisionInput` from the stored POV stream prefix plus
  the input row's tile/player/global state; it imports no target/label class;
- `iter_batches(batch_size, bucket_by_length)` buffers decisions, sorts by sequence
  length (dynamic padding), calls PR-11 `encode_batch`, and yields
  `ShardBatch(batch=ModelBatch, targets=plain dicts, metas, lengths, game_ids, povs,
  decision_indices)`.

Deterministic order is `game_id, pov, decision_index`; bucket mode sorts each buffer by
`(length, game_id, pov, decision_index)`. No target field is passed to the encoder.
File hashing is chunked (`file_sha256`), so opening a 100k-game shard never loads the
whole file.

## 5. CLI

```bash
# default: bounded dry-run (100k), no write
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  .venv/bin/python scripts/rf_build_shards.py \
  --canonical-dir /home/ygsldr/riichi-ai/artifacts/riichiformer/h02/bench_100k \
  --split-manifest /home/ygsldr/riichi-ai/artifacts/riichiformer/h02/split_100k/split_manifest_L_v2.parquet \
  --out /tmp/audit.rfshard --purpose diagnostic --limit 100000

# write a bounded audit/diagnostic shard
... --out /tmp/audit.rfshard --purpose diagnostic --limit 10 --write

# training requires ALLOWED source_rights + split rows; full corpus additionally needs
... --purpose train --all --io-window-approval "<Lead-approved window>"
```

The builder reads the H-02 canonical DB (`--check-canonical-db` default on) plus the
split-v2 manifest, selects `is_representative=True` and `quarantine=False` rows (one per
H-02 component; `component_size` is **not** used to exclude multi-file components).  For
`--purpose train` it requires per-game `split=="train"`, `train_allowed=True`, rights
ALLOWED, and a source ALLOWED in H-03 `configs/riichiformer/source_rights.csv`; any
violation is refused (exit 2) before a writer is opened. `--all` additionally requires
`--io-window-approval` and train purpose.  `--write` is required to write; otherwise
dry-run.

The builder refuses `--out` under `datasets/`, `runs/`, `checkpoints/`, or any linked
checkout's `artifacts/riichiformer/{audit,canonical,split,bench,quality,replay,drift,git_safeguard,audit_logs}`
(multi-root + protected-segment guard; `--repo-root`/`--protected-root` available).  A
write build preflights free disk against `rows x 1,000,000` bytes (measured 9.0 MB / 10
games); dry-run reports the estimate without refusing.

## 6. Dry-run / fixture evidence

100k split-v2 fixture (`bench_100k`, 100,000 representatives):
- dry-run plan: `rows=100000 sources=mixed:tenhou_old+tenhou_recent rights=UNKNOWN
  train_allowed=False dry_run=True` (blocked, so training is refused).
- 10-game bounded audit build (post-P1 rebuild, format `rf-shard/2`):
  - `games=10 decisions=6485 written=10 skipped=0`, **9.0 MB**, ~20.5 s wall (~605 MB RSS)
  - sample hashes unchanged: 1% `e4f2086c…`, 10% `f74d9665…`, 100% `6617036c…`
  - `shard_sha256=5b7e6d5f…` (file hash; includes `created_utc`), `content_sha256=959f4679…`
  - a second identical build has the same `content_sha256` but a different `shard_sha256`
    (only `created_utc` differs)
  - loader: 6,485 decisions; ~100 decisions/s end-to-end encode on this host
    (decode-only ~320/s); not a performance claim.
  - estimated scale: ~0.9 MB/game → ~90 GB for 100k games, ~2.25 TB for 2.51M; the
    builder now refuses to write if free space is below the estimate.
- no writes to `datasets/`, `runs/`, or `checkpoints/`; the write guard enforces this.

Repro:

```bash
cd /home/ygsldr/riichi-ai/worktrees/shard
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q -s tests/rf_test_shards.py
```

## 7. Tests (23 passed)

- four-POV roundtrip: one game row, four POV streams, decisions in one split;
- input/target SQL partition isolation (no target columns in inputs; no state columns in
  targets);
- decision regenerated from stored POV stream: direct PR-11 `encoded_hash` equals the
  loader-reconstructed decision;
- same-game four POVs stay in one game/split;
- 1%/10%/100% decision hash evidence, resume and idempotent re-add;
- explicit-hash requirement (missing metadata hashes → `TypeError`), schema/source
  mismatch fail, blocked-source refusal, quarantine refusal;
- synthetic ALLOWED rights CSV gate (`load_rights`/`source_rights_allowed`);
- H-02 100k fixture: dry-run, bounded audit build, loader, and training-refusal exit 2;
- P1 guard: datasets/runs/checkpoints + main-checkout PR artifacts refused before writer;
- P1 representatives: 2-file component representative included once, duplicate excluded;
- P1 train partition: `val_holdout` row rejected by builder and by training loader, as
  are tampered per-game rights/source rows;
- chunked file hash (no `read_bytes`), streaming `sample_hashes` equals the old
  in-memory definition, reproducible `content_sha256`, pinned canonical/split/component
  hashes.

## 8. Gaps / follow-ups

- **PARTIAL**: H-02 full 2.51M canonical/split is still running; only the 100k fixture is
  used. Full `--all` training-size build not run.
- Only a 10-game audit shard was built (6,485 decisions), not 100k games; the default
  dry-run bound is 100k.
- Loader currently materializes a game's decisions per game and one target query per
  decision; ~100 decisions/s is a starting point, not a performance claim. Streaming
  game rows and batching SQL are PR-12 optimization follow-ups.
- Compression is zlib+base64 in SQLite text; no columnar/Parquet shard yet.
- `source_rights.csv` currently has all H-02 sources blocked; training-path tests use a
  synthetic ALLOWED registry row/metadata, not real Tenhou data.
- No training; no writes to frozen data/run/checkpoint roots.
