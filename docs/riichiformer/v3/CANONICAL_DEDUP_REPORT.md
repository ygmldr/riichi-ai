# H-02 — Canonical / SHA / dedup / split at 2.51 M scale

**Task**: H-02 CANONICAL/DEDUP (BR-03/BR-04/BR-05) · **Branch**: `rf/h02-dedup`
**BASE_SHA**: `59691f57bc7158dcdb2ba98eecd9d102857171f8` (`rf3-base`) · **Worktree**: `/home/ygsldr/riichi-ai/worktrees/h02-dedup`
**Status**: implementation + 10k/100k real-inventory run **DONE**; full 2.51 M run **NOT RUN** (awaiting Lead-approved exclusive-IO window).

> This report never claims corpus-wide duplicate counts. The only corpus-wide evidence here is
> the *method* and the 10k/100k measured fixtures. `duplicate_components = 0` below means
> **0 inside the sample**, not 0 over 2,512,424 logs.

---

## 1. What changed (owned files only)

| File | Change |
|---|---|
| `scripts/rf_build_canonical.py` | `--all`/`--sample` now stream the audit Parquet and canonicalise deterministic **year/chunk shards** into a stdlib-`sqlite3` checkpoint. Resume skips canonicalisation of completed shards but re-streams the audit inventory to rebuild shard boundaries. Atomic final artifacts, deterministic manifest. Unsafe `--out` (datasets / any checkout's PR artifacts) refused across worktrees. `--restart`, `--max-shards` for staged runs. |
| `riichiformer/data/dedup.py` | Duplicate **connected components** over `(game_id, canonical_sha256)` edges; conflict/incomplete → quarantine with no representative; `CanonicalStore` (SQLite schema, shard transactions, atomic finalisation); streaming Parquet/JSONL writers; deterministic streaming stratified sampler. |
| `riichiformer/data/split.py` | New `rf-split/2` registry built from the canonical store: split assigned **per component**, `quarantine` split, representative-only eligibility, PR-05 old-vs-new diff, `verify-v2` leak checks. PR-05 functions/registry untouched. |
| `riichiformer/data/canonicalize.py` | Added `rules_version = mjai.tenhou-4p.v1` to every record (format/protocol tag; **not** a Tenhou rule-parity claim — that is H-04). Canonical bytes are unchanged. |
| `tests/rf_test_canonicalize.py` / `tests/rf_test_split.py` | +negative tests: same game under different filename/gzip/plain/source; same ID different content; cross train/val duplicate; artificial canonical-hash collision; interrupted-resume byte identity; quarantine leak; component straddle; old-vs-new diff; cross-worktree protected-dir guard; blocked-source/non-representative `train_allowed` rejection. 48 passed. |
| `docs/riichiformer/v3/logs/H02_pytest.log` | full pytest + compileall output. |

Deliberately **not** touched: `replay.py`, `player_quality.py`, schema/POV, `.gitignore`, pyproject, `vendor/`, other agents' files, existing `artifacts/riichiformer/{audit,canonical,split}` (read-only inputs).

---

## 2. Method

### 2.1 Streaming canonical index (replaces in-memory `--all`)

1. Read `source_audit.parquet` with `ParquetFile.iter_batches(65536)`; only `kind=leaf` rows
   (~21 MB inventory, never `to_pylist()` over 2.5 M).
2. Canonicalise with PR-03 semantics: gzip/plaintext sniffed by magic, strict per-line JSON,
   sorted-key reserialisation, `canonical_sha256` over UTF-8 canonical NDJSON. No repair.
3. Group selected rows into shards `"<year>:<chunk>"`, `--chunk-rows` default 25 000
   (≈101 shards for 2.51 M). One shard = one SQLite transaction + one `shards` marker row;
   `PRAGMA synchronous=FULL`, WAL checkpoint after each shard. An interruption loses at most
   the in-flight shard; on resume completed shards are skipped without re-canonicalising, but
   the inventory is re-streamed to rebuild the deterministic shard boundaries.
4. Finalise duplicate components in one transaction: only rows that appear in a duplicated
   `id_key` or `canonical_sha256` group are materialised in Python; singleton rows stay on disk.
5. Export by streaming `ORDER BY year, relpath` to Parquet/JSONL through `.tmp` + `os.replace`.
   Given the same inventory + config, a resumed run and a clean run write **byte-identical**
   artifacts (verified on synthetic and real 10k data; see §4.4).

**Store schema (SQLite, `canonical_index.sqlite`)**

```
meta(key, value)                       -- config fingerprint, scan/finalize state
shards(shard_id, year, rows, valid, invalid, status, finished_utc, seconds)
games(id, shard_id, canonical_version, rules_version, game_id, id_key,
      platform_game_key, source_id, year, relpath UNIQUE, root, compression, bytes,
      original_sha256, canonical_sha256, events, kyoku_count, invalid_reason,
      component_id, component_size, is_representative, collision_review,
      cross_key_duplicate, quarantine, quarantine_reason)
components(component_id, size, valid_rows, distinct_game_ids,
           distinct_canonical_shas, cross_key_duplicate, quarantine,
           quarantine_reason, representative_relpath/game_id, canonical_sha256,
           original_sha256, year, sources)
```

Every record keeps **original SHA-256, canonical SHA-256, game_id, id_key, source_id,
year, rules_version, canonical_version, compression, bytes, event/kyoku counts**.
`manifest.json` is the durable deterministic manifest (audit SHA-256, config, shard count,
counts, artifact SHA/logical SHA); `summary.json` is byte-identical for old readers;
`run_log.jsonl` is the volatile per-shard timing/RSS log.

### 2.2 Duplicate graph / quarantine policy

Edges are **all rows sharing `id_key`** (`tenhou:<log id>`, or `tenhou:`+filename key for a
broken file with no canonical hash) **and all rows sharing `canonical_sha256`**; connected
components are computed over the union. Therefore:

* same events in gzip vs plaintext, different filename, different source → one component;
* same content under two Tenhou IDs (`cross_key_duplicate=true`) → **one** component, one
  representative (BR-04 fix; previously only flagged);
* same `game_id` with different canonical SHA → `conflict_same_id_different_canonical_sha`:
  whole component `quarantine`, **no representative**;
* any invalid/incomplete member in the component → `incomplete_component` quarantine;
* invalid singleton → quarantine with its verbatim `invalid_reason`;
* non-quarantined component: exactly one representative, deterministic
  `min(original_sha256, relpath)`; non-representatives are kept for provenance but
  `train_allowed=false`.

### 2.3 Output schemas

`canonical_log_index.parquet` (`rf-log-index/2`) columns:
`index_version, canonical_version, rules_version, game_id, id_key, source_id, year,
relpath, root, compression, bytes, original_sha256, canonical_sha256, platform_game_key,
events, kyoku_count, component_id, component_size, duplicate_count, is_representative,
collision_review, cross_key_duplicate, quarantine, quarantine_reason, invalid_reason`.

`duplicate_components.parquet`:
`component_id, component_size, valid_rows, distinct_game_ids, distinct_canonical_shas,
cross_key_duplicate, quarantine, quarantine_reason, representative_relpath,
representative_game_id, canonical_sha256, original_sha256, year, sources`.

`split_manifest_{L,R}_v2.parquet` (`rf-split/2`) columns:
`split_version, protocol, assignment_mode, game_id, component_id, component_size,
canonical_sha256, original_sha256, source_id, year, month, split, relpath, root,
blocked_source, rights_status, train_allowed, previously_used_in_research, undated,
holdout, is_representative, quarantine, duplicate_group_size, split_source,
representative_relpath, reason`.

### 2.4 Split registry v2

`python -m riichiformer.data.split build-v2` writes **new** files only under the H-02 artifact
directory: `split_registry_v2.json`, `split_manifest_{L,R}_v2.parquet`, `split_diff_v2.json`.
The split is computed once per component (representative's year/month/Tenhou id) and copied to
every member, so a duplicate can never straddle `train`/`val`/`select`; quarantined components
go to the `quarantine` split and can never be `train_allowed`. POV seats do not exist as rows:
one row = one game file, so all four POVs/labels share one split by construction. PR-05's
`split_manifest_{L,R}.parquet` and `split_summary.json` are only read for the diff.
`verify-v2` streams the manifest and fails on component straddle, quarantine leakage, a
quarantine row with `train_allowed=true`, a `blocked_source=true` row with
`train_allowed=true`, or a `is_representative=false` row with `train_allowed=true`.

**Output safety (cross-worktree)**: both output guards resolve the caller's root, this module's
checkout, every linked `git worktree`, and the main checkout from
`git rev-parse --git-common-dir`, plus an optional `--protected-root`; independently, a resolved
path containing `artifacts/riichiformer/{audit,canonical,split,bench,quality,replay,drift,git_safeguard,audit_logs}`
is refused without git. An absolute `--out` aimed at the main checkout's PR-05 registry is
therefore rejected even when the command runs inside a worktree.

---

## 3. Commands

```bash
# 10k / 100k measured fixtures (real inventory, read-only)
AUDIT=/home/ygsldr/riichi-ai/artifacts/riichiformer/audit/source_audit.parquet
.venv/bin/python scripts/rf_build_canonical.py --audit-parquet "$AUDIT" \
    --sample 10000  --seed 20261121 --chunk-rows 25000 --restart \
    --out artifacts/riichiformer/h02/bench_10k
.venv/bin/python scripts/rf_build_canonical.py --audit-parquet "$AUDIT" \
    --sample 100000 --seed 20261121 --chunk-rows 25000 --restart \
    --out artifacts/riichiformer/h02/bench_100k

# new split registry (old PR-05 registry is read-only input)
.venv/bin/python -m riichiformer.data.split build-v2 \
    --canonical-db artifacts/riichiformer/h02/bench_100k/canonical_index.sqlite \
    --canonical-manifest artifacts/riichiformer/h02/bench_100k/manifest.json \
    --out artifacts/riichiformer/h02/split_100k \
    --old-manifest-L /home/ygsldr/riichi-ai/artifacts/riichiformer/split/split_manifest_L.parquet \
    --old-manifest-R /home/ygsldr/riichi-ai/artifacts/riichiformer/split/split_manifest_R.parquet \
    --old-summary   /home/ygsldr/riichi-ai/artifacts/riichiformer/split/split_summary.json
.venv/bin/python -m riichiformer.data.split verify-v2 \
    --manifest artifacts/riichiformer/h02/split_100k/split_manifest_L_v2.parquet \
               artifacts/riichiformer/h02/split_100k/split_manifest_R_v2.parquet

# tests
.venv/bin/python -m pytest -q tests/rf_test_canonicalize.py tests/rf_test_split.py
```

Real-inventory artifacts are under `/home/ygsldr/riichi-ai/artifacts/riichiformer/h02/`
(gitignored). The worktree itself has no production data.

---

## 4. Results

### 4.1 10k / 100k canonical runs (real audit inventory)

| fixture | rows | valid | invalid | duplicate components | quarantine | representatives | wall | build | peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10k stratified | 10,000 | 10,000 | 0 | 0 | 0 | 10,000 | **58.0 s** | 34.6 s | **238 MiB** |
| 100k stratified | 100,000 | 100,000 | 0 | 0 | 0 | 100,000 | **386.7 s** | 358.7 s | **405 MiB** |

Year coverage matches PR-02 (18 strata, 555/5,555 per year; 2009 has 555 not 556 in the
10k draw because of the deterministic hash ranking). Throughput 288.6 → 278.8 files/s
(≈3.4-3.6 ms/game; ~1,100 events/game). Sampling cost ≈23 s per run (2-3 audit-parquet passes);
a full `--all` run does not pay it.

**Old-vs-new canonicalisation**: PR-03's old index was only 1,000 rows (`full_corpus=false`).
All **1,000 old relpaths are present in the new 10k index** (same seed/priority keeps the old
top-55-per-stratum selection inside the new top-555), and every one has **identical
`original_sha256`, `canonical_sha256`, `game_id`, `events`, `kyoku_count` (0 mismatches)**.
Only new metadata columns were added; canonical bytes did not change. The old 1,000-row sample
cannot be compared with the new 100k draw beyond this subset, and
`tests/rf_test_canonicalize.py::test_streamed_sample_matches_in_memory_sample` pins the new
streaming sampler to the old in-memory selection rule.

**Old-vs-new split at 100k** (diff against PR-05 `split_manifest_{L,R}.parquet`, SHA
`5e291ad0…`, `6fbf3465…`):

| protocol | new rows | counts | moved vs PR-05 | quarantined | PR-05 registry hash |
|---|---:|---|---:|---:|---|
| L | 100,000 | train 77,778 / val_holdout 5,556 / external_diagnostic 16,666 | **0** | 0 | `184f34dc…` |
| R | 100,000 | train 88,890 / val 2,713 / select 2,842 / confirm_blocked 5,555 | **0** | 0 | `4ada926b…` |

New v2 registry hashes: L `639cd949df778eb0822a176322814cb7e8de6f1e8e48f3be692d4e22f4432994`,
R `0d92b7af7979c78394422e1ee50fa0229d26e8f60ba76c96bce54685179123f5`.
That 0 moved is expected: the sample contains no duplicate components yet, so component-level
assignment degenerates to the calendar assignment. The synthetic tests prove the duplicate/
quarantine cases that the full corpus may contain (cross-year duplicate moves one file
`val_holdout→train`; conflict rows move `train→quarantine`).

### 4.2 100k split-v2 build cost

- 7.5 s wall, peak RSS **1.10 GiB** (two PR-05 2.5 M-row `relpath→split` maps for the diff;
  the map is released between protocols). 22 MB manifests + 7 KB registry.
- `verify-v2` reads both manifests: OK, 0 component conflicts, 0 quarantine leaks.

### 4.3 Artifact hashes (measured)

| artifact | SHA-256 |
|---|---|
| `bench_10k/manifest.json` | `4a94463c16a106bc61223f079fd2336d2ba101b104e6c9d12a048c5b40b3e4bf` |
| `bench_10k/canonical_log_index.parquet` | `7f2da01432d30e3c324876b9b846c3e30a1ce85e2a6e111e8d414949d403e814` |
| `bench_10k/sample_10000.jsonl` | `629a8ff5bacac761a9629c05a54d35038bfa08d84405e4e1467ce379f42c3d15` |
| `bench_10k/duplicate_components.parquet` | `420e6574c2f73d85d4c43c5ee3dfd72b9864053099fcf6926836f4fd95c5ff83` |
| `bench_100k/manifest.json` | `f8b8eb2364abb1216d38715d910e02a66b5bba90538e5d62bc2146b9f84906a9` |
| `bench_100k/canonical_log_index.parquet` | `42b651c543925a5725e731cd91f2711f87831e89ed8ac201ca31c0663725448e` |
| `split_100k/split_registry_v2.json` | `a8afc2faadd21eb4d0163358906f4a453f8a1a97fa0468d1c08d25621b79ec6d` |
| `split_100k/split_manifest_L_v2.parquet` | `82b61cfe1894ebf25bbf9d01b1753b7c4625b73d1f6b6adb386c8aba4d1e0024` |
| `split_100k/split_manifest_R_v2.parquet` | `d1cccb3c12752164b3ba47dffbb315ee6d3754a76c69a53763e7498dbbaa88ff` |

### 4.4 Resume evidence on real data

10k run interrupted after 5 of 18 shards, then resumed (completed shards are not
re-canonicalised; the inventory is re-streamed to rebuild boundaries). All six deterministic
artifacts are **byte-identical** to a clean 10k run (`manifest.json`, `summary.json`, index, components,
quarantine, `sample_10000.jsonl`, hashes in the table above). The synthetic test also exercises
`--max-shards` interruption and a completed-rerun no-op. A changed inventory SHA refuses to
resume (exit 2, `--restart` required) — no silent mixing of two inventories.

### 4.5 Full-run estimate (not executed)

Measured 278.8 logs/s → **2,512,424 / 278.8 ≈ 9,011 s ≈ 2 h 30 m** canonical CPU, plus ~1-2 min
finalisation/export when there are few duplicate rows. Propose **2.5-3.0 h** wall in an
exclusive-IO window (single process; one core + sequential disk reads of the 11.4 GiB corpus).
Peak RSS on the full run should stay near the 100k figure (~0.4 GiB); finalise memory grows only
with the number of duplicate rows, not the corpus.

Disk extrapolation from the measured 100k bytes-per-row (linear):
`canonical_index.sqlite` ≈ **1.7-2.0 GB**, `canonical_log_index.parquet` ≈ **0.30 GB**,
split v2 L+R manifests ≈ **0.55 GB**, quarantine/registry small. Reserve **≥4 GB free** for the
canonical stage and **≥5 GB** including split v2 (temporary WAL + `.tmp` files included).
No sample_*.jsonl is written for `--all`, so the 53 MB/100k sample JSONL does not scale.

Proposed full-run commands (run from the H-02 worktree; **requires Lead approval**):

```bash
AUDIT=/home/ygsldr/riichi-ai/artifacts/riichiformer/audit/source_audit.parquet
OUT=/home/ygsldr/riichi-ai/artifacts/riichiformer/h02/canonical
.venv/bin/python scripts/rf_build_canonical.py --audit-parquet "$AUDIT" \
    --all --chunk-rows 25000 --out "$OUT"
# resume = rerun the same command (no --restart); restart = add --restart
.venv/bin/python -m riichiformer.data.split build-v2 \
    --canonical-db "$OUT/canonical_index.sqlite" \
    --canonical-manifest "$OUT/manifest.json" \
    --out /home/ygsldr/riichi-ai/artifacts/riichiformer/h02/split \
    --old-manifest-L /home/ygsldr/riichi-ai/artifacts/riichiformer/split/split_manifest_L.parquet \
    --old-manifest-R /home/ygsldr/riichi-ai/artifacts/riichiformer/split/split_manifest_R.parquet \
    --old-summary   /home/ygsldr/riichi-ai/artifacts/riichiformer/split/split_summary.json
.venv/bin/python -m riichiformer.data.split verify-v2 \
    --manifest /home/ygsldr/riichi-ai/artifacts/riichiformer/h02/split/split_manifest_L_v2.parquet \
               /home/ygsldr/riichi-ai/artifacts/riichiformer/h02/split/split_manifest_R_v2.parquet
```

Expected full-run additions to review: corpus-wide duplicate/conflict/quarantine counts (unknown
until run), moved-row table vs PR-05, and the final v2 registry hashes. Until then PR-05's
registry remains the only full-corpus split and must not be treated as SHA-deduplicated.

---

## 5. Evidence level

- **A (directly executed)**: the 10k/100k canonical runs; SQLite schema/checkpoint/resume;
  byte-identical resume; duplicate-component/quarantine unit and CLI tests; split-v2 build and
  `verify-v2` on 100k; PR-05 diff read; all 48 tests; no writes to inputs.
- **B (measured sample, not extrapolated as fact)**: 0 duplicate components / 0 quarantine in
  10k and 100k stratified draws. This is a sample result only.
- **C (estimate)**: full 2.51 M ETA (2.5-3 h), disk (~2.7 GB total), corpus-wide duplicate
  counts, and the size of the train/val/select diff that duplicates will cause.

## 6. P0 / P1 / blockers

- **P0 introduced: none.** Invalid/conflict rows cannot become representatives or
  `train_allowed`; quarantine cannot enter `train`; split assignment is component-level.
- **P1 (open, inherited)**: the full 2,512,424-log canonical/dedup run has **not** been
  executed. Therefore “0 duplicates over 2.51 M” is **not claimed**, and PR-05's split is still
  the only full-corpus registry. H-02 acceptance (“eligible 全量有 canonical hash、重叠为 0、
  冲突全部 quarantine”) remains **PARTIAL** until the full run.
- **P1 (process)**: full run needs the Lead-approved exclusive-IO window; the 100k shard already
  occupies ~6.5 min CPU + ~0.4 GiB RSS, and a concurrent heavy job would distort throughput.
- **Blockers outside H-02**: H-03 rights (`unknown` → no training) still blocks actual training;
  H-01 scoring parity is independent of canonical bytes (no dependency).
- **Not in scope / untouched**: replay scoring, schema/POV, shards/dataloader, Tenhou rule
  parity (H-04). `rules_version` is a format tag, not a rule-equivalence claim.

## 7. Next hard dependencies

1. Lead approval + exclusive-IO window → full canonical run → full split v2 → publish
   `manifest.json`/registry hashes and the old-vs-new diff.
2. H-03 rights gate must clear before any `train_allowed` row can be used, even after H-02.
3. PR-12 shard/dataloader must consume the **representative-only, non-quarantine** rows from the
   v2 manifest (not the raw PR-05 registry).
4. H-04 rule audit remains a separate correctness axis; canonical event bytes are unchanged by it.
