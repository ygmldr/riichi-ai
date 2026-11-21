# PR-17 G3 end-to-end gate — BLOCKED_RIGHTS

**Verdict: BLOCKED_RIGHTS — NOT PASS.**
- **Task**: PR-17 G3 end-to-end gate
- **Worktree/branch**: `/home/ygsldr/riichi-ai/worktrees/pr17` @ `rf/pr17-gate`
- **Base**: `11a92a1b4f0fcbea1828c89ab9caba33853ad628` (PR-08..16, full canonical/split, dedup temp fix)
- **Owned files**: this report, `docs/riichiformer/pr_reports/PR-17.md`, `docs/riichiformer/v3/logs/PR17_verify.log`
- **No training, no `runs/exp/RF3_MVP_S`, no checkpoint, no data/schema/code edits.**

The code/data pipeline is green on synthetic and full-corpus indexing, but **no source is
`train_allowed=ALLOWED`**, so a legal real-data training shard/checkpoint cannot be
created. G3 cannot be claimed; this is an honest rights block, not a model result.

## 1. Component table

| component | status | evidence | gate impact |
|---|---|---|---|
| H-01 scoring | ACCEPTED | replay settlement; synthetic + native anchors | green |
| H-02 canonical/dedup | **ACCEPTED** | full 2,512,424 rows, 109 shards, 0 invalid/dup/quarantine | green |
| H-02 split v2 | **ACCEPTED** | L/R 2,512,424 rows, 0 moved vs PR-05, L/R registry hashes match | green |
| H-03 source rights | **ACCEPTED (fail-closed)** | 11 rows, 0 ALLOWED; Tenhou gate exit 1 | **BLOCKER** |
| H-04 rule anchors | ACCEPTED | 51 tests + subtests; R02/R25/R26 flagged | green |
| PR-08 schema v3/2 | ACCEPTED | 66 tests | green |
| PR-09 POV | ACCEPTED | 28 tests; header/visibility audit | green |
| PR-10 DecisionSnapshot | ACCEPTED | 10k sample, 0 illegal; tiles_left fixed | green |
| PR-11 encoder | ACCEPTED | 16 tests; 144 real-log byte-equal | green |
| PR-12 shard/loader | ACCEPTED (code) | 11 tests; rights gate blocks real train | green code / blocked data |
| PR-13 MVP-S model | ACCEPTED | 51 tests + 15 subtests; short GPU smoke | green |
| PR-14 trainer | ACCEPTED (synthetic) | 11 tests; synthetic loss 2.379641→0.000445 | green synthetic |
| PR-15 offline diagnostics | ACCEPTED (synthetic) | 7 tests; real diagnostics NOT_RUN | green synthetic |
| PR-16 arena adapter | ACCEPTED (mock) | 17 tests + 17 subtests; real arena blocked | green mock |
| **PR-17 G3** | **BLOCKED_RIGHTS** | full code/data green; no legal train source | **not passed** |

## 2. Full canonical evidence

- **2,512,424 rows**, **2,512,424 valid**, **0 invalid**, **0 invalid reasons**.
- **2,512,424 components**, **0 duplicate components/groups/rows**, **0 quarantine
  rows/components**, **109 shards**.
- `full_corpus=true`, `canonical_version=rf-canon/1`, `index_version=rf-log-index/2`,
  `rules_version=mjai.tenhou-4p.v1`.
- `canonical_log_index.parquet`: rows 2,512,424; sha256
  `21d9077f04836a123025a5336e38a3746801ea5e2d5a63d45793cc8714efae4c`;
  logical sha256 `a06175c096d0f08ad606f1c4bd45734b36340ba395c387fce6ee7411e563434e`.
- `summary.json`/`manifest.json` sha256
  `25e1a43d506ba70f20a0040943f04eccd7e0f00022905fa5755b583111e9fdce`.
- source rows: `tenhou_old=1,968,147`, `tenhou_recent=544,277`.
- years 2009..2026 counts match the summary; no missing/invalid bucket.

## 3. Full split v2 evidence

| protocol | rows | quarantine | train_allowed_rows | registry hash | manifest sha256 |
|---|---:|---:|---:|---|---|
| L | 2,512,424 | 0 | **0** | `86d7b5578bdb133d40596cf4a87659008a5d231ddb4fd775a5d54aa2ec482170` | `5032d050d52f3c43cc4d054a13891e0bf51e9c11747f1511fad1e2d925c6d1b7` |
| R | 2,512,424 | 0 | **0** | `7ffbe7c672b387c5c76773a74a45295c84faf002b40b6a42b54f95316ec118de` | `37dc7d579dae6cb18420d773f072a0e0f9097a6e3439f49ca7d49d87b7d2af7c` |

- L counts: train 1,968,147; val_holdout 168,777; external_diagnostic 375,500.
- R counts: train 2,321,348; val 89,021; select 89,867; confirm_blocked_previously_used
  12,188.
- **moved_rows vs PR-05 = 0** for both L/R; same/compared rows 2,512,424; 0 old-only rows.
- split registry sha256
  `1afc11759daae0cc4ab5d07abd745a21a0c2c30f094acbe7ead54790c18b51c5`.
- split run log sha256
  `1d86844344c0e7349a54920bfc2a787e7c181f3419b3082c0187eb6e11422e49`.
- split_version `rf-split/2`; no quarantine emitted.

## 4. Test suite (integration tree)

Command:

```bash
cd /home/ygsldr/riichi-ai
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q \
  $(git ls-files 'tests/rf_test_*.py' | tr '\n' ' ')
```

Result: **471 passed, 2 skipped, 242 subtests passed in 155.75s** (exit 0).
The `pr17` worktree has no `vendor/mortal` submodule, so source-contract tests must be
run on the integration tree; the command above is the authoritative run.

## 5. Rights gate — the blocker

`configs/riichiformer/source_rights.csv`: **11 rows, 0 ALLOWED for training**.
- `train_allowed`: 6 RESTRICTED, 5 UNKNOWN; row `status`: 8 RESTRICTED, 3 UNKNOWN.
- Tenhou gate:

```text
rf_check_source_rights.py --source tenhou_old --purpose train --json
{"allowed": false, "reason": "status=RESTRICTED (need ALLOWED; train_allowed=RESTRICTED)", ...}
exit=1
```

Same denial for `tenhou_recent`, `tenhou_raw_logs`, `tenhou_to_mjai_dataset`,
`teacher_local`, `selfplay_own`, `majsoul_optional`, and `third_party_mirror_hf`.
- `PR-14` trainer refusal on a blocked audit shard:

```text
PR14_REFUSED: rights gate refused shard: training loader refuses purpose='diagnostic' shard
exit=2
```

The earlier PR-09 audit P0-2 and H-03 P0-1/P1-3 identify the same rights/registry gap;
PR-12/14 now enforce the gate in the actual shard/training path. This confirms the gate is
wired in; it fails closed before model construction. The H-02 split has **0** `train_allowed_rows`, so even code-complete pipeline
cannot legally produce a real training shard.

## 6. G3 criteria

| criterion | result | evidence |
|---|---|---|
| full canonical clean | **GREEN** | 2,512,424 rows, 0 invalid/dup/quarantine, 109 shards |
| full split v2 stable | **GREEN** | L/R 2,512,424; 0 moved; hashes match |
| code/test integration | **GREEN** | 471 passed, 2 skipped, 242 subtests |
| synthetic model/trainer smoke | **GREEN** | PR-13/14/15 tests; synthetic loss/metrics |
| legal real training source | **BLOCKED** | 0 ALLOWED rows; Tenhou gate exit 1 |
| legal real checkpoint | **NOT_RUN** | cannot create training shard/checkpoint |
| arena strength | **NOT_RUN** | no permitted trained checkpoint |
| confirmation / Tenhou-equivalent claim | **NOT_RUN / forbidden** | no rights, no arena evidence |

**G3 = BLOCKED_RIGHTS**, not PASS.

## 7. What a human must provide

1. **C-EGG/Tenhou written permission** covering the exact scope: which archives/years and
   derived MJAI conversion, model training, storage/retention, redistribution, attribution,
   contact/approval date. Official manual: unrelated-service use requires prior contact
   (`support@c-egg.com`); redistribution of the log index/files is prohibited. An email or
   agreement must be recorded in H-03, not inferred from the converter's code licence.
2. **Mortal/teacher weight licence and provenance** for any distillation/teacher use.
   The Mortal code is AGPL; local checkpoints have no recorded weight licence, and the
   author has not released weights. Code licence ≠ weight licence.
3. **A legal source alternative**: own self-play corpus generated by our own engine (not
   Tenhou/Majsoul ranked games), plus human review, H-04/G2 legality validation, and an
   ALLOWED H-03 row; or another source with explicit written permission.
4. **H-03 CSV update + gate re-run**: mark the permitted purpose ALLOWED with scope,
   evidence URL/quote, retrieval date and a human (non-AI) reviewer; then regenerate split
   rows with `train_allowed=true` and re-run the shard/trainer gate.
5. **Resolve the converter NOTICE/LICENSE-DATA contradiction** (CC BY claim vs no licence)
   and record the restrictive reading; do not rely on NOTICE.
6. **Lead-approved IO/training window** for any full-corpus build/training, with the
   source gate passing before model construction.

## 8. What is already green

- Full H-02 canonical/dedup/split and 471-test integration suite.
- PR-08..PR-16 code paths, including POV visibility, decision snapshots, byte-equal
  offline/online encoder, shard format/loader, MVP-S forward, synthetic BC trainer,
  synthetic offline diagnostics, and **mock** arena adapter.
- H-04 rule anchors and the R02/R25/R26 caveats.
- The rights gate itself (fail-closed, exit 1/2) and the training-path refusal.

## 9. NOT_RUN / blocked descendants

**NOT_RUN**: real Tenhou training, real checkpoint production, real arena strength,
confirmation, Tenhou-equivalent behaviour claim.

**Blocked by BLOCKED_RIGHTS (real-data-dependent)**: PR-18/19/20 data quality/recency/
weighting A/B; PR-21 teacher/distillation; PR-22/23 memory experiments; PR-24 onward
real-data training/eval/scale/arena runs through PR-35. Mock/code-only work (e.g. PR-16
mock adapter, synthetic tests) may continue, but **no result from it is G3/arena evidence**.

No Tenhou-equivalent claim is made. No training was run; no checkpoint, `runs/exp/RF3_MVP_S`,
or frozen data was written.
