# GPT web-review instruction — RiichiFormer V3.3 Wave 2

> **Current publication status (2026-11-21):** the V3.3 docs are **not pushed**.
> `origin` heads are still `a3ccd85` (main), `a27df099`
> (`rf/v3.2-policy-parallel-plan`) and `a4492e4`
> (`riichiformer-v3-filtered`, V3.1). The local integration is
> `rf/v3.3-integration` at `378eb6b` / tag `rf3.3-w2`; it is a local development
> history and must **not** be pushed unfiltered.
>
> Use **Mode A (uploaded bundle)** now, or replace the `{{DOCS_*}}` placeholders
> after a docs-only branch is published (Mode B). If the URLs 404, do not pretend
> you reviewed them.

## Mode A — no web publication (recommended until a docs branch is approved)

Upload `/tmp/RiichiFormer_V3_3_Review_Bundle.zip` (or the individual files listed
below) to GPT and use the prompt in §3 with `MODE = UPLOADED_BUNDLE`. Mark every
code/test claim that is only asserted by a document as **NOT_VERIFIABLE** unless
the bundle contains the source/log.

## Mode B — docs-only web branch

Only after the owner explicitly approves publishing a **docs-only snapshot**
(no unfiltered history, no private data/checkpoints), use:

- `{{REPO_URL}}` = `https://github.com/ygmldr/riichi-ai`
- `{{DOCS_BRANCH}}` = e.g. `review/v3.3-w2-docs`
- `{{DOCS_COMMIT}}` = the docs snapshot commit
- `{{BASE_COMMIT}}` = `e76022600018e9adc6536ccbb904db8c9dc498b3`
- `{{INTEGRATION_COMMIT}}` = `378eb6b` (local; not claimable as public unless the
  code branch is also published)

GitHub blob pattern:

`{{REPO_URL}}/blob/{{DOCS_COMMIT}}/docs/riichiformer/v3_3/WAVE2_REVIEW.md`

## 1. What the reviewer is reviewing

A research-engineering wave, not a strength result. The project is a RiichiFormer
fork of Equim-chan/Mortal. V3.3 Wave 2 delivered:

- PR-46: frozen self-trained Mortal-5D anchor `M5D-R0`, native 1v3 arena,
  relative-PT scorer and a weak-pilot RF bridge smoke.
- PR-45/47: 100-game native self-play validation corpus, read-only diagnostics,
  and selfplay→POV→shard→loader→train→checkpoint roundtrip.
- PR-48: frozen B0 training protocol, validation/selection/resume code, CPU smoke.
- PR-40R: clean-vendor rebuild and executable R29 ippatsu fixture.

Hard facts to check, not assume:

- Actual local base is `e760226` (tag `rf3.2-w2`); `00e6576` is an ancestor, not
  the tip. The pack's older `00e6576`/707-test text is historical.
- `M5D-R0 = checkpoints/mortal.pth`, SHA-256
  `0a44613cb9c7edae1057cf6a2c160c3971c78130e18e07f28bb43e8f62abe532`,
  396,000 steps, pure CNN `c192/b40`, `use_event_branch=false`.
- Final CPU suite: **841 passed, 6 skipped, 319 subtests**, 271.95 s, CUDA
  disabled. A prior intermediate run accidentally executed the repository's
  pre-existing tiny CUDA benchmark test; this is disclosed, not hidden.
- No full B0 epoch, no Trend/Select/Confirm arena, no strength/dan/Tenhou claim.
- Third-party rights: 11 rows, 0 `ALLOWED`; `operator_private_research` is not a
  licence and does not permit publication/redistribution/online play.

## 2. Files to read, in order

If the docs-only branch is available, prepend
`{{REPO_URL}}/blob/{{DOCS_COMMIT}}/` to each path.

1. `docs/riichiformer/v3_3/WAVE2_REVIEW.md`
2. `docs/riichiformer/v3_3/STATUS.md`
3. `docs/riichiformer/v3_3/REVIEW_CHECKLIST.md`
4. `docs/riichiformer/v3_3/INTEGRATION_LOG.md`
5. `docs/riichiformer/v3_3/PR46_M5D_ARENA.md`
6. `docs/riichiformer/v3_3/PR45_VALIDATION.md`
7. `docs/riichiformer/v3_3/PR47_SELFPLAY_ROUNDTRIP.md`
8. `docs/riichiformer/v3_3/PR48_B0_PROTOCOL.md`
9. `docs/riichiformer/v3_3/PR40R_NATIVE_RULES.md`
10. `docs/riichiformer/v3_3/EVAL_POOL_REGISTRY.md`,
    `GPU_BUDGET.md`, `resource_locks.yaml`
11. `docs/riichiformer/v3_3/pack/01_MASTER_PLAN.md` and
    `03_EXPERIMENT_PROTOCOL.md` (the gate definitions)
12. `docs/riichiformer/v3_3/logs/V3_3_W2_full_suite_final.log`,
    `V3_3_A0_100_selfcontrol_run.json`,
    `V3_3_A0_100_selfcontrol_report.json`
13. Historical context: `docs/riichiformer/v3_2/PR38_LOADER_INTEGRATION.md`,
    `PR44_B0_PILOT.md`, `PR44_B0_PREREG.md`, `PR40_RULES.md`; R29/R02/R25/R26 in
    `docs/riichiformer/v3/TENHOU_RULE_PARITY.md`.

If the code branch is also available, inspect:
`riichiformer/eval/{mortal_anchor.py,relative_pt.py,m5d_arena.py}`,
`riichiformer/validation/{e2e.py,reader.py,shard_build.py,disjoint.py}`,
`riichiformer/train/b0.py`,
`scripts/{rf_eval_m5d.py,rf_train_b0.py,rf_build_validation.py,rf_offline_diag.py}`,
`tests/rf_test_{mortal_anchor,relative_pt,m5d_arena}.py`,
`tests/rf_test_validation_{disjoint,metrics,e2e}.py`,
`tests/rf_test_b0_{protocol,cli}.py`,
`tests/rf_test_native_rules_r29.py`,
`patches/0015-rule-r29-ippatsu-fixture.patch`.

## 3. Copy-paste prompt for GPT

```text
You are an independent, adversarial reviewer for RiichiFormer V3.3 Wave 2.
You are read-only: do not edit code, do not write patches, do not accept a claim
because a document says PASS. You have web/file access; use the URLs or uploaded
files listed below. If a URL is unreachable, say so and mark the check
NOT_VERIFIABLE instead of guessing.

MODE: {{UPLOADED_BUNDLE | DOCS_WEB}}
REPO: {{REPO_URL}}
DOCS_BRANCH: {{DOCS_BRANCH}}
DOCS_COMMIT: {{DOCS_COMMIT}}
BASE_COMMIT: e76022600018e9adc6536ccbb904db8c9dc498b3 (tag rf3.2-w2)
LOCAL_INTEGRATION: 378eb6b (tag rf3.3-w2; local-only unless explicitly published)
ANCHOR: checkpoints/mortal.pth
ANCHOR_SHA256: 0a44613cb9c7edae1057cf6a2c160c3971c78130e18e07f28bb43e8f62abe532
FINAL_SUITE_CLAIM: 841 passed, 6 skipped, 319 subtests, 271.95 s, CUDA disabled
STATUS_CLAIM: full B0 / Trend / Select / Confirm / strength = NOT_RUN

Review these files in the order given:
[list from §2]

Your job:
1. FACT CHECK. Verify branch/commit/tag relationships, anchor SHA, suite counts,
   shard/checkpoint hashes, lane SHAs, and whether each quoted log actually
   supports the sentence around it. Flag any stale `TBD`, `UNVERIFIED`, old hash,
   or contradictory claim (e.g. PR-46 lane registry hash vs merged registry hash).
2. GATE LOGIC. Check G-V33-W2, A0, D0, B0, C0. A0 requires a 100-game 4-seat
   self-control smoke; D0 requires a disjoint 100-200-game validation corpus,
   read-only diagnostics and E2E; B0 code/CPU smoke is not full B0; C0 requires a
   separately registered confirm pool and a frozen winner. Reject any upgrade of
   `NOT_RUN` to PASS.
3. EVIDENCE QUALITY. Are the logs committed and consistent? Does the A0 JSON show
   25 sets / 100 games / ΔPT 0 / hist [25,25,25,25] / 0 illegal? Does the 100-game
   validation shard have purpose=diagnostic, train_allowed=false, zero B0
   intersection? Does E2E train only on a separate train-purpose fixture and keep
   validation read-only? Does B0 isolation reject train-purpose validation?
4. STATISTICS. For PR-46 check the 5D profile [75,30,0,-105], the 7D auxiliary
   [90,45,0,-135], delta = 4/3 * pt5_C, complete four-seat sets, cluster bootstrap
   over sets, tie rule, and failures on incomplete/duplicate/key/score defects.
   Check that CE/loss is never used as strength.
5. CLAIMS AND RIGHTS. Flag any claim of dan/rank, "九段/天凤位", Mortal
   superiority, official Tenhou equivalence, publication permission, or a licence
   that does not exist. Check GPU disclosure and frozen-path writes.
6. CODE-LEVEL (only if source is available): validation-isolation gate,
   train/validation game-id disjointness, no train-on-diagnostic in E2E, no
   hidden-tile access in the RF bridge, no silent prefix fallback in B0, R29
   fixture actually observes ippatsu, no submodule pointer change.
7. OUTPUT EXACTLY:
   VERDICT: PASS | PASS_WITH_P1 | FAIL
   GATES: W2/A0/D0/B0/C0 with one-line evidence
   P0 (blocking): numbered, with file:line or doc anchor and why
   P1 (must fix): numbered, with precise reproduction or missing evidence
   P2 (non-blocking): numbered
   EVIDENCE CHECKED: commands/URLs and what they actually showed
   NOT_VERIFIABLE / MISSING: explicit
   CONTRADICTIONS: exact quotes
   FINAL: one paragraph, no praise, no edits
Do not claim you ran tests unless the bundle/log shows an execution; document-only
review cannot certify runtime behavior.
```

## 4. Specific traps to look for

- The V3.3 pack says base `00e6576`, 707 tests and “Reviewer still running”. The
  actual tree is `e760226`, 708 tests at base, 841 after Wave 2. Treat the pack text
  as historical planning input, not current status.
- `PR46_M5D_ARENA.md` intentionally keeps Dev-A lane-time hashes
  (`registry_hash=4df5c825…`, build `ceb9287c…`). The merged integration registry
  is `70f420a8…`, build `582e367…`; this is documented, not a contradiction.
- `PR45_VALIDATION.md`/`PR47_SELFPLAY_ROUNDTRIP.md` contain historical 2-step
  numbers from before the P1 correction. The current E2E trains on a separate
  train-purpose fixture; do not read the old table as current behavior.
- `WAVE2_REVIEW.md` still has `NOT_RUN` for full B0, Trend/Select/Confirm,
  strength, four-riichi/triple-ron, and full Board R29. Do not treat
  `CODE_PASS`, `PASS at smoke scale`, or a “P2” as a full gate.
- The repository has a public filtered branch `riichiformer-v3-filtered` at
  `a4492e4`; it does **not** contain V3.3. Do not assume the V3.3 docs are public.
- `operator_private_research` / `local_research_only` is a local routing decision,
  not a copyright licence. The external registry still has 0 `ALLOWED` rows.

## 5. Publishing note for the owner

Do **not** push `rf/v3.3-integration` directly: it is unfiltered development
history. A safe web-review publication is a **docs-only orphan snapshot** (or a
private repo), containing only `docs/riichiformer/v3_3/**` and the small evidence
JSON/logs, with no checkpoints, shards, absolute private paths, or `.git` history
from the development branch. Push only after explicit approval, then replace
`{{DOCS_BRANCH}}`/`{{DOCS_COMMIT}}` above.
