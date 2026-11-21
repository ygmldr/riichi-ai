# Independent Review Record — VNext

## Review 1

**Reviewer**: independent subagent (read-only)
**Snapshot**: `138a77b0d39818f97195641859dff8cdce41d79e` (branch later moved)
**Verdict**: **FAIL** — not a code merge verdict; it identified blocking defects.
**Caveat**: the reviewer used a `git archive` of `138a77b` while the Lead was
committing; it explicitly flagged the stale-tree caveat.

### P0 findings and disposition

1. **Rank target used as row index** in `vnext/train.py`.
   - Reviewer reproduction: 2 decisions with ranks `[0,3]` raised `IndexError`.
   - Fixed in `63ffd3b`: `rank_loss` selects the **supervised decision rows**
     and passes their rank classes to `F.cross_entropy`; it never indexes the
     decision axis by label.  New tests cover ranks `[3,0]` and a rank-only
     batch.
2. **Critic gradient isolation not in the training path; deployed critic was a
   bare Linear**.
   - Fixed in `63ffd3b`: `VNextPolicy` uses the two-layer
     `DecisionRankCritic`; `VNextConfig.critic_grad_mode` defaults to
     `detach_actor`; the model-level test proves an Actor gradient norm of zero
     on a critic-only backward while the critic head still learns.
3. **Tile state accumulated prior-kyoku rivers** because the feature builder
   iterated the full visible match while the event stream filtered current
   kyoku.
   - Fixed in `63ffd3b`: tile feature construction filters
     `event.kyoku_idx == current_kyoku`; a two-kyoku test proves a prior
     shimocha discard is absent and a current-kyoku discard is present.

### P1 findings and disposition

4. **Aka channels were broadcast scalars; genbutsu duplicated river**.
   - Aka channels are now per-tile (only row 4/13/22 carries its colour);
     genbutsu now requires that opponent's accepted riichi plus a river tile,
     so it is not a duplicate river count.  Test asserts `[13,6] == 0` and
     `[4,6] == 1`.
5. **No p=0 test**.
   - Added a manual BOS-only `VNextModelBatch` forward test; cross-attention
     offset 0 works.
6. **Test-count/log inconsistency**.
   - The final committed evidence is now `72 passed` for
     `tests/rf_test_vnext_*.py` and `125 passed, 6 deselected, 6 subtests` for
     the combined VNext + B0 + arena-synthetic run; logs are committed under
     `logs/`.  The stale untracked 60-test log is not cited.
7. **No end-to-end arena multi-kan test**.
   - `decode_native_action`/mask-helper tests exist; the full
     `RFMjaiLogEngine._react` native multi-kan path remains `NOT_RUN` because
     the VNext worktree has no initialized native submodule/checkpoint.  This is
     stated in `PR01_P1_FIXES.md`.
8. **PR-11 requested training while rights gate was blocked**.
   - `PR11_PREREG.md` now says no authorization is requested while the gate is
     `BLOCKED_FOR_NEW_TRAINING`; any future request is conditional on a
     rights-cleared audited subset.
9. **B0 validation sync-count wording overstated**.
   - Docstring now says vectorized per batch with a small fixed number of
     batch-level conversions, not "one scalar per batch".
10. **Rank term/accumulation inconsistency**.
   - Rank CE is now an additive **sum** over supervised rows; denominator is
     the window BC weight, else the BC weight, else the rank-row count; a
     rank-only batch no longer errors on a zero BC weight.

### P2 noted

* `select_kan` now requires `decision.legal_mask == kan_mask`.
* `VNextBatchBuilder` pads event type with `PAD_EVENT_ID`.
* `assert_forward_safe` also rejects a bare `rank` field name.
* `pbrs_rewards` is documented as an alternative recording of the same
  terminal objective, not a different algorithm.

## Review 2

**Reviewer**: second independent read-only reviewer
**Snapshot**: `63ffd3b` blobs (contracts `bbacec6`, model `3cb316f`, observable
`a6ad1be`, train `c953b87`, arena `ad2c1f5`)
**Verdict**: **PASS_WITH_P1** — no P0 established by code reading.

The reviewer confirmed: future events rejected by `DecisionInput`; current-kyoku
filter; causal backbone + prefix key mask; native 46/aka order; 24 permutations
and zero-sum PT; fail-closed EMA shadow.  It could not execute tests in its
session and explicitly marked counts/hashes/native runs `NOT_VERIFIABLE`.

### P1 dispositions after the review

1. **Rank-only batch reported loss but did not backward.**  Fixed: `train_step`
   now backprops when `bc.valid > 0 or rank is not None`; the rank-only test
   asserts `optimizer_stepped` and non-None parameter gradients.
2. **Targets could be misaligned with stream-grouped batch rows.**  Fixed:
   `VNextBatchAdapter` now reorders target metadata by explicit
   `decision_id` into the `VNextModelBatch` order and `VNextBCBatch` validates
   unique `decision_ids`; an interleaved two-stream reader test proves the
   mapping.
3. **No end-to-end `RFMjaiLogEngine._react` multi-kan test.**  Still `NOT_RUN`
   (native submodule/checkpoint absent); helper and second-stage decode are
   tested, and the limitation is documented.
4. **Aka-kan action-index skew.**  Added a real `"5mr"` kakan test: the mask
   contains index 34 and `KAN_SELECTION` decodes `"5mr"` with phase
   `KAN_SELECTION`.
5. **PBRS lacked zero-sum Phi guard.**  Added the same guard as the delta form
   and a negative test; also added a general `permute_prob_perm` test.
6. **Opponent ankan could be counted as public meld.**  Fixed: ankan melds are
   skipped; test asserts the opponent-meld channel remains zero.
7. **No real-EMA arena run.**  Still `NOT_RUN` (no VNext checkpoint/native
   build); the synthetic EMA shadow test remains.

P2 items were addressed where cheap: default GRP PT id, external marginal
validation, B0 sync wording, `select_kan` exact mask, event-type PAD padding,
removal of the overbroad bare-`rank` guard, and correcting the GRP
permutation-consistency doc.

The final Lead evidence is `72 passed` for VNext-only and `125 passed, 6
deselected, 6 subtests` for the combined CPU suite.  This is still not a native
Arena or a strength claim.

## Review 3 — native / GPU changes

**Reviewer**: third independent read-only reviewer (forked context)
**Snapshot**: `9ce1071` + uncommitted native/GPU work (before the Lead's final
review fixes)
**Verdict**: **PASS_WITH_P1**; no P0 established.

The reviewer confirmed the RoPE inference-cache fix, the native smoke's
`synthetic_fixture`/`strength_claim: NONE` labelling, and the absence of a
future/target leak in the reviewed bridge.

### P1 dispositions

1. **Pinned 3-arg engine compatibility.**  Confirmed: the clean upstream engine
   has `react_batch(obs, masks, invisible)` while the patched build adds
   `events`.  Fixed: `MortalCheckpointAdapter` detects the keyword at
   construction, passes `events` only when supported, and raises a clear
   `AnchorError` if a new driver supplies events to an old engine.  Added
   `test_old_three_arg_engine_fallback`.
2. **Native smoke could report `PASS` unconditionally.**  Fixed: the script now
   asserts game count, `rf_candidate == RUN`, synthetic mode, non-self-control,
   zero illegal/fallback, and `kan_selections` agreement.  It emits
   `not_run: ["multi_kan_native_end_to_end"]` when the second stage was absent.
3. **Anchor identity gate weakened.**  Not treated as a PASS: the frozen
   registry was not rewritten.  The smoke uses a local registry; tests skip on
   driver/native hash drift and the report records
   `ANCHOR_GATE: ENV_MISMATCH_REQUIRES_REBASE`.  Reissuing M5D-R0 is a separate
   decision.
4. **Unsupported exact sync count / mixed before-after logs.**  Fixed: the
   precise `6,570` claim was removed; the profiler observation is described
   qualitatively, and the after-fix table uses medians from three runs with the
   observed range.  The old pre-fix GPU smoke log is labelled history.
5. **Multi-kan remains `NOT_RUN`.**  Accepted.  The 4-game smoke had
   `kan_selections=0`; PR-05 is marked `PARTIAL`, not complete for
   `KAN_SELECTION`.

### P2 dispositions

* `_load_mortal_modules` no longer inserts the vendor directory into
  `sys.path`; it loads `event_data` explicitly and restores the prior module.
* Anchor skips now carry the explicit frozen-registry/`ANCHOR_REBASE` reason.
* Native smoke semantics are machine-checked rather than prose-only.

Lead suite after these fixes: **142 passed, 6 skipped, 13 subtests**
(`logs/VNEXT_COMBINED_FINAL.log`).  These fixes were not re-reviewed.

## Review 4 — real-data trainer

**Reviewer**: independent read-only reviewer
**Snapshot**: `0365c94`/`fb4896c` plus current files at `95eb619`
**Verdict**: **PASS_WITH_P1**; no P0 established.

The reviewer confirmed private-shard authentication, rank normalization,
the decoded-stream cache (direct-vs-cached decision hash test passes), and the
pilot metrics.  P1 findings and Lead fixes:

1. **Resume replayed the prefix and used non-strict loading.**  Fixed:
   checkpoints store `batches_seen` plus train/val shard hashes in `extra`;
   resume uses `strict_resume=True`, requires matching shard hashes, and skips
   exactly `batches_seen` batches.  A 5→10 step resume test passes with
   `start_batches=5`.
2. **Output guard allowed frozen run dirs and overwrote existing dirs.**  Fixed:
   refuse repo root, `runs/1v3`, `runs/variance`, `runs/exp/PINNED`, datasets,
   checkpoints, vendor, `.git`, and any non-empty dir without `--resume`; tests
   cover `runs/1v3`/`runs/variance` and a busy dir.
3. **Validation shard not authenticated.**  Fixed: require
   `purpose=diagnostic`, reject the train shard hash, and reject a val shard
   sharing the train split manifest; a negative test covers a train-as-val
   attempt.
4. **Empty/all-censored shard could spin.**  Fixed: 1,000 consecutive
   no-update batches raise a hard error; wall-cap runs report `STOPPED`, not
   `COMPLETED`.
5. **Wall-cap status and provenance were incomplete.**  Fixed: reports
   `stopped_reason`/`target_steps`; checkpoint provenance and `run.json` include
   train/val shard SHA-256.
6. **Rank 4 and malformed ranks untested.**  Added `normalise_rank` with strict
   1..4→0..3 mapping and tests.

P2 items: duplicate final `metrics.jsonl` save removed via `last_saved_step`;
LR args validated.

Lead suite after fixes: **77 VNext tests**, **146 combined, 6 skipped,
13 subtests** (`logs/VNEXT_REAL_TRAIN_SUITE.log`,
`logs/VNEXT_COMBINED_REAL.log`).  These fixes were not re-reviewed.
