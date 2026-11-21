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
   - The final committed evidence is now `68 passed` for
     `tests/rf_test_vnext_*.py` and `120 passed, 6 deselected, 6 subtests` for
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

A second independent reviewer was asked to re-check the post-fix tree but did
not return a verdict before its run was stopped.  There is therefore **no
second-review PASS**; this document must not be read as a PASS.  The Lead's own
regression evidence is `VNEXT_COMBINED_CPU.log` (120 passing), and the first
reviewer's P0/P1 items have explicit dispositions above.
