# V3.2 Wave 1 — Lead status

Updated: 2026-11-21 (Asia/Shanghai). Lead single-writer status; subagent claims are
not accepted until the Lead reruns the commands.

## Accepted base

| Item | Value |
|---|---|
| Remote branch | `origin/rf/v3.2-policy-parallel-plan` |
| Remote SHA | `a27df099e9b395a742ebc3a239e003d0f8896b92` |
| Accepted dev base | `rf/v3.2-local` `cbf166a356f83c739b31f0b5a0b241b954ef9e0a` |
| Acceptance record | `docs/riichiformer/v3_2/ACCEPTANCE_LEAD.md` |
| Test result | 494 passed, 2 skipped, 242 subtests (145.41 s) |
| Training-chain result | PASS_WITH_P1; real H-02 10-game private shard; 6-layer 5,217,582-param 2-step GPU smoke; strict resume; split mismatch refused |

The base was created from the remote branch, not from the old V3.2 pack code. The old
pack remains reference material under `/tmp/v32pack`; it was not copied over the remote
tree. The `vendor/mortal` submodule in the accepted worktree is pinned at
`0cff2b52982be5b1163aa9a62fb01f03ce91e0d2`.

## Lanes and worktrees (all from `cbf166a`)

| Lane | Task | Worktree | Branch | Owner (subagent id) | Owned files |
|---|---|---|---|---|---|
| A | PR-37 follow-up | `worktrees/v32-policy` | `rf/v32-policy` | `e1ab6d31-53e7-419e-8f7b-efa9de87dcb8` | `riichiformer/policy/**`, `data/shard.py`, `data/loader.py`, `scripts/rf_build_shards.py`, `scripts/rf_train.py`, policy tests/config, `docs/riichiformer/v3_2/**` |
| B | PR-38 + PR-39 | `worktrees/v32-compute` | `rf/v32-compute` | `7a3fd7f1-dcc1-4678-b88d-5c3d051ac5ad` | `models/**`, `data/encode.py`, new `data/multi_decision.py`, `train/**`, model/training tests |
| C | PR-40 | `worktrees/v32-rules` | `rf/v32-rules` | `8b81322a-7442-49d2-9621-cec8d9126ad3` | `patches/**` (new), rules tests, `docs/riichiformer/v3/TENHOU_RULE_PARITY.md`, new V3.2 rules doc |
| D | PR-41 + PR-42 | `worktrees/v32-selfplay` | `rf/v32-selfplay` | `55672f97-54b6-4761-8aff-29f91dbab019` | new `riichiformer/selfplay/**`, `eval/opponents.py`, `eval/arena.py`, new selfplay/arena scripts/tests |

No lane may edit another lane's owned files. `scripts/rf_train.py` is A-owned this wave;
B supplies library APIs only. C must not commit a `vendor/mortal` pointer change.

## Merge order (Wave 2)

1. A (PR-37) first: policy config authentication and fail-closed default. Rerun the
   policy/shard/loader negative suite and the full suite.
2. B (PR-38/39) second: multi-decision forward and trainer numerics. Rebase on A if
   `data/encode.py`/`models/**` interfaces changed; joint-vs-prefix parity must be rerun.
3. C (PR-40) can merge before or after B only if the patch does not touch shared
   schemas; the Lead applies the patch to the main dirty vendor and rebuilds
   `libriichi.so`, then reruns rules parity.
4. D (PR-41/42) last: wire the accepted model/loader interfaces into selfplay/arena;
   synthetic fallback must stay labelled as such.
5. New integration branch from `rf/v3.2-local` (or the current integration branch);
   no force-push, no direct push to `main`, no merge of the unfiltered history.

Each merge requires an independent reviewer report. The Lead alone updates this file,
`PARALLEL_STATUS.md`, and `INTEGRATION_LOG.md`.

## Resource policy

- One GPU training/evaluation process at a time. This wave is CPU/static work; no agent
  may run the 5070 Ti without the Lead's explicit lock.
- Heavy IO (full 2.5M rebuild), Rust rebuilds, and confirm-seed consumption are
  serialized by the Lead. The accepted 10-game shards are fixtures, not a new full
  corpus.
- Seed/opponent pools: train/select/confirm separated by D; no confirm pool reuse.
- No network uploads, no public model/data release, no Tenhou/雀魂 online AI play.

## Open P1s from acceptance

1. Private shard policy hash is not authenticated against a policy config.
2. Builder defaulted to the permissive private profile.
3. No committed policy config / operator attestation artifact.
4. Stale trainer docstring and missing `--init-weights-only`.
5. Mixed-source metadata reports an aggregate rights status (per-game status retained).

A owns 1–3 and the `rf_train.py` pass-through; B owns 4's library side. These must be
closed before PR-43 claims any private-research training chain.

## Next gates

- Wave 1: four lane reports + independent reviewer.
- Wave 2: merged integration, full suite, PR-38 parity, PR-39 large-batch equivalence,
  PR-40 native fixture evidence, PR-41/42 0-illegal/score-conservation.
- PR-43: separate `CODE_PASS`, `SYNTHETIC_E2E_PASS`, `OWN_SELFPLAY_TRAIN_PASS` (only if
  actually run), `EXTERNAL_LICENSE_STATUS` (unchanged facts), `STRENGTH_NOT_RUN`.

## Wave 1 result (2026-11-21)

All four lanes committed and were merged into `rf/v32-integration`; the accepted base
was not rewritten and no public branch was pushed.

| Lane | Commit | Result |
|---|---|---|
| A PR-37 | `87afe78` | PASS_WITH_P1; P1 bool-coercion fixed in `c426ac7`; policy config/hash/allowlist authentication in place |
| C PR-40 | `8248496` | PASS_WITH_P1; R02/R25/R26 implemented, R29 UNVERIFIED; evidence P1s fixed in `c426ac7` |
| B PR-38/39 | `4e4754b` | merged; PR-38 PASS, PR-39 PASS (partial-override P1 fixed and re-verified); 6-layer CLI, strict resume, accumulation and parity tests |
| D PR-41/42 | `4e380a3` | merged; PR-41/42 PASS (P2 only); real local-engine selfplay and duplicate arena with explicit `tenhou_approx` labels |

Integration full suite after all merges and P1 fixes: **683 passed, 2 skipped, 281
subtests** (187.73 s); after PR-38 loader/trainer integration: **707 passed, 2 skipped,
281 subtests** (182.20 s), log `docs/riichiformer/v3/logs/V3_2_W1_full_suite.log`.

G3/PR-43 split result is in `PR43_GATE.md`: CODE_PASS and private H-02 GPU smoke PASS;
own-selfplay train and strength are NOT_RUN. Third-party rights remain 11 rows / 0
ALLOWED. Remaining P1s: R29/continuation evidence, PR-38 loader/trainer wiring,
selfplay POV→shard conversion, and applying 0014 to a fresh vendor.


## Wave 2 start — PR-38 loader/trainer and B0 pilot (2026-11-21)

- `rf/v32-b0` `ce5a290` (merged as `45eefa1` code + final report) wires
  `iter_multi_decision_batches`, `training_step_multi`, `--multi-decision` and adaptive
  `max_events`/`max_positions`; legacy prefix path remains default.
- Full suite on the patched integration tree: **707 passed, 2 skipped, 281 subtests**
  (182.20 s), log `docs/riichiformer/v3/logs/V3_2_W2_full_suite.log`.
- 500-game deterministic B0 sample and shard: 333,224 decisions, shard sha
  `f474a3f1…`; two 500-step GPU pilot seeds completed (26,918 decisions / 432,147
  tokens per seed), checkpoint shas `f21c8a3e…` / `ac7346d1…`; strict resume to step
  501. Report `PR44_B0_PILOT.md`. This is a pilot, not full B0 and not strength.
