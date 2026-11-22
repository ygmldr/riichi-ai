# RiichiFormer VNext Core — External Audit Packet

**Snapshot**: branch `rf/vnext-core`, commit `46c3b83` (code snapshot; this
packet is a later docs-only commit)
**Base**: `8a915d1` / tag `rf3.3-w2`
**Remote**: `git@github.com:ygmldr/riichi-ai.git`
**Branch URL**: https://github.com/ygmldr/riichi-ai/tree/rf/vnext-core
**Compare URL**: https://github.com/ygmldr/riichi-ai/compare/8a915d1...rf/vnext-core
**PR creation link**: https://github.com/ygmldr/riichi-ai/pull/new/rf/vnext-core

---

## Reviewer prompt (copy this to GPT)

> You are an independent senior reviewer for a mahjong-AI training repo. Audit
> the `rf/vnext-core` branch for correctness, contract violations, data leakage,
> rights-gate bypass, checkpoint/resume integrity, and overclaiming. Treat every
> numeric claim below as a hypothesis to verify against the linked code, logs,
> and artifacts; do not infer strength from training loss. The raw Tenhou data is
> NOT included and cannot be redistributed. Focus on whether the code and
> evidence justify the stated claims, and whether any P0/P1 issue is hidden by
> the documentation. Report findings as P0/P1/P2 with exact file:line and a
> concrete reproduction or counterexample.

---

## 1. Scope

The branch adds the VNext implementation and the real-data BC path:

- VNext contracts, observable event/state encoder, model, critic, GRP24, RL toy
  contracts (`riichiformer/vnext/`).
- Real `rf-shard/3` BC adapter and trainer:
  `riichiformer/vnext/data.py`, `riichiformer/vnext/train.py`,
  `scripts/rf_train_vnext.py`.
- Operator-authorized local data pipeline: audit → canonical → `rf-split/2` →
  `rf-shard/3`, plus the 2009 full-year shard and 2023 held-out val shard.
- Read-only Census (`scripts/rf_vnext_census.py`), native checkpoint smoke
  (`scripts/rf_vnext_native_checkpoint_smoke.py`), 2009 launcher
  (`scripts/run_vnext_2009.sh`).
- Cross-cutting fixes in `riichiformer/data/loader.py`,
  `riichiformer/eval/m5d_arena.py`, `riichiformer/eval/mortal_anchor.py`,
  `riichiformer/models/local_transformer.py`, `riichiformer/train/b0.py`,
  `scripts/rf_train_b0.py`.
- 9,810 insertions / 61 deletions across 101 files from `8a915d1` to `46c3b83`.

This is **not** a claim of playing strength, dan, Mortal parity, or Arena
performance. `Confirm` pools were not used.

---

## 2. What is claimed

| Claim | Evidence | Status |
|---|---|---|
| VNext CPU contracts/model/trainer tests pass | `logs/VNEXT_REAL_TRAIN_SUITE.log`: 79 passed | PASS |
| Combined VNext/B0/arena/anchor tests pass | `logs/VNEXT_COMBINED_REAL.log`: 149 passed, 6 skipped, 13 subtests | PASS |
| Real pilot learns on held-out val | 1,000-game train / 500-game val; 10.5k-step checkpoint 500-batch BC CE 1.4744, rank CE 1.1562 | PASS (loss only) |
| 2009 full-year shard is complete | 6,897 games, 4,568,777 decisions, `written=6897 skipped=0`, SHA `f411c9e2...` | PASS |
| 2009 full-year BC completed | 71,387 steps, 9,121.0 s, final checkpoint `step_071387.pt`, SHA `6d16c4bc...` | PASS |
| Final checkpoint validation | 500-batch / 32,000 decisions: BC CE 1.3500, rank CE 1.1276 | PASS (loss only) |
| Final checkpoint drives native bridge | 16 games, 2,827 decisions, 0 illegal, 0 fallback | PASS (integration only) |
| No Confirm/strength claim | docs and logs; no Confirm shard/run | PASS by construction; verify independently |
| 2.51M full corpus one epoch | **NOT_RUN** | open/limitation |

“PASS” here means the stated bounded check passed; it does not mean the policy is
strong or correct in all states.

---

## 3. Data / rights — audit this first

The local data is operator-authorized local research, not an upstream licence.
The external rights registry remains restrictive (`UNKNOWN`/`RESTRICTED`).

Shard manifest for 2009:

```json
{
  "format_version": "rf-shard/3",
  "game_count": 6897,
  "decision_count": 4568777,
  "shard_sha256": "f411c9e2c219ffc3f38fca26c9acaf8195362baee2e833bdc2a8180c9ffaa76e",
  "content_sha256": "8abe4028d6c0e69263e96126e34aa7c881624bab3a3fa880a4d99616511bbd36",
  "metadata": {
    "source_id": "tenhou_old",
    "purpose": "train",
    "research_policy": "operator_private_research",
    "research_policy_hash": "f0d472ee710455672fc8866452c723d802b3e4e44d92ef2bbd7ac6aa976ce2bc",
    "train_allowed": false,
    "train_eligible_under_policy": true,
    "rights_status": "UNKNOWN",
    "external_license_status": "UNKNOWN"
  }
}
```

**Audit question**: `train_allowed=false` but `train_eligible_under_policy=true`.
Is the local policy an explicit operator routing decision or an unintended
rights-gate bypass? The project’s position is the former; the external rights
registry was not rewritten. Raw data is not committed and cannot be shipped with
this packet.

---

## 4. 2009 full-year run

Artifacts are local-only under `runs/` (gitignored):

```text
runs/vnext_bc_2009/metrics.json
runs/vnext_bc_2009/validation.jsonl
runs/vnext_bc_2009/candidates/step_071387.pt
runs/vnext_bc_2009/candidates/step_071387.pt.json
```

`metrics.json`:

```json
{
  "batches_seen": 71387,
  "steps": 71387,
  "target_steps": 71387,
  "status": "COMPLETED",
  "stopped_reason": "target_reached",
  "exhausted": true,
  "wall_seconds": 9121.015974726004,
  "max_prefix_tokens": 4096,
  "long_prefix_skipped": 460,
  "loss_first": 2.46868896484375,
  "loss_last": 2.5181171894073486,
  "loss_mean": 1.553654037598731,
  "last_validation": {"step": 70000, "bc_ce": 1.3461721225611927,
                       "rank_ce": 1.2503219991455843}
}
```

The final checkpoint sidecar reports:

```text
format: rf-checkpoint/1
sha256: 6d16c4bc6dd97f744feab05d8cfa2d77e4646dde3a15358b4ff7cf85e6ac391c
step: 71387
decisions: 4568765
tokens: 32391386
provenance_hash: 65b2cd7ae8e095aab0feea62254e2de9878f339fa17c77d9b9d638b6c22f2864
```

Final independent 500-batch / 32,000-decision eval:

```json
{"bc_ce": 1.349995789815339,
 "rank_ce": 1.1275775090599245,
 "total_loss": 1.4627535407213315,
 "checkpoint_sha256": "6d16c4bc6dd97f744feab05d8cfa2d77e4646dde3a15358b4ff7cf85e6ac391c"}
```

**Two caveats the reviewer must check**:

1. `loss_last` (2.5181) is higher than `loss_first` (2.4687); this is the last
   *batch* loss and is noisy. Validation, not train loss, is the reported metric.
2. The 460 skipped decisions are absent from batches. The adapter therefore
   exhausted the shard at about 71,379 batches and restarted for ~450 decisions
   (0.01%) to reach 71,387. `exhausted=true` records this. This is disclosed, not
   silent; assess whether the selection/replay is acceptable.

---

## 5. Validation curve

50-batch in-training samples are noisy; 500-batch re-evals are the stable
numbers for checkpoints where available.

| step | 50-batch BC / rank | 500-batch BC / rank |
|---:|---:|---:|
| 5,000 | 1.4891 / 1.2881 | 1.5019 / 1.1787 |
| 10,000 | 1.4666 / 1.2870 | 1.4895 / 1.1472 |
| 15,000 | 1.4539 / 1.2557 | 1.4720 / 1.1363 |
| 20,000 | 1.4396 / 1.2705 | 1.4551 / 1.1400 |
| 25,000 | 1.4299 / 1.2768 | 1.4426 / 1.1363 |
| 30,000 | 1.4211 / 1.2763 | 1.4311 / 1.1434 |
| 35,000 | 1.4025 / 1.2535 | 1.4182 / 1.1360 |
| 40,000 | 1.4002 / 1.2457 | 1.4089 / 1.1231 |
| 45,000 | 1.3808 / 1.2401 | 1.3912 / 1.1420 |
| 50,000 | 1.3674 / 1.2649 | 1.3766 / 1.1345 |
| 55,000 | 1.3612 / 1.2673 | 1.3675 / 1.1295 |
| 60,000 | 1.3567 / 1.2551 | 1.3609 / 1.1244 |
| 65,000 | 1.3515 / 1.2600 | — |
| 70,000 | 1.3462 / 1.2503 | — |
| **71,387 (final)** | — | **1.3500 / 1.1276** |

The final 500-batch result is better than the 70k 50-batch sample on rank CE but
slightly worse on BC CE; the difference is within the expected sample/step
variation. Do not describe this as a monotone “improvement” without checking
the exact samples.

---

## 6. Tests and commands

```bash
export PYTHONPATH="$PWD/vendor/mortal/target/release:$PWD/vendor/mortal:$PWD"

# VNext only (79 passed)
python -m pytest -q tests/rf_test_vnext_*.py

# VNext + B0 + arena/bridge + anchor (149 passed, 6 skipped)
python -m pytest -q tests/rf_test_vnext_*.py tests/rf_test_b0*.py \
  tests/rf_test_m5d_arena.py tests/rf_test_mortal_anchor.py
```

Relevant tests:

- `tests/rf_test_vnext_train.py`: rank normalization, adapter batching,
  prefix-cap skip, BC/rank loss, all-censored behavior, checkpoint round-trip.
- `tests/rf_test_vnext_train_cli.py`: dry-run, output guards, val
  authentication, resume CLI.
- `tests/rf_test_vnext_contracts.py`, `..._model.py`, `..._runtime.py`,
  `..._grp.py`, `..._critic.py`, `..._rl.py`, `..._pr01.py`.
- `tests/rf_test_m5d_arena.py`: native bridge lifecycle, backend reset,
  illegal/fallback gates.

Known test gap: no end-to-end multi-kan (`KAN_SELECTION`) native test. The final
16-game smoke had `kan_selections=0`; single-candidate ankan/kakan appear in
earlier smoke logs, but that is not the same as a real multi-candidate path.

---

## 7. High-risk code to attack

| File | Why it matters |
|---|---|
| `riichiformer/vnext/data.py` | Event-stream batching, rank normalization, 4,096 prefix cap; a batching bug silently corrupts targets. |
| `riichiformer/vnext/train.py` | Loss masking, rank CE, checkpoint save/load, provenance. |
| `scripts/rf_train_vnext.py` | Resume `batches_seen`, shard-hash auth, val-purpose auth, output guards, wall cap. |
| `riichiformer/data/loader.py` | Cached decoded POV streams; verify direct-vs-cached decisions are identical and no game/POV leakage. |
| `riichiformer/eval/m5d_arena.py` | New `_ensure_backend_game` reset in `_react`; could change other engines’ behavior. |
| `configs/riichiformer/local_research_policy.yaml` | Local rights policy; confirm it is not a silent licence rewrite. |
| `scripts/rf_vnext_census.py` | Uses `reader._conn` and `PRAGMA temp_store=MEMORY`; verify read-only and no shard mutation. |
| `scripts/run_vnext_2009.sh` | Preflight 6,897-game check, Census, absolute venv path; verify fail-closed behavior. |

Cross-cutting changes outside `vnext/` are legitimate review targets: loader,
arena, anchor, B0, and the local transformer.

---

## 8. Known limitations / non-claims

- **2009 full-year only**, not the 2.51M-match corpus. At the final measured
  rate (~501 decisions/s), 2.51M one epoch is ~38 days compute; shard build at
  ~0.7 games/s is ~41 days. `NOT_RUN`.
- **One epoch**, no second epoch; no model-selection/Arena promotion.
- **No strength/dan/Mortal/Arena claim.** Native smoke is integration only.
- **Confirm pool untouched.**
- **No full-corpus Census.**
- **460 / 4,568,777 decisions (0.01%) skipped** above 4,096 tokens; +~0.01%
  replay to hit target.
- **External native build mismatch**: local libriichi build and patch manifest
  differ from the frozen M5D registry; the smoke uses a local smoke-only
  registry. It proves ABI/runtime integration, not frozen-Arena reproduction.
- Raw data and `runs/` are gitignored/local-only; the external reviewer can
  verify code, logs, manifest hashes, and checkpoint sidecar, but not replay the
  raw corpus.

---

## 9. Reproduction notes

Pipeline commands are in `docs/riichiformer/vnext/PR12_PIPELINE.md`. The 2009
launcher is `scripts/run_vnext_2009.sh`; it preflights `game_count == 6897` and
runs the Census before training.

The final full-year model was trained with:

```text
batch 64, rank_base 1, warmup 1000, cosine min-lr 0.1,
steps 71387, save/val every 5000, val-batches 50, max-hours 5,
max-prefix-tokens 4096, CUDA FP32, seed 20261122
```

Checkpoint files are 75 MB each; they are not in git. Raw shards are ~6.2 GB and
must not be uploaded to an external reviewer.

---

## 10. Evidence file map

- `docs/riichiformer/vnext/STATUS.md` — final status and caveats.
- `docs/riichiformer/vnext/PR12_2009_PREREG.md` — pre-registration + result.
- `docs/riichiformer/vnext/PR12_REAL_PILOT.md` — pilot and native smoke.
- `docs/riichiformer/vnext/PR12_PIPELINE.md` — data commands.
- `docs/riichiformer/vnext/TASK_BOARD.md` — task state.
- `docs/riichiformer/vnext/REVIEW.md` — independent review history; final
  full-year run was **not** externally reviewed.
- `docs/riichiformer/vnext/logs/VNEXT_CENSUS_2009.json` — full-year Census.
- `docs/riichiformer/vnext/logs/VNEXT_VALIDATE_2009_FINAL_500BATCH.log`.
- `docs/riichiformer/vnext/logs/VNEXT_NATIVE_2009_FINAL_16G.log`.
- `docs/riichiformer/vnext/logs/VNEXT_REAL_TRAIN_SUITE.log` (79 passed).
- `docs/riichiformer/vnext/logs/VNEXT_COMBINED_REAL.log` (149 passed).
- `docs/riichiformer/vnext/AUDIT_EVIDENCE.json` — machine-readable snapshot.

---

## 11. Required review output

Please return:

1. **P0/P1/P2 findings** with `file:line`, why it matters, and a concrete
   reproduction or test.
2. **Claim audit**: for each claim table row, say `supported`, `overstated`, or
   `unsupported`.
3. **Rights-gate opinion**: is `train_allowed=false` +
   `train_eligible_under_policy=true` an explicit local policy or a bypass?
4. **Leakage check**: is there any 2023-val → 2009-train overlap or target
   leakage in the adapter/cache?
5. **Resume/checkpoint check**: can a wrong-shard checkpoint be loaded or can
   replay silently duplicate data?
6. **Test-gap list**, prioritised.
7. **One-line verdict**: `AUDIT_PASS`, `AUDIT_PASS_WITH_P1`, or `AUDIT_BLOCK`.
