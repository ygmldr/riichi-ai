# PR-47 — native self-play → POV → shard → loader → train → checkpoint roundtrip

Task: RiichiFormer V3.3 Wave 2, Dev-B (PR-47 end-to-end).
Worktree: `/home/ygsldr/riichi-ai/worktrees/v33-b` (`rf/pr45-47-validation`).
Base SHA: `e76022600018e9adc6536ccbb904db8c9dc498b3`.
Status: **executed on CPU with native games**; no GPU, no strength claim.

## 1. Pipeline stages

| Stage | Implementation | Evidence |
|---|---|---|
| Native self-play game | `riichiformer.selfplay.rollout` (`libriichi.arena.OneVsThree.py_vs_py`), weak local bots, full-info gzip engine log | 100-game corpus in `PR45_VALIDATION.md` |
| Four POV projections | `data.pov.POVProjector` via `data.shard.make_game_record`; opponent concealed tiles become `UNKNOWN` | visibility audit: 400 streams, 481,504 events, 0 violations |
| Decision snapshots | `data.decision.build_decisions` with `mortal_arena_pinned` / `arena_selfplay` (R02/R25/R26 per decision, R29 shard-level) | 73,841 decisions in the 100-game shard |
| Shard v3 | `data.shard.ShardWriter`, `purpose=diagnostic`, `train_eligible_under_policy=False`, `split=validation` | shard SHA `26c85d6b…` (100-game), row hashes re-verified |
| Multi-decision loader | `ShardReader.iter_multi_decision_batches` through `ValidationReader` (read-only) | `optimizer_steps=0` |
| Six-layer CPU train | `Policy` MVP-S **5,217,582 params**, `training_step_multi`, AdamW | 2 optimizer steps, finite losses |
| Checkpoint | `train.checkpoint.save_checkpoint` (schema/action/normalizer hashes, model-config hash, provenance, shard SHA in `extra`) | `.pt` + `.json` manifest |
| Reload / inference | `load_checkpoint` into a fresh model, `forward_multi`, compare logits | max abs difference **0.0** |
| Duplicate arena | PR-46 (Dev-A) | **NOT_RUN** in this lane; PR-47 ends at reload/inference |

`data/shard.py`, `data/loader.py` and `selfplay/rollout.py` were not modified.

## 2. Executed E2E smoke

Two native 4-game walls selected from the real 100-game corpus to cover the
event classes the pipeline must handle (a coverage selection for a smoke test,
not a strength subset). Both use the corpus `seed_key = 0x73aef4318e7bc77d`,
the real engine logs, the frozen MVP-S and 2 optimizer steps on CPU.

| Wall | Scenario focus | Shard SHA-256 | Games | Decisions | 2-step losses | Checkpoint SHA-256 | Reload Δlogit | Wall time |
|---|---|---|---:|---:|---|---|---|---:|
| `300002` | kakan, daiminkan, multi-ron | `78c4b836d7f7b3b4bd4b433346509c99e4e319412283467dff4fc64761b14e01` | 4 | 2,611 | 1.747654 / 2.584580 | `241298ebd68a6f3a571dce2c6e9ca08a2fd042c3a937b3b5ac7c7e2902189f06` | 0.0 | 9.65 s |
| `300006` | ankan, end-of-game kyotaku | `b9fd0710b15ea95e34e9d519c63becac4b58355df562d22b6a7ee3881ab3b6fd` | 4 | 3,169 | 0.877124 / 2.237095 | `b9a8b675cd2f18aa0cee7c98461a7ff16ba88c58e53281c1c3d7b93fedd74449` | 0.0 | 12.34 s |

Combined wall time **23.40 s**, peak RSS **1,387,920 KiB (1.32 GiB)**, CPU
float32, no GPU. Each wall trained 2 steps × 2 decisions (4 supervised decisions;
31 and 33 joined tokens respectively), saved a checkpoint and reloaded it into a
fresh `Policy` with identical logits.

The pytest suite performs the same roundtrip automatically on its own 4-game
native fixture (`tests/rf_test_validation_e2e.py::test_pr47_roundtrip`):
**12 passed in 22.14 s**, including the roundtrip, reader, visibility and
negative-control tests.

All runs report: `gpu_used=false`, `parameter_count=5217582`,
`training_loader_refusal=ShardRightsError`, `reload_logit_max_abs=0.0`.

## 3. Assertions

| Assertion | Where | Result |
|---|---|---|
| Score conservation (`sum(scores)==100,000`) | builder, per game; `replay_scores(strict_protocol=False)` cross-check | 100/100 games |
| Ruleset/version provenance | metadata `ruleset_id=mortal_arena_pinned`, `source_kind=arena_selfplay`, `ruleset_deviations=R02,R25,R26,R29`, `projection_version=rf-pov-v3/2`, schema/action/normalizer hashes | asserted |
| Training loader cannot optimize the validation shard | `ShardReader(purpose="train")` | `ShardRightsError` |
| Reader performs no optimization | `evaluate()` under `no_grad`, asserts no parameter grad, `optimizer_steps=0` | asserted |
| `UNKNOWN` only on opponents' hidden events | `assert_visibility_contract` over every POV stream | 0 violations; unknown = 169,581 hidden + 10,668 padding |
| Future events cannot change an earlier decision | tamper a public `dahai` after decision 0's cutoff and re-encode | identical state features, mask, logits |
| Prefix tampering is refused | `before_event_idx` −1 + recomputed row hash | `ShardLoadError` |
| Label tampering is refused | injected `expert_action` with stale `target_sha256` | `IntegrityError` |
| Public-state tampering is refused | `kyotaku` +1 + recomputed row hash | `ShardLoadError` |
| Checkpoint reload is exact | reload into fresh model, forward same batch | logits max abs 0.0 |

## 4. Scenario coverage

Engine-generated classes observed in the 100-game corpus (see
`PR45_VALIDATION.md` §7): normal win 462, draw 483, riichi 862, chi 826, pon 490,
daiminkan 21, ankan 2, kakan 14, multi-ron 3, end-of-game kyotaku 25.
The E2E wall `300002` contains kakan=1, daiminkan=1, multi-ron=1, hora=19,
ryukyoku=17, reach=33, chi=31, pon=20; wall `300006` contains ankan=1,
end_kyotaku=1, hora=19, ryukyoku=21, reach=40, chi=28, pon=21 (counts from the
per-wall reports in `/tmp/v33_b_smoke/e2e_coverage/e2e_coverage.json`).

| Required class | Status |
|---|---|
| Normal win | PASS (E2E shard contains `hora` decisions; score deltas replayed) |
| Draw | PASS (`ryukyoku` in corpus/E2E wall) |
| Riichi | PASS (`reach`/`reach_accepted` in corpus) |
| Call | PASS (chi/pon in E2E shard; daiminkan in the kan wall) |
| Kan | PASS (kakan/daiminkan/ankan generated; `kan_selection` decisions present) |
| Four-riichi / abort | **NOT_RUN** — engine generated 0 in 100 games |
| Triple ron | **NOT_RUN** — engine generated 0 in 100 games (multi-ron=3 are double ron) |
| End-of-game kyotaku | PASS (25 occurrences; R25 proxy recorded) |

No result is fabricated for the two missing rare classes; they remain
`NOT_RUN` until a targeted fixture or a larger corpus produces them.

## 5. Reproduction

```bash
export PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release
cd /home/ygsldr/riichi-ai/worktrees/v33-b

# 4-game native roundtrip (corpus logs from PR-45)
.venv/bin/python - <<'PY'
import json
from riichiformer.policy.config import load_policy_config
from riichiformer.validation.corpus import generate_seed_key, run_validation_selfplay, ValidationWindow
from riichiformer.validation.shard_build import ValidationShardSpec
from riichiformer.validation.e2e import run_roundtrip
window = ValidationWindow(seed_start=300200, games=4, seed_key=generate_seed_key())
out = "/tmp/pr47-e2e"
run_validation_selfplay(out, window)
manifest = json.load(open(out + "/validation_corpus.json"))
spec = ValidationShardSpec(seed_start=window.seed_start, games=4,
                           seed_key=window.seed_key,
                           lineup=tuple(manifest["lineup"]),
                           generator_sha256=manifest["generator_sha256"])
policy = load_policy_config("configs/riichiformer/local_research_policy.example.yaml")
report = run_roundtrip(games_dir=out + "/games",
                       shard_path=out + "/e2e.rfshard",
                       workdir=out + "/work", policy=policy, spec=spec,
                       steps=2, batch_size=2)
print(json.dumps(report, indent=2, sort_keys=True))
PY

# tests
.venv/bin/python -m pytest -q tests/rf_test_validation_e2e.py
```

Logs: `/tmp/v33_b_smoke/e2e_coverage.log` and
`/tmp/v33_b_smoke/e2e_coverage/e2e_coverage.json` (recorded run).

## 6. P0 / P1 / P2

* **P0:** none.
* **P1:** none in the executed path.
* **P2:** (a) `make_game_record` projects with the default `ProjectionPolicy`
  (tenhou/human) while decisions are built with the arena policy; both use
  `reveal_opponent_ankan=False`, so the event streams are identical — a future
  policy change that starts revealing opponent kans must pass the same policy to
  both sides. (b) The E2E coverage walls are selected post hoc; the corpus itself
  is the full 100-game window. (c) R29 remains shard-level only (see PR-45 §3).
  (d) A single CPU run of each E2E wall was performed; repeated-run determinism
  of the full generator was not separately measured beyond the seed ledger and
  deterministic encoder tests.

## 7. NOT_RUN

* Four-riichi/abort and triple-ron scenarios: not produced by the engine in 100
  games.
* GPU/CUDA path, VRAM, throughput: not run.
* Duplicate arena (PR-46) and relative PT: not run.
* Strength/dan/rank/Mortal comparison: not run; 2 CE steps on a random init are
  not evidence of strength.

## 13. Post-merge P1 correction (Lead, 2026-11-21)

The independent reviewer found that the original `run_roundtrip` executed its two
optimizer steps on the diagnostic validation shard (`purpose=diagnostic`,
`train_allowed=false`). That is a data-isolation defect even though it used a fresh
random model and no result was selected. The merged implementation now **never
optimizes the validation shard**:

- validation shard: built `purpose=diagnostic`, opened only through
  `ValidationReader`, used for forward/inference and diagnostics;
- optimizer steps: a separate labelled synthetic **train-purpose** shard
  (`purpose=train`, `split=train`, `source_id=synthetic_allowed`) by default, or an
  explicitly supplied independent `train_shard` (e.g. a real B0 train shard);
- train/validation game-id intersection is refused; the diagnostic shard is still
  refused by `ShardReader(purpose="train")`;
- checkpoint provenance comes from the train shard, and `extra` records both the
  train and validation shard hashes.

The historical 2-step numbers in §? above were produced before this correction and
are retained as pipeline evidence only. `tests/rf_test_validation_e2e.py::test_pr47_roundtrip`
now exercises the corrected path; full suite after the fix is recorded in
`WAVE2_REVIEW.md`.
