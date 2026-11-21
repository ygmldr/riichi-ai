# RiichiFormer V3.3 Wave 2 — Lead review

Status: **MERGED, AWAITING INDEPENDENT REVIEWER**. This file records operations
actually run on this host. It is not a strength report and not a licence.

## Base and freeze

- Actual integration base: `e760226` (`docs: fix W2 full-suite log pointer`).
  `00e6576` is an **ancestor**, not the tip. Tag `rf3.2-w2` (annotated `c96b0eb`)
  points at `e760226`; tag `rf3.3-base` marks the same commit.
- W2 full suite at the base: **708 passed, 2 skipped, 281 subtests**, 182.23 s.
- Reviewer P1 (legacy checkpoint cross-path resume) fixed at `2601685`; the
  independent follow-up is recorded in `PR38_LOADER_INTEGRATION.md` §7 and
  `PR44_B0_PILOT.md` §4.1. No P0.
- M5D-R0 is the user-confirmed `checkpoints/mortal.pth`
  (`0a44613c…`, 396k steps, pure CNN `c192/b40`, `use_event_branch=false`).

## Lane commits (base `e760226`)

| Lane | PR | Commits | Merge | Tests / smoke | Self-verdict |
|---|---|---|---|---|---|
| A | PR-46A/B | `0e6cdbc`, `dd3f39a`, `9980de7` | `cf3da45` | 50+14+43 tests; 8 native sets/32 games CPU, 0 illegal/fallback; self-control ΔPT 0.0; weak 500-step RF pilot ΔPT −98.75 (not strength) | PASS_WITH_P1 |
| B | PR-45/47 | `a717db5`, `45a476e` | `53e500f` | 38 tests; **100 native games**, 73,841 decisions, shard `26c85d6b…`; CE 1.937 (random-init, weak-bot corpus); E2E two walls | PASS |
| C | PR-48 | `efd16f7`, `7b1359f`, `8ada…`, `7fa7523` | `ee17664` | 41 tests; CPU 2 seeds × 2 steps, 5,217,582 params, 2.65 s; full epoch NOT_RUN | PASS_WITH_P1 |
| D | PR-40R | `f147d4c`, `9434db9`, `2385533` | `4be7e7f` | cargo 35 passed; R29 4 native tests; targeted pytest 19 passed; build `582e367…` | PASS |

The registry YAML conflicted on merge because A's registry was built against the
0014-only `.so` and the integration tree uses D's 0015 build. The Lead regenerated
it from the merged sources and the rebuilt `.so`:
`libriichi_build_sha256 = 582e36734b10c3d3b1230a834eacc494f94e40f6d642c712eadc23f6cabfaf4e`,
`registry_hash = 42fd9a930532956cceee704e8e9fd1b4bb4b92b9455fd7c696b99ed4879f22ba`.
A's lane report lists the 0014-only `ceb9287c…` hash; the merged value is the
authoritative one. Registry tests pass after regeneration.

## Actual CPU smoke (no strength claim)

| Run | Sets | Games | Wall | Result |
|---|---:|---:|---:|---|
| Lead scratch, M5D self-control | 1 | 4 | 25.97 s | ranks `[1,1,1,1]`, ΔPT 0 |
| Lead scratch, 2k RF pilot | 4 | 16 | 58.37 s | ranks `[0,1,2,13]`, weak pilot |
| Lane A M5D self-control | 4 | 16 | 43.106 s | ΔPT `0.0 [0.0,0.0]`, hist `[4,4,4,4]` |
| Lane A weak RF pilot | 4 | 16 | 129.576 s | ΔPT `−98.75 [−131.25,−60.0]`, hist `[1,0,3,12]`, 2,298 decisions, 0 rejected |
| Lane B corpus generation | — | 100 | 278.63 s | 73,804 engine decisions; shard 73,841 |
| Lane B full diagnostic | — | 100 | 550.64 s | CE 1.937, top1 0.297, top3 0.511, illegal/non-finite 0 |

The RF candidate is the **500-step preliminary B0 pilot**, not a B0 reference and
not evidence of strength. No arena result is a Tenhou dan/rank.

## Full suite

- Final suite (CUDA disabled) at the merged tree: **839 passed, 6 skipped, 319 subtests**, 326.62 s, exit 0 (`docs/riichiformer/v3/logs/V3_3_W2_full_suite.log`). The 6 skips are the CUDA-only/pre-existing skips; no GPU was used.
- Earlier full suite at an intermediate merged tree: **804 passed, 3 skipped,
  319 subtests**, 412.32 s. That run accidentally executed the pre-existing tiny
  `rf_test_bench.py::test_gpu_bf16_smoke` because CUDA was visible. It did not
  train or run an arena, and left no compute process; the final suite is rerun with
  `CUDA_VISIBLE_DEVICES=""` so the GPU test is skipped. This is disclosed as a
  process deviation, not hidden.

## Gate matrix

| Gate | Evidence | Status |
|---|---|---|
| G-V33-W2 | tag, base suite, reviewer P1 fix | **PASS** |
| G-V33-A0 | anchor identity, 4-seat rotation, self-control, hidden-tile tests, real native smoke | **PASS for smoke scale**; 4–20 sets only |
| G-V33-D0 | 100-game disjoint validation, read-only diagnostics, E2E | **PASS for data/pipeline**; four-riichi/triple-ron NOT observed |
| G-V33-B0 | protocol/selection/resume code + CPU smoke | **CODE_PASS; full train NOT_RUN** |
| G-V33-C0 | no confirm pool, no strength | **NOT_RUN** |

## Not run

- Full B0 epoch (6,191 batches × 2 seeds): **NOT_RUN** (no GPU authorization).
- Trend/Select/Confirm: **NOT_RUN**; no confirm pool created or consumed.
- Strength, dan, Tenhou rank, Mortal superiority: **NOT_RUN**.
- Four-riichi abort and triple-ron in the 100-game corpus: not observed
  (`NOT_RUN`, not PASS).
- Full `Board`-level R29 scripted game: **NOT_RUN**; the fixture is
  `PlayerState`-level (the state machine that owns `at_ippatsu`).

## PR-49+ recommendations

1. **PR-49 data quality/time**: re-run the no-weight vs quality-weight and
   time-forward axes at matched decisions, with game-level disjointness; report CE
   and arena ΔPT separately.
2. **PR-50 candidate head**: compare 46-action linear head with a
   candidate-conditioned head under identical data/steps; strict legality parity.
3. **PR-51 risk/belief**: only with a mask/hidden-tile counterfactual proving zero
   leakage; train-only auxiliary labels, never forward inputs.
4. **PR-52 rank/PT auxiliary head**: expected placement/PT auxiliary loss; select by
   arena ΔPT, not loss.
5. **PR-53 memory**: cross-kyoku summary with a same-parameter no-memory negative
   control.
6. **PR-54 distillation**: only after the M5D anchor and a real candidate are
   stable; keep teacher data in a separate provenance stream.
7. **PR-55 GPU Trend**: only after a new select pool is registered and the user
   approves a bounded budget. Do not reuse the smoke windows
   `[241000,241004)`/`[241004,241008)` as confirm.

`STRENGTH_NOT_RUN` remains true. The external rights facts are unchanged (11 rows,
0 ALLOWED); `operator_private_research` is not a licence.
