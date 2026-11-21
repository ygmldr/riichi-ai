# V3.3 integration log

Append-only. Times are Asia/Shanghai. Each entry records commands actually run.

## 2026-11-21 — base verification

- `git rev-parse HEAD` on the user-reported tree: `c28c457` (old `rf/v3` checkout);
  `git rev-parse rf/v32-integration` = `e760226`; `git show 00e6576` confirms it is an
  ancestor of `e760226`, not the tip.
- `git tag --list 'rf3.2*'` → `rf3.2-w1` (`5242e5e`), `rf3.2-w2` (`e760226`).
- W2 full suite at `e760226`, patched vendor + `.so`:
  `708 passed, 2 skipped, 281 subtests passed in 182.23s`, exit 0.
- GPU check via `/usr/lib/wsl/lib/nvidia-smi`: 2,848 MiB baseline, 0 compute apps.
- Created five worktrees from `e760226`: `v33-integration`, `v33-a`, `v33-b`, `v33-c`,
  `v33-d`; copied the patched vendor source and the `v32-rules` `libriichi.so`.
- M5D-R0 confirmed by the user as `checkpoints/mortal.pth`
  (`0a44613c…`, 396k, pure CNN).

## 2026-11-21 — V3.3 Wave 2 merges and tests

- Merged `rf/pr40r-r29` (`2385533`) as `4be7e7f`; applied `patches/0015-rule-r29-ippatsu-fixture.patch`
  to the integration vendor and copied the rebuilt `libriichi.so`
  (`582e36734b10c3d3b1230a834eacc494f94e40f6d642c712eadc23f6cabfaf4e`).
  Targeted rules tests: 18 passed, 1 skipped, 25 subtests.
- Merged `rf/pr48-b0` (`7fa7523`) as `ee17664`; B0 tests 41 passed.
- Merged `rf/pr45-47-validation` (`45a476e`) as `53e500f`; validation tests 38 passed.
- Merged `rf/pr46-m5d` (`9980de7`) as `cf3da45`; A tests 43 passed (non-native).
- Rebuilt the M5D registry after merge (0015 `.so` + merged driver files):
  `libriichi_build_sha256=582e367…`, `registry_hash=42fd9a93…`; A naturally had
  the 0014-only `ceb9287c…` hash in its lane tree.
- Fixed a real PR-48 integration defect found by the Lead:
  `--emit-prereg-template` without `--device` failed, and an omitted `--max-steps`
  hit a None comparison. Commit `e81ed1f`: default template device to `cpu`, require
  a bounded `--max-steps`, add a regression test. `tests/rf_test_b0_cli.py` 14 passed.
- Real PR-45/48 integration dry-run (CPU, no model/optimizer/checkpoint):
  template with `/tmp/b0_500.rfshard` + `/tmp/v33_b_smoke/val100.rfshard` →
  `parameters=5217582`, seeds `[20261121,20261122]`, validation_steps=25,
  candidate_steps=7; `B0_DRY_RUN_OK` with `training_path=multi_decisions`, device `cpu`.
- Full suite (CUDA disabled) at `e81ed1f`: **TBD**; the earlier merged-tree run
  (before the CLI fix) was 839 passed, 6 skipped, 319 subtests, 326.62 s.
- The earlier 708-test base suite was 708 passed, 2 skipped, 281 subtests, 182.23 s.
