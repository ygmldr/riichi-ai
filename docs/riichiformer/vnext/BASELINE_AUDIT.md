# VNext Baseline Audit (PR-00)

**Date**: 2026-11-22 (Asia/Shanghai)  
**Lead worktree**: `/home/ygsldr/riichi-ai/worktrees/vnext-core`  
**Branch**: `rf/vnext-core`  
**Base**: `rf3.3-w2` = `8a915d1df89ced8f322e5f2e6e664b3a0e1fcf85`

## 1. Actual local state (no reset, no checkout of the pack's old SHA)

The repository root was **not** the V3.3 tip:

| Location | HEAD | Branch | Meaning |
|---|---|---|---|
| `/home/ygsldr/riichi-ai` | `c28c457e16c4446029e8204e6fb13eb259bb3ee8` | `rf/v3` | older V3.1/G3 history; not the V3.3 baseline |
| `/home/ygsldr/riichi-ai/worktrees/v33-integration` | `8a915d1df89ced8f322e5f2e6e664b3a0e1fcf85` | `rf/v3.3-integration` | actual V3.3 Wave 2 tree |
| `/home/ygsldr/riichi-ai/worktrees/vnext-core` | `8a915d1...` | `rf/vnext-core` | clean new worktree cut from the frozen tag |

`rf3.3-w2` is an annotated tag object `cf2717b...` whose peeled commit is
`8a915d1...`.  The pack's reported `8a915d1` matches the integration tree, not
the root's current `rf/v3` branch.  I did not reset, rebase, or overwrite either
tree.

The integration worktree has one dirty path: `vendor/mortal` (the pinned
submodule with the locally applied patches).  The new `vnext-core` worktree was
created with `git worktree add -b rf/vnext-core ... rf3.3-w2`; it did not inherit
that dirty submodule.

## 2. Frozen action/schema identity

Collected from the real tree, not from the ZIP text:

| Item | Value |
|---|---|
| `ACTION_SPACE` | **46** |
| `action_map_hash` | `6cf1efc5ba99533ec4844f38184ef51a7941174164f9588f21d33ca70a873a2e` |
| `schema_version` | `rf-schema-v3/2` |
| `schema_hash` | `12503d36e2571449d4e4bdd4c8780e00f285d3a2ec81f9e26af184d24c8ef43c` |
| `normalizer_hash` | `37df861b08b3958c3030c53dcbfa27762892c41332b383060caf46788f9ad444` |
| action entries | 46 (0..45, unique) |
| phases | `SELF_TURN`, `CALL_RESPONSE`, `KAN_SELECTION` |

The full 46-entry table, phase legality sets and lock hash are in
[`ACTION_MAP_LOCK.json`](ACTION_MAP_LOCK.json).

## 3. Native / checkpoint facts

| Item | Value |
|---|---|
| pinned `vendor/mortal` commit | `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2` |
| `libriichi.so` build SHA-256 | `582e36734b10c3d3b1230a834eacc494f94e40f6d642c712eadc23f6cabfaf4e` |
| M5D-R0 checkpoint | `/home/ygsldr/riichi-ai/checkpoints/mortal.pth` |
| M5D-R0 SHA-256 | `0a44613cb9c7edae1057cf6a2c160c3971c78130e18e07f28bb43e8f62abe532` |
| checkpoint size | 130,781,915 bytes |
| anchor architecture | `resnet{conv_channels=192,num_blocks=40}`, `use_event_branch=false`, 396k steps |

The new `vnext-core` worktree has an **uninitialized** `vendor/mortal`
submodule, so native tests that hard-code
`worktrees/vnext-core/vendor/mortal/target/release/libriichi.so` cannot run
there.  I used the built extension from the verified integration worktree for
the synthetic/native-import smoke tests, and left a full native run for a
worktree where the submodule is initialized.  This is an environment gap, not a
code PASS.

## 4. CPU baseline execution

Command tried in the new worktree:

```bash
CUDA_VISIBLE_DEVICES="" \
PYTHONPATH="$PWD:/home/ygsldr/riichi-ai/worktrees/v33-integration/vendor/mortal/target/release" \
/home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q tests/rf_test_*.py
```

Result: `31 failed, 799 passed, 6 skipped, 11 errors` in 233.65 s, exit 1.
The failures are concentrated in tests that require the missing local
`vendor/mortal` build/checkpoint paths, not in the new VNext modules.  Full log:
`logs/vnext/baseline_v33_cpu_rf.log`.

The previously recorded V3.3 evidence on the **real** integration worktree is
`841 passed, 6 skipped, 319 subtests` in 271.95 s
(`docs/riichiformer/v3_3/logs/V3_3_W2_full_suite_final.log`), produced with CUDA
disabled.  This audit treats that as the inherited V3.3 baseline and does not
claim the new worktree's failed collection as a new PASS or a new regression.

A fresh run on the integration worktree completed after this audit:

```text
841 passed, 6 skipped, 1 warning, 319 subtests passed in 332.29s
```

Log: `logs/V3_3_INTEGRATION_BASELINE.log` (archived from the fresh run; it does
not overwrite the historical V3.3 log).  This confirms the inherited V3.3
baseline is green in its real worktree with the native extension available.

## 5. Rights gate

`configs/riichiformer/source_rights.csv` has 11 rows:

* status: 8 `RESTRICTED`, 3 `UNKNOWN`, **0 `ALLOWED`**;
* `train_allowed`: 6 `RESTRICTED`, 5 `UNKNOWN`, **0 `ALLOWED`**;
* `redistribute_allowed`: 8 `RESTRICTED`, 3 `UNKNOWN`, **0 `ALLOWED`**.

Gate decision: **`BLOCKED_FOR_NEW_TRAINING`**.  Local private-research policy is
not a licence.  See [`SOURCE_RIGHTS_GATE.json`](SOURCE_RIGHTS_GATE.json).

## 6. Pack claims rejected or corrected

* `8a915d1` is **not** the root branch HEAD; it is the integration worktree and
  the peeled `rf3.3-w2` tag target.
* The pack's V3.3 841-test count is an inherited report; VNext has no completed
  GPU/Arena/strength result.
* The ZIP contains no local checkpoint, shard, or dataset; none was downloaded.
* `00e6576` is an ancestor of `e760226`, not the current tip.

## 7. PR-00 conclusion

`PASS` for read-only source/action/rights/hardware inventory, with the explicit
caveat that the fresh new-worktree CPU suite is `ENV_BLOCKED` on the
uninitialized submodule and is **not** reported as a code failure or PASS.
