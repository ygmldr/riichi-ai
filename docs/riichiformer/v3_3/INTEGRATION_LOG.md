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
