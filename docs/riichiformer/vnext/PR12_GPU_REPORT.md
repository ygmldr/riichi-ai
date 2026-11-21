# PR-12 Resource Report — Native + Synthetic GPU Validation

**Status**: `SYNTHETIC_ONLY`
**Updated**: 2026-11-22
**Branch**: `rf/vnext-core`
**User authorization**: GPU use granted on 2026-11-22; the rights gate remains
`BLOCKED_FOR_NEW_TRAINING`, so no restricted corpus was read.

This report closes the *tooling* gaps that do not require training data:

1. the pinned native `libriichi` build now exists in the VNext worktree;
2. a real native `OneVsThree` set ran with a VNext checkpoint;
3. a bounded CUDA profile measured throughput and VRAM;
4. synthetic BC/GRP/RL objectives ran on CUDA and the checkpoint round-tripped.

It does **not** close the data-rights gate, the 2.51M-match Census, a real
Train/Val epoch, a real Arena comparison, or any strength claim.

## 1. Native build and smoke

| Item | Value |
|---|---|
| submodule | `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2` |
| local `libriichi.so` SHA-256 | `171bd45a5859d89a5e06175785e4eb39004aa9a4f4d15543a1574c8e7c6875ff` |
| local patch manifest SHA-256 | `fb2d57f19b8ffb2be8de7785225ca96064ef0b25162c2aa46ec121529b1ae55f` |
| local smoke-registry hash | `b3752eb3d61d3618fbfad40bf1826f3157af1c0d489b442775bdcb06b7b78fc7` |
| games | 4 (one complete rotated set) |
| seed window | `[990000, 990001]` |
| seed key | `0x5eedf00d5eedf00d` |
| decisions | 579 |
| illegal / fallback | 0 / 0 |
| wall time | 21.009 s |
| checkpoint SHA-256 | `34d0508da080dbd101aeadc5710a96413fe331f917e7b8467436a0ccd5fe41ea` |

The checkpoint is a **randomly initialised** small VNext model
(`d_model=64`, one layer, `model_config_hash=64a28b1c...`), saved through the
real `save_vnext_checkpoint` ABI and loaded through `load_vnext_checkpoint`.
The four native logs contain an `ankan` and a `kakan`, but each had one
candidate: `kan_selections=0`.  The second `KAN_SELECTION` call therefore was
**not** exercised by this run; the multi-kan end-to-end path remains `NOT_RUN`.

The frozen `configs/riichiformer/m5d_baseline.yaml` was not rewritten.  A local
smoke-only registry was generated in the scratch directory because this
worktree's patch manifest/build differs from that frozen registry.  This is also
why the inherited `NativeSmokeTests` are skipped here: the frozen registry check
is an environment gate, not a VNext failure.  Explicit status:
`ANCHOR_GATE: ENV_MISMATCH_REQUIRES_REBASE`; this report does not label the
smoke as reproducing the frozen M5D anchor.

### Local native setup (not committed)

The submodule is pinned but its patched working tree is intentionally not a
commit of this branch.  A reproducible local build requires, in this order:

```bash
git submodule update --init vendor/mortal
cd vendor/mortal
for p in 0001-train-grp-dataloader-num-workers-0 0002-train-remove-per-step-cuda-sync \
         0003-gamma-discount-labels 0004-agari-guard-consistency \
         0005-train-holdout-val-loss 0006-event-stream-tokens \
         0007-model-event-branch; do
  git apply "/path/to/riichi-ai/patches/$p.patch"
done
git apply /path/to/riichi-ai/patches/0010-arena-event-tokens.patch
git apply /path/to/riichi-ai/patches/0008-rl-policy-kl.patch
git apply /path/to/riichi-ai/patches/0012-1v3-seed-start.patch
git apply /path/to/riichi-ai/patches/0013-online-tail-event-pad.patch
cargo build -p libriichi --lib --release
```

`0009` targets the repository-root `configs/config.toml`, not the vendor tree,
and was intentionally not applied.  The local build/patch hashes differ from the
frozen registry; that is the documented anchor-rebase gate, not a silent pass.

Evidence: `docs/riichiformer/vnext/logs/VNEXT_NATIVE_SMOKE.log`,
`runs/vnext_native_smoke/smoke_report.json`.

## 2. GPU profile

Synthetic contract-valid batch, 128 decisions × 256 tokens, 20–30 measured
iterations after 3 warm-ups, FP32, RTX 5070 Ti 16 GB.  Three after-fix runs
were made; the table uses medians.

| Metric | Before sync fix | After sync fix | CPU reference (same shape) |
|---|---:|---:|---:|
| forward decisions/s | 323.8 | **12,433.7** (median of 3) | 1,638.0 |
| forward tokens/s | 82,894 | **3,183,023** (median) | 419,328 |
| train decisions/s | 274.0 | **1,635.6** (median of 3) | 413.4 |
| train tokens/s | 70,135 | **418,713** (median) | 105,825 |
| inference peak VRAM | 144.6 MiB | **144.6 MiB** | 987.7 MiB RSS |
| train peak VRAM | 331.9 MiB | **331.9 MiB** | 987.7 MiB RSS |

The first GPU run was dominated by host synchronisation: a profiler trace showed
`aten::item` / `_local_scalar_dense` as the top cost, because
`VNextModelBatch.__post_init__` re-ran on every forward.  `VNextPolicy.forward`
now fails closed on a device mismatch instead of re-validating or silently
copying.  This was a measured bug/perf fix, not a new architecture.

These are microbench numbers, not an epoch projection.  They show the GPU path
is now faster than the CPU reference at this shape, with a small VRAM footprint.

Evidence: `logs/VNEXT_GPU_PROFILE.json` (before),
`logs/VNEXT_GPU_PROFILE_FINAL*.json` (after; three runs show run-to-run variance),
`logs/VNEXT_CPU_PROFILE_128x256.json`.  The pre-fix numbers remain in the report table only; the after-fix log is the
cited one.

## 3. Synthetic CUDA BC / GRP / RL smoke

| Check | Result |
|---|---|
| BC steps | 100 (64 decisions × 192 tokens) |
| BC finite | yes |
| BC steps/s | 7.23 |
| checkpoint roundtrip state SHA | `a2e56a52...` unchanged |
| GRP steps | 50; finite; loss 3.1847 → 2.2953 |
| behavior μ sampling | 32 actions, all legal-mask respected |
| AWR / PPO / PBRS / delta / GAE | finite; shapes valid |
| peak VRAM | 233.4 MiB |
| strength claim | **NONE** |

The BC target was a single fixed synthetic action and is not a convergence test.
The GRP loss decrease is on random rank labels and is not a calibration result.

Evidence: `logs/VNEXT_GPU_SMOKE_AFTER_SYNC_FIX.log`,
`runs/vnext_gpu_smoke/smoke_report.json`.

## 4. Remaining blockers

* `SOURCE_RIGHTS_GATE.json`: 11 rows, 0 `ALLOWED`,
  `BLOCKED_FOR_NEW_TRAINING`.
* Full 2.51M-match Census and a real Train/Val BC epoch: `NOT_RUN`.
* Full B0 epoch, broad GRP fit, AWR/PPO league, Trend/Select/Confirm:
  `NOT_RUN`.
* Real paired Arena / strength claim: `NOT_RUN`.
* The native smoke has only 4 games and a random policy; it cannot support a
  win-rate, dan, Tenhou or Mortal comparison.

## 5. Next gate

A real BC pilot requires a rights-cleared, audited match-level Train subset and
a fresh preregistration.  GPU availability alone is not a data licence.  Until
that gate opens, the only legitimate next work is contract/test hardening,
rights review, or more synthetic profiles.
