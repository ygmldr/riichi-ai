# RiichiFormer V3.3 — Wave 2 status (Lead)

Updated: 2026-11-21 18:00 +0800. This file is Lead-owned and records only operations
actually run on this host. It is not a strength report.

## Base facts (verified, not assumed)

| Item | Value |
|---|---|
| User-reported HEAD | `00e6576` |
| Actual latest integration HEAD | **`e760226`** (`docs: fix W2 full-suite log pointer`) |
| Relationship | `00e6576` is an **ancestor** of `e760226`; `e760226` adds the PR-38 legacy-resume P1 fix (`2601685`), the 708-test count refresh, and the reviewer follow-up record |
| Tag | `rf3.2-w2` (annotated `c96b0eb`) points at `e760226`; `rf3.2-w1` points at `5242e5e` |
| W2 full suite at base | **708 passed, 2 skipped, 281 subtests** in 182.23 s, exit 0 (`docs/riichiformer/v3/logs/V3_3_W2_freeze_full_suite.log`) |
| Reviewer | PR-38 loader/trainer + PR-44 pilot: `PASS_WITH_P1`; P1 (legacy checkpoint cross-path resume) fixed in `2601685`; independent recheck recorded in `PR38_LOADER_INTEGRATION.md` §7 and `PR44_B0_PILOT.md` §4.1 |
| GPU | RTX 5070 Ti, **2,848 MiB baseline / 16,303 MiB, 0 compute processes**; no RF training/eval process. One intermediate full-suite run executed the pre-existing tiny `rf_test_bench.py::test_gpu_bf16_smoke` because CUDA was visible (disclosed in `WAVE2_REVIEW.md`); the final suite runs with `CUDA_VISIBLE_DEVICES=""`. |
| Vendor | pinned `vendor/mortal` `0cff2b5` + `patches/0014-rule-r02-r25-r26.patch`; each V3.3 worktree carries the patched source and a copied `libriichi.so` for CPU/static tests |
| W2 freeze | effective at `e760226`; no commit before it was overwritten and no remote code was pulled over it |

**Important:** the pack's statement “HEAD `00e6576`, Reviewer still running, 707 passed”
was written before the reviewer P1 fix and the final 708-test run. This Lead uses the
actual later HEAD `e760226` and the existing `rf3.2-w2` tag. It does **not** reset to
`00e6576` or merge the old remote `a27df099` tree.

## M5D-R0 (user-confirmed)

- Path: `/home/ygsldr/riichi-ai/checkpoints/mortal.pth`
- SHA-256: `0a44613cb9c7edae1057cf6a2c160c3971c78130e18e07f28bb43e8f62abe532`
- 396,000 steps, 10,818,641 state-dict elements, pure CNN `resnet{conv_channels=192,num_blocks=40}`,
  `[model].use_event_branch=false`; byte-identical to `runs/exp/PINNED/mortal.pth.at-396000`.
- User explicitly selected this path from four candidates on 2026-11-21; it is not a
  name-similarity guess. `best.pth` (220k) and the 148.8k/175.2k/328k/382k snapshots are
  **not** M5D-R0.
- Intended inference: deterministic argmax, native legal mask, CPU for smoke; GPU only
  after a separate user authorization.
- Permission: `local_research_only`; this is not a third-party licence and does not
  authorize publication, redistribution, or online Tenba/雀魂 play.

## Lanes and worktrees (all from `e760226`)

| Lane | Task | Worktree | Branch | Owned surface |
|---|---|---|---|---|
| A | PR-46A/B M5D anchor, native 1v3, relative PT | `worktrees/v33-a` | `rf/pr46-m5d` | `riichiformer/eval/mortal_anchor.py`, `relative_pt.py`, `m5d_arena.py`, `scripts/rf_eval_m5d.py`, PR-46 tests/docs |
| B | PR-45/47 validation, diagnostics, selfplay roundtrip | `worktrees/v33-b` | `rf/pr45-47-validation` | new `riichiformer/validation/**`, `scripts/rf_validation*`, PR-45/47 tests/docs |
| C | PR-48 B0 protocol, selection, resume | `worktrees/v33-c` | `rf/pr48-b0` | `riichiformer/train/**`, `scripts/rf_train_b0.py`, PR-48 tests/docs |
| D | PR-40R R29/0014 native rules | `worktrees/v33-d` | `rf/pr40r-r29` | new rules tests/patch/docs; rebuilds only its own vendor |
| Lead | integration, tags, evidence | `worktrees/v33-integration` | `rf/v3.3-integration` | V3.3 docs, merge commits, final suite |

No lane may edit another lane's files, `runs/1v3`, `runs/variance`, `checkpoints/`, or
the root `vendor/`. No lane may start a GPU job. Dev-A may run a **CPU** 4–20 set native
smoke; Dev-B may run a bounded CPU selfplay/validation batch; Dev-C is code/CPU tests
only; Dev-D is CPU Rust build/tests only.

## Current gates

| Gate | Status |
|---|---|
| `G-V33-W2` | **PASS** — reviewer P1 closed, tag `rf3.2-w2`, 708 tests at `e760226` |
| `G-V33-A0` | **PASS** — 100-game/25-set self-control ΔPT 0.0; RF bridge 2,298 decisions / 0 rejected |
| `G-V33-D0` | **PASS for data/pipeline** — 100 native games, 73,841 decisions, disjoint, diagnostic shard, CE 1.937, E2E two walls |
| `G-V33-B0` | **CODE_PASS / FULL_TRAIN NOT_RUN** — 43 B0 tests, 2 seeds × 2 CPU steps, validation-isolation gate; full 6,191-step epoch not run |
| `G-V33-C0` | **NOT_RUN** — no confirm pool, no strength claim |

Final tagged state: `rf3.3-w2`; independent reviewer: **PASS**, no P1; full B0/arena/strength NOT_RUN.

`STRENGTH_NOT_RUN` remains true for the project. No ΔPT, rank, dan, or Mortal-superiority
claim is made from loss or mock runs.

## Evaluation-pool discipline

The pack's historical windows (`[10000,13000)`, `[20000,23000)`, `[30000,33000)`,
`[50000,53000)`, `[70000,73000)`, `[120000,126000)`) are treated as **consumed**. New
smoke/select/confirm pools need a fresh `seed_key` and a written audit before use. The
B0 training sample (500 games, seed `20261121`) is **train**, never validation.

## Next actions

1. Collect four lane reports and independent reviewer findings.
2. Merge in order: PR-46 scoring contract → PR-45/47 validation reader → PR-48 trainer
   protocol → PR-40R native evidence.
3. Run the full RF suite, deterministic replay, pycompile, and seed-pool audit.
4. Run only the approved CPU smoke (4–20 sets) and record measured runtime/VRAM=0.
5. Ask the user for the next GPU budget; do not start B0/Trend/Confirm.
