# A1 VERDICT — river (discard-pile) ORDER is used; recency weighting is not load-bearing

- Date: 2026-09-17 (runs 17:38–19:46 CST, serialized; PREREG sha256
  `71f91196661ecef5c7f29240da044bc7d34603721fab9c98d6816c76ccaadf88`, written before the runs)
- Plan: `.omo/plans/fork-roadmap-steps-1-10.md` todo 12; script `scripts/exp_a1_shuffle.py`;
  analysis `scripts/exp_a1_delta.py` (imports `scripts/eval_report.py` unchanged)
- Protocol: challenger `runs/exp/PINNED/mortal.pth.at-175200` vs champion
  `checkpoints/mortal.pth.at-148800`; key `0x5dfaa4cef265cd7`; seed window `[30000, 33000)`,
  3000 sets x 4 = 12000 games per arm; inference-time intervention on the challenger only.

## 1. Pre-registered decision applied

> **any arm whose Δavg-rank 95% CI crosses 0 ⇒ that arm shows no effect**

| comparison | Δavg_rank | 95% CI | decision |
|---|---|---|---|
| `shuffle − original` | **+0.045833** | **[+0.024000, +0.068917]** | **EFFECT** (CI excludes 0) |
| `recency_zero − original` | −0.014583 | [−0.032002, +0.003671] | **NO detectable effect** (CI crosses 0) |

Δ = arm − original, computed per same seed set (paired), set-level bootstrap over the 3000
paired differences, B=2000, rng seed 0. Positive Δavg_rank = challenger worse.

- **`shuffle`: the model uses discard ORDER.** Permuting the item slots of every river window
  while preserving each block's exact slot multiset (None/empty slots included) makes the
  pinned challenger significantly worse: +0.046 rank, ~1.9x the 12k MDD (0.0239). Secondary
  Δpt = −4.095, CI [−5.910, −2.434] (also excludes 0). Since `kawa_overview_*`
  (order-insensitive tile-set summaries) were left intact, order information in the window
  blocks is load-bearing *in addition to* the tile-set overview.
- **`recency_zero`: no rank effect at 12k.** Zeroing the 4 recency plane groups moves avg_rank
  by −0.015 (CI crosses 0; |point| < 0.0239 MDD). Secondary Δpt = +1.444, CI [+0.064, +2.749]
  is the only non-null secondary signal and points *in the challenger's favour*; it is below
  the 1.86 unpaired MDD, is not part of the pre-registered rule, and is reported as a weak
  signal only. Recency weighting is not load-bearing for finishing rank at this resolution.

## 2. Per-arm reports (`report.csv`, `scripts/eval_report.py`)

| arm | dist(1/2/3/4) | avg_rank | avg_rank 95% CI | pt | pt 95% CI | wall |
|---|---|---|---|---|---|---|
| `original` | `[2975 2977 3058 2990]` | 2.505250 | [2.487917, 2.521002] | −0.161250 | [−1.436438, +1.132594] | 35m00s |
| `shuffle` | `[2869 2910 2960 3261]` | 2.551083 | [2.534165, 2.569583] | −4.256250 | [−5.685188, −2.928750] | 43m55s |
| `recency_zero` | `[2977 3043 3095 2885]` | 2.490667 | [2.473415, 2.508252] | +1.282500 | [−0.041344, +2.617594] | 37m03s |

All three: `sets=3000 games=12000 incomplete_sets=0 ignored_files=0`, 12000 `*.json.gz` each.

Determinism cross-check: the `original` arm reproduces `runs/variance/pool_3/games/` **exactly**
(same seed window/key), confirming the wrapper's passthrough is faithful and that the arena is
deterministic; `original` is therefore pool_3 semantics, not an independent rerun.

## 3. Self-check evidence (asserted in-script, every patched observation)

- `shuffle`: 1,790,089 obs patched, 1,758,409 changed (98.2%); 14,320,712 blocks checked
  (8/obs), 13,622,890 changed (95.1%), 697,822 uniform (4.9%, all-empty/river-empty windows
  correctly asserted UNCHANGED); binary fast path 14,320,712 / 0 fallback.
- Per block asserted: exact slot multiset (item-vector multiset), per-element value multiset
  and nonzero counts preserved; non-uniform blocks asserted to differ; blast radius confined to
  the window blocks; recency planes asserted all-zero.
- `recency_zero`: 1,817,565 obs patched, 1,786,102 (98.3%) had a nonzero recency plane zeroed;
  blast radius confined to the recency channels.
- `--selftest` (47 checks, incl. negative control that a corrupted multiset is caught) passed
  before any arena run.
- Frozen corpus: `runs/1v3/*.json.gz == 2004` before/after every arm; no writes to
  `vendor/**`, `checkpoints/**`, `datasets/**`, `runs/variance/**`.

## 4. Caveats

- Patching overhead is real but bounded: shuffle +535 s (43m55s vs 35m00s), recency_zero +123 s.
- Paired common-random-numbers design makes the Δ CIs tighter than the unpaired 12k MDD; the
  MDD is quoted only as the resolution scale (effects <0.0239 rank are not individually
  resolvable).
- The `recency_zero` Δpt CI lower bound (+0.064) is extremely close to 0; treat as unresolved
  rather than as an improvement.
- Interference targets only inference; no weights were trained, fine-tuned or altered.

## 5. Artifacts

- `runs/exp/A1/PREREG.md` (pre-run, sha256 above), `runs/exp/A1/VERDICT.md` (this file)
- `runs/exp/A1/delta_report.json`, `runs/exp/A1/{original,shuffle,recency_zero}_run.json`
- `runs/exp/A1/games_{original,shuffle,recency_zero}/report.csv` (+ 12000 game logs each, not
  committed — `runs/` is gitignored)
- `runs/exp/A1/logs/{original,shuffle,recency_zero}.log` (nohup stdout/stderr)
- Layout: `runs/analysis/obs_layout.json` (sha256 in each `*_run.json`); offsets read from it.
