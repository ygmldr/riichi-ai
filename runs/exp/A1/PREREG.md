# A1 PREREG — does the pinned challenger use river (discard-pile) ORDER?

- Date: 2026-09-17 (written BEFORE the three 12k data runs; a 1-seed engineering smoke of
  the wrapper - 4 games/arm, temp dir, no repo artifacts - ran 17:35-17:37 CST before this file)
- Plan: `.omo/plans/fork-roadmap-steps-1-10.md` todo 12
- Layout source of truth: `runs/analysis/obs_layout.json` (todo 11, verified; offsets are read
  from that file at runtime, never re-derived)
- Script: `scripts/exp_a1_shuffle.py`; analysis: `scripts/eval_report.py` + `scripts/exp_a1_delta.py`

## 1. Question

The v4 observation tensor encodes each river as ordered item windows (recent-6 `oldest_first`,
old-18 `newest_first`; stride == item_channels) plus recency planes. Does the pinned challenger
`runs/exp/PINNED/mortal.pth.at-175200` actually **read that order**, or is the river window
functionally a **bag of tiles**?

## 2. Arms (challenger only; champion never wrapped)

| arm | intervention |
|-----|--------------|
| `original` | observations pass through untouched (paired reference) |
| `shuffle` | in `react_batch`, BEFORE obs reach `brain`: every river WINDOW block has its item **slots** permuted by a fixed seed; the per-block slot multiset is preserved exactly (incl. all-zero None/empty slots as their own class) |
| `recency_zero` | the 4 recency plane groups are zeroed: self `[131]`, shimocha `[324,327)`, toimen `[519,522)`, kamicha `[714,717)` |

Permutation targets (window blocks, `(offset, items, item_channels)`):
`(35,6,4) (59,18,4) (132,6,8) (180,18,8) (327,6,8) (375,18,8) (522,6,8) (570,18,8)`.
`kawa_overview_*` blocks are order-insensitive tile-set summaries (`group: river_overview`) and are
**not** permuted.

**Fixed permutation seed: `10575597` (`0xA15EED`)**. Per observation the RNG is
`numpy.random.default_rng([0xA15EED, react_batch_call_index, obs_index_in_batch])`
(PCG64; one draw stream per observation, one `rng.permutation(items)` per block, in the block
order above). The seed is a constant in the script; there is no CLI override.

## 3. Protocol (paired)

- challenger `runs/exp/PINNED/mortal.pth.at-175200` (name `mortal`), champion
  `checkpoints/mortal.pth.at-148800` (name `old148800`), both `enable_rule_based_agari_guard=true`.
- `[1v3].games_per_iter=12000`, `iters=1`, `seed_start` = value in `configs/config.toml` at
  launch (**30000** at PREREG time), `seed_key = 0x5dfaa4cef265cd7`; seed window
  `[seed_start, seed_start+3000)`, 3000 sets x 4 splits = 12000 games per arm.
- All three arms use the **same seed window and key** (paired/common-random-numbers); the only
  difference between arms is the wrapper behaviour. Config difference per arm: `[1v3].log_dir`
  only, written atomically; backup `configs/config.toml.bak-12`.
- Log dirs (exact): `runs/exp/A1/games_original/`, `runs/exp/A1/games_shuffle/`,
  `runs/exp/A1/games_recency_zero/`. Each must contain exactly 12000 `*.json.gz` at completion.
- `runs/1v3/*.json.gz` frozen corpus invariant: exactly **2004** files before and after.

Commands (serialized; one arena at a time):

```bash
python3 scripts/exp_a1_shuffle.py --arm original      # log_dir -> runs/exp/A1/games_original/
python3 scripts/exp_a1_shuffle.py --arm shuffle       # log_dir -> runs/exp/A1/games_shuffle/
python3 scripts/exp_a1_shuffle.py --arm recency_zero  # log_dir -> runs/exp/A1/games_recency_zero/
```

## 4. Metrics and decision rule (pre-registered)

- Primary: challenger `avg_rank` (rank 1..4, lower is better) over the same 3000 sets.
- Secondary: `pt` (RANK_PTS 90/45/0/-135).
- Delta: `d_rank[set] = set_avg_rank(arm) - set_avg_rank(original)` (and analogously `d_pt`),
  CI = set-level **paired bootstrap of the differences**, B=2000, `numpy` rng seed 0 — the exact
  `eval_report.py` procedure, computed by `scripts/exp_a1_delta.py`.
- **Decision rule: any arm whose Δavg-rank 95% CI crosses 0 ⇒ that arm shows no effect.**
  (Same reading for Δpt as secondary.) An arm whose CI excludes 0 is an effect; direction is read
  from the point estimate.
- Resolution: at 12k games the 80%-power MDD is **Δavg_rank ≈ 0.0239 / Δpt ≈ 1.86**
  (`runs/variance/README.md`). Effects below that scale cannot be resolved by this design.

## 5. In-script self-checks (asserted on every patched observation)

- per permuted block: exact item-slot multiset preserved (bit-packed lexicographic compare;
  `np.unique(axis=0)` fallback for non-binary data), plus per-element value multiset / nonzero
  counts; None/empty slots participate as their own class;
- a non-uniform block is asserted to DIFFER after the permutation (proves it applied); uniform
  (all-empty) windows are asserted UNCHANGED; `blocks_changed > 0` and `obs_changed > 0` at end;
- blast radius: differing channels are contained in the window blocks (`shuffle`) or recency
  planes (`recency_zero`); recency planes asserted all-zero after zeroing;
- negative control in `--selftest`: a deliberately corrupted multisets is caught.

## 6. Expectations / interpretation

- If the model treats the river as a bag of tiles, `shuffle` shows no effect (CI crosses 0).
- `recency_zero` degrades the challenger only if recency weighting is load-bearing.
- A null result is a valid, expected deliverable: it is reported as "no effect at 12k resolution",
  not as a failure. Only a crash / missing artifact counts as failure (record stderr, rerun arm).

## 7. Amendment log

- 2026-09-17T17:38+08:00: decimal of the registered seed corrected to 10575597 (=0xA15EED; the
  earlier 10559213 annotation was an arithmetic slip, no run used the other value).
- 2026-09-17T17:38+08:00: header wording amended to state the smoke-run ordering (see Date).
  No data run had started; `seed_start`=30000, key 0x5dfaa4cef265cd7 confirmed.
