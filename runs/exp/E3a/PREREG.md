# E3a Pre-Registration — CNN + residual event-Transformer fusion arm

Plan: `.omo/plans/fork-roadmap-steps-1-10.md` todo 19 · Roadmap `docs/FORK_ROADMAP.md` step 8.
Status: **written before training and before the arm's evaluation.** All numbers in §3/§4 are
pre-run *measurements of the model definition* (todo 19a), not training or evaluation results.

Repo HEAD at pre-registration: `8ec93a3` (todo 18's commit); vendor tree = pin `0cff2b5` +
patches 0002/0003/0004/0005/0006/0010/0012 + **0007** (this todo).

---

## 1. Arm definition (E3a, the treatment arm)

`mortal/model.py` (patch 0007):

- **Event branch** `EventBranch`: 4 `nn.TransformerEncoderLayer` blocks, `d_model=128`,
  4 heads, FFN `dim_feedforward=512` (= 4·d), `dropout=0.0`, **causal** self-attention
  (`src_mask` = strictly-upper-triangular bool; verified position-wise in the self-check).
  Input = the patch-0006/0010 token stream `(B, T, 5)` int16, embed per column
  (type 20 / tile 35 / actor 5 / turn 385 / flags 8 vocab) + learned positional embedding
  (≤384), pooled at the last real token (`lengths`). Padding needs no mask: attention is
  causal and pooling reads the last real token, asserted in `scripts/exp_e3a_selfcheck.py`.
- **Fusion**: residual, zero-initialised —
  `phi_out = phi + W_zero(h_event)`, `W_zero = nn.Linear(128, 1024, bias=False)` with
  `nn.init.zeros_(weight)`. **Not** `concat → Linear(1152→1024)`. A freshly built branch is
  therefore exactly the identity: `max|phi_out − phi_base| = 0.000e+00` for both all-zero and
  real token streams (threshold 1e-5; `runs/exp/E3a/selfcheck.json`).
- **Toggle**: `[model] use_event_branch` (default **false**). The branch is instantiated
  **only** when true; otherwise `Brain.event_branch is None` and `Brain.forward` is bit-for-bit
  the old CNN path (`events=None` stays inert).
- **Toggle source of truth (plan decision #10)**: load points read the toggle from the
  **checkpoint's own saved config** (`cfg.get('model', {}).get('use_event_branch', False)` via
  `brain_from_config(cfg)`, `cfg = state['config']`); the live config only decides how
  *training* builds the model. If a checkpoint carries `event_branch.*` weights while the
  constructed `Brain` resolves to false, `load_brain_state_dict` **raises RuntimeError** —
  `strict=False` is never allowed to silently drop them (that would degrade E3a into E2 and
  produce a false "no difference" verdict).

## 2. Load points (all updated to the guarded `strict=False` helper)

`mortal/train.py:116`, `mortal/one_vs_three.py:39,61`, `mortal/player.py:26,88`,
`mortal/mortal.py:45`, `mortal/client.py:49`. Verified individually by
`scripts/exp_e3a_verify_load.py` in four scenarios (evidence:
`.omo/start-work/evidence/T19.txt`):

- `e3a` (E3a checkpoint everywhere): 8/8 loads `missing=[]`, `unexpected=[]`,
  `branch_present=True`, branch `|W|` sum > 0.
- `old` (at-175200 CNN checkpoint): 8/8 loads `missing=[]`, `unexpected=[]`,
  `branch_present=False` (toggle resolves false), 0 branch params.
- `guard` / `guard2` (checkpoint has branch weights, config resolves false): every load point
  raises `RuntimeError: ... refusing to drop them silently ...`.

## 3. Parameter counts (measured `sum(p.numel())`, version 4)

| object | numel |
|---|---|
| baseline CNN (c=192, b=40) | **10,787,456** |
| E3a added by the event branch | **1,031,296** (9.56 % of the baseline) |
| **E3a total** | **11,818,752** |

Analytic estimate in the plan was ≈0.93 M for the added branch; the measured 1.031 M includes
the 5 token embeddings + positional embedding (107,136 params) that the estimate omitted.

**Input for todo 17 (E2)**: pick `conv_channels` such that
`|total_E2 − 11,818,752| / 11,818,752 ≤ 5 %` (analytic estimate ≈ 201). E2 must use the same
**20,000-step** budget (§5), `[model] use_event_branch = false` explicitly, and
`enable_compile = false`.

## 4. Model-definition self-checks (pre-run, todo 19a acceptance)

`scripts/exp_e3a_selfcheck.py` → `runs/exp/E3a/selfcheck.json`, all PASS:

- zero-init identity < 1e-5 (measured 0.0) for zero tokens and real tokens;
- residual actually wired (perturbing `W_zero` changes `phi_out`, 6.4e-01);
- causal mask: changing a future token leaves earlier positions unchanged (0.0), changing an
  earlier token moves later positions (2.0);
- padding-token contents do not change the pooled output (0.0);
- toggle off ⇒ no branch; toggle on + `events=None` ⇒ ValueError;
- `W_zero` into a branch-less Brain ⇒ RuntimeError; CNN state into an E3a Brain ⇒ 55 missing
  keys, all `event_branch.*`, no unexpected keys.

**Step-cost measurement (pre-run, RTX 5070 Ti, batch 1024, T=143, autocast, synthetic inputs)**:
CNN-only forward+backward+optimizer 133 ms/step vs E3a 220 ms/step (raw ratio 1.65×; the real
loop also pays data loading, so the expected end-to-end ratio is lower). The QA fallback
("step time > 2× or OOM ⇒ 2 layers, d=64") is **not triggered**; the arm uses 4 layers, d=128.
VRAM peak measured 6.5 GiB for the synthetic step (16 GiB card, ~6 GiB already held by other
contexts on this host).

## 5. Training protocol (declared BEFORE launch)

- **Step budget: 20,000 optimizer steps** (batch 1024, `opt_step_every=1`, i.e. 20.48 M
  samples). Todo 17's E2 arm, todo 21's A7 and A8, and todo 20's A9 **must reuse 20,000**.
- **From scratch** (no resume): both arms must start from random init because E2's
  `conv_channels` differs from 192 and cannot warm-start from any existing checkpoint; a
  warm-start for E3a only would break the comparison. It is understood that 20 k steps is
  ~1.5 % of one epoch over the 1.97 M-file corpus, so both arms are expected to be weak in
  absolute terms — the experiment is a *matched-budget* architecture comparison, not a
  strength claim.
- Config: `runs/exp/E3a/train.toml`, generated by `scripts/exp_e3a_make_configs.py` from
  `configs/config.toml.bak-19` (pre-0007 repo config) with exactly these changes:
  state/best/tensorboard/test_play/train_play/1v3 paths under `runs/exp/E3a/`;
  `save_every=2000`; `test_every=100000` (no test play); `batch_size=1024`;
  `num_workers=3`; `enable_compile=false` (the event length T is dynamic and would recompile;
  E2 must match); `[model] use_event_branch=true` + branch hyper-parameters; dataset globs
  2009–2022 with 2023 holdout unchanged; `file_index` = the existing
  `checkpoints/file_index.pth` (read-only).
- **Safety**: the four live paths (`state_file`, `best_state_file`, `tensorboard_dir`,
  `test_play.log_dir`) are asserted non-live by the generator immediately before launch, and
  the mtimes of `checkpoints/mortal.pth` / `best.pth` are checked before/after. The live
  checkpoint is never trained on or overwritten.
- Training log: `runs/exp/E3a/train.log`; checkpoints `runs/exp/E3a/state.pth` (every 2 k
  steps) and TB scalars `runs/exp/E3a/tb`.

## 6. Arm evaluation protocol (pool A, declared BEFORE launch)

- 12,000 games = 3,000 sets (one set = one seed × splits a/b/c/d), `seed_key =
  0x5dfaa4cef265cd7`, **`seed_start = 10000`** (pool A; pool B `50000` is reserved for todo
  22 and must not be consumed here).
- challenger = `runs/exp/E3a/state.pth` (the arm); champion =
  `checkpoints/mortal.pth.at-148800` (same champion as every other pool-A arm).
- own `log_dir` = `runs/exp/E3a/games/` (per-arm directory; no other arm shares it).
- `MORTAL_EVT_DEBUG=/tmp/opencode/t19_e3a_evt_debug.jsonl` is set for the whole eval:
  every engine-evaluated decision must appear with a non-empty token count `n ≥ 1`, i.e. the
  branch is provably fed during *this* arm's eval. (A branch-enabled model raises on
  `events=None`, so a silently branch-less run cannot complete; the debug file makes the feed
  observable.) Decision points: the arena additionally evaluates the 60 pre-empted `can_act`
  points per 12 hanchan that the dataset skips (`runs/exp/EVT/SPIKE.md`).
- Report: `scripts/eval_report.py --games-dir runs/exp/E3a/games --seed-key
  0x5dfaa4cef265cd7 --seed-range 10000-13000 --checkpoint state.pth --source-dir
  runs/exp/E3a --out runs/exp/E3a/report.csv`.
- **Resolution context** (must accompany any reading of this arm's numbers): at 12 k games the
  minimum detectable difference at 80 % power is ≈ **0.0239 avg_rank / 1.86 pt**
  (`runs/variance/README.md`, measured noise floor σ_set = 0.4678 / 36.37). Effects below
  that are *not* evidence of no effect.
- **Interpretation**: the plan's architecture verdict (todo 22) is E3a vs E2 with the primary
  criterion "Δpt 95 % CI lower bound > 0" measured **once** on pool B; this arm's pool-A
  `report.csv` feeds the five-arm ranking only. A null/negative outcome here is a valid result.
- Frozen-corpus invariant: `runs/1v3/*.json.gz` stays at exactly 2,004 files; no game logs are
  written there (the arm's games go to `runs/exp/E3a/games/`).

## 7. Artefacts

`runs/exp/E3a/{PREREG.md,train.log,state.pth,report.csv,selfcheck.json}`, configs
`runs/exp/E3a/train.toml`, scripts `scripts/exp_e3a_{selfcheck,verify_load,make_configs}.py`,
patch `patches/0007-model-event-branch.patch`, evidence `.omo/start-work/evidence/T19.txt`.

---

## 8. Addendum — interruption and resume (2026-09-18)

The first training process (pid 91707, started 2026-09-17 23:22:35) was terminated by an
**external kill** during an agent-session interruption at ~step 13,027 of the declared 20,000.
The last completed save is `runs/exp/E3a/state.pth` at **step 12,000** (2026-09-18 01:09:31,
holdout_loss 3.465353). The run was resumed from that checkpoint with the *same*
`runs/exp/E3a/train.toml`: optimizer / scheduler / scaler / best_perf state all come from the
checkpoint, and the **declared 20,000-step budget is unchanged**.

This is a resume of the arm's *own* interrupted run, not a warm start from a foreign or
pretrained checkpoint: PREREG §5's from-scratch rule forbids starting either arm from an
existing trained model (which would break the E3a-vs-E2 comparison), and no such checkpoint
enters this training. One honest caveat, recorded here because it is a property of the
interruption rather than of the protocol: `train.py` shuffles the file list and slices it
across DataLoader workers per process, so the **sample order after resume does not continue
the pre-kill order** (the same number of steps is run, over a fresh random pass of the corpus).
E2/todo 17 and A7/A8/A9 keep the same 20,000-step budget; they are unaffected by this note.
The resumed segment is logged to `runs/exp/E3a/train.log.resume` (the original `train.log`
is kept intact); both together are the arm's training log.
