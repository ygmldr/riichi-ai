# Decisions — fork-roadmap-steps-1-10

Architectural choices and rationales discovered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---

## [2026-09-17T05:15Z] Session start — operational decisions (binding, from plan TL;DR)

- Delivery mode: **direct** in `/home/ygsldr/riichi-ai` (no worktree, no PR) — `/start-work` invoked with no flags.
- Verdict evals use the **pinned** `checkpoints/mortal.pth.at-175200` (copied to `runs/exp/PINNED/`), NEVER the live `checkpoints/mortal.pth` (328k steps as of 02:04).
- Every eval arm gets its own `[1v3].log_dir`; `configs/config.toml` edits serialize `3 → 5 → 6 → 7 → 17 → 22` with `configs/config.toml.bak-<todo#>` backups before each edit.
- Pool A = `seed_start=10000` (selection), Pool B = `seed_start=50000` (final confirm, exactly once).
- Wave-1 dispatch: parallel lanes {T1 = patch 0003 + release-only build, T4 = eval_report.py, T8 = analyze_alllast.py}; T2/T3 serialize after T1 (shared submodule patch-gen + config edit).

---

## [2026-09-17T08:30Z] MEASURED resolution of the 12k protocol (from T6) — binding for all later verdicts

- Pooled set-level dispersion over three disjoint 12k pools (T6): **σ_set(avg_rank) = 0.4678**, **σ_set(pt) = 36.37**.
- **MDD at 80% power: 12k games ⇒ 0.0239 avg_rank / 1.86 pt**; 6100 ⇒ 0.0336 / 2.61; 24000 ⇒ 0.0169 / 1.32. Rule of thumb: **n ≈ 6.87 / Δrank²** and **n ≈ 41525 / Δpt²**.
- **T5's pool-A "challenger better, CI excludes parity" did NOT replicate** on pools 2 and 3 (both CIs include parity: 2.4966 [2.4797,2.5133], 2.5053 [2.4879,2.5210]); the observed pool-A effect (0.0232 rank / +1.86 pt) sits at ~1.0× MDD.
- ⇒ Rules now in force for T22 (architecture) and T26 (A14): (a) always report the comparison against the MDD at the n used; (b) a single pool's "CI excludes 0" at 12k is **not** sufficient evidence for a claim of the order of 0.02 rank / 2 pt; (c) the pool-B confirm must be treated as the only confirmation, and its result must be reported with its CI width.
- **Arena determinism (measured)**: identical `(seed, seed_key, players, weights)` ⇒ byte-identical game logs except `meta.eval_time_ns`. So a repeated seed window is a **replicate, not an independent sample** — pool_1 ≡ pool A.
- The plan's inherited figure "12000 局可分辨 ≈0.05 顺位" was ~2× too pessimistic; the measured MDD is ≈0.024.

## [2026-09-17T08:30Z] A2 probe outcome (from T13) — belief line leans BLOCKED

- Pre-registered probe `opp_any_tenpai` on the frozen 1024-d `phi`: AUC 0.9334 (CI [0.9312,0.9354]), ECE 0.0099, random-label control 0.5007 (gate passes, |Δ|=0.0007).
- **But a 24-feature public-information baseline scores 0.9399 — higher than the embedding**; on a genuinely hidden target (`opp_any_aka_in_hand`) public features also win (0.6158 vs 0.6055). Positive control (visible-meld label) gave only 0.7947 < the pre-registered 0.90 ⇒ prereg verdict `pipeline_valid=false`, `gate_belief=block`.
- Post-hoc supplementary shows the measurement path is sound (exact meld planes → AUC 1.0000; permuted labels ≈ 0.5), so the shortfall is a property of the frozen encoder.
- ⇒ T15 is expected to emit **`gate_belief=block`**; the plan explicitly permits skipping T28/T29 in that case (record the skip in `docs/exp-gates.md`). T14 (GRP/Q calibration) can still reopen the line only via the independent ECE>0.05 criterion.

---

## [2026-09-17T22:00Z] Config state after T12 + dispatch rule for later eval tasks

- `configs/config.toml` is currently **clean vs git** (committed by T6) and carries `[1v3].seed_start = 30000` and `[1v3].log_dir = runs/variance/pool_3/games/`.
- The plan's canonical pool A is `seed_start = 10000`; pool B is `50000`. **Every eval task must set its own `seed_start` + per-arm `log_dir` explicitly** — never inherit the leftover 30000 window.
- Consumed seed windows so far (key `0x5dfaa4cef265cd7` unless noted): pool A / variance pool_1 / A1-original-shuffle-recency_zero = `[10000,13000)` (pool A ≡ variance pool_1 ≡ A1 `original`, all byte-identical — arena determinism); variance pool_2 = `[20000,23000)`; variance pool_3 = `[30000,33000)`; akochan ruler = `[60000,60040)`-ish (chunked, poisoned seeds skipped). **Pool B `[50000,53000)` remains UNCONSUMED and is reserved for T22 and T26, one shot each.**
- Another verified rule: `runs/1v3/` stays frozen at **2004** `*.json.gz` (ISSUE-1). Every eval dispatch must assert this before/after.
- Issue to clean up at T30: `docs/FORK_ROADMAP.md:266` now cites line numbers that were accurate pre-patch but drifted after patch 0005 (ISSUE-3 in issues.md). Re-derive them from the patched `train.py` at wrap-up.

---

## [2026-09-19T16:00Z] POOL-A RANKING OF THE FIVE ARCHITECTURE ARMS (all verified by Atlas re-report)

Protocol: key `0x5dfaa4cef265cd7`, seed window `[10000,13000)` = 3000 sets / 12000 games per arm, champion `checkpoints/mortal.pth.at-148800`, challenger = each arm's own `state.pth`, per-arm `log_dir`, B=2000 set-level bootstrap. **Pool B `[50000,53000)` is still UNCONSUMED and is reserved for todo 22's single E3a-vs-E2 shot.**

| arm | avg_rank [95% CI] | pt [95% CI] |
|---|---|---|
| A7 (c=192 CNN, same 20k budget) | **2.5372** [2.5191, 2.5552] | −3.4125 [−4.7369, −2.0474] |
| E3a (CNN + residual event Transformer) | 2.5506 [2.5327, 2.5685] | −4.4138 [−5.8351, −3.1012] |
| E2 (pure CNN c=202, param-matched) | 2.5631 [2.5450, 2.5797] | −4.0163 [−5.3401, −2.6288] |
| A9 (A7/E3a arch, history shuffled in training) | 2.6023 [2.5849, 2.6200] | −8.1938 [−9.5818, −6.8775] |
| A8 (E3a arch, `phi` zeroed ⇒ event-only) | 3.6073 [3.5957, 3.6192] | −89.625 [−90.885, −88.380] |
| (reference) pinned at-175200 baseline | 2.4768 [2.4599, 2.4938] | +1.8600 [+0.5663, +3.2026] |

Readings, with the measured resolution in view (**12k games ⇒ 80 %-power MDD ≈0.0239 rank / 1.86 pt**):
- The three CNN-bearing arms (A7/E3a/E2) span only 0.026 rank, i.e. right at the MDD ⇒ **pool A cannot separate them**; the E3a-vs-E2 verdict needs pool B (todo 22).
- A8's collapse (pt −89.6, far beyond any MDD) is decisive on its own: **the event branch alone cannot replace the CNN trunk** (roadmap A8's question → "no", for this instantiation).
- A9 is significantly worse than E3a in a **paired** comparison (Δrank +0.0517 CI [+0.0279,+0.0751]; Δpt −3.78 CI [−5.566,−1.905], both excluding 0 at ≈2× MDD) ⇒ shuffling history during training hurts, consistent with A1's inference-time result. Declared confound: training permuted / evaluation natural, so train-test order mismatch is included in the effect.

## [2026-09-19T19:35Z] T22 VERDICT — CLOSE the river-transformer line (pool-B single confirm)

PREREG: `runs/exp/ARCH_B/PREREG.md` (written 17:49:27, before the first pool-B arena start 17:49:58; single-shot declared).
Precondition feed gates (re-read and asserted before running): `runs/exp/EVT/EVAL_FEED.json`, `runs/exp/E3a/EVAL_FEED_E3A.json`, `runs/exp/A8/EVAL_FEED_A8.json`, `runs/exp/A9/EVAL_FEED_A9.json` — all assertions true, `failure_count=0` in all four.

Pool B (`seed_start=50000`, key `0x5dfaa4cef265cd7`, 12000 games/arm, per-arm `log_dir`), one shot per arm:
- E3a: dist `[2775 2943 2985 3297]`, avg_rank 2.5670 [2.5498,2.5848], pt −5.2425 [−6.6113,−3.8963]
- E2: dist `[2642 3188 3104 3066]`, avg_rank 2.5495 [2.5318,2.5668], pt −2.7225 [−4.0501,−1.3759]

Paired per-set bootstrap Δ(E3a − E2) (B=2000, rng seed 0): **Δavg_rank +0.0175 [−0.0059,+0.0409]** (crosses 0); **Δpt −2.5200 [−4.3427,−0.6446]** (entirely < 0).
Primary pre-registered criterion “Δpt 95 % CI lower bound > 0” **FAILS** (lower bound −4.3427).

**Decision: CLOSE the river-transformer (event-Transformer) line** under the current Q/CQL loss and the 20k-step matched budget. Keep the CNN trunk (A8's collapse is beyond any MDD; A7 shows the plain CNN at the same frames is at least as good on pool A). Re-open only with a materially different objective (e.g. todo 23 RL/KL) or different scale, and only with a fresh pre-registration + param-matched E3a-vs-E2-style control on an unconsumed seed pool. Pool B for E3a/E2 is now consumed; do not reuse it.
Honest caveats recorded in `runs/exp/ARCH_VERDICT.md`: single shot (quote CI widths: 3.70 pt / 0.0468 rank), pool-A vs pool-B disagree in rank sign while pt direction is consistent, A9's declared train/eval order-mismatch confound, E2 param-match ≠ FLOP-match, and the disclosed EVT smoke use of seeds [50000,50010).
Acceptance: five pool-A `report.csv` source dirs pairwise distinct (`runs/exp/{A7,A8,A9,E2,E3a}`) — asserted; `runs/1v3/*.json.gz` = 2004 before/after both runs; live checkpoint mtimes untouched.

---

## [2026-09-19T20:00Z] Resolution of two T23-surfaced conflicts (bindings for todos 24-27)

**C-1 — pool B is SPENT.** Todo 22 consumed pool B (`seed_start=50000`, window `[50000,53000)`) as the single confirmatory shot for the architecture verdict (E3a vs E2, paired Δpt −2.52 CI [−4.34,−0.64] → CLOSE). The plan's decision #6 also names pool B for todo 26 (A14), which can no longer be satisfied literally without re-using confirmation seeds.
**Binding resolution for todo 26**: use **pool A (`seed_start=10000`) for the trend**, and a **fresh never-used window for the confirmation** — `seed_start=70000` (`[70000,73000)`), documenting in PREREG that (a) pool B was spent by todo 22, (b) `60000` was consumed by the akochan ruler (T16), (c) `70000` has never been touched, and (d) the confirmation window is therefore still unseen-by-selection data, preserving the plan's intent even though the label "pool B" no longer applies. Any deviation from this must be argued in writing, not improvised.

**C-2 — patch 0008 breaks a closed-but-reusable script.** `scripts/exp_a2b_calibration.py:406` calls `q = dqn(mish, masks)` and is now tuple-incompatible with `DQN.forward → (q, v)`. It is on the do-not-modify list (todo 14 is closed and its artifacts are committed), so it was left untouched. Consequence: **re-running A2b requires a one-line unpack fix first**. Recorded here and scheduled for the wrap-up (todo 30) to fix or explicitly document as `known-breaking-after-0008`. It does NOT affect F2 (`py_compile` passes — this is a runtime type error, not a syntax error).

---

## [2026-09-19T23:15Z] T26 VERDICT — CLOSE the self-play (online RL) line under the declared recipe

PREREG: `runs/exp/A14/PREREG.md` (written 20:54:08, before the first training step at 21:04:23).
Protocol as pre-registered: one full pool cycle (cycle 1, freshly sampled) = 8,000 games
(4 members × 2,000), K=2, `online=true`, **`use_pg=true`** (lam_v=1.0, lam_kl=0.05, anchor =
at-148800), trainee initialised from a byte copy of `checkpoints/mortal.pth.at-148800`, plain CNN,
save_every=100, one generation only.

- Confirm window (fresh, never used): **`seed_start=70000`** (`[70000,73000)`), key
  `0x5dfaa4cef265cd7`, 12,000 games, champion at-148800, log_dir `runs/exp/A14/eval_confirm/games`.
  Trainee: dist `[2649 2487 2664 4200]`, **avg_rank 2.701250 [2.682165,2.719833]**,
  **pt −18.056250 [−19.500188,−16.589906]**; paired Δ vs the anchor seats
  **Δrank +0.268333 [+0.242886,+0.293111]**, **Δpt −24.075000 [−26.000250,−22.119875]**.
- Primary criterion (`pt_ci_lo > 0`) **FAILS**: the CI does not cross 0 but lies entirely *below*
  0 — the trainee is **detectably worse** than the anchor it started from (≈10× the 12k MDD of
  1.86 pt). Auxiliary criterion (`avg_rank_ci_hi < 2.500`) also **FAILS** (2.7198). Trend on pool A
  agrees: 2.692083 / −17.051250 (`runs/exp/A14/report_poolA.csv`).
- **Decision: CLOSE the online self-play line under this recipe and budget; A14b is NOT started**
  (roadmap stop-loss: A14 non-significant ⇒ do not add mixed replay). Mechanisms are recipe-level
  and were declared/observed: (i) upstream online-resume semantics skip the optimizer state when
  the checkpoint's own config has `online=false` ⇒ fresh AdamW at lr 1e-4 for 1,300 steps;
  (ii) online disables CQL (`train.py:378`); (iii) KL(λ=0.05)+next_rank(0.2) too weak. All losses
  were finite and optimised; this is objective mis-alignment, not divergence. **Not evidence that
  self-play RL cannot help** — re-open only with a new pre-registration that controls (i)/(ii)
  (warm-start/optimizer handling, CQL online, stronger KL) on an unconsumed seed pool.
- Integrity: `use_pg=true` proven by trainer log + finite TB scalars (pg/value_huber/kl_anchor over
  13 saves, steps 148,900–150,100); 8,000 games in 4 drains; final checkpoint step 150,100
  (sha256 `ac95620f…`); patch 0013 held live (no torch.cat error); `runs/1v3`=2004; live mtimes
  untouched. Artefacts in `runs/exp/A14/**`, evidence `.omo/start-work/evidence/T26.txt`.
- Commit message deviation (authorised by the task): the plan's `feat(exp): A14 self-play (pool B
  confirm)` is factually wrong now (pool B was spent by T22), so the commit uses
  `feat(exp): A14 self-play (confirm on fresh window 70000; pool B spent)` with the substitution
  stated in the body.

## [2026-09-21] Resume 决定 — 从 328k 续训纯 CNN 离线基线（M2）

- **依据**：M1 测量（池 A 12k）显示 148.8k/175.2k/328k 互相不可分辨；负结论臂已闭环（事件分支 E3a −2.52 pt/A8 崩溃；online 自博弈 Δpt −24.08；RL/PG 仅 A14 recipe 用过且失败）。用户指令"保留正结论、去掉负结论"。
- **配置**：新建 `configs/resume_baseline_328k.toml`（**不改** `configs/config.toml`，后者保留已完成计划的 toggles 作为记录）：由 `config.toml.sp-bak` 复制，唯一语义改动 `[model].use_event_branch=false`；继承 sp-bak 的 `online=false`、`use_pg=false`、`freeze_bn.mortal=false`（与 328k checkpoint 自带 config 的离线 regime 一致）；`state_file=checkpoints/mortal.pth`（续跑同一 live 文件）；`best_state_file`/`tensorboard_dir`/`test_play.log_dir`/`train_play.default.log_dir` 与 live 一致；`globs`=2009–2022、`val_globs`=2023、`file_index=checkpoints/file_index.pth`（重制版，1,968,147 条、0 条 /2023/）；`[resnet]` 与 optimizer/scheduler 不动。
- **启动前断言（全过）**：flags、Brain numel=10,787,456/event_branch=None、loss 位相同（Δ=0.0）、ckpt steps=328,000 且 sha==pin、pin 未动、`.bak` 校验、baseline.pth 存在、`runs/1v3`=2004、GPU 空闲。
- **锁与运行**：`/tmp/opencode/train.lock`（owner.txt 记 M2 resume）；`setsid nohup bash scripts/run_train.sh configs/resume_baseline_328k.toml`，TRAIN PID **3215**（15:18:29 启动），日志 `runs/dqn_train.log`（旧历史已在 `.bak`）。**训练有意保持 RUNNING**；期间禁止并发评估。
- **首个 interval 验收**：15:28:03 `holdout_loss 3.083202` + `total steps: 330,000`；15:36:27 `holdout_loss 2.985855` + `total steps: 332,000`；ckpt mtime 15:28→15:36（就地增长，含新 config）；实测 12,688→14,278 steps/h，无事件 collate 回归。下一里程碑：step 400,000 的 3000 局 test_play（≈20:30）。
