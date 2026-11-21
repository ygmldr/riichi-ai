# RiichiFormer V3.1 — INTEGRATION_LOG

按时间追加；记录 Lead 实际执行的合并、测试、hash 与 gate，不记录子代理自述为事实。

## 2026-11-21 — W0：H-00 通过

- **输入**：V3.1 pack（`docs/riichiformer/v3_1/riichiformer_v3_1_parallel_pack/`，用户上传解压）。
- **基线**：审计 HEAD `f23bb0dc9a6f483686e3c0eb195b36d302d3b980`；本地 `main` `aceb2816…`；`origin/main` `a3ccd853…`（filtered history）。`git merge-base origin/main HEAD` 无输出/退出码 1 → 无共同祖先。
- **H-00 实现提交**：`e585f46`（`feat(h00): git safeguard, baseline snapshot and parallel worktree policy`）。
- **快照**：`artifacts/riichiformer/git_safeguard/20261120T194540Z/`
  - manifest SHA-256 `ee8036b3a8811df7fc044e1c8c4927363fe8613c7406bcfc938ec9bf5d9c4775`
  - bundle SHA-256 `705665a63ee8f24370cf9779bb6549e8450388af046c86e0628dd7eca3ba26af`
  - `verify`：49 entries，0 failures。
  - 用户内容：4 dirty tracked（含 `.gitignore` 修改）+ 24 untracked + vendor 11 modified/3 untracked；未删除、未 stash。
- **`.gitignore`**：`/data/` 与 `/worktrees/` 锚定；`ignored-check` 通过（`git check-ignore --no-index`）。
- **干净工作树复现**：`git worktree add --detach worktrees/w0-verify e585f46` → `git ls-files riichiformer/data` 列出 7 个第一方文件；`git status --porcelain` 空；`pytest -q tests/rf_test_git_safeguard.py` → **8 passed**（0.31 s）；`ignored-check` rc=0。验证后移除临时 worktree。
- **验收提交**：本文件与 `PARALLEL_STATUS.md` 所在提交；tag `rf3-base`。
- **Gate W0**：PASS。允许从 `rf3-base` 创建 `worktrees/h01..h04` 并派发四条 lane。
- **未做**：未 merge/reset/clean/push -f；未触碰 vendor 补丁；未启动训练/下载。

## 2026-11-21 — W0 基线回归（rf/v3）

- `git ls-files 'tests/rf_test_*.py'` 的提交范围测试：**106 passed**，45.84 s（含 H-00 新增 8 项）。
- 日志：`docs/riichiformer/v3/logs/H00_committed_suite.log`（sha256 见提交）。
- `compileall`：`riichiformer/` + `scripts/rf_git_safeguard.py` 退出码 0。
- 说明：该运行不含未提交的 PR-07 drift 草稿（`tests/rf_test_drift.py`），因此与审计时的 98/107 口径不同。

## 2026-11-21 — Wave 1 合并（rf/v3）

- 分支 `rf/v3` 从 `rf3-base` 前进；按顺序 cherry-pick `-x`：
  - H-01 `450eb1a` → `9d59d68`
  - H-02 `c8fd6b1` → `98fd5e9`
  - H-03 `2c80c7c` → `26de532`
  - H-04 `1cf7f04` → `f019d5c`
  - PR-07R `011b5b8` → `de27d41`
- 全量提交范围测试（11 个 `tests/rf_test_*.py`）：**233 passed, 1 skipped, 109 subtests passed**，45.35 s；日志 `docs/riichiformer/v3/logs/W1_full_suite.log`；`compileall` 退出码 0。
- 独立 Reviewer（H-01/H-03）：H-01 **PASS_WITH_P1**（P1-1 真实 Tenhou fixture 片段属于再分发限制，公开合并/发布前必须换成 synthetic 或运行时生成；P1-2 `eval_report.py` 仍有第二套结算；P1-3 PR-06 质量过期）；H-03 **PASS_WITH_P1**（P1-1 gate 对 reviewer=`AI`/`LLM` 等 fail-open；P1-2 gate 尚未接入 shard/training 路径）。无 P0。
- H-02 全量 2.51M canonical 已由 Lead 启动：job `bash-9138f154f85d4d10`，日志 `artifacts/riichiformer/h02/canonical/run_log.jsonl`，输出 `artifacts/riichiformer/h02/canonical/`；预计 ~2.5 h。
- PR-08 Schema 由 Schema Owner 在 `worktrees/schema` / `rf/pr08-schema` 起草；PR-09 未开始；H-03 gate 修复已派回原 owner。
- **未**改变 source_rights：所有 11 行仍然非 `ALLOWED`，真实训练继续 blocked。

## 2026-11-21 — Gate W1 通过（tag `rf3-w1`）

- 修复合入：H-02 guard/verify `c4f6b36`→`0fa8391`；H-03 gate 加固 `1a003e3`→`741513b`；H-04 证据锚点 `a32e73a`→`dae0f41`；PR-07R S4 分类修复 `b6be041`→`da4b802`；PR-08 schema 候选 `bc07334`→`885b14f`（仍待独立 schema review）。
- 全量测试（已跟踪的 111 个 `tests/rf_test_*.py`）：**306 passed, 1 skipped, 210 subtests passed**，44.74 s；日志 `docs/riichiformer/v3/logs/W1_full_suite.log`；compileall 退出码 0。
- **tag `rf3-w1` = `da4b8020216c80b1481817aad70096e933a6c78e`**。
- Reviewer-2 结论：H-02 PASS_WITH_P1（全量未完成 + guard 已修）；H-04 PASS（P2 证据契约已修）；PR-07R PASS_WITH_P1（S4 已修并重跑）。
- PR-08 schema 独立 review 进行中；PR-09 POV 与 PR-13 synthetic model 已在独立 worktree 开发。
- H-02 全量 canonical job `bash-9138f154f85d4d10` 仍在运行（约 2 h ETA），完成后用 `0fa8391` 代码跑 split-v2。

## 2026-11-21 — Wave 2 接口门（PR-08/PR-09/PR-13）

- PR-08 独立 review 首轮 **PASS_WITH_P1**；修复提交 `9111365`（cherry-pick `edf5720`）：
  - `DecisionInput` 精确类/字段集；`expected_hashes` 必须三键齐全；补 `verify_code_consistency`；`deltas_order` 进入 hora/ryukyoku，版本升至 **`rf-schema-v3/2`**。
  - 新 hashes：schema `12503d36e2571449d4e4bdd4c8780e00f285d3a2ec81f9e26af184d24c8ef43c`；action `6cf1efc5…`；normalizer `37df861b…`。schema tests **66 passed**。
- PR-09 独立 review **PASS_WITH_P1**（无泄漏）；follow-up `a8a2b06` → cherry-pick `8c7cbe0`：
  - `project_event` 设 `deltas_order="relative"`，factory 保持 absolute；新增 `ProjectedStreamHeader(pov, projection_version, schema_version, ruleset_id, source_kind)`；`extend()` 原子化；隐藏 ankan 先 mask 再 parse。
  - POV tests **28 passed**（schema+POV 合并 94 passed）。
- PR-13 合成模型 `b18ae4f` + 报告 `a4be4a9` 已合入；`pytest tests/rf_test_models.py` **43 passed, 7 subtests**；5,217,582 params；合成 overfit 收敛；未使用真实数据/训练/checkpoint。
- PR-10 已从 `8c7cbe0` 起步（`worktrees/decision`，branch `rf/pr10-decision`），尚未提交。
- H-02 全量 job `bash-9138f154f85d4d10` 仍在运行（约 44/101 shards）。
- W2 集成测试：**382 passed, 1 skipped, 217 subtests passed**，63.23 s；日志 `docs/riichiformer/v3/logs/W2_full_suite.log`。

## 2026-11-21 — PR-10/PR-11/PR-13 合入，PR-12 启动

- PR-10 `1daddba`（cherry-pick `80df45d`）：10k decisions、**0 illegal labels**、50 censored（29 call / 21 ron）、phases 8000/1990/10；19 tests。
- PR-11 `3e467ff`（cherry-pick `dc14f7f`）：离线/在线编码同映射，12 logs × 4 POV × 3 decisions = 144 组 byte-equal；15 tests。
- PR-13 `8f58ac9` + `31773dd`（合入 `90380d7`/`8de6639`）：ModelBatch 可见性 fail-closed；51 tests；5,217,582 params；仅合成 forward/backward。
- 集成全量：**424 passed, 1 skipped, 225 subtests**，144.38 s；日志 `docs/riichiformer/v3/logs/W2_full_suite.log`。
- PR-12 在 `worktrees/shard` / `rf/pr12-shards` 启动；只允许 100k fixture，train shard 必须 source_rights ALLOWED，否则拒绝。
- H-02 全量 80/101 shards（2024），估计 ~30 min 完成。
- Reviewer P1 修复：PR-10 `tiles_left` 不再 `+kan_count`（`f691002`→合入 `5db301e`）；PR-11 用修复后的 state 验证（`dc61f16`→`2afb1ac`）；决策 14 tests、encoder 16 tests 通过。
- PR-12 在重置到 `2afb1ac` 后重建；PR-16 mock arena 已在 `worktrees/arena` 启动。
- PR-12 `8ff21f0`（合入 `af7cfbe`）：games row + 四 POV 流 + 分离 targets + 显式 hashes + source-rights 训练门；11 tests；10 局 audit build 6,485 decisions，shard SHA `b8a8203b…`。仍为 PARTIAL，等 H-02 全量。
- PR-14 trainer 与 PR-16 mock arena 已在独立 worktree 启动；PR-12 由 H-02 owner 独立 review。

## 2026-11-21 — H-02 全量完成

- Canonical 2,512,424 rows，109 shards，`valid=2512424`、`invalid=0`、`duplicate_components=0`、`quarantine=0`；index `21d9077f…`，manifest 见 `artifacts/riichiformer/h02/canonical/manifest.json`。
- 首次 finalize 因默认 SQLite temp 目录不可写失败；已用 `SQLITE_TMPDIR` 重跑 finalize，并修复 `CanonicalStore` 固定写 `sqlite_tmp/`（`11a92a1`）。
- Full split-v2：L/R 各 2,512,424 行，`moved=0`、`quarantine=0`；registry `86d7b557…` / `7ffbe7c6…`。
- PR-15/16 已合入（7 / 17 tests）；PR-17 报告按 **BLOCKED_RIGHTS** 起草，不伪造训练/arena 结果。

## 2026-11-21 — W3 组件测试与 PR-12 review

- 全量测试 **471 passed, 2 skipped, 242 subtests**，165.89 s；日志 `docs/riichiformer/v3/logs/W3_full_suite.log`。
- PR-12 review **PASS_WITH_P1**：write-root guard、representative 选择（不能要求 component_size==1）、train split/source 校验、loader 大文件流式 hash/sample 需修；已派回 H-02 owner。
- PR-14/15/16 已合入：trainer 11 tests（synthetic）、offline 7 tests（synthetic）、arena mock 17 tests（10/100/1000 hands, illegal=0）。
- Full canonical/split 完成：2,512,424 rows，0 invalid/duplicate/quarantine；split registry L `86d7b557…`、R `7ffbe7c6…`。
- PR-17 报告按 **BLOCKED_RIGHTS** 起草中；不伪造训练/arena 强度。

## 2026-11-21 — PR-17 G3 = BLOCKED_RIGHTS

- `f31b415`（合入）记录 G3 **BLOCKED_RIGHTS**，不是 PASS：代码/数据链 green，但 H-03 11 个来源 0 个 ALLOWED，不能创建合法训练 shard/checkpoint。
- 权威测试：**471 passed, 2 skipped, 242 subtests**（155.75 s）；full canonical 2,512,424 / 109 shards / 0 invalid；split-v2 L/R 2,512,424 / 0 moved。
- 报告：`docs/riichiformer/v3/PR17_GATE.md`、`docs/riichiformer/pr_reports/PR-17.md`。
- 未运行：真实训练、checkpoint、arena 强度、confirm。PR-18 之后真实实验均被 rights 阻塞。

## 2026-11-21 — G3 代码链收尾

- PR-12 P1 修复 `6b5a742`（合入 `2af7e13`）：write-root guard、representative 选择、per-game train split/rights/source 校验、流式 hash/sample、content hash、fsync；23 tests。
- PR-14/PR-15 fixture 对齐 PR-12 `rf-shard/2` 严格 gate：`3c87e83` / `3c44714`；training 11 + shard 23 + offline 7 全绿。
- 最终集成全量：**484 passed, 1 skipped, 242 subtests**，164.55 s；日志 `docs/riichiformer/v3/logs/W3_full_suite.log`。
- **公开推送注意**：HEAD 的 H-01 fixture 已是 synthetic，但历史提交 `9d59d68` 等仍含旧真实牌谱片段；公开 push 前必须 filtered-history 重写，不能直接推送本分支历史。

## 2026-11-21 — V3.2 remote branch accepted, Wave 1 opened

- 从 GitHub `origin/rf/v3.2-policy-parallel-plan` 拉取到 `worktrees/v3.2-policy`（分支 `rf/v3.2-local`），没有用旧 V3.2 包覆盖远程树。
- 全量测试 **494 passed, 2 skipped, 242 subtests**（145.41 s）。首次运行缺少 `vendor/mortal` 子模块导致 25 个 source-contract 环境失败；初始化 pin `0cff2b5` 后通过。
- 真实 H-02 10 局私有 shard：`rf-shard/3`、8,504 decisions、policy `operator_private_research`、`rights_status=UNKNOWN`/`train_allowed=false` 原样保留；`licensed_only` 仍拒绝。
- 6 层 5,217,582 参数 GPU 两步 smoke：loss 2.193→1.712；strict resume 到 step 3；protocol-R split mismatch 被拒绝。
- 结论 **PASS_WITH_P1**：policy hash 未配置认证、builder 默认过宽、无 `--policy-config`。报告 `docs/riichiformer/v3_2/ACCEPTANCE_LEAD.md`。
- 建立四个独立 worktree（`v32-policy/compute/rules/selfplay`）并派发 PR-37 follow-up、PR-38/39、PR-40、PR-41/42；状态见 `docs/riichiformer/v3_2/WAVE1_STATUS.md`。
- 本轮不运行真实训练/强度/发布；source_rights 仍 11 行 0 ALLOWED。

## 2026-11-21 — V3.2 Wave 1 merged

- 四个 Lane 均从 `cbf166a` 派生并合并到 `rf/v32-integration`：PR-37 `87afe78`、PR-40 `8248496`、PR-38/39 `4e4754b`、PR-41/42 `4e380a3`。
- 独立 Reviewer：PR-37 `PASS_WITH_P1`、PR-40 `PASS_WITH_P1`；P1 已在 `c426ac7` 修复（bool gate 强转、R02/R26 证据 tautology、R02 continuation 标注 source-read、R25 引用）。
- 集成全量：**683 passed, 2 skipped, 281 subtests**（187.73 s），日志 `docs/riichiformer/v3/logs/V3_2_W1_full_suite.log`。
- `rf_train.py` 默认改为 6 层 MVP-S（5,217,582 参数），新增 `--tiny-smoke` / `--init-weights-only` / `--model-config`；`rf-shard/3` 私有策略要求显式 policy config/hash + allowlist。
- PR-43 分项：`CODE_PASS`、`SYNTHETIC_E2E_PASS`、`PRIVATE_H2_TRAIN_SMOKE` PASS；`OWN_SELFPLAY_TRAIN_PASS=NOT_RUN`、`STRENGTH_NOT_RUN`；外部权利仍 11 行 0 ALLOWED。报告 `docs/riichiformer/v3_2/PR43_GATE.md`。
- 未推送开发分支、未改公开 main、未训练真实 B0；0014 rules patch 只在本 worktree 本地应用，未提交 vendor 指针。


## 2026-11-21 — V3.2 Wave 2: PR-38 loader/trainer + B0 pilot

- `rf/v32-b0` `ce5a290`（合入 `45eefa1` + 最终报告）：`iter_multi_decision_batches`、`training_step_multi`、`--multi-decision`、长 POV 自适应 `max_events/max_positions`；旧 prefix 路径仍为默认。
- 集成全量 **708 passed, 2 skipped, 281 subtests**（235.27 s），日志 `docs/riichiformer/v3/logs/V3_2_W2_full_suite.log`。
- 500 局确定性 B0 sample/shard：333,224 decisions，shard `f474a3f1…`；两个 seed 各 2,000 步 GPU pilot（107,026 decisions / 1,766,242 tokens），checkpoint `f3879cb8…` / `3aa54cbe…`，strict resume 到 step 501。报告 `docs/riichiformer/v3_2/PR44_B0_PILOT.md`。
- 这是 pilot，不是完整 B0；无验证集、无 arena 强度、无段位/Mortal 结论。
