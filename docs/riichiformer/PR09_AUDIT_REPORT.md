# PR-09 阶段审计报告（实际状态：PR-06 完成，PR-07 草稿，PR-08/09 未实现）

**审计时间 (UTC)**：2026-11-20T19:07Z  
**审计基线 HEAD**：`aceb281602fb2703a1ed2067a006992cd9ddb680`（branch `main`）  
**审计分支**：`riichiformer-v2-audit-pr09`（审计产物提交；commit SHA 见最终 handoff）  
**目标仓库**：`git@github.com:ygmldr/riichi-ai.git`  
**审计性质**：只读代码/数据/测试审计；本轮**没有**改模型架构、**没有**训练、**没有**下载数据。

> **关键更正（必须先读）**：任务描述称“已推进至 PR-09”，但工作区真实状态是：
> **PR-00～PR-06 已有实现与证据；PR-07 只有未提交的 drift 草稿（9 个本地测试通过，但没有报告、
> 没有运行产物）；PR-08（Frozen V2 schema + 46 action mapping）和 PR-09（POV 投影器）尚未创建任何文件。**
> 因此本报告是“PR-06 完成 + PR-07 草稿”的审计，不是对已完成 PR-09 的验收。所有 PR-08/09 相关项
> 都按 `MISSING / NOT_IMPLEMENTED` 记录，不把设计文档当成实现。

---

## 0. 冻结现状（Task 1）

### 0.1 Git / 工作区

| 项 | 值 |
|---|---|
| branch | `main`（审计前）；审计产物推送分支 `riichiformer-v2-audit-pr09` |
| base HEAD | `aceb281602fb2703a1ed2067a006992cd9ddb680` |
| `vendor/mortal` | pinned = index = `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2`；工作树 dirty（11 改 + 3 untracked，patches 应用态） |
| tracked 改动 | `.gitignore`（本工作新增 12 行）+ 既有用户 `.omo/notepads/*` 两文件 + submodule gitlink |
| 用户既有改动 | **未触碰、未提交**（`.omo/notepads/fork-roadmap-steps-1-10/{decisions,learnings}.md`） |
| 未跟踪的非本项目文件 | `.zcode/`、根/子目录 `AGENTS.md`、`package*.json`、两份既有中文 docs、`vendor/AGENTS.md` —— **不纳入本次提交** |
| 工作区产物 | `artifacts/riichiformer/**` 全部 gitignored；没有把牌谱/权重纳入 git |

冻结原始输出：`docs/riichiformer/PR09_TEST_LOGS/freeze_state.txt`。

### 0.2 PR 状态（以文件/测试/产物为准）

| PR | 状态 | 证据 | 测试 |
|---|---|---|---|
| PR-00 | **完成** | `scripts/rf_preflight.py`、`BASELINE_LOCK.md`、`EVAL_POOL_REGISTRY.md`、`pr_reports/PR-00.md`；preflight 0 blockers | 21 passed |
| PR-01 | **完成** | `scripts/rf_bench.py`、`configs/riichiformer/benchmark.yaml`、`GPU_ENV.md`；4/4 shape PASS | 16 passed |
| PR-02 | **完成** | `riichiformer/schemas/provenance.py`、`riichiformer/data/scan.py`、`rf_data_audit.py`、`DATA_REPORT.md`；2,512,424 logs / 0 bad | 13 passed |
| PR-03 | **部分** | `canonicalize.py`、`dedup.py`、`rf_build_canonical.py`；仅 1,000 场 sample（`full_corpus=false`） | 16 passed |
| PR-04 | **部分** | `replay.py`、`rf_validate_replays.py`；1,000 场 / 660,568 decisions / 0 illegal | 6 passed |
| PR-05 | **部分** | `split.py`、`rf_build_split.py`、`SPLIT_REGISTRY.md`；2,512,424 行 × L/R，`duplicate_game_ids=0`；但 canonical/SHA 全量去重未做 | 12 passed |
| PR-06 | **完成（工具）** | `player_quality.py`、`rf_quality.py`、`quality_audit.md`；100k 场 / 400k rows；可靠性 `INCONCLUSIVE` | 14 passed |
| PR-07 | **草稿，未完成** | `drift.py`、`rf_drift.py`、`test_drift.py` **untracked**；9 tests pass；运行被中止，`DRIFT_REPORT.md`/CSV 不存在 | 9 passed（本地，未提交） |
| PR-08 | **未开始** | `schemas/event_v2.py`、`dataset_v2.py`、`FEATURE_SCHEMA_V2.md`、`rf_test_schema.py` 均不存在 | — |
| PR-09 | **未开始** | `data/pov.py`、`rf_test_pov.py` 不存在 | — |

完成度：**PR-00～PR-06 的“已记录 PASS（含部分限定）”共 7 张卡，PR-07 草稿，PR-08/09 空白。**
没有任何 V2 模型、训练器、shard、checkpoint、arena 代码；`scripts/rf_train.py`/`rf_eval.py`/
`rf_build_shards.py` 不存在。

---

## 1. P0 / P1 / P2 发现

严重性沿用 `04_ACCEPTANCE_MATRIX.md`：P0 = 隐藏信息/未来数据/非法行动/权利未知而训练；
P1 = 不能复现/样本错配/实验对照失效；P2 = 优化建议。

### P0-1（阻塞 PR-10/11/13）POV 投影与 ObservableEventV2 完全不存在

- **事实**：`riichiformer/data/pov.py`、`riichiformer/schemas/event_v2.py` 不存在（见 §0.2）。
  现有数据层是**原始全量事件**：
  - `canonicalize.py:63 parse_events` 接受并保留所有字段；`canonicalize.py:120 canonical_text`
    对所有 event 逐字段重序列化（含 `start_kyoku.tehais` 四家手牌、所有 `tsumo.pai`、`hora.ura_markers`）；
  - `scan.py:112 parse_mjai_sample` 只统计事件类型/`start_kyoku` 数量，不做 POV 投影；
  - `replay.py:92 replay_game` 返回 native `Gameplay` 的 actions/masks，**不是**模型输入，
    也没有 observable token stream。
- **影响**：现在**没有任何“模型输入”可以被审计**；因此不能说 POV 泄漏“已修复”或“已通过”。
  由于也没有训练代码，这**不是已上线的泄漏**；但它是 PR-10/11/13 的硬阻塞。
- **修复**：先做 PR-08 schema，再做 PR-09 `project_event(raw_event, pov)` 的 allowlist 投影与
  byte-equal 性质测试（见 `PR09_INTERFACE_CONTRACT.md`），然后 PR-10 才可消费。

### P0-2 来源权利未知 → 新训练仍被禁止

- `provenance.py:46` 仅允许 `ALLOWED`；`provenance.py:103` 默认 5 个来源中 4 个 `UNKNOWN`、
  `majsoul_optional` 为 `RESTRICTED`；split 全量 2,512,424 行 `blocked_source=true`、
  `train_allowed=0`（`PR-05.md`）。
- 这是审计矩阵定义的 P0 条件：**许可未核实不得下载/训练**。本审计没有触发训练，也没有改变该状态。
- 修复：人工审核并改写 `source_rights.csv` + PREREG；在 `ALLOWED` 之前 PR-13+ 不得启动。

### P1-1 全量 canonical/SHA 去重未完成，split 的“0 duplicate”只是 log-id 层面

- `PR-03.md` 明确 `full_corpus=false`（1,000/2,512,424）；`PR-05.md` 明确“canonical/SHA 全量去重
  NOT_RUN”；split 的 `duplicate_game_ids=0` 只证明 2.5M 个 Tenhou log id 唯一。
- 影响：不同 log id 的同一局转换副本不会被识别，可能跨 split；PR-12 shard 之前必须补全量 hash/canonical
  dedup，否则训练/验证集合可能泄漏。
- 证据：`split.py` 的 `duplicate_group_size` 仅按 `game_id_from_relpath`；`PR-05.md` §5。

### P1-2 46 动作空间没有冻结的 ActionMapping / action_map_hash

- 现在只有 libriichi 运行时 mask（`replay.py:118-119`），`ACTION_SPACE=46` 由
  `libriichi.consts.ACTION_SPACE` 断言；没有 `riichiformer` 内的 action map 文件、index→语义表、
  `action_map_hash`、checkpoint fail-closed 检查（PR-08 内容）。
- 1000 场 / 660,568 decisions / 0 illegal（`PR-04.md`）证明**当前样本**与 Rust 一致，但不构成
  “PR-08 映射已冻结”；PR-10 的 Pass/立直后弃牌/杠候选分支也未验证（见 P1-3）。

### P1-3 决策快照、censor、Pass、杠候选、label/input 类型隔离均未实现

- `replay.py` 把 `actions`、`masks`、`final_scores` 放在同一个返回 dict；它是 label/oracle API，
  不是模型输入。没有 `DecisionSnapshotV2`、`supervision_valid`、`censor_reason`、`executed_action`
  字段（PR-10 内容）。
- 现有 1,000 场随机 replay 没有专门构造：被抢和/碰杠导致的 censored 窗口、可 Pass 的响应、
  立直后摸切/手切、暗杠/加杠候选、双荣/三家和了。`PR-04.md` §3 已承认 double-ron 未单独证明。
- 影响：现在若直接接 PR-11/13，会把 “mask + action + 未来分数” 混在同一结构里，容易在未来把
  target 放进 input；必须在 PR-08/10 用类型层隔离。

### P1-4 PR-03/PR-04 的“全量”仍缺失

- PR-03 仅 1,000 场 sample；PR-04 仅 1,000 场（660,568 decisions）。全量 2,512,424 场 replay
  与 canonical index 都 NOT_RUN（PR-04 报告明确 sample）。
- 影响：合法率、重复率、invalid 统计不能外推到全量；PR-12 之前必须跑全量或明确分片抽样协议。

### P1-5 `.gitignore` 的 `data/` 规则曾静默忽略第一方包目录（已修复）

- 原 `.gitignore:14` 是未锚定的 `data/`，会匹配任意层级的 `data` 目录；首次 `git add`
  报错并发现 `riichiformer/data/{scan,replay,…}.py` 全部被忽略。若直接提交，会得到一个
  “代码在本地可用但远端缺失数据层”的不可复现分支。
- 审计中改为根锚定 `/data/`；`git check-ignore -v riichiformer/data/scan.py` 不再命中。
  这是提交完整性修复，不改变数据语义；需在 review 时确认远端分支确实包含该目录。

### P2-1 2026 只有 1 月且已被旧研究使用，confirm 不可用

- `DATA_REPORT.md`：2026 = 12,188 logs / 1 个月；`SPLIT_REGISTRY.md`：
  `confirm_blocked_previously_used=12,188`；不得称 virgin。
- 2009 从 2 月开始；年度覆盖 2010–2025 为完整 12 月。

### P2-2 玩家质量分可靠性 `INCONCLUSIVE`

- `quality_audit.md`：Q1 vs Q4 平均顺位差 −0.0617，CI 大幅重叠、非单调；`top25_seat` 只能作为
  候选 flag，不能当已验证强度。且 56.89% 行历史不足是“只抽 10 万场”造成的。

### P2-3 PR-07 drift 只有草稿

- 9 个测试通过，但运行被中止（`artifacts/riichiformer/drift/` 为空），没有 `DRIFT_REPORT.md`/
  `drift_cohorts.csv`。本轮审计**不**把 PR-07 计为完成。

### P2-4 审计证据的存放方式

- 大体量产物（`source_audit.parquet` 21.6MB、split parquet、quality parquet、bench JSON）仍在
  gitignored `artifacts/`；本审计已把**小型测试日志**复制到
  `docs/riichiformer/PR09_TEST_LOGS/` 并提交，便于离线 review。大数据是否上传需单独决定。

---

## 2. Task 2：数据与 POV 接口逐项审计

| # | 审计项 | 结论 | 证据 / 缺口 |
|---|---|---|---|
| 1 | 不同 POV 只可见该时刻合法信息；对手起手/摸入/隐藏状态不泄漏 | **NOT_IMPLEMENTED（P0）** | 没有 `pov.py`/`ObservableEventV2`；canonicalize 保留全部 `tehais`/`tsumo.pai`/`ura_markers`；没有 byte-equal POV 测试。无模型输入，故无已发布泄漏，但阻塞 PR-10/11/13。 |
| 2 | 全局事件顺序；摸切/手切、赤、副露构成、立直、结算 | **PARTIAL** | `canonicalize.py` 严格逐行保留顺序/字段；`replay.py` 用 native state 重放 1,000 场；但 V2 事件投影不存在。`src/state`/`mjai/event.rs` 是权威，未复制到 Python。 |
| 3 | 自家/公开普通牌与赤牌分别计数；公开区不重复计同一物理牌 | **NOT_IMPLEMENTED（P0/P1）** | 没有 visible-tile counter、没有 aka 计数、没有公开牌去重；raw `consumed`/`pai` 保留 aka tile id（Rust tile 0..36），但 V2 字段不存在。 |
| 4 | 决策输入不含目标/未来；训练/在线编码一致 | **NOT_IMPLEMENTED（P1）** | 没有模型输入；`replay_game` 同时返回 labels/masks/scores，未做类型隔离；quality 的 `target_rank_for_eval` 是 eval-only（有测试），但那是质量分不是决策模型。 |
| 5 | action label / 46 mask / Pass / 立直后弃牌 / 多候选杠 与 Rust 一致 | **PARTIAL（P1）** | 1,000 场 0 illegal（`PR-04.md`）；mask 来自 `GameplayLoader`；但没有冻结 action map、没有 censor/Pass/杠候选专项测试，且未全量。 |
| 6 | 时间划分、去重、来源/许可可追溯 | **PARTIAL（P1/P0）** | split 2,512,424 × L/R，`duplicate_game_ids=0`，registry hashes；`source_rights.csv` 全 unknown；全量 canonical/SHA 去重未做；2026 仅 1 月。 |

补充：旧 `patches/0006` 的 event tokenizer 会把对手 `tsumo.pai` 写进 token（见 `BASELINE_LOCK.md` R-01）。
当前部署权重 `checkpoints/mortal.pth` 的 `use_event_branch=false` 且无 event 权重，**不能**说旧 CNN
线上模型泄漏；V2 也尚未复用该 tokenizer（因为 PR-09 不存在），后续 PR-09 必须独立实现，不得复用 0006。

---

## 3. Task 3：架构接口可支持性

| 目标能力 | 当前是否支持 | 说明 / 缺口 |
|---|---|---|
| Local Causal Transformer | **否（未实现）** | 无事件 token、无序列布局、无 attention mask/causal 测试；只有 raw logs + replay。 |
| 结构化 State Encoder | **否** | 无 `DecisionSnapshotV2`（手牌计数、分数、宝牌、合法 mask）；PR-10 未做。 |
| 46 维动作空间 | **仅运行时** | libriichi mask 46 与 1,000 场一致；没有冻结 mapping/`action_map_hash`（PR-08）。 |
| Candidate-based Policy Head | **否** | 没有候选动作特征/负样本；现有 mask 只能做 legal/illegal，不能做 per-candidate logits。 |
| 对手听牌/待牌/风险监督 | **否** | 没有 belief 标签；`quality_scores` 只是玩家质量，不是 belief。 |
| 跨局历史记忆 | **否** | 无 kyoku summary、无 memory fence 测试；PR-22/23 未开始。 |

结论：**当前数据层与接口不足以支持任一 V2 模型实验**；不是“已经支持只是没训练”。但已完成的数据审计/
split/replay/quality 可作为基础复用（见 `PR09_CHANGE_IMPACT.md`）。

---

## 4. 审计产物与复现

### 4.1 测试结果（真实执行）

- 已提交范围（PR-00～06）：**98 passed / 0 failed**，44.57s。
- 含 PR-07 草稿（未提交）：**107 passed / 0 failed**，44.93s。
- `compileall` exit 0。
- 日志：`docs/riichiformer/PR09_TEST_LOGS/`（per-file + full_suite_committed + freeze_state + compileall；
  `wip_pr07_drift.log` 是草稿日志）。

### 4.2 关键产物 SHA-256

| 产物 | SHA-256 |
|---|---|
| `artifacts/riichiformer/preflight.json` | `4d72961a40b156bc77e3394459057edb5e3e8b101e51dfcb7c72afaa801deef2` |
| `artifacts/riichiformer/bench/gpu_baseline.json` | `db844fb30c5b365b94abe49928abf4ec6ebf0133b1ad5c740b1640963983cc37` |
| `artifacts/riichiformer/audit/source_audit.parquet` | `ebb90909558e16e57a221b0e773c923a5d19a3df84dc48806d13094b16543f52` |
| `split_manifest_L.parquet` | `5e291ad09aa35ab6a392c24c25a4e5b1be97289aad4c6c938253215755f3f27b` |
| `split_manifest_R.parquet` | `6fbf3465eee1be29cfcbba40c47876e0c2a092a24381e65481eb22c8129cb44e` |
| `quality_scores.parquet` | `d18c879ac23d2c38536b2ea82476e47f8e625bd0e97f3938c11332671f0a7e74` |

### 4.3 相关文件

- 已完成卡证据：`docs/riichiformer/pr_reports/PR-00.md` … `PR-06.md`；
  `BASELINE_LOCK.md`、`EVAL_POOL_REGISTRY.md`、`DATA_REPORT.md`、`GPU_ENV.md`、`SPLIT_REGISTRY.md`、
  `quality_audit.md`。
- 接口契约：`docs/riichiformer/PR09_INTERFACE_CONTRACT.md`。
- 变更影响：`docs/riichiformer/PR09_CHANGE_IMPACT.md`。
- 测试矩阵：`docs/riichiformer/PR09_TEST_MATRIX.md`。
- base commit：`aceb281602fb2703a1ed2067a006992cd9ddb680`；审计分支与 commit 见最终 handoff。
- diff 范围：仅新增第一方 `riichiformer/`、`scripts/rf_*.py`、`tests/rf_test_*.py`、
  `configs/riichiformer/`、`docs/riichiformer/` 与 `.gitignore`；**未**改 `vendor/mortal`、checkpoints、
  configs/config.toml、patches、datasets。

---

## 5. 审计结论与停止点

- **实际完成度：PR-00～PR-06 有证据；PR-07 草稿；PR-08/09 未开始。** 因此“PR-09 审计”无法给出
  POV/action/schema 通过结论；这些项目前是 P0/P1 阻塞项，而不是“已验证”。
- 已完成部分中，**数据审计、split、canonicalisation、native replay、quality privacy** 可复用；
  不要因为 v2.1 架构调整重写它们。
- 本次审计**没有**执行 Task 4 的模型改动，也没有开始 PR-08/09/13。
- 下一步应由人审核本报告与 `PR09_CHANGE_IMPACT.md`，然后决定：
  1) 先补 PR-08/09（推荐，任何模型实验的前置）；
  2) 再补 PR-03/04 全量与 canonical dedup；
  3) 权利核实前不训练。
- **审计结束，停止。** 不自动开始架构重构或训练。
