# RiichiFormer V3.2 — 并行研发实施路线（远程精简版）

**状态：计划而非已执行代码。** 2026-11-21。当前本地 `rf/v3` / `rf3-g3` 记录 PR-17=BLOCKED_RIGHTS，远程 filtered tree `a4492e4d4e9a90bab9fb66f0b269ba77a97ea712`。不要直接合并无共同历史的分支，不要公开旧历史里含有牌谱片段的提交。

## 1. 项目决定：内部研究操作政策 != 第三方授权

用户要求更宽松的内部研究政策，不应让软件代码、合成测试、自有 selfplay 一起被第三方牌谱许可冻结。新的设计保留两份独立状态：`external_license_status` 是已核实第三方许可证/条款的真实证据；`operator_research_policy` 是项目所有者的内部研发选择与 allowlist。后者**不是**第三方授予的版权许可，不能将 `RESTRICTED` 或 `UNKNOWN` 伪造成 `ALLOWED`，也不能充当模型公开发布或在天凤凤凰桌运行 AI 的许可。第三方训练权利仍需要按适用条款核实。

## 2. 并行组织

**1 Lead + 4 Dev + 1 独立 Reviewer**。每名写入者一条 Git worktree 和 branch；Lead 独有状态表、资源锁、seed pool、合并权限；Reviewer 只读核查。共享接口 `shard.py`/`loader.py`/`rf_build_shards.py` 由 Lead 指定唯一修改者并序列化合并。

| Lane | 首轮工单 | 工作范围 | 合并验收 |
|---|---|---|---|
| A Policy | PR-37 Gate 解耦 | 新 PolicyDecision API、研究 profile、每游戏策略 hash、shard-v3、loader、train view | 外部事实原封不动；private 与 release 严格分离；旧 schema 不被静默改写 |
| B Compute | PR-38 / PR-39 | multi-decision shared causal forward、gradient accumulation、MVP-S CLI defaults、strict resume | Joint vs isolated-prefix logits/gradient parity；无未来信息；GPU benchmark |
| C Rules | PR-40 | R02 triple ron、R25 南四供托、R26 四家立直、R29 一发 | Native fixture；规则标签修正；不得破坏 dirty vendor |
| D Selfplay/Eval | PR-41 / PR-42 | 自有规则 bot 离线 rollout、OpponentRegistry、4-seat duplicate arena、seed ledger | 牌山及四席轮换、0 illegal、分数守恒，不调用在线天凤 |

## 3. 硬依赖

```text
PR-36 W0 安全基线
 ├── PR-37 Policy/API ───────┐
 ├── PR-38 Shared Forward ──┼── PR-39 Trainer/Resume ─┐
 ├── PR-40 Rules Fix ────────┼── PR-42 Arena ──────────┼── PR-43 G3 smoke
 └── PR-41 Selfplay ────────┴── PR-42 ────────────────┘
                                     PR-43 -> PR-44 B0
                                     PR-44 -> Data / Candidate / Risk / RL / optional Memory
                                                  -> PR-34/35 strong-opponent confirm
```

Wave 0 Lead 必须先核验本地 SHA、dirty submodule、H-00 备份与现有测试记录，更新老旧 PARALLEL_STATUS；锁定同一 BASE_SHA 后才能 fan-out。

Wave 1 四位 Dev **同时开发**不同文件，Reviewer 独立审计；不得为了并行修改未冻结接口或相同文件。

Wave 2 先合并 Policy 的 API 和迁移，再合并新的多决策模型、训练数值/strict resume，后面合并规则适配/自有 rollout/Arena。模型结构或 shard 格式变更要 bump schema，旧 checkpoint fail-closed，不能强行吞掉版本不一致。

## 4. PR-37：逐层清理重复 Gate

目前 `source_rights.csv` 11/0 ALLOWED → `split-v2` 每行 train_allowed false → `rf_build_shards.py` 检查训练许可 → `ShardWriter` → `ShardReader` → `rf_train.py`，是真正的多层硬锁。仅修改 CSV 不够。

新增纯函数 `PolicyDecision`，输入 source、action、profile、原始许可状态；输出 operational_allow、source_rights_status、policy_hash、reason。为 synthetic/selfplay 用 generator SHA/seed/ruleset 本地证明来源，而非 HTTP 许可伪造。迁移到 `rf-shard/3`，分开 `external_license_status` 和 `train_eligible_under_policy`，source/split/POV/合法动作检测仍为 P0 fail closed。默认严格 profile；私有研究 profile 需要显式参数，并在全部 checkpoint/shard 留存 provenance。公开发行始终用独立真实版权检查。

## 5. PR-38：共享 Transformer 计算

现状的 shard 只存一次 4 POV 事件，但 `iter_decisions` 使用各个 prefix，`encode_batch` 独立编码每次 decision，Transformer 因而重复计算。新增一条 full causal event stream + 多个 decision marker/state；每一决策只能看到当前以前事件，且目标/专家标签只能用于 loss，不得进入 forward。对同一新模型测 joint-forward 与 isolated-prefix-forward 的 logits **和梯度一致性**，还要测跨局、Pass、kan、censor、opponent hidden mutation、未来事件扰动。记录 RTX 5070 Ti 的真实吞吐和 VRAM，不预设加速倍数。

## 6. PR-39：训练正确性

修复 `training_step` 的多 microbatch 梯度累积权重：用所有有效样本权重之和归一化，包含 all-censored microbatch 的负控；验证与单大 batch 参数更新一致。修复 `rf_train.py` CLI 默认为 1 层 tiny 而非设计的 6 层 5.22M MVP-S：明确 `--tiny-smoke` 和 `--model-config`。Strict resume 还须验证 model/source/split/policy/canonical/config hash；`--init-weights-only` 是独立路径，不恢复 optimizer。

## 7. PR-40 / PR-41 / PR-42

Native 规则 R02/R25/R26/R29 必须有可执行 fixture；vendor dirty patch 先提交给 Lead。自有 bot 可随机合法、shanten/ukeire 与基础防守，不依赖第三方牌谱即可做本地 selfplay。每份样本记录 generator、ruleset、seed、四 POV、source SHA、split。Arena 采用四席旋转和 seed-set block bootstrap，独立 train/select/confirm 池。

## 8. PR-43 门槛与后续研究

按实际执行结果分别记录 `G3_CODE_PASS`、`SYNTHETIC_E2E_PASS`、`OWN_SELFPLAY_TRAIN_PASS`、`THIRD_PARTY_SOURCE_STATUS`、`STRENGTH_NOT_RUN`。先做 tiny synthetic → 完整六层 synthetic → own selfplay train → 四席 arena；不能因 mock 或训练 loss 降低而宣称强度。

B0 reference 固定数据/配置/策略/规则/模型/seed hashes。后续可同时开发：PR-18/19/20 Data、PR-31 Candidate + PR-25A Capacity、PR-32→27→33 Belief/Risk/Rank、PR-28/29 Selfplay RL，Memory 作为独立可选研究。**正式单 GPU 消融串行**；Lead 负责预注册、预算、独立强对手池、确认池不可复用。九段等效必须有规则校准、独立对手强度锚点、真实四席结果及置信区间；天凤位是长期目标非保证。

## 9. 运行与审计

RTX 5070 Ti 16GB，训练/正式评测同一时刻只能有一个 GPU owner，重 IO 单队列，夜间限制并行编译（如 `CARGO_BUILD_JOBS=4`）而不禁用散热保护。每位开发者交付 TASK_ID、BASE_SHA、COMMIT_SHA、文件、测试命令/exit codes、P0/P1、NOT_RUN、工件 SHA、GPU/IO 使用、下一个 hard gate。Reviewer 独立验证，只有 Lead 可以合并并更新状态。

**详细可执行文档和四位子代理完整提示词见 V3.2 ZIP 文件。**
