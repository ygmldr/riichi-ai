# PR-09 变更影响评估（v2.1 设计调整，不执行）

**前提**：本文件只做影响分析。**不实现**任何模型改动，不开始 PR-08/09/13 的代码，不训练。
基线 HEAD：`aceb281602fb2703a1ed2067a006992cd9ddb680`；实际完成度见 `PR09_AUDIT_REPORT.md`。

## 1. v2.1 决策（来自任务，未执行）

1. 保留 **Local Causal Transformer** 作为主 backbone。
2. State Encoder 优先轻量化；Set Transformer 作为对照。
3. Candidate-based Policy Head 与普通 Linear Head 做受控消融。
4. Belief 从“听牌概率”扩展到“待牌概率 + 候选弃牌风险”。
5. Cross-Kyoku Memory 降级为“需要实验证明有效”的可选模块。
6. Mortal 仅作为“现有五段水平工程基线”，不默认强教师；蒸馏必须有单独实验。
7. 策略能力以独立 duplicate arena 评测为准，不以 BC loss / Mortal agreement 代替。

## 2. 对尚未开始 PR 的影响

| PR | 当前状态 | v2.1 影响 | 需要的具体改动 |
|---|---|---|---|
| PR-08 Frozen schema / action map | 未开始 | **高**（所有下游依赖） | 在 schema 中预留 candidate 描述、belief target 类型；action map 冻结；schema/action/normalizer hash；版本化。 |
| PR-09 POV projector | 未开始 | **中** | 仍按 POV allowlist；新增字段（如 visible tile counts、candidate 所需公开信息）必须在可见性表内；不得因 candidate/belief 需要而放宽隐藏牌。 |
| PR-10 DecisionSnapshot / censor | 未开始 | **高** | Snapshot 增加 candidate-set 输入（动作候选/特征）与 belief 标签的**分离结构**；保持 label/input 类型隔离；Pass/censor/杠选择测试。 |
| PR-11 offline/online encoder | 未开始 | **高** | 统一 encoder 接口需同时适配：local token Transformer、轻量 state encoder、candidate head；Set Transformer 作为可替换 state encoder。 |
| PR-12 shard/dataloader | 未开始 | **中** | shard 增加 schema/candidate/belief 版本字段；仍按 game/POV 组织；不重复前缀。 |
| PR-13 MVP-S 模型 | 未开始 | **高** | Local backbone + 轻量 state encoder；Linear head 与 candidate head 均可插拔；参数量/FLOPs 实测。 |
| PR-14 BC trainer/checkpoint | 未开始 | **中** | checkpoint 带 schema/action/normalizer/candidate 版本；resume 校验；candidate loss 开关。 |
| PR-15 offline eval | 未开始 | **中** | 输出 policy NLL/accuracy 仅作诊断；不得当强度结论；新增 candidate-head 校准指标。 |
| PR-16 arena adapter | 未开始 | **中** | 46 action 解码必须支持 candidate/linear 两种 head；Pass/censor 在线一致。 |
| PR-17 G3 | 未开始 | **低** | 门槛改为“可训练/可出合法动作/arena smoke”，不以 BC loss 作为强弱。 |
| PR-18 sampler | 未开始 | **中** | 若要按质量/年代采样，需明确 champion/quality 标签；PR-06 的 `top25` 目前 INCONCLUSIVE，不能当既定质量。 |
| PR-19 近期数据审计 | 未开始 | **低** | 仍受权利/2026 覆盖限制。 |
| PR-20 数据消融 | 未开始 | **低** | 依赖 PR-18；不改模型。 |
| PR-21 教师蒸馏 | 未开始 | **高** | Mortal 仅 baseline；蒸馏必须是独立 arm、明确许可与 Q 尺度；不能默认“强教师”。 |
| PR-22 局摘要 | 未开始 | **降级** | Memory 可选；仍应保留 kyoku 边界与 summary 数据格式，但不必进 MVP。 |
| PR-23 Main-H | 未开始 | **降级/可选** | 先证明 local 模型与 candidate/state 方案；memory 作为后续实验，不阻塞 G3。 |
| PR-24 Flat-F | 未开始 | **低** | 仍是长序列对照；评估标准改为 arena，不只 NLL。 |
| PR-25 KV cache | 未开始 | **低** | 只有在 memory/长上下文方案被保留时才需要。 |
| PR-26 历史长度消融 | 未开始 | **降级** | 若 memory 不保留，可延后/缩范围。 |
| PR-27 Belief 数据隔离/校准 | 未开始 | **高** | 目标从 tenpai 扩展到 waits + risk；需要 PR-08 预留 target schema、PR-10 分离结构；严格 POV。 |
| PR-28 自博弈环境 | 未开始 | **中** | 依赖 arena/action parity；Mortal 只作对手基线。 |
| PR-29 保守 PPO | 未开始 | **中** | 只在 BC 与 arena 稳定后；behavior logprob 必须真实。 |
| PR-30 最终确认 | 未开始 | **高** | 主判据改为独立 duplicate arena；确认池需新建（2026 已 blocked）；不得用 BC loss/agreement 代替。 |

## 3. 已完成代码的可复用性

| 已完成模块 | 复用性 | 说明 |
|---|---|---|
| PR-00 preflight | **原样复用** | 只读环境/基线检查；与新架构无关。 |
| PR-01 benchmark | **原样复用** | 合成 SDPA 容量；PR-13/17 重测真实模型。 |
| PR-02 audit scan | **原样复用** | 文件/压缩/年份/zip CRC；不依赖模型。 |
| PR-03 canonicalize/dedup | **复用，需补全量** | 无损规范化与去重逻辑独立于架构；全量 index 仍 `full_corpus=false`。 |
| PR-04 replay adapter | **复用/扩展** | native action/mask 是权威；PR-10 需扩展为决策快照/censor，不要重写规则。 |
| PR-05 split | **复用** | game_id 级 split、L/R registry；补全量 canonical dedup 后可冻结新 hash。 |
| PR-06 player quality | **可选复用** | HMAC/时间前缀/隐私正确；但可靠性 INCONCLUSIVE，不能当已验证质量；v2.1 数据筛选需单独预注册。 |
| PR-07 drift（草稿） | **可复用（未完成）** | 事件聚合/CI 代码可继续；没有报告/运行，不计完成。 |

**没有可复用的模型/训练/checkpoint 代码**——因此 v2.1 架构调整不会作废任何 V2 权重；旧 Mortal
checkpoint 只作 baseline，不受影响。

## 4. 数据/接口变更点（高优先级）

1. **Schema 版本升级**：建议直接以 v2.1 为 PR-08 首版，避免“先实现旧 PR-08 再改一次”。
   schema 必须覆盖：事件字段/可见性、DecisionSnapshot、candidate descriptor、belief target、
   action map、normalizer、shard/checkpoint metadata。
2. **Candidate features 的来源**：候选动作只能由**POV 可见状态 + 合法动作**构造；禁止使用
   target 动作、未来摸牌、对手手牌、未公开宝牌。candidate 负样本的标签也要与 input 分离。
3. **Belief 标签**：待牌/风险的 target 必须来自未来终局/对手私有信息，只能进 label 分支；
   不要把 belief target 放进 snapshot/event tensor。
4. **Memory 可选**：数据仍按 game/kyoku 边界保存，但 MVP 不要求 summary；PR-13 可先 local-only。
5. **评估**：PR-30 主判据为独立 duplicate arena；PR-15 的 BC NLL/accuracy 仅诊断。
6. **权利**：任何 v2.1 训练仍受 `source_rights=unknown` 的 P0 阻塞；需人工审核。

## 5. 风险与成本

- **低风险**：已完成的数据审计/split/replay/quality 工具不依赖模型结构，可保留。
- **中风险**：直接实现旧 PR-08/09 会重复劳动；建议先定 v2.1 schema 再写 PR-08。
- **高风险**：candidate head + belief 会扩大 PR-08/10/11 的范围；若同时改数据/模型/RL，
  无法归因。建议单轴推进：先 local backbone + state encoder（PR-13），candidate/linear 消融，
  再 belief，最后 memory/RL。
- **不能做**：不得用旧 `patches/0006` event tokenizer；不得把 Mortal agreement/BC loss 当强度；
  不得用 2026 当 virgin confirm。

## 6. 建议（不执行）

1. 审核本报告后，决定是否采用 v2.1；若采用，先重写 PR-08 设计（schema 版本 v2.1），
   再实现 PR-09/10；不要先补旧 PR-08 再改。
2. 在 PR-08 之前补 PR-03/04 全量 canonical/replay（至少 100k 分层 + 全量 dedup）。
3. 在权利核实前保持 `train_allowed=false`，不启动 PR-13+。
4. 评估顺序：PR-13 local/state 小模型 → PR-16 arena smoke → PR-17 G3；candidate 与 belief 各做
   独立消融；memory 仅在消融显示收益后进入 PR-23。

**本文件不实现任何变更；审计结束。**
