# RiichiFormer V3 — 从审计状态到安定九段 / 天凤位级别的完整迭代路线

**版本**：v3.1（并行开发修订）/ 2026-11-21。**任务类别**：研发管理和实现方案，并非已完成的模型或保证能达到指定段位。**硬件约束**：一张 RTX 5070 Ti（16GB VRAM），训练机环境见审计 PR-01。**唯一规则权威**：Rust `libriichi` 对合法动作/牌局推进负责，但必须有天凤官方规则差异清单。

> 阅读次序：先读 `00_REVIEW_FINDINGS.md` 处理已发现问题；再按本文件**依赖门**而非 PR 数字盲目执行；执行具体 PR 前加载 `02_AGENT_EXECUTION_INSTRUCTIONS.md`；**并行派工先读 `05_PARALLEL_EXECUTION_PLAN.md`**，按其 Waves、硬依赖、资源和文件所有权执行；派工直接复制 `06_SUBAGENT_DISPATCH_PROMPTS.md`；实验按 `03_EVALUATION_PROTOCOL.md`。

## 0. 成功定义与边界

### M0：可复现工程（不等于实力）
无 POV 泄漏、合法动作标签正确、离线/在线完全一致、数据来源与 split 可追溯、结算统计正确、端到端稳定打完完整半庄。只有通过这些门，才有资格报告实力变化。

### M1：超过旧 Mortal（内部相对实力）
从已有 5 段上下的旧权重到更强的新策略，使用同规则、同 seed-set、四席旋转、独立 challenger vs champion 和多样对手池。对 Mortal 的胜率/均顺位仅证明相对优势，不代表真实安定段位。

### M2：安定九段“凤凰桌等效”的证据门槛
官方四麻凤凰桌安定段位为 `D=(60*p1+30*p2)/(10*p4)-2`。统计检验最好避免在 p4 很小时直接对比比率：定义 `A9 = 60*p1 + 30*p2 - 110*p4`，九段长期无亏损对应 `A9>=0`；使用**完整 seed-set block bootstrap** 对 `A9` 给出单侧 95% 下限，要求在与凤凰桌人类对手强度经过校准的**独立评测**中 `lower_CI(A9)>0`，并报告 `D` 的区间。若缺乏可靠校准对手，只能宣称“内部代理赛场达到某值”，不得宣称凤凰桌安定九段。

### M3：天凤位级别的证据门槛
十段东南战 `[+90,+45,0,-180]`；定义 `A10=60*p1+30*p2-120*p4`。除了 `lower_CI(A10)>0`，还要模拟官方十段 2000→4000 晋升、降至0失败的吸收随机游走，报告在假设稳定对局分布下的晋升概率和置信区间。对手必须与顶级凤凰桌强度经过独立标定。即使通过也只能称“天凤位级别实力的间接证据”，不是取得官方称号；官方禁止 AI 在凤凰桌实战，除非未来规则/许可改变并有明确授权。

### M4：长期研究路线
如果 M2/M3 未达标，不视为失败，而是从失分诊断中决定继续投入数据、价值训练、风险模型、搜索或更强对手。对单 GPU 不设毫无依据的时间/局数/成功保证。

## 1. 技术路线与模块分层

### 数据层（必须先于学习）
- `RawGame`：完整全知游戏日志，仅在受权限控制的 audit/target 生成器中出现。
- `POVEvent`：按决策时刻、玩家视角投影的公开事件 + 自己私有可见事件；对手摸牌为 unknown sentinel。
- `DecisionInput`：只含合法可见状态；`DecisionTarget`：专家动作、终局 rank、privileged belief 标注；两者必须是**不同类型/序列化 key**。模型 `forward()` 从类型层禁止接收 target。
- `RuleAction`：保留 46 动作映射 + `decision_phase`；在二次杠候选阶段，0..36 表示杠牌选择，非一般弃牌。
- `CandidateFeature`：由合法状态、候选动作本身经确定性规则/单机表计算；永不引用真值暗牌或训练目标。
- `EpisodeShard`：整场 game + 四 POV + 每局边界 + 决策事件 offset；不为每次决策重复存储整个前缀。

### MVP-S（先验证）
`Input events -> event embeddings -> causal Transformer(6×256, 4 heads) -> DECISION hidden`，当前时刻的 `StructuredState` 由 MLP 投影为 256 并注入 DECISION。`LinearPolicyHead(256,46)`，合法 mask 来自 Rust。只用 BC 开始训练，输出真实在线 MJAI 动作（无对手暗牌）。

### Candidate-S（独立受控实验）
完全固定相同 backbone、dataset、初始化/预算，与 LinearPolicyHead 比较：`logit_a = SharedMLP([h_decision, E(action_type), E(tile_kind), F_visible(state,a)])`。输出仍是 46 维 logits。开始只加**确保有依据的**候选特征，如动作类别/被弃牌/赤牌/是否立直/可知向听数与有效进张；不要把 oracle risk 或非法未来特征写入 Policy。

### Belief/Risk（独立受控实验）
由局内 `h_decision` 和候选牌输出 `(p_tenpai[3], p_wait[3,34], p_ron_possible[3,37], predicted_ron_loss[3,37])` 的适当子集。标签可以使用观测时刻的**全知游戏状态**生成，输入不能；`can_ron_now` 不等于 `opponent_would_ron`，多家荣和须按规则处理；先做概率校准与风险解释性检查，再做 `risk->policy` 显式融合实验。不要一次启用所有头。

### Rank-aware Value（独立受控实验）
终局 rank 分布 `[4]` + 可选顺位分 Value。BC+Rank 先与 BC-only 对照；RL 时使用明确的行为策略 logprob、优势与 value baseline；训练奖励与安定九段的评测口径分开记录。冻结旧 `[90,45,0,-135]` 作为旧 benchmark，但长期九段/十段口径分别计分，不允许在实验中途变更 reward 后直接比较旧指标。

### 可选 Cross-Kyoku Memory
先对照无历史 vs 简单历史统计（只读取已结束局），证明存在**条件于当前公开状态**的跨局收益，再训练 learned summary（每局全局 + 三家，或 1/4/8 tokens 消融）。训练时注意跨局 BPTT/摘要梯度；局末摘要绝不可流回本局之前的决策。不给 Memory 预留必需训练配额。

### 后期 RL 与 Planning
先对齐行为策略与训练 logprob、固定 value/环境奖励、搭建旧策略 pool/自博弈 league、做 PPO/V-trace 等合法的 policy update；再考虑小预算的候选行动搜索、privileged critic/oracle guiding、多教师。不得照搬此前 A14 的 mismatch。

## 2. PR 工作单（全部有产出、测试和过门条件）

### H-00：保护本地开发史与审计资产（立刻）
**依赖**：无；**文件**：`docs/riichiformer/v3/GIT_BASELINE.md`、`scripts/rf_git_safeguard.py`（只读优先）。
**实现**：记录 `git remote -v`、`branch -avv`、`status`、`rev-parse`、`ls-tree`、子模块版本与 patched diff/untracked；保留用户私有 `.omo` 文件；对照 `main=a3ccd...` 与审计分支 `f23bb...`，记录无共同祖先。开发主分支以真实本地工作树为准。要将成果发布到公开 `main`，用既有 `publish_public.sh` 在临时克隆中 *只新增经过审查的第一方文件*，禁止直接 merge 不相关历史。
**测试**：备份读回 hash；在干净临时工作树复现所需第一方代码目录；`.gitignore` 不再忽略 `riichiformer/data`；仓库 grep 无隐私、数据或 checkpoint 意外提交。
**验收**：保留未跟踪/dirty 内容、哈希有据可查、没有破坏旧评测数据；`GIT_BASELINE.md` 明确 Agent 将从哪一分支继续。

### H-01：修复 final score/kyotaku 一致性（立刻）
**依赖**：H-00；**文件**：`riichiformer/data/replay.py`、`tests/rf_test_replay.py`、`docs/riichiformer/v3/SCORE_AUDIT.md`。
**实现**：写 `replay_scores(events, ruleset_id)`：以 `start_kyoku.scores` 为锚；`reach_accepted` 扣1000、kyotaku+1；`hora.deltas` 的供托 inclusion 与各数据源协议明确建模，不自动叠加；`end_game` 时未兑现供托按规则分配；下一局 `start_kyoku.scores` 严格对照上一局结算；双荣、多荣、流局和点数同分单独 fixture。把 `final_scores_from_events` 改成单一权威入口，被质量分析等一致调用。`len(gameplays)==0` 的非正常 replay 显式失败；kyoku 数按事件边界统计。
**测试**：`pytest tests/rf_test_replay.py`；增加自动由 Rust arena 生成的随机 1,000+ 全局、覆盖所有结束类别的测试；真实多来源日志 100+ 个带供托样例，逐局分数一致性率必须 100% 或列明规则偏差并阻止使用。
**验收**：没有未解释的 score mismatch；修复后重算 PR-06 的历史 q 与质量可靠性报告，记录前后变化；未修复前禁用 `top25_seat` 训练。

### H-02：全量 canonical、SHA、去重和 split 强化
**依赖**：H-00；与 H-01 可并行（score 修复不影响 canonical bytes）；**文件**：`scripts/rf_build_canonical.py`、`riichiformer/data/dedup.py`、`riichiformer/data/split.py`、`tests/rf_test_canonicalize.py`、`tests/rf_test_split.py`。
**实现**：将 `--all` 改为按年份/分片流式处理；磁盘中间 manifest 包含 original/canonical hash、game_id、规则版本、来源；使用外部 hash join 或 SQLite/duckdb 等本机可用数据库（需记录依赖），形成 duplicate graph：同 game ID、同 canonical SHA 为边，连通组整体只能进入同一 split 或 quarantine；相同 ID 不同 canonical hash 标为 conflict，默认拒绝。分片幂等、断点续跑、只读牌谱目录。生成唯一 representative 列表，再重建 L/R registry 和 hashes，不得直接覆盖旧 registry，改为新版本文件。
**测试**：同一局不同文件名/压缩/来源、同 ID 内容冲突、跨 train-val 重复、哈希冲突逻辑（人为 fixture）、中断续跑输出 hash 相同；全量开始前用 10k、100k 确认内存峰值和吞吐。
**验收**：eligible 全量有 canonical hash、重叠为0、冲突全部 quarantine、新旧 split 差异表完整；绝不把 scan-only=0 bad 等同 full parse=0 invalid。

### H-03：使用权和数据来源审查（与 H-02 并行）
**依赖**：H-00；**文件**：`configs/riichiformer/source_rights.csv`、`docs/riichiformer/v3/SOURCE_RIGHTS_REVIEW.md`。
**实现**：逐一记录 Tenhou 官方规则、第三方转换源的权限链、数据存储用途、模型训练是否允许、再分发是否允许、隐私要求；用户/法律授权判定记录证据链接及有效日期。`unknown/restricted` 保持 fail-closed；如果没有训练许可，建立明确许可的替代数据、自建规则机器人、自博弈 bootstrap 管线，不能“先拿数据训练后补手续”。
**测试**：`rf_build_shards --source tenhou_old` 在 unknown 状态必定拒绝；不影响只读元数据审计；新来源未配置权限也拒绝。
**验收**：训练启动之前所选来源全部具备适用 scope 的 allowed 证据；否则只能进入不使用该来源的训练分支。

### H-04：独立规则差异审计与修复计划
**依赖**：H-00；**文件**：`docs/riichiformer/v3/TENHOU_RULE_PARITY.md`、`tests/rf_test_ruleset_parity.py`。
**实现**：比对 tenhou.net 官方东南战规则、当前 pinned Mortal 和任何替代环境。记录 triple-ron 流局、双荣结算、赤牌、杠宝、同分顺位、南四终止、西入、供托、亲家连庄、振听等；对于每个差异创建可重复模拟 fixture 和 expected score/continuation。若要修改上游，优先第一方独立规则适配或清晰 patch，不破坏旧 Mortal 可复现基线。
**测试**：固定牌墙/触发案例对照，随机牌局统计 trigger rate；无法完全复现的规则保持 `KNOWN_DIFFERENCE` 标签，不参与“严格天凤等效”声明。
**验收**：规则符合性矩阵完整；所有有实力影响的高频差异已经修复或明确隔离评价。

### PR-07R：完成年代漂移报告
**依赖**：H-00（H-02 的 canonical 完成后可更新数据质量统计）；**文件**：`riichiformer/data/drift.py`、`scripts/rf_drift.py`、`tests/rf_test_drift.py`、`DRIFT_REPORT.md`。
**实现**：保留已有 9 tests；跑真实分年/分阶段 drift 分析，至少包含动作类别、局长、副露/立直、对局终结与牌谱覆盖；按样本量给 CI，注明混杂因素：平台规则/玩家池/数据年份和 AI 污染来源未知。
**验收**：真实 CSV、报告、seed 和样本量；不把“近期=更强”当结论。

### PR-08 V3：冻结 Schema v3 和动作契约
**依赖**：H-00；**文件**：`riichiformer/schemas/event_v3.py`、`riichiformer/schemas/decision_v3.py`、`riichiformer/schemas/action_v3.py`、`configs/riichiformer/schema_v3.json`、`docs/riichiformer/FEATURE_SCHEMA_V3.md`。
**实现**：定义 `ObservableEventV3(kind, actor_rel, target_rel, tile_kind[0..33 or unknown], aka, consumed[4], consumed_aka[4], tsumogiri, seq_idx, kyoku_idx, timing)`；`RuleStateInput` 含 34 牌种 counts + `own_aka[3]` + `public_aka[3]` + `public_counts_by_zone`（同物理牌只计一次）+四家状态+可见宝牌+局况。`DecisionInput`、`DecisionTarget` 类型分离；后者承载 `label`, `censor_reason`, `rank`, `belief_oracle` 且不可序列化进 forward input。`ActionPhase` 显式 `SELF_TURN`, `CALL_RESPONSE`, `KAN_SELECTION`，冻结 46 mapping 和 hash。开放 `candidate_schema_version` 和 `belief_label_schema_version` 但先不生产昂贵标签。
**测试**：每类 mjai 事件 roundtrip；0..36 弃牌和杠候选双重语义；38..40 吃结构；赤/普通五独立；未知类型 fail closed；schema/normalizer/action hash 与 checkpoint/shard 加载互认。
**验收**：`FEATURE_SCHEMA_V3.md`、schema hash、action hash 固定；任何修改必须升级版本并重建相关 shard。

### PR-09 V3：逐 POV 的可观测事件投影
**依赖**：PR-08；**文件**：`riichiformer/data/pov.py`、`tests/rf_test_pov.py`、`docs/riichiformer/v3/VISIBILITY_TABLE.md`。
**实现**：先编写事件类型 allowlist；`start_kyoku` 仅自家13张，其他牌种恒 unknown；`tsumo` 对手牌种/aka 清空，自摸保留；弃牌的 `tsumogiri`、牌种/aka、所有公开副露信息保留；未公开里宝牌不能提前出现在状态；对暗杠牌张展示遵守规则源实际可见细节，不能盲目复制全知 `consumed`；结束事件只有在结束时间点以后开放。去掉真实姓名/牌山 seed 和未来结果。离线/在线使用同一投影。
**测试**：四 POV 对称，隐藏手扰动字节相同，合法自己手牌扰动可见，未来事件截断等价，aka 公开牌迁移不重复，杠/里宝切换时序，已完成局对后续局可见。
**验收**：100% property tests 通过；任何未知事件 type 报错；已知旧 0006 opponent `tsumo.pai` 泄漏专门回归测试通过。

### PR-10 V3：原生重放驱动 Decision Snapshot、censor 与公开牌簿记
**依赖**：H-01、PR-08/09；**文件**：`riichiformer/data/decision.py`、`riichiformer/data/tile_counter.py`、`tests/rf_test_decision.py`、`tests/rf_test_tile_counter.py`。
**实现**：以 Rust `PlayerState` 为规则权威，每次 action opportunity 生成 `DecisionInput` 与独立 `DecisionTarget`。保留事件 offset 到行动标签的对应，优先使用原生合法 mask。区别主动 `Pass` 与被其他人优先动作抢断的 censored 反应；未知/含糊样本保留 trajectory 但设 `supervision_valid=false`。对九种九牌/自摸/荣和/摸切与手切，记录 phase、actor、触发牌。维护物理牌源 `own_hand/river/meld/indicator/other_public` 的去重 ledger：被吃碰的牌由河迁移到副露，不能同时计入两处。
**测试**：每类动作有独立 fixture；多候选暗杠/加杠、立直宣告+后续弃牌、双荣、三荣、被抢断 Pass；手牌计数守恒、赤牌数量不超过 rule config、公开去重；所有有效监督标签在 mask 内。
**验收**：100k 分层样本非法动作 0；有明确 censor 比例和缺失原因分布；仅输入端不含任何未来 `target_rank`。

### PR-11 V3：单一离线/在线编码器与因果断言
**依赖**：PR-10；**文件**：`riichiformer/data/encode.py`、`riichiformer/runtime/encoder.py`、`tests/rf_test_encoder_parity.py`、`tests/rf_test_causal.py`。
**实现**：以同一纯函数编码 event/state/candidate，不分别造训练 token 和 arena token；事件字段 embedding IDs、unknown、padding、相对座位、赤牌、局内位置、全局 seq、局界。`DECISION` token 只能看到自身及其前缀；每局 reset Local，半庄 reset Memory；valid keys 随 schema hash 保存。
**测试**：随机前缀 offline vs arena byte equal；截断后在前面 decision 的 logits/embeddings 不变（在 dropout=0 eval 下）；未来自摸改变不影响过去；反常 L 的 padding 不影响有效输出；全 POV 各 seat 一致。
**验收**：对抗性 leakage 用例 100% PASS，任一 mismatch 阻塞 PR-12/13。

### PR-12 V3：可重放、可恢复的 Sequence Shard 与 DataLoader
**依赖**：H-02/03、PR-11；**文件**：`riichiformer/data/shard.py`、`riichiformer/data/loader.py`、`scripts/rf_build_shards.py`、`tests/rf_test_shards.py`。
**实现**：按 `game_id→four POV→kyoku→events+decision_offsets` 存储，不重复 prefix；标签单独分区或键下并通过类型隔离；保存 native source、schema/action/normalizer/canonical/split hash；按历史长度分桶、动态 padding、长度采样，避免数据读写成为瓶颈；按 game_id 冻结 split，别混 4 POV；支持流式生成和中断恢复。`source_rights` 未被允许时 refuse train shard。
**测试**：shard roundtrip；同游戏四 POV 不跨 split；同一个决策可从原始牌谱再生；1%/10%/100% 样本 hash 匹配；变 schema 或 source 立即 fail closed。
**验收**：训练 sample 的全部有效决策 label∈mask、数据溯源完整，异常按 game_id quarantine；测量磁盘占用与 data-loader throughput。

### PR-13 V3：Local Transformer MVP-S
**依赖**：PR-11（可先只用 synthetic fixture 验证网络 forward）、PR-12（真实训练前）；**文件**：`riichiformer/models/event_embedding.py`、`state_encoder.py`、`local_transformer.py`、`policy.py`、`configs/riichiformer/mvp_s.yaml`、`tests/rf_test_models.py`。
**实现**：`state = MLP(34 tile features + 4 player features + global)->256`，事件 type/actor/tile/aka/tsumogiri/phase embedding 聚合为 `d=256`；6 个 Pre-RMSNorm causal block、4 heads、RoPE、SwiGLU，决策 token 注入 state；`linear_policy -> 46 logits` + legal mask；只输出模型真实可见输入。接口 `forward(batch: EncodedPOVGame) -> PolicyOutput(logits, selected_decision_offsets)`；不默认加入四 Player Query 或 Cross-Kyoku Memory。
**测试**：batch shape、不合法动作概率=0、因果 prefix 等价、不同 POV 不串、所有梯度 finite、独立模型 state_dict 可严格加载、参数量与显存实测。至少在小 synthetic 数据上过拟合有限动作以证明训练链能学。
**验收**：离线 forward/backward 稳定；单 GPU 16GB 无隐性溢出；`parameters`, `tokens/s`, `decisions/s`, `peak_vram` 入报告。

### PR-14 V3：BC Trainer + checkpoint 协议
**依赖**：PR-12/13、H-03 放行训练来源；**文件**：`riichiformer/train/bc.py`、`train/optimizer.py`、`train/checkpoint.py`、`scripts/rf_train.py`、`tests/rf_test_training.py`。
**实现**：`loss=-sum(valid_decision_weights*logprob[expert])/sum(valid_decision_weights)`；censored/未知标签不计；强制动作可作为上下文但可不计 policy loss；每局多个决策一起反传（不重复每决策的整段前缀）。AdamW、BF16、SDPA、长度分桶、梯度累积、梯度裁剪、LR warmup，所有 loss/NaN 检测。checkpoint 保留完整 optimizer/scheduler/RNG/scaler 或 BF16 状态、schema/action/split/source hash、训练 token/decision step、当前权重与 EMA（如启用）。自动保存 immutable pre-reg seed/config。
**测试**：两次同种子小样本轨迹一致；从中断 checkpoint 恢复的 optimizer/next batch 语义正确；label 污染检查；训练只接受来源允许的 shard。
**验收**：训练稳定+可恢复，样本数与有效决策数可核对，没有把 rank/oracle/未来信息送入模型输入。

### PR-15 V3：离线诊断和难点分布
**依赖**：PR-14；**文件**：`riichiformer/eval/offline.py`、`scripts/rf_offline_eval.py`、`docs/riichiformer/v3/OFFLINE_DIAGNOSTICS.md`。
**实现**：NLL、Top1/Top3、按 action phase、立直/碰杠、对手威胁、南四、分数差/向听、强制动作、不同年代分层；另外各模块计算预算和 logits 分布；禁止只用总体 BC accuracy 排名。数据加权后保留原分布无权重测量，避免伪进步。
**测试**：每类分桶 synthetic + 样本真实计数；`assert sum(per_category)=total`（互斥维度）或显式报告重叠。
**验收**：报告覆盖哪些决策真正变好/变坏，冻结选择指标，不宣称 NLL 降低等于天凤段位提高。

### PR-16 V3：Arena 推理适配器 + 规则一致性
**依赖**：PR-11/13、H-04；**文件**：`riichiformer/runtime/agent.py`、`runtime/mjai_adapter.py`、`scripts/rf_eval.py`、`tests/rf_test_arena.py`。
**实现**：支持独立 RiichiFormer checkpoint 和固定 Mortal 1v3；每局合法信息流，46 维 action decode 与 Rust 一致；多杠候选 stage 支持二次索引；和牌 guard 规则一致；对手摸牌永不进入 token；每局清理 Local KV 但完整半庄上下文按模块管理。4 席座位轮换、种子 key/范围和输出目录由显式配置提供，禁止默认写冻结窗口。
**测试**：固定 10/100/1000 半庄逐级 smoke（数量非最终统计功效保证），无非法行动、无 None 死循环、无截断静默、无 CPU/GPU 非确定性导致的重复不一致；官方规则差异被明确标注。
**验收**：可连续打完对局、每局有完整可信 rank 分布，独立报告 `relative strength only`。

### PR-17 V3：G3 端到端基线门
**依赖**：PR-13..16；**文件**：`docs/riichiformer/pr_reports/PR-17.md`、`runs/exp/RF3_MVP_S/PREREG.md`、`REPORT.md`。
**实现**：冻结一版 MVP-S（非最终最强），记参数量、训练 compute、合规样本量、评测窗口；与随机/规则 bot、旧 Mortal 对照，定位主要弱点。此门只要求模型可复现、合法、评测稳定，不要求直接超越 Mortal。
**验收**：G3 pass -> 可进入架构与数据 A/B；若不稳定优先修复数据或优化器，而不是直接放大模型。

### PR-18/19/20 V3：数据质量、近期数据、加权采样 A/B
**依赖**：H-01/02/03、PR-17、PR-07；**文件**：`data/sample_weights.py`、`docs/riichiformer/v3/DATA_ABLATION_PREREG.md`、实验 configs。
**实现**：D0 全部获授权来源、原始分布；D1 多级历史玩家 q 只作为候选（需先验证时间前缀/分数）；D2 按决策类型适度重加权，专门关注 push/fold、立直、南四和非强制动作；D3 仅加入核实可用的近期数据。比较时严格控制 unique game count / effective decision count / optimizer tokens / training budget，每次只改一轴；避免按本局赢家/终局分数直接筛选训练轨迹。对近期数据按规则变化、对手强度、AI 来源 `confirmed/unknown` 区分，不猜封号人数。
**测试**：采样器 reproducibility；sample hash；无 select/confirm 泄漏；不经训练来源许可的样本不得入批；抽样权重/分组覆盖有统计报表。
**验收**：必须在独立 arena 看到可重复增益才提升为主数据策略；现有 PR-06 的 `top25` 可靠性不足，不能预设最好。

### PR-31 NEW：Candidate-based Policy Head A/B
**依赖**：PR-17（或 PR-13/14 已稳定即可开始 A/B）；**文件**：`models/candidate_head.py`、`data/candidate_features.py`、`tests/rf_test_candidate_head.py`。
**实现**：输出 46 logits，但为每个 legal action 单独打分。状态 `h_decision` + 可见 `tile_kind` embedding + red flag + phase + legality/向听等规则合法特征；与线性头共享相同 backbone 结构、初始化 seed、数据和预算。注意在 `KAN_SELECTION` 0..36 是杠牌选择，不是普通弃牌；`PASS` 不允许获取对手私有信息。不要在第一版硬塞理论风险分数。
**测试**：候选顺序置换协变、合法 mask 一致、赤/普通五产生不同候选、单候选 head 特殊动作正确；输入结构不依赖 label。
**验收**：若只改善 NLL 不改善 arena，则保留最简单线性头；提升必须有 effect/CI 与算力延迟对比。

### PR-32 NEW：Belief/Risk Oracle 标签生成器（独立隔离）
**依赖**：PR-10、H-04、H-03（对数据源的可计算标注需有许可）；**文件**：`riichiformer/targets/oracle_risk.py`、`tests/rf_test_risk_targets.py`、`docs/riichiformer/v3/RISK_TARGET_DEFINITION.md`。
**实现**：离线重放到每个决策时刻，构造 opponents concealed state for labels only；对每个 legal discard candidate 检查按该瞬间规则是否可合法 ron，区分 `in_tenpai`, `wait_kind`, `can_ron_now`, `ron_score_if_accept`, `unknown`，区分已振听/无役/同巡/该牌赤属性等；`can_ron_now` 是能力，不是主动调用概率。多家可能荣和时不能把互斥与有序结算简单相加；要由模拟器求结算分支。仅保存目标分区，永不进入 `DecisionInput`，并把 oracle 防护纳入 runtime 打包/导出测试。
**测试**：当前 hand known target 与规则引擎逐项匹配；同样 input 扰动隐藏对手手，input 不变、target 可以变；多荣/赤牌/振听/未公开宝牌 fixture；oracle head 不可 import 到在线 encoder。
**验收**：无 label→input 泄漏；各目标覆盖率、unknown 比例和规则不适用原因报告完整。

### PR-27 V3：风险模型训练与概率校准
**依赖**：PR-32、PR-17；**文件**：`models/belief_heads.py`、`train/belief.py`、`eval/risk_calibration.py`。
**实现**：先 `tenpai[3]` 单独 BCE；在合规数据上扩为 `wait[3,34]` 多标签；再做 `can_ron[3,37]` 与点数损失条件回归。分头记录 NLL、Brier、ECE、按巡目/玩家/立直/副露划分校准，不要用 AUROC 单独作为强度证明。先冻结 BC 主干训练 head，后做 joint fine-tune 防辅助任务压垮策略。
**测试**：被遮蔽的风险 target 不会影响 policy input；高置信标签可复查；正负样本比例记录；防止把未来和牌结果当作当时手牌。
**验收**：出现可复现的 calibration 增益；若 risk 只准但 policy 无增益，仍作为诊断分支，不自动合并。

### PR-33 NEW：Risk-aware / Rank-aware Policy A/B
**依赖**：PR-27、PR-31（可选）；**文件**：`models/risk_fusion.py`、`models/rank_head.py`、`eval/decision_strata.py`。
**实现**：A 纯 policy；B 辅助 risk loss 不在 policy 中显式接线；C 在候选 logits 旁加入预测的风险/期望损失（注意训练推理都只能使用 model prediction，不能用 oracle target）并重新学习融合；D 加终局 rank distribution `[4]` 或 9/10 段顺位分 Value。各 arm 固定 backbone/数据/训练预算；比较押退、立直、南四、4位率和终局 rank。加入 Value 时训练目标严格来自比赛最终结果，仅作为 label。
**测试**：强制把 risk prediction 清零/打乱应改变融合结果但不可改变原始输入；独立 CI；更换 reward 公式时 checkpoint/metrics 明确版本。
**验收**：至少一个对关键决策有明显提升、整体 arena 不退化的 arm 才能进入候选 champion。

### PR-25A NEW：网络容量与 State Encoder 消融
**依赖**：PR-17；**文件**：`configs/riichiformer/architecture_ablations/*.yaml`、`ARCH_REPORT.md`。
**实现**：固定数据和计算配额，比较 `MLP state`、轻量花色共享 1D block、两层 tile-set attention；Local 6×256、8×384（以及预算允许的小型控制）；报告实际参数数、训练 FLOPs/tokens、显存、吞吐、CI；只选 1 个进入下一轮，不因参数更大盲目通过。
**验收**：模型选择基于同条件实力收益和成本，不以视觉上“更像 Transformer”为标准。

### PR-21 V3：弱 Mortal 蒸馏（非默认）
**依赖**：PR-17、来源/权重许可核实；**文件**：`train/distill.py`、`DISTILL_PREREG.md`。
**实现**：Mortal V4 只能作为弱/偏差教师。D0 无教师，D1 小权重软 logits/KL，D2 仅校准高置信关键局面教师引导，全部比较原始人类 BC +独立 arena。若教师自身的策略在关键押退/立直有系统偏差，学生不得因蒸馏回归旧能力上限。旧 Q 值须先对 action_mask、动作时点和温度做校准。
**验收**：缺许可或没独立增益直接 skip/kill；不能把 teacher agreement 当最终成绩。

### PR-22/23/24/26 V3：历史长度与 Memory 研究支线（选做）
**依赖**：PR-17 + 有证据的历史价值诊断；**文件**：`models/memory.py`、`train/memory.py`、`eval/history_ablations.py`。
**实现**：先 Local-only；再已结束局的手工统计（局数、立直、鸣牌、押退等时点条件指标）；再 learned summary 每局 4 tokens、1/8 对照；再 Flat-F 全半庄 causal control。所有比赛在**当前手牌、当前局、当前点数一致**的前提下，对比真实历史/置零历史/受控错配历史，不能改变游戏本身导致信息不等价。摘要需要可训练的梯度路径：2–4 局 TBPTT 或局末辅助监督；无过去局的状态以 no-memory special case 处理。
**验收**：在独立 seed-set 里有可信净收益且推理/训练成本合理才保留 Memory；失败就用 Local-only，不影响其他主线。

### PR-28 V3：合法的自博弈环境、策略池与生产能力
**依赖**：PR-16、H-04、PR-17；**文件**：`riichiformer/rl/rollout.py`、`rl/league.py`、`rl/trajectory_schema.py`、`tests/rf_test_rl_env.py`。
**实现**：对每条 trajectory 记录 `policy_version`, state hash, legal_mask, action, exact behavior_logprob, sampler hyperparams, score/rank outcome, ruleset_hash, seed`; opponent pool 包括合法可用的不同旧版本、规则/搜索 bot、当前 champion、近期策略快照；禁止在线天凤无授权 AI 对局。要先测环境 games/sec、actor/learner 吞吐与 GPU 使用率，否则 RL 研究可能被 CPU 环境瓶颈拖垮。
**测试**：10k+ 独立可复现对局无非法动作和 fatal error；随机墙、四席对称；天凤已知规则差异被修复或显式标记；behavior logprob 与 rollout 采样频率匹配；没有评测 confirm seed 进入训练。
**验收**：日志可逐动作回放并一致复算概率和 rank；在 5070Ti 上可稳定采集轨迹。

### PR-29 V3：保守 RL（可用 PPO/V-trace，但机制正确优先）
**依赖**：PR-28、PR-17（优先选择当前强 champion）；**文件**：`rl/update.py`、`rl/advantage.py`、`tests/rf_test_policy_gradient.py`。
**实现**：指定 `pi_behavior`，严格记录 logprob；`ratio=exp(logpi_new-logpi_behavior)`，支持 masked entropy/KL/clipping、value bootstrap、rank reward；若 epsilon-greedy/Boltzmann 混合采样，必须使用**混合策略真实概率**而非错误 `log_softmax(Q)`；在小型可算 toy MDP 上对比手工目标。采用固定版本的 frozen opponent pool，定期独立 arena 评测。每次 RL 可回滚原有 checkpoint；过度训练或 policy drift 触发 kill。初步可对照七/九/十段奖励，不得一次改多种变量。
**测试**：小规模策略梯度有限差分/手工概率；停止训练后完全可复现；on/off policy 更新分布被报告；advantage mask 和 forced/censored 处理正确。
**验收**：强度在独立 select 测试中可靠提升且未引入四位率灾难；旧 A14 式损失风险明确消除。

### PR-34 NEW：强对手池与实力标定（贯穿前中后期）
**依赖**：H-04、PR-16，涉及人类/平台数据的来源另受 H-03；**文件**：`eval/opponent_registry.py`、`eval/anchor_calibration.py`、`docs/riichiformer/v3/ANCHOR_EVIDENCE.md`。
**实现**：至少分三个对手层：L1 已复现 Mortal（低强度工程对照）、L2 强且独立的获许可模型/规则搜索 bots、L3 与凤凰桌强度有独立可信证据的强锚点（可用授权人类专家的完整在线赛事或其他允许的对战协议）。不因 L2 对 L1 赢很多就默认 L2=九段。报告牌局规则差异、评测延迟、随机种子、对手策略版本、可能的对手训练污染；如 L3 不可用，评价只能称内部相对 Elo/rank。
**测试**：anchors 之间互相对战、座位轮换、稳定性/跨规则差异、与真实人类关联证据来源；不能通过同一个模型自己给自己“九段标定”。
**验收**：只有当 L3 信度充分、对局环境规则一致，才允许使用“凤凰桌等效安定段位”标签。

### PR-30/35 NEW：预注册最终九段检验、候选 champion 冻结
**依赖**：PR-17+、PR-34、各成功 arm；**文件**：`eval/final_confirm.py`、`runs/exp/RF3_CONFIRM/PREREG.md`、`REPORT.md`、`EVAL_POOL_REGISTRY.md`。
**实现**：将通过 select 的唯一 model checkpoint、schema、规则、对手池、测试原种子池、最终样本量/停止法、primary statistic `A9` 或 `A10` 等预先冻结。使用与训练/选择完全隔离的新 seed_key/seed range，四座轮换一个 set。以 seed-set block bootstrap 报告 p1..p4、D、A9/A10 区间、均顺位、推理延迟、规则差异和可复现源码哈希。批量评测与 confirm 验证不得读取任何未来样本的结果进行再优化。
**测试**：确认池未消费/隔离；完整 set 比例、缺失规则预注册；score parser 与另一个独立实现交叉验证；固定 seeds 重跑单个 set 完全一致。
**验收**：强度未经校准不得宣称九段；若 A9 的下限不过 0，明确未达目标并进入误差诊断；confirm 消费后不可复用作新的独立确认。

### PR-36：天凤位级别持续改进（长期课题，不设必成期限）
**依赖**：PR-35 完成九段目标，仍可并行探索潜力；**实现**：针对九→十以上差距做真实高强度问题诊断：押退、立直、南四损益、风险估计、搜牌价值、规则差异、数据 shift；按单轴实验继续优化；在有完整前置与概率校准时尝试 belief-constrained hidden-state sampling + limited search、privileged critic、oracle-guided RL、league 自博弈。每个技术都以同一规则/强对手/新 select pool 判定，目标 `A10>0` 的独立下界、十段晋升概率模拟与更强人类锚点证据。若官方未来有明确新的 AI 专用许可，再单独研究官方认证可行性；禁止将平台封号规避写成技术需求。
**验收**：只报告真实支持的强度级别和不确定性，不承诺一张 GPU 必定实现天凤位。

## 3. 分层并行执行 DAG（不要按 PR 编号串行）

> **以 `05_PARALLEL_EXECUTION_PLAN.md` 的文字硬依赖/接口所有权和 Waves 为权威**。本节是概要；可提前开发 mock/synthetic 但不能提前消费真实产物。Dev 可并行编写，单张 GPU 上正式训练及对照实验由 Lead 排队调度。

```text
W0 必须独占：H-00 (真实工作树/备份/共同开发 SHA/并行 worktrees)
      │
      ├─ Wave 1A：H-01 scores/replay ───────────────────┐
      ├─ Wave 1B：H-02 canonical/dedup/split ────────────┤ 互不覆盖的源码可并行
      ├─ Wave 1C：H-03 source rights ────────────────────┤ (PR-07R 由空闲 Agent 接力)
      ├─ Wave 1D：H-04 Tenhou rule parity ──────────────┘
      └─ PR-08 schema draft → frozen PR-08 → PR-09 POV → PR-10 snapshot → PR-11 parity → PR-12 shards
                              └─ [PR-13 synthetic model 可以提前开发]

         H-02 full dedup + PR-11 parity + source gate ─→ PR-12 real dataset
                      PR-12 + PR-13 + source gate ─────→ PR-14 BC trainer
                      PR-11 + PR-13 + H-04 ────────────→ PR-16 arena
                      PR-14 + PR-15 + PR-16 ───────────→ PR-17 MVP integrated gate
                                                                    │
           ┌─ PR-18/19/20 data A/B       [各支线开发可并行，GPU实验顺序执行]
           ├─ PR-31 Candidate / PR-25A Capacity A/B
           ├─ PR-32 oracle targets → PR-27 risk → PR-33 risk/rank A/B
           ├─ PR-28 rollout / PR-34 opponents → PR-29 RL
           ├─ PR-21 weak teacher (许可后选做)
           └─ PR-22/23/24/26 history memory (价值证据出现后选做)
                                                                    │
          approved champion + calibrated opponents + one-shot prereg/seed gate
                                                                    ↓
                              PR-30/35 final confirmation → PR-36 research
```

### 可并行与必须串行的关键区别

| 可并行开发 | 不能越过的串行/安全门 |
|---|---|
| H-01 结算、H-02 去重、H-03 权利、H-04 规则审计 | H-00 在前；不要两个 Agent 同时写 `replay.py` / `schema` |
| PR-07R 漂移与 PR-08 草稿 | PR-08 必须冻结和验收后 PR-09 才能消费 |
| PR-13 合成模型、PR-34 评测框架、PR-10 专项 fixtures | PR-09→PR-10→PR-11 的真实数据语义和 parity 要顺序验收 |
| PR-14 训练器、PR-16 Arena、PR-15 诊断模块 | PR-17 真正集成验收需权利、POV、数据、规则/模型链路齐全 |
| Candidate、Risk、数据采样、Memory、RL 环境各支线代码 | 正式 A/B 只改一个变量；GPU 独占，select 可复用但 confirm 仅一轮 |

推荐同时 **最多 4 个写代码子代理 + 1 Reviewer + 1 Lead**。每位在独立 worktree 从 H-00 approved BASE_SHA 开始；共享资源锁和合并由 Lead 负责。已核查公开 `main` 与审计分支无共同祖先，不能直接 merge。每个 PR 的执行指令与第一轮四张任务卡见 `06_SUBAGENT_DISPATCH_PROMPTS.md`。

## 4. 管理原则（防止项目无限膨胀）

1. **单轴实验**：每个 A/B 只改一个独立变量；保持 checkpoint 起点、训练源、相同 unique-games/有效 decisions/训练 tokens、座位与 seed-set 共同设置。
2. **先信息契约再网络复杂度**：P0 导致任何后续模型实验无效。网络性能争论不能跳过数据、规则、权限。
3. **不能拿五段模型充当九段教师**：Mortal 可加速冷启动但永远不是终局标签标准。
4. **定时冻结、事前门槛**：预注册成功/失败/无效阈值与样本大小；不可看到结果之后重写指标。
5. **杀死无效方向**：Candidate、Belief、Memory、教师、RL 必须在独立训练/对局中证明边际价值，否则保留代码作为研究 artifact，不进入下一 champion。
6. **公私分离**：Git 仓库只发布代码/公开可分享的元数据与小型证据；牌谱、私密玩家标识、权重按法律/授权与项目策略独立存储。
7. **不隐匿 blockers**：`SKIP`、`NOT_IMPLEMENTED`、`INCONCLUSIVE`、`BLOCKED_RIGHTS` 四种状态不同，不能用 PASS 代表所有通过。
