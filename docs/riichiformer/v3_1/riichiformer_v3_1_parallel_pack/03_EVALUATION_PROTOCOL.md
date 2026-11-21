# RiichiFormer V3 — 实力评测、统计与段位声明协议

## 0. 绝对不能混淆的四种指标

1. **正确性**：POV、防泄漏、合法动作、分数结算、在线等价。正确性是 0/1 硬门，不是强度。
2. **离线预测**：BC NLL/Top1、对手听牌和待牌校准、风险回归误差。它们不是胜率，也不是天凤安定段位。
3. **内部相对对战**：在当前（可能存在规则偏差）的 arena，与 Mortal v4 /规则 bot /经授权其他模型等的差异，属于 relative strength，不得自动转换成凤凰桌段位。
4. **凤凰桌等效实力**：只有在对手强度与人类凤凰桌池经过独立可信标定、对局规则与统计协议相容且确认池未被消费时，才可谨慎报告具有误差区间的“等效安定段位”。

## 1. 十段制度和最终目标（2026-11-21 核对天凤官方）

官方： https://tenhou.net/man/

四麻凤凰桌：

```text
      +90 p1 +45 p2 +0 p3 - (15*(d+2)) p4
E_d = ------------------------------------------------ 
                       1 game
```

其中 `d=7 -> -135`, `d=9 -> -165`, `d=10 -> -180`。设 `p_k = N_k / N`。

安定段位（凤桌四麻）：

```text
D_stable = (60 p1 + 30 p2) / (10 p4) - 2
A9       = 60 p1 + 30 p2 - 110 p4
A10      = 60 p1 + 30 p2 - 120 p4
E9       = 1.5 * A9
E10      = 1.5 * A10
```

针对 D_stable 的 `p4` 小样本分母不稳定问题，主要统计门使用线性 `A9/A10`，配合完整顺位分布及 D 的 bootstrap 区间。目标 M2 为在**锚点合法**的 confirm 中 `lower_one_sided_95_CI(A9)>0`。M3 则在更强的同类确认中 `lower_CI(A10)>0`，另报告达到真实十段 pt 晋升阈值的模拟概率与假设范围。只达到 A10>0 点估计并不保证获得天凤位。

天凤官方 AI 规则：不开放新的 AI 专用 ID，禁止 AI 在凤凰桌对战；因此默认不尝试通过自动化上天凤拿真实称号，目标是公平合法的实力等效研究。

## 2. 主对战协议

**四座一组**：同一个 `(seed, seed_key)` 构成 4 split（挑战者分别坐东/南/西/北），四局为一个 set。该 set 是统计上的成组单位，而不是四个独立样本。

主数据保存：`eval_id, ruleset_hash, challenger_hash, opponent_pool_version, seed_key, seed, split, physical_seat, game_id, final_scores[4], rank, eval_latency, status, full_log_hash, parser_hash`。不完整 set 的处理在预注册中明确：主确认统计默认只收完整 set，残缺报告应另列；不可静默丢掉不利局。

比较 A/B：当可行，使用同一批 seed-set 分别跑对手配置完全相同的 A 与 B，计算 **set 级 paired differences**，用 block bootstrap/paired randomization test。`B=2000` 是旧实现的自举次数，不是 2,000 个对局 set；最终高精度确认可增加重采样次数，但无法用更多 resamples 替代真实样本量。

**阶段性评测不是固定统计功效承诺**：
- smoke 40–1,000 局只证运行合法/延迟；
- screening 4k–12k 局用于排除大幅退化、发现方向；
- intermediate 12k–40k 局用于选 1 个 champion；
- final 由 pilot 的 paired set variance、预注册 MDE 与 power 算法决定，可能需要 100k 甚至更多。不能因 CI 很宽就宣称成功/失败，也不能无限看结果直到显著。

## 3. 三等级对手池

| 层 | 例子 | 能支持什么结论 |
|---|---|---|
| L1 | 旧 Mortal checkpoint、规则 bot | 工程正确、相对强度和旧 benchmark 对比 |
| L2 | 来源权利明确、训练独立的较强 bot、多元 league | 横向强度、策略稳健性，**不自动支持九段** |
| L3 | 有独立合法证据联系到现代凤凰桌人类高段强度的锚点 | 在规则一致/无评测污染条件下支持“凤凰桌等效”推断 |

L3 如何落地：经明确许可的人类强者/赛事对战，或已有独立、可复核、实战成绩且允许接入的顶级模型；至少包括不止一个策略谱系，以避免风格克制和训练谱系重叠。若无法找到 L3，项目只能报告内部相对等级，不能绕过该限制。

注意 `1v3` 与与人类四家随机混桌不同；模型同时面对三份同一 bot 更可能学到该 bot 的风格。最终应有更接近目标人群的混合对手池、规则统一的多谱系对手、座位轮换和跨环境交叉测试。

## 4. 三层数据测试池和 eval registry

1. **train**：用于学习任何权重、采样器、风险模型、奖励预测器、环境模型；包含所有输入和标签已合规的数据。
2. **select**：用于多臂模型选择，允许反复看，但每次试验必须登记、保留结果并校正多重比较/选择偏差。
3. **confirm**：独立一次性、预先冻结，使用新的未消费 `seed_key` 和 seeds。之前 pool A/B、A14、M3、EX1 不能重命名为 virgin；2026-01 历史数据曾被旧研究使用，也不能叫 virgin temporal holdout。

每个 confirm 使用前必须在 `EVAL_POOL_REGISTRY.md` 追加预注册 hash，结束后 append consumption row；任何实际落盘的 confirm 数据不能再次作为新确认池。可用作后续诊断但必须标成 consumed。

## 5. 公共诊断指标和错误归因

全局：`p1,p2,p3,p4,avg_rank,A9,A10,D_stable,rank_pts7/9/10,win_rate,deal_in_rate,ron_loss,scores_by_phase,model_latency_p50/p95/p99`。

难例：
- 押退：对手立直前后，按其对局威胁/放铳风险分类（目标检测逻辑需预先冻结）；
- 立直：是否立直、放弃立直、立直后局面与顺位收益；
- 吃碰杠：动作者有机会却 Pass（censored 必须排除监督）；
- 南三/南四：当前顺位、点数差和目标跨越距离；
- 赤牌：普通/赤五区分、已经公开的赤牌剩余数、宝牌影响；
- 牌效率：不同向听数和巡目，但不得使用对手真实隐藏牌计算输入；
- 规则异常：三家和了、双荣、南四结束、西入、供托残留；
- 模型与训练：policy KL/entropy，behavior logprob mismatch，belief calibration，GPU throughput。

## 6. 统计报告模版

```yaml
eval_id: RF3_EVAL_EXPERIMENT_X
purpose: select  # smoke / select / confirm
model_sha256: REQUIRED
schema_hash: REQUIRED
action_map_hash: REQUIRED
ruleset_hash: REQUIRED
opponent_pool_hash: REQUIRED
seed_key: REQUIRED
seed_range: [start, end_exclusive]
rotation: four_seat_per_set
primary_metric: paired_delta_E10  # or A9 for threshold
secondary_metrics: [p1,p2,p3,p4,avg_rank,A9,A10,D_stable,p95_latency]
confidence_method: paired_set_block_bootstrap
resamples: 10000
missing_set_policy: pre_registered
sample_size_justification: pilot_variance_and_MDE
max_looks: 1  # unless valid sequential testing protocol preregistered
promotion_gate: predefined_lower_confidence_bound
```

**统计规则**：所有 rank 来自原始最终分数，打平按规则座位 tie-break；决策过程不允许使用测试集的真实隐藏状态；对照与新模型使用同一 parser、同一规则和相同 opponents。只报告成功的样本/模型违反预注册。

## 7. 达标报告用语

- 无 L3 校准：`在本项目规则版本 X、对手池 Y 的 duplicate 赛场，A9=...，不能推出实际凤凰桌段位。`
- 有 L3 且 A9 单侧 95% 下限>0：`在经审计的凤凰桌等效假设下，有支持安定九段的统计证据；不构成官方段位。`
- 10段积分模拟：`基于固定 p1..p4 的模型，初始十段2000pt，模拟在先跌至0之前升至4000的概率 ...；结果依赖对手池和对局独立同分布假设。`
- 禁止：`击败了5段Mortal所以已经9段`，`bc loss 更低所以天凤位`，`代理赛场达标即真实拿到天凤位`。
