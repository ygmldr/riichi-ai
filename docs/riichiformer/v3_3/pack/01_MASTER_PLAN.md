# RiichiFormer V3.3 — Mortal-5D 相对 PT 驱动的并行迭代总方案

**日期**：2026-11-21　　**方案版本**：v3.3/plan-1　　**状态**：设计与交接，非已经执行的训练或评测。

## 0. 适用范围和事实边界

用户提供的本地最新事实（尚未由本助手访问其本地 worktree 重跑）：

- V3.2：本地 `rf/v3.2-local` / `rf/v32-integration` 头为 `00e6576`；PR-37～42、PR-38 multi-decision loader/trainer 已合并，707 passed、2 skipped、281 subtests。
- PR-44 B0 自博弈 500 局，333,224 decisions，样本 seed=20261121；2 个训练随机 seed，各 2,000 steps、107,026 decisions、1,766,242 shared tokens；pilot loss 约 1.65→1.295。完整 6,191-step 作业已终止且没有保存 checkpoint。
- PR-38 loader+B0 pilot reviewer 尚在执行；`rf3.2-w2` 未打。GPU 应保持空闲直到用户明确授权正式计算。
- 历史评估框架 `scripts/eval_report.py` 使用 `(90,45,0,-135)` 以及四席轮换每个 seed 组成一个 set；`scripts/exp_a14_delta.py` 计算相对锚点的 paired ΔPT 和 set-level bootstrap。这些是旧仓库代码，不能认定 V3.2 最新实现与之相同。
- 本助手可访问的 GitHub 公开最新分支仍是 `rf/v3.2-policy-parallel-plan`，远程 HEAD `a27df099…`，**不包含**用户刚汇报的本地 `00e6576`。Lead 必须以当前本地最终测试与 SHA 为准，不能拿远程旧树覆盖。

### 本方案目标

**一个固定基准、两个独立实验池、三个验收层级**：

1. 冻结自训 Mortal-5D 为 Reference Anchor `M5D-R0`，得到可复现的 1v3 相对 PT；
2. 将 `validation`（训练选模型用）、`arena_select`（研发用）和 `arena_confirm`（一次性独立最终确认）严格分离；
3. 各模块按 `CORRECTNESS → STATISTICAL_VALIDITY → STRENGTH` 分层通过，禁止因 loss 下降、mock arena 或不完整 seed set 宣称真实段位。

### 不在此阶段做的事

- 不在天凤段位/凤凰桌上线 AI；不把 `tenhou_approx` 的仿真结果当真实天凤段位；不把 Mortal-5D 标签当作客观强度证明。
- 不经用户许可自动启动 full epoch、长时间 GPU 训练或 12k 场 Arena；不覆盖已有 H00/PR17/旧 `runs`、`checkpoints`、`vendor`。
- 不对用户提供的本地 PR-37～42 自述代替独立 Review，不自动假设 `rf3.2-w2` 已通过。

## 1. 基准评分规范

### 1.1 两个明确命名的 PT 定义

- `TOKUJOU_5D_HANCHAN = [75,30,0,-105]`：**主指标**；对应天凤五段在特上卓四人东南战的名次 PT。
- `HOUOU_7D_HANCHAN = [90,45,0,-135]`：**辅助历史指标**；旧 `eval_report.py` 的 `RANK_PTS` 采用它。
- 请不要使用泛化而含混的 `pt` 列名，必须输出 `pt_5d`、`pt_7d`，并在 `score_policy_id`/`score_policy_hash` 明示版本。
- 最终排名的平分规则要与当前 native arena 的 `rankings.rs` 一致，旧 `eval_report.py` 按分数降序、同分座位号小者优先；Lead 必须重放固定 tie fixture 检查。

天凤官方手册： https://tenhou.net/man/ 的段位点数章节（四人东南战；5 段四位 -105、特上一位 75 二位 30，凤凰一位 90 二位 45）。本文件不复制第三方页面原文。

### 1.2 每场和每个四席 set 的计算

每场一个候选 RiichiFormer (`C`) 对阵 3 个固定 `M5D-R0`：

```
pt5_C(g) = [75,30,0,-105][rank_C(g)-1]
pt5_M(g) = sum(pt5 for three Mortal seats) / 3
D5(g)    = pt5_C(g) - pt5_M(g)
        = 4/3 * pt5_C(g)     # 因为每桌 4 个名次分之和为 0
```

四个 seat rotation 的独立 set `s`：

```
PT5_s     = mean_{rotation=0..3}(pt5_C(g_s,rotation))
DeltaPT5_s= mean_{rotation=0..3}(D5(g_s,rotation))
          = (4/3)*PT5_s
```

主估计量 `DeltaPT5 = mean_s(DeltaPT5_s)`，单位 **pt / 半庄**；辅助 `avg_rank`、`rank1_rate`、`rank4_rate`、`pt_7d`、相同公式的 `delta_pt_7d`。请同时报告 `pt5_C` 原始值，不能只报告差分。

**强度解释**：`DeltaPT5>0` 表示在此版本规则/此固定 1v3 对手池中优于三席 Mortal 平均结果；**不**等于达到或超过真人天凤五段，也**不**可由 ΔPT 映射为“九段”。

### 1.3 置信区间 / 敏感性分析

- **主要方法**：按 **完整 seed-set**（一个墙种子对应 4 个 seat rotation）cluster bootstrap，固定 bootstrap seed=由 PREREG 派生并记录，B=2000～10000；报告双侧 95% percentile CI；验证集固定则报告模型已见/未见状态。
- 不将每场半庄或每个 decision 当作独立样本；同一墙种子四场存在关联。
- `Mortal-5D vs 3×同版 Mortal-5D` 对照应回到接近 0（理想确定性轮换严格为 0）；另做两个不同模型但随机替换标签的 permutation/seat bias 负控。
- 报告 `std_set`、`sem_set`、`n_complete_sets`、`n_games`、`missing_sets`、`failed_games`；若缺失一场，整个 set 不进入正式 point estimate / CI，错误与重试记录留存。
- 失败局不能挑选性移除，应重跑同 seed/同 config，失败仍存在则中止正式评估、给出失败清单。
- 多个对手/多个指标中预注册唯一主指标，避免在见到结果后挑显著的口径。

## 2. 基准冻结（PR-46A，第一优先级）

**输入**：用户明确指认自训五段 Mortal 的 checkpoint；不要擅自把 `checkpoints/mortal.pth` 或 `*.at-148800` 当成所谓“五段”权重。历史 `BASELINE_LOCK.md` 报告 `checkpoints/mortal.pth` SHA=`0a44613c…` 是 396k-step 的纯 CNN，这是一个候选，但用户必须确认具体是哪一份。不要读取 `best_perf={1.0,90}`，它是旧占位值。

`baseline_registry.yaml` 至少包含：

```
anchor_id: M5D-R0
role: frozen_opponent
name: mortal_selftrained_5d
checkpoint_path: /abs/path/to/actual_5d_checkpoint.pth
checkpoint_sha256: <actual_sha>
model_arch: <from checkpoint>
model_config_hash: <sha>
python_driver_git_sha: <sha>
libriichi_git_sha: <sha>
libriichi_build_sha256: <sha>
patch_manifest_hash: <sha>
tenhou_approx_rules_hash: <sha>
inference_mode: deterministic_argmax # 需要按实际 driver 确认
action_guard: <same for anchor and C or document difference>
seed_policy: per_game_and_seat
permission: local_research_only
```

必须做 **100 个完整 seed-set 以内的小型校准**：加载三席独立 Mortal 实例；同 checkpoint 在每席的合法 mask 检查；确认没有隐藏手牌泄露；同 seed+版本重复结果字节稳定。验证时可用极少 seed，重复次数由 Lead 决定。先做 4～20 场 smoke 再扩大。

**P0**：任何作弊信息、无法加载权重、rank/pt 口径错、四席不轮换、baseline 实际发生权重改变、不同模型推理条件不一致、对局结束得分/顺位错误。

## 3. 与 B0 训练同时构建的独立验证系统（PR-45/47）

### 3.1 三个池必须真正独立

1. `bc_validation`: 由**另一个完整 game-id/seed 范围**生成的 100～200 局自博弈（若当前仍是自有规则 bot 数据），只用于校验 imitation 的 CE、top1/top3、合法动作、phase breakdown；与已有 500 局 B0 train 在 `game_id`, generator seed, sample hash 上不交叠。
2. `arena_select`: 早期模型晋升/调参；需新 wall `seed_key`，不得复用历史已消费评测窗口。能多次评估，但逐次登记，不当 confirm。
3. `arena_confirm`: 模型和配置冻结后再启用的**只消费一次**的 virgin wall seed+key，所有用途（包括 debug、查看某几局）预注册，不能在实验后重置为 virgin。

留意 `docs/riichiformer/EVAL_POOL_REGISTRY.md` 中历史已消费的 `[10000,13000)`,`[20000,23000)`,`[30000,33000)`,`[50000,53000)`,`[70000,73000)`,`[120000,126000)` 及 EX1 等窗口。**新窗口不能凭没看到文件就称 virgin**；Lead 必须做全库历史记录审核、记录证据和生成新 `seed_key`。训练随机 seed `20261121` 不是评测墙池 key，两者不能混淆。

### 3.2 离线诊断指标

- 每步：`NLL/CE`（仅合法且监督有效）、top1/top3、模型输出非有限率、非法动作率、动作分布 KL；
- 具体场景：`self-turn`, `call-response`, `kan-selection`, forced/censored；riichi yes/no、pon/chi/kan、弃和/防守、最后一巡、南四名次压力；
- 每局：平均 NLL + sequence length histogram + left/right truncation 与 chunk boundary coverage；
- 在 `multi_decision` 与 `prefix` 两条实现相同模型权重下进行对照，logits/gradient parity 的 fixture 是正确性证据，但不把训练吞吐差异当强度证据；
- 标签来自弱自博弈规则 Bot 时，**低 CE 仅说明复刻弱策略**；必须由 Arena PT 评价强度。

### 3.3 PR-47 端到端闭环

`selfplay game → four POV projections → decision snapshots → shard v3 → multi-decision loader → six-layer train → checkpoint → inference → duplicate arena`。

必须覆盖：正常和局、流局、立直、鸣牌、杠、四家立直、三家和了、末局供托、R29 一发取消待补场景；分数守恒、规则版本完整记录、`UNKNOWN` 只能在对家隐藏事件中使用。禁止 training target进入 forward；验证数据前缀改变/未来事件改变的负控。

## 4. 1v3 Arena 执行设计（PR-46 适配器、计分与统计）

### 4.1 单场

- 切换到当前 `tenhou_approx` 的本地离线四人东南战规则引擎；加载候选 checkpoint 作为 seat 指定位置，三个座位同一 `M5D-R0`；四个 Agent 的随机种子从 `(wall_seed, seat, agent_hash)` 派生，且改变模型名不应改变牌山。
- 每步从 POV-only 事件和可见状态建立观察；使用 native legal-action mask；推理类型与 baseline 预算明示；无非法动作 fallback 隐藏。遇到非法动作，完整 trace + failed run，中止正式评估。
- 写 `run.json`, `games.jsonl` / NDJSON 压缩、每 set 汇总、rules_deviation flags；不要向旧 `runs/1v3` 等冻结目录写入。

### 4.2 每个 seed set

- 同一预注册 wall seed，4 个 seat-rotation (`a`,`b`,`c`,`d`)。
- `C@seat 0/1/2/3`；另外三个全是相同 `M5D-R0`。所有四场必须成功，报告时按 set 聚合。
- 同一游戏内模型切换不能影响随机流，若使用随机策略，推理 RNG 与 wall RNG 分离。
- 两张显卡无需并行，5070 Ti 一次只有一个正式评测/训练 GPU job；CPU arena 可以分 worker，但重 CPU 任务与 native build 资源需 Lead 调度。

### 4.3 三个阶段，不要一开始就跑 12,000 场

| 阶段 | 独立 set | 半庄数 | 作用 | 数据性质 |
| --- | ---: | ---: | --- | --- |
| Smoke | 5～25 | 20～100 | 适配器/规则/日志/崩溃/性能 | debug，非强度 |
| Trend | 100～250 | 400～1,000 | 获取方差、明显方向、初始速度 | trend，可重复但登记 |
| Select | 500～1,000 | 2,000～4,000 | 在模型候选之间选择 | select，见过即可不再 virgin |
| Confirm | 2,000～3,000 | 8,000～12,000 | 对**冻结的唯一 winner**一次性确认 | 未见过的 confirm |

这些是规划预算，不是每次实验必须打完的数。**先用 Trend 实测 set 方差、对局耗时与故障率，再冻结主实验所需 sample size**。避免把预设 3,000 set 视为无条件 GPU 运行命令。

### 4.4 对照和可比性

- 控制组 `M5D-R0 vs 3×M5D-R0`，不同场次 `M5D-R0` 扮演挑战者的 seat 标签；平均 ΔPT 应为 0 附近；发现固定系统偏差即 P0。
- 同一新评测池对相邻 champion（原 B0 与 B1）做 paired comparison，减少墙随机波动；但这相当于**已使用**评测池，不得再做 confirm。
- 基线长期固定，即使将来使用 Mortal 强版、akochan 或其他策略，保留对 `M5D-R0` 的趋势坐标。
- 旧 `eval_report.py` 的固定 500-set EXACT_GATE 和 A14 的 3,000-set 检查不适合新自定义游戏数量：新建 `rf_eval_m5d.py` 或通用 scored protocol 类，不要硬改老 gate 或覆写旧结果。

## 5. B0 正式化（PR-48，含先前 PR-44 pilot）

### 5.1 先补当前 pilot 的证据

- 等 PR-38 loader/trainer+B0 pilot Reviewer 只读审计完成，解决 P0/P1 后打 `rf3.2-w2`。
- 对已有 2×2000-step pilot 校验 SHA、真实 config、两个不同 seed、训练样本划分、相同步数；不能拿缺 checkpoint 的 6191-step 中断作业充当 B0。
- B0 保留两个种子：独立训练的随机初始权重、optimizer RNG；同数据同预算，模型 `6-layer 5,217,582 params`；**`--multi-decision` 显式启动**，不能让默认旧 prefix 路径与共享路径混用。
- 各 seed 在一致训练步数保存候选（示例 0/250/500/1000/2000；只有满足 I/O预算时加密）；用验证 CE 与稳定性挑选，规则必须在开始前冻结。

### 5.2 训练默认设置与容量规划

根据 V3.1 MVP-S：6 层 × hidden 256，4 heads、FFN 704、State MLP 172→512→256、AdamW/BF16。不要根据网络输出宣称稳定的 tokens/s；PR38 Shared-Forward 在全长度、半庄切块下需重新 benchmark 16 GB 显存。推荐梯度累积以**有效监督权重总和**归一化，并记录 0-valid batch 的处理策略。必须检查和单一大 batch 更新数值等价。优先实现 EMA/模型选择/early stop 的正确记录，再扩容。

**实验顺序**：2 seed pilot retrospective → B0 validation smoke → B0 select arena → 经用户批准的完整 B0 训练 → freeze B0/confirm。若 pilot 完全弱于启发式/无法合法完成对局，应先修策略/数据，不投入 full epoch。

### 5.3 正式 B0 的明确预注册

`PRE_REG`: train/game SHA、two model RNG seeds、lr/warmup、optimizer、max step 与 early stop、EMA、validation schedule、only-one model selection rule、checkpoint retention、budget/time/VRAM stop、arena wall pool、M5D anchor SHA、PT profile、primary test/CI。任何调参后须换实验 ID，不能在同一个 confirm 上重复选模。

## 6. 后续研究路线：从强弱证据选优先级

### Tier 1：低成本数据/训练改进（先做）

1. **PR-18 数据质量**：旧 PR-06 玩家质量打分在 100k 样本上时序预测不单调，`top25` 未证实更强，不能直接作为默认加权；先重算全量 time-forward/校验混淆，做 no-weight vs quality-weight 严格单轴实验。
2. **PR-19 时间/来源**：比较近年份、全历史、时间衰减、不同规则区间；匹配总训练决策数、同 optimizer/seed，`train/val` game 级无交叉。
3. **PR-20 行为标签/动作权重**：监督有效、非 forced、phase balance、罕见鸣牌/立直事件；稀有动作重权重不能改变标签合法性。报告 CE 与 Arena 两类指标。
4. 正式化数据加载吞吐：Shared Forward 多决策 chunk、长序列处理、GPU 显存/CPU IO 基准，任何数据模块改动需重跑 POV 和 prefix-joint parity。

### Tier 2：结构/策略改进（可并行开发，GPU 正式 A/B 串行）

- **Candidate/Action Representation**：比较 46 action linear head 与 candidate-conditioned head；动作合法性、红五、kan 决策必须 strict parity；固定训练数据/步数验证是否提升 ΔPT。
- **Risk/Belief**：从可见特征预测对手听牌、危险度、放铳损失；只允许把训练标签作为 loss 辅助目标，不能注入推理 forward；先做 mask/hidden-tile counterfactual 证明零泄漏，后做线上策略效益 A/B。
- **Rank/Value head**：训练 expected placement/expected PT 的辅助 head，最终策略应在相同规则和测试预算下验证 PT；避免由局部和牌分数误导到四位率上升。
- **Memory**：跨局 carry/summary，和不带 memory 的相同参数/计算预算负控。优先证明 feature 有增益，不盲目扩大 context 长度。
- **Teacher/distillation**：使用已固定的 Mortal 权重生成对局或推理标签时，明确样本流与生成成本，不能混同“模仿 Mortal”与“超越 Mortal”。自有数据的模型训练不等于必须用教师权重。

### Tier 3：RL（到达可用 B0 后）

- 基线门槛：合法对局率 100%、独立评估仪表可复现、同样本重复训练一致、弱 Bot 上明显能打、冻结 anchor；
- 自博弈对手池：B0、最近 champion、保守/激进 Bot、M5D-R0（作为 eval；是否纳入训练 opponent 要版本化标记），轮换座位和 wall key；
- PPO/离线+在线混合：预注册 KL target、clip、entropy、rank/PT reward 标准化、policy drift/invalid-action kill switch；
- 不仅看自博弈胜率：固定 5D anchor ΔPT 和上一代 champion vs current paired tests；出现历史 champion 回归即拒绝晋升。

## 7. 里程碑与判断门槛

| 里程碑 | 必要证据 | 失败处置 |
| --- | --- | --- |
| `G-V33-W2` | Reviewer P0/P1 关闭，`rf3.2-w2` 或后续新 tag，所有 RF tests 可复现 | 无 tag；先修 |
| `G-V33-A0` | M5D-R0 权重和 adapter 固定，M5D self-control、4-seat 100 games smoke，错误 0 | 不启大评测 |
| `G-V33-D0` | 100～200 局独立 validation、game/seed hash disjoint、端到端 roundtrip | 不正式化 B0 |
| `G-V33-B0` | B0 2 seed、同配置、val 曲线、checkpoint strict resume、M5D Trend vs frozen opponent | 未证实强度时仍叫 B0-pilot |
| `G-V33-C0` | 选模规则冻结；1 个最终 winner 使用独立 Confirm；Set-bootstrap PT 95% CI、原始逐游戏日志与 hash | 不可宣称强于 M5D |
| `G-V33-R0` | 多实验复现有改善、CI/效果量支持、原 champion 与 anchor 不回归 | 不做重 GPU RL |

## 8. 报告与禁止事项

所有结果应同时打印：`anchor checkpoint sha`、`candidate checkpoint sha`、规则和 adapter build sha、`pt_profile`、`seed_key`、池身份、已消费状态、完成的 set、失败的 game、单 seat rank 分布、`raw_pt`、`delta_pt`、95% CI、训练 source/steps/数据 hashes、非法动作与 fallback 数量。任何缺一项要在 Gate 写 `UNKNOWN/NOT_RUN`，不得用 0 或 PASS 填充。

**用户确认长期默认策略**：在未明确批准某次训练/评测 GPU 预算前，只允许代码、单测、少量 CPU smoke 和评估数据审计；当前空闲 GPU 不意味着已授权后台占用。
