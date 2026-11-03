# 讲谱 AI 前置工作 — 分析引擎(W1)设计文档

- 日期:2026-09-22
- 状态:**草案,待用户批准**(brainstorming 流程 §5 已呈现,等逐节确认)
- 项目仓库:`/home/ygsldr/riichi-ai`
- 上位目标:讲谱 AI(对牌谱做量化分析并生成讲解)
- 覆盖范围:**仅 W1(分析引擎)**。W2(实力与可验证尺子)、W3(LLM 讲解层)只给出定位与依赖,各自另立 spec。
- 实现状态(2026-11-03):W1 引擎已落地;`opp_tenpai_prob` 已作为独立 H1 provider 在 `report_sub` 评估(**ship_low_confidence**,控制口径见 `docs/EXPERIMENTS.md` §9 与 `runs/exp/OPP_TENPAI/CONTROL_REVIEW.md`);`danger`/`called_risk`/`post_call_risk` 仍 `unavailable`。

## 1. 背景与目标

### 1.1 三层目标结构与依赖

用户已确认的终局:先把前置工作做扎实,再微调 `Qwen3.5-2B` 作讲解层,产出面向人类玩家的讲谱产品**(目标 b)**,自复盘**(目标 a)**是其第一步。

```
W1 分析引擎(本 spec)  ──┐
  牌谱 → 逐决策指标 → 稳定契约 │
                              ├──▶ W3 LLM 讲解层(微调 Qwen + grounded 评测)
W2 实力与尺子(另立 spec) ────┘
  W2a 补完 tenhou 個室 live smoke(需 operator)
  W2b 建立外部可验证锚(牌谱锚定 / 個室 ladder)
  W2c 训练杠杆(A 清单:#1 修 confound / #2+#3 打包 / #5 oracle guiding)
```

**依赖**:W1 与 W2 **互相独立**,可并行;**W3 依赖 W1 的输出契约**(以及 W2 的强度)。

### 1.2 本期(W1)目标

给**任意一份立直麻将牌谱**,产出**分层、可复现、可追溯**的逐决策指标 JSON,它同时服务三个下游:

1. 目标 (a):自复盘的人读产物;
2. W3:Qwen 的**唯一输入**(LLM 不吃原始张量);
3. W2c:比较训练杠杆的**量化量尺**。

指标集合 = 用户明确的 5 项 + 补充项:

| # | 指标 | 现状(载荷事实,见 §2.2) |
|---|---|---|
| 1 | 其他家的听牌概率 `opp_tenpai_prob[j]` | **不存在,需新建头** |
| 2 | 牌对其他家的铳率 `deal_in_prob[j][tile]` | **不存在,需新建头** |
| 3 | 自己的和率 `own_win_prob` | 存在(SP 表),未暴露 |
| 4 | 打出每张牌后的期望顺位分布 | **部分**:GRP 给出**状态级**顺位矩阵;「打出**某张牌后**」的动作级反事实不存在 |
| 5 | 打出每张牌的局收支估计 `q_rel` | **已随牌谱落盘**(日志内 `meta.q_values`),缺解码层 |
| 补 6 | 关键局面识别 `key_moments` | 新增 → 单元 4 `aggregator` |
| 补 7 | 悪手度/regret 量化 `regret` | 新增 → 单元 3f |
| 补 8 | 结果论解耦(判断对但结果差)`judgement_vs_outcome` | 新增 → 单元 4 + 契约 §4 |
| 补 9 | 自身听牌率/向听进展(进攻侧对称) | 存在 → 单元 3b(SP 表) |
| 补 10 | 待ち/当たり牌的具体枚举 `dangerous_tiles[]` | 由 #1/#2 派生 → 单元 4 |
| 补 11 | 被鸣风险 `called_risk` | **未决**(见 §9 Q3) |

> **量纲警告(必须在契约里区分)**:`#5 q_rel` 是 **pt-EV 尺度**(顺位点的价值差),而 SP 表的
> `exp_values`(#3 / 补 9)是 **normalized_ev 尺度**(归一化期望值)—— **两者不可互换、不可比大小**。
> 契约中必须**分字段命名**并在 schema 里写明单位(对应 §3.2 D6)。

### 1.3 明确不做(YAGNI)

- **不做** UI / 可视化(W3 阶段再议)。
- **不做**自然语言生成(属 W3)。
- **不改**训练 / 评估管线;不改 `libriichi` 观测与动作编码。
- **不碰** `runs/1v3/` 冻结语料(2004 个 `*.json.gz`,只读)。
- **不引入**新平台依赖(雀魂等);**不做** 3 人麻将。
- **不在本期**训练任何"讲解"能力——引擎只出结构化数字。

## 2. 约束与既有事实

### 2.1 约束

| 约束 | 值 |
|---|---|
| 推理环境 | 仓库 `.venv`(Python 3.12,torch 2.11 + libriichi release `.so`) |
| 硬件 | 单张 RTX 5070 Ti 16GB;离线批量分析以 **CPU 单线程**即可(见性能底数) |
| 许可 | 本 fork = **AGPL-3.0**;`mjai-reviewer` = Apache-2.0(可复用);Mortal 须留在**进程边界外** |
| 输出可复现 | 同一输入两次运行必须**逐字节相同** |
| 不静默缺省 | 算不出的指标必须显式进 `unavailable[]` |

### 2.2 决定本设计的既有事实(均已实测/已核对,非推测)

1. **#5 已在日志里**:`Metadata{q_values, mask_bits, is_greedy, shanten, at_furiten}` 随对局落盘
   (`vendor/mortal/libriichi/src/mjai/event.rs:130-150`,`agent/mortal.rs` `gen_meta`);
   实测 `runs/1v3/10219_579411421376059130_d.json.gz` 每局 544 个决策点均带 `q_values`。
   动作索引:`0–33` 基本牌打牌、`34/35/36`=5mr/5pr/5sr、`37` 立直、`38/39/40` 吃(低/中/高)、
   `41` 碰、`42` 槓决定、`43` 和了、`44` 流局、`45` pass(`consts.rs` `ACTION_SPACE=46`)。
   **语料 provenance(实测,决定门 1 的可行性)**:`runs/1v3/` 的 2000 个主文件 key = `0x80a7b9e2dec96fa`
   (= `579411421376059130`),mtime **2026-09-15 18:44**;另 4 个 legacy 文件 key `423244137402621143`、
   mtime 2026-09-16 20:26,其 **`meta` 仅含 `eval_time_ns/shanten`(无 `q_values`)**。
   据 `runs/1v3/PREREG.md:86`,2000 局当时 challenger 为 **live `checkpoints/mortal.pth`(未 pin、步数未知)**
   ⇒ **该权重很可能已不存在**,故门 1 不得依赖此语料(详见 §6.1)。
2. **#3/#9 藏在观测里**:SP 表 `tenpai_probs / win_probs / exp_values`(逐候选打牌 × 逐未来巡目)已烤进观测张量
   (`obs_repr.rs:563-692`),逐字节偏移见 `scripts/obs_layout.py:545-577`。
3. **#4 是状态级、非动作级**:`GRP`(`checkpoints/grp.pth`,steps=78000)由
   `model.py:400-453` 定义,`calc_matrix` 出 `(4,4)` 顺位概率矩阵,review 模式已以 `phi_matrix` 输出;
   `aux_net = Linear(1024,4)`(`model.py:352-359`)预测**行动者自己**在该局结束后的顺位。
   **两者都只吃状态、不吃动作** ⇒ 动作级反事实需新建(见 §9 Q1)。
4. **Q 值不可当概率**(硬约束,来自我们自己的审计):
   A2b 实测 `softmax(Q/τ)` 的 **ECE:τ=1 时 0.0364,τ=0.05 时 0.1897**;Q 局内排序
   **Spearman ρ = 0.145**;Mortal 官方 FAQ(`mjai-reviewer/faq.md:22-23`)明载
   **"Mortal 从来没有算过放铳率"**、**非最优动作的 Q 不可靠**。
5. **公共特征已足够**:A2 实测 frozen φ 线性探针 `opp_any_tenpai` **AUC 0.9334 [0.9312,0.9354]、ECE 0.0099**,
   而 **24 维公共特征基线 AUC 0.9399 > 0.9334** ⇒ φ 未编码超出公共信息的隐藏状态。
6. **两个新指标的标注是免费的**:放銃(放銃方/放銃牌)与听牌(他家手牌)都能由**牌谱回放直接观测**;
   `scripts/exp_a2_probe.py:143 verify_and_label` 已从 oracle 张量解出 `opp_any_tenpai`
   (787,017 观测,0 critical violation)。⇒ **不需要人工标注**。
7. **噪声地板换算**:`MDD_rank ≈ 2.6213/√n`(σ_set=0.467788,80% 功效);
   `MDD_pt ≈ 1.315·√(24000/n)`。分辨 0.02 顺位需 ≈17.2k 局、0.01 需 ≈68.7k 局。
8. **性能底数**(tenhou 计划实测,CPU 单线程):单步 `p50 16.9–20.9 ms`、`p95 31–69 ms`
   (重载最坏 103 ms,零 >200 ms 停顿)。一局约 270 决策 ⇒ 单模型 pass ≈ 5–10 s,离线批量可行。

## 3. 总体架构

### 3.1 组件划分(六单元,各自单一职责、可独立测试)

| # | 单元 | 职责 | 复用/依赖 | 接口 |
|---|---|---|---|---|
| 1 | `record_reader` | 任意来源 → 统一 mjai 事件流 + 完整性/身份校验(恰一个 `start_game…end_game`;names/seed 核对) | `scripts/analyze_alllast.py:decode_events`;人类牌譜前置转换走 `convlog`(可选) | `read_record(path) -> Record{events, source, names, seed}` |
| 2 | `replayer` | 用 `libriichi.PlayerState` **只推进状态**(不用 `Bot`,它会自行选动作),产出逐决策上下文 | libriichi `state` | `iter_decisions(rec) -> Iterator[Decision{player, obs, mask, legal_actions, public_state}]` |
| 3 | `metric_providers` | 一个指标一个 provider,各自声明版本与校准元数据 | — | `provider(ctx) -> payload \| Unavailable(reason)` |
| 3a | ├ `q_ev` | DQN 前向 → 46 维掩码 Q → **相对分**(定义 `q_rel_i = q_i − max_j q_j`,即同决策点内对最优动作的差) | `model.py` DQN | `{q_rel[], action_index_map, unit:"pt_ev", confidence}` |
| 3b | ├ `sp_table` | 按文档偏移切出 SP 表(**该表语义 = 自家按期限抵达听牌/和了的概率**,非"对手当前听牌似然") | `obs_layout.py:545-577` | `{own_tenpai_prob, own_win_prob, own_ev, by_turn[], unit:"normalized_ev"}` |
| 3c | ├ `rank_dist` | 跑 GRP → (4,4) 顺位矩阵(**状态级**) | `grp.pth` + `calc_matrix` | `{matrix[4][4], expected_rank[4]}` |
| 3d | ├ **`opp_tenpai`**〔新〕 | 每对手听牌概率(逐巡目) | 回放自动标注(事实 6) | `{opp_tenpai_prob[4], by_turn[]}` |
| 3e | ├ **`danger`**〔新〕 | 每对手 × 每张牌 的放铳概率 + 期望失点 | 回放自动标注(事实 6) | `{deal_in[j][34], exp_loss[j][34]}` |
| 3f | └ `regret` | 悪手度 = 实打 vs 最优的价值差(**带置信标记**) | 依赖 `q_ev` | `{delta_rel, is_blunder, confidence}` |
| 4 | `aggregator` | 逐决策 → 每局摘要 → 整局摘要 + **关键局面 top-k** + 结果论解耦 + 待ち/危険牌枚举 | — | `{kyoku_summaries[], game_summary{judgement_vs_outcome}, key_moments[], dangerous_tiles[]}` |
| 5 | `contract` | 版本化 JSON Schema + 校验器(`unavailable[]` 显式) | jsonschema | `validate(doc) -> ok \| errors` |
| 6 | `cli` | `analyze_record.py --record X --checkpoint Y --out Z` | 以上全部 | 退出码:0 成功 / 非 0 校验失败 |

**关键局面 top-k 的选法**(单元 4 的核心):两条排序取**并集去重** ——
(a) 按 `|regret|` 降序;(b) 按「`expected_rank` 在该决策点的摆动幅度」降序,该幅度定义为
`|expected_rank(after) − expected_rank(before)|`,并在同一局内归一化后再排序。
一局约 270 打,真正值得讲的通常 3–5 打——这是给 W3 划定"讲什么"的输入。
**结果论解耦**:`game_summary.judgement_vs_outcome` 显式给出"该局**判断质量**评分(基于 regret 聚合)"
与"**实际**得失点"两个**相互独立**的字段,并标注二者背离的局面(判断好但结果差 / 判断差但结果好)。

### 3.2 关键架构决策

| # | 决策 | 理由 |
|---|---|---|
| **D1** | **Q 值只作相对排序呈现**,并带 `low_confidence` 标记(Q 差距 < 阈值时) | 事实 4:Q 局内排序 ρ=0.145、τ=0.05 ECE 0.1897、Mortal FAQ 明载非最优 Q 不可靠。当绝对 EV 展示会教坏用户 |
| **D2** | 两个新头的**默认基线 = 仅公共特征**;融合 φ 只作**消融项**,不得作为默认 | 事实 5(公共特征 AUC 0.9399 ≥ frozen φ 0.9334)+ 事实 6(标注免费)⇒ 不值得为标注付人力;消融用于验证"φ 是否真无增量" |
| **D3** | 人类牌譜路径复用 `mjai-reviewer` 的 `convlog`(Apache-2.0),Mortal 留在**进程边界外** | 许可侦察:Mortal/libriichi=AGPL(本 fork 已是);`convlog`=Apache-2.0 可放心用;进程边界即许可边界 |
| **D4** | 契约**自带分层摘要**(逐决策明细 + 每局摘要 + key_moments),不给下游原始张量 | 一局 270 决策 × 5+ 指标远超 2B 模型上下文;压缩必须在引擎侧完成 |
| **D5** | 每个 provider 输出必须带 `{version, checkpoint_sha256, calibration}` | 让 W2c 的"换了检查点/杠杆后指标怎么变"可归因;也让 W3 语料可重放 |
| **D6** | **两套 EV 尺度必须分字段、并在 schema 写明单位**(Q = pt-EV vs SP = normalized_ev) | 二者不可比;混用会让讲解出现"数字对但结论错"的错误 |
| **D7** | `low_confidence` 阈值**来自校准产物**(provider 元数据),**不得硬编码** | 阈值是 Q 校准的函数;硬编码会在换检查点后失效 |

## 4. 数据流

```
牌谱文件 ──1 record_reader──▶ events
events ──2 replayer──▶ decisions[obs, mask, public_state]
decisions ──3 providers──▶ 逐决策 payload(各模型/切片各自跑,互不依赖)
                                    │
              4 aggregator ◀────────┘ ──▶ 每局摘要 + 整局摘要 + key_moments
                                    │
              5 contract ◀──────────┘ ──▶ 校验后的 JSON(下游唯一输入)
```

**契约顶层字段(草案)**:
`schema_version` / `engine_version` / `checkpoint{path, sha256, steps}` / `record{path, source, names, seed}` /
`decisions[]` / `kyoku_summaries[]` / `game_summary{}` / `key_moments[]` / `unavailable[]`。

## 5. 错误处理与鲁棒性

| 情形 | 行为 |
|---|---|
| 未知事件类型 | **硬报错**(静默跳过会污染状态,宁可不跑) |
| 人类牌譜缺 `meta` | **正常路径**(走自推理),不是错误 |
| 某指标算不出 | 写入 `unavailable[]` + reason;**不填 0、不省略** |
| Schema 校验失败 | 非零退出,**产物不落盘** |
| 同一输入两次运行结果不同 | 视为 bug(见 §6 门 4) |

## 6. 验证策略(四道门,全部可自动执行、零人工判断)

| 门 | 内容 | 强度来源 |
|---|---|---|
| **门 1 · 语义一致性(强,但必须先自建参照集)** | **先用一个 pin 生成参照日志集**(`one_vs_three.py` + pin,如 `at-396000`),再对同一 pin 重推理,逐决策比对 `meta.q_values` vs 引擎 Q:argmax 必须**完全**相同,Q 必须在**已实测标定的容差**内 | 端到端正确性断言,**不需要任何标注**。**禁止用 `runs/1v3/` 冻结语料当参照**(见 §6.1 的两条硬理由) |
| **门 2 · 指标不变量** | SP 表切片与 `obs_layout.py` 偏移一致;GRP 矩阵每行和 = 1;`opp_tenpai_prob` 在已立直对手上必须显著更高(单调性);`deal_in[j][現物] ≈ 0`;`deal_in` 的 34 张和落在合理区间(**不强制 = 1**,见 §9 Q5) | 用麻将领域不变量抓实现错误,覆盖率高 |
| **门 3 · 性能** | 一局端到端 CPU 单线程 p95 上限(底数见事实 8) | 保证离线批量可用 |
| **门 4 · 契约与可复现** | JSON Schema 校验 + 同输入**两次运行逐字节相同** | 让 W3 语料可重放、可回归 |

### 6.1 门 1 的两条硬理由与两个必须显式化的细节(实测得出)

**为何禁止把 `runs/1v3/` 当参照(两条硬理由)**:
1. **provenance 不可复现**:该语料 2000 个主文件 mtime = **2026-09-15 18:44**,key `0x80a7b9e2dec96fa`;
   `runs/1v3/PREREG.md:86` 明载当时 challenger 为 **live `checkpoints/mortal.pth`(未 pin、步数未知)**,
   而 live 检查点此后已被 328k → 382k → 396k 多次覆写 ⇒ **产生它的权重很可能已不存在**,无法重推理比对。
2. **字段不齐**:同目录另有 **4 个 legacy akochan 文件**(key `423244137402621143`,mtime 2026-09-16 20:26),
   其 `meta` **只有 `eval_time_ns/shanten`,没有 `q_values`**。

**两个必须显式化的细节**:
- (i) **容差来源**:推理开 `enable_amp = true`(fp16 autocast),且日志中的 Q 由**批量**算出(`meta.batch_size` 记录);
  故容差**必须实测标定**(建议先取 ~200 个决策点,统计 `max|Δq|` 分布再定阈值),**禁止假设完全相等**。
- (ii) **过滤规则**:只取主语料 key 的文件;`meta` 缺 `q_values` 的文件必须**显式跳过并计数**,
  **不得静默混入**,跳过数写入产物的 `unavailable[]`。

## 7. 里程碑

| 里程碑 | 内容 | 交付物 | 是否需新模型 |
|---|---|---|---|
| **M1** | `record_reader` + `replayer` + `q_ev`/`sp_table`/`rank_dist` + `contract` + **门 1/3/4** | 可对任意检查点分析 1v3 日志;三个既有指标落地 | **否** |
| **M2** | `aggregator` + `regret` + `key_moments` | "哪几打值得讲、错多大"的量化 | 否 |
| **M3** | 训练 `opp_tenpai` / `danger` 两个头 + 门 2 单调性 | 5/5 指标齐(#4 是否升级为动作级取决于 §9 Q1 裁定) | 是(小头) |
| **M4** | 人类牌譜路径(`convlog` 接入) | 可吃天鳳/雀魂/一番街牌譜 → 真为 (b) 服务 | 否 |

> **实现计划拆分建议**:M1+M2 构成一个连贯的实现计划(引擎核心 + 既有指标 + 分层摘要,**零新模型**);
> M3(新头训练)与 M4(人类牌谱集成)**建议各自另立计划**,避免单个计划横跨 基建 / 训练 / 集成 三类工作。

## 8. 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| 动作级期望顺位分布(用户 #4 的字面要求)需新模型,而非接现成头 | 中:M3 范围可能扩大 | §9 Q1 请用户裁定:接受"状态级分布"还是必须"动作级反事实" |
| Q 值得不可靠,导致 regret/悪手度 误报 | **高**(会教坏用户) | D1 强制相对化 + `low_confidence` 标记;M2 需用 门 1 交叉验证 |
| 新头在人类牌譜上分布外(我们的头在自对弈/訓練分布上训) | 中 | M3 需**按来源分层评测**,并报告 OOD 指标 |
| 人类牌譜解析质量(convlog 是第三方) | 中 | M4 做"重放一致性"断言(同一牌譜经 convlog 与经我们的 reader 得到相同状态序列) |
| 单局耗时放大到批量(数千局) | 低 | 门 3 + 可并行进程池 |

## 9. 未决事项(需用户拍板)

| # | 问题 | 选项与我的建议 |
|---|---|---|
| **Q1** | **#4 的语义**:要"**状态级**期望顺位分布"(每个决策点一个分布,接现成 GRP,**便宜**)还是"**动作级**反事实"(打出每张牌后各自的分布,**需新模型**) | 建议**先状态级**(M1 即可交付),动作级列为 M3+ 的独立增量;因为动作级需要动作条件化头或逐候选前向/rollout,成本高一个量级 |
| **Q2** | 排期:先 W1、先补 tenhou 個室、还是并行 | 建议**并行**:個室只差你开号+开個室做一次 smoke(我要补 F3 收尾);W1 与它零依赖 |
| **Q3** | 是否要 `被鸣风险`(#11)与 `鸣牌/立直判断评估`(#13) | 建议**放 M3+**,不在 M1/M2 范围(避免范围膨胀) |
| **Q4** | 目标 (b) 的人类牌譜来源优先级(天鳳 / 雀魂 / 一番街) | 建议**天鳳优先**(公开牌譜质量最高,且与 W2b 的"段位锚定"同一数据源) |
| **Q5** | `deal_in_prob[j][tile]` 的归一化口径:是否强制 34 张和为 1 | 建议**不强制**(两面听牌违反该假设,koba 已指出 和为 1 只是近似);引擎额外输出 `sum` 供校准诊断。若要与商业产品的数字直接比较,再改为归一化 |

## 10. 附:本设计的证据出处

- 事实 1/2:`vendor/mortal/libriichi/src/mjai/event.rs:130-150`、`agent/mortal.rs`(`gen_meta`)、
  `state/obs_repr.rs:563-692`、`scripts/obs_layout.py:545-577`、`runs/1v3/10219_579411421376059130_d.json.gz`
- 事实 3:`vendor/mortal/mortal/model.py:352-359`(AuxNet)、`:400-453`(GRP/`calc_matrix`)、
  `checkpoints/grp.pth`(steps=78000)
- 事实 4:`runs/exp/A2b/README.md`(ECE 0.0364/0.1897,ρ=0.145)、`mjai-reviewer/faq.md:22-23`
- 事实 5:`runs/exp/A2/README.md`(AUC 0.9334/ECE 0.0099;公共 24 维基线 0.9399)
- 事实 6:`scripts/exp_a2_probe.py:143`(`verify_and_label`;787,017 观测,0 critical violation)
- 事实 7:`runs/variance/README.md` + `runs/exp/M3/REPORT.md`(MDD/CI 口径)
- 事实 8:`.omo/plans/tenhou-integration.md`(性能门实测值)
- 侦察背景:`.omo/notepads/fork-roadmap-steps-1-10/learnings.md`(2026-09-22 讲谱 AI 立项前侦察摘要)
