# 立直麻将 AI 研究路线交叉评审报告

- 状态：**v1（交付）** · 研究 run：`20260916-004434`
- 评审对象：`docs/立直麻将 AI 下一代训练路线探索.md` + `docs/麻将 AI 研究路线交叉评审 Prompt.md`
- 本地基线：`vendor/mortal` @ `0cff2b5`（= 上游 HEAD，上游自 2025-09-28 冻结）

> **证据分级**（全文使用）：
> `[CODE]` 本仓库源码/跑码实测 · `[官方]` 项目方一手（论文/官方博文/官方账号/作者发言）· `[评审]` 文献（同行评审论文或 arXiv 预印本） · `[单源]` 未评审但细节充分（单源例外，已标注） · `[推断]` 本报告综合推断 · `[假设]` 明示为估算前提
> 数字标注：**MEASURED**（来源直述或实测）/ **DERIVED**（由测量推导，给公式）/ **ASSUMED**（假设值）。
> 引用编号 `[S#]` 对应附录 A；`[CODE]` 类本地事实的验证记录见附录 B（V1–V7）。本报告只对**外部来源**标 `[S#]`。附录 B 编号为 V1–V7（含子验证 V1b）。另：本文 `局` 与 `半荘` 同义，规范写法用"半荘"。

---

## 摘要

1. **Mortal 的真实身份**是"Dueling DQN + 仅离线 CQL + GRP 塑形奖励"，**不是 AlphaZero，也没有 PPO**：obs `(1012,34)`、46 动作、`b40c192` 模型共 10,839,727 参数 `[CODE]`（附录 B V1）。作者 Equim **在开源前用过 PPO/actor-critic，调了数月打不过监督基线后放弃**，并**自陈不能排除是自己实现有误** `[官方]`——"DQN 更优"并非作者的主张。
2. **你的 §五 前提不成立**：Mortal v4 的 obs **已经**按位置编码了弃牌顺序与时序（first-6 顺序 + 反序-last-18 + 全局 recency-decay plane）`[CODE]`；而**立直领域不存在"等参数、等算力、以对局强度为指标"的 CNN vs Transformer 对比**——唯一符合这些条件的同行评审实验（Tjong，国标）中 Transformer 只**打平 ViT/GRU**，其招牌精度来自层次化决策头而非注意力 `[评审]`。
3. **Pure Zero**：没有任何同行评审的立直 agent 做到"零人类数据 + 随机初始化 → 顶尖人类强度"；从零的最强正例（RVR，4 人国标）**用了塑形与 oracle 价值网** `[评审]`。**严格意义的"零塑形纯零"仍未被证伪，也未被证明**——它是一条代价高昂的未开垦路线。
4. **Self-play 能加，但加法很小**：本战役唯一被独立复现的正面结果是一个**简单 actor-critic（REINFORCE + Huber 价值基线 + forward-KL 缰绳 λ=0.05）**，pooled n=36 blocks / 72,000 局，raw score **+0.230/game（95%CI +0.144~+0.316）**；而且**增益只体现在"平均和了番数更高"，胜率与放铳率不变，且已进入平台**；同一仓库的 **PPO-clip+entropy+PFSP 方案 18 次快照全部失败** `[单源]`。
5. **Opponent belief / 对手手牌分布预测没有强度证据**：不存在任何超人级不完信息 AI 使用"对手隐藏手牌监督预测器"；**三家独立团队**（Mortal 作者、hongdp、SuuTTT）的 oracle/辅助头实验**全部为 null**；RiichiEnv 声明的 `HandPredConfig` **代码模块根本不存在**（未实现，而非未验证）`[评审+CODE]`。**"校准"角度已有先例**（Sasaki 已对麻将手牌概率做过 calibration curve 与 ECE `[S63]`），但**"校准感知的训练目标 + 对局强度验收"这一组合仍未见先例**——见 §9.2 的 A2b 零训练审计。
6. **你当前真正的瓶颈是本地管线，不是架构**（跑码实证）：① `online=false`，**从未跑过自对弈**；② DQN 训练 globs 覆盖 **2009–2026 全部年份**且 `reserve_ratio=0` → **DQN 侧没有任何离线验证集**；③ `baseline.pth` 是 43.69MB 的随机初始化 →`test_play` 饱和 100%，毫无信息量；④ P3 的 "+4.11 顺位分" **显著性取决于口径**（单样本口径 z≈2.16；双样本差口径 z≈1.60、CI 跨 0），**正确的配对 duplicate 检验尚未做**——这正是必须先修评估的原因；⑤ 三个上游未修 bug（#111/#114/#115）都影响你的 pin `[CODE]`。
7. **"4 人立直无法搜索"按字面被推翻**：MeowCaTS（重要性采样 MCTS）夺 Computer Olympiad 2023，VeryLongCat/KF-TREE 亦以搜索型方案夺冠 `[评审]`。可辩护的弱形式是"vanilla MCTS 不能直接套用，且尚无搜索型 4 人立直达到 Suphx 级强度"。
8. **LuckyJ 归属纠正**：腾讯"绝艺 LuckyJ"属实，但你说它用 "Neural CFR" **不准确**——策略优化是 **ACH（NW-CFR 系遗憾最小化 actor-critic）**，搜索是**另一个**组件；且**其同行评审证据全在 1v1（vs 冠军 p=0.19，不显著）**，4 人立直部分只有新闻稿 `[官方+评审]`。
9. **立刻该做的三件事**（按 ROI）：① 修本地评估/训练管线（无验证集 + 饱和指标 + 2023 年留出重叠）；② 用一次**打乱弃牌顺序**的推理期对照**零成本证伪**整条 river-transformer 命题；③ 把算力投在**人类牌谱的质量与覆盖**上，而不是纯自对弈——这是本报告证据最强的结论。
10. 全文所有"应该/不应该"都附了**可执行的证伪实验**（§九、§十）；红队审计发现的最弱结论与"不应写出的声明"已单列（§十 尾声、§十一）。

---

# 一、背景与研究范围

## 1.1 你的现状（本地取证）

| 项 | 实测值 | 来源 |
|---|---|---|
| 硬件 | 单 RTX 5070 Ti（16GB） | 设计文档 |
| 数据 | 天鳳 2009–2026，`datasets/extracted/` 含 18 个年份目录 | `[CODE]` |
| 上游 | `vendor/mortal` @ `0cff2b5` = 上游 HEAD（2025-09-28 冻结） | `[CODE]` |
| 训练 | 离线 DQN+CQL，**193,200+ steps**，实测 **3.3–5.5 steps/s**（batch 512） | `[CODE]` 附录 B V6 |
| GRP | 78,000 步达验证顺位准确率 **22.8%**，已冻结 | `[CODE]` 附录 B V5 |
| online | `config.toml:3 online=false` —— **从未跑过自对弈** | `[CODE]` V5 |
| 验证集 | **无**（`reserve_ratio=0`，DQN globs 覆盖 2009–2026） | `[CODE]` V5 |
| 评估 | 1v3 duplicate，500 sets/2000 半荘，挑战者 2.4705 顺位 / 3.0825pt | `[CODE]` V4 |
| baseline | `baseline.pth` 43.69MB（随机初始化）→ `test_play` 饱和 100% | `[CODE]` 附录 B V5 |

## 1.2 评审范围与方法

25 个研究 worker（3 波：13 + 9 + 3）+ 1 个红队 Oracle + 8 项跑码/只读验证（含 1 项子验证）；覆盖 Mortal/Suphx/NAGA/LuckyJ 本体、Pure Zero 理论、搜索与解概念、Transformer 序列建模、belief 建模、算法选型、OSS 生态、日文与中文地域情报。**综合稿为 `.omo/ulw-research/20260916-004434/SYNTHESIS.md`，全部 worker 原文在同目录 `raw/`。**

---

# 二、现有 Mahjong AI 独立分析

## 2.1 Mortal

**网络结构** `[CODE]`：`Brain` = `Conv1d(in→192, k3)` → **40 × ResBlock**（`BatchNorm(1e-3)` + `Mish` + `Conv1d(192→192, k3)` ×2 + **ChannelAttention（CBAM，ratio=16）** + 残差）→ BN/Mish → `Conv1d(192→32, k3)` → Flatten → `Linear(1088→1024)`。DQN 头 v4 = 单层 `Linear(1024→1+46)`（v/a 共用一层、masked-mean 扣减）。参数量 **10,839,727**（Brain 10,787,456 / DQN 48,175 / Aux 4,096），实测确认。
- 命名规则 `mortal{version}-b{blocks}c{channels}-t{YYMMDDHH}`；官方文档示例快照 **`mortal1-b40c192-t22040618`**；`config.example.toml` 默认 `num_blocks=40, conv_channels=192` → **b40c192 就是上游默认** `[官方]` `[S1,S9]`。
- 社区流传的 "v3/v4 用 b24c512"：来自**已删除的上游文档遗留项**；Mortal-Policy 的维护者明确说该设置**"不是推荐"** `[官方]` `[S1,S56]`。

**observation representation** `[CODE]` `[S10]`：v4 = **`(1012,34)`**（实测）。其 plane 组按顺序大致为：手牌/赤/点数/名次/场况（约 35）、宝牌与可见性（约 20）、**自家牌河 97**、**三家牌河 585**、概览类（牌河概览 28 + 副露概览 80 + 暗杠 4）、对手行为史（last_tedashis 9 + riichi_sutehai 9 + riichi_declared/accepted 6）、自身状态（waits/furiten/shanten/riichi_accepted[0] 共 10）、行动可行域（约 20），以及 **v4 单人（SP）搜索特征 123（包含在这 1012 之内）**。各组计数存在取整与归类差异，**唯一可依赖的总数是实测的 `(1012,34)`**；逐项精确值以 `libriichi/src/state/obs_repr.rs` 为准（首轮逐项枚举曾出现 14 个 plane 的口径差，故本报告不再给出"逐项相加 = 1012"的断言）。
- **关键反证（直接动摇 §五 前提）**：**弃牌顺序与时序已经被编码**——自家与对手牌河都按"前 6 张按序、最近 18 张按反序"的 turn-indexed 通道堆叠，另有全局 `exp(-0.2·(max_len-1-turn))` 的**时间衰减 plane**。代价是**同牌多次打出会互相覆盖**（碰撞），且自家牌河**不编码 tsumogiri/立直标记**（对手牌河有 tedashi/riichi）。**没有任何显式 genbutsu/safe-tile plane**，现物信息需从对手牌河概览隐式推导 `[CODE]` `[S10]`。
- **v4 的单人搜索特征（123 plane，已包含在 1012 内）**是 EV 表、进张需求表与 3×17 的和了/听牌概率表——即 **Mortal v4 已把"手牌向听/进张搜索"这类手工特征喂进网络**，评估"再加新特征/新头"的边际价值时必须先扣除这部分。

**action representation** `[CODE]` `[S10]`：`ACTION_SPACE=46`（实测）：`0–33` 普通弃牌、`34–36` 赤 5 弃牌（`5mr`/`5pr`/`5sr`，`libriichi/src/tile.rs:12-19`）；`37` 立直；`38–40` 吃（低/中/高）；`41` 碰；`42` 杠宣言；`43` 和；`44` 流局；`45` pass。**注意区分**：杠宣言（42）之后的"杠/暗杠择牌"是**另一套编码**——复用 `0–33` 的牌张索引重新置 mask（`obs_repr.rs:423-427`，`at_kan_select` 分支），不是 46 维动作空间里的新槽位；赤 5 候选在择牌阶段去赤映射后同样走 `0–33`（`obs_repr.rs:431-448`）。合法性为 bool mask，推理时 `q.masked_fill(~mask, -inf)`。

**reward design** `[CODE]` `[S10]`：奖励 = **GRP 预测期望顺位点的每局增量**：`Φ_k = (rank_prob_matrix @ pts)[player]`（`pts` 默认 `[6,4,2,0]`），`reward = Φ(x^k) − Φ(x^{k-1})`，并在终局把**真实顺位 one-hot** 拼进 GRP 输入做校正。Q 目标为 **MC 回报** `gamma^steps × kyoku_reward`，`gamma=1`，`steps` 只在 `apply_gamma` 的步上累计。**无 target network、无 TD bootstrap、无 twin-Q、无 entropy 项** `[CODE]`。

**offline RL 方法 / CQL** `[CODE]` `[S10]`：`cql_loss = logsumexp(q).mean() − q.mean()`（CQL(H)），权重 `min_q_weight=5`；**只在离线启用**；总损失 = `0.5·MSE(q, q_target_mc) + 5·cql_loss + 0.2·CE(aux_rank)`。
- **CQL 为什么有效**：离线数据只覆盖人类策略分布，`logsumexp − mean` 对**分布外动作**施加下压，抑制离线 RL 的**外推误差**（extrapolation error）导致的虚高 Q 值 `[推断]`（机制来自 CQL 论文 + 代码实现 + 社区复述）。注意：Mortal 实现的是**简化 CQL(ρ)**（单批、无采样动作集、无温度/拉格朗日），不是论文完整形式。

**人类牌谱如何进入训练** `[CODE]`：MJAI 格式 `.mjson(.gz)` 逐事件流；`file_index` 缓存；`file_batch_size=5`；`reserve_ratio`；`player_names_files` 可过滤牌手；数据增强 = **花色置换（m↔p）**（保留赤牌）。**v1 数据规模为"10–20M 半荘"的社区回忆，Equim 从未确认 v2–v4 规模** `[官方]` `[S6]`。

**online training / self-play** `[CODE]`：`server.py`（参数服务器 + replay buffer，capacity 1600 文件）/ `client.py`（worker：下载参数 → 用 `TrainPlayer` 跑 1v3 自对弈 → 提交日志）；`submit_every=400`、`save_every=400`、`test_every=20000`、`history_window=50`；**无 KL/trust-region**，online 时关 CQL、冻结 BN；`sample_reuse_rate/threshold` 是**死配置**（全仓库无引用）。**你的 fork 从未启用**。

**为什么使用 DQN 而不是 PPO** `[官方]` `[S2,S4]`：Equim 本人（discussion #30 comment 4222166, Maintainer）说明 **Mortal 开源前是 policy-based、用 PPO 训练，数月的调参没有击败一个监督基线**，于是转向 value-based。他强调的是**经验上足够/更可取**，**并明确不能排除自己实现有误**。社区后续三份独立重测**互相冲突**（hyskylord：~50M vs ~10M 局后 policy 反超；Nitasurin 的 policy fork 排序更高；另一测试者报告 policy"显著更差"）→ **"DQN 优于 PPO"在本案中无定论**。

**Mortal 的主要局限** `[官方]`：① **无一手 RL 文档**（issue #93，"Failed Attempts" 章为空）；② **从未发布自训权重**（"I have not released any weights trained by myself to the public yet"）`[S5]`；③ **三个未修正确性缺陷**：#115（kan 被排除在 gamma 折扣外，**已实证** `gameplay.rs:429`）、#111（分数/超车编码）、#114（`player.py` 里 `enable_rule_based_agari_guard` 关闭）——**上游 HEAD 已冻结，三者都会影响你的 pin** `[S7]`；④ online self-play 增益有限 + **灾难性遗忘**；⑤ 牌河编码有碰撞与不对称；⑥ **v4 obs 把分数 clip 到 30,000**（大点数局面信息有损）。

## 2.2 Suphx

**监督预训练** `[官方]` `[S11]`：五个模型**分别**用 (state, action) 对训练——Discard 15M / Riichi 5M / Chow 10M / Pong 10M / Kong 4M 样本（Table 3）。**数据来源需纠正**：论文正文只说 "top human players collected from the Tenhou platform"，**致谢只说 Tenhou 提供了 "expert game playing logs"**——**"鳳凰卓 7 段+"是三方推断，论文从未这样写** `[官方]`。**SL 数据的段位门槛无法一手证实。**
- 架构：输入 **34×838**（discard）/ **34×958**（其余四模型）；256 通道、**50 个 residual block**（约 100 卷积层）；手牌用 **34×1×4 的通道轴**编码；赤牌占 3 通道；另有 **100+ look-ahead 特征**（只对**自己**的摸切做 DFS）`[官方]`。

**policy gradient RL** `[官方]`：REINFORCE 式策略梯度 + **importance sampling**（应对异步分布漂移）+ **自适应熵正则**（α 自动调）。奖励 = **GRP 增量** `Φ(x^k) − Φ(x^{k-1})`。**只有 discard 模型用 RL，其余四个冻结在 SL**——常被忽略的重大局限。无 PPO clip、无 GAE、无 value loss 披露。

**self-play** `[官方]`：分布式（多 self-play worker + 参数服务器 + replay buffer，有界 staleness）；**论文未披露自对弈对手构成**（"grinding version / 历史 checkpoint"等说法无一手出处）。

**Global Reward Prediction (GRP)** `[官方]`：GRU 从"每局状态序列"预测**最终顺位分布**，训练在**人类牌谱**上（非自对弈），用作**稠密奖励**——因为原始局分/终局奖励信号太差。

**Oracle Guiding** `[官方]` `[S11,S17]`：训练时给"完美信息"的 oracle，再把特权信息**退火掉**。**必须纠正一个常见误读**：论文**只声称它是加速器**（"learning could be very slow"），**并没有说"纯 RL 从零会失败"**，也**没有任何 from-scratch 失败的 ablation**。其理论后续是 **VLOG（ICLR 2022）**——对 KL 学一个对偶变量（`kld_target=50`），并发布麻将离线数据集。

**人类牌谱在 Suphx 中的角色**：① 五个模型的 SL 初始化；② GRP 的训练集。**没有**把人类数据混进自对弈 replay。

**为什么有效** `[官方]`：SL 提供强先验（跳过冷启动），GRP 把稀疏终局信号变成每局稠密信号，oracle 加速早期学习，自对弈再往上顶。消融（Fig. 8）显示 SL → RL-basic → RL-1 → RL-2 逐级提升。

**结果与算力**：9 段 1,652 局（2019-04-24）、十段 5,137 局（2019-06-22 达成，官方博客 06-24 公告）、**稳定段位（安定段位）8.740** `[官方]` `[S13,S14]`；**离线自对弈 ~2,000 万局 vs 在线仅 5,760 局**（Tie-Yan Liu, PingWest `[S15]`）；**每个 agent 用 44 GPU × 2 天**（40×K80 + 4×Titan XP，评估 20×K80×2 天）`[S16]`；Suphx = RL-2 ≈ 250 万局 `[官方]`。
- **"5 年 / 一张 3090" 的准确含义**：一手来源是 kanachan 作者 Cryolite 的 gist（2022-04-15）——**指 Suphx 训练所用的 150 万半荘"模拟"在 RTX 3090 上约 5 年、电费 40 万日元**，是**电算式估算**（≈34 半荘/时，与 Mortal 宣称的 40k 半荘/时差 ~1000×，说明瓶颈在**朴素逐决策 NN 推理**而非游戏逻辑）`[官方]` `[S18]`。**它不能作为"Suphx 级自对弈不可行"的一般化论证。**

## 2.3 NAGA

`[官方]` `[S19,S20,S21]`
- **路线**：纯监督模仿（Dwango 官方文章与问答只描述"用公开鳳凰卓牌谱做监督学习"，**无 RL**）。*注：此前有引证误用 Suphx 论文作为"NAGA 不用 RL"的依据——全文 grep 确认 **Suphx 论文根本不含 "NAGA" 一词**，该引证无效。*
- **架构**：**无任何公开的架构论文/技术报告**。
- **数据**：官方只称使用公开对局；具体规模未公开。
- **强度**：**天鳳特上卓十段**（已确认；但**AI 账号被禁止进鳳凰卓**，故其十段**不可与鳳凰卓安定段位直接比较**）；官方称当前五种类型的模型明显强于 2021 年的十段模型（无公开数值），社区估"约九段当量"（数值仅二手）。**"26,598 局/2021-07"无法从任何可检索源证实**。
- **恶手率（日文原文：悪手率）**：官方定义 = 与推荐匹配度 **<5%** 的弃牌占比；开发者承认 **5% 阈值是任意的**；**不含立直/鸣牌决策**。
- **相对优势/局限**：优势是**极简单、可解释、以行为一致度作为可交付指标**（何切る 教学）；局限是① 天花板受人类教师分布限制（无探索机制）；② 只有"行为匹配"没有"期望胜负"目标；③ 无公开强度数字，无法严谨对比。

## 2.4 LuckyJ

`[官方]` `[S22,S23,S24,S25,S26,S27,S28]`
- **实体**：**腾讯 AI Lab"绝艺LuckyJ"**（天鳳账号 `ⓝLuckyJ`），属绝艺决策 AI 家族；负责人 **Haobo Fu（傅浩波）**。
- **路线（1v1 版本）**：**零人类数据、从零自对弈**（4 人立直版本仅有新闻稿）；策略优化 = **ACH（Actor-Critic Hedge）**，由 **NW-CFR**（神经加权 CFR）衍生——本质是**遗憾最小化的 actor-critic**（Hedge + 采样优势，避免 `1/f_p(s)` 的方差爆炸），而非 "Deep CFR"；另有**独立的不完全信息搜索**（腾讯描述为"基于乐观价值估计"，树变换+剪枝），且**搜索结果作为"特征"喂入策略网**（而非直接选动作）——这是其最值得借鉴的架构思想。开源实现：`Liuweiming/ACH_poker`（Apache-2.0）、`yata0/Mahjong`（MIT）。
- **战绩**：2023-05-30 特上卓十段，**稳定段位 10.68，仅 1,321 局**；vs Suphx p=0.02883。
- **对你文档的纠正**：① "Neural CFR" 不是其方法的准确名称（ACH 属遗憾最小化谱系）；② 搜索与策略是**两个组件**；③ **ACH 的同行评审验证只在 1v1 麻将**（ICLR 2022），4 人立直版本①无论文②无代码③其搜索方法（ICML 2023 的 OLSS）**只在 2 人麻将验证**；④ 常被引用的"1v1 击败世界冠军"实为 **margin 0.82±0.96、p=0.19（单尾）不显著**，作者主张是"冠军无法剥削它"`[评审]` `[S28]`。
- **CFR 类为何"可能"适合麻将**：遗憾最小化在多人非零和里**没有收敛保证**（`[评审]` `[S34,S35]`），但它产生**更不易被剥削的混合策略**且**模型自由**（不需模拟器遍历所有动作——这正是 Deep CFR 在麻将不可行而 ACH 可行的关键）。因此"CFR 理论上更适合麻将"是**过度解读**：它的成功是经验性的，理论只"部分继承"（腾讯自己的措辞）。

---

# 三、核心研究问题：Pure Zero 是否值得做

**一句话结论**：对四人立直，"零人类数据 + 零塑形 + 随机初始化 → 顶尖人类强度"**没有被任何同行评审工作证明可行，也没有被证伪**；已证明可行的从零案例**都依赖塑形或 oracle**。Pure Zero 的正确价值是**科研 baseline**，不是达到强度的最短路径。

1. **Reward sparsity**：立直每局都有小分变动，但真正的目标是**顺位/最终点数**，与单局得失并不一致——用终局奖励则极度稀疏，用局分奖励则目标错位。Suphx 因此必须发明 GRP；Mortal 的奖励本质就是**学习出来的塑形**（GRP 期望顺位点增量）`[CODE]`。DouZero 证明终局奖励在小规模牌类可学 `[评审]`，但立直信息集（~10⁴⁸ 平均）远大于斗地主（~10³）。
2. **Exploration difficulty**：无人类先验时，要在"立直/副露/弃牌"巨大组合空间同时发现攻防；实测显示**派生特征（向听/进张）从零训练显著有害**（−73.9，≈4.8σ；机制=副露崩塌：副露率 95.5%、立直率 1.2%）`[单源]`。
3. **Cold start**（最硬）：纯自对弈 Elo **1118** vs 人类先验 BC **1443** vs Mortal **1465**；纯自对弈接近 BC 冠军需 **130–500M 局** `[单源]`。对照 RVR 从零 3 天/8 GPU 达"前 16 的 62.5%"`[评审]`——但**带了 oracle 全局信息价值网 + 学习型奖励**。
4. **Hidden information**：无法反事实地重放信息集——这正是 AlphaZero 式方法失效的根因（见下）。
5. **Opponent modeling**：从零的自对弈对手就是自己的历史，缺少**人类策略多样性**（human prior ≫ pure self-play：0.7966 vs 0.2034 `[单源]`）。
6. **Non-stationarity**：多人自对弈无 Nash 收敛保证，收敛到**中介/相关均衡**（需 constant-sum-polymatrix + subgame-stability 才更强）`[评审]`。**理论张力**：立直**总点数守恒**（constant-sum）是有利结构，但"rank-based 得分（马/冲）"使**效用层面是 general-sum**。
7. **Multi-agent instability**：纯自对弈会周期性退化；本战役唯一的正面 RL 结果必须靠 **forward-KL 缰绳**才稳住 `[单源]`。
8. **Sample efficiency**：Suphx 离线 2,000 万局 + 44 GPU×2 天/agent；纯自对弈实测标度律 **对数线性 ~+1.5–2pp/倍增**，0.356 → 0.50 需 **7–9 次倍增（1.3–5 亿局，$1.5k–6k）** `[单源]`。
9. **Computational cost**：单卡"150 万半荘模拟 ≈ 5 年"只是**朴素实现的电算估算**（瓶颈是逐决策推理）`[官方]`；GPU 向量化环境能显著压低该项（Mahjax 声称 8×A100 下 env-only 2M steps/s），**但尚无消费级 GPU 实测** `[评审]`。
10. **是否需要 reward shaping**：**需要，但要小心**。已证明的系统**均使用塑形或学习型奖励**（Suphx GRP / Mortal GRP 增量 / RVR 学习型期望奖励网）。而从零条件下**手工派生特征塑形显著有害**、**PBRS 向听塑形只改变到达平台的速度、不改变平台高度**；非势能塑形会**改变最优解** `[评审]`。→ 塑形应放在"**学习出来的、对齐最终目标的**"形式（GRP 类），而非手写指标。
11. **是否需要 curriculum learning**：证据薄弱。现有最强课程是"人类先验 → 自对弈"这一**隐式课程** + 联赛式对手池（PFSP/历史 checkpoint）；没有证据支持显式阶段课程有额外收益 `[推断]`。
12. **是否需要 MCTS/CFR/search**：**不是必需，但也非无用**。"4 人立直无法搜索"是**错误**的（见 §七 尾声）；但**所有已知搜索型方案的强度上限都未达 Suphx 级** `[评审]`。
13. **是否存在类似 AlphaZero 的可行路径**：**不存在"直接照搬"的路径**（理论硬伤见下）。

**为什么 AlphaZero 能从零学围棋，麻将 Zero 有本质区别？**

| 维度 | 围棋 / AlphaZero | 四人立直 |
|---|---|---|
| 信息结构 | 完美信息 | **不完美信息** |
| 动作价值含义 | `Q(s,a)` 唯一 | **`Q(s,a)` 依赖"该动作被选中的概率"**（策略化）→"用搜索访问计数当策略目标"**理论上不成立** `[评审]` |
| 搜索 | 正规博弈树、可复用子树 | **回合顺序不固定**（吃/碰/杠/立直打断），**无法构造正规博弈树**，MCTS 不能直接套用 `[官方]` |
| 信息集规模 | 10¹⁷⁰ 状态但结构可枚举 | **~10⁴⁸ 平均信息集**、总量 10¹⁴² 量级 `[评审]` |
| 对手 | 单一对手（自己） | 3 个同时学习的对手 + 随机摸牌（运气方差大） |
| 博弈类型 | 两人零和，有收敛保证 | **4 人 general-sum（效用层面）**，无 Nash 收敛保证 `[评审]` |
| 奖励 | 胜负清晰 | 顺位制、非传递、需要塑形 |

正确表述不是"麻将不能有 Zero"，而是："**麻将不能直接照搬 AlphaZero 的理论框架；zero-style 必须重新定义价值/搜索/均衡目标**（如 α-Rank/α-PSRO 这类多人解概念，或 regret-minimization 家族）。"

---

# 四、Human → RL → Self-play 是否更合理

## 4.1 四路线对照

| 维度 | **A: Pure Zero** | **B: Human → SL → self-play RL** | **C: Human → Offline RL → self-play** | **D: Human + Self-play 混合 replay** |
|---|---|---|---|---|
| 最终性能（证据） | 未证明；纯自对弈实测 Elo 1118 | **最强**（Suphx 十段/8.740） | **强**（Mortal 为最实用开源基线） | 强，比例未定 |
| sample efficiency | 极差（1.3–5 亿局） | 好（跳过冷启动） | **最好**（直接用现成牌谱） | 中 |
| training stability | 差（需 KL 缰绳） | 中 | **好**（CQL 离线稳定） | 中 |
| 探索能力 | 理论最强（无人为偏置） | 弱（受人类先验束缚） | 弱于 A | 中 |
| 策略多样性 | 高（但常退化） | 低-中 | 中 | 中-高 |
| 人类策略 bias | 无 | **高** | 高 | 中（可调） |
| human ceiling | 理论无 | 有，但自对弈可突破（8.74 > 人类 ~7.46） | 同 B | 同 B |
| 训练成本 | **最高**（同强度下） | 高（44 GPU×2 天/agent） | **最低**（单卡可行） | 中高 |

## 4.2 两个目标下的选择

- **目标 = 最强麻将 AI（单卡 + 250 万局）** → **选 C（你正在做的），在证据允许时叠加 D**。在现有证据下，C 是本报告中证据最强、单位算力产出最高的路线；A 在你的硬件上需 1.3–5 亿局才接近人类先验 BC 的水平，性价比极低。**但必须修正 C 的实现**：你现在没有验证集、指标饱和、训练数据无留出——"提升"不可测。
- **目标 = 最有科研价值** → 两条并行：① 保留**小额预算**的 Pure Zero 侧臂（10⁷–10⁸ steps，明确记录 scaling），用于回答"人类先验的价值有多大"；② 主战场放在**尚未被任何人占据**的问题上（§十一 给出 3 个候选，最强候选是"**在 4 人立直上把 regret-minimization（ACH 系）与 value-based 做等预算头对头**"——ACH 唯一评审证据是 1v1，其 4 人价值完全未知）。

---

# 五、Transformer 是否真的有必要

1. **Mortal 现有 feature plane 是否已隐式编码牌河历史？** —— **是，而且相当充分** `[CODE]`：弃牌按 turn 索引（first-6 顺序 / last-18 反序）+ 全局 recency-decay plane + 对手牌河的 tedashi/riichi/赤/dora 标记 + 牌河概览。**唯一缺陷**是"同牌重复打出会互相覆盖"与"自家牌河无 tedashi 标记"。另有自算估计：**约 72%（≈728/1012）plane 与牌河/弃牌历史相关** `[推断]`（自算、无外部枚举，应视为"论证"而非"测量"）。
2. **Transformer 能否提供额外信息？** —— **理论上能补上"被覆盖的顺序信息"与"跨家事件关系"，但没有任何立直实验证明这转化为强度** `[评审]`：立直领域最"Transformer 化"的项目 kanachan **不发布任何强度数字**；唯一等参数真实对局对比（Tjong，国标）中 Transformer 只打平 ViT/GRU。
3. **river token 应如何定义？** —— **证据支持"单一全局时间序事件流 + 显式 actor embedding"**（kanachan 与 Mahjax 都如此），因为吃/碰/杠/立直会打断回合顺序，**跨家交错本身携带信息**。
4. **每次弃牌一个 token，还是每玩家一条序列？** —— **推荐"每原子动作一个 token 的全局流"**（token 可打包 tile/动作类型/是否摸切/巡目）。pymahjong v5 的实现即"**每个原子动作一个 100-dim packed-bitset token，19 种事件类型**" `[CODE]`。
5. **是否需要 positional encoding？** —— **需要**（否则顺序丢失）。注意 **pymahjong v5 默认 `use_pos_emb=False`**（靠事件 token 内含信息），**kanachan 明确要有 PE**（wiki：无 PE 则 progression 序列顺序丢失）。工程上建议**显式加**并做 PE 消融。
6. **如何处理 chi/pon/kan/riichi/tsumo？** —— 作为**独立事件类型**编码进 token（含被鸣的牌、谁对谁鸣、是否立直宣言牌）；**不要**把 TSUMOGIRI 记成"不带牌的 action id"——**Mahjax issue #12 正是这个 bug，会让观测变成非马尔可夫**（该 issue 现已修复）`[CODE]`。
7. **Transformer 的计算成本是否值得？** —— **在当前证据下不值得优先投入**：无立直强度证据、无消费级 GPU 实测、收益边际。**更便宜的替代**是修 Mortal 现有编码的两个缺陷（碰撞、自家牌河缺 tedashi），成本几十行 Rust。
8. **如何设计 ablation？**（8 条不可省的对照）`[评审]`：① **等参数**；② **等算力/等训练帧数**（AlphaStar Unplugged：无记忆 1e10 帧更好、Transformer 到 2e10 帧才反超）；③ 等数据/等预训练；④ **frozen encoder + 只训 fuse/head**；⑤ **打乱历史控制**；⑥ **参数匹配的 LSTM / 无记忆 MLP 对照**；⑦ **多 seed + IQM + 分层 bootstrap CI**（"5 是最低要求，宣称优越性需 10+"）；⑧ 以 **duplicate 对局强度** 而非动作精度为最终指标。

---

# 六、Opponent Hidden State / Belief State

1. **设计是否合理？** —— 作为表征设计合理，作为**强度来源的假设缺乏证据**。三家独立团队的同向 null（Mortal 作者试过后**移除** oracle 观测与排位辅助头；hongdp 的 oracle critic/听牌 aux 头均 null——exp69 为 **3×34** 待张多标签头，头本身学到了（`aux_sep` 0.44），但对局 A vs P **0.4938±0.0035** = 无增益；SuuTTT 的听牌 AUC 0.887 **没换来分数**，且其 `belief-weighted PIMC` 被实测为**有害**——该报告给的机制是 "search exploits the belief model's systematic errors"）。
2. **hidden state 是否应解释为 belief state？** —— **是**。更准确的术语是"**对隐藏状态的学习型充分统计量**"。它不是"网络猜的对手真实手牌"。
3. **是否应显式预测对手手牌分布？** —— **不建议作为第一优先级**：① **没有超人级不完信息 AI 这样做**（Pluribus/Libratus/DeepStack/ReBeL 都是"对自己策略做 Bayes"得到 belief 喂给 value/search）；② **标签部分可观测**（只有"他家荣和"才暴露手牌）——社区危险度预测的 AUC 上限 ≈0.83 且为自发布非评审；③ **每对手独立 34×5 计数头会与"每种牌 4 张"约束联合不一致**，而已知生成式工作都**联合建模 4 家**（4×2×34）。
4. **如何避免错误的 certainty？** —— 用**校准指标**（ECE/Brier）而非准确率；用**分布/粒子表示**；对辅助头**低权重 + 退火**；并且**用下游强度判优劣**（SuuTTT 的 0.887 AUC 换不来分数即反例）。**"校准"不是处女地**：Sasaki 等的扩散手牌模型已报告 calibration curve + Platt scaling + 加权 ECE（扩散 0.00188 / 自回归 0.00390 / 直接预测 0.000801）`[S63]`；note.com 的立直局结果模型报告 multiclass ECE 0.008511 `[S66]`；VLOG（ICLR 2022）已在麻将上学隐状态**分布**但未报校准 `[S64]`。**尚未见先例的是**"校准感知的训练目标（proper scoring rule 当 loss）+ 对局强度验收"这一**组合**——若做，先跑 §9.2 的 **A2b 零训练校准审计**。
5. **如何训练 opponent belief？** —— ① **隐式**（RL 中自行形成；Suphx 的 oracle guiding 本质就是隐式 belief 学习）；② **显式辅助损失**（证据 null 居多）。**若要做，先做线性探针**：检验 Mortal 现有 1024-d embedding 是否**已经**能解码出对手向听/危险度——若能，加专用头是冗余（红队给的最便宜证伪实验）。
6. **是否用 Bayesian / particle / distributional？** —— **理论上更对**（DeepStack 的 belief refinement 即 Bayesian 粒子），但**在麻将规模下没有成功案例**；成本是主要障碍。
7. **是否需要 opponent-specific recurrent state？** —— **不是必需**。已定位的前例项目中，对手建模模块都是 recurrent 的，但超人系统（Pluribus）**完全没有**对手特定状态。**先证明"共享对手编码 + 座位/角色 embedding"不够，再上 per-opponent 状态**。
8. **Transformer 还是 RNN？** —— **任务相关**。Transformer 在 POMDP 状态追踪上有**已被证明的表达力差异**（RNN 需 Ω(N) 宽度做 associative recall，两层 Transformer 不需要——不是"Transformer 更差"，而是分工不同）；AlphaStar Unplugged 显示**低算力下无记忆反而更好**。**建议：对手序列先用 RNN/LSTM（便宜），Transformer 留给全局事件流**。
9. **是否应预测 34×5 tile-count distribution？** —— **不要用三个独立头**（联合不一致）。若要做，**联合建模四家**（如 4×2×34 的"每张牌是否在某家手上"），并保证"每张牌总数 ≤4"的可行性投影。
10. **belief 最终应如何影响 policy/value？** —— **证据支持"喂给 value/search"，而非"直接条件化 policy"**：在**已考察的**超人案例中，belief 的消费者都是 value 或搜索根节点。这与 LuckyJ 一致（**搜索结果作为特征喂回策略网**）——**这是本报告认为最值得借鉴的单一架构思想**。**但有一个已实测的警告**：SuuTTT 把 belief 接入 PIMC 搜索（`belief-weighted worlds`）后**变差**，其归因为"搜索会利用 belief 模型的系统性误差"`[S46]`——**在 belief 未校准前把它喂给搜索会放大错误**，这与本节第 4 条的校准审计互为因果。

---

# 七、PPO vs DQN vs AWR vs CFR

## 7.1 对照（你要求的 8 个维度）

| 算法 | 离线数据兼容 | self-play 兼容 | 隐藏信息 | 动作空间（变长+mask） | 探索 | 稳定性 | 算力 | 样本效率 |
|---|---|---|---|---|---|---|---|---|
| **Dueling DQN + CQL** | **最好**（Mortal 已证） | 需改造（off-policy，易遗忘） | 无特殊处理 | 天然（Q 逐动作打分） | ε-greedy/boltzmann | **好**（离线） | **低** | **高** |
| **PPO** | 差（on-policy） | **好** | 无特殊处理 | 需 mask | 熵正则 | 中 | 中 | 低 |
| **AWR/AWAC** | **好**（专为此设计） | 好（可微调） | 无 | 需 mask | 依赖数据 | 中 | 中 | 中 |
| **PG（Suphx 式）** | 需 SL 初始化 | **好** | oracle guiding（隐式） | 需 mask | 自适应熵 | 中（异步） | 中 | 中 |
| **CFR / 神经 CFR** | 弱（需遍历） | 中（self-play 天然） | **原理正确** | **10⁴⁸ 信息集 → 全树不可行** | 内在（遗憾） | 中 | **高** | 低 |
| **ReBeL** | 弱 | 中 | **正确（belief 空间价值搜索）** | 仅两人零和（2p0s）有保证 | — | — | **很高** | 低 |
| **MuZero** | 中 | 好 | **论文自述 "not directly addressed"** | MCTS 不可直接套用 | — | 中 | 高 | 中 |

## 7.2 推荐："如果从 Mortal 开始改，下一步最应该尝试什么？"

**按证据强度排序（不是按时髦度）：**

1. **先修评估与数据管线，再谈算法**（§九）。指标无信息量时换算法等于盲飞。**唯一零风险、必做的一步。**
2. **做一次"简单 actor-critic + KL 缰绳"的小规模自对弈实验**，而不是直接上 PPO。理由：本战役最强经验证据是 **PPO 失败 18 次，而 REINFORCE + Huber 价值基线 + forward-KL(λ=0.05) 成功**；且该成功**依赖训练良好的价值基线**（同仓库另一条线用 PopArt 修好了 value 却仍失败，说明 **KL 缰绳**很可能是关键）`[单源]`。
3. **若要在算法层做真正有增量的研究：做 ACH（regret-minimization）在 4 人立直上的等预算头对头**——**空白**（唯一评审证据是 1v1 且不显著），已有开源实现可抄。
4. **AWR 只在"想从离线平滑过渡到在线"时才试**，不要指望它比 CQL 强（唯一公开比较是"略低于 CQL"，n=1）。
5. **不要为"现代"而换 PPO**；也不要因"Mortal 用 DQN 失败过 PPO"就断言 DQN 理论更优（作者本人不这么说）。

**关于搜索**：若要做搜索，正确做法**不是** vanilla MCTS/CFR，而是**在 2 人零和之外重新定义问题**——**α-Rank/α-PSRO 是目前已知可推广到 general-sum 多人游戏的解概念（之一）**，并已在 **3–5 人扑克**上优于近似 Nash 求解器；它是**元求解器（策略群体的均衡），不是逐决策搜索**，成本 O(N_t²) 元博弈内存——对麻将规模**可行性未知但值得一试**。此外，"4 人立直不能搜索"是错误说法：MeowCaTS（重要性采样 MCTS）夺 Computer Olympiad 2023，VeryLongCat/KF-TREE 亦为搜索型夺冠——**可辩护的弱形式是"vanilla MCTS 不能直接套用，且尚无搜索型 4 人立直达到 Suphx 级强度"**。

---

# 八、新模型设计

> **设计原则（由证据倒推，而非由愿望倒推）**：
> ① 保留 Mortal 已被验证的部分（CNN 编码、CQL 离线、GRP 奖励、duplicate 评估）；
> ② **任何新增组件都必须先有"零成本证伪实验"**（§九）；
> ③ **不引入已被三家独立团队证伪的组件**（显式对手手牌预测、oracle 头、return-conditioning）；
> ④ 先在**单卡可承受**的尺度上做等预算对照，再决定是否放大。

## 8.1 数据流与 tensor shape

```
① Input
   ├── obs        : float32  (B, 1012, 34)      # 沿用 Mortal v4；P1 可先修两处缺陷
   ├── mask       : bool     (B, 46)            # 合法动作
   └── events     : int32    (B, L=256, 8)      # ★新增：全局时间序事件流
        # 每 token 8 个字段 = [tile_id(0-36), action_type(0-18), actor(0-3),
        #   turn, is_tsumogiri, is_riichi, is_called_by, reserved]
        # 事件类型 19 种：摸牌/弃牌(手切|摸切)/吃/碰/明杠/加杠/暗杠/立直宣言/
        #   立直成立/和了(自摸|荣和)/流局/宝牌翻出/点数变动 等
        # 长度 256 覆盖极绝大多数半荘；超长截断（记录比例，若 >1% 再处理）

② Encoder（沿用 Mortal）
   CNN(1012,34) → h_cnn : float32 (B, 1024)

③ River/Event Transformer（★唯一新增的表示学习组件，先做小）
   Embedding(8 字段 → 各 16d，concat 96d) + PE(learned)
   → 2~4 × TransformerEncoder(d_model=128, heads=4, ffn=512, dropout=0.1)
   → masked mean pool → h_river : (B, 128)

④ Opponent Belief（★默认关闭，作为消融臂而非核心）
   共享编码 + 座位 embedding → h_opp : (B, 3, 128)（可选，若探针证明必要）
   不预测 34×5 手牌分布；只保留**可选**的低权重辅助头（默认 off）

⑤ Fusion
   concat[h_cnn(1024), h_river(128)] → LayerNorm → Linear(1152→1024) → Mish → h (B,1024)

⑥ Heads
   Policy/Q 头 : Linear(1024 → 46)                    # 离线阶段直接沿用 Q + mask
   Value 头    : Linear(1024→256) → Mish → Linear(256→1) → V(s)
   Aux 头（低权重，默认只保留 Mortal 已有的 rank 头）
     ├── next_rank : Linear(1024 → 4)   CE 权重 0.2（Mortal 已有，保留）
     ├── shanten   : Linear(1024 → 7)   权重 0.05（可选，仅离线用）
     └── （不做）opponent hand distribution / oracle / danger 预测

⑦ 训练目标
   离线（阶段 A，主）：L = 0.5·MSE(Q, q_target_mc) + 5·CQL(H) + 0.2·CE(rank)
       q_target_mc = γ^steps · 每局 GRP 期望顺位点增量（沿用 Mortal，γ=1）
   在线（阶段 B，仅在 A 达标后）：L = PG − λ_v·Huber(V, R) + λ_kl·KL(π‖π_frozen)
       R = 本局最终 raw 点（terminal-only，per-seat 标量），batch 内 z-normalize
       λ_v = 1.0，λ_kl = 0.05（forward KL，= 已验证的正向配方）
       **无 PPO clip、无 GAE、无 entropy 项**（先复现已验证配方，再单变量微调）

⑧ Self-play 算法（阶段 B）
   - 4 个 learner seat 全学；对手池 = {历史 checkpoint, 冻结 anchor, 少量启发式}
   - 每 N 次更新把当前策略快照入池（PFSP 风格权重）
   - anchor 固定为阶段 A 的离线模型，KL 缰绳把它当"信任域中心"
   - 规模：先在 5–20M 步量级验证"是否比 anchor 更好"，再谈放大

⑨ Replay buffer
   - 阶段 A：人类牌谱（MJAI）→ 复用 Mortal 的 dataloader/索引（唯一改动：**
     显式划分年份 holdout**，见 §一/§九）
   - 阶段 B：recent self-play(70%) + 历史 checkpoint(20%) + 人类(10%) 作为起始
     比例（**ASSUMED，必须通过消融调**；hongdp 证据显示人类比例对早期强度至关重要）

⑩ Evaluation（见 §九）
   - duplicate 1v3/2v2（libriichi 自带），≥6,100 半荘才能分辨 Δavg-rank 0.05
   - 多 seed + 分层 bootstrap CI；对手固定为 akochan 与冻结的 Mortal 基线
```

## 8.2 该设计的"证据对齐"说明

| 组件 | 证据状态 | 处理 |
|---|---|---|
| Mortal CNN + CQL 离线 | **已验证**（Mortal 本身） | 直接保留 |
| GRP 奖励 | **已验证**（Suphx + Mortal） | 保留 |
| 全局事件流 Transformer | **无强度证据**（立直领域） | 作为**可证伪的消融臂**，预算受限 |
| per-opponent belief | **无强度证据 + 三家 null** | **默认关闭**，只在探针证明 embedding 无法解码时才做 |
| 对手手牌分布预测 | **被证否/无先例** | **不做** |
| oracle heads | **三家 null + 作者移除** | **不做**（仅作为"加速器"实验，且不作为强度来源） |
| 在线 RL | **唯一正例：REINFORCE + Huber + KL=0.05** | 复现该配方，不直接上 PPO |
| 搜索 | 可行但未达 Suphx 级 | 只作为独立研究支线（α-Rank/ACH 方向） |
| 容量放大 | 容量扫描全 null | **不放大**，先做等预算对照 |

---

# 九、实验设计

## 9.1 统计功效（先解决"测不出来"的问题）

**DERIVED**（Wave3 评估统计轴推导）：duplicate 对局的挑战者 rank-pt 每半荘标准差 **σ ≈ 85**（`[90,45,0,−135]` 标度；公开模型卡佐证下界 σ≈68）。
- 必要样本量 `n = (z_{1−α/2} + z_{1−β})² σ² / Δ²`；80% power、95% 置信下：
  - Δ = 0.10 / 0.05 / 0.02 rank-pt → **≈5.8M / 23M / 145M 半荘**（**ASSUMED** 依赖 σ）
  - 换到"平均顺位"单位（1↔4 名互换给出 Δpt = 75·Δrank）：**n ≈ 10.3 / Δrank²**
    → Δavg-rank 0.05 → **≈4,100 半荘**；0.02 → **≈25,800 半荘**
- **标度不影响 n**：纯缩放下 σ²/Δ² **近似不变**（各标度 ≈0.138–0.146，残差来自取整与 uma 结构）——**改 uma/oka 不是杠杆**。
- **你的现状**：2000 半荘只能分辨 **Δavg-rank ≈ 0.072**。P3 的 +4.11 rank-pt（≈0.055 avg-rank）的**显著性取决于检验口径**：按**单样本**（SE≈1.90）算得 z≈2.16、CI 下界 >0；按**两名挑战者之差的保守双样本口径**算得 z≈1.60、95%CI ≈[−0.92, +9.14]（跨 0）。**正确的做法是配对 duplicate 检验**（同一 seed 下两模型的局面配平），其方差应低于上述两种口径；**在该检验完成前，不应宣称 P3 通过**。按双样本口径，要确认需 **≈6,100 半荘**（单样本口径约 4,100）。

**结论：在换任何架构之前，先把评估扩到 ≥6,100 半荘并固定协议**（否则所有"提升"都无法判定）。

## 9.2 Ablation 矩阵（16 项，按优先级）

**第一优先（零/低成本证伪，必须先做）**

| # | 实验 | 控制/对照 | 指标 | 判定 | 淘汰条件 |
|---|---|---|---|---|---|
| A1 | **打乱弃牌顺序**（推理期 shuffle 牌河 plane/事件） | 原模型自身 | 1v3 顺位 | 强度不变 ⇒ 顺序未被使用 | 若不变，**整条 river-transformer 命题失去 headroom** |
| A2 | **线性探针**：Mortal 1024-d embedding 解码对手向听/危险度 | 随机标签基线 | AUC/ECE | 能解码 ⇒ 专用 belief 头冗余 | 若已解码，砍掉 §六 所有显式头 |
| A2b | **校准审计（零训练）**：A2 探针概率 + GRP 24 排列 + Q/价值 的 reliability curve + ECE/Brier | 随机标签基线 + Sasaki 公开 ECE（0.00188 / 0.000801）`[S63]`、note.com（0.008511）`[S66]` | ECE/Brier | 校准差 ⇒ belief 的校准目标有正当性 | 已良好校准 ⇒ 关闭 belief 头路线 |
| A3 | **评估扩样**：2000 → 6,100 半荘 | 同挑战者/冠军 | z、CI | 复现 P3 是否显著 | 若仍不显著，说明此前"进步"是噪声 |
| A4 | **holdout 修复**：划出独立年份/牌手做验证 | 现状（无 holdout） | 验证 loss 曲线 | 暴露"训练 loss 下降但泛化不涨" | — |

**第二优先（架构与算法，等参数/等算力）**

| # | 实验 | 控制 | 指标 |
|---|---|---|---|
| A5 | Mortal baseline 复现（先对齐上游配置） | — | 1v3 顺位达到预期 |
| A6 | Mortal + River Transformer（**等参数**：把 CNN 缩到同参数量） | A5 | 1v3 顺位、Δavg-rank |
| A7 | Mortal + River Transformer（**等算力/等帧数**） | A5 | 同上（防"参数量假象"） |
| A8 | River-only（去掉 CNN） | A6 | 同上（验证"CNN 是否可被替代"） |
| A9 | 打乱历史控制（shuffled-history arm） | A6 | 若 A9 ≈ A6 ⇒ 模型没用顺序 |
| A10 | 参数匹配的 LSTM 替代 Transformer | A6 | 1v3 顺位、Δavg-rank（比对 attention vs 其它序列算子） |
| A11 | Mortal + per-opponent belief（默认关闭的消融臂） | A5 | 1v3 顺位、Δavg-rank（前置条件：A2 证明 embedding 无法解码） |
| A12 | 容量扫描（16/24/40 blocks × 128/192/256 ch） | 等 token 预算 | 1v3 顺位、Δavg-rank（验证"更大≠更强"） |

**第三优先（训练范式，仅在阶段 A 达标后）**

| # | 实验 | 配方 | 判定门槛 |
|---|---|---|---|
| A13 | 离线 CQL（阶段 A）→ 冻结 | Mortal 配方 | baseline |
| A14 | **简单 actor-critic + forward-KL(0.05) 自对弈**（= 唯一正例配方复现） | REINFORCE + Huber(V) + KL | 相对 A13 的 anchor：**placement 95%CI 下界 > 2.500**（复现该战役的判定规则） |
| A15 | PPO-clip 对照（同预算） | clip + entropy + PFSP | 与 A14 等预算头对头（预期失败，但必须实测） |
| A16 | ACH（regret-minimization）对照 | NW-CFR 系 actor-critic | 与 A14/A15 等预算头对头（**这是真正的科研空白**） |

**可淘汰的实验**（不做）：reward shaping 的手写指标变体（已证有害）；更大模型（已证 null）；独立 per-opponent 34×5 头（约束不一致）；return-conditioning（随机环境可证失败）；显式 oracle 头（三家 null）。

## 9.3 指标与协议

- **主指标**：duplicate 1v3 的 **平均顺位**（Δavg-rank）与其 95%CI；辅以 **rank-pt**（注明标度）。
- **副指标**：和牌率、放铳率、立直率、副露率、**平均和了番数**（本次正例的唯一显著变化量）、策略熵、与人类策略的 KL。
- **必须**：多 seed + 分层 bootstrap CI + **选择/确认集分离**（本战役有 +0.0055→+0.0006 的 winner's curse 前车之鉴）。
- **禁止**：用动作预测精度代替对局强度（Tjong/SuuTTT 的反例）；用饱和的 `test_play` 当进度（你的现状）；用"赢过随机初始化基线"当证据。

---

# 十、挑战假设（逐条判定）

> 本节采纳红队 Oracle 的独立判定（原文：`raw/RED.txt`），并交叉本报告证据。

| # | 假设 | 判定 | 最强反证 | 最便宜证伪 | 可能被推翻的因素 |
|---|---|---|---|---|---|
| 1 | "Transformer 一定比 CNN 更适合牌河" | **REFUTED** | Mortal v4 已编码顺序（first-6/last-18/recency）；唯一等参数真实对局对比（Tjong）中 Transformer 打平 ViT/GRU；SuuTTT tile-transformer val-acc 最高但对局 tie-or-below、集成 −57/game；kanachan 无强度数字 | A1（打乱顺序对照） | 若"前例清单"轴出现"立直 Transformer 有公布强度"的线索 → 降为 CONDITIONALLY SUPPORTED |
| 2 | "Opponent hidden state 一定有用" | **UNRESOLVED** | Pluribus 超人且**不做**对手适应；Libratus 刻意避免剥削；无顶级 AI 把 belief 直接给 policy（都走 value/search）；Mortal 已有 AuxNet/oracle/GRP | A2（线性探针） | 若探针证明 embedding 无法解码对手状态 |
| 3 | "预测 opponent hidden hand distribution 一定有用" | **REFUTED（按原表述）** | 无超人 AI 这样做；标签部分可观测（只有荣和是 GT，AUC 上限 ~0.83 自发布）；三家 34×5 独立头违反"每种牌 4 张"；Sasaki 扩散生成不出听牌（但**已报告待张概率的 ECE** 0.00188/0.000801 `[S63]`——"校准"有先例）；hongdp exp69 **3×34** 待张头 A vs P 0.4938±0.0035 = null；SuuTTT 听牌 AUC 0.887 零收益、`belief-weighted PIMC` **有害** | 34 路二分类探针 + ECE（+ A2b 校准审计） | 若审计显示 belief 已良好校准且探针可解码 → 该方向彻底关闭 |
| 4 | "PPO 比 DQN 更现代，所以应该更强" | **REFUTED（该推理）**；PPO-vs-DQN **UNRESOLVED** | Equim 数月 PPO 打不过监督基线**且自陈可能实现有误**；所有麻将 PPO 结果都需 SL/BC 初始化或单局模式；SuuTTT warm PPO 40-block ~50 min/iter 不可行；**三份独立报告互相冲突** | A15（等预算 PPO vs A14） | A14 若复现正向、A15 若也正向 |
| 5 | "Pure Zero 一定能突破人类" | **REFUTED（无证据；也未被证伪）** | 无评审级立直零数据从零到顶尖；纯自对弈 Elo 1118 ≪ BC 1443；需 1.3–5 亿局；SuuTTT 自对弈多次 null。**反-反证**：RVR 4 人国标从零 3 天/8 GPU 达前 16 的 62.5%，但**用了 oracle 全局价值网 + 学习型奖励** | （建议不做的）Mahjax 小额纯零侧臂 | 若出现"严格无塑形纯零"正例 |
| 6 | "最终 reward 足够，不需要 reward shaping" | **CONDITIONALLY SUPPORTED（窄义）** | 已证明系统**全都塑形/学习奖励**（Suphx GRP、Mortal GRP 增量、RVR 学习型期望奖励）；从零的手写派生特征塑形 −73.9（4.8σ）；PBRS 只改到达速度不改平台；DouZero 是小规模反例 | 用训练自由的 `placement` 势函数替换 GRP，重跑 1v3 | 严格纯零若成功则说明终局奖励够用 |
| 7 | "Self-play 一定比 human data 更好" | **REFUTED（此硬件/项目下）** | human prior ≫ pure self-play（0.7966 vs 0.2034；Elo 1443 vs 1118）；RL 叠加 BC **打平**（z=1.79，预注册 ≥0.53 未达）；已证明的系统都有人类数据锚定 | A14 vs A5 | A14 若显著超越 anchor 且纯自对弈也追上 |
| 8 | "麻将最终可以直接照搬 AlphaZero" | **REFUTED** | AlphaZero 在不完信息**理论上不成立**（动作价值依赖被选概率）；MuZero 自述 IIG "not directly addressed"；Suphx：回合顺序不固定、无法构造正规博弈树；ReBeL / Student of Games（SoG）仅两人零和有保证；多人收敛到中介均衡 | — | — |
| 9 | "CFR 一定比 PPO 更适合麻将" | **CONDITIONALLY SUPPORTED（仅神经遗憾最小化谱系）** | 表格/Deep CFR 在 ~10⁴⁸ 信息集不可行；CFR 的收敛保证仅限两人零和（2p0s）；CFR-p 是 2 人重抽象。支持面：ACH/NW-CFR 是少数在 1v1 上战胜过世界冠军的方法（**p=0.19，不显著**），且支撑 LuckyJ 的 4 人新闻稿。α-Rank/α-PSRO 可推广到 general-sum 多人游戏，但属于元求解器 | A16（4 人 ACH vs DQN 等预算） | A16 若 ACH 明显更优/更差 |
| 10 | "更大的模型一定更强" | **REFUTED** | hongdp"容量叙事死亡"（11M SEres < 2M ConvFormer）；SuuTTT 容量扫描（16/24/40/56 blocks × 128/192/256 ch）全 tie-or-below；AlphaStar Unplugged 无记忆在 1e10 帧更好；b24c512 被作者标为"非推荐"；你的本地容量信号被饱和的 test_play 掩盖 | A12（等 token 预算扫描） | — |

**逐条被点名"不应写出"的声明**（红队清单，已从本报告剔除）：
"Transformer 是下一代方向" · "river transformer 能带强度提升" · "belief 是超人必要组件" · "预测对手手牌能提升强度" · "PPO 更现代更强" / "Mortal 用 PPO 失败证明 DQN 更优" · "Pure Zero 能突破人类上限" **与** "纯自对弈是死路" · "self-play 一定优于人类数据" · "麻将可照搬 AlphaZero" · "CFR 一定更适合" · "更大模型更强" · "LuckyJ 证明 4 人立直从零可行" · "SuuTTT 证明自对弈 RL 无用" · "未払い/gokujan 是最强麻将 AI" · "Suphx 需 5 年/一张 3090" · "NAGA 用 Suphx 论文证明不用 RL" · "Mortal v4 有 72% 输入是牌河"（当测量用） · "P3 验收通过、175.2k 明显强于 148.8k"。

---

# 十一、最终推荐

## 如果我是你，我会：

> 可执行版本（每项的配置改动点、依赖顺序、预注册判定门）见 **`docs/FORK_ROADMAP.md`**。

### 第一阶段（0–2 周）：把"测量能力"修好，冻结一切架构改动

1. **修评估**：1v3 duplicate 扩到 **≥6,100 半荘**（Δavg-rank 0.05 可分辨），固定 seed 与协议，报告 CI；把 `test_play` 从进度指标里删除（它对着随机初始化基线，饱和 100%）。
2. **修数据留出**：从 `datasets/extracted/**` 明确划出**验证年份或验证牌手**（当前 `reserve_ratio=0` 且训练覆盖全 18 年 → 没有泛化信号）。**同时修掉 GRP 验证集与 DQN 训练集在 2023 年的重叠**。
3. **修/隔离三个上游 bug**：#114（`enable_rule_based_agari_guard`）对评估影响最直接，先补；#111（分数/超车编码）影响特征与目标；#115（kan 的 gamma 折扣）影响离线目标。上游已冻结，**这三个 issue 的修复必须由你的 fork 自己维护**。**状态（2026-09-17 更新）**：本 fork **#114 已修**（`patches/0004`，`test_play`/`train_play` 两边都开保险丝）、**#115 已修**（`patches/0003`，`label <= 36 || label == 42`）；**#111 不动代码，仅完成量化**（`docs/quant-111-alllast.md`），是否进下次重训另议。
4. **跑 A1（打乱弃牌顺序）+ A2（线性探针）**：两个都是小时级实验，却分别能证伪你 §五 与 §六 的核心假设。
5. **把当前 175.2k 模型与 148.8k 模型在 ≥6,100 半荘下重评**——大概率会得到"两者无显著差异"的结论，早点知道比晚知道好。

### 第二阶段（2–8 周）：在**已验证的**地基上做单变量实验

6. **把 Mortal baseline 复现到你自己可稳定复现的水平**（A5），确认工具链（libriichi 吞吐、1v3 协议、数据管线）与你的一致。
7. **做 A6/A7/A9 三个 river-transformer 对照**（等参数 / 等算力 / 打乱顺序）。**只有当 A6/A7 优于 A5 且 A9 明显更差时**，才继续投入这条线；否则把它写成一篇有价值的负面结果。
8. **做 A14（简单 actor-critic + forward-KL=0.05 自对弈）**——这是唯一被复现的正面 RL 结果，配方明确、成本可控。判定规则照抄：**placement 95%CI 下界 > 2.500**，且必须在**与选择集分离**的确认集上重测。
9. **不要**在此时引入显式 belief 头、oracle 头、对手手牌预测、或更大模型。

### 第三阶段（2–6 个月）：在真正空白的地方做科研增量

10. **A16：4 人立直上 ACH（regret-minimization）与简单 AC、PPO 的等预算头对头**。已有的开源 ACH 实现可直接移植到 `libriichi` 的自对弈循环；这是**没有同行评审先例**的问题，且直接回应用户"CFR 是否更适合麻将"。
11. **评估方法论本身的贡献**：把 §9.1 的功效分析、**选择/确认集分离**、以及"动作精度≠强度"的反例写成可复用的协议——这个领域目前大量结论不可比（Suphx 与 SuuTTT 的对比就是活例）。
12. 视资源决定是否保留一个 **Pure Zero 侧臂**（明确记录 scaling 与失败模式），用于回答"人类先验的价值有多大"。

## 我认为最有希望的最终架构

```text
Human logs (MJAI, 2009-2026)
        |
        v
+--------------------------------------------------------------+
| Stage A: Offline CQL (Dueling DQN) + GRP reward   [PROVEN]   |
+--------------------------------------------------------------+
        |   frozen anchor  ->  trust-region center for Stage B
        v
+--------------------------------------------------------------+
| obs (1012,34) -----------> Mortal CNN (192ch x 40) -> h_cnn   |
| events (L<=256, 19 types) -> Transformer (128d x 2-4) ->      |
|                              h_river 128     [ABLATION ARM]   |
+--------------------------------------------------------------+
        |
        v
   LayerNorm -> Linear(1152 -> 1024) -> Mish   (fusion)
        |
        +------------------------+------------------------+
        v                                                 v
  Q/Policy head (46, masked)                        Value head V(s)
        |                                                 |
        +------------------------+------------------------+
                                 v
        Stage B: REINFORCE + Huber(V) + forward-KL (lambda=0.05)
        opponent pool = {snapshots, frozen anchor, heuristics}
                                 v
        duplicate eval (>=6100 hanchan, multi-seed, disjoint select/confirm)
```

## 最值得做、最可能产生真正科研增量的 3 个方向

1. **4 人立直上的 regret-minimization vs value-based 等预算头对头**（ACH/NW-CFR 系）。
   理由：唯一评审证据在 1v1 且不显著（p=0.19），4 人方法是纯新闻稿；而 4 人恰恰是 CFR 失去理论保证的地方——**理论与实践张力较大、空白较明确、且有开源实现可复用**。
2. **多人麻将的均衡目标本身**：把 α-Rank/α-PSRO 这类"general-sum 唯一且可解"的解概念落到 4 人立直（成本 O(N_t²) 元博弈），并检验它与"人类先验 + 顺位目标"的经验策略差距。这是一条**可能回答"麻将 AI 的收敛目标究竟是什么"**的路线。
3. **麻将 AI 评估科学**：把 duplicate 功效分析、选择/确认集分离、以及"predictive accuracy ≠ gameplay strength"的实证写成协议与工具。理由：本报告发现的多个"结论"其实都因评估不可比而无法判定（Suphx 的 SL 数据段位、SuuTTT 的 val-acc vs 对局、你的 P3 验收）——**这块的边际科研价值被严重低估**。

---

# 附录 A：来源清单

**A. 一手代码 / 仓库 / 官方文档**

1. Equim-chan/Mortal（GitHub，pin `0cff2b5`，= 上游 HEAD）— https://github.com/Equim-chan/Mortal
2. Mortal discussion #30（PPO 起源，Equim 本人 comment 4222166）— https://github.com/Equim-chan/Mortal/discussions/30
3. Mortal discussion #102（oracle guiding 被移除 / aux 任务无效）— .../discussions/102
4. Mortal discussion #91（hyskylord 的 policy 重测：~50M vs ~10M 局）— .../discussions/91
5. Mortal discussion #97（作者未发布权重）— .../discussions/97
6. Mortal discussion #88（数据规模未答）— .../discussions/88
7. Mortal issues #93（无文档）/ #111 / #114 / #115（三个未修缺陷）— .../issues/115 等
8. mortal.ekyu.moe 文档（40K 半荘/时脚注：RTX 4090 + batch 2000）— https://mortal.ekyu.moe/
9. mortal.ekyu.moe/user/docker.html（`mortal1-b40c192-t22040618`）— https://mortal.ekyu.moe/user/docker.html
10. 本地源码：`vendor/mortal/libriichi/src/state/obs_repr.rs`、`.../state/consts.rs`、`.../dataset/gameplay.rs`、`mortal/{model,train,reward_calculator,dataloader,server,client}.py`、`config.example.toml`
11. Suphx 论文：Li et al., *Suphx: Mastering Mahjong with Deep Reinforcement Learning*, arXiv:2003.13590
12. Microsoft Research Suphx 项目页 / MSR Asia 中文稿
13. 天鳳官方博客（AI 账号许可 2019-03-01；九段 04-24；十段 06-24）— https://blog.tenhou.net/
14. mj-news.net（十段 / 5,137 局 / 稳定段位 8.740）
15. PingWest / thepaper（Tie-Yan Liu：离线自对弈 ~2,000 万局 vs 在线 5,760 局）
16. 36kr（44 GPU：40×K80 + 4×Titan XP；1.5M 局；评估 20×K80）
17. VLOG: Han et al., *Variational Oracle Guiding for RL*, ICLR 2022 — https://openreview.net/forum?id=pjqqxepwoMy ; 代码 https://github.com/FrostHan/vlog
18. Cryolite gist《kanachan について》(2022-04-15，"5 年/RTX 3090" 一手出处) — https://gist.github.com/Cryolite/519c8ecc042732a3459a39a1f1256599
19. Dwango 官方 NAGA 文章（dmv.nico）+ https://naga.dmv.nico/naga_report/top/
20. @NAGA025（2022-04"九段超え"、2023-05-01 混合联赛、恶手率定义）+ note.com/naga_project
21. 渡辺太×麻雀AI"NAGA"（KADOKAWA，ISBN 9784048112741，卷末强度表付费）— https://www.kadokawa.co.jp/product/322311000197/
22. 腾讯官方 LuckyJ 公告（2023-07-11）— https://news.qq.com/rain/a/20230711A03ZZ500
23. Haobo Fu 主页（LuckyJ / ACH / OLSS 作者列表）— https://haobofu.github.io/
24. ACH: Fu et al., *Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game*, ICLR 2022 — https://openreview.net/forum?id=DTXZqTNV5wN
25. OLSS: Liu et al., *Opponent-Limited Online Search for Imperfect Information Games*, ICML 2023 — https://proceedings.mlr.press/v202/liu23k/liu23k.pdf
26. 开源实现：https://github.com/Liuweiming/ACH_poker（Apache-2.0）、https://github.com/yata0/Mahjong（MIT）
27. modern-jan（LuckyJ 日文授权翻译，ACH 归属）— https://modern-jan.com/blog/luckyj_article_ja/
28. JueJong 评估（margin 0.82±0.96、p=0.19）— arXiv:2601.08211

**B. 文献（含同行评审论文与 arXiv 预印本）**

29. AlphaZero — arXiv:1712.01815
30. Pluribus（Brown & Sandholm, Science 2019）
31. ReBeL — arXiv:2007.13544
32. MuZero — arXiv:1911.08265（"IIG not directly addressed"）
33. DeepStack — arXiv:1701.01724
34. Gibson et al.（n 人一般和"strong guarantees are lost"）— arXiv:1305.0034
35. Ganzfried（多人 CFR 经验失效）— arXiv:2001.11165
36. MacQueen & Wright（多人收敛到 mediated/correlated 均衡）— arXiv:2310.11518
37. *You Can't Count on Luck*（return-conditioning 在随机环境可证失败）— arXiv:2205.15967
38. AlphaStar Unplugged（无记忆 vs 记忆随算力翻转）— arXiv:2308.03526
39. MeowCaTS（IS-MCTS 夺 Computer Olympiad 2023）— doi:10.1109/TG.2025.3535740
40. RVR（Tencent/CAS/清华，从零 4 人国标）— doi:10.1109/CoG51982.2022.9893584
41. α-Rank / α-PSRO — doi:10.1038/s41598-019-45619-9；arXiv:1909.12823
42. DouZero（终局奖励 + LSTM 历史编码，无 ablation）— arXiv:2106.06135
43. Ng, Harada & Russell（势能式塑形）ICML 1999
44. CQL — arXiv:2006.04779；Dueling — arXiv:1511.06581；PPO — arXiv:1707.06347；AWR — arXiv:1910.00177；AWAC — arXiv:2006.09359；Deep CFR — arXiv:1809.04040

**C. 开源工程与第三方材料（多数未经同行评审，已标注单源）**

45. Mahjax — arXiv:2605.20577 + https://github.com/nissymori/mahjax（issue #12：TSUMOGIRI tokenization bug，已修）
46. SuuTTT/IJCAI-mahjong — https://github.com/SuuTTT/IJCAI-mahjong（`HANDOFF_2027.md`、`FINDINGS_2026-06-14.md`、`SEARCH_WRITEUP.md`、`rl_selfplay_v2.py`）
47. hongdp/LLM_Mahjong — https://github.com/hongdp/LLM_Mahjong（HandRiverFormer/exp30、exp41/46/67/69/70/71/73/74）
48. Agony5757/mahjong（pymahjong v5 + VLOG 数据集）— https://github.com/Agony5757/mahjong（`pymahjong/rl/v5/`、`test/test_mortal_v5.py`）
49. RiichiEnv — https://github.com/smly/riichienv（Apache-2.0；`riichienv-ml` TransformerActorCritic；`HandPredConfig` 模块缺失）
50. kanachan — https://github.com/Cryolite/kanachan（BERT 几何 Transformer + IQL/AWR；**无 license、无强度数字、从未与 Mortal 对比**）
51. Tjong — doi:10.1049/cit2.12298（CAAI TIT 2024；唯一等参数真实对局对比，国标）
52. UEC 层次化 Transformer — IPSJ TOD 19(1):55–65, 2026（数字 embargo 至 2028-01-26）
53. IJCAI 2024 Mahjong AI Competition + Axioms 2023 — doi:10.3390/a16050235（信息集规模）
54. gokujan/極雀 — https://gokujan.com/ + note.com/shabazou/n/n8a2a06f2232f（开发者访谈）+ note.com/qwert227（二手）
55. Kurita & Hoki（akochan 算法，多 MDP 抽象 + 搜索）— arXiv:1904.07491
56. Nitasurin/Mortal-Policy — https://github.com/Nitasurin/Mortal-Policy（discussion #1；policy 路线）
57. hhim8826/mortal3-b40c192 — https://huggingface.co/hhim8826/mortal3-b40c192（第三方 v3 权重 + `best_perf` 语义警告）
58. *Elevating Styled Mahjong Agents with Learning from Demonstration* — arXiv:2506.16995（演示比例 ~5–6%，2×3090）
59. MJ-DLVAT — doi:10.1109/cog60054.2024.10645601（价值评估估计器，−45.5% 方差/n）
60. kobaab 電脳麻将 duplicate 实验 — https://blog.kobalab.net/entry/2025/07/27/221043
61. SIREN（选择感知的重复划分报告，治 winner's curse）— arXiv:2605.05973 + https://github.com/jznmsl/siren
62. 天鳳 AI 账号规则（AI 仅开放特上卓及以下）— https://tenhou.net/sc/raw/
63. Sasaki 等（扩散模型手牌生成：**报告待张概率的 calibration curve + Platt scaling + 加权 ECE = 0.00188（扩散）/ 0.00390（自回归）/ 0.000801（直接预测）**；两种生成模型都生成不出听牌）— dmv.nico 技术稿，**非评审**
64. VLOG: Han et al., *Variational Latent Oracle Guiding*（麻将上学隐状态**分布**/belief，ICLR 2022 **评审级**，未报校准）— https://openreview.net/forum?id=pjqqxepwoMy
65. *Confident at the moment of action: belief miscalibration in LLM play under hidden information*（"预测更准 ≠ 信念更对"的同行表述）— arXiv:2608.24691
66. note.com「AIによる先制リーチ研究①」（立直局结果分布：Brier + multiclass ECE = 0.008511）— https://note.com/lithe_gerbil3668/n/nc7404f729c4d
67. *Hybrid Multi-Agent AI/MCTS Systems*（"28" 牌类游戏：belief 网络 calibration error 0.024）— doi:10.22541/au.176405957.77536600/v1

---

# 附录 B：跑码 / 只读验证记录（Phase 3）

| # | 验证 | 方法与命令 | 结果 | 判定 |
|---|---|---|---|---|
| V1 | Mortal obs / 动作空间 | `PYTHONPATH=vendor/mortal/target/release .venv/bin/python` 导入 libriichi | `obs_shape(4)=(1012,34)`、`ACTION_SPACE=46`、`GRP_SIZE=7` | **CONFIRMED** |
| V1b | libriichi.so 是否陈旧 | 比对 `.so` mtime 与 pin commit 日期 | `.so` 为 **2026-08-10 本机构建**，返回值与 pin 源码一致 | **REFUTED**（附录 B V1b：worker 的"陈旧"怀疑不成立，无需重建） |
| V2 | CQL/奖励超参 | 读 `mortal/train.py`、`config.example.toml`、`reward_calculator.py` | `cql_loss=logsumexp−mean`、`min_q_weight=5`、`pts=[6,4,2,0]`、`gamma=1`、无 target net/twin/entropy | **CONFIRMED** |
| V3 | #115 kan gamma | 读 `libriichi/src/dataset/gameplay.rs:429` | `self.apply_gamma.push(label <= 37)` | **CONFIRMED** |
| V4 | 1v3 评估 | 读 `runs/1v3/eval_20260915_1835.log` + 独立复算 | 500 sets/2000 半荘；挑战者 `[508 504 527 461] (2.4705, 3.0825pt)`；`(1·508+2·504+3·527+4·461)/2000=2.4705` ✓ | **CONFIRMED**（冠军侧未能从日志复现 → P3 的 +4.11 仅存 `parse_eval.py`） |
| V5 | 数据留出 / baseline | 读 `configs/config.toml`、`datasets/grp_val_globs.txt`、`ls datasets/extracted/`、`ls -la checkpoints/` | DQN globs 覆盖 2009–2026、`reserve_ratio=0`、`online=false`；GRP val=2023（**无统一留出**）；`baseline.pth` 43.69MB（随机初始化）→ `test_play` 饱和 100% | **CONFIRMED** |
| V6 | 训练吞吐 / 重启 | 读 `runs/dqn_train.log`（149 行）时间戳 | **MEASURED 3.3–5.5 steps/s**（batch 512）；外部 `version: 4` 重启约每 5 分钟，重载仅 ~5s（**2–4% 开销**）；19:19→00:41 存在 5h22m 空档 | **CONFIRMED**（附录 B V6：修正 worker 关于"摧毁吞吐"的表述） |
| V7 | 统计功效 | 应用 `n = (z_{1−α/2} + z_{1−β})² σ² / Δ²`，σ≈85，Δav-rank 换算法 `n≈10.3/Δrank²` | Δavg-rank 0.05 → **≈4,100 半荘**；0.02 → ≈25,800；2000 半荘仅能分辨 ≈0.072；P3 的 +4.11pt → **z≈1.60**，需 ≈6,100 半荘 | **DERIVED**（σ 为 ASSUMED） |

---

# 附录 C：方法论

- **编排**：ULW-Research 模式；3 波并行 worker 共 **25 个**（Wave1 = 13：Mortal/Suphx/NAGA/LuckyJ/Pure Zero 理论/算法选型/Transformer 序列/belief 建模/OSS 生态/日文扫/中文扫/Mortal 源码/本地状态；Wave2 = 9：Mahjax/SuuTTT/pymahjong/極雀/Transformer 证据/搜索与解概念/Mortal 实操/NAGA 强度/OSS 前例；Wave3 = 3：SuuTTT 正面配方/前例清单/评估统计）。
- **红队**：1 个 Oracle 对抗审计（逐条判定 10 假设、证据质量审计、最弱结论、"不应写出的声明"清单），原文 `raw/RED.txt`。
- **验证**：8 项跑码/只读（附录 B，含 1 项子验证），其中 2 项**推翻了 worker 的初判**（libriichi.so 陈旧；"摧毁吞吐"）。
- **自我纠错记录**：① NAGA "无 RL" 的引证被纠正（Suphx 论文不含 "NAGA"）；② discussion #30 归属由悬置变为 Equim 本人；③ SuuTTT "全负"被其自身后续正向结果修正；④ Suphx "凤凰卓数据源"由"论文所述"降为"三方推断"；⑤ "5 年/3090" 的范围被校正为"150 万半荘模拟的电算估算"。
- **单源例外（已标注，未进 verified-claims 白名单）**：hongdp/LLM_Mahjong、SuuTTT 战役、gokujan 数值、0.83 危险度 AUC。它们支撑的结论在文中均标 `[单源]`。
- **未解问题（gaps）**：LuckyJ 4 人算法/搜索论文不存在；消费级 GPU 上 Mahjax/立直 transformer 无实测；立直领域无等参数 CNN-vs-Transformer 对局强度对比（UEC 精确数字 embargo 至 2028）；hongdp n=1200 与 SuuTTT 的 n=36 blocks（非样本量）均无法从公开产物独立复现；NAGA 各版本强度表（付费书）不可读。
- **全部原始材料**：`.omo/ulw-research/20260916-004434/`（`wave-0-brief.md`、`wave-1-summary.md`、`wave-2-summary.md`、`phase3-verification.md`、`claim-graph.md`、`intent-diff.md`、`observation-manifest.md`、`SYNTHESIS.md`、`raw/*.txt`）。

