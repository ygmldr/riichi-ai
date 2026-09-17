# 麻将 AI 研究路线交叉评审 回答

> 本文回应《麻将 AI 研究路线交叉评审 Prompt.md》的全部 11 节问题。
> 撰写日期：2026-09-16。撰写环境：riichi-ai 仓库（已 vendor Mortal 全量代码并跑通离线训练 + 1v3 评估）。

## 证据等级标注体系

全文所有关键论断均标注来源等级：

| 标注 | 含义 |
|---|---|
| 【事实·代码】 | 直接从 Mortal 源码 / 本仓库代码核实，给出 `文件:行号` |
| 【事实·论文】 | 论文原文明确陈述，给出 arXiv 链接 |
| 【事实·文档】 | 作者官方文档 / gist / 官方公告明确陈述，给出链接 |
| 【事实·媒体】 | 媒体或官方新闻报道（非技术细节来源，可信度次一级） |
| 【推测】 | 我的推断，有推理依据但无一手证据 |

重要发现先说：**Mortal 没有论文**。其权威资料只有 GitHub 代码、一份写作中且大部分页面为空的文档站（包括「Reinforcement Learning」「Failed Attempts」「Comparison with Other AIs」等关键章节均为空占位），以及作者 Equim-chan (eqs) 的少量公开讨论。**因此"分析 Mortal"的最可靠途径是读代码本身——本文对 Mortal 的所有架构结论都直接从 vendored 源码核实。**

---

# 一、背景与本仓库现状（评审落点）

Prompt 假设的"高速 Rust 环境 + 大量人类牌谱 + 多 GPU"在本仓库的现实是：

- **已有**：Tenhou 公开牌谱 2009–2022（训练）+ 2023（验证），12.78GB zip / 解压 40–60GB（`RUNBOOK.md`）；GRP 已训练（`checkpoints/grp.pth`）；离线 DQN v4 已训练至 175.2k 步（`checkpoints/mortal.pth.at-175200`）；1v3 duplicate 评估管线已跑通
- **已有但未启用**：完整的 online self-play 分布式基建（参数服务器 + worker 回放提交，`vendor/mortal/mortal/server.py`、`mortal.py`、`client.py`），当前 `online = false`（`configs/config.toml:3`）
- **现实约束**：单张 5070Ti（16GB）训练机，非多 GPU

【事实·代码】当前训练配置与结果：
- v4 观测 `(1012, 34)`（`vendor/mortal/libriichi/src/consts.rs:25`），动作空间 46（`consts.rs:7-15`：34 普通打牌 + 3 赤 5 打牌（`5mr/5pr/5sr`，`tile.rs:12-19`）+ 立直 + 3 吃 + 碰 + 杠宣言 + 和牌 + 流局 + 跳过；杠宣言后的"杠/暗杠择牌"复用 0–33 索引，不占新槽位）
- ResNet：192 通道 × 40 blocks，1D Conv(k=3) + ChannelAttention(SE 式) + Mish + pre-activation BN（`vendor/mortal/mortal/model.py:30-106`），flatten 后 Linear(32×34 → 1024) 得 state embedding
- Dueling DQN v4 头：单 Linear(1024 → 1+46)，masked dueling（`model.py:217-231`）
- 损失：`0.5·MSE(Q, γ^steps·R_kyoku) + 5·(logsumexp Q − Q(s,a)) + 0.2·CE(next_rank)`（`vendor/mortal/mortal/train.py:228-247`）——Q-target 是**本局内 Monte-Carlo 回报**（γ=1，无 TD bootstrap），reward 来自 GRP
- GRP：GRU(输入 7 维 = [局数, 本场, 供托, 4 家点数] → 64×2 层) → 24 类终局座次排列（`model.py:233-266`）；每局 reward = ΔE[顺位 pt]，pt=[6,4,2,0]，最后一局用真实终局座次锚定（`vendor/mortal/mortal/reward_calculator.py:39-58`）
- 最新代际评估（`runs/1v3/eval_20260915_1835.log`）：当前 challenger vs 148.8k champion，2000 半庄，平均顺位 2.4705，**+3.08 pt/局**——注意：按本文第九节的统计功效计算，该差值约 1.7σ，**尚未达到 95% 显著**

这个现状直接影响后文所有建议的排序：本文不假设"从零起步"，而是假设"在已跑通的 Mortal 管线上做增量研究与改造"。

---

# 二、请首先独立分析现有 Mahjong AI

## 2.1 Mortal

**总体定性**：Mortal（凡夫）是 Equim-chan 2021–2022 年开发的免费开源麻将 AI（AGPL-3.0），核心是"人类牌谱 → 离线 RL(CQL) → 在线自博弈"两阶段管线，是当前唯一**完整开源**训练栈的顶级麻将 AI。【事实·文档】

来源：[GitHub - Equim-chan/Mortal](https://github.com/Equim-chan/Mortal)、[官方文档](https://mortal.ekyu.moe/)、[权重发布讨论 gist](https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56)

### 网络结构
【事实·代码】`vendor/mortal/mortal/model.py`：
- Brain = 1D ResNet（Conv1d k=3，per-block ChannelAttention 注意力，Mish，pre-act BatchNorm(momentum=0.01)），末端 Conv→32ch→flatten→Linear(32×34→1024)，输出 1024 维 embedding（v2/3/4 直接出激活后 embedding；v1 另有 mu/logsig 隐变量头，是早期 VQ/隐变量实验的痕迹，`model.py:126-131`）
- DQN 头：v4 为单 Linear(1024→1+46) 的 masked dueling 结构 `Q = V + A − mean(A)`，非法动作填 −inf（`model.py:221-231`）
- AuxNet：Linear(1024→4)，预测**下一局结束时的自己最终顺位**（4 类 CE）（`model.py:188-195`、`train.py:240-241`）
- GRP：独立小模型，GRU(7→64, 2 层) → FC → 24 类（4 家终局座次的全排列），用于 reward 稠密化（`model.py:233-286`）
- 推理端 `MortalEngine` 有 `enable_rule_based_agari_guard`（和牌决策的规则保险丝）与 top-p/boltzmann 采样（`vendor/mortal/mortal/engine.py`）

### Observation representation
【事实·代码】v4 = 1012 通道 × 34 牌种，tile-centric。关键细节（`vendor/mortal/libriichi/src/state/obs_repr.rs`）：
- 自手牌张数平面(4) + 赤牌指示(3)；四家点数(归一化) + 自家顺位 one-hot(4) + 局数(4) + 本场/供托 RBF 编码 + 宝牌指示牌
- **牌河编码是"窗口化"的**：每家先编最早 6 巡（逐项 8 通道：吃碰对 one-hot、杠、打牌、赤、是否宝牌、是否手切、是否立直），再倒序编最近 18 巡；自家另有全河**时间衰减平面** `exp(−0.2·距现在巡数)`（v3/4，`obs_repr.rs:231-238`）；他家另有 v2 式 3 行×6 巡摘要表（`obs_repr.rs:253-260`）
- 副露（含来源家）、立直宣言牌、被鸣走的牌均有专门通道
- oracle 通道：另加 217 通道（`consts.rs:32`），数据管线在 `oracle=true` 时用 `Invisible::new(events, trust_seed)` 生成完全信息平面（`libriichi/src/dataset/gameplay.rs:164`），Brain 支持拼接输入（`model.py:152-155`）——**Suphx 式 oracle guiding 的基础设施在代码里是存在的**

### Action representation
【事实·代码】46 维（`consts.rs:7-15`）：34 普通打牌（`0–33`）+ 3 赤 5 打牌（`34–36` = `5mr`/`5pr`/`5sr`，`tile.rs:12-19`）+ 1 立直 + 3 吃 + 1 碰 + 1 杠宣言 + 1 和牌 + 1 流局 + 1 跳过。**注**：杠宣言后的"杠/暗杠择牌"不占新槽位——复用 `0–33` 的牌张索引重新置 mask（`obs_repr.rs:423-427`），赤 5 候选去赤映射后同样走 `0–33`（`:431-448`）。mask 由 Rust 端状态机给出。

### Reward design
【事实·代码】不是"终局 reward"，而是 **GRP 稠密化 reward**：每局结束后，reward = ΔE[顺位 pt]（pt 表 [6,4,2,0] 可配），其中顺位分布由 GRP 根据"截至当前的局数/本场/供托/四家点数"预测（24 排列 → 4×4 顺位概率矩阵）；整场半庄的最后一局用**真实终局座次**替换预测（`reward_calculator.py:39-58`）。于是 Q-target = γ^steps × (本局起剩余各局 reward 之和) = 到半庄结束的 Monte-Carlo 期望顺位点回报（`train.py:228`）。这同时解决了两个问题：单局内 reward 稀疏（和牌/流局才有结算）+ 半庄级 credit assignment。

### 人类牌谱如何进入训练
【事实·代码】`FileDatasetsIter`（`vendor/mortal/mortal/dataloader.py` + `libriichi/src/dataset/`）读 Tenhou 牌谱，逐局重放生成 (obs, mask, action, kyoku_reward, player_rank) 转移；支持按玩家名过滤（`player_names_files`，可只学凤凰桌玩家）、seat rotation 数据增强（`enable_augmentation`）、文件索引缓存。**人类牌谱提供的是"动作 + 后续自然发展"构成的离线 RL 转移，而非 BC 目标**——训练目标是 Q 而非 π。

### Offline RL / CQL 为什么有效
【事实·论文】CQL（Kumar et al., NeurIPS 2020, [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)）的核心：离线数据只覆盖策略分布的一小部分，标准 Q-learning 会对分布外(OOD)动作外推出不切实际的高 Q 值，且误差被 bootstrapping 迭代放大。CQL 在损失里加 `logsumexp(Q) − Q(s,a*)`，把没被执行的动作的 Q 往下压，使学到的价值函数在数据支撑范围内保守。
【事实·代码】Mortal 的实现是固定权重(5)的 `logsumexp − mean` 形式（`train.py:238`），**online 阶段该项自动关闭**（`train.py:237`：`if not online`）——自博弈数据由当前策略产生，不存在 OOD 外推问题，这正是 CQL 只属于 offline 阶段的语义体现。
【推测】此外 Mortal 的 Q-target 是 MC 回报而非 TD bootstrap，本身就没有"自举放大外推误差"的链条，CQL 在这里更多是防止单步内对 46 个动作的相对高估，这解释了为什么固定权重 5 就够用、不需要 CQL 论文里的 Lagrangian 自适应版本。

### Online training / self-play 如何进行
【事实·代码】完整基建已存在且本仓库未启用：
- `server.py`：ThreadingTCPServer 参数服务器，维护"最新参数版本 + 回放文件缓冲区"，worker `get_param` / `submit_replay`，trainer `submit_param` / `drain`；`force_sequential` 控制是否强制 on-policy
- `mortal.py <id>`：worker 进程，加载参数跑 mjai Bot 自博弈，按 `train_play.default` 配置采样：**boltzmann_epsilon=0.005, boltzmann_temp=0.05, top_p=1.0**——即近乎贪心 + 极小温度的"低熵自博弈"
- `train.py` online 分支：从 drain 目录读回放（`sample_reuse_rate=0` 即样本基本不重用，接近 on-policy），CQL 关闭，每 400 步向服务器提交新参数；每 `test_every` 步与 baseline 做 1v3 test_play
- 已知问题：作者自己注释了 online 模式下 test_play 会卡死的 bug，用子进程重启绕过（`train.py:381-386`）

### 为什么使用 DQN 而不是 PPO
【事实·文档】**作者没有公开明确陈述过这个选择**（文档「Reinforcement Learning」「Failed Attempts」章节至今为空）。
【推测】从代码与问题结构看，合理性在于：(1) 离线数据上做 PG 需要重要性采样修正且方差大，value-based + CQL 是离线 RL 更自然的入口；(2) 46 维离散动作 + 完备 mask 使 Q-learning 干净可行；(3) Q 顺带产出 policy improvement 需要的价值基准；(4) MC-target + 稠密 GRP reward 让 value 回归本身就是监督学习，稳定。PPO 的优势（随机策略、熵控制、on-policy 稳定性）在**离线阶段无关紧要，在自博弈阶段才有意义**——而 Mortal 的自博弈采用低熵近似贪心采样，本质上仍以 Q 为核心。

### 强度证据
【事实·文档】[官方 strength 页](https://mortal.ekyu.moe/perf/strength.html)：1v3 duplicate（[90,45,0,-135] pt，种子由 "Mortal the mahjong AI" 的 SHA-256 派生，每对局 1M 半庄起）：
- Mortal(4.x) vs akochan 3.1：**+1.891 pt/局**，平均顺位 2.477 vs 2.568（~13k 局 x1 / 40k 局 x3）
- Mortal(4.x) vs Mortal(1.0)：+1.677 pt/局
- 环境吞吐：RTX 4090 + Ryzen 9 7950X、batch 2000 时 **40k 半庄/小时**（[文档首页](https://mortal.ekyu.moe/)）
【事实·文档】权重经[公开伦理讨论](https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56)后部分释出，社区有 [VoidShine/mortal-298k](https://huggingface.co/VoidShine/mortal-298k) 等镜像。作者未发布过天凤凤凰桌爬分记录。

### Mortal 当前方法的主要局限
1. **无搜索**：纯前向网络一步出 Q，推理时不做任何 lookahead/determinized MC（akochan 有搜索，天凤内与 Mortal 互有胜负的传闻无官方对局数据）【事实·代码 + 推测】
2. **观测工程密集**：1012 通道手工平面是大量领域知识的结晶，也是实验迭代的门槛——每加一类信息要动 Rust 状态机【事实·代码】
3. **1024 维单向量瓶颈**：全部决策信息压到 1024 维； belief/反事实推理所需的结构化中间表示没有显式位置【推测】
4. **离线阶段受数据支撑限制**：CQL 保守性使策略难以系统性超越人类分布，超越必须靠 online 阶段，而 online 基建有已知 bug 且参数敏感【事实·代码 + 推测】
5. **对手建模完全隐式**：无任何显式 belief/对手状态结构，全部依赖平面编码 + CNN 隐式提取【事实·代码】
6. **文档不全**：训练细节（尤其失败尝试）未公开，复现依赖读代码【事实·文档】

## 2.2 Suphx

来源：[Suphx: Mastering Mahjong with Deep Reinforcement Learning](https://arxiv.org/abs/2003.13590)（Microsoft Research Asia；下引页码/表格均出自 ar5iv HTML 版原文核实）

### 管线与关键技术
【事实·论文】Suphx = SL 预训练 → 策略梯度 RL 自博弈 → (部署时) pMCPA 运行时适应。四个核心组件：

1. **SL 预训练**：天凤牌谱，5 个输出头（打牌/立直/吃/碰/杠），top-1 准确率打牌 **76.7%**（此前最好 68.8%）、立直 85.7%、吃 95.0%、碰 91.9%、杠 94.0%。论文明确警告这些数字与先前工作"因训练/测试数据不同不可直接比较"。
2. **策略梯度 RL + 自博弈**：importance sampling + **熵正则**（动态 α：论文原话"熵太小则 RL 快速收敛、自博弈不再显著改进；熵太大则训练不稳定"）+ multi-step return。**每个 RL agent 训练用 150 万局，44 块 GPU 跑约 2 天；最终 Suphx 用了约 250 万局**。
3. **Global Reward Prediction (GRP)**：用人类牌谱训练一个模型，从局内状态预测**半庄终局顺位分布**，为每局提供中间 reward——与 Mortal 的 GRP 同名同思路（Mortal 是否受其启发未证实【推测】）。
4. **Oracle guiding**：训练初期给 agent 完全信息（各家手牌+牌山），随后以概率 γt 逐步 drop 掉 oracle 特征（γt 从 1 衰减到 0），"使 oracle agent 逐渐过渡为 normal agent"；γt 归零后继续训练需配合学习率 ÷10 + importance weight 截断，否则"持续训练不稳定且不再提升"。论文强调直接蒸馏 oracle 不行——普通 agent 很难模仿超强 oracle，**渐变 dropout 让 normal agent 比只用可见信息的标准 RL 快得多**。

### 为什么这种 pipeline 有效（Prompt 重点）
【事实·论文】论文的内在逻辑链，每环都有实验支撑：
1. **SL 预训练提供"可学习的起点"**——策略梯度从随机初始化出发在麻将上无信号可用，76.7% 打牌准确率的先验让 RL 从"会打牌"开始优化"打得好"；
2. **熵正则 + 动态 α 防止自博弈过早塌缩**——论文明确给出双边失败模式（熵太小不改进、太大不稳定），这是自博弈能持续推进的前提；
3. **GRP 把"半庄级延迟奖励"变成"局级即时奖励"**——降低方差、改善 credit assignment，这是策略梯度在高延迟奖励下能收敛的关键；
4. **Oracle guiding 解决"探索好策略比找到好策略更难"**——先用完全信息学"什么局该做什么"，再渐退特征逼策略在普通信息下重建这些判断，论文原话是 normal agent 的提升速度远超普通 RL；
5. **Dropout policy + pMCPA 让策略适配"模拟 rollout"的运行时分布**——训练时人为制造手牌不确定性，推理时的 Monte-Carlo 适应才不至于分布错位。
【推测】一句话总结：Suphx 的每个组件都在解决策略梯度应用于 4 人 POMDP 的一个具体失败模式，pipeline 的有效性来自"组件与失败模式一一对应"，而非某个单一秘方。

### 运行时适应 pMCPA
【事实·论文】Parametric Monte-Carlo Policy Adaptation：每一局开局后，采样对手手牌+牌山做 K 次 rollout（论文称 K"不需要很大"），用这些轨迹以 importance-weighted reward 加权对策略参数做少量梯度更新，然后用适应后的参数打这一局；下一局重置回离线参数。

### Dropout policy
【事实·论文】训练时对**自己的手牌特征**做随机 drop（模拟"看不清自己手牌"），迫使策略对不确定信息鲁棒——这直接服务于运行时的 Monte-Carlo 采样 rollout 场景。

### 人类牌谱在 Suphx 中扮演什么角色（Prompt 重点）
【事实·论文】三重角色：(1) 策略网络初始化（SL 预训练是 RL 的起点，不是可选装饰）；(2) GRP 模型的训练数据；(3) 评估参照。**Suphx 没有做过 zero 初始化的对照实验，论文无法回答"没有人类数据会怎样"。**

### 结果
【事实·论文】Table 4（已逐字复核）：天凤凤凰桌四麻，Suphx 5760 局 **最高段位十段、稳定段位 8.74**；对照：顶级人类 7.46（8031 局）、NAGA 6.64（9649 局）、Bakuuchi 6.59（30516 局）。Table 5：Suphx 顺位分布 29.3/27.5/24.4/18.7%，和了率 22.83%，放铳率 10.06%。论文称"超过 99.99% 的人类玩家"。

## 2.3 NAGA

来源：[Dwango Media Village 官方技术文章（2019-01）](https://dmv.nico/ja/articles/mahjong_ai_naga/)、[NAGA 牌谱解析服务](https://naga.dmv.nico/naga_report/top/)、[官方 X @NAGA025](https://x.com/NAGA025)

【事实·文档】NAGA（Neural Architectural Game Agent）由 Dwango 系 DMV 开发：
- **纯监督学习**：从天凤凤凰桌公开牌谱深度学习"打牌/副露/立直/杠"的选择——**由 4 个独立 CNN 组成**（打牌、副露、立直、杠模型），输入包含自家手牌、四家牌河、副露、点数、余牌数等
- 训练采用**置信度估计**（低置信样本加权），并用 Guided Backpropagation 做决策归因可视化
- 只有极少量手工规则（"オーラ斯避开确定垫底的和了"判断），攻防判断完全由学习得出
- 2018-10 上天凤特上卓，2019-01 报道时**最高八段**
- Suphx 论文 Table 4 给的第三方数据：NAGA 稳定段位 **6.64**（9649 局，最高 8 段）【事实·论文】

### 为什么不使用 RL（Prompt 重点）
【事实·文档】官方文章**通篇未提及 RL，也未解释为何不用**。文章唯一的方法论对比是：此前研究用期望顺位+蒙特卡洛（Mizukami 2015），NAGA 改用深度学习。
【推测】合理解释：(1) 2018 年时 4 人非完全信息自博弈的工程风险远高于 SL，SL 路线成本最低、效果可预期（后来 NAGA 达到顶级水平证明上限不低）；(2) NAGA 的商业形态是牌谱解析服务，SL 模型的决策分布贴近凤凰桌玩家，解释性好；(3) 自博弈需要高质量快速环境，当年没有 libriichi 这样的基建。

### NAGA 路线 vs Suphx/Mortal
- **优势**：无需快速环境与自博弈基建；无非平稳性/自博弈崩溃问题；直接贴合顶尖人类策略，行为可解释、可归因；训练成本最低【事实 + 推测】
- **局限**：模仿学习天花板受数据质量与容量约束——它学到的是"凤凰桌玩家的平均最优"而非"博弈最优"；无法发现超越人类的反直觉策略；4 个独立 CNN 缺少跨决策头的状态一致性【推测】
- **关键启示**：高质量 SL 本身就能产生"稳定段位 6.6+/十段级"的 AI，这为任何 RL 路线提供了必须击败的下限基准

## 2.4 LuckyJ

来源：[腾讯官方战报·证券时报](https://www.stcn.com/article/detail/918748.html)、[官方 Twitter @LuckyJ1443836](https://twitter.com/LuckyJ1443836)、[知乎讨论](https://zhuanlan.zhihu.com/p/642827719)、学术引用方式见 [Self-Play Survey, arXiv:2408.01072](https://arxiv.org/abs/2408.01072)

【事实·媒体】绝艺 LuckyJ（腾讯 AI Lab）：
- 2023-07-11 官宣：天凤四麻凤凰桌**稳定段位 10.68 段**，从零爬分**仅 1321 局即达十段**（比 Suphx 少 4052 局），为天凤平台 AI 最好公开成绩
- **没有公开论文**。学术论文引用它时只能引其 Twitter 账号（见上 survey 的引用格式）——**没有任何架构、训练方法、数据规模的一手信息**
- 【推测】社区普遍猜测其用了大规模自博弈 RL（绝艺团队在围棋/星际等完全信息博弈有成熟 self-play 经验），网传"CFR 类方法"**无法从一手来源证实**；1321 局爬十段的样本量也偏小，存在运气成分（知乎有此质疑）
- 【推测】10.68 vs Suphx 8.74 的稳定段位差若可信，说明 2020–2023 年间大规模 self-play + 更大算力带来了约 2 段的进步——但无法归因到任何具体技术

**结论：LuckyJ 是"存在性证明"（更强性能可达）而非"方法论来源"。Prompt 中要求研究的"Neural CFR / search / belief modeling / 与普通 self-play 的区别 / 为什么 CFR 类方法可能适合麻将"这组问题，因无一手资料**全部无法从 LuckyJ 本身得到回答**；CFR 与麻将适配性的正面分析见本文第七节与第十节第 9 条，结论是理论保证不覆盖 4 人且计算不可行。**

## 2.5 对比小结

| | Mortal | Suphx | NAGA | LuckyJ |
|---|---|---|---|---|
| 信息完整度 | 代码全开源 | 论文详尽 | 官方博客 | 几乎为零 |
| 初始化 | 人类牌谱(离线RL) | 人类牌谱(SL) | 人类牌谱(SL) | 未知 |
| RL | 离线 CQL-DQN + 自博弈 | PG 自博弈+oracle guiding | 无(纯SL) | 未知 |
| 搜索 | 无 | pMCPA 局内微调 | 无 | 未知 |
| 天凤稳定段位 | 未爬凤凰桌(官方) | 8.74 | 6.64 | 10.68 |
| 训练算力(已知) | 单机级可复现 | 44 GPU × 天级 ×250万局 | 未公开 | 未知 |

【事实+推测汇总】三条已验证的共同点：**人类数据 warm start + RL 自博弈突破 + 某种形式的稠密化 reward（GRP/oracle guiding/dropout policy）**。没有任何一家验证过 zero 初始化。

---

# 三、核心研究问题：Pure Zero 是否值得做？

**总答案：作为性能路线，当前不值得重注；作为科研 baseline，值得做一个有预算上限、有 kill switch 的诚实实验。** 理由逐条展开：

### 1. Reward sparsity
【事实】麻将半庄 8–12 局，每局以和牌/流局结算，且最终目标（顺位点）只在整场结束才有真值。【推测】但注意 Mortal/Suphx 都已证明"GRP 式稠密化"可以从**人类牌谱**学到局间价值估计——Pure Zero 没有 GRP 的训练数据，需要 bootstrap：先用终局真值训练粗糙 GRP → 用 GRP reward 训练策略 → 策略变强后再迭代 GRP。这个自举链条在随机初始化下能否收敛是核心未知数。

### 2. Exploration difficulty
【事实·代码】46 维动作空间本身不大，但有效策略需要长程组合行为（立直到立直、完全generalase防守），一次半庄 ~70 巡 × 4 家 ≈ 数百个决策点，随机策略几乎不可能"碰巧"做出结构化好行为。【推测】ε-greedy/熵探索在高维组合行为上效率极低，这是 Pure Zero 最大的实际障碍。

### 3. Cold start
【事实·代码】Mortal 的解法（离线 CQL 直接吃人类动作）完全依赖人类数据。【推测】Zero 的冷启动需要替代品：课程学习（如先打 1 局东风局、先学完全信息版本）、oracle guiding（Suphx 式，但 oracle 数据来自自博弈而非人类）、shaping reward（向听/进张辅助损失）。每一项都有失败风险，且**"Oracle guiding 但 oracle 也从零开始"** 是没有先例的组合。

### 4. Hidden information
【事实·论文】ReBeL（[arXiv:2007.13544](https://arxiv.org/abs/2007.13544)）证明了belief state + RL + search 在 2 人扑克上从零可行——但那是 2 人、公开信念状态可精确计算、单局奖励。【推测】麻将的信念空间（3 家手牌 × 13 张 + 牌山）无法精确枚举，只能学近似 belief，且 4 人博弈中"对手也在适应"使 belief 目标非平稳。

### 5. Opponent modeling
【推测】自博弈理论上能涌现对手建模（对手池 = 自己的历史版本），但涌现出的模型与"对人类/对外部 AI 的建模"可能不同——对固定基准的迁移性未知。

### 6. Non-stationarity
【事实·论文】Suphx 论文明确记录了自博弈训练的不稳定性对策（熵正则、oracle 之后的 lr/10+截断）。【推测】Pure Zero 下所有 agent 同时从零学，非平稳性叠加冷启动，更容易出现循环克制（非传递性策略环）与能力崩塌；需要 league/PSRO 式机制，成本再翻数倍。

### 7. Multi-agent learning instability
同上。【事实·媒体/推测】LuckyJ 被普遍认为用了大规模自博弈且极强，暗示这些工程问题可解——但没有公开细节，无法借鉴其配方。

### 8. Sample efficiency
【事实·论文】Suphx 从 SL 初始化出发，RL 阶段 250 万局 / 44 GPU。【推测】zero 初始化的经验规律（Atari/go 各类基准）通常需要 10–100 倍样本；麻将可能需要千万–亿局级。当前最强的开源环境吞吐 40k 半庄/小时（4090），亿局 ≈ 2900 GPU·小时纯生成——单卡不现实，中小集群吃力。

### 9. Computational cost
如上。【推测】对单卡 5070Ti：Pure Zero 在有意义的算力预算内**不可行**；对 8×A100 级别：勉强可试，但预期产出/成本远差于 hybrid。

### 10. 是否需要 reward shaping
**需要**（auxiliary loss 形式为主：向听数/进张数预测、自家手牌自编码），小权重 process reward 为辅，且要有衰减与防 hacking 设计（本文档第五节详述）。

### 11. 是否需要 curriculum learning
**需要**。【推测】候选课程：东1局only → 半庄；无点棒 → 完整点棒；oracle 特征渐退（Suphx 式）；固定弱对手 → 自博弈。每种课程本身都是研究问题。

### 12. 是否需要 MCTS/CFR/search
【推测】Pure Zero 大概率**必须**依赖某种搜索来补足前向网络的表征不足——akochan 式 determinized MC 是工程现实，ReBeL 式 belief search 是理论正解但麻将上无先例。这使 Pure Zero 的真实成本 = AlphaZero(搜索) × POMDP(belief) × 4人(nonstationary)。

### 13. 理论上类似 AlphaZero 的可行路径是否存在
【推测】存在一条"理论上的路"：belief-state 自博弈 + 渐进 oracle/课程 + league + 大量算力。但**每一段都没有麻将先例**，整条链的失败概率是各段之积。与"Human → Offline → Self-play"（每一段都有麻将成功先例）相比，作为主路线性价比很低。

### 特别回答：为什么 AlphaZero 能从零学围棋，而麻将 Zero 难度有本质区别？

| 维度 | 围棋 (AlphaZero) | 麻将 |
|---|---|---|
| 信息 | 完全信息，状态即信息集 | 非完全信息：70 张暗牌分布在 3 家手牌+牌山，**策略必须定义在信息集上**【事实】 |
| 转移 | 确定性（落子即转移） | 随机（每巡摸牌是 chance node）；MCTS 的"树"概念被 chance 打散，评估期望需要采样【事实】 |
| 对手 | 2 人零和、单对手 | 4 人、目标=顺位（序数奖励，非线性）、存在垫底保护/避战等非传递动态；自博弈收敛语义不清【事实+推测】 |
| 搜索价值 | 单步搜索可验证战术真值，MCTS 是"推理时计算"的完美场景 | 好坏手要靠数千局统计验证；搜索需先解决信息集建模（determinize 会引入信息泄露，需 belief 加权）【事实+推测】 |
| 奖励信号 | 胜/负/和 二值清晰 | 4 级顺位 × 点数 × 半庄累积；信号更弱更慢【事实】 |
| 自博弈数据效率 | 每局双方都是学习信号（2/2） | 每局 4 家，但就"我的策略改进"而言有效样本密度被 chance 与他人稀释【推测】 |
| 算力事实 | 5000 TPU 量级 | 无同类投入先例；Suphx 44 GPU 已是已公开麻将最大 RL 投入之一【事实】 |

**本质区别一句话**：围棋的 AlphaZero 是"完全信息 + 确定性 + 2 人"三个**可搜索性友好**条件的叠加；麻将在这三个轴上全部处于不利端，任何 AlphaZero 式方案都必须先把"搜索 + belief + 多智能体"三个子系统各自造出来——这就是 ReBeL/Student-of-Games 在扑克上做的事情，而扑克是 2 人且没有立直/副露这种长程组合行为。

---

# 四、Human → RL → Self-play 是否更合理？（A/B/C/D 比较）

### 四条路线定义
- **A** Random → Zero self-play
- **B** Human → SL → Self-play RL（Suphx 式）
- **C** Human → Offline RL → Self-play（Mortal 式）
- **D** Human + Self-play mixed replay（B/C 的强化版：回放池混合人类与自博弈样本）

### 比较

| 维度 | A Zero | B SL→RL | C Offline→Online | D Mixed |
|---|---|---|---|---|
| 最终性能(预期) | 未知/风险最高 | 已验证 8.74 段 | 已验证(官方 vs akochan +1.89pt) | 预期 ≥ B/C，未验证 |
| Sample efficiency | 最差(10-100×) | 好 | 最好(人类转移直接可用) | 好 |
| Training stability | 最差(冷启动+非平稳) | 中(熵/崩塌需技巧) | 好(CQL+MC 稳定) | 中(两种分布混合) |
| 探索能力 | 理论最强实际最浪费 | 受 SL 先验拖累 | CQL 在 offline 段保守，online 段放开 | 好(人类先验+SP 探索) |
| 策略多样性 | 高(但多为噪声) | 低(模仿偏置) | 中 | 中高 |
| 人类策略 bias | 无 | 最重 | 中(CQL 锚定人类动作，Q 可超出) | 随训练稀释 |
| Human ceiling | 理论无 | RL 段可超越(Suphx 证明) | online 段理论无上限 | 同左 |
| 训练成本 | 极高 | 高(44GPU×250万局) | 中(单机可复现) | 中高 |
| 麻将先例 | **无** | Suphx | Mortal(+LuckyJ疑似) | 无完整公开先例 |

关键论断：
1. **不存在理论上的 human ceiling**：B/C/D 的 RL 阶段都在人类数据支撑之外继续优化。真正的"天花板"是工程性的：CQL 只在 offline 段存在，online 段自博弈数据分布由自己产生，改进上限只受算法/算力约束。【事实·代码 + 推测】
2. **但存在"人类偏置的惯性"**：SL/CQL 学到的价值结构带有凤凰桌风格先验，自博弈可能长期在人类策略盆地内打转。D 路线（mixed replay）是打破惯性又不丢常识的折中。【推测】
3. **A 的真实价值不是性能**：是回答"人类牌谱到底提供了多少不可替代信息"（Prompt 第 13 节问题一）的唯一干净实验。

### 如果目标是"最强麻将 AI"
选 **C 为主干 + D 为强化**（Mortal 式离线起步 → online 自博弈 + mixed replay + 代际 league）。理由：起步成本最低（本仓库已走到这一步）、每一段有先例、online 阶段无理论天花板。B 作为对照可选。

### 如果目标是"最有科研价值"
选 **C/D 为主干 + A 为受控 baseline**。因为：
- C/D 能回答"self-play 能否、何时突破人类盆地"（增量科学）
- A 能回答"人类牌谱的不可替代性"（根目录科学问题）
- 单独做 A（无对照）科学价值有限——无论成败都需要 B/C 作参照系

---

# 五、Transformer 是否真的有必要？

先给结论：**"Mortal 平面已经编码了牌河历史"这个说法只对了一部分；Transformer 有真实的增量信息空间，但收益未经证明，且必须与参数量匹配的 CNN 对照。优先级排在 belief 之后。**

### 1. Mortal 当前 feature-plane 是否已隐式编码牌河历史？
【事实·代码】部分编码，且**有明确的信息损失**（`libriichi/src/state/obs_repr.rs:202-260`）：
- 每家牌河：最早 6 巡 + 最近 18 巡的逐项编码（8 通道/项，含吃碰对、是否手切/立直/宝牌）；**中间超过 24 巡的部分不逐项保留**
- 自家牌河另有全量时间衰减平面 `exp(−0.2·巡差)`（v3/4）
- 他家有 v2 式 3 行×6 巡摘要表
- **跨玩家的全局事件顺序（谁在什么巡数对谁鸣牌、立直与放铳的先后因果链）没有任何通道直接保留**，只能由窗口片段间接推断

### 2. Transformer 能否提供额外信息？
【推测】能提供的是：(a) 超出 24 巡窗口的远期历史；(b) 跨家事件的**交错因果序列**（A 立直 → B 切 1p → C 碰 1p 这类事件链）；(c) 距今任意远的事件对的可微关系。但 40 层 CNN 的感受野在 34 宽度上早已全覆盖，问题不是"看不见"而是"抽象层级"——这正是需要 ablation 而不是信仰的地方。

### 3. River sequence 的 token 定义
【推测·推荐】**事件级 token，全局单流**。每个公开事件 = 1 token：
```text
token = Embed_tile(34) + Embed_player(4) + Embed_action_type(8: 摸切/手切/立直/吃/碰/大明杠/加杠/暗杠)
      + Embed_source(4, 鸣牌来源) + Linear([巡数, is_dora, is_aka, 局内进度, 点数差...])
```
一条半庄打到决策点约为 100–250 个事件 token，全剧 ≤384。

### 4. 每次打牌一个 token，还是每家一条序列？
【推测·推荐】**全局交错单流 + player-id embedding**，而不是 4 条独立序列：鸣牌事件天然连接两家，拆开会丢掉这些跨家边；Transformer 的 self-attention 恰好擅长建模"A 的行为对 B 的含义"。若嫌全长贵，可退化为"每家独立 encoder + 决策时 cross-attention 融合"的两级方案。

### 5. 是否需要 positional encoding？
【推测·需要】三套都要：全局巡数 embedding（绝对时间）+ 各家局内事件序号（相对时间）+ kyoku 边界标记。** causal mask 沿全局事件序**，保证任意决策点只见前缀（与 mjai 事件流一致，防未来信息泄露）。

### 6. 如何处理非打牌事件（吃碰杠立直和牌）？
【推测】全部作为一等 token：吃/碰/杠携带"从谁鸣了什么"（source player + tile），立直携带宣言牌与是否振听语义，和牌/流局作为局边界 token（跨局保留分数上下文但**牌河 attention mask 按局重置**——上一局的牌河对新局无物理意义，点数/供托才跨局有效）。

### 7. 计算成本是否值得？
【事实·代码+推测】成本结构：Mortal 环境 40k 半庄/h 是 Rust 端吞吐；加 Python 端 Transformer 会打断 batch 推理管线。可行方案：(a) 事件流也在 Rust 端产出（新写 encoder，工程量中等）；(b) Transformer 只做 4–8 层 d≈256，FLOPs 增量 ≈ CNN 主干的 30–60%；(c) KV-cache 复用使逐步推理成本可控。**训练成本不是主要障碍，Rust 端重写事件 encoder 才是。**

### 8. 如何设计 ablation 才能证明它真的有效？
【推测·关键设计】三组对照缺一不可：
```text
E2: param-matched 更大 CNN（把 Transformer 的参数量还给 CNN：如 192→256 通道或 +10 blocks）
E3a: Mortal 原样 + River Transformer 融合（concat 后过 MLP）
E3b: Transformer-only（去掉 CNN 主干，纯事件流）——风险实验，预期不敌 CNN，用于定位信息究竟在平面还是事件流
```
判定标准：1v3 duplicate ≥12k 局（见第九节功效计算），E3a 比 E2 高 ≥0.5pt 才算"Transformer 有独立贡献"。**E2 是最容易被省略也最不能省的一组**——没有它，任何涨点都可能是"参数变多"的伪发现。

---

# 六、Opponent Hidden State / Belief State

先给结论：**方向正确，且是本仓库性价比最高的研究增量——因为 Mortal 代码里 oracle 监督的基建已经存在（217 通道 invisible_obs + `Invisible` 生成器），只欠一个 belief 头和损失。但"hidden state 必须被解释为分布，而非猜测的手牌"。**

### 1. 这个设计是否合理？
合理，但要正确定位：h_opp_i 应是"对手公开历史的 learned sufficient statistic"，其价值要靠**下游 aux 任务**来锚定，否则只是多一份冗余 embedding。【事实·代码】Mortal 已有共享的对手信息通道（四家牌河/副露平面），缺的是"专门为对手建模而组织"的表示与监督信号。

### 2. hidden state 是否应该被解释为 belief state？
**是，且必须**。【事实·论文】这是 ReBeL（[arXiv:2007.13544](https://arxiv.org/abs/2007.13544)）的核心语义：策略定义在 public belief state（所有信息集上的分布）上。【推测】若不显式管理这个语义，网络会学出"确定性伪信念"——在自博弈中碰巧对自家人有效、对外部对手迁移时崩溃。

### 3. 是否应该显式预测 opponent hand distribution？
**应该**，且监督信号免费：【事实·代码】`Invisible::new(events, trust_seed)` + `oracle_obs_shape=(217,34)`（`gameplay.rs:164`、`consts.rs:32`）在数据管线里就能拿到任意时刻 3 家的真实手牌与牌山。训练时用它做 dense supervision，推理时它自然退化为模型的 belief——这正是 Suphx oracle guiding 的监督来源，Mortal 的代码预留了同款能力但当前训练未使用。

### 4. 如何避免模型学习出错误的 certainty？
【推测·四条】(a) 目标设为**分布**（见第 9 问的 34×5 直方图）而非手牌点估计；(b) 用 proper scoring rule（CE 对直方图、BCE 对事件）+ 训练后做 ECE 校准检查；(c) belief 头不直接 argmax 进决策，而是以 embedding 参与 fusion（保留软信息）；(d) 对"确定不可能"事件（如已被 4 张全见的牌）显式给结构约束，让"确定"有真实语义。

### 5. 如何训练 opponent belief？
【事实·代码 + 推荐方案】人类牌谱上：h_opp_i 由该对手截至 t 的公开事件编码，用 oracle 通道拿真值监督（手牌直方图 / 立直待牌 / 危险度）。自博弈上：同样可生成 oracle 标签（环境知道所有牌），零成本。两个阶段监督都不断供。

### 6. 是否应该使用 Bayesian / particle / distributional representation？
【推测·务实答案】三层方案的性价比排序：**直方图头（34×5 分类）> distributional latent > 粒子**。粒子/贝叶斯滤波在麻将的组合空间（C(70,13) 级）上维护成本失控；直方图头 + 校准是工程与理论的最佳平衡点。若研究目标是"分布表示"本身，可以加一个 4–8 个 component 的 mixture 头做对照。

### 7. 是否需要 opponent-specific recurrent state？
【推测·不需要（第一版）】麻将跨局状态只有点数/供托（已编码），局内事件流用 causal Transformer 天然承载"截至现在"；RNN 的增量推理优势对训练无用、对部署收益有限。除非未来做 Rust 端流式部署，否则先不上 RNN。

### 8. Transformer 和 RNN 哪个更合适？
【推测】建模质量 Transformer 优（长程 suji/筋关系、并行训练）；部署效率 RNN 优。鉴于本文档推荐 Transformer 分支存在与否本身待 ablation，**belief 头建议设计成"编码器可替换"**：同一组 aux 头，喂 CNN 分支 embedding 或喂事件 Transformer embedding 都能跑，这样 belief 实验与架构实验解耦。

### 9. 是否应该预测 34×5 的 tile-count distribution？
【推测·强烈推荐，这是本文档认为最漂亮的一个设计】对每个对手 i（及牌山），预测**每种牌 x 在该对手手里的张数分布**：`P(count=x | tile, opponent_i)`，x∈{0,1,2,3,4}，形状 34×5（×4 个归属槽）。理由：
- 直接给出安全牌推断所需的全部统计（放铳率 = Σ_tile P(opponent 等该牌) × 我切该牌后的收益）
- 与"自己手里有几张该牌"联动（我持有 2 张则对手至多 2 张，可作硬约束）
- 可校准、可解释、可直接可视化（论文图好做）
- 真值免费（oracle 通道）

### 10. belief state 最终应该如何影响 policy/value？
【推测·两段式】第一阶段：belief 头只做 aux 任务（正则化 + 表征塑造），fusion 只进 value/advantage 通路；第二阶段（若 aux 有效）：belief 显式 condition policy（cross-attention 到 Q 头），重点观察**攻防边界行为**（立直后弃牌选择、副露时机）的变化——belief 的价值最可能在这些"对手依赖"决策上兑现，而不是在手牌效率上。

---

# 七、PPO vs DQN vs AWR vs CFR

### 逐算法比较（四人非完全信息麻将）

| 算法 | offline 兼容 | self-play 兼容 | 隐藏信息处理 | 动作空间 | 探索 | 稳定性 | 算力 | 样本效率 |
|---|---|---|---|---|---|---|---|---|
| Dueling DQN + CQL | ★★★(专为此设计) | ★★(低熵贪心自博弈可用) | 无特殊(靠网络) | 天然适配离散+mask | 弱(贪心) | 好(MC target) | 低 | 好 |
| PPO | ★(需IS修正/BC约束) | ★★★(原生) | 无特殊 | 适配(mask+熵) | ★★★(可控熵) | 中(4人非平稳) | 中 | 差 |
| AWR | ★★(offline 政策提取) | ★★(可做蒸馏) | 无特殊 | 适配 | 中 | 好(回归式) | 低 | 好 |
| Vanilla PG/REINFORCE | ★ | ★★ | 无特殊 | 适配 | 大 | 差(高方差) | 中 | 差 |
| CFR / Neural CFR | ★(off-policy 原生) | ★★(自博弈迭代) | ★★(regret 按信息集) | 理论适配但 regret 传播成本爆炸 | 好 | 好(收敛保证) | 极高 | 差 |
| ReBeL/search | ★ | ★★ | ★★★(public belief state) | 适配 | 好(搜索内) | 好 | 极高 | 中 |
| MuZero-like | ★ | ★★ | 无特殊 | 需改造(随机+多人) | 中 | 差(模型学习+多人) | 极高 | 中 |

注：★ 为定性评级，依据下列一手事实：
- 【事实·论文】CFR 的收敛保证（ε-Nash）严格成立于**两人零和**；四人（即使零和）保证不直接适用，且麻将每半庄数百决策点的 regret 传播在现代硬件上不现实。Neural CFR 家族（ESCHER [arXiv:2308.04035] 等）的成功案例集中在扑克（2 人）。
- 【事实·论文】ReBeL 在 2 人扑克从零可行，但需要 public belief state 的精确表示与巨额算力；Student of Games（[arXiv:2305.03449](https://arxiv.org/abs/2305.03449)）同理。
- 【事实·论文】Suphx 用 PG 自博弈达到 8.74 段，但配了熵正则、oracle guiding、lr/10 截断等大量稳定化技巧——"PG 自博弈开箱即用"不成立。
- 【事实·媒体】GOKU（IPA 未踏 2022，[评估资料](https://www.ipa.go.jp/jinzai/mitou/it/2022/ps6vr70000014g43-att/hyouka_sd-1.pdf)）用 PPO + 人类数据自博弈到天凤七段——PPO 可行但未见超越 Suphx 级别的证据。
- 【事实·代码】Mortal 自博弈的采样配置（boltzmann_temp=0.05）说明其作者在 online 阶段也未采用大熵探索的 PG 式自博弈。

### 如果从 Mortal 开始改，下一步最应该尝试什么？

**按性价比排序，不是按"新颖度"排序：**

1. **启用 oracle 监督的 belief aux 头**（第六节方案）——零环境改动、基建已在代码里、直接产生可发表的校准/消融结果
2. **打通 online 自博弈 + mixed replay**（D 路线）——基建已在，回答"self-play 能否突破离线平台期"，这是当前最有信息量的实验
3. **param-matched 的架构对照组**（CNN 加宽 vs +River Transformer vs Transformer-only）——第五节方案
4. **oracle guiding 小试**（γ 0.5→0 渐退，离线阶段 50–100k 步）——验证 Suphx 技巧在本管线的迁移性
5. PPO/AWR 政策头——只在 online 阶段出现价值瓶颈（如低熵自博弈多样性不足）后再上

---

# 八、请设计一个完整的新模型

按 Prompt 要求给全管线。**分两档：理想多 GPU 档 与 本仓库现实（单卡 5070Ti）可行档。** 两档共用设计，只缩放规模。

### 8.1 理想档（8×A100 级）

```text
Input
├── Obs planes v4+(1012, 34)                    # 沿用 Mortal 编码，冻结其 Rust 实现
└── Event stream (B, T≤384, d_raw≈64)           # 新增：Rust 端逐事件编码

Encoder
├── CNN branch: ResNet1D-BB
│     conv_channels=256, blocks=40, CA+Mish
│     → Conv(256→32) → Flatten → Linear(32×34→1024) = h_state (B, 1024)
└── Event branch: Event Transformer
      6 layers, d_model=256, heads=8, ffn=1024, pre-LN, causal mask
      pos = Embed(global_turn) + Embed(per-player idx) + kyoku-boundary token
      → 取最后事件位聚合 = h_river (B, 256)

Opponent Belief（共享权重，3 个 learned query）
├── h_opp_i = CrossAttn(q_i ∈ R^{3×256}, KV=event tokens)  (B, 3, 256)
└── heads（每个 h_opp_i 上）:
    ├── hand histogram: Linear(256→34×5)  softmax over count ∈ {0..4}
    ├── riichi wait:    Linear(256→34)    （立直宣言后启用，否则 mask）
    └── danger:         Linear(256→1)     BCE（oracle 真值：切此牌是否放铳）

Fusion
└── concat(h_state, h_river, h_opp_1..3) (B, 2048) → MLP(2048→1024) Mish

Policy head（新增，服务 online/AWR/未来 PPO）
└── π = masked softmax(Linear(1024→46))

Value head（保持 Mortal 语义）
└── Dueling: Linear(1024→1+46)，Q = V + A − mean(A)，mask −inf

Auxiliary heads
└── next-rank: Linear(1024→4) CE（Mortal 原有，保留）
```

### Loss（离线阶段）

```text
G_grp          = Σ_k γ^k · ΔE[pts]_k            # Mortal 原有 MC 回报，γ=1
L_dqn          = 0.5 · MSE(Q(s,a), G_grp)
L_cql          = logsumexp_a Q(s,a) − Q(s,a_human)      # w=5
L_belief_hist  = CE(hist_opp, true_counts)              # oracle 真值，w≈0.5
L_belief_wait  = CE(wait_opp, true_wait)                # w≈0.1
L_belief_dang  = BCE(danger, dealin)                    # w≈0.1
L_next_rank    = CE(next_rank, player_rank)             # w=0.2
L_awr          = −Σ exp((Q−V)/β) · log π(a|s)           # β=1, w≈0.1，可选

L = L_dqn + 5·L_cql + 0.5·L_hist + 0.1·L_wait + 0.1·L_dang + 0.2·L_next_rank (+0.1·L_awr)
```

自博弈阶段（online）：去掉 L_cql，人类回放占比 30%（mixed replay），熵正则可选，其余不变。

### 训练流程（理想档）

```text
0. GRP 不变（已有），事件 encoder 的 Rust 重写 = 主要工程量
1. Offline 阶段: batch 1024, lr 1e-4 cosine, 300–600k 步, 8×A100 ≈ 3–5 天
2. 生成 gate: 1v3 duplicate 50k 局 vs 离线基线；不过 gate 不进自博弈
3. Online 阶段: 40–80 CPU workers + 2 GPU 推理;
   league = {最新 challenger, 冻结的上一代 champion, 离线基线} 按比例采样对手;
   boltzmann τ 0.2→0.05 退火; buffer: 30% 人类 + 70% 近期自博弈(容量 200 万转移)
4. 每代: 1v3 50k 局 + akochan 锚点 10k 局；胜过 champion 才换代
5. 重复 3–4 至收益平台
```

### 8.2 单卡现实档（5070Ti 16GB，本仓库直接可执行）

```text
规模缩减: CNN 40×192(现检查点可直接 warm-start) ; Transformer 4 层 d=192;
          belief 头输入 256 维; batch 512 AMP
预算:     offline 300k 步 ≈ 28–34 h(实测 ~2.7 it/s, runs/dqn_train.log);
          自博弈生成 ~20k 半庄/晚(按 4090 40k/h 的 50% 折算【推测】)
          每代评估 12k 局 ≈ 1–2 h
现实节奏: 一代(offline 增强 + 自博弈 100k 局 + 评估) ≈ 3–5 天
```

其余设计（loss、league、gate）与理想档一致，只是把"50k 局 gate"降为"12k 局 gate（CI ±1.5pt）"。

---

# 九、设计实验（12 个 ablation + 统计功效）

### 9.1 先算清楚统计功效（这是所有实验的前提）

【事实·本地数据】从我们 2000 局 1v3 结果反推：challenger 顺位分布 [508, 504, 527, 461] → 顺位 pt 方差 sd ≈ 82 pt（pt=[90,45,0,−135]）。1v3 duplicate 的 game 级方差未被 seat 配对消除：

```text
n = 2000  局:  se ≈ 1.84 pt  →  95% CI ≈ ±3.6 pt   （我们当前的 +3.08pt 仅 ~1.7σ，不显著!）
n = 12000 局:  95% CI ≈ ±1.5 pt
n = 30000 局:  95% CI ≈ ±0.9 pt
n = 1e6   局:  95% CI ≈ ±0.16 pt  （Mortal 官方 strength 页的规模）
```
【事实·文档】Mortal 官方对比用 1M 半庄（种子 10000–260000，[strength 页](https://mortal.ekyu.moe/perf/strength.html)）。
【推测·结论】**单代差异 <1.5pt 的实验在 12k 局以下无法区分于噪声**；所有架构消融必须 ≥12k 局或使用"多 seed 重复 + 合并"设计。检验用 per-hanchan pt 的 Welch t / bootstrap percentile CI，辅以顺位多项分布 χ²。评估用时：12k 局 ≈ 1–2 h（单卡），完全可承受。

### 9.2 实验矩阵

| # | 实验 | 回答的问题 | 控制变量 | 主指标 | 判定(淘汰线) | 优先级 |
|---|---|---|---|---|---|---|
| E0 | 基线复现 + 方差刻度 | 我们管线的 run-to-run 方差多大 | 同配置 3 seeds | 评估 CI 宽度 | — | ★★★ |
| E1 | 评估功效升级(纯评估) | +3.08pt 是否真实 | n: 2k→12k | 95% CI | — | ★★★ |
| E2 | param-matched 大 CNN | "更大=更强"伪对照 | 参数量=FLOPs 匹配 E3a | 1v3 pt | — | ★★★ |
| E3a | + River Transformer | 事件流是否有增量 | 同 E2 参数量, 同数据 | 1v3 pt ≥12k 局 | <E2+0.5pt → 淘汰 | ★★ |
| E3b | Transformer-only | 信息到底在平面还是事件流 | 与 E2/E3a 同数据 | 1v3 pt | 预期淘汰(定位实验) | ★ |
| E4 | + belief aux 头 | oracle 监督表征是否有增量 | 架构同 E2, 仅加 aux | 1v3 pt + ECE | ECE>0.15 或无 pt 增益 → 淘汰 | ★★★ |
| E5 | belief condition policy | belief 进决策通路的价值 | E4 基础上加 fusion | 攻防行为统计 + pt | 攻防统计无差异 → 淘汰 | ★★ |
| E6 | oracle guiding | Suphx 技巧迁移性 | γ: 0.5→0, 50k 步 | 1v3 pt | 与对照组持平 → 淘汰 | ★★ |
| E7 | SL-only BC 下限 | 纯模仿天花板参照 | 同数据同骨干 | 1v3 pt | — (参照系) | ★★ |
| E8 | SL→AWR vs DQN | 政策学习是否需要 Q | 同骨干 | 1v3 pt + 训练时长 | 无优势 → 淘汰 | ★ |
| E9 | offline→online mixed replay | **self-play 能否突破离线平台** | 30% 人类混合, 代际 league | 代际 pt 曲线 | 2 代无增益 → 停 | ★★★ |
| E10 | GRP reward 消融 | GRP 到底值多少 | 真终局 pt(稀疏) vs GRP | 收敛速度 + 最终 pt | — | ★★ |
| E11 | Pure Zero (预算制) | 人类牌谱不可替代性 | 5M 局上限, 1 局制课程+aux shaping | 是否学会立直/和牌 | 2M 局无立直 → 终止(仍可发表) | ★ |

### 9.3 执行原则

- **最重要三个**：E1（否则一切结论建立在噪声上）、E4（性价比最高的科学增量）、E9（回答核心研究问题"self-play 能否超越"）
- **最先淘汰的**：E3b、E8——信息定位型实验，产出是"负结果+理解"，不投入大预算
- **控制变量总则**：同一数据管线（同 file_index）、同训练超参基线、同 12k 局评估协议、同 champion（148.8k）；架构实验一律带 E2 参数量对照组
- **游戏量与统计意义**：如 9.1 所算，12k 局是单代结论的下限；跨实验比较用同 seed 池 + 配对检验

---

# 十、请特别挑战我的假设（逐条）

**1. "Transformer 一定比 CNN 更适合牌河"**
不成立（目前）。【事实·代码】v4 平面已含窗口化+衰减的牌河编码，40 层 CNN 感受野覆盖全部空间宽度；"看不见 24 巡前的中间历史"是真的，但该信息对决策的边际价值未证明。三个最强麻将 AI（Suphx/Mortal/NAGA）没有一个靠 attention 达到当时 SOTA。判定：**待 E3 检验的假设，不是前提**。

**2. "Opponent hidden state 一定有用"**
部分成立。隐式通道已让 Mortal 达到顶级；显式 belief 的增量最可能兑现于攻防边界决策，且必须靠 aux 任务锚定（否则学到的是冗余 embedding）。判定：**值得做，但预期是"结构化改进+可解释性"，不是质变**。

**3. "预测 opponent hidden hand distribution 一定有用"**
不必然。aux 准确率提升 ≠ 端到端胜率提升（辅助任务可以只当正则化器）。它的确定性价值在校准与安全（放铳控制）与可解释性。判定：**做，但用 E4/E5 的双指标（pt + 攻防行为）验收，别用 aux 准确率自我安慰**。

**4. "PPO 比 DQN 更现代，所以应该更强"**
不成立。离线数据上 PPO 需要重要性修正；自博弈上 PPO 的成功案例（GOKU，七段）低于 Suphx/Mortal 级别；DQN+CQL 已被验证到凤凰桌级强度。PPO 的真实适用场景是 online 阶段需要大熵探索时。判定：**"现代"不是理由，问题结构才是**。

**5. "Pure Zero 一定能突破人类"**
无任何证据支持。四人 POMDP 从零自博弈没有先例；ReBeL 也是 2 人 + 巨算力 + 精确 belief 表示。判定：**作为科研问题值得回答（E11），作为性能路线不成立**。

**6. "最终 reward 足够，不需要 reward shaping"**
错。【事实·论文+代码】Suphx（GRP）与 Mortal（GRP 稠密 reward）的成功都依赖中间价值估计；纯终局 reward 意味着 8–12 局才有一个信号，sample efficiency 会差数倍。但 shaping 的正确形式是 **auxiliary loss 为主 + 小权重 process reward 为辅**，且防 hacking（向听数不能当 reward 主项——会被"无脑逼近听"利用）。判定：**完全去 shaping 是倒退，全 shaping 是 hacking，分层是正解**。

**7. "Self-play 一定比 human data 更好"**
假二分。【事实】所有已验证顶级 AI 都是 hybrid：人类数据管冷启动与先验，self-play 管超越。Pure self-play 无先例，Pure human（NAGA）有明显上限。判定：**问题是"比例与切换时机"（E9），不是"选边"**。

**8. "麻将最终可以直接照搬 AlphaZero"**
不成立。非完全信息 + 随机性 + 4 人三个维度同时不友好（第三节详表）。判定：**可行的远景是 ReBeL 式 belief search，不是 AlphaZero 式 MCTS；且那是第三阶段以后的事**。

**9. "CFR 一定比 PPO 更适合麻将"**
未证明且大概率不可行。CFR 收敛保证只严格成立于 2 人零和；四人 + 数百决策点的 regret 传播算不起；"LuckyJ 用 CFR"是网络传闻，无一手证据。判定：**CFR 的真正遗产是"按信息集组织 regret"的思想，它在麻将的对应物是 belief 建模，不是 CFR 算法本身**。

**10. "更大的模型一定更强"**
不成立。【事实+推测】离线阶段受数据 support 约束（更大模型更容易过拟合人类分布、放大 CQL 与数据的拉锯）；自博弈阶段受推理吞吐约束（40k 局/h 的环境瓶颈下，大模型的采样成本直接拖慢迭代）。Mortal 自己的 b40→mortar 扩展是配合数据/算力同步的。判定：**规模要和数据与自博弈量一起放大，单独放大是浪费**。

---

# 十一、最终推荐

### 如果我是你，我会：

**第一阶段（1–2 个月，单卡，零环境改动）**
1. E1：评估协议升级到 12k 局 + CI 报告（1–2 天，把现有 +3.08pt 变成可发表的数字或证伪它）
2. E4：belief aux 头（34×5×4 直方图 + 立直待牌 + 危险度，oracle 真值），warm-start 现有 175k 检查点，200k 步
3. E6：oracle guiding 小试（γ 0.5→0，50k 步）
4. E0：3-seed 方差刻度
> 产出：麻将 AI 的**校准 opponent belief** 消融——科研增量 #1

**第二阶段（3–6 个月，单卡节奏 + 可选扩卡）**
5. E9：启用现成的 server/client 自博弈基建，mixed replay（30% 人类），两代一 league，每代 12k 局 gate——回答"self-play 能否突破离线平台期"
6. E3a/E2：River Transformer 与 param-matched CNN 对照
7. akochan 作为外部强度锚（与自家 baseline 脱钩的绝对标尺）
> 产出：**Human→Offline→Online 全流程的严格消融**——科研增量 #2

**第三阶段（研究性，设预算上限）**
8. E11：Pure Zero 课程实验（1 局制 + aux shaping，5M 局 kill switch）——回答"人类牌谱的不可替代性"
9. 推理时 belief search：把 belief 网络接入 akochan 式 determinized MC 的采样权重（belief-weighted determinization）——向 ReBeL 语义靠拢的第一步
> 产出：**零牌谱基线 + belief-weighted search**——科研增量 #3

### 我认为最有希望的最终架构

```text
                    ┌────────────────────────────────┐
  Obs planes v4+    │  ResNet1D-BB (40 × 256, CA+Mish)│ → h_state (1024)
  (1012 × 34)  ───→ │  (v4 兼容, 旧检查点可 warm-start) │
                    └────────────────────────────────┘
  Event stream ───→ Event Transformer (6L, d256, causal, 全局交错单流)
  (≤384 tokens)         ├──→ h_river (256)
                        └──→ CrossAttn(3 queries) → h_opp ×3 (256)
                                 ├── hand histogram 34×5×4   ← oracle 监督
                                 ├── riichi wait 34          ← oracle 监督
                                 └── danger (BCE)            ← oracle 监督

  concat(h_state, h_river, h_opp×3) → MLP(2048→1024)
       ├── Dueling Q (1+46)  ← MC-GRP target + CQL(offline) + mixed replay(online)
       ├── π (46, masked)    ← AWR 蒸馏(offline) / 小熵 PPO(online, 可选)
       └── next-rank (4)     ← 保留 Mortal aux

  L = L_dqn + 5·L_cql(仅offline) + 0.5·L_hist + 0.1·L_wait + 0.1·L_dang
      + 0.2·L_next_rank (+ 0.1·L_awr, online 可选)
```

设计原则一句话：**CNN 管牌型空间结构，事件 Transformer 管跨家时序因果，belief 头管隐藏信息——三者的贡献各自可被独立消融，任何一路无效都能无痛摘除。**

### 最值得做、最可能产生真正科研增量的 3 个方向

1. **Oracle 监督的校准麻将 belief 建模**：34×5×4 tile-count 直方图 + ECE 校准评估 + belief-conditioned 攻防决策。基建已存在于 Mortal 代码（oracle 通道 + Invisible 生成器），没有任何已发表工作做过麻将 belief 的校准研究——这是低成本高确定性的增量。
2. **单卡可复现的 Offline→Online mixed-replay 全流程与代际消融**：公开社区至今没有一份严格的"self-play 何时/能否超越离线基线"的量化研究；Mortal 的 online 基建 + 我们已有的代际评估管线正好构成实验平台。配套贡献是**评估统计协议**（12k 局 / CI / 配对检验——现有文献普遍用几百局下结论）。
3. **人类牌谱不可替代性的受控度量**（Pure Zero 课程 vs Human-warm-start 的同预算对比）：这是麻将 AI 里最清晰、答案最有普适价值的开放问题（对其它 4 人 POMDP 也有外推价值），且我们的负结果也会有人引用。

---

# 附录：参考链接

**一手来源**
- Mortal 代码（AGPL-3.0）：https://github.com/Equim-chan/Mortal
- Mortal 官方文档（大部分页面空置）：https://mortal.ekyu.moe/ ；strength 页：https://mortal.ekyu.moe/perf/strength.html
- Mortal 权重发布讨论（作者 gist）：https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56
- 社区权重镜像：https://huggingface.co/VoidShine/mortal-298k
- Suphx 论文：https://arxiv.org/abs/2003.13590
- NAGA 官方技术文章（DMV, 2019）：https://dmv.nico/ja/articles/mahjong_ai_naga/ ；服务页：https://naga.dmv.nico/naga_report/top/
- LuckyJ 官宣（媒体）：https://www.stcn.com/article/detail/918748.html ；官方 Twitter：https://twitter.com/LuckyJ1443836 ；知乎讨论：https://zhuanlan.zhihu.com/p/642827719
- Meowjong（3 人麻将 SL+REINFORCE 小规模研究，未引用 Mortal）：https://arxiv.org/abs/2202.12847
- GOKU（PPO+人类数据，天凤七段，IPA 未踏 2022）：https://www.ipa.go.jp/jinzai/mitou/it/2022/ps6vr70000014g43-att/hyouka_sd-1.pdf
- akochan：https://github.com/critter-mj/akochan
- mjai 工具集（同作者）：https://mjai.ekyu.moe/

**方法论文献**
- CQL（Kumar et al., 2020）：https://arxiv.org/abs/2006.04779
- AWR（Peng et al., 2019）：https://arxiv.org/abs/1912.07729
- IQL（Kostrikov et al., 2021）：https://arxiv.org/abs/2110.15155
- AlphaZero：https://arxiv.org/abs/1712.01815
- MuZero：https://arxiv.org/abs/1911.08265
- ReBeL（Brown et al., 2020）：https://arxiv.org/abs/2007.13544
- Student of Games（Schmid et al., 2023）：https://arxiv.org/abs/2305.03449
- ESCHER（Neural CFR 变体）：https://arxiv.org/abs/2308.04035
- Self-Play Survey（含 LuckyJ 引用方式）：https://arxiv.org/abs/2408.01072
- 教师学习+RL 两段式麻将 AI（九州大）：https://api.lib.kyushu-u.ac.jp/opac_download_md/4740670/4740670.pdf

**本仓库证据**
- 模型/损失：`vendor/mortal/mortal/model.py`、`vendor/mortal/mortal/train.py`
- 观测编码：`vendor/mortal/libriichi/src/state/obs_repr.rs`、`vendor/mortal/libriichi/src/consts.rs`
- oracle 通道：`vendor/mortal/libriichi/src/dataset/gameplay.rs`
- 自博弈基建：`vendor/mortal/mortal/server.py`、`mortal.py`、`client.py`、`engine.py`
- 当前评估：`runs/1v3/eval_20260915_1835.log`（+3.08pt / 2000 局，95% CI ±3.6pt）
