# RiichiFormer V2 — Agent 可执行工程说明书

**版本**：v1.0 · 2026-11-21  
**目标仓库**：`https://github.com/ygmldr/riichi-ai`  
**设备**：NVIDIA RTX 5070 Ti 16 GiB，单卡；训练/开发环境由 Agent 现场检测  
**文档定位**：直接交给 Coding Agent 执行的 **规范和验收契约**，不是已完成的代码或已经跑过的实验。  
**适用范围**：四人立直麻将东南半庄；合法 POV 可观测信息；本地规则引擎、离线专家数据、后续自博弈。  
**优先级**：正确性 > 统计可信度 > 单卡可运行 > 性能提升 > 规模扩展。

---

## 0. 开发 Agent 执行约定（每一个任务都要遵守）

### 0.1 权限与边界

1. **不得直接覆盖现有 Mortal 基线、训练配置、checkpoint、既有实验报告、牌谱**；新代码一律放入第一方 `riichiformer/`、`scripts/rf_*.py`、`tests/rf_*`、`configs/riichiformer/`、`docs/riichiformer/`。需要扩展 Rust API 时优先在自己的分支维护独立 patch 或 wrapper，**不可偷偷改 submodule 的未追踪内容**。
2. 默认禁止自动抓取新的天凤/雀魂玩家牌谱，禁止登陆线上账户、注入游戏客户端、规避封禁；数据许可证确认前，只用当前已经有权访问的本地牌谱做**索引/审计**。如来源不允许该用途，停止该来源相关处理。
3. 不使用未授权的 AI 教师批量推理服务；本地已授权 checkpoint 可列为独立来源，但需检查模型许可和派生使用条款。
4. 训练不得在没经过 GPU benchmark、没有锁定数据 split、没过 POV 防泄漏测试的前提下启动。
5. 遇到依赖不可用、GPU OOM、Rust/PyO3 API 变更时，**输出 BLOCKED、复现实例及候选处理方案**，不要伪造 PASS，也不要悄悄跳过损坏牌谱。
6. 一次只做一个 PR。PR 输出包括：`git diff --stat`、测试命令与完整退出码、manifest/checksum、未解决风险、下一张卡片编号。**不要因阶段过长就自行继续下一个 PR**。
7. 若 Agent 能运行测试，则实际运行；若只有仓库读权限，清楚标注“未执行”。所有 `rf_*` 命令是本方案要求**实现的新 CLI**，绝非仓库当前已有命令。

### 0.2 一次任务的标准工作流

```text
1. git status --short && git rev-parse HEAD && git submodule status
2. 读本任务卡 + 本说明相关契约 + 所依赖上游真实代码
3. 写一个最小失败测试（RED）；确认真实失败而非配置缺失
4. 实现最小正确功能（GREEN），不得删旧测试
5. 本地 unit/integration/fixture 测试；记录真实命令和退出码
6. 静态审查：泄漏、未来数据、动作映射、schema、许可、I/O 安全
7. 输出 docs/riichiformer/pr_reports/PR-XX.md（含证据路径及 SHA）
8. 停止，等待下一任务
```

任务报告模板：

```markdown
# PR-XX Evidence
base_commit: <sha>   work_commit: <sha or uncommitted>
implemented: ...
files_changed: ...
commands_run:
  - cmd: '...'
    exit_code: 0
    evidence: '...'
checks: {unit: PASS|FAIL|NOT_RUN, integration: ..., gpu: ...}
artifacts: [{path: '...', sha256: '...'}]
metrics: {rows: 0, decisions: 0, invalid_pct: 0.0, ...}
known_risks: ...
next_ticket: PR-XX
```

### 0.3 阶段门与停止规则

- **G0（冻结）**：现有基线权重/源码/旧测试和评测协议均已固定，不能改写。
- **G1（数据）**：数据可追溯、来源权限明确、100% 样本来源和 split 可解释，非法输入 fail-closed。
- **G2（可见性）**：POV redaction、逐时刻因果、动作 index、训练与 arena parity 全部通过。**P0：任何可见性违规即阻塞训练**。
- **G3（小模型）**：可复现、无 NaN、明显优于随机/多数类、可出合法动作、跑通本地 arena，记录真实显存。
- **G4（数据选择）**：同预算数据质量/年代实验完成，明确改善是否具有统计和实际意义。
- **G5（结构）**：局内 vs 跨局 vs flat 在统一评测协议下完成，复杂度收益有数字。
- **G6（后续研究）**：Belief 和 RL 各自先通过独立测试，才准加入主策略。

通过一个阶段仅意味着**代码和证据满足当阶段 gate**，并不表示已经达到天凤某段位。

---

## 1. 现有代码真实接口及迁移原则（已核对 2026-11-21 仓库）

| 组件 | 现状 | V2 做法 |
|---|---|---|
| `vendor/mortal` | Mortal 固定 submodule，指向 pinned upstream `0cff2b5...`，仓库另有 `patches/0001–0013` | 不覆盖。沿用 Rust 规则引擎、动作定义、原生 eval 框架 |
| `libriichi/src/mjai/event.rs` | `Event::StartKyoku` 带全体 `tehais`；`Event::Tsumo` 带真实 `pai` | 输入模型前按 POV 投影；不能直接序列化原始事件 |
| `libriichi/src/dataset/gameplay.rs` | `GameplayLoader::load_events` / `load_gz_log_files` 产生**逐动作**数据；在 `extend_from_event_window` 用 future next 判断历史动作 | 用于 oracle 对齐测试，不能直接作为全半庄流数据结构 |
| `libriichi/src/state/player_state.rs` | `PlayerState::update` 更新动作状态；原有 `Gameplay` 提供 actions/masks | 规则真值与动作 mask 的唯一权威来源 |
| `mortal/engine.py` | `MortalEngine.react_batch()` 输出 Q、greedy 采样 | 只作冻结基线/教师；新策略输出 masked policy logits，不把 Q 当策略 |
| `scripts/eval_report.py` | 按 `(seed, seed_key)` 的四个 split 配对，set-level bootstrap，点数 `(90,45,0,-135)` | 复用统计，另做新引擎的评估适配，不改变历史报告 |
| `configs/config.toml` | 旧 DQN/GRP/online；路径为作者机器绝对路径 | 新实验只读自己的 YAML，不改变旧 config |
| `scripts/download_data.py` | 年度 2009–2026 第三方归档下载器 | 默认不调用；只有来源/许可复核通过再执行 |
| `scripts/verify_data.py` | 统计的是年度 `.mjson` **文件数**，命名为“局数” | 审计报告改称 `game_logs`，实际 `kyoku_count` 由重放统计 |
| `RUNBOOK.md` | 旧环境和 scripts 调用方式 | 新增独立 v2 runbook；不与旧后台训练脚本混用 |

**证实的旧缺陷**：`patches/0006-event-stream-tokens.patch` 的 `encode_event_token(Event::Tsumo {actor,pai})` 无论是否为自己摸牌都记录 `pai`；`encode_event_log(log,pov)` 的 `pov` 只用于计数 turn。因此旧 Event Transformer **泄漏对手摸牌**，同时未编码自家初始 13 张手牌。不允许在新管线复用这段 tokenizer。此判断不代表旧纯 CNN Mortal 线上版本必然有该泄漏；具体部署模型需按 checkpoint 配置确认。

**参考原型**：先前提供的 `RiichiFormer-S-v0.1.zip` 仅是可反向传播的 skeleton，**没有 MJAI tokenizer、在线适配、全数据实测**；可以借鉴其 `model.py`，不可直接宣称完成新模型。须按本方案 v2 schema 对齐。

---

## 2. 目标目录树（Agent 应逐 PR 创建，不要求先全部生成）

```text
riichi-ai/
├── riichiformer/
│   ├── __init__.py
│   ├── schemas/
│   │   ├── event_v2.py              # Frozen schema + enums + feature names
│   │   ├── dataset_v2.py            # Typed manifest/shard interfaces
│   │   └── provenance.py            # data rights + source classification
│   ├── data/
│   │   ├── scan.py                  # index archives / file magic
│   │   ├── canonicalize.py          # normalizing events w/o changing semantics
│   │   ├── replay.py                # libriichi adapter
│   │   ├── manifest.py
│   │   ├── dedup.py
│   │   ├── split.py
│   │   ├── player_quality.py
│   │   ├── drift.py
│   │   ├── pov.py                   # ***redaction-only, isolated***
│   │   ├── decision.py              # legal action + snapshot
│   │   ├── encode.py                # packed visible tokens
│   │   ├── shard.py                 # per-game 4 POV, NO duplicated prefixes
│   │   ├── sampler.py               # quality/time strata
│   │   └── collate.py               # length bucket, correct attention masks
│   ├── models/
│   │   ├── common.py                # RMSNorm/RoPE/SwiGLU/SDPA
│   │   ├── local.py                 # MVP-S
│   │   ├── memory.py                # cross-kyoku summary/memory
│   │   ├── hierarchical.py          # RiichiFormer-H
│   │   ├── flat.py                  # full-history control
│   │   └── heads.py                 # policy/rank/belief
│   ├── train/
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   ├── checkpoint.py
│   │   ├── distill.py
│   │   ├── belief.py
│   │   └── rl.py                   # *only* after G5
│   ├── arena/
│   │   ├── adapter.py               # 46-action decode & engine bridge
│   │   ├── incremental.py           # KV cache
│   │   └── audit.py                 # parity
│   └── eval/
│       ├── offline.py
│       ├── duplicate.py
│       ├── uncertainty.py
│       └── strata.py
├── scripts/
│   ├── rf_preflight.py
│   ├── rf_data_audit.py
│   ├── rf_build_manifest.py
│   ├── rf_build_shards.py
│   ├── rf_train.py
│   ├── rf_bench.py
│   ├── rf_eval.py
│   ├── rf_quality.py
│   └── rf_distill.py
├── configs/riichiformer/
│   ├── data_v2.yaml
│   ├── mvp_s.yaml
│   ├── main_h.yaml
│   ├── flat_f.yaml
│   ├── eval.yaml
│   └── ablations.yaml
├── tests/rf_...               # synthetic / public fixture, no leaked private data
└── docs/riichiformer/
    ├── FEATURE_SCHEMA_V2.md
    ├── DATA_REPORT.md
    ├── SPLIT_REGISTRY.md
    ├── ABLATION_REGISTRY.md
    └── pr_reports/
```

输出目录**不入 git**：`artifacts/riichiformer/{manifest,shards,checkpoints,teacher,eval,bench}`；在 `.gitignore` 增加规则，不提交牌谱、权重、账号映射、盐和在线凭据。凡涉及上游 Rust 改动，建立 `patches/rf_v2/XXXX-....patch` 并固定重放顺序。

---

## 3. 数据科学：旧、近期、专家、教师、自博弈必须分开

### 3.1 原始数据真相与待验证字段

- 旧训练集：`configs/config.toml` 指向 2009–2022 年度，旧报告 `file_index.pth` 记录约 **1,968,147** 训练日志文件，2023 约 **168,777** 个 holdout 文件；这是**牌谱文件数**，不是局数，不直接代表可训练决策数。
- 第三方 `NikkeTryHard/tenhou-to-mjai` README 列有 2009–2026 年、四人凤凰桌东南半庄数据；其公开 README 标注的 2026 年文件约 13,639 场、仅 1 月。**本地数据可能更新，现场扫描和 hash 是权威，不得用文档代替本地事实。**
- 2025/2026 年度 `.mjson` 可能不使用 gzip，旧 `RUNBOOK.md` 已注明；读文件必须检测 magic bytes，规范化输出之后再交给 loader。不得只根据扩展名猜压缩方式。
- MJAI `StartGame` 可能只有 `names` 和 `seed`；**原始 mjai 本身未必附带账号在对局时的段位、R、付费资格或是否 AI**。元数据缺失须存 `null/unknown`，不允许从最后得分推算强度标签。

### 3.2 来源矩阵

| dataset_id | 来源 | 任务 | 前置条件 | 不允许做的事 |
|---|---|---|---|---|
| `tenhou_old` | 2009–2022 凤凰桌 | Baseline / BC 主要分布 | 核实既有使用许可 | 直接视作“都是真人且完美” |
| `tenhou_recent` | 2023–2026 已核实覆盖 | 年代/分布漂移、近期训练 | 许可、规则、元数据、去重 | “2026 = AI 更强”无证据断言 |
| `teacher_local` | 合法可使用的本地强模型 | 教师 Q / action preference | 权重来源与许可、冻结 SHA | 用 Q-softmax 宣称“真实动作概率” |
| `selfplay_own` | 自建环境合规自博弈 | RL/学习新策略 | 同步 version/logprob/wall seed | 丢失行为策略概率或重标环境结果 |
| `majsoul_optional` | 雀魂 | 独立跨平台 OOD 或单轴消融 | 许可证、规则差异、完整重放 | 未审计混合，尤其有信息不全的转换日志 |

**环境事实**：官方天凤公开牌谱使用条款可能限制外部用途和再分发；第三方转换器 Apache/MIT 许可证不等于牌谱获权。2026 年天凤账号封禁力度是否增加，不能从传闻直接构造“确定 AI”标签。平台若限制 AI 参与特定桌，不可用本项目指导违反规则的真人对战自动化。

### 3.3 原始 manifest 行契约

推荐 Parquet (Arrow schema 单独锁定)；一个 `game_id` 一行：

```json
{
  "game_id": "sha256:...",
  "source_id": "tenhou_old",
  "source_ref": "local:dataset/year/shard (no account info)",
  "year_month": "2021-05",
  "timestamp_utc": null,
  "source_rights": "reviewed|allowed|unknown|restricted",
  "ruleset_id": "tenhou_4p_hanchan_aka_v1",
  "original_sha256": "...",
  "canonical_sha256": "...",
  "parser_version": "mjai-normalizer-v2",
  "game_valid": true,
  "invalid_reason": null,
  "kyoku_count": 9,
  "decision_count": 600,
  "player_hmac": ["...", "...", "...", "..."],
  "player_dan_at_time": [null, null, null, null],
  "player_rating_at_time": [null, null, null, null],
  "player_provenance": ["unknown", "unknown", "unknown", "unknown"],
  "final_scores": [25000, 25000, 25000, 25000],
  "split": "train",
  "cohort": "old"
}
```

注意：上面 JSON 是**字段示例**，不是现实比赛。`decision_count` 必须来自合法重放；`game_id` 从规范化原始事件 ID/sha 算出而非随机文件名；相同对局四个座位必须共享一个 `game_id` 和 split。生成 HMAC 盐只保存在本地 secret store 或环境变量 `RF_PLAYER_HMAC_KEY`；无盐则拒绝生成长效 player ID，不可默默退化成无盐 SHA。训练样本不包含玩家名字或 HMAC 作为神经网络输入。

### 3.4 数据时间切分：两个互不相容的协议

**Protocol L（复现历史）**：Train=2009–2022；Validation=2023；2024–2026 为 OOD 诊断，但其中一部分已经被旧仓库 EX1 等研究使用过，**不称为 pristine confirmation**。

**Protocol R（近期训练）**：Train=2009–2024；Val=2025 H1；Model Select=2025 H2；Final Confirm=2026 **通过来源审计且先前从未被新项目调参消耗的样本**。如果 2026 数据只覆盖 1 月或样本太少，不能为了公布胜出而把 confirm 并进训练；只做内部探索，等待有合法新来源或使用预注册、隔离的新评测墙种子。

**时间外 vs 从未用过（严格区别）**：2026 牌谱可以用于评价在 2025 之前训练的模型的时间外泛化；如果模型或数据方案通过窥看 2026 的对局损失挑选过，它已是 select，不再是未见确认集。旧 EX1 使用过 2024–2026，不可声称这些数据对全项目历史是绝对全新。

**缺失信息处理**：没有可靠 2025/2026 日粒度日期就退回到可验证的年度切分；没有准确时间戳，不得在某年内部伪造 H1/H2 分组。可改成事先哈希固定的随机验证/选择集，但标明**非时间外**。

### 3.5 玩家质量评分与“疑似 AI 混入”

1. 建立 `player × time` 序列，只使用训练分区的**历史**对局估计实力；对每个被评分目标赛局，只允许使用此前日期的其他游戏，不使用该场结果，也不使用之后测试年份数据。
2. 可获得对局时段位/R 则作为**弱先验**；无法获得时别虚构。推荐初版经验贝叶斯 shrink：
   `q_i=(n_i * mean_reward_i + k * prior_mean)/(n_i+k)`，`k` 初始 100 场（候选超参，必须独立审计）。`mean_reward_i` 应使用 placement 和对手分布尽量做分层，不能把高段位直接等同好动作。
3. 若 `n_i < 50`，标记 `insufficient_history`；初版不做“顶尖 25%”认证，允许返回全量池。后续做玩家质量和行为/奖励关联的可靠性曲线。
4. 强模型判错指标：`regret=Q(best_legal)-Q(chosen_legal)`，但这是**裁判模型的估计**，不是因果实战收益。按巡目、局况、名次、分差、动作类型和日期分层，比较多裁判共识；仅作为候选筛选特征。
5. **不按单局赢牌、点数领先、偶然宝牌多来过滤**；不直接删除 AI 风格异常玩家。只有来自官方/合法可核验来源的明确证据，才将 `player_provenance=confirmed_ai/confirmed_human`，否则为 `unknown`。
6. 记录每个月的动作分布：`riichi_rate, pon/chi/call_rate, fold_vs_reach, deal_in_rate, action_agreement, avg_rank, seat_distribution, kyoku_phase, game_length`。分布漂移分析同时控制规则、当时局况、对手强度和来源。

### 3.6 训练数据配方：只允许逐轴实验

数据实验固定小模型 `MVP-S`、随机种子、训练决策和优化 token 预算，不同时换骨干。

| arm | 描述 | 重要对照 |
|---|---|---|
| D0 `ALL` | 所有合规合法凤凰桌 seat-game；基础均匀抽样 | 旧年代数据主基线 |
| D1 `TOP25_SEATS` | 只重加权/筛高可信玩家对应的座位决策；另三家仍作为环境事件 | 同独立 seat-game 数、相同场景覆盖 |
| D2 `MIX` | 60% ALL、25% 高可信、15% 近期/稀有场景（预注册候选） | 采样重复率、有效样本数必须报告 |
| D3 `RECENT` | 同质量分层下提高近期比例 | 对比 D0：控制样本量、年代、规则变动 |
| D4 `DISTILL` | 冻结人类 BC 数据，增加合法教师软标签 | 检验是否强于仅多训练相同步数 |
| D5 `MAJSOUL` | 严格审计后**可选**跨平台来源 | 独立实验不默认合并 |

**对照不是只控制 steps 就够**：同时报告独立 games、独立 seat-games、独立 players、训练 labeled decisions、non-padding tokens、gradient updates、墙钟时间；主要对照按 `effective labeled decisions + updates` 配平，另做 equal-token/compute 分析。小池重复采样报告 effective sample size，防止所谓“高质量”仅靠重复过拟合。

公开 TanyaoDojo 在特定 CNN Q-line 报告 Top25 更好，但随后 online 增益未保留；新增雀魂数据甚至可能伤害防守。这些是**值得预注册的实验线索**，不能当项目事实或放宽统计门槛。

---
## 4. V2 输入契约：必须先冻结再写数据

### 4.1 核心原则

**一条训练样本 = 一场 game 的一个 POV**（最多四条训练样本），其中的所有局按时间顺序排列。**一个 decision token = 此 POV 在这一时刻确实拥有合法行动选择的机会**。不把全部历史前缀复制成几十万个逐决策小样本。

训练输入分两层：

- **observable**：只能包含此时 `POV` 亲见的事件 + 自己手牌/公开局况 + 合法动作。
- **target**：人类已执行动作、终局奖励、用于辅助学习的对手私有标签。目标必须是另一个独立结构，**不可将 target 的字段混入 event/state tensors**。

对手摸牌的牌种：**公开发生一次摸牌，不公开摸入哪张**。自家摸牌：公开给自己。不能通过 `hand_count`、`red_count` 或其余变换间接泄露对手隐蔽信息。

### 4.2 `ObservableEventV2`（推荐 Rust enum / Python TypedDict 一一映射）

```python
class ObservableEventV2(TypedDict):
    kind: str                 # fixed enum, version=2
    actor_rel: int            # 0=self, 1=next, 2=opposite, 3=previous, 4=none
    target_rel: int           # same 0..4
    tile: int                 # 0..33, 34=NONE_OR_UNKNOWN
    aka: bool                # opponent draw aka=false, no hidden flags
    consumed: list[int]      # <=4 tiles, 34 PAD; ONLY publicly exposed
    consumed_aka: list[bool] # aligned with consumed
    tsumogiri: int           # -1=NA, 0=false, 1=true
    seq_idx: int             # monotonically increasing game event index
    kyoku_idx: int           # 0..K-1, never silently reset history
    # public state of last observed event; optional fields present only when public
```

约定 `seat_rel=(absolute_seat - pov + 4)%4`，`actor_rel=4` 表示无 actor。`tile` sentinel 34 不可被当作真实牌种。赤 5 的 tile-kind 是对应 5，`aka` 单独标志。`consumed` **必须保留完整公共副露构成**，不能只取 representative tile。对手暗杠：按官方可见范围构造公开牌面，不能直接复制内部 `consumed` 的私有构成（是否能知 4 张种类需按事件规范逐项测试）。

类型最小集合：`START_GAME, START_KYOKU, OWN_HAIPAI, TSUMO_PUBLIC, TSUMO_SELF, DAHAI, CHI, PON, DAIMINKAN, ANKAN, KAKAN, DORA, REACH, REACH_ACCEPTED, HORA, RYUKYOKU, END_KYOKU, END_GAME, DECISION, KYOKU_SUMMARY`；`PAD=0`。所有 enums/tile indices/action indices 冻结并写 `schema.json` + SHA-256。**不允许训练时默默改 enum 顺序或 normalizer**。

### 4.3 决策快照 `DecisionSnapshotV2`

```python
@dataclass(frozen=True)
class DecisionSnapshotV2:
    game_id: str
    pov: int
    decision_id: int
    before_event_idx: int  # 决策之前的事件前缀截止位置
    kyoku_idx: int
    own_hand_counts: tuple[int, ...]       # length 34
    own_aka_counts: tuple[int, ...]        # length 3
    visible_tiles_counts: tuple[int, ...]  # length 34, optional derived public
    scores_relative: tuple[int, ...]       # length 4; centered / normalized model-side
    bakaze: int                            # public
    kyoku_number: int                      # 1..4
    honba: int
    kyotaku: int
    dealer_relative: int
    turn_count: int
    riichi_states: tuple[int, ...]         # length 4
    own_furiten: bool                      # if known from PlayerState
    dora_markers: tuple[int, ...]          # public dora at decision
    legal_mask: tuple[bool, ...]           # length=46; existing libriichi truth
    executed_action: int | None            # -1/None if censored
    supervision_valid: bool
    censor_reason: str | None
```

固定字段含义、值域、标准化方式；把 `executed_action` 从输入特征 projection 中**明确剔除**。分数中心化/scale（如 `score / 10000`）必须写进 schema 并保存 normalizer version。每个决策应至少有一个合法动作；强制唯一动作可收录为环境 trajectory，但默认从 BC loss 中排除（可做辅助评估）。

**局开始 13 张 + 庄家首次第 14 张**：`StartKyoku.tehais` 在 upstream 是每人 13 张；庄家通常靠后续 `Tsumo` 获得第 14 张。对异常的其他转换器，不能猜哪张是庄家摸牌；先证实来源能否无歧义转成统一时序，不能就拒绝该局的监督。

**事件/状态顺序**：

```text
start_kyoku observable  →  POV own_haipai  →  public actions and own tsumo
                                                      ↓
                                     PlayerState update + cans + legal mask
                                                      ↓
                                  [DECISION] token (state snapshot attached)
                                                      ↓
                                     logged action from future event ONLY as LABEL
                                                      ↓
                              commit observed event to chronological input
```

`libriichi` 的历史 GameplayLoader 用未来 next event 来产生**标签**，这是允许的；禁止未来事件影响**当前的模型输入**。`can_act` 由状态机给出；合法 `PASS` 的标签只在能确证这个玩家的响应窗口/优先级条件时加入。对被更高优先级的和/碰/杠抢断的窗口，记为 `censored` 且无 BC loss，不伪造 PASS。

### 4.4 序列布局和模型 forward 接口

每个半庄由多个局片段组成：

```text
[START_GAME]
[START_KYOKU_0][OWN_HAIPAI][EVT_0][DECISION_A][EVT_1]...[END_KYOKU]
    -> [KYOKU_SUMMARY global][KYOKU_SUMMARY opp1]...[opp3]
[START_KYOKU_1][OWN_HAIPAI] ... [DECISION_B] ...
...
[END_GAME]
```

- `Local` 只看**当前局**中的 causal token，并通过 query 获取前面**已完成局**的 memory output；局末生成 4 枚 summary tokens。
- `Memory` 按局号顺序消费历史摘要；第 `k` 局决策最多可读取 `0..k-1` 的摘要。不能把第 k 局的完成摘要传回第 k 局决策。
- `Flat` 对整场直接做单条 causal Transformer，是对照；超长不能静默截断。可以按长度分桶或 microbatch 1。
- 离线计算每个 POV 的每个局片段一次 forward，并在该局所有 `DECISION` 位置同时计算损失；不重复编码每一决策的前缀。

**统一 `ModelOutput`**：

```python
{
 "decision_logits": FloatTensor[N_decisions, 46],
 "rank_logits": FloatTensor[N_decisions, 4],  # can be disabled in pure-BC run
 "tenpai_logits": FloatTensor[N_decisions, 3],  # optional, disabled initially
 "decision_locations": LongTensor[N_decisions, 3],  # batch, kyoku, local_pos
 "memory_state": ... ,  # optional, versioned
}
```

`legal_mask` **只在策略 head 上应用**，用 `-inf`（BF16 数值和空集要测试）；所有非法行动在 softmax 后概率应为 0。规则 `chi/pon/reach/kan` 的 index 必须来自同一 `libriichi` action mapping，不要按麻将直觉自行重排。

### 4.5 信息可见性表（编码器的逐事件代码分支必须与之对应）

| 原始 MJAI 事件 | 当前 POV 可输出 | 不准输入 | 校验 |
|---|---|---|---|
| `start_game` | 游戏开始、座位映射；必要时规则元数据 | `seed`、牌山、真实名字 | 修改 seed/姓名模型输入不变 |
| `start_kyoku` | 场风、局数、本场、供托、亲家、点数、表宝牌、**自己的** 13 张 | 另外三家的 `tehais` | 修改三家手牌前缀输入相同 |
| `tsumo self` | actor、自家摸到的牌与赤标志 | 牌山位置与剩余墙内容 | 与实时 arena 一致 |
| `tsumo opponent` | actor、摸牌发生一次；tile=34、aka=false | `pai`、赤标志、派生隐藏数量 | 改 opponent pai 输入逐字节不变 |
| `dahai` | actor、弃牌、赤、是否摸切（可见时） | 牌谱后续的隐藏摸牌信息 | 对照官方展示含义 |
| `chi/pon/kan` | 公开的副露构成、来源、赤、开闭状态 | 规则下不公开的牌身份或未来信息 | 逐条正常/异常转换器测 |
| `dora` | **已经翻出的**宝牌指示牌 | 尚未出现的杠宝牌/里宝牌 | 在事件出现前不可读 |
| `reach` | 立直宣言、已经确认的供托变动 | 对手听牌真值作为输入 | 公开/私有严格分离 |
| `hora/ryukyoku` | 结算后公开的得点变动、确认的翻牌信息 | 回填到此前决策 | 改结算结果不得影响早期预测 |
| `end_kyoku` | 本局完成、才可产生记忆摘要 | 不可回填当前局历史决策 | memory fence 测试 |
| `end_game` | 对局结束 | 下个对局历史 | 重置 memory/cache |

### 4.6 必须写的安全测试（测试名建议固定，任何失败阻塞）

1. `test_opponent_tsumo_redacted`: 直接对比 `actor!=pov` 的两组原始事件：`pai=1m` vs `9p`，结果字节相同，赤标志相同常量；自摸必须不同。
2. `test_start_kyoku_hands_redacted`: 保持自己的 13 张不变，替换另外三家的 13 张，当前可见 token 一样；禁止保留 raw fields via `.dict()`。
3. `test_no_future_information`: 处理事件 `0..t` 的输出不依赖 `t+1..end` 和终局标签；Transformer CPU eval outputs `allclose(atol=1e-6,rtol=1e-6)`。
4. `test_cross_kyoku_fence`: 修改第 k 局之后的历史，所有 `<=k` 局 decision logits 不变；修改前一局的**公开**历史允许后续局输出改变。
5. `test_legal_action_mapping`: 对随机重放的真实决策，46 bit mask 与原生 GameplayLoader / PlayerState 一致；已执行标签必须合法且与原生 action id 一致；不合格日志计数而非 silent pass。
6. `test_censored_reaction`: 抢和/高优先级反应导致的未发生动作不应被标为 PASS；对无歧义 pass 允许正确标注。
7. `test_dataset_arena_identical`: 同一事件流由 offline 路径和在线 adapter 输入，**可见 token tensors + 决策状态 + mask 逐字段一致**；至少每类事件各一 fixture。
8. `test_schema_stability`: snapshot schema 文件 hash 相同；模型 checkpoint 若 schema/hash 不符必须拒绝加载。
9. `test_no_padding_attention`: 添加不同 padding 长度不应改变有效 token 的预测；注意 PyTorch SDPA 的 bool mask 语义是 True=允许注意，和部分其他 API 不同。
10. `test_reset`: start_game 清空前一半庄 cache，反复复用相同对象结果一致。

**测试技巧**：简单修改隐藏牌可能产生物理不合法的牌局；`redaction` 单元测试直接在事件 DTO 层做，不能要求无条件通过完整重放；`replay parity` 测试使用各自合法的固定 fixture，在**前缀尚未产生公共差异**的位置比较输出。两种测试互补，不可用一个替代另一个。

---

## 5. 模型定义与损失

### 5.1 MVP-S（先验证，而非先堆 22M）

- Decoder-only causal Local Transformer：6 blocks、`d_model=256`、`n_heads=4`、`head_dim=64`、`ffn_dim=768`、`SwiGLU`、pre-`RMSNorm`、RoPE、dropout 0 起步。具体参数量由脚本打印，不把“5–8M”当已经实现的事实。
- 每条训练序列是**一局的一个 POV**，一局里全部决策位置同时求 loss；不跨局。
- 输入：离散特征 embeddings（事件 type/actor/tile/flag 等按 field 分别编码，不能把多字段整数当一个无意义 token id），可观察状态中的手牌/分数等通过独立 MLP 注入 `DECISION` token。
- Head：46 actions + optional rank 4（首轮先关闭 rank loss）。
- 初始目标：`L = CrossEntropy(masked_46_logits, expert_action)`；每个 seat-game 的 loss 以其有效决策数归一，避免长半庄主导梯度。默认排除单合法动作和 censored；计数都要报告。

### 5.2 Main-H（门 G3 过后）

- Local：8 blocks，`d=384, heads=6, ffn=1024`。
- Memory：4 blocks，`d=384, heads=6, ffn=1024`，每个**完成**的局产生固定 4 枚 summary（global + 3 opponents）。Memory 接收可变长度已完成局摘要；外部 gate 保证不读取当前局未来。
- Fusion：先用 `gate=sigmoid(W[h_local;h_mem])`，`h=LayerNorm(h_local + gate*W_m h_mem)` 的可复现实验作为简单融合；**有无历史时保持相同参数/计算对照**，无历史用定义明确的零记忆，而不是未初始化 token。
- Summary 生成：把 4 个 summary query 放在当前局 `END_KYOKU` **之后**，允许它们 attend 该局全程可观察局内 token；全部摘要进入**下一局** memory。MVP 阶段先 stop-grad 摘要以简化 GPU 图；跨局 BPTT 或 learned compression 梯度流的消融在后续独立进行。**注意：如果 summary 编码器本身不接受任何任务监督、完全 stop-grad 且未预训练，会导致无法学习有效压缩**；因此 Main-H 的首版应该允许本局的辅助摘要损失或给摘要模块独立训练信号，或者采用分段训练后在局级 memory 路径训练总结器。不要声称仅 stop-grad 就能学会对手风格。
- 目标：以 `BC` 为主；增加 rank CE 由单独实验开关控制（推荐探索 `lambda_rank in {0,0.1}`）；Belief head 另开关，不能悄然引入。

### 5.3 Flat-F（成本对照）

- 12 blocks、`d=384`、`heads=6`、`ffn=1024`，全半庄严格 causal；该规模是理论对照，**参数并不天然和 Main-H 完全匹配**，Agent 应输出精确参数、train FLOPs、peak VRAM、wallclock，再建立相同模型规模和等计算预算的对照。
- 长度桶按实际数据 p90/p99 和 1024/2048/4096 基准测试；单卡 OOM 的真实样本不能丢失后仍称为 `full history`。可以单独报告 `Flat-2048-truncated` 实验臂，但标记明确、统计截断率，**不允许与 full-memory 直接宣称公平**。

### 5.4 辅助任务与奖励（不同阶段单独控制）

```text
BC:      CE(student_logits masked to legal, played_action)
Rank:    CE(rank_logits, final_rank_index 0..3)    # target only; never input
Belief:  BCEWithLogits(opp_tenpai_logits[valid], opp_tenpai_truth)
Distill: T² * KL(softmax(Q_teacher/T), softmax(student/T)) on legal subset
RL:      PPO with actual logp_behavior and current logp (later stage only)
```

- `rank` 是最终座位顺位，不是每局中间名次；最终得点模型要有单独 head，避免与顺位奖励混淆。
- 不把 `teacher_Q/T` 当真实采样策略。蒸馏需要保存教师版本、Q 的合法掩码、温度及 Q 尺度；若教师 Q 只在少量动作可比，不用全量 KL，改做 top-k / pairwise preference 控制实验。
- 一次用一条主训练目标，新增 `lambda_aux` 必须有预注册和关闭对照。
- 初步 `rank_pts=(90,45,0,-135)` 用于复用现有 eval；模型是否优化这套值必须在 config 明确，不应和天凤官方段位分数混淆。

### 5.5 单卡显存与性能策略

1. 现场检查 `nvidia-smi`, PyTorch CUDA/BF16 支持和 GPU 空闲显存；记录驱动、Python/CUDA/Torch、CPU/RAM、WSL 与否。
2. 首先使用 PyTorch 内置 `torch.nn.functional.scaled_dot_product_attention`、BF16 autocast、AdamW。**不要求先编译 FlashAttention 或启用 torch.compile**。
3. 按有效 token 数分桶，参考微批：`MVP(512,8)`、`Main-H(local<=512,batch=4)`、`Flat(1024,4)/(2048,2)`，均为待测起点，不是容量保证。
4. OOM：保存完整 error 与 memory stats → 调低 microbatch → 启用 activation checkpoint → 确保 gradient accumulation 合理 → 仍不行则缩模型/使用层级记忆；**不能默默删除历史**。
5. 每个 checkpoint 存 `model,optimizer,scheduler,global_step,tokens_seen,decisions_seen,RNG_python/numpy/torch/cuda,config,gitsha,schema_hash,manifest_hash,split_registry_hash,action_map_hash`。加载 mismatch fail closed。
6. 每次报告 `trainable_params, max_memory_allocated,max_memory_reserved,step_ms,nonpad_token/s,labeled_decisions/s,p50/p95 decision latency`。只报告真实测量；若测试机不是 5070 Ti，需明确标注。

---
