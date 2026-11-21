# RiichiFormer V3.1 — 多 Agent 并行开发计划（可直接交给 Lead）

**状态**：实施计划，不代表任务已执行。**基于**：V3 `00–04`、四份 PR09 审计资料和审核分支现状。**研究目标**：安定九段最低研究门槛；天凤位级别长期目标。**资源**：单台 WSL/Linux 开发机，RTX 5070 Ti 16GB；GPU 独占训练，CPU/磁盘共享。**当前实际完成**：PR-00～06 已有代码/有限验证，PR-07 未完成，PR-08/09 未实现。

## 0. 结论：并行策略及绝不能做的事

- **建议组织：1 Lead + 4 Dev 子代理（分别负责数据/规则/模型/评测）+ 1 Reviewer 子代理**。前期 Dev 数量可以动态降至 2–3，不能为填满席位制造工作；Reviewer 为只读独立角色，不计入写代码的 4 人。建议 **同时最多 4 个会修改代码的 Agent**。
- **H-00 必须单线程、先完成**：真正的本地开发基线和备份确认前，不创建并行工作树、不做任何清理、重置、公开 `main` 合并或 force-push。审计分支与公开 `main` 无共同祖先，所有子代理必须从 H-00 认可的**同一个开发基线 SHA** 派生工作树。
- 可以并行写不同模块、单元测试和文档；不能并行操作共享 GPU、同一 shard 输出目录、同一 eval seed pool、同一个核心接口文件或需要串行归因的正式 A/B 实验。
- **开发完成 ≠ 可合并 ≠ 可训练 ≠ 实验有效**。每一步有不同的门。PR-13 可用合成数据实现/跑 forward/backward；没有获许可的数据和去重/POV 闭环，PR-14 不得用相应牌谱训练。
- 当前 repo 的 `PR-08/09` 未实现，不得由多 Agent 假装一方已完成再让下游消耗未冻结的 API。

## 1. 固定 Agent 角色和职责

| 身份 | 主要所有权 | 首要任务 | 不允许触碰 |
|---|---|---|---|
| Lead（编排/集成） | `docs/riichiformer/v3/PARALLEL_STATUS.md`、`INTEGRATION_LOG.md`、集成分支、seed/resource registry | H-00 主控；派工；批准接口变更；逐 PR 集成；调度 GPU 和评测池 | 不直接覆盖子代理私有工作树；不擅自解锁 rights |
| Dev-DATA | `riichiformer/data/{canonicalize,dedup,split,drift,...}`，相关测试与脚本 | H-02；PR-07R；后续 PR-12、PR-18～20 | H-01 的 `replay.py`（首轮）；PR-08 schema 冻结文件；其他 Agent 输出目录 |
| Dev-ENGINE | `riichiformer/data/replay.py`、rules parity 测试、`decision.py`、`tile_counter.py` | H-01；H-04（可由第二位只读研究 Agent 辅助）；后续 PR-10 | `data/pov.py` 由 POV Owner 独占；训练器、模型目录 |
| Dev-MODEL | `riichiformer/models/**`，模型配置/测试 | PR-13 合成数据版；后续 PR-31、PR-25A、PR-27 的模型头 | 尚未冻结的 `schemas/**`；`data/pov.py`；正式训练中的共享 checkpoint |
| Dev-EVAL | `riichiformer/eval/**`、`runtime/**`（PR-16 接口所有权需明确）、评测协议及测试 | H-04 规则文档/测试可协助；PR-34 框架；后续 PR-16/15 | 无主 Agent 授权不得分配 confirm 池，不得在官方凤凰桌部署 AI |
| Reviewer（只读） | 不拥有实现文件 | 独立验收提交、测试日志、接口对齐、负向测试、资源日志 | 不直接修改开发者分支；不将未执行命令记为 PASS |

**动态所有权重分配**：PR-08/09 属于串行且强耦合的安全接口，Lead 应指定一名 **Schema/POV Owner**（可暂时从 MODEL 或 EVAL 借调）。其余 Agent 不得同时写 `schemas/**`/`data/pov.py`。在 Wave 2 中，ENGINE 转做 PR-10 前须等待 PR-09 验收。

## 2. 并行 Waves（每一轮的硬依赖、实际派工和合并顺序）

### Wave 0 — 唯一先行项：H-00

**唯一写入者**：Lead/Dev-ENGINE 二选一；其余只读。

完成工作区保全、开发分支选定、备份与 hash 验证、`vendor/mortal` patched/untracked 状态记录、现有用户未提交文件保全；建立集成分支，锁定共同 `BASE_SHA`、固定 `RF_SHARED_LOCK_ROOT`。

**Gate W0**：`GIT_BASELINE.md` + 可恢复备份 + 共同基线 SHA + 根目录代码不被 `.gitignore` 漏掉 + 并行 worktree 可创建。Lead 审阅/批准后才能 fan-out。

### Wave 1 — W0 之后同时派 4 条独立流水线

| Lane | 首个工单 | 代码/产物所有权 | 允许并行 | 特别限制 |
|---|---|---|---|---|
| A：Score | **H-01** | `data/replay.py`, `rf_test_replay.py`, `SCORE_AUDIT.md` | 与 B/C/D | 只改结算/重放；覆盖多荣、留存供托、0-seat；依赖质量分析需后续重算 |
| B：Index | **H-02** | `data/{canonicalize,dedup,split}.py`, `rf_build_canonical.py`, `split` tests | 与 A/C/D | 先 10k/100k；全量 2.51M canonical/hash 做分片与断点，磁盘独占调度；**H-01 不是 canonical 的硬依赖** |
| C：Rights | **H-03** | `source_rights.csv`, `SOURCE_RIGHTS_REVIEW.md`，独立验证脚本 | 与 A/B/D | 只能提交证据与建议；**更改 ALLOWED 需要有权限的人审阅**，不能自行放行 |
| D：Rules | **H-04** | `TENHOU_RULE_PARITY.md`, `rf_test_ruleset_parity.py` | 与 A/B/C | 官方规则差异/fixture，禁止把 patched Mortal 当 Tenhou 绝对等效 |

**可插入的轻任务**：PR-07R 漂移代码与轻量测试（第五任务但不增加写代码 Agent 数量，等一条 Lane 空闲再接）；PR-08 Schema **draft 设计**可与 Wave 1 并行，但 schema 的最终 freeze 必须让 H-01/H-04 的关键语义反馈进入 review，并在任何 PR-09 开始前集成。

**Gate W1**：四条 PR 分别通过 Reviewer；本地开发集成分支在合并后重跑全套测试；H-02 全量任务可以长时间延续但至少已有流式/可恢复机制与跨 split 去重正确性测试。H-03 没有获批不阻塞合成数据开发，但阻塞以受限语料启动训练。

### Wave 2 — 关键串行安全主链 + 允许的侧向并行

```text
[固定 PR-08 Schema V3]
          ↓
[PR-09 POV allowlist]
          ↓
[PR-10 Decision Snapshot / legal mask / censor / aka ledger]
          ↓
[PR-11 offline/online byte-equal + causal parity]
          ↓
[PR-12真实 shard/dataloader]
```

**主链必须依序验收**，但以下工作可同时进行：

- Schema Owner 实现 PR-08→PR-09（同一 owner 更安全），不得并行修改 `data/pov.py`。
- ENGINE 在 PR-09 等待期间可补 H-01 难例、H-04 fixtures、PR-10 的**独立 toy fixtures/设计**；PR-09 通过后才接入 PR-10 真实事件。
- MODEL 在 PR-08 字段与维度冻结后，可以并行开发 PR-13 的 **synthetic forward/backward/causal attention**，不碰真实数据和共享 encoder 的实现文件。真集成必须等 PR-11 接口通过。
- EVAL 独立开发 PR-34 的 `opponent_registry`、PR-16 的 **mock arena contract** 和种子池账本；正式 arena 实现必须等 PR-11+PR-13。
- DATA 继续 H-02 / PR-07R；PR-12 真正批量生成必须等 H-02 完整 hash/split，以及 PR-11 parity 和 source gate。

**Gate W2**：Schema/Action Hash 冻结；POV 隐藏扰动测试；真实日志的 action/Pass/censor/二次杠语义正确；离线/在线输入 byte-equal；真实 shards 溯源完整。任何 P0 阻塞完整链路。

### Wave 3 — 第一版训练和 arena 允许并行实施，但不能跳过集成

| Dev | 可开始的工单 | 硬依赖 | 完成标志 |
|---|---|---|---|
| MODEL | PR-13 6×256 Local + MLP State + Linear 46 logits | Frozen PR-08；真实数据 forward 接入需 PR-11 | 合成数据训练、因果测试、模型参数/吞吐 |
| DATA | PR-12 shard/dataloader | H-02 + PR-11；实际训练来源 H-03 allowed | 可恢复、无 split 污染、game/POV 整段格式 |
| ENGINE/Train | PR-14 BC Trainer | PR-12 + PR-13 + H-03 的相应训练许可 | deterministic resume、完整 checkpoint hash；GPU smoke 需统一排队 |
| EVAL | PR-16 arena；PR-15 offline 的独立诊断模块 | PR-11 + PR-13 + H-04；PR-15 完成需 PR-14 | 4-seat adapter、合法动作零错误，诊断分桶可信 |

**Gate W3 / PR-17**：Lead 必须先在一个**集成分支**拿到真实训练数据来源许可、数据/模型/arena 全链路通过，然后冻结首个可复现 MVP-S。Reviewer 需要独立审计 P0/版本 hash/规则差异；PR-17 的实际 GPU 运行和 arena 对战由 Lead 排他调度。初版不要求超过 Mortal，但不得用 NLL 当作胜率证据。

### Wave 4 — PR-17 后可并行的研究支线（**代码并行，实验串行**）

| 研究 lane | 子任务 | 可同时开发 | GPU/决策顺序 |
|---|---|---|---|
| A：数据 | PR-18/19/20（数据质量、近期数据、决策重加权） | 与 B/C/D | 默认先建立 D0 固定基线，再单轴 D1/D2/D3 实验；不得同时改模型结构 |
| B：Policy | PR-31 Candidate Head、PR-25A State/Capacity | 与 A/C/D | Candidate vs Linear 必须同 backbone/数据/预算；容量实验单独一组 |
| C：Risk | PR-32 Oracle Target → PR-27 Belief → PR-33 Risk/Rank A/B | 与 A/B/D；该 lane 内有依赖顺序 | 风险标签不得进入 forward input；校准先于 policy 融合 |
| D：Infra/Eval | PR-28 自博弈 rollout 基础、PR-34 强对手池/标定 | 与 A/B/C | 规则与 rollout semantics 先测试；RL update PR-29 必须等 PR-28、稳定 champion |

`PR-21` 弱 Mortal 蒸馏仅在来源许可、独立实验空间充裕时执行。`PR-22/23/24/26` Memory 先做无 GPU 的 history-value 诊断，观察到条件历史信号且基础策略稳定后才实施 learned summary/Flat-F；两者不是 G3 或九段目标的必需依赖。

**Gate W4**：每条研究 lane 单独提交预注册、源代码、可靠验证、相同预算的 select 结果与消融/负控。仅 Lead 根据**独立的预先指定 select** 决定是否晋升 champion。多组实验探索仅为选择，确认集不能用作调参。

### Wave 5 — 强度确认与长期方向

- `PR-34` 的强对手标定必须独立于模型候选选择；校准不可靠，不得称“凤凰桌等效九段”。
- `PR-30/35`：单个 champion、版本 hash、规则与对手池锁定后，Lead **独占**分配新 confirm seed-key/区间；Reviewer 在首次消费前核验，完成后永久登记 consumed。
- `PR-29` RL 与 `PR-36` 长期搜索/Memory，可以在**另一开发分支**继续研究，但不能反向修改正在确认的候选模型；确认期间模型和评测配置完全冻结。
- 当前硬件无法保证达到九段或天凤位，所有里程碑按真实评测证据推进。

## 3. 两种依赖：hard gate vs interface preview

- **Hard gate**：依赖 PR 被 Lead 集成、Reviewer PASS、所需 artifact hash 固定；后继任务消费真实生产输入或启动实验必须满足。
- **Interface preview**：可读冻结 schema/公开接口，提前开发 mock/test/skeleton，**但不能把它当作功能实现/训练资格**。例：PR-13 合成模型可在 PR-12 前开发；PR-16 mock arena 可在 PR-11 前编写，但完整 arena 要等 PR-11 和 PR-13。
- 依赖项 **BLOCKED** 时：子代理仍可在独立分支开发不消费该依赖的测试/文档，不能绕过权限/数据门冒充成功。

### 实用 PR DAG（括号内为可并行起点）

```text
H00 [唯一串行]
 ├─ H01(score) ──────────┬─ PR10(snapshot) ─ PR11(parity) ─ PR12(shard)
 ├─ H02(canonical/dedup)┼───────────────────────────────────────┘
 ├─ H03(rights) ────────┼─────────────────────┐                ├─ PR14(real train)
 ├─ H04(rules) ──────────┼── PR16(arena) ──────┼────────────────┤
 ├─ PR07R(drift) ────────┤                     │                │
 └─ PR08(schema) ─ PR09(POV) ─ PR10 ─ PR11 ───┼────────────────┤
           └─ [PR13 synthetic model] ─────────┼── PR13 tested ─┘
                               PR12 + PR13 + H03 → PR14
                               PR11 + PR13 + H04 → PR16
                               PR14 + PR15 + PR16 → PR17 (gate)
                                                       ↓
              ┌─ PR18/19/20 data (independent A/B)
              ├─ PR31 candidate → head A/B; PR25A capacity A/B
              ├─ PR32 oracle → PR27 risk → PR33 policy A/B
              ├─ PR34 anchor / PR28 rollout → PR29 RL
              ├─ PR21 teacher (optional, rights gate)
              └─ PR22/23/24/26 memory (optional, conditional)
                                                       ↓
                   champion selected + PR34 anchor evidence
                                                       ↓
                           PR30/35 one-shot confirm → PR36
```

**文字依赖表是权威**。ASCII 图意在展示 fan-out；一项出现两次只是强调汇合点，不表示重复执行。

## 4. 任务文件独占、并行冲突矩阵

**严格单 writer 的文件/接口**：

| 资源 | Owner | 非 Owner 行为 | 合并前检查 |
|---|---|---|---|
| `riichiformer/schemas/**`, `schema_v3.json` | Schema Owner | 可以提 RFC/测试，不直接编辑 | schema、action_map、normalizer hashes 变更明确版本升级 |
| `riichiformer/data/pov.py` | PR-09 Owner | 只提供负例 fixture | 对手 tsumo/起手隐藏、未来截断、赤牌/副露可见性回归 |
| `riichiformer/data/replay.py` | H-01 Owner | 其他任务不并行修改 | 供托/分数、下一局分数、末局供托/多荣验证 |
| `riichiformer/data/decision.py`,`tile_counter.py` | PR-10 Owner | 其他任务提供独立测试 | snapshot/label 隔离、censor/kan/赤牌账本 |
| `riichiformer/data/encode.py`, `runtime/encoder.py` | PR-11 Owner | 模型/arena 用 frozen contract mock | train/arena bytes 同一映射 |
| `riichiformer/models/local_transformer.py`, `state_encoder.py` | PR-13 Owner | Candidate/Risk 使用稳定公开接口 | baseline model outputs 不被悄然改写 |
| `scripts/rf_train.py`, `train/checkpoint.py` | PR-14 Owner | 其他 Agent 仅新增插件或实验配置 | resume + source/schema hash |
| `scripts/rf_eval.py`, `runtime/agent.py` | PR-16 Owner | PR-34 通过 API 使用 | arena action legal、seed rotation |
| `EVAL_POOL_REGISTRY.md`, final confirm PREREG | Lead | 只允许提出申请 | 强度确认集合只由 Lead 分配；一次性消费 |

**同一个测试文件不可在两个活跃 PR 同时修改**；测试通过也不能掩盖未提交内容。公共 `__init__.py`、依赖锁文件、`pyproject` 等交由 Lead 在集成点集中更新，杜绝相互覆盖。

## 5. Git worktree 工作流（先 H-00 认可本地分支）

H-00 确认 `BASE_SHA`、`DEV_BRANCH` 之后，使用同一 Git object store 的独立 worktree。下面的命令是**示例**，不是当前可直接安全执行的自动迁移命令：

```bash
# Lead 先完成 H-00，明确 DEV_BRANCH 与 BASE_SHA，不使用 public main 作默认底座。
BASE_SHA=<H00_APPROVED_DEV_SHA>
mkdir -p ../rf-worktrees
# 每个任务创建唯一 branch + worktree；不能在多个 agent 共享同一工作区。
git worktree add -b rf/h01 ../rf-worktrees/h01 "$BASE_SHA"
git worktree add -b rf/h02 ../rf-worktrees/h02 "$BASE_SHA"
git worktree add -b rf/h03 ../rf-worktrees/h03 "$BASE_SHA"
git worktree add -b rf/h04 ../rf-worktrees/h04 "$BASE_SHA"
```

Lead 要负责开发环境可用性（如共享只读数据根、构建依赖、`.venv` 配置）；创建 worktree **不会自动包含未经提交的改动、ignored artifacts 或 dirty vendor 补丁**。对这些资产必须按 H-00 的安全方案单独处理，并校验 hash，不能直接复制用户未提交文件到所有子代理。

每个子代理：在自己的 worktree 上写代码，生成**唯一任务名**的报告和小型 fixture；输出 `BRANCH`, `COMMIT`, `BASE_SHA`, `DIFF_FILES`, `TEST_COMMANDS`, `TEST_RESULT`, `ARTIFACT_HASH`, `BLOCKERS`。Lead 将其集成到专用开发集成分支，顺序执行 `git cherry-pick <task_sha>`（或在有共同祖先且满足开发库策略的情况下使用常规合并），严禁直接与没有共同祖先的公开 main merge。提交代码后重跑综合测试，才算一个任务完成。

**公开发布**沿用 H-00 审批过的 filtered-history publish 流程，在隔离发布工作树执行，并审计敏感文件清单。

## 6. 物理资源锁：代码可以并行，训练和统计不能争抢

| 资源 | 调度策略 | 理由 |
|---|---|---|
| RTX 5070 Ti 16GB | **一个正式 GPU worker**；实验 A/B 按预注册队列依次跑。模型开发者可用 CPU / 小合成 GPU 测试预约独占窗口 | 多 trainer 并行会共享 VRAM、干扰吞吐，甚至不公平比较 |
| 2.51M 游戏源数据 / 磁盘 | 全量 canonical/dedup 和全量 shard/replay 等重 IO 作业错峰；输入只读，输出隔离并断点 | 避免磁盘瓶颈、半写入、I/O 拥塞 |
| 原始/更新 split registry | 只有 DATA Owner 生成、Lead 审核冻结新版 hash，旧 registry 不覆盖 | 防止同一 game 落入 train 与 val |
| 训练 checkpoint / tensorboard logs | 每个实验独立 `runs/exp/<EXP_ID>` 和 `artifacts/<EXP_ID>`，写入前检查目录不存在 | 避免 checkpoint 混写、同名覆盖 |
| 评测 select seed pool | 可以预注册分配不重叠窗口；评测时由 Lead 管理 worker | 避免把两个 arm 随机条件弄得无法比较 |
| 新 confirm 池 | Lead + Reviewer 双人放行，**锁定后消费一次** | 不可因结果不好反复换 seed 当确认 |

可将 GPU 任务包装成系统级共享文件锁（`RF_SHARED_LOCK_ROOT` 必须为所有 worktree 相同的绝对路径）：

```bash
# PR-14 实现以后才能运行；本例只展示资源隔离方式。
export RF_SHARED_LOCK_ROOT=/absolute/shared/riichiformer_locks
mkdir -p "$RF_SHARED_LOCK_ROOT"
flock -x "$RF_SHARED_LOCK_ROOT/gpu_exclusive.lock" \
  /absolute/path/to/approved/venv/bin/python scripts/rf_train.py --config path/to/experiment.yaml
```

不要把 lock 存在各 worktree 自己的相对路径中，否则不同 Agent 以为都取得了 GPU 独占权。

## 7. 合并 / 验收协议：每个 Wave 汇合后只能由 Lead 开门

```
子代理任务提交 ─→ Reviewer 只读 P0/P1/P2 + 负面测试
                          ↓
                   Lead 检查文件所有权
                          ↓
               integrate/cherry-pick 到共同基线
                          ↓
               全套相关测试 + ABI/hash parity
                          ↓
               更新 TASK_STATUS 与资源账本
                          ↓
                放开下游 hard dependencies
```

每个提交必须有：

- 起始 `BASE_SHA` 和精确 diff；任何跨 PR 修改须事前 RFC。
- targeted tests + affected regressions 的完整 stdout/exit code；Reviewer 可复算测试。
- PR 对应 `PREREG` / `ACCEPTANCE` 和 NOT_RUN/BLOCKED 明示，不得把写了的测试说成运行过。
- 共享输入与产物 SHA、source rights/规则版本、schema/action hash；数据大文件仅保留可审计摘要，不能误推公开库。
- 1 段回滚方案：出错时 revert 单个任务提交，不清理用户工作树、不重写旧结果。

如出现接口变更：**提出 `RFC-SCHEMA-###.md` → Lead 批准 → Schema Owner 修改与升级 hash → 相关 Agent rebase 到集成 SHA → 重做 parity**。严禁模型 Agent 自己新增隐藏信息字段并默默改 Snapshot。

## 8. 前两轮建议立即派出的指令（具体、有限、不会互相覆盖）

**第一条只给 Lead**：完成 H-00，并记录 `BASE_SHA`、用户未提交内容、子模块补丁状态、publish 流程和 `RF_SHARED_LOCK_ROOT`。**确认 H-00 PASS 后，才派下列子代理：**

- `Agent-A-H01-SCORE`：仅修改 `riichiformer/data/replay.py` 与相关 `rf_test_replay.py`；对照 native `hora.deltas`/供托结算，覆盖最终供托、双荣、无 seat/replay；提供修复前失败后通过证据。**不得修改 canonical/schema。**
- `Agent-B-H02-DEDUP`：仅负责 canonical/dedup/split 及对应脚本测试；先在 10k/100k 证明流式断点逻辑、duplicate graph 跨 split 不泄漏，再向 Lead 申请独占 IO 跑全量。**不得修改 replay.py。**
- `Agent-C-H03-RIGHTS`：核实每一种数据源和相关条款、输出逐源证据与权利矩阵；不自动更改为 `allowed`；把阻塞清晰记录。**不得启动训练。**
- `Agent-D-H04-RULES`：检查上游规则与 Tenhou 规则差异，新增三家和了/点数同分/连庄/末局供托等 fixture；只输出差异和建议修复，不在此 PR 重写 vendor/mortal。**不得修改 replay.py。**

**第二轮（在 Wave 1 有成果后）**：让一个 Agent 独占 PR-08 Schema；DATA 去完成 PR-07R；ENGINE 先写 PR-10 的 fixtures 等待 PR-09；MODEL 在 frozen schema 上写 PR-13 合成网络；EVAL 建立 PR-34 的空 registry/校准验证工具。Lead 审阅 PR-08 后，独占启动 PR-09，避免大面积冲突。

## 9. 每日任务状态（Agent 必须更新）

文件 `docs/riichiformer/v3/PARALLEL_STATUS.md`（Lead 单写）：

| task_id | owner | branch | base_sha | hard_deps | status | test | next_gate | GPU/IO reservation | last_commit |
|---|---|---|---|---|---|---|---|---|---|
| H-00 | Lead | TBD | TBD | — | NOT_STARTED | NOT_RUN | W0 | none | — |
| H-01 | A | TBD | TBD | H-00 | BLOCKED_DEP | NOT_RUN | W1 | none | — |
| H-02 | B | TBD | TBD | H-00 | BLOCKED_DEP | NOT_RUN | W1 | heavy-IO later | — |
| H-03 | C | TBD | TBD | H-00 | BLOCKED_DEP | NOT_RUN | rights gate | none | — |
| H-04 | D | TBD | TBD | H-00 | BLOCKED_DEP | NOT_RUN | W1 | none | — |

推荐状态枚举：`NOT_STARTED`, `BLOCKED_DEP`, `BLOCKED_RIGHTS`, `IN_PROGRESS`, `REVIEW`, `MERGED`, `ACCEPTED`, `FAILED`, `INCONCLUSIVE`, `SKIPPED`。**汇总状态按真实 merge 和 gate，不按照 Agent 自述。**

## 10. 并行的科学实验约束：不同 Agent 可写代码，不可同时改变多个变量来声称增益

- Candidate/Linear、数据 D0/D1、Memory/no-memory、Risk on/off、Value on/off 均使用唯一标识、冻结起始模型、相同预算及座位旋转。不同研究 Agent 可在不同工作树同时准备，但 Lead 用 **一个 GPU/评测资源队列** 完成实验。
- 评测**不是**“并行占用同一 confirm window”，而是相同 select 条件下可配对对照；最终 confirm 只对唯一冠军执行。
- 与旧 Mortal 相对收益仅作内部指标；没有可靠人类强度锚点不能宣称九段、天凤位级别。
- 一张 GPU 下如出现更高价值的瓶颈（环境 rollouts 或牌谱清洗），优先暂停低价值 Memory/大模型搜索，不盲目追求子代理满负荷。
