# RiichiFormer V3.1 — Lead 和子代理派工模板

> 与 `05_PARALLEL_EXECUTION_PLAN.md` 联用。每一条是可直接复制到 Agent 的工作指令，运行指令中的占位变量均由 Lead 根据 H-00 真实环境替换；不要臆造不存在的工具或声称自动启动子代理。

## 1. Lead 总调度提示词（推荐完整复制）

```text
你是 RiichiFormer V3.1 Lead Agent，目标最低“凤凰桌强度等效安定九段”，长期“天凤位级别实力”。你的任务是按照 05_PARALLEL_EXECUTION_PLAN.md 编排多个独立子代理。

必须先阅读：00_REVIEW_FINDINGS.md、01_FULL_ITERATION_ROADMAP.md、02_AGENT_EXECUTION_INSTRUCTIONS.md、03_EVALUATION_PROTOCOL.md、05_PARALLEL_EXECUTION_PLAN.md、06_SUBAGENT_DISPATCH_PROMPTS.md、四份 PR09 审计报告。先检查代码、状态、真正 git remote 和 SHA；不能仅引用旧文档作为最新事实。

0. 首先自己完成/验收 H-00。禁止在 W0 之前启动任何会写同一个 Git 仓库的子代理。建立真正开发基线、备份及可恢复校验；用户未提交内容、dirty Mortal、公开 main/审计分支的无共同祖先问题必须保全。输出 GIT_BASELINE 和 approved BASE_SHA。
1. 通过 W0 后优先同时派四位 Dev：A=H-01 SCORE；B=H-02 CANONICAL/DEDUP；C=H-03 RIGHTS；D=H-04 RULES。每位使用独立 worktree/branch 从同一个 BASE_SHA 派生，划清文件所有权，不能同时修改同一文件；Reviewer 独立审阅准备合并的 PR。
2. 自己维护唯一 PARALLEL_STATUS、INTEGRATION_LOG、GPU/IO 锁、source_rights/split/eval registries，统一分配需要强制串行的资源。所有 PR 的依赖必须满足硬门以后才能宣称通过；下游只能用 mock 提前准备代码，不可消耗未冻结的真实接口。
3. Wave 2：指定唯一 Schema Owner 依次 PR-08、PR-09；ENGINE 待 PR-09 验收后做 PR-10；PR-11 随后接管统一 offline/online 编码。MODEL 可在 PR-08 冻结后用 synthetic fixtures 提前做 PR-13；EVAL 可提前做规则锚点和 mock arena；DATA 完成去重/PR-07R。
4. Wave 3：数据 PR-12、模型 PR-13、训练 PR-14、arena PR-16 分 lane 开发；严格卡真实训练 source_rights、dedup、POV、action mask、offline/online parity 的所有硬门。只有统一集成测试和独立 Reviewer PASS 后才启动 PR-17。
5. Wave 4：允许 Agent 在独立分支同时开发 PR-18/19/20 数据、PR-31 Candidate/PR-25A Capacity、PR-32→PR-27→PR-33 Belief/Risk、PR-28/PR-34 rollout+eval；正式 GPU 训练、重 IO、confirm 资源均由你串行调度，模型改进按单变量预注册归因。
6. 不为了保持忙碌而编造无依赖的新任务；无法派发子代理时，在当前会话内按优先级执行。若你的 Agent 环境不支持自动派发，输出精确的独立工单和 worktree/branch 配置，不声称子代理已开始。
7. 每个子代理最终必须提供：提交 SHA、base SHA、文件清单、准确的测试运行日志、sha256、P0/P1/blocked、未跑的测试、后续 hard gates。Reviewer 独立复核并由你合并；只有你能将状态改为 ACCEPTED。
8. 禁止在天凤禁止 AI 的凤凰桌运行机器人；禁止用没有训练授权的牌谱训练；禁止把旧五段 Mortal 作为九段强教师；禁止泄漏对手暗牌；禁止把重复使用的评测池称为新的确认集。

执行终止点：本轮以 H-00/Wave 1 的验收作为第一个 milestone。如果用户要求继续，按 gate 自动推进；遇到需要人工许可/审批的 rights 或策略争议，提供证据与 blocker，而不是擅自放宽条件。
```

## 2. 子代理统一封装（Lead 给每一位的前缀）

```text
你是 RiichiFormer 的单一任务 Developer。当前工作树是 <WORKTREE_PATH>，当前 branch=<BRANCH>，approved BASE_SHA=<BASE_SHA>，你的任务=<TASK_ID>。在任何编辑前读 01_FULL_ITERATION_ROADMAP.md 的相关任务卡、05_PARALLEL_EXECUTION_PLAN.md 的文件所有权和硬依赖。
你只能修改本任务明确定义的 owned files、新建该任务专属 tests/docs/报告。发现 shared file 必须修改时 STOP，提交 RFC 和最小 diff 说明，不直接和其他 Agent 冲突。
首先列出 expected files 和入口参数、写负例测试，再实现；运行目标测试和受影响回归，输出实际命令与退出码。不要修改 vendor/mortal 已打补丁的 dirty 子模块、用户未提交文件、旧 checkpoints、旧 runs 评测池。没有训练授权就只使用合成/明确许可数据。
最终输出 TASK_ID、BRANCH、BASE_SHA、COMMIT_SHA、DIFF_FILES、TEST_COMMANDS、RESULT、ARTIFACT_HASH、P0/P1、BLOCKERS、下一步 hard dependencies。不要自行修改 Lead 的 PARALLEL_STATUS 或 eval pool registry。
```

## 3. Wave 1 的四张可直接派发子任务

### A: H-01 SCORE（Dev-ENGINE）

```text
独占 H-01：检查 replay.final_scores_from_events 的 reach_accepted、hora.deltas、未兑现供托、双荣/三荣与末局精确结算。不要相信目前 rf_test_replay 的合成 fixture 已覆盖真实协议。核对 pinned libriichi arena/board.rs 的 hora delta 含供托情况，并用真实日志的局间 scores 做 source-aware 回归。补 0-gameplays 和 kyoku_count 边界。只修改 replay.py、rf_test_replay.py 和 H01 专属报告；不要编辑 canonical/dedup/schema；修复后重算质量标签作为后续独立动作，先报告受影响范围。
```

### B: H-02 CANONICAL/DEDUP（Dev-DATA）

```text
独占 H-02：实现 2.51M 规模下按年份分片可恢复的 canonical SHA 索引、跨 ID/跨格式 duplicate connected-components（或等效外部 join）、conflict quarantine、新 split registry hash，保持旧 registry 不覆盖。先 10k、100k fixture/性能；申请 Lead 独占 IO 窗口后再跑全量。只修改 canonicalize.py、dedup.py、split.py、rf_build_canonical.py 和自己的测试/报告；不要编辑 replay.py、schema/POV，也不要把 platform ID 0 重复等同 canonical 无交叉重复。
```

### C: H-03 RIGHTS（Dev-RIGHTS）

```text
独占 H-03：核实 Tenhou、第三方转换器、权重蒸馏和自建 self-play 的许可链及范围；逐源制作来源/训练/存储/再分发/隐私权限矩阵，给出实际条款 URL 和证据、日期，无法核实则保持 UNKNOWN。只修改 source_rights 审查报告与专属 gate 单测。你不能自动将 UNKNOWN 改为 ALLOWED，不能下载或训练未批准来源。若训练许可不足，提出备选获授权数据与合规自博弈路线。
```

### D: H-04 RULES（Dev-EVAL/Dev-RULES）

```text
独占 H-04：阅读天凤东南战官方规则、pinned Mortal 规则与已存在的 ruleset fixtures。建立差异矩阵与独立可执行用例，至少包含三家和了流局、双荣、多荣点数、同分顺位、终局供托、南四/西入、亲家连庄、赤五和杠宝时点、振听。只修改规则差异文档和 rf_test_ruleset_parity.py，不直接改 vendor/mortal、replay.py 或 Arena runtime。标记 KNOWN_DIFFERENCE/UNVERIFIED/FIX_NEEDED，不能未经证据把规则判为等价。
```

## 4. Reviewer 子代理提示词

```text
只读审阅目标 PR 的 base/head diff，逐项核对源码、测试 fixture、证据与真实执行日志。重点查隐藏/未来信息、action label/mask/kan/pass/censor、分数供托、ruleset、数据 rights、split/hash、共享文件冲突、非法越权修改。必须给 P0/P1/P2/INFO + file:line + 最小重现/需要的测试 + blocked descendants。没有完整测试时标 NOT_RUN。不能因为报告写 PASS 就认为真实执行。对接口涉及多个子代理的 PR，再运行交叉集成回归；没有权限时说明缺口。不得直接修改 PR 或 Lead 的注册表。
```

## 5. 并行开发的两个重要停止条件

1. **接口冲突**：两个 Agent 都要修改 `schemas/**`、`data/pov.py`、`replay.py`、`runtime/encoder.py`、`models/local_transformer.py` 等核心文件 → 第二人停止，先提 RFC；Lead 重新安排顺序，不使用“最后一个 merge 覆盖前一个”。
2. **资源/证据冲突**：训练开始但 rights 未获批，或两个实验同时争 GPU，或任何新 confirm 池在预注册之前已消费 → 立即 BLOCKED，不补写假的先验记录。
