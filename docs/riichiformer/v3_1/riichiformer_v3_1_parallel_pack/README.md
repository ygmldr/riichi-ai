# RiichiFormer V3.1 — 审阅、迭代与并行 Agent 交接包

**日期**：2026-11-21。**研究目标**：先证明工程可靠和超过旧 Mortal，再争取可被强度校准支持的四麻凤凰桌安定九段，长期追求天凤位级别策略。**现状**：审计分支中 PR-00～06 有实现，PR-07 草稿，PR-08/09 未开始；没有新 V2 模型或训练 checkpoint。**本包是方案，不是已训练结果。**

文件次序：

1. `00_REVIEW_FINDINGS.md` — 独立源码审计结论，新发现的分数结算/分支隔离/规则差异/去重风险。
2. `01_FULL_ITERATION_ROADMAP.md` — 全部续期任务的依赖 DAG、逐 PR 任务、代码路径、测试与验收。
3. `02_AGENT_EXECUTION_INSTRUCTIONS.md` — Lead/Developer/Reviewer 可复制的提示词和第一批工单。
4. `03_EVALUATION_PROTOCOL.md` — 安定九段、十段和天凤位级别的统计定义、对手校准、四席 duplicate 评测。
5. `04_IMPLEMENTATION_RECIPES.md` — 代码结构和伪代码，帮助 Agent 从审计数据直接接到 Transformer / Risk / PPO。
6. **`05_PARALLEL_EXECUTION_PLAN.md` — 并行 Waves、依赖 DAG、子代理文件所有权、Git worktree、资源锁、冲突与合并门。**
7. **`06_SUBAGENT_DISPATCH_PROMPTS.md` — Lead 调度提示词、首轮四位子代理工单、独立 Reviewer 提示词。**

**首批行动**：**H-00 单线程安全保全 Git 工作树**（公开 `main` 与审计分支无共同祖先）；验收后 Lead 在独立 worktree 并行派发 H-01 分数、H-02 去重、H-03 权利、H-04 规则；空闲 Agent 接 PR-07R；PR-08/09 按安全接口门顺序实施。未经来源许可与去重/POV 等硬门，不启动使用该来源的真实监督训练。

**推荐并发度**：1 Lead + 最多 4 个写代码 Dev + 1 只读 Reviewer。正式 GPU 训练和统计决策按预注册顺序由 Lead 独占调度，不可同时修改多个变量。

主要证据：用户上传的四份 PR-09 审计文件、`https://github.com/ygmldr/riichi-ai/tree/riichiformer-v2-audit-pr09`、Mortal 上游 pinned hash `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2`、天凤官方手册 `https://tenhou.net/man/`。
