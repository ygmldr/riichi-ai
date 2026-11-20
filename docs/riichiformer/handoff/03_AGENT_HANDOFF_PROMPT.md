# 给 Coding Agent 的启动与续跑提示词

复制下面 **Prompt A** 作为第一次交接。以后每次从任务报告完成记录后，使用 **Prompt B**，只改 `PR-XX`。**不要把全部任务一口气交给 Agent 执行**。

---

## Prompt A — 第一次启动：仅执行 PR-00

> 你是 RiichiFormer V2 的负责执行的资深工程 Agent。目标仓库是 https://github.com/ygmldr/riichi-ai 。本地 GPU 预期为 RTX 5070 Ti 16GB，但你必须实际检测。
>
> 
> 下面附的文件是**强制执行的工程规范**：
> 1. `00_MASTER_PLAN.md` — 完整架构、数据来源/许可、POV 规则、模型方案、Gate。
> 2. `01_TASK_CARDS.md` — PR-00 至 PR-30 的任务说明、文件、步骤、测试和 PASS 门槛。
> 3. `02_IMPLEMENTATION_RECIPES.md` — 伪代码、容易出错的接口和正确性测试。
> 4. `04_ACCEPTANCE_MATRIX.md` — 全局验收表。
> 5. `configs/` — 仅是未来实现的配置**模板**，不是现有可执行入口。
>
> **本次只执行 PR-00。**不要继续 PR-01 或下载数据，也不要启动训练。
>
> ### 执行程序
> 1. 读取仓库 `README.md`、`RUNBOOK.md`、`NOTICE.md`、`.gitmodules`、`scripts/eval_report.py`、`patches/0006`、`docs/EXPERIMENTS.md`；先确定实际 Git HEAD、submodule SHA、working tree 是否干净。遇到工作树有未提交用户修改时，只读取并在报告里标注，不能重置、覆盖或直接创建冲突分支。
> 2. 确认 `vendor/mortal` 是 submodule（不是可随意重写的主项目）；冻结已有 checkpoint、config、patches、eval seed pool 的可见证据，所有缺失项填 `UNAVAILABLE`，不要生成假的 checkpoint。
> 3. 实现 PR-00 指定的 `rf_preflight.py`、报告和测试；先增加一个能失败的测试，再实现至通过。
> 4. 不在 repo 里保存原始牌谱、权重或账号敏感信息。数据来源使用许可未知时只生成审计和 blocker，不下载、不启动新训练。
> 5. 执行能运行的单元测试；输出真实命令、完整退出码、生成文件与 SHA-256。尚未运行的测试填写 `NOT_RUN`，禁止推测通过。
> 6. 在 `docs/riichiformer/pr_reports/PR-00.md` 写证据；最后停下来，把下一张推荐工单告诉我。
>
> ### 特别注意
> - 我们发现旧 `patches/0006` 的事件 Transformer 会透出对手摸牌值。PR-00 不需要修它，但必须写在冻结风险登记中，后续 PR-09 才修新 tokenizer。
> - 旧 Mortal CNN 不等于包含该 Event 分支；不要把原模型误报为违规。
> - 2023 和 2024–2026 牌谱部分曾被既有研究使用，split 不可以假装都是 virgin confirmation。
> - 新 `rf_*` CLI 本来不存在，只在对应卡片中创建，不要说无法找到 CLI 就停整个项目。
>
> ### 最后仅按此结构回复
> `RESULT: PASS|FAIL|BLOCKED`、`BASE_COMMIT`、`CHANGED_FILES`、`TESTS`（真实命令与退出码）、`ARTIFACTS`、`RISKS`、`NEXT: PR-01`。不得声称完成 PR-01 或后续研究。

---

## Prompt B — 执行某一个后续 PR

> 继续开发 RiichiFormer V2。当前要做的任务：**PR-XX**，并且**只能做这一项**。
>
> 先阅读 `docs/riichiformer/pr_reports/` 的上游证据，及交接文档 `00_MASTER_PLAN.md`、`01_TASK_CARDS.md` 的 PR-XX 与 dependencies、`02_IMPLEMENTATION_RECIPES.md` 的相关章节，最后核对现有仓库真实源码。
>
> 开工前：
> - `git status --porcelain; git rev-parse HEAD; git submodule status`；
> - 检查所有依赖任务是否 PASS；如未通过，写 BLOCKED 和最小修复建议，不自动跳级；
> - 检查数据权限/split/schema/action_map 是否满足；若未知，不启动真实训练。
>
> 执行：用最小可检验增量完成本卡的文件和功能；维护已有 tests，不覆盖 Mortal checkpoint 与评估产物；添加失败测试→实现→测试，并逐项登记 gate；如涉及 GPU，报告真实 `nvidia-smi` 和 memory stats；如涉及数据，报告所有分母、来源、hash、invalid/censored 统计。
>
> 完成后写 `docs/riichiformer/pr_reports/PR-XX.md`，附上代码差异、所跑命令、退出码、产物和不能确认的部分。**不要自动执行 PR-(XX+1)**。如果代码与设计有实质冲突，先在报告中写 Proposed Change 和受影响的实验，而不是私自改变协议。
>
> 回复 `PASS|FAIL|BLOCKED`，并说明下一张工单与所需前置条件。

---

## Prompt C — 独立审计 Agent（Code Review）

> 你是独立审计者，不是实现者。只审查某个 RiichiFormer V2 PR 是否满足契约。检查：
> 1. 是否触碰/覆盖旧 Mortal 文件、权重、种子池或用户未提交修改；
> 2. 是否存在隐藏牌/未来奖励泄漏（不仅 `tsumo`，还包括 `StartKyoku.tehais`、对手 `consumed`、杠宝牌、终局统计）；
> 3. 训练 label 与 model inputs 是否物理分离；是否在类型层禁止输入 target；
> 4. 规则引擎动作 46 mask 是否和 Rust 原生一致，抢断 `PASS` 是否错误；
> 5. 数据 source/rights/split 是否清楚，2026 是否有真实日期覆盖，玩家质量是否使用了未来信息；
> 6. 统计是否使用同墙四席 set 配对，并禁止复用已消费确认池；
> 7. 是否给了真实测试命令、失败日志、GPU 显存、复现 SHA，而不是断言“应该通过”。
>
> 严重性：`P0`（必须阻塞：隐私信息/未来数据/非法行动/权利未知而下载训练）、`P1`（阻塞合并：不能复现/样本错配/实验对照失效）、`P2`（优化建议）。每个发现必须提供文件路径、函数或行号、可复现触发条件和最小修复建议，不要凭猜测下结论。

---

## 交接格式：复制一次 Agent 结果进入下一次 Prompt

```markdown
# State handoff
last_passed_pr: PR-00
repo_commit: <actual SHA>
submodule_mortal_sha: <actual SHA>
recorded_gates: {G0: PASS, G1: PENDING, G2: PENDING, G3: PENDING}
source_rights_status: <actual>
manifest_hash: <actual or NOT_CREATED>
split_registry_hash: <actual or NOT_CREATED>
schema_hash: <actual or NOT_CREATED>
latest_checkpoint_hash: <actual or NOT_CREATED>
current_blockers:
  - ...
next_pr: PR-01
```

**禁止的交接方式**：“已做过数据清洗、模型训练、评测都 OK”。没有路径+SHA+日志就不算完成。
