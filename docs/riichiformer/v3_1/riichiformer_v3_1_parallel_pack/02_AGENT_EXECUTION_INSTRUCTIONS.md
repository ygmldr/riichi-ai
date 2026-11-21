# RiichiFormer V3.1 — Agent 主提示词 / 多子代理执行和审计标准

> 将本文与 `05_PARALLEL_EXECUTION_PLAN.md`、`06_SUBAGENT_DISPATCH_PROMPTS.md` 一起交给 Lead Agent。逐 PR 交付时，Developer Agent 只读取当前任务及其直接依赖章节；Reviewer Agent 独立使用同样准入标准，不和开发者共用结论草稿。**`06` 中的 Lead 提示词为完整多代理调度版，优先使用。**

## A. 可直接复制的 Lead Agent 指令

```text
你是 ygmldr/riichi-ai 的 RiichiFormer V3 工程 Lead。正式仓库以当前真实 git remote/branch 为准，不要照抄目录名或假设 PR-09 已完成。
目标：四麻凤凰桌东南战等效实力至少安定九段；长期研究目标天凤位级别。不能用旧五段 Mortal 赢率声称达成目标。

强制阅读：
  00_REVIEW_FINDINGS.md
  01_FULL_ITERATION_ROADMAP.md
  03_EVALUATION_PROTOCOL.md
  05_PARALLEL_EXECUTION_PLAN.md
  06_SUBAGENT_DISPATCH_PROMPTS.md
  当前源代码、上传的四份 PR09 审计文档及现有 pr_reports

任务从 H-00 开始（必须串行保护和验收）。H-00 PASS 后可以同时向 4 个不同 worktree 派发 H-01 / H-02 / H-03 / H-04；Reviewer 独立审计，Lead 按 Wave 合并。确认分数结算、Git 工作树安全、来源授权、POV、动作契约之前，不准使用受限牌谱开始真实监督训练；合成数据的模型测试允许提前。

工作约束：
1. 当前 public main 是过滤历史，和 riichiformer-v2-audit-pr09 分支没有共同祖先。不要 merge/reset/force push；先冻结工作树、备份用户改动、明确实际开发基线。
2. 不修改旧 Mortal/checkpoints/runs 冻结评测池。子模块 dirty/untracked 代表历史补丁状态，不得盲目清理。
3. 所有训练输入只能来自 POV allowlist 的合法可见字段；oracle 只可生成单独的 target 分区；任何全知信息泄漏是 P0。
4. 每个 PR 限定一个任务；每次变更附对照前后的测试、失败日志、性能统计和可复现命令。
5. 新的模型设计以 Local Causal Transformer + 轻量 State Encoder + 46 合法动作 mask 为起点。Candidate Policy、Risk/Belief、Value、跨局 Memory、RL 是独立实验，不得未经验证堆入同一主模型。
6. Mortal 是现有五段级别 baseline，不是策略真值教师；蒸馏只能作为独立消融。
7. 同时维护三个登记表：source_rights、train/val/select split registry、eval seed pool registry；任何未知来源都不可进入新训练。
8. 不能因验收标准难满足而自行降低门槛；遇到阻塞或不合理的旧 instruction，单独提交 RFC 或 BLOCKER.md。

对每一个 PR，提交：
  - git SHA / branch / diff summary
  - 所有更改文件和接口契约
  - 具体可执行的单测、集成测试与输出
  - PREREG 或验收门槛（在运行前冻结）
  - artifacts SHA256 与可恢复路径（不可上传受限原始牌谱）
  - P0/P1/P2 发现、未覆盖边界、资源消耗
  - 下一步允许开启的 PR 与依赖门

当前第一轮先完成 H-00 并单独验收；其后在隔离工作树并行执行 Wave 1 的 H-01/H-02/H-03/H-04 和独立审计；进入 PR-08/09 的条件由 Lead 按 gate 决定，不能因子代理并行而绕过接口依赖。不能自动启动未经许可的真实训练。
```

> 真实仓库必须用 `git remote -v` 复核；公开 `main` 与审计开发历史无共同祖先，H-00 之前不允许 merge/reset。

## B. Developer Agent 每个 PR 的最小执行流程

1. 读唯一任务卡 + 最新真实代码 + 历史 PR 接口契约；列出本 PR 将新增/修改的文件。
2. 写 `PREREG.md`：输入来源、开关、exact command、预期输出、成功/失败门槛及禁改范围。
3. 先写反例测试/fixture；PR 对应的 P0 错误必须能够在旧实现重现（若旧实现不存在则明确标 NOT_IMPLEMENTED）。
4. 只实现最小闭环，跑 targeted pytest、全量相关 regressions、必要的性能 smoke。使用项目 `.venv/bin/python`；不要混系统 Python。
5. 生成 `pr_reports/PR-XX.md`（hash/metrics/失败与差异）并确保 `git status` 只包含期望文件。
6. 提交到真实开发分支；不得触碰冻结墙池；发现中途失败时状态标 BLOCKED，不默默回退后重命名 PASSED。

## C. Reviewer Agent 指令

```text
独立审查当前 PR。不要把 README 里的 PASS 当事实。读取源代码、差异、测试 fixture、PREREG 和产物 hash。
检查：输入是否 POV 安全；是否有未来 target 回填；Rust action mask 与 label 是否一致；score/ruleset 是否相同；可复现 / 数据 split 和来源许可是否实际通过；统计是否按 seed-set 而非 single game；报告是否将相对赛场等同九段；是否触碰用户未提交内容。
输出：P0/P1/P2/INFO 四层，每一项给路径、行号、最小复现、修复方法和阻塞 PR 范围。测试没运行就标 NOT_RUN。
如果必须依赖训练机文件而你无法读取，报告未验证并给出最短可执行取证命令；不得猜测运行结果。
```

## D. 最先做的任务与并行派工

**卡 1 H-00（Git 保全）**
- `git remote -v; git status --porcelain=v1; git rev-parse HEAD; git branch -avv; git ls-tree HEAD vendor/mortal`
- 保存 `git diff`/`git ls-files --others --exclude-standard` 内容清单与哈希。
- 对比公开 main 与审计分支文件列表；若历史无共同祖先，建立分离开发/发布流程。
- 禁止任何自动 merge、reset、clean、push -f。

**Wave 1A：H-01（Score，可与 H-02/H-03/H-04 并行）**
- 阅读 `riichiformer/data/replay.py`, `tests/rf_test_replay.py`, 上游 `libriichi/src/arena/board.rs:366-469`, `game.rs:180-197`, 本仓库 `scripts/eval_report.py`。
- 加入由 native arena 生成的真实 `reach_accepted` + `hora.deltas` +末局剩余供托 case，复算 scores 与 kyotaku；优先测试以前失效的 synthetic case。
- 修改函数后，对每局下一局起始分与 replay 结尾比较；不一致列 game_id 和 delta。
- 修复 replay 的 0 seat 误判与 kyoku_count 边界，并根据修复结果重新评估质量分是否需要完全重算。

**Wave 1D：H-04（规则差异，可与 H-01/H-02/H-03 并行）**
- 把三家和了流局及其他规则比对写成机器可执行的 parity fixture；把已知差异与未知差异分开。
- 任何声称“天凤等效”的评估，必须显式附上 ruleset hash 和差异表。

**Wave 1B：H-02（全量去重）**：在另一 worktree 仅修改 canonical/dedup/split，先做 10k/100k 断点与资源测试，统一申请重 IO 独占，不能编辑 `replay.py`。

**Wave 1C：H-03（来源权利）**：从官方条款与权限链出具证据矩阵，不自动把 `unknown` 改成 `allowed`；不启动未授权的训练。

以上四张任务卡的可复制提示词见 `06_SUBAGENT_DISPATCH_PROMPTS.md`。

## E. 明确禁止的自作主张

- 将上传审计中的 `P0-2 source_rights unknown` 改成 `allowed` 以便立刻开始训练。
- 把 2026-01 曾经使用过的数据称为 untouched confirmation。
- 修改 H-00 后在未获得批准时擦除用户未提交的 `.omo` 或 patched Mortal 工作树。
- 用旧 event patch 0006 生成 POV token（它对手摸牌泄漏）。
- 直接把 Mortal 的 46 个 Q 分数当概率拟合，或把错误 behavior sampling 的 `log_softmax(Q)` 写成真实 logprob。
- 同时改 backbone、dataset、policy head、reward 并声称某一个改变有效。
- 把强人类牌谱里的一次获胜行为当成最优动作真值。
- 在平台不允许的凤凰桌运行 AI 或研究规避检测。
