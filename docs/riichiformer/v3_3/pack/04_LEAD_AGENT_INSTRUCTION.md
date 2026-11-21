# 给 Lead Agent 的直接执行指令（V3.3）

你现在作为 RiichiFormer V3.3 的唯一集成 Lead。用户报告本地 `rf/v3.2-local` / `rf/v32-integration` HEAD 为 `00e6576`，Wave 1 PR37～42 已合并，707 passed、2 skipped、281 subtests；B0 500 selfplay games、2 个 seed 各 2000 GPU steps；PR38/B0 reviewer 仍在运行；full epoch 已停且用户要求 GPU 空闲。上述均是用户提供的状态，必须在当前本机实际核对。

**必须先完成**：

1. `git status`, `git rev-parse HEAD`, 当前 worktree/branch/tag、`rf3.2-w1` 与将来 `rf3.2-w2`、dirty vendor/submodule、现有 checkpoint sha 与 PR44 pilot 日志的只读审计。先辨识真实本地分支与公开 remote `rf/v3.2-policy-parallel-plan` 的历史差异，**禁止拉取远程旧分支覆盖当前本地新提交**。若确需 fetch：`git fetch origin` 仅更新 refs；不得 reset/force-push/merge 无共同历史。
2. 等正在运行的 PR38 loader 与 B0 pilot Reviewer 结束；有 P0/P1 则修复、重跑测试后才打 W2 tag；没有报告不得谎称 Reviewer PASS。
3. 本轮**不要启动 GPU 正式训练**，不运行 6191-step epoch，不生成 12k Arena eval；用户会单独授权资源预算。
4. 并行派发四条互不重叠 worktree：A PR46 M5D checkpoint+adapter+PT 1v3，B PR45 Validation/PR47 E2E，C B0 trainer protocol，D R29/0014 native rules。明确 owned files 和依赖；Reviewer 只读。
5. Lead 需要找出用户说的“我们训练的5d Mortal”准确 checkpoint 路径、SHA 与推理设置，不能猜 `mortal.pth`；独立记录 `M5D-R0` 并冻结。
6. A 的主指标采用五段特上东南战 `[75,30,0,-105]`，辅指标 `[90,45,0,-135]`；一组 `wall_seed` 必须四席轮换完整；相对 PT `delta=4/3*challenger_raw_pt`；cluster bootstrap by complete seed set，CI、negative controls。
7. 参照旧 `scripts/eval_report.py`、`scripts/exp_a14_delta.py`，但是不要覆写历史 `runs/1v3`、`runs/variance`、旧 champion 权重；旧脚本固定 7D 计分、500/3000-set 守卫，必须实现带 scorer version 的新模块。
8. 维护 `EVAL_POOL_REGISTRY.md`，新 key 和 virgin confirm 审计在运行前冻结。PR44 的 500 game train seed 不能进入验证集；新验证集按完整 game_id/seed 隔离。
9. 只完成代码、文档、测试和小量必要 CPU smoke，验收后提交 `WAVE2_REVIEW.md`、明确各门绿/黄/红、实际耗时与 GPU 实验预算，**向用户申请下一次 GPU 运行的授权**。

**每个 PR 的标准交付**：`BASE_SHA`, `COMMIT_SHA`, `FILES`, `TEST_CMD`, `EXIT_CODE`, `NEW_HASHES`, `P0/P1`, `NOT_RUN`, `RULES_DIFF`, `GPU/IO usage`, `NEXT_GATE`。失败则提交清晰的 blocker，而不是修改测试以通过。

**目标是可信测量，而不是报告漂亮分数。** B0 即使相对 5D Mortal 是负的，也应完整记录并作为未来改进曲线起点。确实没有强度数据时请写 `STRENGTH_NOT_RUN`。
