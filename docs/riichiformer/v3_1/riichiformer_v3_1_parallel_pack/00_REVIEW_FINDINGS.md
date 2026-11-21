# RiichiFormer V3 — 独立审阅结论（2026-11-21）

> 本文对应用户上传的 `PR09_AUDIT_REPORT(1).md`、`PR09_CHANGE_IMPACT(1).md`、`PR09_INTERFACE_CONTRACT(1).md`、`PR09_TEST_MATRIX(1).md`，以及对 GitHub 仓库 `ygmldr/riichi-ai` 的 `riichiformer-v2-audit-pr09` 分支和上游 `Equim-chan/Mortal` pin `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2` 的源码交叉核对。没有在用户训练机上实际重新运行代码。审计报告中所记 98 passed、2,512,424 logs 是**审计 Agent 的记录**，不是本次独立执行的测试结果。

## 0. 决策摘要

- 不推倒 PR-00～06；完成的扫描、数据规范化、native replay、split、质量工具可以复用。
- PR-07 未完成；PR-08/09 尚未实现。不得把 PR-09 当作已经通过安全门。
- 先处理 **BR-01 分数结算**、**BR-02 分支/仓库迁移**、**BR-03 全量去重与 split**、**BR-04 权限核实**、**BR-05 POV 和动作契约**、**BR-06 规则仿真差异**。
- 不把 Mortal 当作强策略教师。允许作为代码基准和弱对手；其教师蒸馏须单独 A/B 证明净收益。
- 新架构先做小型 Local Transformer + 轻量状态编码 + 46 mask，随后分别检验 Candidate Policy、Belief/Risk、Value、跨局 Memory、自博弈。
- 研究目标：内部达到足以支持凤凰桌**安定九段**推断，再追求**天凤位级别**策略；绝不能将低段位对手的胜率直接转换为天凤段位。

## 1. 已确认的工程现状

| 原 PR | 事实状态 | 备注 |
|---|---|---|
| 00、01、02 | 主要已完成 | preflight/GPU mock bench/数据扫描；bench 不等于真实 Transformer 吞吐 |
| 03 | 代码完成，数据仅 1,000 场 | `full_corpus=false`；canonical hash 全量缺失 |
| 04 | 代码完成，replay 仅 1,000 场 | 660,568 decision / 0 illegal 为样本现象，且分数函数需修正 |
| 05 | split 已生成 2,512,424 行 | 只有 game_id 去重；跨来源 canonical 重复未排除 |
| 06 | 质量工具已完成 | 可靠性结论 `INCONCLUSIVE`，不能按 top25 为高强度事实 |
| 07 | 草稿 | 单测 9 passed，但缺 DRIFT_REPORT 和实际运行 |
| 08、09 | 空白 | 无 schema/action hash/POV 投影文件 |
| 10 以后 | 未开始 | 无 V2 模型、训练器、在线 encoder 或 arena adapter |

## 2. 独立源码审阅新增发现

### BR-01 [高，阻塞依赖最终顺位的标签/质量筛选] `final_scores_from_events()` 疑似双计供托

文件：`riichiformer/data/replay.py` 的 `final_scores_from_events()`：

```python
elif etype in ("hora", "ryukyoku"):
    scores += deltas
    if etype == "hora" and kyotaku:
        scores[actor] += 1000 * kyotaku  # <-- 额外再加一次
```

上游 `libriichi/src/arena/board.rs` 在 `handle_hora` 构造 `Event::Hora { deltas }` 时，已经写入 `deltas[actor] = point.ron + kyotaku_point + ...`（tsumo 分支同样计入）。因此对使用该约定的日志，额外加分会重复结算。当前测试 `tests/rf_test_replay.py::test_reach_and_kyotaku_are_accounted` 使用与规则引擎构造不一致的人工 deltas，无法发现此类错误。函数还缺少结束时剩余供托分配给第一名的逻辑。

**必须操作**：不要盲目改一个 `+=` 就结束。创建由真实 native engine 导出的带立直和供托的逐局事件 fixture（含和牌、流局、末局供托、双荣），确认具体来源/版本的 `hora.deltas` 约定；对每次局末重放结果与下一局 `start_kyoku.scores` 的四家分数进行精确比较；若最终一局无下一局，优先使用独立权威结算/末局分数导出交叉验证。测试必须覆盖正负样本。修复后重新生成 PR-06 质量测量。

### BR-02 [高，阻塞安全移交] `main` 与审计分支没有共同祖先

GitHub API 读取到：公开 `main` HEAD=`a3ccd85380880c1b97ef230a5e8f7b4ec35ce5a3`，审计分支 `riichiformer-v2-audit-pr09` HEAD=`f23bb0dc9a6f483686e3c0eb195b36d302d3b980`。GitHub compare 返回 **No common ancestor**。公开 `main` 经过 filtered history 发布；审计分支保留第一方 `riichiformer/` 代码，目前 `main` 不含该目录。

**必须操作**：代理在真实训练机先 `git status --porcelain=v1`、`git rev-parse HEAD`、`git branch -avv`、`git ls-tree`、备份未跟踪代码和打补丁的子模块，然后决定继续本地原历史开发分支或创建新开发分支；若希望把成果发布到过滤历史的公开 `main`，必须使用经审查的 *create-only staging/publish workflow*，不可 `git merge`、`reset --hard` 或强制推送。审计分支不是普通 PR 的可快进基线。

### BR-03 [高] native replay 把 0 seats 接受为成功

`replay.py` 判断 `if len(gameplays) not in (0, 4): ...`，因此 `len(gameplays)==0` 继续保持 `ok=True`；对正常完整四麻，0 席动作样本应显式报警/拒绝，除非有明确且经过 fixture 验证的特殊例外。加 `test_zero_seat_rejected`。另 `kyoku_count` 由 seat0 的 `at_kyoku` 去重计数，未行动的局会漏计，应从原始边界事件计算并与 native 比较。

### BR-04 [高] 跨 ID 的 canonical 重复虽然标记但未自动隔离

`dedup.py` 在同一 Tenhou ID 内分组，但不同 ID 内容相同仅设置 `cross_key_duplicate=true` 供人工审查；当前 split 在所有 ID 唯一时仍可能把同一游戏的不同导出副本放入不同 split。要求在 final manifest 中对同 canonical hash 的所有记录采用统一 duplicate group，冲突的 game_id/数据不完整的 group **quarantine**；同 ID 不同内容也 quarantine、不得任选代表后继续训练。

### BR-05 [高] 全量 canonicalize 的内存与 I/O 风险

`rf_build_canonical.py --all` 将 2.51M inventory `to_pylist()`，随后积累 `records`、`index_rows` 并一次写出 JSONL。它有 `--all` 开关不等于可稳定完成全量。需要按年份/月分片写 Parquet/SQLite 或外部排序 join，checkpoint/resume/worker quarantine，保证与小样本 in-memory 结果逐字一致。全量 native replay 可先做 >100k 分层定向压力验证，最终进入训练的子集必须通过逐局规则重放，不能以 1,000 场 0 illegal 外推。

### BR-06 [高，对安定段位测量至关重要] 官方天凤规则与训练 arena 有已知差异

上游 `libriichi/src/arena/board.rs` 自注释：`No triple-ron ryukyoku`；天凤官方段位四麻规则列有三家和了流局。对局规则偏差会影响强度标定。再审计：赤五各 1、双荣供托/本场归属、南四亲家听牌/和了止、飞牌结束、西入、杠宝即时/延迟、四家立直/四杠流局、振听、立直后暗杠、同分顺位、最终供托分配。每项提供 `upstream behavior` vs `Tenhou behavior` vs fixture vs 业务影响。不能凭默认 arena 声称天凤完全等效。

### BR-07 [中] 数据覆盖与质量用语需要更严格

- “扫描 2,512,424 logs 0 bad”代表扫描层文件格式/可读性检查，不代表这 251 万局均已完整 JSON 解析、规则重放、合法性认证。
- `quality_audit.md` 的 q 非单调，平均顺位 Q1/Q4 区分度不足，且抽样导致历史不足；不能基于现有 q 过滤掉 75% 数据。要先修结算并复跑完整时序历史。
- 2026 仅有 1 月（12,188 局）且旧 EX1 已使用，不能称新鲜独立 confirm。2024–26 不能自动视为更强或 AI 纯净数据。
- `source_rights` 默认 unknown，因此 `train_allowed=0` 是项目所设的 fail-closed **操作门禁**，不等于独立法律判定；需根据官方条款与授权记录决定实际允许用途。

## 3. 安全与使用许可：两条独立轨道

天凤官方说明：目前不接受新的 AI 专用 ID 申请、禁止 AI 参加凤凰桌；牌谱公开主要用于支持公平游玩，面向与天凤对战无关服务的再利用受限制；如用于一般麻将研发，官方要求先联系运营方核实（https://tenhou.net/man/#AI 和 https://tenhou.net/man/#PAIFU）。不能因已从第三方下载转换牌谱，就认定可训练/再分发。

- **轨道 A（明确许可）**：人工复核资料/获得必要授权 → `source_rights` 按真实允许范围置 `allowed` → 才可按 frozen split 建立 supervised shard、进行训练。
- **轨道 B（未获授权）**：继续做本地只读元数据与规则一致性审计；训练只使用明确获授权来源以及经核实的独立规则环境生成数据。优先创建合法的自博弈/规则 bot/搜索式 curriculum；不得将 `unknown` 来源或用它们计算的教师标签偷偷并入训练。

## 4. 审计证据与级别

- **A：源码直接核验**：分数函数、0 seat、canonical dedup grouping、`--all` 载入方式、上游 arena 特殊规则。
- **B：上传报告与 GitHub 分支一致，未独立执行**：98 passed、2.51M 扫描、100k 质量样本、split SHA 与训练机环境。
- **C：待实验确认**：上述 bug 对 q/最终 rank 的影响程度、规则差异在大量对局中的实际频率、模型是否能跨越五段至九段。

所有后续 PR 报告必须显式使用这三种证据级别，不得把 C 说成 A。

## 5. 立即修复顺序

`H-00 Git safeguard` → `H-01 scoring parity` → `H-02 canonical/dedup` 与 `H-03 rights review`（可并行） → `PR-07 drift completion` → `PR-08 schema` → `PR-09 POV` → `PR-10 snapshot/censor` → `PR-11 online parity` → `PR-12 train shards` → `PR-13+ model`。

若目标是尽早编码模型，H-02 **全量任务**可与 PR-08/09 开发并行，但**训练用 shard 与数据门禁**不得在 H-02 和 H-03 完成前放行。
