# RiichiFormer V2 — Agent 实施包 v1.0

这是面向 `https://github.com/ygmldr/riichi-ai` 的**工程实施文档**，用于把「16GB 5070Ti 上的立直麻将 Transformer + 近期/质量数据研究」交给 AI Coding Agent 分 PR 执行。它不是已集成的模型、不是已经测量的 GPU 性能，也不包含原始牌谱/训练权重。

**阅读顺序**：

1. `00_MASTER_PLAN.md` — 总规范：仓库实况、架构、信息可见性、数据策略、权限、训练与研究边界。
2. `01_TASK_CARDS.md` — **PR-00 至 PR-30：31 张具体实施卡**，每张写依赖、文件、实现步骤、测试与验收。
3. `02_IMPLEMENTATION_RECIPES.md` — 伪代码、Rust 对齐、数据格式、padding/causal、记忆梯度、统计注意事项。
4. `03_AGENT_HANDOFF_PROMPT.md` — **直接复制**的首个 Agent 提示词、续跑提示词和独立代码审查提示词。
5. `04_ACCEPTANCE_MATRIX.md` — Gate、安全测试、预注册及报告模板。
6. `configs/*.yaml` — 未来 `rf_*` CLI 的建议配置 schema（尚未由程序解析，所有数值需实测验证）。

**正确的启动**：把本压缩包解到 repo 外或 `docs/riichiformer/handoff/`，复制 `03_AGENT_HANDOFF_PROMPT.md` 中的 Prompt A 给 Agent，**只要求执行 PR-00**。Agent 的每个 PR 必须自己产生真实测试和证据，然后你审核通过才继续下一张卡。早期 PR 主要围绕审计与 POV 安全；到 PR-13/14 才开始 BC 训练，PR-19 才涉及近期数据源补充，PR-29 才允许 RL。

### 为什么不是直接让 Agent 一口气完成

- 数据是历史牌谱和未来终局标签的混合来源，泄漏很容易发生而且模型照样能“学好”。
- `libriichi` 的 46 维动作 mask、抢断反应、赤牌和暗杠特例都很容易在重新编码时产生偏差。
- 单张 16GB GPU 的训练预算必须先由 `nvidia-smi + torch` benchmark 得出，而不能拍脑袋决定 batch/时间。
- 近期牌谱存在未知来源/许可、质量和可能 AI 混入，不能先下载混在一起再解释结果。
- 架构、数据质量和强化学习三条轴都变化时，即使对局改善也无法归因。

### 文档生成状态

- **Verified by source inspection**：原仓库文件结构、Mortal pinned source `mjai/event.rs`、`dataset/gameplay.rs`、`state/player_state.rs`、`engine.py` 与当前 scripts。详情见 `00_MASTER_PLAN.md`。
- **Not executed**：任何 V2 新 CLI、Rust/PyO3 binding 修改、5070 Ti GPU benchmark、模型训练、数据质量实验、对局实测。文档中 `rf_*` 命令是目标接口。
- **Prior prototype**：`RiichiFormer-S-v0.1` skeleton 仅作参考，和此包无运行时依赖。

### 推荐的第一个交付物

`docs/riichiformer/BASELINE_LOCK.md` + `artifacts/riichiformer/preflight.json` + `docs/riichiformer/pr_reports/PR-00.md`，其余训练先不开始。
