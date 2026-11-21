# V3.3 Wave 2/3 并行工单与交付契约

## 1. 角色、锁与目录

- Lead（唯一集成 writer）：`docs/riichiformer/v3_3/STATUS.md`, `INTEGRATION_LOG.md`, `EVAL_POOL_REGISTRY.md`, `resource_locks.yaml`、release tags、GPU 与重 IO 权限。
- Dev-A Arena（PR-46）：MortalCheckpointAdapter、1v3 Duplicate Arena 桥接、rank/PT scorer、M5D anchor registry。独占 `riichiformer/eval/mortal_anchor.py`, `riichiformer/eval/relative_pt.py`, `scripts/rf_eval_m5d.py`。
- Dev-B Data/Validation（PR-45/PR-47）：独立 selfplay validation、split/seed/game hash、防泄漏、Selfplay→POV→Shard→Loader 端到端；独占新 `riichiformer/validation/**`、`tests/rf_test_validation_*.py`。如需修改 `data/shard.py`/`loader.py`，先与 Lead 协调与 Dev-A/C 串行。
- Dev-C Train/Model（PR-48）：B0 config、multi-decision 训练数值复核、配置/metadata Checkpoint 生命周期、val/test 度量和 model selection；独占 `riichiformer/train/**`、`scripts/rf_train_b0.py`，不可修改 Dev-A 的 evaluator。
- Dev-D Rules/QA（PR-40R）：R29/0014 native fixture/rebuild、DuplicateArena vs Mortal-native 规则差异清单、构建/动作/模型回放性能审计；独占 `tests/rf_test_native_rules_*.py` 和独立补丁草稿，不直接改 dirty vendor。
- Reviewer：跨 lane 只读审计，不改 Dev 的实现，不当 reviewer 报告为运行事实；由 Lead 收集独立日志。

**共同基线**：Lead 先验证 `00e6576` 实际 HEAD、`rf3.2-w1`、Reviewer 结果、修复后新 W2 tag。所有 worktree 从同一 `BASE_SHA` 创建。若本地分支未推送，先在用户本地保全并确认远程状态，不应从 `a27df099` 的旧公开 HEAD 覆盖 `00e6576`。

## 2. Wave 2：并发 4 lanes

### Dev-A / PR-46A：冻结 Mortal-5D 与 Adapter

交付：`baseline_registry.yaml`、Mortal-5D checkpoint `sha256`、native driver/config/patch hash、4×seat 同权重读取一致性、active local rules version。验证 `MortalAdapter` 的合法动作来源、call/ron/kan/riichi/aka/forced/censored 一致；和 RiichiFormer Agent 交替调用无状态串扰。失败不允许自动使用随机动作掩盖；trace 必须可诊断。

**验收**：`test_5d_checkpoint_identity`, `test_native_action_roundtrip`, `test_4seat_rotation`, `test_no_hidden_tile_access`, `test_riichi_kan`, `test_repeated_seed_determinism`, `test_anchor_self_play_is_balanced`, `test_no_old_frozen_run_write`；每个测试真实 exit=0，保存 4～20 场样本压缩日志与 hashes。`M5D-R0` 只能在用户确认具体权重后冻结。

### Dev-A / PR-46B：Relative PT Scorer + Arena

写清楚 `(75,30,0,-105)` 与 `(90,45,0,-135)`。旧 A14 scorer 可复用计算思路但别引用其固定 3000-set/旧 seed_KEY gate。输出每 set 的 `ranks/pts/seat_rotations` 和 bootstrap；缺 set 必须 fail，SEED_POOL 不明拒绝标 confirm。检查同分排名和 1v3 零和；5D selfplay control near 0。

**验收**：手造排名向量、全顺位平均、tie、坏分数、错哈希、缺 1 场、重复 seed、同 seed 不同墙、非固定 anchor、repeat/confirm 泄漏等负控；所有 metrics 与老 `eval_report` 在对应旧 7D 模式一致（只读 fixture）。

### Dev-B / PR-45（独立验证与离线诊断）+ PR-47（Selfplay E2E）

生成与 B0 train 不重叠的 100～200 局，四 POV、保持 `selfplay_own` 来源与训练/验证不同的 split 语义。生产 Validation 不应因 `--research-policy` 覆盖分区：train loader 不得打开 val shard 进行优化；val reader 若当前只支持 audit/diagnostic，需增显式 read-only eval mode，避免把 val 伪标 train。

**验收**：500 training games / validation 之间 hash/seed/game/distribution intersection=0；人工注入标签、未来事件、隐藏牌必须触发数据契约异常；同序列多决策与 prefix 对齐；结算、全场 score conservation、copy-on-write。自博弈 weak rule bots 不提供“高段位真值”。

### Dev-C / PR-48：B0 正式训练、选模方案

补数据加载与训练模式显式 pin；两个 seed 的 config fingerprint、AdamW/BF16/EMA、gradient accumulation、strict resume 的 provenance、每 250/500 step valid CE；B0 pilot 的 checkpoint load、未完成 6191-step 不要误用。训练任何 epoch 前生成 `PREREG.md` 并明确预计 GPU 秒/显存/IO/最大成本与 kill 条件。

**验收**：同 seed 重跑 10-step 权重一致、同参数不同 seed 不相同、大 batch 与 microbatch 梯度一致、resume 与 uninterrupted 一致；验证数据不回传 optimizer；val CE 仅作为候选模型筛选；trainer 不在当前阶段调用 12k 场 arena。

### Dev-D / PR-40R：Rules / runtime

PR-40 R02/R25/R26 据用户报告已修；只核对原 vendor+0014 真实重建、fixed fixtures 与 version fingerprint，不要重做。R29 仍 UNVERIFIED：补顺序事件 fixture、明确 `KNOWN_DIFFERENCE` 或 `MATCH`，不允许“文档上勾绿”。Arena rule scope 继续标 `tenhou_approx`。

**验收**：dirty vendor hash/patch manifest、一发、三家和了、末局供托、四家立直、三人荣和 score invariants、国产 Rust native rebuild binary sha 与 Python binding 一致性报告；修改 vendor 需先 Lead 批准和备份。

## 3. Lead 串行集成顺序

1. Reviewer 完成现有 V3.2 W2 验收 → 修 P0/P1 → `rf3.2-w2`（否则冻结阻塞）；
2. 合 PR-46A/46B 的评分合同（本阶段禁正式 GPU）；
3. 合 PR-47 validation 与 PR-48 Trainer 的新 read-only eval 接口（由单 writer 解决 shared loader）；
4. 合 PR-40R native evidence；
5. 全套 `pytest` + 100% 确定性 replay + pycompile + file hash + seed pool audit；
6. 在用户许可下仅运行 `5～25 seed` smoke（20～100 局），评测耗时/错误率；
7. B0 pilot 验证/选模→ Trend →正式 Train 的单独请求许可→独立 Confirm；
8. 输出 `G-V33-A0/D0/B0/C0` 分开判定，不把一条绿链伪作全部 pass。

## 4. 所有子代理的提交格式

```
TASK_ID
BASE_SHA
WORKTREE
OWNED_FILES
COMMIT_SHA
INPUT_HASHES / OUTPUT_HASHES
TEST_COMMANDS + EXIT_CODE + DURATION
RUNTIME_RULES_HASH / MODEL_CHECKPOINT_HASH
P0 / P1 / P2 + REPRODUCTION
NOT_RUN
GPU_LOCK_USED? yes/no, actual duration
IO_LOCK_USED? yes/no, actual bytes
DO_NOT_CLAIM
REVIEWER_VERDICT
NEXT_DEPENDENCY
```

提交和报告只允许声称真正运行过的操作；`NOT_RUN` 不得作为 `PASS`。Reviewer 尤其要审计是否有传统 prefix 静默变成默认、是否使用历史已消费墙池以及“把弱 imitation loss 当实力”的论证。

## 5. GPU / IO 排队规则

- 当前按用户要求不启动训练。Lead 可读日志、整理 PR、跑轻量测试。
- 一旦用户批准某个 bounded task，所有 GPU training/arena 的任务进入同一 FIFO，不能训练和大评测同时占卡。
- 首个 GPU job 只允许 tiny 2-step smoke 或小批 CPU/IO 并行，确认风扇和温度控制后再升级；夜间按原策略限制 Rust/C++ 并行 worker（例如 CARGO_BUILD_JOBS=4），禁止禁用温控。
- 每项耗时在本机测量后制定预算，不在 PREREG 中凭主观经验编造 ETA。
