# Agent 验收矩阵与研究结果填表

## 1. Gate 级总览

| Gate | 检验主题 | 应有证据 | 错误处理 | 负责人 |
|---|---|---|---|---|
| G0 | 原始基线冻结 | SHA、旧 eval/patch 注册、preflight JSON | 缺失记录 BLOCKED，不生成假权重 | PR-00/01 |
| G1 | 数据/来源/切分 | Audit + replay + rights + split hashes | 违规数据退出训练，不自动下载 | PR-02..07 |
| G2 | POV 和 action parity | 10 个硬测试、100 game × 4 POV replay parity | 任何泄漏 P0 阻塞 | PR-08..12 |
| G3 | MVP-S 真机可训练 | Loss、resume、合法行动、arena smoke、VRAM/tps | No-go 时不得扩大模型 | PR-13..17 |
| G4 | 数据质量/近期/教师配方 | 同预算 arm + 配对CI + 训练样本统计 | 不显著则 INCONCLUSIVE | PR-18..21 |
| G5 | 分层 vs Flat 长历史 | 统一 data、params/compute、实战效应、p95 | 无增益则回退当前局 | PR-22..26 |
| G6 | Belief/RL | 校准、behavior logprob 复算、外部对手确认 | 早停/回滚冻结 BC | PR-27..29 |
| FINAL | 最终交付 | 新独立 confirm pool、model card、复现实验 | 不能推断线上段位 | PR-30 |

## 2. 不容绕过的 P0

- [ ] 不同对手隐藏摸牌、起手在同公共前缀下，POV token 与模型预测保持不变。
- [ ] 所有未来终局标签、未来事件不可能进入历史决策的 input；历史记忆只来自完成的局。
- [ ] `46` 维合法动作和原生 Rust 引擎对齐，输出 action 必合法；censored 反应不伪造 Pass。
- [ ] 训练/验证/选择/确认按完整 game_id 和 duplicate 来源隔离，不把未见标签用于训练。
- [ ] 数据权限按来源核验；不公开玩家身份，合法来源不明确时不继续新的采集和训练。
- [ ] 不规避平台 AI 限制，不以在线天凤自动挂机/封禁规避为评测路径。

## 3. 漏洞与回归测试的最小覆盖

| Test name | 输入变更 | 预期 |
|---|---|---|
| `test_opponent_tsumo_redacted` | 替换对手摸牌牌种/赤 | token 完全相同 |
| `test_self_tsumo_visible` | 替换自己摸牌 | token 有变化 |
| `test_haipai_redacted` | 替换其他三家的 13 张 | token 完全相同 |
| `test_future_causal` | 修改未来动作/结束结果 | 当前 decision logits 不变 |
| `test_memory_fence` | 修改第 k+1 局 | 第 k 局 decision 不变 |
| `test_legal_index_parity` | 对比 Rust GameplayLoader | 46 mask/action 完全对齐 |
| `test_censored_pass` | 被抢断抢碰/和窗口 | 标签无效而非 Pass |
| `test_online_offline_parity` | 相同 game/POV/前缀双路径 | token、mask、状态一致 |
| `test_checkpoint_schema_mismatch` | schema/action hash 变化 | 显式拒绝加载 |
| `test_padding_causality` | 改有效序列之外 padding | logits 不变 |
| `test_ppo_behavior_logp` | 混合探索策略 | logprob 与真实行为分布一致 |
| `test_reward_seat_rotation` | 固定墙四席轮换 | 排名得分/方向一致 |

## 4. 统计研究预注册模板

```yaml
experiment_id: RF-D1-001
registered_at_utc: <timestamp-before-training>
hypothesis: "top-quality seat-game labels improve actual strength at equal budget"
base_git_sha: <hash>
model_config_hash: <hash>
dataset_manifest_hash: <hash>
split_registry_hash: <hash>
sample_unit: seat_game
primary_metric: paired_delta_pt_per_hanchan
secondary_metrics:
  - avg_rank
  - deal_in_rate
  - riichi_rate
training_budget:
  labeled_decisions: <integer>
  optimizer_steps: <integer>
  compute_cap_gpu_hours: <float>
  min_unique_seat_games: <integer>
model_seeds: [<two-or-more-actual-fixed-seeds>]
arena:
  opponent: frozen_mortal_checkpoint_sha
  seed_key: <new seed key>
  select_range: <pre-registered and not already consumed>
  confirm_range: <virgin and held inaccessible until candidate frozen>
  seat_rotations: 4
  bootstrap_unit: wall_seed_set
  bootstrap_resamples: 2000
  no-peeking: true
selection_rule: <explicit minimal-interest + CI rule>
early_stopping_rule: <explicit catastrophe guard>
```

**重要**：只有确认之前登记的 `confirm_range` 且没有被任何候选选择消耗，才属于 confirm。Old baseline `A14` 的 `[70000,73000)` 已经消费。预算数值应结合 PR-01、PR-17 的真实 GPU 性能和模型对模型的配对方差设定，不可在看过正式效果后再改。

## 5. 每次训练必留日志字段

`run_id, utc_start, git_sha, all config hashes, data manifest hash, schema hash, action hash, split hash, source rights snapshot, torch/cuda/gpu, trainable params, effective decisions, source mix, unique games, unique players (private aggregate), sampler seed, tokenizer length histogram, lr, optimizer step, BC NLL, masked illegal rate, val NLL, checkpoint hash, peak VRAM, tokens/s, decision/s, status, exit_code`。

## 6. 每次评测必留日志字段

`candidate SHA, frozen opponent SHA, arena/ruleset version, seed_key, wall ranges, complete sets, incomplete sets, placements 1..4, pt/rank, paired delta, CI method/unit, source/old sample overlap, p50/p95 inference latency, battle failure count, source license assumptions`。

## 7. 结果结论词汇的固定含义

- `PASS_CORRECTNESS`: 软件正确、安全和可复现，**不代表强度**。
- `POSITIVE`: 有独立确认的统计+实际意义收益，且未违反预注册。
- `NO_DETECTABLE_EFFECT`: 按指定样本和精度未观察到可检出差异，**不等于证明零效果**。
- `INCONCLUSIVE`: CI/统计功效不足，或资源/数据限制使比较无法定性。
- `NEGATIVE`: 确认性检测到退步、超过预先设定的风险阈值。
- `BLOCKED`: 许可/接口/数据缺失或重要测试失败；禁止发布训练结果。

## 8. 系统规模的层级验收（不应混为一谈）

1. **数据工程成功**：4 POV 合法、零泄漏、复现一致。
2. **模型工程成功**：本地单卡可以学、可以推理、可以完整打完合法对局。
3. **研究假设通过**：跨局记忆/数据筛选有确认性收益。
4. **麻将实力通过**：对独立强对手的最终对局强度不低于冻结 Mortal 并有 CI 支持。
5. **真实平台段位**：另一个外部目标，不通过自动进入受限真人天凤桌测试；需要遵守平台规则与授权。
