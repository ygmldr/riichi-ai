# V3.3 M5D-R0 评测正式协议（可复制为每个 EXP-xxx/PREREG.md）

## A. 实验元数据

```
experiment_id: EXP-M5D-<unique>
status: PLANNED | RUNNING | SELECTED | CONFIRM_CONSUMED | INVALID
registered_before_execution_utc: <timestamp>
owner: Lead
candidate_model_sha256: <sha>
candidate_config_sha256: <sha>
anchor_id: M5D-R0
anchor_checkpoint_sha256: <sha>
anchor_driver_sha256: <sha>
engine_version: <git_sha>
engine_binary_sha256: <sha>
ruleset: tenhou_approx
rule_deviation_list: <Rxx>
scorer_sha256: <sha>
pt_profile_primary: TOKUJOU_5D_HANCHAN
pt_profile_secondary: HOUOU_7D_HANCHAN
arena_mode: one_candidate_vs_three_same_frozen_anchor
model_inference: deterministic / sample (exactly defined)
wall_seed_key: <new_hex_key>
wall_seed_window: [<start>,<stop>)
seed_pool_role: smoke|trend|select|confirm
seed_pool_registry_entry_sha: <sha>
seats: [0,1,2,3]
blocks: one_complete_seed_x_four_rotations
max_retries_per_game: 1_or_2_fixed_before_execution
invalid_game_policy: fail_set_and_investigate
expected_complete_sets: <N>
bootstrap_seed: <fixed_value>
bootstrap_resamples: 5000
primary_metric: delta_pt_5d_per_hanchan
secondary_metrics: [raw_pt_5d, pt_7d, avg_rank, rank1_rate, rank4_rate]
selection_rule: <frozen_before_result>
predeclared_effect_size: <e.g. +3 pt/half_game, engineering target not fact>
max_gpu_wall_budget: <approved>
max_cpu_wall_budget: <approved>
stop_conditions: [illegal_action, hidden_info_leak, rule_crash, mismatch_SHA, missing_seed_set, OOM, thermal_limit, user_stop]
release_policy: local_results_only
```

## B. 统计与检验过程

1. 汇总每完整 `seed_key+seed` 的 4 场，以各个 seat 的候选分数计算 `pt5`,`pt7`,`delta_pt5`,`delta_pt7`,`avg_rank`,`last_rate`。
2. 主估计量是 set average；bootstrap 必须重采样 set，而不是重采样游戏；输出双侧 95% CI；附加 1～4 位概率的 Wilson 或 set bootstrap CI。
3. 若只有 20 set，用作 plumbing smoke，**不**做强度显著性声明；Trend 只估方差，Select 可用于选模型；Confirm 在候选及分析方案冻结后才执行。
4. `delta_pt5` 与锚点对局中的相对 ΔPT 只表示该 1v3 分布；正值与 5D 真人段位无直接映射。
5. 预注册最小有意义效应量建议暂设 `+3 pt/半庄` 作为工程阈值，确认之前用 trend 的实测方差再调整和冻结；如果只是 B0 baseline，则即使负值也作为未来实验参考，不强行设“B0 必须赢”。
6. 强度晋升条件建议：同一独立 confirm 池只比较**预先选出的一个候选 vs 冻结的上一代 champion**；主要 paired ΔPT 的 CI 下界 > 0 且无合法性/故障回归，达到工程阈值再宣布“显著且有实际价值”；如果选择多候选则必须换未消费 confirm 或采用预注册多重比较校正。
7. 允许有限的 pilot 中断，记录原因与 last completed set；不能挑选好的半局/report；若中断且无冻结停止策略，标 `INCOMPLETE/EXPLORATORY`。

## C. 数据输出契约

每场 `game` 的最小日志字段：

```
run_id, model_sha, anchor_sha, engine_sha, rules_hash, scorer_hash,
seed_key, wall_seed, seat_rotation, candidate_seat, point_scores[4], ranks[4],
pt5_candidate, pt5_mortal_seats[3], pt7_candidate, delta_pt5, delta_pt7,
wall_trace_hash, event_trace_hash, illegal_actions, fallback_actions,
exception_class, rules_deviation, completed, elapsed_ms
```

每组 `seed_set`：

```
seed_key, wall_seed, four_game_hashes[4], completed_4_of_4,
set_avg_rank, set_raw_pt5, set_delta_pt5, set_delta_pt7, last_rate,
errors, retry_count
```

每次 `report`：

```
experiment_id, prereg_hash, baseline_registry_hash, n_complete_sets,
n_games, failed_games, avg_rank, rank_hist, raw_pt5, delta_pt5,
95ci_delta_pt5, raw_pt7, delta_pt7, 95ci_delta_pt7,
per_seat_table, cpu_s, gpu_s, full_source_hashes, NOT_RUN, status
```

压缩原始牌谱按内容 hash 审计；公布报告时去除真实玩家标识和任何无发布许可内容。

## D. 种子池/数据泄漏审计

- `seed_pool_role=confirm` 的新 `seed_key` 必须在首次计算前完成全库使用历史审计并冻结，不能凭“新文件夹”就认定未消费。
- 评估墙 RNG vs selfplay 训练 game RNG vs 训练参数 RNG 三套独立；如需跨模型配对使用同一个 select pool 可以，但已经不是 confirm。
- 生成新池使用本地安全随机源与独立登记，而不是公开示例中的硬编码 seed；历史 `runs/1v3` 与已消费 pool 严格只读。

## E. 必需的负控

- 人工 `rank=(1,2,3,4)`：四席 PT 总和恰好 0；候选在某席每种名次一次：`pt5=0`, `delta_pt5=0`；
- 候选四场均第一：每场 raw `+75`, delta `+100`；候选四场均第四：每场 raw `-105`, delta `-140`；
- `M5D-R0` 自对照：完全相同 checkpoint、固定 wall/rng 下四席结果不应系统偏离 0；
- 接口丢失一场/文件名不一致/写重复一场/seed key 不一致/出现 fallback：按 P0 fail；
- old 7D scorer 对已归档 1v3 gate 的 `(90,45,0,-135)` 输出必须与只读历史结果数值吻合；
- Replay 的最终分数与 engine 完成事件一致，发生 R29 特定一发场景必须标 deviation，否则无法无条件声称规则完全等价。
