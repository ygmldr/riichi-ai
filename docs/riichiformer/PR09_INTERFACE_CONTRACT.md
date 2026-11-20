# PR-09 接口契约（当前实现 vs 必需契约）

**状态说明**：本文件区分两类内容：

- **[IMPLEMENTED]**：当前仓库真实存在、有测试/产物的接口；
- **[MISSING]**：PR-08/09/10/11 尚未实现的目标契约（只作设计约束，**不是**已实现事实）。

审计基线 HEAD：`aceb281602fb2703a1ed2067a006992cd9ddb680`。  
本文件不实现任何模型改动。

---

## 1. 当前已实现的数据接口 [IMPLEMENTED]

### 1.1 审计 inventory：`source_audit.parquet`

一行 = 一个发现的文件或 zip 容器（gitignored artifact，PR-02）：

| 列 | 类型 | 含义 |
|---|---|---|
| `root` | string | 本地根路径 |
| `relpath` | string | 相对路径（Tenhou log 名） |
| `year` | int16 | 2009–2026；由显式年份目录/文件名前缀推出 |
| `source_id` | string | `tenhou_old` / `tenhou_recent` |
| `kind` | string | `leaf` / `zip` |
| `compression` | string | `gzip` / `json` / `zip` / `empty` / `unknown`（按 magic） |
| `bytes` | int64 | 未压缩/容器字节 |
| `mtime_utc` | string | 本地文件 mtime（不是对局时间） |
| `sha256` | string/null | 仅抽样/显式 hash 模式填充 |
| `member_count` | int64/null | zip 容器成员数 |
| `invalid_reason` | string/null | 空文件/未知 magic/zip CRC 等 |

### 1.2 canonical index：`canonical_log_index.parquet`（sample only）

一行 = 一场 game（PR-03，1,000 场 sample；`full_corpus=false`）：

| 列 | 类型 | 含义 |
|---|---|---|
| `game_id` | string | `tenhou:<log id>`，否则 `sha256:<canonical>` |
| `source_id` / `year` / `relpath` / `compression` | string/int | provenance |
| `original_sha256` | string | 原始字节 SHA |
| `canonical_sha256` | string | 规范化事件文本 SHA |
| `platform_game_key` | string/null | Tenhou log id |
| `events` / `kyoku_count` | int | 事件数 / 局数 |
| `duplicate_group` / `duplicate_count` / `is_representative` | string/int/bool | 去重 |
| `collision_review` / `cross_key_duplicate` | bool | 冲突/cross-key 审查标记 |
| `invalid_reason` | string/null | 拒绝原因 |

**注意**：这是**元数据索引**，不保存完整事件流，也不是模型输入。

### 1.3 native replay：`replay_game()` [IMPLEMENTED]

`riichiformer/data/replay.py:92`，返回：

```text
{
  ok, error_kind, error, event_count, kyoku_count, final_scores,
  seats: [{seat, player_id, decisions, invalid_actions, mask_shape_bad,
           min_legal_count, max_legal_count}],
  masks: optional [seat][decision][46 bool]
}
```

`actions/masks` 来自 `GameplayLoader`（`replay.py:118-119`）；`final_scores` 来自公开事件
（`replay.py:40-70`）。**这是 label/oracle API，不是 observable input**；没有 POV projection、
决策快照或 censor 标记。

### 1.4 split manifest：`split_manifest_L/R.parquet` [IMPLEMENTED]

一行 = 一个 `game_id`（PR-05）：

```text
split_version, protocol, assignment_mode, game_id, source_id, year, month,
split, relpath, root, blocked_source, rights_status, train_allowed,
previously_used_in_research, undated, holdout, duplicate_group_size, reason
```

已冻结 hash：L `184f34dc…`、R `4ada926b…`；所有行 `blocked_source=true`。
当前 `duplicate_group_size` 仅基于 Tenhou log id（全量 canonical/SHA 去重未做）。

### 1.5 quality rows：`quality_scores.parquet` [IMPLEMENTED]

一行 = 一个 seat-game（PR-06）：

```text
game_id, seat, player_hmac, match_time, year, source_id,
n_prior_games, mean_rank_prior, q, confidence, insufficient_history,
temporal, target_rank_for_eval, is_last_appearance, top25_seat
```

只有 HMAC 标识；`target_rank_for_eval` 明确是 eval-only。可靠性 `INCONCLUSIVE`，
不能作为已认证 label。

---

## 2. 必需但缺失的模型输入契约 [MISSING]

### 2.1 `ObservableEventV2`

建议字段（来自 `00_MASTER_PLAN.md` §4.2；**当前不存在**）：

| 字段 | dtype | 值域 | 可见性 |
|---|---|---|---|
| `kind` | enum | 20 个 EventKind；`PAD=0` | public |
| `actor_rel` | int8 | 0=self, 1=next, 2=opposite, 3=prev, 4=none | public |
| `target_rel` | int8 | 同上 | public |
| `tile` | int8 | 0..33 牌种；34=SENTINEL（对手摸牌/未知） | public，但**实际 tile 仅自家或公开弃牌** |
| `aka` | bool | 赤 5 标志；对手摸牌恒 false | public/self-only 分支 |
| `consumed` | int8[4] | 公开副露构成；34=PAD | public |
| `consumed_aka` | bool[4] | 与 consumed 对齐 | public |
| `tsumogiri` | int8 | -1=NA, 0=false, 1=true | public（有记录时） |
| `seq_idx` | int32 | 全局单调，不因新局重置 | public |
| `kyoku_idx` | int16 | 0..K-1；不可静默清零 | public |

**信息可见性表**（设计约束；实现必须逐分支 allowlist）：

| 原始事件 | 可输出 | 禁止进入输入 |
|---|---|---|
| `start_game` | 座位映射/规则元数据 | `seed`、真实 `names` |
| `start_kyoku` | 场风、局数、本场、供托、亲家、点数、表宝牌、**仅自己** 13 张 | 另外三家 `tehais` |
| `tsumo` self | actor、tile、aka | 牌山/剩余墙 |
| `tsumo` opponent | actor、发生一次；tile=34、aka=false | `pai`、隐藏数量 |
| `dahai` | actor、tile、aka、tsumogiri（可见时） | 后续隐藏摸牌 |
| `chi/pon/kan` | 公开副露构成/来源/aka/开闭 | 不公开的牌身份/未来 |
| `dora` | 已翻出的指示牌 | 未翻出的杠宝牌/里宝牌 |
| `reach/reach_accepted` | 宣言、已确认供托 | 对手听牌真值 |
| `hora/ryukyoku` | 结算后公开的得点/翻牌 | 回填到之前决策 |
| `end_kyoku/end_game` | 完成/结束 | 回填当前局或下局 |

### 2.2 `DecisionSnapshotV2`（PR-10，MISSING）

必需字段：`game_id`, `pov`, `decision_id`, `before_event_idx`, `kyoku_idx`,
`own_hand_counts[34]`, `own_aka_counts[3]`, `visible_tiles_counts[34]`,
`scores_relative[4]`, `bakaze`, `kyoku_number`, `honba`, `kyotaku`, `dealer_relative`,
`turn_count`, `riichi_states[4]`, `own_furiten`, `dora_markers`, `legal_mask[46]`,
`executed_action`（**label，不在输入 projection**）, `supervision_valid`, `censor_reason`。

当前不存在；因此“决策输入不含目标/未来”**无法认证**。在实现前，训练不得开始。

### 2.3 46 动作映射（PR-08，MISSING；仅 Rust 运行时存在）

从 `vendor/mortal/libriichi/src/state/obs_repr.rs` 与 `consts.rs` 确认的映射（应冻结为
`schema_v2.json` 的 `action_map` 并生成 `action_map_hash`）：

| id | 含义 | 证据 |
|---:|---|---|
| 0..33 | 普通牌种弃牌（m/p/s 1-9 + 东南西北白发中） | `obs_repr.rs:446` `mask[t]` |
| 34..36 | 赤 5m/5p/5s 弃牌（tile id 34/35/36） | `tile.rs` 34-36 + `mask[t]` |
| 37 | 立直 | `obs_repr.rs:481` |
| 38 | 吃（low） | `obs_repr.rs:489` |
| 39 | 吃（mid） | `obs_repr.rs:495` |
| 40 | 吃（high） | `obs_repr.rs:501` |
| 41 | 碰 | `obs_repr.rs:509` |
| 42 | 杠（大明杠/暗杠/加杠的统一决策位） | `obs_repr.rs:517/530/543` |
| 43 | 和了（自摸/荣和） | `obs_repr.rs:551` |
| 44 | 流局/九种九牌 | `obs_repr.rs:559` |
| 45 | Pass | `obs_repr.rs:424`（`ACTION_SPACE-1`） |

注意：杠选择模式（`at_kan_select`）把候选牌放在 **tile id 0..36** 的 mask 位，而不是 42；
PR-10 必须单独处理该模式，否则会把“选择哪张杠”误读成弃牌。当前没有实现/测试。

### 2.4 标签与非法动作

必需：

- `executed_action` 必须是整数 action id；训练 sample 必须 `legal_mask[executed_action]==True`；
- 被更高优先级和/碰/杠抢断的反应窗口：`supervision_valid=false`, `censor_reason=preempted_or_ambiguous`，
  **不得**写成 Pass；
- 单合法动作可从 BC loss 排除但保留 trajectory；
- `target_rank`、终局得分、`ura_markers` 只能是 label/target 结构，不得进入 input。

### 2.5 版本/哈希 fail-closed

checkpoint/shard 必须至少带：

```text
schema_version, schema_hash, action_map_hash, normalizer_hash,
obs_version=4, config_hash, git_sha, split_registry_hash
```

加载时任何一项不匹配 → 异常拒绝，不得自动迁移。当前没有 checkpoint/shard loader。

---

## 3. 当前接口能否支持目标架构（Task 3）

| 能力 | 当前 | 缺口 |
|---|---|---|
| Local Causal Transformer | 否 | 无 observable token 序列/位置/attention mask |
| 结构化 State Encoder | 否 | 无 DecisionSnapshot；raw `obs` 是 1012×34 native oracle obs，不是 POV-safe 结构 |
| 46 action | 仅运行时 | 无冻结 action map/哈希；mask 可用但未持久化 |
| Candidate policy | 否 | 无候选动作特征/负样本；mask 只有 legal/illegal |
| Belief（听牌/待牌/风险） | 否 | 无标签；quality 不是 belief |
| Cross-kyoku memory | 否 | 无 summary/memory fence |

**结论**：现有接口**不能**直接支撑任一 V2 模型实验；PR-08/09/10/11/12 是必需前置。

---

## 4. 实施顺序约束（供 PR-08/09 使用）

1. PR-08：先冻结 `schema_v2.json`（事件字段/可见性/动作映射/snapshot），生成 `schema_hash`、
   `action_map_hash`、`normalizer_hash`；测试至少每类 event fixture + action 46 对齐。
2. PR-09：严格 allowlist `project_event`，用 byte-equal 性质测试证明“同公共前缀、不同隐藏牌”输出相同；
   未知事件类型必须报错，不得 fallthrough raw dict。
3. PR-10：native `PlayerState` 推进 + 快照 + censor；只有通过 `test_legal_action_mapping` 与
   `test_censored_reaction` 的样本才进入训练。
4. PR-11/12：统一 offline/online encoder 与 shard；shard/checkpoint 带全部 hash。
5. 在这之前**不得**实现/训练 PR-13+。

> 本契约是**审计产物**，不实现 schema/POV；也**不**采用 v2.1 架构修改（见
> `PR09_CHANGE_IMPACT.md`）。
