# PR-08 / PR-09 接口设计 brief（Lead → Schema Owner）

**状态**：DRAFT，不是冻结 schema；由 Schema Owner 在 Wave 2 实现并以 `schema_v3.json` 的 hash 冻结。
**依据**：V3.1 pack `01_FULL_ITERATION_ROADMAP.md`（PR-08/09）、旧 V2 task card PR-08/09、H-01/H-04 的待验收结论。
**原则**：先信息契约，后网络复杂度；任何 target/oracle/未来信息都不能进入 forward input。

## 1. 标识与编码

- `TileKind = 0..33`：`1m..9m, 1p..9p, 1s..9s, E,S,W,N,P,F,C`；`UNKNOWN = 34`。
- 赤五不占独立 kind：`aka ∈ {0,1,2}` 只对 5m/5p/5s 有效；普通五与赤五必须可区分。
- Tenhou/libriichi 内部 `34..36` 是赤五 id、`37` 是 `?`；schema 层必须显式降为 `(kind, aka)`，不得把 37 当普通 kind 透传。
- `ActorRel`：`0=self, 1=shimo, 2=toimen, 3=kami, 4=none/unknown`。
- `Phase`：`SELF_TURN`, `CALL_RESPONSE`, `KAN_SELECTION`。
- 动作 46：`0..33` 通常弃牌 kind、`34..36` 赤五弃牌、`37` riichi、`38..40` chi low/mid/high、`41` pon、`42` kan、`43` agari、`44` ryukyoku、`45` pass。`KAN_SELECTION` 时 `0..36` 解释为杠牌种类（含赤 id），不是弃牌；必须与 phase 一起解码。最终以 `libriichi` binding 的 legal mask 与 fixture 为准。

## 2. 事件（ObservableEventV3）

每个事件是显式 dataclass/枚举分支，字段白名单；未知 event type **报错**，不得 fallthrough 原始 dict。

至少覆盖：`start_game`, `start_kyoku`, `tsumo`, `dahai`, `chi`, `pon`, `daiminkan`, `kakan`, `ankan`, `reach`, `reach_accepted`, `dora`, `hora`, `ryukyoku`, `end_kyoku`, `end_game`。

建议公共字段：`kind`, `seq_idx`, `kyoku_idx`, `actor_rel`, `target_rel`, `tile_kind`, `aka`, `consumed`（4 个 `(kind,aka)` 或 `UNKNOWN`）, `tsumogiri`, `phase`, `timing`。

硬约束：
- `start_kyoku` 只保存公共局况/分数/宝牌指示牌 + **自家** 13 张（`own_haipai` 专用字段）；其他三家手牌不得出现在对象里。
- 对手 `tsumo`：`tile_kind=UNKNOWN`, `aka=0`，只保留 actor/seq；自家 tsumo 保留 tile/aka。
- `dahai`/副露/立直/宝牌是公开事件，字段按实际可见时点保留；暗杠的公开细节由 PR-09 按规则决定，schema 不能假设全知 `consumed`。
- `hora`/`ryukyoku`/`end_*` 只能在事件发生后追加；`ura_markers` 只在和牌后可见，不能提前注入状态。
- 禁止字段：玩家姓名、账号、牌山 seed、未来动作、对手暗牌、未来里宝、`rank_final`。

## 3. 状态输入（RuleStateInput）

- `own_counts[34]`, `own_aka[3]`
- `public_counts[34]`, `public_aka[3]`：河 + 副露 + 已翻指示牌，同一物理牌只计一次；被吃碰的牌从河迁移到副露，不重复。
- `dora_indicators`：只含当前已翻出的指示牌；杠宝翻出时点由 H-04 结论决定。
- 四家公开状态：立直、副露、河牌数、分数、门风/自风；不包含对手手牌。
- 局况：bakaze/kyoku/honba/kyotaku/本场、剩余牌数、自风/场风。
- `ruleset_id` 与 `schema_version` 必须随样本保存。

不变量（测试必须覆盖）：
- `own_counts[t] >= own_aka[color(t)]`；赤牌数 ≤ ruleset 允许。
- `own_counts[t] + public_counts[t] <= 4`。
- 被副露消费的牌不能同时出现在 river 与 meld 的 public 计数中。

## 4. DecisionInput / DecisionTarget 隔离

`DecisionInput`：`events`, `tile_state`, `players_public`, `global_public`, `legal_mask[46]`, `phase`, `before_event_idx`。

`DecisionTarget`：`expert_action: int|None`, `supervision_valid: bool`, `censor_reason: str|None`, `rank_final: int|None`, `oracle_labels: dict|None`。

- 两个类型放在不同模块（例如 `decision_v3.py` 内两个类可以，但 runtime 只 import input；更安全是 target 单独模块）。
- `forward()` 只接受 input；训练器从 batch 分开取；在线推理不得 import target 模块。
- 所有有效 label 必须在 `legal_mask` 内；被抢断的 pass 是 `censored`，不计监督。
- `DecisionInput` 不得含 `rank_final`/`oracle_labels`/未来事件。

## 5. 版本与 hash

- `configs/riichiformer/schema_v3.json`：规范化 JSON（sorted keys, UTF-8, no whitespace variance），包含 event/field/dtype/normalizer/action map。
- 计算并冻结：
  - `schema_hash = sha256(canonical schema JSON)`
  - `action_map_hash = sha256(canonical action map)`
  - `normalizer_hash = sha256(canonical normalizer spec)`
- loader 在 shard/checkpoint 加载时三者任一不匹配 → 立即异常；不得自动迁移。
- `candidate_schema_version`, `belief_label_schema_version` 预留，但先不生产昂贵标签。

## 6. PR-09 POV 投影

- `project_event(raw, pov, context) -> ObservableEventV3`，逐类型 allowlist 构造新对象，不 deepcopy 后删字段。
- `encode_pov` 输出在“公共前缀相同、仅对手暗牌不同”的两种输入上必须 byte-equal；自家手牌变化必须改变；未来事件截断必须不改变过去输入。
- 旧 patch 0006 的 opponent `tsumo.pai` 泄漏要有专门回归测试。
- 未知 event type 报错；不得静默跳过。

## 7. 验收测试（最低）

- 每类 mjai event 的 roundtrip + unknown type fail-closed。
- 0..36 弃牌 vs KAN_SELECTION 双重语义；赤/普通五独立；chi 38..40 结构。
- schema/action/normalizer hash 与 checkpoint/shard 互认；篡改一个字节 → 加载失败。
- 四 POV 对称；隐藏手扰动 byte-equal；未来截断等价；里宝/杠宝时序。
- target 不可序列化进 forward input；runtime 包不 import target。
- 46 mask 与 `libriichi` 的 legal mask 在真实日志上一致；所有标签 ⊆ mask。

## 8. 待 H-01/H-04 回填的开放问题

1. `hora.deltas` 是否含供托（H-01 证据）——影响分数结算，不改变事件字段，但决定 quality 重算。
2. 暗杠公开哪些牌、杠宝翻出是即时还是延后（H-04）——决定 `ankan`/`dora` 事件的可见字段与时点。
3. 三家和了流局/双荣的终局事件顺序（H-04）——决定 PR-10 phase/censor 与 PR-32 多荣分支。
4. `start_kyoku` 的 `kyotaku` 与上一局结算不一致时以哪个为准（H-01）——schema 只记录，不修补。

## 9. H-04 规则反馈（2026-11-21，待并入 PR-08）

H-04 结论：pinned arena **不是 Tenhou 等价**，PR-16/PR-17 只能报告相对强度。

- `ruleset_id` 至少区分 `tenhou_4p_ranked`（真实牌谱标签）与 `mortal_arena_pinned`（本仓库 arena 行为）。
- **R02 三家和了**：arena 照常结算三家荣和，Tenhou 为流局。任何由 arena 生成/验证的样本必须带 `ruleset_deviation="triple_ron"`，训练 shard 默认排除或显式降权；不能把它当 Tenhou 标签。
- **R25 末局亲家和了止**：arena 的 30000 点判断包含刚支付的供托，Tenhou 要求扣除供托。终局 continuation/排名样本必须记录该偏差；不要用它推断 Tenhou 的终局。
- **R26 四家立直流局**：arena 在第四家立直后多摸一张才流局（日志层面差异，分数不变）；POV/事件序列必须允许该差异，不能假设与 Tenhou 完全同序。
- `DecisionInput`/shard 元数据应包含 `ruleset_id`, `ruleset_deviation[]`, `source_kind`（human log / arena self-play / synthetic），供 PR-12/PR-16 过滤。
- 在 R02/R25 修复或明确隔离前，任何“天凤等效九段”声明都被 H-04 阻断；模型可以训练，但只能称内部相对强度。
