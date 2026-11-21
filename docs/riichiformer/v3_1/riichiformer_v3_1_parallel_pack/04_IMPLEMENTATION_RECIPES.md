> 并行实施提示：接口/源码文件所有权、Waves 和资源锁以 `05_PARALLEL_EXECUTION_PLAN.md` 为准；PR-13 合成模型可提前编写，但真正的训练必须经过数据/许可门。

# RiichiFormer V3 — 关键实现契约、伪代码与验收实例

此文件用于 PR 卡实施时避免“描述正确，接口接不上”。**下面是方案伪代码，不是已提交仓库代码。** Rust 原生规则接口以当前 pinned `libriichi` 的实际可调用 API 为准，不得假设有不存在的 `get_ron_score()` 等方法。

## Recipe A：分数结算真值回归（H-01）

```python
# 示意：注意这不是完整的日麻打点/局末结算器
# 由对应数据源的真实 event deltas 协议决定是否需要额外处理 kyotaku。

def settle_one_kyoku(start_scores, start_kyotaku, events, protocol):
    scores = list(start_scores)
    kyotaku = start_kyotaku
    for ev in events:
        if ev['type'] == 'reach_accepted':
            scores[ev['actor']] -= 1000
            kyotaku += 1
        elif ev['type'] in ('hora', 'ryukyoku'):
            if ev.get('deltas') is None:
                raise ScoreContractError('missing deltas')
            scores = [x + y for x, y in zip(scores, ev['deltas'])]
            if ev['type'] == 'hora':
                # 当 deltas 已含供托（上游 Mortal arena）时只清零，不再次加给和牌者
                # 对数据源未知/不兼容的情况必须 fail closed，不能猜
                if protocol.hora_deltas_include_kyotaku is not True:
                    raise ScoreContractError('unverified hora delta convention')
                kyotaku = 0
    return scores, kyotaku

# 每一局结算后的结果应与下一局 start_kyoku.scores 交叉校验
# 最后一局若有未领供托，按经核验的 ruleset 另行结算
```

**应新增 fixtures**：`riichi_then_hora`, `riichi_then_ryukyoku`, `kyotaku_carried_next_kyoku`, `double_ron_order`, `last_kyoku_leftover_kyotaku`, `rank_tiebreak`, `empty_native_gameplay_rejected`；其中事件中的 `hora.deltas` 优先来自 native arena 或真实已复核牌谱，不能为凑预期任意编造。

## Recipe B：严格输入-标签隔离（PR-08～10）

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class VisibleTileState:
    own_counts: tuple[int, ...]        # 34
    own_aka: tuple[int, int, int]       # 5m,5p,5s
    public_counts: tuple[int, ...]     # 34，不重复物理牌
    public_aka: tuple[int, int, int]
    dora_indicators: tuple[int, ...]   # 仅当前已翻出的指示牌
    ruleset_id: str

@dataclass(frozen=True)
class DecisionInput:
    events: tuple  # POVEventV3, 仅历史及当前可见事件
    tile_state: VisibleTileState
    players_public: tuple             # 四家相对座位状态
    global_public: tuple
    legal_mask: tuple[bool, ...]      # 46
    phase: Literal['SELF_TURN','CALL_RESPONSE','KAN_SELECTION']
    before_event_idx: int

@dataclass(frozen=True)
class DecisionTarget:
    expert_action: int | None
    supervision_valid: bool
    censor_reason: str | None
    rank_final: int | None
    oracle_labels: dict | None
```

注意：`DecisionInput` 与 `DecisionTarget` 即使保存到同一 shard，也必须通过不同反序列化/`forward()` 接口读取，`forward` 不接受 target。训练器从 `batch.inputs` 与 `batch.targets` 分别取值；在线推理只有 inputs，根本不导入目标类/文件。

**关键不变量**：
- `own_counts[t] >= own_aka[color(t)]`，后者仅适用于 5m/p/s；
- `public_counts[t]` 只统计外部可见物理牌（河+副露+指示牌，依据规则定义），被吃碰迁移不得重复；
- `own_counts[t] + public_counts[t] <= 4`（物理牌种数量），赤牌总量不得超 ruleset；
- 不允许从 `next_event` 的实际动作或未来抽牌生成任何 `DecisionInput` 字段。

## Recipe C：事件 allowlist（PR-09）

```python
def project_event(raw, pov, revealed_context):
    typ = raw['type']
    if typ == 'start_kyoku':
        return ProjectedStartKyoku(
            scores=raw['scores'], bakaze=raw['bakaze'], kyoku=raw['kyoku'],
            dora_marker=raw['dora_marker'],
            own_tehai=tuple(raw['tehais'][pov]),
            # 三家他人手牌绝不保存；不是用原字段改几个 ?，而是不同类型
        )
    if typ == 'tsumo':
        if raw['actor'] == pov:
            return OwnDraw(actor=0, tile=raw['pai'])
        return OpponentDraw(actor=rel(raw['actor'], pov), tile=UNKNOWN)
    if typ == 'dahai':
        return PublicDiscard(actor=rel(raw['actor'],pov),
                             tile=raw['pai'], tsumogiri=raw['tsumogiri'])
    if typ in ('chi','pon','daiminkan','kakan','ankan'):
        return project_revealed_meld(raw, pov, revealed_context)  # 先核实暗杠公开细节
    if typ in ('reach','reach_accepted','dora', 'hora','ryukyoku',
               'end_kyoku','end_game','start_game'):
        return explicitly_project_each_field_by_event_type(raw, pov)
    raise UnknownEventType(typ)  # no fallthrough raw dictionary!
```

测试对任何 `pov`：只改变对手当前隐牌但保持公开 prefix，`hash(encode(input_before_t))` 不变；只改变当前尚未发生的事件，输入不变。若动作是已经发生的 `dahai`，它对应的牌身份和手切/摸切属于公开信息，可保留。

## Recipe D：决策边界与 BC 标注（PR-10～12）

```python
for game in permitted_canonical_games:
    for pov in range(4):
        native = PlayerState(pov)
        visible = POVProjector(pov)
        for event_idx, event in enumerate(game.events):
            visible_ev = visible.project(event)
            opportunity = native.update(event)  # 适配实际 Rust/PyO3 API
            if opportunity.can_act:
                input_ = build_input(native.public_view(), visible.events,
                                     native.legal_mask, phase=opportunity.phase,
                                     before_event_idx=event_idx)
                target = infer_action_from_protocol_with_censor(
                    game.events, event_idx, pov, opportunity
                )
                assert target.expert_action is None or (
                    0 <= target.expert_action < 46 and
                    input_.legal_mask[target.expert_action]
                )
                yield input_, target
```

**事件更新顺序是强约束**：此伪代码的 `native.update(event)` 是原生 PlayerState 接口的概念形态；具体根据是否需要先更新再决策及 mjai react 触发时间，必须与 `GameplayLoader` 的 `events.windows(4)` 行为逐一对齐。不能仅按 Python 循环 `next_event` 猜 Pass；抢断窗口必须设 censored。**训练 sample 的事件截点**是“可做决策、但还未发生所选动作”的精确时刻，在线 encoder 必须同源。

## Recipe E：最小模型的张量与掩码（PR-13）

```text
Input batch:
  event_type_id      [B,L]
  event_actor_rel     [B,L]
  event_tile_kind     [B,L]
  event_aka           [B,L]
  event_tsumogiri     [B,L]
  phase/pos/kyoku     [B,L]
  state_features      [N,F_state]       # 在 N 个决策位置对应
  decision_indices    [N,2]             # (batch_idx, token_idx)
  legal_masks         [N,46]
  labels               [N]              # 只在 target 类中

Encoder:
  E_event -> [B,L,d]
  E_event[decision_indices] += Project(StateMLP(state_features))
  CausalTransformer -> [B,L,d]
  select decision -> H [N,d]
  logits = Linear(H) -> [N,46]
  logits[~legal_masks] = -inf
  logprobs = log_softmax(logits)
  loss = masked_cross_entropy(logprobs, labels, supervision_valid)
```

**需要验证一个微妙点**：`state_features` 包含当前自家手牌，对局结束才可见的 `ura_markers` 等不能提前注入，且事件序列中存在多次 DECISION 时每次只注入对应时刻状态。第一版单局 `L` 动态，不能把超过 512 的事件静默截断。

```yaml
model:
  architecture: local_causal
  hidden: 256
  layers: 6
  heads: 4
  ffn: 704                # 示意；真实参数量以统计结果为准
  state_encoder: mlp
  policy_head: linear46
  memory: false
train:
  precision: bf16
  optimizer: adamw
  grad_accumulation: auto_by_valid_decisions
  activation_checkpointing: auto_on_memory_pressure
  max_sequence_policy: dynamic_bucket_no_silent_truncate
  source_rights_gate: required
```

Candidate Head **不能**让 `action_feat` 来自 `oracle`。它可以利用 `StateMLP` 编码的 per-tile 信息，也可用于 normal/aka 差异。对于 `KAN_SELECTION`，合法索引被解释为杠牌牌种，必须携带 `phase`，不能误用普通弃牌特征。

## Recipe F：Belief 标签 vs Risk 定义（PR-32/27）

```text
input  : POVEvent history and CURRENT observable state
oracle : true opponent hand AT THE SAME TIME t  [never input]
output :
  opponent_tenpai[3]
  opponent_waits[3,34]                     # 真实结构听牌/可胡候选的定义要明确
  ron_legality_if_discard[3,37]           # 当前弃哪个牌能被合法荣和
  point_transfer_if_ron[3,37,4]            # 四家得点变化，只有适用分支有效
  valid_oracle_label_mask[3,37]
```

`waits[tile]` 并不必然等于 `can_ron_now`: 无役、振听、局况限制都会造成差异。`would_ron` 不在完整信息中直接可观察，它需要对对手策略的假设；不要把它与 `can_ron_now` 混用。训练时可先只做前两项；Risk 对 Policy 产生收益后才加点数回归。

## Recipe G：正确记录 PPO 行为策略（PR-28/29）

如果 rollout 真正按混合策略行动：

```text
with probability (1-epsilon): a = deterministic argmax Q
with probability epsilon:     a ~ Categorical(softmax(Q / temperature))
```

则需要记录：

```text
pi_b(a) = (1-epsilon)*1[a==argmax] + epsilon*softmax(Q/temperature)[a]
log_behavior_prob = log(pi_b(chosen_action))
```

若探索还带 `top_p`，探索分布必须先按照 top-p 后的重归一化概率计算；如 argmax 有特殊 tie-break 也固定并审计。不能用 `log_softmax(Q)` 代替 `log(pi_b)`。后续 PPO 更新中的 `ratio=exp(log_pi_new-log_pi_behavior)`，masked action 上无非法概率。需要针对 `epsilon=0`, `epsilon=1`, `top_p<1`, one-legal-action 和 ties 做单测。

## Recipe H：Memory 的正确训练与评估（PR-22/23）

**不要**直接将未受监督的摘要 `detach()` 后使用。训练中保留来自未来局的策略损失到前局 summary writer 的梯度，或使用有对应解释力的辅助目标；可用 2–4 局 truncated BPTT。做 Counterfactual History Test 时只替换“原局之前的摘要”，不改变当前局事件、当前状态、分数、宝牌和牌墙；否则无法把模型输出改变归因给历史信息。需要证明：对真实在线对局有好处，而不是只在被替换历史的奇怪 OOD 状态里产生不同 logit。

## Recipe I：什么样的测试报告值得 Lead 接受

```text
PR: XX
DEV HEAD: <sha> ; MAIN/REVIEW HEAD: <sha>
Changed files: ...
Repro command: .venv/bin/python -m pytest -q tests/rf_test_*.py
Test count: ... passed / ... failed / ... skipped
Known untested conditions: ...
Training source license: allowed / blocked (evidence)
New artifact SHA-256: ...
Baseline model hash: ...
New model hash: ...
Eval protocol + seed_key/range: ...
Primary metric + set-level CI: ...
GPU peak VRAM / tokens/s / decisions/s: ...
P0 blockers: ...
Verdict: PASS | BLOCKED | INCONCLUSIVE | FAIL
Allowed next PR: ...
```

不可因标记为 `PASS` 就删除历史失败日志，也不可在未执行测试时从其他人的结果填入“本轮通过”。
