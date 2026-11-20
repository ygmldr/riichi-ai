# 实现范式、伪代码与必须通过的测试场景

本文提供**代码级实现思路**，并非可直接复制即可运行的真实项目代码。Agent 必须根据已固定的 `libriichi` Rust/PI API、实际 schema 和版本完成适配，且用测试证明一致性。**严禁将示意函数伪称已存在。**

---

## A. 数据重放与 POV 投影：分离标签路径与观察路径

对每个 `game`，构造一个只接受该 POV 可见事件的 tokenizer，与一个读取原始完整 log、仅输出监督标签的 `label oracle`。后者永远不可被模型 `forward` 读取。

```python
# PSEUDOCODE: do not import actual unavailable symbols as if implemented.
def build_game_pov(raw_game, pov: int, schema: FrozenSchema):
    assert 0 <= pov < 4
    source_events = parse_complete_mjai(raw_game)     # raw, potentially omniscient
    state = make_libriichi_player_state(pov)           # native authoritative rules
    visible_stream, supervised = [], []

    for i, ev in enumerate(source_events):
        # A: state used for legal moves; ensure feature getter itself is POV-safe
        cans = state.update(ev)
        # B: visible stream only gets explicit PUBLIC or SELF fields
        for vis_ev in project_event_allowlist(ev, pov):
            visible_stream.append(vis_ev)

        if cans.can_act():
            obs = snapshot_from_pov_only(state, visible_stream, pov)
            legal = authoritative_action_mask(cans)
            visible_stream.append(make_decision_token(obs))
            label = infer_logged_action_or_censor(source_events, i, pov, cans)
            # label oracle can read future events; these are NEVER placed in obs
            supervised.append({
                "position": len(visible_stream) - 1,
                "mask": legal,
                "played_action": label.action if label.valid else -1,
                "supervision_valid": label.valid,
                "reason": label.reason,
            })

    assert validate_visible_stream(visible_stream)
    return GamePovSample(visible_stream, supervised)
```

**注意**：先推进 `state.update(ev)` 再插入决策位置是否精确对应当前 Rust `GameplayLoader`，由 PR-10 的完整 replay parity 决定；若引擎要求在更新前/后设置 callback，以原生 `cans` 语义调整，并锁定测试。过去 `GameplayLoader` 借助 4-event window 判断事件标签，在抢断反应中容易发生歧义；本实现必须显式记录 `censor_reason`。

### A.1 禁止的写法

```python
# BROKEN — embeds the actual tile from an opponent tsumo
if ev["type"] == "tsumo":
    token["tile"] = tile_id(ev["pai"])

# BROKEN — raw event to model: includes opponent initial hands
features.update(ev)

# BROKEN — labels can leak with broad dict comprehension
model_input = {k: v for k, v in record.items() if k != "target"}

# BROKEN — naive pass label after opponent higher-priority response
if can_act and next_event.actor != pov:
    action = PASS
```

### A.2 推荐的安全实现

```python
def public_tsumo(ev: dict, pov: int) -> dict:
    actor = int(ev["actor"])
    if actor == pov:
        return {
            "kind": "TSUMO_SELF",
            "actor_rel": 0,
            "tile": tile_kind(ev["pai"]),
            "aka": tile_is_aka(ev["pai"]),
        }
    return {
        "kind": "TSUMO_PUBLIC",
        "actor_rel": (actor - pov) % 4,
        "tile": 34,
        "aka": False,
    }
```

每个事件分支都用新 dict 的 allowlist；`StartGame` 不输出原始 `seed` 或 `names`，`StartKyoku` 只选 `tehais[pov]` 并**在模型 state 中保存 hand counts 而非内部全体手牌**。

### A.3 为什么双通道必要

有监督训练必然需要“专家最终选择哪个动作”或“未来哪家听牌”，这些标签天然来自决策后的事件。因此不能禁止训练预处理读取未来；必须禁止未来通过 feature projection、normalizer、summary、teacher logits 等路径进入此前决策的 **input**。允许 label oracle 读未来，不允许 model input 读未来。

---

## B. 决策动作和 MJAI 事件对齐（最容易写错的工程环节）

**现有** `libriichi/src/dataset/gameplay.rs` 使用 `GameplayLoader::load_log` 得到逐 seat gameplay；含 `actions`, `masks` 和 `obs`。新的 `ReplayAdapter` 在 pilot 1000 场比对行为：

```python
for raw_game in pilot:
    old = existing_gameplay_loader(raw_game, oracle=False)
    new = build_all_povs_v2(raw_game)
    for pov in range(4):
        old_dec = collect_old_valid_actions_masks(old[pov])
        new_dec = collect_new_supervised_decisions(new[pov])
        # align by (kyoku,event_index,action-candidate signature), not position alone
        issues = align_and_compare(old_dec, new_dec)
        if issues: write_debug_json(game_id, pov, issues)
```

要特别处理：

- 同巡多个玩家同时可响应；低优先级玩家因其他人的和牌/鸣牌被抢断，并不代表玩家主动选择 PASS。
- `ReachAccepted` 和 `Dora` 可能夹在真正的动作决定与下一个动作事件之间；不能只看 `events[i+1]`。
- `Pon` 和 `Chi` 反应的耗用牌构成可能多选，且赤 5 构成会影响 action id；以 Rust 游戏状态中候选集合为权威。
- 强制唯一动作（如只能 `PASS`）进入时间流，但不贡献 BC 的有效 non-forced 决策。
- 原生 action index `37` 是 Reach（按 pinned upstream mapping），其他 index **不得按推断硬编码**。必须由原生动作候选或固定 ActionId 映射导出完整 46 项表，并创建 golden fixtures。
- `start_kyoku` 庄家 13/14 张的不同原始转换格式要先审计；禁止将新数据转化失败的情况下仍强行 label。

---

## C. Shard 格式与训练节省计算的关键

**不要**生成 `每个 decision -> 从 start_game 起的历史全部复制` 的巨大逐决策文件。建议的逻辑结构：

```python
@dataclass
class KyokuSlice:
    token_offset: int
    token_count: int
    decision_indices: list[int]      # index in this kyoku slice
    has_terminal_reveal: bool

@dataclass
class GamePOVShard:
    game_id: str
    pov: int
    visible_events: PackedArray     # chronological, no duplicated prefixes
    kyoku_slices: list[KyokuSlice]
    action_targets: list[int]        # aligned with decision indices
    supervision_mask: list[bool]
    legal_masks: PackedBoolArray     # (n_decisions, 46)
    final_rank_target: int
    metadata: dict                  # schema hash, split hash, provenance
```

训练器对于 Local-S：把 `KyokuSlice` 变长 packed batch，针对所有决策取最终隐藏向量并计算 CE；对于 Main-H：同 POV 的所有 slice 依序运行，局末压缩历史。

`max_seq_len` 不能用于**静默裁剪**。如果 local kyoku 长度超过配置，dynamic microbatch 或明确 OOM / overflow 分支。所有溢出计入 `train_overflow_count`。DataLoader 计算 `decisions/s` 时只用真正有监督的 decisions 数，不把所有 event tokens 叫作决策数。

---

## D. PyTorch attention：padding 与 causality 的正确组合

以布尔 `allowed[b,q,k] = valid_key[b,k] AND (k<=q)` 为基准；**padding query** 的输出不会参与 loss，应该仍处理得数值稳定（例如将无效 query 临时允许 attend 自身后最终强制置零），而不产生全屏蔽一行导致的 NaN。与 `torch.nn.functional.scaled_dot_product_attention` 的 bool mask 语义核对，版本不同要实际执行回归测试。

```python
# PSEUDOCODE, masks shape [B, L, L]
def causal_padding_allowed(valid_token: Tensor) -> Tensor:
    B, L = valid_token.shape
    t = torch.arange(L, device=valid_token.device)
    causal = t[:, None] >= t[None, :]
    visible = valid_token[:, None, :] & causal[None, :, :]
    # For padding query, allow its own self-key only for numerical stability;
    # ignore/crop outputs at padding locations afterward.
    diag = torch.eye(L, device=valid_token.device, dtype=torch.bool)
    visible = torch.where(valid_token[:, :, None], visible, diag[None, :, :])
    return visible
```

**对于有左 padding 的 batch**，RoPE position index 必须根据局内有效 token 的累计位置，而不是物理 tensor 下标，否则添加 padding 会改变输出。优先固定仅右 padding，后续支持左 padding 时补测试。使用 `attn_mask` 与 `is_causal` 时须按真实 Torch 版本要求传参，别在未确认兼容性时把两个都设为 true。

---

## E. 损失与梯度的数值测试

```python
# N decisions, each label belongs to a game/POV instance.
valid = supervision_valid & (legal_mask.sum(-1) > 1)
assert torch.all(legal_mask[valid].gather(1, played_action[valid, None]))
per_dec_loss = F.cross_entropy(masked_logits[valid].float(),
                               played_action[valid], reduction="none")
# per seat-game mean, then batch mean — choose/freeze one normalization.
loss = group_mean_then_average(per_dec_loss, group_id[valid])
```

**极端测试**：

1. `valid.sum()==0`：训练器应跳过该 microbatch，报告 `empty_supervision_batches`，避免 `nan` 的空平均。
2. 仅 1 个合法 action：不计算 BC 主损失，若选择保留该决策在序列中，mask 和 state 仍正确。
3. 单游戏 1000 决策 vs 另一游戏 10 决策：在 `per_seat_game` 归一化下两游戏总权重应相近。
4. 教师 Q 值全相等、高尺度、含非有限、只有一个合法动作：软标签须可稳定化或明确丢弃该 teacher supervision，原人类标签不受影响。
5. `Q` 不是 calibrated action probability；softmax 温度是模型蒸馏超参，不能报告为真实选择概率。

---

## F. 半庄记忆的实现选择和计算图

比较务实的第一版：

```text
for game POV in chronological kyoku order:
    historical_memory = state.completed_kyoku_summaries
    curr_hidden = local_transformer(curr_kyoku_visible)
    for every curr decision:
        h = fuse(curr_hidden[decision], memory_transformer(historical_memory))
        compute policy loss / aux loss
    if end_kyoku_observed:
        summary = summarize_completed_kyoku(curr_hidden, terminal_visible)
        state.completed_kyoku_summaries.append(summary)
```

**重要梯度问题**：如果每次都 `summary.detach()`，又不给压缩器任何训练目标，它不能学会让先前局信息有利于后续决策。可选正确实现有：

- **Option 1（推荐首试）**：只对最近 1–2 个已完成局保留跨局梯度，其余更早的摘要使用 detach。每半庄的 memory window 动态，通过 `G5` 比较显存与收益。
- **Option 2**：先训练 Local backbone，再使用可解释的中间层统计目标训练 summary，再冻结摘要并优化 memory/fusion（分阶段法，可能性能较低）。
- **Option 3**：全半庄跨局 BPTT（16GB 可能成本高，仅 benchmark 后实验）。

不允许在同一局 `END_KYOKU` 之后生成摘要再拿来帮助它**此前**发生的 decision。训练时如果通过 tensor reshape / join 导致时间越界属于 P0。

---

## G. 统计评估：与现有仓库一致的关键细节

现有 `scripts/eval_report.py` 已处理：

- 一 set = 同 `(seed, seed_key)` 的 `a/b/c/d` 四种座位轮换局。对 **set** bootstrap，不对单游戏 bootstrap。
- 顺位得点 `(90,45,0,-135)`，同点分数按现有 libros 排序规则处理。
- 不完整 set 可报告游戏数量，但不能参与 set bootstrap。

新的候选比较必须在相同墙/seed pool 与固定对手上运行，输出**配对 delta**。不要将两个模型单独的 95% CI 是否重叠当作显著性判断。

```text
prior to training/evaluation:
    write prereg: frozen checkpoint hashing, model/data arm definitions,
    primary metric, minimum interesting difference, seed pool allocation,
    early-stop rule, planned n_sets, bootstrap method, contingency actions

after evaluation:
    compute per set (candidate_pt - baseline_pt)
    bootstrap paired deltas across set IDs
    report point estimate and confidence interval
```

12k 是旧仓库对部分效果的经验规模，不是对**任何**对模型都足够。开发中 16/64 局仅验证软件正确性；2k/4k 只是灾难筛查，12k/24k 是探索选择；真正强度声明需要未看过的确认池，并根据预期收益和真实方差设计样本量。旧 A14 `[70000,73000)` 已使用，不能复用为 virgin confirm。

---

## H. 性能门槛和止损建议（必须在第一次正式实验前确认）

这份说明不伪称已经在用户 5070 Ti 上测量过。建议先生成下列表：

| 模型 | 局内/半庄 tokens | microbatch | BF16 peak VRAM | labeled decisions/s | ms/update | p95 decision latency | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| MVP-S | 256/512/1024 | 8/4/2? | TODO | TODO | TODO | TODO | 真实测 |
| Main-H | 局内 p50/p99 | 4/2/1? | TODO | TODO | TODO | TODO | memory 4*K |
| Flat-F | 1024/2048/4096 | 4/2/1? | TODO | TODO | TODO | TODO | OOM 显式 |

候选阈值（不是既定事实）：

- P0 安全性：**0 容忍**。
- replay/合法性：对纳入训练样本 **100%**；不合法样本剔除并完整报告。大量不可解释错误阻止正式训练。
- GPU：16GB 内没有 OOM/NaN；吞吐与 p95 延迟以 benchmark 容量计划作为工程决策。
- Model learning：小合成样本过拟合 >99% 仅用于验证梯度；不能据此判断策略强度。
- Belief：必须既有校准指标优势，也能在对局决策上证明增益，否则先不加复杂网络。
- RL：相对冻结 BC 基线不允许发生未解释灾难下降，否则停止；小规模 selfplay 的样本随机性不能支持“超过人类高手”的表述。

---

## I. 真实项目读码索引（为 Agent 减少盲搜）

```text
现有仓库第一方
README.md
RUNBOOK.md
NOTICE.md
configs/config.toml
configs/config.example.toml
scripts/run_train.sh
scripts/run_eval.sh
scripts/eval_report.py
scripts/parse_eval.py
scripts/download_data.py
scripts/normalize_gzip.py
scripts/verify_data.py
patches/0006-event-stream-tokens.patch    # known P0 event leak
patches/0007-model-event-branch.patch
patches/0008-rl-policy-kl.patch
patches/0010-arena-event-tokens.patch
runs/exp/A1/VERDICT.md
runs/exp/A14/VERDICT.md
runs/exp/ARCH_VERDICT.md
docs/EXPERIMENTS.md

上游 Mortal pinned submodule
libriichi/src/mjai/event.rs
libriichi/src/dataset/gameplay.rs
libriichi/src/state/player_state.rs
libriichi/src/arena/one_vs_three.rs
libriichi/src/agent/
mortal/dataloader.py
mortal/engine.py
mortal/one_vs_three.py
```

注意：上述是读码索引，并非声明每个 API 已通过编译或所有 Rust 绑定可直接被 Python 引用。**当读到不兼容处，先补最小绑定再逐项验证**。
