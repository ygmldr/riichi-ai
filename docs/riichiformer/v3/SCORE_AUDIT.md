# H-01 — final score / kyotaku 结算与 replay 边界审计（RiichiFormer V3.1）

**TASK_ID**：H-01（SCORE）
**分支 / 基线**：`rf/h01-score`，BASE_SHA `59691f57bc7158dcdb2ba98eecd9d102857171f8`
**工作树**：`/home/ygsldr/riichi-ai/worktrees/h01-score`
**Python / 原生库**：`/home/ygsldr/riichi-ai/.venv/bin/python` +
`PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release`（pinned `libriichi.so`，未重建）
**状态**：PASS（无未解释 score mismatch；真实日志 final kyoku 的独立锚点 = NOT_VERIFIABLE，见 §7）

---

## 0. 结论摘要

1. **供托被双计**：`hora.deltas` 在 Tenhou-derived MJAI 日志中已经包含 `1000 × kyotaku`；旧
   `final_scores_from_events()` 又无条件 `scores[actor] += 1000 * kyotaku`，每个“带供托的和牌”
   都会多算 1000~N×1000。已删除额外加项，并把 `sum(deltas) == 1000 × kyotaku` 变成显式校验；
   不一致即 fail closed。
2. **终局供托**：旧函数不处理 `end_game` 时留在桌上的供托。按 `arena/game.rs:182-184`，
   终局供托归最高分者（同分取最小座位号）；已实现，并用 native arena 的 `end_game` 分数锚定。
3. **四家立直流局**：Tenhou-derived 日志中第 4 个立直没有 `reach_accepted`，但 1000 点确实扣除、
   供托 +1。已按“pending reach + 3 accepted + 全零 deltas”建模；旧实现漏扣 1000。
4. **0 seat 被接受为成功**：旧 `len(gameplays) not in (0, 4)` 使 0 seat 继续 `ok=True`；
   现要求必须恰好 4 seat。
5. **`kyoku_count` 错误**：旧实现先 `take_at_kyoku()` 后再次 take，实际恒为 0；即使修掉，
   按 seat 0 去重也会漏计“seat 0 未行动的局”。现从原始 `start_kyoku` 边界计数，并与 native
   `at_kyoku` 流交叉核对（不匹配记录 flag/error）。
6. **协议 fail-closed**：直播天凤 POV 日志的 `hora` 携带绝对 `scores` 而非 `deltas`，该协议未被
   本项目验证；`replay_scores(strict_protocol=True)` 抛 `ScoreProtocolError`，兼容入口
   `final_scores_from_events()` 返回 `None`，绝不编造分数。
7. **真实日志交叉验证**：2025+2024 共 **614 局**含立直/供托的真实牌谱、**6,352 个局间转移**，
   每个 `start_kyoku.scores`/`kyotaku` 全部一致，**0 mismatch**。250 局完整 native replay
   （162,071 decisions）0 illegal、0 kyoku 交叉不一致。native arena 160/160 POV 日志最终分与
   `end_game` 原生分数完全一致。
8. **PR-06 影响**：614 局中 341 局终局分数变化，45 个 seat-game 的顺位变化（22 局）；历史 q /
   质量报告必须由 Lead 安排重算，本任务**未改写**任何 quality 产物。

---

## 1. 证据等级定义

| 等级 | 含义 |
|---|---|
| **A** | pinned native 源码 + 可重复运行的真实日志/native 引擎锚点，可直接作为实现依据 |
| **B** | 由 A 推导、但缺少独立外部锚点（例如真实 Tenhou 终局官方分数）或规则等价性未独立验证 |
| **C / NOT_VERIFIABLE** | 本任务未覆盖、样本不足或依赖 H-04 规则审计 |

本任务各项的证据等级：

| 项目 | 等级 | 依据 |
|---|---|---|
| `hora.deltas` 含供托 | **A** | `board.rs` 源码 + 6,352 次真实转移 + 160 次 native 最终分 |
| 第二荣/多荣不再重复供托 | **A** | `board.rs:406-422`（`kyotaku_point = 0` after first）+ 2025 样本 21 次、2024 样本 16 次多荣 |
| `ryukyoku.deltas` 不含供托 | **A** | `board.rs` exhaustive/abortive 路径 + 真实日志 `sum(deltas)==0` |
| 四家立直 pending reach 扣 1000 | **A** | 真实日志 1 例 + 下一局锚点；native 源码路径不同（见 §6.3） |
| 终局供托归最高分者 | **A**（native）/ **B**（真实天凤） | `game.rs:182-184` + seed 20005 原生 `end_game` 锚点；dataset 无终局分数可对照 |
| `kyoku_count` 原始边界 | **A** | 250 局完整 replay 与 native `at_kyoku` 一致 |
| 真实 dataset 的终局独立分数 | **C / NOT_VERIFIABLE** | `end_game` 只有 `{"type":"end_game"}`，无 `scores`；见 §7 |
| 三家和了流局、同分顺位、祝仪/顺位点 | **C** | H-04 范围；本任务只做分数结算，不做最终名次规则/祝仪 |

---

## 2. 变更文件（仅 H-01 owned files）

| 文件 | 变更 |
|---|---|
| `riichiformer/data/replay.py` | 新增唯一权威入口 `replay_scores()` + `ScoreReplay`/`ScoreProtocolError`；`final_scores_from_events()` 改为委托；`replay_game()` 使用新结算、0-seat 拒绝、raw/native kyoku 交叉核对 |
| `tests/rf_test_replay.py` | 14 个 H-01 回归用例（synthetic 结算 shape + native fixture + 0-seat/kyoku boundary/protocol fail-closed；真实日志交叉验证在运行时读取本地 datasets，不复制进仓库），保留原有 integration tests |
| `docs/riichiformer/v3/SCORE_AUDIT.md` | 本文件 |
| `docs/riichiformer/v3/logs/H01_pytest.log` | 修复前失败 + 修复后通过完整 stdout/exit code |

未修改：`canonicalize.py`、`dedup.py`、`split.py`、`provenance.py`、schema/POV、
`scripts/rf_validate_replays.py`、`riichiformer/data/player_quality.py`、`vendor/mortal`、
`.gitignore`、`pyproject`。`rf_validate_replays.py` 的调用签名（`replay_game`/`summarize_replay`）
保持不变，无需 RFC。

---

## 3. 结算语义：source of truth

### 3.1 native 源码

`vendor/mortal/libriichi/src/arena/board.rs`：

- `check_riichi_accepted()`（342-350）：`scores[actor] -= 1000; kyotaku += 1`，并写
  `reach_accepted`。
- `handle_hora()`（360-467）：
  - 376-377：`let mut kyotaku_point = board.kyotaku as i32 * 1000; board.kyotaku = 0;`
  - 417：`deltas[actor] = point.ron + kyotaku_point + honba_left * 300;`
  - 422 之后：`kyotaku_point = 0; honba_left = 0;` — 多荣时只有**第一个**和牌者拿供托。
  - 453-454：自摸 `deltas[actor] = point.tsumo_total(...) + kyotaku_point + honba_left * 300`。
- `abortive_ryukyoku()`（504-509）：写 `ryukyoku.deltas = [0; 4]`，不碰供托。
- 因此：**`hora.deltas` 的供托 inclusion 是协议的一部分，不能再额外加 `1000*kyotaku`。**

`vendor/mortal/libriichi/src/arena/game.rs`：

- 122-124：每局结束 `kyotaku = kyoku_result.kyotaku_left`，`scores = kyoku_result.scores`。
- 154-170：连庄/进局；`abortive_ryukyoku` 不 `kyoku += 1`（同局重开，honba+1）。
- 182-184：`commit()` 在 `ended` 时
  `*scores.iter_mut().min_by_key(|s| -**s).unwrap() += kyotaku * 1000`，即终局供托给最高分
  （同分取最小 index），随后才把 `GameResult.scores` 交给 `end_game` 回调。

### 3.2 真实日志验证（不是文档转述）

对 `/home/ygsldr/riichi-ai/datasets/extracted/{2025,2024}` 的真实牌谱逐事件验证：

- `sum(hora.deltas) == 1000 × kyotaku_before_hora`（首个和牌；第二荣为 0）；
- `sum(ryukyoku.deltas) == 0`（供托不进入 deltas）；
- 下一局 `start_kyoku.scores` 与 `kyotaku` 等于上一局结算结果。

计数见 §5。这是“deltas 已含供托”的直接实证；旧的额外 `+1000*kyotaku` 在 341/614 局中改变
终局分数，并使终局总分不等于 100000。

### 3.3 旧实现的反例

旧 `final_scores_from_events()`：

```python
elif etype in ("hora", "ryukyoku") and scores is not None:
    ...
    if etype == "hora" and kyotaku:
        scores[actor] += 1000 * kyotaku   # ← double count
        kyotaku = 0
```

真实 2025 日志 `2025010100gm-00a9-0000-15efc65c.mjson`（S2 honba 1，供托 1）：

- 真实 `hora.deltas = [-1100, 4300, -1100, -1100]`，sum = +1000；
- 正确终局：`[23300, 29400, 28100, 19200]`；
- 旧实现：`[23300, 30400, 28100, 19200]`（多 +1000）；
- 新实现：`[23300, 29400, 28100, 19200]`。

---

## 4. 修复内容

### 4.1 唯一权威入口

```python
replay_scores(events, *, protocol="auto", ruleset_id="tenhou",
              award_final_kyotaku=True, strict_protocol=True) -> ScoreReplay
```

`ScoreReplay` 返回：`scores`、`scores_before_final_kyotaku`、`kyotaku_left`、`kyoku_count`、
`hora_records`（每和牌的 delta sum/kyotaku）、`transitions`（每次局间比对）、`mismatches`、
`errors`、`warnings`、`flags`、`multi_ron_count`、`final_kyotaku_award_seat`。

`final_scores_from_events(events)` 现在只有一行委托：

```python
result = replay_scores(events, strict_protocol=False)
return result.scores if result.ok else None
```

不再有第二份结算循环。

### 4.2 关键规则

1. **锚点**：每个 `start_kyoku.scores/kyotaku` 重置状态；在下一个 `start_kyoku` 时比对上一局
   结算的 `scores` 与 `kyotaku`。不一致 → `kyoku_transition` mismatch，`ok=False`，但继续用新
   锚点结算（报告全部 mismatch）。
2. **reach**：只有 `reach_accepted` 扣 1000、供托 +1；被荣和的 `reach` 不扣。
3. **hora**：`deltas` 必填；`sum(deltas)` 必须等于 `1000 × kyotaku_before`（首和/自摸/双荣首和），
   随后 `kyotaku=0`。第二荣期望 0；不一致 → `hora_delta_sum_mismatch`。
4. **ryukyoku**：`deltas` 必填且 `sum==0`；供托保留。全零 + pending reach + 3 accepted →
   四家立直，补扣 pending 并 `kyotaku += 1`（flag `four_riichi_pending_reach_charged`）。
5. **end_game**：若当前局尚未结算（无 `hora`/`ryukyoku`/`end_kyoku`）→ `unsettled_end_game`
   fail closed，不把桌上供托发给任意“领先者”。否则若有供托 → 给最高分（同分最小 seat），
   `final_kyotaku_award_seat` 记录。
6. **总分不变量**：初始 `sum(scores)+1000*kyotaku` 必须等于终局 `sum(scores)`（正常情况下
   100000）；不满足 → `score_total_mismatch`。满贯/自摸/双荣/流局/终局供托都满足该不变量。
7. **协议**：`hora` 全部有 `deltas` → `tenhou_mjai_deltas_v1`；全部只有绝对 `scores` →
   `absolute_hora_scores`（识别但拒绝）；混合/缺失 → `ScoreProtocolError`。`strict_protocol=False`
   时返回 `ok=False, scores=None`。
8. **kyoku_count**：`start_kyoku` 原始事件计数（含 abortive 同局重开）；`replay_game()` 从四个
   seat 的 `at_kyoku` 取 `max+1` 与 raw 比对，记录 `kyoku_cross_check`；native 更多 → error；
   native 更少（末尾零 decision 的流局）→ flag；seat 0 去重更少 → flag
   `seat0_kyoku_undercount`（旧实现即此处漏计，且旧代码因二次 take 恒为 0）。
9. **0 seat**：`len(gameplays) != 4` 一律 `ok=False, error_kind="seat_count"`，含 0。

---

## 5. 真实日志交叉验证（200+ 局要求）

### 5.1 命令

```bash
cd /home/ygsldr/riichi-ai/worktrees/h01-score
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q -s \
  tests/rf_test_replay.py tests/rf_test_quality.py
```

以及独立汇总脚本（采样 2025/2024 各前 500 个文件，筛出含 `kyotaku>0` 的局）：

```bash
/home/ygsldr/riichi-ai/.venv/bin/python \
  /home/ygsldr/riichi-ai/artifacts/riichiformer/h01/crosscheck.py
# → artifacts/riichiformer/h01/crosscheck_2025_2024.json
```

完整 native replay：

```bash
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python \
  /home/ygsldr/riichi-ai/artifacts/riichiformer/h01/replay_games_check.py
```

### 5.2 结果

| 年份 | 含供托局数 | 局间转移 | transition mismatch | hora 总数 | 带供托 hora | 多荣（额外 hora） | 四家立直 | 终局留供托 | `end_game.scores` 锚点 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 | 293 | 3,072 | **0** | 2,705 | 1,733 | 21 | 1 | 10 | 0 |
| 2024 | 321 | 3,280 | **0** | 2,890 | 1,851 | 16 | 0 | 20 | 0 |
| **合计** | **614** | **6,352** | **0** | 5,595 | 3,584 | 37 | 1 | 30 | **0** |

- 每个 mismatch 都会被记录 game_id/kyoku/settled/next；本次为 **0**，无需逐条列出。
- pytest 内置的 `RealCorpusCrossCheckTests` 对 2025 前 400 个文件运行同一检查：
  `games=234, transitions=2461, hora_with_kyotaku=1392, games_with_leftover=8`，全部通过。
- 全部 614 局终局 `sum(scores) == 100000`（新实现）；旧实现 **341/614** 局总分不等于 100000。
- 250 局完整 `replay_game()`：`games=250 ok=250 decisions=162071 illegal=0`，
  `kyoku_cross_check` 0 不一致；11 局出现 `seat0_kyoku_undercount` flag（旧 seat-0 去重会少计）；
  16 局多荣、10 局终局供托、250 局全部 `hora_kyotaku_in_deltas`。

### 5.3 修复前后失败

修复前（BASE 的 `replay.py`，新测试已加入）：

```
14 failed, 4 passed in 3.25s   (exit 1)
```

代表值：

| 用例 | 旧值 | 正确值 |
|---|---|---|
| reach+hora（供托 1） | `[23300, 30400, 28100, 19200]` | `[23300, 29400, 28100, 19200]` |
| 双荣（供托 2） | `[15400, 22100, 32500, 32000]` | `[15400, 20100, 32500, 32000]` |
| 四家立直 pending | `[31000, 22000, 22000, 22000]` | `[30000, 22000, 22000, 22000]` |
| 终局留供托（native 锚点） | `[25000, 25000, 24000, 25000]` | `[26000, 25000, 24000, 25000]` |
| 0-seat loader | `ok=True` | `ok=False, error_kind="seat_count"` |
| kyoku_count（seat0 未行动局） | `0` | `12 == raw == native` |
| 绝对 scores 协议 | 返回 `[25000]*4` | `ScoreProtocolError` / `None` |

修复后：

```
31 passed, 1 skipped in 3.44s   (exit 0)
```

唯一 skip 是 `tests/rf_test_quality.py` 的 CLI case（worktree 相对路径下
`datasets/extracted/2024` 不存在），与 H-01 无关；quality 代码未改，13 passed。

完整 stdout/exit code：`docs/riichiformer/v3/logs/H01_pytest.log`。原始修复前日志：
`artifacts/riichiformer/h01/logs/H01_pytest_old.log`。

---

## 6. native 引擎锚点

### 6.1 生成方式（Python binding 的 exact API gap）

`libriichi` 的 Python binding **没有**暴露 `arena::Board`/`GameResult`/分数查询 API。
`libriichi.arena` 只暴露 `OneVsThree` / `TwoVsTwo`（`py_vs_py` 返回 rankings），
`dataset.GameplayLoader` 只返回 actions/masks/at_kyoku/at_turns/obs，不返回 board scores。
这是确切的 API gap：**无法用 `GameplayLoader` 直接导出终局分数**。

可用替代：`OneVsThree.py_vs_py` 接受实现了 `engine_type="mjai-log"` 的 Python agent；
Rust 的 `end_game(game_idx, scores)` 回调会收到 arena `GameResult.scores`（已含终局供托）。
我们用 `mjai-log` 的 `h01` agent（能立直、能自摸/荣和，否则摸切）生成日志并捕获该分数：

```bash
cd /home/ygsldr/riichi-ai/worktrees/h01-score
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python \
  /home/ygsldr/riichi-ai/artifacts/riichiformer/h01/gen_native.py
# seeds 10000..10039, seed_key 0x5DFAA4CEF265CD7, log_dir=/tmp/h01/native_logs

PYTHONPATH=... .venv/bin/python \
  /home/ygsldr/riichi-ai/artifacts/riichiformer/h01/one_seed.py 20005 /tmp/h01/seed20005
PYTHONPATH=... .venv/bin/python \
  /home/ygsldr/riichi-ai/artifacts/riichiformer/h01/one_seed.py 20115 /tmp/h01/seed20115
```

### 6.2 结果

- **160/160** POV 日志（40 seeds × 4 rotations）的 `replay_scores().scores` 与 native
  `end_game` 分数逐位相等，0 mismatch。
- **seed 20005**（终局留供托 1，最后局 `W4 honba11`，start `[25000,25000,24000,25000]`，
  `ryukyoku [0,0,0,0]`，end_game）：
  native final `[26000,25000,24000,25000]` = 重放结果；`kyotaku_left=1`，award seat 0。
- **seed 20115**（S3 honba6，供托 1，`hora actor=1 target=3 deltas=[0,10800,0,-9800]`，
  sum=+1000）：下一局 `start_kyoku.scores=[25000,34800,25000,15200]`，重放结果与整局 native
  final `[25000,34800,25000,15200]` 一致；`kyotaku_left=0`。
- 这两个 anchor 分别覆盖“终局供托分配”和“和牌吸收供托”；fixture 中只保留 score 事件，
  deltas/start scores 均为 native 日志原文（非手造）。

### 6.3 四家立直：两种来源的差异（已分别覆盖）

- **Tenhou-derived MJAI 日志**：第 4 个 `reach` 后直接 `ryukyoku [0,0,0,0]`，没有
  `reach_accepted`；但下一局 `kyotaku` 从 3→4、四家各 -1000。我们的 pending-reach 规则
  复现该行为（真实日志 `2025061413gm-00a9-0000-80d770aa.mjson`）。
- **native arena**：`board.rs:561` 在 `Event::None` 分支先 `check_riichi_accepted()`，所以第 4 个
  `reach_accepted` 会写入日志，`accepted_riichis==4` 后下一步才 abort；旧规则不会触发
  pending 分支，正常结算 4 根供托。两种来源都已覆盖，不会双重收取。

---

## 7. 终局（final kyoku）独立锚点状态

- **Tenhou-derived dataset（2025/2024）**：`end_game` 只有 `{"type": "end_game"}`，没有
  `scores`；614 局中 `with_end_game_scores_anchor=0`。因此**真实日志的终局绝对分数 =
  NOT_VERIFIABLE**。我们能验证的是：(a) 每次局间 `start_kyoku` 转移 100% 一致；
  (b) 终局总分不变量 `sum == 100000`（含终局供托分配后）；(c) 与 native `end_game` 锚点
  160/160 一致。**没有**把 native 结论冒充为“真实天凤终局已独立核对”。
- **live-Tenhou POV 日志**（`runs/tenhou/*/events.mjson`）有 `end_game.scores`，但它的
  `hora` 用绝对 `scores`（协议不同）；本任务按 fail-closed 拒绝，未用它做最终分对照。
  若后续需要真实天凤终局锚点，应单独验证该协议的 `hora.scores`/`end_game.scores` 语义后再接入。
- **native arena**：`end_game` 回调的分数是独立权威（A 级），已覆盖终局供托；但这只证明
  我们与 pinned native 一致，不证明 native 与官方天凤规则在终局名次/祝仪上完全一致（H-04）。

---

## 8. PR-06 / quality 影响范围（不改写产物）

`riichiformer/data/player_quality.py::extract_appearances()` 调用
`replay.final_scores_from_events()` 并用 `ranks_from_scores()` 生成 rank。修复后：

- 在 614 局含供托真实牌谱中，**341 局（55.5%）终局分数变化**；
- **45 个 seat-game 的 rank 变化，涉及 22 局（3.6%）**（2025: 26 seats；2024: 19 seats）；
- 终局供托分配只增加最高分者的分数，在同一 tie-break 下不改变名次，但改变终局分数；
- `q` 的历史 mean-rank 由过去所有局的 rank 计算，因此单个 rank 变化会传播到该玩家后续
  所有 target 的 q；**受影响范围大于 45 个 seat-game**；
- 因此现有 PR-06 的 `quality_audit.md` / q / reliability 结果对含供托局已 **stale**，
  必须由 Lead 在 H-01 合入后重算；本任务按约束**没有**重写任何 quality 产物。
- `top25_seat` 在重算完成前继续保持禁用（roadmap H-01 验收条件）。
- 旧 `final_scores_from_events` 的 fail-open 行为（协议不明也返回分数）已改为 fail closed；
  若未来真实日志出现未建模的结算怪例，`extract_appearances` 会返回 `missing_scores` 并把该局
  排除，而不是带病进入 quality。

---

## 9. 其他调用方与遗留问题

| 调用方 | 状态 |
|---|---|
| `riichiformer/data/player_quality.py` | 自动受益；无改动；`tests/rf_test_quality.py` 13 passed / 1 skipped |
| `scripts/rf_validate_replays.py` | 签名不变；`replay_game` 现在对 score mismatch/protocol/kyoku error fail closed；`summarize_replay` 新增字段向后兼容 |
| `scripts/eval_report.py` | **未改（非 owned）**。它的 `replay_one()` 已经正确地“deltas 含供托、终局供托给最高分”，但 (a) 不处理四家立直 pending reach，(b) 不验证局间转移，(c) 对 `hora` 无 deltas 的来源会静默用 0。建议后续改为调用 `replay_scores()`；当前是独立重复实现，属于 **P1 follow-up** |
| `scripts/rf_build_canonical.py` / schema / POV | 不受影响（H-02/PR-08/09 独立） |

### P0 / P1

- **P0（已关闭）**：供托双计、终局供托缺失、0-seat 接受、kyoku_count 恒 0/seat-0 漏计、
  四家立直漏扣、协议 fail-open。
- **P1（本任务不阻塞，需后续安排）**：
  1. PR-06 quality 全量重算（341/614 分数变化、22 局 rank 变化；传播范围更大）。
  2. `scripts/eval_report.py` 去重 + 四家立直修复（建议委托 `replay_scores`）。
  3. 真实 dataset 终局独立锚点缺失；建议未来保留 `end_game.scores` 或天凤 `owari` 分数。
  4. 三家和了流局/同分顺位/祝仪等规则差异（H-04）；本任务只覆盖分数结算。
- **BLOCKERS**：无。

---

## 10. 复现与产物

```bash
# 1) H-01 tests + affected quality tests
cd /home/ygsldr/riichi-ai/worktrees/h01-score
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q -s \
  tests/rf_test_replay.py tests/rf_test_quality.py
# => 31 passed, 1 skipped; exit 0

# 2) real-log cross-check 2025/2024 (614 games, 6352 transitions)
/home/ygsldr/riichi-ai/.venv/bin/python \
  /home/ygsldr/riichi-ai/artifacts/riichiformer/h01/crosscheck.py

# 3) full native replay 250 games
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python \
  /home/ygsldr/riichi-ai/artifacts/riichiformer/h01/replay_games_check.py

# 4) native anchor generation + targeted seeds
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python \
  /home/ygsldr/riichi-ai/artifacts/riichiformer/h01/gen_native.py
```

产物（本机，artifacts 不提交）：

| 文件 | 说明 |
|---|---|
| `artifacts/riichiformer/h01/logs/H01_pytest.log` | 与本仓库 `docs/.../logs/H01_pytest.log` 同内容 |
| `artifacts/riichiformer/h01/logs/H01_pytest_old.log` | 修复前 14 failed / 4 passed 原始输出 |
| `artifacts/riichiformer/h01/crosscheck_2025_2024.json` | 614 局统计与 examples |
| `artifacts/riichiformer/h01/native_anchor_summary.json` | native 160/160 + seed 20005/20115 锚点 |
| `artifacts/riichiformer/h01/crosscheck.py` 等 | 可复现脚本 |

测试 fixture 证据：

- 运行时真实日志样本（不进入仓库）：`2025010100gm-00a9-0000-15efc65c.mjson`、`2025040518gm-00a9-0000-d802454f.mjson`、
  `2025061413gm-00a9-0000-80d770aa.mjson`；边界局 fixture
  `2025061404gm-00a9-0000-c5b3fa50.mjson`（seat 0 在 E2 honba3 未行动，raw/native=12，seat0=11）。
- native fixture：seed 20005 最后局 + native final；seed 20115 S3 hora + native final transition。
- protocol fail-closed fixture：`runs/tenhou/20260919T092832Z/events.mjson` 的绝对 `hora.scores`
  片段。

---

## 11. 未解释 mismatch / 残余风险

- 614 局真实日志、6,352 次局间转移：**0 mismatch**。
- 250 局完整 native replay：**0 illegal、0 kyoku 交叉不一致**。
- native 160/160 最终分：**0 mismatch**。
- 已知残余：
  1. 真实日志终局绝对分数 NOT_VERIFIABLE（§7）；
  2. 四家立直只捕获 1 例真实样本（12k 扫描中 14 例；本次 614 局中 1 例）；若其他转换器
     对第 4 个立直的处理不同，会触发 `unverified_pending_reach_on_ryukyoku` 并 fail closed，
     不会静默错算；
  3. 多荣只验证双荣（真实 37 例、native 0 例）；三荣未在样本中出现，native 也注释
     “No triple-ron ryukyoku”（H-04）；
  4. 同分顺位/祝仪不在本任务范围；`ranks_from_scores` 的同分 tie-break 沿用 seat index，
     与 native `min_by_key(-s)` 一致，但不声称与官方天凤顺位规则完全等价。
