# RiichiFormer V2 — 评测种子池登记表 (EVAL POOL REGISTRY)

**建立日期 (UTC)**：2026-11-20  
**建立者**：PR-00（只读审计，未删除/未移动任何对局，未重跑任何评测）  
**数据来源**：① `runs/**` 下的 `.json.gz` **文件名**（不读内容）；② `docs/`、`runs/*/PREREG|REPORT|VERDICT`、
`runs/akochan_eval/TREND.md`；③ `scripts/eval_report.py` 的 `EXACT_GATE`；④ `scripts/exp_a14_run.py`。

> 目的：让后续 PR（尤其 PR-05 split registry、PR-17/26 确认实验）**不能把已消费窗口重新说成 virgin**。
> 本表只登记“有磁盘证据/文档证据”的消费；推断不到的一律写 `UNKNOWN`。

---

## 1. 口径

- **seed_key**：libriichi 1v3 的墙随机密钥；`seed` 是墙号，`1 seed = 1 set = 4 半庄`（split a/b/c/d = 挑战者坐 0/1/2/3）。
- **窗口**：`[seed_start, seed_start + games/4)`，半开区间。
- **consumed（已消费）**：该窗口的对局已落盘，并且被用于报告/判定/对照（不是“跑过但丢弃”）。
- **confirm（确认）**：在看结果前预注册、只允许消费一次的窗口；一旦有对局落盘即视为消费，**不可回滚**。
- **select（选择）**：可用于调参/多臂比较，可重复，但必须全量报告、不许挑选性重跑。
- **frozen corpus（冻结语料）**：只读目录，任何新工具不得写入。

---

## 2. seed_key 清单

| seed_key | 用途 | 证据 |
|---|---|---|
| `0x5dfaa4cef265cd7` (= `423244137402621143`) | 主评测/训练曲线/自博弈的默认 key | `configs/config.toml:90`；`runs/1v3/PREREG.md:21` |
| `0x80a7b9e2dec96fa` | `runs/1v3/` 的 2000 局 gate 语料（500 sets，种子 `[10000,10500)`） | `scripts/eval_report.py:64`（`EXACT_GATE`） |
| `0x455831` | EX1 研究报告的独立语料（`900000000+` 三个窗口） | `runs/exp/EX1/corpus/{calib,select,report}_sub/` |
| `0x923b8015a6879ba9` | 上游 bug #114 的 200 局定向探针 | `runs/exp/114/games` |
| `0x2000` / `0x4187fa4a2a7a20b8` | `runs/exp/114/{spy_test_play,spy_train_play}` 各 4 局 | 同上 |
| 每个 worker 一个随机 key（如 `0x2acae9f7cbc244fd`、`0x26a8b6d08f49513f`、`0x6356bc801a66fc4e` …） | **自博弈训练流量**（server drain/buffer），不是评测池 | `runs/exp/SELFPLAY/`、`runs/exp/A14/selfplay/` |

> 自博弈 worker 的 key 每次生成都不同，且目录里是训练样本而不是 1v3 评测；不得把它们当成“新 confirm 池”。

---

## 3. 已消费窗口（按区间）

| 窗口 | 角色 | 消费方 / 目录（每个目录的对局数） | 可否再作 confirm |
|---|---|---|---|
| `[10000,13000)` | **select**（池 A / variance pool_1） | `runs/variance/pool_1/games`（12,000）；`runs/exp/A7|A8|A9|E2|E3a|EVAL_A_175200|M328/vs148800|M328/vs175200|A14/eval_poolA` 各 12,000 | **禁止** |
| `[10000,10500)` | gate（500 sets） | `runs/1v3/`（2,000 局，key `0x80a7…`；另有 4 个 key `0x5dfa…` 的 legacy 文件） | **禁止**（且目录冻结） |
| `[20000,23000)` | 噪声地板 | `runs/variance/pool_2/games`（12,000） | **禁止** |
| `[30000,33000)` | 噪声地板 + A1 | `runs/variance/pool_3/games`、`runs/exp/A1/games_{original,shuffle,recency_zero}`（4×12,000） | **禁止** |
| `[40000,40030)` | akochan 早期探索 | `runs/akochan_eval/games_extra`（seed 40022 的 4 局）；`TREND.md:92` 记录 40000–40029 的 3 批 40 局 + 分块串行 | **禁止**（含未落盘批次） |
| `[50000,53000)` | **池 B，confirm（恰好一次）** | `runs/exp/ARCH_B/{E3a,E2}/games`（各 12,000）；`runs/exp/EVT/smoke_games` 44 局子集 | **禁止**（见 §5 流程缺陷） |
| `60000–63001` | akochan 趋势尺 | `runs/akochan_eval/games`（40 局）；`TREND.md:85-93`：成功 10 seeds（60000,60003,60006,61000,61002,61005,62001,62002,63000,63001），毒局 8 seeds（60001,60002,60004,60005,61001,61003,61004,62000） | **禁止**（毒局还会确定性 `bad_alloc`） |
| `[70000,73000)` | **A14 confirm（已消费）** | `runs/exp/A14/eval_confirm/games`（12,000）；`FORK_ROADMAP.md:289` 记录 Δpt −24.075 | **禁止** |
| `[120000,126000)` | **M3 confirm（已消费，EXACTLY ONCE）** | `runs/exp/M3/games`（24,000，6,000 sets）；`M3/REPORT.md`、`M3/PREREG.md` | **禁止** |
| `900000000+` / `901000000+` / `902000000+` | EX1 内部语料（key `0x455831`） | `runs/exp/EX1/corpus/{calib,select,report}_sub`（400/400/800 局） | 与主协议隔离；不建议复用 |

### 3.1 只读目录（**禁止写入**）

| 目录 | 内容 | 证据 |
|---|---|---|
| `runs/1v3/` | 2,004 个 `.json.gz`：2,000 主 gate（key `0x80a7…`）+ 4 个 legacy（key `0x5dfa…`） | `PREREG.md:55`；`EXACT_GATE.ignored_files=4` |
| `runs/variance/` | 3 个互斥噪声池，共 36,000 局 | `runs/1v3/PREREG.md:33`；`pool_{1,2,3}` |
| `runs/akochan_eval/games*` | akochan 尺子产物；含毒局记录 | `TREND.md` |
| `runs/exp/PINNED/` | 4 个钉住权重（非对局） | `configs/config.toml:138` |

> `runs/1v3/` 允许追加 `.md`/`.log`，**绝不**允许写入 `*.json.gz`。

---

## 4. 本地**没有**对局产物的区间（≠ virgin）

按文件名扫描（`runs/**/*.json.gz`，含 4 段式 self-play drain 文件名）得到以下空档：

```text
[0,10000)  [13000,20000)  [23000,30000)  [33000,40000)
[41000,50000)  [53000,60000)  [64000,70000)  [73000,120000)
[126000,900000000)
```

**只能说明“本机没有落盘评测对局”**，不能证明从未被离线分析/人工看过：
- 旧仓库的 offline eval、训练 holdout、EX1 报告都可能不保存 1v3 对局；
- 文档/配置里也可能出现过某些数字（M3 的 PREREG §5 就是靠全库文本审计 + 落盘扫描双重确认）。

因此：**本表不认证任何区间为 virgin。** 新确认池必须同时满足：

1. 使用**新 seed_key**（不要复用 `0x5dfa…` / `0x80a7…`）；
2. 在 `runs/exp/<PR>/PREREG.md` 里预注册窗口、指标、样本量、停止规则；
3. 启动前做**全库**文本 + 文件名审计，并把命令与输出粘进 PREREG；
4. 结果落盘后立刻在本表新增一行（窗口、命令、SHA、结果、日期），之后永远标 consumed；
5. 不得把已消费窗口“改名/换 key”后重新称为 confirm。

---

## 5. 流程缺陷与矛盾（必须记录，不得回写历史）

1. **池 B 的消费记录没有按 PREREG 要求追加**：`runs/1v3/PREREG.md:31-32` 要求“若池 B 被消费，必须在此文件追加一行记录”；
   但该文件最后一次修改只到池 A 结论（2026-09-17），而池 B 已被 `runs/exp/ARCH_B` 的 E3a−E2 单次确认消费
   （`EXPERIMENTS.md:183` 明确写“池 B 已被 E-18 一次性消费”）。
   → **本表接管该事实**；PR-05 应在新的 split registry 中补记，不要篡改旧 PREREG。
2. **A14 PREREG 把 confirm 从池 B 改到 `[70000,73000)`**，理由是池 B 已被消费；该替换有 commit body 与
   `decisions.md` C-1 证据（`EXPERIMENTS.md:183`）。`[70000,73000)` 之后确实被 A14 消费（12,000 局 + Δpt −24.075）。
3. **M3 的 `[120000,126000)` 不在旧 AGENTS/交接文档的“冻结语料”清单里**（该清单只列 A/B/A14 confirm），
   但它是已消费的 confirm（24,000 局）。任何后续 Agent 不得把它当新池。
4. **`configs/config.toml` 的 `[1v3].seed_start=30000` 指向 variance pool_3**；它只描述“下一次若直接跑
   `run_eval.sh` 会写哪里”，不代表该窗口还可用。新实验必须显式改窗口 + 显式 log_dir。
5. `runs/akochan_eval/TREND.md` 的成功 seed 只有 10 个，但窗口 `[60000,63002)` 内有 8 个毒局；
   后续若扩大 akochan 样本，必须避开这些 seed（否则重复 `bad_alloc`）。

---

## 6. 给 PR-05（split registry）的硬约束

- 数据 split（train/val/select/confirm）必须按 `game_id` 与来源隔离；**评测墙池**另立 registry，两者不可互相代替。
- 2023 牌谱已被旧仓库用作 holdout、2024–2026 曾被 EX1 使用；不得声称“对所有历史研究都是 virgin”。
- 任何新 confirm 池：先预注册、再用新 key、再做全库审计、最后一次性消费；消费后本表 + split registry 同步更新。
- 已消费的墙池可以继续用于**诊断/趋势**（如池 A），但必须在报告中标注“非 confirm”，并全量报告所有重复运行。

---

## 7. 复现本表的命令（只读文件名，不读对局内容）

```bash
# 1) 全库对局文件名聚合（含 4 段式 self-play drain）
python3 - <<'PY'
import os, re
from collections import Counter, defaultdict
pat = re.compile(r"^(?:\d+_)?(\d+)_(\d+)_([abcd])\.json\.gz$")
buckets, keys = Counter(), Counter()
for dirpath, _, files in os.walk("runs"):
    for f in files:
        m = pat.match(f)
        if m:
            buckets[int(m.group(1)) // 1000 * 1000] += 1
            keys[int(m.group(2))] += 1
print(sorted(buckets.items()))
print({hex(k): v for k, v in keys.items()})
PY

# 2) gate 语料核对
python3 scripts/eval_report.py --games-dir runs/1v3 --seed-key 0x80a7b9e2dec96fa \
    --seed-range 10000-10500 --checkpoint at-175200 --source-dir runs/exp/EVAL_A_175200

# 3) 文档证据
grep -n "池 B\|70000\|120000\|60000" runs/1v3/PREREG.md docs/EXPERIMENTS.md \
     runs/exp/M3/PREREG.md runs/akochan_eval/TREND.md
```

> 本文件是 PR-00 的审计结论，不是统计结果。它不改变任何历史报告；它只规定“哪些窗口不能再被叫作 virgin”。
