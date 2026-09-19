# SELFPLAY 对手池：成员、权重与轮换（plan todo 25）

单一事实源：`runs/exp/SELFPLAY/pool.json`（成员 + 显式权重 + assign 计划）。
启动器：`scripts/selfplay_launch.py`（`validate` / `sample` / `prepare` / `launch` / `status` / `stop`）。
本轮 smoke 产物：`runs/exp/SELFPLAY/cycle_0/wave_{0,1}/`（早期生成代保留在 `cycle_0/history/`）。

## 1. 池成员（仅本地快照；无启发式对手；不改 server 协议）

| id | 路径 | steps | bytes | sha256 前 12 | 状态 | 出处 |
|---|---|---:|---:|---|---|---|
| `best-220000` | `checkpoints/best.pth` | 220000 | 130780507 | `7fe084d98f84` | active | 本仓训练 run 的 best（2026-09-16 02:59:45），池内最新 |
| `at-175200` | `runs/exp/PINNED/mortal.pth.at-175200` | 175200 | 130780507 | `f0a611b97f31` | active | 175.2k 快照的 PINNED 副本（2026-09-17 13:55）；与 `checkpoints/mortal.pth.at-175200` 逐字节相同 |
| `at-148800` | `checkpoints/mortal.pth.at-148800` | 148800 | 130780507 | `73ec8108f1b9` | active | 148.8k 快照；同时是 `[rl].anchor_state_file` KL 锚与 1v3/pool-B champion |
| `baseline` | `checkpoints/baseline.pth` | — | 43691001 | `83fd3929035d` | active | `tag='random-baseline'`（2026-08-10），仅含 config/mortal/current_dqn 的权重导出；历史冻结对手，最弱成员（地板锚） |

- **别名（不是成员）**：`checkpoints/mortal.pth.at-175200` 与 `at-175200` sha256 相同；重复成员会重复采样同一权重，故只登记在 `aliases`。
- **排除**：`checkpoints/mortal.pth`（live、328k 步、mtime 2026-09-17 02:03:09，**禁止**当对手或任何 `state_file` 目标）、`checkpoints/grp.pth`（只有 DQN 键，无法作为 Mortal 对手加载）、`file_index.pth`/`grp_file_index.pth`（数据集索引）。
- 枚举时间与完整 sha256 见 `pool.json` 的 `members`/`excluded` 字段。

## 2. 权重规则（显式概率，和 = 1；可测试）

- 公式：`w_i = 2 ** (-rank_i / half_life_ranks)`，参数 `half_life_ranks = 1.0`；`rank` 按快照 steps 降序（无 steps 者排在最后，用 mtime 打破平局）。
- 数值（存进 `pool.json`，不由运行期计算；`validate` 断言 `|sum-1| <= 1e-6`）：

| id | rank | 显式权重 |
|---|---:|---:|
| `best-220000` | 0 | 0.533333 |
| `at-175200` | 1 | 0.266667 |
| `at-148800` | 2 | 0.133333 |
| `baseline` | 3 | 0.066667 |
| | 合计 | **1.000000** |

- 采样器：`weighted_without_replacement_sequential`，`seed=20260925`，`rng=random.Random(f'{seed}:cycle:{cycle}')`。
  一个 **cycle** = `ceil(N_active / k)` 个 wave：从剩余集合按 `w_i / sum(w_remaining)` 逐个抽签、抽出即移除，直到全部 active 各出现一次；抽签序列按 k 切段即各 wave。
  ⇒ wave 内成员互异；每个 active 成员在每个 cycle **恰好出场一次**。`worker_order`：抽签位置 = worker 下标。
- `k_per_wave=2`（K=2 是实测显存上限：峰值 ~12.9 GiB / MemAvailable 低 4.5 GB，见 todo 24）。
- 已存储计划：`assign.cycle_0 = [["at-175200","at-148800"], ["best-220000","baseline"]]`。
  复现命令 `python3 scripts/selfplay_launch.py sample --cycle 0` 的输出与存储逐字一致（`pool.json` 中 `assign` 即采样器产物，不是手写例外）。

## 3. 新快照晋升 / 成员退役

`promotion` 规则（pool.json 内为结构化步骤）：

1. 新快照冻结到本地路径（如 `runs/exp/PINNED/mortal.pth.at-<steps>`），记录 `bytes` + `sha256`（**不得**用 live 路径）；
2. 以 `status=active` 追加，填 `steps/mtime/provenance`；
3. 用同一 `rank_key` 对“新快照 + 现役成员”重排名，用同一公式重算原始权重；
4. 若 active 数 > `max_active=4`：**退役权重最低者**（`status=retired`、`retired_at`、`retired_reason='displaced by <new-id> (max_active=4)'`）；文件保留在盘上；
5. 对剩余 active 重新归一化并写入显式权重（和必须为 1）；
6. 为**下一个** cycle 采样新 assign（进行中的 cycle 保持已存计划不变）；更新 `cycle_length_waves`。

示例：晋升 250k 快照 ⇒ 排名 `[new, best-220000, at-175200, at-148800, baseline]`，退役 `baseline`，存留 4 人权重变为 `{new:0.533333, best-220000:0.266667, at-175200:0.133333, at-148800:0.066667}`。

退役成员保留 path/bytes/sha256，权重视为 0、采样器跳过；重新启用 = 显式改 `status` 后重排名/重算权重/重新采样后续 cycle。

## 4. 缺失或损坏快照 = 启动即硬失败（禁止静默降级）

`validate` / `prepare` / `launch` 都先跑 `validate_pool`：任一 active 成员**文件缺失、bytes 不符、sha256 不符**，或权重和 ≠ 1，即打印 `[FAIL]` 明细并以非 0 退出（`REFUSING: pool validation failed`）。演示（证据 T25 §D，均 rc=1）：
- 缺失：`best-220000 -> /tmp/.../does-not-exist.pth`；
- 哈希漂移：把 `at-175200` 的 sha256 改成全 0；
- 字节漂移：把 `baseline` 路径换成 `best.pth`。

`launch` 额外断言根 `worker_map.json` 里的 `pool_source_sha256` 与当前 `pool.json` 一致——prepare 之后改池不重跑会被拒绝。

## 5. 与 `worker_map.json` 的兼容（todo 24 → todo 25）

- todo 24 的根 `worker_map.json` 已归档为 **`worker_map.todo24.json`**（首次 `prepare` 自动逐字节备份，其后不再改动）。
- 根 `worker_map.json` 仍保留原 schema 的全部键与语义：`experiment/todo/k/games_per_session/trainee/opponent_seats_per_game/pool/workers`；新增 `cycle/wave/pool_source/pool_source_sha256/logs_dir/configs/assign`；`pool[]` 增 `sha256/steps/status/weight`；`workers[]` 增 `opponent_sha256`。每个 wave 目录内另有一份同内容的 `worker_map.json`。**注意**：根文件现在表示“最近一次 prepare 的 wave”，历史 wave 的映射看各 wave 目录。
- `pids.json` 结构：进程 PID 全部在顶层已知角色键（`server`/`trainer`/`worker_<k>`）下；`run.{cycle,wave}` 为元数据。`status`/`stop` 只把已知角色键当 PID（防 `killpg(0)` 事故）。

## 6. 目录布局与命令

```
runs/exp/SELFPLAY/
  pool.json
  worker_map.json            # 最近一次 prepare 的 wave（兼容 todo24 schema + 扩展）
  worker_map.todo24.json     # todo24 归档
  cycle_0/
    history/                 # 早期 smoke 生成代（保留）
    wave_0/{worker_0.toml,worker_1.toml,worker_0/,worker_1/,server.toml,trainer.toml,logs/,server/{buffer,drain},trainer/,pids.json,worker_map.json}
    wave_1/...
```

```bash
source scripts/activate_env.sh
python3 scripts/selfplay_launch.py validate --pool runs/exp/SELFPLAY/pool.json
python3 scripts/selfplay_launch.py sample  --cycle 1 --k 2 --write   # 为下一 cycle 生成/落盘 assign
python3 scripts/selfplay_launch.py prepare --cycle 1 --wave 0 --k 2 --games 4
python3 scripts/selfplay_launch.py launch      # 需先持 /tmp/opencode/train.lock
python3 scripts/selfplay_launch.py status
python3 scripts/selfplay_launch.py stop        # 释放锁：先 rm owner.txt 再 rmdir
```

轮换语义：一个 worker 进程生命周期内只加载配置里的那一个对手快照（**无逐局轮转**）；换对手 = 下一次 prepare/launch（新 wave 或新 cycle）。

## 7. 配置卫生（live checkpoint 不进任何 state_file）

- worker 与 server 配置中的 `[control].state_file / best_state_file / tensorboard_dir` 一律指向 wave 内 `unused-*.pth` / `unused-*-tb` scratch；trainer 的 `[control]` 指向 wave 内 `trainer/state.pth`。
- 依据：`client.py` 只读 `[control].device/version`；`player.py` 只读 `version/enable_cudnn_benchmark`；`server.py` **完全不引用** `control` 或 `state_file`（`grep -n "control\|state_file" server.py` 无命中，rc=1）。唯一真正被加载为对手的 state_file 是池成员快照。
- 最终 smoke 配置扫描：`grep -rn 'checkpoints/mortal.pth"' cycle_0/wave_*` 无命中（`best.pth` 只以“成员对手”身份出现）。

## 8. smoke 证据摘要（2026-09-19 20:39–20:42 +0800，K=2，games/session=4）

| cycle/wave | worker | 成员 | sha256 前 12 | 产出目录 | 局数 | 日志行 |
|---|---|---|---|---|---|---|
| 0/0 | 0 | `at-175200` | `f0a611b97f31` | `cycle_0/wave_0/worker_0/` | 4 | `trainee rankings` + `logs have been submitted` |
| 0/0 | 1 | `at-148800` | `73ec8108f1b9` | `cycle_0/wave_0/worker_1/` | 4 | 同上 |
| 0/1 | 0 | `best-220000` | `7fe084d98f84` | `cycle_0/wave_1/worker_0/` | 4 | 同上 |
| 0/1 | 1 | `baseline` | `83fd3929035d` | `cycle_0/wave_1/worker_1/` | 4 | 同上 |

覆盖断言（脚本）：`assign.cycle_0` 恰为 active 集合的一次全排列；每个成员在产物目录中有 4 个 `*.json.gz`（`zcat -t` 全部通过）。每局 `start_game.names = ['trainee','baseline','baseline','baseline']`（受训方 1 席）。

## 9. 冻结不变量（每次运行前后核对）

- `ls runs/1v3/*.json.gz | wc -l` == **2004**；
- `checkpoints/mortal.pth` mtime `2026-09-17 02:03:09`、`checkpoints/best.pth` mtime `2026-09-16 02:59:45` 不变；
- 本池只写 `runs/exp/SELFPLAY/**`；从不写 live checkpoint / `runs/1v3/` / 其他臂目录。
