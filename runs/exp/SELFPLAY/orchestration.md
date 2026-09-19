# SELFPLAY 在线自对弈编排（plan todo 24）

产物：`runs/exp/SELFPLAY/{orchestration.md,worker_map.json,worker_<k>.toml,server.toml,trainer.toml,pids.json,logs/}`
配置改动：`configs/config.toml` 的 `[control].online = true`、`[freeze_bn].mortal = true`（备份 `configs/config.toml.sp-bak`，补丁 `patches/0009-online-selfplay-config.patch`）
启动器：`scripts/selfplay_launch.py`（prepare / launch / status / stop）

## 1. 数据流（未改 server 协议）

```
train.py (trainer.toml)                 server.py (server.toml)              K × client.py (worker_<k>.toml)
  submit_param(is_idle=True) ──┐              ┌── buffer_dir ── drain_dir ──→ drain() → 训练 → 回传参数
  训练 → submit_param ─────────┼── get_param ─┘                                   ↑
                               └── submit_replay ←──────── 每个 session 结束后提交 replay
```

- 每个 worker 一份配置副本，经环境变量 `MORTAL_CFG` 注入（`config.py` 只读该变量），因此 **每 worker 有独立的**
  `[baseline.train].state_file`（冻结对手快照）与 `[train_play.default].log_dir`（`runs/exp/SELFPLAY/worker_<k>/`）。
- trainee 占 1 席：`one_vs_three.rs` 的 challenger 每局 1 席、champion 3 席（split A–D 轮换）；在线模式下
  `train.py` 用 `player_names=['trainee']`（**不是 `[]`**），REINFORCE 只使用 trainee 自己的决策，对手的
  off-policy 决策不会折进损失。
- **无逐局快照轮转**：一个 worker 进程在生命周期内始终只加载配置里那一个对手快照；换对手 = 换 worker 配置重启进程。
- 冻结语料不受影响：本编排只写 `runs/exp/SELFPLAY/**`（scratch buffer/drain、scratch state/tb）。

## 2. 危险点：`player.py:137-138` 会 `shutil.rmtree(log_dir)`

训练玩家在每个 session 开始时**删除自己的 log_dir**。所以每个 worker 的 `log_dir` 必须是
`/home/ygsldr/riichi-ai/runs/exp/SELFPLAY/worker_<k>/` 下的**新目录**（不存在或为空）。
`prepare` 与 `launch` 都会打印断言（路径前缀 + fresh）并在不满足时拒绝启动；**绝不**指向
`runs/1v3/`、既有 `runs/exp/*` 臂、`runs/variance/`、`runs/akochan_eval/`。

## 3. 准备：生成 K 份配置 + worker_map.json

```bash
cd /home/ygsldr/riichi-ai
source scripts/activate_env.sh          # .venv: py3.12.3, torch 2.11.0+cu128
python3 scripts/selfplay_launch.py prepare --k 2 --games 100
```

本次 smoke（K=2）的 K 份配置路径：

| worker | 配置 | 冻结对手 `[baseline.train].state_file` | `log_dir` |
|---|---|---|---|
| 0 | `runs/exp/SELFPLAY/worker_0.toml` | `runs/exp/PINNED/mortal.pth.at-175200`（steps=175200, sha256 `f0a611b9…`） | `runs/exp/SELFPLAY/worker_0/` |
| 1 | `runs/exp/SELFPLAY/worker_1.toml` | `checkpoints/mortal.pth.at-148800`（steps=148800, sha256 `73ec8108…`） | `runs/exp/SELFPLAY/worker_1/` |

两份配置与仓库配置的差异**只有** 3 个键：`baseline.train.state_file`、`train_play.default.log_dir`、
`train_play.default.games`（脚本内断言 pairwise distinct 且每个 log_dir 都在 SELFPLAY 下）。
另有两份 scratch 配置：`server.toml`（buffer/drain 在 `runs/exp/SELFPLAY/server/`）、
`trainer.toml`（state/best/tb 在 `runs/exp/SELFPLAY/trainer/`，不写 live checkpoint）。
`worker_map.json` 记录 worker → 对手快照 → log_dir（字段含 `pool_member_id`/`opponent_state_file`/
`log_dir`/`games_per_session`，与 todo 25 的 pool.json 对齐）。

## 4. 启动（先持 `/tmp/opencode/train.lock`，一次只跑一个 GPU 作业）

```bash
cd /home/ygsldr/riichi-ai
source scripts/activate_env.sh
if mkdir /tmp/opencode/train.lock 2>/dev/null; then
    echo "$$ $(date -Is) selfplay-smoke" > /tmp/opencode/train.lock/owner.txt
else
    echo "BUSY: $(cat /tmp/opencode/train.lock/owner.txt)" && exit 1
fi
python3 scripts/selfplay_launch.py launch
```

`launch` 的启动顺序（全部 `start_new_session`，detached，日志在 `logs/`，PID 记入 `pids.json`）：

1. `server.py`（`MORTAL_CFG=runs/exp/SELFPLAY/server.toml`）→ 等 127.0.0.1:5000 可连；
2. `train.py`（`MORTAL_CFG=runs/exp/SELFPLAY/trainer.toml`）→ 等日志出现 `param has been submitted`
   （worker 只有在 server 有参数后才能 `get_param` 成功）；
3. K × `client.py`（`MORTAL_CFG=runs/exp/SELFPLAY/worker_<k>.toml`）。

手工等价命令（launcher 内部即此形式，cwd=`vendor/mortal/mortal`）：

```bash
MORTAL_CFG=/home/ygsldr/riichi-ai/runs/exp/SELFPLAY/server.toml  python3 server.py
MORTAL_CFG=/home/ygsldr/riichi-ai/runs/exp/SELFPLAY/trainer.toml python3 train.py
MORTAL_CFG=/home/ygsldr/riichi-ai/runs/exp/SELFPLAY/worker_0.toml python3 client.py
MORTAL_CFG=/home/ygsldr/riichi-ai/runs/exp/SELFPLAY/worker_1.toml python3 client.py
```

## 5. 监控

```bash
python3 scripts/selfplay_launch.py status   # PID 存活/RSS、每 worker 当前 session 文件数/已提交 session 数、buffer/drain 计数、GPU
```

注意：Rust arena 在**整个 session 结束时**才一次性 dump 日志，session 中途 `worker_<k>/` 为空属正常；
worker 进度看 `logs/worker_<k>.log` 的 `trainee rankings` / `logs have been submitted` 行。

## 6. 停止与清理

```bash
python3 scripts/selfplay_launch.py stop     # 对每个 pgid 发 SIGTERM，20s 后 SIGKILL，并校验 pgrep 无残留
pgrep -af 'python3 (server|client|train)\.py'   # 应为空
rm -f /tmp/opencode/train.lock/owner.txt && rmdir /tmp/opencode/train.lock
```

（`train.py` 在线模式由父进程再 spawn 一个子进程；同 pgid，`killpg` 一并回收。释放锁前必须先删 `owner.txt`，
否则 `rmdir` 因目录非空失败——T22 教训。）

## 7. 200 局 smoke 结果（2026-09-19 19:52–19:59 +0800）

- 每个 worker 完成 2 个 session × 100 局：worker_0（对手 at-175200）200 局、worker_1（对手 at-148800）200 局，**共 400 局**；
  其中 300 局走完"提交 → buffer → drain → 训练"管线，100 局（worker_1 的第 2 个 session）在提交前被 `SIGSTOP` 截住，
  作为 `worker_<k>/` 目录级证据保留。
- 提交/训练循环（trainer 日志）：`file list size: 200` → TRAIN 37 步 → `param has been submitted`（19:57:42）→
  再次 drain（100 局）→ 继续训练。worker 日志显示第二个 session 已取到更新后的参数（版本轮转正常）。
- 证据（`logs/`、`worker_<k>/`、`server/drain|buffer`）：见 `.omo/start-work/evidence/T24.txt`；
  每局 `start_game.names = ['trainee','baseline','baseline','baseline']`（trainee 1 席）。
- 资源：K=2 峰值 GPU **12,945 MiB / 16,303 MiB**（其中 ~4,074 MiB 是 WSL 基线幻影占用），
  MemAvailable 最低 **4.5 GB**；因此**保持 K=2**，K=3 未尝试（余量不足，OOM 风险）。逐进程 RSS：
  worker 各 ~1.9 GB、trainer 子进程 ~4.7 GB、server ~0.6 GB。冒烟期间无 OOM、无重启。
- 全程 `runs/1v3/*.json.gz` = 2004；`checkpoints/mortal.pth`/`best.pth` mtime 未变。

## 8. 交接给 todo 25 / 26

- **todo 25 已落地（见 `pool.md`）**：对手池 = `pool.json`（4 个本地快照 + 显式权重 + per-cycle assign，全为本地配置、无 server 协议改动）；`worker_map.json` 现为“最近一次 prepare 的 wave”，原 todo24 文件归档在 `worker_map.todo24.json`。`prepare` 需要 `--cycle G --wave W`（配置/日志写入 `cycle_<g>/wave_<w>/`），并可 `validate`/`sample`；`launch` 会断言 pool 完整性（缺失/哈希漂移即拒绝启动）与 `pool_source_sha256`（防止 prepare 后改池不重跑）。
- todo 26（A14）在 `trainer.toml` 基础上：`[rl].use_pg = true`（`lam_kl=0.05`、`lam_v=1.0` 已是仓库默认），
  `[control].state_file` 指向要续训的 mortal.pth 的 scratch 副本（**不要**写 live checkpoint），
  并把 `[train_play.default].games` 提到真实 session 预算；K 与对手权重按 `pool.md` 的池规则定
  （K=2 实测上限；cycle 内每成员出场一次，下一 cycle 用 `sample --cycle N --write` 生成）。
