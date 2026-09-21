# M3 预注册(PREREG)— 382k vs 328k 直接头对头(24k 局)

- 日期:2026-09-21(写入时间 **21:15:38 +0800**,先于任何 arena 启动)
- 状态:**已冻结**。本文件是事前注册:结论只允许**追加**,不得改写既有条目。
- 目的:判决"续训 54,000 步(328,000 → 382,000)**是否带来棋力提升**"——用**直接头对头**测量
  (382k 作 challenger vs 328k 作 champion),取代跨窗口/跨对手的间接比较。
- 纪律来源:`runs/1v3/PREREG.md`(池登记、"不得污染 `runs/1v3`")、`runs/variance/README.md`
  (噪声地板与 n-局数表)、`runs/exp/M328/REPORT.md`(前次 328k 测量;12k 分辨率不足)。

## 0. 一句话

24,000 局 / 6,000 sets 单次判决;事前声明的 80% 功效 **MDD = 0.0169 ≈ 0.017 顺位 / 1.315 ≈ 1.3 pt**;
判定规则 = 95% CI 是否排除 parity(challenger **avg_rank 2.5** / **pt 0**)。

## 1. 评估对象(两个 pin,只读)

| 臂 | 角色 | path | steps | bytes | sha256 |
|----|------|------|-------|-------|--------|
| at-382000 | **challenger** | `runs/exp/PINNED/mortal.pth.at-382000` | 382,000 | 130,781,915 | `fddf5c4859e0e5a39842177155065ab28f995b0c07da8820aea993abdff07dd3` |
| at-328000 | **champion** | `runs/exp/PINNED/mortal.pth.at-328000` | 328,000 | 130,780,507 | `b14e77ed9b79bdaa02ad21d269e378efb398e0aeec3721842975541c082a3511` |

- 两者均为纯 CNN `resnet{c=192,b=40}`、`version=4`;382k 由 328k 离线续训 +54,000 步得到
  (M2:15:18:32 → 19:00:10 存档,零损失停止;见 notepad learnings M2/M3 条目)。
- 两个 pin **只读**;禁止指向 live `checkpoints/mortal.pth`。
- 启动前两个 sha256 已逐字核对一致(见 §6)。

## 2. 协议

唯一改动 = `runs/exp/M3/eval.toml` 的下列键;其余与模板 `runs/exp/A7/eval.toml` **逐字一致**:

| 项 | 值 |
|----|----|
| `[1v3].seed_key` | `423244137402621143`(`0x5dfaa4cef265cd7`) |
| `[1v3].seed_start` | **120000** |
| seed 区间 | **`[120000, 126000)`** = 6,000 seeds |
| `[1v3].games_per_iter` / `iters` | **24000** / **1** |
| sets | **6000**(1 seed = 1 set = split a/b/c/d 共 4 局) |
| games | **24000** |
| `[1v3].log_dir` | `runs/exp/M3/games/` |
| challenger | `at-382000`(`name = "at382000"`) |
| champion | `at-328000`(`name = "at328000"`) |
| `[1v3.akochan].enabled` | `false`(保持模板值) |

- 布局固定为 1 challenger 席 vs 3 champion 席;每个 set 内两臂共享同一 seed(同牌局),
  "配对"即"同一批 set 内两臂结果";统计单元 = set(先对 4 局取均值)。
- **parity:** 若两臂等强,challenger 期望 avg_rank = **2.5**(1 席 vs 3 等价席),期望 pt = **0**
  (RANK_PTS 90/45/0/−135)。
- 统计口径:set 级均值 + 百分位 bootstrap(B=2000,rng seed=0),报告由**未修改**的
  `scripts/eval_report.py` 生成(与池 A/池 B/方差研究完全同一口径)。

## 3. 噪声地板与事前 MDD(来自 `runs/variance/README.md`)

三互斥 12k 池实测合并值(同协议、同 seed_key、同 champion 家族;是发布前可得的最佳地板):

- σ_set(avg_rank) = **0.467788**(CSV 反推;精确重放 0.475096,交叉验证差 1.5%);
  σ_set(pt) = **36.368346**。
- n = 24000 局 = 6000 sets:SE = 2σ/√n ⇒ rank **0.006039**、pt **0.469512**。
- 95% CI 半宽:rank **±0.01184**、pt **±0.92023**。
- 80% 功效 MDD(`(z₀.₉₇₅+z₀.₈₀)=2.801585`,双侧 α=0.05):
  **rank = 0.016919 ≈ 0.017 顺位**、**pt = 1.315374 ≈ 1.3 pt**
  (精确重放 σ 版 rank 0.01718;两者同报 ≈0.017)。
- 与 12k 对照:12k MDD = 0.0239 / 1.86 ⇒ 24k 把可分辨粒度压到 **≈0.7×**。

历史参照(不当作判据):M328 在池 A 12k 的配对 Δ(328k−175.2k)= **+0.0036 顺位 [−0.0179,+0.0270]**、
−0.87 pt [−2.655,+0.803],两 CI 均跨 0 ⇒ 12k 下 148.8k/175.2k/328k 互相不可分辨。M3 的分辨率(0.017)
低于 M328 的 CI 半宽(0.0225),故若 +54k 步有 ≥0.017 顺位量级的真实效应,本设计应能检出。

## 4. 判定规则(预注册,不得事后改门)

**主判据 = challenger `avg_rank` 的 95% CI 是否排除 parity 2.5:**

1. CI 完全 **< 2.5** ⇒ 判定 **382k 有可检测提升**(续训有增益);
2. CI 完全 **> 2.5** ⇒ 判定 **382k 有可检测下降**(续训有害);
3. CI **含 2.5** ⇒ 判定 **"no detectable difference at 24k"**(不写成"无差异");
   必须同时报告 `|点估计| / MDD` 比值与 CI 宽度,并注明"小于 MDD 的效应不可分辨"。

**副判据 = pt 的 95% CI 是否排除 0**(方向与 rank 一致时作为支持证据;冲突时以 avg_rank 为主并如实标注)。

**报告必备项:** 顺位分布(1/2/3/4)、sets/games/incomplete_sets/ignored_files、
bootstrap 参数、两指标的 95% CI 与 parity 的关系、以及 NDJSON 逐局 arm 名核对(`at382000`/`at328000`)。

## 5. 窗口来源论证:120000 未被任何既有实验消费

已消费窗口(截至 2026-09-21;除注明外 key 均为 `0x5dfaa4cef265cd7`):

| 窗口 | 消费方(证据) |
|------|----------------|
| `[10000,13000)` | 池 A / variance pool_1 / A1 original;A7/A8/A9/E2/E3a/M328 各臂 |
| `[20000,23000)` | variance pool_2 |
| `[30000,33000)` | variance pool_3 / A1 original |
| `40010` 起(单点) | akochan 尺子(`runs/akochan_eval/config.at-175200.toml:66`) |
| `[50000,53000)` | 池 B(T22 / ARCH_B 单次确认消费) |
| `60000` 一带(分块) | akochan 趋势尺(T16;`runs/akochan_eval/TREND.md`) |
| `[70000,73000)` | A14 确认窗(todo 26) |

- `120000` **不在任何一行**;`[120000,126000)` 与最接近的已消费窗口 `[70000,73000)` 相距 47,000 seeds,
  **零交集**。
- 全库文本审计(启动前):`grep -rn "120000" --include="*.toml|*.md|*.py|*.sh|*.json|*.txt|*.csv"`
  (范围 `runs/ configs/ scripts/ docs/ .omo/ patches/`)⇒ **0 命中**。
- 特别声明:**不得**改用 40000 一带——`40010` 已被 `runs/akochan_eval/config.at-175200.toml` 占用。
- 因此本窗口是**新鲜、未被选择过程看过**的数据;按**确认性**测量对待:
  结果落盘前不得调参/换种子重跑;本窗口 **EXACTLY ONCE**。

## 6. 启动前基线证据(2026-09-21,启动前采集)

| 项 | 值 |
|----|----|
| `ls runs/1v3/*.json.gz \| wc -l` | **2004**(冻结语料;主臂 2000 + 4 legacy) |
| sha256 `runs/exp/PINNED/mortal.pth.at-382000` | `fddf5c4859e0e5a39842177155065ab28f995b0c07da8820aea993abdff07dd3`(**匹配期望**) |
| sha256 `runs/exp/PINNED/mortal.pth.at-328000` | `b14e77ed9b79bdaa02ad21d269e378efb398e0aeec3721842975541c082a3511`(**匹配期望**) |
| `stat -c '%y %s' checkpoints/file_index.pth` | mtime `2026-09-17 16:39:22.835975729 +0800`,size `185010303`(评估不得写它) |
| 运行中的 arena | 无(`pgrep -af one_vs_three` 仅匹配命令自身) |

## 7. 操作约束(MUST NOT)

- 不得把任何 `*.json.gz` 写进 `runs/1v3/`(只允许追加 `.md`/`.log`);
  `scripts/run_eval.sh` 固定追加 `runs/1v3/eval_<ts>.log`(非对局文件,允许)。
- 不得修改 `configs/config.toml`、`checkpoints/mortal.pth`、`checkpoints/baseline.pth`、
  `runs/exp/PINNED/*`、任何 `scripts/*.py`(尤其 `scripts/eval_report.py` 与 `scripts/run_eval.sh` 原样)。
- 单 GPU 串行:运行期间不得启动第二个 arena 或训练。
- 预期墙钟:12,000 局 ≈33 min ⇒ **24,000 局 ≈66–75 min**(arena 只在整批结束时 dump 牌谱,
  运行中途 `games/` 文件数可能长时间为 0,属正常)。

## 8. 运行记录(启动后**追加**,只允许追加)

- (待追加:启动时间、launcher PID、arena PID、日志路径、runner 打印的 seed 区间/sets/games、
  实测速率与 ETA、启动后 1v3 计数复核。)

### [2026-09-21 21:16:22 +0800] 启动记录（追加）

- 启动命令：`setsid nohup bash scripts/run_eval.sh runs/exp/M3/eval.toml > runs/exp/M3/nohup.log 2>&1 < /dev/null &`（仓库根；`< /dev/null` 为仓库既有 ops 硬化，不影响语义）
- 进程链（实测）：launcher `9938`（setsid，`eval.pid` 初值）→ 同一 PID 为 `bash scripts/run_eval.sh`（session leader）→
  **arena PID `9976` = `python3 one_vs_three.py`**（pgid 9938）；`tee` PID 9977 → `runs/1v3/eval_20260921_2116.log`。
  `runs/exp/M3/eval.pid` 已按 `runs/variance/` 约定改记为真实 arena PID **9976**。
- runner 自身打印（逐字）：
  `2026-09-21 21:16:27,095     INFO one_vs_three.rs:131  seed: [120000, 126000) w/ 0x5dfaa4cef265cd7, start 6000 sets, 24000 hanchans`
  ⇒ seed 区间 / sets / games 与 §2 完全一致。
- **文件增长口径修正（提前声明，防事后误判）**：`vendor/mortal/libriichi/src/arena/one_vs_three.rs` 的 `run_batch` 在
  `batch_game.run()`（打完**整批** 24000 局）之后才打印 `dumping game logs` 并一次性落盘；本次 `iters=1` 只有一个批次。
  历史同管线日志（A7 / pool_1 / M328）在非 TTY 下无进度条输出（首行 seed 之后直接是 dump 行）。
  ⇒ play 阶段 `runs/exp/M3/games/` 文件数**恒为 0 属预期行为**，不能作为速率判据；牌谱将在结束后数秒~数十秒内齐现。
- 活动实测（启动后 1–2.5 min）：`/proc/9976/stat` CPU 时间 21:17:27 ≈487 s → 21:18:34 ≈1,027 s（60 s 窗口 +541 s ≈ 8.1 核满载）；
  21:18:54 `TIME=00:20:07`；state 在 S/D 间切换；39 threads；VmRSS ≈ 8.9 GB；GPU（WSL 口径）util 7%→15%。
- 速率 / ETA（同协议、同 key、单卡串行的历史实测；本 run 的 live 速率不可直读）：A7 12k = **32m47s**（≈6.10 games/s）；
  pool_1 12k = **35m00s**（≈5.71 games/s）；T22 E3a 12k = 35m49s ⇒ 24k ≈ **66–75 min**，预计 `dumping game logs` ≈ **22:22–22:31 CST**。
- 启动前后不变量：`runs/1v3/*.json.gz` = **2004 → 2004**；`runs/exp/M3/games/*.json.gz` = **0**（见上）；
  `checkpoints/file_index.pth` mtime/size 未变（`2026-09-17 16:39:22.835975729 +0800` / `185010303`）；未启动训练 / 第二个 arena。
