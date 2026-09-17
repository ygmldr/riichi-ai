# 1v3 评估预注册(PREREG)— 12k 局协议

- 日期:2026-09-17
- 状态:**已冻结**。本文件是事前注册:结论只允许**追加**,不得改写既有条目。
- 配置来源:`configs/config.toml`(`[1v3]` / `[1v3.challenger]` / `[1v3.champion]`)

## 规模决策

**12000 局 = 3000 seeds × 4 splits**。1 个 seed = 1 个 set = 4 个半荘(split a/b/c/d 各 1 局),
故 `seeds_per_iter = games_per_iter // 4 = 3000`,seed 区间长度为 3000。

运行期验证(runner 自身代码路径,`one_vs_three.seed_interval`):

```
planned seed interval: [10000, 13000)
sets=3000  games=12000  games_per_iter=12000  iters=1
```

- `games_per_iter = 12000`
- `iters = 1`
- `seed_key = 0x5dfaa4cef265cd7`(十进制 `423244137402621143`,与 akochan 基准使用的固定 key 相同)
- `seed_start` 为新增配置键(补丁 `patches/0012-1v3-seed-start.patch`);缺省回退 `10000`(向后兼容)

## 种子池

| 池 | `seed_start` | seed 区间 | 用途 | 使用次数 |
|----|--------------|-----------|------|----------|
| A | `10000` | `[10000, 13000)` | **选择池**:12k 判定跑在这里 | 可重复(注:同一池重跑不得选择性汇报,需全量记录) |
| B | `50000` | `[50000, 53000)` | **最终确认池** | **恰好一次(EXACTLY ONCE)** |

- 池 B 不得用于调参、选择、迭代或"再看看";只有在池 A 结论落盘后才允许消费。
- 若池 B 被消费,必须在此文件追加一行记录(时间、命令、结果),不得回写。
- 关联:`runs/variance/` 将承载三个**互斥**噪声地板池 `pool_1|2|3`(`seed_start = 10000 / 20000 / 30000`,各 12k 局);
  池 B 在 `50000`,与上述四者均不相交(池 A 与 variance pool_1 同起点是计划预期,见 todo 6)。

## 评估对象

- **champion**:`/home/ygsldr/riichi-ai/checkpoints/mortal.pth.at-148800`(`name = "old148800"`,内部 save 步数 148,800)
- **challenger**:`/home/ygsldr/riichi-ai/runs/exp/PINNED/mortal.pth.at-175200`(`name = "mortal"`,快照自步数 175,200)
  - 大小 130,780,507 B;sha256 `f0a611b97f318e0b32fd7ce97965da0a9299b959742da54f64051677c7dc93a9`
  - **钉住快照,不可变**。禁止把 `[1v3.challenger].state_file` 指向 live `checkpoints/mortal.pth`(该文件已训练至约 328k 步)。
  - 方差标定(todo 6)、A1 实验(todo 12)等后续任务复用同一 `runs/exp/PINNED/` 快照。

## 每臂 `log_dir` 约定

- 单臂评估:`runs/exp/<ID>/games/`(例:todo 5 的 `runs/exp/EVAL_A_175200/games/`)
- 多臂对比:`runs/exp/<ID>/games_<arm>/`(例:todo 12 的 `runs/exp/A1/games_original|games_shuffle|games_recency_zero/`)
- **`runs/1v3/` 是冻结的 2004 文件对照语料**(2000 主 + 4 legacy `*.json.gz`),
  任何新评估一律不得把 `*.json.gz` 写进 `runs/1v3/`;只允许在此追加 `.md` / `.log` 类非对局文件。

## 运行命令

```bash
# 仓库根目录
source scripts/activate_env.sh          # .venv;PYTHONPATH 指向 release libriichi
bash scripts/run_eval.sh                # 默认 CFG=configs/config.toml
# run_eval.sh 内部: export MORTAL_CFG=<abs config>; cd vendor/mortal/mortal && python3 one_vs_three.py
# 评估日志(run_eval.sh 固定写入): runs/1v3/eval_<YYYYmmdd_HHMM>.log(.log,非对局文件)
```

12k 基准耗时参考:`eval_20260915_1835.log` 中 2000 半荘 = 8 分 49 秒,单卡外推 12k ≈ 1 小时量级。

## 已知跨任务影响

- `scripts/eval_akochan.sh`(todo 16 所有)内置字符串断言 `games_per_iter = 2000` 与 `seed_key = -1`;
  本配置变更后该脚本会断言失败,需由 todo 16 更新后才能使用。
- `scripts/run_eval.sh` 的 `LOG_DIR` 硬编码为 `runs/1v3`,每次评估会向冻结语料目录追加 1 个 `eval_*.log`
  (`.log` 不计入 `eval_report.py` 的 `ignored_files`,后者只统计 `*.json.gz`)。
- `seed_key = 0x5dfaa4cef265cd7` 与 `runs/1v3/` 中 4 个 legacy akochan 文件同 key,且池 A 起点同为 seed 10000;
  因各臂写入独立目录,不会污染冻结语料;但若未来把两个目录合并送进 `eval_report.py`,legacy 4 文件会与
  新评估的 `(10000, key)` set 归为同组——不要这么做。

## 结论(todo 5 — 池 A 12k 局;2026-09-17 追加)

- 运行:2026-09-17 14:01:15 → 14:33:51 CST(墙钟 **32m36s**;`dumping game logs` 14:33:45 → 14:33:51)。
  命令 `bash scripts/run_eval.sh configs/config.toml`,`[1v3].log_dir = runs/exp/EVAL_A_175200/games/`(本任务唯一改动);
  运行日志 `runs/1v3/eval_20260917_1401.log`(`run_eval.sh` 固定写向 `runs/1v3/`,非对局文件)。期间无其他 GPU 作业。
- 数据范围:seed `[10000, 13000)` × key `0x5dfaa4cef265cd7` = **3000 sets / 12000 局**
  (`runs/exp/EVAL_A_175200/report.csv`:sets=3000, games=12000, incomplete_sets=0, ignored_files=0)。
  arm:challenger = **`runs/exp/PINNED/mortal.pth.at-175200`**(步数 175,200 钉住快照,sha256 `f0a611b9…93a9`),
  champion = `checkpoints/mortal.pth.at-148800`;NDJSON 逐局核对:challenger 名 `mortal` ×12000,champion 均为 `old148800`。
- 结果(B=2000 set 级 bootstrap,固定 rng seed=0):挑战者顺位分布(1/2/3/4 位)**`[3010 3149 2950 2891]`**,
  **avg_rank = 2.4768**(95% CI **[2.4599, 2.4938]**);**pt = +1.8600**(95% CI **[+0.5663, +3.2026]**)。
  两个区间均不含 parity(avg_rank 2.5 / pt 0)。
- 判读:池 A 上 **at-175200 优于 at-148800**(95% CI 排除 parity),但幅度温和——avg_rank 仅比 parity 好 0.023。
  与 `runs/1v3/` 冻结语料的 2000 局历史对照(**不同 `seed_key`,不可直接比较**:当时 challenger 为 live
  `checkpoints/mortal.pth`,且 `eval_report.py` 的精确复现门只对 `runs/1v3` + `0x80a7b9e2dec96fa` + `10000-10500` 生效)。
- **caveat:池 A 是选择池(selection data)**——本结论可用于决策与迭代,但**不得**当作最终确认;
  最终确认必须**恰好一次**消费池 B(`seed_start=50000`,`[50000, 53000)`),池 B 至今未消费。
- 卫生:本臂 12000 个 `.json.gz`(492 MB)全部写入 `runs/exp/EVAL_A_175200/games/`;`runs/1v3/*.json.gz`
  运行前后均断言为 **2004**(冻结语料不变)。
