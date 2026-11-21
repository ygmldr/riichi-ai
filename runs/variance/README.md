# 评估噪声地板 —— 三互斥种子池标定(todo 6)

- 日期:2026-09-17
- 来源:`.omo/plans/fork-roadmap-steps-1-10.md` todo 6;`runs/1v3/PREREG.md`(池定义)
- 目的:同一对选手、同一 12k 局协议下,**换一批互斥种子重跑**,量化"12k 判定"有多可靠,
  并给出"给定 n 局可分辨效应"表,供后续实验选 n。
- 一句话结论:同一挑战者的三池均值在 avg_rank 2.4768→2.5053、pt +1.86→−0.16 之间摆动;
  12k 局上可分辨的效应约 **Δavg_rank ≈ 0.024 / Δpt ≈ 1.86**(80% 功效),池 A 观察到的效应恰好在这个边界上。

## 1. 协议(固定不变项)

| 项 | 值 |
|----|----|
| challenger | `runs/exp/PINNED/mortal.pth.at-175200`(name `mortal`,sha256 `f0a611b9…93a9`) |
| champion | `checkpoints/mortal.pth.at-148800`(name `old148800`) |
| `[1v3].seed_key` | `0x5dfaa4cef265cd7`(十进制 `423244137402621143`) |
| `[1v3].games_per_iter` / `iters` | `12000` / `1` |
| set 定义 | 1 seed = 1 set = split a/b/c/d 共 4 局;3000 seeds = 12000 局 |
| 每池唯一改动 | `[1v3].log_dir`、`[1v3].seed_start` |

三个池互斥(seed 区间不相交),已用文件名与 NDJSON 逐项核对(`runs/variance/pool_disjointness.txt`):
pool_1 ∩ pool_2 = pool_1 ∩ pool_3 = pool_2 ∩ pool_3 = **0 seeds**;每池 `unique_seeds=3000`,`key` 唯一,`splits=abcd` 齐全。

## 2. 运行记录(串行执行,任一时刻只有 1 个 arena)

命令模板(每池相同,只改 config 两行):

```bash
cd /home/ygsldr/riichi-ai
mkdir -p runs/variance/pool_<N>
setsid nohup bash scripts/run_eval.sh configs/config.toml > runs/variance/pool_<N>/nohup.log 2>&1 &
echo $! > runs/variance/pool_<N>/eval.pid    # 事后替换为 arena(python) 真实 PID
```

| 池 | `seed_start` | seed 区间 | 开始→结束(CST) | 墙钟 | eval 日志 |
|----|--------------|-----------|------------------|------|-----------|
| pool_1 | `10000` | `[10000,13000)` | 14:40:55 → 15:16:05 | **35m10s** | `runs/1v3/eval_20260917_1440.log` |
| pool_2 | `20000` | `[20000,23000)` | 15:17:16 → 15:50:08 | **32m52s** | `runs/1v3/eval_20260917_1517.log` |
| pool_3 | `30000` | `[30000,33000)` | 15:51:07 → 16:21:40 | **30m33s** | `runs/1v3/eval_20260917_1551.log` |

- 均值 32m52s,与池 A 参考值 32m36s 一致。三池运行期间各有一个 `nice -n 19` 的单核重放作业并行,
  未观察到 VRAM/RAM 压力(16 GB VRAM 只跑单 arena);墙钟波动在±7% 内。
- 每池 `games/` 均为 12000 个 `*.json.gz`;`incomplete_sets=0`,`ignored_files=0`。

## 3. 三池结果(B=2000 set 级 bootstrap,固定 rng seed=0)

报告由 `scripts/eval_report.py` 生成,命令行中的 `--seed-range` 与池区间一致
(例 pool_2:`--seed-range=20000-23000`)。

| 池 | sets/games | 顺位分布(1/2/3/4) | avg_rank | avg_rank 95%CI | pt | pt 95%CI |
|----|------------|--------------------|----------|----------------|-----|----------|
| pool_1 | 3000/12000 | `[3010 3149 2950 2891]` | 2.476833 | `[2.459912, 2.493754]` | **+1.860000** | `[+0.566250, +3.202594]` |
| pool_2 | 3000/12000 | `[2952 3080 3025 2943]` | 2.496583 | `[2.479748, 2.513252]` | **+0.581250** | `[-0.701344, +1.901250]` |
| pool_3 | 3000/12000 | `[2975 2977 3058 2990]` | 2.505250 | `[2.487917, 2.521002]` | **−0.161250** | `[-1.436438, +1.132594]` |
| 池 A(参考,已有) | 3000/12000 | `[3010 3149 2950 2891]` | 2.476833 | `[2.459912, 2.493754]` | +1.860000 | `[+0.566250, +3.202594]` |

判读(诚实汇报):
- **pool_1 与池 A 完全一致**(同 seed 区间,确定性,见 §5)——pool_1 不是池 A 的独立复现,只作确定性对照。
- pool_2/pool_3 的 avg_rank 与 pt 的 95% CI **都包含 parity**(2.5 / 0);池 A 那条"挑战者更优且 CI 排除 parity"的
  结论在两批新种子上**没有重现**。
- 三池均值散布(样本 SD):avg_rank `0.014564`(预测 SE `0.008674`,比值 1.68,χ² 检验 p≈0.060);
  pt `1.022412`(预测 SE `0.663756`,比值 1.54,p≈0.093)。3 个池子检验力弱,**没有**实质证据表明池间有系统性差异,
  但也不能排除"种子窗口异质性"让真实池间方差略大于池内 σ 的估计——后续若做多池合并,需记住这一点。
- 极差:avg_rank `2.505250−2.476833 = 0.028417`;pt `+1.86−(−0.16125) = 2.021250`。

## 4. 噪声地板:set 级 SD

口径:统计量按 **set** 计算(每个 set 先对 4 局取均值),set 才是独立重采样单元;
所以"n 局"= n/4 个 set,且组内相关已被 set 均值吸收。

### 4.1 精确重放(ddof=1,全量 12000 局/池)

命令(逐池):

```bash
python3 runs/variance/measure_sd.py --games-dir runs/variance/pool_1/games \
    --seed-key 0x5dfaa4cef265cd7 --seed-range 10000-13000      # pool_2/3 改目录与区间
```

| 池 | σ_set(avg_rank) | σ_set(pt) |
|----|-----------------|-----------|
| pool_1 | 0.477185 | 36.509978 |
| pool_2 | 0.468818 | 35.949106 |
| pool_3 | 0.479221 | 36.603709 |
| **合并(等权 RMS)** | **0.475096** | **36.355413** |

### 4.2 由 `report.csv` 反推(可仅凭 CSV 复算)

bootstrap 百分位区间宽度 ≈ `2·z₀.₉₇₅·SE`,`SE = σ_set/√sets`,故
`σ_set ≈ (ci_hi − ci_lo)·√sets / (2·z₀.₉₇₅)`(`z₀.₉₇₅ = 1.959964`)。

| 池 | σ_set(avg_rank) | σ_set(pt) |
|----|-----------------|-----------|
| pool_1 | 0.472862 | 36.837027 |
| pool_2 | 0.468146 | 36.365446 |
| pool_3 | 0.462295 | 35.896485 |
| **合并(等权 RMS)** | **0.467788** | **36.368346** |

两种口径一致性:rank 差 ~1.5%,pt 差 ~0.04%。下表(§4.3)用 **CSV 反推的合并 σ**,以便仅凭三份
`report.csv` 复算;§4.4 给出精确重放版本作为交叉验证。

## 5. 给定 n 局可分辨效应表

公式(全部量都以 set 为单元,n 为局数):

```
σ_set  = set 级统计量的样本 SD(每 set 已对 4 局取均值)
SE(n)  = σ_set / sqrt(n/4) = 2·σ_set / sqrt(n)                    # n 局均值的标准误
±95%   = z₀.₉₇₅ · SE(n),           z₀.₉₇₅ = 1.959964               # 95% 置信半宽
Δ_MDD  = (z₀.₉₇₅ + z₀.₈₀) · SE(n) = 2.801585 · SE(n)              # 双侧 α=0.05、功效 80% 的最小可分辨效应
```

`Δ_MDD` 是"与一个固定参考值(如 parity)比较"的口径;若是**两个各跑 n 局的独立臂互比**,把 SE 再乘 `√2`。

### 5.1 avg_rank 标尺(合并 σ_set = 0.467788)

| n 局 | SE(n) | ±95% CI | Δ_MDD(80% 功效) |
|------|-------|---------|------------------|
| 6100 | 0.01198 | ±0.02348 | **0.03356** |
| 12000 | 0.00854 | ±0.01674 | **0.02393** |
| 24000 | 0.00604 | ±0.01184 | **0.01692** |

反解所需局数:`n ≈ 6.8701 / Δ_avg_rank²`(如 Δ=0.02 → 17175 局;Δ=0.05 → 2748 局)。

### 5.2 pt 标尺(合并 σ_set = 36.368346,单位 = RANK_PTS 90/45/0/−135)

| n 局 | SE(n) | ±95% CI | Δ_MDD(80% 功效) |
|------|-------|---------|------------------|
| 6100 | 0.93130 | ±1.82531 | **2.60911** |
| 12000 | 0.66399 | ±1.30140 | **1.86023** |
| 24000 | 0.46951 | ±0.92023 | **1.31538** |

反解所需局数:`n ≈ 41525.5 / Δ_pt²`(如 Δ=2 pt → 10381 局;Δ=5 pt → 1661 局)。

### 5.3 复算命令(仅用 CSV,可逐格复现上表)

```bash
cd /home/ygsldr/riichi-ai
python3 runs/variance/measure_sd.py --pooled-csv \
    runs/variance/pool_1/report.csv runs/variance/pool_2/report.csv runs/variance/pool_3/report.csv
# 输出存档: runs/variance/pooled_table.txt
```

### 5.4 精确重放版交叉验证(σ 来自 §4.1)

| n 局 | Δ_MDD avg_rank | Δ_MDD pt |
|------|----------------|----------|
| 6100 | 0.03408 | 2.60818 |
| 12000 | 0.02430 | 1.85957 |
| 24000 | 0.01718 | 1.31491 |

与 §5.1/§5.2 相比,最大格差 +1.5%(rank@12000),pt 差 −0.04%:
CSV 反推的近似足以支撑决策;需要更精确时用 §4.1 的逐池重放。

### 5.5 与继承规划数字的关系

- 研究阶段的规划口径(继承,非本任务测量):`n ≈ 10.3 / Δ_avg_rank²`,per-set `σ ≈ 85 pt`。
  在本文约定下,`10.3/Δ²` 对应 `σ_set(avg_rank) ≈ 0.573`(比实测 0.4751 / 0.4678 大约 +21%),
  即**继承估值偏保守**,实测噪声地板略好于计划。
- 继承的 `σ≈85 pt` 未注明 pt 标尺,与本文 `σ_set(pt)=36.37`(RANK_PTS 90/45/0/−135)不可直接比较,只作数量级对照。
- 结论:本 README 的表全部来自本次三池实测(§4.1/§4.2),继承数字仅作 cross-check。

### 5.6 对照池 A 判定

池 A 观察效应:avg_rank 偏离 parity `2.5−2.476833 = +0.023167`(≈0.97× Δ_MDD(12000)),pt `+1.86`(≈1.00× Δ_MDD(12000))。
即"池 A 的挑战者优势"**正好落在 12k 局可分辨边界上**;换两批种子(pool_2/3)后点估计落到 parity 两侧
且 CI 含 parity。因此:**12k 局不足以对 <0.024 avg_rank / <1.86 pt 的真实效应给出稳定判定**;
若要可靠分辨池 A 量级的效应,按 §5.1 需要约 17k 局以上(或合并多个互斥池)。

## 6. 确定性交叉核对:pool_1 vs 池 A(同 seed 区间、同 key、同选手)

这是本任务的 first-class finding。

**结论:评估管线是确定性的(逐局语义完全一致);唯一的运行间差异是壁钟计时字段 `meta.eval_time_ns`。**

证据(`runs/variance/determinism_pool1_vs_poolA.txt`):

| 层次 | 结果 |
|------|------|
| 文件名集合 | 12000 vs 12000,**完全相同** |
| 原始 `.json.gz` sha256 | 12000/12000 **全部不同**(唯一原因见下) |
| 去掉 `meta.eval_time_ns` 后的逐事件规范化 sha256 | 12000/12000 **全部一致** |
| `report.csv` 数值字段 | 全部一致(sets/games/incomplete/dist/avg_rank/CI/pt/CI/ignored) |

- 抽样单局 diff 显示:每处差异都是 `"eval_time_ns":<ns>`(单次推理的壁钟纳秒),而
  `pai`/`tsumogiri`/`q_values`/`mask_bits`/`shanten`/`deltas`/`hori` 等全部逐字节相同,1105 行行数也相同。
- 报告的标签列差异仅两处:`source_dir`(`runs/variance/pool_1/games` vs `runs/exp/EVAL_A_175200`)、
  `checkpoint`(`-` vs `at-175200`)——属于写 CSV 时传入的标签,非统计量。
- 复算命令:

```bash
cd /home/ygsldr/riichi-ai
python3 runs/variance/determinism_check.py runs/variance/pool_1/games runs/exp/EVAL_A_175200/games
# 输出存档: runs/variance/determinism_pool1_vs_poolA.txt ;退出码 0 = 全部一致
```

含义:(a) 同 seed 重跑可用于故障复现与审计,不需要担心漂移;(b) 池 A 与 pool_1 是同一份数据的两次落盘,
**不是独立复现**;(c) 想量化噪声必须换互斥种子(本任务的 pool_2/pool_3 就是)。

## 7. 冻结语料不变量

`ls runs/1v3/*.json.gz | wc -l` 在三池运行前后与期间逐次检查:

| 时点 | 计数 |
|------|------|
| 每池启动前 | 2004 |
| `pool_1` 结束后 | 2004 |
| `pool_2` 结束后 | 2004 |
| `pool_3` 结束后 | 2004 |

新对局全部写入 `runs/variance/pool_<N>/games/`;`run_eval.sh` 只向 `runs/1v3/` 追加 `.log`(非对局文件),
符合 PREREG「不得把 `*.json.gz` 写进 `runs/1v3/`」的约束。

## 8. 产物与复算清单

| 文件 | 内容 |
|------|------|
| `runs/variance/pool_<N>/games/` | 12000 个 `*.json.gz`(每池) |
| `runs/variance/pool_<N>/report.csv` | `eval_report.py` 报告(本 README 表 3) |
| `runs/variance/pool_<N>/report_stdout.txt` | 报告终端输出(含 bootstrap 摘要与 arm 核对) |
| `runs/variance/pool_<N>/nohup.log`、`times.txt`、`eval.pid` | 运行记录与 arena PID |
| `runs/variance/measure_sd.py` | §4/§5 复算工具(`--pooled-csv` / `--games-dir` / `--from-csv`) |
| `runs/variance/determinism_check.py` | §6 复算工具(两目录参数) |
| `runs/variance/pooled_table.txt`、`crosscheck_stats.txt`、`pool_disjointness.txt` | 复算输出存档 |
| `runs/variance/exact_sd_output.txt` | §4.1 精确重放(三池)原始输出 |
| `runs/variance/determinism_pool1_vs_poolA.txt` | §6 确定性核对原始输出 |

## 9. Caveats

- 池 A 是**选择池**(PREREG);pool_1 与其同种子,故本任务提供的**独立** 12k 复现只有 pool_2/pool_3 两批。
- 噪声大/CI 宽不是 failure(计划 todo 6 QA 明确);本文件如实记录。
- 池 B(`seed_start=50000`)保留给最终确认,本次**未消费**。
- 三池均值散布略大于单池 SE 预测(rank 1.68×,p≈0.06;pt 1.54×,p≈0.09),3 池样本下不显著;
  若后续做多池合并,建议继续观察是否存在种子窗口异质性。
