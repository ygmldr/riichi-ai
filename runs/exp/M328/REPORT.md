# M328 测量报告 — live 328k 对 at-175200 / at-148800(池 A 12k)

- 日期:2026-09-20(两场 arena:vs148800 `00:53:38→01:30:13`;vs175200 `01:34:07→02:10:01` CST)
- 对象:`runs/exp/PINNED/mortal.pth.at-328000`(即 live `checkpoints/mortal.pth`,steps=328000)
- 协议:3000 sets / 12000 局 / 场,seed_key `0x5dfaa4cef265cd7`(`423244137402621143`),
  seed 窗口 `[10000,13000)` = 池 A,champion 固定为 `checkpoints/mortal.pth.at-148800`(name `old148800`)
  或 `runs/exp/PINNED/mortal.pth.at-175200`;报告由未修改的 `scripts/eval_report.py` 生成
  (set 级 bootstrap B=2000,rng seed=0)。
- 噪声地板:`runs/variance/README.md`(todo 6)——同一 12k 协议、三互斥池,80% 功效可分辨效应
  **MDD ≈ 0.0239 avg_rank / 1.86 pt**。

## 0. 结论(一句话)

**多训练的 153k 步(175.2k→328k)没有带来任何可检测的强度变化**:配对 Δ(328k−175.2k)
= **+0.0036 顺位 / −0.87 pt**,两个 95% CI 均跨 0,|点估计| ≤ 0.15×MDD(顺位)/ 0.47×MDD(pt);
同种子同 champion 的单次运行均值差(未配对框架)与配对框架给出的点估计一致,
但两者都远在 12k 分辨率地板之下 —— 328k 与 175.2k(以及 148.8k)**互相不可分辨**。

## 1. 血统与身份

| 快照 | steps | 角色 | sha256(前 16) | 本报告中的测量 |
|---|---|---|---|---|
| `checkpoints/mortal.pth.at-148800` | 148800 | 前冠军(两次测量的共同对手) | `73ec8108f1b9c4ab…` | 仅作 champion |
| `runs/exp/PINNED/mortal.pth.at-175200` | 175200 | 挑战者(归档测量)/ champion(vs175200) | `f0a611b97f318e0b…` | EVAL_A_175200;M328/vs175200 champion |
| `runs/exp/PINNED/mortal.pth.at-328000` = live `checkpoints/mortal.pth` | 328000 | 挑战者 | `b14e77ed9b79bdaa…` | M328/vs148800 + M328/vs175200 |

- 328k pin 与 live `checkpoints/mortal.pth` 的 sha256 **相同**(`b14e77ed…a3511`),pin 未被本任务改动。
- 架构一致性:三者均为 `resnet {c=192, b=40}` 纯 CNN(config 无 `[model]` key,`online=false`,version=4)。

## 2. 三次 12k 测量(各自 report.csv,未配对)

| 挑战者 | champion | 顺位分布(1/2/3/4) | avg_rank [95%CI] | pt [95%CI] | CI 跨 parity? |
|---|---|---|---|---|---|
| at-175200 | at-148800 | `[3010 3149 2950 2891]` | 2.476833 [2.459912, 2.493754] | +1.860000 [+0.566250, +3.202594] | rank 跨 2.5;pt **不跨 0** |
| at-328000 | at-148800 | `[3096 3028 2891 2985]` | 2.480417 [2.463329, 2.497338] | +0.993750 [−0.270563, +2.328844] | rank 跨 2.5;pt 跨 0 |
| at-328000 | at-175200 | `[3177 2896 2849 3078]` | 2.485667 [2.466998, 2.503169] | +0.060000 [−1.309031, +1.481250] | rank 跨 2.5;pt 跨 0 |

- 全部 `sets=3000 games=12000 incomplete_sets=0 ignored_files=0`;challenger/champion 名字由
  NDJSON `start_game.names` 逐局验证(`at-328000`/`mortal` vs `old148800`/`at-175200`)。
- 只有 175.2k-vs-148.8k 这一条的 pt CI 排除 0(+1.86);后续方差研究(`runs/variance/README.md`)
  证明该 +1.86 是三池中的最高值、且不可复现,此处 328k 对同一 champion、同一池的 +0.99 也未排除 0。

### 2.1 与同 champion 对手的未配对均值差(框架 A)

两次挑战者测量共享 champion `at-148800` 与同一 seed 窗口,故均值差即:

- Δavg_rank(328k−175.2k)= 2.480417 − 2.476833 = **+0.003583**(328k 略差)
- Δpt(328k−175.2k)= 0.993750 − 1.860000 = **−0.866250**(328k 略差)

## 3. 配对比较(主判据):同种子、同 champion 的 set 级配对 bootstrap

目录:`runs/exp/M328/vs148800/games`(328k)vs `runs/exp/EVAL_A_175200/games`(175.2k),
双方均为 3000 seeds / 12000 局,champion 均为 `old148800`。
过程:`scripts/exp_a1_delta.py`(导入、未修改)—— 逐 seed 配对差值,B=2000,rng seed=0
(与 `eval_report.py` 完全同一 bootstrap 口径)。复现:

```bash
python3 runs/exp/M328/paired_delta_328k_minus_175200.py
# -> runs/exp/M328/paired_delta_328k_minus_175200.json
```

| 指标 | Δ(328k − 175.2k) | 95% CI | 跨 0? | 点估计 / MDD | CI 半宽 / MDD |
|---|---|---|---|---|---|
| avg_rank | **+0.003583** | [−0.017923, +0.027002] | **是** | 0.15× | 0.94× |
| pt | **−0.866250** | [−2.655188, +0.802688] | **是** | 0.47× | 0.93× |

- 配对后 CI 半宽(顺位 0.0225 / pt 1.73)约等于 1×MDD ⇒ **本 12k 设计连"分辨 1×MDD 效应"都只能勉强做到**,
  而观察到的效应远小于此。
- 结论:**不满足"328k 比 175.2k 更强"的任何阈值**;预注册式判词
  `challenger_328k: NO detectable effect on avg_rank (95% CI crosses 0)` 对 pt 同样成立。

### 3.1 符号/框架差异(如实记录)

- **原始符号不一致**:Δrank 为正而 Δpt 为负。这不是两个框架互相矛盾 ——
  rank 是成本(越小越好)、pt 是收益(越大越好),两者都指向同一方向:328k **略差**,
  差幅 0.0036 顺位 / 0.87 pt,均在噪声内。
- **配对 vs 未配对**:点估计相同(同 seeds 的均值差),差别只在 CI ——
  配对 CI 半宽 0.0225 / 1.73,与两独立 CI 合成的朴素半宽(≈0.0240 / 1.87)几乎相同;
  说明"同种子配对"在本协议下**没有**兑现出可观的方差削减(同 seed 的两条对战轨迹
  因政策不同而高度发散,组内相关很低)。两种框架下的判决一致:无差异。

## 4. 判决

1. **328k vs 175.2k:不可分辨**(配对 Δrank CI 与 Δpt CI 均跨 0;|点估计| ≤ 0.47×MDD)。
2. **328k vs 148.8k(live vs 前冠军):不可分辨**(rank CI 含 2.5;pt CI [−0.27, +2.33] 含 0)。
3. **175.2k vs 148.8k:也不可分辨**(rank CI 含 2.5;池 A 的 pt +1.86 未被多池复现)。
   ⇒ 148.8k / 175.2k / 328k 三代快照在 12k 分辨率下**互相不可分辨**。
4. 因此:**继续训练 153k 步未产生可检测的强度提升**;若后续要判断 resume 后是否有提升,
   12k/次的最小可分辨效应是 ≈0.024 顺位 / 1.86 pt,单次 12k 判定不足,应按 `runs/variance/README.md`
   的 n-局数表或多池合并设计。

## 5. 产物与备注

| 产物 | 说明 |
|---|---|
| `runs/exp/M328/vs148800/{report.csv,report_stdout.txt,eval.log,games}` | 328k vs at-148800,12000 局 |
| `runs/exp/M328/vs175200/{report.csv,report_stdout.txt,eval.log,games}` | 328k vs at-175200,12000 局 |
| `runs/exp/M328/paired_delta_328k_minus_175200.{py,json}` | §3 配对分析及其 JSON 结果 |
| `.omo/start-work/evidence/M1.txt` | 完整证据链(预检、config、PID、日志、本报告数字) |

- 已知良性异常:`vs175200/runner.log` 在 01:35:03 记录 `FAIL: no 'challenger rankings'`,
  是启动器的存活检查误判(它在 60s 时误以为 arena 已退出);arena 实际继续运行到
  02:10:01 完成,eval.log 与 `report.csv`(3177/2896/2849/3078,2.485667,0.06pt)完全一致,
  12000 个 `*.json.gz` 齐全并通过 `eval_report.py` 的全部自洽断言。**不需重跑。**
- 本次测量期间未启动训练;`runs/1v3/` 冻结语料保持 2004 个 `*.json.gz`。
