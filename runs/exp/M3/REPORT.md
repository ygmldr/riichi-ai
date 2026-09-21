# M3 测量报告 — 382k 直接头对头 328k(全新窗口 `[120000,126000)`,24,000 局)

- 日期:2026-09-21;arena 墙钟 **21:16:22 → 23:19:18 CST**(2h02m56s)
- 对象:challenger `runs/exp/PINNED/mortal.pth.at-382000`(steps=382000);champion
  `runs/exp/PINNED/mortal.pth.at-328000`(steps=328000);两者均为纯 CNN `resnet{c=192,b=40}`、version=4。
- 协议:6,000 sets / 24,000 局,seed_key `0x5dfaa4cef265cd7`,窗口 `[120000,126000)`(未消费);
  1 challenger 席 vs 3 champion 席,challenger 席位随 split a/b/c/d 轮转(每席 6,000 局)。
- 报告由**未修改**的 `scripts/eval_report.py` 生成(set 级 bootstrap B=2000,rng seed=0,
  `git status --porcelain -- scripts/eval_report.py` 为空,mtime `2026-09-17 13:29:19.866788050`,size 17434)。
- 事前注册:`runs/exp/M3/PREREG.md`(birth **2026-09-21 21:15:51**,先于 21:16:22 启动);本报告只判决,不改规则。

## 0. 结论(一句话)

**382k 与 328k 在本次 24k 局分辨率下不可分辨。** 主判据 challenger `avg_rank` = **2.4936,95% CI
[2.4816, 2.5050] 含 parity 2.5** ⇒ 按 PREREG §4 判词为 **`no detectable difference at 24k`**(不是"无差异")。
副判据 pt = **+1.2300,95% CI [+0.3429, +2.1435] 排除 0**,方向与主判据点估计一致(382k 略优),按 PREREG §4
作为支持证据记录,不改变主判据。|点估计| = **0.38×MDD**(rank)/ **0.94×MDD**(pt);
小于事前 MDD(0.0169 顺位 / 1.315 pt)的效应不可分辨。

## 1. 血统与身份(启动前 / 结束后各核对一次,逐字相同)

| 角色 | 文件 | steps | bytes | sha256 |
|---|---|---|---|---|
| challenger `at382000` | `runs/exp/PINNED/mortal.pth.at-382000` | 382000 | 130,781,915 | `fddf5c4859e0e5a39842177155065ab28f995b0c07da8820aea993abdff07dd3` |
| champion `at328000` | `runs/exp/PINNED/mortal.pth.at-328000` | 328000 | 130,780,507 | `b14e77ed9b79bdaa02ad21d269e378efb398e0aeec3721842975541c082a3511` |

- 382k pin 由 328k 续训 +54,000 步得到(M2:15:18:32 → 19:00:10 存档,零损失停止);
  两个 pin 全程只读,未被本任务改动(结束后 `sha256sum` 复核逐字匹配)。
- NDJSON 逐局核对(`/tmp/opencode/m3_identity_audit.py`,独立于 `eval_report.py`):
  24,000 个文件 = 24,000 条 start 记录;每局 names 恰为 1×`at382000` + 3×`at328000`;
  challenger 席位分布 a/b/c/d = **[6000, 6000, 6000, 6000]**;
  `seed_pos` ∈ [120000,125999] 共 6,000 个不同值(每值 4 局 = 4 splits),
  seed_key 恒为 `423244137402621143`;`IDENTITY_AUDIT: PASS`。

## 2. 协议复核

- **窗口新鲜性**:`[120000,126000)` 与已消费窗口(`[10000,13000)`/`[20000,23000)`/`[30000,33000)`/
  `40010`(akochan)/`[50000,53000)`(池 B,已恰好消费一次)/`60000` 一带(akochan)/`[70000,73000)`(A14))
  零交集;启动前全库文本审计 `120000` 零命中(见 PREREG §5)。本窗口 **EXACTLY ONCE**。
- **eval.toml 与模板 `runs/exp/A7/eval.toml` 的完整 diff**(实测,共 4 个 hunk):
  1. `[control].state_file/best_state_file`:A7 state → 382k pin(键重定向;`one_vs_three.py` 只读
     `cfg['champion']/cfg['challenger'].state_file`,实测 grep 确认 **`[control]` 在 1v3 路径上不被读取**,
     故该项无语义影响);
  2. `[1v3].seed_start 10000 → 120000`、`games_per_iter 12000 → 24000`;
  3. `[1v3].log_dir → runs/exp/M3/games`;
  4. `[1v3.challenger].name/state_file = at382000/382k pin`;`[1v3.champion].name/state_file = at328000/328k pin`。
  `seed_key`、`iters=1`、`[1v3.akochan].enabled=false`、`[resnet]`、`[cql]`、`[aux]`、`[model] use_event_branch=false`、
  `[env]` 与模板逐字一致;无 reward/架构键改动。
- **时间线**:PREREG birth 21:15:51:031 → launcher 21:16:22 → runner 首行
  `seed: [120000, 126000) w/ 0x5dfaa4cef265cd7, start 6000 sets, 24000 hanchans`(21:16:27.095)→
  arena `python3 one_vs_three.py` PID 9976 进程启动 21:25:42(launcher 后 ≈9m20s,模型加载/预热)→
  `dumping game logs`(23:18:57.127)→ `>>> 1v3 评估结束`(23:19:18)。
- **runner 自打印与报告一致性(交叉验证)**:`challenger rankings: [5895 6113 6243 5749] (2.4935833333333335, 1.23pt)`
  与 `eval_report.py` 输出逐值一致。

## 3. 结果

| 指标 | 值 | 95% CI(B=2000,set 级重采样) | 与 parity 关系 |
|---|---|---|---|
| 顺位分布(1/2/3/4) | `[5895 6113 6243 5749]` | — | — |
| challenger `avg_rank`(parity **2.5**) | **2.493583** | **[2.481622, 2.505003]** | **CI 含 2.5** |
| challenger `pt`(parity **0**) | **+1.230000** | **[+0.342891, +2.143453]** | **CI 完全 > 0** |

- `sets=6000 games=24000 incomplete_sets=0 ignored_files=0 candidates=24000 matched=24000`(全部自洽断言 PASS)。
- bootstrap:resample sets(6,000),B=2000,rng seed=0,percentile [2.5, 97.5];与池 A/池 B/方差研究同一口径。
- 实测精度 vs 事前:rank SE 观测 **0.005965**(事前 0.006039,−1.2%);pt SE 观测 **0.459327**
  (事前 0.469512,−2.2%)⇒ 噪声模型与事前一致,CI 半宽 **0.011691 / 0.900281**。
- 观测 CI 半宽/MDD:rank **0.69×MDD**、pt **0.68×MDD**(MDD 定义见 §6)。

## 4. 判决(严格按 PREREG §4;不事后改门)

1. **主判据(challenger `avg_rank` 95% CI 是否排除 2.5)**:CI = [2.4816, 2.5050] **含 2.5** ⇒
   判词 = **`no detectable difference at 24k`**(PREREG §4 情形 3;不写成"无差异")。
   按规则同时报告:|点估计 − 2.5| / MDD = **0.38**;CI 宽度 = 0.0234 = **1.38×MDD**;
   小于 MDD 的效应不可分辨(§6)。
2. **副判据(pt 95% CI 是否排除 0)**:CI = [+0.3429, +2.1435] **完全 > 0**,方向与主判据点估计
   (2.4936 < 2.5)一致 ⇒ 作为"382k 可能略优"的**支持证据**记录;按 PREREG §4,主判据优先,判决不变。
   如实标注:这是本实验唯一排除 0 的区间,两条指标出自同一局结果的不同矩(rank 为有界序数,pt 对
   4 位施加 −135 重罚),出现"一条排除、一条含 parity"属边界配置,不宜过度解读。
3. **仍未排除的最小效应**(CI 内最接近 parity 的真实值):
   rank **+0.0050**(即 382k 略差 0.005 顺位仍未被 95% CI 排除);
   pt **+0.3429**(382k 略优 0.34 pt 仍未被排除,且所有 ≤0.34 pt 的效应含 0 与负方向均已被排除)。

## 5. 与 M328(12k)对照

| | M328(池 A 12k) | M3(新窗口 24k) |
|---|---|---|
| 对局 | 328k vs 175.2k(配对 Δ) | **382k vs 328k(直接 H2H)** |
| 主指标 | Δavg_rank **+0.0036 [−0.0179, +0.0270]** 跨 0 | challenger avg_rank **2.4936 [2.4816, 2.5050]** 含 2.5 |
| 副指标 | Δpt **−0.866 [−2.655, +0.803]** 跨 0 | challenger pt **+1.230 [+0.343, +2.143]** 排除 0 |
| 分辨率 | MDD 0.0239 顺位 / 1.86 pt | **MDD 0.0169 顺位 / 1.315 pt**(≈0.71×) |
| 判词 | 不可分辨 | **no detectable difference** |

- 两次测量的效应量都在 ~0.4×MDD(rank)以内;M3 把可分辨粒度压到 M328 的约 0.7×(CI 半宽 0.0117 vs 0.0225),
  仍未检出 ≥0.017 顺位量级的续训增益。
- 差异提示:两次实验的 champion/窗口/统计对象都不同(M328 测 328k 相对 175.2k 的增量;M3 测 382k 相对 328k
  是否偏离 parity),**不可**把两行的 pt 数值直接相减当作"续训曲线"。可说的只有:M3 的 pt 点估计
  (+1.23)是迄今单次测量中最大的正向提示,与"54k 步续训带来 ≤~1.3 pt 的小幅正向效应"相容,
  但低于/接近事前 MDD,且主判据未过门。

## 6. 分辨力边界(事前,来自 `runs/variance/README.md`)

- σ_set = **0.467788 顺位 / 36.368346 pt**;n = 24,000 局(6,000 sets)⇒ SE = **0.00604 / 0.4695**,
  95% CI 半宽 = **±0.01184 / ±0.92023**。
- 80% 功效 MDD(双侧 α=0.05)= **0.016919 ≈ 0.017 顺位 / 1.315374 ≈ 1.3 pt**;
  12k 时为 0.0239 / 1.86。
- 结论口径:**本设计对 <0.017 顺位(≈<1.3 pt)的真实效应没有 80% 检出功效**;本次 CI 也不排除
  382k 真实优势最大到 **0.0184 顺位**(CI 下界 2.4816 对应 2.5 − 0.0184)。
  若未来要判定该量级,应使用多池合并或更大 n(按 `runs/variance/README.md` 的 n-局数表)。

## 7. 墙钟与吞吐(异常,如实记录;不影响判决有效性)

- **实测**:launcher→结束 **7,378 s = 2h02m56s**;arena 进程启动→dump **6,795 s = 1h53m15s**。
  等效吞吐 = **3.253 games/s**(launcher 口径)/ **3.532 games/s**(进程口径)。
- **历史同协议**(A7 12k = 32m47s ≈ 6.10 g/s;pool_1 12k = 35m00s ≈ 5.71 g/s;T22 E3a 12k = 35m49s ≈ 5.58 g/s):
  24k 预计 **66–75 min**;实际 113.3–123.0 min = 历史外推的 **1.64–1.86×**,
  只有历史吞吐的 **53.3–63.3%**。PREREG 中"dump ≈22:22–22:31"的 ETA 未兑现(实际 23:18:57),
  前序会话 110 分钟等待预算因此耗尽(interim 记录见 notepad)。
- **非挂死证据**:轮询期间 CPU ticks 持续增长(23:15:08→23:18:51:`3,553,692 → 3,666,084` ticks,
  每 30s 增量 13.2–18.8k ≈ 4.4–6.3 核);结束时累计 CPU ≈ **36,660.8 s ≈ 10h11m**,均值 ≈ **5.40 核**;
  RSS 观测范围 2,364,094–2,519,953 页 ≈ **9.02–9.61 GiB**;39 threads;state 在 R/S 间切换。
- GPU 健康(23:12 观测,前序会话实测):WSL nvidia-smi 内存 15,774 MiB(含宿主侧,勿当 WSL 占用)、
  util 66–99%、79–96 W、2842 MHz;CUDA 探针 4096³ matmul = 13.60 ms ≈ 10.11 TFLOP/s。
- **原因未确定(不粉饰)**:已否证的假设 —— #114 agari guard(落盘 `e4043eb` 09-17 **13:40**)
  **早于**方差基线(09-17 **14:40–16:21**,当时吞吐为历史水平),故不是它导致本次变慢。
  进程健康、GPU 在算、GPU 资源充足(单作业)⇒ 本报告只记录"实测远慢于历史",不宣称根因。
  对策建议(实验记录层面):未来 24k 量级的 ETA 按 **2h** 而非 70 min 计预算。
- 有效性说明:慢只影响墙钟;种子与策略决定牌局,24,000 局、6,000 sets、4 splits 全部齐备且身份正确,
  测量本身不受影响。

## 8. 不变量复核(全部通过)

| 项 | 值 |
|---|---|
| `ls runs/1v3/*.json.gz \| wc -l` | **2004**(与启动前一致,冻结语料只增 `.log`) |
| `find runs/1v3 -name '*.json.gz' -newermt '2026-09-21 21:00' \| wc -l` | **0** |
| 两个 pin sha256(结束后) | `fddf5c48…7dd3` / `b14e77ed…3511`(**匹配期望**) |
| `stat -c '%y %s' checkpoints/file_index.pth` | `2026-09-17 16:39:22.835975729 +0800` / `185010303`(**未变**) |
| `runs/exp/M3/games/*.json.gz` | **24000**(a/b/c/d 各 6000) |
| 第二 GPU 作业 | 无(启动训练前已复核,arena 已退出) |

## 9. 产物与复现

| 产物 | 说明 |
|---|---|
| `runs/exp/M3/PREREG.md` | 事前注册(birth 21:15:51,先于启动) |
| `runs/exp/M3/eval.toml`,`eval.pid`,`nohup.log` | 协议配置、arena PID 9976、launcher 日志 |
| `runs/exp/M3/report.csv`,`report_stdout.txt` | 未修改 `scripts/eval_report.py` 的输出 |
| `runs/exp/M3/games/*.json.gz` | 24,000 局牌谱(970 MB;`runs/` 已被 `.gitignore` 忽略,不入库) |
| `/tmp/opencode/m3_identity_audit.py|.out` | 独立 NDJSON 身份审计(仓库外,临时) |

```bash
python3 scripts/eval_report.py --games-dir runs/exp/M3/games \
  --seed-key 0x5dfaa4cef265cd7 --seed-range 120000-126000 \
  --checkpoint at382000 --source-dir runs/exp/M3/games \
  --out runs/exp/M3/report.csv | tee runs/exp/M3/report_stdout.txt
```

判决记录:主判据 `no detectable difference at 24k`(avg_rank CI 含 2.5);副判据 pt CI 排除 0(正向,支持性)。
