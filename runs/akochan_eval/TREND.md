# akochan 基线趋势尺 — TREND

> 日期: 2026-09-17 ｜ 计划: `.omo/plans/fork-roadmap-steps-1-10.md` todo 16(Wave 3 基线复现)
> 脚本: `scripts/eval_akochan.sh`(本次改为 config-driven + 分块 + 并行重试)
> 结论速览: **尺子可用(粗粒度)**;40 局分辨率 ≈ ±8.1pt/席,**读不出该计划实验关注的 1–2pt 级差异**;
> 40 局点估计: 我方每席位 **+6.75pt**(95% CI `[-1.13, +15.00]`,跨 0);
> akochan 二进制有**确定性毒局**(实测 ~14%/局),单批 40 局必崩 → 必须分块跑。

## 1. 这柄尺子量什么

- **对手**: akochan(`vendor/akochan/system.exe`,critter-mj/akochan 的 C++ AI)+
  `tactics.json`(jun_pt `[90,45,0,-135]`,与 mjai-reviewer / upstream strength.md 一致)。
- **被测**: `[1v3.challenger].state_file` = `runs/exp/PINNED/mortal.pth.at-175200`
  (计划决策 #4 钉住的不可变快照;**不用** live `checkpoints/mortal.pth`,它已被训练推进到 328k 步)。
  评估前快照到 `runs/akochan_eval/mortal.pth.at-175200`(与源文件 `cmp` 逐字节一致)。
- **席位**: akochan 坐 1 席(challenger),我方模型坐 3 席(champion)。
- **读法**: `python3 scripts/parse_eval.py <log> --akochan` → "我方每席位均分 > 0 即强于 akochan"。
  上游参照: Mortal 3.1 时代 akochan 每席位 -5.67pt;v4 推算 -8~-9pt(RUNBOOK:155-156)。

## 2. 配置来源(脚本不再硬编码)

`scripts/eval_akochan.sh` 只读 `configs/config.toml`,生成临时配置注入 `MORTAL_CFG`:

| 键 | 来源 | 本次值 |
|---|---|---|
| `[1v3].seed_key` | 仓库配置(可 `AKO_SEED_KEY` 覆盖) | `0x5dfaa4cef265cd7` |
| `[1v3].seed_start` | 仓库配置(可 `AKO_SEED_START` 覆盖) | 60000(见 §5) |
| `[1v3].challenger.state_file` | 仓库配置(可 `AKO_CHALLENGER` 覆盖) | `runs/exp/PINNED/mortal.pth.at-175200` |
| `[1v3].games_per_iter` | CLI 参数 | 每块局数(默认分块 4) |
| `[1v3].log_dir` | **脚本改写** | `runs/akochan_eval/games/`(+ worker 子目录) |

- 临时配置生成在 `/tmp/opencode/akochan_eval/`;写入后用 tomllib **读回逐项断言**
  (games_per_iter / iters / seed_key / seed_start / log_dir / akochan.enabled /
  challenger.state_file / akochan.dir / tactics)。旧脚本断言仓库里的字面量 `2000`/`-1`
  (todo 3 改值后必失效),现在断言的是**脚本实际注入的值**。
- **不改仓库配置**:`configs/config.toml` 全程只读。其 `[1v3].log_dir` 当时指向
  `runs/variance/pool_3/games/`(T6 池目录),照抄会把牌谱写进别人的实验目录,故必须改写。
- **不走 `run_eval.sh`**:其 `:12` 硬编码 `LOG_DIR=runs/1v3`,会向冻结语料目录追加
  `eval_*.log`;本脚本直接调 `vendor/mortal/mortal/one_vs_three.py`,日志留在 `runs/akochan_eval/`。
  冻结不变量 `runs/1v3/*.json.gz == 2004` 运行前后一致;`runs/variance/pool_3/games` 12000 文件未动。

## 3. 为什么分块(chunk)+ 并行

**arena 只在整批对局全部结束后一次性落盘牌谱**(`libriichi/src/arena/one_vs_three.rs:195-220`);
而 akochan 在毒局上确定性崩溃(§4)。实测代价:

- 单批 40 局(2026-09-17 17:24 / 17:51 / 18:42 三次尝试): 撞毒局即整批报废、**0 个牌谱落盘**,
  已算的 42–85% 全部作废;
- 改为**每块 4 局(=1 set)**独立调用 arena: 每块独立落盘,毒块只损失 4 局;
- 默认 **4 个并行 worker**(`AKO_PARALLEL=4`),各自独立 seed 区间/子目录/日志,结束时合并
  子目录并校验总数;每 worker 尝试次数有界(`4*chunks+8`);
- 失败窗口由脚本写入 `windows_*.txt` 与合并日志,**换后续 seed 窗口重试**(不重试同一窗口),
  凑满请求局数才 exit 0;凑不满则 exit 1 并打印"趋势不可用"。

## 4. akochan 毒局记录(趋势尺的已知噪声源)

崩溃形态:system.exe `terminate called after throwing an instance of
'boost::container::bad_alloc'`(signal 6 / WSL CaptureCrash);Rust 侧报
`RuntimeError: failed to read from akochan: unexpected EOF`。

**定位方法(可复用)**: `dmesg | grep CaptureCrash` 得到崩溃 PID → 与 `AKOCHAN_PIPE_TRACE=<dir>`
的 `pipe_<pid>.log` 同名 → trace 文件 PID 升序 = arena 生成顺序 = game index →
`seed = seed_start + idx//4`,`split = idx%4`,akochan 座位 = split 序号。

| # | seed | split | 座位 | 崩溃时该进程已响应 | 触发事件 |
|---|---|---|---|---|---|
| 1 | 40007 | b | 1 | 1 | 自己的 `tsumo` |
| 2 | 40019 | c | 2 | ~34(全批 1346 actions) | `tsumo` |
| 3 | 40020 | c | 2 | 43 | `tsumo` |
| 4 | 40021 | a | 0 | 174 | `tsumo` |
| 5 | 40023 | a | 0 | 短(1.7 min 内) | `tsumo` |
| 6 | 40024 | c | 2 | 短 | — |
| 7 | 40026 | — | — | 8 秒内 | — |
| 8-15 | 60001/60002/60004/60005/61001/61003/61004/62000 | — | — | 1–15 min 内 | — |

- **确定性**: 窗口 [40000,40010) 连续 4 次运行、[40010,40020) 2 次、以及单 seed 定向复现(40007)
  → 同一牌局必崩,PID→(seed,split,seat) 映射一致,不是随机抖动。
- **触发点**: 已确认的 4 例都发生在该席位**自己的 `tsumo`**(摸牌后的打牌搜索 `ai()`);
  静态复现(手工重放牌谱喂 system.exe)在 `tehai_ana.cpp:263` 撞 aka 断言 → 疑为接口/边界 bug。
- **非全机内存不足**: 崩溃时 44 进程总 RSS 1.9 GB、MemAvailable ≥ 8.7 GB(已测,排除 OOM)。
- **实测毒局率**: 40 局验收运行 18 块尝试 **8 块失败(44%)** → 约 **14% 的单局**会让 akochan 崩
  (0.56^0.25≈0.86)。历史 seed 10000 与 smoke seed 40100 属于少数干净局。
- **不能修**: 任务约定不改/不重建 `vendor/akochan`(预构建依赖);尺子以"分块 + 跳过毒局 + 记录窗口"应对。

## 5. seed 窗口与消费记录

- 本次验收运行的 seed 起点 `AKO_SEED_START=60000`;4 个 worker 起点 60000/61000/62000/63000
  (`AKO_SEED_STRIDE=1000`),共消费 18 个 seed,成功 10 个:
  **成功**: 60000, 60003, 60006, 61000, 61002, 61005, 62001, 62002, 63000, 63001
  **失败(毒局,已跳过)**: 60001, 60002, 60004, 60005, 61001, 61003, 61004, 62000
  (逐 worker 明细见 `windows_at-175200_40g_seed60000_20260917_2020.txt`)
- 早期探索消费(未计入本次结果,记录备查): 40000–40029 区间(3 次 40 局批运行 + 分块串行),
  首个成功块 seed 40022 的 4 局留在 `runs/akochan_eval/games_extra/`,不进本次 40 局统计。
- T6 噪声池已占用 [10000,13000)/[20000,23000)/[30000,33000);**池 B 预留 [50000,53000)
  只测一次** —— 本尺子未触碰。

## 6. 实测结果(2026-09-17,40 局验收运行)

命令(接受标准): `AKO_SEED_START=60000 AKO_PARALLEL=4 bash scripts/eval_akochan.sh 40` → **exit 0**

```
iters=10 局数=40 分布(1/2/3/4 名)=[8, 11, 6, 15]
akochan(挑战者席位): 均顺位 2.7000 | 均顺位分 -20.2500
我方模型(3 席位)   : 均顺位 2.4333 | 均顺位分 +6.7500
我方每席位均分 +6.7500 (>0 即强于 akochan) | 我方强于 akochan(验收通过)
```

- **set 级 bootstrap(B=2000,重采样 set,rng seed=0,10 sets)**:akochan 均顺位 95%CI `[2.4000, 3.0000]`,
  均顺位分 95%CI `[-45.00, +3.375]` → **我方每席位均分 95%CI `[-1.13, +15.00]`**(= -akochan/3)。
- **逐 set(seed → akochan 顺位/分 → 我方每席位分)**:

| seed | akochan 均顺位 | akochan 均顺位分 | 我方每席位分 |
|---|---|---|---|
| 60000 | 2.25 | +11.2 | -3.75 |
| 60003 | 3.50 | -90.0 | +30.00 |
| 60006 | 2.00 | +22.5 | -7.50 |
| 61000 | 3.25 | -56.2 | +18.75 |
| 61002 | 3.00 | -45.0 | +15.00 |
| 61005 | 2.25 | +11.2 | -3.75 |
| 62001 | 2.75 | -33.8 | +11.25 |
| 62002 | 2.25 | +33.8 | -11.25 |
| 63000 | 2.50 | 0.0 | 0.00 |
| 63001 | 3.25 | -56.2 | +18.75 |

- **噪声实测**: set 级 SD(ddof=1, n=10 sets) 顺位 **0.5244** / 顺位分 **41.35**。
  对照 T6 的 mortal-vs-mortal 12k 局噪声地板(σ_set 顺位 0.475 / 顺位分 36.36,合并三池):
  同一量级(40 局样本的 SD 估计本身 ±25% 不确定),即 akochan 尺子的波动与自家对局一致。
  40 局 → 我方每席位 95%CI 半宽 **±8.1pt**(RUNBOOK 说 ±5~8pt,吻合);按 1/√n,100 局 ≈ ±5.1pt。
- **可分辨性**: 以 σ=13.78/席(本次 set 级 SD)与 80% 功效、双侧 5% 计,MDD = 2.80σ/√n_sets:
  要看出 **10pt** 差需 ≈ 15 sets(60 局);要看出 **5pt** 差需 ≈ 60 sets(240 局)。
- **吞吐(实测)**:
  - 本次 40 局: **20:20:43 → 21:29:52 = 69.2 min 墙钟**(4 worker 并行,机器空闲;含 8 次毒块重试)。
  - 18 次块尝试:10 次成功(时长中位 **13.5 min**,范围 9.8–18.6);8 次失败(1–15 min)。
  - 折算 **≈1.73 min/局**(含重试);单次成功块(4 局)≈13.5 min,与 RUNBOOK:148 的
    "1 set ≈13.5 min" 一致 —— 该数字此前被误读为"40 局≈2.3h",实际单批 40 局不可行(§3/§4)。
- **可用性结论**: 尺子**可用,但只能当粗粒度绝对锚**。
  ① 40 局 CI 半宽 ±8.1pt,**读不出 1–2pt 级差异**(计划中各 arm 的效应量级),不能用于判决性评估
  (计划已规定判决用 12000 局自家冠军池);
  ② 本次点估计 +6.75pt/席(CI 跨 0)→ "未分出与 akochan 的高低",与上游参照(-5.67/-8.5)相比
  点估计偏乐观但证据强度弱;
  ③ 每 50 万步重复本命令即可得到可比趋势(同 key、窗口从文档记录的下一个未用 seed 段继续)。
- **100 局扩展: 未执行**。原因: 40 局已需 18 次块尝试/69 min;毒局率 44%/块,60 局扩展预计再耗
  2–3.5h,超出本次预算。接受标准要求的 40 局已完成并 exit 0。

## 7. akochan 二进制来源/版本(不入库)

- `vendor/akochan/`(**未 commit**,gitignore 内): `git clone https://github.com/critter-mj/akochan`
  (C++/GNU Make),本地构建;`RUNBOOK.md:100-123` 记录重建步骤与踩坑。
- 本次使用哈希(2026-09-17):
  - `system.exe` sha256 `e347fa1154e99fa03acfbfdffd91b785b3fa4a769f61a5a2836db1fb88c42a35`(11,187,392 B, mtime 2026-09-16 19:29)
  - `libai.so` sha256 `4a6362af0706d24312d4ccc1f33f02e6a45b9589a2e408989a7bd6ca7a5399d7`
  - `tactics.json` sha256 `b118a51d3f637d9240238e48b01246a4717e32a837fe15c035510cee706df8c4`
- 本地补丁(`main.cpp`,未提交):仅新增 `AKOCHAN_PIPE_TRACE` 协议 trace(opt-in,默认关闭),
  不改决策逻辑;毒局定位全靠它。
- Boost 依赖免 root 装于 `vendor/akochan/.boost_local`;binary 与 `params/` 同为构建产物。

## 8. 复现

```bash
# 依赖: .venv、vendor/akochan/system.exe(见 §7)、vendor/mortal/target/release/libriichi.so
AKO_SEED_START=60000 AKO_PARALLEL=4 bash scripts/eval_akochan.sh 40   # exit 0
python3 scripts/parse_eval.py runs/akochan_eval/akochan_eval_at-175200_40g_seed60000_*.log --akochan
python3 scripts/eval_report.py --games-dir runs/akochan_eval/games \
    --seed-key 0x5dfaa4cef265cd7 --checkpoint at-175200 --out runs/akochan_eval/report_40g.csv
# 自检(不跑对局): AKO_DRY_RUN=1 bash scripts/eval_akochan.sh 40
```
