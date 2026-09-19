# Architecture verdict — todo 22: five-arm pool-A ranking + single pool-B confirm (E3a vs E2)

- 日期: 2026-09-19(池 B 运行:E3a 17:49:58→18:25:44;E2 18:32:39→19:12:43)
- 计划: `.omo/plans/fork-roadmap-steps-1-10.md` todo 22(lines 215–220)
- PREREG: `runs/exp/ARCH_B/PREREG.md`(**17:49:27 写入**,早于首个池 B arena 启动 17:49:58;
  单次确认声明在运行前落盘)
- 状态: **判决达成**(前置门通过;池 B 对每臂恰好使用一次;不变量全程保持)

## 0. 结论(一句话)

前置投喂门全部通过后,五臂池 A 排序显示三条含 CNN 的臂(A7/E3a/E2)挤在 12k 分辨率地板
(0.026 顺位 ≈ 1.1×MDD)内、无法互相分辨;**唯一一次池 B 确认(配对,3000 sets/12000 局/臂)
给出 Δ(E3a−E2)pt = −2.52,95% CI [−4.34,−0.64](主判据要求下界 > 0)——不满足**
⇒ **正式关闭 river-transformer(事件 Transformer 分支)线**(限于当前损失与 20k 步预算;
重开条件见 §7)。附带:主判据之外,池 B 的 Δavg_rank = +0.0175,CI [−0.0059,+0.0409] 跨 0,
两指标方向不一致处如实记录。

## 1. 前置门(评估投喂门)——必须先过,已过

以下四份记录均在池 B 运行**之前**重新读取并断言(`failure_count == 0` 且断言为真):

| 记录 | 内容 | 断言 | 判定 |
|---|---|---|---|
| `runs/exp/EVT/EVAL_FEED.json`(todo 18 竞技场管线门,smoke40) | 双方 `enable_quick_eval=False`;25,177/25,177 引擎样本与牌谱重放 ground truth 对齐;engine_calls chal=845 / champ=926 | `all_assertions_pass=true`,①非空非零 ②长度==公开事件数 ③同局内单调 | **PASS** |
| `runs/exp/E3a/EVAL_FEED_E3A.json` | `evaluate_calls=2,964`,`entries=7,225,501`,`n∈[2,158]` | ①非空 ②首 token=start_kyoku ③末 token ∈ {3,4,5,6,8,11},`failure_count=0` | **PASS** |
| `runs/exp/A8/EVAL_FEED_A8.json` | `evaluate_calls=2,757`,`entries=7,489,737`,`n∈[2,157]` | 同上,`failure_count=0` | **PASS** |
| `runs/exp/A9/EVAL_FEED_A9.json` | `evaluate_calls=2,682`,`entries=7,214,959`,`n∈[2,160]` | 同上,`failure_count=0` | **PASS** |

机制说明(为什么投喂门 + 跑完即证明未被旁路):带分支的 Brain 在 `events=None` 时抛错,
且 `MORTAL_EVT_DEBUG` 的计数与 `react_batch` 实参同步 ⇒ 任一事件臂若静默退化为 branch-less
都不可能跑完 12k。**无任何缺失或失败项,判决未被阻断,未列出补齐命令。**

## 2. 池 B 单次确认的纪律与"未被消费"审计

- 池 B = `seed_start=50000`,`seed_key=0x5dfaa4cef265cd7`,`[50000,53000)` = 3000 sets/12000 局/臂,
  champion `checkpoints/mortal.pth.at-148800`,challenger = 各臂自己的 20k 步 `state.pth`,
  per-arm `log_dir=runs/exp/ARCH_B/<arm>/games/`。**每臂只跑一次,不重跑、不换窗**(PREREG §1)。
- 运行前 grep 审计(PREREG §2):全部历史 toml 无 `seed_start=50000`;唯一命中是
  `runs/exp/EVT/eval_feed_smoke.*` 的 40 局**投喂门 smoke**(seed `[50000,50010)`,
  当时的挑战者不是 E3a/E2——两臂在 2026-09-17 尚不存在),属管线验证而非判决样本。
  其余 12k 判决运行只用 10000/20000/30000。⇒ 对本比较而言池 B 是一次性确认池。
- E3a/E2 之外的三臂(A7/A8/A9)**未**在池 B 上运行(计划只限制 E3a-vs-E2 的确认)。

## 3. 池 A 五臂排序(各臂自己的 report.csv,均 seed_start=10000,12000 局)

来源目录两两不同(**验收项,已断言**):
`runs/exp/A7`、`runs/exp/A8`、`runs/exp/A9`、`runs/exp/E2`、`runs/exp/E3a`
(逐臂 `games/` 目录亦两两不同:runs/exp/<arm>/games)。

| 臂 | 顺位分布(1/2/3/4) | avg_rank [95% CI] | pt [95% CI] |
|---|---|---|---|
| A7(c=192 纯 CNN,同 20k 帧数) | `[2928 2930 2910 3232]` | **2.5372** [2.5191, 2.5552] | −3.4125 [−4.7369, −2.0474] |
| E3a(CNN + 残差零初始化事件 Transformer) | `[2860 2958 2897 3285]` | 2.5506 [2.5327, 2.5685] | −4.4138 [−5.8351, −3.1012] |
| E2(纯 CNN c=202,参数匹配) | `[2625 3150 3068 3157]` | 2.5631 [2.5450, 2.5797] | −4.0163 [−5.3401, −2.6288] |
| A9(同架构,训练侧打乱历史) | `[2672 2908 2941 3479]` | 2.6023 [2.5849, 2.6200] | −8.1938 [−9.5818, −6.8775] |
| A8(同架构,`phi` 置零 ⇒ event-only) | `[144 730 2820 8306]` | 3.6073 [3.5957, 3.6192] | −89.625 [−90.885, −88.380] |
| (参考,非臂)pin at-175200 基线 | `[3010 3149 2950 2891]` | 2.4768 [2.4599, 2.4938] | +1.8600 [+0.5663, +3.2026] |

读法与诚实限定:
- 三条 CNN 臂(A7/E3a/E2)的 avg_rank 极差仅 **0.026**,正好压在 12k 的 80% 功效
  MDD(0.0239)上;且 E3a 相对 E2 在池 A 上 rank 略优(−0.0125)而 pt 略差(−0.3975),
  **符号不一致** ⇒ 池 A 无分辨力,这正是需要池 B 的原因。
- A7 是池 A 最好的臂(2.5372 / −3.4125,且其 pt 好于 E3a/E2)——同帧数下"事件/参数都不加"
  并不吃亏,本身即弱证据。
- A8 的坍缩(pt −89.6,CI 与所有含 CNN 臂完全不重叠)远超任何 MDD,单臂即可判定:
  **event-only 不能替代 CNN 主干**(roadmap A8 之问的答案是"否")。
- A9 相对 E3a 的配对差显著为负(Δrank +0.0517 CI [+0.0279,+0.0751];Δpt −3.78
  CI [−5.566,−1.905],≈2× MDD);**已声明的混淆**:训练用固定置换、评估用自然顺序,
  退化中含 train/test 序错配成分,不能单臂区分"学习需要顺序"与"学到的排列不迁移"。

## 4. 池 B 确认(主判据)——E3a vs E2,配对 per-set bootstrap

独立报告(B=2000,set 级,rng seed 0):

| 臂 | 顺位分布 | avg_rank [95% CI] | pt [95% CI] | sets/games/incomplete/ignored |
|---|---|---|---|---|
| E3a | `[2775 2943 2985 3297]` | 2.5670 [2.5498, 2.5848] | −5.2425 [−6.6113, −3.8963] | 3000/12000/0/0 |
| E2 | `[2642 3188 3104 3066]` | 2.5495 [2.5318, 2.5668] | −2.7225 [−4.0501, −1.3759] | 3000/12000/0/0 |

**配对差 Δ(E3a − E2)**(`scripts/exp_arch_b_delta.py`,与 `scripts/exp_a1_delta.py` /
`eval_report.bootstrap_ci` 同 procedure;B=2000,rng seed 0;3000 个同 seed 配对 set):

| 量 | 点估计 | 95% CI | 判定 |
|---|---|---|---|
| Δavg_rank | **+0.0175** | [−0.0059, +0.0409](宽 0.0468) | 跨 0,不显著 |
| Δpt | **−2.5200** | [−4.3427, −0.6446](宽 3.70) | **完全 < 0** |

**主判据(预注册):Δpt 95% CI 下界 > 0。实测下界 = −4.3427 ⇒ 不满足。**
因此:**无确认性证据支持事件分支;本次确认在 pt 上反而给出 E3a 劣于 E2 的证据**。
(单次、12k/臂,不得过度外推;详见 §8。)

## 5. 分辨率语境(MDD)

- 实测噪声地板(`runs/variance/README.md` §4/§5):σ_set(avg_rank)=0.4678、σ_set(pt)=36.37;
  **12k 局 ⇒ 与固定参考比,80% 功效 MDD ≈ 0.0239 avg_rank / 1.86 pt**(与计划继承数字
  0.05 相比实测约好 2×)。两独立臂互比 MP 需 ×√2 ≈ 0.0338 / 2.63。
- 池 A 内部三 CNN 臂的差异(≤0.026 rank、≤1.0 pt)**就在该地板附近**,单池不足以定论;
  池 B 用配对设计(同 seed 两臂共享局面)把 Δpt 的 95% CI 收窄到 ±1.85 pt 半宽,
  仍不足以分辨 <1.8 pt 级的真实效应——但主判据问的是下界是否 > 0,答案明确为否。
- **池 B 的 null / 负结果是合法且被如实报告的结果**(计划 todo 22 QA 明文:CI 重叠不算 failure)。

## 6. 池 A 与池 B 并排(即使不一致也照报)

| 对比 | 池 A(非配对读数) | 池 B(配对确认) |
|---|---|---|
| Δrank(E3a−E2) | −0.0125(E3a 略优,远低于 MDD) | +0.0175(跨 0) |
| Δpt(E3a−E2) | −0.3975(E3a 略差,远低于 MDD) | **−2.5200,CI 完全 < 0** |

两池方向在 pt 上一致(都是 E3a 更差),但幅度在池 B 上更大;rank 方向翻转且始终不显著。
**不采用"池 A 的符号更有利"之类的择优叙述;主判据只看池 B 的预注册 CI。**

## 7. 明确语句:是否关闭 river-transformer 线

> **关闭(CLOSE)。**
> 在**当前训练损失(Q/CQL 头,无 RL、无 KL 正则,patch 0008 未实现)**与 **20k 步匹配预算**
> 下,E3a 的残差零初始化事件 Transformer 分支没有被池 B 确认优于参数匹配的纯 CNN(E2):
> 主判据 Δpt 95% CI 下界 −4.34 ≤ 0(点估计 −2.52)。结合 A7(同帧数纯 CNN 在池 A 即最好)
> 与 A8(event-only 坍缩,CI 不重叠),**不再为事件 Transformer 分支投入新臂/预算**;
> 保留 CNN 主干路线。
>
> 作用域与重开条件(避免过度外推):
> 1. 该关闭只针对"在 CNN 主干上追加事件 Transformer 分支"这一架构线;不否定 A1 的
>    "推理期顺序信息是承重的"发现(CNN 以固定平面消费河/事件信息,顺序敏感性由 A1 单独证明),
>    也不否定序列建模在其他损失/规模下的可能性。
> 2. 若要重开:必须先有**实质不同的训练目标**(例如 todo 23 的 REINFORCE+Huber+KL)或
>    **不同规模/预算**,并**重新预注册**一个参数匹配的 E3a-vs-E2 式对照(且使用未消费种子池)。
>    仅凭"再试一次"不得重开,池 B 对 E3a/E2 已消费。
> 3. 本判决不涉及 patch 0008 的设计本身;todo 23 的 RL 损失仍可按计划推进(其锚点为
>    `checkpoints/mortal.pth.at-148800`)。

## 8. 限界(写于看数据之后,仍不事后更改主判据)

- **单次确认**:池 B 对每臂只跑一次,本 Δ 无独立复现;quote CI 宽度(Δpt 3.70,Δrank 0.0468)。
  12k/臂仅能对约 ≥1.8 pt(配对口径)的效应给出稳定读数,更小真实增益不可排除。
- E2 的"参数匹配"不等于同 FLOPs/同步时(T17:实测端到端 E2 306 ms/step vs E3a 313 ms/step,
  ≈1.02×)。
- 池 B 种子 `[50000,50010)` 曾被 EVT 40 局投喂门 smoke 使用(不同挑战者、非判决用途,
  PREREG §2 已披露);`[50010,53000)` 为全新。
- A9 的 train/test 序错配混淆只影响 A9 读数,不影响 E3a-vs-E2 确认。
- 所有池 B 运行:单 GPU 作业 + `train.lock` 全程持有;`runs/1v3/*.json.gz` 前/后 = **2004**;
  live `checkpoints/mortal.pth` mtime `2026-09-17 02:03:09`、`best.pth` `2026-09-16 02:59:45`
  全程未变。

## 9. 复算命令

```bash
cd /home/ygsldr/riichi-ai && source scripts/activate_env.sh
# 池 B 配置(断言无 live 路径)
python3 scripts/exp_arch_b_make_configs.py
# 两次评估(严格串行;MORTAL_CFG 由 run_eval.sh 注入)
setsid nohup bash scripts/run_eval.sh runs/exp/ARCH_B/E3a/eval.toml > runs/exp/ARCH_B/E3a/nohup.log 2>&1 &
# …完成后再跑 E2(同上,config 换 E2)
# 独立报告
python3 scripts/eval_report.py --games-dir runs/exp/ARCH_B/E3a/games \
    --seed-key 0x5dfaa4cef265cd7 --seed-range 50000-53000 --checkpoint state.pth \
    --source-dir runs/exp/ARCH_B/E3a --out runs/exp/ARCH_B/E3a/report.csv
# 配对差
python3 scripts/exp_arch_b_delta.py   # -> runs/exp/ARCH_B/delta_e3a_vs_e2.json
```

## 10. 产物

| 文件 | 内容 |
|---|---|
| `runs/exp/ARCH_B/PREREG.md` | 运行前单次确认预注册(含池 B 审计与主判据) |
| `runs/exp/ARCH_B/{E3a,E2}/eval.toml` | per-arm 临时配置(路径全部重定向;生成器断言) |
| `runs/exp/ARCH_B/{E3a,E2}/report.csv`、`report_stdout.txt` | 池 B 独立报告(sets=3000/games=12000/incomplete=0/ignored=0) |
| `runs/exp/ARCH_B/{E3a,E2}/nohup.log`、`runs/1v3/eval_20260919_1749.log`、`_1832.log` | 运行记录(arena 启动/结束/排名行) |
| `runs/exp/ARCH_B/delta_e3a_vs_e2.json`、`delta_stdout.txt` | 配对 per-set bootstrap(B=2000,rng seed 0) |
| `scripts/exp_arch_b_make_configs.py`、`scripts/exp_arch_b_delta.py` | 新增工具(未改任何既有脚本) |
| `.omo/start-work/evidence/T22.txt` | 本 todo 证据 |

验收核对(对照计划 todo 22 Accept):
- [x] 判决文件含每臂池 A CI(§3)、E3a-vs-E2 的**池 B** CI(§4)、**关闭 river-transformer 线**明确语句(§7)、`EVAL_FEED*.json` 引用(§1)
- [x] 五臂 `report.csv` 来源目录两两不同(§3,已用脚本断言)
- [x] 事件臂投喂门记录齐全且 `failure_count=0`(§1)
- [x] 池 B 只出现一次(PREREG §1/§2;无重跑)
