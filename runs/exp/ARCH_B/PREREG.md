# PREREG — todo 22 pool-B single confirmatory shot: E3a vs E2

- 日期: 2026-09-19T17:50+08:00 (**本文件先于任何池 B 运行写入**)
- 任务: `.omo/plans/fork-roadmap-steps-1-10.md` todo 22 —— 架构判决
- 前置门状态(先于本文件逐项复核): `runs/exp/EVT/EVAL_FEED.json` (all_assertions_pass,
  failure_count=0)、`runs/exp/E3a/EVAL_FEED_E3A.json` (2,964 calls / 7,225,501 entries /
  3 assertions true / failure_count=0)、`runs/exp/A8/EVAL_FEED_A8.json` (2,757 calls /
  7,489,737 entries / 3 true / 0)、`runs/exp/A9/EVAL_FEED_A9.json` (2,682 calls /
  7,214,959 entries / 3 true / 0) —— **全部通过**。

## 1. 单次确认声明(the whole point)

**池 B 对每个臂只运行一次。** 本次是 E3a vs E2 的**唯一一次**确认性测量;无论结果为正、
为负还是零,都必须如实报告,不得重跑、不得换窗口、不得增量延长。A7/A8/A9 **不在池 B
上运行**(计划只限制 E3a-vs-E2 的确认)。

池 B 定义(继承 `runs/1v3/PREREG.md` / `runs/variance/README.md`):

| 项 | 值 |
|----|----|
| `[1v3].seed_key` | `0x5dfaa4cef265cd7` (423244137402621143) |
| `[1v3].seed_start` | **50000** (池 B,区间 `[50000,53000)`,3000 sets) |
| `[1v3].games_per_iter` / `iters` | `12000` / `1` |
| champion | `checkpoints/mortal.pth.at-148800` (name `old148800`) |
| challenger | 各臂自己的 `runs/exp/<arm>/state.pth`(20k 步、匹配预算,E3a=`mortal`,E2=`e2`) |
| 每臂独立 log_dir | `runs/exp/ARCH_B/<arm>/games/` |

## 2. 池 B "未被消费" 的审计(先 grep 再声明)

命令与结果(2026-09-19 17:47,运行前):

```bash
grep -rn "50000" runs/ --include="*.toml"        # → 无命中(无任何 toml 用 seed_start=50000)
grep -rn "50000" runs/exp/EVT/eval_feed_smoke.*  # → seed_start=50000 (40 局投喂门 smoke)
ls runs/exp/EVT/smoke_games | wc -l              # → 44 个 *.json.gz(10 sets/40 局,seed [50000,50010))
```

**诚实披露**: 种子 `[50000,50010)` 曾被 todo 18 的 40 局**投喂门 smoke**用过(挑战者不是
E3a/E2,那两臂当时还不存在;用途是管线验证,不是判决)。除此之外:
- 全部历史 12k 判决运行使用 `10000`(池 A / pool_1 / A1 original ≡ 同一份)、`20000`、`30000`;
- **没有任何** 12k 局规模的测量、也没有任何 E3a/E2 对比使用过 `[50000,53000)`;
- arena 对 `(seed, key, 权重)` 确定性 ⇒ 这 11 个 seed 在不同挑战者权重下是不同对局,不构成
  对 E3a/E2 比较的样本消费;
- 因此**对本比较而言池 B 是单次**:本次运行后不得再将其用于其他 E3a/E2 判决。

## 3. 主判据 (pre-registered, 不得事后更改)

令 set = 同一 seed 的 4 个 split(a/b/c/d,挑战者各坐一席),共 3000 sets / 12000 局/臂。
对每个 set 计算平均顺位分 `pt` 与平均顺位 `avg_rank`,并对两臂**按同 seed 配对**求差:

```
Δpt[set]  = pt_E3a[set]  − pt_E2[set]
Δrank[set]= rank_E3a[set] − rank_E2[set]
B=2000 有放回 set 级 bootstrap,rng = numpy.random.default_rng(0),
95% CI = 重采样均值的 2.5/97.5 百分位(procedure 同 scripts/eval_report.py / exp_a1_delta.py)
```

- **主判据: Δpt 的 95% CI 下界 > 0 ⇒ 确认事件分支(E3a)优于参数匹配纯 CNN(E2)
  ⇒ river-transformer 线保持开放,不关闭。**
- 若 Δpt CI 下界 ≤ 0(含跨 0 或显著为负)⇒ **无确认性证据支持事件分支的增益 ⇒ 按计划
  关闭 river-transformer 线**(在当前损失、20k 步预算、本实例下),在判决书中写明。
- 次要读数: Δavg_rank 及其 CI(方向一致性只作为叙述,不改变主判据)。
- 零结果 (null) 是合法且必须如实报告的结果;CI 重叠不是 failure(计划 QA 明文)。

分辨率语境(`runs/variance/README.md` §5,实测): σ_set(avg_rank)=0.4678、σ_set(pt)=36.37;
12k 局 ⇒ 与固定参考比 80% 功效 MDD ≈ **0.0239 avg_rank / 1.86 pt**;两独立臂互比 MP 要乘 √2
(≈0.0338 / 2.63)。池 B 为**配对**设计,故按配对 SE 读;CI 宽度必须原样引用。

## 4. 运行与安全(写于运行前)

- 单 GPU 作业(评估与训练互斥),全程持 `/tmp/opencode/train.lock`(原子 `mkdir`,
  BUSY 即退出;当前 lock 不存在、无 `train.py`/`one_vs_three.py` 存活)。
- `free -g` 启动前要求 `MemAvailable ≥ 6 GB`;两臂**严格串行**。
- 每臂一份临时 config:`runs/exp/ARCH_B/<arm>/eval.toml`,其中 `[1v3].log_dir`、
  `[control].state_file/best_state_file/tensorboard_dir`、`[test_play].log_dir`、
  `[train_play.default].log_dir` 全部指向 `runs/exp/ARCH_B/<arm>/`;
  `[1v3].challenger.state_file` 指向该臂真实 `state.pth`(只读输入)。
  生成脚本 `scripts/exp_arch_b_make_configs.py` 内断言:以上四项均非 live 路径、
  champion ≡ at-148800 pin、seed_start=50000。
- 分离启动:`(setsid nohup bash scripts/run_eval.sh <cfg> > nohup.log 2>&1 &)`,
  记录真实 arena PID;每 ~5 min 轮询(arena 只在批次结束落盘,`games/` 中途为空是正常的)。
- 不变量:每次运行前后 `ls runs/1v3/*.json.gz | wc -l` == **2004**;live
  `checkpoints/mortal.pth` mtime `2026-09-17 02:03:09`、`best.pth` `2026-09-16 02:59:45` 不变。

## 5. 预注册的产物

- `runs/exp/ARCH_B/{E3a,E2}/report.csv`(eval_report.py;sets=3000/games=12000/incomplete=0/ignored=0)
- `runs/exp/ARCH_B/delta_e3a_vs_e2.json`(配对 bootstrap,B=2000,rng seed 0)
- `runs/exp/ARCH_VERDICT.md`(五臂池 A 表 + 池 B CI + 关闭/开放语句 + EVAL_FEED 引用)
- 证据 `.omo/start-work/evidence/T22.txt`

## 6. 预注册的限界 (limits, 写于看数据之前)

- 池 B 是**每个臂一次**的 12k 局;它的一条 CI 只能确认/否定不小于其分辨率的效应,不能排除
  更小的真实增益(quote CI width)。
- E2 的"参数匹配"不等于同 FLOPs/同步时(实测端到端 E2 306 ms/step vs E3a 313 ms/step,
  ≈1.02×;PREREG T17 附录已声明)。
- A9 的已声明混淆(训练置换、评估自然序)不影响本确认;A8 的坍缩远超任何 MDD,单独成立。
- 本判决只针对**当前损失**(CQL/Q 头,无 RL/无 KL 正则,patch 0008 未实现)——"关闭
  river-transformer 线"的作用域就是该设置;若未来引入 RL 损失,该结论不自动外推
  (判决书会写明这一限定)。

签字: T22 worker(todo 22),2026-09-19T17:50+08:00,在启动任何池 B 运行**之前**。
