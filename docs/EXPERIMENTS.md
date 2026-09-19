# 实验索引 EXPERIMENTS (fork-roadmap-steps-1-10)

- 计划: `.omo/plans/fork-roadmap-steps-1-10.md`(30 个实现 todo,覆盖 roadmap 第 1-10 步)
- 生成: 2026-09-19(收尾 todo 30);每条都从磁盘产物实读,不凭记忆;缺失处标"未完成"并列补齐命令
- 统计口径: `runs/variance/README.md`(噪声地板、确定性);门控依据: `docs/exp-gates.md`
- 状态词: **closed** = 已有判决/产物关闭,不再投入(负结果同样是交付物); **open** = 仍未做; **gate-skipped** = 门控未放行,按计划跳过
- 约定: 每条实验的对局只写进自己的 `runs/exp/<ID>/games*/`;`runs/1v3/` 是冻结语料,全程 `*.json.gz = 2004`(实测复核见 `.omo/start-work/evidence/T30.txt`)

## 0. 公共快照与语料

| 名称 | 路径 | 说明 |
|---|---|---|
| pinned at-175200 | `runs/exp/PINNED/mortal.pth.at-175200` | 冻结挑战者(130,780,507 B,sha256 `f0a611b9…`);永不用 live `checkpoints/mortal.pth` |
| at-148800 | `checkpoints/mortal.pth.at-148800` | 池 A/池 B 评估的固定冠军(name `old148800`) |
| 冻结语料 | `runs/1v3/` | 恰好 2004 个 `*.json.gz`;主语料 key `0x80a7b9e2dec96fa`,另有 4 个 legacy akochan 文件 |
| 12k 协议 | `runs/1v3/PREREG.md` | key `0x5dfaa4cef265cd7`;1 seed = 1 set = 4 局;池 A `[10000,13000)`、池 B `[50000,53000)`、A14 确认 `[70000,73000)` |
| live checkpoint 不变量 | `checkpoints/mortal.pth`(2026-09-17 02:03:09)、`best.pth`(2026-09-16 02:59:45) | 本轮所有任务前后 mtime 未变 |

---

## 1. 尺子与基础设施

### E-01 12k 评估协议 + `seed_start` + pinned 挑战者(todo 3/5)
- 产物目录: `runs/exp/EVAL_A_175200/`(report.csv、games/ 12000 局、nohup.log);协议冻结于 `runs/1v3/PREREG.md`
- PREREG: `runs/1v3/PREREG.md`
- 来源快照: challenger `runs/exp/PINNED/mortal.pth.at-175200`;champion `checkpoints/mortal.pth.at-148800`
- 头条结果: 3000 sets/12000 局,dist `[3010 3149 2950 2891]`,avg_rank **2.476833** CI [2.4599,2.4938],pt **+1.8600** CI [+0.5663,+3.2026];效应量恰好 ≈1.0x 12k MDD
- 状态: **closed(不可判定)**。池 A 是选择池;两批独立种子(pool_2/pool_3)的 CI 都含 parity(见 E-02) ⇒ "175.2k 更强"不再作为结论引用,`+3.08`/`+4.11` 退役

### E-02 评估噪声地板:三个互斥种子池(todo 6)
- 产物目录: `runs/variance/pool_1`、`runs/variance/pool_2`、`runs/variance/pool_3`(各 12000 局 + report.csv);复算工具 `runs/variance/{README.md,measure_sd.py,determinism_check.py}`
- PREREG: 无独立 PREREG;协议冻结于 `runs/1v3/PREREG.md`,判定口径见 `runs/variance/README.md`
- 来源快照: 同 E-01
- 头条结果: 合并 σ_set(rank)=0.4678、σ_set(pt)=36.37(CSV 反推);**12k 局 80% 功效 MDD ≈ 0.0239 avg_rank / 1.86 pt**(6100 局 0.0336/2.61;24000 局 0.0169/1.32);反解 `n≈6.87/Δrank²`、`n≈41525/Δpt²`。确定性: pool_1 与池 A 同 seed 区间,规范化后 12000/12000 逐字节一致(唯 `meta.eval_time_ns` 不同) ⇒ 是复制,不是独立复现
- 状态: **closed(尺子已建立;后续所有判决强制引用其 MDD)**

### E-03 holdout:年份 globs 排除 2023 + `holdout_loss` 可见(todo 7)
- 产物目录: `runs/exp/T7/`(smoke.log、state_smoke.pth、tb/、monitor.log);补丁 `patches/0005-train-holdout-val-loss.patch`
- PREREG: 无独立 PREREG;验收 = 计划 todo 7 + 补丁规格
- 来源快照: 重建后的 `checkpoints/file_index.pth`(185,010,303 B;训练 1,968,147 文件,**0 条 2023 路径**;holdout 168,777 文件);smoke 从零 2000 步
- 头条结果: `holdout_loss = 4.201218`(dqn 0.556935 / cql 0.692359 / next_rank 0.912450),TB 可读;`val_globs` 为空 glob 时按设计 `RuntimeError`(fail 路径实测);2024-2026 年数据不在任何集合(已声明)
- 状态: **closed(交付:训练可见留出损失,排除 2023 泄漏)**

### E-04 #111 量化:all-last 超车影响面(只测不修)(todo 8/9)
- 产物目录: `runs/analysis/alllast.json`(5,343,688 B);报告 `docs/quant-111-alllast.md`;脚本 `scripts/analyze_alllast.py`
- PREREG: 无(对 `runs/1v3/` 冻结语料的纯解析)
- 来源快照: `runs/1v3/` 全部 2004 个 `*.json.gz`(at-175200 vs at-148800 的 1v3 验收臂)
- 头条结果: 1576/2004 局到 all-last;末局和了者仍第 4 = **27/159 = 17.0%**(终局口径;at-S4-end 37/181 = 20.4%);3rd-4th 分差 < 8000 = **45.1%**;4th 距 1st < 8000 仅 4/1576
- 状态: **closed(仅量化;#111 未修,`obs_repr.rs` 分数编码未动)**

### E-05 akochan 趋势尺:40 局绝对锚(todo 16)
- 产物目录: `runs/akochan_eval/`(TREND.md、report_40g.csv、windows_*.txt、games/ 40 局);脚本 `scripts/eval_akochan.sh`(config-driven)
- PREREG: 无独立 PREREG;协议与验收 = 计划 todo 16,实读记录 `runs/akochan_eval/TREND.md`
- 来源快照: `runs/exp/PINNED/mortal.pth.at-175200`(评估前快照到 `runs/akochan_eval/mortal.pth.at-175200`,cmp 一致)
- 头条结果: 40 局,我方每席位 **+6.75 pt**,95% CI **[−1.13,+15.00]**(跨 0);akochan 二进制有确定性毒局(实测 ~14%/局,44% 块失败),必须分块+跳过;40 局 CI 半宽 ±8.1 pt,只能当粗粒度锚
- 状态: **closed(尺子可用但无判决力;100 局扩展未跑,预算原因)**

### E-06 #114 修复验证:agari guard 一致(todo 2)
- 产物目录: `runs/exp/114/`(verify.txt、games/ 200 局 smoke、spy_test_play/、spy_train_play/);补丁 `patches/0004-agari-guard-consistency.patch`
- PREREG: 无(运行时 spy + smoke,证据 `.omo/start-work/evidence/T2.txt`)
- 来源快照: 200 局 smoke 用挑战者 `checkpoints/mortal.pth`(只读) vs `mortal.pth.at-148800`;四场景 spy 用临时配置
- 头条结果: 4/4 `MortalEngine` 构造 `guard=True`(baseline/mortal/baseline/trainee 各 2 处);`grep -c` 2→4;200 局 smoke exit 0;补丁 apply -R/apply 往返与 patch 文件逐字节一致
- 状态: **closed(#114 已修;训练规则 = 验收规则)**

### E-07 #115 修复验证 + release-only 构建(todo 1)
- 产物目录: `patches/0003-gamma-discount-labels.patch`;`scripts/build_libriichi.sh`;证据 `.omo/start-work/evidence/T1.txt`
- PREREG: 无(补丁规格 = 计划 todo 1;验收含 `libriichi.__file__` 断言)
- 来源快照: pin `0cff2b5` + patch 0003;release `.so` 重建(23,281,568 B,mtime 晚于补丁)
- 头条结果: 折扣条件 `label <= 37` → `label <= 36 || label == 42`(立直移出、杠宣言移入);γ=1 下数值不变;release 构建后 `import libriichi` 指向 patch 后的 `target/release/libriichi.so`,smoke exit 0(debug/release 陷阱已修)
- 状态: **closed(#115 已修且补丁真生效)**

---

## 2. Wave 2:证伪与校准

### E-08 A1 推理期顺序消融(打乱牌河 / recency 置零)(todo 11/12)
- 产物目录: `runs/exp/A1/`(games_original、games_shuffle、games_recency_zero、delta_report.json、VERDICT.md)
- PREREG: `runs/exp/A1/PREREG.md`
- 来源快照: challenger `runs/exp/PINNED/mortal.pth.at-175200` vs champion `checkpoints/mortal.pth.at-148800`;seed `[30000,33000)`,3000 sets/12000 局/臂
- 头条结果: shuffle−original Δrank **+0.045833** CI **[+0.024000,+0.068917]**(排除 0,≈1.9x MDD);Δpt −4.095 CI [−5.910,−2.434];recency_zero−original Δrank −0.014583 CI [−0.032002,+0.003671](跨 0,12k 未检出);original 臂逐项复现 `runs/variance/pool_3`(包装器忠实、arena 确定性)
- 状态: **closed(正面结论: 顺序信息是承重的,`kawa_overview` 概览之外仍承重)**

### E-09 A2 线性探针:1024-d 是否已编码对手隐藏信息(todo 13)
- 产物目录: `runs/exp/A2/`(PREREG.md、results.json、supplementary.json、README.md)
- PREREG: `runs/exp/A2/PREREG.md`
- 来源快照: `runs/exp/PINNED/mortal.pth.at-175200` 冻结 phi(无训练);400 文件、262,339 决策点
- 头条结果: 主探针 `opp_any_tenpai` AUC **0.9334** CI [0.9312,0.9354]、ECE 0.0099;随机标签对照 0.5007;**24 特征公开基线 AUC 0.9399 > 0.9334**(隐藏 aka 目标 0.6055 vs 0.6158);阳性对照 0.7947 < 0.90 ⇒ 机器判定 `pipeline_valid=false`、`gate_belief=block`
- 状态: **closed(负结果: 没有超出公开信息的隐藏状态证据;喂入 E-11 门控)**

### E-10 A2b 校准审计(零训练,三组对象)(todo 14)
- 产物目录: `runs/exp/A2b/`(PREREG.md、results.json、README.md、a2b_*.npz);脚本 `scripts/exp_a2b_calibration.py`
- PREREG: `runs/exp/A2b/PREREG.md`(sha256 `92cd0d37…`,运行前写入)
- 来源快照: phi = `runs/exp/PINNED/mortal.pth.at-175200`;GRP = `checkpoints/grp.pth`;Q = 同 checkpoint 的 `current_dqn`;主样本 8000 局/85,144 轮
- 头条结果: 三组 gate-bearing ECE 全 ≤0.05: (a) A2 探针 0.00987 CI [0.0086,0.0180]、(b) GRP top-1 0.02422 CI [0.0214,0.0282]、(c) DQN τ=1 0.03643 CI [0.0346,0.0382];τ=0.05 部署温度臂 0.1897 仅透明度报告;自检合成完美校准 ECE 9.3e-18
- 状态: **closed(负结果: 无校准缺口;喂入 E-11 门控)**
- 收尾修复(C-2): patch 0008 后 `DQN.forward` 返回 `(q, v)`,脚本 :406 的 `q = dqn(...)` 已改为 `q, _ = dqn(...)`(单行,todo 30 授权修改);验证: `py_compile` PASS、`--selftest` PASS(输出与 T14 记录逐值一致)、对真实 v4 DQN 执行同式解包得 q(3,46) 有限。另注: patch 0007 后 `mortal/model.py` 顶层 `from event_data import ...`,从仓库根重跑需让 `event_data` 可导入且不遮蔽 `mortal` 包(本次用 `PYTHONPATH=/tmp/opencode/a2b_env` + 仅含 `event_data.py` 符号链接的目录;该环境前提在 T14 时尚不存在,属 0007 的连带影响)

### E-11 Wave 2 门控判定:belief 线不开工(todo 15)
- 产物目录: `docs/exp-gates.md`(判定行 `gate_belief=block`)
- PREREG: 判定规则来自计划 todo 15 与 A2/A2b 各自的 PREREG(非事后选择)
- 来源快照: 汇总 E-08/E-09/E-10 的实读产物
- 头条结果: A2(AUC 路线)与 A2b(ECE 路线)两条独立路线都不放行;`gate_belief=block`。todo 28/29 **gate-skipped**: 交付门(Δpt CI + ECE≤0.15)不适用;`patches/0011` 因此不存在
- 状态: **closed(门控判定落盘;belief 线未开、无需重开)**

---

## 3. Wave 3:架构对照

### E-12 EVT 事件流 tokenizer 投喂门(patch 0006 + 0010)(todo 18)
- 产物目录: `runs/exp/EVT/`(SPIKE.md、EVAL_FEED.json、dataset_smoke.json、train_smoke.log、smoke_games/)
- PREREG: `runs/exp/EVT/SPIKE.md`(三段验收 18-A/18-B/18-C 规格)
- 来源快照: 40 局投喂 smoke 双方 `enable_quick_eval=False`;训练 smoke 用临时 config(不写 live)
- 头条结果: 引擎收到的 **25,177/25,177(100%)** 样本与牌谱重放 ground truth 对齐(非空、长度==公开事件数、同局单调);`failure_count=0`;训练 smoke `event tokens: (64,144,5)`、EXIT=0
- 状态: **closed(前置投喂门通过;是 E-13/A8/A9 可解释的前提)**

### E-13 E3a 融合臂:CNN + 残差零初始化事件 Transformer(todo 19)
- 产物目录: `runs/exp/E3a/`(PREREG.md、report.csv、EVAL_FEED_E3A.json、TRAIN_SUMMARY.md、state.pth)
- PREREG: `runs/exp/E3a/PREREG.md`
- 来源快照: 从随机初始化训练 20k 步(≈2h39m);challenger = 本臂 state.pth;champion = `checkpoints/mortal.pth.at-148800`;池 A `[10000,13000)`;参数量 11,818,752(新增 1,031,296)
- 头条结果: dist `[2860 2958 2897 3285]`,avg_rank 2.5506 CI [2.5327,2.5685],pt −4.4138 CI [−5.8351,−3.1012];投喂门 2964 calls / 7,225,501 entries / failure=0
- 状态: **closed(池 A 仅排序;架构判决见 E-18)**

### E-14 E2 对照臂:参数精确匹配的更大 CNN(todo 17)
- 产物目录: `runs/exp/E2/`(PREREG.md、params.json、report.csv、state.pth)
- PREREG: `runs/exp/E2/PREREG.md`
- 来源快照: 从随机初始化训练 20k 步;champion `checkpoints/mortal.pth.at-148800`;池 A;`conv_channels=202` ⇒ 总参数 11,775,996(与 E3a 差 −0.3618%);`use_event_branch=false` 显式、分支从未实例化
- 头条结果: dist `[2625 3150 3068 3157]`,avg_rank 2.5631 CI [2.5450,2.5797],pt −4.0163 CI [−5.3401,−2.6288];步时 306 ms vs E3a 313 ms(参数匹配 ≠ FLOPs 匹配,已声明)
- 状态: **closed(池 A 无分辨力;主判据归 E-18)**

### E-15 A7 等帧数纯 CNN 参照臂(todo 21)
- 产物目录: `runs/exp/A7/`(PREREG.md、report.csv、TRAIN_SUMMARY.md、state.pth;含一次被 WSL 重启杀死的 eval.log.killed-wsl-restart)
- PREREG: `runs/exp/A7/PREREG.md`
- 来源快照: E3a 的 CNN 主干(c=192/40 blocks)同 20k 步;champion at-148800;池 A
- 头条结果: dist `[2928 2930 2910 3232]`,avg_rank **2.5372** CI [2.5191,2.5552],pt −3.4125 CI [−4.7369,−2.0474](池 A 三 CNN 臂里最好,差异 <MDD,不能据此下结论)
- 状态: **closed(池 A 排序证据;架构判决见 E-18)**

### E-16 A8 `phi` 置零的 event-only 臂(todo 21)
- 产物目录: `runs/exp/A8/`(PREREG.md、report.csv、EVAL_FEED_A8.json、phi_zero_proof_{train,eval}.json、state.pth)
- PREREG: `runs/exp/A8/PREREG.md`
- 来源快照: E3a 架构 + 融合前 `phi=0`(launcher hook,零 vendor 改动);champion at-148800;池 A
- 头条结果: dist `[144 730 2820 8306]`,avg_rank **3.6073** CI [3.5957,3.6192],pt **−89.625** CI [−90.885,−88.380](CI 与所有含 CNN 臂不重叠) ⇒ **event-only 不能替代 CNN 主干**
- 状态: **closed(决定性负结果,单臂即可判)**

### E-17 A9 训练侧打乱历史(todo 20)
- 产物目录: `runs/exp/A9/`(PREREG.md、report.csv、delta_vs_E3a.json、EVAL_FEED_A9.json、state.pth)
- PREREG: `runs/exp/A9/PREREG.md`(含 §7 中断恢复附录)
- 来源快照: E3a 架构 + collate 固定置换(指纹 sha256 `bf16f51c…`);champion at-148800;池 A
- 头条结果: dist `[2672 2908 2941 3479]`,avg_rank 2.6023 CI [2.5849,2.6200],pt −8.1938 CI [−9.5818,−6.8775];与 E3a 配对差 Δrank **+0.051667** CI [+0.0279,+0.0751]、Δpt **−3.78** CI [−5.566,−1.905](≈2x MDD)。声明的混淆: 训练置换/评估自然顺序
- 状态: **closed(顺序在训练与推理两侧都承重;与 E-08 配对读 D5 表)**

### E-18 架构判决:五臂池 A + 池 B 单次确认(todo 22)
- 产物目录: `runs/exp/ARCH_B/`(PREREG.md、E3a/、E2/、delta_e3a_vs_e2.json);判决书 `runs/exp/ARCH_VERDICT.md`
- PREREG: `runs/exp/ARCH_B/PREREG.md`(17:49:27 写入,早于首个 arena 17:49:58;单次声明)
- 来源快照: challenger = `runs/exp/{E3a,E2}/state.pth`;champion `checkpoints/mortal.pth.at-148800`;池 B `[50000,53000)`,每臂只跑一次
- 头条结果: 池 B E3a pt −5.2425 CI [−6.6113,−3.8963]、E2 pt −2.7225 CI [−4.0501,−1.3759];配对 Δ(E3a−E2) Δrank +0.0175 CI [−0.0059,+0.0409]、Δpt **−2.5200** CI **[−4.3427,−0.6446]**;主判据(下界 >0)不满足
- 状态: **closed(river-transformer 事件分支线关闭;重开需新目标/新规模 + 新预注册 + 未消费池)**

---

## 4. Wave 4:自对弈

### E-19 在线自对弈编排 + 对手池(todo 23/24/25)
- 产物目录: `runs/exp/SELFPLAY/`(orchestration.md、pool.json、pool.md、worker_map.json、cycle_0/、run1/);补丁 `patches/0008`(RL 损失)、`patches/0009`(online config)、`patches/0013`(在线尾部 event padding)
- PREREG: 无独立 PREREG;编排协议见 `runs/exp/SELFPLAY/orchestration.md`,池权重规则见 `runs/exp/SELFPLAY/pool.md`
- 来源快照: 4 个本地 v4 快照(best-220000、at-175200、at-148800、baseline),权重 `2^(-rank/1)` 和为 1;K=2 worker 各 100 局 smoke
- 头条结果: worker_0/1 各 100 局、两 key 不同、`zcat -t` 全过;server/trainer/worker 协议跑通;`rmtree` 竞态与 `pids.json` 整数键事故已记录并被 A14 规避;K=2 峰值 VRAM 12,945/16,303 MiB
- 状态: **closed(基础设施交付;被 E-20 使用;`patches/0013` 是超出计划 0003-0012 范围的前置发现)**

### E-20 A14 最小在线自对弈 + 确认(todo 26)
- 产物目录: `runs/exp/A14/`(PREREG.md、VERDICT.md、report.csv、report_poolA.csv、delta_confirm.json、delta_trend.json、state_final.pth)
- PREREG: `runs/exp/A14/PREREG.md`(20:54:08 写入,早于首步训练 21:04:23)
- 来源快照: trainee 从 `checkpoints/mortal.pth.at-148800` 逐字节副本起步(sha256 断言);KL 锚 = 同快照;确认窗 = **`seed_start=70000` `[70000,73000)`**(池 B 已被 E-18 消费,见 `decisions.md` C-1);state_final.pth step 150,100、sha256 `ac95620f…`
- 头条结果: 8,000 局/1,300 步(27.8 min);确认窗 avg_rank 2.701250 CI [2.682165,2.719833]、pt −18.056250 CI [−19.500188,−16.589906];配对 Δ vs anchor: Δrank **+0.268333** CI [+0.242886,+0.293111]、Δpt **−24.075000** CI **[−26.000250,−22.119875]**(≈10x MDD,显著更差)
- 状态: **closed(该 recipe/预算下自对弈线关闭;A14b 不启动。机制级混淆: online 恢复不载 optimizer / 在线关 CQL / KL 权重弱,均已在 VERDICT §4 声明;不等于"self-play RL 无用")**

---

## 5. 偏离计划字面的地方(供审计)

| # | 计划字面 | 实际执行 | 依据 |
|---|---|---|---|
| 1 | 架构五臂按 todo 单条提交 | **逐臂提交**(E3a、E2、A7、A8、A9 各自 commit) | 当天机器两次重启,A7/A8 完成即提交以免丢工作;`learnings.md` T21 记录 |
| 2 | A14 最终用确认池 B `[50000,53000)` | 确认窗改为**从未使用的 `[70000,73000)`**;趋势用池 A | 池 B 已被 E-18(todo 22)一次性消费;`decisions.md` C-1;A14 PREREG §0 与 commit body 均声明 |
| 3 | 补丁范围 0003-0012 | 增设 **`patches/0013-online-tail-event-pad.patch`** | 在线尾部 event 批长度不齐导致 `torch.cat` 崩溃,是 todo 26 的前置发现;A14 首次真机验证通过 |
| 4 | `patches/0011`(belief head) | **不存在(未创建)** | todo 28/29 门控跳过(见 E-11);`docs/exp-gates.md` 记录在案 |
| 5 | todo 26 commit message `feat(exp): A14 self-play (pool B confirm)` | 改为 `feat(exp): A14 self-play (confirm on fresh window 70000; pool B spent)` | 同上 C-1;原 message 已与事实不符,body 说明替换原因 |
| 6 | A2b 脚本属"已关闭、不可改" | todo 30 授权下**单行修复** :406 解包 | C-2;见 E-10 与 `.omo/start-work/evidence/T30.txt` |

## 6. 运行事故记录(不影响语义,但影响可复现性)

- 2026-09-17 02:40 前后: E3a 的池 A eval 与 E2 的训练进程被杀,20:30 重启;两者均从**自己的 state.pth** 恢复并在 PREREG 追加恢复说明。
- 2026-09-19 10:44 与 ~11:37: 两次 WSL 重启,内存 19 GB → 15 GB;A9 run1(~600 步,未到 2000 步存档点)被杀死后从零重训,A7 的第一次 eval 被杀后完整重跑(arena 确定性 ⇒ 复制而非新样本);均有 PREREG 附录。
- 2026-09-19 12:22: A8 启动脚本在锁 BUSY 时未退出,短暂并发 32 步后自检发现并终止;A9 未受影响,记录于 T21 evidence。
- 全程: 每个恢复点都从本臂自己的 checkpoint 继续;`runs/1v3/*.json.gz = 2004`、live checkpoint mtime 未变(每次任务边界断言)。

## 7. 复算入口

```bash
cd /home/ygsldr/riichi-ai && source scripts/activate_env.sh
# 尺子
python3 runs/variance/measure_sd.py --pooled-csv runs/variance/pool_1/report.csv runs/variance/pool_2/report.csv runs/variance/pool_3/report.csv
# 单臂报告(以 A7 为例)
python3 scripts/eval_report.py --games-dir runs/exp/A7/games --seed-key 0x5dfaa4cef265cd7 --seed-range 10000-13000 --checkpoint state.pth --source-dir runs/exp/A7 --out runs/exp/A7/report.csv
# 配对差
python3 scripts/exp_a1_delta.py            # A1 -> runs/exp/A1/delta_report.json
python3 scripts/exp_arch_b_delta.py        # ARCH_B -> runs/exp/ARCH_B/delta_e3a_vs_e2.json
python3 scripts/exp_a14_delta.py           # A14 -> runs/exp/A14/delta_confirm.json
# 门控与量化文档
cat docs/exp-gates.md; python3 scripts/analyze_alllast.py runs/1v3
```

## 8. 缺失产物

无。上述全部条目的产物均在磁盘实读;无"未完成"项需要补齐命令。
