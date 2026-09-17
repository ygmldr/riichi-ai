# Wave-2 门控判定（todo 15）

- 日期：2026-09-17
- 计划：`.omo/plans/fork-roadmap-steps-1-10.md` todo 15；输入为 todo 12（A1）、13（A2）、14（A2b）的产物
- 用途：本文件是 todo 28/29（belief 头）是否开工的唯一门控依据；三条结论各自引用数字与产物路径

**判定行（机器可读）：**

```
gate_belief=block
```

一句话：A2（AUC 路线）触发预注册的 `block`，A2b（ECE 路线）的 triage 门没有触发（`ECE > 0.05 ⇒ 放行` 不成立），两条独立路线都放行不了 ⇒ belief 线不开工。A1 与 belief 门无关，单独按下文表述。

## ① A2 线性探针：1024-d 是否已编码对手隐藏信息（决定 todo 28 是否开工）

产物：`runs/exp/A2/results.json`、`runs/exp/A2/README.md`、`runs/exp/A2/PREREG.md`、`runs/exp/A2/supplementary.json`；脚本 `scripts/exp_a2_probe.py`（补充分析 `scripts/exp_a2_supp.py`）。

- 预注册主探针（`opp_any_tenpai`，262,339 决策点，209,610 train / 52,729 val，oracle 张量解码）：**AUC 0.9334**，bootstrap 95% CI **[0.9312, 0.9354]**，ECE **0.0099**，Brier 0.0943；随机标签对照 **0.5007**（3 seeds 0.4994 / 0.5002 / 0.5025）。
- 预注册放行规则（`results.json → verdict.rule`）：`allow iff |auc_random-0.5|<0.02 AND ctrl_auc>=0.90 AND auc>=0.65 AND CI_lo>0.5`。随机对照门通过，但**阳性对照 `ctrl_opp_any_meld_visible` AUC 0.7947 < 0.90** ⇒ 机器结论 `pipeline_valid=false`、`gate_belief=block`（运行后未改写）。
- 实质负结果（比对照技术细节更重要）：24 特征**公开信息**基线在同一 split 上 **AUC 0.9399 > 0.9334**；真正隐藏的目标 `opp_any_aka_in_hand` 仅 0.6055，公开基线 0.6158。即该表示没有超出公开信息的对手隐藏状态证据。
- 口径说明（诚实标注）：`pipeline_valid=false` 来自一个可论证为设定过高的阳性对照；post-hoc 补充证据表明测量路径本身是好的（原始 meld 平面 ⇒ AUC 1.0000；置换标签 ≈ 0.5；`Mish(phi)` 0.7725）。但两条读法都收敛到 `block`，因此判定不依赖这个歧义。

**结论 ①：A2 不放行，todo 28 不开工。**

## ② A2b 校准审计：校准缺口（门：ECE > 0.05 ⇒ 放行）

产物：`runs/exp/A2b/results.json`、`runs/exp/A2b/README.md`、`runs/exp/A2b/PREREG.md`（三个 gate-bearing ECE 在 `results.json → summary_ece`）；脚本 `scripts/exp_a2b_calibration.py`。

| 对象 | ECE | 95% CI | 与 0.05 门的关系 |
|---|---|---|---|
| (a) `a_probe_primary`（A2 探针概率） | 0.00987 | [0.0086, 0.0180] | ≤ 0.05 |
| (b) `b_grp_top1`（GRP 24 排列 top-1） | 0.02422 | [0.0214, 0.0282] | ≤ 0.05 |
| (c) `c_q_actionconf_tau1`（DQN Q，τ=1） | 0.03643 | [0.0346, 0.0382] | ≤ 0.05 |

- 三者**全部 ≤ 0.05**，预注册 triage 规则（`ECE > 0.05 ⇒ allow`）**不触发** ⇒ 没有值得在 Wave 4 专门建 belief 头去修的校准缺口。`results.json → gate` 的机器结论：`gate_three_default=block`、`gate_two_models=block`。
- **两个阈值为什么不同**（`results.json → gate.threshold_audit` / `threshold_delivery`）：0.05 是**开工前 triage**，问的是"是否存在足够大的校准缺口，值得为它启动一条训练线"；0.15 是 todo 29 的**交付门**（`Δpt 95%CI 不跨 0` 且 `ECE ≤ 0.15`），衡量一个已经训出来的模型绝对质量是否可交付。triage 更严（更小）是刻意的过滤器，两者不可互相替代。
- 不得当作门触发器的数字：τ=0.05（部署温度）臂 ECE **0.1897**，是决策规则把 Q 压成近 0/1 的产物（`boltzmann_temp=0.05`），不是模型概率，仅作透明度报告；若误读它会把两条门都误判为触发。
- 自检：合成完美校准数据 ECE **9.3e-18**；A2 探针概率**逐位复现**（`delta_auc = delta_ece = delta_brier = 0.0`）。

**结论 ②：ECE 路线同样不放行。**

## ③ A1 牌河顺序实验：表述（与 belief 门无关）

产物：`runs/exp/A1/VERDICT.md`、`runs/exp/A1/delta_report.json`、`runs/exp/A1/PREREG.md`；脚本 `scripts/exp_a1_shuffle.py`、`scripts/exp_a1_delta.py`。协议：challenger `runs/exp/PINNED/mortal.pth.at-175200` vs champion `checkpoints/mortal.pth.at-148800`，seed 窗口 `[30000,33000)` = 3000 sets / 12000 局每臂。

| 对比 | Δavg_rank | 95% CI | 判定 |
|---|---|---|---|
| `shuffle − original` | **+0.045833** | **[+0.024000, +0.068917]** | CI 排除 0：**模型确实使用打牌顺序** |
| `recency_zero − original` | −0.014583 | [−0.032002, +0.003671] | CI 跨 0：12k 分辨率下未检出 rank 效应 |

- **正面发现**：按窗口内 item 槽位做保持多重集的置换，会让固定挑战者显著变差；+0.046 ≈ 1.9 倍 12k MDD（0.0239）。因 `kawa_overview_*`（顺序不敏感的字牌集合概览）保持完整，说明**顺序信息在 tile-set 概览之外仍是承重的**。次要指标 Δpt = −4.095，CI [−5.910, −2.434]，方向一致。
- **零结果**：`recency_zero` 的 rank 点估计 0.0146 小于 12k MDD 0.0239，CI 跨 0；这是"此分辨率下未检出"，不是"recency 无意义"。次要 Δpt = +1.444，CI [+0.064, +2.749] 是**弱非门控信号**（低于 1.86 非配对 MDD，方向对挑战者有利，记为 unresolved，不参与判定）。
- **噪声地板（配对所有 null 读数）**：`runs/variance/README.md`，12k 局只能分辨 ≈ **0.0239 avg_rank / 1.86 pt**（80% 功效）；σ_set(avg_rank) 0.4678、σ_set(pt) 36.37；反解 `n ≈ 6.87/Δrank²`、`n ≈ 41525/Δpt²`。任何小于该尺度的效应都以"未检出"计。

**结论 ③：打牌顺序是承重信息（A1 的正面结果）；recency 置零在 12k 下没有可检出的 rank 效应，受上述地板限制。**

## 门控后果（standing decision）

- `gate_belief=block` ⇒ **todo 28 跳过**（todo 28 是双锁门控，前置写明 `gate_belief=allow`），**todo 29 随之不跑**。两者是**门控跳过（gate-skipped）**，属计划记录在案的有效结果，不是静默丢弃；Wave 4 执行到它们时按本文件执行跳过，并在本文件确认。
- todo 29 的交付门（`Δpt 95%CI 不跨 0` 且 `ECE ≤ 0.15`）只在 belief 线真正运行时适用，本次**不适用**（不是"未过交付门"，是"门控未放行、线未开"）。
- 本判定不阻断 Wave 3；计划依赖图中 `15 → 28 → 29` 的被跳过部分即在此终止。

## 补齐命令（仅当上述产物缺失时需要；本次三组产物齐全，全部为实读）

```bash
# A2（~13 min）与 post-hoc 补充（~8 min）
source scripts/activate_env.sh
python scripts/exp_a2_probe.py; python scripts/exp_a2_supp.py
# A2b（~4 min）
python scripts/exp_a2b_calibration.py --selftest && python scripts/exp_a2b_calibration.py
# A1 三臂（每臂 ~35-44 min）+ 报告
python scripts/exp_a1_shuffle.py --arm <original|shuffle|recency_zero>
python scripts/exp_a1_delta.py
```

## 证据

本次判定实读的产物：`runs/exp/A2/{results.json,README.md,PREREG.md,supplementary.json}`、`runs/exp/A2b/{results.json,README.md,PREREG.md}`、`runs/exp/A1/{VERDICT.md,delta_report.json,PREREG.md}`、`runs/variance/README.md`。证据存档 `.omo/start-work/evidence/T15.txt`。
