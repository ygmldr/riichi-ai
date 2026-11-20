# NOTICE

riichi-ai — 自制立直麻将 AI
Copyright (C) 2026 ygmldr and contributors

本程序是自由软件：你可以依据自由软件基金会发布的 GNU Affero 通用公共许可证
（第 3 版或你选择的任何更新版本）的条款重新发布和/或修改它。本程序按"现状"提供，
不附带任何担保。详见 `LICENSE`（AGPL-3.0 全文）。

## 上游与 fork 关系

本仓库是 [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal) 的 fork：

- 上游许可证：AGPL-3.0
- 本仓库 pin 的上游提交：`0cff2b52982be5b1163aa9a62fb01f03ce91e0d2`（`vendor/mortal`，git submodule）
- 本仓库的所有修改同样以 AGPL-3.0 发布

`vendor/mortal` 是 submodule，指向**未打补丁**的上游提交；本仓库对上游的每一处改动都
以 `patches/` 下的编号补丁记录。克隆后需按 `vendor/README.md` 的顺序应用补丁
（`patches/0001` … `0013`，其中 `0011` 因实验门控跳过而不存在）。

## 对上游的修改（摘要）

| 补丁 | 内容 |
|---|---|
| 0001 | GRP dataloader 的 `num_workers` 可配置 |
| 0002 | 去掉训练每步的隐式 CUDA 同步（legality assert 移到 CPU 张量、H2D 用 non_blocking） |
| 0003 | gamma 折扣标签修正（上游 #115）：立直移出、杠宣言移入折扣条件 |
| 0004 | agari guard 一致性（上游 #114）：训练侧与评估侧都开启和牌保险丝 |
| 0005 | `holdout_loss`（2023 年留出集）+ 年份 globs，训练期可见验证损失 |
| 0006 | 事件流 tokenizer（dataset 侧），与 arena 共用同一编码器 |
| 0007 | 事件分支模型（零初始化残差 Transformer）+ `brain_from_config` 严格加载 |
| 0008 | RL policy/KL 损失（在线自对弈用，默认关闭） |
| 0009 | 在线自对弈配置段 |
| 0010 | 事件流 tokenizer（arena/评估侧） |
| 0012 | 1v3 评估的 `seed_start` |
| 0013 | 在线尾部 event 批 padding（修 `torch.cat` 长度不齐崩溃） |

核对某补丁是否已应用：`grep -rn "\[patch " vendor/mortal`（不要用 `git apply --reverse`，
后续补丁会重叠前面的 hunk）。

## 数据与权重（均不随本仓库发布）

- 训练数据：Tenhou 公开牌谱（2009–2022），由上游的数据下载脚本获取。**本仓库不包含牌谱数据。**
- 留出集：2023 年牌谱，仅用于 `holdout_loss`，不参与训练。
- 训练权重（checkpoint）由上述代码训练得到，**不随本仓库发布**；如需分发请自行训练并遵守
  AGPL-3.0 与上游数据的使用条款。

## 第三方组件

- `vendor/mortal`：Equim-chan/Mortal（AGPL-3.0），见其自带 `LICENSE`。
- akochan（评估对手，critter-mj/akochan）以独立仓库形式本地使用，**不包含在本仓库中**，
  也不在发布产物里。其许可证见上游项目。
- `mjai-reviewer/` 是独立的嵌套仓库，不属于本仓库的发布内容。
