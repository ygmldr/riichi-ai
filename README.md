# riichi-ai

自制立直麻将 AI —— 基于 [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal) 的 fork，
Dueling DQN + offline CQL + GRP 奖励，在 RTX 5070 Ti 上训练。

本仓库只发布**代码**：训练/评估流水线、实验脚本、对上游的补丁与实验记录。
训练权重和牌谱数据不在仓库里——`checkpoints/`、`datasets/`、`runs/` 都是本地产物。

## 训练状态

训练停在 396,000 步。12,000 / 24,000 局的评估显示，约 15 万步后进入平台期，
继续离线训练**没有可检测的提升**（判据和 CI 见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)、
[`docs/FORK_ROADMAP.md`](docs/FORK_ROADMAP.md)）。

## 快速开始

```bash
git clone --recursive https://github.com/ygmldr/riichi-ai.git && cd riichi-ai
bash scripts/setup_gpu_env.sh          # 训练机环境（CUDA/Rust/Python）
# 按 vendor/README.md 的顺序应用 patches/（submodule 指向未打补丁的上游）
bash scripts/build_libriichi.sh        # 构建 Rust/PyO3 的 libriichi.so

# 训练 / 评估前准备自己的 checkpoint（训练流程见 RUNBOOK.md）
# 并编辑 configs/config.toml（里面是训练机的绝对路径，必须改成你的）：
#   [grp.dataset] globs、[1v3.challenger].state_file、[1v3.champion].state_file
bash scripts/run_eval.sh configs/config.toml
python3 scripts/parse_eval.py runs/1v3/eval_*.log
```

训练还需要 `checkpoints/baseline.pth`（上游 Mortal 的基线权重，不在本仓库）；
配置项的逐条说明见 [`RUNBOOK.md`](RUNBOOK.md)。

## 仓库结构

```
scripts/     第一方代码：训练/评估/下载流水线 + exp_* 实验脚本
configs/     MORTAL_CFG TOML（config.toml = 当前可用配置）
patches/     对 vendor/mortal 的每一处改动（0001–0013，0011 不存在）
vendor/      mortal（submodule，pin 0cff2b5）+ akochan（本地，不发布）
docs/        研究路线、实验索引、门控判定
runs/        实验产物（大部分 gitignore；报告与小体积证据在库内）
```

## 相关文档

- 设计：[`docs/superpowers/specs/2026-08-09-riichi-mahjong-ai-design.md`](docs/superpowers/specs/2026-08-09-riichi-mahjong-ai-design.md)
- 计划：[`docs/superpowers/plans/2026-08-09-riichi-mahjong-ai-p0-p3.md`](docs/superpowers/plans/2026-08-09-riichi-mahjong-ai-p0-p3.md)
- 运行手册：[`RUNBOOK.md`](RUNBOOK.md)
- 实验索引：[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)

## 许可

AGPL-3.0。fork 关系、补丁清单与数据出处见 [`NOTICE.md`](NOTICE.md)。
