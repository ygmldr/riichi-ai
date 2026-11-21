# RiichiFormer V3.2 — 本地研究权限清理与验收

状态：远程代码已提交，尚未运行用户本机完整测试。日期：2026-11-21。

用户希望当前项目用于个人非商业离线研究；无需再因旧权限登记表中 0 ALLOWED 停止所有训练开发。当前实现将“项目本地研究运行政策”与“第三方真实许可状态”拆分，允许记录了 operator_private_research 的训练 shard 通过原先重复的阻塞，而来源权利原样如实保留。

## 已经修改的代码

- riichiformer/data/shard.py：格式升级 rf-shard/3，ShardMetadata 含 research_policy、research_policy_hash。私有研究下 train_allowed=False 和 rights_status=RESTRICTED/UNKNOWN 不再自动拒绝；licensed_only 仍保留旧 Gate。
- scripts/rf_build_shards.py：CLI 默认 --research-policy operator_private_research，也可显式 --research-policy licensed_only。只有原 split=train、非隔离的代表记录能够写入训练 shard，未知 source_id 拒绝。旧 split registry 原样保留，不把真实权限改成 ALLOWED。
- riichiformer/data/loader.py：从 shard 元数据读取政策，只在私有研究模式下放开旧外部许可检查；目的类型、partition、source、schema/hash 检查保留。
- scripts/rf_train.py + riichiformer/train/checkpoint.py：记录研究策略、权利原始状态、source/split/provenance，并在 checkpoint resume 前严格比对来源记录。
- tests/rf_test_shards.py：新增政策保真、训练可读、holdout 拒绝、必要哈希测试；原来验证严格 Gate 的用例显式加 --research-policy licensed_only。

## 仍然不做的事

- 不将 configs/riichiformer/source_rights.csv 的第三方记录虚改成 ALLOWED；该表中的许可事实还是 UNKNOWN 或 RESTRICTED。
- 不将诊断、验证、选择或确认池混入训练；不因训练可启动就放开对外数据、权重的再分发，也不允许天凤在线凤凰桌 AI 对局。
- 不修改冻结 PR17 G3 报告。新 PR43 将分开记录 LOCAL_RESEARCH_TRAIN_SMOKE、EXTERNAL_RIGHTS_STATUS 和 ACTUAL_STRENGTH。

## Lead 首轮验证步骤

1. 保存用户未提交的代码和 vendor 子模块变更；核对当前本地 rf3-g3 基线。远程公开的是 filtered history，不能直接 merge 本地开发史。
2. **必须先从 GitHub 拉取最新远程 V3.2 分支**，再用独立 worktree 开发。不要把远程 filtered history 直接 merge/cherry-pick 到本地 rf/v3；不要 reset、clean 或强制推送。下面命令假设仓库主目录是 /home/ygsldr/riichi-ai 且远程名是 origin，Lead 应先检查 git remote -v 并根据实际目录适配：

   ```bash
   cd /home/ygsldr/riichi-ai
   MAIN_ROOT="$PWD"
   git remote -v
   git status --short
   git fetch origin refs/heads/rf/v3.2-policy-parallel-plan:refs/remotes/origin/rf/v3.2-policy-parallel-plan
   git rev-parse origin/rf/v3.2-policy-parallel-plan
   git worktree add -b rf/v3.2-local worktrees/v3.2-policy origin/rf/v3.2-policy-parallel-plan
   cd worktrees/v3.2-policy
   git rev-parse HEAD
   PYTHONPATH="$MAIN_ROOT/vendor/mortal/target/release" "$MAIN_ROOT/.venv/bin/python" -m pytest -q tests/rf_test_shards.py tests/rf_test_training.py tests/rf_test_source_rights.py tests/rf_test_research_policy.py
   ```

   如果本地已有 rf/v3.2-local 或 worktrees/v3.2-policy，先检查状态并复用或另起名字，**不要**删除现有分支/目录。这里使用的是远程 filtered 分支的树，不能擅自替代冻结的 rf3-g3；与原本地数据及 vendor 依赖应通过 MAIN_ROOT 的绝对路径引用。进入 worktree 后需重新设置 MAIN_ROOT 为真正的本地主工作树目录（不要误指新 worktree）。

3. 使用本地 H02 full 数据运行只有 10 场的 dry-run（本地原仓库根目录，**不是**新的 worktree 目录）：

   .venv/bin/python scripts/rf_build_shards.py --canonical-dir artifacts/riichiformer/h02/canonical --split-manifest artifacts/riichiformer/h02/split_full/split_manifest_L_v2.parquet --purpose train --limit 10 --out /tmp/rf32_private10.rfshard --research-policy operator_private_research --dry-run
4. 确认选中的 game_id 全属于训练分区，再将 --dry-run 换为 --write。校验 manifest 的 research_policy、research_policy_hash、source_id 和 rights_status，并用 ShardReader(purpose=train) 抽验。
5. 在 5070 Ti 上进行两步六层模型 smoke：
   .venv/bin/python scripts/rf_train.py --shard /tmp/rf32_private10.rfshard --train --out /tmp/rf32_private10-smoke.pt --device cuda --max-steps 2 --batch-size 4 --layers 6 --ffn 704 --state-hidden 512
6. 确保同来源同策略 checkpoint 能 resume，不同源/split/policy 不可 resume。最终 Lead 提交测试退出码、模型输出和新的 PR43 gate 状态。

注意：本会话没有在用户机器运行训练/真实数据测试，因此新代码尚不可视为验收通过。非商业目的也不自动赋予第三方许可；原始记录始终保留，外部使用依旧独立核查。