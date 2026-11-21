# H-00 — Git / 工作树安全基线（RiichiFormer V3.1）

**状态**：W0 PASS（Lead 已核对快照读回哈希）
**生成时间**：2026-11-21（Asia/Shanghai；快照 UTC `2026-11-20T19:45:40Z`）
**工具**：`scripts/rf_git_safeguard.py`（只读采集，不 clean/reset/stash/checkout）
**快照目录**：`artifacts/riichiformer/git_safeguard/20261120T194540Z/`（gitignored，本机）
**manifest SHA-256**：`ee8036b3a8811df7fc044e1c8c4927363fe8613c7406bcfc938ec9bf5d9c4775`
**repo.bundle SHA-256**：`705665a63ee8f24370cf9779bb6549e8450388af046c86e0628dd7eca3ba26af`
**读回验证**：49 个条目（42 个用户文件 + 6 个元数据 + bundle）全部 SHA-256 一致。

## 1. 真实 Git 事实（不是文档转述）

| 项目 | 值 |
|---|---|
| 工作树 | `/home/ygsldr/riichi-ai` |
| 采集时 HEAD | `f23bb0dc9a6f483686e3c0eb195b36d302d3b980` |
| 采集时分支 | `riichiformer-v2-audit-pr09`（= `origin/riichiformer-v2-audit-pr09`） |
| 本地 `main` | `aceb281602fb2703a1ed2067a006992cd9ddb680`（是审计分支的祖先） |
| `origin/main` | `a3ccd85380880c1b97ef230a5e8f7b4ec35ce5a3`（filtered-history 发布结果） |
| `merge-base origin/main HEAD` | **无**（`git merge-base` 退出码 1，无输出） |
| `origin/main` 是否 HEAD 祖先 | 否 |
| `vendor/mortal` gitlink | `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2` |
| `vendor/mortal` 工作树 | 11 个已修改文件 + 3 个未跟踪文件（历史补丁状态，未清理） |

**结论**：公开 `origin/main` 与本地审计/开发历史没有共同祖先。任何 `git merge`、`git reset --hard`、`git push -f` 都不允许；发布只能沿用 `scripts/publish_public.sh` 的 filtered-history create-only 流程。

## 2. 已保全的用户内容

快照复制了以下内容（完整清单见 `baseline.json` 与 `manifest.json`）：

- dirty tracked 4 项：`.gitignore`（H-00 新增 `/worktrees/`）、`.omo/notepads/fork-roadmap-steps-1-10/{decisions,learnings}.md`、`vendor/mortal`（submodule 指针）。
- 未跟踪 24 项，包括 `AGENTS.md`（根 + configs/docs/patches/scripts/vendor）、`package.json`、`package-lock.json`、两份中文研究文档、PR-07 草稿（`riichiformer/data/drift.py`、`scripts/rf_drift.py`、`tests/rf_test_drift.py`）、V3.1 pack、H-00 工具/测试。
- `vendor/mortal` 的 11 个修改文件、3 个未跟踪文件全文 + `worktree.diff`；可用 `git -C vendor/mortal apply` 恢复。
- 敏感文件不进入备份：`.env`、`*player_map*`、`*player_hmac*`、`*.salt`、`*.key`、`docs/riichiformer/private/`；本次实际跳过 0 个。

**未做**：没有删除、移动、stash 或提交任何用户未提交内容；没有改 vendor 子模块。

## 3. `.gitignore` 回归

历史 bug：`data/` 无锚点，会忽略第一方 `riichiformer/data/`。当前规则已改为 `/data/`，并新增 `/worktrees/`。`rf_git_safeguard.py ignored-check` 使用 `git check-ignore --no-index`（避免 tracked 文件永远显示“未忽略”的假阴性）：

| 探测路径 | 期望 | 实际 | 规则 |
|---|---|---|---|
| `riichiformer/data/replay.py` | 不忽略 | 不忽略 | — |
| `riichiformer/schemas/provenance.py` | 不忽略 | 不忽略 | — |
| `artifacts/riichiformer/probe` | 忽略 | 忽略 | `.gitignore:45` |
| `data/probe` | 忽略 | 忽略 | `.gitignore:14:/data/` |
| `datasets/probe` | 忽略 | 忽略 | `.gitignore:15` |
| `worktrees/probe` | 忽略 | 忽略 | `.gitignore:56:/worktrees/` |

## 4. 开发基线与并行工作树

- **DEV_BRANCH**：`rf/v3`（集成分支，从审计 HEAD 派生；不触碰 `main` 与审计分支）。
- **AUDIT_BASE_SHA**：`f23bb0dc9a6f483686e3c0eb195b36d302d3b980`。
- **H-00 BASE_SHA**：H-00 提交本身，以 annotated tag **`rf3-base`** 固定；子代理统一从 `git rev-parse rf3-base` 派生。
- **worktree 根**：`/home/ygsldr/riichi-ai/worktrees/`（已 gitignore，与主工作树共享 object store）。
- **worktree/branch**：
  - `worktrees/h01` → `rf/h01-score`（H-01）
  - `worktrees/h02` → `rf/h02-dedup`（H-02）
  - `worktrees/h03` → `rf/h03-rights`（H-03）
  - `worktrees/h04` → `rf/h04-rules`（H-04）
- **共享锁根**：`/home/ygsldr/riichi-ai/artifacts/riichiformer/locks`（所有 worktree 使用同一绝对路径；GPU/重 IO 任务用 `flock`）。
- 环境：所有 Python 命令使用 `/home/ygsldr/riichi-ai/.venv/bin/python`；需要 `libriichi` 时设 `PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release`。

## 5. 备份恢复步骤

```bash
# 1) 克隆完整 refs（bundle 含 main / 审计分支 / 本地 refs）
git clone artifacts/riichiformer/git_safeguard/20261120T194540Z/repo.bundle /tmp/rf-restore
# 1b) bundle 不含 submodule 内容；恢复 vendor 源码需初始化 pinned submodule
git -C /tmp/rf-restore submodule update --init vendor/mortal
# 2) 恢复 dirty/untracked 文件
cp -a artifacts/riichiformer/git_safeguard/20261120T194540Z/files/worktree/. /tmp/rf-restore/
# 3) vendor 补丁状态
git -C /tmp/rf-restore/vendor/mortal apply \
  /home/ygsldr/riichi-ai/artifacts/riichiformer/git_safeguard/20261120T194540Z/vendor/worktree.diff
# 4) 读回校验
.venv/bin/python scripts/rf_git_safeguard.py verify \
  artifacts/riichiformer/git_safeguard/20261120T194540Z
```

注意：bundle 只含已提交 refs，不含 dirty/untracked；后两者必须从 `files/` 恢复。备份与工作树在同一块磁盘，属于“可恢复”而非异地容灾；在获得外置介质前不得删除该目录。

## 6. W0 检查表

| 检查 | 结果 |
|---|---|
| 保全用户未提交/dirty 内容 | PASS（42 文件，SHA-256 读回一致） |
| 记录真实 remote/branch/SHA/子模块 | PASS（`baseline.json`） |
| 公开 main 与开发史无共同祖先的证据 | PASS（merge-base 为空，bundle 含两者 refs） |
| 第一方代码不被 `.gitignore` 漏掉 | PASS（`ignored-check`） |
| 可创建并行 worktree | PASS（H-00 提交后 `git worktree add` 验证） |
| 干净工作树可复现第一方代码 | PASS（见 `INTEGRATION_LOG.md` 的 W0 复现记录） |
| 未执行 merge/reset/clean/push -f | PASS |

**Lead 批准**：W0 PASS，允许从 `rf3-base` 向 H-01/H-02/H-03/H-04 四条 lane 派发子代理。
