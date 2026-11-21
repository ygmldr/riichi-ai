# RiichiFormer V3.1 — 最终交接（G3 = BLOCKED_RIGHTS）

**日期**：2026-11-21（Asia/Shanghai）
**集成分支**：`rf/v3`
**冻结 tag**：`rf3-g3` = `a251872d70abf0fee93cdcd9156602f03f7b8373`
**测试**：484 passed, 1 skipped, 242 subtests（164.55 s）
**结论**：代码/数据链到 PR-17 全部完成；真实训练被 H-03 source_rights 硬阻塞，G3 **不是 PASS**。用户已选择暂停于此、先解决授权。

## 1. 已完成资产

| 组件 | 证据 |
|---|---|
| H-00 Git 保全 | 快照 manifest `ee8036b3…`、bundle `705665a6…`；未动用户未提交内容 |
| H-01 分数结算 | 供托双计/0-seat/kyoku_count/四家立直 pending 修复；614 局 / 6,352 转移 0 mismatch |
| H-02 canonical/dedup/split | 2,512,424 rows / 109 shards / 0 invalid / 0 duplicate / 0 quarantine；L/R split v2 各 2,512,424，moved=0 |
| H-03 权利 | 11 行、0 ALLOWED；fail-closed gate；人类未授权 |
| H-04 规则 | 51 tests + 197 subtests；R02/R25 FIX_NEEDED、R26 KNOWN_DIFFERENCE |
| PR-07R 漂移 | 3,200 局 / 2,106,070 decisions；S4 分类修复 |
| PR-08 Schema | `rf-schema-v3/2`；schema `12503d36…`、action `6cf1efc5…`、normalizer `37df861b…`；66 tests |
| PR-09 POV | 28 tests；ProjectedStreamHeader、相对 deltas、可见性 property tests |
| PR-10 Snapshot | 10k decisions / 0 illegal / 50 censored；tiles_left 已修；20 tests |
| PR-11 Encoder | 16 tests；144 组 offline/online byte-equal |
| PR-12 Shard | 23 tests；write-root/train gate/流式 hash；10 局 6,485 decisions |
| PR-13 Model | 5,217,582 params；51 tests；synthetic forward/backward |
| PR-14 Trainer | 11 tests；synthetic loss 2.379641→0.000445；rights-blocked 拒绝启动 |
| PR-15 Offline | 7 tests；真实 diagnostics NOT_RUN |
| PR-16 Arena | 17 tests + 17 subtests；mock 10/100/1000 hands，illegal=0 |
| PR-17 Gate | `BLOCKED_RIGHTS`；`PR17_GATE.md` / `PR-17.md` |

## 2. 关键产物路径（本机，gitignored）

- `artifacts/riichiformer/h02/canonical/{canonical_index.sqlite,canonical_log_index.parquet,summary.json,manifest.json}`
- `artifacts/riichiformer/h02/split_full/{split_registry_v2.json,split_manifest_L_v2.parquet,split_manifest_R_v2.parquet,split_run_log.jsonl}`
- `artifacts/riichiformer/git_safeguard/20261120T194540Z/`
- 日志：`docs/riichiformer/v3/logs/`

## 3. 真实训练为何不能开始

`configs/riichiformer/source_rights.csv` 的 11 个来源（Tenhou raw/old/recent、converter data/code、Mortal code、teacher weights、akochan、self-play、Majsoul、HF mirror）全部为 `UNKNOWN` 或 `RESTRICTED`，**0 个 `ALLOWED`**。因此：

- 没有合法 `train_allowed=True` 的 shard；PR-12/PR-14 的 rights gate 会拒绝。
- 不能产生真实 checkpoint、arena 强度、confirm 结果。
- 任何“九段/天凤位/超过 Mortal”的声明都缺乏合法数据与证据。

## 4. 解除阻塞

按 [RIGHTS_UNBLOCK_CHECKLIST.md](RIGHTS_UNBLOCK_CHECKLIST.md)：

1. 取得 C-EGG/Tenhou 书面授权，覆盖年份/派生 MJAI/训练/存储/再分发/隐私/期限；或
2. 明确许可的替代语料；或
3. 自建规则环境自博弈语料，通过 H-04/G2/H-03 人工复核。
4. 人类更新 `source_rights.csv`，运行 `rf_check_source_rights.py --purpose train` 必须 exit 0。
5. 重生成 split-v2 的 `train_allowed` 行；再用 `rf_build_shards.py --purpose train --write` 构建合法 shard。
6. PR-14 GPU smoke → 重新跑 PR-17；之前 G3 保持 BLOCKED_RIGHTS。

## 5. 禁止事项

- 不得把 `UNKNOWN/RESTRICTED` 来源或由它计算的教师标签并入训练。
- 不得把旧五段 Mortal 当作九段教师或强度证据。
- 不得在凤凰桌运行 AI 或规避平台检测。
- 不得直接公开推送当前 git 历史：HEAD 的 H-01 fixture 已 synthetic，但历史提交仍含旧真实牌谱片段；发布前必须 filtered-history 重写。
- 不得在未授权时启动 PR-18 之后的真实 A/B/训练/confirm。

## 6. 恢复开发的最小步骤

```bash
cd /home/ygsldr/riichi-ai
git switch rf/v3
git rev-parse rf3-g3
PYTHONPATH=vendor/mortal/target/release .venv/bin/python -m pytest -q \
  $(git ls-files 'tests/rf_test_*.py' | tr '\n' ' ')
# 阅读 docs/riichiformer/v3/PARALLEL_STATUS.md 与 INTEGRATION_LOG.md
# 授权后：source_rights.csv -> gate -> split train_allowed -> shards -> PR-14 -> PR-17
```

**状态**：等待人工授权；不启动真实训练。
