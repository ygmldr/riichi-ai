# RiichiFormer V3.1 — PARALLEL_STATUS

**PAUSED_BY_USER（2026-11-21）**：用户选择暂停在 `BLOCKED_RIGHTS`、先解决 source_rights 授权。暂停期间不启动真实训练、不推进 PR-18+ 真实实验；仅保留本地只读审计与代码证据。

**Lead 单写**。状态只按真实提交、真实测试与 gate 更新；子代理自述不能直接改这里。
**最后更新**：2026-11-21（PR-08..17 代码完成，484 passed；G3=BLOCKED_RIGHTS；PR-12/14/15 修复合入）

| task_id | owner | branch | base_sha | hard_deps | status | test | next_gate | GPU/IO reservation | last_commit |
|---|---|---|---|---|---|---|---|---|---|
| H-00 | Lead | `rf/v3` | `f23bb0d` | — | **ACCEPTED** | 8 passed | W0 closed | none | `rf3-base` (`59691f5`) |
| H-01 | Dev-A | `rf/h01-score` | `rf3-base` | H-00 | **MERGED / P1** | 18 passed + 31/1 real-corpus | PR-06 recompute; public-release fixture fix | none | `9d59d68` |
| H-02 | Dev-B | `rf/h02-dedup` | `rf3-base` | H-00 | **ACCEPTED (full)** | 48 passed; 2,512,424 rows; 0 dup/quarantine | full split v2 done | heavy-IO done | `11a92a1` |
| H-03 | Dev-C | `rf/h03-rights` | `rf3-base` | H-00 | **MERGED / ACCEPTED** | 40 passed, 13 subtests | gate wiring in PR-12 | none | `741513b` |
| H-04 | Dev-D | `rf/h04-rules` | `rf3-base` | H-00 | **MERGED / ACCEPTED** | 51 passed, 1 skipped, 197 subtests | PR-16 must flag R02/R25/R26 | none | `dae0f41` |
| PR-07R | Dev-DATA | `rf/pr07r-drift` | `rf3-base` | H-00 | **MERGED / ACCEPTED** | 16 passed; 3,200 games / 2,106,070 decisions | cite after W1 tag | light IO done | `da4b802` |
| PR-08 V3 | Schema Owner | `rf/pr08-schema` | `rf3-base` | H-00; H-01/H-04 feedback | **ACCEPTED (v3/2)** | 66 passed | PR-10/11 use v3/2 hashes | none | `edf5720` |
| PR-09 V3 | Schema Owner | `rf/pr09-pov` | `885b14f` | PR-08 | **ACCEPTED** | 28 passed (94 with schema) | PR-10 consumes header | none | `8c7cbe0` |
| PR-13 V3 | Dev-MODEL | `rf/pr13-model` | `885b14f` | PR-08 | **ACCEPTED** | 51 passed, 15 subtests | PR-14 consumes ModelBatch | short GPU smoke done | `8de6639` |
| PR-10 V3 | Dev-ENGINE | `rf/pr10-decision` | `8c7cbe0` | PR-08/09 + H-01 | **ACCEPTED** | 19 tests; 10k/0 illegal | PR-12 consumes samples | none | `80df45d` |
| PR-11 V3 | Dev-ENGINE | `rf/pr11-encoder` | `924e9f6` | PR-10 | **ACCEPTED** | 15 tests; 144 byte-equal | PR-12 | none | `dc14f7f` |
| PR-12 V3 | Dev-DATA | `rf/pr12-shards` | `2afb1ac` | PR-11 + H-02 split | **ACCEPTED (code) / training blocked** | 23 tests; 10-game 6,485 decisions | rights gate blocks real train | light IO | `2af7e13` |
| PR-14 V3 | Dev-ENGINE | `rf/pr14-trainer` | `af7cfbe` | PR-12/13 + H-03 | **ACCEPTED (synthetic)** | 11 tests | real train blocked by H-03 | short CPU smoke | `88479f9` |
| PR-15 V3 | Dev-ENGINE | `rf/pr15-offline` | `88479f9` | PR-14 | **ACCEPTED (synthetic)** | 7 tests | real diagnostics NOT_RUN | none | `5dd5123` |
| PR-16 V3 | Dev-EVAL | `rf/pr16-arena` | `de85040` | PR-11/13 + H-04 | **ACCEPTED (mock)** | 17 tests, 17 subtests | real arena blocked on checkpoint | none | `9da4786` |
| PR-17 G3 | Lead | `rf/pr17-gate` | `11a92a1` | PR-13..16 | **BLOCKED_RIGHTS** | 471 passed, 2 skipped | human rights approval required | none | `f31b415` |

## 登记表

- **开发基线**：`rf3-base` = `59691f57…`；集成分支 `rf/v3` HEAD 见 `INTEGRATION_LOG.md`。
- **公开 main**：`a3ccd853…`，与本地开发史无共同祖先；禁止 merge/reset/force-push。
- **source_rights**：11 行，0 个 `ALLOWED`；真实监督训练继续 blocked。`configs/riichiformer/source_rights.csv` 是唯一新 registry。
- **split registry**：PR-05 旧版保留；H-02 生成 `split_registry_v2`（100k fixture），全量待 canonical job 完成。
- **eval seed pools**：A `[10000,13000)`、B `[50000,53000)`、A14 confirm `[70000,73000)` 已消费/冻结；新 confirm 未分配。
- **GPU/IO 锁根**：`/home/ygsldr/riichi-ai/artifacts/riichiformer/locks`。
- **当前写代码 Agent**：Schema Owner（PR-08）；H-03 owner 修 gate；其余等待 gate。
- **全量 canonical job**：`bash-9138f154f85d4d10`，不要在该 job 结束前启动第二个重 IO 全量任务。
