# PR-09 审计测试矩阵（实际状态）

**基线 HEAD**：`aceb281602fb2703a1ed2067a006992cd9ddb680`  
**环境**：`.venv/bin/python` (3.12.3)，torch `2.11.0+cu128`，CUDA 12.8，RTX 5070 Ti；pytest 9.2.0 仅在 `.venv`。
系统 `/usr/bin/python3` 无 torch/pytest，因此所有命令使用 `.venv/bin/python`。

## 1. 单元/集成测试（已提交范围 PR-00～PR-06）

| 测试文件 | 数量 | 命令 | 结果 | 日志 |
|---|---:|---|---|---|
| `tests/rf_test_preflight.py` | 21 | `.venv/bin/python -m pytest -q tests/rf_test_preflight.py` | PASS（0） | `PR09_TEST_LOGS/rf_test_preflight.log` |
| `tests/rf_test_bench.py` | 16 | `.venv/bin/python -m pytest -q tests/rf_test_bench.py` | PASS（0） | `PR09_TEST_LOGS/rf_test_bench.log` |
| `tests/rf_test_scan.py` | 13 | `.venv/bin/python -m pytest -q tests/rf_test_scan.py` | PASS（0） | `PR09_TEST_LOGS/rf_test_scan.log` |
| `tests/rf_test_canonicalize.py` | 16 | `.venv/bin/python -m pytest -q tests/rf_test_canonicalize.py` | PASS（0） | `PR09_TEST_LOGS/rf_test_canonicalize.log` |
| `tests/rf_test_replay.py` | 6 | `.venv/bin/python -m pytest -q tests/rf_test_replay.py` | PASS（0） | `PR09_TEST_LOGS/rf_test_replay.log` |
| `tests/rf_test_split.py` | 12 | `.venv/bin/python -m pytest -q tests/rf_test_split.py` | PASS（0） | `PR09_TEST_LOGS/rf_test_split.log` |
| `tests/rf_test_quality.py` | 14 | `.venv/bin/python -m pytest -q tests/rf_test_quality.py` | PASS（0） | `PR09_TEST_LOGS/rf_test_quality.log` |
| **合计** | **98** | `.venv/bin/python -m pytest -q`（上述 7 文件） | **98 passed / 0 failed**（44.57s） | `PR09_TEST_LOGS/full_suite_committed.log` |

### PR-07 草稿（未提交，不计入上表）

| 测试文件 | 数量 | 命令 | 结果 | 日志 |
|---|---:|---|---|---|
| `tests/rf_test_drift.py` | 9 | `.venv/bin/python -m pytest -q tests/rf_test_drift.py` | PASS（0）但功能未完成 | `PR09_TEST_LOGS/wip_pr07_drift.log` |

含草稿的全量：**107 passed / 0 failed**（44.93s，`artifacts/riichiformer/audit_logs/full_suite.log`）。
**注意**：9 个 drift 测试通过只证明当前草稿的聚合/CI 单元正确，不代表 PR-07 完成；实际 drift 运行被中止，
没有 `DRIFT_REPORT.md`/CSV。

## 2. 真实产物/端到端检查（审计期间复核，不是单元测试）

| 检查 | 命令 | 结果 | 证据 |
|---|---|---|---|
| 只读 preflight | `python3 scripts/rf_preflight.py --output artifacts/riichiformer/preflight.json` | exit 0；status=WARN；0 blockers；G0=PASS_WITH_WARNINGS | preflight SHA `4d72961a…` |
| GPU benchmark | `.venv/bin/python scripts/rf_bench.py --config configs/riichiformer/benchmark.yaml --out artifacts/riichiformer/bench/gpu_baseline.json` | exit 0；4/4 shape PASS；无 OOM | bench SHA `db844fb3…` |
| 数据审计 | `.venv/bin/python scripts/rf_data_audit.py --roots datasets/ --mode scan-only --out artifacts/riichiformer/audit --sample-per-year 200` | exit 0；2,512,424 leaves / 18 zip / 0 bad；925.76s | audit SHA `ebb90909…` |
| canonical sample | `.venv/bin/python scripts/rf_build_canonical.py --sample 1000` | exit 0；1,000/1,000 valid；`full_corpus=false` | canonical summary |
| native replay | `.venv/bin/python scripts/rf_validate_replays.py --manifest artifacts/riichiformer/replay/manifest_1000.parquet --sample 1000 --workers 8` | exit 0；1,000/1,000；660,568 decisions；0 illegal | `PR-04.md` |
| split | `.venv/bin/python scripts/rf_build_split.py build ...` + `assert-no-overlap` | exit 0；2×2,512,424 行；`duplicate_game_ids=0` | `SPLIT_REGISTRY.md` |
| quality | `RF_PLAYER_HMAC_KEY=… .venv/bin/python scripts/rf_quality.py --sample 100000` | exit 0；400,000 rows；可靠性 INCONCLUSIVE | `quality_audit.md` |
| 静态编译 | `python3 -m compileall -q riichiformer scripts/rf_*.py tests/rf_*.py` | exit 0 | `PR09_TEST_LOGS/compileall.log`（空） |

## 3. 审计覆盖矩阵（Task 2）

| 断言/主题 | 已存在测试 | 结果 | 未覆盖 |
|---|---|---|---|
| 对手摸牌/起手 POV redaction | **无**（PR-09 未实现） | NOT_RUN（无实现） | 全部；需 `test_opponent_tsumo_redacted`、`test_start_kyoku_hands_redacted`、byte-equal |
| 自家摸牌可见 | 无 | NOT_RUN | 全部 |
| 未来/终局信息不进历史 | 无模型输入；quality 有 `target_rank_for_eval` 分离测试 | PARTIAL | Transformer/decision 层 `test_no_future_information` 缺失 |
| 跨局 memory fence | 无 | NOT_RUN | PR-22 |
| 46 mask / action 对齐 | PR-04 的 1,000 场 replay（0 illegal） | PASS（样本） | 全量、Pass/censor、立直后弃牌、杠候选 |
| action map 冻结/哈希 | 无 | NOT_RUN | PR-08 |
| schema roundtrip/版本拒绝 | 无 | NOT_RUN | PR-08 |
| censored 反应不伪造 Pass | 无 | NOT_RUN | PR-10 |
| 在线/离线 encoder 一致 | 无 | NOT_RUN | PR-11 |
| padding/因果 attention | 无 | NOT_RUN | PR-11/13 |
| 奖励四席轮换 | 无 | NOT_RUN | PR-16/30 |
| 数据来源/权利 | PR-02 audit + `source_rights.csv` | PASS（记录）/BLOCKED（训练） | 人工许可审核 |
| 时间切分 | PR-05 L/R registry | PASS（game_id 互斥） | 全量 canonical dedup |
| 去重 | PR-03 sample + split log-id | PARTIAL | 2.51M 全量 |
| 2026 完整性 | audit | PASS（记录：1 月） | 无 |

## 4. 日志清单

提交到仓库的日志（`docs/riichiformer/PR09_TEST_LOGS/`）：

```text
freeze_state.txt
test_matrix.txt
compileall.log
full_suite_committed.log
rf_test_preflight.log
rf_test_bench.log
rf_test_scan.log
rf_test_canonicalize.log
rf_test_replay.log
rf_test_split.log
rf_test_quality.log
wip_pr07_drift.log
```

未提交的大产物（gitignored）：`artifacts/riichiformer/{preflight.json,bench/,audit/,split/,quality/}`；
`artifacts/riichiformer/audit_logs/` 另存一份运行日志。PR-07 的 `artifacts/riichiformer/drift/` 为空
（运行被用户要求暂停时中止）。

## 5. 环境注意事项

- `nvidia-smi` 不在 PATH，实际在 `/usr/lib/wsl/lib/nvidia-smi`；Rust 在 `~/.cargo/bin`。
- GPU 非独占：测量时 torch 看到 15,009 MiB free 而 nvidia-smi 看到 ~4.8 GiB used；benchmark 数字
  不是独占性能。
- `.venv` 新增 PyYAML 6.0.3/pytest 9.2.0/pyarrow 22.0.0；preflight 本身不安装任何东西。
- 系统 Python 缺 torch/pytest，`python3 -m pytest` 不可用；报告中的命令全部用 `.venv/bin/python`。
