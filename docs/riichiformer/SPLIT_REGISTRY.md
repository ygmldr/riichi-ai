# RiichiFormer V2 — SPLIT REGISTRY (PR-05)

- 生成时间：2026-11-20T18:34:44Z
- inventory：`/home/ygsldr/riichi-ai/artifacts/riichiformer/audit/source_audit.parquet`（SHA-256 `ebb90909558e16e57a221b0e773c923a5d19a3df84dc48806d13094b16543f52`）
- schema：`rf-split/1`；一行 = 一个 `game_id`（四 POV 与全部标签共享同一 split）
- **所有来源当前 `unknown` → `train_allowed=false`；本文件不授权训练。**
- 输入 inventory：`source_audit.parquet` 的 2,512,424 个叶子日志（18 个 zip 的 2,512,424 个
  member 只用于 CRC/覆盖，不重复入 split）。`game_id` 来自 Tenhou log id；**duplicate_game_ids=0**。
- `assignment_mode=calendar`（月来自文件名 `YYYYMMDDHH…`）；`hash_fallback` 已实现但**未使用**。

## Protocol L（mode=calendar）

- rows：**2,512,424**
- `registry_hash`：`184f34dcbd26174c72f222a2136d3727c660e75c2a77c2811c9b01796a9af27f`
- `manifest_sha256`：`5e291ad09aa35ab6a392c24c25a4e5b1be97289aad4c6c938253215755f3f27b`
- blocked_source rows：2,512,424；train_allowed rows：0

| split | rows |
|---|---:|
| `external_diagnostic` | 375,500 |
| `train` | 1,968,147 |
| `val_holdout` | 168,777 |

## Protocol R（mode=calendar）

- rows：**2,512,424**
- `registry_hash`：`4ada926b931f94432b7bb7684f64c80677d1f62bd81198e076d4d2948dd069ec`
- `manifest_sha256`：`6fbf3465eee1be29cfcbba40c47876e0c2a092a24381e65481eb22c8129cb44e`
- blocked_source rows：2,512,424；train_allowed rows：0

| split | rows |
|---|---:|
| `confirm_blocked_previously_used` | 12,188 |
| `select` | 89,867 |
| `train` | 2,321,348 |
| `val` | 89,021 |

## 覆盖与互斥验证

- 每个协议内 `game_id` → 唯一 split；`undated` 与 `unknown_id` 均为 **0**（全部文件名可解析）。
- `assert-no-overlap` 实测：

```bash
.venv/bin/python scripts/rf_build_split.py assert-no-overlap \
  --manifest artifacts/riichiformer/split/split_manifest_L.parquet \
  --manifest artifacts/riichiformer/split/split_manifest_R.parquet
# OK ... rows=2512424 counts={...} duplicate_game_ids=0   (exit 0)
```

- 与旧仓库数字的**独立吻合**：旧 `file_index.pth` 报告约 1,968,147 个训练 logs、
  2023 约 168,777 个 holdout —— 本 registry 的 L-train=1,968,147、L-val=168,777 **逐字相等**。
  这证明年度边界与本地文件计数一致，但不代表旧训练用过哪些具体 game。
- 2025：val 89,021 + select 89,867 = 178,888（与审计 2025 文件总数一致）。
- 2026 confirm 只有 12,188（1 月），且 `previously_used_in_research=true`，因此被单独标成
  `confirm_blocked_previously_used`，**不能当 virgin confirm**。
- 2023–2026 共 544,277 行标记 `previously_used_in_research=true`。

## 规则

- **L（复现历史）**：train 2009–2022；val_holdout 2023；2024–2026 = `external_diagnostic`
  （旧 EX1 已使用，**不是 pristine confirmation**）。
- **R（近期训练）**：train 2009–2024；val 2025 H1；select 2025 H2；
  2026 = `confirm_blocked_previously_used`（本地仅 1 月，且 2024–2026 已被 EX1 消费）。
- month 来自 Tenhou 文件名 `YYYYMMDDHH…`；缺失 = `undated`，不猜。
- 非 Tenhou 文件名 = `unknown_id`，fail closed；不进入 train。
- `hash_fallback` 仅用于无日期语料，**不是时间外**；必须显式选择并记录。
- 每个协议内 `game_id` 必须唯一 split；`--assert-no-overlap` 验证。

> **覆盖缺口（必须保留）**：PR-03 的全量 canonical index 仍是 `full_corpus=false`（只做了
> 1,000 场 sample）；本 registry 的重复检测依赖 **Tenhou log id 唯一性**（本机 0 重复），
> **未**对 2.5M 场做 SHA 级/canonical 级去重。PR-05 的 train/val/select/confirm 集合互斥已用
> game_id 验证；若未来发现同一 game 的转换副本，必须在重建 registry 时并入同一 split。
>
> 权利状态是 registry hash 的一部分：一旦 source_rights 从 `unknown` 变为 `allowed`，
> `registry_hash` 会变化，必须重新冻结并记录新 hash；当前 hash 只对应“全 blocked”的状态。
