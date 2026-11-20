# RiichiFormer V2 — DATA_REPORT (PR-02, read-only)

**状态**：`PASS`（审计完成，0 个坏文件）· **training_blocked = true**（来源权利未知）  
**生成时间**：2026-11-20T18:22Z（UTC）· 耗时 **925.76 s**  
**git HEAD**：`aceb281602fb2703a1ed2067a006992cd9ddb680`  
**命令**：

```bash
.venv/bin/python scripts/rf_data_audit.py --roots datasets/ --mode scan-only \
    --out artifacts/riichiformer/audit --sample-per-year 200
```

**机读产物**（`artifacts/riichiformer/audit/`，均 gitignored）：

| 文件 | SHA-256 |
|---|---|
| `source_audit.parquet`（2,512,442 行） | `ebb90909558e16e57a221b0e773c923a5d19a3df84dc48806d13094b16543f52` |
| `yearly_counts.csv` | `eb0da93a1c1ecbfc70458b95b17b4feadc5700fec04ca7a7cd894f31d75ef5bd` |
| `source_rights.csv` | `604e4b9ffb5f690b15e527d3c7ee131a32d5c5e60f9fff80f61a87052446bf89` |
| `audit_summary.json` | `1052da649a39d5e408109fd9a724e7b3248f7363eea302a6a2e33d2ad1a3d61f` |
| `bad_files.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`（空文件） |
| `samples.jsonl` | `f7cbbb985adf72ca5d0db89158a858f3f6a62ea39b346ecf128a2feed7648e03` |

---

## 1. 方法（只读；无网络；无数据改写）

- `--roots datasets/`：先扫 18 个年度 zip（`ZipFile` central directory + 每个 member 的 CRC 由
  `testzip()` 全量校验），再扫 `datasets/extracted/<year>/` 的叶子日志；`grp_*_globs.txt` 等非日志文件跳过。
- 压缩类型**按 magic bytes** 判定（`1f 8b`=gzip，`PK`=zip，`{`=明文 JSON），**不看扩展名**；
  每个叶子文件都实际读取了文件头，因此 gzip/plaintext 统计是 **100% 实测**，不是抽样推断。
- 每一年用确定性哈希优先序抽 200 个叶子做严格 NDJSON 解析（逐行 `json.loads`，要求
  `start_game/start_kyoku/end_kyoku/end_game` 完整），另对每个 zip 抽 3 个 member；共 **3,654** 个 deep sample。
- 年度由路径中的显式年份目录或文件名 `20YY...` 前缀推出；**不**从 Tenhou 16 进制后缀猜年份
  （例如 `...-e2011cd6.mjson` 不会被误判为 2011）。
- 不提取玩家名/身份；`source_audit.parquet` 只有路径、年份、来源、大小、mtime、压缩、成员数。
- 没有下载、没有解压、没有 gzip 规范化、没有删除任何 zip。

## 2. 覆盖结果

| 年 | 月份（文件时间） | 叶子文件 | gzip | 明文 | 字节 | zip members | 抽样 kyoku |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2009 | 02-12 | 6,897 | 6,897 | 0 | 30,470,055 | 6,897 | 2,188 |
| 2010 | 01-12 | 70,228 | 70,228 | 0 | 309,994,504 | 70,228 | 2,126 |
| 2011 | 01-12 | 100,814 | 100,814 | 0 | 447,816,081 | 100,814 | 2,232 |
| 2012 | 01-12 | 116,985 | 116,985 | 0 | 519,244,883 | 116,985 | 2,194 |
| 2013 | 01-12 | 131,838 | 131,838 | 0 | 584,365,908 | 131,838 | 2,207 |
| 2014 | 01-12 | 141,132 | 141,132 | 0 | 624,382,401 | 141,132 | 2,177 |
| 2015 | 01-12 | 152,278 | 152,278 | 0 | 671,952,157 | 152,278 | 2,123 |
| 2016 | 01-12 | 159,663 | 159,663 | 0 | 703,138,281 | 159,663 | 2,109 |
| 2017 | 01-12 | 169,382 | 169,382 | 0 | 745,385,399 | 169,382 | 2,183 |
| 2018 | 01-12 | 173,546 | 173,546 | 0 | 762,599,797 | 173,546 | 2,148 |
| 2019 | 01-12 | 171,628 | 171,628 | 0 | 752,783,851 | 171,628 | 2,164 |
| 2020 | 01-12 | 218,816 | 218,816 | 0 | 961,252,523 | 218,816 | 2,190 |
| 2021 | 01-12 | 187,313 | 187,313 | 0 | 823,017,099 | 187,313 | 2,110 |
| 2022 | 01-12 | 167,627 | 167,627 | 0 | 1,062,401,680 | 167,627 | 2,174 |
| 2023 | 01-12 | 168,777 | 168,777 | 0 | 1,152,901,947 | 168,777 | 2,174 |
| 2024 | 01-12 | 184,424 | 184,424 | 0 | 1,254,717,029 | 184,424 | 2,166 |
| 2025 | 01-12 | 178,888 | 178,888 | 0 | 820,394,058 | 178,888 | 2,163 |
| 2026 | **01** | 12,188 | 12,188 | 0 | 56,000,498 | 12,188 | 2,189 |
| **合计** | | **2,512,424** | **2,512,424** | **0** | **12,282,818,151** | **2,512,424** | **39,017** |

**结论**：

1. 本地 **18 个年度 zip 全部通过 CRC**（`testzip()` 返回 `None`），`zip_members` 与解压后的叶子文件
   **逐年 1:1 完全相等**（2,512,424 vs 2,512,424）；`bad_files.jsonl` 为空。
2. **本地所有 2,512,424 个叶子日志都是 gzip**（magic `1f 8b`），包括 2023–2026。
   2025/2026 的 zip member **未压缩大小**分别约 9.67 GB / 0.66 GB，而解压目录仅 0.82 GB / 0.056 GB ——
   说明它们原本是明文，已被 `normalize_gzip.py` 就地 gzip 化。**zip 内的原始明文仍是明文**，
   PR-03 必须两种都正确解码。
3. **2026 只有 1 月**（12,188 个文件 / 1 个月份），2025 为完整 12 个月；2009 从 2 月开始。
   这与交接文档里的“2026 覆盖约 1 月”一致，但**不能**把 2026 当完整年度。
4. 抽样 3,654 场全部解析成功（39,017 个 `start_kyoku`，平均 10.68 局/场），
   `start_game/end_game` 均存在；没有损坏 gzip、损坏 zip、未知 magic 或空文件。
5. `file_index.pth` 旧报告中的“1,968,147 训练 logs / 168,777 个 2023 holdout”与本地文件数同量级
   （2023 = 168,777），但那是**文件数**，不是局数；本报告不把文件数称为“局数”。

## 3. 来源与权利（阻塞新训练）

所有本地数据的来源均为第三方转换的 Tenhou 年度归档；**本次审计离线、未核实 Tenhou 条款**：

| source | rights | allows_new_training | terms |
|---|---|---|---|
| `tenhou_old` (2009–2022) | `unknown` | **false** | https://tenhou.net/term.html |
| `tenhou_recent` (2023–2026) | `unknown` | **false** | https://tenhou.net/term.html |
| `teacher_local` | `unknown` | **false** | 本地 Mortal 权重，蒸馏条款未审计 |
| `selfplay_own` | `unknown` | **false** | 暂无自博弈数据；先过 G2 |
| `majsoul_optional` | `restricted` | **false** | 无许可，禁止采集 |

因此 `training_blocked=true`，`block_reasons` 明确列出 5 条。**本次审计不授权训练**；
G1 仍需要人工复核权利并把对应来源改为 `allowed`，且该操作必须有证据与复核日期。

## 4. 已知限制（不得省略）

- **只对抽样做了完整 JSON 解析**；全部 2.5M 文件的 magic/大小/mtime 是实测，但“每行 JSON 都能解析”
  只由 3,654 场抽样支持。PR-04 的 1,000 场 native replay 是另一个独立证据，不是全量。
- 2025/2026 的“明文 → gzip”发生在解压目录；**zip 内仍是原始明文**。任何后续工具必须 sniff magic，
  不能依赖 `.mjson` / `.mjson.gz` 扩展名（PR-03 的 canonicalizer 已按此实现）。
- 没有玩家名/账号/身份推断；`player_quality` 所需元数据（段位/R/日期）本地不存在，PR-06 只能返回
  `unknown/insufficient_history`，不能编造。
- `datasets/` 只有 11.44 GiB extracted（18 个 zip 共 11.90 GiB），与 AGENTS 里“40–60 GB”的预期不符；
  实际文件数与 zip member 数 1:1，说明当前是**完整解压**，差异来自压缩率/数据版本，不是缺年份。
- 磁盘扫描期间未删除或移动任何 zip；`datasets/` 的 mtime 未变（只读命令）。

## 5. 对后续 PR 的直接输入

- PR-03：叶子总数 **2,512,424**；建议先跑 1,000 场分层 sample 验证 canonicalize/dedup，再决定
  全量 canonical index 的运行时预算（全量 = 2.5M 场 × 约 1,582 事件的严格 JSON 解析）。
- PR-05：可直接用 `source_audit.parquet` 的 `root+relpath+year+source_id` 建 split；但
  **在来源权利改为 allowed 之前，train/val 只能用于审计与离线研究，不能启动新训练**。
- PR-04：从本报告可直接生成 1,000 场分层 replay manifest；native replay 结果见 `PR-04.md`。

## 6. 复现

```bash
# 审计（约 15 分钟，2.5M 文件；只读）
.venv/bin/python scripts/rf_data_audit.py --roots datasets/ --mode scan-only \
    --out artifacts/riichiformer/audit --sample-per-year 200

# 测试（13 个）
.venv/bin/python -m pytest -q tests/rf_test_scan.py
```

> 本报告没有下载、没有改写数据、没有推断身份。所有“未知/缺失”都保持 `unknown/NOT_FOUND`。
