# RiichiFormer V2 — 原始基线冻结 (G0 / PR-00)

**冻结日期 (UTC)**：2026-11-20T17:39:10Z  
**目标仓库**：`git@github.com:ygmldr/riichi-ai.git`（工作目录 `/home/ygsldr/riichi-ai`）  
**证据机**：host `ccpc`，WSL2（`Linux-6.6.87.2-microsoft-standard-WSL2-x86_64`），RTX 5070 Ti 16 GB  
**机读证据**：`artifacts/riichiformer/preflight.json`（gitignored；其 SHA-256 记录在 `pr_reports/PR-00.md`）

> 本文件的职责只有一个：**证明 PR-00 时刻的代码、配置、权重、补丁、评测协议是什么**。
> 它不授权训练，也不表示 G1–G6 的任何门已通过。所有缺失项都写 `NOT_FOUND` / `UNAVAILABLE`，
> 本项目**没有、也不会**为了让它“通过”而生成任何假权重。

---

## 1. Git 与 submodule 冻结

| 项 | 值 | 说明 |
|---|---|---|
| repo HEAD | `aceb281602fb2703a1ed2067a006992cd9ddb680` | branch `main` |
| HEAD 提交 | `feat(publish): scripted filtered-history push with backup and verification` | 2026-11-20 |
| `vendor/mortal` index SHA | `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2` | `git ls-tree HEAD vendor/mortal` |
| `vendor/mortal` HEAD SHA | `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2` | 与 index 相同，未换 commit |
| submodule 工作树 | **dirty**：11 个 tracked 修改 + 3 个 untracked 新文件 | 这是 `patches/0001–0013` 被应用后的预期状态 |
| `[patch NNNN]` 标记数 | **29** | 12 个文件内，见下表 |
| submodule tracked diff SHA-256 | `9e30e96734295c50626061a778b4208ee93574e1308a5ff5cf72f61d48f1199b` | `git -C vendor/mortal diff \| sha256sum` |
| submodule 全量工作树 SHA-256 | `5dc5ba3bd5af6104c91c71b09592bde302ca1875ae09531f0dfdf5c357dca606` | tracked diff + 3 个 untracked 文件的拼接哈希 |

**主仓库脏文件（只记录，绝不 reset/覆盖）**：

```text
 M .gitignore                                          # PR-00 追加的忽略规则（本 PR 的改动）
 M .omo/notepads/fork-roadmap-steps-1-10/decisions.md  # 既有会话记录（用户改动）
 M .omo/notepads/fork-roadmap-steps-1-10/learnings.md  # 既有会话记录（用户改动）
 M vendor/mortal                                       # submodule 内容 dirty（补丁状态）
```

**submodule 脏文件清单**（`git -C vendor/mortal status --porcelain`）：

```text
 M libriichi/src/agent/mortal.rs
 M libriichi/src/dataset/gameplay.rs
 M libriichi/src/mjai/mod.rs
 M mortal/client.py
 M mortal/dataloader.py
 M mortal/engine.py
 M mortal/model.py
 M mortal/mortal.py
 M mortal/one_vs_three.py
 M mortal/player.py
 M mortal/train.py
?? libriichi/src/mjai/tokens.rs
?? mortal/event_data.py
?? mortal/rl_loss.py
```

3 个 untracked 文件（由 patch 0006/0008 引入，但**不在 `patches/` 里**，见风险 R-02）：

| 文件 | SHA-256 |
|---|---|
| `vendor/mortal/libriichi/src/mjai/tokens.rs` | `bc6b25f17987064dd4cbc72ad2835e86dbba9f1d1bdd4a7f7daedbeaa5fffe11` |
| `vendor/mortal/mortal/event_data.py` | `0e1ca9d7911cb5106f3d0d824f8acae361bb3e73209c9bbe9d4997006834d6ce` |
| `vendor/mortal/mortal/rl_loss.py` | `ba35461a23464511179c89978159aa13aa31f0568d9dee336ff1b17abb9b523d` |

`[patch NNNN]` 标记分布（29）：

| 文件 | 标记数 | 文件 | 标记数 |
|---|---|---|---|
| `mortal/train.py` | 6 | `mortal/model.py` | 8 |
| `libriichi/src/dataset/gameplay.rs` | 3 | `libriichi/src/agent/mortal.rs` | 2 |
| `mortal/one_vs_three.py` | 2 | `mortal/player.py` | 2 |
| `libriichi/src/mjai/tokens.rs` | 1 | `mortal/client.py` | 1 |
| `mortal/dataloader.py` | 1 | `mortal/engine.py` | 1 |
| `mortal/mortal.py` | 1 | `mortal/rl_loss.py` | 1 |

> `patches/0011` 从未存在（历史已知空号），不计为异常。
>
> **不要把 marker 数当作“全部补丁已应用”的证明**：12 个补丁里只有 5 个带 `[patch NNNN]`
> 标记（0002/0006/0007/0008/0010），`0001/0003/0004/0005/0009/0012/0013` 的补丁文件内
> **0 个** marker。补丁已应用的状态由 §1 的 dirty diff（11 改 + 3 新增）与全量哈希证明；
> 另按 `AGENTS.md`，`git apply --reverse` 的重叠 hunk 检查不可靠，禁止用它“验证干净”。

---

## 2. 补丁冻结（`patches/`）

| 补丁 | bytes | SHA-256 |
|---|---:|---|
| `0001-train-grp-dataloader-num-workers-0.patch` | 470 | `52a2c5142ec17e5d45e199b369342fd7a2c2f6a8d72099b096d97c3c2d71003f` |
| `0002-train-remove-per-step-cuda-sync.patch` | 1907 | `24e183445fe625f2c393cb5adfec2821040ba0ddcb9c031cd570af786be3169d` |
| `0003-gamma-discount-labels.patch` | 689 | `ce2555f20ee71b9eda14e47cb8a28c4db22f9ffd1534a53eed2541925bba79a7` |
| `0004-agari-guard-consistency.patch` | 588 | `9a9a4437c9fc87d2e8cc85669428cac018bddbd993d5a5b923f6b5f4e59a09cf` |
| `0005-train-holdout-val-loss.patch` | 6001 | `1df6f5fffbd0f1f5b347064cd2f499472b2044ea5f67c44c68abbca54c84d6f0` |
| `0006-event-stream-tokens.patch` | 20748 | `fd5f554679e10b08855ff1b75f06bc89b0522bd91c722e47af3d97c3b3c29ef3` |
| `0007-model-event-branch.patch` | 17255 | `f6017db8224497189f5e25c7b1e396355091999601988692c339cbe4ac1692ca` |
| `0008-rl-policy-kl.patch` | 10846 | `f05e96d6b55e4278fa1d2ab939d175b1552b871d017546edc91edb7835d7cb4c` |
| `0009-online-selfplay-config.patch` | 760 | `caff02d7cfb0cc7c253bc4d1dd26ff3c3edff53f36c1989b70d93fa7eb016a3d` |
| `0010-arena-event-tokens.patch` | 9930 | `e2b2a4b12a5d17e56c4932a6eccc68b684a2618734e97a184ba9dca380211327` |
| `0012-1v3-seed-start.patch` | 1171 | `c81a309776c72984266b9eb1f9e7d4744872a96d5fc46a3fbacfa096a83e99dc` |
| `0013-online-tail-event-pad.patch` | 2325 | `467dd03f6708208f292122d8d0b489be1f15554c7e254cd5ac770b18587ba454` |

**Rust 侧已构建产物**：`vendor/mortal/target/release/libriichi.so`，23,432,688 bytes，
SHA-256 `fc05371f8035283ce61147e440c82eac16506cbb45584410607746afcbd91eba`；
`.venv` 内 `import libriichi` 成功，`ACTION_SPACE=46`（与 V2 动作契约一致）。

---

## 3. 旧配置冻结（`configs/config.toml`）

- SHA-256：`c12bc5f319a615b65bd5aa71783897f871a981a5c5f5a69ac1934aa3699f9ee3`
- 6,071 bytes，177 行；所有路径为绝对路径 `/home/ygsldr/riichi-ai/...`。
- 关键字段（原文行号）：

| 字段 | 值 | 行 |
|---|---|---|
| `[control].online` | `true`（patch 0009 自博弈） | 5 |
| `[model].use_event_branch` | **`true`**（训练时构建 event 分支） | 52 |
| `[rl].use_pg` | `false` | 70 |
| `[1v3].seed_key` | `0x5dfaa4cef265cd7` | 90 |
| `[1v3].seed_start` | `30000` | 91 |
| `[1v3].games_per_iter` | `12000` | 92 |
| `[1v3].log_dir` | `runs/variance/pool_3/games/`（**冻结语料目录**，勿写入） | 94 |
| `[1v3.challenger].state_file` | `runs/exp/PINNED/mortal.pth.at-175200` | 138 |
| `[1v3.champion].state_file` | `checkpoints/mortal.pth.at-148800` | 147 |
| `[dataset].globs` | 2009–2022，共 14 个年度 glob | 24 |
| `[dataset].val_globs` | 2023（年度 holdout） | 27 |
| `[baseline.*].state_file` | `checkpoints/baseline.pth` | 116, 121 |

> **重要区分**：`config.toml` 的 `[model].use_event_branch = true` 只决定**新训练**如何搭 Brain；
> 评测/加载读 checkpoint 自己保存的 config（见 §4）。当前 live 权重 `mortal.pth` 的该值为
> `false`，即部署模型是**纯 CNN**，不包含 event 分支。

---

## 4. Checkpoint 冻结（`checkpoints/`）

6 个模型权重全部在 CPU 上真实 `torch.load(..., weights_only=False)` 成功；2 个 `file_index` 只做哈希、未作为模型加载。下表 `params` 为 `mortal`
state_dict 张量元素总数，不是模型 forward 的 FLOPs。

| 文件 | bytes | SHA-256 | steps | 保存时间 (UTC) | params | 说明 |
|---|---:|---|---:|---|---:|---|
| `checkpoints/mortal.pth` | 130,781,915 | `0a44613cb9c7edae1057cf6a2c160c3971c78130e18e07f28bb43e8f62abe532` | 396,000 | 2026-09-21 16:22:46 | 10,818,641 | **live 部署基线**；`use_event_branch=false`；无 event 权重 |
| `checkpoints/best.pth` | 130,780,507 | `7fe084d98f8475967ea0602b9d74980b3d5fc439ca78b8b9701110fc0dd5be77` | 220,000 | 2026-09-15 18:51:06 | 10,818,641 | 旧 best；saved config 无 `[model]` 段；无 event 权重 |
| `checkpoints/baseline.pth` | 43,691,001 | `83fd3929035d4aa0d4df3423a16dd028267e40bb54f7dc3f831e18f6bbccc6c9` | — | 2026-08-10 05:29:50 | 10,818,641 | tag=`random-baseline`，train.py 无条件加载 |
| `checkpoints/grp.pth` | 1,419,230 | `009014e4a4883434d7809ad32fd4fefa7840806494dc4cfa950ed5e6bc25aa29` | 78,000 | 2026-08-10 05:28:25 | 58,776 | GRP 网络（14 张量），非 DQN |
| `checkpoints/mortal.pth.at-148800` | 130,780,507 | `73ec8108f1b9c4ab3775107c7192dcbf72b97bbd47b64dc09a7da9adfd551f4e` | 148,800 | 2026-08-10 17:24:07 | 10,818,641 | 旧 champion (`old148800`)；无 event 权重 |
| `checkpoints/mortal.pth.at-175200` | 130,780,507 | `f0a611b97f318e0b32fd7ce97965da0a9299b959742da54f64051677c7dc93a9` | 175,200 | 2026-09-14 19:02:37 | 10,818,641 | PINNED challenger；与 `runs/exp/PINNED/mortal.pth.at-175200` **逐字节相同** |
| `checkpoints/file_index.pth` | 185,010,303 | `11bf5f644eb502b4cf6ad89b646164c7f3aabcc619eac8d4b9f10372e296048e` | — | — | — | dataloader 文件缓存，未作为模型加载；改 globs 即失效 |
| `checkpoints/grp_file_index.pth` | 200,875,735 | `c57dae7d6e29b504cefd11f4e488f3d24ce6a17d4b7b3597a79499a16b0cbc3b` | — | — | — | 同上（GRP） |

补充冻结事实：

- `mortal.pth` 与 `best.pth` **不是**同一文件（SHA 不同，steps 396k vs 220k）；
  `mortal.pth` 与 `mortal.pth.at-175200` 也不同；
- `runs/exp/PINNED/` 另有 `at-328000`（`b14e77ed…`）、`at-382000`（`fddf5c48…`）、
  `at-396000`（`0a44613c…`，与 live `mortal.pth` 相同）——这些是评测产物，不属于 git，
  由本表钉住；
- **legacy 快照没有 `[model]` 段**：`best.pth` / `at-148800` / `at-175200` / `baseline.pth` 的
  saved config 只有 `[control]` + `[resnet]`（`baseline` 另有 `tag`），因此加载时 event branch
  按关闭处理；只有 `mortal.pth` 带 `[model].use_event_branch=false`（且 `event_dim=128` 等键存在，
  但 state_dict 无 event 权重）；
- `best_perf` 在 `best.pth` / `at-175200` / `mortal.pth` 中都是 `{avg_rank: 1.0, avg_pt: 90.0}`
  的占位值，**不是**实测评估，禁止引用为性能；
- `file_index.pth` 顶层键为 `file_list`，`grp_file_index.pth` 为 `train_file_list`/`val_file_list`；
  二者是 dataloader 缓存，不是模型权重；
- `checkpoints/` 中另有 `mortal.pth.at-175200` / `at-148800` 两个快照（已列）。
- **UNAVAILABLE / NOT_FOUND：无**。PR-00 所需的 6 个权重/快照均在本地存在且可读。

> **泄漏结论必须精确**：旧 `patches/0006` 的 event tokenizer 会把对手摸到的牌写进 token
> （`Event::Tsumo` 不看 actor），且未编码自家起手 13 张。当前 live 权重
> `use_event_branch=false` 且 state_dict 无 event 权重，所以**不能**说“线上 Mortal CNN 违规”；
> 该缺陷属于未来 V2 若复用 0006 tokenizer 的风险（PR-09 负责新 tokenizer）。

---

## 5. 旧评测协议冻结（`scripts/eval_report.py`）

- SHA-256：`1fe95382fda3956efccbaf1876c799e1e5a7e2ae0c2f0803fb18140014f213b6`
- 17,434 bytes；解析 `<seed>_<seed_key十进制>_<split>.json.gz`，split a/b/c/d = 座位 0/1/2/3，
  按 `(seed, seed_key)` 的 4 局 set 做 2000 次 bootstrap，`RANK_PTS=(90,45,0,-135)`。
- `EXACT_GATE`（第 62–75 行）：

```python
EXACT_GATE = {
    "dir_basename": "1v3",
    "seed_key": 0x80A7B9E2DEC96FA,
    "seed_range": (10000, 10500),
    "sets": 500, "games": 2000,
    "dist": [508, 504, 527, 461],
    "avg_rank": 2.4705, "pt": 3.0825,
    "ignored_files": 4,
    "challenger": "mortal", "champion": "old148800",
    "source": "runs/1v3/eval_20260915_1835.log",
}
```

种子池/已消费区间的完整登记见 **`docs/riichiformer/EVAL_POOL_REGISTRY.md`**。
本文件不重复推断；任何“virgin/未消费”的主张必须在那里有证据。

---

## 6. 环境冻结（实际检测，非文档假设）

| 项 | 实际值 | 检测方式 |
|---|---|---|
| Python（调用） | 3.12.3, `/usr/bin/python3` | `python3 -V` |
| 项目解释器 | `.venv/bin/python` (3.12.3) | preflight 自动优先 `.venv` |
| PyTorch | `2.11.0+cu128` | 子进程 `import torch` |
| CUDA runtime | 12.8；`torch.cuda.is_available()=True` | 同上 |
| GPU | NVIDIA GeForce RTX 5070 Ti, `sm_120` (12.0), BF16=True | torch props + nvidia-smi |
| VRAM | torch: 17,066,360,832 B ≈ 16,276 MiB；nvidia-smi total 16,303 MiB | 两个来源都记录 |
| nvidia-smi | 不在 PATH，实际路径 `/usr/lib/wsl/lib/nvidia-smi`；driver 621.01 | WSL 特性 |
| WSL | 是（`6.6.87.2-microsoft-standard-WSL2`） | `platform.release()` |
| CPU / RAM | 14 threads / 15.6 GiB | `/proc/meminfo` |
| 磁盘 | 挂载点剩余 772.3 GiB / 总量 1,006.5 GiB | `shutil.disk_usage` |
| Rust | `rustc 1.97.1` / `cargo 1.97.1`（`~/.cargo/bin`，不在 PATH） | 显式路径探测 |
| pytest | `.venv` 内 9.2.0（PR-00 安装）；系统 Python **无** | `python -m pytest --version` |

> PR-01 的正式 benchmark 必须重新实测；上表只是 PR-00 的环境事实，**不是**性能数字。

---

## 7. 冻结风险登记（PR-00 不修复，后续 PR 处理）

| ID | 级别 | 风险 | 证据 | 处置 |
|---|---|---|---|---|
| R-01 | **P0（未来训练）** | `patches/0006` 的 event tokenizer 透出对手摸牌 `pai`，且未编码自家起手；任何复用它的 V2 训练都会泄漏隐藏信息 | `patches/0006-event-stream-tokens.patch`；master plan §1 已证实 | PR-09 重写 tokenizer + POV redaction 硬测试；**禁止**在新管线复用 0006 |
| R-02 | P1（复现） | submodule 有 3 个 untracked 源文件（`tokens.rs`/`event_data.py`/`rl_loss.py`），不在 `patches/` 内；单靠 `patches/` 无法完整重建当前树 | §1 哈希表 | 后续把它固化成编号 patch（不在 PR-00 改 vendor）；在此之前以 §1 的两个树哈希为准 |
| R-03 | P1（操作） | `git submodule update --init --force` / `git checkout` 会清掉已应用的补丁，使 G0 证据失效 | §1 dirty 状态 | **禁止**重置/覆盖；恢复只能靠 §1 哈希与补丁重放 |
| R-04 | P2（协作） | 主工作树有用户未提交修改（`.omo/*`） | `git status` | 只记录，不 reset、不 stash、不提交用户文件 |
| R-05 | P1（评测） | A14 confirm 池 `[70000,73000)` 已消费，不能再称 virgin | 见 EVAL_POOL_REGISTRY | PR-05 建立 split registry 时必须登记为 consumed |
| R-06 | P1（训练） | live `config.toml` 的 `[model].use_event_branch=true`，而部署权重是纯 CNN；若直接训练会新建 event 分支（并触碰 R-01） | §3, §4 | 训练前必须显式决定 arms；G2 未过不得训练 |
| R-07 | P2（评测） | `best.pth`(220k) 比 `mortal.pth`(396k) 旧且内容不同，混用会导致对照失效 | §4 | 评测必须写明所用 SHA，禁止只写“best/mortal” |
| R-08 | P2（数据） | `file_index.pth` 是 globs 缓存；改 `[dataset].globs` 后不删会读到旧索引 | AGENTS.md 既有约定 | 改 globs 必须删缓存（PR-05 后由新 manifest 取代） |
| R-09 | P2（产物） | `runs/exp/PINNED/`、`runs/1v3/`、`runs/variance/` 是冻结语料，未入 git | §4；config `[1v3].log_dir` | 只读；禁止写入，见 EVAL_POOL_REGISTRY |
| R-10 | P2（设备） | preflight 时 GPU 已有 ~4.9 GB 占用；单卡独占假设不成立 | nvidia-smi `memory.used` | PR-01/训练前必须持 `/tmp/opencode/train.lock` 并重测空闲显存 |
| R-11 | P1（实验安全） | `configs/config.toml` 是 **live 配置**：`state_file/best_state_file/runs/test_play/runs/train_play/[1v3].log_dir` 全指向 live 产物或冻结语料 `runs/variance/pool_3/games/` | 本文件 §3；20 个路径键全为绝对路径 | 任何实验必须 deep-copy + 重定向 + 读回断言；禁止直接把该文件交给新 arm（违背项目 anti-pattern） |
| R-12 | P2（证据） | `best_perf={1.0,90.0}` 是占位符，不是真实评估；若被当作 baseline 成绩会误导 | 见 §4 补充 | 只引用 `report.csv`/VERDICT，不引用 `best_perf` |

---

## 8. 复现与校验配方

```bash
# 1) 重新生成机读证据（只写 artifacts/，不下载/不训练）
python3 scripts/rf_preflight.py --output artifacts/riichiformer/preflight.json
#    -> 本文件 §1–§4 的所有 hash 都来自该 JSON

# 2) 测试（两种都能跑）
.venv/bin/python -m pytest -q tests/rf_test_preflight.py
python3 tests/rf_test_preflight.py

# 3) 手工核对
git rev-parse HEAD
git -C vendor/mortal rev-parse HEAD
git -C vendor/mortal diff | sha256sum
sha256sum checkpoints/*.pth
```

**基线聚合哈希**（required 文件 + `patches/*.patch` + `checkpoints/*.pth`，按
`sha256("relpath:sha256\n")` 排序拼接）：

```
baseline_files_hash = a5ea24edc00777f7df3cb1108030b743e83a0f574f9fedbd6afc8b33a5bb0a9b
```

该值在 PR-00 合并后应保持不变；任何变化都意味着基线被改动，必须解释。
（`preflight.json` 本身 gitignored；其生成时的 SHA 记录在 `pr_reports/PR-00.md`。）

---

## 9. 本 PR 的边界

- 未下载任何新数据，未启动训练/评测，未改 `vendor/mortal`，未改任何 checkpoint/config/patch。
- 新增文件：`scripts/rf_preflight.py`、`tests/rf_test_preflight.py`、本文件、
  `EVAL_POOL_REGISTRY.md`、`pr_reports/PR-00.md`、`handoff/`（规范副本）；`.gitignore` 仅追加。
- G0 状态：**PASS_WITH_WARNINGS**。warnings 仅来自“工作树本就有用户改动 + submodule 补丁态”，
  不是基线缺失。G0 通过**不**代表 G1/G2 通过，更不允许开始训练。
