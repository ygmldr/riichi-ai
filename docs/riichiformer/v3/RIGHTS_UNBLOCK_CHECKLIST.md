# Rights unblock checklist（G3 = BLOCKED_RIGHTS）

**当前结论**：代码与数据链到 PR-17 已完成；`source_rights.csv` 11 个来源 **0 个 ALLOWED**，因此没有合法真实训练 shard/checkpoint。本文是人类解除训练门所需的最小清单，不是法律意见。

## 1. 当前冻结证据

- 集成 tag：`rf3-g3` = `a251872d70abf0fee93cdcd9156602f03f7b8373`
- 测试：**484 passed, 1 skipped, 242 subtests**；日志 `docs/riichiformer/v3/logs/W3_full_suite.log`
- Full canonical：2,512,424 rows / 109 shards / 0 invalid / 0 duplicate / 0 quarantine
  - index sha256 `21d9077f04836a123025a5336e38a3746801ea5e2d5a63d45793cc8714efae4c`
  - summary/manifest sha256 `25e1a43d506ba70f20a0040943f04eccd7e0f00022905fa5755b583111e9fdce`
- Full split-v2：L/R 2,512,424 rows，moved=0，quarantine=0
  - L registry `86d7b5578bdb133d40596cf4a87659008a5d231ddb4fd775a5d54aa2ec482170`
  - R registry `7ffbe7c672b387c5c76773a74a45295c84faf002b40b6a42b54f95316ec118de`
- G3 报告：`docs/riichiformer/v3/PR17_GATE.md`、`docs/riichiformer/pr_reports/PR-17.md`

## 2. 需要取得的书面授权（至少一项）

### A. Tenhou / C-EGG 牌谱授权
必须覆盖以下每一项，写明年份/来源范围与有效期：
- 原始牌谱与第三方转换后的 MJAI 派生数据
- 本地存储、保留、删除
- 监督训练、模型权重、蒸馏/派生模型
- 内部评测与自博弈
- 再分发（代码、权重、样本、元数据）
- 署名/赞助/商业或非商业用途
- 玩家隐私/性别/账号等字段的处理
- 撤回与终止后的删除义务

联系方式记录：`support@c-egg.com`（官方手册要求一般麻将研究先联系）。

### B. 明确许可的替代语料
- 许可必须覆盖训练与预期再分发；仅有“可下载”或转换器代码许可证不够。
- 更新 `configs/riichiformer/source_rights.csv` 的 `storage_allowed/train_allowed/redistribute_allowed/status`，提交证据 URL、逐字引文、日期、人类 reviewer（不得含 AI/draft 标记）。

### C. 自建规则环境自博弈
- 必须由本项目规则引擎生成，不得通过天凤/雀魂排位自动对战。
- 先通过 H-04 规则差异隔离与 G2 合法性测试，再由人类复核 provenance/隐私。
- 生成后仍需把 `selfplay_own` 行改为 ALLOWED，并由人类 reviewer 签署。

### D. 教师权重（仅 PR-21）
- Mortal 代码 AGPL 不等于权重许可；需要作者书面许可或可复现的公开权重许可。
- 本地旧 checkpoint 的 provenance 未记录，不能默认用于蒸馏。

## 3. 授权后的操作顺序

1. 人工更新 `configs/riichiformer/source_rights.csv`，`rf_check_source_rights.py --purpose train` 返回 0。
2. 重新生成 split-v2 的 `train_allowed` 行（当前所有行 `train_allowed=0`）。
3. 用 `scripts/rf_build_shards.py --purpose train --write` 构建合法 shard；确认 per-game split/rights/source 全为 ALLOWED。
4. PR-14 `scripts/rf_train.py` 跑 GPU smoke → PR-17 重新验收；此前 G3 仍为 BLOCKED_RIGHTS。
5. 真实 arena/confirm 必须用新 seed 池并登记；不得复用已消费池。
6. 公开分支前必须做 filtered-history：HEAD 的 H-01 fixture 已是 synthetic，但历史提交仍含旧真实牌谱片段。

## 4. 未授权时允许/禁止

允许：
- 只读本地索引/审计、规则/代码测试、synthetic/mock 单元测试。
- 现有 2.51M canonical/split 的本地只读审计。

禁止：
- 真实监督训练、checkpoint、蒸馏、arena 强度结论、PR-30 confirm。
- 把任何 `UNKNOWN/RESTRICTED` 来源或由其计算的教师标签并入训练。
- 公开推送含旧真实牌谱片段的 git 历史。
