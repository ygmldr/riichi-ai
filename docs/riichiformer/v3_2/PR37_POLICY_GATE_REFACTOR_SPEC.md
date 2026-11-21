# PR-37：权限 Gate 去耦与本地研究策略——技术实施规格

**负责人**：Dev-POLICY（独占模块）、Lead 负责批准；Reviewer 独立验收。**前置**：从 `rf/v3` 本地集成线的冻结 tag `rf3-g3` 派生新工作树；若远程只访问到 `riichiformer-v3-filtered`，使用其 SHA 作为审阅基线，不直接与本地历史 merge。**目标**：用户同意的内部研究不被旧 H-03 重复 Gate 误卡；外部权限证据不被伪造；标签、版本、泄漏保护不被削弱。

## A. 必须实现的概念分离

现有 `source_rights.csv` 原样保留为事实登记，不再同时充当“每个运行任务的操作许可”。引入明确的 `ResearchPolicy` 和 `PolicyDecision`。

```python
@dataclass(frozen=True)
class PolicyDecision:
    source_id: str
    action: Literal['audit','compile','synthetic_train','private_train','distribute']
    selected_profile: Literal['licensed','operator_private_research']
    operational_allow: bool   # 能否进入此项目运行链；不是第三方授权
    source_rights_status: Literal['ALLOWED','UNKNOWN','RESTRICTED']
    basis: Literal['verified_external','operator_attestation','self_generated']
    policy_hash: str
    evidence_ref: str | None
    reason: str
```

新增独立配置（建议用户本地文件，仅提交示例，不提交任何私密审批证明）：

```yaml
policy_version: rf-local-policy/1
profile: operator_private_research
scope: local_nonpublic_only
operator_decision: explicit_user_request_2026-11-21
allowed_local_sources:
  - tenhou_old
  - tenhou_recent
  - tenhou_raw_logs
  - tenhou_to_mjai_dataset
  - tenhou_to_mjai_code
  - mortal_code
  - teacher_local
  - akochan_code
  - selfplay_own
  - majsoul_optional
  - third_party_mirror_hf
release_allowed: false
network_upload_allowed: false
```

说明：用户已表示内部项目中以上五类全部允许，但上表的 allowlist 是**他的研发操作选择**，不是第三方授权证明，也不能覆盖官方平台 AI 对战禁令。对第三方 `RESTRICTED` 数据的实际训练有条款/版权风险；Agent 不得将它描述为已获得官方许可，应在执行前向用户展示真实来源状态及影响。本地运行的司法/合同判断由项目所有者自行承担；如有明确禁令应先寻求授权或独立使用自有数据。

实现层面不强制所有来源在 `source_rights.csv` 变成 `ALLOWED`，也不允许把记录为 `RESTRICTED` 的证据改成更宽松状态。对 `selfplay_own`、`synthetic_*` 允许 `basis=self_generated`，凭本地生成器版本/commit/seed 哈希证明来源，而不要求一个无意义的 HTTP 证据 URL。

## B. 行为矩阵（必须作为单元测试 fixture）

| 来源、目的 | verified / licensed profile | operator_private_research | 说明 |
|---|---|---|---|
| 本地代码构建/模块测试 | 依据软件代码许可，不读数据训练 Gate | 同左 | 例如 Apache2/AGPL/akochan 分开评估 |
| 合成模型单元测试 | ALLOW | ALLOW | 先验不会被天凤牌谱阻断 |
| 自有生成自博弈数据，私有训练 | 若可验证自有来源则 ALLOW | ALLOW | 生成器、种子、规则版哈希 |
| `RESTRICTED/UNKNOWN` 第三方牌谱，索引/审计 | 按适用使用范围单独判断 | 审计可单独建本地受限工件 | 不自动下载、不上传 |
| `RESTRICTED/UNKNOWN` 第三方牌谱，训练 | DENY，直到有授权 | 必须显式选择 operator policy 并将风险状态写入工件；**不标示已获许可** | 不等于法律授权，采用前需要自行评估条款 |
| 将第三方牌谱/衍生样本公开发布 | DENY，除非明确有权 | DENY | 不得由本地实验 profile 覆盖 |
| 外部天凤凤凰桌 AI 对局 | DENY | DENY | 平台规则，不在此项目执行 |

**避免模棱两可**：不建议在所有地方把 `train_allowed=True` 解释为“第三方已经授权”。新的 `train_eligible_under_policy=True` 只能说明本地策略通过；`external_license_status` 仍独立、可追溯。因旧 SQLite shard 格式 `rf-shard/2` 把两者合并，须版本化升级为 `rf-shard/3`；不能悄悄复用旧文件。

## C. 修改顺序、文件所有权

1. **Policy Owner 唯一写入 `riichiformer/policy/**`**：建 `research_policy.py`（纯函数、无网络、无隐式环境变量）和 `tests/rf_test_research_policy.py`；输入显式的 profile + source + action + evidence；输出 `PolicyDecision`。序列化、规范 hash、稳定错误码。
2. **Schema Owner（按 Lead 独占锁）**：更新 `riichiformer/data/shard.py` 的 `ShardMetadata/GameRecord`，引入 `external_license_status`, `training_policy`, `policy_hash`, `train_eligible_under_policy`。明确迁移升级为 `rf-shard/3`，禁止加载 old train shard 冒充新 profile。旧 audit shard 可以以只读迁移工具转换，保留原 sha 与 origin。
3. **Data Owner**：`scripts/rf_build_shards.py` 从已冻结的 split / source manifest 中保留 game_id、split、source、日期、原始状态；只对 `split=train` + 非 quarantine/representative 行计算训练策略决策。不要无条件篡改 `split_manifest_L/R_v2.parquet` 中旧 `train_allowed`；优先新建 `training_view_registry`，以 game_id → `policy_decision` join 产生 train view 和散列。
4. **Loader Owner**：`riichiformer/data/loader.py` 在加载前验证 `training_policy` 与 `policy_hash` 一致；默认只接受 `licensed`，显式 `--policy operator_private_research` 才可消费该 profile 的 train shard；依旧验证全部 per-game split、quarantine、source membership 和 schema/action/normalizer 哈希。只要 action/POV/split 不一致，仍然 P0 FAIL。
5. **Trainer Owner**：`scripts/rf_train.py` 透传显式 profile/config 文件，训练日志打印 `source_rights_status`、`operational_allow`、`basis` 与警示；checkpoint 保存 policy hash、source manifest hash、split hash、canonical hash、模型 config hash；严格 resume 必须逐项匹配。
6. **Lead 集成**：用合成 fixture + 自生成 4 POV 小牌谱做完整 end-to-end；第三方真实数据的实际使用遵循项目所有者的独立许可/条款判断，不作为发布 Gate 的通过证据。
7. **Publish Owner**：文档与 CI 独立检查任何对外输出，不受 local policy 影响；不能把 `operator_attestation` 当作可再分发授权。代码发行遵守 Apache/AGPL/akochan 各自条款，输出权重的派生许可单独审查。

## D. 命令行目标（规格，不代表目前仓库已有命令）

```bash
# 合成及自有数据默认可用于本地测试
python scripts/rf_check_policy.py --source selfplay_own --action private_train \
  --profile operator_private_research --config configs/riichiformer/local_research_policy.yaml

# 本地非公开研究，明确输出真实的第三方权限状态
python scripts/rf_build_shards.py --purpose train --policy operator_private_research \
  --policy-config configs/riichiformer/local_research_policy.yaml --write ...
python scripts/rf_train.py --train --policy operator_private_research \
  --policy-config configs/riichiformer/local_research_policy.yaml \
  --shard ... --model-config configs/riichiformer/mvp_s.yaml --out ...

# 任何 publish/export 不得接受私有研究 profile 提供的所谓许可
python scripts/rf_check_policy.py --source tenhou_old --action distribute \
  --profile operator_private_research
# -> DENY, report external_license_status separately
```

建议 config 采用“不提交人名、不保存秘密”的用户选择文件；可提交不含个人信息的示例。真实使用时需由用户手动存入本地配置，hash 写入产物。

## E. 实验/兼容/安全验收

- `strict_without_license_rejects_thirdparty_train`、`operator_profile_preserves_restricted_status`、`selfplay_provenance_no_http_required`。
- `operator_profile_does_not_allow_publish`、`wrong_policy_hash_fails_closed`、`wrong_source_hash_fails_closed`。
- `per_game_split_not_train_rejected`、`quarantine_not_trainable`、`opponent_hidden_tile_leak_fails`、`target_in_forward_fails`。
- `rf-shard/2` 只能按显式迁移计划读取为旧版本；不允许升级时自动删失 evidence。
- `checkpoint_resume_policy_mismatch_refused`、`checkpoint_resume_canonical_split_mismatch_refused`、`init_weights_does_not_resume_optimizer`。
- 不允许 local profile 自动下载 Tenhou、上传样本、执行云端训练、上传 HuggingFace 模型或进行平台 AI 对局；网络行为必须是独立用户授权/执行。
- Gate 报告区分 `G3_CODE_PASS`, `G3_SYNTHETIC_PASS`, `G3_PRIVATE_RESEARCH_ALLOWED_BY_OPERATOR`, `G3_LICENSED_TRAIN_PASS`，**不得把前两者写成真实训练成功**。

## F. 风险与回滚

- 单独新分支，不改公开 `main`、不 force push、不改现有本地 `rf3-g3` tag。
- 如果从 `rf/v3` 到远程 filtered branch 的历史不同，不用普通 `git merge`；按文件树或 cherry-pick -x 到指定共同基线重建。
- 每份实验记录携带：git SHA, `schema_hash`, `action_map_hash`, `normalizer_hash`, `canonical_hash`, `split_hash`, `policy_hash`, `source_ids`, `original_rights_status`, seed, date, dataset mode, license caveat。
- 新 policy code 一旦导致 P0 掩蔽隐藏信息、错误 split、未标注来源，立即回滚到旧 profile。

## G. 研发不再依赖哪些人为门槛

- 不要求每一行软件代码来源都作为训练数据来源审批；代码是否可用按其真实软件许可遵守即可。
- 不要求自有生成的测试数据拥有外部 HTTP 权利证明。
- 不要求已完成的 Schema、POV、BC smoke、Arena mock、规则 fixture 等开发等待 Tenhou 回复。
- 不要求新 split 恢复为全量 train_allowed，以便让 synthetic 自有数据开发；训练视图只对实际参与的 game_id 产生。
- 仍坚持来源跟踪、隐藏信息隔离、合法动作、安全模型恢复、对外发布范围和平台 AI 禁令。