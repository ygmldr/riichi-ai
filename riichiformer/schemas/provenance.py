"""Provenance and source-rights primitives for RiichiFormer V2.

Design rules (from `00_MASTER_PLAN.md` §3 and `04_ACCEPTANCE_MATRIX.md`):

* a file's **year** is derived from its local path, not from the Tenhou log id;
* a source's **rights** are never inferred from the converter's open-source license;
* `unknown` / `restricted` never enter *new* training data; `reviewed` alone is not
  enough either -- only an explicit `allowed` permits new training;
* player identity and names are **not** part of this schema; PR-06 may add an HMAC
  in a private local store, never here.

This module is stdlib-only so the audit can run without torch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

YEAR_MIN = 2009
YEAR_MAX = 2026
AUDIT_SCHEMA_VERSION = "rf-source-audit/1"

# Official Tenhou manual (the old term.html URL is 404 as of 2026-11-21;
# the paifu/AI clauses live in the manual page, without a stable #PAIFU anchor).
# The audit is offline, so this URL is *recorded*, not fetched.
TENHOU_TERMS_URL = "https://tenhou.net/man/"
# Converter project that produced the local zip layout; Apache/MIT code != log licence.
TENHOU_CONVERTER_URL = "https://github.com/NikkeTryHard/tenhou-to-mjai"


class SourceId(str, Enum):
    TENHOU_OLD = "tenhou_old"
    TENHOU_RECENT = "tenhou_recent"
    TEACHER_LOCAL = "teacher_local"
    SELFPLAY_OWN = "selfplay_own"
    MAJSOUL_OPTIONAL = "majsoul_optional"
    UNKNOWN = "unknown"


class RightsStatus(str, Enum):
    REVIEWED = "reviewed"
    ALLOWED = "allowed"
    UNKNOWN = "unknown"
    RESTRICTED = "restricted"


TRAIN_ALLOWED_STATUSES = frozenset({RightsStatus.ALLOWED})


def rights_allow_new_training(status) -> bool:
    """Only an explicit ``allowed`` permits adding a source to new training data."""
    try:
        return RightsStatus(status) in TRAIN_ALLOWED_STATUSES
    except ValueError:
        return False


def year_is_supported(year) -> bool:
    return isinstance(year, int) and YEAR_MIN <= year <= YEAR_MAX


def source_for_year(year) -> SourceId:
    """Protocol split used by the audit; unknown years are not silently bucketed."""
    if not isinstance(year, int):
        return SourceId.UNKNOWN
    if 2009 <= year <= 2022:
        return SourceId.TENHOU_OLD
    if 2023 <= year <= 2026:
        return SourceId.TENHOU_RECENT
    return SourceId.UNKNOWN


@dataclass(frozen=True)
class SourceRights:
    """One row of the source-rights registry (``source_rights.csv``)."""

    source_id: SourceId
    display_name: str
    terms_url: str
    rights_status: RightsStatus
    rights_scope: str
    reviewed_at_utc: str | None
    evidence: str
    notes: str = ""

    def __post_init__(self):
        if not isinstance(self.source_id, SourceId):
            raise ValueError(f"source_id must be SourceId, got {self.source_id!r}")
        if not isinstance(self.rights_status, RightsStatus):
            raise ValueError(f"rights_status must be RightsStatus, got {self.rights_status!r}")

    @property
    def allows_new_training(self) -> bool:
        return rights_allow_new_training(self.rights_status)

    def as_row(self) -> dict:
        row = asdict(self)
        row["source_id"] = self.source_id.value
        row["rights_status"] = self.rights_status.value
        row["allows_new_training"] = self.allows_new_training
        return row


def default_source_rights() -> list[SourceRights]:
    """The truthful starting registry: no licence has been reviewed in this project.

    The local data arrives as Tenhou archives converted by a third-party project.
    Neither the platform terms nor the actual permitted use has been verified by
    this repository, therefore both sources are `unknown` and block **new** training.
    Existing checkpoints/experiments are not retroactively re-licensed by this row;
    they are recorded in `BASELINE_LOCK.md` and `EVAL_POOL_REGISTRY.md`.
    """

    evidence = (
        "本地文件由第三方转换器生成；本次审计只读本地文件、无网络，"
        "未核实 Tenhou 使用条款/再分发范围。converter 的 Apache/MIT 代码许可不等于牌谱授权。"
    )
    common_notes = (
        "unknown 不进入新训练数据；仅允许做本地索引/审计。"
        "若后续 PR 要训练，必须先由人工核实条款并改成 allowed，记录复核日期与范围。"
    )
    return [
        SourceRights(
            source_id=SourceId.TENHOU_OLD,
            display_name="Tenhou 凤凰桌年度归档 (2009-2022, via tenhou-to-mjai)",
            terms_url=TENHOU_TERMS_URL,
            rights_status=RightsStatus.UNKNOWN,
            rights_scope="offline local audit only; redistribution/training rights unverified",
            reviewed_at_utc=None,
            evidence=f"{evidence} converter: {TENHOU_CONVERTER_URL}",
            notes=common_notes,
        ),
        SourceRights(
            source_id=SourceId.TENHOU_RECENT,
            display_name="Tenhou 近期年度归档 (2023-2026, via tenhou-to-mjai)",
            terms_url=TENHOU_TERMS_URL,
            rights_status=RightsStatus.UNKNOWN,
            rights_scope="offline local audit only; redistribution/training rights unverified",
            reviewed_at_utc=None,
            evidence=f"{evidence} converter: {TENHOU_CONVERTER_URL}",
            notes=common_notes,
        ),
        SourceRights(
            source_id=SourceId.TEACHER_LOCAL,
            display_name="本地教师权重 (Mortal checkpoints)",
            terms_url="local:checkpoints/",
            rights_status=RightsStatus.UNKNOWN,
            rights_scope="derived-use/distillation terms of upstream Mortal weights unverified",
            reviewed_at_utc=None,
            evidence="Mortal is MIT-licensed code; weight provenance/licence for distillation not audited",
            notes="PR-21 之前不得把教师 Q 用于新训练；先用本地冻结权重做只读参考。",
        ),
        SourceRights(
            source_id=SourceId.SELFPLAY_OWN,
            display_name="自建环境自博弈",
            terms_url="local:riichiformer",
            rights_status=RightsStatus.UNKNOWN,
            rights_scope="generated locally; ruleset legality still to be validated (PR-16/PR-28)",
            reviewed_at_utc=None,
            evidence="no self-play data exists yet at PR-02 time",
            notes="新生成数据必须先过 G2 动作/POV 测试；不得用线上平台自动对战生成。",
        ),
        SourceRights(
            source_id=SourceId.MAJSOUL_OPTIONAL,
            display_name="雀魂 (可选，跨平台 OOD)",
            terms_url="UNKNOWN",
            rights_status=RightsStatus.RESTRICTED,
            rights_scope="not authorised for this project; do not collect",
            reviewed_at_utc=None,
            evidence="no local Majsoul data and no permission; explicitly out of scope",
            notes="在书面许可前禁止采集/训练。",
        ),
    ]


def rights_block_reasons(rights: list[SourceRights]) -> list[str]:
    """Human-readable reasons why new training is blocked by the rights registry."""
    reasons = []
    for row in rights:
        if not row.allows_new_training:
            reasons.append(f"source_rights_{row.rights_status.value}:{row.source_id.value}")
    return reasons
