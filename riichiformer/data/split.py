"""Frozen train/val/select/confirm split registry for RiichiFormer V2 (PR-05).

Two incompatible protocols are supported (see `00_MASTER_PLAN.md` §3.4):

* **L (reproduce history)**: train 2009–2022, validation/holdout 2023,
  2024–2026 are external diagnostics, *not* a pristine confirmation set.
* **R (recent training)**: train 2009–2024, val 2025 H1, model-select 2025 H2,
  confirm 2026 — but the local 2026 data is only January and 2024–2026 was already
  used by prior research (EX1), so 2026 is labelled
  ``confirm_blocked_previously_used`` and **must not** be called virgin.

Calendar months come from the Tenhou filename prefix (``YYYYMMDDHH…``); there is no
per-game timestamp inside the logs.  If a month is missing in protocol R, the game is
``undated`` (fail closed) rather than guessed.  A deterministic ``hash_fallback`` mode
exists for corpora without usable dates, but it is explicitly **not** temporal and is
never selected silently.

Split assignment is per ``game_id`` (one game = four POVs + all decision labels), so
no POV can leak across splits.  Rights are part of the registry hash: while sources are
``unknown`` every row has ``blocked_source=true`` and ``train_allowed=false``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from riichiformer.schemas.provenance import (
    RightsStatus,
    SourceId,
    default_source_rights,
    source_for_year,
)

SPLIT_VERSION = "rf-split/1"
PROTOCOL_L = "L"
PROTOCOL_R = "R"
PROTOCOLS = (PROTOCOL_L, PROTOCOL_R)
TENHOU_ID_RE = re.compile(r"^\d{10}gm-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{8}$")
MANIFEST_COLUMNS = [
    "split_version", "protocol", "assignment_mode", "game_id", "source_id", "year",
    "month", "split", "relpath", "root", "blocked_source", "rights_status",
    "train_allowed", "previously_used_in_research", "undated", "holdout",
    "duplicate_group_size", "reason",
]


class SplitError(ValueError):
    """Invalid split configuration / overlap / fail-closed condition."""


@dataclass(frozen=True)
class SplitConfig:
    protocol: str
    mode: str = "calendar"          # calendar | hash_fallback
    hash_seed: int = 20261121
    version: str = SPLIT_VERSION

    def __post_init__(self):
        if self.protocol not in PROTOCOLS:
            raise SplitError(f"protocol must be one of {PROTOCOLS}, got {self.protocol!r}")
        if self.mode not in ("calendar", "hash_fallback"):
            raise SplitError(f"mode must be calendar|hash_fallback, got {self.mode!r}")
        if self.mode == "hash_fallback":
            # Not a temporal split; recorded in every row and in the registry.
            pass


def month_from_relpath(relpath: str):
    """Tenhou names start with ``YYYYMMDDHH``; return a valid 1..12 month or None."""
    base = Path(relpath).name
    match = re.match(r"^(\d{4})(\d{2})\d{2}", base)
    if not match:
        return None
    month = int(match.group(2))
    return month if 1 <= month <= 12 else None


def game_id_from_relpath(relpath: str):
    """Return ``(game_id, verified)``; unverified ids are fail-closed into ``unknown_id``."""
    base = Path(relpath).name
    for suffix in (".mjson.gz", ".mjson", ".json.gz", ".json"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    if TENHOU_ID_RE.match(base):
        return f"tenhou:{base}", True
    digest = hashlib.sha256(relpath.encode("utf-8")).hexdigest()
    return f"sha256:{digest}", False


def _rights_status_map():
    return {row.source_id.value: row.rights_status.value for row in default_source_rights()}


def _previously_used(year) -> bool:
    # 2023 was the old holdout; 2024–2026 were consumed by EX1.  2009–2022 is the
    # historical training corpus, not a confirmation set.
    return isinstance(year, int) and year >= 2023


def assign_split(protocol: str, year, month, mode: str = "calendar",
                 game_id: str = "", hash_seed: int = 20261121):
    """Pure assignment: returns ``(split, reason)``.

    ``undated`` and ``unknown_id`` are first-class split labels; nothing is silently
    pushed into train.  ``hash_fallback`` is order-preserving only in the sense that
    it is deterministic; it is **not** a temporal split.
    """
    if mode == "hash_fallback":
        digest = int.from_bytes(
            hashlib.sha256(f"{hash_seed}\x00{game_id}".encode("utf-8")).digest()[:8], "big"
        )
        bucket = digest % 100
        if protocol == PROTOCOL_L:
            split = "train" if bucket < 80 else ("val_holdout" if bucket < 90 else "external_diagnostic")
        else:
            split = "train" if bucket < 80 else ("val" if bucket < 90 else "select")
        return split, "hash_fallback_not_temporal"

    if not isinstance(year, int):
        return "undated", "missing_year"
    if protocol == PROTOCOL_L:
        if 2009 <= year <= 2022:
            return "train", "protocol_L_train_2009_2022"
        if year == 2023:
            return "val_holdout", "protocol_L_holdout_2023"
        if 2024 <= year <= 2026:
            return "external_diagnostic", "protocol_L_ood_2024_2026_not_pristine"
        return "undated", f"year_out_of_range:{year}"

    if 2009 <= year <= 2024:
        return "train", "protocol_R_train_2009_2024"
    if year == 2025:
        if month is None:
            return "undated", "protocol_R_missing_month_2025"
        if 1 <= month <= 6:
            return "val", "protocol_R_val_2025_H1"
        return "select", "protocol_R_select_2025_H2"
    if year == 2026:
        # Local 2026 is January only and was used by EX1; never call it virgin.
        return "confirm_blocked_previously_used", "protocol_R_confirm_2026_previously_used"
    return "undated", f"year_out_of_range:{year}"


def build_split_rows(inventory: list[dict], cfg: SplitConfig) -> list[dict]:
    """Map audit/canonical inventory rows to split rows (one row per game)."""
    rights = _rights_status_map()
    rows = []
    id_counts = defaultdict(int)
    for item in inventory:
        gid, verified = game_id_from_relpath(item["relpath"])
        id_counts[gid] += 1
        year = item.get("year")
        month = month_from_relpath(item["relpath"])
        source = item.get("source_id") or source_for_year(year).value
        status = rights.get(source, RightsStatus.UNKNOWN.value)
        blocked = status != RightsStatus.ALLOWED.value
        split, reason = assign_split(cfg.protocol, year, month, cfg.mode, gid, cfg.hash_seed)
        if not verified:
            split, reason = "unknown_id", "unverified_game_id"
        rows.append({
            "split_version": cfg.version,
            "protocol": cfg.protocol,
            "assignment_mode": cfg.mode,
            "game_id": gid,
            "source_id": source,
            "year": year,
            "month": month,
            "split": split,
            "relpath": item["relpath"],
            "root": item.get("root"),
            "blocked_source": blocked,
            "rights_status": status,
            "train_allowed": (split == "train") and (not blocked),
            "previously_used_in_research": _previously_used(year),
            "undated": split == "undated",
            "holdout": split in ("val", "val_holdout", "select"),
            "duplicate_group_size": 1,  # filled below
            "reason": reason,
        })
    for row in rows:
        row["duplicate_group_size"] = id_counts[row["game_id"]]
    rows.sort(key=lambda r: (r["split"], r["game_id"]))
    return rows


def split_counts(rows: list[dict]) -> dict:
    counts = defaultdict(int)
    for row in rows:
        counts[row["split"]] += 1
    return dict(sorted(counts.items()))


def _split_digest(rows: list[dict]) -> dict:
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row["game_id"])
    digests = {}
    for split, ids in sorted(by_split.items()):
        ids.sort()
        digest = hashlib.sha256()
        for gid in ids:
            digest.update(gid.encode("utf-8"))
            digest.update(b"\n")
        digests[split] = {"count": len(ids), "sha256": digest.hexdigest()}
    return digests


def registry_hash(rows: list[dict], cfg: SplitConfig, rights_snapshot: dict | None = None) -> str:
    payload = {
        "split_version": cfg.version,
        "protocol": cfg.protocol,
        "mode": cfg.mode,
        "hash_seed": cfg.hash_seed if cfg.mode == "hash_fallback" else None,
        "counts": split_counts(rows),
        "split_digests": _split_digest(rows),
        "rights": rights_snapshot if rights_snapshot is not None else _rights_status_map(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def assert_no_overlap(rows: list[dict]) -> None:
    """Every game_id must belong to exactly one split within a protocol."""
    seen = {}
    conflicts = []
    for row in rows:
        gid = row["game_id"]
        previous = seen.get(gid)
        if previous is None:
            seen[gid] = row["split"]
        elif previous != row["split"]:
            conflicts.append((gid, previous, row["split"]))
    if conflicts:
        sample = "; ".join(f"{gid}: {a} vs {b}" for gid, a, b in conflicts[:8])
        raise SplitError(f"{len(conflicts)} game_id(s) cross split boundaries: {sample}")


def manifest_schema():
    import pyarrow as pa

    return pa.schema([
        ("split_version", pa.string()), ("protocol", pa.string()), ("assignment_mode", pa.string()),
        ("game_id", pa.string()), ("source_id", pa.string()), ("year", pa.int16()),
        ("month", pa.int8()), ("split", pa.string()), ("relpath", pa.string()),
        ("root", pa.string()), ("blocked_source", pa.bool_()), ("rights_status", pa.string()),
        ("train_allowed", pa.bool_()), ("previously_used_in_research", pa.bool_()),
        ("undated", pa.bool_()), ("holdout", pa.bool_()),
        ("duplicate_group_size", pa.int32()), ("reason", pa.string()),
    ])


def write_manifest(rows: list[dict], path: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = manifest_schema()
    table = pa.Table.from_pydict(
        {name: [row.get(name) for row in rows] for name in schema.names}, schema=schema
    )
    pq.write_table(table, path, compression="zstd")
    return table.num_rows


def read_manifest(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def assert_manifest_no_overlap(path: Path) -> dict:
    rows = read_manifest(path)
    assert_no_overlap(rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "counts": split_counts(rows),
        "duplicate_game_ids": sum(1 for row in rows if row["duplicate_group_size"] > 1),
    }
