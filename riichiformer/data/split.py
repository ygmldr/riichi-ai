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
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from riichiformer.data import dedup

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


# ======================================================================================
# H-02: duplicate-component-aware split registry v2
# ======================================================================================
#
# PR-05's registry is built per filename/game_id from the audit inventory and must not
# be overwritten.  v2 is built from the H-02 canonical index and assigns the split at
# **duplicate-component** level: every file sharing a Tenhou id or a canonical SHA gets
# exactly one split; conflict/incomplete components are ``quarantine`` and are never
# training-eligible.  It is a new file set under the H-02 artifact directory.

SPLIT_V2_VERSION = "rf-split/2"
QUARANTINE_SPLIT = "quarantine"
V2_MANIFEST_COLUMNS = [
    "split_version", "protocol", "assignment_mode", "game_id", "component_id",
    "component_size", "canonical_sha256", "original_sha256", "source_id", "year",
    "month", "split", "relpath", "root", "blocked_source", "rights_status",
    "train_allowed", "previously_used_in_research", "undated", "holdout",
    "is_representative", "quarantine", "duplicate_group_size", "split_source",
    "representative_relpath", "reason",
]

_PROTECTED_DIRS = ("audit", "canonical", "split", "bench", "quality", "replay",
                   "drift", "git_safeguard", "audit_logs")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _unique_paths(paths) -> list[Path]:
    unique: list[Path] = []
    seen = set()
    for path in paths:
        resolved = Path(path).resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def _git_checkout_roots(start: Path) -> list[Path]:
    """Checkout roots visible from ``start`` (main checkout + every linked worktree)."""
    roots: list[Path] = []
    try:
        proc = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--git-common-dir", "--show-toplevel"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if proc.returncode == 0:
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if lines:
                common = Path(lines[0])
                common = common if common.is_absolute() else (start / common)
                common = common.resolve()
                roots.append(common.parent if common.name == ".git" else common)
            for line in lines[1:]:
                path = Path(line)
                roots.append((path if path.is_absolute() else start / path).resolve())
        proc = subprocess.run(
            ["git", "-C", str(start), "worktree", "list", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("worktree "):
                    roots.append(Path(line[len("worktree "):].strip()).resolve())
    except (OSError, subprocess.SubprocessError):
        pass
    return roots


def repository_roots(repo: Path, extra_roots=()) -> list[Path]:
    """Every checkout root whose ``artifacts/riichiformer/*`` must stay read-only."""
    package_root = Path(__file__).resolve().parents[2]
    candidates = [Path(repo), package_root, *extra_roots]
    candidates.extend(_git_checkout_roots(Path(repo)))
    return _unique_paths(candidates)


def _protected_artifact_name(path: Path) -> str | None:
    """Detect ``artifacts/riichiformer/<protected>`` in any checkout, without git."""
    parts = path.parts
    for index in range(len(parts) - 2):
        if (parts[index] == "artifacts" and parts[index + 1] == "riichiformer"
                and parts[index + 2] in _PROTECTED_DIRS):
            return parts[index + 2]
    return None


def assert_safe_v2_out(out_dir: Path, repo: Path, protected_roots=()) -> None:
    resolved = out_dir.resolve()
    # Repository-independent guard: an absolute --out under *any* checkout's
    # artifacts/riichiformer/{audit,canonical,split,...} is refused even when the
    # caller passes a worktree as --repo-root (main checkout not on that path).
    artifact_name = _protected_artifact_name(resolved)
    if artifact_name is not None:
        raise SplitError(
            "refusing to write under protected PR artifact directory "
            f"artifacts/riichiformer/{artifact_name}: {resolved}; "
            "write under artifacts/riichiformer/h02/ instead"
        )
    for root in repository_roots(repo, protected_roots):
        datasets = (root / "datasets").resolve()
        if resolved == datasets or _is_relative_to(resolved, datasets):
            raise SplitError(f"refusing to write inside read-only datasets/: {resolved}")
        for name in _PROTECTED_DIRS:
            protected = (root / "artifacts" / "riichiformer" / name).resolve()
            if resolved == protected or _is_relative_to(resolved, protected):
                raise SplitError(
                    f"refusing to overwrite existing PR artifact directory {protected}; "
                    "write under artifacts/riichiformer/h02/ instead"
                )


def v2_manifest_schema():
    import pyarrow as pa

    return pa.schema([
        ("split_version", pa.string()),
        ("protocol", pa.string()),
        ("assignment_mode", pa.string()),
        ("game_id", pa.string()),
        ("component_id", pa.string()),
        ("component_size", pa.int32()),
        ("canonical_sha256", pa.string()),
        ("original_sha256", pa.string()),
        ("source_id", pa.string()),
        ("year", pa.int16()),
        ("month", pa.int8()),
        ("split", pa.string()),
        ("relpath", pa.string()),
        ("root", pa.string()),
        ("blocked_source", pa.bool_()),
        ("rights_status", pa.string()),
        ("train_allowed", pa.bool_()),
        ("previously_used_in_research", pa.bool_()),
        ("undated", pa.bool_()),
        ("holdout", pa.bool_()),
        ("is_representative", pa.bool_()),
        ("quarantine", pa.bool_()),
        ("duplicate_group_size", pa.int32()),
        ("split_source", pa.string()),
        ("representative_relpath", pa.string()),
        ("reason", pa.string()),
    ])


def _component_assignments(store, cfg: SplitConfig) -> dict[str, dict]:
    """Split/quarantine decision for every component with more than one file."""
    assignments: dict[str, dict] = {}
    for component in store.iter_components():
        component_id = component["component_id"]
        representative = component.get("representative_relpath")
        if component["quarantine"]:
            assignments[component_id] = {
                "split": QUARANTINE_SPLIT,
                "reason": component.get("quarantine_reason") or "quarantined_component",
                "split_source": "quarantine",
                "representative_relpath": representative,
                "verified": False,
            }
            continue
        if not representative:
            raise SplitError(f"component {component_id} has no representative")
        _, verified = game_id_from_relpath(representative)
        if not verified:
            assignments[component_id] = {
                "split": "unknown_id", "reason": "unverified_game_id",
                "split_source": "component",
                "representative_relpath": representative, "verified": False,
            }
            continue
        split, reason = assign_split(
            cfg.protocol, component.get("year"), month_from_relpath(representative),
            cfg.mode, component.get("representative_game_id") or "", cfg.hash_seed,
        )
        assignments[component_id] = {
            "split": split, "reason": reason, "split_source": "component",
            "representative_relpath": representative, "verified": True,
        }
    return assignments


def _rights_for(source_id, year, rights: dict) -> tuple[str, str, bool]:
    source = source_id or source_for_year(year).value
    status = rights.get(source, RightsStatus.UNKNOWN.value)
    return source, status, status != RightsStatus.ALLOWED.value


def build_v2_row(row: dict, cfg: SplitConfig, rights: dict,
                 assignments: dict[str, dict]) -> dict:
    """Map one canonical-index row to a v2 manifest row (component-aware)."""
    relpath = row["relpath"]
    year = row.get("year")
    component_id = row["component_id"]
    is_representative = bool(row["is_representative"])
    quarantine = bool(row["quarantine"])
    assignment = assignments.get(component_id)
    if assignment is not None:
        split = assignment["split"]
        reason = assignment["reason"]
        split_source = assignment["split_source"]
        representative = assignment["representative_relpath"]
        verified = assignment["verified"]
    else:
        game_id, verified = game_id_from_relpath(relpath)
        if verified:
            split, reason = assign_split(
                cfg.protocol, year, month_from_relpath(relpath), cfg.mode,
                row.get("game_id") or game_id, cfg.hash_seed,
            )
        else:
            split, reason = "unknown_id", "unverified_game_id"
        split_source = "own"
        representative = relpath
    if quarantine and split != QUARANTINE_SPLIT:
        split = QUARANTINE_SPLIT
        reason = row.get("quarantine_reason") or "quarantined_record"
        split_source = "quarantine"
    if not verified and split != QUARANTINE_SPLIT:
        split, reason, split_source = "unknown_id", "unverified_game_id", split_source

    game_id = row.get("game_id")
    if not game_id:
        game_id, _ = game_id_from_relpath(relpath)
    source, status, blocked = _rights_for(row.get("source_id"), year, rights)
    train_allowed = (
        split == "train" and (not blocked) and is_representative and (not quarantine)
    )
    return {
        "split_version": SPLIT_V2_VERSION,
        "protocol": cfg.protocol,
        "assignment_mode": cfg.mode,
        "game_id": game_id,
        "component_id": component_id,
        "component_size": int(row.get("component_size") or 1),
        "canonical_sha256": row.get("canonical_sha256"),
        "original_sha256": row.get("original_sha256"),
        "source_id": source,
        "year": year,
        "month": month_from_relpath(relpath),
        "split": split,
        "relpath": relpath,
        "root": row.get("root"),
        "blocked_source": blocked,
        "rights_status": status,
        "train_allowed": train_allowed,
        "previously_used_in_research": _previously_used(year),
        "undated": split == "undated",
        "holdout": split in ("val", "val_holdout", "select"),
        "is_representative": is_representative,
        "quarantine": quarantine,
        "duplicate_group_size": int(row.get("component_size") or 1),
        "split_source": split_source,
        "representative_relpath": representative,
        "reason": reason,
    }


def _xor_game_digest(accumulator: dict, split: str, game_id: str) -> None:
    value = int.from_bytes(hashlib.sha256(game_id.encode("utf-8")).digest()[:16], "big")
    accumulator[split] = accumulator.get(split, 0) ^ value


def _digest_map(accumulator: dict) -> dict:
    return {
        split: {"count": None, "xor_sha256": hashlib.sha256(value.to_bytes(16, "big")).hexdigest()}
        for split, value in sorted(accumulator.items())
    }


def _v2_registry_hash(cfg: SplitConfig, counts: dict, representative_counts: dict,
                      digests: dict, canonical_meta: dict, rights: dict) -> str:
    payload = {
        "split_version": cfg.version,
        "protocol": cfg.protocol,
        "mode": cfg.mode,
        "hash_seed": cfg.hash_seed if cfg.mode == "hash_fallback" else None,
        "counts": counts,
        "representative_counts": representative_counts,
        "representative_digests": digests,
        "canonical": canonical_meta,
        "rights": rights,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_v2_manifest_stream(rows, path: Path, batch_size: int = 20000) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = v2_manifest_schema()
    path = Path(path)
    count = 0
    batch: list[dict] = []
    with dedup.atomic_path(path) as tmp:
        writer = pq.ParquetWriter(tmp, schema, compression="zstd")
        try:
            for row in rows:
                batch.append(row)
                count += 1
                if len(batch) >= batch_size:
                    table = pa.Table.from_pydict(
                        {name: [item.get(name) for item in batch] for name in schema.names},
                        schema=schema,
                    )
                    writer.write_table(table)
                    batch.clear()
            if batch:
                table = pa.Table.from_pydict(
                    {name: [item.get(name) for item in batch] for name in schema.names},
                    schema=schema,
                )
                writer.write_table(table)
        finally:
            writer.close()
    return count


def load_old_manifest_map(path: Path) -> dict[str, str]:
    """``relpath -> split`` from a PR-05 manifest (read-only)."""
    import pyarrow.parquet as pq

    mapping: dict[str, str] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=200000, columns=["relpath", "split"]):
        for relpath, split in zip(batch.column("relpath").to_pylist(),
                                  batch.column("split").to_pylist()):
            mapping[relpath] = split
    return mapping


def _new_diff_state(old_manifest: Path | None) -> dict | None:
    if old_manifest is None:
        return None
    if not Path(old_manifest).is_file():
        return {"status": "old_manifest_missing", "path": str(old_manifest)}
    return {
        "status": "compared",
        "old_manifest": str(old_manifest),
        "old_manifest_sha256": dedup.sha256_file(old_manifest),
        "old_map": load_old_manifest_map(Path(old_manifest)),
        "compared": 0,
        "same": 0,
        "moved": 0,
        "moved_by_pair": defaultdict(int),
        "moved_samples": [],
        "moved_xor": 0,
        "quarantined_rows": 0,
        "quarantined_by_old_split": defaultdict(int),
        "not_in_old": 0,
    }


def _diff_row(state: dict | None, row: dict) -> None:
    if not state or state.get("status") != "compared":
        return
    old_split = state["old_map"].get(row["relpath"])
    state["compared"] += 1
    if old_split is None:
        state["not_in_old"] += 1
        return
    new_split = row["split"]
    if old_split == new_split:
        state["same"] += 1
    else:
        state["moved"] += 1
        state["moved_by_pair"][f"{old_split}->{new_split}"] += 1
        state["moved_xor"] ^= int.from_bytes(
            hashlib.sha256(
                f"{row['relpath']}\x00{old_split}\x00{new_split}".encode("utf-8")
            ).digest()[:16], "big",
        )
        if len(state["moved_samples"]) < 500:
            state["moved_samples"].append({
                "relpath": row["relpath"], "old_split": old_split, "new_split": new_split,
                "component_id": row["component_id"],
                "component_size": row["component_size"],
                "quarantine_reason": row.get("quarantine"),
            })
    if new_split == QUARANTINE_SPLIT:
        state["quarantined_rows"] += 1
        state["quarantined_by_old_split"][old_split] += 1


def _diff_summary(state: dict | None) -> dict:
    if state is None:
        return {"status": "not_requested"}
    if state.get("status") != "compared":
        return {"status": state.get("status"), "path": state.get("path")}
    return {
        "status": "compared",
        "old_manifest": state["old_manifest"],
        "old_manifest_sha256": state["old_manifest_sha256"],
        "old_rows": len(state["old_map"]),
        "compared_rows": state["compared"],
        "same_rows": state["same"],
        "moved_rows": state["moved"],
        "moved_by_pair": dict(sorted(state["moved_by_pair"].items())),
        "moved_digest_xor_sha256": hashlib.sha256(
            state["moved_xor"].to_bytes(16, "big")
        ).hexdigest(),
        "moved_samples": state["moved_samples"],
        "not_in_old_rows": state["not_in_old"],
        "old_only_rows": len(state["old_map"]) - (state["compared"] - state["not_in_old"]),
        "quarantined_rows": state["quarantined_rows"],
        "quarantined_by_old_split": dict(sorted(state["quarantined_by_old_split"].items())),
    }


def build_registry_v2(store, out_dir, protocols=("L", "R"), canonical_meta: dict | None = None,
                      old_manifests: dict | None = None, old_summary: dict | None = None):
    """Build ``split_registry_v2.json`` + per-protocol manifests from a canonical store.

    ``store`` is a :class:`riichiformer.data.dedup.CanonicalStore` (duck-typed for
    tests).  Old PR-05 manifests are only read for the diff; nothing is overwritten.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if store.get_meta("finalized") != "1":
        raise SplitError("canonical store is not finalised; run rf_build_canonical first")
    config = store.get_meta_json("config") or {}
    if canonical_meta.get("audit_parquet_sha256") and config.get(
        "audit_parquet_sha256"
    ) not in (None, canonical_meta["audit_parquet_sha256"]):
        raise SplitError("canonical manifest and store audit hashes differ")
    if canonical_meta.get("index_rows") is not None:
        if store.count_rows() != int(canonical_meta["index_rows"]):
            raise SplitError(
                f"canonical manifest says {canonical_meta['index_rows']} rows but the "
                f"store has {store.count_rows()}"
            )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rights = _rights_status_map()
    canonical_meta = canonical_meta or {}
    old_manifests = old_manifests or {}
    old_summary_hashes = {}
    if old_summary:
        for protocol_summary in old_summary.get("protocols", []):
            old_summary_hashes[protocol_summary["protocol"]] = {
                "registry_hash": protocol_summary.get("registry_hash"),
                "manifest_sha256": protocol_summary.get("manifest_sha256"),
                "counts": protocol_summary.get("counts"),
            }

    summaries = []
    diffs = {}
    for protocol in protocols:
        cfg = SplitConfig(protocol=protocol, version=SPLIT_V2_VERSION)
        assignments = _component_assignments(store, cfg)
        manifest_path = out_dir / f"split_manifest_{protocol}_v2.parquet"
        schema = v2_manifest_schema()
        counts: dict[str, int] = defaultdict(int)
        representative_counts: dict[str, int] = defaultdict(int)
        year_split: dict[str, int] = defaultdict(int)
        digest_acc: dict[str, int] = defaultdict(int)
        blocked_rows = train_allowed_rows = representative_rows = quarantined_rows = 0
        component_rows = unknown_id_rows = 0
        diff_state = _new_diff_state(old_manifests.get(protocol))
        rows_written = 0

        with dedup.atomic_path(manifest_path) as tmp:
            writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            batch: list[dict] = []
            try:
                for index_row in store.iter_index_rows():
                    row = build_v2_row(index_row, cfg, rights, assignments)
                    _diff_row(diff_state, row)
                    counts[row["split"]] += 1
                    year_split[f"{row['year']}|{row['split']}"] += 1
                    quarantined_rows += 1 if row["quarantine"] else 0
                    component_rows += 1 if row["component_size"] > 1 else 0
                    unknown_id_rows += 1 if row["split"] == "unknown_id" else 0
                    blocked_rows += 1 if row["blocked_source"] else 0
                    train_allowed_rows += 1 if row["train_allowed"] else 0
                    if row["is_representative"]:
                        representative_rows += 1
                        representative_counts[row["split"]] += 1
                        _xor_game_digest(digest_acc, row["split"], row["game_id"] or "")
                    batch.append(row)
                    rows_written += 1
                    if len(batch) >= 20000:
                        writer.write_table(pa.Table.from_pydict(
                            {name: [item.get(name) for item in batch]
                             for name in schema.names}, schema=schema))
                        batch.clear()
                if batch:
                    writer.write_table(pa.Table.from_pydict(
                        {name: [item.get(name) for item in batch]
                         for name in schema.names}, schema=schema))
            finally:
                writer.close()

        digests = _digest_map(digest_acc)
        for split in digests:
            digests[split]["count"] = representative_counts[split]
        counts = dict(sorted(counts.items()))
        representative_counts = dict(sorted(representative_counts.items()))
        registry_hash = _v2_registry_hash(
            cfg, counts, representative_counts, digests, canonical_meta,
            {key: rights[key] for key in sorted(rights)},
        )
        summary = {
            "protocol": protocol,
            "mode": cfg.mode,
            "split_version": SPLIT_V2_VERSION,
            "manifest": manifest_path.name,
            "manifest_sha256": dedup.sha256_file(manifest_path),
            "rows": rows_written,
            "counts": counts,
            "representative_counts": representative_counts,
            "representatives": representative_rows,
            "quarantined_rows": quarantined_rows,
            "duplicate_component_rows": component_rows,
            "unknown_id_rows": unknown_id_rows,
            "blocked_source_rows": blocked_rows,
            "train_allowed_rows": train_allowed_rows,
            "split_digests": digests,
            "registry_hash": registry_hash,
            "year_split": dict(sorted(year_split.items())),
        }
        diff = _diff_summary(diff_state)
        if protocol in old_summary_hashes:
            diff["old_pr05"] = old_summary_hashes[protocol]
        diffs[protocol] = diff
        diff_state = None  # release the old-manifest map before the next protocol
        summaries.append(summary)

    registry = {
        "split_version": SPLIT_V2_VERSION,
        "canonical": canonical_meta,
        "rights_snapshot": {key: rights[key] for key in sorted(rights)},
        "protocols": summaries,
        "diff_vs_pr05": diffs,
        "rules": {
            "unit": "duplicate component (same game_id or same canonical sha256)",
            "representative": "min(original_sha256, relpath); quarantined => none",
            "conflict": "same id/incomplete component => quarantine, never training-eligible",
            "pov": "one row per game; POV seats are derived later and share this split",
            "old_registry": "PR-05 split_manifest_{L,R}.parquet is read-only input, never overwritten",
        },
    }
    registry_text = json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with dedup.atomic_path(out_dir / "split_registry_v2.json") as tmp:
        tmp.write_text(registry_text, encoding="utf-8")
    diff_text = json.dumps(diffs, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with dedup.atomic_path(out_dir / "split_diff_v2.json") as tmp:
        tmp.write_text(diff_text, encoding="utf-8")
    return registry


def assert_no_component_overlap(rows) -> None:
    """No duplicate component may appear in two splits."""
    seen: dict[str, str] = {}
    conflicts = []
    for row in rows:
        if int(row.get("duplicate_group_size") or 1) <= 1:
            continue
        component_id = row.get("component_id")
        previous = seen.setdefault(component_id, row["split"])
        if previous != row["split"]:
            conflicts.append((component_id, previous, row["split"], row.get("relpath")))
    if conflicts:
        sample = "; ".join(
            f"{cid}: {a} vs {b} ({relpath})" for cid, a, b, relpath in conflicts[:8]
        )
        raise SplitError(f"{len(conflicts)} component split conflict(s): {sample}")


def verify_v2_manifest(path: Path) -> dict:
    """Stream a v2 manifest: component disjointness + quarantine cannot leak."""
    import pyarrow.parquet as pq

    counts: dict[str, int] = defaultdict(int)
    duplicate_counts: dict[str, int] = defaultdict(int)
    component_splits: dict[str, str] = {}
    conflicts = []
    leaks = []
    rows = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100000):
        for row in batch.to_pylist():
            rows += 1
            counts[row["split"]] += 1
            if int(row.get("duplicate_group_size") or 1) > 1:
                duplicate_counts[row["split"]] += 1
                previous = component_splits.setdefault(row["component_id"], row["split"])
                if previous != row["split"]:
                    conflicts.append((row["component_id"], previous, row["split"]))
            if row["quarantine"]:
                if row["split"] != QUARANTINE_SPLIT:
                    leaks.append((row["relpath"], row["split"], "quarantine_row_in_split"))
                if row["train_allowed"]:
                    leaks.append((row["relpath"], row["split"], "quarantine_train_allowed"))
            if row["split"] == QUARANTINE_SPLIT and not row["quarantine"]:
                leaks.append((row["relpath"], row["split"], "quarantine_split_non_quarantine_row"))
            # Only the single component representative may be training-eligible, and a
            # rights-blocked source can never be train_allowed (fail closed).
            if row["train_allowed"]:
                if not row["is_representative"]:
                    leaks.append((row["relpath"], row["split"],
                                  "non_representative_train_allowed"))
                if row["blocked_source"]:
                    leaks.append((row["relpath"], row["split"],
                                  "blocked_source_train_allowed"))
    if conflicts:
        raise SplitError(f"{len(conflicts)} component(s) straddle splits: {conflicts[:5]}")
    if leaks:
        raise SplitError(f"{len(leaks)} quarantine leak(s): {leaks[:5]}")
    return {
        "path": str(path),
        "rows": rows,
        "counts": dict(sorted(counts.items())),
        "duplicate_component_rows": dict(sorted(duplicate_counts.items())),
        "components_checked": len(component_splits),
        "ok": True,
    }


def load_canonical_meta(manifest_path: Path, store) -> dict:
    """Read the H-02 canonical manifest and cross-check it against the store."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise SplitError(f"canonical manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = store.get_meta_json("config") or {}
    audit_sha = data.get("audit_parquet_sha256")
    if audit_sha and config.get("audit_parquet_sha256") not in (None, audit_sha):
        raise SplitError(
            "canonical manifest and canonical store were built from different audit "
            "inputs; refusing to bind a split registry to them"
        )
    artifacts = data.get("artifacts") or {}
    index_info = artifacts.get("canonical_log_index.parquet") or {}
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": dedup.sha256_file(manifest_path),
        "audit_parquet_sha256": audit_sha,
        "index_sha256": index_info.get("sha256"),
        "index_rows": data.get("rows"),
        "canonical_version": data.get("canonical_version"),
        "index_version": data.get("index_version"),
        "full_corpus": data.get("full_corpus"),
        "representatives": data.get("representatives"),
        "quarantine_rows": data.get("quarantine_rows"),
    }


def _v2_utc_now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv=None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="RiichiFormer H-02 duplicate-component split registry v2"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-v2", help="build split_registry_v2.json + L/R manifests")
    build.add_argument("--canonical-db", default="artifacts/riichiformer/h02/canonical/canonical_index.sqlite")
    build.add_argument("--canonical-manifest",
                       default="artifacts/riichiformer/h02/canonical/manifest.json")
    build.add_argument("--out", default="artifacts/riichiformer/h02/split")
    build.add_argument("--protocol", choices=["L", "R", "both"], default="both")
    build.add_argument("--old-manifest-L",
                       default="artifacts/riichiformer/split/split_manifest_L.parquet")
    build.add_argument("--old-manifest-R",
                       default="artifacts/riichiformer/split/split_manifest_R.parquet")
    build.add_argument("--old-summary",
                       default="artifacts/riichiformer/split/split_summary.json")
    build.add_argument("--no-diff", action="store_true")
    build.add_argument("--protected-root", action="append", default=[],
                       help="extra checkout root whose artifacts/riichiformer/* is read-only "
                            "(repeatable; git roots are discovered automatically)")
    build.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent.parent))

    verify = sub.add_parser("verify-v2", help="stream-check component disjointness/quarantine")
    verify.add_argument("--manifest", nargs="+", required=True)

    args = ap.parse_args(argv)
    if args.command == "verify-v2":
        try:
            for manifest in args.manifest:
                report = verify_v2_manifest(Path(manifest))
                print(f"OK {manifest} rows={report['rows']} counts={report['counts']} "
                      f"components_checked={report['components_checked']}")
        except SplitError as exc:
            print(f"split: {exc}", file=sys.stderr)
            return 1
        return 0

    repo = Path(args.repo_root).resolve()
    try:
        out_dir = Path(args.out)
        out_dir = out_dir if out_dir.is_absolute() else (repo / out_dir)
        assert_safe_v2_out(out_dir, repo, args.protected_root)
        db_path = Path(args.canonical_db)
        db_path = db_path if db_path.is_absolute() else (repo / db_path)
        manifest_path = Path(args.canonical_manifest)
        manifest_path = manifest_path if manifest_path.is_absolute() else (repo / manifest_path)
        store = dedup.CanonicalStore(db_path, create=False)
        try:
            canonical_meta = load_canonical_meta(manifest_path, store)
            protocols = ("L", "R") if args.protocol == "both" else (args.protocol,)
            old_manifests = {}
            if not args.no_diff:
                for protocol in protocols:
                    candidate = getattr(args, f"old_manifest_{protocol}")
                    old_manifests[protocol] = (
                        Path(candidate) if Path(candidate).is_absolute() else repo / candidate
                    )
            old_summary = None
            if args.old_summary and not args.no_diff:
                summary_path = Path(args.old_summary)
                summary_path = summary_path if summary_path.is_absolute() else repo / summary_path
                if summary_path.is_file():
                    old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            registry = build_registry_v2(
                store, out_dir, protocols=protocols, canonical_meta=canonical_meta,
                old_manifests=old_manifests, old_summary=old_summary,
            )
        finally:
            store.close()
        print(f"split v2: out={out_dir} protocols={','.join(protocols)}")
        for summary in registry["protocols"]:
            print(f"  {summary['protocol']}: rows={summary['rows']} "
                  f"counts={summary['counts']} reps={summary['representatives']} "
                  f"quarantine={summary['quarantined_rows']} "
                  f"registry_hash={summary['registry_hash']}")
            diff = registry["diff_vs_pr05"].get(summary["protocol"], {})
            if diff.get("status") == "compared":
                print(f"    diff vs PR-05: moved={diff['moved_rows']} "
                      f"quarantine={diff['quarantined_rows']} "
                      f"old_rows={diff['old_rows']}")
        with dedup.atomic_path(out_dir / "split_run_log.jsonl") as tmp:
            tmp.write_text(json.dumps({
                "ts": _v2_utc_now(),
                "canonical": canonical_meta,
                "registry_sha256": dedup.sha256_file(out_dir / "split_registry_v2.json"),
            }, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    except (SplitError, dedup.DedupError, FileNotFoundError) as exc:
        print(f"split: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
