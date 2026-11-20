"""Deterministic duplicate grouping for canonical MJAI games (PR-03).

Priority (per task card):

1. ``platform_game_key`` (Tenhou log id) when present;
2. else ``canonical_sha256``.

Rules:

* identical canonical events encoded as gzip vs plaintext share ``canonical_sha256`` and
  are therefore one group (original byte SHA stays different, preserved per provenance row);
* the same Tenhou id with a **different** canonical hash is a collision and is flagged for
  review -- never silently merged, never silently split;
* the same canonical content under two different Tenhou ids is flagged
  ``cross_key_duplicate`` (converter/re-export artefact) for a separate review;
* one representative per group is chosen deterministically (smallest original SHA, then
  smallest relpath), so the training corpus keeps exactly one copy of a game.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

INDEX_VERSION = "rf-log-index/1"
INDEX_COLUMNS = [
    "index_version", "game_id", "source_id", "year", "relpath", "compression",
    "bytes", "original_sha256", "canonical_sha256", "platform_game_key",
    "events", "kyoku_count", "duplicate_group", "duplicate_count",
    "is_representative", "collision_review", "cross_key_duplicate", "invalid_reason",
]


def _group_key(record: dict):
    platform = record.get("platform_game_key")
    if platform:
        return ("tenhou", platform)
    canonical = record.get("canonical_sha256")
    if canonical:
        return ("canon", canonical)
    return None


def _group_id(key) -> str:
    return hashlib.sha256(f"{key[0]}:{key[1]}".encode("utf-8")).hexdigest()[:16]


def assign_duplicate_groups(records: list[dict]) -> list[dict]:
    """Return new rows with duplicate/collision fields filled in (input not mutated)."""
    groups = defaultdict(list)
    invalid = []
    for record in records:
        key = _group_key(record)
        if key is None:
            invalid.append(record)
        else:
            groups[key].append(record)

    # Which canonical hashes appear under more than one Tenhou key?
    canon_to_platforms = defaultdict(set)
    for (kind, value), items in groups.items():
        if kind != "tenhou":
            continue
        for item in items:
            canon_to_platforms[item["canonical_sha256"]].add(value)

    output = []
    for key, items in groups.items():
        group = _group_id(key)
        canon_hashes = {item["canonical_sha256"] for item in items}
        collision = len(canon_hashes) > 1
        representative = min(
            items, key=lambda r: (r["original_sha256"] or "", r["relpath"] or "")
        )
        for item in items:
            row = dict(item)
            row["duplicate_group"] = group
            row["duplicate_count"] = len(items)
            row["is_representative"] = item is representative
            row["collision_review"] = collision
            row["cross_key_duplicate"] = (
                key[0] == "tenhou"
                and len(canon_to_platforms.get(item["canonical_sha256"], set())) > 1
            )
            output.append(row)

    for item in invalid:
        row = dict(item)
        row["duplicate_group"] = None
        row["duplicate_count"] = 1
        row["is_representative"] = False
        row["collision_review"] = False
        row["cross_key_duplicate"] = False
        output.append(row)

    output.sort(key=lambda r: (r.get("year") or 0, r.get("relpath") or ""))
    return output


def summarize_index(rows: list[dict]) -> dict:
    valid = [r for r in rows if not r.get("invalid_reason")]
    invalid = [r for r in rows if r.get("invalid_reason")]
    groups = defaultdict(list)
    for row in valid:
        if row.get("duplicate_group"):
            groups[row["duplicate_group"]].append(row)
    return {
        "index_version": INDEX_VERSION,
        "rows": len(rows),
        "valid": len(valid),
        "invalid": len(invalid),
        "invalid_reasons": _count(invalid, "invalid_reason"),
        "duplicate_groups": len(groups),
        "duplicate_rows": sum(len(items) for items in groups.values() if len(items) > 1),
        "collision_groups": sum(1 for items in groups.values() if items[0].get("collision_review")),
        "cross_key_duplicate_rows": sum(1 for row in rows if row.get("cross_key_duplicate")),
        "representatives": sum(1 for row in rows if row.get("is_representative")),
        "years": _count(valid, "year"),
        "sources": _count(valid, "source_id"),
    }


def _count(rows: list[dict], field: str) -> dict:
    counts = defaultdict(int)
    for row in rows:
        counts[str(row.get(field))] += 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def index_rows(records: list[dict]) -> list[dict]:
    """Normalise records to ``INDEX_COLUMNS`` order after duplicate grouping."""
    rows = assign_duplicate_groups(records)
    out = []
    for row in rows:
        out.append({
            "index_version": INDEX_VERSION,
            "game_id": row.get("game_id"),
            "source_id": row.get("source_id"),
            "year": row.get("year"),
            "relpath": row.get("relpath"),
            "compression": row.get("compression"),
            "bytes": row.get("bytes"),
            "original_sha256": row.get("original_sha256"),
            "canonical_sha256": row.get("canonical_sha256"),
            "platform_game_key": row.get("platform_game_key"),
            "events": row.get("events"),
            "kyoku_count": row.get("kyoku_count"),
            "duplicate_group": row.get("duplicate_group"),
            "duplicate_count": row.get("duplicate_count"),
            "is_representative": bool(row.get("is_representative")),
            "collision_review": bool(row.get("collision_review")),
            "cross_key_duplicate": bool(row.get("cross_key_duplicate")),
            "invalid_reason": row.get("invalid_reason"),
        })
    return out


def index_schema():
    import pyarrow as pa

    return pa.schema([
        ("index_version", pa.string()),
        ("game_id", pa.string()),
        ("source_id", pa.string()),
        ("year", pa.int16()),
        ("relpath", pa.string()),
        ("compression", pa.string()),
        ("bytes", pa.int64()),
        ("original_sha256", pa.string()),
        ("canonical_sha256", pa.string()),
        ("platform_game_key", pa.string()),
        ("events", pa.int32()),
        ("kyoku_count", pa.int16()),
        ("duplicate_group", pa.string()),
        ("duplicate_count", pa.int32()),
        ("is_representative", pa.bool_()),
        ("collision_review", pa.bool_()),
        ("cross_key_duplicate", pa.bool_()),
        ("invalid_reason", pa.string()),
    ])


def write_index_parquet(rows: list[dict], path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = index_schema()
    table = pa.Table.from_pydict(
        {name: [row.get(name) for row in rows] for name in schema.names}, schema=schema
    )
    pq.write_table(table, path, compression="zstd")
    return table.num_rows


def read_index_parquet(path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def stratified_sample(rows: list[dict], max_total: int = 1000, seed: int = 20261121,
                      strata_fields=("year", "source_id")) -> list[dict]:
    """Deterministic hash-stratified sample over ``rows`` (same seed => same sample)."""
    if max_total <= 0:
        return []
    buckets = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field)) for field in strata_fields)
        priority = int.from_bytes(
            hashlib.sha256(f"{seed}\x00{key}\x00{row.get('relpath')}".encode("utf-8")).digest()[:8],
            "big",
        )
        buckets[key].append((priority, row))
    strata = sorted(buckets)
    quota = max(1, max_total // max(1, len(strata)))
    selected = []
    for key in strata:
        chunks = sorted(buckets[key], key=lambda item: item[0])[:quota]
        selected.extend(row for _, row in chunks)
    if len(selected) < max_total:
        chosen = {id(row) for row in selected}
        rest = sorted(
            (row for bucket in strata for _, row in buckets[bucket] if id(row) not in chosen),
            key=lambda row: hashlib.sha256(
                f"{seed}\x00{row.get('relpath')}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(rest[: max_total - len(selected)])
    return selected[:max_total]
