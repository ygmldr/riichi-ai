"""Deterministic duplicate components for canonical MJAI games (PR-03 / H-02).

The unit of identity is a **component**, not a single filename:

* every row that shares a ``game_id`` (Tenhou log id) is an edge;
* every row that shares a ``canonical_sha256`` is an edge;
* connected components are computed over the union of both edge kinds.

Consequences:

* the same events stored as gzip vs plaintext, under a different filename, under a
  different source id and/or under a different Tenhou log id form **one** component
  (``cross_key_duplicate=True`` records the last case);
* the same ``game_id`` with two different canonical hashes is a **conflict**; the whole
  component is quarantined and no representative is chosen;
* a component containing an unparseable/incomplete member is quarantined as
  ``incomplete_component``;
* a component gets exactly one representative (smallest ``original_sha256`` then
  ``relpath``) unless it is quarantined;
* a component never straddles train/val/select -- ``split.py`` assigns the split at
  component level and quarantined components are excluded from every training split.

Two implementations are provided and kept semantically parallel:

``assign_duplicate_groups``
    in-memory, for samples and tests (also the reference for the SQLite path);
``CanonicalStore``
    stdlib ``sqlite3``-backed, shard/checkpoint/resume storage used at 2.51 M scale.
    Only duplicate rows are materialised in Python memory during finalisation; the
    full inventory never is.

The index writer keeps the PR-03 columns and adds the provenance/decision fields
(``canonical_version``, ``rules_version``, ``component_id``, ``component_size``,
``quarantine``, ``quarantine_reason``, ...).  Outputs are written atomically and, given
the same input inventory + config, byte-identically (see ``H02_pytest.log``).
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import sqlite3
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

from riichiformer.data import canonicalize as canon

INDEX_VERSION = "rf-log-index/2"
STORE_VERSION = "rf-canon-store/1"
COMPONENT_VERSION = "rf-component/1"

INDEX_COLUMNS = [
    "index_version", "canonical_version", "rules_version", "game_id", "id_key",
    "source_id", "year", "relpath", "root", "compression", "bytes",
    "original_sha256", "canonical_sha256", "platform_game_key", "events", "kyoku_count",
    "component_id", "component_size", "duplicate_count", "is_representative",
    "collision_review", "cross_key_duplicate", "quarantine", "quarantine_reason",
    "invalid_reason",
]

COMPONENT_COLUMNS = [
    "component_id", "component_size", "valid_rows", "distinct_game_ids",
    "distinct_canonical_shas", "cross_key_duplicate", "quarantine",
    "quarantine_reason", "representative_relpath", "representative_game_id",
    "canonical_sha256", "original_sha256", "year", "sources",
]

GAMES_TABLE_FIELDS = [
    "shard_id", "canonical_version", "rules_version", "game_id", "id_key",
    "platform_game_key", "source_id", "year", "relpath", "root", "compression",
    "bytes", "original_sha256", "canonical_sha256", "events", "kyoku_count",
    "invalid_reason", "component_id", "component_size", "is_representative",
    "collision_review", "cross_key_duplicate", "quarantine", "quarantine_reason",
]

# Column order used by every INSERT into ``games`` (the two trailing columns use SQL
# defaults; they are listed here for documentation/readability).
GAMES_INSERT_SQL = (
    "INSERT INTO games (shard_id, canonical_version, rules_version, game_id, id_key,"
    " platform_game_key, source_id, year, relpath, root, compression, bytes,"
    " original_sha256, canonical_sha256, events, kyoku_count, invalid_reason,"
    " component_id, component_size, is_representative, collision_review,"
    " cross_key_duplicate, quarantine, quarantine_reason)"
    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)

SCHEMA_SQL = """
PRAGMA foreign_keys = OFF;
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shards (
    shard_id TEXT PRIMARY KEY,
    year INTEGER,
    rows INTEGER NOT NULL,
    valid INTEGER NOT NULL,
    invalid INTEGER NOT NULL,
    status TEXT NOT NULL,
    finished_utc TEXT NOT NULL,
    seconds REAL
);
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY,
    shard_id TEXT NOT NULL,
    canonical_version TEXT NOT NULL,
    rules_version TEXT NOT NULL,
    game_id TEXT,
    id_key TEXT,
    platform_game_key TEXT,
    source_id TEXT,
    year INTEGER NOT NULL,
    relpath TEXT NOT NULL UNIQUE,
    root TEXT,
    compression TEXT,
    bytes INTEGER,
    original_sha256 TEXT,
    canonical_sha256 TEXT,
    events INTEGER,
    kyoku_count INTEGER,
    invalid_reason TEXT,
    component_id TEXT,
    component_size INTEGER NOT NULL DEFAULT 1,
    is_representative INTEGER NOT NULL DEFAULT 0,
    collision_review INTEGER NOT NULL DEFAULT 0,
    cross_key_duplicate INTEGER NOT NULL DEFAULT 0,
    quarantine INTEGER NOT NULL DEFAULT 0,
    quarantine_reason TEXT
);
CREATE INDEX IF NOT EXISTS games_id_key ON games(id_key);
CREATE INDEX IF NOT EXISTS games_canonical_sha ON games(canonical_sha256);
CREATE INDEX IF NOT EXISTS games_year_relpath ON games(year, relpath);
CREATE TABLE IF NOT EXISTS components (
    component_id TEXT PRIMARY KEY,
    component_size INTEGER NOT NULL,
    valid_rows INTEGER NOT NULL,
    distinct_game_ids INTEGER NOT NULL,
    distinct_canonical_shas INTEGER NOT NULL,
    cross_key_duplicate INTEGER NOT NULL,
    quarantine INTEGER NOT NULL,
    quarantine_reason TEXT,
    representative_relpath TEXT,
    representative_game_id TEXT,
    canonical_sha256 TEXT,
    original_sha256 TEXT,
    year INTEGER,
    sources TEXT
);
"""


class DedupError(RuntimeError):
    """Inconsistent store / configuration; always fail closed."""


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_id_key(record: dict) -> str | None:
    """The game-identity key used for duplicate edges.

    Valid records carry ``game_id`` (``tenhou:<log id>`` or ``sha256:<canonical>``);
    invalid records have ``game_id=None`` but still carry the Tenhou filename key, so a
    broken copy of a game is grouped with its valid copy and quarantines the component.
    """
    key = record.get("id_key")
    if key:
        return key
    game_id = record.get("game_id")
    if game_id:
        return game_id
    platform = record.get("platform_game_key")
    return f"tenhou:{platform}" if platform else None


def component_id_for_relpaths(relpaths) -> str:
    """Deterministic component id for a real (size > 1) component."""
    joined = "\n".join(sorted(relpaths))
    return "c" + _sha256_text(f"{COMPONENT_VERSION}\n{joined}")[:20]


def singleton_component_id(relpath: str) -> str:
    return "s" + _sha256_text(f"{COMPONENT_VERSION}\nsingle\n{relpath}")[:20]


class _UnionFind:
    __slots__ = ("parent",)

    def __init__(self):
        self.parent: dict[int, int] = {}

    def add(self, item: int) -> None:
        if item not in self.parent:
            self.parent[item] = item

    def find(self, item: int) -> int:
        parent = self.parent
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        self.add(left)
        self.add(right)
        a, b = self.find(left), self.find(right)
        if a != b:
            # Smaller index wins: deterministic roots independent of union order.
            if a < b:
                self.parent[b] = a
            else:
                self.parent[a] = b


def _quarantine_reason(conflict: bool, incomplete: bool, invalid_reason: str | None) -> str | None:
    reasons = []
    if conflict:
        reasons.append("conflict_same_id_different_canonical_sha")
    if incomplete:
        reasons.append("incomplete_component")
    if reasons:
        return ";".join(reasons)
    return invalid_reason or None


def _component_decision(members: list[dict]) -> dict:
    """Decide component identity/quarantine/representative from member records."""
    valid = [m for m in members if not m.get("invalid_reason")]
    canons = sorted({m.get("canonical_sha256") for m in valid if m.get("canonical_sha256")})
    id_keys = sorted({record_id_key(m) for m in valid if record_id_key(m)})
    conflict = len(canons) > 1
    incomplete = len(valid) != len(members)
    quarantine = conflict or incomplete
    reason = _quarantine_reason(conflict, incomplete, None)
    representative = None
    if not quarantine and valid:
        representative = min(
            valid, key=lambda m: (m.get("original_sha256") or "", m.get("relpath") or "")
        )
    if len(members) == 1:
        component_id = singleton_component_id(members[0]["relpath"])
    else:
        component_id = component_id_for_relpaths([m["relpath"] for m in members])
    return {
        "component_id": component_id,
        "component_size": len(members),
        "valid_rows": len(valid),
        "distinct_game_ids": len(id_keys),
        "distinct_canonical_shas": len(canons),
        "cross_key_duplicate": len(id_keys) > 1,
        "conflict": conflict,
        "incomplete": incomplete,
        "quarantine": quarantine,
        "quarantine_reason": reason,
        "representative": representative,
        "canonical_sha256": representative.get("canonical_sha256") if representative else None,
        "original_sha256": representative.get("original_sha256") if representative else None,
        "year": representative.get("year") if representative else None,
        "sources": sorted({m.get("source_id") for m in valid if m.get("source_id")}),
    }


# --------------------------------------------------------------------------------------
# In-memory implementation (samples / tests / reference semantics)
# --------------------------------------------------------------------------------------


def assign_duplicate_groups(records: list[dict]) -> list[dict]:
    """Return new rows with component/quarantine fields filled in (input not mutated)."""
    if not records:
        return []

    union = _UnionFind()
    first_seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        union.add(index)
        keys = []
        id_key = record_id_key(record)
        if id_key:
            keys.append(("id", id_key))
        canonical_sha = record.get("canonical_sha256")
        if canonical_sha:
            keys.append(("canon", canonical_sha))
        for key in keys:
            previous = first_seen.get(key)
            if previous is None:
                first_seen[key] = index
            else:
                union.union(previous, index)

    by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        by_root[union.find(index)].append(index)

    output: list[dict] = []
    for indices in by_root.values():
        members = [records[i] for i in indices]
        decision = _component_decision(members)
        representative = decision["representative"]
        for record in members:
            row = dict(record)
            row["component_id"] = decision["component_id"]
            row["component_size"] = decision["component_size"]
            row["duplicate_group"] = (
                decision["component_id"] if decision["component_size"] > 1 else None
            )
            row["duplicate_count"] = decision["component_size"]
            row["is_representative"] = record is representative
            row["collision_review"] = decision["conflict"]
            row["cross_key_duplicate"] = decision["cross_key_duplicate"]
            quarantine = decision["quarantine"] or bool(record.get("invalid_reason"))
            row["quarantine"] = quarantine
            if quarantine:
                row["quarantine_reason"] = (
                    decision["quarantine_reason"]
                    or record.get("invalid_reason")
                    or "invalid_record"
                )
            else:
                row["quarantine_reason"] = None
            output.append(row)

    output.sort(key=lambda r: (r.get("year") or 0, r.get("relpath") or ""))
    return output


def index_rows(records: list[dict]) -> list[dict]:
    """Normalise records to ``INDEX_COLUMNS`` order after component grouping."""
    rows = assign_duplicate_groups(records)
    out = []
    for row in rows:
        out.append({
            "index_version": INDEX_VERSION,
            "canonical_version": row.get("canonical_version") or canon.CANONICAL_VERSION,
            "rules_version": row.get("rules_version") or canon.RULES_VERSION,
            "game_id": row.get("game_id"),
            "id_key": record_id_key(row),
            "source_id": row.get("source_id"),
            "year": row.get("year"),
            "relpath": row.get("relpath"),
            "root": row.get("root"),
            "compression": row.get("compression"),
            "bytes": row.get("bytes"),
            "original_sha256": row.get("original_sha256"),
            "canonical_sha256": row.get("canonical_sha256"),
            "platform_game_key": row.get("platform_game_key"),
            "events": row.get("events"),
            "kyoku_count": row.get("kyoku_count"),
            "component_id": row.get("component_id"),
            "component_size": row.get("component_size"),
            "duplicate_count": row.get("duplicate_count"),
            "is_representative": bool(row.get("is_representative")),
            "collision_review": bool(row.get("collision_review")),
            "cross_key_duplicate": bool(row.get("cross_key_duplicate")),
            "quarantine": bool(row.get("quarantine")),
            "quarantine_reason": row.get("quarantine_reason"),
            "invalid_reason": row.get("invalid_reason"),
        })
    return out


def summarize_index(rows: list[dict]) -> dict:
    valid = [r for r in rows if not r.get("invalid_reason")]
    invalid = [r for r in rows if r.get("invalid_reason")]
    duplicate_components = {
        r["component_id"] for r in valid
        if r.get("component_id") and (r.get("component_size") or 1) > 1
    }
    return {
        "index_version": INDEX_VERSION,
        "rows": len(rows),
        "valid": len(valid),
        "invalid": len(invalid),
        "invalid_reasons": _count(invalid, "invalid_reason"),
        "components": len({r["component_id"] for r in rows if r.get("component_id")}),
        "duplicate_groups": len(duplicate_components),
        "duplicate_components": len(duplicate_components),
        "duplicate_rows": sum(
            1 for r in rows if r.get("component_id") and (r.get("component_size") or 1) > 1
        ),
        "conflict_components": len({
            r["component_id"] for r in rows if r.get("collision_review")
        }),
        "incomplete_components": len({
            r["component_id"] for r in rows if r.get("quarantine_reason") == "incomplete_component"
        }),
        "cross_key_duplicate_rows": sum(1 for row in rows if row.get("cross_key_duplicate")),
        "representatives": sum(1 for row in rows if row.get("is_representative")),
        "quarantine_rows": sum(1 for row in rows if row.get("quarantine")),
        "years": _count(valid, "year"),
        "sources": _count(valid, "source_id"),
    }


def _count(rows: list[dict], field: str) -> dict:
    counts = defaultdict(int)
    for row in rows:
        counts[str(row.get(field))] += 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


# --------------------------------------------------------------------------------------
# SQLite store (sharded, checkpointed, resumable)
# --------------------------------------------------------------------------------------


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@contextmanager
def atomic_path(path: Path):
    """Write ``path.tmp`` then atomically replace ``path`` (never a half-written file)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():  # leftover from a hard kill; never trust a stale partial artifact
        tmp.unlink()
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


class CanonicalStore:
    """Durable canonical record store keyed by ``relpath``.

    One SQLite file is the checkpoint: a shard is a single transaction plus a ``shards``
    marker row, so an interrupted run loses at most the in-flight shard and a resume
    re-processes exactly that shard.  ``finalize_components`` is also one transaction.
    """

    def __init__(self, path: Path, create: bool = True):
        self.path = Path(path)
        if not self.path.exists() and not create:
            raise FileNotFoundError(f"canonical store not found: {self.path}")
        # SQLite's default temp directory is not writable in every sandbox; a
        # GROUP BY that needs a temp B-tree then fails with "unable to open
        # database file" only at finalize time.  Pin SQLITE_TMPDIR next to the
        # store so full-corpus summaries are reproducible.
        sqlite_tmp = self.path.parent / "sqlite_tmp"
        try:
            sqlite_tmp.mkdir(exist_ok=True)
        except OSError:
            pass
        else:
            os.environ.setdefault("SQLITE_TMPDIR", str(sqlite_tmp))
        self.con = sqlite3.connect(str(self.path), timeout=120, isolation_level=None)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=FULL")
        self.con.execute("PRAGMA temp_store=FILE")
        self.con.execute("PRAGMA cache_size=-65536")
        if create:
            self.con.executescript(SCHEMA_SQL)
            existing_version = self.get_meta("store_version")
            if existing_version is not None and existing_version != STORE_VERSION:
                raise DedupError(
                    f"canonical store {self.path} has version {existing_version!r}, "
                    f"expected {STORE_VERSION!r}; rebuild with --restart"
                )
            self.set_meta({"store_version": STORE_VERSION})

    # -- metadata -----------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def get_meta_json(self, key: str, default=None):
        raw = self.get_meta(key)
        return default if raw is None else json.loads(raw)

    def set_meta(self, values: dict) -> None:
        with self._transaction():
            for key, value in values.items():
                if not isinstance(value, str):
                    value = json.dumps(value, sort_keys=True) if not isinstance(value, (int, float)) else str(value)
                self.con.execute(
                    "INSERT INTO meta(key, value) VALUES (?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, value),
                )

    @contextmanager
    def _transaction(self):
        self.con.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.con.execute("ROLLBACK")
            raise
        else:
            self.con.execute("COMMIT")

    # -- shard ingestion ----------------------------------------------------------
    def shard_status(self) -> dict[str, dict]:
        out = {}
        for shard_id, rows, valid, invalid, status in self.con.execute(
            "SELECT shard_id, rows, valid, invalid, status FROM shards"
        ):
            out[shard_id] = {
                "rows": rows, "valid": valid, "invalid": invalid, "status": status,
            }
        return out

    def add_shard(self, shard_id: str, year: int, records: list[dict], seconds: float) -> dict:
        """Insert all records of one shard atomically, then mark the shard done."""
        if not records:
            raise DedupError(f"shard {shard_id} has no records")
        valid = sum(1 for record in records if not record.get("invalid_reason"))
        invalid = len(records) - valid
        tuples = [self._record_tuple(shard_id, record) for record in records]
        with self._transaction():
            try:
                self.con.executemany(GAMES_INSERT_SQL, tuples)
            except sqlite3.IntegrityError as exc:  # duplicate relpath -> bad resume/input
                raise DedupError(f"duplicate relpath while inserting shard {shard_id}: {exc}") from exc
            self.con.execute(
                "INSERT INTO shards(shard_id, year, rows, valid, invalid, status, "
                "finished_utc, seconds) VALUES (?,?,?,?,?,?,?,?)",
                (shard_id, year, len(records), valid, invalid, "done", _utc_now(), seconds),
            )
        # Keep the WAL bounded and checkpoint durable after every shard.
        self.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return {"shard_id": shard_id, "year": year, "rows": len(records),
                "valid": valid, "invalid": invalid, "seconds": seconds}

    @staticmethod
    def _record_tuple(shard_id: str, record: dict):
        invalid_reason = record.get("invalid_reason")
        is_valid = not invalid_reason
        return (
            shard_id,
            record.get("canonical_version") or canon.CANONICAL_VERSION,
            record.get("rules_version") or canon.RULES_VERSION,
            record.get("game_id"),
            record_id_key(record),
            record.get("platform_game_key"),
            record.get("source_id"),
            int(record.get("year") or 0),
            record["relpath"],
            record.get("root"),
            record.get("compression"),
            record.get("bytes"),
            record.get("original_sha256"),
            record.get("canonical_sha256"),
            record.get("events"),
            record.get("kyoku_count"),
            invalid_reason,
            None,                              # component_id
            1,                                 # component_size
            1 if is_valid else 0,              # is_representative (singleton default)
            0,                                 # collision_review
            0,                                 # cross_key_duplicate
            0 if is_valid else 1,              # quarantine
            None if is_valid else (invalid_reason or "invalid_record"),
        )

    # -- component finalisation ---------------------------------------------------
    def finalize_components(self) -> dict:
        """Compute connected components once, atomically; resumable/idempotent."""
        cached = self.get_meta_json("finalize_stats")
        if self.get_meta("finalized") == "1" and cached is not None:
            return cached

        stats = {}
        with self._transaction():
            self.con.execute("DELETE FROM components")
            self.con.execute(
                "UPDATE games SET component_id=NULL, component_size=1,"
                " is_representative=CASE WHEN invalid_reason IS NULL THEN 1 ELSE 0 END,"
                " collision_review=0, cross_key_duplicate=0,"
                " quarantine=CASE WHEN invalid_reason IS NULL THEN 0 ELSE 1 END,"
                " quarantine_reason=invalid_reason"
            )
            self.con.execute("DROP TABLE IF EXISTS temp.dup_id_keys")
            self.con.execute("DROP TABLE IF EXISTS temp.dup_canon_keys")
            self.con.execute("DROP TABLE IF EXISTS temp.dup_ids")
            self.con.execute(
                "CREATE TEMP TABLE dup_id_keys(key TEXT PRIMARY KEY)"
            )
            self.con.execute(
                "INSERT INTO dup_id_keys SELECT id_key FROM games"
                " WHERE id_key IS NOT NULL GROUP BY id_key HAVING COUNT(*) > 1"
            )
            self.con.execute(
                "CREATE TEMP TABLE dup_canon_keys(key TEXT PRIMARY KEY)"
            )
            self.con.execute(
                "INSERT INTO dup_canon_keys SELECT canonical_sha256 FROM games"
                " WHERE canonical_sha256 IS NOT NULL"
                " GROUP BY canonical_sha256 HAVING COUNT(*) > 1"
            )

            union = _UnionFind()
            involved: set[int] = set()

            def consume(sql: str) -> None:
                previous_key = object()
                previous_id = None
                for key, row_id in self.con.execute(sql):
                    if key != previous_key:
                        previous_key, previous_id = key, None
                    union.add(row_id)
                    involved.add(row_id)
                    if previous_id is not None:
                        union.union(previous_id, row_id)
                    previous_id = row_id

            consume(
                "SELECT g.id_key, g.id FROM games g JOIN dup_id_keys d"
                " ON g.id_key=d.key ORDER BY g.id_key, g.id"
            )
            consume(
                "SELECT g.canonical_sha256, g.id FROM games g JOIN dup_canon_keys d"
                " ON g.canonical_sha256=d.key ORDER BY g.canonical_sha256, g.id"
            )

            groups: dict[int, list[dict]] = defaultdict(list)
            if involved:
                self.con.execute("CREATE TEMP TABLE dup_ids(id INTEGER PRIMARY KEY)")
                self.con.executemany(
                    "INSERT INTO dup_ids(id) VALUES (?)", ((row_id,) for row_id in involved)
                )
                for row in self.con.execute(
                    "SELECT g.id, g.relpath, g.game_id, g.id_key, g.canonical_sha256,"
                    " g.invalid_reason, g.original_sha256, g.source_id, g.year"
                    " FROM games g JOIN dup_ids d ON g.id=d.id"
                ):
                    member = {
                        "id": row[0], "relpath": row[1], "game_id": row[2],
                        "id_key": row[3], "canonical_sha256": row[4],
                        "invalid_reason": row[5], "original_sha256": row[6],
                        "source_id": row[7], "year": row[8],
                    }
                    groups[union.find(row[0])].append(member)

            updates = []
            component_rows = []
            conflicts = incomplete = cross_key = 0
            for members in groups.values():
                decision = _component_decision(members)
                conflicts += 1 if decision["conflict"] else 0
                incomplete += 1 if decision["incomplete"] else 0
                cross_key += 1 if decision["cross_key_duplicate"] else 0
                representative = decision["representative"]
                component_rows.append((
                    decision["component_id"], decision["component_size"],
                    decision["valid_rows"], decision["distinct_game_ids"],
                    decision["distinct_canonical_shas"],
                    1 if decision["cross_key_duplicate"] else 0,
                    1 if decision["quarantine"] else 0,
                    decision["quarantine_reason"],
                    representative["relpath"] if representative else None,
                    representative["game_id"] if representative else None,
                    decision["canonical_sha256"], decision["original_sha256"],
                    decision["year"], json.dumps(decision["sources"]),
                ))
                for member in members:
                    updates.append((
                        decision["component_id"], decision["component_size"],
                        1 if member is representative else 0,
                        1 if decision["conflict"] else 0,
                        1 if decision["cross_key_duplicate"] else 0,
                        1 if decision["quarantine"] else 0,
                        decision["quarantine_reason"],
                        member["id"],
                    ))
            if component_rows:
                self.con.executemany(
                    "INSERT INTO components (" + ",".join(COMPONENT_COLUMNS) + ")"
                    " VALUES (" + ",".join("?" * len(COMPONENT_COLUMNS)) + ")",
                    component_rows,
                )
            if updates:
                self.con.executemany(
                    "UPDATE games SET component_id=?, component_size=?,"
                    " is_representative=?, collision_review=?, cross_key_duplicate=?,"
                    " quarantine=?, quarantine_reason=? WHERE id=?",
                    updates,
                )
            stats = {
                "component_count": len(groups),
                "duplicate_rows": sum(len(members) for members in groups.values()),
                "conflict_components": conflicts,
                "incomplete_components": incomplete,
                "cross_key_components": cross_key,
            }
            self.con.execute(
                "INSERT INTO meta(key, value) VALUES ('finalized','1')"
                " ON CONFLICT(key) DO UPDATE SET value='1'"
            )
            self.con.execute(
                "INSERT INTO meta(key, value) VALUES ('finalize_stats',?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (json.dumps(stats, sort_keys=True),),
            )
        self.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return stats

    # -- iteration ----------------------------------------------------------------
    def iter_games(self, batch_size: int = 5000):
        cursor = self.con.execute("SELECT * FROM games ORDER BY year, relpath")
        names = [description[0] for description in cursor.description]
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                yield dict(zip(names, row))

    def iter_index_rows(self, batch_size: int = 5000):
        for row in self.iter_games(batch_size):
            yield index_row_from_store(row)

    def iter_components(self, batch_size: int = 5000):
        cursor = self.con.execute(
            f"SELECT {','.join(COMPONENT_COLUMNS)} FROM components ORDER BY component_id"
        )
        names = [description[0] for description in cursor.description]
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                item = dict(zip(names, row))
                item["quarantine"] = bool(item["quarantine"])
                item["cross_key_duplicate"] = bool(item["cross_key_duplicate"])
                item["sources"] = json.loads(item["sources"] or "[]")
                yield item

    def count_rows(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM games").fetchone()[0]

    def summary(self) -> dict:
        con = self.con
        rows = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        valid = con.execute("SELECT COUNT(*) FROM games WHERE invalid_reason IS NULL").fetchone()[0]
        invalid = rows - valid
        invalid_reasons = {
            str(reason): count for reason, count in con.execute(
                "SELECT invalid_reason, COUNT(*) FROM games"
                " WHERE invalid_reason IS NOT NULL GROUP BY invalid_reason"
                " ORDER BY invalid_reason"
            )
        }
        component_count = con.execute("SELECT COUNT(*) FROM components").fetchone()[0]
        singleton_count = con.execute(
            "SELECT COUNT(*) FROM games WHERE component_id IS NULL"
        ).fetchone()[0]
        duplicate_rows = con.execute(
            "SELECT COUNT(*) FROM games WHERE component_id IS NOT NULL"
        ).fetchone()[0]
        conflict_components = con.execute(
            "SELECT COUNT(*) FROM components"
            " WHERE quarantine_reason LIKE '%conflict_same_id_different_canonical_sha%'"
        ).fetchone()[0]
        quarantine_rows = con.execute(
            "SELECT COUNT(*) FROM games WHERE quarantine=1"
        ).fetchone()[0]
        quarantine_components = con.execute(
            "SELECT COUNT(*) FROM components WHERE quarantine=1"
        ).fetchone()[0]
        representatives = con.execute(
            "SELECT COUNT(*) FROM games WHERE is_representative=1"
        ).fetchone()[0]
        cross_key_rows = con.execute(
            "SELECT COUNT(*) FROM games WHERE cross_key_duplicate=1"
        ).fetchone()[0]
        years = {
            str(year): count for year, count in con.execute(
                "SELECT year, COUNT(*) FROM games GROUP BY year ORDER BY year"
            )
        }
        sources = {
            str(source): count for source, count in con.execute(
                "SELECT source_id, COUNT(*) FROM games GROUP BY source_id ORDER BY source_id"
            )
        }
        incomplete_components = con.execute(
            "SELECT COUNT(*) FROM components WHERE quarantine_reason LIKE '%incomplete_component%'"
        ).fetchone()[0]
        return {
            "index_version": INDEX_VERSION,
            "rows": rows,
            "valid": valid,
            "invalid": invalid,
            "invalid_reasons": invalid_reasons,
            "components": component_count + singleton_count,
            "duplicate_groups": con.execute(
                "SELECT COUNT(*) FROM components WHERE component_size > 1"
            ).fetchone()[0],
            "duplicate_components": con.execute(
                "SELECT COUNT(*) FROM components WHERE component_size > 1"
            ).fetchone()[0],
            "duplicate_rows": duplicate_rows,
            "conflict_components": conflict_components,
            "incomplete_components": incomplete_components,
            "cross_key_duplicate_rows": cross_key_rows,
            "representatives": representatives,
            "quarantine_rows": quarantine_rows,
            "quarantine_components": quarantine_components,
            "years": years,
            "sources": sources,
        }

    def close(self) -> None:
        try:
            self.con.close()
        except sqlite3.Error:
            pass


def index_row_from_store(row: dict) -> dict:
    """Map a ``games`` row to the public index schema (singleton component ids)."""
    component_id = row.get("component_id") or singleton_component_id(row["relpath"])
    component_size = row.get("component_size") or 1
    quarantine = bool(row.get("quarantine"))
    return {
        "index_version": INDEX_VERSION,
        "canonical_version": row["canonical_version"],
        "rules_version": row["rules_version"],
        "game_id": row.get("game_id"),
        "id_key": row.get("id_key"),
        "source_id": row.get("source_id"),
        "year": row.get("year"),
        "relpath": row["relpath"],
        "root": row.get("root"),
        "compression": row.get("compression"),
        "bytes": row.get("bytes"),
        "original_sha256": row.get("original_sha256"),
        "canonical_sha256": row.get("canonical_sha256"),
        "platform_game_key": row.get("platform_game_key"),
        "events": row.get("events"),
        "kyoku_count": row.get("kyoku_count"),
        "component_id": component_id,
        "component_size": component_size,
        "duplicate_count": component_size,
        "is_representative": bool(row.get("is_representative")),
        "collision_review": bool(row.get("collision_review")),
        "cross_key_duplicate": bool(row.get("cross_key_duplicate")),
        "quarantine": quarantine,
        "quarantine_reason": row.get("quarantine_reason"),
        "invalid_reason": row.get("invalid_reason"),
    }


# --------------------------------------------------------------------------------------
# Schemas / writers (streaming, atomic, byte-stable)
# --------------------------------------------------------------------------------------


def index_schema():
    import pyarrow as pa

    return pa.schema([
        ("index_version", pa.string()),
        ("canonical_version", pa.string()),
        ("rules_version", pa.string()),
        ("game_id", pa.string()),
        ("id_key", pa.string()),
        ("source_id", pa.string()),
        ("year", pa.int16()),
        ("relpath", pa.string()),
        ("root", pa.string()),
        ("compression", pa.string()),
        ("bytes", pa.int64()),
        ("original_sha256", pa.string()),
        ("canonical_sha256", pa.string()),
        ("platform_game_key", pa.string()),
        ("events", pa.int32()),
        ("kyoku_count", pa.int16()),
        ("component_id", pa.string()),
        ("component_size", pa.int32()),
        ("duplicate_count", pa.int32()),
        ("is_representative", pa.bool_()),
        ("collision_review", pa.bool_()),
        ("cross_key_duplicate", pa.bool_()),
        ("quarantine", pa.bool_()),
        ("quarantine_reason", pa.string()),
        ("invalid_reason", pa.string()),
    ])


def component_schema():
    import pyarrow as pa

    return pa.schema([
        ("component_id", pa.string()),
        ("component_size", pa.int32()),
        ("valid_rows", pa.int32()),
        ("distinct_game_ids", pa.int32()),
        ("distinct_canonical_shas", pa.int32()),
        ("cross_key_duplicate", pa.bool_()),
        ("quarantine", pa.bool_()),
        ("quarantine_reason", pa.string()),
        ("representative_relpath", pa.string()),
        ("representative_game_id", pa.string()),
        ("canonical_sha256", pa.string()),
        ("original_sha256", pa.string()),
        ("year", pa.int16()),
        ("sources", pa.list_(pa.string())),
    ])


def _rows_to_table(rows: list[dict], schema):
    import pyarrow as pa

    return pa.Table.from_pydict(
        {name: [row.get(name) for row in rows] for name in schema.names}, schema=schema
    )


def write_index_parquet(rows: list[dict], path) -> int:
    """In-memory writer (samples/tests): one table, atomic replace."""
    import pyarrow.parquet as pq

    path = Path(path)
    schema = index_schema()
    with atomic_path(path) as tmp:
        pq.write_table(_rows_to_table(rows, schema), tmp, compression="zstd")
    return len(rows)


def read_index_parquet(path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def write_store_index_parquet(store: CanonicalStore, path, batch_size: int = 20000) -> dict:
    """Stream the full index from SQLite to Parquet; returns row count + digests."""
    import pyarrow.parquet as pq

    schema = index_schema()
    path = Path(path)
    digest = hashlib.sha256()
    batch: list[dict] = []
    rows = 0
    with atomic_path(path) as tmp:
        writer = pq.ParquetWriter(tmp, schema, compression="zstd")
        try:
            for row in store.iter_index_rows(batch_size=batch_size):
                digest.update(_logical_index_line(row))
                batch.append(row)
                rows += 1
                if len(batch) >= batch_size:
                    writer.write_table(_rows_to_table(batch, schema))
                    batch.clear()
            if batch:
                writer.write_table(_rows_to_table(batch, schema))
        finally:
            writer.close()
    return {"rows": rows, "logical_sha256": digest.hexdigest()}


def _logical_index_line(row: dict) -> bytes:
    payload = "\x1f".join(str(row.get(name) or "") for name in (
        "index_version", "canonical_version", "rules_version", "game_id", "id_key",
        "source_id", "year", "relpath", "root", "compression", "bytes",
        "original_sha256", "canonical_sha256", "platform_game_key", "events",
        "kyoku_count", "component_id", "component_size", "is_representative",
        "collision_review", "cross_key_duplicate", "quarantine", "quarantine_reason",
        "invalid_reason",
    ))
    return (payload + "\n").encode("utf-8")


def write_store_components_parquet(store: CanonicalStore, path) -> dict:
    import pyarrow.parquet as pq

    schema = component_schema()
    path = Path(path)
    hash_digest = hashlib.sha256()
    batch: list[dict] = []
    rows = 0
    with atomic_path(path) as tmp:
        writer = pq.ParquetWriter(tmp, schema, compression="zstd")
        try:
            for component in store.iter_components():
                hash_digest.update(
                    json.dumps(component, sort_keys=True, separators=(",", ":")).encode()
                    + b"\n"
                )
                batch.append(component)
                rows += 1
                if len(batch) >= 10000:
                    writer.write_table(_rows_to_table(batch, schema))
                    batch.clear()
            if batch:
                writer.write_table(_rows_to_table(batch, schema))
        finally:
            writer.close()
    return {"rows": rows, "logical_sha256": hash_digest.hexdigest()}


def write_store_quarantine_jsonl(store: CanonicalStore, path) -> dict:
    path = Path(path)
    rows = 0
    digest = hashlib.sha256()
    with atomic_path(path) as tmp:
        with open(tmp, "w", encoding="utf-8") as handle:
            for row in store.iter_index_rows():
                if not row["quarantine"]:
                    continue
                record = {
                    "relpath": row["relpath"],
                    "year": row["year"],
                    "source_id": row["source_id"],
                    "game_id": row["game_id"],
                    "component_id": row["component_id"],
                    "component_size": row["component_size"],
                    "quarantine_reason": row["quarantine_reason"],
                    "invalid_reason": row["invalid_reason"],
                    "canonical_sha256": row["canonical_sha256"],
                }
                line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                handle.write(line)
                digest.update(line.encode("utf-8"))
                rows += 1
    return {"rows": rows, "logical_sha256": digest.hexdigest()}


def write_index_jsonl(rows, path) -> dict:
    path = Path(path)
    count = 0
    digest = hashlib.sha256()
    with atomic_path(path) as tmp:
        with open(tmp, "w", encoding="utf-8") as handle:
            for row in rows:
                payload = {
                    "component_id": row["component_id"],
                    "component_size": row["component_size"],
                    "game_id": row["game_id"],
                    "relpath": row["relpath"],
                    "year": row["year"],
                    "compression": row.get("compression"),
                    "canonical_sha256": row["canonical_sha256"],
                    "original_sha256": row.get("original_sha256"),
                    "is_representative": row["is_representative"],
                    "collision_review": row["collision_review"],
                    "cross_key_duplicate": row["cross_key_duplicate"],
                    "quarantine": row["quarantine"],
                    "quarantine_reason": row["quarantine_reason"],
                    "invalid_reason": row["invalid_reason"],
                }
                line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
                handle.write(line)
                digest.update(line.encode("utf-8"))
                count += 1
    return {"rows": count, "logical_sha256": digest.hexdigest()}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Deterministic streaming stratified sample
# --------------------------------------------------------------------------------------


def stratum_key(row: dict, fields=("year", "source_id")) -> tuple:
    return tuple(str(row.get(field)) for field in fields)


def sample_priority(seed: int, key: tuple, relpath: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\x00{key}\x00{relpath}".encode("utf-8")).digest()[:8], "big"
    )


def fill_priority(seed: int, relpath: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\x00{relpath}".encode("utf-8")).digest()[:8], "big"
    )


class StreamingStratifiedSampler:
    """Same selection rule as :func:`stratified_sample` with O(sample) memory.

    Per ``(year, source_id)`` stratum keep the ``quota`` rows with the smallest
    ``sha256(seed \\0 stratum \\0 relpath)``; the shortfall is filled by the globally
    smallest ``sha256(seed \\0 relpath)`` among the rest (a second pass over the input).
    """

    def __init__(self, max_total: int, seed: int = 20261121,
                 fields=("year", "source_id")):
        self.max_total = max_total
        self.seed = seed
        self.fields = fields
        self.quota = None
        self._heaps: dict[tuple, list] = defaultdict(list)
        self._sequence = 0

    def configure(self, strata_count: int) -> None:
        self.quota = max(1, self.max_total // max(1, strata_count))

    def feed(self, row: dict) -> None:
        if self.quota is None:
            raise DedupError("sampler.configure() must be called before feed()")
        key = stratum_key(row, self.fields)
        self._sequence += 1
        priority = sample_priority(self.seed, key, row["relpath"])
        heap = self._heaps[key]
        heapq.heappush(heap, (-priority, -self._sequence, row["relpath"], row))
        if len(heap) > self.quota:
            heapq.heappop(heap)

    def selected(self) -> list[dict]:
        rows = [item[3] for heap in self._heaps.values() for item in heap]
        rows.sort(key=lambda row: (row.get("year") or 0, row["relpath"]))
        return rows


def stratified_sample(rows: list[dict], max_total: int = 1000, seed: int = 20261121,
                      strata_fields=("year", "source_id")) -> list[dict]:
    """Deterministic hash-stratified sample over ``rows`` (same seed => same sample)."""
    if max_total <= 0:
        return []
    import heapq

    buckets = defaultdict(list)
    for order, row in enumerate(rows):
        key = stratum_key(row, strata_fields)
        priority = sample_priority(seed, key, row["relpath"])
        buckets[key].append((priority, order, row["relpath"], row))
    strata = sorted(buckets)
    quota = max(1, max_total // max(1, len(strata)))
    selected = []
    for key in strata:
        chunks = sorted(buckets[key], key=lambda item: item[0])[:quota]
        selected.extend(item[3] for item in chunks)
    if len(selected) < max_total:
        chosen = {row["relpath"] for row in selected}
        rest = sorted(
            (row for key in strata for _, _, _, row in buckets[key]
             if row["relpath"] not in chosen),
            key=lambda row: (fill_priority(seed, row["relpath"]), row["relpath"]),
        )
        selected.extend(rest[: max_total - len(selected)])
    selected.sort(key=lambda row: (row.get("year") or 0, row["relpath"]))
    return selected[:max_total]


def select_streamed(rows, max_total: int, seed: int = 20261121,
                    strata_fields=("year", "source_id")) -> list[dict]:
    """Two-pass equivalent of :func:`stratified_sample` over an iterable of rows.

    The iterable must be re-iterable (a list or a factory) because the fill pass is a
    second scan.  Memory is O(max_total).
    """
    if max_total <= 0:
        return []
    strata = {stratum_key(row, strata_fields) for row in rows}
    sampler = StreamingStratifiedSampler(max_total, seed, strata_fields)
    sampler.configure(len(strata))
    for row in rows:
        sampler.feed(row)
    selected = sampler.selected()
    if len(selected) < max_total:
        chosen = {row["relpath"] for row in selected}
        remaining = max_total - len(selected)
        heap = []
        sequence = 0
        for row in rows:
            if row["relpath"] in chosen:
                continue
            sequence += 1
            priority = fill_priority(seed, row["relpath"])
            heapq.heappush(heap, (-priority, -sequence, row["relpath"], row))
            if len(heap) > remaining:
                heapq.heappop(heap)
        selected.extend(item[3] for item in heap)
        selected.sort(key=lambda row: (row.get("year") or 0, row["relpath"]))
    return selected[:max_total]
