#!/usr/bin/env python3
"""PR-45 disjointness audit: B0 train sample and consumed wall windows.

The validation corpus must be provably disjoint from

* the frozen **B0 training sample** (500 human Tenhou games, sampled by
  ``sha256("20261121:" + game_id)``), and
* every **historical evaluation wall window** recorded in
  ``docs/riichiformer/EVAL_POOL_REGISTRY.md`` (never a "virgin" claim).

Two independent notions are checked:

1. **Game identity.**  B0 rows are H-02 human games with ``tenhou:<...>`` ids;
   self-play games are ``<seed>:<key>:<rotation>``.  The audit records the
   literal set intersection (must be zero) *and* the namespace evidence, so a
   future change that starts emitting human ids into the validation corpus is
   caught instead of being hidden by the namespace mismatch.
2. **Seed space.**  A new wall window ``[seed_start, seed_start + games/4)``
   must not intersect any consumed window, and the new ``seed_key`` must not be
   one of the historically used keys.

Nothing here reads another worktree or writes to a frozen path; a B0 artifact is
opened read-only (SQLite ``mode=ro`` or a parquet read).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..selfplay.provenance import utc_now

__all__ = [
    "B0_SAMPLE_SEED",
    "HISTORICAL_SEED_WINDOWS",
    "RESERVED_SEED_KEYS",
    "SeedWindow",
    "audit_seed_window",
    "build_intersection_report",
    "game_id_intersection",
    "load_b0_game_ids",
]

#: Fixed B0 sample seed (PR-44 prereg).  It is a *sampling* seed, not a wall key;
#: the two must never be conflated.
B0_SAMPLE_SEED = 20261121
B0_SAMPLE_SIZE = 500


@dataclass(frozen=True)
class SeedWindow:
    label: str
    start: int
    stop: int
    role: str

    def as_dict(self) -> dict:
        return {"label": self.label, "start": self.start, "stop": self.stop,
                "role": self.role}


#: Consumed evaluation wall windows from ``EVAL_POOL_REGISTRY.md`` §3 (half-open).
#: An EX1 corpus is registered as ``900000000+`` (100 walls), ``901000000+``
#: (100 walls) and ``902000000+`` (200 walls); the stop values below are the
#: documented game counts converted to walls, intentionally conservative.
HISTORICAL_SEED_WINDOWS: tuple[SeedWindow, ...] = (
    SeedWindow("variance_pool_1", 10000, 13000, "consumed/select"),
    SeedWindow("gate_1v3", 10000, 10500, "consumed/gate"),
    SeedWindow("variance_pool_2", 20000, 23000, "consumed/select"),
    SeedWindow("variance_pool_3", 30000, 33000, "consumed/select"),
    SeedWindow("akochan_extra", 40000, 40030, "consumed/trend"),
    SeedWindow("pool_b", 50000, 53000, "consumed/confirm"),
    SeedWindow("akochan_trend", 60000, 63002, "consumed/trend"),
    SeedWindow("a14_confirm", 70000, 73000, "consumed/confirm"),
    SeedWindow("m3_confirm", 120000, 126000, "consumed/confirm"),
    SeedWindow("ex1_calib", 900000000, 900000100, "consumed/internal"),
    SeedWindow("ex1_select", 901000000, 901000100, "consumed/internal"),
    SeedWindow("ex1_report", 902000000, 902000200, "consumed/internal"),
)

#: Historically used ``seed_key`` values (registry §2).  A new validation key
#: must not equal any of them, even though these keys belonged to evaluation or
#: training drains rather than to a corpus like ours.
RESERVED_SEED_KEYS: dict[int, str] = {
    0x5DFAA4CEF265CD7: "main eval/train/selfplay default key",
    0x80A7B9E2DEC96FA: "runs/1v3 2000-game gate",
    0x455831: "EX1 internal corpus",
    0x923B8015A6879BA9: "upstream bug #114 200-game probe",
    0x2000: "upstream bug #114 spy_test_play",
    0x4187FA4A2A7A20B8: "upstream bug #114 spy_train_play",
    B0_SAMPLE_SEED: "selfplay CLI default key / B0 sample seed (do not reuse)",
}

#: Local self-play smoke walls observed in this checkout's artifacts (PR-41/43).
#: Recorded so a later validation run cannot reuse the same walls by accident.
KNOWN_SELFPLAY_SEED_WINDOWS: tuple[SeedWindow, ...] = (
    SeedWindow("pr41_pr43_smoke", 1000, 1001, "selfplay/smoke"),
)


class DisjointnessError(RuntimeError):
    """A corpus is not provably disjoint from B0 or the consumed windows."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# B0 sample loading (read-only)
# ---------------------------------------------------------------------------
def _load_ids_from_shard(path: Path) -> list[str]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [row[0] for row in conn.execute("SELECT game_id FROM games ORDER BY game_id")]
    finally:
        conn.close()


def _load_ids_from_parquet(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["game_id"])
    return [str(value) for value in table.column("game_id").to_pylist()]


def _load_ids_from_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("game_ids", "games", "selected_game_ids"):
            if isinstance(payload.get(key), list):
                values = payload[key]
                return [str(item.get("game_id") if isinstance(item, dict) else item)
                        for item in values]
        raise DisjointnessError(f"{path}: no game-id list in JSON payload")
    if isinstance(payload, list):
        return [str(item.get("game_id") if isinstance(item, dict) else item)
                for item in payload]
    raise DisjointnessError(f"{path}: unsupported JSON payload")


def _candidate_b0_paths() -> list[Path]:
    return [
        Path("/tmp/b0_sample500.parquet"),
        Path("/tmp/b0_500.rfshard"),
        Path("/tmp/b0_sample500.json"),
    ]


def load_b0_game_ids(path: str | Path | None = None) -> dict:
    """Load the frozen B0 training game ids read-only.

    ``path=None`` tries the known local artifacts in order.  Returns a dict with
    ``ids`` (frozenset), ``source``, ``sha256``, ``format`` and ``loaded``; when
    nothing is found it returns ``loaded=False`` and the caller must report the
    direct intersection as ``NOT_RUN`` rather than as a pass.
    """
    candidates = [Path(path)] if path is not None else _candidate_b0_paths()
    tried = []
    for candidate in candidates:
        tried.append(str(candidate))
        if not candidate.exists():
            continue
        if candidate.suffix == ".parquet":
            ids = _load_ids_from_parquet(candidate)
            kind = "parquet"
        elif candidate.suffix in (".rfshard", ".sqlite", ".db"):
            ids = _load_ids_from_shard(candidate)
            kind = "rfshard"
        elif candidate.suffix == ".json":
            ids = _load_ids_from_json(candidate)
            kind = "json"
        else:
            continue
        return {
            "loaded": True,
            "source": str(candidate),
            "format": kind,
            "sha256": _file_sha256(candidate),
            "game_count": len(ids),
            "ids": frozenset(ids),
        }
    return {"loaded": False, "source": None, "format": None, "sha256": None,
            "game_count": 0, "ids": frozenset(), "tried": tried}


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def audit_seed_window(seed_start: int, games: int, seed_key: int, *,
                      windows: Sequence[SeedWindow] = HISTORICAL_SEED_WINDOWS,
                      reserved_keys: dict[int, str] | None = None) -> dict:
    """Audit one half-open wall window and its key against consumed history."""
    if isinstance(games, bool) or not isinstance(games, int) or games <= 0 \
            or games % 4 != 0:
        raise DisjointnessError(f"games must be a positive multiple of 4, got {games!r}")
    if isinstance(seed_start, bool) or not isinstance(seed_start, int) or seed_start < 0:
        raise DisjointnessError(f"seed_start must be a non-negative int, got {seed_start!r}")
    if isinstance(seed_key, bool) or not isinstance(seed_key, int) or not 0 < seed_key < 2 ** 64:
        raise DisjointnessError(f"seed_key must be a 64-bit non-zero int, got {seed_key!r}")
    walls = games // 4
    start, stop = seed_start, seed_start + walls
    overlaps = [window.as_dict() for window in windows
                if start < window.stop and window.start < stop]
    reserved = reserved_keys if reserved_keys is not None else RESERVED_SEED_KEYS
    key_label = reserved.get(seed_key)
    return {
        "seed_start": start,
        "seed_stop": stop,
        "walls": walls,
        "games": games,
        "seed_key": seed_key,
        "seed_key_hex": f"0x{seed_key:016x}",
        "window": f"[{start},{stop})",
        "window_overlaps": overlaps,
        "window_ok": not overlaps,
        "seed_key_reserved": key_label,
        "seed_key_ok": key_label is None,
        "checked_windows": len(windows),
    }


def game_id_intersection(validation_ids: Iterable[str],
                         b0_ids: Iterable[str]) -> dict:
    left = set(validation_ids)
    right = set(b0_ids)
    return {
        "validation_games": len(left),
        "b0_games": len(right),
        "intersection_count": len(left & right),
        "intersection_examples": sorted(left & right)[:10],
    }


def _namespace(value: str) -> str:
    if value.startswith("tenhou:"):
        return "tenhou"
    if value.count(":") == 2 and value.split(":")[0].isdigit():
        return "selfplay"
    return "other"


def build_intersection_report(*, seed_start: int, games: int, seed_key: int,
                              validation_game_ids: Iterable[str],
                              generator_sha256: str,
                              b0_artifact: str | Path | None = None,
                              historical_windows: Sequence[SeedWindow]
                              = HISTORICAL_SEED_WINDOWS) -> dict:
    """Full PR-45 disjointness report (JSON-serialisable).

    ``ok`` is ``True`` only when every check ran and passed; an unavailable B0
    artifact makes ``complete=False`` and ``ok=None`` (``NOT_RUN``), never a
    silent pass.
    """
    game_ids = sorted(set(validation_game_ids))
    window = audit_seed_window(seed_start, games, seed_key, windows=historical_windows)
    b0 = load_b0_game_ids(b0_artifact)
    if b0["loaded"]:
        direct = game_id_intersection(game_ids, b0["ids"])
        b0_check = {
            "status": "RUN",
            "game_id_intersection": direct["intersection_count"],
            "ok": direct["intersection_count"] == 0,
            "details": direct,
        }
    else:
        b0_check = {
            "status": "NOT_RUN",
            "game_id_intersection": None,
            "ok": None,
            "reason": "no B0 sample artifact found (pass --b0-artifact)",
            "tried": b0.get("tried", []),
        }
    namespaces = sorted({_namespace(game_id) for game_id in game_ids})
    b0_namespaces = sorted({_namespace(game_id) for game_id in b0["ids"]}) \
        if b0["loaded"] else ["tenhou"]
    namespace_check = {
        "validation_namespaces": namespaces,
        "b0_namespaces": b0_namespaces,
        "namespace_intersection": sorted(set(namespaces) & set(b0_namespaces)),
        "ok": not (set(namespaces) & set(b0_namespaces)),
    }
    # B0's deterministic sample is keyed on sha256("<seed>:<game_id>") for H-02
    # game ids.  Recompute it for our ids and record that none of our ids can be
    # a B0 sample member because the sample selection only ever read H-02 rows.
    sampled_hashes = {_sha256_text(f"{B0_SAMPLE_SEED}:{game_id}") for game_id in game_ids}
    b0_membership_check = {
        "definition": 'sha256("20261121:" + game_id) over H-02 train rows',
        "checked": len(sampled_hashes),
        "hits": 0 if b0["loaded"] else None,
        "ok": (True if b0["loaded"] else None),
        "basis": ("B0 sampling read only H-02 human rows; self-play ids are not "
                  "members of that row universe"),
    }
    if b0["loaded"]:
        b0_hashes = {_sha256_text(f"{B0_SAMPLE_SEED}:{game_id}") for game_id in b0["ids"]}
        hits = sampled_hashes & b0_hashes
        b0_membership_check["hits"] = len(hits)
        b0_membership_check["ok"] = not hits
        b0_membership_check["examples"] = sorted(hits)[:5]
    checks = {
        "seed_window": {
            "status": "RUN",
            "ok": window["window_ok"],
            "overlaps": window["window_overlaps"],
        },
        "seed_key": {
            "status": "RUN",
            "ok": window["seed_key_ok"],
            "reserved_label": window["seed_key_reserved"],
            "key": window["seed_key_hex"],
        },
        "b0_game_ids": b0_check,
        "game_id_namespace": namespace_check,
        "b0_sample_membership": b0_membership_check,
        "generator_sha256": {
            "status": "RUN",
            "validation": generator_sha256,
            "b0_pipeline": "human-log shard build (no self-play generator)",
            "ok": len(generator_sha256) == 64,
            "note": ("the B0 500-game shard is a human-log pipeline; it has no "
                     "riichiformer.selfplay generator hash, so equality is not "
                     "expected and is not treated as a pass criterion"),
        },
    }
    values = [entry["ok"] for entry in checks.values()]
    complete = all(value is not None for value in values)
    overall = all(values) if complete else None
    return {
        "format": "rf-validation-intersection/1",
        "created_utc": utc_now(),
        "purpose": "prove PR-45 validation corpus is disjoint from B0 and history",
        "validation": {
            "seed_key": seed_key,
            "seed_key_hex": f"0x{seed_key:016x}",
            "seed_start": window["seed_start"],
            "seed_stop": window["seed_stop"],
            "games": games,
            "walls": window["walls"],
            "window": window["window"],
            "game_id_count": len(game_ids),
            "game_id_examples": game_ids[:5],
            "generator_sha256": generator_sha256,
        },
        "b0": {
            "sample_seed": B0_SAMPLE_SEED,
            "sample_size": B0_SAMPLE_SIZE,
            "artifact": b0.get("source"),
            "artifact_sha256": b0.get("sha256"),
            "artifact_format": b0.get("format"),
            "game_count": b0["game_count"],
            "loaded": b0["loaded"],
        },
        "historical_windows_checked": [entry.as_dict() for entry in historical_windows],
        "checks": checks,
        "complete": complete,
        "ok": overall,
        "not_run": ([] if complete else ["b0_game_ids"]),
        "note": (
            "This report proves data separation only.  Weak local heuristic bots "
            "are not a strong teacher and these games are not Tenhou truth."),
    }


__all__ += ["DisjointnessError", "KNOWN_SELFPLAY_SEED_WINDOWS"]
