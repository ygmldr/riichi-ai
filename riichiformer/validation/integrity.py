#!/usr/bin/env python3
"""PR-45 read-only shard integrity + POV visibility audit.

``ShardReader`` intentionally does not re-hash the per-decision payloads: the
training path trusts the writer and checks prefix/state consistency instead.  The
validation path is an *audit* path, so it does two extra checks the training
loader never performs:

1. **Row hashes.**  Recompute ``input_sha256``/``target_sha256`` from the stored
   columns exactly as ``shard.DecisionRecord`` does and compare them with the
   recorded digest.  Injecting a label into ``targets.expert_action`` without
   rewriting ``target_sha256`` is therefore detected (negative control).
2. **Visibility.**  Walk all four projected POV streams and assert that every
   ``UNKNOWN`` tile belongs to an opponent hidden event (opponent ``tsumo`` or
   opponent ``ankan``) and that no concealed field (``tehais``/``ura_markers``)
   survived projection.  ``UNKNOWN`` on a public tile, or a known tile on an
   opponent draw, is a hidden-information violation.

Neither check mutates the shard; both open SQLite with ``mode=ro``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from ..data.shard import ShardMetadata, read_manifest, unpack_json
from ..schemas.action_v3 import UNKNOWN_TILE_KIND
from ..schemas.decision_v3 import ACTION_SPACE

__all__ = [
    "IntegrityError",
    "VisibilityError",
    "assert_visibility_contract",
    "recompute_input_sha256",
    "recompute_target_sha256",
    "verify_shard_integrity",
    "visibility_counts",
]

#: Keys that would carry concealed information if they ever reached a POV stream.
#: ``ura_markers`` is deliberately *not* here: the hora settlement event may expose
#: them (schema visibility ``AFTER_EVENT``), which is not a hidden-information leak.
_FORBIDDEN_PROJECTED_KEYS = frozenset({
    "tehais", "hand", "hands", "wall", "seed", "names", "paishu",
    "opponent_hands", "concealed_hands",
})

_TILE_FIELDS = ("tile", "dora_marker")
_TILE_LIST_FIELDS = ("own_haipai", "consumed", "ura_markers")
#: Real (unpadded) consumed-tile count per call event.  ``event_v3`` pads the
#: ``consumed`` tuple to four slots; slots beyond this count are placeholders and
#: must be UNKNOWN without meaning "hidden".
_REAL_CONSUMED = {"chi": 2, "pon": 2, "daiminkan": 3, "kakan": 3, "ankan": 4}


class IntegrityError(RuntimeError):
    """A stored hash does not match its payload (tamper/corruption)."""


class VisibilityError(RuntimeError):
    """The POV projection leaks or over-masks hidden information."""


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def recompute_input_sha256(row) -> str:
    payload = {
        "before_event_idx": int(row["before_event_idx"]),
        "kyoku_idx": int(row["kyoku_idx"]),
        "phase": str(row["phase"]),
        "legal_mask": unpack_json(row["legal_mask"]),
        "tile_state": unpack_json(row["tile_state"]),
        "players_public": unpack_json(row["players_public"]),
        "global_public": unpack_json(row["global_public"]),
    }
    return _sha(_canonical(payload))


def recompute_target_sha256(row, *, schema_version: str) -> str:
    # ``DecisionTarget.to_dict()`` includes schema_version + oracle_labels; the
    # decision builder never sets oracle labels, so the writer hashed this exact
    # six-key payload and only four columns were persisted.
    payload = {
        "schema_version": schema_version,
        "expert_action": row["expert_action"],
        "supervision_valid": bool(row["supervision_valid"]),
        "censor_reason": row["censor_reason"],
        "rank_final": row["rank_final"],
        "oracle_labels": None,
    }
    return _sha(_canonical(payload))


def verify_shard_integrity(path: str | Path, *, metadata: ShardMetadata | None = None,
                           strict: bool = True) -> dict:
    """Recompute every row hash; ``strict`` raises on the first mismatch class."""
    shard_path = Path(path)
    if metadata is None:
        metadata = ShardMetadata.from_dict(read_manifest(shard_path)["metadata"])
    conn = sqlite3.connect(f"file:{shard_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        input_count = 0
        target_count = 0
        input_mismatch: list[str] = []
        target_mismatch: list[str] = []
        illegal_mask = 0
        orphan_targets = 0
        for row in conn.execute("SELECT * FROM inputs ORDER BY decision_id"):
            input_count += 1
            if len(unpack_json(row["legal_mask"])) != ACTION_SPACE:
                illegal_mask += 1
            digest = recompute_input_sha256(row)
            if digest != row["input_sha256"]:
                input_mismatch.append(row["decision_id"])
        query = ("SELECT t.*, i.decision_id AS input_id FROM targets t "
                 "LEFT JOIN inputs i ON i.decision_id=t.decision_id "
                 "ORDER BY t.decision_id")
        for row in conn.execute(query):
            target_count += 1
            if row["input_id"] is None:
                orphan_targets += 1
            digest = recompute_target_sha256(row, schema_version=metadata.schema_version)
            if digest != row["target_sha256"]:
                target_mismatch.append(row["decision_id"])
        result = {
            "shard": str(shard_path),
            "purpose": metadata.purpose,
            "inputs": input_count,
            "targets": target_count,
            "input_hash_mismatches": len(input_mismatch),
            "target_hash_mismatches": len(target_mismatch),
            "input_mismatch_examples": input_mismatch[:5],
            "target_mismatch_examples": target_mismatch[:5],
            "bad_legal_mask_width": illegal_mask,
            "orphan_targets": orphan_targets,
            "ok": (not input_mismatch and not target_mismatch
                   and not illegal_mask and not orphan_targets),
        }
    finally:
        conn.close()
    if strict and not result["ok"]:
        raise IntegrityError(
            "shard integrity failure: "
            f"input_hash_mismatches={result['input_hash_mismatches']} "
            f"target_hash_mismatches={result['target_hash_mismatches']} "
            f"bad_mask={result['bad_legal_mask_width']} "
            f"orphans={result['orphan_targets']}")
    return result


# ---------------------------------------------------------------------------
# visibility
# ---------------------------------------------------------------------------
def _tile_unknown(tile: dict | None) -> bool | None:
    if not isinstance(tile, dict):
        return None
    if "kind" not in tile:
        return None
    return int(tile["kind"]) == UNKNOWN_TILE_KIND


def _iter_tiles(event: dict):
    """Yield ``(field, tile_dict, position)`` for every tile carried by an event."""
    for name in _TILE_FIELDS:
        tile = event.get(name)
        if isinstance(tile, dict):
            yield name, tile, None
    for name in _TILE_LIST_FIELDS:
        values = event.get(name)
        if isinstance(values, list):
            for index, tile in enumerate(values):
                if isinstance(tile, dict):
                    yield name, tile, index


def visibility_counts(streams) -> dict:
    """Count unknown tiles per event type/actor without judging them."""
    counts: dict[str, int] = {}
    total_unknown = 0
    for stream in streams:
        for event in stream:
            kind = event.get("type")
            for field, tile, _index in _iter_tiles(event):
                if _tile_unknown(tile):
                    total_unknown += 1
                    key = f"{kind}.{field}"
                    counts[key] = counts.get(key, 0) + 1
    return {"unknown_total": total_unknown, "by_slot": dict(sorted(counts.items()))}


def assert_visibility_contract(pov_streams, *, game_id: str = "",
                               strict: bool = True) -> dict:
    """Assert ``UNKNOWN`` only on opponents' hidden events.

    ``pov_streams`` is the stored four-stream structure: a sequence of four
    sequences of projected event dicts (POV relative seats: 0 is the viewer).
    ``UNKNOWN`` is allowed only for

    * an opponent ``tsumo`` tile, and
    * the four tiles of an opponent ``ankan``;

    plus the structural padding of ``consumed`` (chi/pon carry 2 real tiles,
    daiminkan/kakan 3, ankan 4, and the frozen event schema pads to four slots).
    Every other unknown tile is a visibility violation; every known opponent draw
    or opponent ankan tile is a leak.

    Returns a report; with ``strict`` raises :class:`VisibilityError` on any
    violation instead of only reporting it.
    """
    violations: list[str] = []
    streams = 0
    events = 0
    unknown_total = 0
    hidden_unknown = 0
    padding_unknown = 0
    for pov, stream in enumerate(pov_streams):
        streams += 1
        for seq, event in enumerate(stream):
            events += 1
            for key in event:
                if key in _FORBIDDEN_PROJECTED_KEYS:
                    violations.append(
                        f"POV {pov} event[{seq}] carries concealed key {key!r}")
            kind = event.get("type")
            actor_rel = event.get("actor_rel")
            real_consumed = _REAL_CONSUMED.get(kind)
            for field, tile, index in _iter_tiles(event):
                unknown = _tile_unknown(tile)
                if unknown is None:
                    continue
                location = f"POV {pov} {kind}[{seq}] {field}"
                if index is not None:
                    location += f"[{index}]"
                if unknown:
                    unknown_total += 1
                    if kind == "tsumo" and field == "tile" and actor_rel != 0:
                        hidden_unknown += 1
                    elif (kind == "ankan" and field == "consumed"
                          and actor_rel != 0 and index is not None
                          and index < real_consumed):
                        hidden_unknown += 1
                    elif (field == "consumed" and index is not None
                          and real_consumed is not None and index >= real_consumed):
                        padding_unknown += 1
                    else:
                        violations.append(
                            f"{location} is UNKNOWN but not an opponent hidden event")
                else:
                    if kind == "tsumo" and field == "tile" and actor_rel != 0:
                        violations.append(f"{location} reveals an opponent draw")
                    elif (kind == "ankan" and field == "consumed"
                          and actor_rel != 0 and index is not None
                          and index < real_consumed):
                        violations.append(
                            f"{location} reveals an opponent ankan tile")
    result = {
        "game_id": game_id,
        "streams": streams,
        "events": events,
        "unknown_tiles": unknown_total,
        "unknown_on_hidden_events": hidden_unknown,
        "unknown_padding_slots": padding_unknown,
        "violations": violations[:20],
        "violation_count": len(violations),
        "ok": not violations,
    }
    if strict and violations:
        raise VisibilityError(
            f"visibility contract violated for {game_id or 'shard'}: "
            + "; ".join(violations[:5]))
    return result


__all__ += ["_FORBIDDEN_PROJECTED_KEYS"]
