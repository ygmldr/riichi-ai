"""Loader and hash verifier for ``configs/riichiformer/schema_v3.json`` (PR-08).

Hashes are computed over **canonical JSON** (sorted keys, no whitespace, UTF-8), not over
raw file bytes, so formatting alone cannot change a hash; any semantic change does.

* ``schema_hash``: sha256 of the schema object with the ``schema_hash`` key removed;
* ``action_map_hash``: sha256 of ``schema["action_map"]``;
* ``normalizer_hash``: sha256 of ``schema["normalizer"]``.

A shard/checkpoint records all three; :func:`load_schema` refuses a mismatch instead of
migrating silently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

SCHEMA_VERSION = "rf-schema-v3/2"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "configs" / "riichiformer" / "schema_v3.json"
SCHEMA_HASH_KEY = "schema_hash"
ACTION_MAP_HASH_KEY = "action_map_hash"
NORMALIZER_HASH_KEY = "normalizer_hash"
HASH_KEYS = (SCHEMA_HASH_KEY, ACTION_MAP_HASH_KEY, NORMALIZER_HASH_KEY)


class SchemaError(ValueError):
    """Base class for schema-file errors."""


class SchemaVersionMismatch(SchemaError):
    pass


class SchemaHashMismatch(SchemaError):
    pass


class SchemaFileError(SchemaError):
    pass


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_schema_hash(schema: Mapping) -> str:
    payload = {key: value for key, value in schema.items() if key != SCHEMA_HASH_KEY}
    return sha256_text(canonical_json(payload))


def compute_action_map_hash(schema: Mapping) -> str:
    if "action_map" not in schema:
        raise SchemaError("schema has no action_map")
    return sha256_text(canonical_json(schema["action_map"]))


def compute_normalizer_hash(schema: Mapping) -> str:
    if "normalizer" not in schema:
        raise SchemaError("schema has no normalizer")
    return sha256_text(canonical_json(schema["normalizer"]))


def verify_schema(schema: Mapping, *, expected_hashes: Mapping | None = None) -> dict:
    """Verify version + all three hashes; return the schema unchanged."""
    if not isinstance(schema, Mapping):
        raise SchemaError(f"schema must be a mapping, got {type(schema).__name__}")
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"schema_version {schema.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    for key in HASH_KEYS:
        value = schema.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise SchemaHashMismatch(f"{key} missing or not a sha256 hex digest")
    recomputed = {
        SCHEMA_HASH_KEY: compute_schema_hash(schema),
        ACTION_MAP_HASH_KEY: compute_action_map_hash(schema),
        NORMALIZER_HASH_KEY: compute_normalizer_hash(schema),
    }
    for key, value in recomputed.items():
        if schema[key] != value:
            raise SchemaHashMismatch(
                f"{key} mismatch: file={schema[key]} canonical={value}"
            )
    if expected_hashes is not None:
        if not isinstance(expected_hashes, Mapping):
            raise SchemaHashMismatch(
                f"expected_hashes must be a mapping, got {type(expected_hashes).__name__}"
            )
        expected_keys = set(expected_hashes)
        required_keys = set(HASH_KEYS)
        if expected_keys != required_keys:
            raise SchemaHashMismatch(
                "expected_hashes must contain exactly "
                f"{sorted(required_keys)}; missing={sorted(required_keys - expected_keys)} "
                f"unknown={sorted(expected_keys - required_keys)}"
            )
        for key in HASH_KEYS:
            if expected_hashes[key] != schema[key]:
                raise SchemaHashMismatch(
                    f"{key} mismatch against checkpoint/shard: "
                    f"expected={expected_hashes[key]} actual={schema[key]}"
                )
    return dict(schema)


def load_schema(path: Path | str | None = None,
                *, expected_hashes: Mapping | None = None,
                verify_code: bool = True) -> dict:
    """Load, hash-verify and (optionally) code-consistency-check the schema file."""
    target = Path(path) if path is not None else SCHEMA_PATH
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaFileError(f"cannot read schema {target}: {exc}") from exc
    try:
        schema = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaFileError(f"schema {target} is not valid JSON: {exc}") from exc
    verified = verify_schema(schema, expected_hashes=expected_hashes)
    if verify_code:
        verify_code_consistency(verified)
    return verified


def verify_code_consistency(schema: Mapping) -> None:
    """The frozen JSON must match the code that constructs samples."""
    from . import action_v3, decision_v3, event_v3

    if schema.get("schema_version") != action_v3.SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"schema_version {schema.get('schema_version')!r} != "
            f"action_v3.SCHEMA_VERSION {action_v3.SCHEMA_VERSION!r}"
        )
    if schema.get("phases") != [phase.value for phase in action_v3.ActionPhase]:
        raise SchemaHashMismatch("schema phases do not match action_v3.ActionPhase")
    expected_phase_actions = {
        phase.value: sorted(action_v3.PHASE_LEGAL_ACTIONS[phase])
        for phase in action_v3.ActionPhase
    }
    if schema.get("phase_legal_actions") != expected_phase_actions:
        raise SchemaHashMismatch(
            "schema phase_legal_actions do not match action_v3.PHASE_LEGAL_ACTIONS"
        )
    expected_tile_kinds = {
        "count": action_v3.TILE_KIND_COUNT,
        "unknown": action_v3.UNKNOWN_TILE_KIND,
        "names": list(action_v3.TILE_KIND_NAMES),
    }
    if schema.get("tile_kinds") != expected_tile_kinds:
        raise SchemaHashMismatch("schema tile_kinds do not match action_v3")
    aka = schema.get("aka", {})
    if (aka.get("colors") != {"5m": 0, "5p": 1, "5s": 2}
            or aka.get("kinds") != {"5m": 4, "5p": 13, "5s": 22}
            or aka.get("per_color") != action_v3.AKA_PER_COLOR):
        raise SchemaHashMismatch("schema aka do not match action_v3")
    rulesets = schema.get("rulesets", {})
    if set(rulesets) != {"tenhou_4p_ranked", "mortal_arena_pinned"}:
        raise SchemaHashMismatch("schema rulesets drifted")
    if rulesets["tenhou_4p_ranked"].get("allowed_deviations") != []:
        raise SchemaHashMismatch("tenhou_4p_ranked must not allow deviations")
    if rulesets["mortal_arena_pinned"].get("allowed_deviations") != ["R02", "R25", "R26"]:
        raise SchemaHashMismatch("mortal_arena_pinned deviations drifted")
    expected_deviations = {
        code.value: {"description": description}
        for code, description in decision_v3.RULESET_DEVIATIONS.items()
    }
    if schema.get("ruleset_deviations") != expected_deviations:
        raise SchemaHashMismatch(
            "schema ruleset_deviations do not match decision_v3.RULESET_DEVIATIONS"
        )
    if schema.get("source_kinds") != [kind.value for kind in decision_v3.SourceKind]:
        raise SchemaHashMismatch("schema source_kinds do not match decision_v3.SourceKind")
    versioning = schema.get("versioning", {})
    if versioning.get("version_string") != SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"schema versioning.version_string {versioning.get('version_string')!r} != "
            f"{SCHEMA_VERSION!r}"
        )
    if schema.get("action_map") != action_v3.canonical_action_map():
        raise SchemaHashMismatch(
            "schema action_map does not match action_v3.canonical_action_map()"
        )
    if schema.get(ACTION_MAP_HASH_KEY) != action_v3.ACTION_MAP_HASH:
        raise SchemaHashMismatch(
            "action_map_hash does not match action_v3.ACTION_MAP_HASH"
        )
    if schema.get("event_types") != event_v3.event_type_specs():
        raise SchemaHashMismatch(
            "schema event_types do not match event_v3.event_type_specs()"
        )
    if tuple(schema.get("decision_input_fields", ())) != decision_v3.DECISION_INPUT_FIELDS:
        raise SchemaHashMismatch("schema decision_input_fields drifted")
    if tuple(schema.get("decision_target_fields", ())) != decision_v3.DECISION_TARGET_FIELDS:
        raise SchemaHashMismatch("schema decision_target_fields drifted")
    if set(schema.get("forward_input_forbidden_fields", ())) != set(
            decision_v3.FORBIDDEN_FORWARD_KEYS):
        raise SchemaHashMismatch("schema forward_input_forbidden_fields drifted")


__all__ = [
    "ACTION_MAP_HASH_KEY", "HASH_KEYS", "NORMALIZER_HASH_KEY", "SCHEMA_HASH_KEY",
    "SCHEMA_PATH", "SCHEMA_VERSION", "SchemaError", "SchemaFileError",
    "SchemaHashMismatch", "SchemaVersionMismatch", "canonical_json",
    "compute_action_map_hash", "compute_normalizer_hash", "compute_schema_hash",
    "load_schema", "sha256_text", "verify_code_consistency", "verify_schema",
]
