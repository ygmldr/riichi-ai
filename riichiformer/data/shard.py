#!/usr/bin/env python3
"""PR-12 immutable sequence shard: one game row, four POV streams, split I/O.

On disk a shard is a SQLite database with three disjoint sections:

* ``games``  — one row per game/component: four POV-projected event streams, per-kyoku
  offsets and decision counts.  The full prefix is stored **once** per POV stream;
* ``inputs`` — one row per decision: cutoff/phase/mask/visible state, no target fields;
* ``targets`` — one row per decision: expert/censor/rank only, keyed by decision_id.

A shard also has a JSON sidecar manifest.  Writes go to ``<path>.tmp`` and are atomically
renamed on :meth:`ShardWriter.finalize`; resume opens the temp/final DB and is idempotent
per game_id.  Licensed training shards require ``train_allowed=True`` and
rights_status ALLOWED.  An ``operator_private_research`` shard keeps the real
``rights_status`` (UNKNOWN/RESTRICTED) and records a policy-config hash plus
``train_eligible_under_policy``; the loader authenticates that hash against an
explicit operator policy before any training row is yielded.  Audit/diagnostic shards
are explicit and carry the same metadata plus a loud purpose.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sqlite3
import zlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ..policy.config import POLICY_VERSION
from ..schemas.action_v3 import ACTION_MAP_HASH, SCHEMA_VERSION, ActionPhase
from ..schemas.decision_v3 import DecisionInput, RulesetDeviation, RulesetId, SourceKind
from ..schemas.event_v3 import event_from_dict
from ..schemas.schema_v3 import load_schema
from . import pov

FORMAT_VERSION = "rf-shard/3"
PURPOSES = ("train", "audit", "diagnostic")
FROZEN_SCHEMA = load_schema()
SCHEMA_HASH = FROZEN_SCHEMA["schema_hash"]
NORMALIZER_HASH = FROZEN_SCHEMA["normalizer_hash"]
ACTION_HASH = FROZEN_SCHEMA["action_map_hash"]
if ACTION_HASH != ACTION_MAP_HASH:  # pragma: no cover - contract drift
    raise AssertionError("schema action_map_hash != action_v3.ACTION_MAP_HASH")

#: Input table columns; a target/label word must never appear here.
INPUT_COLUMNS = (
    "decision_id", "game_id", "pov", "decision_index", "before_event_idx",
    "kyoku_idx", "phase", "legal_mask", "tile_state", "players_public",
    "global_public", "input_sha256",
)
TARGET_COLUMNS = (
    "decision_id", "expert_action", "supervision_valid", "censor_reason",
    "rank_final", "target_sha256",
)


class ShardError(RuntimeError):
    """Malformed shard/record; fail closed."""


class ShardRightsError(ShardError):
    """The shard/source may not be used for the requested purpose."""


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pack_json(value) -> str:
    """JSON+zlib+base64 for repeatable shard payloads (prefix ``z:``)."""
    raw = _canonical(value).encode("utf-8")
    return "z:" + base64.b64encode(zlib.compress(raw, 6)).decode("ascii")


def unpack_json(text: str):
    if not isinstance(text, str):
        raise ShardError("compressed shard field must be text")
    if text.startswith("z:"):
        return json.loads(zlib.decompress(base64.b64decode(text[2:])).decode("utf-8"))
    return json.loads(text)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Chunked file SHA-256 (never loads a whole shard into memory)."""
    return _file_sha(Path(path))


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


#: Deterministic content tables for the reproducible ``content_sha256`` (created_utc is
#: deliberately excluded: it changes every run, the rows must not).
_CONTENT_TABLES = (
    ("games", ("game_id", "component_id", "component_sha256", "split", "source_id",
               "train_allowed", "rights_status", "pov_streams", "kyoku_offsets",
               "decision_count"), "game_id"),
    ("inputs", ("decision_id", "game_id", "pov", "decision_index", "before_event_idx",
                "kyoku_idx", "phase", "legal_mask", "tile_state", "players_public",
                "global_public", "input_sha256"), "decision_id"),
    ("targets", ("decision_id", "expert_action", "supervision_valid", "censor_reason",
                 "rank_final", "target_sha256"), "decision_id"),
)


def content_sha256(path: str | Path, metadata: "ShardMetadata") -> str:
    """Reproducible content hash: metadata minus created_utc + every table row."""
    digest = hashlib.sha256()
    meta = metadata.as_dict()
    meta.pop("created_utc", None)
    digest.update(f"rf-shard-content/{FORMAT_VERSION}\n".encode("ascii"))
    digest.update(_canonical(meta).encode("utf-8") + b"\n")
    conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        for table, columns, order in _CONTENT_TABLES:
            digest.update(f"{table}\n".encode("ascii"))
            sql = f"SELECT {','.join(columns)} FROM {table} ORDER BY {order}"
            for row in conn.execute(sql):
                digest.update(_canonical(list(row)).encode("utf-8") + b"\n")
    finally:
        conn.close()
    return digest.hexdigest()


@dataclass(frozen=True)
class ShardMetadata:
    purpose: str
    source_id: str
    rights_status: str
    train_allowed: bool
    canonical_manifest_sha256: str
    canonical_index_sha256: str
    split_manifest_sha256: str
    split_version: str
    schema_version: str
    schema_hash: str
    action_map_hash: str
    normalizer_hash: str
    ruleset_id: str = RulesetId.TENHOU_4P_RANKED.value
    ruleset_deviations: tuple[str, ...] = ()
    source_kind: str = SourceKind.HUMAN_LOG.value
    projection_version: str = pov.POV_PROJECTION_VERSION
    research_policy: str = "licensed_only"
    research_policy_hash: str = ""
    #: Canonical policy-config version that produced ``research_policy_hash``.
    #: Empty on legacy shards (or licensed-only builds without a config file).
    policy_version: str = ""
    #: Independent external-rights fact; never rewritten by a local policy.
    #: Empty means "inherit ``rights_status``" (normalised in ``__post_init__``).
    external_license_status: str = ""
    #: True only when a policy decision (not merely a local file) made this
    #: training shard eligible.  External licence facts remain in
    #: ``rights_status``/``external_license_status`` and are never overwritten.
    train_eligible_under_policy: bool = False
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def __post_init__(self):
        if self.purpose not in PURPOSES:
            raise ShardError(f"purpose must be one of {PURPOSES}, got {self.purpose!r}")
        if not self.source_id:
            raise ShardError("source_id must be non-empty")
        if not isinstance(self.train_allowed, bool):
            raise ShardError("train_allowed must be bool")
        status = str(self.rights_status).upper()
        if status not in {"ALLOWED", "RESTRICTED", "UNKNOWN"}:
            raise ShardError(f"rights_status must be ALLOWED/RESTRICTED/UNKNOWN, got {self.rights_status!r}")
        if self.research_policy not in {"licensed_only", "operator_private_research"}:
            raise ShardError("unknown research policy")
        if self.research_policy == "operator_private_research":
            if len(self.research_policy_hash) != 64:
                raise ShardError("private research requires a 64-character policy hash")
        elif self.research_policy_hash:
            raise ShardError("licensed_only must not carry a private research hash")
        if self.policy_version not in {"", POLICY_VERSION}:
            raise ShardError(
                f"unsupported policy_version {self.policy_version!r}; expected "
                f"{POLICY_VERSION!r} or empty for legacy shards")
        if not isinstance(self.train_eligible_under_policy, bool):
            raise ShardError("train_eligible_under_policy must be bool")
        external = str(self.external_license_status or "").upper()
        if external:
            if external not in {"ALLOWED", "RESTRICTED", "UNKNOWN"}:
                raise ShardError(
                    "external_license_status must be ALLOWED/RESTRICTED/UNKNOWN, "
                    f"got {self.external_license_status!r}")
            if external != status:
                raise ShardError(
                    "external_license_status must match rights_status; a local "
                    "policy can never rewrite the external fact")
        else:
            external = status
        object.__setattr__(self, "external_license_status", external)
        if self.train_eligible_under_policy and self.research_policy == "licensed_only" \
                and (not self.train_allowed or status != "ALLOWED"):
            raise ShardError(
                "licensed_only train_eligible_under_policy=True requires "
                "train_allowed=True and rights_status=ALLOWED")
        if self.purpose == "train" and self.research_policy == "licensed_only" \
                and (not self.train_allowed or status != "ALLOWED"):
            raise ShardRightsError(
                "licensed training shards require train_allowed=True and rights_status=ALLOWED; "
                f"got train_allowed={self.train_allowed}, rights_status={status}")
        if self.purpose == "train" \
                and self.research_policy == "operator_private_research" \
                and not self.train_eligible_under_policy:
            raise ShardRightsError(
                "private training shards require train_eligible_under_policy=True; "
                "an unauthenticated private shard cannot claim training eligibility")
        for name in ("canonical_manifest_sha256", "canonical_index_sha256",
                     "split_manifest_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ShardError(f"{name} must be a 64-char sha256")
        if self.schema_version != SCHEMA_VERSION:
            raise ShardError(f"schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}")
        if self.schema_hash != SCHEMA_HASH:
            raise ShardError(f"schema_hash {self.schema_hash!r} != frozen {SCHEMA_HASH!r}")
        if self.action_map_hash != ACTION_HASH:
            raise ShardError("action_map_hash mismatch")
        if self.normalizer_hash != NORMALIZER_HASH:
            raise ShardError("normalizer_hash mismatch")
        if self.ruleset_id not in {value.value for value in RulesetId}:
            raise ShardError(f"unknown ruleset_id {self.ruleset_id!r}")
        if self.source_kind not in {value.value for value in SourceKind}:
            raise ShardError(f"unknown source_kind {self.source_kind!r}")
        if self.projection_version != pov.POV_PROJECTION_VERSION:
            raise ShardError(
                f"projection_version {self.projection_version!r} != "
                f"{pov.POV_PROJECTION_VERSION!r}")
        if len(set(self.ruleset_deviations)) != len(self.ruleset_deviations):
            raise ShardError("duplicate ruleset deviations")

    def as_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "purpose": self.purpose,
            "source_id": self.source_id,
            "rights_status": str(self.rights_status).upper(),
            "train_allowed": self.train_allowed,
            "research_policy": self.research_policy,
            "research_policy_hash": self.research_policy_hash,
            "policy_version": self.policy_version,
            "external_license_status": self.external_license_status,
            "train_eligible_under_policy": self.train_eligible_under_policy,
            "canonical_manifest_sha256": self.canonical_manifest_sha256,
            "canonical_index_sha256": self.canonical_index_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "split_version": self.split_version,
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "action_map_hash": self.action_map_hash,
            "normalizer_hash": self.normalizer_hash,
            "ruleset_id": self.ruleset_id,
            "ruleset_deviations": list(self.ruleset_deviations),
            "source_kind": self.source_kind,
            "projection_version": self.projection_version,
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "ShardMetadata":
        if value.get("format_version") != FORMAT_VERSION:
            raise ShardError(f"unsupported shard format {value.get('format_version')!r}")
        # Boolean gate fields must stay booleans.  ``bool("false")`` is True, so a
        # silent coercion here would turn a JSON string into an eligible training
        # shard and bypass the explicit contract in ``__post_init__``.
        train_allowed = value["train_allowed"]
        if not isinstance(train_allowed, bool):
            raise ShardError(
                f"train_allowed must be a JSON boolean, got {type(train_allowed).__name__}")
        eligible = value.get("train_eligible_under_policy", False)
        if not isinstance(eligible, bool):
            raise ShardError(
                "train_eligible_under_policy must be a JSON boolean, got "
                f"{type(eligible).__name__}")
        return cls(
            purpose=value["purpose"], source_id=value["source_id"],
            rights_status=value["rights_status"], train_allowed=train_allowed,
            canonical_manifest_sha256=value["canonical_manifest_sha256"],
            canonical_index_sha256=value["canonical_index_sha256"],
            split_manifest_sha256=value["split_manifest_sha256"],
            split_version=value["split_version"], schema_version=value["schema_version"],
            schema_hash=value["schema_hash"], action_map_hash=value["action_map_hash"],
            normalizer_hash=value["normalizer_hash"], ruleset_id=value["ruleset_id"],
            ruleset_deviations=tuple(value.get("ruleset_deviations", ())),
            source_kind=value["source_kind"], projection_version=value["projection_version"],
            research_policy=value.get("research_policy", "licensed_only"),
            research_policy_hash=value.get("research_policy_hash", ""),
            policy_version=value.get("policy_version", ""),
            external_license_status=value.get("external_license_status", ""),
            train_eligible_under_policy=eligible,
            created_utc=value.get("created_utc", "1970-01-01T00:00:00Z"),
        )


@dataclass(frozen=True)
class DecisionRecord:
    pov: int
    decision_index: int
    before_event_idx: int
    kyoku_idx: int
    phase: str
    legal_mask: tuple[bool, ...]
    tile_state: dict
    players_public: tuple[dict, ...]
    global_public: dict
    target: dict

    def __post_init__(self):
        if not 0 <= self.pov <= 3:
            raise ShardError(f"pov {self.pov} outside 0..3")
        if self.decision_index < 0 or self.before_event_idx < 1:
            raise ShardError("invalid decision offsets")
        if self.phase not in {phase.value for phase in ActionPhase}:
            raise ShardError(f"unknown phase {self.phase!r}")
        if len(self.legal_mask) != 46 or not all(isinstance(v, bool) for v in self.legal_mask):
            raise ShardError("legal_mask must be 46 bools")
        if len(self.players_public) != 4:
            raise ShardError("players_public must have 4 entries")
        target_keys = set(self.target)
        if target_keys & {"events", "tile_state", "players_public", "global_public",
                          "legal_mask", "before_event_idx", "phase"}:
            raise ShardError("target record contains input fields")
        if set(self.tile_state) & {"expert_action", "rank_final", "censor_reason",
                                   "supervision_valid"}:
            raise ShardError("input record contains target fields")

    def input_payload(self) -> dict:
        return {
            "before_event_idx": self.before_event_idx,
            "kyoku_idx": self.kyoku_idx,
            "phase": self.phase,
            "legal_mask": list(self.legal_mask),
            "tile_state": self.tile_state,
            "players_public": [dict(player) for player in self.players_public],
            "global_public": dict(self.global_public),
        }

    def target_payload(self) -> dict:
        return dict(self.target)

    @property
    def input_sha256(self) -> str:
        return _sha(_canonical(self.input_payload()))

    @property
    def target_sha256(self) -> str:
        return _sha(_canonical(self.target_payload()))


@dataclass(frozen=True)
class GameRecord:
    game_id: str
    component_id: str
    component_sha256: str
    split: str
    source_id: str
    pov_streams: tuple[tuple[dict, ...], ...]
    kyoku_offsets: tuple[tuple[dict, ...], ...]
    decisions: tuple[DecisionRecord, ...]
    quarantine: bool = False
    #: Per-game rights provenance (H-02 split-v2 row).  ``purpose=train`` requires
    #: both fields, not just the shard-level metadata, so a mixed/corrupt shard cannot
    #: silently carry a blocked or val_holdout row into training.
    train_allowed: bool = False
    rights_status: str = "UNKNOWN"

    def __post_init__(self):
        if self.quarantine:
            raise ShardError("quarantined games must never enter a shard")
        if not isinstance(self.train_allowed, bool):
            raise ShardError("game train_allowed must be bool")
        status = str(self.rights_status).upper()
        if status not in {"ALLOWED", "RESTRICTED", "UNKNOWN"}:
            raise ShardError(
                f"game rights_status must be ALLOWED/RESTRICTED/UNKNOWN, got "
                f"{self.rights_status!r}")
        if self.train_allowed and status != "ALLOWED":
            raise ShardError("game train_allowed=True requires rights_status=ALLOWED")
        if len(self.pov_streams) != 4 or len(self.kyoku_offsets) != 4:
            raise ShardError("a game must carry four POV streams/offsets")
        if not self.decisions:
            raise ShardError("a game must carry at least one decision")
        for pov_index, stream in enumerate(self.pov_streams):
            if not stream:
                raise ShardError(f"POV {pov_index} stream is empty")
        for decision in self.decisions:
            if decision.pov < 0 or decision.pov > 3:
                raise ShardError("decision POV outside 0..3")
            if decision.before_event_idx > len(self.pov_streams[decision.pov]):
                raise ShardError("decision cutoff beyond POV stream")

    def decision_count(self) -> int:
        return len(self.decisions)


def make_game_record(raw_events: Sequence[dict], samples, *,
                     game_id: str, component_id: str, component_sha256: str,
                     split: str, source_id: str, train_allowed: bool = False,
                     rights_status: str = "UNKNOWN") -> GameRecord:
    """Build one game row from PR-10 samples; stores only projected POV streams."""
    if not game_id or not component_id:
        raise ShardError("game_id/component_id required")
    if len(component_sha256) != 64:
        raise ShardError("component_sha256 must be 64-char sha256")
    streams = []
    offsets = []
    for seat in range(4):
        projector = pov.POVProjector(seat)
        events = projector.extend(raw_events)
        stream = tuple(event.to_dict() for event in events)
        streams.append(stream)
        current = -1
        start = 0
        rows = []
        for index, event in enumerate(stream):
            if event.get("type") == "start_kyoku":
                if current >= 0:
                    rows.append({"kyoku_idx": current, "start": start, "end": index})
                current += 1
                start = index
        if current >= 0:
            rows.append({"kyoku_idx": current, "start": start, "end": len(stream)})
        offsets.append(tuple(rows))
    decisions = []
    for sample in sorted(samples, key=lambda s: (s.header.pov, s.decision_index)):
        decision_input = sample.input
        if decision_input.before_event_idx > len(streams[sample.header.pov]):
            raise ShardError("sample cutoff beyond projected stream")
        decisions.append(DecisionRecord(
            pov=sample.header.pov,
            decision_index=sample.decision_index,
            before_event_idx=decision_input.before_event_idx,
            kyoku_idx=decision_input.events[-1].kyoku_idx,
            phase=decision_input.phase.value,
            legal_mask=tuple(bool(v) for v in decision_input.legal_mask),
            tile_state=decision_input.tile_state.as_dict(),
            players_public=tuple(p.as_dict() for p in decision_input.players_public),
            global_public=decision_input.global_public.as_dict(),
            target=sample.target.to_dict(),
        ))
    return GameRecord(
        game_id=game_id, component_id=component_id,
        component_sha256=component_sha256, split=split, source_id=source_id,
        pov_streams=tuple(streams), kyoku_offsets=tuple(offsets),
        decisions=tuple(decisions), train_allowed=train_allowed,
        rights_status=rights_status,
    )


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS games (
  game_id TEXT PRIMARY KEY, component_id TEXT NOT NULL, component_sha256 TEXT NOT NULL,
  split TEXT NOT NULL, source_id TEXT NOT NULL, train_allowed INTEGER NOT NULL,
  rights_status TEXT NOT NULL, pov_streams TEXT NOT NULL,
  kyoku_offsets TEXT NOT NULL, decision_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS inputs (
  decision_id TEXT PRIMARY KEY, game_id TEXT NOT NULL, pov INTEGER NOT NULL,
  decision_index INTEGER NOT NULL, before_event_idx INTEGER NOT NULL,
  kyoku_idx INTEGER NOT NULL, phase TEXT NOT NULL, legal_mask TEXT NOT NULL,
  tile_state TEXT NOT NULL, players_public TEXT NOT NULL, global_public TEXT NOT NULL,
  input_sha256 TEXT NOT NULL,
  UNIQUE(game_id, pov, decision_index)
);
CREATE TABLE IF NOT EXISTS targets (
  decision_id TEXT PRIMARY KEY, expert_action INTEGER, supervision_valid INTEGER NOT NULL,
  censor_reason TEXT, rank_final INTEGER, target_sha256 TEXT NOT NULL,
  FOREIGN KEY(decision_id) REFERENCES inputs(decision_id)
);
CREATE INDEX IF NOT EXISTS inputs_game ON inputs(game_id, pov, decision_index);
"""


class ShardWriter:
    """Build one atomic shard; ``resume=True`` appends to an existing shard."""

    def __init__(self, path: str | Path, metadata: ShardMetadata,
                 *, resume: bool = False):
        if not isinstance(metadata, ShardMetadata):
            raise ShardError("metadata must be ShardMetadata")
        self.path = Path(path)
        self.metadata = metadata
        self.tmp_path = Path(str(self.path) + ".tmp")
        self.manifest_path = Path(str(self.path) + ".manifest.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume and self.path.exists() and not self.tmp_path.exists():
            shutil.copy2(self.path, self.tmp_path)
        elif self.path.exists() and not resume and not self.tmp_path.exists():
            raise ShardError(f"shard already exists (use resume=True): {self.path}")
        self.conn = sqlite3.connect(self.tmp_path)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_CREATE_SQL)
        row = self.conn.execute("SELECT value FROM meta WHERE key='metadata'").fetchone()
        if row is None:
            self.conn.execute("INSERT INTO meta(key,value) VALUES('metadata',?)",
                              (_canonical(metadata.as_dict()),))
        else:
            existing = ShardMetadata.from_dict(json.loads(row[0]))
            if existing != metadata:
                # ``created_utc`` is the only allowed difference on resume.
                if replace(existing, created_utc=metadata.created_utc) == metadata:
                    self.metadata = existing
                else:
                    raise ShardError("resume metadata mismatch")
        self.conn.commit()
        self._closed = False

    @property
    def game_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM games").fetchone()[0])

    @property
    def decision_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM inputs").fetchone()[0])

    def _allowed_sources(self) -> set[str]:
        source = self.metadata.source_id
        if source.startswith("mixed:"):
            return {item for item in source[len("mixed:"):].split("+") if item}
        return {source}

    def add_game(self, game: GameRecord) -> None:
        if not isinstance(game, GameRecord):
            raise ShardError("game must be GameRecord")
        if self.metadata.purpose == "train":
            if self.metadata.research_policy == "licensed_only" and (
                    not self.metadata.train_allowed or self.metadata.rights_status.upper() != "ALLOWED"):
                raise ShardRightsError("cannot write licensed training shard from blocked source")
            eligible = self.metadata.train_eligible_under_policy or (
                self.metadata.research_policy == "licensed_only"
                and self.metadata.train_allowed
                and self.metadata.rights_status.upper() == "ALLOWED")
            if not eligible:
                raise ShardRightsError(
                    "cannot write a training shard that is not eligible under policy")
            # Per-game gate: metadata alone cannot smuggle a val/blocked row into train.
            if game.split != "train":
                raise ShardRightsError(
                    f"training shard refuses non-train split {game.split!r} "
                    f"for {game.game_id}")
            if self.metadata.research_policy == "licensed_only" and (
                    not game.train_allowed or game.rights_status.upper() != "ALLOWED"):
                raise ShardRightsError(
                    f"licensed training shard refuses per-game rights "
                    f"train_allowed={game.train_allowed}, "
                    f"rights_status={game.rights_status!r} for {game.game_id}")
            if game.source_id not in self._allowed_sources():
                raise ShardRightsError(
                    f"training shard refuses source {game.source_id!r} not covered by "
                    f"metadata source {self.metadata.source_id!r}")
        if game.quarantine:
            raise ShardError("quarantined game rejected")
        with self.conn:
            # Idempotent replace: targets first, then inputs, then the game row.
            ids = [row[0] for row in self.conn.execute(
                "SELECT decision_id FROM inputs WHERE game_id=?", (game.game_id,))]
            for decision_id in ids:
                self.conn.execute("DELETE FROM targets WHERE decision_id=?", (decision_id,))
            self.conn.execute("DELETE FROM inputs WHERE game_id=?", (game.game_id,))
            self.conn.execute("DELETE FROM games WHERE game_id=?", (game.game_id,))
            self.conn.execute(
                "INSERT INTO games(game_id,component_id,component_sha256,split,source_id,"
                "train_allowed,rights_status,pov_streams,kyoku_offsets,decision_count)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (game.game_id, game.component_id, game.component_sha256, game.split,
                 game.source_id, 1 if game.train_allowed else 0,
                 str(game.rights_status).upper(),
                 pack_json([list(stream) for stream in game.pov_streams]),
                 pack_json([list(rows) for rows in game.kyoku_offsets]),
                 len(game.decisions)))
            for decision in game.decisions:
                decision_id = f"{game.game_id}:{decision.pov}:{decision.decision_index}"
                payload = decision.input_payload()
                self.conn.execute(
                    "INSERT INTO inputs(decision_id,game_id,pov,decision_index,"
                    "before_event_idx,kyoku_idx,phase,legal_mask,tile_state,players_public,"
                    "global_public,input_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (decision_id, game.game_id, decision.pov, decision.decision_index,
                     decision.before_event_idx, decision.kyoku_idx, decision.phase,
                     pack_json(list(decision.legal_mask)),
                     pack_json(decision.tile_state),
                     pack_json(list(decision.players_public)),
                     pack_json(decision.global_public), decision.input_sha256))
                self.conn.execute(
                    "INSERT INTO targets(decision_id,expert_action,supervision_valid,"
                    "censor_reason,rank_final,target_sha256) VALUES(?,?,?,?,?,?)",
                    (decision_id, decision.target.get("expert_action"),
                     1 if decision.target.get("supervision_valid") else 0,
                     decision.target.get("censor_reason"),
                     decision.target.get("rank_final"), decision.target_sha256))

    def finalize(self) -> dict:
        if self._closed:
            raise ShardError("writer already closed")
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('game_count',?)",
                              (str(self.game_count),))
            self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('decision_count',?)",
                              (str(self.decision_count),))
        self.conn.commit()
        game_count = self.game_count
        decision_count = self.decision_count
        self.conn.close()
        shard_sha = _file_sha(self.tmp_path)
        content_sha = content_sha256(self.tmp_path, self.metadata)
        manifest = {
            "format_version": FORMAT_VERSION,
            "metadata": self.metadata.as_dict(),
            "game_count": game_count,
            "decision_count": decision_count,
            "shard_sha256": shard_sha,
            "content_sha256": content_sha,
            "files": {self.path.name: shard_sha},
            "schema_hash": SCHEMA_HASH,
            "action_map_hash": ACTION_HASH,
            "normalizer_hash": NORMALIZER_HASH,
        }
        tmp_manifest = Path(str(self.manifest_path) + ".tmp")
        tmp_manifest.write_text(_canonical(manifest) + "\n", encoding="utf-8")
        # Publish the DB first, then its manifest: a reader can never see a manifest
        # newer than the final DB.  fsync both files and the directory before rename.
        _fsync_file(self.tmp_path)
        os.replace(self.tmp_path, self.path)
        _fsync_dir(self.path.parent)
        _fsync_file(tmp_manifest)
        os.replace(tmp_manifest, self.manifest_path)
        _fsync_dir(self.manifest_path.parent)
        self._closed = True
        return manifest

    def close(self) -> None:
        if not self._closed:
            self.conn.close()
            self._closed = True


def read_manifest(path: str | Path) -> dict:
    manifest_path = Path(str(path) + ".manifest.json")
    if not manifest_path.exists():
        raise ShardError(f"missing manifest {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def sample_hashes(path: str | Path, fractions=(0.01, 0.10, 1.0)) -> dict:
    """Deterministic 1/10/100% decision-sample hashes (streaming, O(1) memory).

    Exact compatibility with the old in-memory definition: the hash is
    ``sha256(canonical_json([selected rows]))``; rows are selected in
    ``decision_id`` order and only the first ``round(total*fraction)`` rows count.
    """
    conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM inputs i JOIN targets t USING(decision_id)"
        ).fetchone()[0]
        counts = {
            fraction: (max(1, int(round(total * fraction))) if total else 0)
            for fraction in fractions
        }
        digests = {fraction: hashlib.sha256() for fraction in fractions}
        started = {fraction: False for fraction in fractions}
        for fraction in fractions:
            digests[fraction].update(b"[")
        query = (
            "SELECT i.decision_id, i.input_sha256, t.target_sha256 "
            "FROM inputs i JOIN targets t USING(decision_id) ORDER BY i.decision_id"
        )
        for index, row in enumerate(conn.execute(query)):
            for fraction in fractions:
                if index >= counts[fraction]:
                    continue
                if started[fraction]:
                    digests[fraction].update(b",")
                else:
                    started[fraction] = True
                digests[fraction].update(_canonical({
                    "decision_id": row[0],
                    "input_sha256": row[1],
                    "target_sha256": row[2],
                }).encode("utf-8"))
        out = {}
        for fraction in fractions:
            digests[fraction].update(b"]")
            out[f"{fraction:g}"] = digests[fraction].hexdigest()
        return out
    finally:
        conn.close()


__all__ = [
    "ACTION_HASH", "DecisionRecord", "FORMAT_VERSION", "FROZEN_SCHEMA", "GameRecord",
    "INPUT_COLUMNS", "NORMALIZER_HASH", "PURPOSES", "SCHEMA_HASH", "ShardError",
    "ShardMetadata", "ShardRightsError", "ShardWriter", "TARGET_COLUMNS",
    "content_sha256", "file_sha256", "make_game_record", "pack_json", "read_manifest",
    "sample_hashes", "unpack_json",
]
