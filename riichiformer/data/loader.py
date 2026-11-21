#!/usr/bin/env python3
"""PR-12 streaming shard reader: ModelBatch rows + separate targets.

The loader imports no target/label class.  Targets stay plain dictionaries from the
``targets`` table; only the forward side is converted to PR-13 tensors.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..models.batch import ModelBatch
from ..schemas.action_v3 import ActionPhase
from ..schemas.decision_v3 import (
    DecisionInput,
    GlobalPublicState,
    MeldKind,
    PlayerPublicState,
    PublicMeld,
    RiichiState,
    RulesetDeviation,
    RulesetId,
    SourceKind,
    VisibleTileState,
)
from ..schemas.event_v3 import TileRef, event_from_dict
from .encode import EncodedBatch, StreamMeta, encode_batch, encode_decision
from .shard import (
    ACTION_HASH,
    FROZEN_SCHEMA,
    NORMALIZER_HASH,
    SCHEMA_HASH,
    ShardError,
    ShardMetadata,
    ShardRightsError,
    file_sha256,
    read_manifest,
    unpack_json,
)

PURPOSES = ("train", "audit", "diagnostic")


class ShardLoadError(ShardError):
    """The shard cannot be read under the requested purpose/hash/source."""


def _tile(value: dict) -> TileRef:
    return TileRef(int(value["kind"]), value.get("aka"))


def _meld(value: dict) -> PublicMeld:
    return PublicMeld(
        meld_kind=MeldKind(value["meld_kind"]),
        tiles=tuple(_tile(tile) for tile in value["tiles"]),
        from_rel=int(value["from_rel"]),
    )


def _tile_state(value: dict) -> VisibleTileState:
    return VisibleTileState(
        own_counts=tuple(int(v) for v in value["own_counts"]),
        own_aka=tuple(int(v) for v in value["own_aka"]),
        public_counts=tuple(int(v) for v in value["public_counts"]),
        public_aka=tuple(int(v) for v in value["public_aka"]),
        public_counts_by_zone=tuple(tuple(int(v) for v in zone)
                                    for zone in value["public_counts_by_zone"]),
        dora_indicators=tuple(_tile(tile) for tile in value["dora_indicators"]),
        ruleset_id=RulesetId(value["ruleset_id"]),
    )


def _player(value: dict) -> PlayerPublicState:
    return PlayerPublicState(
        seat_rel=int(value["seat_rel"]),
        score=int(value["score"]),
        wind=_tile(value["wind"]),
        is_oya=bool(value["is_oya"]),
        riichi=RiichiState(value["riichi"]),
        river_count=int(value["river_count"]),
        melds=tuple(_meld(meld) for meld in value["melds"]),
    )


def _global(value: dict) -> GlobalPublicState:
    return GlobalPublicState(
        bakaze=_tile(value["bakaze"]),
        kyoku=int(value["kyoku"]),
        honba=int(value["honba"]),
        kyotaku=int(value["kyotaku"]),
        tiles_left=int(value["tiles_left"]),
        self_wind=_tile(value["self_wind"]),
        oya_rel=int(value["oya_rel"]),
        ruleset_id=RulesetId(value["ruleset_id"]),
        ruleset_deviation=tuple(RulesetDeviation(code)
                                for code in value["ruleset_deviation"]),
        source_kind=SourceKind(value["source_kind"]),
    )


def _decision_input(stream: tuple[dict, ...], payload: dict) -> DecisionInput:
    before = int(payload["before_event_idx"])
    if before < 1 or before > len(stream):
        raise ShardLoadError(f"bad decision cutoff {before} / stream {len(stream)}")
    events = tuple(event_from_dict(event) for event in stream[:before])
    if events[-1].seq_idx != before - 1:
        raise ShardLoadError("stored POV stream is not contiguous at the cutoff")
    return DecisionInput(
        events=events,
        tile_state=_tile_state(payload["tile_state"]),
        players_public=tuple(_player(player) for player in payload["players_public"]),
        global_public=_global(payload["global_public"]),
        legal_mask=tuple(bool(v) for v in payload["legal_mask"]),
        phase=ActionPhase(payload["phase"]),
        before_event_idx=before,
        schema_version=payload.get("schema_version", FROZEN_SCHEMA["schema_version"]),
    )


@dataclass(frozen=True)
class ShardBatch:
    """One deterministic batch: forward ModelBatch + plain target dicts."""

    batch: ModelBatch
    targets: tuple[dict, ...]
    metas: tuple[StreamMeta, ...]
    lengths: tuple[int, ...]
    game_ids: tuple[str, ...]
    povs: tuple[int, ...]
    decision_indices: tuple[int, ...]

    def __post_init__(self):
        n = len(self.targets)
        for name in ("metas", "lengths", "game_ids", "povs", "decision_indices"):
            if len(getattr(self, name)) != n:
                raise ShardLoadError(f"{name} length != targets length")
        if self.batch.n_decisions != n:
            raise ShardLoadError("ModelBatch decisions != target rows")


class ShardReader:
    """Read a finalized shard; validate metadata before yielding anything."""

    def __init__(self, path: str | Path, *, purpose: str = "train",
                 expected_hashes: dict | None = None,
                 expected_source_id: str | None = None,
                 expected_component_sha256: dict[str, str] | None = None,
                 verify_file_hash: bool = True):
        if purpose not in PURPOSES:
            raise ShardLoadError(f"purpose must be one of {PURPOSES}")
        self.path = Path(path)
        self.purpose = purpose
        if not self.path.exists():
            raise ShardLoadError(f"shard not found: {self.path}")
        self.manifest = read_manifest(self.path)
        self.metadata = ShardMetadata.from_dict(self.manifest["metadata"])
        if verify_file_hash:
            # Chunked, never path.read_bytes(): a 100k-game shard can be ~90 GB.
            digest = file_sha256(self.path)
            if digest != self.manifest.get("shard_sha256"):
                raise ShardLoadError("shard file hash does not match manifest")
        if expected_hashes is not None:
            required = {"schema_hash", "action_map_hash", "normalizer_hash"}
            optional = {"canonical_manifest_sha256", "canonical_index_sha256",
                        "split_manifest_sha256", "schema_version"}
            if not required <= set(expected_hashes):
                raise ShardLoadError(
                    "expected_hashes must contain schema_hash/action_map_hash/"
                    "normalizer_hash")
            if set(expected_hashes) - required - optional:
                raise ShardLoadError(
                    f"unsupported expected_hashes keys: "
                    f"{sorted(set(expected_hashes) - required - optional)}")
            for key, expected in expected_hashes.items():
                if expected != getattr(self.metadata, key):
                    raise ShardLoadError(f"{key} mismatch")
        if expected_source_id is not None and self.metadata.source_id != expected_source_id:
            raise ShardLoadError(
                f"source_id {self.metadata.source_id!r} != expected {expected_source_id!r}")
        if expected_component_sha256 is not None:
            if not isinstance(expected_component_sha256, dict):
                raise ShardLoadError("expected_component_sha256 must be a dict")
            for game_id, digest in expected_component_sha256.items():
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ShardLoadError(
                        f"expected component sha for {game_id!r} must be 64-char sha256")
        self._expected_component_sha256 = dict(expected_component_sha256 or {})
        if self.metadata.schema_hash != SCHEMA_HASH \
                or self.metadata.action_map_hash != ACTION_HASH \
                or self.metadata.normalizer_hash != NORMALIZER_HASH:
            raise ShardLoadError("shard schema/action/normalizer hash mismatch")
        if purpose == "train":
            if self.metadata.purpose != "train":
                raise ShardRightsError(
                    f"training loader refuses purpose={self.metadata.purpose!r} shard")
            if not self.metadata.train_allowed or self.metadata.rights_status.upper() != "ALLOWED":
                raise ShardRightsError("training loader refuses blocked source shard")
        self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._conn.close()
            self._closed = True

    def _stream_meta(self, pov_index: int) -> StreamMeta:
        return StreamMeta(
            pov=pov_index,
            projection_version=self.metadata.projection_version,
            schema_version=self.metadata.schema_version,
            schema_hash=self.metadata.schema_hash,
            ruleset_id=self.metadata.ruleset_id,
            source_kind=self.metadata.source_kind,
        )

    def _allowed_sources(self) -> set[str]:
        source = self.metadata.source_id
        if source.startswith("mixed:"):
            return {item for item in source[len("mixed:"):].split("+") if item}
        return {source}

    def _assert_train_game(self, game: dict) -> None:
        if game["split"] != "train":
            raise ShardRightsError(
                f"training loader refuses game {game['game_id']} from split "
                f"{game['split']!r}")
        if not game["train_allowed"]:
            raise ShardRightsError(
                f"training loader refuses game {game['game_id']} with "
                "per-game train_allowed=False")
        if str(game["rights_status"]).upper() != "ALLOWED":
            raise ShardRightsError(
                f"training loader refuses game {game['game_id']} with rights_status "
                f"{game['rights_status']!r}")
        if game["source_id"] not in self._allowed_sources():
            raise ShardRightsError(
                f"training loader refuses game {game['game_id']} source "
                f"{game['source_id']!r} not covered by metadata source "
                f"{self.metadata.source_id!r}")

    def _validate_expected_component(self, game: dict) -> None:
        expected = self._expected_component_sha256.get(game["game_id"])
        if expected is not None and expected != game["component_sha256"]:
            raise ShardLoadError(
                f"component sha mismatch for {game['game_id']}: "
                f"{game['component_sha256']} != expected {expected}")

    def games(self) -> Iterator[dict]:
        # Stream game rows: a 100k-game shard is ~90 GB, so fetchall() would defeat
        # the chunked-hash fix and load every compressed POV stream into memory.
        rows = self._conn.execute("SELECT * FROM games ORDER BY game_id")
        for row in rows:
            game = {
                "game_id": row["game_id"], "component_id": row["component_id"],
                "component_sha256": row["component_sha256"], "split": row["split"],
                "source_id": row["source_id"],
                "train_allowed": bool(row["train_allowed"]),
                "rights_status": str(row["rights_status"]).upper(),
                "pov_streams": unpack_json(row["pov_streams"]),
                "kyoku_offsets": unpack_json(row["kyoku_offsets"]),
                "decision_count": row["decision_count"],
            }
            self._validate_expected_component(game)
            if self.purpose == "train":
                self._assert_train_game(game)
            yield game

    def iter_decisions(self):
        """Yield ``(input, target_dict, meta, game_id, pov, decision_index)``."""
        for game in self.games():
            streams = tuple(tuple(stream) for stream in game["pov_streams"])
            inputs = self._conn.execute(
                "SELECT * FROM inputs WHERE game_id=? ORDER BY pov,decision_index",
                (game["game_id"],)).fetchall()
            for row in inputs:
                payload = {
                    "before_event_idx": row["before_event_idx"],
                    "kyoku_idx": row["kyoku_idx"],
                    "phase": row["phase"],
                    "legal_mask": unpack_json(row["legal_mask"]),
                    "tile_state": unpack_json(row["tile_state"]),
                    "players_public": unpack_json(row["players_public"]),
                    "global_public": unpack_json(row["global_public"]),
                }
                target = self._conn.execute(
                    "SELECT * FROM targets WHERE decision_id=?",
                    (row["decision_id"],)).fetchone()
                if target is None:
                    raise ShardLoadError(f"missing target {row['decision_id']}")
                target_dict = {
                    "expert_action": target["expert_action"],
                    "supervision_valid": bool(target["supervision_valid"]),
                    "censor_reason": target["censor_reason"],
                    "rank_final": target["rank_final"],
                }
                if bool(target["supervision_valid"]) and target_dict["censor_reason"] is not None:
                    raise ShardLoadError("supervised target has censor reason")
                if not bool(target["supervision_valid"]) and target_dict["censor_reason"] is None:
                    raise ShardLoadError("censored target has no reason")
                decision = _decision_input(streams[row["pov"]], payload)
                yield (decision, target_dict, self._stream_meta(row["pov"]),
                       game["game_id"], int(row["pov"]), int(row["decision_index"]))

    def iter_batches(self, batch_size: int = 8,
                     bucket_by_length: bool = True) -> Iterator[ShardBatch]:
        """Deterministic batches; sorts each buffer by sequence length (dynamic padding)."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ShardLoadError("batch_size must be a positive int")
        buffer = []
        for item in self.iter_decisions():
            buffer.append(item)
            if len(buffer) >= batch_size:
                yield self._make_batch(buffer, bucket_by_length)
                buffer = []
        if buffer:
            yield self._make_batch(buffer, bucket_by_length)

    def _make_batch(self, items, bucket_by_length: bool) -> ShardBatch:
        if bucket_by_length:
            items = sorted(items, key=lambda item: (
                len(item[0].events), item[3], item[4], item[5]))
        decisions = [item[0] for item in items]
        targets = tuple(item[1] for item in items)
        metas = tuple(item[2] for item in items)
        encoded: EncodedBatch = encode_batch(decisions, metas)
        return ShardBatch(
            batch=encoded.to_model_batch(), targets=targets, metas=metas,
            lengths=encoded.lengths,
            game_ids=tuple(item[3] for item in items),
            povs=tuple(item[4] for item in items),
            decision_indices=tuple(item[5] for item in items),
        )


__all__ = ["ShardBatch", "ShardLoadError", "ShardReader"]
