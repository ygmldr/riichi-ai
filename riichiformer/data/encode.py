#!/usr/bin/env python3
"""PR-11 single pure encoder: DecisionInput + POV provenance -> ModelBatch contract.

The same function is used offline (PR-12 shards) and online (arena); there is no second
token layout.  It consumes only PR-10's ``DecisionInput`` (visible prefix + mask + state)
and a ``StreamMeta``/PR-09 ``ProjectedStreamHeader``.  Targets, labels, rank, oracle and
future events have no parameter and no field.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from ..models.batch import PHASE_INDEX, ModelBatch
from ..models.state_encoder import (
    STATE_FEATURE_DIM,
    StateSnapshot,
    state_features_from_snapshot,
)
from ..schemas.action_v3 import (
    ACTION_SPACE,
    SCHEMA_VERSION,
    UNKNOWN_TILE_KIND,
    ActionPhase,
    canonical_json,
    validate_mask_for_phase,
)
from ..schemas.decision_v3 import DecisionInput, RiichiState, RulesetId, SourceKind
from ..schemas.event_v3 import (
    EVENT_TYPE_IDS,
    AnkanEvent,
    ChiEvent,
    DahaiEvent,
    DaiminkanEvent,
    DoraEvent,
    EVENT_CLASSES,
    KakanEvent,
    PonEvent,
    StartKyokuEvent,
    TsumoEvent,
    UNKNOWN_TILE,
    TileRef,
)
from ..schemas.schema_v3 import load_schema
from .pov import POV_PROJECTION_VERSION, ProjectedStreamHeader

#: Frozen schema hash; a caller-provided hash must equal this exactly.
_FROZEN_SCHEMA = load_schema()
SCHEMA_HASH: str = _FROZEN_SCHEMA["schema_hash"]

#: Event embedding IDs from PR-13 (PAD = 16, actor PAD = 5, phase PAD = 3).
EVENT_TYPE_PAD = 16
ACTOR_REL_PAD = 5
TILE_PAD = UNKNOWN_TILE_KIND
PHASE_PAD = 3
KYOKU_POS_VOCAB = 512
MAX_EVENTS_DEFAULT = 4096

RIICHI_INDEX = {
    RiichiState.NONE: 0,
    RiichiState.DECLARED: 1,
    RiichiState.ACCEPTED: 2,
}

#: Tensor fields copied from EncodedDecision into a ModelBatch row.
EVENT_TENSOR_FIELDS = (
    "event_type_id", "event_actor_rel", "event_tile_kind", "event_aka",
    "event_tsumogiri", "event_phase", "event_kyoku_pos",
)


class EncoderError(ValueError):
    """Encoder input/provenance violates the frozen contract (fail closed)."""


@dataclass(frozen=True)
class StreamMeta:
    """Provenance carried with an encoded stream/shard sample."""

    pov: int
    projection_version: str
    schema_version: str
    schema_hash: str
    ruleset_id: str
    source_kind: str

    def __post_init__(self):
        if isinstance(self.pov, bool) or not isinstance(self.pov, int) \
                or not 0 <= self.pov <= 3:
            raise EncoderError(f"pov must be int 0..3, got {self.pov!r}")
        if self.projection_version != POV_PROJECTION_VERSION:
            raise EncoderError(
                f"projection_version {self.projection_version!r} != "
                f"{POV_PROJECTION_VERSION!r}")
        if self.schema_version != SCHEMA_VERSION:
            raise EncoderError(
                f"schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}")
        if self.schema_hash != SCHEMA_HASH:
            raise EncoderError(
                f"schema_hash {self.schema_hash!r} != frozen {SCHEMA_HASH!r}")
        if self.ruleset_id not in {value.value for value in RulesetId}:
            raise EncoderError(f"unknown ruleset_id {self.ruleset_id!r}")
        if self.source_kind not in {value.value for value in SourceKind}:
            raise EncoderError(f"unknown source_kind {self.source_kind!r}")

    @classmethod
    def from_header(cls, header: ProjectedStreamHeader) -> "StreamMeta":
        if not isinstance(header, ProjectedStreamHeader):
            raise EncoderError("expected ProjectedStreamHeader")
        return cls(
            pov=header.pov,
            projection_version=header.projection_version,
            schema_version=header.schema_version,
            schema_hash=SCHEMA_HASH,
            ruleset_id=header.ruleset_id,
            source_kind=header.source_kind,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping) -> "StreamMeta":
        if not isinstance(mapping, Mapping):
            raise EncoderError("stream_meta must be StreamMeta/header/mapping")
        required = ("pov", "projection_version", "schema_version",
                    "ruleset_id", "source_kind")
        missing = [key for key in required if key not in mapping]
        if missing:
            raise EncoderError(f"stream_meta missing keys {missing}")
        schema_hash = mapping.get("schema_hash", SCHEMA_HASH)
        if schema_hash != SCHEMA_HASH:
            raise EncoderError(
                f"schema_hash {schema_hash!r} != frozen {SCHEMA_HASH!r}")
        return cls(
            pov=mapping["pov"],
            projection_version=mapping["projection_version"],
            schema_version=mapping["schema_version"],
            schema_hash=schema_hash,
            ruleset_id=mapping["ruleset_id"],
            source_kind=mapping["source_kind"],
        )

    @classmethod
    def coerce(cls, value) -> "StreamMeta":
        if isinstance(value, cls):
            return value
        if isinstance(value, ProjectedStreamHeader):
            return cls.from_header(value)
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        raise EncoderError(
            f"stream_meta must be StreamMeta/ProjectedStreamHeader/mapping, "
            f"got {type(value).__name__}")

    def as_dict(self) -> dict:
        return {
            "pov": self.pov,
            "projection_version": self.projection_version,
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "ruleset_id": self.ruleset_id,
            "source_kind": self.source_kind,
        }


def _event_tile(event) -> TileRef:
    if isinstance(event, (TsumoEvent, DahaiEvent, ChiEvent, PonEvent,
                          DaiminkanEvent, KakanEvent)):
        return event.tile
    if isinstance(event, AnkanEvent):
        if event.actor_rel == 0 and event.consumed and not event.consumed[0].is_unknown:
            return event.consumed[0]
        return UNKNOWN_TILE
    if isinstance(event, (StartKyokuEvent, DoraEvent)):
        return event.dora_marker
    return UNKNOWN_TILE


def _check_input(decision: DecisionInput, meta: StreamMeta, max_events: int) -> tuple:
    if not isinstance(decision, DecisionInput):
        raise EncoderError(f"expected DecisionInput, got {type(decision).__name__}")
    if decision.schema_version != SCHEMA_VERSION:
        raise EncoderError(
            f"DecisionInput.schema_version {decision.schema_version!r} != "
            f"{SCHEMA_VERSION!r}")
    if decision.before_event_idx != len(decision.events):
        raise EncoderError(
            "DecisionInput must be the exact visible prefix: "
            f"before_event_idx={decision.before_event_idx}, events={len(decision.events)}")
    if not decision.events:
        raise EncoderError("decision has an empty visible prefix")
    if len(decision.events) > max_events:
        raise EncoderError(
            f"sequence length {len(decision.events)} > max_events {max_events}; "
            "refusing silent truncation")
    for index, event in enumerate(decision.events):
        if type(event) not in EVENT_CLASSES.values():
            raise EncoderError(f"event[{index}] is not an exact registered event class")
        if event.seq_idx != index:
            raise EncoderError(
                f"event[{index}].seq_idx={event.seq_idx}; visible prefix must be "
                "contiguous from 0")
    if not isinstance(decision.phase, ActionPhase):
        raise EncoderError("decision.phase must be ActionPhase")
    if len(decision.legal_mask) != ACTION_SPACE:
        raise EncoderError(f"legal_mask width {len(decision.legal_mask)} != 46")
    try:
        validate_mask_for_phase(decision.legal_mask, decision.phase)
    except Exception as exc:  # noqa: BLE001
        raise EncoderError(f"legal_mask/phase invalid: {exc}") from exc
    return decision.events


def _state_snapshot(decision: DecisionInput) -> StateSnapshot:
    tile_state = decision.tile_state
    players = decision.players_public
    public = decision.global_public
    dora_counts = [0] * UNKNOWN_TILE_KIND
    for marker in tile_state.dora_indicators:
        if marker.is_unknown:
            raise EncoderError("dora indicator cannot be UNKNOWN")
        dora_counts[marker.kind] += 1
    phase_index = PHASE_INDEX[decision.phase]
    return StateSnapshot(
        own_counts=tuple(tile_state.own_counts),
        public_counts=tuple(tile_state.public_counts),
        own_aka=tuple(tile_state.own_aka),
        public_aka=tuple(tile_state.public_aka),
        player_scores=tuple(player.score for player in players),
        player_riichi=tuple(RIICHI_INDEX[player.riichi] for player in players),
        player_river_counts=tuple(player.river_count for player in players),
        player_meld_counts=tuple(len(player.melds) for player in players),
        player_is_oya=tuple(player.is_oya for player in players),
        player_winds=tuple(player.wind.kind for player in players),
        bakaze_kind=public.bakaze.kind,
        kyoku=public.kyoku,
        honba=public.honba,
        kyotaku=public.kyotaku,
        tiles_left=public.tiles_left,
        oya_rel=public.oya_rel,
        dora_indicator_counts=tuple(dora_counts),
        phase=phase_index,
    )


@dataclass(frozen=True)
class EncodedDecision:
    """One decision's exact PR-13 row (no batch padding)."""

    event_type_id: Tensor
    event_actor_rel: Tensor
    event_tile_kind: Tensor
    event_aka: Tensor
    event_tsumogiri: Tensor
    event_phase: Tensor
    event_kyoku_pos: Tensor
    padding_mask: Tensor
    state_features: Tensor
    legal_mask: Tensor
    decision_time_index: int
    phase_index: int
    meta: StreamMeta

    @property
    def event_count(self) -> int:
        return int(self.event_type_id.numel())

    def to_model_batch(self) -> ModelBatch:
        return ModelBatch(
            **{name: getattr(self, name).unsqueeze(0) for name in EVENT_TENSOR_FIELDS},
            padding_mask=self.padding_mask.unsqueeze(0),
            state_features=self.state_features.unsqueeze(0),
            decision_batch_index=torch.zeros(1, dtype=torch.long),
            decision_time_index=torch.tensor([self.decision_time_index], dtype=torch.long),
            legal_mask=self.legal_mask.unsqueeze(0),
            decision_phase=torch.tensor([self.phase_index], dtype=torch.long),
        )


@dataclass(frozen=True)
class EncodedBatch:
    """Padded batch of decisions; one row per decision's visible prefix."""

    event_type_id: Tensor
    event_actor_rel: Tensor
    event_tile_kind: Tensor
    event_aka: Tensor
    event_tsumogiri: Tensor
    event_phase: Tensor
    event_kyoku_pos: Tensor
    padding_mask: Tensor
    state_features: Tensor
    decision_batch_index: Tensor
    decision_time_index: Tensor
    legal_mask: Tensor
    decision_phase: Tensor
    metas: tuple[StreamMeta, ...]
    lengths: tuple[int, ...]

    def to_model_batch(self) -> ModelBatch:
        return ModelBatch(
            **{name: getattr(self, name) for name in EVENT_TENSOR_FIELDS},
            padding_mask=self.padding_mask,
            state_features=self.state_features,
            decision_batch_index=self.decision_batch_index,
            decision_time_index=self.decision_time_index,
            legal_mask=self.legal_mask,
            decision_phase=self.decision_phase,
        )


def _pad_tensor(rows: Sequence[Tensor], fill: int) -> Tensor:
    max_len = max(int(row.numel()) for row in rows)
    out = torch.full((len(rows), max_len), fill, dtype=torch.long)
    for index, row in enumerate(rows):
        out[index, :row.numel()] = row
    return out


def encode_decision(decision: DecisionInput, stream_meta,
                    max_events: int = MAX_EVENTS_DEFAULT) -> EncodedDecision:
    """Pure mapping from one visible prefix to one PR-13 row."""
    if not isinstance(max_events, int) or isinstance(max_events, bool) or max_events <= 0:
        raise EncoderError(f"max_events must be a positive int, got {max_events!r}")
    meta = StreamMeta.coerce(stream_meta)
    events = _check_input(decision, meta, max_events)
    length = len(events)
    event_type = torch.empty(length, dtype=torch.long)
    actor = torch.empty(length, dtype=torch.long)
    tile_kind = torch.empty(length, dtype=torch.long)
    aka = torch.empty(length, dtype=torch.long)
    tsumogiri = torch.empty(length, dtype=torch.long)
    event_phase = torch.zeros(length, dtype=torch.long)
    kyoku_pos = torch.empty(length, dtype=torch.long)

    position = 0
    for index, event in enumerate(events):
        if isinstance(event, StartKyokuEvent):
            position = 0
        if position >= KYOKU_POS_VOCAB:
            raise EncoderError(
                f"kyoku_pos {position} >= {KYOKU_POS_VOCAB} at event {index}; "
                "refusing silent clamp")
        event_type[index] = EVENT_TYPE_IDS[event.EVENT_TYPE]
        actor[index] = int(getattr(event, "actor_rel", 4))
        if not 0 <= int(actor[index]) <= 4:
            raise EncoderError(f"actor_rel {int(actor[index])} outside 0..4")
        tile = _event_tile(event)
        if not isinstance(tile, TileRef):
            raise EncoderError(f"event {index} tile is not TileRef")
        tile_kind[index] = int(tile.kind)
        if tile.is_unknown:
            aka[index] = 0
        else:
            if tile.is_aka and tile.aka not in (0, 1, 2):
                raise EncoderError(f"event {index} invalid aka colour {tile.aka!r}")
            # PR-13 aka ids are 1..3 for 5m/5p/5s (0 = normal/none).
            aka[index] = (1 + int(tile.aka)) if tile.is_aka else 0
        tsumogiri[index] = int(bool(event.tsumogiri)) if isinstance(event, DahaiEvent) else 0
        kyoku_pos[index] = position
        position += 1

    phase_index = PHASE_INDEX[decision.phase]
    event_phase[-1] = phase_index
    state_features = state_features_from_snapshot(_state_snapshot(decision))
    if int(state_features.numel()) != STATE_FEATURE_DIM:
        raise EncoderError(
            f"state feature width {int(state_features.numel())} != {STATE_FEATURE_DIM}")
    legal_mask = torch.tensor(tuple(bool(value) for value in decision.legal_mask),
                              dtype=torch.bool)
    return EncodedDecision(
        event_type_id=event_type,
        event_actor_rel=actor,
        event_tile_kind=tile_kind,
        event_aka=aka,
        event_tsumogiri=tsumogiri,
        event_phase=event_phase,
        event_kyoku_pos=kyoku_pos,
        padding_mask=torch.ones(length, dtype=torch.bool),
        state_features=state_features,
        legal_mask=legal_mask,
        decision_time_index=length - 1,
        phase_index=phase_index,
        meta=meta,
    )


def encode_batch(decisions: Sequence[DecisionInput], stream_metas: Sequence,
                 max_events: int = MAX_EVENTS_DEFAULT) -> EncodedBatch:
    """Encode many decisions with dynamic left-aligned padding (no truncation)."""
    if not decisions:
        raise EncoderError("encode_batch needs at least one decision")
    if len(stream_metas) != len(decisions):
        raise EncoderError("stream_metas length must match decisions length")
    rows = [encode_decision(decision, meta, max_events=max_events)
            for decision, meta in zip(decisions, stream_metas)]
    max_len = max(row.event_count for row in rows)
    batch_size = len(rows)
    tensors = {
        "event_type_id": _pad_tensor([row.event_type_id for row in rows], EVENT_TYPE_PAD),
        "event_actor_rel": _pad_tensor([row.event_actor_rel for row in rows], ACTOR_REL_PAD),
        "event_tile_kind": _pad_tensor([row.event_tile_kind for row in rows], TILE_PAD),
        "event_aka": _pad_tensor([row.event_aka for row in rows], 0),
        "event_tsumogiri": _pad_tensor([row.event_tsumogiri for row in rows], 0),
        "event_phase": _pad_tensor([row.event_phase for row in rows], PHASE_PAD),
        "event_kyoku_pos": _pad_tensor([row.event_kyoku_pos for row in rows], 0),
    }
    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for index, row in enumerate(rows):
        padding_mask[index, :row.event_count] = True
    return EncodedBatch(
        **tensors,
        padding_mask=padding_mask,
        state_features=torch.stack([row.state_features for row in rows], dim=0),
        decision_batch_index=torch.arange(batch_size, dtype=torch.long),
        decision_time_index=torch.tensor([row.decision_time_index for row in rows],
                                         dtype=torch.long),
        legal_mask=torch.stack([row.legal_mask for row in rows], dim=0),
        decision_phase=torch.tensor([row.phase_index for row in rows], dtype=torch.long),
        metas=tuple(row.meta for row in rows),
        lengths=tuple(row.event_count for row in rows),
    )


def encoded_payload(encoded: EncodedDecision | EncodedBatch) -> dict:
    """Deterministic JSON-able array payload used for the offline/online hash."""
    if isinstance(encoded, EncodedDecision):
        fields = {
            name: getattr(encoded, name).detach().cpu().tolist()
            for name in EVENT_TENSOR_FIELDS
        }
        fields["padding_mask"] = encoded.padding_mask.detach().cpu().tolist()
        fields["state_features"] = encoded.state_features.detach().cpu().tolist()
        fields["legal_mask"] = encoded.legal_mask.detach().cpu().tolist()
        fields["decision_time_index"] = encoded.decision_time_index
        fields["phase_index"] = encoded.phase_index
        fields["meta"] = encoded.meta.as_dict()
        return fields
    if isinstance(encoded, EncodedBatch):
        fields = {
            name: getattr(encoded, name).detach().cpu().tolist()
            for name in EVENT_TENSOR_FIELDS
        }
        fields.update({
            "padding_mask": encoded.padding_mask.detach().cpu().tolist(),
            "state_features": encoded.state_features.detach().cpu().tolist(),
            "decision_batch_index": encoded.decision_batch_index.detach().cpu().tolist(),
            "decision_time_index": encoded.decision_time_index.detach().cpu().tolist(),
            "legal_mask": encoded.legal_mask.detach().cpu().tolist(),
            "decision_phase": encoded.decision_phase.detach().cpu().tolist(),
            "metas": [meta.as_dict() for meta in encoded.metas],
            "lengths": list(encoded.lengths),
        })
        return fields
    raise EncoderError(
        f"encoded_payload expects EncodedDecision/EncodedBatch, got {type(encoded).__name__}")


def encoded_hash(encoded: EncodedDecision | EncodedBatch) -> str:
    """SHA-256 over canonical JSON arrays (byte parity witness)."""
    return hashlib.sha256(
        canonical_json(encoded_payload(encoded)).encode("utf-8")).hexdigest()


__all__ = [
    "ACTOR_REL_PAD", "EncodedBatch", "EncodedDecision", "EncoderError",
    "EVENT_TENSOR_FIELDS", "EVENT_TYPE_PAD", "MAX_EVENTS_DEFAULT", "PHASE_PAD",
    "SCHEMA_HASH", "StreamMeta", "TILE_PAD", "encode_batch", "encode_decision",
    "encoded_hash", "encoded_payload",
]
