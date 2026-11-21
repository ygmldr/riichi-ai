"""POV-safe current-kyoku tokenization and structured visible state (PR-03).

The encoder is intentionally a pure function of
:class:`riichiformer.schemas.decision_v3.DecisionInput`.  It never imports a target,
rank, oracle, replay or checkpoint module and it never reads an opponent's hand.
The event stream starts with :data:`~riichiformer.vnext.contracts.BOS_EVENT_ID`
and contains only events from the decision's current kyoku.  One native visible
event maps to exactly one token (the versioned
``current_kyoku_bos/1`` convention); a future tokenizer must update
``prefix_token_offset`` through :class:`VNextDecisionIndex` rather than reusing
the native event count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from torch import Tensor

from ..schemas.action_v3 import (
    ACTION_SPACE,
    AKA_KINDS,
    TILE_KIND_COUNT,
    UNKNOWN_TILE_KIND,
    ActionPhase,
)
from ..schemas.decision_v3 import DecisionInput
from ..schemas.event_v3 import (
    AnkanEvent,
    ChiEvent,
    DahaiEvent,
    DaiminkanEvent,
    DoraEvent,
    EVENT_TYPE_IDS,
    HoraEvent,
    KakanEvent,
    PonEvent,
    ReachAcceptedEvent,
    ReachEvent,
    RyukyokuEvent,
    StartKyokuEvent,
    TileRef,
    TsumoEvent,
    UNKNOWN_TILE,
)
from .contracts import (
    BOS_EVENT_ID,
    CANDIDATE_FEATURE_DIM,
    CANDIDATE_FEATURE_NAMES,
    EVENT_ACTOR_VOCAB,
    EVENT_AKA_VOCAB,
    EVENT_KYOKU_POS_VOCAB,
    EVENT_TILE_VOCAB,
    EVENT_TSUMOGIRI_VOCAB,
    EVENT_TYPE_VOCAB,
    GLOBAL_FEATURE_DIM,
    GLOBAL_FEATURE_NAMES,
    PAD_EVENT_ID,
    TILE_FEATURE_DIM,
    TILE_FEATURE_NAMES,
    VNextDecisionIndex,
    VNextModelBatch,
    VNextVisibleState,
)

PHASE_ORDER: tuple[ActionPhase, ...] = (
    ActionPhase.SELF_TURN,
    ActionPhase.CALL_RESPONSE,
    ActionPhase.KAN_SELECTION,
)
PHASE_INDEX = {phase: index for index, phase in enumerate(PHASE_ORDER)}
AKA_COLOR = {kind: color for color, kind in enumerate(AKA_KINDS)}


class EncoderContractError(ValueError):
    """Visible input cannot be encoded under the frozen VNext contract."""


# ---------------------------------------------------------------------------
# Visible state
# ---------------------------------------------------------------------------


def _tile_feature_matrix(decision: DecisionInput) -> Tensor:
    ts = decision.tile_state
    own = [float(v) for v in ts.own_counts]
    public = [float(v) for v in ts.public_counts]
    own_aka = [0.0] * 3
    public_aka = [0.0] * 3
    for colour, count in enumerate(ts.own_aka):
        own_aka[colour] = float(count)
    for colour, count in enumerate(ts.public_aka):
        public_aka[colour] = float(count)

    # Per-opponent public rivers and post-riichi discards are rebuilt from visible
    # Dahai events; `players_public[*].river_count` alone has no tile identities.
    rivers = [[0] * TILE_KIND_COUNT for _ in range(3)]
    post_riichi = [[0] * TILE_KIND_COUNT for _ in range(3)]
    riichi_started = [False, False, False]
    for event in decision.events:
        if isinstance(event, ReachAcceptedEvent) and 1 <= event.actor_rel <= 3:
            riichi_started[event.actor_rel - 1] = True
        if isinstance(event, DahaiEvent) and 1 <= event.actor_rel <= 3:
            offset = event.actor_rel - 1
            rivers[offset][event.tile.kind] += 1
            if riichi_started[offset]:
                post_riichi[offset][event.tile.kind] += 1

    melds = [[0] * TILE_KIND_COUNT for _ in range(3)]
    for seat_rel in (1, 2, 3):
        for meld in decision.players_public[seat_rel].melds:
            for tile in meld.tiles:
                if not tile.is_unknown:
                    melds[seat_rel - 1][tile.kind] += 1

    indicators = [0] * TILE_KIND_COUNT
    for marker in ts.dora_indicators:
        indicators[marker.kind] += 1

    features = torch.zeros((TILE_KIND_COUNT, TILE_FEATURE_DIM), dtype=torch.float32)
    for kind in range(TILE_KIND_COUNT):
        values = [
            own[kind] / 4.0,
            1.0 if own[kind] >= 1 else 0.0,
            1.0 if own[kind] >= 2 else 0.0,
            1.0 if own[kind] >= 3 else 0.0,
            1.0 if own[kind] >= 4 else 0.0,
            public[kind] / 4.0,
        ]
        values.extend(own_aka)
        values.extend(public_aka)
        values.extend(rivers[seat][kind] / 4.0 for seat in range(3))
        # A tile in an opponent river is genbutsu for that opponent; this is a
        # public-information fact, not an inference from a hidden hand.
        values.extend(rivers[seat][kind] / 4.0 for seat in range(3))
        values.extend(post_riichi[seat][kind] / 4.0 for seat in range(3))
        values.extend(melds[seat][kind] / 4.0 for seat in range(3))
        values.append(indicators[kind] / 4.0)
        values.append(1.0 if indicators[kind] else 0.0)
        if len(values) != TILE_FEATURE_DIM:  # pragma: no cover
            raise AssertionError("tile feature width drifted")
        features[kind] = torch.tensor(values, dtype=torch.float32)
    return features


def _global_feature_vector(decision: DecisionInput) -> Tensor:
    players = decision.players_public
    public = decision.global_public
    scores = [float(player.score) for player in players]
    scale = 25_000.0
    rel = [(score - 25_000.0) / scale for score in scores]
    leader = max(scores)
    last = min(scores)
    diff_self_leader = (scores[0] - leader) / scale
    diff_self_last = (scores[0] - last) / scale
    diff_leader_last = (leader - last) / scale
    bakaze = public.bakaze.kind - 27
    self_wind = public.self_wind.kind - 27
    values = list(rel)
    values.extend([diff_self_leader, diff_self_last, diff_leader_last])
    values.extend([public.kyoku / 4.0, public.honba / 255.0,
                   public.kyotaku / 255.0, public.tiles_left / 70.0])
    values.extend(1.0 if bakaze == index else 0.0 for index in range(4))
    values.extend(1.0 if self_wind == index else 0.0 for index in range(4))
    values.extend(1.0 if public.oya_rel == index else 0.0 for index in range(4))
    values.extend(1.0 if decision.phase is phase else 0.0 for phase in PHASE_ORDER)
    for seat in range(4):
        values.append(1.0 if players[seat].riichi.value != "none" else 0.0)
    for seat in range(4):
        values.append(min(len(players[seat].melds), 4) / 4.0)
    for seat in range(4):
        values.append(min(players[seat].river_count, 70) / 70.0)
    if len(values) != GLOBAL_FEATURE_DIM:  # pragma: no cover
        raise AssertionError(f"global feature width {len(values)} != {GLOBAL_FEATURE_DIM}")
    return torch.tensor(values, dtype=torch.float32)


def _candidate_features() -> Tensor:
    values = torch.zeros((37, CANDIDATE_FEATURE_DIM), dtype=torch.float32)
    for kind in range(34):
        values[kind, 0] = 0.0
        values[kind, 1] = 0.0
        values[kind, 2] = kind / 33.0
        values[kind, 3] = 1.0
    for color, kind in enumerate(AKA_KINDS):
        row = 34 + color
        values[row, 0] = 1.0
        values[row, 1] = color / 2.0
        values[row, 2] = kind / 33.0
        values[row, 3] = 1.0
    return values


_CANDIDATE_FEATURES = _candidate_features()


def encode_visible_state(decision: DecisionInput) -> VNextVisibleState:
    """Build the exact visible state for one decision."""
    if not isinstance(decision, DecisionInput):
        raise EncoderContractError("expected DecisionInput")
    if decision.before_event_idx != len(decision.events):
        raise EncoderContractError("DecisionInput must be the exact visible prefix")
    phase = PHASE_INDEX.get(decision.phase)
    if phase is None:
        raise EncoderContractError("unknown ActionPhase")
    legal = torch.tensor(tuple(bool(v) for v in decision.legal_mask), dtype=torch.bool)
    if legal.numel() != ACTION_SPACE or not bool(legal.any()):
        raise EncoderContractError("legal mask must be 46 bools and non-empty")
    return VNextVisibleState(
        tile_features=_tile_feature_matrix(decision),
        global_features=_global_feature_vector(decision),
        candidate_features=_CANDIDATE_FEATURES.clone(),
        legal_mask=legal,
        phase=phase,
    )


# ---------------------------------------------------------------------------
# Current-kyoku event stream
# ---------------------------------------------------------------------------


def _event_tile(event) -> TileRef:
    if isinstance(event, (TsumoEvent, DahaiEvent, ChiEvent, PonEvent,
                          DaiminkanEvent, KakanEvent)):
        return event.tile
    if isinstance(event, StartKyokuEvent):
        return event.dora_marker
    if isinstance(event, DoraEvent):
        return event.dora_marker
    if isinstance(event, AnkanEvent):
        if event.actor_rel == 0 and event.consumed \
                and not event.consumed[0].is_unknown:
            return event.consumed[0]
        return UNKNOWN_TILE
    return UNKNOWN_TILE


def _event_actor(event) -> int:
    value = getattr(event, "actor_rel", None)
    return 4 if value is None else int(value)


def _event_tsumogiri(event) -> int:
    return 1 if isinstance(event, DahaiEvent) and event.tsumogiri else 0


@dataclass
class VNextEventStream:
    """One POV's BOS-prefixed current-kyoku token stream.

    A stream may own several decisions with non-decreasing token offsets; this is
    how a ``SELF_TURN`` decision and a following ``KAN_SELECTION`` decision share
    one visible prefix.  The stream never appends an event after a decision is
    committed; callers build a fresh stream for a later prefix or use
    :class:`VNextBatchBuilder`.
    """

    match_id: str
    kyoku_id: int
    pov: int
    event_type_id: list[int] = field(default_factory=lambda: [BOS_EVENT_ID])
    event_actor_rel: list[int] = field(default_factory=lambda: [4])
    event_tile_kind: list[int] = field(default_factory=lambda: [UNKNOWN_TILE_KIND])
    event_aka: list[int] = field(default_factory=lambda: [0])
    event_tsumogiri: list[int] = field(default_factory=lambda: [0])
    event_kyoku_pos: list[int] = field(default_factory=lambda: [0])
    decisions: list[tuple[VNextDecisionIndex, VNextVisibleState]] = field(
        default_factory=list)

    def __post_init__(self) -> None:
        if not self.match_id:
            raise EncoderContractError("match_id must be non-empty")
        if self.pov not in (0, 1, 2, 3):
            raise EncoderContractError("pov must be 0..3")
        if self.kyoku_id < 0:
            raise EncoderContractError("kyoku_id must be >= 0")
        if len(self.event_type_id) != len(self.event_actor_rel) \
                or len(self.event_type_id) != len(self.event_tile_kind) \
                or len(self.event_type_id) != len(self.event_aka) \
                or len(self.event_type_id) != len(self.event_tsumogiri) \
                or len(self.event_type_id) != len(self.event_kyoku_pos):
            raise EncoderContractError("event token columns have unequal lengths")

    @property
    def n_tokens(self) -> int:
        return len(self.event_type_id)

    @property
    def last_offset(self) -> int:
        return self.n_tokens - 1

    def append_event(self, event) -> int:
        """Append one current-kyoku native event and return its token offset."""
        if int(event.kyoku_idx) != self.kyoku_id:
            raise EncoderContractError(
                f"event kyoku {event.kyoku_idx} != stream kyoku {self.kyoku_id}")
        pos = len(self.event_type_id) - 1
        if pos >= EVENT_KYOKU_POS_VOCAB:
            raise EncoderContractError("kyoku event position exceeds 511; refuse truncation")
        tile = _event_tile(event)
        if tile.is_unknown and tile.aka is not None:
            raise EncoderContractError("UNKNOWN event tile cannot carry aka")
        self.event_type_id.append(EVENT_TYPE_IDS[event.EVENT_TYPE])
        self.event_actor_rel.append(_event_actor(event))
        self.event_tile_kind.append(tile.kind)
        self.event_aka.append(0 if tile.aka is None else tile.aka + 1)
        self.event_tsumogiri.append(_event_tsumogiri(event))
        self.event_kyoku_pos.append(pos)
        return len(self.event_type_id) - 1

    def add_decision(self, decision: DecisionInput, *, decision_id: str,
                     decision_order: int) -> VNextDecisionIndex:
        if not isinstance(decision, DecisionInput):
            raise EncoderContractError("decision must be DecisionInput")
        state = encode_visible_state(decision)
        current = [event for event in decision.events
                   if int(event.kyoku_idx) == self.kyoku_id]
        # The stream is built from the decision prefix; append only the suffix not
        # already represented.  A changed prefix is a hard failure, never a rewind.
        expected = self.n_tokens - 1
        if len(current) < expected:
            raise EncoderContractError("decision prefix is shorter than the stream")
        if len(current) > expected:
            for event in current[expected:]:
                self.append_event(event)
        offset = len(current)
        if offset >= self.n_tokens:
            raise EncoderContractError("prefix offset is outside the stream")
        index = VNextDecisionIndex(
            match_id=self.match_id, kyoku_id=self.kyoku_id, pov=self.pov,
            decision_id=decision_id, decision_order=decision_order,
            phase=state.phase, prefix_event_count=len(current),
            prefix_token_offset=offset,
        )
        self.decisions.append((index, state))
        return index

    def to_batch(self) -> VNextModelBatch:
        if not self.decisions:
            raise EncoderContractError("stream has no decisions")
        length = self.n_tokens
        pad = length
        def pad_col(column, value):
            return column + [value] * (pad - len(column))
        tensors = {
            "event_type_id": pad_col(self.event_type_id, PAD_EVENT_ID),
            "event_actor_rel": pad_col(self.event_actor_rel, 0),
            "event_tile_kind": pad_col(self.event_tile_kind, UNKNOWN_TILE_KIND),
            "event_aka": pad_col(self.event_aka, 0),
            "event_tsumogiri": pad_col(self.event_tsumogiri, 0),
            "event_kyoku_pos": pad_col(self.event_kyoku_pos, 0),
        }
        valid = torch.zeros((1, pad), dtype=torch.bool)
        valid[0, :length] = True
        states = [state for _index, state in self.decisions]
        return VNextModelBatch(
            event_type_id=torch.tensor([tensors["event_type_id"]], dtype=torch.long),
            event_actor_rel=torch.tensor([tensors["event_actor_rel"]], dtype=torch.long),
            event_tile_kind=torch.tensor([tensors["event_tile_kind"]], dtype=torch.long),
            event_aka=torch.tensor([tensors["event_aka"]], dtype=torch.long),
            event_tsumogiri=torch.tensor([tensors["event_tsumogiri"]], dtype=torch.long),
            event_kyoku_pos=torch.tensor([tensors["event_kyoku_pos"]], dtype=torch.long),
            event_valid_mask=valid,
            decision_to_sequence=torch.zeros(len(states), dtype=torch.long),
            prefix_token_offsets=torch.tensor(
                [index.prefix_token_offset for index, _state in self.decisions],
                dtype=torch.long),
            phases=torch.tensor([index.phase for index, _state in self.decisions],
                                dtype=torch.long),
            x_tile=torch.stack([state.tile_features for state in states]),
            x_global=torch.stack([state.global_features for state in states]),
            candidate_features=torch.stack(
                [state.candidate_features for state in states]),
            legal_mask=torch.stack([state.legal_mask for state in states]),
        )


@dataclass
class VNextBatchBuilder:
    """Combine several POV streams into one padded batch (no cross-stream KV)."""

    streams: list[VNextEventStream] = field(default_factory=list)

    def add_stream(self, stream: VNextEventStream) -> int:
        if not isinstance(stream, VNextEventStream):
            raise EncoderContractError("stream must be VNextEventStream")
        self.streams.append(stream)
        return len(self.streams) - 1

    def to_batch(self) -> VNextModelBatch:
        if not self.streams:
            raise EncoderContractError("no streams")
        length = max(stream.n_tokens for stream in self.streams)
        batch = len(self.streams)
        flat = {
            "event_type_id": [], "event_actor_rel": [], "event_tile_kind": [],
            "event_aka": [], "event_tsumogiri": [], "event_kyoku_pos": [],
        }
        valid = torch.zeros((batch, length), dtype=torch.bool)
        decisions: list[tuple[int, VNextDecisionIndex, VNextVisibleState]] = []
        for seq, stream in enumerate(self.streams):
            valid[seq, :stream.n_tokens] = True
            for name, column in (
                ("event_type_id", stream.event_type_id),
                ("event_actor_rel", stream.event_actor_rel),
                ("event_tile_kind", stream.event_tile_kind),
                ("event_aka", stream.event_aka),
                ("event_tsumogiri", stream.event_tsumogiri),
                ("event_kyoku_pos", stream.event_kyoku_pos),
            ):
                flat[name].append(column + [0] * (length - stream.n_tokens))
            for index, state in stream.decisions:
                decisions.append((seq, index, state))
        if not decisions:
            raise EncoderContractError("batch has no decisions")
        return VNextModelBatch(
            event_type_id=torch.tensor(flat["event_type_id"], dtype=torch.long),
            event_actor_rel=torch.tensor(flat["event_actor_rel"], dtype=torch.long),
            event_tile_kind=torch.tensor(flat["event_tile_kind"], dtype=torch.long),
            event_aka=torch.tensor(flat["event_aka"], dtype=torch.long),
            event_tsumogiri=torch.tensor(flat["event_tsumogiri"], dtype=torch.long),
            event_kyoku_pos=torch.tensor(flat["event_kyoku_pos"], dtype=torch.long),
            event_valid_mask=valid,
            decision_to_sequence=torch.tensor([d[0] for d in decisions], dtype=torch.long),
            prefix_token_offsets=torch.tensor(
                [d[1].prefix_token_offset for d in decisions], dtype=torch.long),
            phases=torch.tensor([d[2].phase for d in decisions], dtype=torch.long),
            x_tile=torch.stack([d[2].tile_features for d in decisions]),
            x_global=torch.stack([d[2].global_features for d in decisions]),
            candidate_features=torch.stack(
                [d[2].candidate_features for d in decisions]),
            legal_mask=torch.stack([d[2].legal_mask for d in decisions]),
        )


def encode_stream(decision: DecisionInput, *, decision_id: str = "d0",
                  decision_order: int = 0, match_id: str = "local",
                  pov: int = 0,
                  existing: VNextEventStream | None = None) -> VNextEventStream:
    """Build (or extend) one current-kyoku stream from a decision prefix."""
    if existing is None:
        current = [event for event in decision.events
                   if int(event.kyoku_idx) >= 0]
        if not current:
            raise EncoderContractError("decision has no current-kyoku events")
        kyoku_id = int(current[-1].kyoku_idx)
        stream = VNextEventStream(match_id=match_id, kyoku_id=kyoku_id, pov=pov)
    else:
        stream = existing
    stream.add_decision(decision, decision_id=decision_id,
                        decision_order=decision_order)
    return stream


__all__ = [
    "EncoderContractError", "PHASE_INDEX", "PHASE_ORDER", "VNextBatchBuilder",
    "VNextEventStream", "encode_stream", "encode_visible_state",
]
