"""POV (per-seat) observable-event projector for schema v3 (PR-09).

PR-08 defines the event dataclasses and a POV-safe event *factory*
(``event_v3.event_from_mjai``).  This module adds the PR-09 projection layer:

* an explicit allowlist dispatcher :func:`project_event` (unknown type raises);
* a stateful :class:`POVProjector` that assigns ``seq_idx``/``kyoku_idx``, tracks the
  kyoku boundaries, seals the stream after ``end_game``, and exposes only the visible
  prefix;
* policy for rule-dependent reveals (opponent ankan), with documented evidence required;
* relative-seat score deltas for ``hora``/``ryukyoku`` (raw MJAI deltas are absolute
  seat order; the projected event uses self/shimo/toimen/kami like ``scores_rel``);
* deterministic ``encode_visible(before_event_idx)`` for the hidden-hand / future-cut
  property tests.

The raw dict is never deep-copied and never stored: every field is read through the
PR-08 per-type allowlist.  Names, account ids, wall seeds, log URLs, opponent hands and
future events cannot enter the projected stream.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from ..schemas.action_v3 import SCHEMA_VERSION, canonical_json
from ..schemas.decision_v3 import RulesetId, SourceKind
from ..schemas.event_v3 import (
    DELTAS_ORDER_RELATIVE,
    EVENT_CLASSES,
    AnkanEvent,
    HoraEvent,
    ObservableEventV3,
    RyukyokuEvent,
    UNKNOWN_TILE,
    UnknownEventType,
    event_from_mjai,
)

POV_PROJECTION_VERSION = "rf-pov-v3/2"


class ProjectionError(ValueError):
    """Base class for PR-09 projection errors."""


class ProjectionOrderError(ProjectionError):
    """An event appeared before ``start_kyoku`` or outside the stream order."""


class ProjectionClosed(ProjectionError):
    """The projector was already sealed by ``end_game``."""


class ProjectionPolicyError(ProjectionError):
    """The reveal policy lacks documented evidence or mixes provenance."""


@dataclass(frozen=True)
class ProjectionPolicy:
    """Ruleset/provenance metadata and the only opt-in reveal in PR-09.

    Opponent ankan tiles are **UNKNOWN by default**.  Revealing them requires both
    ``reveal_opponent_ankan=True`` and a non-empty ``ankan_evidence`` string documenting
    which ruleset/source makes the tiles public.  Without such evidence the projector must
    fail closed rather than copy the omniscient ``consumed`` field.
    """

    ruleset_id: RulesetId = RulesetId.TENHOU_4P_RANKED
    source_kind: SourceKind = SourceKind.HUMAN_LOG
    reveal_opponent_ankan: bool = False
    ankan_evidence: str | None = None

    def __post_init__(self):
        if not isinstance(self.ruleset_id, RulesetId):
            raise ProjectionPolicyError(
                f"ruleset_id must be RulesetId, got {self.ruleset_id!r}"
            )
        if not isinstance(self.source_kind, SourceKind):
            raise ProjectionPolicyError(
                f"source_kind must be SourceKind, got {self.source_kind!r}"
            )
        if self.source_kind == SourceKind.HUMAN_LOG and (
                self.ruleset_id != RulesetId.TENHOU_4P_RANKED):
            raise ProjectionPolicyError("human_log projection requires tenhou_4p_ranked")
        if self.source_kind == SourceKind.ARENA_SELFPLAY and (
                self.ruleset_id != RulesetId.MORTAL_ARENA_PINNED):
            raise ProjectionPolicyError("arena_selfplay projection requires mortal_arena_pinned")
        if not isinstance(self.reveal_opponent_ankan, bool):
            raise ProjectionPolicyError("reveal_opponent_ankan must be bool")
        if self.reveal_opponent_ankan:
            if not isinstance(self.ankan_evidence, str) or not self.ankan_evidence.strip():
                raise ProjectionPolicyError(
                    "reveal_opponent_ankan requires documented ankan_evidence"
                )
        elif self.ankan_evidence is not None:
            raise ProjectionPolicyError(
                "ankan_evidence is only meaningful when reveal_opponent_ankan=True"
            )


DEFAULT_POLICY = ProjectionPolicy()


@dataclass(frozen=True)
class ProjectedStreamHeader:
    """Attribution for a projected stream, so POV/raw mixes are detectable.

    The header is static for one ``POVProjector``; it is included in
    ``encode_visible``/``ProjectedStream.encode`` payloads but is **not** a schema event
    and is not inserted into ``DecisionInput`` (schema ownership).
    """

    pov: int
    projection_version: str
    schema_version: str
    ruleset_id: str
    source_kind: str
    reveal_opponent_ankan: bool

    def __post_init__(self):
        if self.projection_version != POV_PROJECTION_VERSION:
            raise ProjectionError(
                f"projection_version {self.projection_version!r} != "
                f"{POV_PROJECTION_VERSION!r}"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ProjectionError(
                f"schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}"
            )
        _check_pov(self.pov)
        if self.ruleset_id not in {value.value for value in RulesetId}:
            raise ProjectionError(f"unknown ruleset_id {self.ruleset_id!r}")
        if self.source_kind not in {value.value for value in SourceKind}:
            raise ProjectionError(f"unknown source_kind {self.source_kind!r}")
        if not isinstance(self.reveal_opponent_ankan, bool):
            raise ProjectionError("reveal_opponent_ankan must be bool")

    def as_dict(self) -> dict:
        return {
            "pov": self.pov,
            "projection_version": self.projection_version,
            "schema_version": self.schema_version,
            "ruleset_id": self.ruleset_id,
            "source_kind": self.source_kind,
            "reveal_opponent_ankan": self.reveal_opponent_ankan,
        }


@dataclass(frozen=True)
class ProjectedStream:
    """A header + projected events snapshot for one POV; no labels/ranks."""

    header: ProjectedStreamHeader
    events: tuple[ObservableEventV3, ...]

    def __post_init__(self):
        if not isinstance(self.header, ProjectedStreamHeader):
            raise ProjectionError("header must be ProjectedStreamHeader")
        if not isinstance(self.events, tuple):
            raise ProjectionError("events must be a tuple")
        for event in self.events:
            if not isinstance(event, ObservableEventV3):
                raise ProjectionError("events must contain ObservableEventV3")

    @property
    def count(self) -> int:
        return len(self.events)

    def visible(self, before_event_idx: int | None = None) -> tuple[ObservableEventV3, ...]:
        if before_event_idx is None:
            return self.events
        if (not isinstance(before_event_idx, int)
                or isinstance(before_event_idx, bool)
                or before_event_idx < 0):
            raise ProjectionError(
                f"before_event_idx {before_event_idx!r} must be an int >= 0"
            )
        return tuple(event for event in self.events
                     if event.seq_idx < before_event_idx)

    def encode(self, before_event_idx: int | None = None) -> bytes:
        payload = {
            "header": self.header.as_dict(),
            "events": [event.to_dict() for event in self.visible(before_event_idx)],
        }
        return canonical_json(payload).encode("utf-8")


@dataclass(frozen=True)
class ProjectionContext:
    """Per-event coordinates + reveal policy passed to :func:`project_event`."""

    seq_idx: int
    kyoku_idx: int
    policy: ProjectionPolicy = DEFAULT_POLICY

    def __post_init__(self):
        if not isinstance(self.seq_idx, int) or isinstance(self.seq_idx, bool) \
                or self.seq_idx < 0:
            raise ProjectionError(f"seq_idx {self.seq_idx!r} must be an int >= 0")
        if not isinstance(self.kyoku_idx, int) or isinstance(self.kyoku_idx, bool) \
                or self.kyoku_idx < -1:
            raise ProjectionError(f"kyoku_idx {self.kyoku_idx!r} must be an int >= -1")
        if not isinstance(self.policy, ProjectionPolicy):
            raise ProjectionError(f"policy must be ProjectionPolicy, got {self.policy!r}")


def _check_pov(pov) -> int:
    if not isinstance(pov, int) or isinstance(pov, bool) or not 0 <= pov < 4:
        raise ProjectionError(f"pov {pov!r} outside 0..3")
    return pov


def _event_type(raw: Mapping) -> str:
    if not isinstance(raw, Mapping):
        raise ProjectionError(f"raw event must be a mapping, got {type(raw).__name__}")
    type_name = raw.get("type")
    if not isinstance(type_name, str):
        raise ProjectionError(f"raw event type must be str, got {type_name!r}")
    if type_name not in EVENT_CLASSES:
        raise UnknownEventType(f"unknown MJAI event type {type_name!r}")
    return type_name


def rotate_deltas_to_relative(deltas, pov: int):
    """Absolute-seat MJAI deltas -> (self, shimo, toimen, kami)."""
    _check_pov(pov)
    if deltas is None:
        return None
    values = tuple(int(value) for value in deltas)
    if len(values) != 4:
        raise ProjectionError(f"score deltas must have 4 entries, got {len(values)}")
    return tuple(values[(pov + offset) % 4] for offset in range(4))


def rotate_deltas_to_absolute(deltas_rel, pov: int):
    """(self, shimo, toimen, kami) -> absolute-seat order (inverse of above)."""
    _check_pov(pov)
    if deltas_rel is None:
        return None
    values = tuple(int(value) for value in deltas_rel)
    if len(values) != 4:
        raise ProjectionError(f"score deltas must have 4 entries, got {len(values)}")
    absolute = [0, 0, 0, 0]
    for offset, value in enumerate(values):
        absolute[(pov + offset) % 4] = value
    return tuple(absolute)


def project_event(raw, pov: int, context: ProjectionContext) -> ObservableEventV3:
    """Project one full-info MJAI event into a POV-safe ``ObservableEventV3``.

    The raw mapping is read field-by-field through the PR-08 allowlist; it is never
    copied.  Unknown types raise :class:`UnknownEventType`.  ``hora``/``ryukyoku`` score
    deltas are rotated into relative seat order.
    """
    pov = _check_pov(pov)
    if not isinstance(context, ProjectionContext):
        raise ProjectionError(f"context must be ProjectionContext, got {type(context)!r}")
    type_name = _event_type(raw)
    if type_name == "start_kyoku":
        if context.kyoku_idx < 0:
            raise ProjectionOrderError("start_kyoku requires kyoku_idx >= 0")
    elif type_name != "start_game" and context.kyoku_idx < 0:
        raise ProjectionOrderError(
            f"event {type_name!r} appears before start_kyoku (kyoku_idx=-1)"
        )
    if type_name == "ankan":
        actor = raw.get("actor")
        if (not isinstance(actor, int) or isinstance(actor, bool)
                or not 0 <= actor < 4):
            raise ProjectionError(f"ankan actor {actor!r} outside 0..3")
        if actor != pov and not context.policy.reveal_opponent_ankan:
            # Mask before parsing: a hidden opponent's malformed/absent ``consumed``
            # must not fail the projection or reveal anything.
            return AnkanEvent(
                seq_idx=context.seq_idx, kyoku_idx=context.kyoku_idx,
                actor_rel=(actor - pov) % 4,
                consumed=(UNKNOWN_TILE, UNKNOWN_TILE, UNKNOWN_TILE, UNKNOWN_TILE),
            )
    event = event_from_mjai(
        raw,
        pov=pov,
        seq_idx=context.seq_idx,
        kyoku_idx=context.kyoku_idx,
        reveal_opponent_ankan=context.policy.reveal_opponent_ankan,
    )
    if isinstance(event, (HoraEvent, RyukyokuEvent)):
        event = replace(
            event,
            deltas=rotate_deltas_to_relative(event.deltas, pov),
            deltas_order=DELTAS_ORDER_RELATIVE,
        )
    return event


class POVProjector:
    """Stateful one-game projector for one POV."""

    def __init__(self, pov: int, *, policy: ProjectionPolicy = DEFAULT_POLICY):
        self._pov = _check_pov(pov)
        if not isinstance(policy, ProjectionPolicy):
            raise ProjectionError(f"policy must be ProjectionPolicy, got {policy!r}")
        self._policy = policy
        self._events: list[ObservableEventV3] = []
        self._kyoku_idx = -1
        self._closed = False

    @property
    def pov(self) -> int:
        return self._pov

    @property
    def policy(self) -> ProjectionPolicy:
        return self._policy

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def kyoku_idx(self) -> int:
        return self._kyoku_idx

    @property
    def next_seq_idx(self) -> int:
        return len(self._events)

    @property
    def events(self) -> tuple[ObservableEventV3, ...]:
        return tuple(self._events)

    @property
    def header(self) -> ProjectedStreamHeader:
        return ProjectedStreamHeader(
            pov=self._pov,
            projection_version=POV_PROJECTION_VERSION,
            schema_version=SCHEMA_VERSION,
            ruleset_id=self._policy.ruleset_id.value,
            source_kind=self._policy.source_kind.value,
            reveal_opponent_ankan=self._policy.reveal_opponent_ankan,
        )

    def stream(self) -> ProjectedStream:
        return ProjectedStream(header=self.header, events=self.events)

    def reset(self) -> None:
        self._events.clear()
        self._kyoku_idx = -1
        self._closed = False

    def project(self, raw) -> ObservableEventV3:
        """Project and append one event; state is unchanged if projection fails."""
        if self._closed:
            raise ProjectionClosed("projector is sealed by end_game; call reset()")
        type_name = _event_type(raw)
        next_kyoku = self._kyoku_idx + 1 if type_name == "start_kyoku" else self._kyoku_idx
        if type_name == "start_kyoku" and next_kyoku < 0:  # pragma: no cover - defensive
            raise ProjectionOrderError("start_kyoku cannot have a negative kyoku_idx")
        if type_name not in ("start_game", "start_kyoku") and self._kyoku_idx < 0:
            raise ProjectionOrderError(
                f"event {type_name!r} appears before start_kyoku"
            )
        context = ProjectionContext(
            seq_idx=len(self._events), kyoku_idx=next_kyoku, policy=self._policy,
        )
        event = project_event(raw, self._pov, context)
        self._events.append(event)
        self._kyoku_idx = next_kyoku
        if type_name == "end_game":
            self._closed = True
        return event

    def extend(self, raw_events: Sequence) -> tuple[ObservableEventV3, ...]:
        """Project a sequence atomically: on any error, roll back to the prior state."""
        saved_events = list(self._events)
        saved_kyoku = self._kyoku_idx
        saved_closed = self._closed
        added = []
        try:
            for raw in raw_events:
                added.append(self.project(raw))
        except Exception:
            self._events = saved_events
            self._kyoku_idx = saved_kyoku
            self._closed = saved_closed
            raise
        return tuple(added)

    def event_at(self, seq_idx: int) -> ObservableEventV3:
        if not isinstance(seq_idx, int) or isinstance(seq_idx, bool) or seq_idx < 0:
            raise ProjectionError(f"seq_idx {seq_idx!r} must be an int >= 0")
        if seq_idx >= len(self._events):
            raise ProjectionError(
                f"seq_idx {seq_idx} is not visible yet (have {len(self._events)} events)"
            )
        return self._events[seq_idx]

    def visible_events(self, before_event_idx: int | None = None
                       ) -> tuple[ObservableEventV3, ...]:
        """Events with ``seq_idx < before_event_idx`` (all events if ``None``)."""
        if before_event_idx is None:
            return tuple(self._events)
        if (not isinstance(before_event_idx, int)
                or isinstance(before_event_idx, bool)
                or before_event_idx < 0):
            raise ProjectionError(
                f"before_event_idx {before_event_idx!r} must be an int >= 0"
            )
        return tuple(event for event in self._events
                     if event.seq_idx < before_event_idx)

    def encode_visible(self, before_event_idx: int | None = None) -> bytes:
        """Deterministic bytes of header + visible prefix (canonical JSON, sorted keys)."""
        return self.stream().encode(before_event_idx)


def project_stream(raw_events: Sequence, pov: int, *,
                   policy: ProjectionPolicy = DEFAULT_POLICY
                   ) -> tuple[ObservableEventV3, ...]:
    """Convenience wrapper: project a whole one-game stream for one POV."""
    return POVProjector(pov, policy=policy).extend(raw_events)


__all__ = [
    "DEFAULT_POLICY", "POV_PROJECTION_VERSION", "POVProjector", "ProjectedStream",
    "ProjectedStreamHeader", "ProjectionClosed", "ProjectionContext", "ProjectionError",
    "ProjectionOrderError", "ProjectionPolicy", "ProjectionPolicyError", "project_event",
    "project_stream", "rotate_deltas_to_absolute", "rotate_deltas_to_relative",
]
