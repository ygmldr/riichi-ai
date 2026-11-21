#!/usr/bin/env python3
"""PR-38 multi-decision stream: one causal event sequence + explicit decision markers.

The legacy pipeline turns one ``(game, POV)`` stream into ``N`` independent
prefix samples: ``encode_batch`` materialises every prefix and the transformer
recomputes the shared history once per decision.  This module describes the
shared-forward alternative:

``MultiDecisionStream``
    one **full causal event main sequence** plus ``N`` decision descriptors.  A
    descriptor is either a PR-10 :class:`~riichiformer.schemas.decision_v3.DecisionInput`
    (the exact visible prefix must end at its cutoff) or a :class:`DecisionMarker`
    (cutoff + normalized state features + legal mask + phase).
``DecisionMarker``
    the target/oracle-free per-decision payload used by the tensor encoder.  It
    carries that decision's own state vector; nothing in the event sequence is
    modified by it, so decisions never see each other's state or a later event.
``prefix_recompute_profile``
    the profiler required by PR-38: per stream, how many prefix tokens the legacy
    path recomputes versus how many tokens the shared pass reads.

The tensor contract itself (:class:`~riichiformer.models.batch.MultiDecisionBatch`)
and the pure mapping live in ``models/batch.py`` and ``data/encode.py``.  This
module imports no ``data.encode`` symbol so the dependency direction stays
``encode -> multi_decision`` and there is no import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from ..models.batch import PHASE_INDEX
from ..models.state_encoder import STATE_FEATURE_DIM
from ..schemas.action_v3 import ACTION_SPACE, ActionPhase, validate_mask_for_phase
from ..schemas.decision_v3 import DecisionInput
from ..schemas.event_v3 import EVENT_CLASSES, ObservableEventV3


class MultiDecisionError(ValueError):
    """A multi-decision stream violates the causal/POV contract (fail closed)."""


def check_event_sequence(events: Sequence) -> tuple:
    """Validate one projected POV event main sequence.

    The sequence must be the contiguous projection of one game/POV: ``seq_idx``
    ``0..L-1``, exact registered event classes, non-decreasing ``kyoku_idx``.  No
    raw dict, unknown subclass or out-of-order event is accepted.
    """
    events = tuple(events)
    if not events:
        raise MultiDecisionError("multi-decision stream needs a non-empty event sequence")
    previous_kyoku = -1
    for index, event in enumerate(events):
        if type(event) not in EVENT_CLASSES.values():
            raise MultiDecisionError(
                f"event[{index}] is not an exact registered event class"
            )
        if event.seq_idx != index:
            raise MultiDecisionError(
                f"event[{index}].seq_idx={event.seq_idx}; the main sequence must be "
                "contiguous from 0"
            )
        if event.kyoku_idx < previous_kyoku:
            raise MultiDecisionError(
                f"event[{index}].kyoku_idx={event.kyoku_idx} goes back to an "
                "earlier kyoku"
            )
        previous_kyoku = event.kyoku_idx
    return events


def _cutoff_of(decision) -> int:
    if isinstance(decision, DecisionInput):
        return decision.before_event_idx
    if isinstance(decision, DecisionMarker):
        return decision.cutoff_idx
    raise MultiDecisionError(
        f"decision must be DecisionInput or DecisionMarker, got "
        f"{type(decision).__name__}"
    )


def check_prefix(prefix: Sequence, events: Sequence, cutoff: int, *,
                 verify: str) -> None:
    """Fail closed unless ``prefix`` is exactly ``events[:cutoff]``.

    ``verify == "structural"`` (default) compares event class and the two
    identity fields; ``"exact"`` compares the whole frozen dataclass.  Both
    reject a decision whose visible prefix is not the stream prefix, which is the
    property that makes "only events before the cutoff" enforceable.
    """
    if verify not in ("structural", "exact"):
        raise MultiDecisionError(
            f"verify must be 'structural' or 'exact', got {verify!r}"
        )
    if isinstance(prefix, (str, bytes)) or len(prefix) != cutoff:
        raise MultiDecisionError(
            f"decision prefix length {len(prefix)} != cutoff {cutoff}"
        )
    for index in range(cutoff):
        left, right = prefix[index], events[index]
        if type(left) is not type(right):
            raise MultiDecisionError(
                f"decision prefix event[{index}] type {type(left).__name__} != "
                f"stream {type(right).__name__}"
            )
        if left.seq_idx != right.seq_idx or left.kyoku_idx != right.kyoku_idx:
            raise MultiDecisionError(
                f"decision prefix event[{index}] identity "
                f"(seq_idx={left.seq_idx}, kyoku_idx={left.kyoku_idx}) != stream "
                f"(seq_idx={right.seq_idx}, kyoku_idx={right.kyoku_idx})"
            )
        if verify == "exact" and left != right:
            raise MultiDecisionError(
                f"decision prefix event[{index}] differs from the stream event"
            )


@dataclass(frozen=True, eq=False)
class DecisionMarker:
    """Target-free descriptor of one decision inside a shared event stream.

    ``cutoff_idx`` is the number of visible events (the exclusive prefix end).
    ``state_features`` is the documented 172-float vector of
    :mod:`riichiformer.models.state_encoder`; it is injected only into this
    decision's own marker token.
    """

    cutoff_idx: int
    phase: ActionPhase
    legal_mask: tuple[bool, ...]
    state_features: Tensor

    def __post_init__(self) -> None:
        if isinstance(self.cutoff_idx, bool) or not isinstance(self.cutoff_idx, int) \
                or self.cutoff_idx < 1:
            raise MultiDecisionError(
                f"cutoff_idx must be a positive int, got {self.cutoff_idx!r}"
            )
        if not isinstance(self.phase, ActionPhase):
            raise MultiDecisionError(
                f"phase must be ActionPhase, got {type(self.phase).__name__}"
            )
        if not isinstance(self.legal_mask, tuple) or len(self.legal_mask) != ACTION_SPACE:
            raise MultiDecisionError(
                f"legal_mask must be a tuple of {ACTION_SPACE} bools"
            )
        try:
            validate_mask_for_phase(self.legal_mask, self.phase)
        except Exception as exc:  # noqa: BLE001 - re-raise in this module's type
            raise MultiDecisionError(f"legal_mask/phase invalid: {exc}") from exc
        if not isinstance(self.state_features, Tensor) \
                or self.state_features.dim() != 1 \
                or int(self.state_features.numel()) != STATE_FEATURE_DIM:
            raise MultiDecisionError(
                f"state_features must be a [{STATE_FEATURE_DIM}] tensor, got "
                f"{tuple(self.state_features.shape) if isinstance(self.state_features, Tensor) else type(self.state_features).__name__}"
            )
        if not self.state_features.is_floating_point():
            raise MultiDecisionError("state_features must be floating point")
        if not bool(torch.isfinite(self.state_features).all()):
            raise MultiDecisionError("state_features contains NaN/Inf")

    @property
    def phase_index(self) -> int:
        return PHASE_INDEX[self.phase]


@dataclass(frozen=True, eq=False)
class MultiDecisionStream:
    """One ``(game, POV)`` event main sequence plus its decisions.

    ``meta`` is the PR-11 provenance of the projected stream (a
    ``ProjectedStreamHeader``/``StreamMeta`` or a mapping).  It is deliberately
    typed as ``object`` so this module never imports ``data.encode`` (which
    imports this module); the encoder coerces and validates it.
    """

    events: tuple[ObservableEventV3, ...]
    decisions: tuple[DecisionInput | DecisionMarker, ...]
    meta: object = None

    def __post_init__(self) -> None:
        events = check_event_sequence(self.events)
        decisions = tuple(self.decisions)
        if not decisions:
            raise MultiDecisionError("stream needs at least one decision")
        if self.meta is None:
            raise MultiDecisionError("stream needs projected-stream provenance")
        previous_cutoff = 0
        for index, decision in enumerate(decisions):
            cutoff = _cutoff_of(decision)
            if not 1 <= cutoff <= len(events):
                raise MultiDecisionError(
                    f"decision[{index}] cutoff {cutoff} outside [1, {len(events)}]"
                )
            if cutoff < previous_cutoff:
                raise MultiDecisionError(
                    f"decision[{index}] cutoff {cutoff} < previous cutoff "
                    f"{previous_cutoff}; decisions must stay in causal order"
                )
            previous_cutoff = cutoff
            if isinstance(decision, DecisionInput):
                if decision.before_event_idx != len(decision.events):
                    raise MultiDecisionError(
                        "DecisionInput must be the exact visible prefix"
                    )
                check_prefix(decision.events, events, cutoff, verify="exact")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "decisions", decisions)

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_decision_inputs(cls, events, decisions, meta) -> "MultiDecisionStream":
        """Explicit-stream constructor (online/offline shared entry point)."""
        return cls(events=tuple(events), decisions=tuple(decisions), meta=meta)

    @classmethod
    def from_samples(cls, samples) -> "MultiDecisionStream":
        """Build one stream from PR-10 ``DecisionSample`` objects of one game/POV.

        The main sequence is the longest decision prefix: the last decision's
        visible prefix is the complete causal history up to the last decision
        (trailing settlement events can be appended by the caller, they are never
        read by a decision).  Samples must be the contiguous decision order of one
        POV; mixed POVs fail closed.
        """
        samples = tuple(samples)
        if not samples:
            raise MultiDecisionError("need at least one DecisionSample")
        for sample in samples:
            if not hasattr(sample, "header") or not hasattr(sample, "input"):
                raise MultiDecisionError(
                    "samples must expose header/input/decision_index"
                )
        meta = samples[0].header
        pov = meta.pov
        for sample in samples:
            if sample.header.pov != pov:
                raise MultiDecisionError(
                    f"mixed POVs in one stream: {pov} and {sample.header.pov}"
                )
            if sample.header != meta:
                raise MultiDecisionError("samples must share one projected-stream header")
        ordered = sorted(samples, key=lambda sample: sample.decision_index)
        indices = [int(sample.decision_index) for sample in ordered]
        if indices != list(range(len(indices))):
            raise MultiDecisionError(
                f"decision_index must be contiguous 0..N-1, got {indices[:8]}..."
            )
        longest = max((sample.input for sample in ordered),
                      key=lambda decision: len(decision.events))
        return cls(events=longest.events,
                   decisions=tuple(sample.input for sample in ordered),
                   meta=meta)

    # -------------------------------------------------------------- properties
    @property
    def length(self) -> int:
        return len(self.events)

    @property
    def cutoffs(self) -> tuple[int, ...]:
        return tuple(_cutoff_of(decision) for decision in self.decisions)

    @property
    def pov(self) -> int | None:
        return getattr(self.meta, "pov", None)

    def profile(self) -> dict:
        """Per-stream decision count and legacy recomputation cost."""
        cutoffs = self.cutoffs
        prefix_tokens = sum(cutoffs)
        shared_tokens = self.length + len(cutoffs)
        return {
            "pov": self.pov,
            "events": self.length,
            "decisions": len(cutoffs),
            "prefix_tokens": prefix_tokens,
            "shared_tokens": shared_tokens,
            "saved_tokens": prefix_tokens - shared_tokens,
            "ratio": (prefix_tokens / shared_tokens) if shared_tokens else 1.0,
            "duplicate_cutoffs": len(cutoffs) - len(set(cutoffs)),
            "max_cutoff": max(cutoffs),
        }


def streams_from_samples(samples) -> list[MultiDecisionStream]:
    """Group one game's ``DecisionSample`` list into one stream per POV."""
    samples = tuple(samples)
    if not samples:
        raise MultiDecisionError("need at least one DecisionSample")
    by_pov: dict[int, list] = {}
    for sample in samples:
        by_pov.setdefault(sample.header.pov, []).append(sample)
    return [MultiDecisionStream.from_samples(by_pov[pov]) for pov in sorted(by_pov)]


def prefix_recompute_profile(streams: Sequence[MultiDecisionStream]) -> dict:
    """Aggregate the PR-38 profiler over many streams.

    ``prefix_tokens`` is the number of event tokens a per-decision prefix pipeline
    encodes and runs through the transformer (the legacy cost);
    ``shared_tokens`` is ``events + decision markers`` for one shared pass.
    """
    streams = tuple(streams)
    if not streams:
        raise MultiDecisionError("need at least one stream")
    rows = [stream.profile() for stream in streams]
    totals = {
        key: sum(row[key] for row in rows)
        for key in ("events", "decisions", "prefix_tokens", "shared_tokens",
                    "saved_tokens")
    }
    totals["ratio"] = (
        totals["prefix_tokens"] / totals["shared_tokens"]
        if totals["shared_tokens"] else 1.0
    )
    totals["streams"] = len(rows)
    totals["per_stream"] = rows
    return totals


__all__ = [
    "DecisionMarker", "MultiDecisionError", "MultiDecisionStream",
    "check_event_sequence", "check_prefix", "prefix_recompute_profile",
    "streams_from_samples",
]
