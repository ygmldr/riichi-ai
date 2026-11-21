#!/usr/bin/env python3
"""PR-11 online adapter.

This module is deliberately a thin adapter over :func:`riichiformer.data.encode.encode_decision`
/:func:`encode_batch`; it must never grow a second token/state mapping.  It imports no
target/label/rank module and builds only forward-safe arrays.
"""

from __future__ import annotations

from ..data.encode import (
    MAX_EVENTS_DEFAULT,
    EncodedBatch,
    EncodedDecision,
    encode_batch,
    encode_decision,
)
from ..schemas.decision_v3 import DecisionInput


class OnlineEncoder:
    """Live/arena adapter with the same pure mapping as offline encoding."""

    def __init__(self, max_events: int = MAX_EVENTS_DEFAULT):
        if isinstance(max_events, bool) or not isinstance(max_events, int) \
                or max_events <= 0:
            raise ValueError(f"max_events must be a positive int, got {max_events!r}")
        self.max_events = int(max_events)

    def encode(self, decision: DecisionInput, stream_meta) -> EncodedDecision:
        return encode_decision(decision, stream_meta, max_events=self.max_events)

    def encode_batch(self, decisions, stream_metas) -> EncodedBatch:
        return encode_batch(decisions, stream_metas, max_events=self.max_events)


__all__ = ["OnlineEncoder"]
