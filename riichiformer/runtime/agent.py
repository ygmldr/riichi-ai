#!/usr/bin/env python3
"""PR-16 arena inference agent (mock-first, Mortal-compatible seat interface).

The agent is deliberately small and framework-free:

* it accepts one PR-10 :class:`~riichiformer.schemas.decision_v3.DecisionInput` per
  decision (the POV-safe visible prefix + state + native 46-mask + phase);
* it optionally runs the PR-11 :func:`~riichiformer.data.encode.encode_decision`
  encoder (for schema/hash checks and the future PR-14 model), then calls a policy
  callable ``policy(decision) -> int``;
* it decodes the action through :mod:`riichiformer.runtime.mjai_adapter`, which
  validates the native legal mask and the phase before emitting a concrete MJAI
  reaction;
* it fails loudly on a ``None``/empty/out-of-range policy output, an illegal
  action, an ambiguous multi-kan declaration, an encoder truncation, or a visible
  prefix that contains an opponent's concealed tile.

There is no target, label, rank or oracle parameter anywhere on this surface; the
policy sees only the forward-safe ``DecisionInput``.  The Mortal-style
``set_scene``/``get_reaction`` pair is provided for a future 1v3 adapter, while the
mock arena loop calls :meth:`RiichiFormerAgent.act` directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..schemas.action_v3 import ACTION_SPACE, ActionPhase
from ..schemas.decision_v3 import DecisionInput
from ..schemas.event_v3 import AnkanEvent, TsumoEvent, UNKNOWN_TILE_KIND
from .mjai_adapter import (
    ActionContext,
    AdapterError,
    AmbiguousKanError,
    IllegalActionError,
    KanOption,
    action_to_mjai,
    context_from_decision,
    mjai_to_action,
)

try:  # PR-11 encoder + torch are optional for a pure-adapter agent.
    from ..data.encode import EncodedDecision, StreamMeta, encode_decision
except Exception:  # pragma: no cover - exercised when torch is unavailable
    EncodedDecision = None  # type: ignore[assignment]
    StreamMeta = None  # type: ignore[assignment]
    encode_decision = None  # type: ignore[assignment]


class AgentError(RuntimeError):
    """Base class for PR-16 agent failures (always fail closed)."""


class IllegalAgentAction(AgentError):
    """The policy returned a non-integer, out-of-range, masked, or ambiguous action."""


class EmptyPolicyOutput(AgentError):
    """The policy returned ``None`` (or a truncated/empty object)."""


class TruncatedStream(AgentError):
    """The encoder rejected the visible prefix (e.g. ``max_events`` exceeded)."""


class VisibilityError(AgentError):
    """An opponent concealed tile entered the POV token stream."""


@dataclass(frozen=True)
class DecisionRecord:
    decision_index: int
    phase: str
    action: int
    event: Mapping[str, Any]
    encoded_hash: str | None = None


class MockPolicy:
    """Deterministic mock policy: sample uniformly among the legal actions.

    ``seed`` is per seat/hand so the arena loop is reproducible.  ``select_kan``
    resolves a multi-kan declaration explicitly (the second stage).
    """

    name = "mock"

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self._state = int(seed) & 0xFFFFFFFF
        self._calls = 0

    def _next(self) -> int:
        # xorshift32: small, dependency-free, reproducible across Python versions.
        value = self._state or 0x9E3779B9
        value ^= (value << 13) & 0xFFFFFFFF
        value ^= value >> 17
        value ^= (value << 5) & 0xFFFFFFFF
        self._state = value & 0xFFFFFFFF
        return self._state

    def __call__(self, decision: DecisionInput) -> int:
        if not isinstance(decision, DecisionInput):
            raise IllegalAgentAction("policy expected DecisionInput")
        legal = [index for index, allowed in enumerate(decision.legal_mask) if allowed]
        if not legal:
            raise EmptyPolicyOutput("policy received an empty legal mask")
        self._calls += 1
        return legal[self._next() % len(legal)]

    def select_kan(self, options: Sequence[KanOption]) -> KanOption:
        if not options:
            raise EmptyPolicyOutput("no kan options to select")
        return options[self._next() % len(options)]


PolicyCallable = Callable[[DecisionInput], int]


@dataclass
class AgentStats:
    decisions: int = 0
    encoded: int = 0
    illegal: int = 0
    empty: int = 0
    truncated: int = 0
    ambiguous: int = 0
    visibility: int = 0

    def as_dict(self) -> dict:
        return {
            "decisions": self.decisions,
            "encoded": self.encoded,
            "illegal": self.illegal,
            "empty": self.empty,
            "truncated": self.truncated,
            "ambiguous": self.ambiguous,
            "visibility": self.visibility,
        }


class RiichiFormerAgent:
    """One POV seat: policy + optional PR-11 encoder, strict action validation."""

    engine_type = "mortal"  # Mortal's libriichi agent bridge expects this tag.
    version = 1
    is_oracle = False
    enable_quick_eval = False
    enable_rule_based_agari_guard = False

    def __init__(self, seat: int, policy: PolicyCallable, *,
                 encoder: Callable[..., Any] | None = None,
                 stream_meta: Any | None = None,
                 max_events: int = 4096,
                 reveal_opponent_ankan: bool = False,
                 name: str = "rif-mock"):
        if isinstance(seat, bool) or not isinstance(seat, int) or not 0 <= seat <= 3:
            raise AgentError(f"seat must be int 0..3, got {seat!r}")
        if not callable(policy):
            raise AgentError("policy must be callable: policy(decision) -> int")
        if encoder is not None and not callable(encoder):
            raise AgentError("encoder must be callable or None")
        if stream_meta is not None:
            meta_pov = getattr(stream_meta, "pov", None)
            if meta_pov is not None and int(meta_pov) != seat:
                raise AgentError(
                    f"stream_meta.pov={meta_pov} does not match agent seat {seat}")
        self.seat = seat
        self.policy = policy
        self.encoder = encoder
        self.stream_meta = stream_meta
        self.max_events = int(max_events)
        self.reveal_opponent_ankan = bool(reveal_opponent_ankan)
        self.name = str(name)
        self.stats = AgentStats()
        self.records: list[DecisionRecord] = []
        self.last_encoded_hash: str | None = None
        self._decision: DecisionInput | None = None
        self._player_id: int | None = None
        self._game_id = 0
        self._kyoku = -1
        self._last_log_lines = 0

    # -- lifecycle ---------------------------------------------------------
    def reset_game(self) -> None:
        """Clear per-game state (seat identity is immutable)."""
        self._game_id += 1
        self._kyoku = -1
        self._decision = None
        self._player_id = None
        self._last_log_lines = 0
        self.records.clear()
        self.last_encoded_hash = None
        self.stats = AgentStats()

    def reset_kyoku(self) -> None:
        """Clear per-kyoku state (no cross-kyoku Local KV is kept in PR-16)."""
        self._kyoku += 1
        self._last_log_lines = 0
        # Per-kyoku caches are intentionally empty; the full hanchan context is
        # owned by the caller/PR-14, not by this stateless mock agent.

    # -- Mortal-compatible scene interface --------------------------------
    def set_scene(self, player_id: int, log: Any = None, state: Any = None,
                  invisible_state: Any = None) -> None:
        """Mortal-style scene setter.

        ``state`` may be a :class:`DecisionInput` (the mock path) or an object
        exposing ``.decision``.  ``log`` is kept only as a line count; raw logs and
        hidden opponent data are never stored on the agent.
        """
        if isinstance(player_id, bool) or not isinstance(player_id, int):
            raise AgentError(f"player_id must be int, got {player_id!r}")
        self._player_id = int(player_id)
        self._last_log_lines = (
            len(log.splitlines()) if isinstance(log, str) and log else len(log or ()))
        if isinstance(state, DecisionInput):
            self._decision = state
        elif state is not None and isinstance(getattr(state, "decision", None), DecisionInput):
            self._decision = state.decision
        else:
            self._decision = None

    def get_reaction(self) -> str:
        """Return one MJAI event as a compact JSON string."""
        return json.dumps(self.act(), ensure_ascii=False, separators=(",", ":"))

    # -- action ------------------------------------------------------------
    def act(self, decision: DecisionInput | None = None) -> dict:
        decision = decision if decision is not None else self._decision
        if not isinstance(decision, DecisionInput):
            raise AgentError("no DecisionInput scene set")
        self._validate_decision(decision)

        encoded = self._encode(decision)
        try:
            action = self.policy(decision)
        except AgentError:
            raise
        except Exception as exc:  # noqa: BLE001 - policy bugs must be loud
            self.stats.empty += 1
            raise AgentError(f"policy raised {type(exc).__name__}: {exc}") from exc

        if action is None:
            self.stats.empty += 1
            raise EmptyPolicyOutput("policy returned None")
        if isinstance(action, bool) or not isinstance(action, int):
            self.stats.illegal += 1
            raise IllegalAgentAction(f"policy returned non-integer action {action!r}")
        if not 0 <= action < ACTION_SPACE:
            self.stats.illegal += 1
            raise IllegalAgentAction(
                f"action {action} outside 0..{ACTION_SPACE - 1}")
        if not decision.legal_mask[action]:
            self.stats.illegal += 1
            raise IllegalAgentAction(
                f"action {action} is outside the native legal mask")

        context = context_from_decision(decision, self.seat)
        if action == 42 and len(context.kan_options) > 1:
            selector = getattr(self.policy, "select_kan", None)
            if selector is None:
                self.stats.ambiguous += 1
                raise IllegalAgentAction(
                    "multi-kan declaration needs policy.select_kan(options)")
            try:
                choice = selector(context.kan_options)
            except Exception as exc:  # noqa: BLE001
                self.stats.ambiguous += 1
                raise IllegalAgentAction(
                    f"select_kan failed: {type(exc).__name__}: {exc}") from exc
            if not isinstance(choice, KanOption):
                self.stats.ambiguous += 1
                raise IllegalAgentAction("select_kan must return a KanOption")
            if choice not in context.kan_options:
                self.stats.ambiguous += 1
                raise IllegalAgentAction("select_kan returned an unknown option")
            context = context.with_kan_choice(choice.tile)

        try:
            event = action_to_mjai(action, context)
        except AmbiguousKanError as exc:
            self.stats.ambiguous += 1
            raise IllegalAgentAction(str(exc)) from exc
        except (IllegalActionError, AdapterError) as exc:
            self.stats.illegal += 1
            raise IllegalAgentAction(str(exc)) from exc

        try:
            round_trip = mjai_to_action(event, decision.phase)
        except AdapterError as exc:
            self.stats.illegal += 1
            raise IllegalAgentAction(f"decoded event is not reversible: {exc}") from exc
        if round_trip != action:
            self.stats.illegal += 1
            raise IllegalAgentAction(
                f"decoded event maps back to action {round_trip}, not {action}")

        self.stats.decisions += 1
        self.records.append(DecisionRecord(
            decision_index=len(self.records),
            phase=decision.phase.value,
            action=action,
            event=event,
            encoded_hash=self.last_encoded_hash,
        ))
        self._decision = None
        return event

    # -- validation -------------------------------------------------------
    def _validate_decision(self, decision: DecisionInput) -> None:
        if decision.before_event_idx != len(decision.events):
            raise AgentError(
                "DecisionInput must be the exact visible prefix: "
                f"before_event_idx={decision.before_event_idx}, "
                f"events={len(decision.events)}")
        if len(decision.legal_mask) != ACTION_SPACE:
            raise AgentError(
                f"legal_mask width {len(decision.legal_mask)} != {ACTION_SPACE}")
        if not any(decision.legal_mask):
            raise EmptyPolicyOutput("decision has an empty legal mask")
        if not isinstance(decision.phase, ActionPhase):
            raise AgentError(f"invalid phase {decision.phase!r}")
        for event in decision.events:
            if (isinstance(event, TsumoEvent) and event.actor_rel != 0
                    and event.tile.kind != UNKNOWN_TILE_KIND):
                self.stats.visibility += 1
                raise VisibilityError(
                    "opponent tsumo tile is visible in the POV token prefix")
            if (isinstance(event, AnkanEvent) and event.actor_rel != 0
                    and not self.reveal_opponent_ankan
                    and any(tile.kind != UNKNOWN_TILE_KIND for tile in event.consumed)):
                self.stats.visibility += 1
                raise VisibilityError(
                    "opponent ankan tiles are visible in the POV token prefix")

    def _encode(self, decision: DecisionInput) -> Any | None:
        if self.encoder is None:
            return None
        if self.stream_meta is None:
            self.stats.truncated += 1
            raise TruncatedStream("encoder configured without stream_meta")
        try:
            encoded = self.encoder(decision, self.stream_meta,
                                   max_events=self.max_events)
        except Exception as exc:  # noqa: BLE001 - encoder errors are fatal
            self.stats.truncated += 1
            raise TruncatedStream(
                f"encoder rejected the visible prefix: {type(exc).__name__}: {exc}"
            ) from exc
        self.stats.encoded += 1
        try:
            from ..data.encode import encoded_hash
            self.last_encoded_hash = encoded_hash(encoded)
        except Exception:  # pragma: no cover - encoder always provides this
            self.last_encoded_hash = None
        return encoded


__all__ = [
    "AgentError", "AgentStats", "DecisionRecord", "EmptyPolicyOutput",
    "IllegalAgentAction", "MockPolicy", "RiichiFormerAgent", "TruncatedStream",
    "VisibilityError",
]
