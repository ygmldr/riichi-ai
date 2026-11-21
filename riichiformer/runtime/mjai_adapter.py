#!/usr/bin/env python3
"""PR-16: 46-action → MJAI reaction adapter (offline/online single mapping).

The frozen schema action space (``ACTION_SPACE = 46``) is:

=====  ===========================================================
0..33  discard tile kind (1m..9m, 1p..9p, 1s..9s, E,S,W,N,P,F,C)
34..36 discard the red five of 5m/5p/5s (aka colour 0/1/2)
37     riichi declaration
38..40 chi (called tile is low / middle / high of the run)
41     pon
42     kan (ankan, kakan or daiminkan; the concrete meld is explicit)
43     agari (tsumo in SELF_TURN, ron in CALL_RESPONSE)
44     ryukyoku (kyuushu kyuuhai)
45     pass
=====  ===========================================================

``KAN_SELECTION`` reuses 0..36 as the **tile choice** for one kan candidate; a
second kan candidate with the same action still needs an explicit choice.  A
declaration (42) with more than one candidate is therefore ambiguous unless the
caller supplies ``ActionContext.kan_choice`` or calls :func:`select_kan_option`;
the agent does the latter through the policy's optional ``select_kan`` hook.

Every decoded action is checked against :class:`ActionContext.legal_mask` (the
native 46-wide mask) *and* the phase-legal action family.  Illegal, ambiguous, or
out-of-phase actions raise :class:`AdapterError`; this module never returns
``None`` or an illegal event.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from ..schemas.action_v3 import (
    ACTION_SPACE,
    AKA_KINDS,
    PHASE_LEGAL_ACTIONS,
    ActionPhase,
)
from ..schemas.decision_v3 import DecisionInput, MeldKind
from ..schemas.event_v3 import (
    AnkanEvent,
    ChiEvent,
    DahaiEvent,
    DaiminkanEvent,
    HoraEvent,
    KakanEvent,
    PonEvent,
    ReachEvent,
    RyukyokuEvent,
    TileRef,
    TsumoEvent,
    UNKNOWN_TILE,
    tile_from_text,
)

MELD_KIND_NAMES = {
    "ankan": "ankan",
    "kakan": "kakan",
    "daiminkan": "daiminkan",
}


class AdapterError(ValueError):
    """Base class for PR-16 adapter failures (fail closed)."""


class IllegalActionError(AdapterError):
    """Action is outside 0..45, not in the mask, or not legal in the phase."""


class AmbiguousKanError(AdapterError):
    """A kan action (42) has several candidates; an explicit choice is required."""


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def action_tile(action: int) -> TileRef | None:
    """Return the tile for a discard/kan-selection action 0..36 (else None)."""
    if not _is_int(action):
        raise IllegalActionError(f"action must be int, got {type(action).__name__}")
    if 0 <= action < 34:
        return TileRef(action, None)
    if 34 <= action <= 36:
        color = action - 34
        return TileRef(AKA_KINDS[color], color)
    return None


def action_for_tile(tile: TileRef) -> int:
    """Map a visible tile identity to its discard/kan-selection action index."""
    if not isinstance(tile, TileRef):
        raise AdapterError(f"expected TileRef, got {type(tile).__name__}")
    if tile.is_unknown:
        raise AdapterError("UNKNOWN tile has no action index")
    if tile.aka is not None:
        return 34 + int(tile.aka)
    return int(tile.kind)


def _counts(tiles: Iterable[TileRef]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for tile in tiles:
        counts[tile.kind] = counts.get(tile.kind, 0) + 1
    return counts


def _pick_tiles(hand: Sequence[TileRef], kinds: Sequence[int]) -> tuple[TileRef, ...] | None:
    """Pick one physical tile per requested kind, normal five before aka."""
    pool = list(hand)
    picked: list[TileRef] = []
    for kind in kinds:
        candidates = [tile for tile in pool if tile.kind == kind]
        if not candidates:
            return None
        candidates.sort(key=lambda tile: 0 if tile.aka is None else 1)
        chosen = candidates[0]
        pool.remove(chosen)
        picked.append(chosen)
    return tuple(picked)


def derive_chi_options(hand: Sequence[TileRef], called: TileRef) -> dict[int, tuple[TileRef, TileRef]]:
    """Action 38/39/40 → the two hand tiles for a call on ``called``."""
    if called.is_unknown or called.kind >= 27:
        return {}
    suit = called.kind // 9
    options: dict[int, tuple[TileRef, TileRef]] = {}
    for action, kinds in (
        (38, (called.kind + 1, called.kind + 2)),
        (39, (called.kind - 1, called.kind + 1)),
        (40, (called.kind - 2, called.kind - 1)),
    ):
        if any(kind < suit * 9 or kind >= (suit + 1) * 9 for kind in kinds):
            continue
        picked = _pick_tiles(hand, kinds)
        if picked is not None:
            options[action] = (picked[0], picked[1])
    return options


def derive_pon_consumed(hand: Sequence[TileRef], called: TileRef) -> tuple[TileRef, TileRef] | None:
    if called.is_unknown:
        return None
    picked = _pick_tiles(hand, (called.kind, called.kind))
    return None if picked is None else (picked[0], picked[1])


@dataclass(frozen=True)
class KanOption:
    """One concrete kan the 46-action ``42`` may stand for."""

    meld_type: str
    tile: TileRef
    consumed: tuple[TileRef, ...]
    target: int | None = None

    def __post_init__(self):
        if self.meld_type not in MELD_KIND_NAMES:
            raise AdapterError(f"unknown kan meld_type {self.meld_type!r}")
        if not isinstance(self.tile, TileRef) or self.tile.is_unknown:
            raise AdapterError("kan tile must be a known TileRef")
        expected = 4 if self.meld_type == "ankan" else 3
        if len(self.consumed) != expected:
            raise AdapterError(
                f"{self.meld_type} needs {expected} consumed tiles, got {len(self.consumed)}")
        if any(tile.is_unknown for tile in self.consumed):
            raise AdapterError("kan consumed tiles must be known")

    def event(self, seat: int) -> dict:
        if self.meld_type == "ankan":
            return {"type": "ankan", "actor": seat,
                    "consumed": [tile.text for tile in self.consumed]}
        if self.meld_type == "kakan":
            return {"type": "kakan", "actor": seat, "pai": self.tile.text}
        target = self.target
        if target is None:
            raise AdapterError("daiminkan option needs a target seat")
        return {"type": "daiminkan", "actor": seat, "target": target,
                "pai": self.tile.text,
                "consumed": [tile.text for tile in self.consumed]}


@dataclass(frozen=True)
class ActionContext:
    """Everything needed to turn one action index into one concrete MJAI event."""

    seat: int
    phase: ActionPhase
    legal_mask: tuple[bool, ...]
    hand: tuple[TileRef, ...] = ()
    last_kawa_tile: TileRef | None = None
    last_kawa_actor: int | None = None
    last_self_tsumo: TileRef | None = None
    chi_options: tuple[tuple[int, tuple[TileRef, TileRef]], ...] = ()
    kan_options: tuple[KanOption, ...] = ()
    kan_choice: TileRef | None = None
    can_tsumo_agari: bool = False
    can_ron_agari: bool = False
    ron_target: int | None = None
    can_ryukyoku: bool = False

    def __post_init__(self):
        if not _is_int(self.seat) or not 0 <= self.seat <= 3:
            raise AdapterError(f"seat must be int 0..3, got {self.seat!r}")
        if not isinstance(self.phase, ActionPhase):
            raise AdapterError(f"phase must be ActionPhase, got {self.phase!r}")
        if not isinstance(self.legal_mask, tuple) or len(self.legal_mask) != ACTION_SPACE:
            raise AdapterError("legal_mask must be a tuple of 46 bools")
        if not all(isinstance(value, bool) for value in self.legal_mask):
            raise AdapterError("legal_mask entries must be bool")
        if not self.legal_mask or not any(self.legal_mask):
            raise AdapterError("legal_mask has no legal action")
        for action, consumed in self.chi_options:
            if action not in (38, 39, 40) or len(consumed) != 2:
                raise AdapterError("chi_options must map 38/39/40 to two tiles")
        if self.kan_choice is not None and not isinstance(self.kan_choice, TileRef):
            raise AdapterError("kan_choice must be TileRef or None")

    # -- helpers ----------------------------------------------------------
    def is_legal(self, action) -> bool:
        return (_is_int(action) and 0 <= action < ACTION_SPACE
                and bool(self.legal_mask[action]))

    def kan_candidates(self, action: int) -> tuple[KanOption, ...]:
        """Candidates matching a KAN_SELECTION action (0..36, kind or aka)."""
        if 0 <= action <= 33:
            return tuple(option for option in self.kan_options if option.tile.kind == action)
        color = action - 34
        return tuple(option for option in self.kan_options
                     if option.tile.aka == color)

    def with_kan_choice(self, tile: TileRef) -> "ActionContext":
        return replace(self, kan_choice=tile)


def _check_action(action, ctx: ActionContext) -> None:
    if not ctx.is_legal(action):
        raise IllegalActionError(
            f"action {action!r} is not legal in the native mask "
            f"(phase {ctx.phase.value})")
    if action not in PHASE_LEGAL_ACTIONS[ctx.phase]:
        raise IllegalActionError(
            f"action {action} ({_action_name(action)}) is not legal in phase "
            f"{ctx.phase.value}")


def _action_name(action: int) -> str:
    if 0 <= action <= 33:
        return f"discard {action_tile(action).text}"
    if 34 <= action <= 36:
        return f"discard {action_tile(action).text}"
    return {
        37: "riichi", 38: "chi_low", 39: "chi_mid", 40: "chi_high",
        41: "pon", 42: "kan", 43: "agari", 44: "ryukyoku", 45: "pass",
    }.get(action, "?")


def _pick_kan(ctx: ActionContext, action: int) -> KanOption:
    candidates = ctx.kan_options
    if not candidates:
        raise IllegalActionError(
            f"action 42 is set but the context has no kan candidate")
    if ctx.kan_choice is not None:
        chosen = [option for option in candidates if option.tile == ctx.kan_choice]
        if len(chosen) != 1:
            raise AmbiguousKanError(
                f"kan_choice {ctx.kan_choice.text} matches {len(chosen)} candidates")
        return chosen[0]
    if len(candidates) > 1:
        raise AmbiguousKanError(
            f"{len(candidates)} kan candidates and no explicit kan_choice "
            "(multi-kan second-stage selection required)")
    return candidates[0]


def action_to_mjai(action: int, ctx: ActionContext) -> dict:
    """Decode one action index into one concrete MJAI reaction event."""
    _check_action(action, ctx)

    if 0 <= action <= 36:
        tile = action_tile(action)
        assert tile is not None
        if ctx.phase is ActionPhase.KAN_SELECTION:
            candidates = ctx.kan_candidates(action)
            if not candidates:
                raise IllegalActionError(
                    f"kan selection {tile.text} is not in the kan candidates")
            if len(candidates) > 1:
                raise AmbiguousKanError(
                    f"kan selection {tile.text} matches {len(candidates)} candidates")
            return candidates[0].event(ctx.seat)
        if ctx.phase is not ActionPhase.SELF_TURN:
            raise IllegalActionError("tile action requires SELF_TURN or KAN_SELECTION")
        if tile not in ctx.hand:
            raise IllegalActionError(f"discard {tile.text} is not in the own hand")
        tsumogiri = ctx.last_self_tsumo is not None and ctx.last_self_tsumo == tile
        return {"type": "dahai", "actor": ctx.seat, "pai": tile.text,
                "tsumogiri": bool(tsumogiri)}

    if action == 37:
        return {"type": "reach", "actor": ctx.seat}

    if action in (38, 39, 40):
        if ctx.last_kawa_tile is None or ctx.last_kawa_actor is None:
            raise IllegalActionError("chi needs a last discard")
        options = dict(ctx.chi_options)
        consumed = options.get(action)
        if consumed is None:
            derived = derive_chi_options(ctx.hand, ctx.last_kawa_tile)
            consumed = derived.get(action)
        if consumed is None:
            raise IllegalActionError("chi tiles are not available in hand")
        return {"type": "chi", "actor": ctx.seat, "target": ctx.last_kawa_actor,
                "pai": ctx.last_kawa_tile.text,
                "consumed": [tile.text for tile in consumed]}

    if action == 41:
        if ctx.last_kawa_tile is None or ctx.last_kawa_actor is None:
            raise IllegalActionError("pon needs a last discard")
        consumed = derive_pon_consumed(ctx.hand, ctx.last_kawa_tile)
        if consumed is None:
            raise IllegalActionError("pon needs two matching tiles in hand")
        return {"type": "pon", "actor": ctx.seat, "target": ctx.last_kawa_actor,
                "pai": ctx.last_kawa_tile.text,
                "consumed": [tile.text for tile in consumed]}

    if action == 42:
        return _pick_kan(ctx, action).event(ctx.seat)

    if action == 43:
        if ctx.phase is ActionPhase.SELF_TURN:
            if not ctx.can_tsumo_agari:
                raise IllegalActionError("agari in SELF_TURN is not a tsumo win")
            return {"type": "hora", "actor": ctx.seat, "target": ctx.seat}
        if not ctx.can_ron_agari or ctx.ron_target is None:
            raise IllegalActionError("agari in CALL_RESPONSE is not a ron win")
        return {"type": "hora", "actor": ctx.seat, "target": ctx.ron_target}

    if action == 44:
        if not ctx.can_ryukyoku:
            raise IllegalActionError("ryukyoku is not available")
        return {"type": "ryukyoku"}

    if action == 45:
        return {"type": "none"}

    raise IllegalActionError(f"unhandled action {action}")


def select_kan_option(options: Sequence[KanOption], tile: TileRef | None = None) -> KanOption:
    """Explicit second-stage selection for a multi-kan decision."""
    if not options:
        raise IllegalActionError("no kan candidates")
    if tile is not None:
        matched = [option for option in options if option.tile == tile]
        if len(matched) != 1:
            raise AmbiguousKanError(
                f"kan tile {tile.text} matches {len(matched)} candidates")
        return matched[0]
    if len(options) > 1:
        raise AmbiguousKanError("several kan candidates; pass an explicit tile")
    return options[0]


def mjai_to_action(event: dict, phase: ActionPhase = ActionPhase.SELF_TURN) -> int:
    """Inverse mapping used for replay checks and tests (never returns None)."""
    if not isinstance(event, dict):
        raise AdapterError(f"event must be dict, got {type(event).__name__}")
    kind = event.get("type")
    if kind == "none":
        return 45
    if kind == "ryukyoku":
        return 44
    if kind == "hora":
        return 43
    if kind == "reach":
        return 37
    if kind == "dahai":
        return action_for_tile(tile_from_text(event["pai"]))
    if kind == "chi":
        consumed = [tile_from_text(text) for text in event["consumed"]]
        called = tile_from_text(event["pai"])
        low, high = sorted(tile.kind for tile in consumed)
        if called.kind < low:
            return 38
        if called.kind < high:
            return 39
        return 40
    if kind == "pon":
        return 41
    if kind in ("daiminkan", "kakan", "ankan"):
        if phase is ActionPhase.KAN_SELECTION and kind == "ankan":
            # PR-10 labels an ankan selection by tile kind (0..33), not aka.
            return int(tile_from_text(event["consumed"][0]).kind)
        return 42
    raise AdapterError(f"unsupported MJAI event type {kind!r}")


def _hand_from_decision(decision: DecisionInput) -> tuple[TileRef, ...]:
    tiles: list[TileRef] = []
    tile_state = decision.tile_state
    for kind in range(34):
        normal = int(tile_state.own_counts[kind])
        aka = 0
        if kind in AKA_KINDS:
            aka = int(tile_state.own_aka[AKA_KINDS.index(kind)])
        normal -= aka
        if normal < 0:
            raise AdapterError(f"own_aka exceeds own count for kind {kind}")
        tiles.extend([TileRef(kind, None)] * normal)
        if aka:
            tiles.extend([TileRef(kind, AKA_KINDS.index(kind))] * aka)
    return tuple(tiles)


def _kan_options_from_decision(decision: DecisionInput, hand: tuple[TileRef, ...],
                               last_kawa: DahaiEvent | None,
                               seat: int) -> tuple[KanOption, ...]:
    options: list[KanOption] = []
    counts = _counts(hand)
    if decision.phase is not ActionPhase.CALL_RESPONSE:
        for kind, count in sorted(counts.items()):
            if count == 4:
                tiles = tuple(tile for tile in hand if tile.kind == kind)
                representative = next((tile for tile in tiles if tile.aka is None), tiles[0])
                options.append(KanOption("ankan", representative, tiles))
        for meld in decision.players_public[0].melds:
            if meld.meld_kind is not MeldKind.PON:
                continue
            matching = [tile for tile in hand if tile.kind == meld.tiles[0].kind]
            if matching:
                options.append(KanOption("kakan", matching[0], tuple(meld.tiles)))
    elif last_kawa is not None:
        kind = last_kawa.tile.kind
        matching = [tile for tile in hand if tile.kind == kind]
        if len(matching) == 3:
            options.append(KanOption("daiminkan", matching[0], tuple(matching),
                                     target=(seat + last_kawa.actor_rel) % 4))
    return tuple(options)


def context_from_decision(decision: DecisionInput, seat: int = 0) -> ActionContext:
    """Build the adapter context from one PR-10 ``DecisionInput``.

    ``seat`` is the absolute seat of the POV (0..3); relative ``actor_rel`` values
    in the visible prefix are converted to absolute MJAI actors for the events.
    """
    if not isinstance(decision, DecisionInput):
        raise AdapterError("expected DecisionInput")
    hand = _hand_from_decision(decision)
    last_dahai = None
    last_self_tsumo = None
    for event in reversed(decision.events):
        if last_dahai is None and isinstance(event, DahaiEvent):
            last_dahai = event
        if last_self_tsumo is None and isinstance(event, TsumoEvent) and event.actor_rel == 0:
            last_self_tsumo = event.tile
        if last_dahai is not None and last_self_tsumo is not None:
            break
    chi_options = ()
    if (decision.phase is ActionPhase.CALL_RESPONSE and last_dahai is not None
            and last_dahai.actor_rel == 3):
        chi_options = tuple(sorted(derive_chi_options(hand, last_dahai.tile).items()))
    kan_options = _kan_options_from_decision(decision, hand, last_dahai, seat)
    mask = tuple(bool(value) for value in decision.legal_mask)
    target = (seat + last_dahai.actor_rel) % 4 if last_dahai is not None else None
    return ActionContext(
        seat=seat,
        phase=decision.phase,
        legal_mask=mask,
        hand=hand,
        last_kawa_tile=last_dahai.tile if last_dahai is not None else None,
        last_kawa_actor=target,
        last_self_tsumo=last_self_tsumo,
        chi_options=chi_options,
        kan_options=kan_options,
        can_tsumo_agari=bool(mask[43] and decision.phase is ActionPhase.SELF_TURN),
        can_ron_agari=bool(mask[43] and decision.phase is ActionPhase.CALL_RESPONSE),
        ron_target=target,
        can_ryukyoku=bool(mask[44]),
    )


__all__ = [
    "ActionContext", "AdapterError", "AmbiguousKanError", "IllegalActionError",
    "KanOption", "action_for_tile", "action_tile", "action_to_mjai",
    "context_from_decision", "derive_chi_options", "derive_pon_consumed",
    "mjai_to_action", "select_kan_option",
]
