#!/usr/bin/env python3
"""Engine-neutral, POV-safe observation for the local self-play bots.

A :class:`SeatView` is everything one seat may legally see at one decision point:

* its own physical hand (including the drawn tile) and its own melds;
* the public rivers, calls, riichi declarations, dora indicators and scores;
* the frozen 46-action mask and its phase.

It deliberately has **no** field for another seat's concealed hand, no wall order and
no future event.  Both rollout backends (``libriichi.arena.OneVsThree.py_vs_py`` and the
procedural fallback) build the same object, so a bot cannot behave differently on a
hidden-information leak.  The libriichi view builder only parses a whitelisted set of
public MJAI event types and never reads ``start_kyoku.tehais`` or ``hora.ura_markers``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas.action_v3 import (
    ACTION_SPACE,
    PHASE_LEGAL_ACTIONS,
    ActionPhase,
)
from ..schemas.event_v3 import TileRef
from .tiles import (
    AKA_KINDS,
    TILE_KIND_COUNT,
    physical_hand,
    tile_text,
)

__all__ = [
    "SeatView",
    "ViewError",
    "empty_visible_counts",
    "build_scores",
]

SOURCE_LIBRIICHI = "libriichi"
SOURCE_SYNTHETIC = "synthetic"


class ViewError(ValueError):
    """The view is inconsistent with the frozen action contract (fail closed)."""


def empty_visible_counts() -> list[int]:
    return [0] * TILE_KIND_COUNT


def _check_counts(values, name: str) -> tuple[int, ...]:
    values = tuple(values)
    if len(values) != TILE_KIND_COUNT:
        raise ViewError(f"{name} must have {TILE_KIND_COUNT} entries")
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
            raise ViewError(f"{name}[{index}] must be an int 0..4")
    return values


def _check_tile(tile, name: str) -> TileRef | None:
    if tile is None:
        return None
    if not isinstance(tile, TileRef):
        raise ViewError(f"{name} must be a TileRef or None")
    if tile.is_unknown:
        raise ViewError(f"{name} must be a known tile")
    return tile


@dataclass(frozen=True)
class SeatView:
    """One POV decision snapshot (public events + own hand only)."""

    seat: int
    hand_counts: tuple[int, ...]
    hand_akas: tuple[int, int, int]
    drawn: TileRef | None
    legal_mask: tuple[bool, ...]
    phase: ActionPhase

    meld_count: int = 0
    melds: tuple[tuple[TileRef, ...], ...] = ()
    rivers: tuple[tuple[TileRef, ...], ...] = ()
    riichi: tuple[bool, bool, bool, bool] = (False, False, False, False)
    open_melds: tuple[int, int, int, int] = (0, 0, 0, 0)
    scores: tuple[int, int, int, int] = (25000, 25000, 25000, 25000)
    dora_indicators: tuple[TileRef, ...] = ()
    visible_counts: tuple[int, ...] = (0,) * TILE_KIND_COUNT
    last_kawa: TileRef | None = None
    last_kawa_actor: int | None = None
    ankan_candidates: tuple[int, ...] = ()
    kakan_candidates: tuple[tuple[int, int | None], ...] = ()
    daiminkan_tile: tuple[int, int | None] | None = None

    kyoku: int = 1
    honba: int = 0
    kyotaku: int = 0
    tiles_left: int = 70
    self_wind: int = 27
    oya: int = 0

    source: str = SOURCE_SYNTHETIC
    ruleset_id: str = "synthetic_abstract_hanchan/1"
    turn: int = 0

    # -- constructor checks ------------------------------------------------
    def __post_init__(self):
        if isinstance(self.seat, bool) or not isinstance(self.seat, int) or not 0 <= self.seat <= 3:
            raise ViewError(f"seat must be an int 0..3, got {self.seat!r}")
        counts = _check_counts(self.hand_counts, "hand_counts")
        akas = tuple(self.hand_akas)
        if len(akas) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 1
                                 for v in akas):
            raise ViewError("hand_akas must be three 0/1 entries")
        for color, kind in enumerate(AKA_KINDS):
            if akas[color] > counts[kind]:
                raise ViewError(f"aka count {akas[color]} exceeds copies of kind {kind}")
        mask = tuple(self.legal_mask)
        if len(mask) != ACTION_SPACE or not all(isinstance(value, bool) for value in mask):
            raise ViewError("legal_mask must be a tuple of 46 bools")
        if not isinstance(self.phase, ActionPhase):
            raise ViewError(f"phase must be ActionPhase, got {self.phase!r}")
        allowed = PHASE_LEGAL_ACTIONS[self.phase]
        for index, value in enumerate(mask):
            if value and index not in allowed:
                raise ViewError(
                    f"action {index} is masked legal but not legal in phase {self.phase.value}")
        _check_tile(self.drawn, "drawn")
        _check_tile(self.last_kawa, "last_kawa")
        if self.last_kawa_actor is not None and self.last_kawa_actor not in (0, 1, 2, 3):
            raise ViewError("last_kawa_actor must be 0..3 or None")
        if len(self.rivers) != 4:
            raise ViewError("rivers must contain four seats")
        _check_counts(self.visible_counts, "visible_counts")
        if len(self.scores) != 4 or len(self.riichi) != 4 or len(self.open_melds) != 4:
            raise ViewError("scores/riichi/open_melds must contain four seats")
        if mask[42] and not self.kan_choices():
            raise ViewError(
                "action 42 is masked legal but the view carries no concrete kan candidate")

    # -- hand helpers ------------------------------------------------------
    @property
    def hand_size(self) -> int:
        return sum(self.hand_counts)

    @property
    def own_tiles(self) -> tuple[TileRef, ...]:
        return physical_hand(self.hand_counts, self.hand_akas)

    def discard_candidates(self) -> tuple[tuple[int, TileRef], ...]:
        """Legal discard actions (0..36) with the physical tile they would discard."""
        tiles = list(self.own_tiles)
        result: list[tuple[int, TileRef]] = []
        for tile in tiles:
            action = 34 + tile.aka if tile.aka is not None else tile.kind
            if self.legal_mask[action]:
                result.append((action, tile))
        return tuple(result)

    def counts_after_discard(self, tile: TileRef) -> tuple[int, ...]:
        counts = list(self.hand_counts)
        if counts[tile.kind] <= 0:
            raise ViewError(f"cannot discard {tile.text}: not in hand")
        counts[tile.kind] -= 1
        return tuple(counts)

    def remaining_kind(self, kind: int) -> int:
        return max(0, 4 - self.visible_counts[kind])

    def is_legal(self, action: int) -> bool:
        return (isinstance(action, int) and not isinstance(action, bool)
                and 0 <= action < ACTION_SPACE and bool(self.legal_mask[action]))

    def legal_actions(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.legal_mask) if value)

    def tile_text(self, tile: TileRef) -> str:
        return tile_text(tile.kind, tile.aka)

    def kan_choices(self) -> tuple[tuple[str, int, int | None], ...]:
        """Concrete kan candidates when action 42 is legal (from the native state)."""
        choices: list[tuple[str, int, int | None]] = []
        for kind in self.ankan_candidates:
            choices.append(("ankan", kind, None))
        for kind, aka in self.kakan_candidates:
            choices.append(("kakan", kind, aka))
        if self.daiminkan_tile is not None:
            kind, aka = self.daiminkan_tile
            choices.append(("daiminkan", kind, aka))
        return tuple(choices)

    def opponents_with_riichi(self) -> tuple[int, ...]:
        return tuple(seat for seat in range(4) if seat != self.seat and self.riichi[seat])

    def public_visible_counts(self) -> tuple[int, ...]:
        return tuple(self.visible_counts)

    def as_dict(self) -> dict:
        return {
            "seat": self.seat,
            "hand_counts": list(self.hand_counts),
            "hand_akas": list(self.hand_akas),
            "drawn": self.drawn.text if self.drawn else None,
            "meld_count": self.meld_count,
            "riichi": list(self.riichi),
            "scores": list(self.scores),
            "kyoku": self.kyoku,
            "honba": self.honba,
            "kyotaku": self.kyotaku,
            "tiles_left": self.tiles_left,
            "phase": self.phase.value,
            "legal_actions": list(self.legal_actions()),
            "source": self.source,
            "ruleset_id": self.ruleset_id,
        }


def build_scores(start_scores, riichi_seats) -> tuple[int, int, int, int]:
    """Public scores from a ``start_kyoku`` snapshot minus declared riichi sticks."""
    scores = list(start_scores)
    if len(scores) != 4:
        raise ViewError("start_scores must have four entries")
    for seat in riichi_seats:
        scores[seat] -= 1000
    return tuple(scores)
