#!/usr/bin/env python3
"""Local heuristic opponents for RiichiFormer self-play (PR-41/PR-42).

Three dependency-free, weight-free bots are provided:

``legal-random``
    Uniform over the frozen 46-action mask.  It is the weak control opponent: it calls,
    declares riichi and even wins when the mask allows it, so it exercises a wide slice
    of the legality surface without any strategy.

``shanten-ukeire``
    Closed-hand shanten reduction: for every legal discard it computes the exact
    shanten of the resulting 13-tile hand and keeps the minimum.  Ties at tenpai are
    broken by the exact number of remaining winning tiles; ties above tenpai use a
    documented shape heuristic (pairs / adjacent / kanchan) instead of a full ukeire
    enumeration.  It declares riichi whenever the mask allows it and keeps tenpai.

``defense``
    "shanten-ukeire" plus a threat check: against a declared riichi it only pushes when
    already tenpai, otherwise it plays the least dangerous legal tile (genbutsu first,
    then honours/terminals, suji-reduced middle tiles).  It does not declare riichi
    while a threat is live.

Every bot validates its own choice against the native 46-action mask *and* the phase
before returning; :class:`IllegalBotAction` is a bug that the rollout bridges count and
fail on.  No bot loads a checkpoint, calls the network, or uses a third-party weight.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

from ..schemas.action_v3 import ActionPhase
from ..schemas.event_v3 import TileRef
from .shanten import TERMINAL_HONOR_KINDS, shanten, waits
from .tiles import TILE_KIND_COUNT
from .view import SeatView

__all__ = [
    "BOT_KINDS",
    "Bot",
    "BotAction",
    "BotError",
    "IllegalBotAction",
    "LegalRandomBot",
    "SimpleShantenUkeireBot",
    "DefenseBot",
    "BOT_FACTORIES",
    "make_bot",
    "bot_specs",
    "default_lineup",
]

BOT_KINDS = ("heuristic_random", "heuristic_shanten", "heuristic_defense")

BOT_VERSION = "1"

#: Documented licence caveat attached to every local heuristic bot.
LOCAL_BOT_LICENSE = (
    "locally authored heuristic bot; no model weights, no third-party log or "
    "checkpoint; project-private research output only; not Mortal and not a Tenhou "
    "rank claim"
)


class BotError(RuntimeError):
    """Base class for self-play bot failures."""


class IllegalBotAction(BotError):
    """A bot returned an action outside the native 46-action mask."""


@dataclass(frozen=True)
class BotAction:
    """One 46-action choice plus an optional concrete kan selection."""

    index: int
    kan: tuple[str, int, int | None] | None = None
    label: str = ""

    def as_tuple(self) -> dict:
        return {
            "action": self.index,
            "kan": None if self.kan is None else {
                "meld_type": self.kan[0],
                "kind": self.kan[1],
                "aka": self.kan[2],
            },
            "label": self.label,
        }


class Bot:
    """Base class: subclasses implement :meth:`_decide`, never bypass legality."""

    name = "bot"
    kind = "heuristic"
    version = BOT_VERSION
    license_caveat = LOCAL_BOT_LICENSE

    def __init__(self, seed: int = 0):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise BotError(f"seed must be a non-negative int, got {seed!r}")
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.decisions = 0

    # -- public API --------------------------------------------------------
    def act(self, view: SeatView) -> BotAction:
        if not isinstance(view, SeatView):
            raise BotError("act() expects a SeatView")
        action = self._decide(view)
        if not isinstance(action, BotAction):
            raise BotError(f"{type(self).__name__} returned {type(action).__name__}")
        if not view.is_legal(action.index):
            raise IllegalBotAction(
                f"{self.name} returned action {action.index} outside the native mask "
                f"(phase {view.phase.value}, legal {view.legal_actions()})")
        self.decisions += 1
        return action

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.decisions = 0

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _first_legal(view: SeatView, preferred: Sequence[int]) -> int | None:
        for action in preferred:
            if view.is_legal(action):
                return action
        return None

    @staticmethod
    def _pass_or_first(view: SeatView) -> BotAction:
        if view.is_legal(45):
            return BotAction(45, label="pass")
        legal = view.legal_actions()
        if not legal:
            raise IllegalBotAction("empty legal mask")
        return BotAction(legal[0], label="first-legal")

    @staticmethod
    def _best_tenpai_discards(view: SeatView):
        scored = []
        for action, tile in view.discard_candidates():
            after = view.counts_after_discard(tile)
            value = shanten(after, view.meld_count)
            scored.append((value, action, tile, after))
        return scored

    def _decide(self, view: SeatView) -> BotAction:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# legal random
# ---------------------------------------------------------------------------
class LegalRandomBot(Bot):
    name = "legal-random"
    kind = "heuristic_random"

    def _decide(self, view: SeatView) -> BotAction:
        legal = view.legal_actions()
        if not legal:
            raise IllegalBotAction("empty legal mask")
        action = self.rng.choice(legal)
        if action == 42:
            choices = view.kan_choices()
            if choices:
                return BotAction(42, self.rng.choice(choices), label="kan-random")
        return BotAction(action, label="random")


# ---------------------------------------------------------------------------
# shanten / ukeire
# ---------------------------------------------------------------------------
def _shape_score(counts) -> int:
    """Cheap documented tie-break for non-tenpai hands (not a full ukeire count)."""
    score = 0
    for kind in range(TILE_KIND_COUNT):
        copies = counts[kind]
        if copies >= 2:
            score += 4
        if copies >= 3:
            score += 6
    for base in (0, 9, 18):
        for offset in range(8):
            if counts[base + offset] and counts[base + offset + 1]:
                score += 2
        for offset in range(7):
            if counts[base + offset] and counts[base + offset + 2]:
                score += 1
    for kind in TERMINAL_HONOR_KINDS:
        if counts[kind] == 1 and kind >= 27:
            score -= 1
    return score


class SimpleShantenUkeireBot(Bot):
    name = "shanten-ukeire"
    kind = "heuristic_shanten"

    def _decide(self, view: SeatView) -> BotAction:
        if view.is_legal(43):
            return BotAction(43, label="agari")
        if view.phase is ActionPhase.CALL_RESPONSE:
            # Closed-hand style: never call; pass is always legal on a call.
            return self._pass_or_first(view)
        if not view.discard_candidates():
            return self._pass_or_first(view)

        ranked = self._rank_discards(view)
        best = ranked[0]
        if view.is_legal(37) and best[0] == 0:
            return BotAction(37, label="riichi")
        return BotAction(best[1], label=best[4])

    def _rank_discards(self, view: SeatView):
        ranked = []
        for action, tile in view.discard_candidates():
            after = view.counts_after_discard(tile)
            value = shanten(after, view.meld_count)
            if value == 0:
                exact = sum(view.remaining_kind(kind) for kind in waits(after, view.meld_count))
                score = exact
                label = f"tenpai-discard wait_tiles={exact}"
            else:
                score = _shape_score(after)
                label = f"shanten={value} shape={score}"
            ranked.append((value, action, tile, -score, label))
        ranked.sort()
        return ranked


# ---------------------------------------------------------------------------
# defense
# ---------------------------------------------------------------------------
class DefenseBot(SimpleShantenUkeireBot):
    name = "defense"
    kind = "heuristic_defense"

    #: tile number groups used for a coarse danger estimate
    _DANGER_BY_NUMBER = {1: 2, 9: 2, 2: 3, 8: 3, 3: 4, 7: 4, 4: 5, 5: 6, 6: 5}

    def _tile_danger(self, view: SeatView, tile: TileRef, threats: Sequence[int]) -> int:
        danger = 0
        for opponent in threats:
            river = view.rivers[opponent]
            if any(called.kind == tile.kind for called in river):
                continue  # genbutsu: already discarded by that riichi player
            if tile.kind >= 27:
                danger += 1
                continue
            number = tile.kind % 9 + 1
            value = self._DANGER_BY_NUMBER[number]
            suji = 0
            for offset in (-3, 3):
                if 0 <= tile.kind + offset < 27 and (tile.kind + offset) // 9 == tile.kind // 9:
                    if any(called.kind == tile.kind + offset for called in river):
                        suji += 1
            danger += max(1, value - suji)
        return danger

    def _decide(self, view: SeatView) -> BotAction:
        if view.is_legal(43):
            return BotAction(43, label="agari")
        if view.phase is ActionPhase.CALL_RESPONSE:
            return self._pass_or_first(view)
        threats = view.opponents_with_riichi()
        if not view.discard_candidates():
            return self._pass_or_first(view)

        ranked = self._rank_discards(view)
        best_shanten = ranked[0][0]
        if threats and best_shanten > 0:
            best = min(
                ranked,
                key=lambda row: (
                    self._tile_danger(view, row[2], threats),
                    row[0],
                    row[1],
                ),
            )
            danger = self._tile_danger(view, best[2], threats)
            return BotAction(best[1], label=f"defense danger={danger} threats={list(threats)}")
        if not threats and view.is_legal(37) and best_shanten == 0:
            return BotAction(37, label="riichi")
        return BotAction(ranked[0][1], label=ranked[0][4])


# ---------------------------------------------------------------------------
# factories / metadata
# ---------------------------------------------------------------------------
BOT_FACTORIES: dict[str, Callable[..., Bot]] = {
    "legal-random": LegalRandomBot,
    "shanten-ukeire": SimpleShantenUkeireBot,
    "defense": DefenseBot,
}

DEFAULT_LINEUP = ("shanten-ukeire", "defense", "legal-random", "shanten-ukeire")


def make_bot(name: str, seed: int = 0) -> Bot:
    try:
        factory = BOT_FACTORIES[name]
    except KeyError:
        raise BotError(
            f"unknown local bot {name!r}; known: {sorted(BOT_FACTORIES)}") from None
    return factory(seed)


def default_lineup() -> tuple[str, str, str, str]:
    return DEFAULT_LINEUP


def bot_specs() -> list[dict]:
    """Registry rows for the local bots (``checkpoint_sha256`` is always ``None``)."""
    rows = []
    for name, factory in BOT_FACTORIES.items():
        rows.append({
            "name": name,
            "version": f"heuristic-v{BOT_VERSION}",
            "kind": factory.kind,
            "checkpoint_sha256": None,
            "license_caveat": LOCAL_BOT_LICENSE,
            "display_name": f"local {name} heuristic bot",
        })
    return rows

