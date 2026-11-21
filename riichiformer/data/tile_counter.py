#!/usr/bin/env python3
"""PR-10 public/own physical-tile ledger.

The counter is the single place that migrates a called tile from the river to a meld.
It never infers hidden opponent hand tiles and never double-counts a called tile.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas.action_v3 import (
    AKA_KINDS,
    AKA_PER_COLOR,
    TILE_KIND_COUNT,
)
from ..schemas.decision_v3 import RulesetId, VisibleTileState
from ..schemas.event_v3 import TileRef


class TileCounterError(ValueError):
    """The physical ledger is impossible under the ruleset."""


def _check_tile(tile: TileRef, *, known: bool = True) -> TileRef:
    if not isinstance(tile, TileRef):
        raise TileCounterError(f"expected TileRef, got {type(tile).__name__}")
    if tile.is_unknown:
        if known:
            raise TileCounterError("UNKNOWN cannot enter the physical ledger")
        return tile
    if not 0 <= tile.kind < TILE_KIND_COUNT:
        raise TileCounterError(f"tile kind {tile.kind} outside 0..33")
    if tile.is_aka and tile.aka not in (0, 1, 2):
        raise TileCounterError(f"invalid aka colour {tile.aka!r}")
    return tile


@dataclass
class TileCounter:
    """Physical tiles visible to one POV, split into own hand and public zones."""

    ruleset_id: RulesetId = RulesetId.TENHOU_4P_RANKED

    def __post_init__(self):
        if not isinstance(self.ruleset_id, RulesetId):
            raise TileCounterError("ruleset_id must be RulesetId")
        self.own_counts = [0] * TILE_KIND_COUNT
        self.own_aka = [0, 0, 0]
        self.river = [0] * TILE_KIND_COUNT
        self.river_aka = [0, 0, 0]
        self.meld = [0] * TILE_KIND_COUNT
        self.meld_aka = [0, 0, 0]
        self.indicator = [0] * TILE_KIND_COUNT
        self.indicator_aka = [0, 0, 0]
        self.other_public = [0] * TILE_KIND_COUNT
        self.other_public_aka = [0, 0, 0]

    # -- low-level helpers -------------------------------------------------
    def _snapshot(self):
        return (
            self.own_counts[:], self.own_aka[:],
            self.river[:], self.river_aka[:],
            self.meld[:], self.meld_aka[:],
            self.indicator[:], self.indicator_aka[:],
            self.other_public[:], self.other_public_aka[:],
        )

    def _restore(self, snapshot) -> None:
        (self.own_counts, self.own_aka, self.river, self.river_aka,
         self.meld, self.meld_aka, self.indicator, self.indicator_aka,
         self.other_public, self.other_public_aka) = snapshot

    def _add(self, counts, akas, tile: TileRef) -> None:
        tile = _check_tile(tile)
        snapshot = self._snapshot()
        counts[tile.kind] += 1
        if tile.is_aka:
            akas[tile.aka] += 1
        try:
            self.validate()
        except TileCounterError:
            self._restore(snapshot)
            raise

    def _remove(self, counts, akas, tile: TileRef, zone: str) -> None:
        tile = _check_tile(tile)
        if counts[tile.kind] <= 0:
            raise TileCounterError(f"{zone} has no tile kind {tile.kind} to remove")
        if tile.is_aka and akas[tile.aka] <= 0:
            raise TileCounterError(f"{zone} has no aka colour {tile.aka} to remove")
        snapshot = self._snapshot()
        counts[tile.kind] -= 1
        if tile.is_aka:
            akas[tile.aka] -= 1
        try:
            self.validate()
        except TileCounterError:
            self._restore(snapshot)
            raise

    # -- own hand ----------------------------------------------------------
    def add_own(self, tile: TileRef) -> None:
        self._add(self.own_counts, self.own_aka, tile)

    def remove_own(self, tile: TileRef) -> None:
        self._remove(self.own_counts, self.own_aka, tile, "own hand")

    # -- public zones ------------------------------------------------------
    def add_river(self, tile: TileRef) -> None:
        self._add(self.river, self.river_aka, tile)

    def remove_river(self, tile: TileRef) -> None:
        self._remove(self.river, self.river_aka, tile, "river")

    def add_meld(self, tile: TileRef) -> None:
        self._add(self.meld, self.meld_aka, tile)

    def add_indicator(self, tile: TileRef) -> None:
        self._add(self.indicator, self.indicator_aka, tile)

    def add_other_public(self, tile: TileRef) -> None:
        self._add(self.other_public, self.other_public_aka, tile)

    def call_tile(self, tile: TileRef) -> None:
        """Move a called tile river -> meld exactly once (chi/pon/daiminkan)."""
        self.remove_river(tile)
        self.add_meld(tile)

    # -- derived views -----------------------------------------------------
    @property
    def public_counts(self) -> tuple[int, ...]:
        return tuple(
            self.river[k] + self.meld[k] + self.indicator[k] + self.other_public[k]
            for k in range(TILE_KIND_COUNT)
        )

    @property
    def public_aka(self) -> tuple[int, int, int]:
        return tuple(
            self.river_aka[c] + self.meld_aka[c]
            + self.indicator_aka[c] + self.other_public_aka[c]
            for c in range(3)
        )

    def zone_counts(self, zone: str) -> tuple[int, ...]:
        if zone == "river":
            return tuple(self.river)
        if zone == "meld":
            return tuple(
                self.meld[k] + self.other_public[k] for k in range(TILE_KIND_COUNT)
            )
        if zone == "dora_indicator":
            return tuple(self.indicator)
        raise TileCounterError(f"unknown public zone {zone!r}")

    # -- invariants --------------------------------------------------------
    def validate(self) -> None:
        for name, counts in (
            ("own", self.own_counts),
            ("river", self.river),
            ("meld", self.meld),
            ("indicator", self.indicator),
            ("other_public", self.other_public),
        ):
            for kind, count in enumerate(counts):
                if count < 0 or count > 4:
                    raise TileCounterError(f"{name} count for kind {kind} is {count}")
        for kind in range(TILE_KIND_COUNT):
            if self.own_counts[kind] + self.public_counts[kind] > 4:
                raise TileCounterError(
                    f"physical overflow for kind {kind}: "
                    f"own={self.own_counts[kind]} public={self.public_counts[kind]}"
                )
        for color, kind in enumerate(AKA_KINDS):
            own = self.own_aka[color]
            public = self.public_aka[color]
            if own < 0 or public < 0 or own > AKA_PER_COLOR or public > AKA_PER_COLOR:
                raise TileCounterError(f"aka colour {color} out of bounds")
            if own + public > AKA_PER_COLOR:
                raise TileCounterError(
                    f"more than {AKA_PER_COLOR} aka five of colour {color}"
                )
            if own > self.own_counts[kind]:
                raise TileCounterError(f"own aka colour {color} exceeds own five count")
            if public > self.public_counts[kind]:
                raise TileCounterError(
                    f"public aka colour {color} exceeds public five count"
                )
        if sum(self.indicator) > 5:
            raise TileCounterError("more than 5 dora indicators")

    def assert_conservation(self) -> None:
        self.validate()

    def to_visible_tile_state(
        self,
        dora_indicators: tuple[TileRef, ...],
        *,
        ruleset_id: RulesetId | None = None,
    ) -> VisibleTileState:
        self.validate()
        if len(dora_indicators) != sum(self.indicator):
            raise TileCounterError("dora indicator count does not match the ledger")
        exact = [0] * TILE_KIND_COUNT
        for marker in dora_indicators:
            if not isinstance(marker, TileRef) or marker.is_unknown:
                raise TileCounterError("dora_indicators must be known TileRefs")
            exact[marker.kind] += 1
        if exact != self.indicator:
            raise TileCounterError("dora indicator kinds do not match the ledger")
        state = VisibleTileState(
            own_counts=tuple(self.own_counts),
            own_aka=tuple(self.own_aka),
            public_counts=self.public_counts,
            public_aka=self.public_aka,
            public_counts_by_zone=(
                self.zone_counts("river"),
                self.zone_counts("meld"),
                self.zone_counts("dora_indicator"),
            ),
            dora_indicators=tuple(dora_indicators),
            ruleset_id=ruleset_id or self.ruleset_id,
        )
        return state

    def as_dict(self) -> dict:
        return {
            "own_counts": list(self.own_counts),
            "own_aka": list(self.own_aka),
            "public_counts": list(self.public_counts),
            "public_aka": list(self.public_aka),
            "public_counts_by_zone": {
                "river": list(self.river),
                "meld": [
                    self.meld[k] + self.other_public[k]
                    for k in range(TILE_KIND_COUNT)
                ],
                "dora_indicator": list(self.indicator),
            },
            "ruleset_id": self.ruleset_id.value,
        }


__all__ = ["TileCounter", "TileCounterError"]
