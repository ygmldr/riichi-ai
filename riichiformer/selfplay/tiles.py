#!/usr/bin/env python3
"""Tile helpers shared by the local self-play bots and the rollout bridges.

The 34-kind order and the MJAI text form are the frozen ones from
:mod:`riichiformer.schemas.action_v3` / :mod:`riichiformer.schemas.event_v3`; this
module only adds the small count-array helpers the bots need.  Nothing here decodes an
opponent concealed tile.
"""

from __future__ import annotations

from ..schemas.action_v3 import (
    AKA_COLORS,
    AKA_KINDS,
    AKA_PER_COLOR,
    TILE_KIND_COUNT,
    TILE_KIND_NAMES,
)
from ..schemas.event_v3 import TileRef, tile_from_text

__all__ = [
    "AKA_COLORS",
    "AKA_KINDS",
    "AKA_PER_COLOR",
    "TILE_KIND_COUNT",
    "TILE_KIND_NAMES",
    "TileRef",
    "empty_counts",
    "counts_from_tiles",
    "kind_from_text",
    "aka_from_text",
    "text_from_kind",
    "tile_text",
    "physical_hand",
    "add_tile",
    "remove_tile",
    "is_aka_kind",
]

_TILE_TEXT_TO_KIND = {name: index for index, name in enumerate(TILE_KIND_NAMES)}


def empty_counts() -> list[int]:
    return [0] * TILE_KIND_COUNT


def kind_from_text(text: str) -> int:
    return tile_from_text(text).kind


def aka_from_text(text: str) -> int | None:
    return tile_from_text(text).aka


def text_from_kind(kind: int) -> str:
    if isinstance(kind, bool) or not isinstance(kind, int) or not 0 <= kind < TILE_KIND_COUNT:
        raise ValueError(f"tile kind must be an int 0..33, got {kind!r}")
    return TILE_KIND_NAMES[kind]


def tile_text(kind: int, aka: int | None = None) -> str:
    if aka is None:
        return text_from_kind(kind)
    if kind not in AKA_KINDS or AKA_KINDS.index(kind) != aka:
        raise ValueError(f"kind {kind} cannot carry aka colour {aka!r}")
    return text_from_kind(kind) + "r"


def is_aka_kind(kind: int) -> bool:
    return kind in AKA_KINDS


def counts_from_tiles(tiles) -> tuple[list[int], list[int]]:
    counts = empty_counts()
    akas = [0, 0, 0]
    for tile in tiles:
        if not isinstance(tile, TileRef) or tile.is_unknown:
            raise ValueError(f"expected known TileRef, got {tile!r}")
        counts[tile.kind] += 1
        if tile.aka is not None:
            akas[tile.aka] += 1
    return counts, akas


def physical_hand(counts, akas) -> tuple[TileRef, ...]:
    """Expand a 34-count + aka ledger into physical tile identities.

    Normal copies come before the aka copy of the same kind, which is the same order
    :mod:`riichiformer.runtime.mjai_adapter` uses when it picks consumed tiles.
    """
    counts = list(counts)
    akas = list(akas)
    if len(counts) != TILE_KIND_COUNT:
        raise ValueError("counts must have 34 entries")
    if len(akas) != 3:
        raise ValueError("akas must have 3 entries")
    tiles: list[TileRef] = []
    for kind in range(TILE_KIND_COUNT):
        aka = 0
        if kind in AKA_KINDS:
            color = AKA_KINDS.index(kind)
            aka = akas[color]
        normal = counts[kind] - aka
        if normal < 0:
            raise ValueError(f"kind {kind} has {counts[kind]} copies but {aka} aka copies")
        for _ in range(normal):
            tiles.append(TileRef(kind))
        for _ in range(aka):
            tiles.append(TileRef(kind, color))
    return tuple(tiles)


def add_tile(counts, kind: int, aka: int | None = None) -> tuple[list[int], int | None]:
    values = list(counts)
    values[kind] += 1
    return values, aka


def remove_tile(counts, kind: int) -> list[int]:
    values = list(counts)
    if values[kind] <= 0:
        raise ValueError(f"kind {kind} has no copy to remove")
    values[kind] -= 1
    return values
