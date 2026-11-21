#!/usr/bin/env python3
"""Pure-Python shanten / win / wait / ukeire evaluator for local self-play bots.

This is a *heuristic bot helper*, not a rules engine and not a Tenhou-parity claim.
It operates on the frozen 34-kind contract from :mod:`riichiformer.schemas.action_v3`
and never looks at opponent concealed tiles.

Definitions
-----------
``shanten(counts, open_melds=0)`` assumes a **3n+1** concealed hand (13 tiles when the
hand is closed, 13 - 3*open_melds when melds are open) and returns:

* ``-1`` for a complete hand,
* ``0`` for tenpai,
* ``n`` for n-shanten.

For a 14-tile (3n+2) hand callers pass the candidate 13-tile hands after a discard; a
14-tile hand is never treated as complete here.

The standard-form evaluator decomposes each suit with a cached left-to-right search and
combines the suits by dynamic programming.  Chiitoitsu and kokushi are checked only for
fully concealed hands.  A pure 14-tile win check (``is_winning_hand``) is provided for
exact tenpai wait counts.

The module is stdlib-only so the bots, the synthetic fallback and the offline tests all
run on CPU without torch or libriichi.
"""

from __future__ import annotations

from functools import lru_cache

from .tiles import TILE_KIND_COUNT, kind_from_text

__all__ = [
    "TERMINAL_HONOR_KINDS",
    "is_terminal_or_honor",
    "shanten",
    "standard_shanten",
    "chiitoi_shanten",
    "kokushi_shanten",
    "is_winning_hand",
    "shanten_14",
    "waits",
    "ukeire",
    "concealed_tile_count",
]

#: 1m, 9m, 1p, 9p, 1s, 9s, E, S, W, N, P, F, C
TERMINAL_HONOR_KINDS: tuple[int, ...] = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)
_INF = 1 << 20


def is_terminal_or_honor(kind: int) -> bool:
    return kind in TERMINAL_HONOR_KINDS


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _check_counts(counts, *, name: str = "counts") -> tuple[int, ...]:
    if isinstance(counts, (bytes, bytearray)):
        values = tuple(int(value) for value in counts)
    elif isinstance(counts, (tuple, list)):
        values = tuple(counts)
    else:
        raise TypeError(f"{name} must be a 34-length sequence, got {type(counts).__name__}")
    if len(values) != TILE_KIND_COUNT:
        raise ValueError(f"{name} must have {TILE_KIND_COUNT} entries, got {len(values)}")
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 4:
            raise ValueError(f"{name}[{index}] must be an int 0..4, got {value!r}")
    return values


def concealed_tile_count(counts) -> int:
    return sum(_check_counts(counts))


# ---------------------------------------------------------------------------
# standard form
# ---------------------------------------------------------------------------
# A suit state is (melds, partials, head): melds are complete runs/triplets, partials
# are two-tile groups (pairs included), head is 1 iff a pair has been reserved as the
# eyes.  ``melds + partials <= 4`` and at most one head are kept, which is exactly the
# standard-form bound for a 3n+1 hand.
_MAX_SETS = 4


def _merge_suit(states: set[tuple[int, int, int]], add_m: int, add_p: int, add_q: int):
    merged = set()
    for melds, partials, head in states:
        m = melds + add_m
        p = partials + add_p
        q = head + add_q
        if m > _MAX_SETS or p > _MAX_SETS or q > 1 or m + p > _MAX_SETS:
            continue
        merged.add((m, p, q))
    return merged


@lru_cache(maxsize=200_000)
def _suit_states(counts9: tuple[int, ...]) -> frozenset[tuple[int, int, int]]:
    """All achievable ``(melds, partials, head)`` states for one 9-kind suit."""
    counts = list(counts9)

    def walk(start: int) -> set[tuple[int, int, int]]:
        index = start
        while index < 9 and counts[index] == 0:
            index += 1
        if index >= 9:
            return {(0, 0, 0)}
        states: set[tuple[int, int, int]] = set()
        tile = counts[index]

        if tile >= 3:
            counts[index] -= 3
            states |= _merge_suit(walk(index), 1, 0, 0)
            counts[index] += 3

        if index <= 6 and counts[index + 1] and counts[index + 2]:
            counts[index] -= 1
            counts[index + 1] -= 1
            counts[index + 2] -= 1
            states |= _merge_suit(walk(index), 1, 0, 0)
            counts[index] += 1
            counts[index + 1] += 1
            counts[index + 2] += 1

        if tile >= 2:
            counts[index] -= 2
            sub = walk(index)
            states |= _merge_suit(sub, 0, 1, 0)
            states |= _merge_suit(sub, 0, 0, 1)
            counts[index] += 2

        if index <= 7 and counts[index + 1]:
            counts[index] -= 1
            counts[index + 1] -= 1
            states |= _merge_suit(walk(index), 0, 1, 0)
            counts[index] += 1
            counts[index + 1] += 1

        if index <= 6 and counts[index + 2]:
            counts[index] -= 1
            counts[index + 2] -= 1
            states |= _merge_suit(walk(index), 0, 1, 0)
            counts[index] += 1
            counts[index + 2] += 1

        # leave this tile isolated (or as the floating tile of a 3n+1 hand)
        counts[index] -= 1
        states |= walk(index)
        counts[index] += 1

        return states

    return frozenset(walk(0))


def _honor_states(counts7: tuple[int, ...]) -> frozenset[tuple[int, int, int]]:
    melds = partials = 0
    for count in counts7:
        if count >= 3:
            melds += 1
        elif count == 2:
            partials += 1
    states = {(melds, partials, 0)}
    if partials >= 1:
        states.add((melds, partials - 1, 1))
    return frozenset((m, p, q) for m, p, q in states if m + p <= _MAX_SETS)


def _combine(components, open_melds: int) -> int:
    # state grid: (melds, partials, head) reachable
    states: set[tuple[int, int, int]] = {(0, 0, 0)}
    for component in components:
        merged = set()
        for melds, partials, head in states:
            for add_m, add_p, add_q in component:
                m = melds + add_m
                p = partials + add_p
                q = head + add_q
                if m + open_melds > _MAX_SETS or m + open_melds + p > _MAX_SETS:
                    continue
                if q > 1:
                    continue
                merged.add((m, p, q))
        states = merged
        if not states:  # pragma: no cover - guarded by the caller
            return _INF
    best = 8
    for melds, partials, head in states:
        total_melds = melds + open_melds
        value = 8 - 2 * total_melds - partials - head
        if value < best:
            best = value
    return best


def standard_shanten(counts, open_melds: int = 0) -> int:
    """Standard-form shanten for a 3n+1 hand with ``open_melds`` declared melds."""
    values = _check_counts(counts)
    if not isinstance(open_melds, int) or isinstance(open_melds, bool) \
            or not 0 <= open_melds <= 4:
        raise ValueError(f"open_melds must be an int 0..4, got {open_melds!r}")
    if sum(values) + 3 * open_melds not in (1, 4, 7, 10, 13):
        raise ValueError(
            "standard form expects a 3n+1 concealed hand "
            f"(tiles={sum(values)}, open_melds={open_melds})")
    components = [_suit_states(tuple(values[0:9])),
                  _suit_states(tuple(values[9:18])),
                  _suit_states(tuple(values[18:27])),
                  _honor_states(tuple(values[27:34]))]
    return _combine(components, open_melds)


def chiitoi_shanten(counts) -> int:
    values = _check_counts(counts)
    if sum(values) != 13:
        raise ValueError("chiitoitsu is only defined for a 13-tile concealed hand")
    pairs = sum(1 for value in values if value >= 2)
    kinds = sum(1 for value in values if value > 0)
    return 6 - pairs + max(0, 7 - kinds)


def kokushi_shanten(counts) -> int:
    values = _check_counts(counts)
    if sum(values) != 13:
        raise ValueError("kokushi is only defined for a 13-tile concealed hand")
    kinds = sum(1 for kind in TERMINAL_HONOR_KINDS if values[kind] > 0)
    pair = any(values[kind] >= 2 for kind in TERMINAL_HONOR_KINDS)
    return 13 - kinds - (1 if pair else 0)


def shanten(counts, open_melds: int = 0) -> int:
    """Minimum of standard / chiitoitsu / kokushi shanten (3n+1 hand)."""
    values = _check_counts(counts)
    best = standard_shanten(values, open_melds)
    if open_melds == 0:
        best = min(best, chiitoi_shanten(values), kokushi_shanten(values))
    return best


# ---------------------------------------------------------------------------
# wins / waits / ukeire
# ---------------------------------------------------------------------------
@lru_cache(maxsize=200_000)
def _all_melds(counts: tuple[int, ...]) -> bool:
    """True iff every tile in the 34-count decomposes into triplets/sequences."""
    total = sum(counts)
    if total % 3 != 0:
        return False
    if total == 0:
        return True
    values = list(counts)
    index = 0
    while index < 34 and values[index] == 0:
        index += 1
    if index >= 34:
        return True
    if values[index] >= 3:
        values[index] -= 3
        if _all_melds(tuple(values)):
            return True
        values[index] += 3
    if index < 27 and index % 9 <= 6 and values[index + 1] and values[index + 2]:
        values[index] -= 1
        values[index + 1] -= 1
        values[index + 2] -= 1
        ok = _all_melds(tuple(values))
        values[index] += 1
        values[index + 1] += 1
        values[index + 2] += 1
        if ok:
            return True
    return False


def is_winning_hand(counts, open_melds: int = 0) -> bool:
    """True iff ``counts`` is exactly a complete 3n+2 hand."""
    values = _check_counts(counts)
    total = sum(values)
    if total + 3 * open_melds not in (2, 5, 8, 11, 14):
        return False
    if open_melds == 0 and total == 14:
        pairs = sum(1 for value in values if value == 2)
        if pairs == 7:
            return True
        kinds = sum(1 for kind in TERMINAL_HONOR_KINDS if values[kind] > 0)
        pair = any(values[kind] >= 2 for kind in TERMINAL_HONOR_KINDS)
        if kinds == 13 and pair:
            return True
    mutable = list(values)
    for kind in range(TILE_KIND_COUNT):
        if mutable[kind] < 2:
            continue
        mutable[kind] -= 2
        ok = _all_melds(tuple(mutable))
        mutable[kind] += 2
        if ok:
            return True
    return False


def shanten_14(counts, open_melds: int = 0) -> int:
    """Shanten of a 3n+2 hand: the minimum over every legal discard.

    ``-1`` means the 14-tile hand is already complete.  Cost is one 13-tile shanten
    evaluation per distinct tile held, so bots use it only for explicit ukeire counts.
    """
    values = _check_counts(counts)
    if sum(values) + 3 * open_melds not in (2, 5, 8, 11, 14):
        raise ValueError(
            "standard form expects a 3n+2 hand "
            f"(tiles={sum(values)}, open_melds={open_melds})")
    if open_melds == 0 and sum(values) == 14 and is_winning_hand(values, open_melds):
        return -1
    best = 8
    for kind in range(TILE_KIND_COUNT):
        if values[kind] == 0:
            continue
        reduced = list(values)
        reduced[kind] -= 1
        best = min(best, shanten(tuple(reduced), open_melds))
        if best == -1:
            return -1
    return best


def _add_kind(counts: tuple[int, ...], kind: int) -> tuple[int, ...]:
    values = list(counts)
    values[kind] += 1
    return tuple(values)


def waits(counts, open_melds: int = 0) -> tuple[int, ...]:
    """Tiles (0..33) that complete a tenpai 3n+1 hand (exact win check)."""
    values = _check_counts(counts)
    if shanten(values, open_melds) != 0:
        return ()
    result = []
    for kind in range(TILE_KIND_COUNT):
        if values[kind] >= 4:
            continue
        if is_winning_hand(_add_kind(values, kind), open_melds):
            result.append(kind)
    return tuple(result)


def ukeire(counts, open_melds: int, visible_counts) -> int:
    """Remaining tiles that strictly reduce shanten of a 3n+1 hand.

    ``visible_counts`` is the public/own 34-count ledger; remaining copies are
    ``max(0, 4 - visible)``.  This is exact for the supplied ledger.
    """
    values = _check_counts(counts)
    visible = _check_counts(visible_counts, name="visible_counts")
    current = shanten(values, open_melds)
    total = 0
    for kind in range(TILE_KIND_COUNT):
        remaining = 4 - visible[kind]
        if remaining <= 0 or values[kind] >= 4:
            continue
        if shanten_14(_add_kind(values, kind), open_melds) < current:
            total += remaining
    return total


def tile_kind_of_text(text: str) -> int:
    return kind_from_text(text)
