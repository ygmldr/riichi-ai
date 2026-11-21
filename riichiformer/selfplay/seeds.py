#!/usr/bin/env python3
"""PR-41 seed pools: train / select / confirm are disjoint by construction.

A rollout seed identifies one *wall* (one ``(seed, key)`` pair) that is replayed in four
rotated games.  Reusing a seed across pools would leak evaluation walls into training
data, so :class:`SeedLedger` refuses **any** reuse:

* a seed range can be reserved in exactly one pool;
* a seed that has been consumed once cannot be reserved or consumed again;
* reserving the same range twice (even in the same pool) is a conflict.

The ledger is a small JSON file so separate CPU processes can share it under an explicit
path.  There is no lock: rollouts are launched serially by the Lead, and the ledger is
written atomically after each reservation.  A missing ledger means "nothing used yet";
a malformed ledger fails closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .provenance import utc_now, write_json

__all__ = [
    "POOLS",
    "LEDGER_FORMAT",
    "SeedError",
    "SeedConflict",
    "SeedLedger",
    "seed_range",
    "validate_games",
]

POOLS: tuple[str, str, str] = ("train", "select", "confirm")
LEDGER_FORMAT = "rf-selfplay-seed-ledger/1"
ROTATIONS = 4


class SeedError(ValueError):
    """A seed-pool contract violation (always fail closed)."""


class SeedConflict(SeedError):
    """A seed was reused across pools or consumed twice."""


def validate_games(games: int) -> int:
    if isinstance(games, bool) or not isinstance(games, int) or games <= 0:
        raise SeedError(f"--games must be a positive int, got {games!r}")
    if games % ROTATIONS != 0:
        raise SeedError(
            f"--games must be a multiple of {ROTATIONS} (four-seat rotations), got {games}")
    return games


def seed_range(seed_start: int, games: int) -> tuple[int, ...]:
    """Unique wall seeds for a run: ``games / 4`` walls, each replayed 4 times."""
    validate_games(games)
    if isinstance(seed_start, bool) or not isinstance(seed_start, int) or seed_start < 0:
        raise SeedError(f"--seed-start must be a non-negative int, got {seed_start!r}")
    return tuple(range(seed_start, seed_start + games // ROTATIONS))


def _check_pool(pool: str) -> str:
    if pool not in POOLS:
        raise SeedError(f"unknown seed pool {pool!r}; known: {list(POOLS)}")
    return pool


def _normalise(seeds) -> tuple[int, ...]:
    values = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise SeedError(f"seed must be a non-negative int, got {seed!r}")
        values.append(int(seed))
    if not values:
        raise SeedError("no seeds supplied")
    if len(set(values)) != len(values):
        raise SeedError("duplicate seeds in one reservation")
    return tuple(values)


@dataclass
class SeedLedger:
    """Disjoint seed pools persisted as JSON."""

    pools: dict[str, dict[int, dict]] = field(
        default_factory=lambda: {pool: {} for pool in POOLS})
    consumed: dict[int, dict] = field(default_factory=dict)
    path: Path | None = None
    runs: list[dict] = field(default_factory=list)

    # -- queries -----------------------------------------------------------
    @property
    def used_seeds(self) -> frozenset[int]:
        used: set[int] = set(self.consumed)
        for entries in self.pools.values():
            used.update(entries)
        return frozenset(used)

    def pool_of(self, seed: int) -> str | None:
        for pool, entries in self.pools.items():
            if seed in entries:
                return pool
        return None

    def is_consumed(self, seed: int) -> bool:
        return seed in self.consumed

    def reserved(self, pool: str) -> tuple[int, ...]:
        return tuple(sorted(self.pools[_check_pool(pool)]))

    # -- mutations ---------------------------------------------------------
    def reserve(self, pool: str, seeds, *, tag: str = "") -> tuple[int, ...]:
        pool = _check_pool(pool)
        values = _normalise(seeds)
        conflicts = []
        for seed in values:
            other = self.pool_of(seed)
            if other is not None:
                scope = "same pool" if other == pool else f"pool {other!r}"
                conflicts.append(f"{seed} already reserved in {scope}")
            elif seed in self.consumed:
                conflicts.append(f"{seed} already consumed")
        if conflicts:
            raise SeedConflict(
                f"seed reuse refused for pool {pool!r}: " + "; ".join(conflicts[:5])
                + ("" if len(conflicts) <= 5 else f" (+{len(conflicts) - 5} more)"))
        stamp = utc_now()
        for seed in values:
            self.pools[pool][seed] = {"tag": tag, "reserved_utc": stamp}
        self.save()
        return values

    def reserve_range(self, pool: str, seed_start: int, games: int, *,
                      tag: str = "") -> tuple[int, ...]:
        seeds = seed_range(seed_start, games)
        return self.reserve(pool, seeds, tag=tag)

    def consume(self, pool: str, seeds) -> tuple[int, ...]:
        pool = _check_pool(pool)
        values = _normalise(seeds)
        for seed in values:
            if self.pool_of(seed) != pool:
                raise SeedConflict(
                    f"seed {seed} was not reserved in pool {pool!r} "
                    f"(reserved in {self.pool_of(seed)!r})")
            if seed in self.consumed:
                raise SeedConflict(f"seed {seed} was already consumed")
        stamp = utc_now()
        for seed in values:
            self.consumed[seed] = {"pool": pool, "consumed_utc": stamp}
        self.save()
        return values

    def record_run(self, pool: str, seed_start: int, games: int, *,
                   tag: str = "") -> dict:
        """Reserve + consume one run's seeds and append a run record."""
        validate_games(games)
        seeds = self.reserve_range(pool, seed_start, games, tag=tag)
        self.consume(pool, seeds)
        record = {
            "pool": pool,
            "seed_start": seed_start,
            "seed_count": len(seeds),
            "games": games,
            "seeds": list(seeds),
            "tag": tag,
            "recorded_utc": utc_now(),
        }
        self.runs.append(record)
        self.save()
        return record

    # -- persistence -------------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "format": LEDGER_FORMAT,
            "pools": {
                pool: {str(seed): dict(entry) for seed, entry in sorted(entries.items())}
                for pool, entries in sorted(self.pools.items())
            },
            "consumed": {str(seed): dict(entry) for seed, entry in sorted(self.consumed.items())},
            "runs": list(self.runs),
        }

    def save(self, path: str | Path | None = None) -> Path | None:
        target = Path(path) if path is not None else self.path
        if target is None:
            return None
        write_json(target, self.as_dict())
        self.path = target
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SeedLedger":
        target = Path(path)
        if not target.exists():
            return cls(path=target)
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SeedError(f"seed ledger {target} is not valid JSON: {exc}") from None
        if payload.get("format") != LEDGER_FORMAT:
            raise SeedError(
                f"seed ledger {target} has format {payload.get('format')!r}, "
                f"expected {LEDGER_FORMAT!r}")
        pools = {pool: {} for pool in POOLS}
        for pool, entries in (payload.get("pools") or {}).items():
            _check_pool(pool)
            for seed_text, entry in entries.items():
                pools[pool][int(seed_text)] = dict(entry)
        consumed = {int(seed): dict(entry)
                    for seed, entry in (payload.get("consumed") or {}).items()}
        return cls(pools=pools, consumed=consumed,
                   runs=list(payload.get("runs") or []), path=target)
