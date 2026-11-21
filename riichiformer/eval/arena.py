#!/usr/bin/env python3
"""PR-42 duplicate arena: four-seat rotation, paired wall seeds, block statistics.

A *duplicate* here follows the pinned arena's arrangement: one wall ``(seed, key)`` is
replayed four times, each lineup position occupying each absolute seat exactly once.
The four games of one seed are therefore **not** independent samples -- they share a
wall and the seating is deliberately rotated -- so all confidence intervals resample
whole seeds (:func:`bootstrap_seed_blocks`), never individual games.

``GameAudit`` re-checks every game: score conservation (sum == 100000), four POVs, a
seat permutation, ranks 1..4, zero illegal/rejected/fallback actions.

Nothing here converts bot-vs-bot results into an official dan/rank or a Mortal
comparison: every report carries ``official_rank: false`` and ``strength_claim: none``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from ..selfplay.provenance import assert_known_ruleset
from .opponents import OpponentRegistry, OpponentSpec, build_lineup

__all__ = [
    "ArenaError",
    "GameAudit",
    "AuditFinding",
    "DuplicateArena",
    "ArenaReport",
    "bootstrap_seed_blocks",
    "bootstrap_independent_games",
]

EXPECTED_SCORE_SUM = 100_000
DEFAULT_RESAMPLES = 2000


class ArenaError(RuntimeError):
    """An arena precondition/consistency failure (fail closed)."""


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AuditFinding:
    category: str
    game_id: str
    detail: str

    def as_dict(self) -> dict:
        return {"category": self.category, "game_id": self.game_id, "detail": self.detail}


class GameAudit:
    """Score-conservation / legality / ruleset audit over arena games."""

    def __init__(self, *, expected_sum: int = EXPECTED_SCORE_SUM):
        self.expected_sum = int(expected_sum)
        self.findings: list[AuditFinding] = []
        self.games = 0
        self.deviations: dict[str, int] = {}

    def check(self, game) -> list[AuditFinding]:
        """Audit one ``GameRecord``-shaped object; returns the new findings."""
        start = len(self.findings)
        self.games += 1
        game_id = str(getattr(game, "game_id", "?"))

        scores = tuple(getattr(game, "scores", ()))
        if len(scores) != 4:
            self._fail("score_shape", game_id, f"scores {scores!r} is not four values")
        elif sum(scores) != self.expected_sum:
            self._fail("score_conservation", game_id,
                       f"sum {sum(scores)} != {self.expected_sum}")
        ranks = tuple(getattr(game, "ranks", ()))
        if sorted(ranks) != [1, 2, 3, 4]:
            self._fail("rank_permutation", game_id, f"ranks {ranks!r}")
        seats = tuple(getattr(game, "seats", ()))
        if sorted(seats) != [0, 1, 2, 3]:
            self._fail("seat_rotation", game_id, f"seats {seats!r}")
        povs = getattr(game, "povs", ())
        if len(povs) != 4:
            self._fail("pov_count", game_id, f"{len(povs)} POVs, expected 4")
        for field_name in ("illegal", "rejected", "fallbacks"):
            value = int(getattr(game, field_name, 0))
            if value:
                self._fail(f"{field_name}_actions", game_id, f"{value} {field_name}")
        for key, value in dict(getattr(game, "deviation_counts", {}) or {}).items():
            if value < 0:
                self._fail("deviation_count", game_id, f"{key}={value}")
            self.deviations[key] = self.deviations.get(key, 0) + int(value)
        return self.findings[start:]

    def _fail(self, category: str, game_id: str, detail: str) -> None:
        self.findings.append(AuditFinding(category, game_id, detail))

    @property
    def ok(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict:
        categories: dict[str, int] = {}
        for finding in self.findings:
            categories[finding.category] = categories.get(finding.category, 0) + 1
        return {
            "ok": self.ok,
            "games": self.games,
            "expected_score_sum": self.expected_sum,
            "violations": [finding.as_dict() for finding in self.findings],
            "violation_categories": dict(sorted(categories.items())),
            "deviations": dict(sorted(self.deviations.items())),
        }


# ---------------------------------------------------------------------------
# block bootstrap
# ---------------------------------------------------------------------------
def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ArenaError("empty bootstrap sample")
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_seed_blocks(blocks: Sequence[Sequence[float]], *, statistic: Callable,
                          resamples: int = DEFAULT_RESAMPLES, seed: int = 0,
                          ci: float = 0.95) -> dict:
    """Percentile CI resampling **whole seeds**, not the four games of a seed.

    ``blocks[i]`` holds the per-game observations of seed ``i`` (normally four values).
    Each resample draws ``len(blocks)`` seeds with replacement, concatenates their
    games and evaluates ``statistic`` on the concatenation, which preserves the
    within-seed correlation the duplicate design deliberately creates.
    """
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ArenaError("resamples must be an int >= 100")
    if not 0.0 < ci < 1.0:
        raise ArenaError("ci must be in (0, 1)")
    blocks = [list(block) for block in blocks]
    if len(blocks) < 2:
        return {"unit": "seed", "blocks": len(blocks), "resamples": 0,
                "low": None, "high": None, "point": None,
                "note": "block bootstrap needs at least 2 seed blocks"}
    for block in blocks:
        if not block:
            raise ArenaError("each seed block must contain at least one game")
    rng = random.Random(int(seed))
    values: list[float] = []
    for _ in range(resamples):
        sample: list[float] = []
        for _ in range(len(blocks)):
            sample.extend(blocks[rng.randrange(len(blocks))])
        values.append(float(statistic(sample)))
    values.sort()
    point = float(statistic([value for block in blocks for value in block]))
    return {
        "unit": "seed",
        "blocks": len(blocks),
        "resamples": resamples,
        "low": _percentile(values, (1 - ci) / 2),
        "high": _percentile(values, (1 + ci) / 2),
        "point": point,
        "ci": ci,
    }


def bootstrap_independent_games(blocks: Sequence[Sequence[float]], *,
                                statistic: Callable, resamples: int = DEFAULT_RESAMPLES,
                                seed: int = 0, ci: float = 0.95) -> dict:
    """Negative control: resample individual games as if they were independent.

    This is the *wrong* interval for a duplicate arena (the four rotated games of one
    seed share a wall).  It is kept only so tests can prove that
    :func:`bootstrap_seed_blocks` does not collapse to independent-game sampling.
    """
    games = [value for block in blocks for value in block]
    if len(games) < 2:
        return {"unit": "game", "games": len(games), "resamples": 0,
                "low": None, "high": None, "point": None,
                "note": "needs at least 2 games"}
    rng = random.Random(int(seed))
    values = []
    for _ in range(resamples):
        sample = [games[rng.randrange(len(games))] for _ in range(len(games))]
        values.append(float(statistic(sample)))
    values.sort()
    return {
        "unit": "game",
        "games": len(games),
        "resamples": resamples,
        "low": _percentile(values, (1 - ci) / 2),
        "high": _percentile(values, (1 + ci) / 2),
        "point": float(statistic(games)),
        "ci": ci,
        "warning": "invalid for duplicate arenas: treats four rotated games as independent",
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
@dataclass
class ArenaReport:
    ruleset_id: str
    rules_engine: str
    mode: str
    lineup: tuple[str, ...]
    opponents: dict[str, dict]
    seeds: tuple[int, ...]
    games: int
    per_opponent: dict[str, dict]
    audit: dict
    bootstrap: dict
    notes: tuple[str, ...] = field(default_factory=tuple)
    game_rows: tuple[dict, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "task": "PR-42",
            "mode": "duplicate_arena",
            "matchup_mode": self.mode,
            "ruleset_id": self.ruleset_id,
            "rules_engine": self.rules_engine,
            "tenhou_equivalent": False,
            "official_rank": False,
            "strength_claim": "none",
            "lineup": list(self.lineup),
            "opponents": self.opponents,
            "seeds": list(self.seeds),
            "seed_count": len(self.seeds),
            "games": self.games,
            "rotations": 4,
            "per_opponent": self.per_opponent,
            "audit": self.audit,
            "bootstrap": self.bootstrap,
            "not_run": [
                "no official dan/rank estimate (bot-vs-bot is not an official rating)",
                "no Mortal/akochan comparison: no third-party weights are registered",
                "no online Tenhou/Majsoul play and no network access",
            ],
            "notes": list(self.notes),
        }


class DuplicateArena:
    """Run a four-seat rotated duplicate arena over paired wall seeds."""

    ROTATIONS = 4

    def __init__(self, *, registry: OpponentRegistry, runner: Callable[[int], Sequence],
                 ruleset_id: str, expected_engine: str, key: int,
                 resamples: int = DEFAULT_RESAMPLES, bootstrap_seed: int = 0,
                 audit: GameAudit | None = None):
        assert_known_ruleset(ruleset_id)
        if not callable(runner):
            raise ArenaError("runner must be callable: runner(seed) -> 4 game records")
        self.registry = registry
        self.runner = runner
        self.ruleset_id = ruleset_id
        self.expected_engine = str(expected_engine)
        self.key = int(key)
        self.resamples = int(resamples)
        self.bootstrap_seed = int(bootstrap_seed)
        self.audit = audit if audit is not None else GameAudit()

    # -- helpers -----------------------------------------------------------
    def _check_game(self, game, *, seed: int, lineup: tuple[str, ...]) -> None:
        game_seed = int(getattr(game, "seed", -1))
        if game_seed != seed:
            raise ArenaError(f"runner returned game seed {game_seed} for seed {seed}")
        if tuple(getattr(game, "lineup", ())) != lineup:
            raise ArenaError(
                f"{getattr(game, 'game_id', '?')}: lineup "
                f"{getattr(game, 'lineup', ())} != requested {lineup}")
        ruleset = getattr(game, "ruleset_id", None)
        if ruleset != self.ruleset_id:
            raise ArenaError(
                f"{getattr(game, 'game_id', '?')}: ruleset {ruleset!r} != "
                f"{self.ruleset_id!r}")
        engine = getattr(game, "rules_engine", None)
        if engine != self.expected_engine:
            raise ArenaError(
                f"{getattr(game, 'game_id', '?')}: engine {engine!r} != "
                f"{self.expected_engine!r}")

    # -- run ---------------------------------------------------------------
    def run(self, seeds: Iterable[int], *, opponents: Iterable[str] | None = None,
            lineup: Sequence[str] | None = None) -> ArenaReport:
        seed_list = tuple(int(seed) for seed in seeds)
        if not seed_list:
            raise ArenaError("at least one wall seed is required")
        if len(set(seed_list)) != len(seed_list):
            raise ArenaError("duplicate wall seeds in one arena run")
        if lineup is None:
            if opponents is None:
                raise ArenaError("pass either opponents= or lineup=")
            mode, built = build_lineup(opponents, registry=self.registry)
            lineup = built
        else:
            if len(lineup) != 4:
                raise ArenaError("lineup must contain exactly four opponent names")
            for slot in lineup:
                self.registry.get(slot)
            mode = "one_vs_three" if len(set(lineup)) == 2 else "free_for_all"
        lineup = tuple(str(name) for name in lineup)
        specs = self.registry.resolve(sorted(set(lineup)))
        names = sorted(set(lineup))
        positions_by_name = {name: [index for index, slot in enumerate(lineup) if slot == name]
                             for name in names}

        rank_blocks: dict[str, list[list[float]]] = {name: [] for name in names}
        score_blocks: dict[str, list[list[float]]] = {name: [] for name in names}
        rank_counts: dict[str, list[int]] = {name: [0, 0, 0, 0] for name in names}
        totals: dict[str, int] = {name: 0 for name in names}
        wins: dict[str, int] = {name: 0 for name in names}
        top2: dict[str, int] = {name: 0 for name in names}
        games_per: dict[str, int] = {name: 0 for name in names}
        game_rows: list[dict] = []
        games = 0

        for seed in seed_list:
            records = sorted(self.runner(seed), key=lambda record: int(record.rotation))
            if len(records) != self.ROTATIONS:
                raise ArenaError(
                    f"seed {seed}: runner returned {len(records)} games, expected "
                    f"{self.ROTATIONS} rotations")
            rotations = [int(record.rotation) for record in records]
            if rotations != list(range(self.ROTATIONS)):
                raise ArenaError(f"seed {seed}: rotations {rotations} are not 0..3")
            for record in records:
                self._check_game(record, seed=seed, lineup=lineup)
                self.audit.check(record)
                games += 1
                game_rows.append({
                    "game_id": str(getattr(record, "game_id", f"{seed}:{record.rotation}")),
                    "seed": seed,
                    "rotation": int(record.rotation),
                    "scores": list(record.scores),
                    "ranks": list(record.ranks),
                    "illegal": int(getattr(record, "illegal", 0)),
                    "rejected": int(getattr(record, "rejected", 0)),
                })
            for name in names:
                block_ranks: list[float] = []
                block_scores: list[float] = []
                for record in records:
                    for position in positions_by_name[name]:
                        rank = int(record.ranks[position])
                        score = int(record.scores[position])
                        block_ranks.append(float(rank))
                        block_scores.append(float(score))
                        rank_counts[name][rank - 1] += 1
                        totals[name] += score
                        games_per[name] += 1
                        if rank == 1:
                            wins[name] += 1
                        if rank <= 2:
                            top2[name] += 1
                rank_blocks[name].append(block_ranks)
                score_blocks[name].append(block_scores)

        per_opponent: dict[str, dict] = {}
        for name in names:
            count = games_per[name]
            spec: OpponentSpec = self.registry.get(name)
            rank_ci = bootstrap_seed_blocks(
                rank_blocks[name], statistic=_mean, resamples=self.resamples,
                seed=self.bootstrap_seed)
            score_ci = bootstrap_seed_blocks(
                score_blocks[name], statistic=_mean, resamples=self.resamples,
                seed=self.bootstrap_seed)
            per_opponent[name] = {
                "kind": spec.kind.value,
                "version": spec.version,
                "checkpoint_sha256": spec.checkpoint_sha256,
                "license_caveat": spec.license_caveat,
                "games": count,
                "mean_rank": round(sum((index + 1) * value
                                       for index, value in enumerate(rank_counts[name]))
                                   / count, 4) if count else None,
                "rank_counts": {str(index + 1): value
                                for index, value in enumerate(rank_counts[name])},
                "mean_score": round(totals[name] / count, 3) if count else None,
                "win_rate": round(wins[name] / count, 4) if count else None,
                "top2_rate": round(top2[name] / count, 4) if count else None,
                "mean_rank_ci95": rank_ci,
                "mean_score_ci95": score_ci,
            }

        opponents_payload = {
            spec.name: spec.as_dict() for spec in sorted(specs, key=lambda item: item.name)
        }
        return ArenaReport(
            ruleset_id=self.ruleset_id,
            rules_engine=self.expected_engine,
            mode=mode,
            lineup=lineup,
            opponents=opponents_payload,
            seeds=tuple(seed_list),
            games=games,
            per_opponent=per_opponent,
            audit=self.audit.as_dict(),
            bootstrap={
                "unit": "seed",
                "blocks": len(seed_list),
                "resamples": self.resamples,
                "statistic": "per-opponent mean rank / mean score",
                "method": "percentile CI over whole-seed blocks",
                "within_seed_games": self.ROTATIONS,
                "independent_games_forbidden": True,
            },
            notes=(
                "four rotated games share one wall; blocks are wall seeds",
                "bot-vs-bot results are an internal comparison only",
            ),
            game_rows=tuple(game_rows),
        )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")
