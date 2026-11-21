#!/usr/bin/env python3
"""PR-46B relative-PT scorer for the frozen Mortal-5D 1v3 arena.

Two explicitly versioned Tenhou-Hanchan point profiles are defined here:

``TOKUJOU_5D_HANCHAN = [75, 30, 0, -105]``
    **Primary metric.** Tenhou 5-dan, 4-player South-East game, Tokujou table.
``HOUOU_7D_HANCHAN = [90, 45, 0, -135]``
    **Auxiliary historical metric.** The old ``eval_report.py`` ``RANK_PTS``.

For one game ``g`` with the candidate ``C`` against three identical frozen
Mortal seats::

    pt5_C(g)  = profile[rank_C(g) - 1]
    pt5_M(g)  = mean(profile[rank_m(g) - 1] for the three Mortal seats)
    D5(g)     = pt5_C(g) - pt5_M(g) = 4/3 * pt5_C(g)     (ranks sum to zero)

A *complete set* is one wall seed played under four absolute seat rotations; the
candidate therefore occupies each absolute seat exactly once.  Set statistics are
averages over its four games, and confidence intervals resample **whole sets**
(never individual games, never decisions).  The point estimate is the mean of the
per-set means; with equal-size sets this equals the mean over games.

Everything here fails closed: an incomplete set, a duplicate/missing wall seed, a
missing or extra rotation, a game whose final scores do not sum to 100000, a key
mismatch, or a rank vector that disagrees with the scores raises immediately and no
partial estimate is produced.  There is deliberately no historical 3000-set or
500-set gate in this module.

This module makes **no** claim that a positive ``delta_pt5`` maps to a real Tenhou
dan.  A relative result is versus this exact 1v3 opponent pool under the pinned
``tenhou_approx`` rules, nothing more.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "HOUOU_7D_HANCHAN",
    "MAX_SCORE_SUM",
    "PT_PROFILES",
    "PRIMARY_PROFILE",
    "SECONDARY_PROFILE",
    "RelativePTError",
    "IncompleteSetError",
    "DuplicateGameError",
    "DuplicateSeedError",
    "KeyMismatchError",
    "MissingSeedError",
    "RankMismatchError",
    "ScoreSumError",
    "GameIncompleteError",
    "GameScore",
    "CompleteSet",
    "score_game",
    "rank_by_scores",
    "build_complete_sets",
    "aggregate_sets",
    "block_bootstrap_sets",
    "score_policy",
    "score_policy_hash",
    "canonical_seed_key",
]

#: Four-player hanchan score sum.
MAX_SCORE_SUM = 100_000

#: Tenhou 5-dan Tokujou 4p South-East Hanchan (primary).
TOKUJOU_5D_HANCHAN: tuple[int, int, int, int] = (75, 30, 0, -105)
#: Tenhou 7-dan Houou 4p South-East Hanchan (auxiliary / historical eval_report).
HOUOU_7D_HANCHAN: tuple[int, int, int, int] = (90, 45, 0, -135)

PT_PROFILES: dict[str, tuple[int, int, int, int]] = {
    "TOKUJOU_5D_HANCHAN": TOKUJOU_5D_HANCHAN,
    "HOUOU_7D_HANCHAN": HOUOU_7D_HANCHAN,
}
PRIMARY_PROFILE = "TOKUJOU_5D_HANCHAN"
SECONDARY_PROFILE = "HOUOU_7D_HANCHAN"

#: Fixed bootstrap protocol (pre-registered; B=2000, percentile 95%).
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0x52463641  # "RF6A", independent of wall/seed pools and training RNG

ROTATIONS = (0, 1, 2, 3)
TIE_RULE = "scores_descending_then_lower_absolute_seat"
DELTA_FORMULA = "delta_pt = (4/3) * pt_candidate"
SET_UNIT = "complete_seed_set"

SCORE_POLICY_SCHEMA = "rf-relative-pt/1"


class RelativePTError(RuntimeError):
    """Base class for scorer contract failures (always fail closed)."""


class IncompleteSetError(RelativePTError):
    """A wall seed does not have exactly four completed rotations."""


class DuplicateGameError(RelativePTError):
    """The same game id / (seed, rotation) appears more than once."""


class DuplicateSeedError(RelativePTError):
    """A wall seed is declared more than once as an independent set."""


class KeyMismatchError(RelativePTError):
    """A seed key is missing, malformed, or differs from the expected key."""


class MissingSeedError(RelativePTError):
    """An expected wall seed is absent from the collected games."""


class RankMismatchError(RelativePTError):
    """The supplied rank vector disagrees with the supplied final scores."""


class ScoreSumError(RelativePTError):
    """The four final scores do not sum to 100000."""


class GameIncompleteError(RelativePTError):
    """A game is marked as not completed."""


# ---------------------------------------------------------------------------
# canonical helpers
# ---------------------------------------------------------------------------
def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def canonical_seed_key(value) -> str:
    """Normalise a wall seed key to lowercase hex without a ``0x`` prefix.

    The native arena takes a ``u64`` key; the registry records it as hex.  Accepting
    either the integer or a hex string keeps the CLI and the tests unambiguous while
    still refusing empty/oddly typed values.
    """
    if _is_int(value):
        if not 0 <= value < 2**64:
            raise KeyMismatchError(f"seed key {value} is outside u64")
        return f"{value:016x}"
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("0x"):
            text = text[2:]
        if not text or len(text) > 16 or any(ch not in "0123456789abcdef" for ch in text):
            raise KeyMismatchError(f"malformed seed key {value!r}")
        return text.rjust(16, "0")
    raise KeyMismatchError(
        f"seed key must be int or hex str, got {type(value).__name__}")


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def score_policy(profile: str = PRIMARY_PROFILE) -> dict:
    """The frozen scoring contract (tie rule + delta formula + profile)."""
    if profile not in PT_PROFILES:
        raise RelativePTError(f"unknown score profile {profile!r}")
    return {
        "schema": SCORE_POLICY_SCHEMA,
        "primary_profile": PRIMARY_PROFILE,
        "secondary_profile": SECONDARY_PROFILE,
        "profile": profile,
        "pt_table": list(PT_PROFILES[profile]),
        "tie_rule": TIE_RULE,
        "delta_formula": DELTA_FORMULA,
        "unit": SET_UNIT,
        "rotations": list(ROTATIONS),
        "score_sum": MAX_SCORE_SUM,
        "bootstrap": {
            "method": "percentile_over_complete_sets",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "ci": 0.95,
        },
    }


def score_policy_hash(profile: str = PRIMARY_PROFILE) -> str:
    return hashlib.sha256(_canonical_json(score_policy(profile)).encode()).hexdigest()


# ---------------------------------------------------------------------------
# rank / game scoring
# ---------------------------------------------------------------------------
def rank_by_scores(scores: Sequence[int]) -> tuple[int, int, int, int]:
    """Ranks 1..4 by score descending, ties broken by the lower absolute seat.

    The native arena's ``rankings.rs`` orders by score descending and then by seat
    index, which is exactly the old ``eval_report.py`` rule.  A non-100000 sum is a
    contract failure, not a warning.
    """
    values = tuple(scores)
    if len(values) != 4:
        raise ScoreSumError(f"scores must have four entries, got {len(values)}")
    for index, value in enumerate(values):
        if not _is_int(value):
            raise ScoreSumError(f"scores[{index}] must be int, got {value!r}")
    total = sum(values)
    if total != MAX_SCORE_SUM:
        raise ScoreSumError(f"final scores sum to {total}, expected {MAX_SCORE_SUM}")
    order = sorted(range(4), key=lambda seat: (-values[seat], seat))
    ranks = [0, 0, 0, 0]
    for rank, seat in enumerate(order, start=1):
        ranks[seat] = rank
    return tuple(ranks)  # type: ignore[return-value]


@dataclass(frozen=True)
class GameRecord:
    """One completed 1v3 hanchan as written by the arena driver.

    ``seed_key`` identifies the registered wall pool; ``wall_seed`` is one wall in
    it; ``rotation`` is the split index 0..3 (candidate's *relative* starting
    position in the rotated lineup) and ``candidate_seat`` is the candidate's
    absolute seat.  In the native ``OneVsThree`` driver these are equal.
    """

    seed_key: str
    wall_seed: int
    rotation: int
    candidate_seat: int
    scores: tuple[int, int, int, int]
    game_id: str = ""
    completed: bool = True
    ranks: tuple[int, int, int, int] | None = None

    def __post_init__(self):
        object.__setattr__(self, "seed_key", canonical_seed_key(self.seed_key))
        if not _is_int(self.wall_seed) or self.wall_seed < 0:
            raise RelativePTError(f"wall_seed must be a non-negative int, got {self.wall_seed!r}")
        if not _is_int(self.rotation) or self.rotation not in ROTATIONS:
            raise RelativePTError(f"rotation must be 0..3, got {self.rotation!r}")
        if not _is_int(self.candidate_seat) or not 0 <= self.candidate_seat <= 3:
            raise RelativePTError(
                f"candidate_seat must be 0..3, got {self.candidate_seat!r}")
        if not isinstance(self.completed, bool):
            raise RelativePTError("completed must be bool")
        if not self.completed:
            raise GameIncompleteError(
                f"game {self.game_id or (self.wall_seed, self.rotation)} is incomplete")
        object.__setattr__(self, "scores", tuple(self.scores))
        derived = rank_by_scores(self.scores)
        if self.ranks is not None:
            supplied = tuple(self.ranks)
            if sorted(supplied) != [1, 2, 3, 4]:
                raise RankMismatchError(f"ranks {supplied!r} are not a permutation 1..4")
            if supplied != derived:
                raise RankMismatchError(
                    f"supplied ranks {supplied!r} disagree with scores "
                    f"{self.scores!r} under {TIE_RULE}")
        object.__setattr__(self, "ranks", derived)


@dataclass(frozen=True)
class GameScore:
    """Per-game PT metrics (raw values are kept next to the deltas)."""

    game_id: str
    seed_key: str
    wall_seed: int
    rotation: int
    candidate_seat: int
    scores: tuple[int, int, int, int]
    ranks: tuple[int, int, int, int]
    pt5_candidate: int
    pt5_mortal_seats: tuple[int, int, int]
    pt5_mortal_mean: float
    delta_pt5: float
    pt7_candidate: int
    pt7_mortal_mean: float
    delta_pt7: float

    def as_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "seed_key": self.seed_key,
            "wall_seed": self.wall_seed,
            "rotation": self.rotation,
            "candidate_seat": self.candidate_seat,
            "scores": list(self.scores),
            "ranks": list(self.ranks),
            "pt5_candidate": self.pt5_candidate,
            "pt5_mortal_seats": list(self.pt5_mortal_seats),
            "pt5_mortal_mean": self.pt5_mortal_mean,
            "delta_pt5": self.delta_pt5,
            "pt7_candidate": self.pt7_candidate,
            "pt7_mortal_mean": self.pt7_mortal_mean,
            "delta_pt7": self.delta_pt7,
            "avg_rank": float(self.ranks[self.candidate_seat]),
        }


def _profile_values(record: GameRecord, table: Sequence[int]) -> tuple[int, tuple[int, int, int]]:
    candidate_pt = int(table[record.ranks[record.candidate_seat] - 1])
    mortal = tuple(int(table[record.ranks[seat] - 1])
                   for seat in range(4) if seat != record.candidate_seat)
    if len(mortal) != 3:
        raise RelativePTError("a 1v3 game must have exactly three Mortal seats")
    if sum(table) != 0:
        raise RelativePTError(f"PT profile {tuple(table)!r} does not sum to zero")
    return candidate_pt, mortal


def score_game(record: GameRecord) -> GameScore:
    """Score one game; the candidate's raw PT and the 4/3 delta are both returned."""
    if not isinstance(record, GameRecord):
        raise RelativePTError("expected GameRecord")
    ranks = record.ranks
    assert ranks is not None
    pt5_c, pt5_m = _profile_values(record, TOKUJOU_5D_HANCHAN)
    pt7_c, pt7_m = _profile_values(record, HOUOU_7D_HANCHAN)
    mortal_mean5 = sum(pt5_m) / 3.0
    mortal_mean7 = sum(pt7_m) / 3.0
    delta5 = (4.0 / 3.0) * pt5_c
    delta7 = (4.0 / 3.0) * pt7_c
    # The closed form must agree with the direct definition (both are exact integers
    # for these profiles; compare in float with a tight tolerance).
    if not math.isclose(delta5, pt5_c - mortal_mean5, rel_tol=0, abs_tol=1e-9):
        raise RelativePTError("delta_pt5 does not match pt5_C - mean(pt5_M)")
    if not math.isclose(delta7, pt7_c - mortal_mean7, rel_tol=0, abs_tol=1e-9):
        raise RelativePTError("delta_pt7 does not match pt7_C - mean(pt7_M)")
    return GameScore(
        game_id=record.game_id or f"{record.wall_seed}:{record.rotation}",
        seed_key=record.seed_key,
        wall_seed=record.wall_seed,
        rotation=record.rotation,
        candidate_seat=record.candidate_seat,
        scores=record.scores,
        ranks=ranks,
        pt5_candidate=pt5_c,
        pt5_mortal_seats=pt5_m,
        pt5_mortal_mean=mortal_mean5,
        delta_pt5=delta5,
        pt7_candidate=pt7_c,
        pt7_mortal_mean=mortal_mean7,
        delta_pt7=delta7,
    )


# ---------------------------------------------------------------------------
# complete sets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CompleteSet:
    """One wall seed and its four rotated, completed games."""

    seed_key: str
    wall_seed: int
    games: tuple[GameScore, ...]

    def __post_init__(self):
        games = tuple(self.games)
        if len(games) != 4:
            raise IncompleteSetError(
                f"wall seed {self.wall_seed}: {len(games)} games, expected 4")
        if tuple(sorted(game.rotation for game in games)) != ROTATIONS:
            raise IncompleteSetError(
                f"wall seed {self.wall_seed}: rotations "
                f"{sorted(game.rotation for game in games)} != {list(ROTATIONS)}")
        if tuple(sorted(game.candidate_seat for game in games)) != ROTATIONS:
            raise IncompleteSetError(
                f"wall seed {self.wall_seed}: candidate seats "
                f"{sorted(game.candidate_seat for game in games)} are not one full rotation")
        keys = {game.seed_key for game in games}
        if len(keys) != 1:
            raise KeyMismatchError(f"wall seed {self.wall_seed}: mixed seed keys {keys}")
        object.__setattr__(self, "games", tuple(sorted(games, key=lambda g: g.rotation)))

    @property
    def set_pt5_candidate(self) -> float:
        return sum(game.pt5_candidate for game in self.games) / 4.0

    @property
    def set_pt7_candidate(self) -> float:
        return sum(game.pt7_candidate for game in self.games) / 4.0

    @property
    def set_delta_pt5(self) -> float:
        return sum(game.delta_pt5 for game in self.games) / 4.0

    @property
    def set_delta_pt7(self) -> float:
        return sum(game.delta_pt7 for game in self.games) / 4.0

    @property
    def set_avg_rank(self) -> float:
        return sum(game.ranks[game.candidate_seat] for game in self.games) / 4.0

    @property
    def rank_hist(self) -> tuple[int, int, int, int]:
        counts = [0, 0, 0, 0]
        for game in self.games:
            counts[game.ranks[game.candidate_seat] - 1] += 1
        return tuple(counts)  # type: ignore[return-value]

    def as_dict(self) -> dict:
        hist = self.rank_hist
        total = sum(hist)
        return {
            "seed_key": self.seed_key,
            "wall_seed": self.wall_seed,
            "n_games": len(self.games),
            "complete_4_of_4": True,
            "game_ids": [game.game_id for game in self.games],
            "rotations": [game.rotation for game in self.games],
            "candidate_seats": [game.candidate_seat for game in self.games],
            "pt5_candidate": self.set_pt5_candidate,
            "pt7_candidate": self.set_pt7_candidate,
            "delta_pt5": self.set_delta_pt5,
            "delta_pt7": self.set_delta_pt7,
            "avg_rank": self.set_avg_rank,
            "rank_hist": list(hist),
            "rank1_rate": hist[0] / total,
            "rank4_rate": hist[3] / total,
        }


def _dedupe_wall_seeds(expected: Sequence[int]) -> tuple[int, ...]:
    values = []
    for index, seed in enumerate(expected):
        if not _is_int(seed) or seed < 0:
            raise RelativePTError(f"expected_wall_seeds[{index}] must be a non-negative int")
        values.append(seed)
    if len(set(values)) != len(values):
        raise DuplicateSeedError("expected_wall_seeds contains a duplicate seed")
    return tuple(values)


def build_complete_sets(
    records: Iterable[GameRecord],
    *,
    seed_key: str | int | None = None,
    expected_wall_seeds: Sequence[int] | None = None,
) -> tuple[CompleteSet, ...]:
    """Group games into complete four-rotation sets, failing closed on any defect.

    ``seed_key`` pins the whole run to one registered wall key.  When
    ``expected_wall_seeds`` is supplied, the collected set must match it exactly
    (no missing and no extra seeds); this is the guard against dropping a failed
    seed from the denominator.
    """
    records = list(records)
    if not records:
        raise IncompleteSetError("no games were collected")
    expected_key = canonical_seed_key(seed_key) if seed_key is not None else None
    expected_seeds = (None if expected_wall_seeds is None
                      else _dedupe_wall_seeds(list(expected_wall_seeds)))

    by_seed: dict[int, list[GameRecord]] = {}
    seen_game_ids: set[str] = set()
    seen_pairs: set[tuple[int, int]] = set()
    for record in records:
        if not isinstance(record, GameRecord):
            raise RelativePTError("all records must be GameRecord")
        if expected_key is not None and record.seed_key != expected_key:
            raise KeyMismatchError(
                f"wall seed {record.wall_seed} carries key {record.seed_key}, "
                f"expected {expected_key}")
        pair = (record.wall_seed, record.rotation)
        if pair in seen_pairs:
            raise DuplicateGameError(
                f"duplicate (wall_seed, rotation) {pair}")
        seen_pairs.add(pair)
        if record.game_id:
            if record.game_id in seen_game_ids:
                raise DuplicateGameError(f"duplicate game id {record.game_id!r}")
            seen_game_ids.add(record.game_id)
        by_seed.setdefault(record.wall_seed, []).append(record)

    if expected_seeds is not None:
        missing = sorted(set(expected_seeds) - set(by_seed))
        extra = sorted(set(by_seed) - set(expected_seeds))
        if missing:
            raise MissingSeedError(f"missing wall seeds: {missing}")
        if extra:
            raise MissingSeedError(f"unexpected wall seeds: {extra}")
    if len(by_seed) < 1:
        raise IncompleteSetError("no wall seeds collected")

    sets: list[CompleteSet] = []
    for wall_seed in sorted(by_seed):
        games = sorted(by_seed[wall_seed], key=lambda item: item.rotation)
        rotations = [game.rotation for game in games]
        if rotations != list(ROTATIONS):
            raise IncompleteSetError(
                f"wall seed {wall_seed}: rotations {rotations} are not 0..3")
        keys = {game.seed_key for game in games}
        if len(keys) != 1:
            raise KeyMismatchError(f"wall seed {wall_seed}: mixed keys {keys}")
        scored = tuple(score_game(game) for game in games)
        sets.append(CompleteSet(seed_key=next(iter(keys)), wall_seed=wall_seed,
                                games=scored))
    return tuple(sets)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise RelativePTError("empty bootstrap sample")
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


def block_bootstrap_sets(
    values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci: float = 0.95,
    statistic: Callable[[Sequence[float]], float] | None = None,
) -> dict:
    """Percentile CI resampling complete sets (one observation per set)."""
    if isinstance(resamples, bool) or not _is_int(resamples) or resamples < 100:
        raise RelativePTError("resamples must be an int >= 100")
    if not 0.0 < ci < 1.0:
        raise RelativePTError("ci must be in (0, 1)")
    values = [float(value) for value in values]
    if not values:
        raise RelativePTError("block bootstrap needs at least one set")
    fn = statistic or (lambda sample: sum(sample) / len(sample))
    rng = random.Random(int(seed))
    draws = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        draws.append(float(fn(sample)))
    draws.sort()
    return {
        "unit": SET_UNIT,
        "sets": len(values),
        "resamples": resamples,
        "seed": int(seed),
        "ci": ci,
        "method": "percentile_over_complete_sets",
        "low": _percentile(draws, (1 - ci) / 2),
        "high": _percentile(draws, (1 + ci) / 2),
        "point": float(fn(values)),
    }


def aggregate_sets(
    sets: Sequence[CompleteSet],
    *,
    seed_key: str | int | None = None,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    ci: float = 0.95,
) -> dict:
    """Aggregate complete sets into the PR-46B report payload (no strength claim)."""
    sets = tuple(sets)
    if not sets:
        raise IncompleteSetError("at least one complete set is required")
    expected_key = canonical_seed_key(seed_key) if seed_key is not None else None
    wall_seeds = [item.wall_seed for item in sets]
    if len(set(wall_seeds)) != len(wall_seeds):
        raise DuplicateSeedError("duplicate wall seed between complete sets")
    set_keys = {item.seed_key for item in sets}
    if len(set_keys) != 1:
        raise KeyMismatchError(f"mixed seed keys across sets: {sorted(set_keys)}")
    if expected_key is not None and set_keys != {expected_key}:
        raise KeyMismatchError(
            f"set key {sorted(set_keys)[0]} != expected {expected_key}")

    deltas5 = [item.set_delta_pt5 for item in sets]
    deltas7 = [item.set_delta_pt7 for item in sets]
    raw5 = [item.set_pt5_candidate for item in sets]
    ranks = [item.set_avg_rank for item in sets]
    n_sets = len(sets)
    n_games = sum(len(item.games) for item in sets)
    mean_delta5 = sum(deltas5) / n_sets
    mean_delta7 = sum(deltas7) / n_sets
    mean_raw5 = sum(raw5) / n_sets
    mean_rank = sum(ranks) / n_sets
    variance = (sum((value - mean_delta5) ** 2 for value in deltas5) / (n_sets - 1)
                if n_sets > 1 else 0.0)
    std_set = math.sqrt(variance)
    sem_set = std_set / math.sqrt(n_sets) if n_sets else float("nan")

    hist = [0, 0, 0, 0]
    for item in sets:
        for index, value in enumerate(item.rank_hist):
            hist[index] += value
    if sum(hist) != n_games:
        raise RelativePTError("rank histogram does not cover every game")
    ci5 = block_bootstrap_sets(deltas5, resamples=bootstrap_resamples,
                               seed=bootstrap_seed, ci=ci)
    ci7 = block_bootstrap_sets(deltas7, resamples=bootstrap_resamples,
                               seed=bootstrap_seed, ci=ci)
    raw_ci5 = block_bootstrap_sets(raw5, resamples=bootstrap_resamples,
                                   seed=bootstrap_seed, ci=ci)
    rank_ci = block_bootstrap_sets(ranks, resamples=bootstrap_resamples,
                                   seed=bootstrap_seed, ci=ci)

    return {
        "task": "PR-46B",
        "score_policy_id": PRIMARY_PROFILE,
        "score_policy_hash": score_policy_hash(PRIMARY_PROFILE),
        "score_policy": score_policy(PRIMARY_PROFILE),
        "seed_key": next(iter(set_keys)),
        "n_complete_sets": n_sets,
        "n_games": n_games,
        "missing_sets": 0,
        "failed_games": 0,
        "pt5_candidate": mean_raw5,
        "delta_pt5": mean_delta5,
        "delta_pt5_ci95": ci5,
        "pt5_candidate_ci95": raw_ci5,
        "pt7_candidate": sum(item.set_pt7_candidate for item in sets) / n_sets,
        "delta_pt7": mean_delta7,
        "delta_pt7_ci95": ci7,
        "avg_rank": mean_rank,
        "avg_rank_ci95": rank_ci,
        "rank_hist": hist,
        "rank1_rate": hist[0] / n_games if n_games else None,
        "rank4_rate": hist[3] / n_games if n_games else None,
        "std_set": std_set,
        "sem_set": sem_set,
        "per_set": [item.as_dict() for item in sets],
        "official_rank": False,
        "strength_claim": "relative_to_frozen_M5D-R0_pool_only",
        "not_run": [
            "no real-Tenhou dan/rank mapping from delta_pt5",
            "no confirm pool: this is a smoke/trend/select scale estimate",
        ],
    }
