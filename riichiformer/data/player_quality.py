"""Player metadata & empirical-Bayes quality scores (PR-06).

Privacy design:

* names are read from ``start_game.names`` **in memory only**;
* the only persisted identifier is ``HMAC-SHA256(key, name)`` with the key taken from
  ``RF_PLAYER_HMAC_KEY`` (local secret store / env).  Without the key there is no
  player-linked audit; a basic game-level audit still works and writes no names;
* raw names and the key must never appear in an artefact, log line, or exception.

Scoring design (all fields are candidates, documented in ``quality_audit.md``):

* one row = one seat-game;
* history = the *same player's* earlier games **inside the allowed (train) set**,
  excluding the target game itself; if the target has no timestamp, history falls
  back to the full allowed pool and is marked ``temporal=false``;
* ``q = (n*mean_rank_prior + k*prior) / (n + k)`` with ``prior=2.5``, ``k=100``
  (lower q is better; configurable);
* ``n < min_games (50)`` → ``insufficient_history=true`` and low confidence;
* the target game's own result is kept **only** as ``target_rank_for_eval`` and is
  never used to compute q (label/input separation);
* ``top25_seat`` marks seat-games of players whose q is in the best 25 % among
  eligible players.  It never marks the other three seats of the same game;
* validation/select/confirm games are excluded from history by the caller (the CLI
  passes only the protocol's ``train`` split) and can never change a training score.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = "rf-quality/1"
DEFAULT_PRIOR_MEAN_RANK = 2.5
DEFAULT_K = 100
DEFAULT_MIN_GAMES = 50
DEFAULT_TOP_QUANTILE = 0.25
TIME_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})(\d{2})")


class HmacKeyMissing(RuntimeError):
    """RF_PLAYER_HMAC_KEY is absent/too short; player-linked audit is disabled."""


class GameParseError(ValueError):
    pass


def hmac_key_from_env(env=None) -> bytes:
    env = os.environ if env is None else env
    raw = env.get("RF_PLAYER_HMAC_KEY")
    if not raw:
        raise HmacKeyMissing("RF_PLAYER_HMAC_KEY is not set")
    key = raw.encode("utf-8")
    if len(key) < 16:
        raise HmacKeyMissing("RF_PLAYER_HMAC_KEY must be >= 16 bytes")
    return key


def player_hmac(name: str, key: bytes) -> str:
    if not name:
        raise ValueError("empty player name")
    return hmac.new(key, name.encode("utf-8"), hashlib.sha256).hexdigest()


def match_time_from_relpath(relpath: str):
    """Tenhou filename prefix ``YYYYMMDDHH`` → naive datetime (ordering only)."""
    base = Path(relpath).name
    match = TIME_RE.match(base)
    if not match:
        return None
    try:
        return datetime(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def ranks_from_scores(scores: list[int]) -> list[int]:
    """1-based rank; higher score first, ties broken by smaller seat index (eval_report)."""
    order = sorted(range(len(scores)), key=lambda seat: (-int(scores[seat]), seat))
    ranks = [0] * len(scores)
    for position, seat in enumerate(order, 1):
        ranks[seat] = position
    return ranks


@dataclass(frozen=True)
class Appearance:
    game_id: str
    seat: int
    player_hmac: str
    match_time: datetime | None
    rank: int
    year: int | None = None
    source_id: str | None = None


def _percentile(sorted_values: list[float], q: float):
    if not sorted_values:
        return None
    index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * q) - 1))
    return sorted_values[index]


def compute_quality(appearances: list[Appearance], *, prior: float = DEFAULT_PRIOR_MEAN_RANK,
                    k: int = DEFAULT_K, min_games: int = DEFAULT_MIN_GAMES,
                    top_quantile: float = DEFAULT_TOP_QUANTILE,
                    history_game_ids: set[str] | None = None) -> list[dict]:
    """Compute one quality row per appearance (target result never enters its own q)."""
    if k <= 0:
        raise ValueError("k must be > 0")
    if not 0 < top_quantile < 1:
        raise ValueError("top_quantile must be in (0, 1)")
    by_player = defaultdict(list)
    for appearance in appearances:
        by_player[appearance.player_hmac].append(appearance)
    for items in by_player.values():
        items.sort(key=lambda a: (a.match_time or datetime.min, a.game_id, a.seat))

    rows = []
    for player, items in by_player.items():
        last_key = max((a.match_time or datetime.min, a.game_id) for a in items)
        for target in items:
            history = []
            temporal = True
            for other in items:
                if other.game_id == target.game_id:
                    continue
                if history_game_ids is not None and other.game_id not in history_game_ids:
                    continue
                if target.match_time is None or other.match_time is None:
                    temporal = False
                elif other.match_time >= target.match_time:
                    continue
                history.append(other)
            n = len(history)
            mean_rank = sum(item.rank for item in history) / n if n else prior
            q = (n * mean_rank + k * prior) / (n + k)
            rows.append({
                "game_id": target.game_id,
                "seat": target.seat,
                "player_hmac": player,
                "match_time": target.match_time.isoformat() if target.match_time else None,
                "year": target.year,
                "source_id": target.source_id,
                "n_prior_games": n,
                "mean_rank_prior": round(mean_rank, 6),
                "q": round(q, 6),
                "confidence": round(n / (n + k), 6),
                "insufficient_history": n < min_games,
                "temporal": temporal,
                "target_rank_for_eval": target.rank,   # evaluation only; never an input
                "is_last_appearance": (target.match_time or datetime.min, target.game_id) == last_key,
                "top25_seat": False,
            })

    eligible = [row["q"] for row in rows if not row["insufficient_history"]]
    threshold = _percentile(sorted(eligible), top_quantile) if eligible else None
    for row in rows:
        row["top25_seat"] = (
            threshold is not None and not row["insufficient_history"] and row["q"] <= threshold
        )
    rows.sort(key=lambda r: (r["game_id"], r["seat"]))
    return rows


def quality_summary(rows: list[dict], *, k=DEFAULT_K, prior=DEFAULT_PRIOR_MEAN_RANK,
                    min_games=DEFAULT_MIN_GAMES) -> dict:
    eligible = [row for row in rows if not row["insufficient_history"]]
    qs = sorted(row["q"] for row in eligible)
    return {
        "schema": SCHEMA_VERSION,
        "rows": len(rows),
        "unique_players": len({row["player_hmac"] for row in rows}),
        "eligible_rows": len(eligible),
        "insufficient_rows": len(rows) - len(eligible),
        "insufficient_fraction": round((len(rows) - len(eligible)) / len(rows), 4) if rows else None,
        "top25_rows": sum(1 for row in rows if row["top25_seat"]),
        "top25_threshold": _percentile(qs, DEFAULT_TOP_QUANTILE) if qs else None,
        "prior": prior,
        "k": k,
        "min_games": min_games,
        "q_quantiles": {
            "p10": _percentile(qs, 0.10), "p25": _percentile(qs, 0.25),
            "p50": _percentile(qs, 0.50), "p75": _percentile(qs, 0.75),
            "p90": _percentile(qs, 0.90),
        } if qs else None,
        "n_prior_histogram": dict(sorted(Counter(
            "0" if row["n_prior_games"] == 0 else
            "1-9" if row["n_prior_games"] < 10 else
            "10-49" if row["n_prior_games"] < 50 else
            "50-99" if row["n_prior_games"] < 100 else "100+"
            for row in rows
        ).items())),
        "temporal_rows": sum(1 for row in rows if row["temporal"]),
        "non_temporal_rows": sum(1 for row in rows if not row["temporal"]),
        "years": dict(sorted(Counter(str(row["year"]) for row in rows).items())),
        "sources": dict(sorted(Counter(str(row["source_id"]) for row in rows).items())),
    }


def reliability_table(rows: list[dict]) -> list[dict]:
    """Players' last appearance: q (from prior games) vs that game's rank; bucketed.

    This is a *post-hoc* reliability check, never a training feature.  Only rows with
    ``n_prior_games >= 50`` are eligible.
    """
    last = [row for row in rows if row["is_last_appearance"] and not row["insufficient_history"]]
    if not last:
        return []
    values = sorted(row["q"] for row in last)
    cuts = [_percentile(values, 0.25), _percentile(values, 0.50), _percentile(values, 0.75)]
    buckets = defaultdict(list)
    for row in last:
        if row["q"] <= cuts[0]:
            name = "Q1_lowest_q"
        elif row["q"] <= cuts[1]:
            name = "Q2"
        elif row["q"] <= cuts[2]:
            name = "Q3"
        else:
            name = "Q4_highest_q"
        buckets[name].append(row["target_rank_for_eval"])
    out = []
    for name in ("Q1_lowest_q", "Q2", "Q3", "Q4_highest_q"):
        ranks = buckets.get(name, [])
        if not ranks:
            continue
        out.append({
            "bucket": name,
            "players": len(ranks),
            "mean_target_rank": round(sum(ranks) / len(ranks), 4),
            "target_win_rate": round(sum(1 for rank in ranks if rank == 1) / len(ranks), 4),
        })
    return out


def extract_appearances(raw: bytes, relpath: str, game_id: str, year: int | None,
                        source_id: str | None, key: bytes) -> tuple[list[Appearance], str | None]:
    """Parse one log into appearances; returns ``(rows, error)`` and never logs names."""
    from riichiformer.data import canonicalize, replay

    try:
        compression = canonicalize.detect_compression_bytes(raw)
        payload = canonicalize.decode_log_bytes(raw, compression)
        events = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    except Exception as exc:  # noqa: BLE001
        return [], f"decode:{type(exc).__name__}"
    if not events or events[0].get("type") != "start_game":
        return [], "missing_start_game"
    names = events[0].get("names")
    if not isinstance(names, list) or len(names) != 4 or not all(
        isinstance(name, str) and name.strip() for name in names
    ):
        return [], "invalid_names"
    scores = replay.final_scores_from_events(events)
    if scores is None or len(scores) != 4:
        return [], "missing_scores"
    ranks = ranks_from_scores(scores)
    match_time = match_time_from_relpath(relpath)
    rows = []
    for seat, name in enumerate(names):
        try:
            digest = player_hmac(name, key)
        except ValueError:
            return [], "empty_name"
        rows.append(Appearance(
            game_id=game_id, seat=seat, player_hmac=digest, match_time=match_time,
            rank=ranks[seat], year=year, source_id=source_id,
        ))
    return rows, None


def assert_no_names(rows: list[dict]) -> None:
    """Guard used by tests/CLI: public rows must contain only HMAC identifiers."""
    forbidden = ("name", "player_name", "display_name")
    for row in rows:
        for field in row:
            if field in forbidden:
                raise ValueError(f"public quality row contains forbidden field {field!r}")
