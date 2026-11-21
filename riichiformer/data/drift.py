"""Era / rule drift measurement for RiichiFormer V2 (PR-07 / PR-07R).

Metrics are computed from public events (and optionally native replay decisions),
then aggregated by year, by the pre-registered time windows, and by condition:

* basic: games, kyoku, decisions, events/kyoku/decisions per game, and their
  p50/p90/p99/p999;
* action categories per kyoku: discards, chi, pon, kan (daiminkan/ankan/kakan),
  reach declarations and accepted reaches, hora (ron vs tsumo) and ryukyoku;
* rates with normal 95% CI: reach per kyoku, calls per kyoku, hora/ryukyoku per
  kyoku, deal-ins (ron events) per kyoku and per game;
* game termination: share of games that reached South 4 (or a West extension),
  and whether the final kyoku ended with hora or ryukyoku;
* conditional strata: ``window × bakaze × turn bucket`` for reach,
  ``window × bakaze`` for calls, and an explicit ``all_last`` stratum
  (``bakaze=S, kyoku=4``) — raw differences are *not* treated as skill changes;
* players: only an in-memory HMAC set (never names) if a key is provided;
* AI provenance: there is no local confirmed-AI evidence, so all players are
  ``unknown``; this module never emits any account list.

Semantics fixed for PR-07R:

* a ``hora`` is a **deal-in (ron) only when a valid ``target`` differs from
  ``actor``**; Tenhou logs set ``target == actor`` for tsumo, so counting every
  integer ``target`` as a deal-in would silently inflate the rate;
* logs must pass the canonical ``validate_sequence`` contract before they are
  counted as parsed; otherwise the caller reports the failure reason instead of
  silently aggregating a truncated game;
* a reach's "turn bucket" uses the **declaring player's own discard index**
  (junme), not the total number of discards in the kyoku;
* multi-ron is counted per winner (one discard can produce two ron events).

Time windows are fixed before inspecting results:
``2019_2022``, ``2023_2024``, ``2025``, ``2026`` (January only locally).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from riichiformer.data import canonicalize

WINDOWS = ("2019_2022", "2023_2024", "2025", "2026")
TURN_BUCKETS = ("0-5", "6-11", "12+")
KAN_TYPES = ("daiminkan", "ankan", "kakan")


def window_for(year):
    if isinstance(year, int):
        if 2019 <= year <= 2022:
            return "2019_2022"
        if 2023 <= year <= 2024:
            return "2023_2024"
        if year == 2025:
            return "2025"
        if year == 2026:
            return "2026"
    return "pre_2019" if isinstance(year, int) and year < 2019 else "unknown"


def turn_bucket(turn):
    if turn is None:
        return "unknown"
    if turn <= 5:
        return "0-5"
    if turn <= 11:
        return "6-11"
    return "12+"


def percentile(values: list[float], q: float):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(q * len(ordered))) - 1))
    return ordered[index]


def wilson_interval(success: int, total: int, z: float = 1.96):
    if total <= 0:
        return (None, None)
    p = success / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_interval(success: int, exposure: int, z: float = 1.96):
    """Normal-approx CI for an event count per exposure (rate may exceed 1).

    Poisson-style standard error ``sqrt(count)/exposure``; this is a *sampling*
    interval and deliberately does not model clustering by game or kyoku.
    """
    if exposure <= 0:
        return (None, None)
    rate = success / exposure
    se = math.sqrt(max(success, 0)) / exposure
    return (max(0.0, rate - z * se), rate + z * se)


def mean_ci(values: list[float], z: float = 1.96):
    if not values:
        return (None, None, None)
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return (mean, None, None)
    var = sum((value - mean) ** 2 for value in values) / (n - 1)
    half = z * math.sqrt(var / n)
    return (mean, mean - half, mean + half)


def _ci_fields(prefix: str, interval) -> dict:
    low, high = interval
    return {
        f"{prefix}_ci_low": None if low is None else round(low, 6),
        f"{prefix}_ci_high": None if high is None else round(high, 6),
    }


@dataclass
class GameDrift:
    game_id: str
    year: int | None
    month: int | None
    window: str
    events: int = 0
    kyoku_count: int = 0
    decisions: int | None = None
    reach_count: int = 0
    reach_accepted_count: int = 0
    call_count: int = 0
    chi_count: int = 0
    pon_count: int = 0
    daiminkan_count: int = 0
    ankan_count: int = 0
    kakan_count: int = 0
    dahai_count: int = 0
    hora_count: int = 0
    ron_count: int = 0
    tsumo_count: int = 0
    ryukyoku_count: int = 0
    hora_points: list[int] = field(default_factory=list)
    deal_ins: int = 0
    player_hmacs: list[str] = field(default_factory=list)
    kyoku_rows: list[dict] = field(default_factory=list)
    final_kyoku_label: str | None = None
    final_event: str | None = None
    reached_south4: bool = False
    replay_error: str | None = None

    @property
    def kan_count(self) -> int:
        return self.daiminkan_count + self.ankan_count + self.kakan_count

    def as_length(self):
        return {
            "events": self.events,
            "kyoku": self.kyoku_count,
            "decisions": self.decisions,
            "dahai": self.dahai_count,
            "chi": self.chi_count,
            "pon": self.pon_count,
            "kan": self.kan_count,
            "reach": self.reach_count,
            "hora": self.hora_count,
            "ryukyoku": self.ryukyoku_count,
        }


def _hmac(name: str, key: bytes) -> str:
    import hashlib
    import hmac

    return hmac.new(key, name.encode("utf-8"), hashlib.sha256).hexdigest()


def _kyoku_label(start_event: dict) -> str | None:
    bakaze = start_event.get("bakaze")
    kyoku = start_event.get("kyoku")
    if bakaze is None or kyoku is None:
        return None
    return f"{bakaze}{kyoku}"


def parse_game(raw: bytes, relpath: str, game_id: str, year: int | None,
               hmac_key: bytes | None = None, include_decisions: bool = False) -> GameDrift:
    """Parse one log; raises ``canonicalize.CanonicalizeError`` for invalid logs.

    Fail-closed: a log that does not satisfy ``validate_sequence`` is never
    counted as parsed.  The caller (CLI) records the exception's ``reason`` in
    its error table so failures are visible instead of silently dropped.
    """
    record = GameDrift(game_id=game_id, year=year, month=month_from_relpath(relpath),
                       window=window_for(year))
    compression = canonicalize.detect_compression_bytes(raw)
    payload = canonicalize.decode_log_bytes(raw, compression)
    events = canonicalize.parse_events(payload)
    canonicalize.validate_sequence(events)
    record.events = len(events)
    if events and isinstance(events[0].get("names"), list) and hmac_key is not None:
        for name in events[0]["names"]:
            if isinstance(name, str) and name:
                try:
                    record.player_hmacs.append(_hmac(name, hmac_key))
                except Exception:  # noqa: BLE001 - HMAC is optional metadata
                    pass

    current = None
    for event in events:
        etype = event.get("type")
        if etype == "start_kyoku":
            current = {
                "bakaze": event.get("bakaze"), "kyoku": event.get("kyoku"),
                "honba": event.get("honba", 0), "oya": event.get("oya"),
                "scores": event.get("scores"), "reach": 0, "calls": 0,
                "hora": 0, "ryukyoku": 0,
                "reach_turns": [], "discards_by_actor": {},
                "is_last": event.get("bakaze") == "S" and event.get("kyoku") == 4,
            }
            record.kyoku_rows.append(current)
            record.kyoku_count += 1
        elif current is None:
            continue
        elif etype == "dahai":
            record.dahai_count += 1
            actor = event.get("actor")
            if isinstance(actor, int):
                current["discards_by_actor"][actor] = (
                    current["discards_by_actor"].get(actor, 0) + 1)
        elif etype == "reach":
            current["reach"] += 1
            record.reach_count += 1
            actor = event.get("actor")
            own_discards = (current["discards_by_actor"].get(actor, 0)
                            if isinstance(actor, int) else None)
            current["reach_turns"].append(own_discards)
        elif etype == "reach_accepted":
            record.reach_accepted_count += 1
        elif etype == "chi":
            current["calls"] += 1
            record.chi_count += 1
            record.call_count += 1
        elif etype == "pon":
            current["calls"] += 1
            record.pon_count += 1
            record.call_count += 1
        elif etype == "daiminkan":
            current["calls"] += 1
            record.daiminkan_count += 1
            record.call_count += 1
        elif etype == "ankan":
            current["calls"] += 1
            record.ankan_count += 1
            record.call_count += 1
        elif etype == "kakan":
            current["calls"] += 1
            record.kakan_count += 1
            record.call_count += 1
        elif etype == "hora":
            current["hora"] += 1
            record.hora_count += 1
            record.final_event = "hora"
            actor = event.get("actor")
            target = event.get("target")
            deltas = event.get("deltas") or []
            if isinstance(actor, int) and 0 <= actor < len(deltas):
                try:
                    record.hora_points.append(int(deltas[actor]))
                except (TypeError, ValueError):
                    pass
            is_ron = (isinstance(actor, int) and isinstance(target, int)
                      and 0 <= target < 4 and target != actor)
            if is_ron:
                # A ron event is a deal-in by ``target``; multi-ron counts each winner.
                record.ron_count += 1
                record.deal_ins += 1
            else:
                # Tenhou sets target == actor for tsumo; an absent target is also tsumo.
                record.tsumo_count += 1
        elif etype == "ryukyoku":
            current["ryukyoku"] += 1
            record.ryukyoku_count += 1
            record.final_event = "ryukyoku"

    if record.kyoku_rows:
        last = record.kyoku_rows[-1]
        record.final_kyoku_label = _kyoku_label(last)
        # "Reached S4/W" means the last kyoku is South 4 or a West extension.
        # Checking only bakaze == "S" would count games that ended in S1/S2/S3
        # (e.g. a bust) as if they had reached South 4.
        record.reached_south4 = (
            last.get("bakaze") == "W"
            or (last.get("bakaze") == "S" and last.get("kyoku") == 4)
        )

    if include_decisions:
        from riichiformer.data import replay

        try:
            text = payload.decode("utf-8")
            result = replay.replay_game(text, oracle=False)
            if result.ok:
                record.decisions = sum(seat["decisions"] for seat in result.seats)
            else:
                record.replay_error = result.error_kind or "replay_failed"
        except Exception as exc:  # noqa: BLE001 - recorded as data-quality evidence
            record.replay_error = type(exc).__name__
    return record


def month_from_relpath(relpath: str):
    base = Path(relpath).name
    if len(base) >= 6 and base[:6].isdigit():
        month = int(base[4:6])
        return month if 1 <= month <= 12 else None
    return None


def _basic_fields(games: list[GameDrift]) -> dict:
    events = [g.events for g in games]
    kyoku = [g.kyoku_count for g in games]
    decisions = [g.decisions for g in games if g.decisions is not None]
    return {
        "games": len(games),
        "kyoku": sum(kyoku),
        "decisions": sum(decisions) if decisions else None,
        "replay_games": len(decisions),
        "events_per_game_p50": percentile(events, 0.50),
        "events_per_game_p90": percentile(events, 0.90),
        "events_per_game_p99": percentile(events, 0.99),
        "events_per_game_p999": percentile(events, 0.999),
        "kyoku_per_game_p50": percentile(kyoku, 0.50),
        "kyoku_per_game_p90": percentile(kyoku, 0.90),
        "decisions_per_game_p50": percentile(decisions, 0.50),
        "decisions_per_game_p90": percentile(decisions, 0.90),
        "decisions_per_game_p99": percentile(decisions, 0.99),
        "decisions_per_game_p999": percentile(decisions, 0.999),
        "dahai": sum(g.dahai_count for g in games),
        "chi": sum(g.chi_count for g in games),
        "pon": sum(g.pon_count for g in games),
        "daiminkan": sum(g.daiminkan_count for g in games),
        "ankan": sum(g.ankan_count for g in games),
        "kakan": sum(g.kakan_count for g in games),
        "kan": sum(g.kan_count for g in games),
        "reach_accepted": sum(g.reach_accepted_count for g in games),
        "ron_events": sum(g.ron_count for g in games),
        "tsumo": sum(g.tsumo_count for g in games),
        "reached_south4_games": sum(1 for g in games if g.reached_south4),
        "final_hora_games": sum(1 for g in games if g.final_event == "hora"),
        "final_ryukyoku_games": sum(1 for g in games if g.final_event == "ryukyoku"),
        "replay_errors": sum(1 for g in games if g.replay_error),
        "unique_players": len({hmac for g in games for hmac in g.player_hmacs}) if any(
            g.player_hmacs for g in games) else None,
    }


def aggregate_cohorts(games: list[GameDrift]) -> list[dict]:
    """Per-year and per-window rows with denominators, rates and 95% CIs."""
    by_year = defaultdict(list)
    by_month = defaultdict(list)
    by_window = defaultdict(list)
    for game in games:
        if game.year is not None:
            by_year[game.year].append(game)
            if game.month is not None:
                by_month[(game.year, game.month)].append(game)
        by_window[game.window].append(game)

    rows = []
    for scope, groups in (("year", by_year), ("month", by_month), ("window", by_window)):
        for key in sorted(groups):
            group = groups[key]
            basic = _basic_fields(group)
            kyoku = basic["kyoku"]
            reach = sum(g.reach_count for g in group)
            calls = sum(g.call_count for g in group)
            hora = sum(g.hora_count for g in group)
            ryukyoku = sum(g.ryukyoku_count for g in group)
            points = [p for g in group for p in g.hora_points]
            deal_ins = sum(g.deal_ins for g in group)
            reach_ci = rate_interval(reach, kyoku)
            call_ci = rate_interval(calls, kyoku)
            hora_ci = rate_interval(hora, kyoku)
            ryukyoku_ci = rate_interval(ryukyoku, kyoku)
            deal_ci = rate_interval(deal_ins, kyoku)
            mean_points, points_lo, points_hi = mean_ci(points)
            mean_deal, deal_lo, deal_hi = mean_ci([g.deal_ins for g in group])
            south4_ci = wilson_interval(basic["reached_south4_games"], basic["games"])
            final_hora_ci = wilson_interval(basic["final_hora_games"], basic["games"])
            final_ryukyoku_ci = wilson_interval(basic["final_ryukyoku_games"], basic["games"])
            cohort = f"{key[0]:04d}-{key[1]:02d}" if scope == "month" else str(key)
            rows.append({
                "scope": scope,
                "cohort": cohort,
                "window": group[0].window if group else None,
                **basic,
                "reach": reach, "reach_per_kyoku": round(reach / kyoku, 6) if kyoku else None,
                **_ci_fields("reach", reach_ci),
                "calls": calls, "calls_per_kyoku": round(calls / kyoku, 6) if kyoku else None,
                **_ci_fields("calls", call_ci),
                "hora_per_kyoku": round(hora / kyoku, 6) if kyoku else None,
                **_ci_fields("hora", hora_ci),
                "ryukyoku_per_kyoku": round(ryukyoku / kyoku, 6) if kyoku else None,
                **_ci_fields("ryukyoku", ryukyoku_ci),
                "dahai_per_kyoku": round(basic["dahai"] / kyoku, 6) if kyoku else None,
                "chi_per_kyoku": round(basic["chi"] / kyoku, 6) if kyoku else None,
                "pon_per_kyoku": round(basic["pon"] / kyoku, 6) if kyoku else None,
                "kan_per_kyoku": round(basic["kan"] / kyoku, 6) if kyoku else None,
                "ron_per_kyoku": round(deal_ins / kyoku, 6) if kyoku else None,
                "hora_events": len(points),
                "mean_hora_points": None if mean_points is None else round(mean_points, 3),
                "mean_hora_points_ci_low": None if points_lo is None else round(points_lo, 3),
                "mean_hora_points_ci_high": None if points_hi is None else round(points_hi, 3),
                "deal_ins": deal_ins,
                "mean_deal_ins_per_game": None if mean_deal is None else round(mean_deal, 6),
                "deal_in_ci_low": None if deal_lo is None else round(deal_lo, 6),
                "deal_in_ci_high": None if deal_hi is None else round(deal_hi, 6),
                "deal_ins_per_kyoku": round(deal_ins / kyoku, 6) if kyoku else None,
                **_ci_fields("deal_ins", deal_ci),
                "reached_south4_share": (round(basic["reached_south4_games"] / basic["games"], 6)
                                         if basic["games"] else None),
                **_ci_fields("reached_south4", south4_ci),
                "final_hora_share": (round(basic["final_hora_games"] / basic["games"], 6)
                                     if basic["games"] else None),
                **_ci_fields("final_hora", final_hora_ci),
                "final_ryukyoku_share": (round(basic["final_ryukyoku_games"] / basic["games"], 6)
                                         if basic["games"] else None),
                **_ci_fields("final_ryukyoku", final_ryukyoku_ci),
            })
    return rows


def aggregate_strata(games: list[GameDrift]) -> list[dict]:
    """Conditional strata; raw/unadjusted rates are explicitly labelled.

    * ``window_bakaze`` / ``window_bakaze_last``: per-kyoku rates;
    * ``window_bakaze_turn``: share of reach declarations by the declaring
      player's own discard-turn bucket (denominator = all reaches in that
      window/bakaze).
    """
    kyoku_buckets = defaultdict(lambda: {"kyoku": 0, "reach": 0, "calls": 0, "hora": 0})
    last_buckets = defaultdict(lambda: {"kyoku": 0, "reach": 0, "calls": 0, "hora": 0})
    reach_turns = defaultdict(Counter)
    reach_totals = Counter()

    for game in games:
        for row in game.kyoku_rows:
            bakaze = row.get("bakaze")
            for buckets, key in (
                (kyoku_buckets, (game.window, bakaze)),
                (last_buckets, (game.window, bakaze,
                                "all_last" if row.get("is_last") else "other")),
            ):
                bucket = buckets[key]
                bucket["kyoku"] += 1
                bucket["reach"] += int(row.get("reach", 0))
                bucket["calls"] += int(row.get("calls", 0))
                bucket["hora"] += int(row.get("hora", 0))
            for turn in row.get("reach_turns", []):
                reach_turns[(game.window, bakaze)][turn_bucket(turn)] += 1
                reach_totals[(game.window, bakaze)] += 1

    rows = []
    for (window, bakaze), bucket in sorted(
        kyoku_buckets.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        rows.append(_stratum_row("window_bakaze", f"{window}|{bakaze}", "kyoku", bucket))
    for (window, bakaze, label), bucket in sorted(
        last_buckets.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        rows.append(_stratum_row("window_bakaze_last", f"{window}|{bakaze}|{label}",
                                 "kyoku", bucket))
    for (window, bakaze), counts in sorted(reach_turns.items()):
        total = reach_totals[(window, bakaze)]
        for bucket_name in TURN_BUCKETS:
            reach = counts.get(bucket_name, 0)
            ci = wilson_interval(reach, total)
            rows.append({
                "stratum_type": "window_bakaze_turn",
                "stratum": f"{window}|{bakaze}|{bucket_name}",
                "denominator_kind": "reach",
                "kyoku": total,
                "reach": reach,
                "reach_per_kyoku": None,
                "reach_share": round(reach / total, 6) if total else None,
                "reach_ci_low": None if ci[0] is None else round(ci[0], 6),
                "reach_ci_high": None if ci[1] is None else round(ci[1], 6),
                "calls": 0,
                "calls_per_kyoku": None,
                "calls_ci_low": None,
                "calls_ci_high": None,
                "hora": 0,
                "note": "share of reach declarations by the declaring player's own "
                        "discard turn; unadjusted",
            })
    return rows


def _stratum_row(stratum_type: str, stratum: str, denominator_kind: str, bucket: dict) -> dict:
    reach_ci = rate_interval(bucket["reach"], bucket["kyoku"])
    call_ci = rate_interval(bucket["calls"], bucket["kyoku"])
    return {
        "stratum_type": stratum_type,
        "stratum": stratum,
        "denominator_kind": denominator_kind,
        "kyoku": bucket["kyoku"],
        "reach": bucket["reach"],
        "reach_per_kyoku": round(bucket["reach"] / bucket["kyoku"], 6) if bucket["kyoku"] else None,
        "reach_share": None,
        "reach_ci_low": None if reach_ci[0] is None else round(reach_ci[0], 6),
        "reach_ci_high": None if reach_ci[1] is None else round(reach_ci[1], 6),
        "calls": bucket["calls"],
        "calls_per_kyoku": round(bucket["calls"] / bucket["kyoku"], 6) if bucket["kyoku"] else None,
        "calls_ci_low": None if call_ci[0] is None else round(call_ci[0], 6),
        "calls_ci_high": None if call_ci[1] is None else round(call_ci[1], 6),
        "hora": bucket["hora"],
        "note": "unadjusted conditional rate; not a causal skill estimate",
    }


def cohort_windows_with_coverage(games: list[GameDrift]) -> dict:
    by_window = defaultdict(list)
    for game in games:
        by_window[game.window].append(game)
    coverage = {}
    for window in WINDOWS:
        group = by_window.get(window, [])
        years = sorted({g.year for g in group if g.year is not None})
        months = sorted({(g.year, g.month) for g in group if g.month is not None})
        coverage[window] = {
            "games": len(group),
            "years": years,
            "months": len(months),
            "kyoku": sum(g.kyoku_count for g in group),
            "decisions": (sum(g.decisions for g in group
                              if g.decisions is not None)
                          if any(g.decisions is not None for g in group) else None),
            "note": "2026 is January-only locally" if window == "2026" else "",
        }
    return coverage
