"""Read-only libriichi replay adapter for RiichiFormer V2 (PR-04) and H-01 scores.

The Rust engine in the pinned submodule is the **only** authority for legal actions:

* ``GameplayLoader.load_log`` runs the native ``PlayerState`` state machine and returns one
  ``Gameplay`` per seat with ``actions`` and 46-wide ``masks`` at every decision point;
* this wrapper validates every logged action against its mask, records illegal/mismatched
  decisions with the exact decision index, and never repairs or re-ranks an action.

Score settlement (H-01) has exactly one entry point, :func:`replay_scores`:

* ``start_kyoku.scores`` is the authoritative anchor at every kyoku boundary; the replay
  settles each kyoku from its events and cross-checks the result against the *next*
  ``start_kyoku`` (scores **and** kyotaku);
* ``reach_accepted`` deducts 1000 and increments kyotaku.  A ``reach`` declaration that is
  never accepted is not charged, except for the Tenhou four-riichi abort where the fourth
  declaration has no ``reach_accepted`` event but is still paid (verified on real logs);
* ``hora.deltas`` in Tenhou-derived MJAI logs **already include** ``1000 * kyotaku`` for the
  first winner of a settlement (``arena/board.rs`` builds
  ``deltas[actor] = point.ron + kyotaku_point + ...`` and then clears the board kyotaku).
  A second winner of a multi-ron therefore carries no kyotaku.  The old extra
  ``+1000 * kyotaku`` is gone; a delta sum inconsistent with the live kyotaku is a hard
  mismatch instead of being silently repaired;
* ``ryukyoku.deltas`` never include kyotaku (the sticks stay on the table);
* leftover kyotaku at ``end_game`` is awarded to the top scorer (highest score, lowest seat
  index on ties), matching ``arena/game.rs`` ``commit()``;
* a score source whose protocol is not verified (for example live-Tenhou POV logs whose
  ``hora`` carries absolute ``scores`` instead of ``deltas``) is rejected.  The strict
  entry point raises :class:`ScoreProtocolError`; the legacy
  :func:`final_scores_from_events` returns ``None`` (fail closed) instead of inventing
  scores.

``kyoku_count`` is counted from raw ``start_kyoku`` boundaries, never from the decisions of
one seat; :func:`replay_game` cross-checks it against the native ``at_kyoku`` stream and
records a mismatch instead of silently under-counting.

Nothing here writes data, downloads anything, or invents hidden events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

ACTION_SPACE = 46  # libriichi.consts.ACTION_SPACE; asserted at replay time

KYOTAKU_POINTS = 1000

#: Tenhou-derived MJAI logs: ``hora.deltas`` include the live kyotaku, ``ryukyoku``
#: deltas do not, ``start_kyoku`` carries the authoritative scores/kyotaku.
PROTOCOL_TENHOU_DELTAS = "tenhou_mjai_deltas_v1"
#: Live-Tenhou POV logs: ``hora``/``end_game`` carry absolute ``scores``.  Recognised but
#: intentionally unsupported until independently verified (fail closed).
PROTOCOL_ABSOLUTE_SCORES = "absolute_hora_scores"
KNOWN_PROTOCOLS = (PROTOCOL_TENHOU_DELTAS, PROTOCOL_ABSOLUTE_SCORES)

RULESET_TENHOU = "tenhou"
SUPPORTED_RULESETS = (RULESET_TENHOU,)


class ReplayUnavailable(RuntimeError):
    """libriichi is not importable in this interpreter."""


class ScoreProtocolError(ValueError):
    """The event source does not use a verified score protocol."""


def load_libriichi():
    try:
        import libriichi  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ReplayUnavailable(
            "libriichi is not importable; run with "
            "PYTHONPATH=vendor/mortal/target/release and the .venv interpreter"
        ) from exc
    return libriichi


# ---------------------------------------------------------------------------
# Score settlement
# ---------------------------------------------------------------------------


@dataclass
class ScoreReplay:
    """Result of :func:`replay_scores` (scores plus the evidence behind them)."""

    ok: bool = True
    protocol: str | None = None
    ruleset_id: str = RULESET_TENHOU
    protocol_error: bool = False
    scores: list[int] | None = None
    scores_before_final_kyotaku: list[int] | None = None
    kyotaku_left: int = 0
    kyoku_count: int = 0
    hora_count: int = 0
    multi_ron_count: int = 0
    final_kyotaku_award_seat: int | None = None
    transitions: list[dict] = field(default_factory=list)
    hora_records: list[dict] = field(default_factory=list)
    mismatches: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "protocol": self.protocol,
            "ruleset_id": self.ruleset_id,
            "protocol_error": self.protocol_error,
            "scores": list(self.scores) if self.scores is not None else None,
            "scores_before_final_kyotaku": (
                list(self.scores_before_final_kyotaku)
                if self.scores_before_final_kyotaku is not None else None
            ),
            "kyotaku_left": self.kyotaku_left,
            "kyoku_count": self.kyoku_count,
            "hora_count": self.hora_count,
            "multi_ron_count": self.multi_ron_count,
            "final_kyotaku_award_seat": self.final_kyotaku_award_seat,
            "transitions": [dict(item) for item in self.transitions],
            "hora_records": [dict(item) for item in self.hora_records],
            "mismatches": [dict(item) for item in self.mismatches],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "flags": list(self.flags),
        }


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_scores(value) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(_is_int(v) for v in value)


def _valid_deltas(value) -> bool:
    return _valid_scores(value)


def _actor(event: dict):
    actor = event.get("actor")
    return actor if _is_int(actor) and 0 <= actor < 4 else None


def _resolve_protocol(events: list[dict], requested: str) -> str:
    if requested not in ("auto",) + KNOWN_PROTOCOLS:
        raise ScoreProtocolError(
            f"unknown score protocol {requested!r}; expected 'auto' or one of "
            f"{', '.join(KNOWN_PROTOCOLS)}"
        )
    hora_events = [e for e in events if isinstance(e, dict) and e.get("type") == "hora"]
    if not hora_events:
        # No hora means the kyotaku ambiguity is not exercised; ryukyoku deltas and
        # start_kyoku anchors are still verified by the same protocol.
        return requested if requested != "auto" else PROTOCOL_TENHOU_DELTAS
    with_deltas = [e for e in hora_events if _valid_deltas(e.get("deltas"))]
    with_scores = [e for e in hora_events if _valid_scores(e.get("scores"))]
    if len(with_deltas) == len(hora_events):
        detected = PROTOCOL_TENHOU_DELTAS
    elif not with_deltas and len(with_scores) == len(hora_events):
        detected = PROTOCOL_ABSOLUTE_SCORES
    else:
        raise ScoreProtocolError(
            "unverified hora score fields: every hora must carry either deltas "
            f"(got {len(with_deltas)}/{len(hora_events)}) or absolute scores "
            f"(got {len(with_scores)}/{len(hora_events)}); mixed logs are rejected"
        )
    if requested != "auto" and requested != detected:
        raise ScoreProtocolError(
            f"score protocol is {detected!r}, not the requested {requested!r}"
        )
    return detected


def _issue(result: ScoreReplay, kind: str, detail: dict) -> None:
    result.mismatches.append({"kind": kind, **detail})
    result.errors.append(f"{kind}: {detail}")
    result.ok = False


def _replay_tenhou_deltas(events: list[dict], result: ScoreReplay,
                         award_final_kyotaku: bool) -> None:
    scores: list[int] | None = None
    initial_total: int | None = None
    kyotaku = 0
    kyoku_count = 0
    kyoku_open = False
    accepted_reaches = 0
    pending_reach_actor: int | None = None
    hora_in_kyoku = 0
    current_kyoku: dict = {}
    ended = False

    for event in events:
        if not isinstance(event, dict):
            _issue(result, "malformed_event", {"event": repr(event)[:120]})
            continue
        etype = event.get("type")
        if not isinstance(etype, str):
            _issue(result, "malformed_event", {"event": repr(event)[:120]})
            continue

        if etype == "start_kyoku":
            raw_scores = event.get("scores")
            if not _valid_scores(raw_scores):
                _issue(result, "invalid_start_kyoku_scores",
                       {"event": event, "kyoku_index": kyoku_count + 1})
                continue
            new_scores = [int(v) for v in raw_scores]
            raw_kyotaku = event.get("kyotaku")
            if raw_kyotaku is None:
                # All audited real sources carry kyotaku; a minimal fixture defaults to
                # 0 and is surfaced as a warning (not a silent settlement assumption).
                new_kyotaku = 0
                if "missing_start_kyoku_kyotaku" not in result.warnings:
                    result.warnings.append("missing_start_kyoku_kyotaku")
            elif not _is_int(raw_kyotaku) or raw_kyotaku < 0:
                _issue(result, "invalid_start_kyoku_kyotaku",
                       {"event": event, "kyoku_index": kyoku_count + 1})
                new_kyotaku = 0
            else:
                new_kyotaku = int(raw_kyotaku)

            if scores is not None:
                transition = {
                    "kyoku_index": kyoku_count + 1,
                    "previous_kyoku": dict(current_kyoku),
                    "settled_scores": list(scores),
                    "settled_kyotaku": kyotaku,
                    "next_scores": list(new_scores),
                    "next_kyotaku": new_kyotaku,
                    "next_kyoku": {"bakaze": event.get("bakaze"),
                                   "kyoku": event.get("kyoku"),
                                   "honba": event.get("honba")},
                    "match": (list(scores) == new_scores and kyotaku == new_kyotaku),
                }
                result.transitions.append(transition)
                if not transition["match"]:
                    _issue(result, "kyoku_transition", transition)

            scores = new_scores
            if initial_total is None:
                # start_kyoku.scores excludes the sticks still on the table.
                initial_total = sum(scores) + KYOTAKU_POINTS * new_kyotaku
            kyotaku = new_kyotaku
            kyoku_count += 1
            kyoku_open = True
            accepted_reaches = 0
            pending_reach_actor = None
            hora_in_kyoku = 0
            current_kyoku = {"bakaze": event.get("bakaze"), "kyoku": event.get("kyoku"),
                             "honba": event.get("honba")}
            continue

        if etype == "end_kyoku":
            # A kyoku boundary is authoritative even if the source only reports an
            # (empty) settlement via end_kyoku; settle any unresolved declaration state.
            kyoku_open = False
            pending_reach_actor = None
            continue

        if etype == "reach":
            actor = _actor(event)
            if actor is None:
                _issue(result, "invalid_reach_actor", {"event": event})
            elif pending_reach_actor is not None:
                _issue(result, "multiple_pending_reach",
                       {"pending": pending_reach_actor, "event": event})
            else:
                pending_reach_actor = actor
            continue

        if etype == "reach_accepted":
            if scores is None:
                _issue(result, "score_event_before_start_kyoku", {"event": event})
                continue
            actor = _actor(event)
            if actor is None:
                _issue(result, "invalid_reach_actor", {"event": event})
                continue
            scores[actor] -= KYOTAKU_POINTS
            kyotaku += 1
            accepted_reaches += 1
            pending_reach_actor = None
            continue

        if etype == "hora":
            if scores is None:
                _issue(result, "score_event_before_start_kyoku", {"event": event})
                continue
            deltas = event.get("deltas")
            if not _valid_deltas(deltas):
                _issue(result, "invalid_hora_deltas", {"event": event})
                continue
            if _actor(event) is None:
                _issue(result, "invalid_hora_actor", {"event": event})
                continue
            delta_sum = sum(int(v) for v in deltas)
            expected = KYOTAKU_POINTS * kyotaku
            record = {
                "actor": event.get("actor"),
                "target": event.get("target"),
                "delta_sum": delta_sum,
                "expected_delta_sum": expected,
                "kyotaku_before": kyotaku,
                "multi_ron": hora_in_kyoku > 0,
                "kyoku": dict(current_kyoku),
            }
            result.hora_records.append(record)
            result.hora_count += 1
            if delta_sum != expected:
                _issue(result, "hora_delta_sum_mismatch", record)
            if hora_in_kyoku > 0:
                result.multi_ron_count += 1
                if "multi_ron" not in result.flags:
                    result.flags.append("multi_ron")
            elif expected > 0:
                if "hora_kyotaku_in_deltas" not in result.flags:
                    result.flags.append("hora_kyotaku_in_deltas")
            scores = [int(s) + int(d) for s, d in zip(scores, deltas)]
            kyotaku = 0
            hora_in_kyoku += 1
            pending_reach_actor = None
            kyoku_open = False
            continue

        if etype == "ryukyoku":
            if scores is None:
                _issue(result, "score_event_before_start_kyoku", {"event": event})
                continue
            deltas = event.get("deltas")
            if not _valid_deltas(deltas):
                _issue(result, "invalid_ryukyoku_deltas", {"event": event})
                continue
            ds = [int(v) for v in deltas]
            if pending_reach_actor is not None and all(v == 0 for v in ds) \
                    and accepted_reaches == 3:
                # Four-riichi abort (四家立直): Tenhou does not emit reach_accepted for
                # the fourth declaration, but the stick is still paid and carried.
                scores[pending_reach_actor] -= KYOTAKU_POINTS
                kyotaku += 1
                accepted_reaches += 1
                pending_reach_actor = None
                if "four_riichi_pending_reach_charged" not in result.flags:
                    result.flags.append("four_riichi_pending_reach_charged")
            elif pending_reach_actor is not None:
                _issue(result, "unverified_pending_reach_on_ryukyoku",
                       {"pending_reach_actor": pending_reach_actor,
                        "deltas": ds, "accepted_reaches": accepted_reaches,
                        "kyoku": dict(current_kyoku)})
            delta_sum = sum(ds)
            if delta_sum != 0:
                _issue(result, "ryukyoku_delta_sum_nonzero",
                       {"delta_sum": delta_sum, "deltas": ds,
                        "kyotaku": kyotaku, "kyoku": dict(current_kyoku)})
            scores = [int(s) + int(d) for s, d in zip(scores, ds)]
            pending_reach_actor = None
            kyoku_open = False
            continue

        if etype == "end_game":
            if ended:
                continue
            ended = True
            if scores is None:
                _issue(result, "no_start_kyoku", {"event": event})
                continue
            result.scores_before_final_kyotaku = list(scores)
            result.kyotaku_left = kyotaku
            if kyoku_open:
                # An interrupted game has no defined final ranking; refuse instead of
                # awarding live sticks to a possibly arbitrary leader.
                _issue(result, "unsettled_end_game",
                       {"kyoku": dict(current_kyoku), "scores": list(scores),
                        "kyotaku": kyotaku})
                continue
            if kyotaku > 0 and award_final_kyotaku:
                top = max(range(4), key=lambda seat: (scores[seat], -seat))
                scores[top] += KYOTAKU_POINTS * kyotaku
                result.final_kyotaku_award_seat = top
                result.flags.append("final_kyotaku_awarded")
            provided = event.get("scores")
            if _valid_scores(provided):
                provided_scores = [int(v) for v in provided]
                if provided_scores != scores:
                    _issue(result, "end_game_anchor_mismatch",
                           {"provided": provided_scores, "replayed": list(scores)})
            elif provided is not None:
                _issue(result, "invalid_end_game_scores", {"event": event})
            kyotaku = 0
            continue

    if not ended:
        result.warnings.append("no_end_game")
        if scores is not None:
            result.scores_before_final_kyotaku = list(scores)
            result.kyotaku_left = kyotaku

    if scores is None:
        if not any(item["kind"] == "no_start_kyoku" for item in result.mismatches):
            _issue(result, "no_start_kyoku", {"events": len(events)})
        result.scores = None
    else:
        result.scores = scores
        if ended and initial_total is not None:
            expected_total = initial_total
            if not award_final_kyotaku and result.kyotaku_left > 0:
                # Raw inspection mode: the sticks are still on the table.
                expected_total -= KYOTAKU_POINTS * result.kyotaku_left
            if sum(scores) != expected_total:
                _issue(result, "score_total_mismatch",
                       {"initial_total": initial_total, "expected_total": expected_total,
                        "final_total": sum(scores), "scores": list(scores)})
    result.kyoku_count = kyoku_count


def replay_scores(events, *, protocol: str = "auto", ruleset_id: str = RULESET_TENHOU,
                  award_final_kyotaku: bool = True,
                  strict_protocol: bool = True) -> ScoreReplay:
    """Authoritative score settlement for one MJAI event stream.

    Parameters
    ----------
    events:
        Parsed MJAI event dicts (the score events are ``start_kyoku``,
        ``reach``/``reach_accepted``, ``hora``, ``ryukyoku`` and ``end_game``).
    protocol:
        ``"auto"`` (default) detects the verified ``tenhou_mjai_deltas_v1`` protocol;
        an explicit constant is accepted for already-audited callers.
    ruleset_id:
        Only the Tenhou-derived ruleset is modelled here; anything else fails closed.
    award_final_kyotaku:
        Award leftover sticks to the top scorer at ``end_game`` (native rule).
    strict_protocol:
        Raise :class:`ScoreProtocolError` for unverified sources instead of returning
        ``ok=False``.  :func:`final_scores_from_events` passes ``False``.
    """
    events = list(events)
    result = ScoreReplay(ruleset_id=ruleset_id)
    if ruleset_id not in SUPPORTED_RULESETS:
        message = (
            f"unsupported ruleset_id {ruleset_id!r}; only {RULESET_TENHOU!r} settlement "
            "semantics are verified"
        )
        if strict_protocol:
            raise ScoreProtocolError(message)
        result.protocol_error = True
        result.ok = False
        result.errors.append(message)
        return result
    try:
        detected = _resolve_protocol(events, protocol)
    except ScoreProtocolError as exc:
        if strict_protocol:
            raise
        result.protocol_error = True
        result.ok = False
        result.errors.append(str(exc))
        return result
    result.protocol = detected
    if detected != PROTOCOL_TENHOU_DELTAS:
        message = (
            "unsupported_score_protocol: hora carries absolute scores instead of deltas "
            "(live-Tenhou POV logs). The delta/kyotaku convention is not verified for "
            "this source; refusing to derive scores."
        )
        if strict_protocol:
            raise ScoreProtocolError(message)
        result.protocol_error = True
        result.ok = False
        result.errors.append(message)
        return result

    _replay_tenhou_deltas(events, result, award_final_kyotaku)
    if result.errors or result.mismatches:
        result.ok = False
    return result


def final_scores_from_events(events):
    """Legacy compatibility wrapper; delegates to :func:`replay_scores`.

    Returns ``None`` when the source protocol is unverified or the replay found a score
    inconsistency, so callers never rank players from untrusted settlement.
    """
    result = replay_scores(events, strict_protocol=False)
    if not result.ok:
        return None
    return result.scores


# ---------------------------------------------------------------------------
# Action replay
# ---------------------------------------------------------------------------


@dataclass
class ReplayGame:
    ok: bool = True
    error: str | None = None
    error_kind: str | None = None
    event_count: int = 0
    kyoku_count: int = 0
    native_kyoku_count: int = 0
    seat0_kyoku_count: int = 0
    protocol: str | None = None
    kyotaku_left: int = 0
    final_scores: list[int] | None = None
    seats: list[dict] = field(default_factory=list)
    masks: list[list[list[bool]]] | None = None
    kyoku_cross_check: dict = field(default_factory=dict)
    score_mismatches: list[dict] = field(default_factory=list)
    score_flags: list[str] = field(default_factory=list)
    score_errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "error": self.error, "error_kind": self.error_kind,
            "event_count": self.event_count, "kyoku_count": self.kyoku_count,
            "native_kyoku_count": self.native_kyoku_count,
            "seat0_kyoku_count": self.seat0_kyoku_count,
            "protocol": self.protocol, "kyotaku_left": self.kyotaku_left,
            "final_scores": self.final_scores,
            "kyoku_cross_check": dict(self.kyoku_cross_check),
            "score_mismatches": [dict(item) for item in self.score_mismatches],
            "score_flags": list(self.score_flags),
            "score_errors": list(self.score_errors),
            "seats": [dict(seat) for seat in self.seats],
        }


def _mask_to_bools(mask) -> list[bool]:
    return [bool(value) for value in mask]


def replay_game(raw_text: str, oracle: bool = False,
                include_masks: bool = False) -> ReplayGame:
    """Replay one MJAI NDJSON game with libriichi; never raises on invalid games."""
    libriichi = load_libriichi()
    try:
        events = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        return ReplayGame(ok=False, error=f"bad_json: {exc}", error_kind="bad_json")
    result = ReplayGame(event_count=len(events))
    score_result = replay_scores(events, strict_protocol=False)

    try:
        loader = libriichi.dataset.GameplayLoader(4, oracle=oracle)
        gameplays = loader.load_log(raw_text)
    except Exception as exc:  # noqa: BLE001 - native error is evidence
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        result.error_kind = "libriichi_load"
        return result
    if len(gameplays) != 4:
        # A normal full four-player log must yield exactly four seat gameplays.  The old
        # code treated 0 seats as success, which let empty logs pass validation.
        result.ok = False
        result.error = f"expected 4 seats, got {len(gameplays)}"
        result.error_kind = "seat_count"
        result.final_scores = score_result.scores if score_result.ok else None
        return result

    all_masks = []
    seat_at_kyoku: list[list[int]] = []
    for seat_index, gameplay in enumerate(gameplays):
        actions = list(gameplay.take_actions())
        masks = list(gameplay.take_masks())
        kyoku = list(gameplay.take_at_kyoku())
        turns = list(gameplay.take_at_turns())
        player_id = int(gameplay.take_player_id())
        seat_at_kyoku.append([int(value) for value in kyoku])
        seat = {
            "seat": seat_index,
            "player_id": player_id,
            "decisions": len(actions),
            "invalid_actions": [],
            "mask_shape_bad": [],
            "min_legal_count": None,
            "max_legal_count": None,
        }
        if len(masks) != len(actions):
            seat["mask_shape_bad"].append({"actions": len(actions), "masks": len(masks)})
            result.ok = False
            result.error = "actions/masks length mismatch"
            result.error_kind = "mask_length"
        legal_counts = []
        seat_masks = []
        for index, action in enumerate(actions):
            if index >= len(masks):
                break
            mask = _mask_to_bools(masks[index])
            if len(mask) != ACTION_SPACE:
                seat["mask_shape_bad"].append({"index": index, "width": len(mask)})
                result.ok = False
                result.error = f"mask width {len(mask)} != {ACTION_SPACE}"
                result.error_kind = "mask_width"
                break
            legal_counts.append(sum(mask))
            seat_masks.append(mask)
            if not (0 <= int(action) < ACTION_SPACE) or not mask[int(action)]:
                seat["invalid_actions"].append({
                    "index": index,
                    "action": int(action),
                    "legal": bool(0 <= int(action) < ACTION_SPACE and mask[int(action)]),
                    "legal_count": int(sum(mask)),
                    "kyoku": int(kyoku[index]) if index < len(kyoku) else None,
                    "turn": int(turns[index]) if index < len(turns) else None,
                })
        if legal_counts:
            seat["min_legal_count"] = min(legal_counts)
            seat["max_legal_count"] = max(legal_counts)
        if include_masks:
            all_masks.append(seat_masks)
        result.seats.append(seat)

    native_indices = [value for seat_values in seat_at_kyoku for value in seat_values]
    native_max = (max(native_indices) + 1) if native_indices else 0
    native_distinct = len(set(native_indices))
    seat0_distinct = len(set(seat_at_kyoku[0])) if seat_at_kyoku else 0
    result.kyoku_count = score_result.kyoku_count
    result.native_kyoku_count = native_max
    result.seat0_kyoku_count = seat0_distinct
    result.protocol = score_result.protocol
    result.kyotaku_left = score_result.kyotaku_left
    result.score_mismatches = list(score_result.mismatches)
    result.score_flags = list(score_result.flags)
    result.score_errors = list(score_result.errors)
    result.kyoku_cross_check = {
        "raw": score_result.kyoku_count,
        "native_max": native_max,
        "native_distinct": native_distinct,
        "seat0_distinct": seat0_distinct,
        "match": score_result.kyoku_count == native_max,
    }
    if score_result.kyoku_count > 0 and native_max > score_result.kyoku_count:
        result.ok = False
        result.error = (
            "native saw more kyokus than the raw event boundaries "
            f"(raw={score_result.kyoku_count}, native={native_max})"
        )
        result.error_kind = "kyoku_count_mismatch"
    elif score_result.kyoku_count > 0 and native_max < score_result.kyoku_count:
        # Raw boundaries stay authoritative (a trailing zero-decision abortive kyoku has
        # no native at_kyoku entry), but surface the gap instead of hiding it.
        if "kyoku_count_native_lower" not in result.score_flags:
            result.score_flags.append("kyoku_count_native_lower")
    if score_result.kyoku_count > 0 and seat0_distinct < score_result.kyoku_count:
        if "seat0_kyoku_undercount" not in result.score_flags:
            result.score_flags.append("seat0_kyoku_undercount")

    if not score_result.ok:
        result.final_scores = None
        if result.ok:
            result.ok = False
            result.error_kind = ("score_protocol" if score_result.protocol_error
                                 else "score_mismatch")
            result.error = "; ".join(score_result.errors[:3]) or "score settlement mismatch"
    else:
        result.final_scores = score_result.scores

    if include_masks:
        result.masks = all_masks
    if not result.ok and result.error is None:
        result.error = "replay_validation_failed"
    if any(seat["invalid_actions"] for seat in result.seats):
        result.ok = False
        result.error = "illegal_or_mismatched_action"
        result.error_kind = "illegal_action"
    return result


def summarize_replay(result: ReplayGame) -> dict:
    return {
        "ok": result.ok,
        "error_kind": result.error_kind,
        "error": result.error,
        "events": result.event_count,
        "decisions": sum(seat["decisions"] for seat in result.seats),
        "illegal": sum(len(seat["invalid_actions"]) for seat in result.seats),
        "mask_width_errors": sum(len(seat["mask_shape_bad"]) for seat in result.seats),
        "final_scores": result.final_scores,
        "protocol": result.protocol,
        "kyoku_count": result.kyoku_count,
        "native_kyoku_count": result.native_kyoku_count,
        "seat0_kyoku_count": result.seat0_kyoku_count,
        "kyotaku_left": result.kyotaku_left,
        "score_mismatches": len(result.score_mismatches),
        "score_flags": list(result.score_flags),
    }
