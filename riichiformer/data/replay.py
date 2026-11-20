"""Read-only libriichi replay adapter for RiichiFormer V2 (PR-04).

The Rust engine in the pinned submodule is the **only** authority for legal actions:

* ``GameplayLoader.load_log`` runs the native ``PlayerState`` state machine and returns one
  ``Gameplay`` per seat with ``actions`` and 46-wide ``masks`` at every decision point;
* this wrapper validates every logged action against its mask, records illegal/mismatched
  decisions with the exact decision index, and never repairs or re-ranks an action;
* final scores are derived from the raw public events using the same deltas as
  ``scripts/eval_report.py`` (``start_kyoku.scores`` authoritative, ``reach_accepted``
  deducts 1000 into kyotaku, ``hora``/``ryukyoku`` deltas, kyotaku to the winner).

Nothing here writes data, downloads anything, or invents hidden events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ACTION_SPACE = 46  # libriichi.consts.ACTION_SPACE; asserted at replay time


class ReplayUnavailable(RuntimeError):
    """libriichi is not importable in this interpreter."""


def load_libriichi():
    try:
        import libriichi  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ReplayUnavailable(
            "libriichi is not importable; run with "
            "PYTHONPATH=vendor/mortal/target/release and the .venv interpreter"
        ) from exc
    return libriichi


def final_scores_from_events(events: list[dict]):
    """Deterministic final scores from public events; mirrors eval_report.py semantics."""
    scores = None
    kyotaku = 0
    for event in events:
        etype = event.get("type")
        if etype == "start_kyoku":
            value = event.get("scores")
            if isinstance(value, list) and len(value) == 4:
                scores = [int(v) for v in value]
            kyotaku = int(event.get("kyotaku", 0) or 0)
        elif etype == "reach_accepted" and scores is not None:
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                scores[actor] -= 1000
                kyotaku += 1
        elif etype in ("hora", "ryukyoku") and scores is not None:
            deltas = event.get("deltas")
            if isinstance(deltas, list) and len(deltas) == 4:
                scores = [score + int(delta) for score, delta in zip(scores, deltas)]
            if etype == "hora" and kyotaku:
                actor = event.get("actor")
                if isinstance(actor, int) and 0 <= actor < 4:
                    scores[actor] += 1000 * kyotaku
                kyotaku = 0
    return scores


@dataclass
class ReplayGame:
    ok: bool = True
    error: str | None = None
    error_kind: str | None = None
    event_count: int = 0
    kyoku_count: int = 0
    final_scores: list[int] | None = None
    seats: list[dict] = field(default_factory=list)
    masks: list[list[list[bool]]] | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "error": self.error, "error_kind": self.error_kind,
            "event_count": self.event_count, "kyoku_count": self.kyoku_count,
            "final_scores": self.final_scores,
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
    result.final_scores = final_scores_from_events(events)
    try:
        loader = libriichi.dataset.GameplayLoader(4, oracle=oracle)
        gameplays = loader.load_log(raw_text)
    except Exception as exc:  # noqa: BLE001 - native error is evidence
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        result.error_kind = "libriichi_load"
        return result
    if len(gameplays) not in (0, 4):
        result.ok = False
        result.error = f"expected 4 seats, got {len(gameplays)}"
        result.error_kind = "seat_count"
        return result

    all_masks = []
    for seat_index, gameplay in enumerate(gameplays):
        actions = list(gameplay.take_actions())
        masks = list(gameplay.take_masks())
        kyoku = list(gameplay.take_at_kyoku())
        turns = list(gameplay.take_at_turns())
        player_id = int(gameplay.take_player_id())
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

    if result.seats:
        kyoku_values = []
        try:
            kyoku_values = list(gameplays[0].take_at_kyoku())
        except Exception:  # pragma: no cover
            pass
        result.kyoku_count = len(set(int(value) for value in kyoku_values)) if kyoku_values else 0
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
    }
