#!/usr/bin/env python3
"""PR-16 mock arena CLI: deterministic 4-seat hand loop for the inference adapter.

This script deliberately runs a **mock** game, not a real rules engine:

* each seat gets a synthetic, schema-valid :class:`DecisionInput` (visible prefix,
  public state, native-width 46 mask, phase) built from a seeded tile pool;
* the :class:`RiichiFormerAgent` runs the PR-11 encoder (hash/schema checks) and a
  deterministic mock policy, then the adapter decodes the action to a concrete MJAI
  reaction and validates it against the mask;
* hands are 4-seat rotated, seeds/keys are explicit, and the output directory is
  mandatory.  Frozen evaluation directories (``runs/1v3``, ``runs/variance``,
  datasets/checkpoints/vendor) are refused.

No model checkpoint, dataset, libriichi call, network access or training write is
performed here.  Real checkpoint evaluation is PR-14/PR-17 work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.encode import (  # noqa: E402
    SCHEMA_HASH,
    StreamMeta,
    encode_decision,
)
from riichiformer.data.pov import POV_PROJECTION_VERSION  # noqa: E402
from riichiformer.runtime.agent import (  # noqa: E402
    AgentError,
    MockPolicy,
    RiichiFormerAgent,
)
from riichiformer.schemas.action_v3 import (  # noqa: E402
    ACTION_SPACE,
    SCHEMA_VERSION,
    ActionPhase,
)
from riichiformer.schemas.decision_v3 import (  # noqa: E402
    DecisionInput,
    GlobalPublicState,
    PlayerPublicState,
    RiichiState,
    RulesetId,
    SourceKind,
    VisibleTileState,
)
from riichiformer.schemas.event_v3 import (  # noqa: E402
    DahaiEvent,
    StartGameEvent,
    StartKyokuEvent,
    TileRef,
    TsumoEvent,
    tile_from_text,
)

MOCK_NOTE = (
    "MOCK arena only: relative-strength plumbing check, not a Tenhou-equivalent "
    "evaluation and not a strength claim. H-04 rules deviations R02 (triple ron "
    "settled by the pinned arena), R25 (last-dealer stop uses kyotaku-inclusive "
    "scores) and R26 (four-riichi abort after one extra draw) remain in force."
)

#: Directories that must never receive an arena run output.
FROZEN_OUTPUTS = ("runs/1v3", "runs/variance", "datasets", "checkpoints", "vendor")


def guard_out(out: str | Path) -> Path:
    """Resolve and validate the mandatory output directory (no frozen paths)."""
    if out is None or str(out).strip() == "":
        raise SystemExit("rf_eval: --out is required (refusing to guess a path)")
    resolved = Path(out).expanduser().resolve()
    root = REPO_ROOT.resolve()
    forbidden = [root / relative for relative in FROZEN_OUTPUTS]
    if resolved == root or resolved == (root / "runs").resolve():
        raise SystemExit(f"rf_eval: refusing repository root / runs root: {resolved}")
    for frozen in forbidden:
        if resolved == frozen or frozen in resolved.parents:
            raise SystemExit(
                f"rf_eval: refusing frozen path {resolved} (under {frozen})")
    if resolved.exists() and not resolved.is_dir():
        raise SystemExit(f"rf_eval: output path exists and is not a directory: {resolved}")
    return resolved


def _tile(kind: int, aka: int | None = None) -> TileRef:
    return TileRef(kind, aka)


def _hand_tiles(rng: random.Random, count: int = 14) -> list[TileRef]:
    """Seeded hand with at most four copies per kind and at most one aka per suit."""
    pool: list[int] = []
    for kind in range(34):
        pool.extend([kind] * 4)
    rng.shuffle(pool)
    picked: list[int] = []
    counts: Counter[int] = Counter()
    for kind in pool:
        if counts[kind] >= 4:
            continue
        picked.append(kind)
        counts[kind] += 1
        if len(picked) == count:
            break
    if len(picked) != count:  # pragma: no cover - 136-tile pool always suffices
        raise RuntimeError("could not build a mock hand")
    tiles = [_tile(kind) for kind in picked]
    # Every third hand: turn one five into its aka to exercise aka decoding.
    if rng.random() < 1.0 / 3.0:
        for index, tile in enumerate(tiles):
            if tile.kind in (4, 13, 22):
                color = (4, 13, 22).index(tile.kind)
                tiles[index] = _tile(tile.kind, color)
                break
    return tiles


def _own_counts(tiles: Sequence[TileRef]) -> tuple[tuple[int, ...], tuple[int, int, int]]:
    counts = [0] * 34
    aka = [0, 0, 0]
    for tile in tiles:
        counts[tile.kind] += 1
        if tile.aka is not None:
            aka[tile.aka] += 1
    return tuple(counts), tuple(aka)


def _players(scores: Sequence[int]) -> tuple[PlayerPublicState, ...]:
    winds = [tile_from_text(text) for text in ("E", "S", "W", "N")]
    return tuple(
        PlayerPublicState(seat_rel=index, score=int(scores[index]), wind=winds[index],
                          is_oya=(index == 0), riichi=RiichiState.NONE,
                          river_count=0, melds=())
        for index in range(4)
    )


def _tile_state(own: Sequence[TileRef], indicator: TileRef) -> VisibleTileState:
    own_counts, own_aka = _own_counts(own)
    public = [0] * 34
    public[indicator.kind] = 1
    zeros = (0,) * 34
    return VisibleTileState(
        own_counts=own_counts,
        own_aka=own_aka,
        public_counts=tuple(public),
        public_aka=(0, 0, 0),
        public_counts_by_zone=(zeros, zeros, tuple(public)),
        dora_indicators=(indicator,),
        ruleset_id=RulesetId.TENHOU_4P_RANKED,
    )


def _global_public(scores: Sequence[int], tiles_left: int) -> GlobalPublicState:
    return GlobalPublicState(
        bakaze=tile_from_text("E"), kyoku=1, honba=0, kyotaku=0,
        tiles_left=int(tiles_left), self_wind=tile_from_text("E"), oya_rel=0,
        ruleset_id=RulesetId.TENHOU_4P_RANKED, ruleset_deviation=(),
        source_kind=SourceKind.HUMAN_LOG,
    )


def _events(hand: Sequence[TileRef], indicator: TileRef, *, draw: TileRef | None,
            opponent_discard: TileRef | None = None):
    haipai = tuple(hand[:13])
    events: list = [
        StartGameEvent(seq_idx=0, kyoku_idx=-1, aka_flag=True, kyoku_first=0),
        StartKyokuEvent(seq_idx=1, kyoku_idx=0, bakaze=tile_from_text("E"),
                        dora_marker=indicator, kyoku=1, honba=0, kyotaku=0,
                        oya_rel=0, scores_rel=(25000, 25000, 25000, 25000),
                        own_haipai=haipai),
    ]
    if draw is not None:
        events.append(TsumoEvent(seq_idx=2, kyoku_idx=0, actor_rel=0, tile=draw))
    elif opponent_discard is not None:
        events.append(DahaiEvent(seq_idx=2, kyoku_idx=0, actor_rel=3,
                                 tile=opponent_discard, tsumogiri=True))
    else:  # pragma: no cover - callers always supply one
        raise ValueError("mock scene needs a draw or an opponent discard")
    return tuple(events)


def _mask_for(hand: Sequence[TileRef], phase: ActionPhase, *,
              extra: Iterable[int] = (), only: Iterable[int] | None = None
              ) -> tuple[bool, ...]:
    mask = [False] * ACTION_SPACE
    if only is not None:
        for action in only:
            mask[action] = True
    elif phase is ActionPhase.SELF_TURN:
        for tile in hand:
            mask[34 + tile.aka if tile.aka is not None else tile.kind] = True
        terminals = sum(1 for tile in hand
                        if tile.kind in (0, 8, 9, 17, 18, 26) or tile.kind >= 27)
        if terminals >= 9:
            mask[44] = True   # kyuushu kyuuhai (a real 9-terminal/honor hand)
        for action in extra:
            mask[action] = True
    elif phase is ActionPhase.KAN_SELECTION:
        for tile in hand:
            mask[34 + tile.aka if tile.aka is not None else tile.kind] = True
        for action in extra:
            mask[action] = True
    else:
        mask[45] = True   # pass
        for action in extra:
            mask[action] = True
    return tuple(mask)


def _decision(hand: Sequence[TileRef], phase: ActionPhase, *, draw: TileRef | None = None,
              opponent_discard: TileRef | None = None, extra: Iterable[int] = (),
              only: Iterable[int] | None = None, tiles_left: int = 69) -> DecisionInput:
    if draw is not None:
        own = list(hand[:13]) + [draw]
    else:
        own = list(hand[:13])
    counts = Counter(tile.kind for tile in own)
    indicator_kind = next(kind for kind in range(34) if counts[kind] == 0)
    indicator = _tile(indicator_kind)
    if draw is not None:
        events = _events(hand, indicator, draw=draw)
    else:
        events = _events(hand, indicator, draw=None,
                         opponent_discard=opponent_discard)
    mask = _mask_for(own, phase, extra=extra, only=only)
    return DecisionInput(
        events=events,
        tile_state=_tile_state(own, indicator),
        players_public=_players((25000, 25000, 25000, 25000)),
        global_public=_global_public((25000, 25000, 25000, 25000), tiles_left),
        legal_mask=mask,
        phase=phase,
        before_event_idx=len(events),
    )


def _mock_scores(hand: int, start_seat: int) -> list[int]:
    """Deterministic mock scores; sum is exactly 100000 (not a strength signal)."""
    scores = [25000] * 4
    for seat in range(4):
        scores[seat] = 25000 + ((hand * 37 + seat * 101 + start_seat * 13) % 5000) - 2500
    drift = 100_000 - sum(scores)
    scores[start_seat] += drift
    return scores


def _ranks(scores: Sequence[int]) -> list[int]:
    order = sorted(range(4), key=lambda seat: (-scores[seat], seat))
    ranks = [0] * 4
    for rank, seat in enumerate(order, start=1):
        ranks[seat] = rank
    return ranks


def _meta(seat: int) -> StreamMeta:
    return StreamMeta(
        pov=seat,
        projection_version=POV_PROJECTION_VERSION,
        schema_version=SCHEMA_VERSION,
        schema_hash=SCHEMA_HASH,
        ruleset_id="tenhou_4p_ranked",
        source_kind="human_log",
    )


def run_mock(*, hands: int, seed: int, key: int, out: str | Path | None = None,
             decisions_per_seat: int = 2, use_encoder: bool = True,
             max_events: int = 4096) -> dict:
    """Run the deterministic mock arena; return the summary dict.

    ``decisions_per_seat=2`` means one SELF_TURN discard and one CALL_RESPONSE
    pass/pon per seat per hand.  Every third hand also exercises KAN_SELECTION.
    """
    if isinstance(hands, bool) or not isinstance(hands, int) or hands <= 0:
        raise SystemExit("rf_eval: --hands must be a positive integer")
    if decisions_per_seat not in (1, 2, 3):
        raise SystemExit("rf_eval: --decisions-per-seat must be 1, 2 or 3")
    out_path = guard_out(out) if out is not None else None

    lines: list[str] = []
    phase_counts: Counter[str] = Counter()
    seat_starts: Counter[int] = Counter()
    seat_decisions: Counter[int] = Counter()
    rank_hist = [0, 0, 0, 0]
    illegal = none = truncated = ambiguous = visibility = failures = 0
    decisions_done = 0

    for hand in range(hands):
        start_seat = hand % 4
        seat_starts[start_seat] += 1
        scores = _mock_scores(hand, start_seat)
        ranks = _ranks(scores)

        for offset in range(4):
            seat = (start_seat + offset) % 4
            rng = random.Random((seed * 1_000_003) ^ (key * 97) ^ (hand * 4 + seat))
            hand_tiles = _hand_tiles(rng, 14)
            agent = RiichiFormerAgent(
                seat,
                MockPolicy(seed + hand * 31 + seat * 7 + key),
                encoder=encode_decision if use_encoder else None,
                stream_meta=_meta(seat) if use_encoder else None,
                max_events=max_events,
                name=f"mock-hand{hand}-seat{seat}",
            )
            agent.reset_game()
            agent.reset_kyoku()

            planned: list[tuple[ActionPhase, DecisionInput]] = []
            planned.append((ActionPhase.SELF_TURN, _decision(
                hand_tiles, ActionPhase.SELF_TURN, draw=hand_tiles[13])))
            if decisions_per_seat >= 2:
                discard = hand_tiles[rng.randrange(13)]
                planned.append((ActionPhase.CALL_RESPONSE, _decision(
                    hand_tiles, ActionPhase.CALL_RESPONSE, opponent_discard=discard)))
            if decisions_per_seat >= 3 and hand % 3 == 0:
                # 13-tile haipai with three copies + the drawn fourth -> ankan.
                kind = hand_tiles[13].kind
                others = [tile for tile in hand_tiles[:13] if tile.kind != kind][:10]
                used = {tile.kind for tile in others}
                fillers = [_tile(candidate) for candidate in range(34)
                           if candidate != kind and candidate not in used]
                others += fillers[: max(0, 10 - len(others))]
                haipai13 = others[:10] + [_tile(kind), _tile(kind), _tile(kind)]
                planned.append((ActionPhase.KAN_SELECTION, _decision(
                    haipai13, ActionPhase.KAN_SELECTION, draw=_tile(kind),
                    only=(kind,))))

            for phase, decision in planned:
                phase_counts[phase.value] += 1
                seat_decisions[seat] += 1
                try:
                    event = agent.act(decision)
                except AgentError as exc:
                    failures += 1
                    message = str(exc)
                    if "None" in message or "empty" in message.lower():
                        none += 1
                    elif "truncat" in message.lower() or "encoder" in message.lower():
                        truncated += 1
                    elif "visibility" in message.lower() or "concealed" in message.lower():
                        visibility += 1
                    elif "multi-kan" in message.lower() or "ambiguous" in message.lower():
                        ambiguous += 1
                    else:
                        illegal += 1
                    lines.append(json.dumps({
                        "hand": hand, "seat": seat, "phase": phase.value,
                        "error": f"{type(exc).__name__}: {message}"[:400],
                    }, sort_keys=True))
                    continue
                decisions_done += 1
                action = agent.records[-1].action
                if action is None or not decision.legal_mask[action]:
                    illegal += 1
                if not isinstance(event, dict) or not event.get("type"):
                    none += 1
                lines.append(json.dumps({
                    "hand": hand, "seat": seat, "phase": phase.value,
                    "action": action, "event": event,
                    "encoded_hash": agent.records[-1].encoded_hash,
                    "rank": ranks[seat],
                }, sort_keys=True))

        for rank in ranks:
            rank_hist[rank - 1] += 1

    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    summary = {
        "mode": "mock",
        "note": MOCK_NOTE,
        "hands": hands,
        "seed": seed,
        "key": key,
        "decisions": decisions_done,
        "decisions_planned": sum(phase_counts.values()),
        "illegal": illegal,
        "none": none,
        "truncated": truncated,
        "ambiguous": ambiguous,
        "visibility": visibility,
        "failures": failures,
        "phases": dict(phase_counts),
        "seat_starts": {str(seat): count for seat, count in sorted(seat_starts.items())},
        "seat_decisions": {str(seat): count for seat, count in sorted(seat_decisions.items())},
        "rank_histogram": {str(rank + 1): count for rank, count in enumerate(rank_hist)},
        "digest_sha256": digest,
        "tenhou_equivalent": False,
    }
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "decisions.jsonl").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
        (out_path / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PR-16 mock 4-seat arena (no model/data/frozen-pool evaluation)")
    parser.add_argument("--hands", type=int, required=True,
                        help="hand count (smoke: 10, 100, 1000)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--key", type=int, required=True)
    parser.add_argument("--out", required=True, help="explicit output directory")
    parser.add_argument("--decisions-per-seat", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--max-events", type=int, default=4096)
    parser.add_argument("--no-encoder", action="store_true",
                        help="skip the PR-11 encoder (debug only; default is on)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    summary = run_mock(hands=args.hands, seed=args.seed, key=args.key,
                       out=args.out, decisions_per_seat=args.decisions_per_seat,
                       use_encoder=not args.no_encoder, max_events=args.max_events)
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if (summary["illegal"] == 0 and summary["none"] == 0
                 and summary["truncated"] == 0 and summary["failures"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
