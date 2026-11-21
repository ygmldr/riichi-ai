#!/usr/bin/env python3
"""PR-10 decision snapshots: native 46-action opportunities + POV-censored reactions.

The native ``GameplayLoader`` remains the authority for labels/masks.  A small
``PlayerState`` replay reproduces its event-window loop so every opportunity gets an
exact ``before_event_idx``; windows where the native loader intentionally emits no label
(other player's higher-priority call/ron) become censored samples, never fake passes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Sequence

from ..schemas.action_v3 import ACTION_SPACE, ActionPhase
from ..schemas.decision_v3 import (
    DecisionInput,
    GlobalPublicState,
    MeldKind,
    PlayerPublicState,
    PublicMeld,
    RiichiState,
    RulesetDeviation,
    RulesetId,
    SourceKind,
)
from ..schemas.event_v3 import (
    AnkanEvent,
    ChiEvent,
    DaiminkanEvent,
    DahaiEvent,
    DoraEvent,
    EndGameEvent,
    EndKyokuEvent,
    HoraEvent,
    KakanEvent,
    PonEvent,
    ReachAcceptedEvent,
    ReachEvent,
    RyukyokuEvent,
    StartGameEvent,
    StartKyokuEvent,
    TileRef,
    TsumoEvent,
    tile_from_text,
)
from ..schemas.target_v3 import DecisionTarget, TargetNotInMask
from . import pov
from .replay import load_libriichi, replay_scores
from .tile_counter import TileCounter


class DecisionBuildError(RuntimeError):
    """Native/POV alignment failed; the game cannot produce trustworthy samples."""


@dataclass(frozen=True)
class DecisionBuildConfig:
    ruleset_id: RulesetId = RulesetId.TENHOU_4P_RANKED
    source_kind: SourceKind = SourceKind.HUMAN_LOG
    ruleset_deviation: tuple[RulesetDeviation, ...] = ()
    policy: pov.ProjectionPolicy | None = None
    oracle: bool = False

    def __post_init__(self):
        if not isinstance(self.ruleset_id, RulesetId):
            raise DecisionBuildError("ruleset_id must be RulesetId")
        if not isinstance(self.source_kind, SourceKind):
            raise DecisionBuildError("source_kind must be SourceKind")
        if not isinstance(self.ruleset_deviation, tuple):
            raise DecisionBuildError("ruleset_deviation must be a tuple")
        if len(set(self.ruleset_deviation)) != len(self.ruleset_deviation):
            raise DecisionBuildError("duplicate ruleset_deviation")
        for value in self.ruleset_deviation:
            if not isinstance(value, RulesetDeviation):
                raise DecisionBuildError("ruleset_deviation entries must be RulesetDeviation")
        if self.ruleset_id == RulesetId.TENHOU_4P_RANKED and self.ruleset_deviation:
            raise DecisionBuildError("Tenhou samples cannot carry arena deviations")
        if self.source_kind == SourceKind.HUMAN_LOG and (
                self.ruleset_id != RulesetId.TENHOU_4P_RANKED):
            raise DecisionBuildError("human_log must use tenhou_4p_ranked")
        if self.source_kind == SourceKind.ARENA_SELFPLAY and (
                self.ruleset_id != RulesetId.MORTAL_ARENA_PINNED):
            raise DecisionBuildError("arena_selfplay must use mortal_arena_pinned")
        if self.policy is not None:
            if self.policy.ruleset_id != self.ruleset_id:
                raise DecisionBuildError("policy/ruleset_id mismatch")
            if self.policy.source_kind != self.source_kind:
                raise DecisionBuildError("policy/source_kind mismatch")

    @property
    def projection_policy(self) -> pov.ProjectionPolicy:
        if self.policy is not None:
            return self.policy
        return pov.ProjectionPolicy(
            ruleset_id=self.ruleset_id, source_kind=self.source_kind)


@dataclass(frozen=True)
class DecisionSample:
    input: DecisionInput
    target: DecisionTarget
    header: pov.ProjectedStreamHeader
    decision_index: int

    def __post_init__(self):
        if not isinstance(self.input, DecisionInput):
            raise DecisionBuildError("input must be DecisionInput")
        if not isinstance(self.target, DecisionTarget):
            raise DecisionBuildError("target must be DecisionTarget")
        if not isinstance(self.header, pov.ProjectedStreamHeader):
            raise DecisionBuildError("header must be ProjectedStreamHeader")
        if self.input.before_event_idx > len(self.input.events):
            raise DecisionBuildError("before_event_idx exceeds the visible prefix")

    def as_record(self) -> dict:
        """Shard/sample record: header attribution + forward input + side target."""
        return {
            "header": self.header.as_dict(),
            "decision_index": self.decision_index,
            "input": self.input.to_forward_dict(),
            "target": self.target.to_dict(),
        }


@dataclass
class _Opportunity:
    label: int | None
    kan_select: int | None
    before_event_idx: int
    phase: ActionPhase
    mask: tuple[bool, ...]
    at_kyoku: int
    at_turn: int
    censor_reason: str | None = None


def _tile_action(tile: TileRef) -> int:
    if tile.is_aka:
        return 34 + tile.aka
    return tile.kind


def _chi_action(consumed: Sequence[str], pai: str) -> int:
    called = tile_from_text(pai).kind
    a, b = (tile_from_text(text).kind for text in consumed)
    lo, hi = min(a, b), max(a, b)
    if called < lo:
        return 38
    if called < hi:
        return 39
    return 40


def _can_act(cans) -> bool:
    value = cans.can_act
    return bool(value() if callable(value) else value)


def _at_turn(state) -> int:
    value = state.at_turn
    return int(value() if callable(value) else value)


def _mask(state, at_kan_select: bool) -> tuple[bool, ...]:
    _feature, mask = state.encode_obs(4, at_kan_select)
    values = tuple(bool(value) for value in mask)
    if len(values) != ACTION_SPACE:
        raise DecisionBuildError(f"native mask width {len(values)} != {ACTION_SPACE}")
    return values


def _phase(cans, label: int) -> ActionPhase:
    if label == 44:  # kyuushu kyuuhai is a self-turn decision before a draw
        return ActionPhase.SELF_TURN
    if bool(cans.can_discard):
        return ActionPhase.SELF_TURN
    return ActionPhase.CALL_RESPONSE


def _other_call(next_event: dict, pov_index: int) -> bool:
    return (next_event.get("type") in ("chi", "pon", "daiminkan")
            and next_event.get("actor") != pov_index)


def _censor_reason(next_event: dict, pov_index: int) -> str:
    kind = next_event.get("type")
    if kind == "hora" and next_event.get("actor") != pov_index:
        return "preempted_by_ron"
    if kind in ("chi", "pon", "daiminkan"):
        return "preempted_by_call"
    return "unlabeled_opportunity"


def _label_for_next(next_event: dict, wnd: Sequence[dict], cans,
                    pov_index: int) -> tuple[int | None, int | None]:
    """Exact port of libriichi ``extend_from_event_window`` label selection."""
    kind = next_event.get("type")
    if kind == "dahai":
        return _tile_action(tile_from_text(next_event["pai"])), None
    if kind == "reach":
        return 37, None
    if kind == "chi" and next_event.get("actor") == pov_index:
        return _chi_action(next_event["consumed"], next_event["pai"]), None
    if kind == "pon" and next_event.get("actor") == pov_index:
        return 41, None
    if kind == "daiminkan" and next_event.get("actor") == pov_index:
        return 42, tile_from_text(next_event["pai"]).kind
    if kind == "kakan":
        return 42, tile_from_text(next_event["pai"]).kind
    if kind == "ankan":
        return 42, tile_from_text(next_event["consumed"][0]).kind
    if kind == "ryukyoku" and bool(cans.can_ryukyoku):
        return 44, None

    has_any_ron = wnd[1].get("type") == "hora"
    if has_any_ron:
        for event in wnd[1:]:
            if event.get("type") == "end_kyoku":
                break
            if event.get("type") == "hora" and event.get("actor") == pov_index:
                return 43, None
    can_chi = (bool(cans.can_chi_low) or bool(cans.can_chi_mid)
               or bool(cans.can_chi_high))
    if can_chi and kind == "tsumo":
        return 45, None
    if (bool(cans.can_pon) or bool(cans.can_daiminkan)
            or bool(cans.can_ron_agari)) and not has_any_ron:
        return 45, None
    return None, None


def _native_opportunities(raw: Sequence[dict], pov_index: int,
                          libriichi) -> list[_Opportunity]:
    state = libriichi.state.PlayerState(pov_index)
    kyoku_idx = 0
    out: list[_Opportunity] = []
    for index in range(len(raw) - 3):
        wnd = raw[index:index + 4]
        cur, first, second, _third = wnd
        next_event = second if first.get("type") in ("reach_accepted", "dora") else first
        before_event_idx = index + (2 if next_event is second else 1)
        if cur.get("type") == "end_kyoku":
            kyoku_idx += 1
        cans = state.update(json.dumps(cur, ensure_ascii=False))
        if not _can_act(cans):
            continue
        label, kan_select = _label_for_next(next_event, wnd, cans, pov_index)
        if label is None:
            mask = _mask(state, False)
            phase = (ActionPhase.SELF_TURN if bool(cans.can_discard)
                     else ActionPhase.CALL_RESPONSE)
            out.append(_Opportunity(
                label=None, kan_select=None, before_event_idx=before_event_idx,
                phase=phase, mask=mask, at_kyoku=kyoku_idx,
                at_turn=_at_turn(state),
                censor_reason=_censor_reason(next_event, pov_index),
            ))
            continue
        phase = _phase(cans, label)
        censor_reason = None
        if label == 45 and _other_call(next_event, pov_index):
            # The native loader can emit a 45 here, but another seat's chi/pon/
            # daiminkan already took the discard: this was not a free pass.
            censor_reason = "preempted_by_call"
        out.append(_Opportunity(
            label=label, kan_select=kan_select, before_event_idx=before_event_idx,
            phase=phase, mask=_mask(state, False), at_kyoku=kyoku_idx,
            at_turn=_at_turn(state), censor_reason=censor_reason,
        ))
        if kan_select is not None:
            out.append(_Opportunity(
                label=kan_select, kan_select=None, before_event_idx=before_event_idx,
                phase=ActionPhase.KAN_SELECTION, mask=_mask(state, True),
                at_kyoku=kyoku_idx, at_turn=_at_turn(state),
            ))
    return out


def _check_native(opportunities: list[_Opportunity], gameplay) -> None:
    supervised = [item for item in opportunities if item.label is not None]
    actions = [int(value) for value in gameplay.take_actions()]
    masks = list(gameplay.take_masks())
    kyokus = [int(value) for value in gameplay.take_at_kyoku()]
    turns = [int(value) for value in gameplay.take_at_turns()]
    if len(supervised) != len(actions):
        raise DecisionBuildError(
            f"native alignment: generated {len(supervised)} supervised decisions, "
            f"Gameplay has {len(actions)}")
    for index, item in enumerate(supervised):
        if item.label != actions[index]:
            raise DecisionBuildError(
                f"native action mismatch at {index}: {item.label} != {actions[index]}")
        if item.at_kyoku != kyokus[index]:
            raise DecisionBuildError(
                f"native kyoku mismatch at {index}: {item.at_kyoku} != {kyokus[index]}")
        if item.at_turn != turns[index]:
            raise DecisionBuildError(
                f"native turn mismatch at {index}: {item.at_turn} != {turns[index]}")
        native_mask = tuple(bool(value) for value in masks[index])
        if len(native_mask) != ACTION_SPACE or native_mask != item.mask:
            raise DecisionBuildError(f"native mask mismatch at decision {index}")
        item.mask = native_mask


def _final_ranks(raw: Sequence[dict]) -> tuple[int, int, int, int] | None:
    result = replay_scores(raw, strict_protocol=False)
    if not result.ok or result.scores is None:
        return None
    scores = result.scores
    order = sorted(range(4), key=lambda seat: (-scores[seat], seat))
    ranks = [0, 0, 0, 0]
    for rank, seat in enumerate(order, start=1):
        ranks[seat] = rank
    return tuple(ranks)  # type: ignore[return-value]


class _SnapshotTracker:
    """Incremental relative-seat table state for one POV."""

    def __init__(self, config: DecisionBuildConfig):
        self.config = config
        self.started = False
        self.ledger = TileCounter(config.ruleset_id)
        self.scores = [0, 0, 0, 0]
        self.kyotaku = 0
        self.honba = 0
        self.kyoku = 1
        self.bakaze = TileRef(27)
        self.oya_rel = 0
        self.rivers: list[list[TileRef]] = [[], [], [], []]
        self.melds: list[list[PublicMeld]] = [[], [], [], []]
        self.riichi = [RiichiState.NONE] * 4
        self.dora_indicators: list[TileRef] = []
        self.tsumo_count = 0

    def _take_river(self, target: int, tile: TileRef) -> None:
        river = self.rivers[target]
        for index in range(len(river) - 1, -1, -1):
            if river[index] == tile:
                del river[index]
                return
        raise DecisionBuildError(
            f"called tile {tile.text} not found in target river {target}")

    def _add_meld(self, seat: int, meld: PublicMeld) -> None:
        self.melds[seat].append(meld)

    def apply(self, event) -> None:
        if isinstance(event, StartKyokuEvent):
            self.started = True
            self.ledger = TileCounter(self.config.ruleset_id)
            self.scores = list(event.scores_rel)
            self.kyotaku = event.kyotaku
            self.honba = event.honba
            self.kyoku = event.kyoku
            self.bakaze = event.bakaze
            self.oya_rel = event.oya_rel
            self.rivers = [[], [], [], []]
            self.melds = [[], [], [], []]
            self.riichi = [RiichiState.NONE] * 4
            self.dora_indicators = [event.dora_marker]
            self.tsumo_count = 0
            for tile in event.own_haipai:
                self.ledger.add_own(tile)
            self.ledger.add_indicator(event.dora_marker)
            return
        if not self.started:
            return
        if isinstance(event, TsumoEvent):
            self.tsumo_count += 1
            if event.actor_rel == 0:
                self.ledger.add_own(event.tile)
        elif isinstance(event, DahaiEvent):
            if event.actor_rel == 0:
                self.ledger.remove_own(event.tile)
            self.rivers[event.actor_rel].append(event.tile)
            self.ledger.add_river(event.tile)
        elif isinstance(event, (ChiEvent, PonEvent, DaiminkanEvent)):
            target = event.target_rel
            self._take_river(target, event.tile)
            self.ledger.call_tile(event.tile)
            known_count = 3 if isinstance(event, DaiminkanEvent) else 2
            known_consumed = tuple(event.consumed[:known_count])
            for tile in known_consumed:
                if event.actor_rel == 0:
                    self.ledger.remove_own(tile)
                self.ledger.add_meld(tile)
            kind = (MeldKind.CHI if isinstance(event, ChiEvent)
                    else MeldKind.PON if isinstance(event, PonEvent)
                    else MeldKind.DAIMINKAN)
            self._add_meld(event.actor_rel, PublicMeld(
                meld_kind=kind, tiles=(event.tile,) + known_consumed,
                from_rel=target))
        elif isinstance(event, KakanEvent):
            for index, meld in enumerate(self.melds[event.actor_rel]):
                if meld.meld_kind == MeldKind.PON and all(
                        tile.kind == event.tile.kind for tile in meld.tiles):
                    if event.actor_rel == 0:
                        self.ledger.remove_own(event.tile)
                    self.ledger.add_meld(event.tile)
                    self.melds[event.actor_rel][index] = replace(
                        meld, meld_kind=MeldKind.KAKAN,
                        tiles=meld.tiles + (event.tile,))
                    return
            raise DecisionBuildError("kakan without a matching pon")
        elif isinstance(event, AnkanEvent):
            known = all(not tile.is_unknown for tile in event.consumed)
            if event.actor_rel == 0:
                for tile in event.consumed:
                    self.ledger.remove_own(tile)
            if known:
                for tile in event.consumed:
                    self.ledger.add_meld(tile)
            self._add_meld(event.actor_rel, PublicMeld(
                meld_kind=MeldKind.ANKAN, tiles=tuple(event.consumed), from_rel=4))
        elif isinstance(event, ReachEvent):
            self.riichi[event.actor_rel] = RiichiState.DECLARED
        elif isinstance(event, ReachAcceptedEvent):
            self.riichi[event.actor_rel] = RiichiState.ACCEPTED
            self.scores[event.actor_rel] -= 1000
            self.kyotaku += 1
        elif isinstance(event, DoraEvent):
            self.dora_indicators.append(event.dora_marker)
            self.ledger.add_indicator(event.dora_marker)
        elif isinstance(event, HoraEvent):
            for seat, delta in enumerate(event.deltas):
                self.scores[seat] += delta
            self.kyotaku = 0
        elif isinstance(event, RyukyokuEvent):
            for seat, delta in enumerate(event.deltas):
                self.scores[seat] += delta
        elif isinstance(event, (EndKyokuEvent, EndGameEvent, StartGameEvent)):
            return
        else:  # pragma: no cover - event_v3 allowlist is closed
            raise DecisionBuildError(f"unhandled event {type(event).__name__}")

    def _wind(self, seat: int) -> TileRef:
        return TileRef(27 + ((self.bakaze.kind - 27 + seat - self.oya_rel) % 4))

    def build_input(self, prefix: Sequence, opportunity: _Opportunity,
                    header: pov.ProjectedStreamHeader) -> DecisionInput:
        if not self.started:
            raise DecisionBuildError("decision before start_kyoku")
        tile_state = self.ledger.to_visible_tile_state(
            tuple(self.dora_indicators), ruleset_id=self.config.ruleset_id)
        players = tuple(
            PlayerPublicState(
                seat_rel=seat,
                score=self.scores[seat],
                wind=self._wind(seat),
                is_oya=(seat == self.oya_rel),
                riichi=self.riichi[seat],
                river_count=len(self.rivers[seat]),
                melds=tuple(self.melds[seat]),
            )
            for seat in range(4)
        )
        # Native wall: 70 initial live tiles; every tsumo (including rinshan)
        # decrements exactly once.  Kans do NOT add a wall tile here.
        tiles_left = max(0, min(70, 70 - self.tsumo_count))
        global_public = GlobalPublicState(
            bakaze=self.bakaze,
            kyoku=self.kyoku,
            honba=self.honba,
            kyotaku=self.kyotaku,
            tiles_left=tiles_left,
            self_wind=players[0].wind,
            oya_rel=self.oya_rel,
            ruleset_id=self.config.ruleset_id,
            ruleset_deviation=self.config.ruleset_deviation,
            source_kind=self.config.source_kind,
        )
        return DecisionInput(
            events=tuple(prefix),
            tile_state=tile_state,
            players_public=players,
            global_public=global_public,
            legal_mask=tuple(opportunity.mask),
            phase=opportunity.phase,
            before_event_idx=opportunity.before_event_idx,
        )


def _make_target(opportunity: _Opportunity,
                 rank_final: int | None) -> DecisionTarget:
    if opportunity.label is None:
        return DecisionTarget(
            expert_action=None, supervision_valid=False,
            censor_reason=opportunity.censor_reason, rank_final=rank_final)
    label = int(opportunity.label)
    if opportunity.censor_reason is not None:
        return DecisionTarget(
            expert_action=label, supervision_valid=False,
            censor_reason=opportunity.censor_reason, rank_final=rank_final)
    valid = 0 <= label < ACTION_SPACE and bool(opportunity.mask[label])
    if not valid:
        return DecisionTarget(
            expert_action=label, supervision_valid=False,
            censor_reason="action_not_in_mask", rank_final=rank_final)
    return DecisionTarget(
        expert_action=label, supervision_valid=True,
        censor_reason=None, rank_final=rank_final)


def build_decisions(raw_events: Sequence[dict],
                    config: DecisionBuildConfig | None = None) -> list[DecisionSample]:
    """Build every native action opportunity for all four POVs of one game."""
    config = config or DecisionBuildConfig()
    raw = list(raw_events)
    if not raw or not all(isinstance(event, dict) for event in raw):
        raise DecisionBuildError("raw_events must be a non-empty sequence of dicts")
    raw_text = "\n".join(json.dumps(event, ensure_ascii=False) for event in raw)
    libriichi = load_libriichi()
    try:
        loader = libriichi.dataset.GameplayLoader(
            4, oracle=config.oracle, always_include_kan_select=True)
        gameplays = loader.load_log(raw_text)
    except Exception as exc:  # noqa: BLE001
        raise DecisionBuildError(f"libriichi load failed: {type(exc).__name__}: {exc}")
    if len(gameplays) != 4:
        raise DecisionBuildError(f"expected 4 native seats, got {len(gameplays)}")
    ranks = _final_ranks(raw)
    samples: list[DecisionSample] = []
    for seat in range(4):
        projector = pov.POVProjector(seat, policy=config.projection_policy)
        try:
            projected = projector.extend(raw)
        except Exception as exc:  # noqa: BLE001
            raise DecisionBuildError(f"POV projection failed for seat {seat}: {exc}")
        opportunities = _native_opportunities(raw, seat, libriichi)
        _check_native(opportunities, gameplays[seat])
        tracker = _SnapshotTracker(config)
        applied = 0
        for decision_index, opportunity in enumerate(opportunities):
            if opportunity.before_event_idx > len(projected):
                raise DecisionBuildError("decision beyond projected stream")
            while applied < opportunity.before_event_idx:
                tracker.apply(projected[applied])
                applied += 1
            prefix = projected[:opportunity.before_event_idx]
            if not prefix:
                raise DecisionBuildError("decision has empty visible prefix")
            try:
                decision_input = tracker.build_input(prefix, opportunity,
                                                    projector.header)
            except Exception as exc:  # noqa: BLE001
                raise DecisionBuildError(
                    f"snapshot failed seat {seat} before {opportunity.before_event_idx}: "
                    f"{type(exc).__name__}: {exc}")
            target = _make_target(
                opportunity,
                ranks[seat] if ranks is not None else None)
            if target.supervision_valid:
                try:
                    target.validate_against_input(decision_input)
                except TargetNotInMask as exc:
                    raise DecisionBuildError(
                        f"illegal target seat {seat} before "
                        f"{opportunity.before_event_idx}: {exc}")
            samples.append(DecisionSample(
                input=decision_input, target=target,
                header=projector.header, decision_index=decision_index))
    return samples


__all__ = [
    "DecisionBuildConfig", "DecisionBuildError", "DecisionSample", "build_decisions",
]
