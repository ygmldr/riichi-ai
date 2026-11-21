#!/usr/bin/env python3
"""PR-41 four-seat hanchan rollout with per-game provenance.

Two backends share one bot/view contract:

``libriichi``
    A real local rules engine (``libriichi.arena.OneVsThree.py_vs_py`` from the pinned
    Mortal submodule).  Four seats are driven by four *distinct* Python bot instances:
    the arena runs four rotated games per wall seed, the challenger slot walks through
    seats 0..3 and the champion slot fills the other three seats.  Each seat of each
    rotated game gets its own bot instance, so rotation ``k`` puts lineup position
    ``p`` at absolute seat ``(k + p) % 4``.

    The bridge derives the frozen 46-action mask from the native ``ActionCandidate``
    (exactly the indices the pinned engine uses), translates the chosen index with
    :func:`riichiformer.runtime.mjai_adapter.action_to_mjai`, and re-validates every
    reaction with ``PlayerState.validate_reaction`` before returning it.  Illegal
    actions, rejected reactions and fallbacks are counted; a non-zero count fails the
    run.

``procedural``
    An explicit fallback for environments without libriichi: a seeded draw/discard
    hanchan with tenpai/noten payments and riichi sticks, no calls and no wins.  It is
    labelled ``synthetic_abstract_hanchan/1`` everywhere and is never called Tenhou.

Neither backend touches the network, a dataset, a checkpoint or an online platform.
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ..runtime.mjai_adapter import ActionContext, AdapterError, KanOption, action_to_mjai
from ..schemas.action_v3 import ACTION_SPACE, ActionPhase
from ..schemas.event_v3 import TileRef, tile_from_text
from .bots import BOT_FACTORIES, Bot, BotAction, IllegalBotAction, make_bot
from .provenance import (
    RULESET_LIBRIICHI,
    RULESET_SYNTHETIC,
    SELFPLAY_VERSION,
    assert_known_ruleset,
    build_provenance,
    generator_fingerprint,
    ruleset_metadata,
    sha256_file,
    source_metadata,
    write_json,
    write_jsonl,
)
from .seeds import ROTATIONS, SeedLedger, seed_range, validate_games
from .shanten import is_winning_hand, shanten
from .tiles import AKA_KINDS, TILE_KIND_COUNT, physical_hand, tile_text
from .view import SOURCE_LIBRIICHI, SOURCE_SYNTHETIC, SeatView, ViewError

__all__ = [
    "RolloutError",
    "RolloutConfig",
    "SeatPovRecord",
    "GameRecord",
    "libriichi_status",
    "resolve_engine",
    "run_selfplay",
    "run_seed",
]

ENGINE_LIBRIICHI = "libriichi.arena.OneVsThree.py_vs_py"
ENGINE_SYNTHETIC = "none (procedural fallback)"
ENGINE_FOR_RULESET = {
    RULESET_LIBRIICHI: ENGINE_LIBRIICHI,
    RULESET_SYNTHETIC: ENGINE_SYNTHETIC,
}

DEFAULT_KEY = 20261121
SELFPLAY_SOURCE = source_metadata()


class RolloutError(RuntimeError):
    """A rollout precondition/consistency failure (fail closed)."""


# ---------------------------------------------------------------------------
# libriichi discovery
# ---------------------------------------------------------------------------
def _vendor_release_dirs() -> list[Path]:
    repo = Path(__file__).resolve().parents[2]
    candidates = [repo / "vendor" / "mortal" / "target" / "release"]
    env = __import__("os").environ.get("RIICHI_LIBRIICHI_DIR")
    if env:
        candidates.insert(0, Path(env))
    return candidates


def libriichi_status() -> dict:
    """Import status of the pinned local rules engine (no network, no download)."""
    try:
        import libriichi  # noqa: F401

        return {"available": True, "path": getattr(libriichi, "__file__", None),
                "error": None}
    except Exception as first_exc:
        for candidate in _vendor_release_dirs():
            if not candidate.is_dir() or str(candidate) in sys.path:
                continue
            sys.path.insert(0, str(candidate))
            try:
                import libriichi  # noqa: F401

                return {"available": True, "path": getattr(libriichi, "__file__", None),
                        "error": None}
            except Exception:
                continue
        return {
            "available": False,
            "path": None,
            "error": f"{type(first_exc).__name__}: {first_exc}"[:400],
            "hint": "export PYTHONPATH=<repo>/vendor/mortal/target/release",
        }


def resolve_engine(ruleset_id: str) -> str:
    assert_known_ruleset(ruleset_id)
    if ruleset_id == RULESET_LIBRIICHI:
        status = libriichi_status()
        if not status["available"]:
            raise RolloutError(
                "NOT_RUN: ruleset libriichi_tenhou_approx requires the pinned local "
                f"libriichi ({status.get('error')}); no game was played. Use "
                "--ruleset synthetic_abstract_hanchan for the labelled fallback.")
    return ENGINE_FOR_RULESET[ruleset_id]


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass
class SeatPovRecord:
    position: int
    seat: int
    bot: str
    decisions: int = 0
    actions: Counter = field(default_factory=Counter)
    illegal: int = 0
    rejected: int = 0
    fallbacks: int = 0

    def as_dict(self) -> dict:
        return {
            "position": self.position,
            "seat": self.seat,
            "bot": self.bot,
            "decisions": self.decisions,
            "actions": {str(key): value for key, value in sorted(self.actions.items())},
            "illegal": self.illegal,
            "rejected": self.rejected,
            "fallbacks": self.fallbacks,
            "pov_policy": "own hand + public events only",
        }


@dataclass
class GameRecord:
    game_id: str
    seed: int
    key: int
    rotation: int
    ruleset_id: str
    rules_engine: str
    generator_sha256: str
    version: str
    lineup: tuple[str, ...]
    seats: tuple[int, int, int, int]
    scores: tuple[int, int, int, int]
    ranks: tuple[int, int, int, int]
    povs: tuple[SeatPovRecord, ...]
    decisions: int
    illegal: int
    rejected: int
    fallbacks: int
    deviation_counts: dict[str, int] = field(default_factory=dict)
    game_log_path: str | None = None
    game_log_sha256: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def score_sum(self) -> int:
        return sum(self.scores)

    def as_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "key": self.key,
            "rotation": self.rotation,
            "ruleset_id": self.ruleset_id,
            "rules_engine": self.rules_engine,
            "tenhou_equivalent": False,
            "generator_sha256": self.generator_sha256,
            "version": self.version,
            "lineup": list(self.lineup),
            "seats": list(self.seats),
            "scores": list(self.scores),
            "ranks": list(self.ranks),
            "score_sum": self.score_sum,
            "score_conserved": self.score_sum == 100_000,
            "decisions": self.decisions,
            "illegal": self.illegal,
            "rejected": self.rejected,
            "fallbacks": self.fallbacks,
            "povs": [pov.as_dict() for pov in self.povs],
            "deviation_counts": dict(sorted(self.deviation_counts.items())),
            "game_log": {
                "path": self.game_log_path,
                "sha256": self.game_log_sha256,
            } if self.game_log_path else None,
            "source": {
                "source_kind": "selfplay_own",
                "external_license": "none",
                "third_party_logs": False,
                "third_party_weights": False,
                "network": False,
            },
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# 46-action mask from the native candidate set
# ---------------------------------------------------------------------------
def _mask_from_candidate(cans, tehai: Sequence[int], akas: Sequence[int]) -> list[bool]:
    mask = [False] * ACTION_SPACE
    if cans.can_discard:
        for kind in range(TILE_KIND_COUNT):
            normal = tehai[kind]
            if kind in AKA_KINDS:
                normal -= akas[AKA_KINDS.index(kind)]
            if normal > 0:
                mask[kind] = True
        for color in range(3):
            if akas[color]:
                mask[34 + color] = True
    if cans.can_riichi:
        mask[37] = True
    if cans.can_chi_low:
        mask[38] = True
    if cans.can_chi_mid:
        mask[39] = True
    if cans.can_chi_high:
        mask[40] = True
    if cans.can_pon:
        mask[41] = True
    if cans.can_daiminkan or cans.can_ankan or cans.can_kakan:
        mask[42] = True
    if cans.can_tsumo_agari or cans.can_ron_agari:
        mask[43] = True
    if cans.can_ryukyoku:
        mask[44] = True
    if cans.can_pass:
        mask[45] = True
    if not any(mask):
        raise RolloutError("native candidate set produced an empty 46-action mask")
    return mask


# ---------------------------------------------------------------------------
# public event parsing (whitelist; never reads concealed fields)
# ---------------------------------------------------------------------------
_PUBLIC_EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "start_kyoku": ("scores", "kyoku", "honba", "kyotaku", "oya", "dora_marker"),
    "dahai": ("actor", "pai"),
    "chi": ("actor", "target", "pai", "consumed"),
    "pon": ("actor", "target", "pai", "consumed"),
    "daiminkan": ("actor", "target", "pai", "consumed"),
    "ankan": ("actor", "consumed"),
    "kakan": ("actor", "pai", "consumed"),
    "reach": ("actor",),
    "reach_accepted": ("actor",),
    "dora": ("dora_marker",),
    "tsumo": ("actor",),
    "hora": ("actor", "target"),
    "ryukyoku": (),
    "end_kyoku": (),
}


def _public_events(events_json: str) -> list[dict]:
    """Parse only whitelisted public fields; concealed hands/ura dora are dropped."""
    raw = json.loads(events_json)
    if not isinstance(raw, list):
        raise RolloutError("events_json is not a JSON list")
    public: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        kind = entry.get("type")
        fields = _PUBLIC_EVENT_FIELDS.get(kind)
        if fields is None:
            continue
        public.append({key: entry[key] for key in ("type",) + fields if key in entry})
    return public


@dataclass
class _PublicLog:
    rivers: list[list[TileRef]] = field(default_factory=lambda: [[] for _ in range(4)])
    melds: list[list[tuple[TileRef, ...]]] = field(
        default_factory=lambda: [[] for _ in range(4)])
    riichi: list[bool] = field(default_factory=lambda: [False] * 4)
    reach_accepted: list[bool] = field(default_factory=lambda: [False] * 4)
    dora: list[TileRef] = field(default_factory=list)
    visible_extra: list[TileRef] = field(default_factory=list)
    start_scores: tuple[int, int, int, int] = (25000, 25000, 25000, 25000)
    kyoku: int = 1
    honba: int = 0
    kyotaku: int = 0
    oya: int = 0
    draws: int = 0

    def visible_counts(self, hand_counts: Sequence[int]) -> list[int]:
        counts = list(hand_counts)
        for river in self.rivers:
            for tile in river:
                counts[tile.kind] += 1
        for melds in self.melds:
            for meld in melds:
                for tile in meld:
                    counts[tile.kind] += 1
        for tile in self.dora:
            counts[tile.kind] += 1
        for tile in self.visible_extra:
            counts[tile.kind] += 1
        return counts


def _parse_public_log(events_json: str) -> _PublicLog:
    log = _PublicLog()
    for event in _public_events(events_json):
        kind = event["type"]
        if kind == "start_kyoku":
            log.start_scores = tuple(int(value) for value in event["scores"])
            log.kyoku = int(event.get("kyoku", 1))
            log.honba = int(event.get("honba", 0))
            log.kyotaku = int(event.get("kyotaku", 0))
            log.oya = int(event.get("oya", 0))
            log.dora.append(tile_from_text(event["dora_marker"]))
        elif kind == "dahai":
            log.rivers[int(event["actor"])].append(tile_from_text(event["pai"]))
        elif kind in ("chi", "pon", "daiminkan"):
            actor = int(event["actor"])
            consumed = tuple(tile_from_text(text) for text in event["consumed"])
            # the called tile stays in the river, exactly like the pinned engine's kawa
            log.melds[actor].append(consumed)
        elif kind == "ankan":
            actor = int(event["actor"])
            log.melds[actor].append(
                tuple(tile_from_text(text) for text in event["consumed"]))
        elif kind == "kakan":
            # a kakan upgrades an existing pon: it adds one visible tile but no new meld
            log.visible_extra.append(tile_from_text(event["pai"]))
        elif kind == "reach":
            log.riichi[int(event["actor"])] = True
        elif kind == "reach_accepted":
            log.reach_accepted[int(event["actor"])] = True
        elif kind == "dora":
            log.dora.append(tile_from_text(event["dora_marker"]))
        elif kind == "tsumo":
            log.draws += 1
    return log


def _tile_ref(text: str | None) -> TileRef | None:
    if text is None:
        return None
    return tile_from_text(text)


# ---------------------------------------------------------------------------
# libriichi bridge
# ---------------------------------------------------------------------------
class _SeatContext:
    __slots__ = ("role", "index", "split", "seat", "bot", "pov")

    def __init__(self, role: str, index: int, split: int, seat: int, bot: Bot):
        self.role = role
        self.index = index
        self.split = split
        self.seat = seat
        self.bot = bot
        self.pov = SeatPovRecord(position=(seat - split) % 4, seat=seat, bot=bot.name)


class _Recorder:
    def __init__(self, *, seed: int, key: int, lineup: Sequence[str],
                 seat_ids: dict[str, Sequence[int]], verify_bots: bool = False):
        self.seed = seed
        self.key = key
        self.lineup = tuple(lineup)
        self.seat_ids = {role: tuple(ids) for role, ids in seat_ids.items()}
        self.verify_bots = bool(verify_bots)
        self.contexts: dict[tuple[str, int], _SeatContext] = {}
        self.split_scores: dict[int, tuple[int, int, int, int]] = {}
        self.split_kyoku: Counter = Counter()
        self.illegal = 0
        self.rejected = 0
        self.fallbacks = 0
        self.mask_checks = 0
        self.mask_restricted = 0
        self.mask_undercount = 0
        self.agari_guard = 0
        self.agari_checks = 0
        self.agari_mismatches = 0
        self.shanten_checks = 0
        self.shanten_mismatches = 0
        self.bot_errors = 0
        self.views = 0

    # -- setup -------------------------------------------------------------
    def create_context(self, role: str, index: int) -> _SeatContext:
        ids = self.seat_ids[role]
        if not 0 <= index < len(ids):
            raise RolloutError(f"{role} index {index} outside seat map {ids}")
        seat = ids[index]
        split = index // 3 if role == "champion" else index
        position = (seat - split) % 4
        bot = make_bot(self.lineup[position],
                       seed=_bot_seed(self.seed, split, position, role))
        context = _SeatContext(role, index, split, seat, bot)
        self.contexts[(role, index)] = context
        return context

    def context(self, role: str, index: int) -> _SeatContext:
        try:
            return self.contexts[(role, index)]
        except KeyError:
            raise RolloutError(f"no seat context for {role}[{index}]") from None

    def game_contexts(self, split: int) -> list[_SeatContext]:
        result = [ctx for ctx in self.contexts.values() if ctx.split == split]
        if len(result) != 4:
            raise RolloutError(f"split {split} has {len(result)} seat contexts, expected 4")
        return sorted(result, key=lambda ctx: ctx.seat)


def _bot_seed(seed: int, split: int, position: int, role: str) -> int:
    role_index = 0 if role == "challenger" else 1
    return (seed * 1_000_003 + split * 10_007 + position * 101 + role_index) & 0x7FFFFFFF


class _SeatEngine:
    """A ``mjai-log`` engine object the pinned arena can drive from Python."""

    engine_type = "mjai-log"
    version = 1
    is_oracle = False
    enable_quick_eval = False
    enable_rule_based_agari_guard = False

    def __init__(self, role: str, recorder: _Recorder):
        self.role = role
        self.recorder = recorder
        self.name = f"rf-selfplay-{role}"
        self.player_ids: tuple[int, ...] | None = None

    # -- lifecycle ---------------------------------------------------------
    def set_player_ids(self, ids):
        ids = tuple(int(value) for value in ids)
        expected = self.recorder.seat_ids[self.role]
        if ids != expected:
            raise RolloutError(f"{self.role} seat map {ids} != expected {expected}")
        self.player_ids = ids

    def start_game(self, index: int) -> None:
        self.recorder.create_context(self.role, int(index))

    def end_kyoku(self, index: int) -> None:
        context = self.recorder.context(self.role, int(index))
        self.recorder.split_kyoku[context.split] += 1

    def end_game(self, index: int, scores) -> None:
        context = self.recorder.context(self.role, int(index))
        scores = tuple(int(value) for value in scores)
        if len(scores) != 4:
            raise RolloutError(f"engine returned {len(scores)} scores")
        if context.role == "challenger":
            previous = self.recorder.split_scores.get(context.split)
            if previous is not None and previous != scores:
                raise RolloutError(
                    f"split {context.split} returned two different score lines")
            self.recorder.split_scores[context.split] = scores
        else:
            expected = self.recorder.split_scores.get(context.split)
            if expected is not None and expected != scores:
                raise RolloutError("challenger/champion score lines disagree")

    # -- reactions ---------------------------------------------------------
    def react_batch(self, states):
        return [self._react(game_state) for game_state in states]

    def _react(self, game_state) -> str:
        state = game_state.state
        context = self.recorder.context(self.role, int(game_state.game_index))
        if int(state.player_id) != context.seat:
            raise RolloutError(
                f"seat mismatch: engine says {state.player_id}, map says {context.seat}")
        view = _view_from_state(state, game_state.events_json, self.recorder)
        self.recorder.views += 1
        action: BotAction
        try:
            action = context.bot.act(view)
        except IllegalBotAction:
            self.recorder.bot_errors += 1
            context.pov.illegal += 1
            action = _fallback_action(view)
            context.pov.fallbacks += 1
        except Exception:  # pragma: no cover - bug guard
            self.recorder.bot_errors += 1
            context.pov.illegal += 1
            action = _fallback_action(view)
            context.pov.fallbacks += 1
        try:
            text, selected = _encode_action(view, action)
        except (RolloutError, AdapterError):
            self.recorder.fallbacks += 1
            context.pov.fallbacks += 1
            text, selected = _encode_action(view, _fallback_action(view))
        if selected == 43 and not _agari_is_real(view):
            # The native can_agari flag can only be stale if a previous reaction
            # broke the riichi wait invariant; never emit a phantom win.
            self.recorder.agari_guard += 1
            context.pov.fallbacks += 1
            text, selected = _encode_action(view, _fallback_action(view))
        context.pov.decisions += 1
        context.pov.actions[selected] += 1
        try:
            state.validate_reaction(text)
        except Exception:
            self.recorder.rejected += 1
            context.pov.rejected += 1
            context.pov.fallbacks += 1
            text, selected = _encode_action(view, _fallback_action(view))
            context.pov.actions[selected] += 1
            try:
                state.validate_reaction(text)
            except Exception as exc2:  # pragma: no cover - engine-level bug
                raise RolloutError(
                    "fallback reaction also rejected by the native engine: "
                    f"{type(exc2).__name__}: {exc2}") from exc2
        if self.recorder.verify_bots:
            self._verify_bot_view(state, view)
        return text

    def _verify_bot_view(self, state, view: SeatView) -> None:
        """Test-only cross-checks against the native state (never used in a rollout)."""
        cans = state.last_cans
        tehai = tuple(int(value) for value in state.tehai)
        if view.hand_counts != tehai:
            raise RolloutError("view hand differs from the native concealed hand")
        if cans.can_discard and cans.can_tsumo_agari:
            self.recorder.agari_checks += 1
            if not is_winning_hand(tehai, view.meld_count):
                self.recorder.agari_mismatches += 1
        if cans.can_pass and not cans.can_discard:
            self.recorder.shanten_checks += 1
            if shanten(tehai, view.meld_count) != int(state.shanten):
                self.recorder.shanten_mismatches += 1


def _agari_is_real(view: SeatView) -> bool:
    counts = list(view.hand_counts)
    if view.phase is ActionPhase.CALL_RESPONSE:
        if view.last_kawa is None:
            return False
        counts[view.last_kawa.kind] += 1
    return is_winning_hand(counts, view.meld_count)


def _fallback_action(view: SeatView) -> BotAction:
    if view.is_legal(45):
        return BotAction(45, label="fallback-pass")
    candidates = view.discard_candidates()
    if candidates:
        return BotAction(candidates[0][0], label="fallback-discard")
    legal = view.legal_actions()
    if not legal:
        raise RolloutError("no legal fallback action")
    return BotAction(legal[0], label="fallback-first")


def _kan_options(view: SeatView) -> tuple[KanOption, ...]:
    hand = list(view.own_tiles)
    options: list[KanOption] = []
    for meld_type, kind, aka in view.kan_choices():
        if meld_type == "ankan":
            copies = [tile for tile in hand if tile.kind == kind][:4]
            if len(copies) < 4:
                continue
            options.append(KanOption("ankan", copies[0], tuple(copies)))
        elif meld_type == "kakan":
            copies = [tile for tile in hand
                      if tile.kind == kind and (aka is None or tile.aka == aka)]
            if not copies:
                copies = [tile for tile in hand if tile.kind == kind]
            if not copies:
                continue
            tile = copies[0]
            options.append(KanOption("kakan", tile, (tile, tile, tile)))
        elif meld_type == "daiminkan":
            if view.last_kawa is None or view.last_kawa_actor is None:
                continue
            copies = [tile for tile in hand if tile.kind == kind][:3]
            if len(copies) < 3:
                continue
            options.append(KanOption("daiminkan", view.last_kawa, tuple(copies),
                                     target=view.last_kawa_actor))
    return tuple(options)


def _action_context(view: SeatView, kan_choice: TileRef | None = None) -> ActionContext:
    return ActionContext(
        seat=view.seat,
        phase=view.phase,
        legal_mask=tuple(view.legal_mask),
        hand=tuple(view.own_tiles),
        last_kawa_tile=view.last_kawa,
        last_kawa_actor=view.last_kawa_actor,
        last_self_tsumo=view.drawn,
        kan_options=_kan_options(view),
        kan_choice=kan_choice,
        can_tsumo_agari=bool(view.legal_mask[43]) and view.phase is ActionPhase.SELF_TURN,
        can_ron_agari=bool(view.legal_mask[43]) and view.phase is ActionPhase.CALL_RESPONSE,
        ron_target=view.last_kawa_actor,
        can_ryukyoku=bool(view.legal_mask[44]),
    )


def _encode_action(view: SeatView, action: BotAction) -> tuple[str, int]:
    kan_choice: TileRef | None = None
    if action.index == 42:
        options = _kan_options(view)
        if not options:
            raise RolloutError("action 42 without a concrete kan candidate")
        chosen = options[0]
        if action.kan is not None:
            meld_type, kind, aka = action.kan
            matched = [option for option in options
                       if option.meld_type == meld_type and option.tile.kind == kind
                       and (aka is None or option.tile.aka == aka)]
            if matched:
                chosen = matched[0]
        kan_choice = chosen.tile
    context = _action_context(view, kan_choice=kan_choice)
    try:
        event = action_to_mjai(action.index, context)
    except AdapterError as exc:
        raise RolloutError(f"adapter rejected action {action.index}: {exc}") from exc
    if event.get("type") == "kakan" and "consumed" not in event:
        # The frozen runtime adapter emits the minimal kakan payload; the pinned
        # engine deserializer still requires the three meld tiles.  ``consumed`` is
        # not used by the engine for kakan, it only has to be well formed.
        pai = tile_from_text(event["pai"])
        deaka = tile_text(pai.kind, None)
        event["consumed"] = ([event["pai"], deaka, deaka]
                             if pai.aka is not None else [deaka, event["pai"], event["pai"]])
    return json.dumps(event, separators=(",", ":"), sort_keys=True), action.index


def _view_from_state(state, events_json: str,
                     recorder: "_Recorder | None" = None) -> SeatView:
    cans = state.last_cans
    tehai = tuple(int(value) for value in state.tehai)
    akas = tuple(int(value) for value in state.akas_in_hand)
    log = _parse_public_log(events_json)
    # The native 46-action mask is authoritative: it is the only source that also
    # encodes riichi-only discards and kuikae (swap-call) restrictions.  The
    # independent candidate derivation below is kept as a cross-check.
    native = [bool(value) for value in state.encode_obs(1, False)[1]]
    derived = _mask_from_candidate(cans, tehai, akas)
    if recorder is not None:
        recorder.mask_checks += 1
        if [i for i in range(ACTION_SPACE) if native[i] and not derived[i]]:
            recorder.mask_undercount += 1
        if [i for i in range(ACTION_SPACE) if derived[i] and not native[i]]:
            recorder.mask_restricted += 1
    mask = native
    phase = ActionPhase.CALL_RESPONSE if cans.can_pass else ActionPhase.SELF_TURN
    meld_count = (len(state.chis) + len(state.pons) + len(state.minkans)
                  + len(state.ankans))
    last_kawa = _tile_ref(state.last_kawa_tile())
    last_kawa_actor = int(cans.target_actor) if phase is ActionPhase.CALL_RESPONSE else None
    drawn = _tile_ref(state.last_self_tsumo())
    scores = list(log.start_scores)
    for seat in range(4):
        if log.reach_accepted[seat]:
            scores[seat] -= 1000
    visible = log.visible_counts(tehai)
    self_wind = 27 + (int(state.player_id) - log.oya) % 4
    hand_tiles = physical_hand(tehai, akas)
    kakan: list[tuple[int, int | None]] = []
    for text in state.kakan_candidates():
        tile = tile_from_text(text)
        physical = [candidate for candidate in hand_tiles if candidate.kind == tile.kind]
        kakan.append((tile.kind, physical[0].aka if physical else None))
    daiminkan = None
    if cans.can_daiminkan and last_kawa is not None:
        daiminkan = (last_kawa.kind, last_kawa.aka)
    try:
        return SeatView(
            seat=int(state.player_id),
            hand_counts=tehai,
            hand_akas=akas,
            drawn=drawn,
            legal_mask=tuple(mask),
            phase=phase,
            meld_count=meld_count,
            rivers=tuple(tuple(river) for river in log.rivers),
            riichi=tuple(log.riichi),
            open_melds=tuple(len(melds) for melds in log.melds),
            scores=tuple(scores),
            dora_indicators=tuple(log.dora),
            visible_counts=tuple(visible),
            last_kawa=last_kawa,
            last_kawa_actor=last_kawa_actor,
            ankan_candidates=tuple(tile_from_text(text).kind
                                   for text in state.ankan_candidates()),
            kakan_candidates=tuple(kakan),
            daiminkan_tile=daiminkan,
            kyoku=log.kyoku,
            honba=log.honba,
            kyotaku=log.kyotaku,
            tiles_left=max(0, 70 - log.draws),
            self_wind=self_wind,
            oya=log.oya,
            source=SOURCE_LIBRIICHI,
            ruleset_id=RULESET_LIBRIICHI,
            turn=log.draws,
        )
    except ViewError as exc:
        raise RolloutError(f"cannot build a POV-safe seat view: {exc}") from exc


# ---------------------------------------------------------------------------
# game-log based deviation counters (public events only)
# ---------------------------------------------------------------------------
def _deviation_counts(log_rows: Sequence[str]) -> dict[str, int]:
    counts: Counter = Counter()
    multi_ron = triple_ron = four_riichi = 0
    reach_in_kyoku = 0
    ron_after_discard = 0
    last_reach_accept: dict[int, int] = {}
    ippatsu_proxy = 0
    kyotaku = 0
    index = 0
    for line in log_rows:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "start_kyoku":
            reach_in_kyoku = 0
            kyotaku = int(event.get("kyotaku", 0))
            counts["kyoku"] += 1
        elif kind == "reach":
            reach_in_kyoku += 1
            counts["reach"] += 1
            kyotaku += 1
            if reach_in_kyoku == 4:
                four_riichi += 1
        elif kind == "reach_accepted":
            last_reach_accept[int(event["actor"])] = index
        elif kind in ("chi", "pon", "daiminkan"):
            counts["call"] += 1
        elif kind == "dahai":
            ron_after_discard = 0
        elif kind == "tsumo":
            ron_after_discard = 0
        elif kind == "hora":
            ron_after_discard += 1
            counts["hora"] += 1
            kyotaku = 0
            actor = int(event["actor"])
            accepted = last_reach_accept.get(actor)
            if accepted is not None and index - accepted <= 8:
                ippatsu_proxy += 1
        elif kind in ("ryukyoku", "end_kyoku"):
            counts["ryukyoku" if kind == "ryukyoku" else "end_kyoku"] += 1
            if ron_after_discard >= 2:
                multi_ron += 1
                if ron_after_discard >= 3:
                    triple_ron += 1
            ron_after_discard = 0
        index += 1
    counts["R02_multi_ron"] = multi_ron
    counts["R02_triple_ron"] = triple_ron
    counts["R26_four_riichi"] = four_riichi
    counts["R29_ippatsu_proxy_unverified"] = ippatsu_proxy
    # R25 proxy: the game ended while riichi sticks were still on the table, which is
    # the situation where the last-dealer stop depends on kyotaku-inclusive scores.
    counts["R25_end_kyotaku_carry_proxy"] = 1 if kyotaku > 0 else 0
    return dict(counts)


# ---------------------------------------------------------------------------
# libriichi rollout
# ---------------------------------------------------------------------------
_SEAT_IDS = {
    "challenger": (0, 1, 2, 3),
    "champion": (1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2),
}


def _run_seed_libriichi(seed: int, key: int, lineup: Sequence[str], *,
                        out: Path, write_logs: bool,
                        verify_bots: bool = False) -> list[GameRecord]:
    import libriichi  # local import: the caller already resolved availability

    recorder = _Recorder(
        seed=seed,
        key=key,
        lineup=lineup,
        seat_ids=_SEAT_IDS,
        verify_bots=verify_bots,
    )
    challenger = _SeatEngine("challenger", recorder)
    champion = _SeatEngine("champion", recorder)
    log_dir = None
    if write_logs:
        log_dir = out / "games"
        log_dir.mkdir(parents=True, exist_ok=True)
    arena = libriichi.arena.OneVsThree(disable_progress_bar=True,
                                       log_dir=str(log_dir) if log_dir else None)
    arena.py_vs_py(challenger, champion, (int(seed), int(key)), 1)

    if len(recorder.split_scores) != 4:
        raise RolloutError(
            f"seed {seed}: engine finished {len(recorder.split_scores)} of 4 split games")
    generator_sha = generator_fingerprint()["code_sha256"]
    records: list[GameRecord] = []
    for split in range(4):
        scores = recorder.split_scores[split]
        contexts = recorder.game_contexts(split)
        ranks = _ranks(scores)
        log_path = None
        log_sha = None
        deviations: dict[str, int] = {}
        if write_logs:
            candidate = out / "games" / f"{seed}_{key}_{'abcd'[split]}.json.gz"
            if not candidate.exists():
                raise RolloutError(f"missing engine game log {candidate}")
            with gzip.open(candidate, "rt", encoding="utf-8") as handle:
                rows = handle.read().splitlines()
            log_sha = sha256_file(candidate)
            log_path = str(candidate.relative_to(out))
            deviations = _deviation_counts(rows)
        povs = tuple(sorted((context.pov for context in contexts),
                            key=lambda pov: pov.position))
        records.append(GameRecord(
            game_id=f"{seed}:{key}:{split}",
            seed=seed,
            key=key,
            rotation=split,
            ruleset_id=RULESET_LIBRIICHI,
            rules_engine=ENGINE_LIBRIICHI,
            generator_sha256=generator_sha,
            version=SELFPLAY_VERSION,
            lineup=tuple(pov.bot for pov in povs),
            seats=tuple((split + position) % 4 for position in range(4)),
            scores=tuple(scores[(split + position) % 4] for position in range(4)),
            ranks=tuple(ranks[(split + position) % 4] for position in range(4)),
            povs=povs,
            decisions=sum(pov.decisions for pov in povs),
            illegal=sum(pov.illegal for pov in povs),
            rejected=sum(pov.rejected for pov in povs),
            fallbacks=sum(pov.fallbacks for pov in povs),
            deviation_counts=deviations,
            game_log_path=log_path,
            game_log_sha256=log_sha,
            notes=("four-seat rotation of one wall seed; libriichi tenhou-approximate",),
        ))
    if recorder.mask_undercount:
        raise RolloutError(
            "native 46-action mask contained an action the derived candidate set did "
            f"not explain ({recorder.mask_undercount}/{recorder.mask_checks} checks)")
    if recorder.agari_guard:
        raise RolloutError(
            f"{recorder.agari_guard} phantom agari flags were suppressed; the engine "
            "state is inconsistent")
    if recorder.agari_mismatches:
        raise RolloutError(
            f"py win check disagreed with the native agari flag "
            f"{recorder.agari_mismatches}/{recorder.agari_checks} times")
    if recorder.shanten_mismatches:
        raise RolloutError(
            f"py shanten disagreed with the native shanten "
            f"{recorder.shanten_mismatches}/{recorder.shanten_checks} times")
    if recorder.bot_errors:
        raise RolloutError(f"bot returned an illegal action {recorder.bot_errors} times")
    if recorder.rejected:
        raise RolloutError(f"{recorder.rejected} reactions were rejected by the engine")
    records[0].notes = records[0].notes + (
        f"mask_checks={recorder.mask_checks}",
        f"mask_restricted_by_engine={recorder.mask_restricted}",
        f"verify_agari_checks={recorder.agari_checks}",
        f"verify_shanten_checks={recorder.shanten_checks}",
        f"engine_decisions={recorder.views}",
    )
    return records


def _ranks(scores: Sequence[int]) -> tuple[int, int, int, int]:
    order = sorted(range(4), key=lambda seat: (-int(scores[seat]), seat))
    ranks = [0] * 4
    for rank, seat in enumerate(order, start=1):
        ranks[seat] = rank
    return tuple(ranks)


# ---------------------------------------------------------------------------
# procedural fallback
# ---------------------------------------------------------------------------
def _synthetic_view(hand: Sequence[int], *, seat: int, scores, riichi, rivers,
                    dora: TileRef, kyoku: int, honba: int, kyotaku: int,
                    wall_left: int, dealer: int, legal: Sequence[bool]) -> SeatView:
    counts = [0] * TILE_KIND_COUNT
    for kind in hand:
        counts[kind] += 1
    visible = list(counts)
    for river in rivers:
        for kind in river:
            visible[kind] += 1
    visible[dora.kind] += 1
    return SeatView(
        seat=seat,
        hand_counts=tuple(counts),
        hand_akas=(0, 0, 0),
        drawn=TileRef(hand[-1]),
        legal_mask=tuple(legal),
        phase=ActionPhase.SELF_TURN,
        meld_count=0,
        rivers=tuple(tuple(TileRef(kind) for kind in river) for river in rivers),
        riichi=tuple(riichi),
        open_melds=(0, 0, 0, 0),
        scores=tuple(scores),
        dora_indicators=(dora,),
        visible_counts=tuple(visible),
        kyoku=kyoku,
        honba=honba,
        kyotaku=kyotaku,
        tiles_left=wall_left,
        self_wind=27 + (seat - dealer) % 4,
        oya=dealer,
        source=SOURCE_SYNTHETIC,
        ruleset_id=RULESET_SYNTHETIC,
    )


def _synthetic_discard_mask(hand: Sequence[int], *, can_riichi: bool) -> list[bool]:
    mask = [False] * ACTION_SPACE
    for kind in range(TILE_KIND_COUNT):
        if hand[kind] > 0:
            mask[kind] = True
    if can_riichi:
        mask[37] = True
    return mask


def _synthetic_riichi_mask(hand: Sequence[int]) -> list[bool]:
    mask = [False] * ACTION_SPACE
    counts = list(hand)
    for kind in range(TILE_KIND_COUNT):
        if counts[kind] <= 0:
            continue
        counts[kind] -= 1
        if shanten(counts) == 0:
            mask[kind] = True
        counts[kind] += 1
    if not any(mask):  # pragma: no cover - the declared hand was tenpai
        for kind in range(TILE_KIND_COUNT):
            if hand[kind] > 0:
                mask[kind] = True
    return mask


def _run_seed_synthetic(seed: int, key: int, lineup: Sequence[str], *,
                        out: Path) -> list[GameRecord]:
    generator_sha = generator_fingerprint()["code_sha256"]
    records: list[GameRecord] = []
    for rotation in range(ROTATIONS):
        bots = [make_bot(lineup[position],
                         seed=_bot_seed(seed, rotation, position, "challenger"))
                for position in range(4)]
        scores = [25000] * 4
        kyotaku = 0
        decisions = 0
        povs = {position: SeatPovRecord(position=position,
                                        seat=(rotation + position) % 4,
                                        bot=lineup[position]) for position in range(4)}
        for kyoku in range(8):
            dealer = kyoku % 4
            rng = random.Random((seed * 1_000_003) ^ (key * 7_919) ^ (kyoku * 104_729))
            deck = [kind for kind in range(TILE_KIND_COUNT) for _ in range(4)]
            rng.shuffle(deck)
            dora = TileRef(deck[0])
            live = deck[14:]
            hands = [sorted(live[seat * 13:(seat + 1) * 13]) for seat in range(4)]
            wall = live[52:]
            rivers: list[list[int]] = [[] for _ in range(4)]
            riichi = [False] * 4
            total = len(wall)
            while wall:
                seat = (dealer + (total - len(wall))) % 4
                position = (seat - rotation) % 4
                bot = bots[position]
                pov = povs[position]
                hand = hands[seat]
                hand.append(wall.pop())
                counts = [0] * TILE_KIND_COUNT
                for kind in hand:
                    counts[kind] += 1
                tenpai = min(
                    shanten([counts[i] - (1 if i == kind else 0)
                             for i in range(TILE_KIND_COUNT)])
                    for kind in range(TILE_KIND_COUNT) if counts[kind] > 0)
                can_riichi = (not riichi[seat]) and tenpai == 0 and len(wall) >= 4 \
                    and scores[seat] >= 1000
                mask = _synthetic_discard_mask(counts, can_riichi=can_riichi)
                view = _synthetic_view(
                    hand, seat=seat, scores=scores, riichi=riichi, rivers=rivers,
                    dora=dora, kyoku=kyoku + 1, honba=0, kyotaku=kyotaku,
                    wall_left=len(wall), dealer=dealer, legal=mask)
                action = bot.act(view)
                pov.decisions += 1
                pov.actions[action.index] += 1
                decisions += 1
                if not mask[action.index]:
                    raise RolloutError("synthetic bot chose an illegal action")
                if action.index == 37:
                    riichi[seat] = True
                    scores[seat] -= 1000
                    kyotaku += 1
                    restricted = _synthetic_riichi_mask(counts)
                    view2 = _synthetic_view(
                        hand, seat=seat, scores=scores, riichi=riichi, rivers=rivers,
                        dora=dora, kyoku=kyoku + 1, honba=0, kyotaku=kyotaku,
                        wall_left=len(wall), dealer=dealer, legal=restricted)
                    action = bot.act(view2)
                    pov.decisions += 1
                    pov.actions[action.index] += 1
                    decisions += 1
                    if not restricted[action.index]:
                        raise RolloutError("riichi discard escaped the restricted mask")
                kind = action.index
                if kind >= 34 or kind not in hand:
                    raise RolloutError("synthetic fallback only supports kind discards")
                hand.remove(kind)
                rivers[seat].append(kind)
            concealed = []
            for seat in range(4):
                counts = [0] * TILE_KIND_COUNT
                for kind in hands[seat]:
                    counts[kind] += 1
                concealed.append(counts)
            tenpai_seats = [seat for seat in range(4) if shanten(concealed[seat]) == 0]
            if 0 < len(tenpai_seats) < 4:
                tenpai_count = len(tenpai_seats)
                noten_count = 4 - tenpai_count
                per_noten = 3000 // noten_count
                per_tenpai = 3000 // tenpai_count
                for seat in range(4):
                    if seat in tenpai_seats:
                        scores[seat] += per_tenpai
                    else:
                        scores[seat] -= per_noten
        if kyotaku:
            top = min(range(4), key=lambda seat: (-scores[seat], seat))
            scores[top] += kyotaku * 1000
        ranks = _ranks(scores)
        records.append(GameRecord(
            game_id=f"{seed}:{key}:{rotation}",
            seed=seed,
            key=key,
            rotation=rotation,
            ruleset_id=RULESET_SYNTHETIC,
            rules_engine=ENGINE_SYNTHETIC,
            generator_sha256=generator_sha,
            version=SELFPLAY_VERSION,
            lineup=tuple(lineup),
            seats=tuple((rotation + position) % 4 for position in range(4)),
            scores=tuple(scores[(rotation + position) % 4] for position in range(4)),
            ranks=tuple(ranks[(rotation + position) % 4] for position in range(4)),
            povs=tuple(povs[position] for position in range(4)),
            decisions=decisions,
            illegal=0,
            rejected=0,
            fallbacks=0,
            deviation_counts={},
            notes=("procedural fallback: no calls/wins, tenpai payments + riichi sticks only",
                   "not mahjong-equivalent; not a strength measurement"),
        ))
    return records


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RolloutConfig:
    seed_start: int
    games: int
    ruleset_id: str
    out: Path
    key: int = DEFAULT_KEY
    lineup: tuple[str, ...] | None = None
    seed_pool: str = "train"
    ledger: SeedLedger | None = None
    write_logs: bool = True
    verify_bots: bool = False

    def __post_init__(self):
        object.__setattr__(self, "out", Path(self.out))
        if self.lineup is None:
            from .bots import default_lineup

            object.__setattr__(self, "lineup", default_lineup())
        if len(self.lineup) != 4:
            raise RolloutError(f"lineup must have 4 bot names, got {self.lineup!r}")
        for name in self.lineup:
            if name not in BOT_FACTORIES:
                raise RolloutError(
                    f"unknown bot {name!r}; known: {sorted(BOT_FACTORIES)}")
        assert_known_ruleset(self.ruleset_id)
        validate_games(self.games)


def _summary_per_bot(records: Sequence[GameRecord]) -> dict:
    stats: dict[str, dict] = {}
    for record in records:
        for pov in record.povs:
            entry = stats.setdefault(pov.bot, {
                "games": 0, "decisions": 0, "score_total": 0,
                "rank_counts": {"1": 0, "2": 0, "3": 0, "4": 0},
            })
            entry["games"] += 1
            entry["decisions"] += pov.decisions
            entry["score_total"] += record.scores[pov.position]
            entry["rank_counts"][str(record.ranks[pov.position])] += 1
    result = {}
    for name, entry in sorted(stats.items()):
        games = entry["games"]
        result[name] = {
            "games": games,
            "decisions": entry["decisions"],
            "mean_score": round(entry["score_total"] / games, 3) if games else None,
            "mean_rank": round(sum(int(rank) * count
                                   for rank, count in entry["rank_counts"].items()) / games, 4)
            if games else None,
            "rank_counts": entry["rank_counts"],
        }
    return result


def _audit(records: Sequence[GameRecord]) -> dict:
    violations = []
    for record in records:
        if record.score_sum != 100_000:
            violations.append(f"{record.game_id}: score sum {record.score_sum} != 100000")
        if record.illegal:
            violations.append(f"{record.game_id}: {record.illegal} illegal actions")
        if record.rejected:
            violations.append(f"{record.game_id}: {record.rejected} rejected reactions")
        if record.fallbacks:
            violations.append(f"{record.game_id}: {record.fallbacks} fallbacks")
        if len(record.povs) != 4:
            violations.append(f"{record.game_id}: {len(record.povs)} POVs, expected 4")
        if len(set(record.seats)) != 4:
            violations.append(f"{record.game_id}: seat rotation is not a permutation")
        if sorted(record.ranks) != [1, 2, 3, 4]:
            violations.append(f"{record.game_id}: ranks {record.ranks} are not 1..4")
    return {
        "ok": not violations,
        "games": len(records),
        "score_conservation": {
            "expected_sum": 100_000,
            "violations": sum(1 for record in records if record.score_sum != 100_000),
        },
        "illegal_actions": sum(record.illegal for record in records),
        "rejected_actions": sum(record.rejected for record in records),
        "fallbacks": sum(record.fallbacks for record in records),
        "violations": violations,
    }


def _aggregate_deviations(records: Sequence[GameRecord]) -> dict:
    totals: Counter = Counter()
    for record in records:
        totals.update(record.deviation_counts)
    return dict(sorted(totals.items()))


def run_selfplay(config: RolloutConfig, *,
                 progress: Callable[[str], None] | None = None) -> dict:
    """Run the requested rotations, write ``--out`` artifacts and return the summary."""
    engine = resolve_engine(config.ruleset_id)
    out = config.out
    out.mkdir(parents=True, exist_ok=True)
    seeds = seed_range(config.seed_start, config.games)
    if config.ledger is not None:
        config.ledger.reserve_range(config.seed_pool, config.seed_start, config.games,
                                    tag=f"selfplay:{config.ruleset_id}")
    generator = generator_fingerprint()

    records: list[GameRecord] = []
    for seed in seeds:
        if progress is not None:
            progress(f"seed {seed}")
        if config.ruleset_id == RULESET_LIBRIICHI:
            records.extend(_run_seed_libriichi(
                seed, config.key, config.lineup, out=out,
                write_logs=config.write_logs,
                verify_bots=config.verify_bots))
        else:
            records.extend(_run_seed_synthetic(seed, config.key, config.lineup, out=out))

    if config.ledger is not None:
        config.ledger.consume(config.seed_pool, seeds)

    audit = _audit(records)
    ruleset = ruleset_metadata(config.ruleset_id)
    summary = {
        "task": "PR-41",
        "mode": "selfplay",
        "version": SELFPLAY_VERSION,
        "ruleset_id": config.ruleset_id,
        "rules_engine": engine,
        "tenhou_equivalent": False,
        "parity_status": ruleset["parity_status"],
        "strength_claim": "none",
        "seed_start": config.seed_start,
        "key": config.key,
        "seed_count": len(seeds),
        "games": len(records),
        "rotations": ROTATIONS,
        "lineup": list(config.lineup),
        "decisions": sum(record.decisions for record in records),
        "illegal_actions": audit["illegal_actions"],
        "rejected_actions": audit["rejected_actions"],
        "fallbacks": audit["fallbacks"],
        "score_conservation": audit["score_conservation"],
        "per_bot": _summary_per_bot(records),
        "deviations": _aggregate_deviations(records),
        "audit": audit,
        "generator_sha256": generator["code_sha256"],
        "seed_pool": config.seed_pool,
        "not_run": [
            "no network access, no online Tenhou/Majsoul bot play",
            "no external log download and no third-party checkpoint",
            "no Tenhou-parity verification (PR-40 deviations R02/R25/R26/R29)",
            "no official rank / strength claim",
        ],
        "outputs": {
            "summary": "summary.json",
            "ledger": "games.jsonl",
            "provenance": "provenance.json",
            "seed_ledger": "seeds.json" if config.ledger is not None else None,
            "game_logs": "games/" if (config.write_logs
                                      and config.ruleset_id == RULESET_LIBRIICHI) else None,
            "game_log_note": (
                "raw engine logs are full-information (all four haipai and ura "
                "markers); they are not POV-safe training rows"
                if (config.write_logs and config.ruleset_id == RULESET_LIBRIICHI) else None
            ),
        },
    }
    provenance = build_provenance(
        ruleset_id=config.ruleset_id,
        engine=engine,
        generator=generator,
        extra={
            "seed_start": config.seed_start,
            "seed_count": len(seeds),
            "games": len(records),
            "key": config.key,
            "lineup": list(config.lineup),
            "seed_pool": config.seed_pool,
            "libriichi_status": libriichi_status(),
            "source": SELFPLAY_SOURCE,
        },
    )
    write_jsonl(out / "games.jsonl", [record.as_dict() for record in records])
    write_json(out / "summary.json", summary)
    write_json(out / "provenance.json", provenance)
    if config.ledger is not None and config.ledger.path is not None:
        write_json(out / "seeds.json", config.ledger.as_dict())
    return summary


def run_seed(seed: int, key: int, lineup: Sequence[str], *,
             ruleset_id: str, out: Path, write_logs: bool = True) -> list[GameRecord]:
    """Convenience wrapper used by tests and by :class:`DuplicateArena`."""
    assert_known_ruleset(ruleset_id)
    Path(out).mkdir(parents=True, exist_ok=True)
    if ruleset_id == RULESET_LIBRIICHI:
        resolve_engine(ruleset_id)
        return _run_seed_libriichi(seed, key, lineup, out=Path(out),
                                   write_logs=write_logs)
    return _run_seed_synthetic(seed, key, lineup, out=Path(out))
