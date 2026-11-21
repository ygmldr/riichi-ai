#!/usr/bin/env python3
"""PR-46B: native 1v3 duplicate arena with a frozen three-seat M5D-R0 champion.

Champion
    Three identical ``M5D-R0`` seats, served by one batch-capable
    :class:`~riichiformer.eval.mortal_anchor.MortalCheckpointAdapter` exactly as the
    native ``OneVsThree.py_vs_py`` driver expects (the driver assigns the three
    champion player ids of each rotated game to one Python object).

Challenger
    ``--candidate mortal`` is a **self-control only** mode: the same M5D-R0 adapter
    plays the candidate seat.  It measures rotation/seed/arena bias; it is not an RF
    candidate and the report says so.  RF-candidate quality is ``NOT_RUN`` in this
    mode.

    ``--candidate rf`` bridges a RiichiFormer checkpoint policy through an
    ``engine_type='mjai-log'`` seat.  Each reaction receives the native
    ``PlayerState`` plus the current visible MJAI log (``game_state.events_json``);
    the log is re-projected to the seat's POV with :class:`POVProjector`, the
    ``DecisionInput`` is rebuilt from public events only, encoded and forwarded by
    the exact PR-13 :class:`Policy`, and the argmax action is decoded with
    :mod:`riichiformer.runtime.mjai_adapter`.  The decoder's context is built from
    the **native** state (hand, last kawa, last self tsumo, kan candidates) so a
    reconstructed event prefix can never introduce a false ``tsumogiri`` or a stale
    call target.

There is no hidden-tile input anywhere: the projection drops ``start_kyoku.tehais``
of other seats, masks opponent ``ankan`` tiles, and the forward receives only a
``DecisionInput`` (which itself refuses opponent concealed tiles).  An illegal or
rejected reaction aborts the run; there is no random/fallback action.

The historical frozen output directories (``runs/1v3``, ``runs/variance``,
``checkpoints``, ``datasets``, ``vendor``) are refused by :func:`guard_output_dir`.
"""

from __future__ import annotations

import gzip
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..data import pov as pov_module
from ..data.decision import DecisionBuildConfig, _SnapshotTracker
from ..data.encode import StreamMeta, encode_decision
from ..data.replay import replay_scores
from ..runtime.mjai_adapter import (
    ActionContext,
    AdapterError,
    AmbiguousKanError,
    KanOption,
    action_to_mjai,
    derive_chi_options,
)
from ..schemas.action_v3 import ActionPhase
from ..schemas.decision_v3 import DecisionInput, RulesetId, SourceKind
from ..schemas.event_v3 import TileRef, UNKNOWN_TILE_KIND, tile_from_text
from ..selfplay.provenance import (
    RULESET_LIBRIICHI,
    ruleset_metadata,
    sha256_file,
)
from ..selfplay.tiles import physical_hand, tile_text
from . import relative_pt
from .mortal_anchor import (
    ANCHOR_ID,
    DEFAULT_CHECKPOINT_PATH,
    REPO_ROOT,
    MortalCheckpointAdapter,
    load_baseline_registry,
    load_mortal_checkpoint,
)

__all__ = [
    "ArenaError",
    "BridgeError",
    "IllegalReaction",
    "M5DArenaError",
    "ScoreRecorder",
    "PovDecisionStream",
    "RFPolicyBackend",
    "RFMjaiLogEngine",
    "RecordingMortalEngine",
    "ArenaRun",
    "run_arena",
    "plan_run",
    "guard_output_dir",
    "seed_window",
    "DEFAULT_WINDOW_START",
    "MAX_SMOKE_SETS",
]

DEFAULT_WINDOW_START = 200_000
MAX_SMOKE_SETS = 20
_CHALLENGER_IDS = (0, 1, 2, 3)
_FROZEN_COMPONENTS = ("runs/1v3", "runs/variance", "checkpoints", "datasets", "vendor")


class M5DArenaError(RuntimeError):
    """Arena precondition or consistency failure (fail closed)."""


class ArenaError(M5DArenaError):
    """Alias kept explicit for callers/tests."""


class BridgeError(M5DArenaError):
    """The RF policy → MJAI bridge produced an unusable decision."""


class IllegalReaction(M5DArenaError):
    """The native engine rejected a decoded reaction; the run must abort."""


class KanSelectionRequired(M5DArenaError):
    """Action 42 is legal for more than one concrete kan; a second decision is required."""

    def __init__(self, context):
        super().__init__(
            f"action 42 has {len(context.kan_options)} concrete kan candidates; "
            "run a KAN_SELECTION decision before decoding")
        self.context = context
        self.options = tuple(context.kan_options)


# ---------------------------------------------------------------------------
# output safety
# ---------------------------------------------------------------------------
def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def guard_output_dir(path: str | Path, *, repo_root: Path | None = None) -> Path:
    """Refuse frozen/committed locations; return the resolved output directory.

    The arena may write game logs and run manifests only to a caller-owned scratch
    directory.  It never writes into the repository root, the frozen evaluation
    corpora, datasets, checkpoints or the vendor tree.
    """
    root = _resolve(repo_root or REPO_ROOT)
    target = _resolve(path)
    if target == root:
        raise M5DArenaError("refusing to write arena output into the repository root")
    for component in _FROZEN_COMPONENTS:
        frozen = (root / component).resolve()
        try:
            target.relative_to(frozen)
        except ValueError:
            continue
        raise M5DArenaError(f"refusing to write under frozen path {frozen}")
    for name in ("datasets", "checkpoints", "vendor"):
        if name in target.parts:
            raise M5DArenaError(f"refusing to write under a {name!r} path: {target}")
    return target


def seed_window(start: int, sets: int) -> tuple[int, int]:
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise M5DArenaError(f"seed_start must be a non-negative int, got {start!r}")
    if isinstance(sets, bool) or not isinstance(sets, int) or not 1 <= sets <= 100:
        raise M5DArenaError(f"sets must be an int in 1..100, got {sets!r}")
    return start, start + sets


def fresh_seed_key() -> int:
    """A fresh 64-bit wall pool key (never a historical/consumed pool)."""
    return secrets.randbits(64)


def _seed_key_int(value) -> int:
    if isinstance(value, bool):
        raise M5DArenaError("seed_key must be an int or hex string")
    if isinstance(value, int):
        if not 0 <= value < 2**64:
            raise M5DArenaError("seed_key is outside u64")
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("0x"):
            base = 16
            text = text[2:]
        elif any(ch in "abcdef" for ch in text):
            base = 16
        else:
            base = 10
        if not text or any(ch not in "0123456789abcdef" for ch in text):
            raise M5DArenaError(f"malformed seed_key {value!r}")
        try:
            number = int(text, base)
        except ValueError as exc:
            raise M5DArenaError(f"malformed seed_key {value!r}") from exc
        if not 0 <= number < 2**64:
            raise M5DArenaError("seed_key is outside u64")
        return number
    raise M5DArenaError(f"seed_key must be int or hex str, got {type(value).__name__}")


def seed_key_hex(key: int) -> str:
    return f"{_seed_key_int(key):016x}"


# ---------------------------------------------------------------------------
# POV-safe decision input
# ---------------------------------------------------------------------------
_DEFAULT_BUILD_CONFIG = DecisionBuildConfig(
    ruleset_id=RulesetId.MORTAL_ARENA_PINNED,
    source_kind=SourceKind.ARENA_SELFPLAY,
    ruleset_deviation=(),
)
_ARENA_PROJECTION_POLICY = pov_module.ProjectionPolicy(
    ruleset_id=RulesetId.MORTAL_ARENA_PINNED,
    source_kind=SourceKind.ARENA_SELFPLAY,
    reveal_opponent_ankan=False,
)


def _kyoku_identity(event: Mapping) -> tuple:
    """Identity of a kyoku-start event (used only to allow a legitimate reset)."""
    if event.get("type") == "start_kyoku":
        return (
            "start_kyoku",
            event.get("bakaze"),
            event.get("kyoku"),
            event.get("honba"),
            event.get("oya"),
            tuple(event.get("scores", ())),
        )
    if event.get("type") == "start_game":
        return ("start_game", tuple(event.get("seed", ())))
    return ("other", event.get("type"))


@dataclass(frozen=True)
class _Opportunity:
    """The subset of ``data.decision._Opportunity`` used by ``build_input``."""

    label: int | None
    kan_select: int | None
    before_event_idx: int
    phase: ActionPhase
    mask: tuple[bool, ...]
    at_kyoku: int = 0
    at_turn: int = 0
    censor_reason: str | None = None


class PovDecisionStream:
    """Incrementally project one seat's visible arena log.

    The native arena hands the agent ``events_json`` for the **current kyoku**
    (``Board::take_log`` clears it at ``end_kyoku``).  This class keeps the completed
    kyoku logs and rebuilds the projection over the whole visible game history, so
    the RF policy sees the same full-game prefix shape it was trained on.  Within a
    kyoku only the new events are projected.  A rewritten prefix is a contract
    failure, never a silent rewind.
    """

    def __init__(self, pov: int):
        if isinstance(pov, bool) or not isinstance(pov, int) or not 0 <= pov <= 3:
            raise BridgeError(f"pov must be int 0..3, got {pov!r}")
        self.pov = pov
        self.history: list[dict] = []
        self.current: list[dict] = []
        self._reset_projection()

    def _reset_projection(self) -> None:
        self.projector = pov_module.POVProjector(
            self.pov, policy=_ARENA_PROJECTION_POLICY)
        self.tracker = _SnapshotTracker(_DEFAULT_BUILD_CONFIG)
        self.projected: list = []

    def _project_new(self, events: Sequence[Mapping]) -> None:
        for raw_event in events:
            if not isinstance(raw_event, Mapping):
                raise BridgeError("events_json contains a non-object event")
            projected = self.projector.project(dict(raw_event))
            self.tracker.apply(projected)
            self.projected.append(projected)

    def _rebuild(self) -> None:
        self._reset_projection()
        self._project_new(self.history + self.current)

    def update(self, events_json: str) -> tuple[tuple, StreamMeta]:
        try:
            raw = json.loads(events_json)
        except json.JSONDecodeError as exc:
            raise BridgeError(f"events_json is not valid JSON: {exc}") from exc
        if not isinstance(raw, list) or not raw:
            raise BridgeError("events_json is not a non-empty list")
        if not all(isinstance(event, dict) for event in raw):
            raise BridgeError("events_json contains a non-object event")

        if self.current and (len(raw) < len(self.current)
                             or raw[:len(self.current)] != self.current):
            # A new kyoku must start at start_kyoku and must not be a rewrite of the
            # same kyoku (same identity + changed prefix).
            if not (raw and raw[0].get("type") == "start_kyoku"):
                raise BridgeError("arena event log changed its already-visible prefix")
            if (self.current[0].get("type") == "start_kyoku"
                    and _kyoku_identity(raw[0]) == _kyoku_identity(self.current[0])):
                raise BridgeError("arena event log changed mid-kyoku")
            self.history.extend(self.current)
            self.current = raw
            self._rebuild()
        elif not self.current:
            self.current = raw
            self._rebuild()
        elif len(raw) > len(self.current):
            new_events = raw[len(self.current):]
            self.current = raw
            self._project_new(new_events)
        if not self.projected:
            raise BridgeError("no projected events at a decision")
        header = self.projector.header
        return tuple(self.projected), StreamMeta.from_header(header)

    def build_decision(self, events_json: str, *, legal_mask: Sequence[bool],
                       phase: ActionPhase) -> tuple[DecisionInput, StreamMeta]:
        events, meta = self.update(events_json)
        mask = tuple(bool(value) for value in legal_mask)
        if len(mask) != 46 or not any(mask):
            raise BridgeError("native legal mask must be 46 bools with one legal action")
        if not isinstance(phase, ActionPhase):
            raise BridgeError(f"phase must be ActionPhase, got {phase!r}")
        opportunity = _Opportunity(
            label=None,
            kan_select=None,
            before_event_idx=len(events),
            phase=phase,
            mask=mask,
        )
        try:
            decision = self.tracker.build_input(events, opportunity, meta)
        except Exception as exc:  # noqa: BLE001 - every failure aborts the run
            raise BridgeError(
                f"cannot build a POV-safe DecisionInput: {type(exc).__name__}: {exc}"
            ) from exc
        if decision.before_event_idx != len(events):
            raise BridgeError("DecisionInput prefix length drifted")
        return decision, meta


# ---------------------------------------------------------------------------
# native action context / decoding
# ---------------------------------------------------------------------------
def _tile_ref(text: str | None) -> TileRef | None:
    if text is None:
        return None
    return tile_from_text(text)


def _bool_cans(cans, name: str) -> bool:
    value = getattr(cans, name)
    return bool(value() if callable(value) else value)


def _native_hand(state) -> tuple[TileRef, ...]:
    try:
        counts = tuple(int(value) for value in state.tehai)
        akas = tuple(int(value) for value in state.akas_in_hand)
        return physical_hand(counts, akas)
    except Exception as exc:  # noqa: BLE001
        raise BridgeError(f"cannot read the native hand: {type(exc).__name__}: {exc}") from exc


def _native_kan_options(state, hand: Sequence[TileRef], last_kawa: TileRef | None,
                        target: int | None) -> tuple[KanOption, ...]:
    options: list[KanOption] = []
    for text in state.ankan_candidates():
        tile = tile_from_text(text)
        copies = [candidate for candidate in hand if candidate.kind == tile.kind]
        if len(copies) < 4:
            raise BridgeError(
                f"native ankan candidate {text} has {len(copies)} hand copies")
        representative = next((copy for copy in copies if copy.aka is None), copies[0])
        options.append(KanOption("ankan", representative, tuple(copies[:4])))
    for text in state.kakan_candidates():
        tile = tile_from_text(text)
        copies = [candidate for candidate in hand if candidate.kind == tile.kind]
        if not copies:
            raise BridgeError(f"native kakan candidate {text} is not in hand")
        representative = next((copy for copy in copies if copy.aka == tile.aka), copies[0])
        options.append(KanOption("kakan", representative, (representative,) * 3))
    cans = state.last_cans
    if _bool_cans(cans, "can_daiminkan") and last_kawa is not None:
        copies = [candidate for candidate in hand if candidate.kind == last_kawa.kind]
        if len(copies) < 3:
            raise BridgeError("native daiminkan candidate lacks three hand copies")
        options.append(KanOption("daiminkan", last_kawa, tuple(copies[:3]),
                                 target=target))
    return tuple(options)


def context_from_native(
    state,
    *,
    seat: int,
    legal_mask: Sequence[bool] | None = None,
    phase: ActionPhase | None = None,
) -> ActionContext:
    """Build the adapter context from the authoritative native ``PlayerState``.

    The native state is the only source of ``last_self_tsumo``/``last_kawa``/
    ``target_actor``; reconstructing them from the visible event prefix can claim a
    ``tsumogiri`` that the engine does not have (a real bug found during PR-46
    integration).
    """
    if isinstance(seat, bool) or not isinstance(seat, int) or not 0 <= seat <= 3:
        raise BridgeError(f"seat must be int 0..3, got {seat!r}")
    cans = state.last_cans
    if legal_mask is None:
        _feature, native_mask = state.encode_obs(4, False)
        mask = tuple(bool(value) for value in native_mask)
    else:
        mask = tuple(bool(value) for value in legal_mask)
    if len(mask) != 46 or not any(mask):
        raise BridgeError("native/legal mask must be 46 bools with one legal action")
    native_phase = (ActionPhase.SELF_TURN if _bool_cans(cans, "can_discard")
                    else ActionPhase.CALL_RESPONSE)
    if phase is not None and phase is ActionPhase.KAN_SELECTION:
        # The KAN_SELECTION decision shares the native SELF_TURN prefix but has
        # its own policy phase and a tile-only legal mask.  It is not a rewrite of
        # the native phase.
        native_phase = ActionPhase.KAN_SELECTION
    elif phase is not None and phase is not native_phase:
        raise BridgeError(f"phase {phase.value} disagrees with native {native_phase.value}")
    hand = _native_hand(state)
    last_kawa = _tile_ref(state.last_kawa_tile())
    last_tsumo = _tile_ref(state.last_self_tsumo())
    target = int(cans.target_actor) if not _bool_cans(cans, "can_discard") else None
    chi_options = ()
    if native_phase is ActionPhase.CALL_RESPONSE and last_kawa is not None:
        chi_options = tuple(sorted(derive_chi_options(hand, last_kawa).items()))
    kan_options = _native_kan_options(state, hand, last_kawa, target)
    can_tsumo = bool(mask[43] and native_phase is ActionPhase.SELF_TURN)
    can_ron = bool(mask[43] and native_phase is ActionPhase.CALL_RESPONSE)
    return ActionContext(
        seat=seat,
        phase=native_phase,
        legal_mask=mask,
        hand=hand,
        last_kawa_tile=last_kawa,
        last_kawa_actor=target,
        last_self_tsumo=last_tsumo,
        chi_options=chi_options,
        kan_options=kan_options,
        can_tsumo_agari=can_tsumo,
        can_ron_agari=can_ron,
        ron_target=target,
        can_ryukyoku=bool(mask[44]),
    )


def _normalise_kakan_consumed(event: dict) -> dict:
    """The native deserializer requires a well-formed 3-tile kakan ``consumed``."""
    if event.get("type") != "kakan" or "consumed" in event:
        return event
    pai = tile_from_text(event["pai"])
    deaka = tile_text(pai.kind, None)
    event = dict(event)
    event["consumed"] = ([event["pai"], deaka, deaka]
                         if pai.aka is not None else [deaka, event["pai"], event["pai"]])
    return event


def decode_native_action(
    action: int,
    state,
    *,
    seat: int,
    legal_mask: Sequence[bool] | None = None,
    phase: ActionPhase | None = None,
) -> tuple[dict, ActionContext]:
    """Map one policy action to one concrete native-valid MJAI event.

    A bare action 42 with multiple concrete kan candidates is **not** decoded by
    choosing ``kan_options[0]``.  It raises :class:`KanSelectionRequired`; the
    caller must issue a real ``KAN_SELECTION`` decision (tile action 0..36) and
    pass that phase/mask back here.  This makes the second policy call observable
    and prevents silent first-candidate fallback.
    """
    if isinstance(action, bool) or not isinstance(action, int):
        raise BridgeError(f"policy action must be int, got {action!r}")
    context = context_from_native(state, seat=seat, legal_mask=legal_mask, phase=phase)
    if not 0 <= action < 46 or not context.legal_mask[action]:
        raise BridgeError(f"policy action {action} is outside the native legal mask")
    if action == 42 and len(context.kan_options) > 1:
        raise KanSelectionRequired(context)
    try:
        event = action_to_mjai(action, context)
    except AmbiguousKanError as exc:
        raise BridgeError(f"ambiguous kan after deterministic selection: {exc}") from exc
    except AdapterError as exc:
        raise BridgeError(f"adapter rejected action {action}: {exc}") from exc
    event = _normalise_kakan_consumed(event)
    text = json.dumps(event, separators=(",", ":"), sort_keys=True)
    try:
        state.validate_reaction(text)
    except Exception as exc:  # noqa: BLE001
        raise IllegalReaction(
            f"native engine rejected action {action} ({text}): "
            f"{type(exc).__name__}: {exc}") from exc
    return event, context


# ---------------------------------------------------------------------------
# RF policy backend
# ---------------------------------------------------------------------------
def _ema_shadow(payload: Mapping) -> Mapping:
    """Return ``payload['ema']['shadow']`` or fail closed (never Raw fallback)."""
    ema = payload.get("ema")
    if not isinstance(ema, Mapping):
        raise BridgeError(
            "EMA requested but checkpoint has no EMA state; refusing Raw fallback")
    shadow = ema.get("shadow")
    if not isinstance(shadow, Mapping) or not shadow:
        raise BridgeError(
            "EMA requested but checkpoint EMA has no shadow state; refusing Raw fallback")
    return shadow


def _torch():
    import torch

    return torch


@dataclass
class RFPolicyBackend:
    """A checkpointed RiichiFormer ``Policy`` used as one deterministic argmax seat."""

    path: str
    sha256: str
    steps: int
    model_config: dict
    model_config_hash: str
    parameter_count: int
    policy: Any
    use_ema: bool = False
    max_events: int = 4096
    name: str = "rf-candidate"

    def __post_init__(self):
        torch = _torch()
        self.policy = self.policy.to(torch.device("cpu")).eval()
        for parameter in self.policy.parameters():
            if parameter.device.type != "cpu":
                raise BridgeError("RF policy must run on CPU")
            if not torch.isfinite(parameter).all():
                raise BridgeError("RF policy contains a non-finite parameter")

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, use_ema: bool = False,
                        max_events: int = 4096, name: str = "rf-candidate",
                        ) -> "RFPolicyBackend":
        from ..models.policy import MVPConfig, Policy, count_parameters
        from ..train.checkpoint import CHECKPOINT_FORMAT, config_hash

        torch = _torch()
        path = Path(path)
        if not path.is_file():
            raise BridgeError(f"RF checkpoint not found: {path}")
        sha = sha256_file(path)
        # The PR-14 checkpoint is a trusted local research artifact.  Its payload
        # carries RNG state objects (numpy), which `weights_only=True` refuses; the
        # file is hashed above and never downloaded by this module.
        payload = torch.load(str(path), weights_only=False, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise BridgeError("RF checkpoint payload is not a mapping")
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise BridgeError(
                f"RF checkpoint format {payload.get('format')!r} != {CHECKPOINT_FORMAT!r}")
        model_config = payload.get("model_config")
        if not isinstance(model_config, Mapping):
            raise BridgeError("RF checkpoint has no model_config")
        config = MVPConfig.from_mapping(dict(model_config))
        expected_hash = config_hash(dict(model_config))
        if payload.get("model_config_hash") not in (None, expected_hash):
            raise BridgeError("RF checkpoint model_config_hash does not match its config")
        model = Policy(config)
        if use_ema:
            state = _ema_shadow(payload)
        else:
            state = payload.get("model")
        if not isinstance(state, Mapping):
            raise BridgeError("RF checkpoint has no model state_dict")
        try:
            model.load_state_dict(state, strict=True)
        except Exception as exc:  # noqa: BLE001
            raise BridgeError(f"RF state_dict does not match model_config: {exc}") from exc
        counters = payload.get("counters") or {}
        return cls(
            path=str(path.resolve()), sha256=sha, steps=int(counters.get("step", 0)),
            model_config=dict(model_config), model_config_hash=expected_hash,
            parameter_count=count_parameters(model), policy=model, use_ema=bool(use_ema),
            max_events=int(max_events), name=str(name),
        )

    def act(self, decision: DecisionInput, meta: StreamMeta) -> int:
        torch = _torch()
        try:
            encoded = encode_decision(decision, meta, max_events=self.max_events)
        except Exception as exc:  # noqa: BLE001
            raise BridgeError(
                f"encoder rejected the visible prefix: {type(exc).__name__}: {exc}"
            ) from exc
        batch = encoded.to_model_batch()
        with torch.inference_mode():
            output = self.policy(batch)
        action = int(output.masked_logits.argmax(dim=-1)[0])
        if not 0 <= action < 46 or not bool(decision.legal_mask[action]):
            raise BridgeError(f"RF policy returned illegal action {action}")
        return action

    def as_dict(self) -> dict:
        return {
            "mode": "rf",
            "checkpoint_path": self.path,
            "checkpoint_sha256": self.sha256,
            "steps": self.steps,
            "model_config": self.model_config,
            "model_config_hash": self.model_config_hash,
            "parameter_count": self.parameter_count,
            "use_ema": self.use_ema,
            "inference_mode": "deterministic_argmax",
            "device": "cpu",
            "max_events": self.max_events,
        }


# ---------------------------------------------------------------------------
# engines and score capture
# ---------------------------------------------------------------------------
class ScoreRecorder:
    """Final scores reported by one engine role, keyed by the native game index."""

    def __init__(self, role: str):
        self.role = str(role)
        self.scores: dict[int, tuple[int, int, int, int]] = {}

    def record(self, index: int, scores: Sequence[int]) -> None:
        values = tuple(int(value) for value in scores)
        if len(values) != 4:
            raise BridgeError(f"{self.role} end_game returned {len(values)} scores")
        if sum(values) != relative_pt.MAX_SCORE_SUM:
            raise BridgeError(
                f"{self.role} end_game scores sum to {sum(values)}, expected "
                f"{relative_pt.MAX_SCORE_SUM}")
        previous = self.scores.get(int(index))
        if previous is not None and previous != values:
            raise BridgeError(
                f"{self.role} game {index} reported two different score lines")
        self.scores[int(index)] = values


class RFMjaiLogEngine:
    """``engine_type='mjai-log'`` seat backed by the RF checkpoint policy."""

    engine_type = "mjai-log"
    version = 1
    is_oracle = False
    enable_quick_eval = False
    enable_rule_based_agari_guard = False

    def __init__(self, backend: RFPolicyBackend, recorder: ScoreRecorder, *,
                 name: str | None = None):
        self.backend = backend
        self.recorder = recorder
        self.name = name or backend.name
        self.player_ids: tuple[int, ...] | None = None
        self.streams: dict[int, PovDecisionStream] = {}
        self.stats = {
            "decisions": 0, "illegal": 0, "fallback": 0,
            "bridge_errors": 0, "kan_selections": 0,
        }

    def set_player_ids(self, ids) -> None:
        ids = tuple(int(value) for value in ids)
        if not ids or any(value not in _CHALLENGER_IDS for value in ids):
            raise BridgeError(f"invalid challenger seat map {ids!r}")
        if ids != tuple(index % 4 for index in range(len(ids))):
            raise BridgeError(f"challenger seat map is not a 0..3 rotation: {ids!r}")
        self.player_ids = ids

    def start_game(self, index: int) -> None:
        index = int(index)
        self.streams[index] = PovDecisionStream(index % 4)

    def end_kyoku(self, index: int) -> None:
        return None

    def end_game(self, index: int, scores) -> None:
        self.recorder.record(int(index), scores)

    def react_batch(self, game_states) -> list[str]:
        reactions = []
        for game_state in game_states:
            try:
                reactions.append(self._react(game_state))
            except M5DArenaError as exc:
                self.stats["illegal"] += 1
                self.stats["bridge_errors"] += 1
                raise IllegalReaction(
                    f"RF bridge failed on game {game_state.game_index}: {exc}") from exc
        return reactions

    def _react(self, game_state) -> str:
        state = game_state.state
        index = int(game_state.game_index)
        stream = self.streams.get(index)
        if stream is None:
            raise BridgeError(f"no RF stream for game index {index}")
        if int(state.player_id) != stream.pov:
            raise BridgeError(
                f"engine seat {int(state.player_id)} != challenge stream POV {stream.pov}")
        cans = state.last_cans
        phase = (ActionPhase.SELF_TURN if _bool_cans(cans, "can_discard")
                 else ActionPhase.CALL_RESPONSE)
        _feature, native_mask = state.encode_obs(4, False)
        mask = tuple(bool(value) for value in native_mask)
        decision, meta = stream.build_decision(
            game_state.events_json, legal_mask=mask, phase=phase)
        native_counts = tuple(int(value) for value in state.tehai)
        native_akas = tuple(int(value) for value in state.akas_in_hand)
        if tuple(decision.tile_state.own_counts) != native_counts:
            raise BridgeError(
                "projected own hand disagrees with the native PlayerState")
        if tuple(decision.tile_state.own_aka) != native_akas:
            raise BridgeError(
                "projected own aka count disagrees with the native PlayerState")
        action = self.backend.act(decision, meta)
        self.stats["decisions"] += 1
        try:
            event, _context = decode_native_action(
                action, state, seat=stream.pov, legal_mask=mask, phase=phase)
        except KanSelectionRequired as required:
            # Real second policy decision on the unchanged visible prefix.
            kan_mask = [False] * 46
            for option in required.options:
                index = (option.tile.kind if option.tile.aka is None
                         else 34 + option.tile.aka)
                if 0 <= index < 37:
                    kan_mask[index] = True
            if not any(kan_mask):
                raise BridgeError("multi-kan context has no decodable tile action")
            kan_decision, kan_meta = stream.build_decision(
                game_state.events_json, legal_mask=kan_mask,
                phase=ActionPhase.KAN_SELECTION)
            kan_action = self.backend.act(kan_decision, kan_meta)
            self.stats["decisions"] += 1
            self.stats["kan_selections"] += 1
            event, _context = decode_native_action(
                kan_action, state, seat=stream.pov, legal_mask=kan_mask,
                phase=ActionPhase.KAN_SELECTION)
        return json.dumps(event, separators=(",", ":"), sort_keys=True)


class RecordingMortalEngine:
    """``engine_type='mortal'`` wrapper that records final scores for cross-checks."""

    engine_type = "mortal"

    def __init__(self, adapter: MortalCheckpointAdapter, recorder: ScoreRecorder, *,
                 name: str | None = None):
        self.adapter = adapter
        self.recorder = recorder
        self.name = name or adapter.name
        self.version = adapter.version
        self.is_oracle = False
        self.enable_quick_eval = adapter.enable_quick_eval
        self.enable_rule_based_agari_guard = adapter.enable_rule_based_agari_guard

    def react_batch(self, obs, masks, invisible_obs=None):
        return self.adapter.react_batch(obs, masks, invisible_obs)

    def end_game(self, index: int, scores) -> None:
        self.recorder.record(int(index), scores)

    def start_game(self, index: int) -> None:
        return None

    def end_kyoku(self, index: int) -> None:
        return None


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
@dataclass
class ArenaRun:
    mode: str
    self_control_only: bool
    seed_key: int
    seed_start: int
    seed_count: int
    sets: tuple
    report: dict
    rank_counts: tuple[int, int, int, int]
    challenger_scores: dict[int, tuple[int, int, int, int]]
    game_rows: tuple[dict, ...]
    log_files: tuple[dict, ...]
    wall_seconds: float
    candidate: dict
    anchor: dict
    registry_hash: str
    ingested_at_utc: str

    @property
    def games(self) -> int:
        return len(self.game_rows)

    def summary(self) -> dict:
        report = self.report
        return {
            "task": "PR-46B",
            "mode": self.mode,
            "self_control_only": self.self_control_only,
            "seed_key": seed_key_hex(self.seed_key),
            "seed_window": [self.seed_start, self.seed_start + self.seed_count],
            "sets": self.seed_count,
            "games": self.games,
            "wall_seconds": round(self.wall_seconds, 3),
            "games_per_second": round(self.games / self.wall_seconds, 4)
            if self.wall_seconds > 0 else None,
            "delta_pt5": report["delta_pt5"],
            "delta_pt5_ci95": report["delta_pt5_ci95"],
            "avg_rank": report["avg_rank"],
            "pt5_candidate": report["pt5_candidate"],
            "rank_hist": report["rank_hist"],
            "illegal": self.candidate.get("illegal", 0),
            "fallback": self.candidate.get("fallback", 0),
            "anchor_sha256": self.anchor.get("checkpoint_sha256"),
            "candidate_sha256": self.candidate.get("checkpoint_sha256"),
            "rules_hash": self.anchor.get("tenhou_approx_rules_hash"),
            "registry_hash": self.registry_hash,
            "rf_candidate": "NOT_RUN" if self.self_control_only else "RUN",
            "log_files": [item["path"] for item in self.log_files],
        }


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fresh_log_dir(path: Path) -> Path:
    if path.exists():
        if not path.is_dir():
            raise M5DArenaError(f"log path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise M5DArenaError(
                f"log directory is not fresh: {path} (refusing to overwrite game logs)")
    else:
        path.mkdir(parents=True)
    return path


def _read_log(path: Path) -> tuple[list[dict], str]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise M5DArenaError(f"empty arena log {path}")
    # Hash the compressed file itself, so the manifest can be audited with sha256sum.
    return rows, sha256_file(path)


def _ranks(scores: Sequence[int]) -> tuple[int, int, int, int]:
    order = sorted(range(4), key=lambda seat: (-int(scores[seat]), seat))
    ranks = [0, 0, 0, 0]
    for rank, seat in enumerate(order, start=1):
        ranks[seat] = rank
    return tuple(ranks)  # type: ignore[return-value]


def _rank_counts_from_scores(scores: dict[int, tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    counts = [0, 0, 0, 0]
    for index in sorted(scores):
        rotation = index % 4
        counts[_ranks(scores[index])[rotation] - 1] += 1
    return tuple(counts)  # type: ignore[return-value]


def run_arena(
    *,
    seed_key: int | str,
    seed_start: int = DEFAULT_WINDOW_START,
    seed_count: int = 4,
    candidate: str = "mortal",
    candidate_checkpoint: str | Path | None = None,
    candidate_backend: Any | None = None,
    use_ema: bool = False,
    registry_path: str | Path | None = None,
    out_dir: str | Path,
    log_dir: str | Path | None = None,
    bootstrap_resamples: int = relative_pt.BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = relative_pt.BOOTSTRAP_SEED,
    write_logs: bool = True,
    max_events: int = 4096,
    progress: bool = False,
) -> ArenaRun:
    """Run ``seed_count`` complete sets (4 rotated games each) and score them.

    ``candidate='mortal'`` is the anchor self-control; ``candidate='rf'`` requires a
    RiichiFormer checkpoint.  Any bridge/illegal/replay inconsistency aborts before
    a report is returned.
    """
    import libriichi

    if candidate not in ("mortal", "rf"):
        raise M5DArenaError(f"candidate must be 'mortal' or 'rf', got {candidate!r}")
    if not write_logs:
        raise M5DArenaError(
            "native game logs are required for score settlement; write_logs=False "
            "would leave the run without authoritative scores")
    key = _seed_key_int(seed_key)
    start, stop = seed_window(seed_start, seed_count)
    out = guard_output_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if log_dir is None:
        logs = _fresh_log_dir(out / "games")
    else:
        logs = _fresh_log_dir(guard_output_dir(log_dir))

    registry_file = Path(registry_path) if registry_path is not None else (
        REPO_ROOT / "configs" / "riichiformer" / "m5d_baseline.yaml")
    registry = load_baseline_registry(registry_file)
    anchor_adapter = MortalCheckpointAdapter.from_registry(registry)

    candidate_meta: dict
    if candidate == "mortal":
        self_control_only = True
        challenger_adapter = MortalCheckpointAdapter(
            anchor_adapter.loaded, name=f"{ANCHOR_ID}-selfcontrol", seat_index=0)
        challenger_recorder = ScoreRecorder("challenger")
        challenger = RecordingMortalEngine(
            challenger_adapter, challenger_recorder, name=f"{ANCHOR_ID}-selfcontrol")
        candidate_meta = {
            "mode": "mortal_self_control",
            "self_control_only": True,
            "rf_candidate": "NOT_RUN",
            "checkpoint_path": anchor_adapter.identity.path,
            "checkpoint_sha256": anchor_adapter.identity.sha256,
            "model_config_hash": anchor_adapter.identity.model_config_hash,
            "inference_mode": "deterministic_argmax",
            "illegal": 0,
            "fallback": 0,
        }
    else:
        self_control_only = False
        if candidate_backend is None:
            if candidate_checkpoint is None:
                raise M5DArenaError("--candidate rf requires --checkpoint")
            backend = RFPolicyBackend.from_checkpoint(
                candidate_checkpoint, use_ema=use_ema, max_events=max_events)
        else:
            backend = candidate_backend
        challenger_recorder = ScoreRecorder("challenger")
        challenger = RFMjaiLogEngine(backend, challenger_recorder)
        candidate_meta = backend.as_dict()
        candidate_meta.update({"self_control_only": False, "rf_candidate": "RUN",
                               "illegal": 0, "fallback": 0})

    champion = RecordingMortalEngine(anchor_adapter, ScoreRecorder("champion"),
                                     name=ANCHOR_ID)
    started = time.perf_counter()
    arena = libriichi.arena.OneVsThree(
        disable_progress_bar=not progress,
        log_dir=str(logs) if write_logs else None,
    )
    try:
        rank_counts = arena.py_vs_py(
            challenger, champion, (int(start), int(key)), int(seed_count))
    except Exception as exc:  # noqa: BLE001 - preserve the native traceback text
        raise M5DArenaError(
            f"native OneVsThree.py_vs_py failed: {type(exc).__name__}: {exc}") from exc
    wall_seconds = time.perf_counter() - started

    expected = 4 * int(seed_count)

    # The native ``MortalBatchAgent`` does not forward end_game/start_game callbacks
    # to Python (only the mjai-log agent does), so scores are settled from the four
    # dumped native game logs.  That is the shared authoritative path for both
    # candidate modes and is cross-checked against the native rankings array.
    records: list[relative_pt.GameRecord] = []
    game_rows: list[dict] = []
    log_files: list[dict] = []
    scores_by_index: dict[int, tuple[int, int, int, int]] = {}
    for index in range(expected):
        rotation = index % 4
        wall_seed = int(start) + index // 4
        game_id = f"{wall_seed}:{rotation}"
        candidate_path = logs / f"{wall_seed}_{key}_{'abcd'[rotation]}.json.gz"
        if not candidate_path.is_file():
            raise M5DArenaError(f"missing native game log {candidate_path}")
        rows, log_sha = _read_log(candidate_path)
        replay = replay_scores(rows, strict_protocol=False)
        if not replay.ok or replay.scores is None:
            raise M5DArenaError(
                f"replay failed for {candidate_path}: {replay.errors}")
        scores = tuple(int(value) for value in replay.scores)
        if len(scores) != 4 or sum(scores) != relative_pt.MAX_SCORE_SUM:
            raise M5DArenaError(
                f"replayed scores {scores} violate the 100000 invariant: "
                f"{candidate_path}")
        ranks = _ranks(scores)
        scores_by_index[index] = scores
        log_files.append({
            "path": str(candidate_path),
            "sha256": log_sha,
            "game_id": game_id,
            "seed": wall_seed,
            "rotation": rotation,
            "scores": list(scores),
            "ranks": list(ranks),
        })
        records.append(relative_pt.GameRecord(
            seed_key=seed_key_hex(key),
            wall_seed=wall_seed,
            rotation=rotation,
            candidate_seat=rotation,
            scores=scores,
            game_id=game_id,
        ))
        game_rows.append({
            "game_id": game_id,
            "seed_key": seed_key_hex(key),
            "wall_seed": wall_seed,
            "seat_rotation": rotation,
            "candidate_seat": rotation,
            "scores": list(scores),
            "ranks": list(ranks),
            "candidate_rank": ranks[rotation],
            "log_path": str(candidate_path),
            "log_sha256": log_sha,
            "illegal_actions": 0,
            "fallback_actions": 0,
        })
    if len(game_rows) != expected:
        raise M5DArenaError(
            f"settled {len(game_rows)} games, expected {expected}")
    derived_counts = _rank_counts_from_scores(scores_by_index)
    if tuple(int(value) for value in rank_counts) != derived_counts:
        raise M5DArenaError(
            f"native rank counts {list(rank_counts)} != replay-derived "
            f"{list(derived_counts)}")

    expected_seeds = list(range(int(start), int(stop)))
    sets = relative_pt.build_complete_sets(
        records, seed_key=seed_key_hex(key), expected_wall_seeds=expected_seeds)
    report = relative_pt.aggregate_sets(
        sets, seed_key=seed_key_hex(key),
        bootstrap_resamples=bootstrap_resamples, bootstrap_seed=bootstrap_seed)
    if candidate == "rf":
        # The bridge validates every reaction natively and aborts on failure, so a
        # completed run has zero illegal/fallback actions by construction.
        candidate_meta["decisions"] = int(challenger.stats["decisions"])
        candidate_meta["illegal"] = int(challenger.stats["illegal"])
        candidate_meta["fallback"] = int(challenger.stats["fallback"])
        if challenger.stats["decisions"] <= 0:
            raise M5DArenaError("RF challenger made no decisions")

    rules_meta = ruleset_metadata(RULESET_LIBRIICHI)
    anchor_meta = {
        "anchor_id": ANCHOR_ID,
        "checkpoint_path": anchor_adapter.identity.path,
        "checkpoint_sha256": anchor_adapter.identity.sha256,
        "model_config_hash": anchor_adapter.identity.model_config_hash,
        "model_arch": anchor_adapter.identity.model_arch,
        "steps": anchor_adapter.identity.steps,
        "parameter_count": anchor_adapter.loaded.parameter_count,
        "inference_mode": "deterministic_argmax",
        "action_guard": "native_46_action_mask_only",
        "tenhou_approx_rules_hash": registry.get("tenhou_approx_rules_hash"),
        "ruleset_id": RULESET_LIBRIICHI,
        "rules_metadata": rules_meta,
    }
    report.update({
        "mode": candidate_meta["mode"],
        "self_control_only": self_control_only,
        "seed_window": [int(start), int(stop)],
        "wall_seconds": round(wall_seconds, 3),
        "games_per_second": round(expected / wall_seconds, 4) if wall_seconds else None,
        "rank_counts": list(rank_counts),
        "candidate": candidate_meta,
        "anchor": anchor_meta,
        "log_files": log_files,
        "registry_hash": registry.get("registry_hash"),
        "libriichi_build_sha256": registry.get("libriichi_build_sha256"),
        "libriichi_git_sha": registry.get("libriichi_git_sha"),
        "patch_manifest_hash": registry.get("patch_manifest_hash"),
        "not_run": list(report.get("not_run", [])) + (
            ["RF candidate quality: no checkpoint was provided (self-control only)"]
            if self_control_only else []),
        "context_scope": "full_game_visible_prefix",
    })

    run = ArenaRun(
        mode=candidate_meta["mode"],
        self_control_only=self_control_only,
        seed_key=key,
        seed_start=int(start),
        seed_count=int(seed_count),
        sets=sets,
        report=report,
        rank_counts=tuple(int(value) for value in rank_counts),
        challenger_scores=dict(scores_by_index),
        game_rows=tuple(game_rows),
        log_files=tuple(log_files),
        wall_seconds=wall_seconds,
        candidate=candidate_meta,
        anchor=anchor_meta,
        registry_hash=str(registry.get("registry_hash")),
        ingested_at_utc=_utc_now(),
    )
    _write_run(out, run)
    return run


def _write_run(out: Path, run: ArenaRun) -> None:
    manifest = {
        "task": "PR-46B",
        "mode": run.mode,
        "self_control_only": run.self_control_only,
        "seed_key": seed_key_hex(run.seed_key),
        "seed_key_decimal": run.seed_key,
        "seed_window": [run.seed_start, run.seed_start + run.seed_count],
        "sets": run.seed_count,
        "games": run.games,
        "wall_seconds": round(run.wall_seconds, 3),
        "games_per_second": run.report.get("games_per_second"),
        "candidate": run.candidate,
        "anchor": run.anchor,
        "registry_hash": run.registry_hash,
        "rank_counts": list(run.rank_counts),
        "report": run.report,
        "log_files": list(run.log_files),
        "ingested_at_utc": run.ingested_at_utc,
        "not_run": run.report.get("not_run", []),
    }
    (out / "run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with open(out / "games.jsonl", "w", encoding="utf-8") as handle:
        for row in run.game_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (out / "report.json").write_text(
        json.dumps(run.report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plan_run(
    *,
    seed_key: int | str | None = None,
    seed_start: int = DEFAULT_WINDOW_START,
    seed_count: int = 4,
    candidate: str = "mortal",
    candidate_checkpoint: str | Path | None = None,
    registry_path: str | Path | None = None,
    out_dir: str | Path = "/tmp/rf_m5d_plan",
) -> dict:
    """Validate a run plan without playing (used by the CLI's ``--plan-only``)."""
    key = fresh_seed_key() if seed_key is None else _seed_key_int(seed_key)
    start, stop = seed_window(seed_start, seed_count)
    out = guard_output_dir(out_dir)
    registry_file = Path(registry_path) if registry_path is not None else (
        REPO_ROOT / "configs" / "riichiformer" / "m5d_baseline.yaml")
    registry = load_baseline_registry(registry_file)
    candidate_meta: dict = {"mode": "mortal_self_control", "self_control_only": True,
                            "rf_candidate": "NOT_RUN"}
    if candidate == "rf":
        if candidate_checkpoint is None:
            raise M5DArenaError("--candidate rf requires --checkpoint")
        candidate_meta = RFPolicyBackend.from_checkpoint(
            candidate_checkpoint).as_dict()
    elif candidate != "mortal":
        raise M5DArenaError(f"candidate must be 'mortal' or 'rf', got {candidate!r}")
    return {
        "task": "PR-46B",
        "plan": True,
        "seed_key": seed_key_hex(key),
        "seed_window": [start, stop],
        "sets": int(seed_count),
        "games": 4 * int(seed_count),
        "candidate": candidate_meta,
        "out_dir": str(out),
        "registry_hash": registry.get("registry_hash"),
        "anchor_sha256": registry.get("checkpoint_sha256"),
        "rules_hash": registry.get("tenhou_approx_rules_hash"),
        "bootstrap": {
            "resamples": relative_pt.BOOTSTRAP_RESAMPLES,
            "seed": relative_pt.BOOTSTRAP_SEED,
            "unit": "complete_seed_set",
        },
    }
