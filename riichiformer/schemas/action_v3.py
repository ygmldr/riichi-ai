"""Frozen 46-action contract for schema v3 (PR-08).

The mapping is copied from the pinned native engine and is **frozen**:

* ``vendor/mortal/libriichi/src/consts.rs`` defines ``ACTION_SPACE = 37 + 1 + 3 + 1 + 1 + 1 + 1 + 1``;
* ``state/obs_repr.rs`` sets mask indices 0..36 for discards / kan tile choice, 37 riichi,
  38..40 chi low/mid/high, 41 pon, 42 kan, 43 agari, 44 ryukyoku, 45 pass;
* ``dataset/gameplay.rs`` maps a logged action to the same indices.

Indices 0..36 are **phase-dependent**:

* ``SELF_TURN``: discard the tile kind (0..33 normal, 34..36 aka five);
* ``KAN_SELECTION``: the tile kind to kan (0..33 normal, 34..36 aka five);
* ``CALL_RESPONSE``: not legal.

The hash is computed from canonical JSON of :func:`canonical_action_map` (sorted keys,
no whitespace, UTF-8) and is mirrored in ``configs/riichiformer/schema_v3.json``.  Any
change to this file requires a schema-version bump; shards/checkpoints must refuse a
mismatch instead of migrating silently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

SCHEMA_VERSION = "rf-schema-v3/2"
ACTION_SPACE = 46

TILE_KIND_NAMES: tuple[str, ...] = (
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
)
TILE_KIND_COUNT = len(TILE_KIND_NAMES)  # 34
UNKNOWN_TILE_KIND = 34
AKA_COLORS = (0, 1, 2)  # 5m, 5p, 5s
AKA_KINDS = (4, 13, 22)  # 0-based kinds of the five in each suit
AKA_PER_COLOR = 1


class ActionPhase(str, Enum):
    """Decision phase; indices 0..36 must be decoded together with this."""

    SELF_TURN = "SELF_TURN"
    CALL_RESPONSE = "CALL_RESPONSE"
    KAN_SELECTION = "KAN_SELECTION"


class ActionKind(str, Enum):
    DISCARD = "discard"
    DISCARD_AKA = "discard_aka"
    RIICHI = "riichi"
    CHI = "chi"
    PON = "pon"
    KAN = "kan"
    AGARI = "agari"
    RYUKYOKU = "ryukyoku"
    PASS = "pass"


class ChiType(str, Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class ActionSchemaError(ValueError):
    """Base class for action-contract violations."""


class IllegalActionIndex(ActionSchemaError):
    pass


class IllegalActionForPhase(ActionSchemaError):
    pass


class MaskShapeError(ActionSchemaError):
    pass


@dataclass(frozen=True)
class ActionSpec:
    index: int
    name: str
    kind: ActionKind
    tile_kind: int | None
    aka: int | None
    chi_type: ChiType | None
    #: (phase, semantic) pairs; semantic values are stable strings.
    semantics: tuple[tuple[str, str], ...]

    @property
    def phases(self) -> tuple[ActionPhase, ...]:
        return tuple(ActionPhase(phase) for phase, _ in self.semantics)

    def semantic_for(self, phase: ActionPhase) -> str:
        for phase_name, semantic in self.semantics:
            if phase_name == phase.value:
                return semantic
        raise IllegalActionForPhase(
            f"action {self.index} ({self.name}) is not legal in phase {phase.value}"
        )

    def as_entry(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "kind": self.kind.value,
            "tile_kind": self.tile_kind,
            "aka": self.aka,
            "chi_type": self.chi_type.value if self.chi_type is not None else None,
            "semantics": {phase: semantic for phase, semantic in self.semantics},
        }


def _build_action_specs() -> tuple[ActionSpec, ...]:
    specs: list[ActionSpec] = []
    for kind in range(TILE_KIND_COUNT):
        specs.append(ActionSpec(
            index=kind,
            name=f"discard_{TILE_KIND_NAMES[kind]}",
            kind=ActionKind.DISCARD,
            tile_kind=kind,
            aka=None,
            chi_type=None,
            semantics=(("SELF_TURN", "discard"), ("KAN_SELECTION", "kan_tile")),
        ))
    for offset, color in enumerate(AKA_COLORS):
        kind = AKA_KINDS[color]
        specs.append(ActionSpec(
            index=34 + offset,
            name=f"discard_5{'mps'[color]}r",
            kind=ActionKind.DISCARD_AKA,
            tile_kind=kind,
            aka=color,
            chi_type=None,
            semantics=(("SELF_TURN", "discard_aka"), ("KAN_SELECTION", "kan_tile_aka")),
        ))
    specs.append(ActionSpec(37, "riichi", ActionKind.RIICHI, None, None, None,
                            (("SELF_TURN", "riichi"),)))
    for offset, chi_type in enumerate((ChiType.LOW, ChiType.MID, ChiType.HIGH)):
        specs.append(ActionSpec(
            38 + offset, f"chi_{chi_type.value}", ActionKind.CHI, None, None, chi_type,
            (("CALL_RESPONSE", f"chi_{chi_type.value}"),),
        ))
    specs.append(ActionSpec(41, "pon", ActionKind.PON, None, None, None,
                            (("CALL_RESPONSE", "pon"),)))
    specs.append(ActionSpec(42, "kan", ActionKind.KAN, None, None, None,
                            (("SELF_TURN", "kan_decide"), ("CALL_RESPONSE", "kan_decide"))))
    specs.append(ActionSpec(43, "agari", ActionKind.AGARI, None, None, None,
                            (("SELF_TURN", "tsumo_agari"), ("CALL_RESPONSE", "ron_agari"))))
    specs.append(ActionSpec(44, "ryukyoku", ActionKind.RYUKYOKU, None, None, None,
                            (("SELF_TURN", "ryukyoku"),)))
    specs.append(ActionSpec(45, "pass", ActionKind.PASS, None, None, None,
                            (("CALL_RESPONSE", "pass"),)))
    if len(specs) != ACTION_SPACE:
        raise AssertionError(f"action map has {len(specs)} entries, expected {ACTION_SPACE}")
    if [spec.index for spec in specs] != list(range(ACTION_SPACE)):
        raise AssertionError("action indices are not 0..45 in order")
    if len({spec.name for spec in specs}) != ACTION_SPACE:
        raise AssertionError("duplicate action names")
    return tuple(specs)


ACTION_MAP: tuple[ActionSpec, ...] = _build_action_specs()

PHASE_LEGAL_ACTIONS: dict[ActionPhase, frozenset[int]] = {
    ActionPhase.SELF_TURN: frozenset(range(38)) | frozenset({42, 43, 44}),
    ActionPhase.CALL_RESPONSE: frozenset({38, 39, 40, 41, 42, 43, 45}),
    ActionPhase.KAN_SELECTION: frozenset(range(37)),
}
if PHASE_LEGAL_ACTIONS[ActionPhase.SELF_TURN] & PHASE_LEGAL_ACTIONS[ActionPhase.CALL_RESPONSE]:
    # 42/43 overlap is intentional (kan decide / agari have a different semantic per
    # phase); 0..36 must never overlap with CALL_RESPONSE.
    overlap = (PHASE_LEGAL_ACTIONS[ActionPhase.SELF_TURN]
               & PHASE_LEGAL_ACTIONS[ActionPhase.CALL_RESPONSE])
    if overlap - {42, 43}:
        raise AssertionError(f"unexpected SELF_TURN/CALL_RESPONSE overlap: {sorted(overlap)}")

ACTION_INDICES: frozenset[int] = frozenset(range(ACTION_SPACE))


def canonical_json(value) -> str:
    """Canonical JSON used for every schema/action/normalizer hash."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_action_map() -> dict:
    """Canonical, hashable action map (exactly mirrored in ``schema_v3.json``)."""
    return {
        "action_space": ACTION_SPACE,
        "entries": [spec.as_entry() for spec in ACTION_MAP],
    }


ACTION_MAP_HASH: str = sha256_text(canonical_json(canonical_action_map()))


@dataclass(frozen=True)
class DecodedAction:
    index: int
    phase: ActionPhase
    semantic: str
    spec: ActionSpec

    @property
    def tile_kind(self) -> int | None:
        return self.spec.tile_kind

    @property
    def aka(self) -> int | None:
        return self.spec.aka

    @property
    def chi_type(self) -> ChiType | None:
        return self.spec.chi_type

    @property
    def is_tile_choice(self) -> bool:
        return self.semantic in ("discard", "discard_aka", "kan_tile", "kan_tile_aka")


def _coerce_phase(phase) -> ActionPhase:
    if isinstance(phase, ActionPhase):
        return phase
    try:
        return ActionPhase(phase)
    except ValueError as exc:
        raise ActionSchemaError(f"unknown action phase {phase!r}") from exc


def decode_action(index: int, phase) -> DecodedAction:
    """Decode a mask index **with its phase** (0..36 has two meanings)."""
    if not isinstance(index, int) or isinstance(index, bool):
        raise IllegalActionIndex(f"action index must be int, got {index!r}")
    if not 0 <= index < ACTION_SPACE:
        raise IllegalActionIndex(f"action index {index} outside 0..{ACTION_SPACE - 1}")
    phase = _coerce_phase(phase)
    spec = ACTION_MAP[index]
    semantic = spec.semantic_for(phase)
    return DecodedAction(index=index, phase=phase, semantic=semantic, spec=spec)


def action_name(index: int) -> str:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < ACTION_SPACE:
        raise IllegalActionIndex(f"action index {index!r} outside 0..{ACTION_SPACE - 1}")
    return ACTION_MAP[index].name


def mask_from_indices(indices) -> tuple[bool, ...]:
    mask = [False] * ACTION_SPACE
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < ACTION_SPACE:
            raise IllegalActionIndex(f"action index {index!r} outside 0..{ACTION_SPACE - 1}")
        mask[index] = True
    return tuple(mask)


def mask_indices(mask) -> tuple[int, ...]:
    if len(mask) != ACTION_SPACE:
        raise MaskShapeError(f"mask length {len(mask)} != {ACTION_SPACE}")
    return tuple(index for index, value in enumerate(mask) if value)


def validate_mask_for_phase(mask, phase) -> tuple[bool, ...]:
    """Validate a 46-wide legal mask against the phase semantics.

    This is the contract PR-10/PR-11 must satisfy: a mask is never interpreted without
    its phase, and no index outside ``PHASE_LEGAL_ACTIONS[phase]`` may be legal.
    """
    if len(mask) != ACTION_SPACE:
        raise MaskShapeError(f"mask length {len(mask)} != {ACTION_SPACE}")
    phase = _coerce_phase(phase)
    allowed = PHASE_LEGAL_ACTIONS[phase]
    legal = []
    for index, value in enumerate(mask):
        if not isinstance(value, bool):
            raise MaskShapeError(f"mask[{index}] is not bool: {value!r}")
        if value:
            if index not in allowed:
                raise IllegalActionForPhase(
                    f"action {index} ({ACTION_MAP[index].name}) is not legal in "
                    f"phase {phase.value}"
                )
            legal.append(index)
    if not legal:
        raise MaskShapeError(f"empty legal mask in phase {phase.value}")
    return tuple(mask)


__all__ = [
    "ACTION_MAP", "ACTION_MAP_HASH", "ACTION_SPACE", "AKA_COLORS", "AKA_PER_COLOR",
    "AKA_KINDS", "ActionKind", "ActionPhase", "ActionSchemaError", "ActionSpec",
    "ChiType", "DecodedAction", "IllegalActionForPhase", "IllegalActionIndex",
    "MaskShapeError", "PHASE_LEGAL_ACTIONS", "SCHEMA_VERSION", "TILE_KIND_COUNT",
    "TILE_KIND_NAMES", "UNKNOWN_TILE_KIND", "action_name", "canonical_action_map",
    "canonical_json", "decode_action", "mask_from_indices", "mask_indices",
    "sha256_text", "validate_mask_for_phase",
]
