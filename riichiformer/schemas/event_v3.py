"""Explicit, POV-safe observable events for schema v3 (PR-08).

Every MJAI event is a **separate frozen dataclass** with a field allowlist.  There is no
raw-dict passthrough and an unknown ``type`` raises :class:`UnknownEventType`.

Visibility rules enforced by construction:

* ``start_kyoku`` stores only ``own_haipai`` (13 tiles of the POV seat); the other three
  hands are never read into the object;
* an opponent ``tsumo`` becomes ``UNKNOWN``; the own draw keeps its tile;
* an opponent ``ankan`` keeps four ``UNKNOWN`` consumed slots unless the caller explicitly
  opts into the rule-dependent reveal (``reveal_opponent_ankan=True``);
* ``start_game`` keeps only ``aka_flag``/``kyoku_first``: names, account ids, wall seeds and
  log URLs are ignored;
* ``hora``/``ryukyoku`` keep public score deltas; ``hora.ura_markers`` are part of the
  settlement event and must only be exposed at/after it (PR-09 timing rule).

Tiles are ``(kind, aka)``: ``kind`` 0..33, ``UNKNOWN=34``; ``aka`` is the red-five colour
(0=5m, 1=5p, 2=5s) or ``None`` for a normal tile.  Red fives are therefore *not* separate
tile kinds, and a normal five is distinct from an aka five.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import ClassVar, Mapping

from .action_v3 import (
    AKA_COLORS,
    AKA_KINDS,
    AKA_PER_COLOR,
    SCHEMA_VERSION,
    TILE_KIND_COUNT,
    TILE_KIND_NAMES,
    UNKNOWN_TILE_KIND,
)

INITIAL_POINTS = 25_000
KYOTAKU_POINTS = 1_000

DELTAS_ORDER_ABSOLUTE = "absolute"
DELTAS_ORDER_RELATIVE = "relative"
DELTAS_ORDERS = (DELTAS_ORDER_ABSOLUTE, DELTAS_ORDER_RELATIVE)


class EventSchemaError(ValueError):
    """Base class for event-schema violations."""


class UnknownEventType(EventSchemaError):
    """An MJAI event type not covered by the v3 allowlist."""


class SchemaFieldError(EventSchemaError):
    """Missing or unexpected field in a canonical event dict."""


class SchemaValueError(EventSchemaError):
    """A field value violates the v3 contract."""


class EventKind(str, Enum):
    START_GAME = "start_game"
    START_KYOKU = "start_kyoku"
    TSUMO = "tsumo"
    DAHAI = "dahai"
    CHI = "chi"
    PON = "pon"
    DAIMINKAN = "daiminkan"
    KAKAN = "kakan"
    ANKAN = "ankan"
    REACH = "reach"
    REACH_ACCEPTED = "reach_accepted"
    DORA = "dora"
    HORA = "hora"
    RYUKYOKU = "ryukyoku"
    END_KYOKU = "end_kyoku"
    END_GAME = "end_game"


class EventTiming(str, Enum):
    """When the event becomes part of the observable stream."""

    BOUNDARY = "BOUNDARY"        # start/end of a game or kyoku
    TURN = "TURN"                # own turn / public in-turn announcement
    RESPONSE = "RESPONSE"        # call window (chi/pon/daiminkan)
    SETTLEMENT = "SETTLEMENT"    # hora / ryukyoku


VISIBILITY_PUBLIC = "public"
VISIBILITY_SELF_ONLY = "self_only"
VISIBILITY_AFTER_EVENT = "public_after_event"
VISIBILITY_RULE_DEPENDENT = "rule_dependent"
VISIBILITIES = frozenset({
    VISIBILITY_PUBLIC, VISIBILITY_SELF_ONLY,
    VISIBILITY_AFTER_EVENT, VISIBILITY_RULE_DEPENDENT,
})


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------


_TILE_TEXT_TO_KIND: dict[str, int] = {
    name: kind for kind, name in enumerate(TILE_KIND_NAMES)
}


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, order=True)
class TileRef:
    """A visible tile identity: ``kind`` plus optional red-five colour."""

    kind: int
    aka: int | None = None

    def __post_init__(self):
        if not _is_int(self.kind):
            raise SchemaValueError(f"tile kind must be int, got {self.kind!r}")
        if not 0 <= self.kind <= UNKNOWN_TILE_KIND:
            raise SchemaValueError(
                f"tile kind {self.kind} outside 0..{UNKNOWN_TILE_KIND}"
            )
        if self.aka is None:
            return
        if not _is_int(self.aka) or self.aka not in AKA_COLORS:
            raise SchemaValueError(f"aka must be one of {AKA_COLORS} or None, got {self.aka!r}")
        if self.kind == UNKNOWN_TILE_KIND:
            raise SchemaValueError("UNKNOWN tile cannot carry an aka colour")
        if self.kind not in AKA_KINDS:
            raise SchemaValueError(f"aka is only valid on 5m/5p/5s, got kind {self.kind}")
        expected_colour = AKA_KINDS.index(self.kind)
        if self.aka != expected_colour:
            raise SchemaValueError(
                f"aka colour {self.aka} does not match kind {self.kind} "
                f"(expected {expected_colour})"
            )

    @property
    def is_unknown(self) -> bool:
        return self.kind == UNKNOWN_TILE_KIND

    @property
    def is_aka(self) -> bool:
        return self.aka is not None

    @property
    def deaka(self) -> int:
        """Physical tile kind ignoring the red-five flag."""
        return self.kind

    @property
    def text(self) -> str:
        if self.is_unknown:
            return "?"
        base = TILE_KIND_NAMES[self.kind]
        return base + "r" if self.is_aka else base

    def to_dict(self) -> dict:
        return {"kind": self.kind, "aka": self.aka}

    @classmethod
    def from_dict(cls, value) -> "TileRef":
        if not isinstance(value, Mapping):
            raise SchemaValueError(f"tile must be a mapping, got {type(value).__name__}")
        expected = {"kind", "aka"}
        if set(value) != expected:
            raise SchemaFieldError(
                f"tile fields must be exactly {sorted(expected)}, got {sorted(value)}"
            )
        return cls(kind=value["kind"], aka=value["aka"])

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text


UNKNOWN_TILE = TileRef(UNKNOWN_TILE_KIND, None)


def tile_from_text(text: str) -> TileRef:
    """Parse one MJAI tile string (``1m..9m,1p..,1s..,E,S,W,N,P,F,C,5mr,5pr,5sr,?``)."""
    if not isinstance(text, str):
        raise SchemaValueError(f"tile string must be str, got {type(text).__name__}")
    if text == "?":
        return UNKNOWN_TILE
    if text.endswith("r"):
        base = text[:-1]
        if base not in ("5m", "5p", "5s"):
            raise SchemaValueError(f"only 5m/5p/5s can be aka, got {text!r}")
        kind = _TILE_TEXT_TO_KIND[base]
        return TileRef(kind=kind, aka=AKA_KINDS.index(kind))
    if text not in _TILE_TEXT_TO_KIND:
        raise SchemaValueError(f"unknown MJAI tile {text!r}")
    return TileRef(kind=_TILE_TEXT_TO_KIND[text], aka=None)


def tile_to_text(tile: TileRef) -> str:
    if not isinstance(tile, TileRef):
        raise SchemaValueError(f"expected TileRef, got {type(tile).__name__}")
    return tile.text


def _check_delta_values(deltas, name: str) -> tuple[int, int, int, int]:
    if deltas is None:
        return None
    if not isinstance(deltas, (tuple, list)) or len(deltas) != 4:
        raise SchemaValueError(f"{name} must be a sequence of 4 ints")
    values = tuple(_check_int(value, f"{name}[{index}]")
                   for index, value in enumerate(deltas))
    return values


def rotate_deltas_to_relative(deltas, pov: int):
    """Absolute-seat MJAI score deltas -> (self, shimo, toimen, kami).

    This is the sole producer of relative deltas for schema v3.  PR-09
    ``project_event`` must use it and set ``deltas_order="relative"``; raw MJAI events
    default to ``"absolute"``.
    """
    _check_int(pov, "pov", 0, 3)
    values = _check_delta_values(deltas, "deltas")
    if values is None:
        return None
    return tuple(values[(pov + offset) % 4] for offset in range(4))


def rotate_deltas_to_absolute(deltas_rel, pov: int):
    """(self, shimo, toimen, kami) -> absolute-seat order (inverse of above)."""
    _check_int(pov, "pov", 0, 3)
    values = _check_delta_values(deltas_rel, "deltas_rel")
    if values is None:
        return None
    absolute = [0, 0, 0, 0]
    for offset, value in enumerate(values):
        absolute[(pov + offset) % 4] = value
    return tuple(absolute)


# ---------------------------------------------------------------------------
# Dataclass field dtypes / spec
# ---------------------------------------------------------------------------

_DTYPES = frozenset({
    "bool", "u8", "u16", "u32", "i32", "i32x4", "optional_i32x4",
    "deltas_order", "tile", "optional_tile", "tile4", "tile13", "tiles",
    "actor_rel",
})


def _check_int(value, name: str, minimum: int | None = None,
               maximum: int | None = None) -> int:
    if not _is_int(value):
        raise SchemaValueError(f"{name} must be int, got {value!r}")
    if minimum is not None and value < minimum:
        raise SchemaValueError(f"{name}={value} < {minimum}")
    if maximum is not None and value > maximum:
        raise SchemaValueError(f"{name}={value} > {maximum}")
    return value


def _check_actor(value, name: str, *, allow_none: bool = False) -> int:
    if allow_none and value == 4:
        return 4
    return _check_int(value, name, 0, 3)


def _check_bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaValueError(f"{name} must be bool, got {value!r}")
    return value


def _check_tile(value, name: str, *, allow_unknown: bool = True) -> TileRef:
    if not isinstance(value, TileRef):
        raise SchemaValueError(f"{name} must be TileRef, got {type(value).__name__}")
    if value.is_unknown and not allow_unknown:
        raise SchemaValueError(f"{name} must be a known tile")
    return value


def _check_tiles(value, name: str, *, length: int | None = None,
                 allow_unknown: bool = True) -> tuple[TileRef, ...]:
    if not isinstance(value, tuple):
        raise SchemaValueError(f"{name} must be a tuple of TileRef")
    if length is not None and len(value) != length:
        raise SchemaValueError(f"{name} length {len(value)} != {length}")
    for index, tile in enumerate(value):
        _check_tile(tile, f"{name}[{index}]", allow_unknown=allow_unknown)
    return value


def _check_i32x4(value, name: str, *, optional: bool = False):
    if value is None and optional:
        return None
    if not isinstance(value, tuple) or len(value) != 4:
        raise SchemaValueError(f"{name} must be a tuple of 4 ints")
    for index, item in enumerate(value):
        _check_int(item, f"{name}[{index}]")
    return value


def _require_all_unknown(tiles: tuple[TileRef, ...], start: int, name: str) -> None:
    for index in range(start, len(tiles)):
        if not tiles[index].is_unknown:
            raise SchemaValueError(f"{name}[{index}] must be UNKNOWN padding")


def _require_known_prefix(tiles: tuple[TileRef, ...], count: int, name: str) -> None:
    for index in range(count):
        if tiles[index].is_unknown:
            raise SchemaValueError(f"{name}[{index}] must be a known tile")


def _check_same_kind(tiles, kind: int, name: str) -> None:
    for index, tile in enumerate(tiles):
        if tile.is_unknown:
            raise SchemaValueError(f"{name}[{index}] must be known")
        if tile.kind != kind:
            raise SchemaValueError(
                f"{name}[{index}] kind {tile.kind} != called kind {kind}"
            )


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservableEventV3:
    """Common base; concrete subclasses add their allowlisted fields."""

    seq_idx: int
    kyoku_idx: int

    EVENT_TYPE: ClassVar[str]
    TYPE_ID: ClassVar[int]
    TIMING: ClassVar[EventTiming]
    FIELDS: ClassVar[dict[str, str]] = {}
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {}

    def __post_init__(self):
        _check_int(self.seq_idx, "seq_idx", 0)
        _check_int(self.kyoku_idx, "kyoku_idx", -1)

    @property
    def kind(self) -> EventKind:
        return EventKind(self.EVENT_TYPE)

    @property
    def timing(self) -> EventTiming:
        return self.TIMING

    @property
    def type_id(self) -> int:
        return self.TYPE_ID

    def to_dict(self) -> dict:
        out = {
            "type": self.EVENT_TYPE,
            "seq_idx": self.seq_idx,
            "kyoku_idx": self.kyoku_idx,
            "timing": self.TIMING.value,
        }
        for field in fields(self):
            out[field.name] = _encode_value(getattr(self, field.name))
        return out

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, Mapping):
            raise SchemaValueError(f"event must be a mapping, got {type(data).__name__}")
        expected = {"type", "seq_idx", "kyoku_idx", "timing"} | set(cls.FIELDS)
        actual = set(data)
        missing = expected - actual
        extra = actual - expected
        if missing:
            raise SchemaFieldError(f"missing event fields: {sorted(missing)}")
        if extra:
            raise SchemaFieldError(f"unexpected event fields: {sorted(extra)}")
        if data["type"] != cls.EVENT_TYPE:
            raise SchemaFieldError(
                f"expected type {cls.EVENT_TYPE!r}, got {data['type']!r}"
            )
        if data["timing"] != cls.TIMING.value:
            raise SchemaValueError(
                f"expected timing {cls.TIMING.value!r}, got {data['timing']!r}"
            )
        kwargs = {}
        for name, dtype in cls.FIELDS.items():
            kwargs[name] = _decode_value(data[name], dtype, name)
        return cls(seq_idx=data["seq_idx"], kyoku_idx=data["kyoku_idx"], **kwargs)


@dataclass(frozen=True)
class StartGameEvent(ObservableEventV3):
    aka_flag: bool
    kyoku_first: int

    EVENT_TYPE: ClassVar[str] = "start_game"
    TYPE_ID: ClassVar[int] = 0
    TIMING: ClassVar[EventTiming] = EventTiming.BOUNDARY
    FIELDS: ClassVar[dict[str, str]] = {"aka_flag": "bool", "kyoku_first": "u8"}
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {
        "aka_flag": VISIBILITY_PUBLIC, "kyoku_first": VISIBILITY_PUBLIC,
    }

    def __post_init__(self):
        super().__post_init__()
        _check_bool(self.aka_flag, "aka_flag")
        _check_int(self.kyoku_first, "kyoku_first", 0, 255)


@dataclass(frozen=True)
class StartKyokuEvent(ObservableEventV3):
    bakaze: TileRef
    dora_marker: TileRef
    kyoku: int
    honba: int
    kyotaku: int
    oya_rel: int
    scores_rel: tuple[int, int, int, int]
    own_haipai: tuple[TileRef, ...]

    EVENT_TYPE: ClassVar[str] = "start_kyoku"
    TYPE_ID: ClassVar[int] = 1
    TIMING: ClassVar[EventTiming] = EventTiming.BOUNDARY
    FIELDS: ClassVar[dict[str, str]] = {
        "bakaze": "tile", "dora_marker": "tile", "kyoku": "u8", "honba": "u8",
        "kyotaku": "u8", "oya_rel": "actor_rel", "scores_rel": "i32x4",
        "own_haipai": "tile13",
    }
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {
        "bakaze": VISIBILITY_PUBLIC, "dora_marker": VISIBILITY_PUBLIC,
        "kyoku": VISIBILITY_PUBLIC, "honba": VISIBILITY_PUBLIC,
        "kyotaku": VISIBILITY_PUBLIC, "oya_rel": VISIBILITY_PUBLIC,
        "scores_rel": VISIBILITY_PUBLIC, "own_haipai": VISIBILITY_SELF_ONLY,
    }

    def __post_init__(self):
        super().__post_init__()
        _check_tile(self.bakaze, "bakaze", allow_unknown=False)
        if self.bakaze.kind not in (27, 28, 29, 30):
            raise SchemaValueError(f"bakaze must be E/S/W/N, got {self.bakaze.text}")
        _check_tile(self.dora_marker, "dora_marker", allow_unknown=False)
        _check_int(self.kyoku, "kyoku", 1, 4)
        _check_int(self.honba, "honba", 0, 255)
        _check_int(self.kyotaku, "kyotaku", 0, 255)
        _check_actor(self.oya_rel, "oya_rel")
        _check_i32x4(self.scores_rel, "scores_rel")
        _check_tiles(self.own_haipai, "own_haipai", length=13, allow_unknown=False)
        for kind in {tile.kind for tile in self.own_haipai}:
            if sum(1 for tile in self.own_haipai if tile.kind == kind) > 4:
                raise SchemaValueError(f"own_haipai has more than 4 copies of kind {kind}")
        for color in AKA_COLORS:
            if sum(1 for tile in self.own_haipai if tile.aka == color) > AKA_PER_COLOR:
                raise SchemaValueError(
                    f"own_haipai has more than {AKA_PER_COLOR} aka five of colour {color}"
                )
        # start_kyoku.scores exclude the sticks still on the table (H-01).
        if sum(self.scores_rel) + KYOTAKU_POINTS * self.kyotaku != 4 * INITIAL_POINTS:
            raise SchemaValueError(
                "start_kyoku scores+kyotaku violate the 100000 total invariant: "
                f"{sum(self.scores_rel)} + {KYOTAKU_POINTS * self.kyotaku}"
            )


@dataclass(frozen=True)
class TsumoEvent(ObservableEventV3):
    actor_rel: int
    tile: TileRef

    EVENT_TYPE: ClassVar[str] = "tsumo"
    TYPE_ID: ClassVar[int] = 2
    TIMING: ClassVar[EventTiming] = EventTiming.TURN
    FIELDS: ClassVar[dict[str, str]] = {"actor_rel": "actor_rel", "tile": "tile"}
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {
        "actor_rel": VISIBILITY_PUBLIC, "tile": VISIBILITY_SELF_ONLY,
    }

    def __post_init__(self):
        super().__post_init__()
        _check_actor(self.actor_rel, "actor_rel")
        _check_tile(self.tile, "tile", allow_unknown=True)


@dataclass(frozen=True)
class DahaiEvent(ObservableEventV3):
    actor_rel: int
    tile: TileRef
    tsumogiri: bool

    EVENT_TYPE: ClassVar[str] = "dahai"
    TYPE_ID: ClassVar[int] = 3
    TIMING: ClassVar[EventTiming] = EventTiming.TURN
    FIELDS: ClassVar[dict[str, str]] = {
        "actor_rel": "actor_rel", "tile": "tile", "tsumogiri": "bool",
    }
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {
        "actor_rel": VISIBILITY_PUBLIC, "tile": VISIBILITY_PUBLIC,
        "tsumogiri": VISIBILITY_PUBLIC,
    }

    def __post_init__(self):
        super().__post_init__()
        _check_actor(self.actor_rel, "actor_rel")
        _check_tile(self.tile, "tile", allow_unknown=False)
        _check_bool(self.tsumogiri, "tsumogiri")


@dataclass(frozen=True)
class ChiEvent(ObservableEventV3):
    actor_rel: int
    target_rel: int
    tile: TileRef
    consumed: tuple[TileRef, TileRef, TileRef, TileRef]

    EVENT_TYPE: ClassVar[str] = "chi"
    TYPE_ID: ClassVar[int] = 4
    TIMING: ClassVar[EventTiming] = EventTiming.RESPONSE
    FIELDS: ClassVar[dict[str, str]] = {
        "actor_rel": "actor_rel", "target_rel": "actor_rel", "tile": "tile",
        "consumed": "tile4",
    }
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {
        "actor_rel": VISIBILITY_PUBLIC, "target_rel": VISIBILITY_PUBLIC,
        "tile": VISIBILITY_PUBLIC, "consumed": VISIBILITY_PUBLIC,
    }

    def __post_init__(self):
        super().__post_init__()
        _check_actor(self.actor_rel, "actor_rel")
        _check_actor(self.target_rel, "target_rel")
        expected_target = (self.actor_rel - 1) % 4
        if self.target_rel != expected_target:
            raise SchemaValueError(
                f"chi must come from kamicha: target {self.target_rel} != {expected_target}"
            )
        _check_tile(self.tile, "tile", allow_unknown=False)
        _check_tiles(self.consumed, "consumed", length=4, allow_unknown=True)
        _require_known_prefix(self.consumed, 2, "consumed")
        _require_all_unknown(self.consumed, 2, "consumed")
        suits = {tile.kind // 9 for tile in self.consumed[:2]}
        tile_suit = self.tile.kind // 9
        if (3 in suits or tile_suit == 3 or len(suits) != 1
                or tile_suit not in suits):
            raise SchemaValueError("chi tiles must be three consecutive tiles of one suit")
        ranks = sorted([self.tile.kind] + [tile.kind for tile in self.consumed[:2]])
        if ranks[1] - ranks[0] != 1 or ranks[2] - ranks[1] != 1:
            raise SchemaValueError(f"chi tiles are not consecutive: {ranks}")


@dataclass(frozen=True)
class PonEvent(ObservableEventV3):
    actor_rel: int
    target_rel: int
    tile: TileRef
    consumed: tuple[TileRef, TileRef, TileRef, TileRef]

    EVENT_TYPE: ClassVar[str] = "pon"
    TYPE_ID: ClassVar[int] = 5
    TIMING: ClassVar[EventTiming] = EventTiming.RESPONSE
    FIELDS: ClassVar[dict[str, str]] = {
        "actor_rel": "actor_rel", "target_rel": "actor_rel", "tile": "tile",
        "consumed": "tile4",
    }
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {
        "actor_rel": VISIBILITY_PUBLIC, "target_rel": VISIBILITY_PUBLIC,
        "tile": VISIBILITY_PUBLIC, "consumed": VISIBILITY_PUBLIC,
    }

    def __post_init__(self):
        super().__post_init__()
        _check_actor(self.actor_rel, "actor_rel")
        _check_actor(self.target_rel, "target_rel")
        if self.actor_rel == self.target_rel:
            raise SchemaValueError("pon cannot target self")
        _check_tile(self.tile, "tile", allow_unknown=False)
        _check_tiles(self.consumed, "consumed", length=4, allow_unknown=True)
        _require_known_prefix(self.consumed, 2, "consumed")
        _require_all_unknown(self.consumed, 2, "consumed")
        _check_same_kind(self.consumed[:2], self.tile.kind, "consumed")


@dataclass(frozen=True)
class DaiminkanEvent(ObservableEventV3):
    actor_rel: int
    target_rel: int
    tile: TileRef
    consumed: tuple[TileRef, TileRef, TileRef, TileRef]

    EVENT_TYPE: ClassVar[str] = "daiminkan"
    TYPE_ID: ClassVar[int] = 6
    TIMING: ClassVar[EventTiming] = EventTiming.RESPONSE
    FIELDS: ClassVar[dict[str, str]] = {
        "actor_rel": "actor_rel", "target_rel": "actor_rel", "tile": "tile",
        "consumed": "tile4",
    }
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {
        "actor_rel": VISIBILITY_PUBLIC, "target_rel": VISIBILITY_PUBLIC,
        "tile": VISIBILITY_PUBLIC, "consumed": VISIBILITY_PUBLIC,
    }

    def __post_init__(self):
        super().__post_init__()
        _check_actor(self.actor_rel, "actor_rel")
        _check_actor(self.target_rel, "target_rel")
        if self.actor_rel == self.target_rel:
            raise SchemaValueError("daiminkan cannot target self")
        _check_tile(self.tile, "tile", allow_unknown=False)
        _check_tiles(self.consumed, "consumed", length=4, allow_unknown=True)
        _require_known_prefix(self.consumed, 3, "consumed")
        _require_all_unknown(self.consumed, 3, "consumed")
        _check_same_kind(self.consumed[:3], self.tile.kind, "consumed")


@dataclass(frozen=True)
class KakanEvent(ObservableEventV3):
    actor_rel: int
    tile: TileRef
    consumed: tuple[TileRef, TileRef, TileRef, TileRef]

    EVENT_TYPE: ClassVar[str] = "kakan"
    TYPE_ID: ClassVar[int] = 7
    TIMING: ClassVar[EventTiming] = EventTiming.TURN
    FIELDS: ClassVar[dict[str, str]] = {
        "actor_rel": "actor_rel", "tile": "tile", "consumed": "tile4",
    }
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {
        "actor_rel": VISIBILITY_PUBLIC, "tile": VISIBILITY_PUBLIC,
        "consumed": VISIBILITY_PUBLIC,
    }

    def __post_init__(self):
        super().__post_init__()
        _check_actor(self.actor_rel, "actor_rel")
        _check_tile(self.tile, "tile", allow_unknown=False)
        _check_tiles(self.consumed, "consumed", length=4, allow_unknown=True)
        _require_known_prefix(self.consumed, 3, "consumed")
        _require_all_unknown(self.consumed, 3, "consumed")
        _check_same_kind(self.consumed[:3], self.tile.kind, "consumed")


@dataclass(frozen=True)
class AnkanEvent(ObservableEventV3):
    actor_rel: int
    consumed: tuple[TileRef, TileRef, TileRef, TileRef]

    EVENT_TYPE: ClassVar[str] = "ankan"
    TYPE_ID: ClassVar[int] = 8
    TIMING: ClassVar[EventTiming] = EventTiming.TURN
    FIELDS: ClassVar[dict[str, str]] = {
        "actor_rel": "actor_rel", "consumed": "tile4",
    }
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {
        "actor_rel": VISIBILITY_PUBLIC, "consumed": VISIBILITY_RULE_DEPENDENT,
    }

    def __post_init__(self):
        super().__post_init__()
        _check_actor(self.actor_rel, "actor_rel")
        _check_tiles(self.consumed, "consumed", length=4, allow_unknown=True)
        known = [tile for tile in self.consumed if not tile.is_unknown]
        if known and len(known) != 4:
            raise SchemaValueError(
                "ankan consumed must be either four known tiles or four UNKNOWN slots"
            )
        if known:
            _check_same_kind(self.consumed, self.consumed[0].kind, "consumed")


@dataclass(frozen=True)
class ReachEvent(ObservableEventV3):
    actor_rel: int

    EVENT_TYPE: ClassVar[str] = "reach"
    TYPE_ID: ClassVar[int] = 9
    TIMING: ClassVar[EventTiming] = EventTiming.TURN
    FIELDS: ClassVar[dict[str, str]] = {"actor_rel": "actor_rel"}
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {"actor_rel": VISIBILITY_PUBLIC}

    def __post_init__(self):
        super().__post_init__()
        _check_actor(self.actor_rel, "actor_rel")


@dataclass(frozen=True)
class ReachAcceptedEvent(ObservableEventV3):
    actor_rel: int

    EVENT_TYPE: ClassVar[str] = "reach_accepted"
    TYPE_ID: ClassVar[int] = 10
    TIMING: ClassVar[EventTiming] = EventTiming.TURN
    FIELDS: ClassVar[dict[str, str]] = {"actor_rel": "actor_rel"}
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {"actor_rel": VISIBILITY_PUBLIC}

    def __post_init__(self):
        super().__post_init__()
        _check_actor(self.actor_rel, "actor_rel")


@dataclass(frozen=True)
class DoraEvent(ObservableEventV3):
    dora_marker: TileRef

    EVENT_TYPE: ClassVar[str] = "dora"
    TYPE_ID: ClassVar[int] = 11
    TIMING: ClassVar[EventTiming] = EventTiming.TURN
    FIELDS: ClassVar[dict[str, str]] = {"dora_marker": "tile"}
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {"dora_marker": VISIBILITY_PUBLIC}

    def __post_init__(self):
        super().__post_init__()
        _check_tile(self.dora_marker, "dora_marker", allow_unknown=False)


@dataclass(frozen=True)
class HoraEvent(ObservableEventV3):
    actor_rel: int
    target_rel: int
    deltas: tuple[int, int, int, int]
    ura_markers: tuple[TileRef, ...]
    deltas_order: str = DELTAS_ORDER_ABSOLUTE

    EVENT_TYPE: ClassVar[str] = "hora"
    TYPE_ID: ClassVar[int] = 12
    TIMING: ClassVar[EventTiming] = EventTiming.SETTLEMENT
    FIELDS: ClassVar[dict[str, str]] = {
        "actor_rel": "actor_rel", "target_rel": "actor_rel",
        "deltas": "i32x4", "ura_markers": "tiles", "deltas_order": "deltas_order",
    }
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {
        "actor_rel": VISIBILITY_PUBLIC, "target_rel": VISIBILITY_PUBLIC,
        "deltas": VISIBILITY_AFTER_EVENT, "ura_markers": VISIBILITY_AFTER_EVENT,
        "deltas_order": VISIBILITY_PUBLIC,
    }

    def __post_init__(self):
        super().__post_init__()
        _check_actor(self.actor_rel, "actor_rel")
        _check_actor(self.target_rel, "target_rel")
        _check_i32x4(self.deltas, "deltas", optional=False)
        _check_tiles(self.ura_markers, "ura_markers", allow_unknown=False)
        if len(self.ura_markers) > 5:
            raise SchemaValueError("ura_markers has more than 5 tiles")
        if self.deltas_order not in DELTAS_ORDERS:
            raise SchemaValueError(
                f"deltas_order must be one of {DELTAS_ORDERS}, got {self.deltas_order!r}"
            )


@dataclass(frozen=True)
class RyukyokuEvent(ObservableEventV3):
    deltas: tuple[int, int, int, int]
    deltas_order: str = DELTAS_ORDER_ABSOLUTE

    EVENT_TYPE: ClassVar[str] = "ryukyoku"
    TYPE_ID: ClassVar[int] = 13
    TIMING: ClassVar[EventTiming] = EventTiming.SETTLEMENT
    FIELDS: ClassVar[dict[str, str]] = {
        "deltas": "i32x4", "deltas_order": "deltas_order",
    }
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {
        "deltas": VISIBILITY_AFTER_EVENT, "deltas_order": VISIBILITY_PUBLIC,
    }

    def __post_init__(self):
        super().__post_init__()
        _check_i32x4(self.deltas, "deltas", optional=False)
        if self.deltas_order not in DELTAS_ORDERS:
            raise SchemaValueError(
                f"deltas_order must be one of {DELTAS_ORDERS}, got {self.deltas_order!r}"
            )


@dataclass(frozen=True)
class EndKyokuEvent(ObservableEventV3):
    EVENT_TYPE: ClassVar[str] = "end_kyoku"
    TYPE_ID: ClassVar[int] = 14
    TIMING: ClassVar[EventTiming] = EventTiming.BOUNDARY
    FIELDS: ClassVar[dict[str, str]] = {}
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {}


@dataclass(frozen=True)
class EndGameEvent(ObservableEventV3):
    EVENT_TYPE: ClassVar[str] = "end_game"
    TYPE_ID: ClassVar[int] = 15
    TIMING: ClassVar[EventTiming] = EventTiming.BOUNDARY
    FIELDS: ClassVar[dict[str, str]] = {}
    FIELD_VISIBILITY: ClassVar[dict[str, str]] = {}


EVENT_CLASSES: dict[str, type[ObservableEventV3]] = {
    cls.EVENT_TYPE: cls for cls in (
        StartGameEvent, StartKyokuEvent, TsumoEvent, DahaiEvent, ChiEvent, PonEvent,
        DaiminkanEvent, KakanEvent, AnkanEvent, ReachEvent, ReachAcceptedEvent,
        DoraEvent, HoraEvent, RyukyokuEvent, EndKyokuEvent, EndGameEvent,
    )
}
EVENT_TYPE_IDS: dict[str, int] = {
    name: cls.TYPE_ID for name, cls in EVENT_CLASSES.items()
}
if sorted(EVENT_TYPE_IDS.values()) != list(range(len(EVENT_CLASSES))):
    raise AssertionError("event type ids must be a stable 0..15 range")
EVENT_KINDS: tuple[EventKind, ...] = tuple(EventKind(name) for name in EVENT_CLASSES)

COMMON_FIELD_SPECS: tuple[tuple[str, str, bool, str], ...] = (
    ("type", "event_kind", True, VISIBILITY_PUBLIC),
    ("seq_idx", "u32", True, VISIBILITY_PUBLIC),
    ("kyoku_idx", "i32", True, VISIBILITY_PUBLIC),
    ("timing", "event_timing", True, VISIBILITY_PUBLIC),
)


def _encode_value(value):
    if isinstance(value, TileRef):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_encode_value(item) for item in value]
    return value


def _decode_value(value, dtype: str, name: str):
    if dtype not in _DTYPES:
        raise AssertionError(f"unknown dtype {dtype!r} for {name}")
    if dtype == "bool":
        return _check_bool(value, name)
    if dtype in ("u8", "u16", "u32"):
        maximum = {"u8": 255, "u16": 65535, "u32": 2**32 - 1}[dtype]
        return _check_int(value, name, 0, maximum)
    if dtype == "i32":
        return _check_int(value, name, -(2**31), 2**31 - 1)
    if dtype == "actor_rel":
        return _check_actor(value, name, allow_none=False)
    if dtype == "deltas_order":
        if value not in DELTAS_ORDERS:
            raise SchemaValueError(
                f"{name} must be one of {DELTAS_ORDERS}, got {value!r}"
            )
        return value
    if dtype == "tile":
        return TileRef.from_dict(value)
    if dtype == "optional_tile":
        return None if value is None else TileRef.from_dict(value)
    if dtype in ("tile4", "tile13"):
        length = 4 if dtype == "tile4" else 13
        if not isinstance(value, list) or len(value) != length:
            raise SchemaValueError(f"{name} must be a list of {length} tiles")
        return tuple(TileRef.from_dict(item) for item in value)
    if dtype == "tiles":
        if not isinstance(value, list):
            raise SchemaValueError(f"{name} must be a list of tiles")
        return tuple(TileRef.from_dict(item) for item in value)
    if dtype == "i32x4":
        if not isinstance(value, list) or len(value) != 4:
            raise SchemaValueError(f"{name} must be a list of 4 ints")
        return tuple(_check_int(item, f"{name}[{index}]")
                     for index, item in enumerate(value))
    if dtype == "optional_i32x4":
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != 4:
            raise SchemaValueError(f"{name} must be null or a list of 4 ints")
        return tuple(_check_int(item, f"{name}[{index}]")
                     for index, item in enumerate(value))
    raise AssertionError(f"unhandled dtype {dtype!r}")  # pragma: no cover


def event_type_specs() -> dict:
    """Machine-readable event allowlist mirrored in ``schema_v3.json``."""
    out = {}
    for type_name, cls in EVENT_CLASSES.items():
        out[type_name] = {
            "type_id": cls.TYPE_ID,
            "timing": cls.TIMING.value,
            "fields": [
                {
                    "name": name,
                    "dtype": dtype,
                    "required": True,
                    "visibility": cls.FIELD_VISIBILITY[name],
                }
                for name, dtype in cls.FIELDS.items()
            ],
        }
    return out


def event_from_dict(data) -> ObservableEventV3:
    if not isinstance(data, Mapping):
        raise SchemaValueError(f"event must be a mapping, got {type(data).__name__}")
    type_name = data.get("type")
    if type_name not in EVENT_CLASSES:
        raise UnknownEventType(f"unknown ObservableEventV3 type {type_name!r}")
    return EVENT_CLASSES[type_name].from_dict(data)


def event_to_dict(event: ObservableEventV3) -> dict:
    if not isinstance(event, ObservableEventV3):
        raise SchemaValueError(f"expected ObservableEventV3, got {type(event).__name__}")
    return event.to_dict()


# ---------------------------------------------------------------------------
# MJAI -> POV-safe event construction
# ---------------------------------------------------------------------------


def _require(raw: Mapping, key: str, type_name: str):
    if key not in raw:
        raise SchemaValueError(f"{type_name} event is missing required field {key!r}")
    return raw[key]


def _raw_actor(raw: Mapping, key: str, type_name: str) -> int:
    value = _require(raw, key, type_name)
    if not _is_int(value):
        raise SchemaValueError(f"{type_name}.{key} must be int, got {value!r}")
    return value


def _rel(absolute_actor: int, pov: int) -> int:
    if not _is_int(absolute_actor) or not 0 <= absolute_actor < 4:
        raise SchemaValueError(f"absolute actor {absolute_actor!r} outside 0..3")
    if not _is_int(pov) or not 0 <= pov < 4:
        raise SchemaValueError(f"pov {pov!r} outside 0..3")
    return (absolute_actor - pov) % 4


def _tile(raw: Mapping, key: str, type_name: str) -> TileRef:
    return tile_from_text(_require(raw, key, type_name))


def _consumed_tiles(raw: Mapping, type_name: str, count: int) -> tuple[TileRef, ...]:
    consumed = _require(raw, "consumed", type_name)
    if not isinstance(consumed, list) or len(consumed) != count:
        raise SchemaValueError(
            f"{type_name}.consumed must be a list of {count} tiles"
        )
    tiles = [tile_from_text(value) for value in consumed]
    return tuple(tiles + [UNKNOWN_TILE] * (4 - count))


def event_from_mjai(raw: Mapping, *, pov: int, seq_idx: int, kyoku_idx: int,
                    reveal_opponent_ankan: bool = False) -> ObservableEventV3:
    """Project one full-info MJAI event into a POV-safe v3 event.

    ``raw`` is read through an explicit per-type allowlist; extra MJAI fields (names,
    account ids, wall seeds, opponent hands, ...) are never stored.  Unknown types raise.
    """
    if not isinstance(raw, Mapping):
        raise SchemaValueError(f"raw event must be a mapping, got {type(raw).__name__}")
    if not _is_int(pov) or not 0 <= pov < 4:
        raise SchemaValueError(f"pov {pov!r} outside 0..3")
    type_name = raw.get("type")
    if not isinstance(type_name, str):
        raise SchemaValueError(f"raw event type must be str, got {type_name!r}")
    if type_name not in EVENT_CLASSES:
        raise UnknownEventType(f"unknown MJAI event type {type_name!r}")

    common = {"seq_idx": seq_idx, "kyoku_idx": kyoku_idx}

    if type_name == "start_game":
        aka_flag = raw.get("aka_flag", True)
        _check_bool(aka_flag, "start_game.aka_flag")
        kyoku_first = raw.get("kyoku_first", 0)
        if kyoku_first is None:
            kyoku_first = 0
        _check_int(kyoku_first, "start_game.kyoku_first", 0, 255)
        return StartGameEvent(aka_flag=aka_flag, kyoku_first=kyoku_first, **common)

    if type_name == "start_kyoku":
        bakaze = _tile(raw, "bakaze", type_name)
        if bakaze.kind not in (27, 28, 29, 30):
            raise SchemaValueError(f"start_kyoku.bakaze must be E/S/W/N, got {bakaze.text}")
        dora_marker = _tile(raw, "dora_marker", type_name)
        if dora_marker.is_unknown:
            raise SchemaValueError("start_kyoku.dora_marker must be known")
        kyoku = _check_int(_require(raw, "kyoku", type_name), "start_kyoku.kyoku", 1, 4)
        honba = _check_int(_require(raw, "honba", type_name), "start_kyoku.honba", 0, 255)
        kyotaku = _check_int(_require(raw, "kyotaku", type_name),
                             "start_kyoku.kyotaku", 0, 255)
        oya = _raw_actor(raw, "oya", type_name)
        oya_rel = _rel(oya, pov)
        raw_scores = _require(raw, "scores", type_name)
        if not isinstance(raw_scores, list) or len(raw_scores) != 4:
            raise SchemaValueError("start_kyoku.scores must be a list of 4 ints")
        scores = tuple(_check_int(value, f"start_kyoku.scores[{index}]")
                       for index, value in enumerate(raw_scores))
        scores_rel = tuple(scores[(pov + offset) % 4] for offset in range(4))
        raw_tehais = _require(raw, "tehais", type_name)
        if not isinstance(raw_tehais, list) or len(raw_tehais) != 4:
            raise SchemaValueError("start_kyoku.tehais must contain 4 hands")
        if not isinstance(raw_tehais[pov], list) or len(raw_tehais[pov]) != 13:
            raise SchemaValueError("start_kyoku own hand must contain 13 tiles")
        own_haipai = tuple(tile_from_text(value) for value in raw_tehais[pov])
        if any(tile.is_unknown for tile in own_haipai):
            raise SchemaValueError("start_kyoku own haipai must be known")
        return StartKyokuEvent(
            bakaze=bakaze, dora_marker=dora_marker, kyoku=kyoku, honba=honba,
            kyotaku=kyotaku, oya_rel=oya_rel, scores_rel=scores_rel,
            own_haipai=own_haipai, **common,
        )

    if type_name == "tsumo":
        actor = _raw_actor(raw, "actor", type_name)
        tile = _tile(raw, "pai", type_name) if actor == pov else UNKNOWN_TILE
        return TsumoEvent(actor_rel=_rel(actor, pov), tile=tile, **common)

    if type_name == "dahai":
        actor = _raw_actor(raw, "actor", type_name)
        tile = _tile(raw, "pai", type_name)
        if tile.is_unknown:
            raise SchemaValueError("dahai.pai must be a known public tile")
        tsumogiri = raw.get("tsumogiri", False)
        _check_bool(tsumogiri, "dahai.tsumogiri")
        return DahaiEvent(actor_rel=_rel(actor, pov), tile=tile,
                          tsumogiri=tsumogiri, **common)

    if type_name == "chi":
        actor = _raw_actor(raw, "actor", type_name)
        target = _raw_actor(raw, "target", type_name)
        tile = _tile(raw, "pai", type_name)
        consumed = _consumed_tiles(raw, type_name, 2)
        return ChiEvent(actor_rel=_rel(actor, pov), target_rel=_rel(target, pov),
                        tile=tile, consumed=consumed, **common)

    if type_name == "pon":
        actor = _raw_actor(raw, "actor", type_name)
        target = _raw_actor(raw, "target", type_name)
        tile = _tile(raw, "pai", type_name)
        consumed = _consumed_tiles(raw, type_name, 2)
        return PonEvent(actor_rel=_rel(actor, pov), target_rel=_rel(target, pov),
                        tile=tile, consumed=consumed, **common)

    if type_name == "daiminkan":
        actor = _raw_actor(raw, "actor", type_name)
        target = _raw_actor(raw, "target", type_name)
        tile = _tile(raw, "pai", type_name)
        consumed = _consumed_tiles(raw, type_name, 3)
        return DaiminkanEvent(actor_rel=_rel(actor, pov), target_rel=_rel(target, pov),
                              tile=tile, consumed=consumed, **common)

    if type_name == "kakan":
        actor = _raw_actor(raw, "actor", type_name)
        tile = _tile(raw, "pai", type_name)
        consumed = _consumed_tiles(raw, type_name, 3)
        return KakanEvent(actor_rel=_rel(actor, pov), tile=tile,
                          consumed=consumed, **common)

    if type_name == "ankan":
        actor = _raw_actor(raw, "actor", type_name)
        consumed = _consumed_tiles(raw, type_name, 4)
        if actor != pov and not reveal_opponent_ankan:
            consumed = (UNKNOWN_TILE, UNKNOWN_TILE, UNKNOWN_TILE, UNKNOWN_TILE)
        elif any(tile.is_unknown for tile in consumed):
            consumed = (UNKNOWN_TILE, UNKNOWN_TILE, UNKNOWN_TILE, UNKNOWN_TILE)
        return AnkanEvent(actor_rel=_rel(actor, pov), consumed=consumed, **common)

    if type_name in ("reach", "reach_accepted"):
        actor = _raw_actor(raw, "actor", type_name)
        cls = ReachEvent if type_name == "reach" else ReachAcceptedEvent
        return cls(actor_rel=_rel(actor, pov), **common)

    if type_name == "dora":
        marker = _tile(raw, "dora_marker", type_name)
        return DoraEvent(dora_marker=marker, **common)

    if type_name == "hora":
        actor = _raw_actor(raw, "actor", type_name)
        target = _raw_actor(raw, "target", type_name)
        deltas_raw = raw.get("deltas")
        if not isinstance(deltas_raw, list) or len(deltas_raw) != 4:
            raise SchemaValueError("hora.deltas must be a list of 4 ints (H-01)")
        deltas = tuple(_check_int(value, f"hora.deltas[{index}]")
                       for index, value in enumerate(deltas_raw))
        ura_raw = raw.get("ura_markers") or []
        if not isinstance(ura_raw, list):
            raise SchemaValueError("hora.ura_markers must be a list")
        ura = tuple(tile_from_text(value) for value in ura_raw)
        return HoraEvent(actor_rel=_rel(actor, pov), target_rel=_rel(target, pov),
                         deltas=deltas, ura_markers=ura, **common)

    if type_name == "ryukyoku":
        deltas_raw = raw.get("deltas")
        if not isinstance(deltas_raw, list) or len(deltas_raw) != 4:
            raise SchemaValueError("ryukyoku.deltas must be a list of 4 ints (H-01)")
        deltas = tuple(_check_int(value, f"ryukyoku.deltas[{index}]")
                       for index, value in enumerate(deltas_raw))
        return RyukyokuEvent(deltas=deltas, **common)

    if type_name == "end_kyoku":
        return EndKyokuEvent(**common)
    if type_name == "end_game":
        return EndGameEvent(**common)

    raise UnknownEventType(f"unknown MJAI event type {type_name!r}")  # pragma: no cover


def events_from_mjai(raw_events, *, pov: int, reveal_opponent_ankan: bool = False
                     ) -> tuple[ObservableEventV3, ...]:
    """Project a whole full-info MJAI event list; assigns ``seq_idx``/``kyoku_idx``.

    ``kyoku_idx`` is 0-based (native ``at_kyoku`` convention); ``start_game`` is -1.
    """
    if not isinstance(pov, int) or isinstance(pov, bool) or not 0 <= pov < 4:
        raise SchemaValueError(f"pov {pov!r} outside 0..3")
    raw_events = list(raw_events)
    out: list[ObservableEventV3] = []
    kyoku_idx = -1
    for seq_idx, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise SchemaValueError(f"raw event {seq_idx} is not a mapping")
        type_name = raw.get("type")
        if type_name not in EVENT_CLASSES:
            raise UnknownEventType(f"unknown MJAI event type {type_name!r}")
        if type_name == "start_kyoku":
            kyoku_idx += 1
        elif type_name != "start_game" and kyoku_idx < 0:
            raise SchemaValueError(
                f"event {type_name!r} at seq {seq_idx} appears before start_kyoku"
            )
        out.append(event_from_mjai(
            raw, pov=pov, seq_idx=seq_idx, kyoku_idx=kyoku_idx,
            reveal_opponent_ankan=reveal_opponent_ankan,
        ))
    return tuple(out)


__all__ = [
    "AnkanEvent", "ChiEvent", "COMMON_FIELD_SPECS", "DahaiEvent", "DaiminkanEvent",
    "DoraEvent", "EndGameEvent", "EndKyokuEvent", "EVENT_CLASSES",
    "EVENT_KINDS", "EVENT_TYPE_IDS", "EventKind", "EventSchemaError", "EventTiming",
    "HoraEvent", "INITIAL_POINTS", "KakanEvent", "KYOTAKU_POINTS", "ObservableEventV3",
    "DELTAS_ORDERS", "DELTAS_ORDER_ABSOLUTE", "DELTAS_ORDER_RELATIVE",
    "PonEvent", "ReachAcceptedEvent", "ReachEvent", "RyukyokuEvent", "SCHEMA_VERSION",
    "SchemaFieldError", "SchemaValueError", "StartGameEvent", "StartKyokuEvent",
    "TileRef", "TsumoEvent", "UNKNOWN_TILE", "UNKNOWN_TILE_KIND", "UnknownEventType",
    "VISIBILITIES", "event_from_dict", "event_from_mjai", "event_to_dict",
    "event_type_specs", "events_from_mjai", "rotate_deltas_to_absolute",
    "rotate_deltas_to_relative", "tile_from_text", "tile_to_text",
]
