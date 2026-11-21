"""Forward-input types for schema v3 (PR-08).

``DecisionInput`` is the **only** object a forward pass may consume.  The label side
(``DecisionTarget``) lives in :mod:`riichiformer.schemas.target_v3`; this module must not
import it, and the eventual runtime encoder must not import it either (tested by a
subprocess import-isolation test and by ``DecisionInput.to_forward_dict()`` refusing any
target field).

Visibility contract:

* ``events`` contains only already-visible history; every event's ``seq_idx`` is strictly
  less than ``before_event_idx`` (the event at that index is the action being decided);
* ``tile_state`` counts physical tiles once: own hand, river + melds + revealed dora
  indicators, with aka tracked separately;
* ``players_public`` has exactly four relative seats and no hidden hand;
* no names, account ids, wall seeds, future events, future ura, or target labels.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Mapping

from .action_v3 import (
    AKA_KINDS,
    AKA_PER_COLOR,
    ACTION_SPACE,
    ActionPhase,
    ActionSchemaError,
    SCHEMA_VERSION,
    TILE_KIND_COUNT,
    validate_mask_for_phase,
)
from .event_v3 import (
    EVENT_CLASSES,
    INITIAL_POINTS,
    KYOTAKU_POINTS,
    ObservableEventV3,
    SchemaValueError,
    TileRef,
)

TOTAL_POINTS = 4 * INITIAL_POINTS


class DecisionSchemaError(SchemaValueError):
    """A DecisionInput / visible-state violation."""


class TargetLeakError(DecisionSchemaError):
    """A target-only field tried to enter a forward input."""


class RulesetId(str, Enum):
    TENHOU_4P_RANKED = "tenhou_4p_ranked"
    MORTAL_ARENA_PINNED = "mortal_arena_pinned"


class SourceKind(str, Enum):
    HUMAN_LOG = "human_log"
    ARENA_SELFPLAY = "arena_selfplay"
    SYNTHETIC = "synthetic"


class RulesetDeviation(str, Enum):
    """H-04 deviations that must travel with a sample (see FEATURE_SCHEMA_V3.md)."""

    R02 = "R02"  # triple ron settled by the arena instead of Tenhou abortive draw
    R25 = "R25"  # last-dealer agari stop compares kyotaku-inclusive scores
    R26 = "R26"  # four-riichi abort happens after one extra draw (log-level)


RULESET_DEVIATIONS: dict[RulesetDeviation, str] = {
    RulesetDeviation.R02: "triple_ron_settled_instead_of_tenhou_abortive_draw",
    RulesetDeviation.R25: "last_dealer_stop_uses_kyotaku_inclusive_scores",
    RulesetDeviation.R26: "four_riichi_abort_after_one_extra_draw",
}


class PublicZone(str, Enum):
    RIVER = "river"
    MELD = "meld"
    DORA_INDICATOR = "dora_indicator"


PUBLIC_ZONE_ORDER: tuple[PublicZone, ...] = (
    PublicZone.RIVER, PublicZone.MELD, PublicZone.DORA_INDICATOR,
)


class RiichiState(str, Enum):
    NONE = "none"
    DECLARED = "declared"
    ACCEPTED = "accepted"


class MeldKind(str, Enum):
    CHI = "chi"
    PON = "pon"
    DAIMINKAN = "daiminkan"
    KAKAN = "kakan"
    ANKAN = "ankan"


MELD_TILE_COUNTS: dict[MeldKind, int] = {
    MeldKind.CHI: 3,
    MeldKind.PON: 3,
    MeldKind.DAIMINKAN: 4,
    MeldKind.KAKAN: 4,
    MeldKind.ANKAN: 4,
}

#: Fields that must never appear anywhere inside a forward input.
_ALLOWED_EVENT_CLASSES = frozenset(EVENT_CLASSES.values())

FORBIDDEN_FORWARD_KEYS: frozenset[str] = frozenset({
    "expert_action", "supervision_valid", "censor_reason", "rank_final",
    "oracle_labels", "target", "targets", "names", "player_names", "account_id",
    "wall_seed", "seed", "log_url",
})

#: Exact top-level keys of ``DecisionInput.to_forward_dict()``.
DECISION_INPUT_FIELDS: tuple[str, ...] = (
    "schema_version", "events", "tile_state", "players_public", "global_public",
    "legal_mask", "phase", "before_event_idx",
)

DECISION_TARGET_FIELDS: tuple[str, ...] = (
    "schema_version", "expert_action", "supervision_valid", "censor_reason",
    "rank_final", "oracle_labels",
)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_int(value, name: str, minimum: int | None = None,
               maximum: int | None = None) -> int:
    if not _is_int(value):
        raise DecisionSchemaError(f"{name} must be int, got {value!r}")
    if minimum is not None and value < minimum:
        raise DecisionSchemaError(f"{name}={value} < {minimum}")
    if maximum is not None and value > maximum:
        raise DecisionSchemaError(f"{name}={value} > {maximum}")
    return value


def _check_counts(value, name: str, length: int = TILE_KIND_COUNT) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) != length:
        raise DecisionSchemaError(f"{name} must be a tuple of {length} ints")
    for index, count in enumerate(value):
        _check_int(count, f"{name}[{index}]", 0, 4)
    return value


def _check_zone_counts(value, name: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise DecisionSchemaError(f"{name} must be a tuple of 3 zones")
    return tuple(_check_counts(zone, f"{name}[{index}]")
                 for index, zone in enumerate(value))


def _plain_tile(tile: TileRef) -> dict:
    return tile.to_dict()


def _plain_tiles(tiles) -> list[dict]:
    return [tile.to_dict() for tile in tiles]


@dataclass(frozen=True)
class VisibleTileState:
    """Physical tile ledger visible to one POV at one decision point."""

    own_counts: tuple[int, ...]
    own_aka: tuple[int, int, int]
    public_counts: tuple[int, ...]
    public_aka: tuple[int, int, int]
    public_counts_by_zone: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    dora_indicators: tuple[TileRef, ...]
    ruleset_id: RulesetId

    def __post_init__(self):
        _check_counts(self.own_counts, "own_counts")
        _check_counts(self.public_counts, "public_counts")
        if not isinstance(self.own_aka, tuple) or len(self.own_aka) != 3:
            raise DecisionSchemaError("own_aka must be a tuple of 3 ints")
        if not isinstance(self.public_aka, tuple) or len(self.public_aka) != 3:
            raise DecisionSchemaError("public_aka must be a tuple of 3 ints")
        for color, count in enumerate(self.own_aka):
            _check_int(count, f"own_aka[{color}]", 0, AKA_PER_COLOR)
        for color, count in enumerate(self.public_aka):
            _check_int(count, f"public_aka[{color}]", 0, AKA_PER_COLOR)
        zones = _check_zone_counts(self.public_counts_by_zone, "public_counts_by_zone")
        if not isinstance(self.ruleset_id, RulesetId):
            raise DecisionSchemaError(
                f"ruleset_id must be RulesetId, got {self.ruleset_id!r}"
            )
        if not isinstance(self.dora_indicators, tuple):
            raise DecisionSchemaError("dora_indicators must be a tuple of TileRef")
        if len(self.dora_indicators) > 5:
            raise DecisionSchemaError("at most 5 dora indicators are visible")

        for kind in range(TILE_KIND_COUNT):
            if self.own_counts[kind] + self.public_counts[kind] > 4:
                raise DecisionSchemaError(
                    f"physical tile overflow for kind {kind}: "
                    f"own={self.own_counts[kind]} public={self.public_counts[kind]}"
                )
            if sum(zone[kind] for zone in zones) != self.public_counts[kind]:
                raise DecisionSchemaError(
                    f"public_counts_by_zone does not sum to public_counts for kind {kind}"
                )
        for color, kind in enumerate(AKA_KINDS):
            if self.own_aka[color] > self.own_counts[kind]:
                raise DecisionSchemaError(
                    f"own_aka[{color}] > own five count for kind {kind}"
                )
            if self.public_aka[color] > self.public_counts[kind]:
                raise DecisionSchemaError(
                    f"public_aka[{color}] > public five count for kind {kind}"
                )
            if self.own_aka[color] + self.public_aka[color] > AKA_PER_COLOR:
                raise DecisionSchemaError(
                    f"more than {AKA_PER_COLOR} aka five of colour {color} exists"
                )

        indicator_counts: dict[int, int] = {}
        indicator_aka = [0, 0, 0]
        for index, marker in enumerate(self.dora_indicators):
            if not isinstance(marker, TileRef):
                raise DecisionSchemaError(
                    f"dora_indicators[{index}] must be TileRef"
                )
            if marker.is_unknown:
                raise DecisionSchemaError("dora indicators cannot be UNKNOWN")
            indicator_counts[marker.kind] = indicator_counts.get(marker.kind, 0) + 1
            if marker.is_aka:
                indicator_aka[marker.aka] += 1
        if sum(zones[2]) != len(self.dora_indicators):
            raise DecisionSchemaError(
                "dora_indicator zone must contain exactly the visible indicator tiles"
            )
        for kind, count in indicator_counts.items():
            if self.public_counts[kind] < count or zones[2][kind] != count:
                raise DecisionSchemaError(
                    f"dora indicator kind {kind} is not counted exactly once in the zone"
                )
        for color, count in enumerate(indicator_aka):
            if self.public_aka[color] < count:
                raise DecisionSchemaError(
                    f"public_aka[{color}] does not include visible aka indicators"
                )

    @property
    def total_own(self) -> int:
        return sum(self.own_counts)

    def zone_counts(self, zone: PublicZone) -> tuple[int, ...]:
        if not isinstance(zone, PublicZone):
            raise DecisionSchemaError(f"zone must be PublicZone, got {zone!r}")
        return self.public_counts_by_zone[PUBLIC_ZONE_ORDER.index(zone)]

    def as_dict(self) -> dict:
        return {
            "own_counts": list(self.own_counts),
            "own_aka": list(self.own_aka),
            "public_counts": list(self.public_counts),
            "public_aka": list(self.public_aka),
            "public_counts_by_zone": [list(zone) for zone in self.public_counts_by_zone],
            "dora_indicators": _plain_tiles(self.dora_indicators),
            "ruleset_id": self.ruleset_id.value,
        }


@dataclass(frozen=True)
class PublicMeld:
    """A public (or ankan-hidden) meld in a player's state."""

    meld_kind: MeldKind
    tiles: tuple[TileRef, ...]
    from_rel: int

    def __post_init__(self):
        if not isinstance(self.meld_kind, MeldKind):
            raise DecisionSchemaError(f"meld_kind must be MeldKind, got {self.meld_kind!r}")
        expected = MELD_TILE_COUNTS[self.meld_kind]
        if not isinstance(self.tiles, tuple) or len(self.tiles) != expected:
            raise DecisionSchemaError(
                f"{self.meld_kind.value} meld must have {expected} tiles"
            )
        _check_int(self.from_rel, "from_rel", 0, 4)
        for index, tile in enumerate(self.tiles):
            if not isinstance(tile, TileRef):
                raise DecisionSchemaError(f"meld tiles[{index}] must be TileRef")
        unknown = [tile for tile in self.tiles if tile.is_unknown]
        if unknown:
            if self.meld_kind != MeldKind.ANKAN:
                raise DecisionSchemaError("only ankan may hide its tiles")
            if len(unknown) != expected:
                raise DecisionSchemaError(
                    "ankan tiles must be either all known or all UNKNOWN"
                )
            return
        kinds = [tile.kind for tile in self.tiles]
        if self.meld_kind == MeldKind.CHI:
            if any(kind // 9 == 3 for kind in kinds):
                raise DecisionSchemaError("chi cannot contain honour tiles")
            suits = {kind // 9 for kind in kinds}
            if len(suits) != 1:
                raise DecisionSchemaError("chi tiles must share one suit")
            ordered = sorted(kinds)
            if ordered[1] - ordered[0] != 1 or ordered[2] - ordered[1] != 1:
                raise DecisionSchemaError(f"chi tiles are not consecutive: {ordered}")
        elif self.meld_kind in (MeldKind.PON, MeldKind.DAIMINKAN,
                                MeldKind.KAKAN, MeldKind.ANKAN):
            if len(set(kinds)) != 1:
                raise DecisionSchemaError(
                    f"{self.meld_kind.value} tiles must share one kind"
                )

    @property
    def is_open(self) -> bool:
        return self.meld_kind != MeldKind.ANKAN

    def as_dict(self) -> dict:
        return {
            "meld_kind": self.meld_kind.value,
            "tiles": _plain_tiles(self.tiles),
            "from_rel": self.from_rel,
        }


@dataclass(frozen=True)
class PlayerPublicState:
    """What every player can see about one seat (no hidden hand)."""

    seat_rel: int
    score: int
    wind: TileRef
    is_oya: bool
    riichi: RiichiState
    river_count: int
    melds: tuple[PublicMeld, ...]

    def __post_init__(self):
        _check_int(self.seat_rel, "seat_rel", 0, 3)
        _check_int(self.score, "score", -1_000_000, 1_000_000)
        if not isinstance(self.wind, TileRef) or self.wind.is_unknown:
            raise DecisionSchemaError("wind must be a known TileRef")
        if self.wind.kind not in (27, 28, 29, 30) or self.wind.is_aka:
            raise DecisionSchemaError("wind must be E/S/W/N")
        if not isinstance(self.is_oya, bool):
            raise DecisionSchemaError("is_oya must be bool")
        if not isinstance(self.riichi, RiichiState):
            raise DecisionSchemaError(f"riichi must be RiichiState, got {self.riichi!r}")
        _check_int(self.river_count, "river_count", 0, 70)
        if not isinstance(self.melds, tuple) or len(self.melds) > 4:
            raise DecisionSchemaError("melds must be a tuple of at most 4 entries")
        for index, meld in enumerate(self.melds):
            if not isinstance(meld, PublicMeld):
                raise DecisionSchemaError(f"melds[{index}] must be PublicMeld")

    @property
    def meld_count(self) -> int:
        return len(self.melds)

    def as_dict(self) -> dict:
        return {
            "seat_rel": self.seat_rel,
            "score": self.score,
            "wind": _plain_tile(self.wind),
            "is_oya": self.is_oya,
            "riichi": self.riichi.value,
            "river_count": self.river_count,
            "melds": [meld.as_dict() for meld in self.melds],
        }


@dataclass(frozen=True)
class GlobalPublicState:
    """Public table state + ruleset/provenance metadata."""

    bakaze: TileRef
    kyoku: int
    honba: int
    kyotaku: int
    tiles_left: int
    self_wind: TileRef
    oya_rel: int
    ruleset_id: RulesetId
    ruleset_deviation: tuple[RulesetDeviation, ...]
    source_kind: SourceKind

    def __post_init__(self):
        for name, tile in (("bakaze", self.bakaze), ("self_wind", self.self_wind)):
            if not isinstance(tile, TileRef) or tile.is_unknown:
                raise DecisionSchemaError(f"{name} must be a known TileRef")
            if tile.kind not in (27, 28, 29, 30):
                raise DecisionSchemaError(f"{name} must be E/S/W/N")
            if tile.is_aka:
                raise DecisionSchemaError(f"{name} cannot be an aka five")
        _check_int(self.kyoku, "kyoku", 1, 4)
        _check_int(self.honba, "honba", 0, 255)
        _check_int(self.kyotaku, "kyotaku", 0, 255)
        _check_int(self.tiles_left, "tiles_left", 0, 70)
        _check_int(self.oya_rel, "oya_rel", 0, 3)
        if not isinstance(self.ruleset_id, RulesetId):
            raise DecisionSchemaError(
                f"ruleset_id must be RulesetId, got {self.ruleset_id!r}"
            )
        if not isinstance(self.ruleset_deviation, tuple):
            raise DecisionSchemaError("ruleset_deviation must be a tuple")
        for value in self.ruleset_deviation:
            if not isinstance(value, RulesetDeviation):
                raise DecisionSchemaError(
                    f"ruleset_deviation entries must be RulesetDeviation, got {value!r}"
                )
        if len(set(self.ruleset_deviation)) != len(self.ruleset_deviation):
            raise DecisionSchemaError("duplicate ruleset_deviation entries")
        if self.ruleset_id == RulesetId.TENHOU_4P_RANKED and self.ruleset_deviation:
            raise DecisionSchemaError(
                "tenhou_4p_ranked samples cannot carry arena rule deviations"
            )
        if self.ruleset_id == RulesetId.MORTAL_ARENA_PINNED:
            # Any subset of R02/R25/R26 is allowed; unknown codes were rejected above.
            pass
        if not isinstance(self.source_kind, SourceKind):
            raise DecisionSchemaError(
                f"source_kind must be SourceKind, got {self.source_kind!r}"
            )
        if self.source_kind == SourceKind.HUMAN_LOG and (
                self.ruleset_id != RulesetId.TENHOU_4P_RANKED):
            raise DecisionSchemaError("human_log samples must use tenhou_4p_ranked")
        if self.source_kind == SourceKind.ARENA_SELFPLAY and (
                self.ruleset_id != RulesetId.MORTAL_ARENA_PINNED):
            raise DecisionSchemaError("arena_selfplay samples must use mortal_arena_pinned")
        # Relative seat 0 is the dealer, so self wind equals the round wind iff oya_rel=0.
        if (self.self_wind.kind == self.bakaze.kind) != (self.oya_rel == 0):
            raise DecisionSchemaError(
                "self_wind/bakaze/oya_rel are inconsistent"
            )

    def has_deviation(self, code) -> bool:
        try:
            return RulesetDeviation(code) in self.ruleset_deviation
        except ValueError:
            return False

    def as_dict(self) -> dict:
        return {
            "bakaze": _plain_tile(self.bakaze),
            "kyoku": self.kyoku,
            "honba": self.honba,
            "kyotaku": self.kyotaku,
            "tiles_left": self.tiles_left,
            "self_wind": _plain_tile(self.self_wind),
            "oya_rel": self.oya_rel,
            "ruleset_id": self.ruleset_id.value,
            "ruleset_deviation": [value.value for value in self.ruleset_deviation],
            "source_kind": self.source_kind.value,
        }


def assert_forward_safe(value) -> None:
    """Public guard: raise if a target/private key appears anywhere in a payload."""
    _scan_forbidden(value)


def _scan_forbidden(value, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_FORWARD_KEYS:
                raise TargetLeakError(
                    f"forbidden target/private field {key!r} at {path}"
                )
            _scan_forbidden(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_forbidden(item, f"{path}[{index}]")


@dataclass(frozen=True)
class DecisionInput:
    """Forward-pass input at one decision point (history only, no labels)."""

    events: tuple[ObservableEventV3, ...]
    tile_state: VisibleTileState
    players_public: tuple[
        PlayerPublicState, PlayerPublicState, PlayerPublicState, PlayerPublicState
    ]
    global_public: GlobalPublicState
    legal_mask: tuple[bool, ...]
    phase: ActionPhase
    before_event_idx: int
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise DecisionSchemaError(
                f"schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}"
            )
        if not isinstance(self.events, tuple) or not self.events:
            raise DecisionSchemaError("events must be a non-empty tuple")
        previous_seq = -1
        previous_kyoku = -1
        for index, event in enumerate(self.events):
            if not isinstance(event, ObservableEventV3):
                raise DecisionSchemaError(f"events[{index}] is not ObservableEventV3")
            event_cls = type(event)
            if event_cls not in _ALLOWED_EVENT_CLASSES:
                raise DecisionSchemaError(
                    f"events[{index}] uses a rogue ObservableEventV3 subclass "
                    f"{event_cls.__name__}; exact registered classes only"
                )
            expected_fields = ("seq_idx", "kyoku_idx") + tuple(event_cls.FIELDS)
            actual_fields = tuple(field.name for field in fields(event))
            if actual_fields != expected_fields:
                raise DecisionSchemaError(
                    f"events[{index}] field set {actual_fields} != {expected_fields}"
                )
            if event.seq_idx <= previous_seq:
                raise DecisionSchemaError("events must be strictly ordered by seq_idx")
            if event.kyoku_idx < previous_kyoku:
                raise DecisionSchemaError("events must not go back to an earlier kyoku")
            if event.seq_idx >= self.before_event_idx:
                raise DecisionSchemaError(
                    f"event seq {event.seq_idx} is not before before_event_idx "
                    f"{self.before_event_idx} (future event leaked)"
                )
            previous_seq = event.seq_idx
            previous_kyoku = event.kyoku_idx
        _check_int(self.before_event_idx, "before_event_idx", 1)
        if not isinstance(self.tile_state, VisibleTileState):
            raise DecisionSchemaError("tile_state must be VisibleTileState")
        if not isinstance(self.global_public, GlobalPublicState):
            raise DecisionSchemaError("global_public must be GlobalPublicState")
        if self.tile_state.ruleset_id != self.global_public.ruleset_id:
            raise DecisionSchemaError(
                "tile_state.ruleset_id != global_public.ruleset_id"
            )
        if not isinstance(self.players_public, tuple) or len(self.players_public) != 4:
            raise DecisionSchemaError("players_public must contain exactly 4 seats")
        for index, player in enumerate(self.players_public):
            if not isinstance(player, PlayerPublicState):
                raise DecisionSchemaError(f"players_public[{index}] is not public state")
            if player.seat_rel != index:
                raise DecisionSchemaError(
                    f"players_public[{index}].seat_rel {player.seat_rel} != {index}"
                )
        oya = [player.seat_rel for player in self.players_public if player.is_oya]
        if oya != [self.global_public.oya_rel]:
            raise DecisionSchemaError(
                f"players_public dealer {oya} != global_public.oya_rel "
                f"{self.global_public.oya_rel}"
            )
        bakaze_index = self.global_public.bakaze.kind - 27
        for player in self.players_public:
            expected_wind = 27 + (
                (bakaze_index + player.seat_rel - self.global_public.oya_rel) % 4
            )
            if player.wind.kind != expected_wind:
                raise DecisionSchemaError(
                    f"players_public[{player.seat_rel}].wind "
                    f"{player.wind.text} != expected kind {expected_wind}"
                )
        if self.players_public[0].wind.kind != self.global_public.self_wind.kind:
            raise DecisionSchemaError(
                "players_public[0].wind != global_public.self_wind"
            )

        total = sum(player.score for player in self.players_public)
        if total + KYOTAKU_POINTS * self.global_public.kyotaku != TOTAL_POINTS:
            raise DecisionSchemaError(
                "player scores + kyotaku violate the 100000 total invariant: "
                f"{total} + {KYOTAKU_POINTS * self.global_public.kyotaku}"
            )
        if not isinstance(self.phase, ActionPhase):
            raise DecisionSchemaError(f"phase must be ActionPhase, got {self.phase!r}")
        if not isinstance(self.legal_mask, tuple) or len(self.legal_mask) != ACTION_SPACE:
            raise DecisionSchemaError(
                f"legal_mask must be a tuple of {ACTION_SPACE} bools"
            )
        try:
            validate_mask_for_phase(self.legal_mask, self.phase)
        except ActionSchemaError as exc:
            raise DecisionSchemaError(str(exc)) from exc

    @property
    def legal_actions(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.legal_mask) if value)

    def to_forward_dict(self) -> dict:
        """Whitelisted serialization; raises if a target/private field sneaks in."""
        out = {
            "schema_version": self.schema_version,
            "events": [event.to_dict() for event in self.events],
            "tile_state": self.tile_state.as_dict(),
            "players_public": [player.as_dict() for player in self.players_public],
            "global_public": self.global_public.as_dict(),
            "legal_mask": list(self.legal_mask),
            "phase": self.phase.value,
            "before_event_idx": self.before_event_idx,
        }
        if set(out) != set(DECISION_INPUT_FIELDS):
            raise AssertionError("DecisionInput forward field set drifted")
        _scan_forbidden(out)
        return out

    def assert_forward_safe(self) -> None:
        assert_forward_safe(self.to_forward_dict())


__all__ = [
    "AKA_KINDS", "DECISION_INPUT_FIELDS", "DECISION_TARGET_FIELDS",
    "DecisionInput", "DecisionSchemaError", "FORBIDDEN_FORWARD_KEYS", "GlobalPublicState",
    "MELD_TILE_COUNTS", "MeldKind", "PUBLIC_ZONE_ORDER", "PlayerPublicState", "PublicMeld", "PublicZone",
    "RULESET_DEVIATIONS", "RiichiState", "RulesetDeviation", "RulesetId", "SourceKind",
    "TOTAL_POINTS", "TargetLeakError", "VisibleTileState", "assert_forward_safe",
]
