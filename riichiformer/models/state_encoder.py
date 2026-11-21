"""State features and MLP state encoder for the PR-13 MVP-S model.

The state encoder is a 2-layer MLP over a **documented, fixed-layout** vector
(``STATE_FEATURE_DIM = 172``) followed by a LayerNorm to ``d_model``.  Its output is
injected (added) only at the decision token positions, exactly as in the roadmap
Recipe E:

``state = MLP(34 tile counts + own/public aka + 4 player public features + global)``

Layout (all values are finite and normalized; no target/rank/oracle field exists)
-------------------------------------------------------------------------------
============  ====  ===========================================================
segment       dim   normalization
============  ====  ===========================================================
own_counts     34   count / 4
public_counts  34   count / 4 (river + melds + visible dora indicators)
own_aka         3   0/1 per colour (5m, 5p, 5s)
public_aka      3   0/1 per colour
players        44   4 × [score_rel, is_oya, riichi one-hot 3, river/70,
                    meld_count/4, wind one-hot 4]
global         16   bakaze one-hot 4, kyoku/4, honba/255, kyotaku/255,
                    tiles_left/70, self_wind one-hot 4, oya_rel one-hot 4
dora           35   visible indicator counts/34 kinds / 4, num_indicators/5
phase           3   one-hot (SELF_TURN, CALL_RESPONSE, KAN_SELECTION)
============  ====  ===========================================================

``score_rel`` follows the frozen normalizer spec: ``clip((score - 25000) / 25000,
-3, 3)``.  Counts use the schema scale of 4.  Ruleset id / source kind / deviations
are **not** model features (they are provenance metadata); they only affect which
samples a trainer may use.

:class:`StateSnapshot` is the synthetic-first builder input.  PR-11's encoder may
construct it from ``DecisionInput`` later; this module imports no PR-11 ``data`` code
and only reads the frozen tile/aka constants from ``schemas.action_v3``, so the
interface stays replaceable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..schemas.action_v3 import AKA_KINDS, AKA_PER_COLOR, TILE_KIND_COUNT
from .batch import PHASE_CLASSES

INITIAL_POINTS = 25_000
KYOTAKU_POINTS = 1_000
TOTAL_POINTS = 4 * INITIAL_POINTS

#: Feature scales (also the frozen schema normalizer spec).
COUNT_SCALE = 4.0
SCORE_CENTER = 25_000.0
SCORE_SCALE = 25_000.0
SCORE_CLIP_SIGMA = 3.0
RIVER_SCALE = 70.0
MELD_SCALE = 4.0
KYOKU_SCALE = 4.0
HONBA_SCALE = 255.0
KYOTAKU_SCALE = 255.0
TILES_LEFT_SCALE = 70.0
NUM_INDICATOR_SCALE = 5.0

PLAYER_FEATURE_DIM = 11
NUM_PLAYERS = 4
GLOBAL_FEATURE_DIM = 16
DORA_FEATURE_DIM = TILE_KIND_COUNT + 1
RIICHI_CLASSES = 3
WIND_CLASSES = 4


@dataclass(frozen=True)
class StateFeatureSlot:
    name: str
    offset: int
    dim: int
    description: str


def _build_layout() -> tuple[StateFeatureSlot, ...]:
    entries: list[tuple[str, int, str]] = [
        ("own_counts", TILE_KIND_COUNT, "own hand physical counts / 4"),
        ("public_counts", TILE_KIND_COUNT, "public river+meld+indicator counts / 4"),
        ("own_aka", 3, "own red-five count per colour (0/1)"),
        ("public_aka", 3, "public red-five count per colour (0/1)"),
        ("players", NUM_PLAYERS * PLAYER_FEATURE_DIM,
         "per relative seat: score_rel, is_oya, riichi 3, river, melds, wind 4"),
        ("global", GLOBAL_FEATURE_DIM,
         "bakaze 4, kyoku, honba, kyotaku, tiles_left, self_wind 4, oya_rel 4"),
        ("dora", DORA_FEATURE_DIM, "visible indicator counts / 4 and num indicators / 5"),
        ("phase", PHASE_CLASSES, "SELF_TURN / CALL_RESPONSE / KAN_SELECTION one-hot"),
    ]
    layout: list[StateFeatureSlot] = []
    offset = 0
    for name, dim, description in entries:
        layout.append(StateFeatureSlot(name=name, offset=offset, dim=dim,
                                       description=description))
        offset += dim
    return tuple(layout)


STATE_FEATURE_LAYOUT: tuple[StateFeatureSlot, ...] = _build_layout()
STATE_FEATURE_DIM: int = sum(slot.dim for slot in STATE_FEATURE_LAYOUT)
STATE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{slot.name}[{index}]" if slot.dim > 1 else slot.name
    for slot in STATE_FEATURE_LAYOUT
    for index in range(slot.dim)
)


class StateFeatureError(ValueError):
    """A :class:`StateSnapshot` violates the visible-state contract."""


def _check_counts(name: str, value, length: int = TILE_KIND_COUNT) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) != length:
        raise StateFeatureError(f"{name} must be a tuple of {length} ints")
    for index, count in enumerate(value):
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 4:
            raise StateFeatureError(f"{name}[{index}] must be int in 0..4, got {count!r}")
    return value


def _check_int(name: str, value, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise StateFeatureError(f"{name} must be int in [{low}, {high}], got {value!r}")
    return value


@dataclass(frozen=True)
class StateSnapshot:
    """Visible table state at one decision (PR-13's synthetic feature source).

    All fields are public/own information only; there is no label, rank, opponent
    hand, wall or future field.  The ``__post_init__`` invariants mirror the frozen
    schema's ``VisibleTileState`` / ``GlobalPublicState`` checks.
    """

    own_counts: tuple[int, ...]
    public_counts: tuple[int, ...]
    own_aka: tuple[int, int, int]
    public_aka: tuple[int, int, int]
    player_scores: tuple[int, int, int, int]
    player_riichi: tuple[int, int, int, int] = (0, 0, 0, 0)
    player_river_counts: tuple[int, int, int, int] = (0, 0, 0, 0)
    player_meld_counts: tuple[int, int, int, int] = (0, 0, 0, 0)
    player_is_oya: tuple[bool, bool, bool, bool] = (True, False, False, False)
    player_winds: tuple[int, int, int, int] = (27, 28, 29, 30)
    bakaze_kind: int = 27
    kyoku: int = 1
    honba: int = 0
    kyotaku: int = 0
    tiles_left: int = 69
    oya_rel: int = 0
    dora_indicator_counts: tuple[int, ...] = (0,) * TILE_KIND_COUNT
    phase: int = 0

    def __post_init__(self) -> None:
        _check_counts("own_counts", self.own_counts)
        _check_counts("public_counts", self.public_counts)
        if sum(self.own_counts) > 14:
            raise StateFeatureError(
                f"own hand has {sum(self.own_counts)} tiles (> 14): not a mahjong hand"
            )
        for kind in range(TILE_KIND_COUNT):
            if self.own_counts[kind] + self.public_counts[kind] > 4:
                raise StateFeatureError(
                    f"tile overflow for kind {kind}: own={self.own_counts[kind]} "
                    f"public={self.public_counts[kind]}"
                )
        for name, aka in (("own_aka", self.own_aka), ("public_aka", self.public_aka)):
            if not isinstance(aka, tuple) or len(aka) != 3:
                raise StateFeatureError(f"{name} must be a tuple of 3 ints")
            for colour, count in enumerate(aka):
                if isinstance(count, bool) or not isinstance(count, int) \
                        or not 0 <= count <= AKA_PER_COLOR:
                    raise StateFeatureError(
                        f"{name}[{colour}] must be 0..{AKA_PER_COLOR}, got {count!r}"
                    )
        for colour, kind in enumerate(AKA_KINDS):
            if self.own_aka[colour] > self.own_counts[kind]:
                raise StateFeatureError(
                    f"own_aka[{colour}] > own five count for kind {kind}"
                )
            if self.public_aka[colour] > self.public_counts[kind]:
                raise StateFeatureError(
                    f"public_aka[{colour}] > public five count for kind {kind}"
                )
            if self.own_aka[colour] + self.public_aka[colour] > AKA_PER_COLOR:
                raise StateFeatureError(
                    f"more than {AKA_PER_COLOR} aka five of colour {colour} exists"
                )

        for name, value in (
            ("player_scores", self.player_scores),
            ("player_riichi", self.player_riichi),
            ("player_river_counts", self.player_river_counts),
            ("player_meld_counts", self.player_meld_counts),
        ):
            if not isinstance(value, tuple) or len(value) != 4:
                raise StateFeatureError(f"{name} must be a tuple of 4 ints")
        for score in self.player_scores:
            _check_int("player_scores[]", score, -1_000_000, 1_000_000)
        for riichi in self.player_riichi:
            _check_int("player_riichi[]", riichi, 0, RIICHI_CLASSES - 1)
        for river in self.player_river_counts:
            _check_int("player_river_counts[]", river, 0, 70)
        for melds in self.player_meld_counts:
            _check_int("player_meld_counts[]", melds, 0, 4)

        _check_int("bakaze_kind", self.bakaze_kind, 27, 30)
        _check_int("kyoku", self.kyoku, 1, 4)
        _check_int("honba", self.honba, 0, 255)
        _check_int("kyotaku", self.kyotaku, 0, 255)
        _check_int("tiles_left", self.tiles_left, 0, 70)
        _check_int("oya_rel", self.oya_rel, 0, 3)
        _check_int("phase", self.phase, 0, PHASE_CLASSES - 1)

        _check_counts("dora_indicator_counts", self.dora_indicator_counts)
        if sum(self.dora_indicator_counts) > 5:
            raise StateFeatureError("at most 5 dora indicators are visible")

        if not isinstance(self.player_is_oya, tuple) or len(self.player_is_oya) != 4:
            raise StateFeatureError("player_is_oya must be a tuple of 4 bools")
        for index, is_oya in enumerate(self.player_is_oya):
            if not isinstance(is_oya, bool):
                raise StateFeatureError(f"player_is_oya[{index}] must be bool")
        if sum(self.player_is_oya) != 1:
            raise StateFeatureError("exactly one player must be the dealer")
        if not self.player_is_oya[self.oya_rel]:
            raise StateFeatureError("player_is_oya[oya_rel] must be True")

        if not isinstance(self.player_winds, tuple) or len(self.player_winds) != 4:
            raise StateFeatureError("player_winds must be a tuple of 4 wind kinds")
        for seat, wind in enumerate(self.player_winds):
            _check_int(f"player_winds[{seat}]", wind, 27, 30)
            expected = 27 + ((self.bakaze_kind - 27 + seat - self.oya_rel) % 4)
            if wind != expected:
                raise StateFeatureError(
                    f"player_winds[{seat}]={wind} != expected {expected} for "
                    f"bakaze={self.bakaze_kind}, oya_rel={self.oya_rel}"
                )
        total = sum(self.player_scores) + KYOTAKU_POINTS * self.kyotaku
        if total != TOTAL_POINTS:
            raise StateFeatureError(
                f"player scores + kyotaku must total {TOTAL_POINTS}, got {total}"
            )

    @property
    def self_wind_kind(self) -> int:
        return self.player_winds[0]

    def phase_one_hot(self) -> list[float]:
        return [1.0 if self.phase == index else 0.0 for index in range(PHASE_CLASSES)]


def _score_rel(score: int) -> float:
    value = (float(score) - SCORE_CENTER) / SCORE_SCALE
    return max(-SCORE_CLIP_SIGMA, min(SCORE_CLIP_SIGMA, value))


def state_features_from_snapshot(snapshot: StateSnapshot, *,
                                 dtype: torch.dtype = torch.float32) -> Tensor:
    """Flatten a snapshot into the documented 172-float vector."""
    if not isinstance(snapshot, StateSnapshot):
        raise TypeError(f"expected StateSnapshot, got {type(snapshot).__name__}")
    values: list[float] = []
    values.extend(count / COUNT_SCALE for count in snapshot.own_counts)
    values.extend(count / COUNT_SCALE for count in snapshot.public_counts)
    values.extend(float(count) for count in snapshot.own_aka)
    values.extend(float(count) for count in snapshot.public_aka)
    for seat in range(NUM_PLAYERS):
        values.append(_score_rel(snapshot.player_scores[seat]))
        values.append(1.0 if snapshot.player_is_oya[seat] else 0.0)
        riichi = snapshot.player_riichi[seat]
        values.extend(1.0 if riichi == index else 0.0 for index in range(RIICHI_CLASSES))
        values.append(snapshot.player_river_counts[seat] / RIVER_SCALE)
        values.append(snapshot.player_meld_counts[seat] / MELD_SCALE)
        wind = snapshot.player_winds[seat] - 27
        values.extend(1.0 if wind == index else 0.0 for index in range(WIND_CLASSES))
    values.extend(1.0 if snapshot.bakaze_kind - 27 == index else 0.0
                  for index in range(WIND_CLASSES))
    values.append(snapshot.kyoku / KYOKU_SCALE)
    values.append(snapshot.honba / HONBA_SCALE)
    values.append(snapshot.kyotaku / KYOTAKU_SCALE)
    values.append(snapshot.tiles_left / TILES_LEFT_SCALE)
    self_wind = snapshot.self_wind_kind - 27
    values.extend(1.0 if self_wind == index else 0.0 for index in range(WIND_CLASSES))
    values.extend(1.0 if snapshot.oya_rel == index else 0.0 for index in range(4))
    values.extend(count / COUNT_SCALE for count in snapshot.dora_indicator_counts)
    values.append(sum(snapshot.dora_indicator_counts) / NUM_INDICATOR_SCALE)
    values.extend(snapshot.phase_one_hot())
    if len(values) != STATE_FEATURE_DIM:  # pragma: no cover - layout drift guard
        raise AssertionError(
            f"state feature layout produced {len(values)} values, "
            f"expected {STATE_FEATURE_DIM}"
        )
    return torch.tensor(values, dtype=dtype)


def state_features_from_snapshots(snapshots, *,
                                  dtype: torch.dtype = torch.float32) -> Tensor:
    if not snapshots:
        raise StateFeatureError("need at least one snapshot")
    return torch.stack([state_features_from_snapshot(snapshot, dtype=dtype)
                        for snapshot in snapshots], dim=0)


class StateMLP(nn.Module):
    """``F_state -> hidden -> d_model`` MLP with GELU and output LayerNorm."""

    def __init__(self, hidden: int = 512, d_model: int = 256,
                 input_dim: int = STATE_FEATURE_DIM) -> None:
        super().__init__()
        if hidden <= 0 or d_model <= 0 or input_dim <= 0:
            raise ValueError("StateMLP dimensions must be positive")
        if input_dim != STATE_FEATURE_DIM:
            raise ValueError(
                f"input_dim {input_dim} != frozen STATE_FEATURE_DIM {STATE_FEATURE_DIM}"
            )
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.d_model = int(d_model)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, features: Tensor, *, validate: bool = True) -> Tensor:
        if not isinstance(features, Tensor) or features.dim() != 2:
            raise StateFeatureError(
                f"state features must be [N, {self.input_dim}], got "
                f"{tuple(features.shape) if isinstance(features, Tensor) else type(features)}"
            )
        if features.size(1) != self.input_dim:
            raise StateFeatureError(
                f"state feature width {features.size(1)} != {self.input_dim}"
            )
        if not features.is_floating_point():
            raise StateFeatureError(
                f"state features must be floating point, got {features.dtype}"
            )
        if validate and not bool(torch.isfinite(features).all()):
            raise StateFeatureError("state features contain NaN/Inf")
        target_dtype = self.net[0].weight.dtype
        return self.net(features.to(dtype=target_dtype))


# Backwards-friendly alias used by the roadmap/config (`state_encoder: mlp`).
StateEncoder = StateMLP


__all__ = [
    "COUNT_SCALE", "DORA_FEATURE_DIM", "GLOBAL_FEATURE_DIM", "KYOTAKU_POINTS",
    "PLAYER_FEATURE_DIM", "RIICHI_CLASSES", "STATE_FEATURE_DIM", "STATE_FEATURE_LAYOUT",
    "STATE_FEATURE_NAMES", "StateEncoder", "StateFeatureError", "StateFeatureSlot",
    "StateMLP", "StateSnapshot", "TOTAL_POINTS", "WIND_CLASSES", "INITIAL_POINTS",
    "state_features_from_snapshot", "state_features_from_snapshots",
]
