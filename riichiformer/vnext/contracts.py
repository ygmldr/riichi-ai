"""Frozen PR-02 contracts for RiichiFormer VNext.

This module is deliberately dependency-light (stdlib + torch) and is the *single*
place where the VNext ABI is defined.  The native 46-action order is imported
from :mod:`riichiformer.schemas.action_v3`; nothing here re-orders it.

The contracts separate what the Actor may see from what the trainer may use:

* :class:`VNextVisibleState` and :class:`VNextModelBatch` contain **no target,
  rank, oracle, future or hidden-world field**;
* :class:`VNextTargets` is the trainer-only side channel;
* :class:`GRPBoundary` and :class:`RLDecisionRow` are versioned public-data /
  rollout contracts.

A field name containing ``target``, ``label``, ``rank_final`` or ``oracle`` is
rejected in the visible/model objects by :func:`assert_forward_safe`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from ..schemas.action_v3 import (
    ACTION_MAP_HASH,
    ACTION_SPACE,
    AKA_KINDS,
    SCHEMA_VERSION,
    TILE_KIND_COUNT,
    ActionPhase,
    canonical_json,
    validate_mask_for_phase,
)

MODEL_ARCH_ID = "rf-vnext-crossattn/1"
ENCODER_CONTRACT_ID = "current_kyoku_bos/1"
STATE_CONTRACT_ID = "tile34_global/1"
GRP_CONTRACT_ID = "grp24-joint/1"
RL_CONTRACT_ID = "rf-rl-row/1"
PT_TABLE_5D_ID = "pt5d/75,30,0,-105"
PT_TABLE_7D_ID = "pt7d/90,45,0,-135"
PT_TABLE_5D: tuple[int, int, int, int] = (75, 30, 0, -105)
PT_TABLE_7D: tuple[int, int, int, int] = (90, 45, 0, -135)

TILE_CANDIDATES = 37
OTHER_ACTIONS = 9
BOS_EVENT_ID = 16
PAD_EVENT_ID = 17
EVENT_TYPE_VOCAB = 18
EVENT_ACTOR_VOCAB = 6
EVENT_TILE_VOCAB = 35
EVENT_AKA_VOCAB = 4
EVENT_TSUMOGIRI_VOCAB = 2
EVENT_KYOKU_POS_VOCAB = 512
EVENT_MELD_SLOTS = 3

#: Per-tile channel registry.  Every channel is computed from POV-visible data.
TILE_FEATURE_NAMES: tuple[str, ...] = (
    "own_count_over_4",
    "own_ge_1",
    "own_ge_2",
    "own_ge_3",
    "own_ge_4",
    "public_count_over_4",
    "own_aka_m",
    "own_aka_p",
    "own_aka_s",
    "public_aka_m",
    "public_aka_p",
    "public_aka_s",
    "opponent_river_shimocha",
    "opponent_river_toimen",
    "opponent_river_kamicha",
    "opponent_genbutsu_shimocha",
    "opponent_genbutsu_toimen",
    "opponent_genbutsu_kamicha",
    "opponent_post_riichi_shimocha",
    "opponent_post_riichi_toimen",
    "opponent_post_riichi_kamicha",
    "opponent_meld_shimocha_over_4",
    "opponent_meld_toimen_over_4",
    "opponent_meld_kamicha_over_4",
    "dora_indicator_count_over_4",
    "is_visible_dora",
)
TILE_FEATURE_DIM = len(TILE_FEATURE_NAMES)

#: Global feature registry (all public / own-visible).
GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "score_self_rel", "score_shimocha_rel", "score_toimen_rel", "score_kamicha_rel",
    "score_diff_self_leader", "score_diff_self_last", "score_diff_leader_last",
    "kyoku_over_4", "honba_over_255", "kyotaku_over_255", "tiles_left_over_70",
    "bakaze_e", "bakaze_s", "bakaze_w", "bakaze_n",
    "self_wind_e", "self_wind_s", "self_wind_w", "self_wind_n",
    "oya_rel_0", "oya_rel_1", "oya_rel_2", "oya_rel_3",
    "phase_self_turn", "phase_call_response", "phase_kan_selection",
    "riichi_self", "riichi_shimocha", "riichi_toimen", "riichi_kamicha",
    "meld_self_over_4", "meld_shimocha_over_4", "meld_toimen_over_4",
    "meld_kamicha_over_4",
    "river_self_over_70", "river_shimocha_over_70",
    "river_toimen_over_70", "river_kamicha_over_70",
)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)

#: Candidate descriptor (the first 34 are normal kinds, 34..36 aka candidates).
CANDIDATE_FEATURE_NAMES: tuple[str, ...] = (
    "is_aka_candidate", "aka_color_over_2", "base_kind_over_33", "is_playable_tile",
)
CANDIDATE_FEATURE_DIM = len(CANDIDATE_FEATURE_NAMES)

#: Fixed 24 joint final-rank permutations, lexicographic by (seat0..seat3) rank.
GRP_PERMUTATIONS: tuple[tuple[int, int, int, int], ...] = tuple(
    itertools.permutations(range(4))
)
if len(GRP_PERMUTATIONS) != 24 or len(set(GRP_PERMUTATIONS)) != 24:
    raise AssertionError("GRP permutation table must contain exactly 24 unique entries")
GRP_PERMUTATION_INDEX: dict[tuple[int, int, int, int], int] = {
    perm: index for index, perm in enumerate(GRP_PERMUTATIONS)
}

PT_TABLES: dict[str, tuple[int, int, int, int]] = {
    PT_TABLE_5D_ID: PT_TABLE_5D,
    PT_TABLE_7D_ID: PT_TABLE_7D,
}

FORBIDDEN_FORWARD_WORDS: tuple[str, ...] = (
    "target", "label", "oracle", "rank_final", "expert_action", "supervision",
    "censor", "future", "hidden",
)


class ContractError(ValueError):
    """A VNext contract object violates the frozen ABI."""


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_bool_tensor(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor) or value.dtype is not torch.bool:
        raise ContractError(f"{name} must be a bool torch.Tensor")
    if tuple(value.shape) != shape:
        raise ContractError(f"{name} shape {tuple(value.shape)} != {shape}")


def _check_long_tensor(name: str, value: Tensor, shape: tuple[int, ...],
                       *, low: int = 0, high: int | None = None) -> None:
    if not isinstance(value, Tensor) or value.dtype is not torch.long:
        raise ContractError(f"{name} must be a long torch.Tensor")
    if tuple(value.shape) != shape:
        raise ContractError(f"{name} shape {tuple(value.shape)} != {shape}")
    if value.numel() and (int(value.min()) < low
                          or (high is not None and int(value.max()) >= high)):
        upper = (high - 1) if high is not None else "inf"
        raise ContractError(
            f"{name} values must be in [{low}, {upper}], got "
            f"[{int(value.min())}, {int(value.max())}]")


def _check_float_tensor(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ContractError(f"{name} must be a floating torch.Tensor")
    if tuple(value.shape) != shape:
        raise ContractError(f"{name} shape {tuple(value.shape)} != {shape}")
    if not bool(torch.isfinite(value).all()):
        raise ContractError(f"{name} contains NaN/Inf")


def assert_forward_safe(names: Sequence[str], *, owner: str) -> None:
    forbidden = sorted(
        name for name in names
        if any(word in name for word in FORBIDDEN_FORWARD_WORDS)
    )
    if forbidden:
        raise ContractError(
            f"{owner} must not carry target/private field names: {forbidden}")


def _phase(value: Any) -> int:
    if isinstance(value, ActionPhase):
        return tuple(ActionPhase).index(value)
    if _is_int(value) and 0 <= value < len(ActionPhase):
        return int(value)
    raise ContractError(f"phase must be ActionPhase or int 0..2, got {value!r}")


@dataclass(frozen=True)
class VNextConfig:
    """Model and contract configuration; hash is part of every checkpoint."""

    name: str = "vnext"
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    ffn_hidden: int = 704
    dropout: float = 0.05
    max_positions: int = 4096
    cross_attn_layers: int = 1
    cross_attn_heads: int = 4
    tile_embedding: str = "factorized"  # factorized | identity
    head_type: str = "tile_scored"       # tile_scored | linear46
    model_mode: str = "event_state"      # event_state | state_only
    critic_enabled: bool = True
    critic_grad_mode: str = "detach_actor"  # joint | detach_actor
    tile_identity_scale: float = 1.0
    rope_base: float = 10_000.0
    rmsnorm_eps: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.tile_embedding not in {"factorized", "identity"}:
            raise ContractError("tile_embedding must be factorized or identity")
        if self.head_type not in {"tile_scored", "linear46"}:
            raise ContractError("head_type must be tile_scored or linear46")
        if self.model_mode not in {"event_state", "state_only"}:
            raise ContractError("model_mode must be event_state or state_only")
        if self.critic_grad_mode not in {"joint", "detach_actor"}:
            raise ContractError("critic_grad_mode must be joint or detach_actor")
        if self.cross_attn_layers not in {0, 1, 2}:
            raise ContractError("cross_attn_layers must be 0, 1 or 2")
        for name in ("d_model", "n_layers", "n_heads", "ffn_hidden",
                     "max_positions", "cross_attn_heads"):
            value = getattr(self, name)
            if not _is_int(value) or value <= 0:
                raise ContractError(f"{name} must be a positive int, got {value!r}")
        if self.d_model % self.n_heads or self.d_model % self.cross_attn_heads:
            raise ContractError("d_model must be divisible by both head counts")
        if not 0.0 <= self.dropout < 1.0:
            raise ContractError("dropout must be in [0, 1)")
        if self.tile_identity_scale < 0:
            raise ContractError("tile_identity_scale must be >= 0")
        if self.rope_base <= 1.0 or self.rmsnorm_eps <= 0:
            raise ContractError("rope_base must be > 1 and rmsnorm_eps > 0")
        if not isinstance(self.name, str) or not self.name:
            raise ContractError("name must be a non-empty string")

    @property
    def architecture_hash(self) -> str:
        return _sha(self.as_dict())

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping) -> "VNextConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = set(mapping) - allowed
        if unknown:
            raise ContractError(f"unknown VNextConfig fields: {sorted(unknown)}")
        return cls(**dict(mapping))


@dataclass(frozen=True)
class VNextDecisionIndex:
    """One decision's causal position; ``decision_order`` is not a token offset."""

    match_id: str
    kyoku_id: int
    pov: int
    decision_id: str
    decision_order: int
    phase: int
    prefix_event_count: int
    prefix_token_offset: int

    def __post_init__(self) -> None:
        for name in ("match_id", "decision_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractError(f"{name} must be a non-empty string")
        if not _is_int(self.pov) or not 0 <= self.pov <= 3:
            raise ContractError("pov must be int 0..3")
        for name in ("kyoku_id", "decision_order", "phase",
                     "prefix_event_count", "prefix_token_offset"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ContractError(f"{name} must be a non-negative int")
        if self.phase >= len(ActionPhase):
            raise ContractError("phase outside ActionPhase range")
        if self.prefix_token_offset < self.prefix_event_count:
            raise ContractError(
                "prefix_token_offset must be >= prefix_event_count "
                "(one event may tokenize to multiple tokens)"
            )


@dataclass(frozen=True)
class VNextVisibleState:
    """Strictly POV-visible state at one decision.

    ``tile_features`` is ``[34, 26]``; the registry is
    :data:`TILE_FEATURE_NAMES`.  ``candidate_features`` is ``[37, 4]`` and
    distinguishes normal and aka-five candidates.  No target/rank/future field
    is present.
    """

    tile_features: Tensor
    global_features: Tensor
    candidate_features: Tensor
    legal_mask: Tensor
    phase: int

    def __post_init__(self) -> None:
        _check_float_tensor("tile_features", self.tile_features,
                            (TILE_KIND_COUNT, TILE_FEATURE_DIM))
        _check_float_tensor("global_features", self.global_features,
                            (GLOBAL_FEATURE_DIM,))
        _check_float_tensor("candidate_features", self.candidate_features,
                            (TILE_CANDIDATES, CANDIDATE_FEATURE_DIM))
        _check_bool_tensor("legal_mask", self.legal_mask, (ACTION_SPACE,))
        phase = _phase(self.phase)
        if not bool(self.legal_mask.any()):
            raise ContractError("legal_mask must contain at least one action")
        validate_mask_for_phase(tuple(bool(v) for v in self.legal_mask),
                                tuple(ActionPhase)[phase])
        self._validate_candidate_features()

    def _validate_candidate_features(self) -> None:
        cf = self.candidate_features
        if not bool(((cf[:, 0] == 0) | (cf[:, 0] == 1)).all()):
            raise ContractError("candidate is_aka_candidate must be 0/1")
        if not bool(((cf[:, 3] == 0) | (cf[:, 3] == 1)).all()):
            raise ContractError("candidate is_playable_tile must be 0/1")
        # First 34 are normal kinds in native order.
        expected_kind = torch.arange(TILE_KIND_COUNT, dtype=cf.dtype) / 33.0
        if not torch.allclose(cf[:34, 2], expected_kind, atol=1e-6):
            raise ContractError("normal candidate base_kind column drifted")
        for index, colour in enumerate(AKA_KINDS):
            row = 34 + index
            if int(cf[row, 0]) != 1 or abs(float(cf[row, 2]) - colour / 33.0) > 1e-6:
                raise ContractError(f"aka candidate {row} descriptor drifted")

    @classmethod
    def empty(cls) -> "VNextVisibleState":
        """Synthetic empty state for shape tests (not a legal real decision)."""
        tile = torch.zeros((TILE_KIND_COUNT, TILE_FEATURE_DIM), dtype=torch.float32)
        glob = torch.zeros((GLOBAL_FEATURE_DIM,), dtype=torch.float32)
        cf = torch.zeros((TILE_CANDIDATES, CANDIDATE_FEATURE_DIM), dtype=torch.float32)
        cf[:34, 2] = torch.arange(34, dtype=torch.float32) / 33.0
        cf[:34, 3] = 1.0
        for index, colour in enumerate(AKA_KINDS):
            row = 34 + index
            cf[row, 0] = 1.0
            cf[row, 1] = colour / 2.0
            cf[row, 2] = colour / 33.0
            cf[row, 3] = 1.0
        legal = torch.zeros((ACTION_SPACE,), dtype=torch.bool)
        legal[37] = True
        return cls(tile_features=tile, global_features=glob,
                   candidate_features=cf, legal_mask=legal, phase=0)

    def to(self, device: torch.device | str) -> "VNextVisibleState":
        return replace(self,
                       tile_features=self.tile_features.to(device),
                       global_features=self.global_features.to(device),
                       candidate_features=self.candidate_features.to(device),
                       legal_mask=self.legal_mask.to(device))


@dataclass(frozen=True)
class VNextTargets:
    """Trainer-only labels; never passed to the model forward."""

    action: Tensor
    final_rank: Tensor
    supervise_mask: Tensor
    weight: Tensor
    forced: Tensor | None = None

    def __post_init__(self) -> None:
        n = int(self.action.numel())
        _check_long_tensor("action", self.action, (n,), high=ACTION_SPACE)
        _check_long_tensor("final_rank", self.final_rank, (n,), high=4)
        _check_bool_tensor("supervise_mask", self.supervise_mask, (n,))
        _check_float_tensor("weight", self.weight, (n,))
        if bool((self.weight < 0).any()):
            raise ContractError("weight must be non-negative")
        if self.forced is not None:
            _check_bool_tensor("forced", self.forced, (n,))


@dataclass
class VNextModelBatch:
    """One shared event stream per row plus flattened decision positions.

    The model consumes exactly this object; labels live in :class:`VNextTargets`.
    ``event_type_id`` uses :data:`BOS_EVENT_ID` at position 0 and
    :data:`PAD_EVENT_ID` in padded slots.  ``prefix_token_offsets`` are inclusive
    token indices (BOS is 0); the cross-attention key mask is ``key <= offset``.
    """

    event_type_id: Tensor
    event_actor_rel: Tensor
    event_tile_kind: Tensor
    event_aka: Tensor
    event_tsumogiri: Tensor
    event_kyoku_pos: Tensor
    event_meld_tile: Tensor
    event_meld_aka: Tensor
    event_valid_mask: Tensor
    decision_to_sequence: Tensor
    prefix_token_offsets: Tensor
    phases: Tensor
    x_tile: Tensor
    x_global: Tensor
    candidate_features: Tensor
    legal_mask: Tensor

    def __post_init__(self) -> None:
        assert_forward_safe(tuple(self.__dataclass_fields__), owner="VNextModelBatch")
        if not isinstance(self.event_valid_mask, Tensor) or self.event_valid_mask.dtype is not torch.bool:
            raise ContractError("event_valid_mask must be bool")
        if self.event_valid_mask.dim() != 2:
            raise ContractError("event_valid_mask must be [B, L]")
        b, length = (int(v) for v in self.event_valid_mask.shape)
        if b <= 0 or length <= 0:
            raise ContractError("empty event batch")
        _check_long_tensor("event_type_id", self.event_type_id, (b, length),
                           high=EVENT_TYPE_VOCAB)
        _check_long_tensor("event_actor_rel", self.event_actor_rel, (b, length),
                           high=EVENT_ACTOR_VOCAB)
        _check_long_tensor("event_tile_kind", self.event_tile_kind, (b, length),
                           high=EVENT_TILE_VOCAB)
        _check_long_tensor("event_aka", self.event_aka, (b, length),
                           high=EVENT_AKA_VOCAB)
        _check_long_tensor("event_tsumogiri", self.event_tsumogiri, (b, length),
                           high=EVENT_TSUMOGIRI_VOCAB)
        _check_long_tensor("event_kyoku_pos", self.event_kyoku_pos, (b, length),
                           high=EVENT_KYOKU_POS_VOCAB)
        _check_long_tensor("event_meld_tile", self.event_meld_tile,
                           (b, length, EVENT_MELD_SLOTS), high=EVENT_TILE_VOCAB)
        _check_long_tensor("event_meld_aka", self.event_meld_aka,
                           (b, length, EVENT_MELD_SLOTS), high=EVENT_AKA_VOCAB)
        n = int(self.decision_to_sequence.numel())
        if n <= 0:
            raise ContractError("batch must contain at least one decision")
        _check_long_tensor("decision_to_sequence", self.decision_to_sequence, (n,),
                           high=b)
        _check_long_tensor("prefix_token_offsets", self.prefix_token_offsets, (n,),
                           high=length)
        _check_long_tensor("phases", self.phases, (n,), high=len(ActionPhase))
        _check_float_tensor("x_tile", self.x_tile,
                            (n, TILE_KIND_COUNT, TILE_FEATURE_DIM))
        _check_float_tensor("x_global", self.x_global, (n, GLOBAL_FEATURE_DIM))
        _check_float_tensor("candidate_features", self.candidate_features,
                            (n, TILE_CANDIDATES, CANDIDATE_FEATURE_DIM))
        _check_bool_tensor("legal_mask", self.legal_mask, (n, ACTION_SPACE))
        valid = self.event_valid_mask
        lengths = valid.sum(dim=1)
        if int(lengths.min()) <= 0:
            raise ContractError("every sequence needs at least the BOS token")
        expected = (torch.arange(length, device=valid.device).unsqueeze(0)
                    < lengths.unsqueeze(1))
        if not torch.equal(valid, expected):
            raise ContractError("event_valid_mask must be a left-aligned prefix")
        for row in range(n):
            seq = int(self.decision_to_sequence[row])
            offset = int(self.prefix_token_offsets[row])
            if offset >= length or not bool(valid[seq, offset]):
                raise ContractError(
                    f"decision {row} offset {offset} is not a valid token in sequence {seq}")
            if not bool(self.legal_mask[row].any()):
                raise ContractError(f"decision {row} has an empty legal mask")
            validate_mask_for_phase(
                tuple(bool(v) for v in self.legal_mask[row]),
                tuple(ActionPhase)[int(self.phases[row])])
        devices = {
            self.event_type_id.device, self.event_actor_rel.device,
            self.event_tile_kind.device, self.event_aka.device,
            self.event_tsumogiri.device, self.event_kyoku_pos.device,
            self.event_meld_tile.device, self.event_meld_aka.device,
            self.event_valid_mask.device, self.decision_to_sequence.device,
            self.prefix_token_offsets.device, self.phases.device, self.x_tile.device,
            self.x_global.device, self.candidate_features.device, self.legal_mask.device,
        }
        if len(devices) != 1:
            raise ContractError("all VNextModelBatch tensors must share one device")

    @property
    def batch_size(self) -> int:
        return int(self.event_valid_mask.size(0))

    @property
    def seq_len(self) -> int:
        return int(self.event_valid_mask.size(1))

    @property
    def n_decisions(self) -> int:
        return int(self.decision_to_sequence.numel())

    def to(self, device: torch.device | str) -> "VNextModelBatch":
        moved = {
            item.name: getattr(self, item.name).to(device)
            for item in fields(self)
        }
        return replace(self, **moved)

    def select_decisions(self, keep: Tensor) -> "VNextModelBatch":
        if keep.dtype is not torch.long or keep.dim() != 1:
            raise ContractError("keep must be a 1-D long tensor")
        if keep.numel() == 0:
            raise ContractError("keep must select at least one decision")
        return replace(
            self,
            decision_to_sequence=self.decision_to_sequence[keep],
            prefix_token_offsets=self.prefix_token_offsets[keep],
            phases=self.phases[keep], x_tile=self.x_tile[keep],
            x_global=self.x_global[keep],
            candidate_features=self.candidate_features[keep],
            legal_mask=self.legal_mask[keep],
        )


@dataclass
class VNextPolicyOutput:
    logits: Tensor
    masked_logits: Tensor
    logprobs: Tensor
    legal_mask: Tensor
    rank_logits: Tensor | None = None

    def validate(self) -> None:
        n = int(self.logits.size(0))
        _check_float_tensor("logits", self.logits, (n, ACTION_SPACE))
        if not isinstance(self.masked_logits, Tensor) \
                or self.masked_logits.dtype is not self.logits.dtype \
                or tuple(self.masked_logits.shape) != (n, ACTION_SPACE):
            raise ContractError("masked_logits must be a float [N,46] tensor")
        if not isinstance(self.logprobs, Tensor) \
                or self.logprobs.dtype is not self.logits.dtype \
                or tuple(self.logprobs.shape) != (n, ACTION_SPACE):
            raise ContractError("logprobs must be a float [N,46] tensor")
        _check_bool_tensor("legal_mask", self.legal_mask, (n, ACTION_SPACE))
        if not bool(self.legal_mask.any(dim=1).all()):
            raise ContractError("every decision needs a legal action")
        if not bool(torch.isfinite(self.masked_logits[self.legal_mask]).all()):
            raise ContractError("legal masked_logits are not finite")
        if not bool(torch.isinf(self.masked_logits[~self.legal_mask]).all()):
            raise ContractError("illegal masked_logits must be -inf")
        if not bool(torch.allclose(
                torch.logsumexp(self.logprobs, dim=-1),
                torch.zeros(n, dtype=self.logprobs.dtype,
                            device=self.logprobs.device), atol=1e-5, rtol=1e-5)):
            raise ContractError("logprobs do not normalize over legal actions")
        if self.rank_logits is not None:
            _check_float_tensor("rank_logits", self.rank_logits, (n, 4))


# ---------------------------------------------------------------------------
# GRP / PT / rollout contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GRPBoundary:
    """Public kyoku-start state used by the independent 24-way GRP."""

    match_id: str
    kyoku_index: int
    grand_kyoku: int
    honba: int
    kyotaku: int
    scores_abs: tuple[int, int, int, int]
    rules_hash: str
    seat_order_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.match_id, str) or not self.match_id:
            raise ContractError("match_id must be non-empty")
        for name in ("kyoku_index", "grand_kyoku", "honba", "kyotaku"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ContractError(f"{name} must be a non-negative int")
        if not isinstance(self.scores_abs, tuple) or len(self.scores_abs) != 4:
            raise ContractError("scores_abs must be a tuple of 4 ints")
        if not all(_is_int(v) for v in self.scores_abs):
            raise ContractError("scores_abs entries must be ints")
        if sum(self.scores_abs) + 1000 * self.kyotaku != 100_000:
            raise ContractError("scores_abs + kyotaku must total 100000")
        if not self.rules_hash:
            raise ContractError("rules_hash must be non-empty")
        if not isinstance(self.seat_order_hash, str):
            raise ContractError("seat_order_hash must be a string")

    @property
    def input_vector(self) -> tuple[float, ...]:
        scale = 100.0
        return (
            float(self.grand_kyoku), float(self.honba), float(self.kyotaku),
            *(float(score) / scale for score in self.scores_abs),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def permutation_index(rank_by_abs_seat: Sequence[int]) -> int:
    """Index of a final rank tuple in :data:`GRP_PERMUTATIONS` (0..23)."""
    value = tuple(int(v) for v in rank_by_abs_seat)
    if len(value) != 4 or sorted(value) != [0, 1, 2, 3]:
        raise ContractError(f"rank_by_abs_seat must be a permutation of 0..3: {value}")
    return GRP_PERMUTATION_INDEX[value]


def rank_marginals(prob_perm: Tensor) -> Tensor:
    """``[N, 4 seats, 4 ranks]`` marginals from ``[N, 24]`` joint probabilities."""
    if prob_perm.dim() != 2 or prob_perm.size(1) != 24:
        raise ContractError("prob_perm must be [N, 24]")
    table = torch.tensor(GRP_PERMUTATIONS, dtype=torch.long,
                         device=prob_perm.device)
    marginal = torch.zeros((prob_perm.size(0), 4, 4), dtype=prob_perm.dtype,
                           device=prob_perm.device)
    for seat in range(4):
        for rank in range(4):
            cols = table[:, seat] == rank
            marginal[:, seat, rank] = prob_perm[:, cols].sum(dim=1)
    return marginal


def phi_from_marginals(marginal: Tensor, pt_table: Sequence[int] = PT_TABLE_5D) -> Tensor:
    if marginal.dim() != 3 or tuple(marginal.shape[1:]) != (4, 4):
        raise ContractError("marginal must be [N, 4, 4]")
    if not torch.allclose(marginal.sum(dim=-1), torch.ones_like(marginal[..., 0]),
                          atol=1e-4, rtol=1e-4):
        raise ContractError("each seat marginal must sum to 1")
    if not torch.allclose(marginal.sum(dim=1), torch.ones_like(marginal[:, 0, :]),
                          atol=1e-4, rtol=1e-4):
        raise ContractError("each rank marginal must sum to 1 across seats")
    table = torch.tensor(tuple(pt_table), dtype=marginal.dtype,
                         device=marginal.device)
    return (marginal * table.view(1, 1, 4)).sum(dim=-1)


def expected_pt_from_rank_probs(probs: Tensor,
                                pt_table: Sequence[int] = PT_TABLE_5D) -> Tensor:
    if probs.dim() != 2 or probs.size(1) != 4:
        raise ContractError("probs must be [N, 4]")
    table = torch.tensor(tuple(pt_table), dtype=probs.dtype, device=probs.device)
    return (probs * table).sum(dim=-1)


def relative_pt_from_rf_pt(rf_pt: float) -> float:
    """Strict zero-sum 1v3 identity: RF_PT - mean(3 opponents) = (4/3) RF_PT."""
    return (4.0 / 3.0) * float(rf_pt)


@dataclass(frozen=True)
class GuardSpec:
    """Post-policy execution transformation; all fields are explicit and versioned."""

    temperature: float = 1.0
    top_p: float = 1.0
    epsilon_mix: float = 0.0
    forced_action: int | None = None
    effective_mask: tuple[bool, ...] | None = None
    version: str = "guard/1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ContractError("temperature must be finite and > 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ContractError("top_p must be in (0, 1]")
        if not 0.0 <= self.epsilon_mix <= 1.0:
            raise ContractError("epsilon_mix must be in [0, 1]")
        if self.forced_action is not None and not 0 <= self.forced_action < ACTION_SPACE:
            raise ContractError("forced_action outside 0..45")
        if self.effective_mask is not None:
            if len(self.effective_mask) != ACTION_SPACE:
                raise ContractError("effective_mask must have 46 entries")
            if not any(self.effective_mask):
                raise ContractError("effective_mask cannot be empty")


@dataclass
class RLDecisionRow:
    """One learner decision with the **actual executed** behavior distribution."""

    match_id: str
    kyoku_id: int
    seat: int
    decision_id: str
    decision_order: int
    phase: int
    observation_ref: str
    prefix_token_offset: int
    legal_mask_46: tuple[bool, ...]
    effective_mask_46: tuple[bool, ...]
    raw_logits_46: tuple[float, ...]
    behavior_probs_46: tuple[float, ...]
    executed_action: int
    behavior_logprob: float
    guard_reason: str | None
    forced: bool
    policy_sha: str
    opponent_shas_by_seat: tuple[str, str, str, str]
    rules_hash: str
    grp_sha: str
    reward_spec_id: str
    pt_table_id: str
    kyoku_done: bool
    match_done: bool
    contract_id: str = RL_CONTRACT_ID

    def __post_init__(self) -> None:
        if self.contract_id != RL_CONTRACT_ID:
            raise ContractError("unknown RLDecisionRow contract_id")
        for name in ("match_id", "decision_id", "observation_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractError(f"{name} must be non-empty")
        if not _is_int(self.seat) or not 0 <= self.seat <= 3:
            raise ContractError("seat must be int 0..3")
        if not _is_int(self.phase) or not 0 <= self.phase < len(ActionPhase):
            raise ContractError("phase outside ActionPhase")
        if not _is_int(self.executed_action) or not 0 <= self.executed_action < ACTION_SPACE:
            raise ContractError("executed_action outside 0..45")
        for name in ("legal_mask_46", "effective_mask_46"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(value) != ACTION_SPACE \
                    or not all(isinstance(v, bool) for v in value):
                raise ContractError(f"{name} must be 46 bools")
        if not self.effective_mask_46[self.executed_action]:
            raise ContractError("executed_action is not in effective_mask_46")
        if not self.legal_mask_46[self.executed_action]:
            raise ContractError("executed_action is not in legal_mask_46")
        for name in ("raw_logits_46", "behavior_probs_46"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(value) != ACTION_SPACE:
                raise ContractError(f"{name} must have 46 entries")
            if not all(isinstance(v, (int, float)) and math.isfinite(float(v))
                       for v in value):
                raise ContractError(f"{name} must be finite")
        probs = self.behavior_probs_46
        if abs(sum(probs) - 1.0) > 1e-5:
            raise ContractError(f"behavior_probs_46 sum to {sum(probs)}, expected 1")
        if any(p < 0 for p in probs):
            raise ContractError("behavior_probs_46 must be non-negative")
        if any(p > 0 and not self.legal_mask_46[i] for i, p in enumerate(probs)):
            raise ContractError("behavior_probs_46 puts mass on an illegal action")
        if probs[self.executed_action] <= 0:
            raise ContractError("executed_action must have positive behavior mass")
        expected = math.log(probs[self.executed_action])
        if abs(self.behavior_logprob - expected) > 1e-5:
            raise ContractError(
                f"stored behavior_logprob {self.behavior_logprob} != log(mu) {expected}")
        if self.forced and probs[self.executed_action] != 1.0:
            raise ContractError("forced action must have behavior probability 1")
        if not isinstance(self.forced, bool):
            raise ContractError("forced must be bool")
        for name in ("policy_sha", "rules_hash", "grp_sha", "reward_spec_id",
                     "pt_table_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractError(f"{name} must be non-empty")
        if len(self.opponent_shas_by_seat) != 4:
            raise ContractError("opponent_shas_by_seat must have 4 entries")

    @classmethod
    def from_probabilities(cls, **kwargs) -> "RLDecisionRow":
        probs = tuple(float(v) for v in kwargs["behavior_probs_46"])
        action = int(kwargs["executed_action"])
        kwargs["behavior_probs_46"] = probs
        kwargs["behavior_logprob"] = math.log(probs[action])
        return cls(**kwargs)


@dataclass
class RLMatchHeader:
    match_id: str
    wall_seed: int
    seat_assignments: tuple[str, str, str, str]
    rules_hash: str
    engine_hash: str
    deployed_config_sha: str
    train_eval_pool: str
    contract_id: str = RL_CONTRACT_ID

    def __post_init__(self) -> None:
        if not self.match_id or not self.rules_hash or not self.engine_hash:
            raise ContractError("RLMatchHeader strings must be non-empty")
        if not _is_int(self.wall_seed):
            raise ContractError("wall_seed must be int")
        if len(self.seat_assignments) != 4:
            raise ContractError("seat_assignments must have 4 entries")
        if self.train_eval_pool not in {"train", "smoke", "trend", "select", "confirm"}:
            raise ContractError("unknown train_eval_pool")


@dataclass
class RLKyokuBoundary:
    match_id: str
    kyoku_id: int
    score_before: tuple[int, int, int, int]
    score_after: tuple[int, int, int, int]
    phi: tuple[float, float, float, float]
    grp_sha: str
    pt_table_id: str

    def __post_init__(self) -> None:
        if len(self.score_before) != 4 or len(self.score_after) != 4:
            raise ContractError("score tuples must have 4 entries")
        if len(self.phi) != 4:
            raise ContractError("phi must have 4 entries")
        if abs(sum(self.phi)) > 1e-4:
            raise ContractError("zero-sum PT table requires sum(phi) == 0")
        if self.pt_table_id not in PT_TABLES:
            raise ContractError("unknown pt_table_id")


@dataclass
class RLMatchOutcome:
    match_id: str
    final_scores: tuple[int, int, int, int]
    final_rank_by_seat: tuple[int, int, int, int]
    final_pt_by_seat: tuple[float, float, float, float]
    pt_table_id: str
    match_done: bool = True

    def __post_init__(self) -> None:
        if len(self.final_scores) != 4:
            raise ContractError("final_scores must have 4 entries")
        if sorted(self.final_rank_by_seat) != [0, 1, 2, 3]:
            raise ContractError("final_rank_by_seat must be a permutation")
        if self.pt_table_id not in PT_TABLES:
            raise ContractError("unknown pt_table_id")
        if not self.match_done:
            raise ContractError("RLMatchOutcome requires match_done=True")


# ---------------------------------------------------------------------------
# Stable contract manifest
# ---------------------------------------------------------------------------


def contract_manifest(config: VNextConfig | None = None) -> dict:
    config = config or VNextConfig()
    return {
        "model_arch_id": MODEL_ARCH_ID,
        "encoder_contract_id": ENCODER_CONTRACT_ID,
        "state_contract_id": STATE_CONTRACT_ID,
        "grp_contract_id": GRP_CONTRACT_ID,
        "rl_contract_id": RL_CONTRACT_ID,
        "schema_version": SCHEMA_VERSION,
        "action_space": ACTION_SPACE,
        "action_map_hash": ACTION_MAP_HASH,
        "tile_feature_dim": TILE_FEATURE_DIM,
        "tile_feature_names": list(TILE_FEATURE_NAMES),
        "global_feature_dim": GLOBAL_FEATURE_DIM,
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
        "candidate_feature_dim": CANDIDATE_FEATURE_DIM,
        "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
        "bos_event_id": BOS_EVENT_ID,
        "pad_event_id": PAD_EVENT_ID,
        "event_type_vocab": EVENT_TYPE_VOCAB,
        "event_meld_slots": EVENT_MELD_SLOTS,
        "grp_permutations": [list(perm) for perm in GRP_PERMUTATIONS],
        "pt_tables": {key: list(value) for key, value in PT_TABLES.items()},
        "model_config": config.as_dict(),
        "model_architecture_hash": config.architecture_hash,
    }


def contract_hash(config: VNextConfig | None = None) -> str:
    return _sha(contract_manifest(config))


__all__ = [
    "ACTION_SPACE", "BOS_EVENT_ID", "CANDIDATE_FEATURE_DIM", "CANDIDATE_FEATURE_NAMES",
    "ContractError", "EVENT_ACTOR_VOCAB", "EVENT_AKA_VOCAB",
    "EVENT_KYOKU_POS_VOCAB", "EVENT_MELD_SLOTS", "EVENT_TILE_VOCAB",
    "EVENT_TSUMOGIRI_VOCAB",
    "EVENT_TYPE_VOCAB", "ENCODER_CONTRACT_ID", "GLOBAL_FEATURE_DIM",
    "GLOBAL_FEATURE_NAMES", "GRP_CONTRACT_ID", "GRPBoundary",
    "GRP_PERMUTATIONS", "GRP_PERMUTATION_INDEX", "GuardSpec", "MODEL_ARCH_ID",
    "OTHER_ACTIONS", "PAD_EVENT_ID", "PT_TABLES", "PT_TABLE_5D", "PT_TABLE_5D_ID",
    "PT_TABLE_7D", "PT_TABLE_7D_ID", "RLDecisionRow", "RLKyokuBoundary",
    "RLMatchHeader", "RLMatchOutcome", "RL_CONTRACT_ID", "STATE_CONTRACT_ID",
    "TILE_CANDIDATES", "TILE_FEATURE_DIM", "TILE_FEATURE_NAMES", "VNextConfig",
    "VNextDecisionIndex", "VNextModelBatch", "VNextPolicyOutput", "VNextTargets",
    "VNextVisibleState", "assert_forward_safe", "contract_hash", "contract_manifest",
    "expected_pt_from_rank_probs", "permutation_index", "phi_from_marginals",
    "rank_marginals", "relative_pt_from_rf_pt",
]
