"""Tensor batch contract for the PR-13 MVP-S model.

:class:`ModelBatch` is the **only** object ``Policy.forward`` consumes.  It is a
PR-13-owned, synthetic-first interface: PR-11's future ``data/encode.py`` may build
it, but nothing here imports PR-11/PR-12 code, so the interface can be replaced
without touching the network.

Visibility contract (mirrors ``DecisionInput`` and the roadmap Recipe E)
-----------------------------------------------------------------------
* one row = one POV event stream, left-aligned and right-padded;
* ``event_*`` tensors hold already-projected observable events (opponent hidden
  tiles are UNKNOWN, never a real tile);
* ``state_features`` has one row per **decision** (``N`` rows across the batch) and
  is injected only at that decision's token;
* ``legal_mask`` is the 46-wide schema mask for that decision;
* labels/targets/rank/oracle data are deliberately **absent** — the trainer keeps
  them in a separate object (PR-14) so no target can enter forward by accident.

Padding policy: ``padding_mask`` must be a True-prefix per row (no interior holes).
That is stricter than the transformer (which also accepts holey key masks) and
guarantees that a shorter sequence is a real prefix of a longer one, which makes
prefix invariance testable and prevents accidental mid-sequence truncation.
Sequence length is whatever the caller provides: this class never truncates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from ..schemas.action_v3 import (
    ACTION_SPACE,
    AKA_KINDS,
    PHASE_LEGAL_ACTIONS,
    SCHEMA_VERSION,
    TILE_KIND_NAMES,
    UNKNOWN_TILE_KIND,
    ActionPhase,
)
from ..schemas.event_v3 import EVENT_TYPE_IDS
from .event_embedding import (
    ACTOR_REL_VOCAB,
    AKA_VOCAB,
    EVENT_TYPE_VOCAB,
    KYOKU_POS_VOCAB,
    PHASE_CLASSES,
    PHASE_VOCAB,
    TILE_KIND_VOCAB,
    TSUMOGIRI_VOCAB,
)

#: Stable phase order used by every tensor in this package (schema enum order).
PHASE_ORDER: tuple[ActionPhase, ...] = (
    ActionPhase.SELF_TURN,
    ActionPhase.CALL_RESPONSE,
    ActionPhase.KAN_SELECTION,
)
PHASE_INDEX: dict[ActionPhase, int] = {phase: index for index, phase in enumerate(PHASE_ORDER)}
PHASE_INDEX_TO_ENUM: tuple[ActionPhase, ...] = PHASE_ORDER

#: Legal action index sets per phase order entry (from the frozen schema contract).
PHASE_LEGAL_INDEX_SETS: tuple[frozenset[int], ...] = tuple(
    frozenset(PHASE_LEGAL_ACTIONS[phase]) for phase in PHASE_ORDER
)

#: Event ids resolved from the frozen schema, never hardcoded here.
TSUMO_EVENT_TYPE: int = EVENT_TYPE_IDS["tsumo"]
ANKAN_EVENT_TYPE: int = EVENT_TYPE_IDS["ankan"]
DAIMINKAN_EVENT_TYPE: int = EVENT_TYPE_IDS["daiminkan"]
KAKAN_EVENT_TYPE: int = EVENT_TYPE_IDS["kakan"]

#: An aka embedding index ``1..3`` is valid only on the matching five kind.
AKA_INDEX_TO_KIND: tuple[tuple[int, int], ...] = tuple(
    (colour + 1, kind) for colour, kind in enumerate(AKA_KINDS)
)

#: Structure-level guard: a batch may never carry a label/target/oracle field.
FORBIDDEN_BATCH_FIELD_WORDS: tuple[str, ...] = (
    "label", "target", "oracle", "rank_final", "expert_action", "supervision",
    "censor",
)

EVENT_TENSOR_FIELDS: tuple[str, ...] = (
    "event_type_id", "event_actor_rel", "event_tile_kind", "event_aka",
    "event_tsumogiri", "event_phase", "event_kyoku_pos",
)

#: Event-feature vocabularies, in field order (PAD slot included where applicable).
EVENT_TENSOR_VOCABS: tuple[int, ...] = (
    EVENT_TYPE_VOCAB, ACTOR_REL_VOCAB, TILE_KIND_VOCAB, AKA_VOCAB,
    TSUMOGIRI_VOCAB, PHASE_VOCAB, KYOKU_POS_VOCAB,
)


class ModelBatchError(ValueError):
    """A ModelBatch violates the PR-13 tensor contract."""


def _require_tensor(name: str, value, shape: tuple[int, ...], *, dtype=None) -> None:
    if not isinstance(value, Tensor):
        raise ModelBatchError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ModelBatchError(f"{name} shape {tuple(value.shape)} != expected {shape}")
    if dtype is not None and value.dtype is not dtype:
        raise ModelBatchError(f"{name} dtype {value.dtype} != {dtype}")


def _check_long_range(name: str, value: Tensor, low: int = 0,
                      high: int | None = None) -> None:
    if value.numel() == 0:
        raise ModelBatchError(f"{name} must not be empty")
    minimum = int(value.min())
    maximum = int(value.max())
    if minimum < low or (high is not None and maximum >= high):
        upper = (high - 1) if high is not None else "inf"
        raise ModelBatchError(
            f"{name} values must be in [{low}, {upper}], got [{minimum}, {maximum}]"
        )


@dataclass
class ModelBatch:
    """POV event/state/decision tensors for one forward pass.

    Event tensors are ``[B, L]`` ``torch.long`` (padding mask ``[B, L]`` bool).
    Decisions are flattened across the batch: row ``i`` points at
    ``(decision_batch_index[i], decision_time_index[i])``.  ``state_features`` is
    ``[N, STATE_FEATURE_DIM]`` (see :mod:`riichiformer.models.state_encoder`).
    """

    # -- event stream (padded, left-aligned) ---------------------------------
    event_type_id: Tensor
    event_actor_rel: Tensor
    event_tile_kind: Tensor
    event_aka: Tensor
    event_tsumogiri: Tensor
    event_phase: Tensor
    event_kyoku_pos: Tensor
    padding_mask: Tensor

    # -- decision positions (N rows, flattened over B) -----------------------
    state_features: Tensor
    decision_batch_index: Tensor
    decision_time_index: Tensor
    legal_mask: Tensor
    decision_phase: Tensor

    #: Frozen schema version this batch was encoded against (metadata only; never
    #: an input feature and never passed to ``forward`` as a tensor).
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ModelBatchError(
                f"schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}"
            )
        if not isinstance(self.padding_mask, Tensor):
            raise ModelBatchError(
                f"padding_mask must be a torch.Tensor, got "
                f"{type(self.padding_mask).__name__}"
            )
        _require_tensor("padding_mask", self.padding_mask,
                        tuple(self.padding_mask.shape), dtype=torch.bool)
        if self.padding_mask.dim() != 2:
            raise ModelBatchError(
                f"padding_mask must be [B, L], got {tuple(self.padding_mask.shape)}"
            )
        batch_size, seq_len = (int(size) for size in self.padding_mask.shape)
        if batch_size <= 0 or seq_len <= 0:
            raise ModelBatchError(f"empty batch shape {batch_size, seq_len}")
        shape = (batch_size, seq_len)
        for name, vocab in zip(EVENT_TENSOR_FIELDS, EVENT_TENSOR_VOCABS):
            value = getattr(self, name)
            _require_tensor(name, value, shape, dtype=torch.long)
            _check_long_range(name, value, high=vocab)
        lengths = self.padding_mask.sum(dim=1)
        if int(lengths.min()) == 0:
            raise ModelBatchError("every row needs at least one real event token")
        expected_mask = (
            torch.arange(seq_len, device=self.padding_mask.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if not torch.equal(self.padding_mask, expected_mask):
            raise ModelBatchError(
                "padding_mask must be a left-aligned True prefix per row "
                "(no interior holes, no left padding); got non-prefix mask"
            )

        n_decisions = int(self.decision_batch_index.numel())
        if n_decisions <= 0:
            raise ModelBatchError("batch must contain at least one decision")
        _require_tensor("decision_batch_index", self.decision_batch_index, (n_decisions,),
                        dtype=torch.long)
        _require_tensor("decision_time_index", self.decision_time_index, (n_decisions,),
                        dtype=torch.long)
        _require_tensor("decision_phase", self.decision_phase, (n_decisions,),
                        dtype=torch.long)
        if self.state_features.dim() != 2 or self.state_features.size(0) != n_decisions:
            raise ModelBatchError(
                f"state_features must be [N={n_decisions}, F], got "
                f"{tuple(self.state_features.shape)}"
            )
        if not self.state_features.is_floating_point():
            raise ModelBatchError(
                f"state_features must be floating point, got {self.state_features.dtype}"
            )
        if not bool(torch.isfinite(self.state_features).all()):
            raise ModelBatchError("state_features contains NaN/Inf")
        _require_tensor("legal_mask", self.legal_mask, (n_decisions, ACTION_SPACE),
                        dtype=torch.bool)

        device = self.padding_mask.device
        for name in ("state_features", "decision_batch_index", "decision_time_index",
                     "legal_mask", "decision_phase"):
            value = getattr(self, name)
            if value.device != device:
                raise ModelBatchError(
                    f"{name} device {value.device} != padding_mask device {device}"
                )

        _check_long_range("decision_batch_index", self.decision_batch_index,
                          high=batch_size)
        _check_long_range("decision_time_index", self.decision_time_index,
                          high=seq_len)
        _check_long_range("decision_phase", self.decision_phase, high=PHASE_CLASSES)

        # Decisions must point at real (unpadded) tokens, and twice at the same token
        # would mean two different states share one residual slot: forbid it.
        at_valid_token = self.padding_mask[self.decision_batch_index,
                                           self.decision_time_index]
        if not bool(at_valid_token.all()):
            raise ModelBatchError("decision indices must point at non-padded event tokens")
        offsets = self.flat_decision_offsets
        if torch.unique(offsets).numel() != n_decisions:
            raise ModelBatchError("duplicate (batch, time) decision positions")

        # Phase-aware mask validation against the frozen 46-action contract.
        for index in range(n_decisions):
            phase = int(self.decision_phase[index])
            mask_row = self.legal_mask[index]
            if not bool(mask_row.any()):
                raise ModelBatchError(f"legal_mask row {index} has no legal action")
            legal = frozenset(int(i) for i in torch.nonzero(mask_row, as_tuple=False).flatten())
            unexpected = legal - PHASE_LEGAL_INDEX_SETS[phase]
            if unexpected:
                raise ModelBatchError(
                    f"legal_mask row {index} has actions {sorted(unexpected)} illegal "
                    f"in phase {PHASE_ORDER[phase].value}"
                )
        self.validate_visibility()

    def validate_visibility(self) -> None:
        """Enforce POV visibility on the projected event ids (fail closed).

        The model consumes exactly one ``tile_kind`` + ``aka`` per token, so the checks
        are token-local.  Aggregated physical-tile ledgers (``consumed``/haipai counts)
        are PR-09/PR-11 responsibilities and are deliberately not stored here; if a
        future encoder adds consumed ids, those ids must obey the same rules.

        Rules:

        1. ``tile_kind == UNKNOWN`` carries no aka (``aka == 0`` in this encoding);
        2. an aka index 1/2/3 is valid only on 5m/5p/5s respectively (schema kinds
           4/13/22) and therefore fixes the kind — a red flag can never travel with a
           different kind or with UNKNOWN;
        3. an opponent ``tsumo`` is concealed: ``actor_rel != 0`` implies
           ``tile_kind == UNKNOWN`` and ``aka == 0``;
        4. an opponent ``ankan`` is concealed (the schema's ``consumed`` is
           rule-dependent): same requirement.  ``daiminkan``/``kakan`` keep their
           public called/added tile, so a known tile is valid there;
        5. padding slots are ignored (they are not observable events);
        6. no ``ModelBatch`` dataclass field may be a label/target/oracle field.

        ``__post_init__`` runs this, and ``Policy.forward`` calls it again so an
        in-place mutation after construction cannot bypass the check.
        """
        names = set(self.__dataclass_fields__)  # type: ignore[attr-defined]
        forbidden = {
            name for name in names
            if any(word in name for word in FORBIDDEN_BATCH_FIELD_WORDS)
        }
        if forbidden:
            raise ModelBatchError(
                f"ModelBatch must not carry target/label fields: {sorted(forbidden)}"
            )

        visible = self.padding_mask
        event_type = self.event_type_id
        actor = self.event_actor_rel
        tile = self.event_tile_kind
        aka = self.event_aka

        aka_or_unknown = (tile == UNKNOWN_TILE_KIND) & (aka != 0)
        aka_kind_ok = (aka == 0)
        for aka_index, kind in AKA_INDEX_TO_KIND:
            aka_kind_ok = aka_kind_ok | ((aka == aka_index) & (tile == kind))
        aka_kind_bad = ~aka_kind_ok
        opponent_private = (event_type == TSUMO_EVENT_TYPE) | (
            event_type == ANKAN_EVENT_TYPE
        )
        opponent_private_bad = opponent_private & (actor != 0) & (
            (tile != UNKNOWN_TILE_KIND) | (aka != 0)
        )

        bad = (aka_or_unknown | aka_kind_bad | opponent_private_bad) & visible
        if not bool(bad.any()):
            return
        # Only compute the detailed masks on the failure path (one extra device sync
        # per check) so the hot path stays a single reduction.
        if bool((aka_or_unknown & visible).any()):
            raise ModelBatchError(
                "tile_kind UNKNOWN (34) cannot carry an aka flag; use aka=0 (none)"
            )
        if bool((opponent_private_bad & visible).any()):
            raise ModelBatchError(
                "opponent tsumo/ankan tokens must hide the tile: expected "
                f"tile_kind={UNKNOWN_TILE_KIND} (UNKNOWN) and aka=0"
            )
        index, row = (
            int(value) for value in
            torch.nonzero(aka_kind_bad & visible, as_tuple=False)[0].tolist()
        )
        kind = int(tile[index, row])
        kind_name = TILE_KIND_NAMES[kind] if kind < len(TILE_KIND_NAMES) else "UNKNOWN"
        raise ModelBatchError(
            f"aka={int(aka[index, row])} is invalid for tile_kind={kind} "
            f"({kind_name}); aka 1/2/3 must be exactly 5m/5p/5s (kinds 4/13/22) "
            "and aka=0 means none"
        )

    # -------------------------------------------------------------- properties
    @property
    def batch_size(self) -> int:
        return int(self.padding_mask.size(0))

    @property
    def seq_len(self) -> int:
        return int(self.padding_mask.size(1))

    @property
    def lengths(self) -> Tensor:
        return self.padding_mask.sum(dim=1)

    @property
    def n_decisions(self) -> int:
        return int(self.decision_batch_index.numel())

    @property
    def decision_indices(self) -> Tensor:
        """``[N, 2]`` ``(batch_idx, token_idx)`` view (new tensor, cheap)."""
        return torch.stack((self.decision_batch_index, self.decision_time_index), dim=1)

    @property
    def flat_decision_offsets(self) -> Tensor:
        """``[N]`` offsets into a flattened ``B * L`` token axis."""
        return self.decision_batch_index * self.seq_len + self.decision_time_index

    @property
    def decision_phases(self) -> tuple[ActionPhase, ...]:
        return tuple(PHASE_ORDER[int(index)] for index in self.decision_phase)

    # ------------------------------------------------------------------ helpers
    def to(self, device: torch.device | str) -> "ModelBatch":
        """Return a copy with every tensor moved to ``device`` (metadata kept)."""
        moved = {
            field: getattr(self, field).to(device)
            for field in self.__dataclass_fields__  # type: ignore[attr-defined]
            if isinstance(getattr(self, field), Tensor)
        }
        moved["schema_version"] = self.schema_version
        return replace(self, **moved)

    def select_rows(self, keep: Tensor) -> "ModelBatch":
        """Subset the batch by row indices, keeping only those decisions.

        Useful for the POV-independence test (batched vs per-sample forward).  The
        event tensors are sliced as-is (padding stays a prefix).
        """
        if keep.dtype is not torch.long or keep.dim() != 1:
            raise ModelBatchError("keep must be a 1-D long tensor of batch rows")
        if int(keep.min()) < 0 or int(keep.max()) >= self.batch_size:
            raise ModelBatchError("keep row index out of range")
        device = self.decision_batch_index.device
        keep = keep.to(device)
        remap = torch.full((self.batch_size,), -1, dtype=torch.long, device=device)
        remap[keep] = torch.arange(keep.numel(), device=device)
        decision_keep = remap[self.decision_batch_index] >= 0
        new_batch_index = remap[self.decision_batch_index[decision_keep]]
        tensors = {
            field: getattr(self, field)[keep]
            for field in EVENT_TENSOR_FIELDS
        }
        tensors["padding_mask"] = self.padding_mask[keep]
        tensors["state_features"] = self.state_features[decision_keep]
        tensors["decision_batch_index"] = new_batch_index
        tensors["decision_time_index"] = self.decision_time_index[decision_keep]
        tensors["legal_mask"] = self.legal_mask[decision_keep]
        tensors["decision_phase"] = self.decision_phase[decision_keep]
        return ModelBatch(schema_version=self.schema_version, **tensors)


__all__ = [
    "EVENT_TENSOR_FIELDS", "EVENT_TENSOR_VOCABS", "ModelBatch", "ModelBatchError",
    "PHASE_INDEX", "PHASE_INDEX_TO_ENUM", "PHASE_LEGAL_INDEX_SETS", "PHASE_ORDER",
]
