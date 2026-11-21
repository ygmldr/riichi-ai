"""Observable-event embedding for the PR-13 MVP-S local transformer.

Every token is one :class:`riichiformer.schemas.event_v3.ObservableEventV3` in the POV
event stream.  Seven categorical features are embedded and **summed** into one
``d_model`` vector (no concatenation, so the transformer width stays ``d_model``):

====================  ===========================================================
feature               vocabulary
====================  ===========================================================
``event_type_id``     schema ``EVENT_TYPE_IDS`` 0..15, plus ``PAD`` = 16
``actor_rel``         0=self, 1=shimo, 2=toimen, 3=kami, 4=none/unknown, ``PAD`` = 5
``tile_kind``         0..33 + UNKNOWN=34 (schema ``TILE_KIND_COUNT`` + 1)
``aka``               ``0`` = no red five, ``1..3`` = red colour 0/1/2
``tsumogiri``         0/1 (only meaningful on ``dahai``; 0 elsewhere)
``phase``             schema ``ActionPhase`` order 0/1/2, plus ``PAD`` = 3
``kyoku_pos``         0-based ordinal of the event **inside its current kyoku**,
                      bucket 0..511; overflowing this range is an error, never a
                      silent clamp/truncation
====================  ===========================================================

The padded slots are zeroed after the sum (they may carry any in-range id; a
``PAD`` id is recommended for readability).  Padding is therefore a hard zero and
never reaches attention as a non-zero value.

This module is self-contained and synthetic-first: it consumes integer tensors only
and has no dependency on the future PR-11 ``data/encode.py``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

#: Schema event type ids 0..15 (``event_v3.EVENT_TYPE_IDS``).
EVENT_TYPE_CLASSES: int = 16
EVENT_TYPE_PAD: int = EVENT_TYPE_CLASSES
EVENT_TYPE_VOCAB: int = EVENT_TYPE_CLASSES + 1

#: Relative seats 0..3, 4 = none/unknown (schema ``ActorRel``).
ACTOR_REL_CLASSES: int = 5
ACTOR_REL_PAD: int = ACTOR_REL_CLASSES
ACTOR_REL_VOCAB: int = ACTOR_REL_CLASSES + 1

#: Schema tile kinds 0..33 plus UNKNOWN=34; UNKNOWN doubles as the pad tile.
TILE_KIND_VOCAB: int = 35

#: ``0`` = normal tile / no aka, ``1..3`` = red five of colour 0/1/2.
AKA_VOCAB: int = 4

TSUMOGIRI_VOCAB: int = 2

#: Schema ``ActionPhase`` order 0..2, plus ``PAD`` = 3.
PHASE_CLASSES: int = 3
PHASE_PAD: int = PHASE_CLASSES
PHASE_VOCAB: int = PHASE_CLASSES + 1

#: Position of the event inside its kyoku (0-based event ordinal).
KYOKU_POS_VOCAB: int = 512


class EventEmbeddingError(ValueError):
    """Malformed event-id tensor (fail closed; never silently clamp)."""


def _check_tensor(name: str, value: Tensor, shape: tuple[int, int]) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.shape != shape:
        raise EventEmbeddingError(
            f"{name} shape {tuple(value.shape)} != expected {shape}"
        )
    if value.dtype != torch.long:
        raise EventEmbeddingError(f"{name} dtype must be torch.long, got {value.dtype}")
    if value.numel() == 0:
        raise EventEmbeddingError(f"{name} must not be empty")


def _check_bounds(name: str, value: Tensor, vocab: int) -> None:
    # aminmax is a single kernel + one host sync instead of two; the bound check is
    # only used on the standalone/debug path (ModelBatch already guards the model).
    minimum, maximum = (int(bound) for bound in torch.aminmax(value))
    if minimum < 0 or maximum >= vocab:
        raise EventEmbeddingError(
            f"{name} values must be in [0, {vocab - 1}], got [{minimum}, {maximum}]"
        )


class EventEmbedding(nn.Module):
    """Sum of seven categorical embeddings, masked to zero on padding."""

    def __init__(self, d_model: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.d_model = int(d_model)
        self.event_type = nn.Embedding(EVENT_TYPE_VOCAB, d_model)
        self.actor_rel = nn.Embedding(ACTOR_REL_VOCAB, d_model)
        self.tile_kind = nn.Embedding(TILE_KIND_VOCAB, d_model)
        self.aka = nn.Embedding(AKA_VOCAB, d_model)
        self.tsumogiri = nn.Embedding(TSUMOGIRI_VOCAB, d_model)
        self.phase = nn.Embedding(PHASE_VOCAB, d_model)
        self.kyoku_pos = nn.Embedding(KYOKU_POS_VOCAB, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.vocab_sizes: dict[str, int] = {
            "event_type": EVENT_TYPE_VOCAB,
            "actor_rel": ACTOR_REL_VOCAB,
            "tile_kind": TILE_KIND_VOCAB,
            "aka": AKA_VOCAB,
            "tsumogiri": TSUMOGIRI_VOCAB,
            "phase": PHASE_VOCAB,
            "kyoku_pos": KYOKU_POS_VOCAB,
        }

    def forward(
        self,
        event_type_id: Tensor,
        event_actor_rel: Tensor,
        event_tile_kind: Tensor,
        event_aka: Tensor,
        event_tsumogiri: Tensor,
        event_phase: Tensor,
        event_kyoku_pos: Tensor,
        padding_mask: Tensor,
        *,
        check_ids: bool = True,
    ) -> Tensor:
        """Return ``[B, L, d_model]`` embeddings; padded positions are exact zeros.

        All id tensors are ``torch.long`` ``[B, L]``; ``padding_mask`` is a bool
        ``[B, L]`` with ``True`` for real tokens.  Out-of-vocabulary ids raise
        :class:`EventEmbeddingError` instead of wrapping or clamping.
        ``check_ids=False`` skips the (device-synchronising) bound check; the model
        path uses it because :class:`ModelBatch` has already validated every id.
        """
        if not isinstance(padding_mask, Tensor) or padding_mask.dtype is not torch.bool:
            raise EventEmbeddingError("padding_mask must be a bool torch.Tensor")
        if padding_mask.dim() != 2:
            raise EventEmbeddingError(
                f"padding_mask must be [B, L], got {tuple(padding_mask.shape)}"
            )
        shape = (padding_mask.size(0), padding_mask.size(1))
        if shape[0] <= 0 or shape[1] <= 0:
            raise EventEmbeddingError(f"empty event batch {shape}")
        named_ids = (
            ("event_type_id", event_type_id, EVENT_TYPE_VOCAB),
            ("event_actor_rel", event_actor_rel, ACTOR_REL_VOCAB),
            ("event_tile_kind", event_tile_kind, TILE_KIND_VOCAB),
            ("event_aka", event_aka, AKA_VOCAB),
            ("event_tsumogiri", event_tsumogiri, TSUMOGIRI_VOCAB),
            ("event_phase", event_phase, PHASE_VOCAB),
            ("event_kyoku_pos", event_kyoku_pos, KYOKU_POS_VOCAB),
        )
        for name, value, vocab in named_ids:
            _check_tensor(name, value, shape)
            if check_ids:
                _check_bounds(name, value, vocab)

        x = (
            self.event_type(event_type_id)
            + self.actor_rel(event_actor_rel)
            + self.tile_kind(event_tile_kind)
            + self.aka(event_aka)
            + self.tsumogiri(event_tsumogiri)
            + self.phase(event_phase)
            + self.kyoku_pos(event_kyoku_pos)
        )
        x = self.norm(x)
        x = self.dropout(x)
        return x * padding_mask.unsqueeze(-1).to(x.dtype)


__all__ = [
    "ACTOR_REL_CLASSES", "ACTOR_REL_PAD", "ACTOR_REL_VOCAB", "AKA_VOCAB",
    "EVENT_TYPE_CLASSES", "EVENT_TYPE_PAD", "EVENT_TYPE_VOCAB", "EventEmbedding",
    "EventEmbeddingError", "KYOKU_POS_VOCAB", "PHASE_CLASSES", "PHASE_PAD",
    "PHASE_VOCAB", "TILE_KIND_VOCAB", "TSUMOGIRI_VOCAB",
]
