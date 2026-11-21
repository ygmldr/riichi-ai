"""VNext policy: current-kyoku causal Transformer + decision-query cross-attention.

The default architecture is the frozen ``rf-vnext-crossattn/1`` contract:

* a six-layer, d=256, four-head pre-RMSNorm causal event Transformer with RoPE;
* ``[BOS]`` plus one token per visible current-kyoku event;
* a 34-tile structured state encoder and a factorized tile embedding;
* one decision-query cross-attention layer whose key/value mask is exactly
  ``key_token <= prefix_token_offset`` and never crosses a sequence row;
* a 37-candidate tile head plus a 9-way non-tile head, merged into the native
  46-action order.

The model consumes only :class:`VNextModelBatch`.  Targets are deliberately absent
from this module's forward signature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..models.local_transformer import LocalTransformer
from ..schemas.action_v3 import ACTION_SPACE, AKA_KINDS, TILE_KIND_COUNT
from .critic import DecisionRankCritic
from .contracts import (
    EVENT_ACTOR_VOCAB,
    EVENT_AKA_VOCAB,
    EVENT_KYOKU_POS_VOCAB,
    EVENT_MELD_SLOTS,
    EVENT_TILE_VOCAB,
    EVENT_TSUMOGIRI_VOCAB,
    EVENT_TYPE_VOCAB,
    GLOBAL_FEATURE_DIM,
    TILE_CANDIDATES,
    TILE_FEATURE_DIM,
    VNextConfig,
    VNextModelBatch,
    VNextPolicyOutput,
)

OTHER_OFFSET = 37


class VNextModelError(ValueError):
    """Malformed VNext model input or configuration."""


class VNextEventEmbedding(nn.Module):
    """Additive categorical event embedding; BOS is a real token id.

    Padding rows are zeroed by the valid mask.  ``BOS_EVENT_ID`` (16) is a real
    event-type embedding, not the PAD id (17).
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.event_type = nn.Embedding(EVENT_TYPE_VOCAB, d_model)
        self.actor = nn.Embedding(EVENT_ACTOR_VOCAB, d_model)
        self.tile = nn.Embedding(EVENT_TILE_VOCAB, d_model)
        self.aka = nn.Embedding(EVENT_AKA_VOCAB, d_model)
        self.tsumogiri = nn.Embedding(EVENT_TSUMOGIRI_VOCAB, d_model)
        self.kyoku_pos = nn.Embedding(EVENT_KYOKU_POS_VOCAB, d_model)
        self.meld_tile = nn.Embedding(EVENT_TILE_VOCAB, d_model)
        self.meld_aka = nn.Embedding(EVENT_AKA_VOCAB, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: VNextModelBatch) -> Tensor:
        x = (
            self.event_type(batch.event_type_id)
            + self.actor(batch.event_actor_rel)
            + self.tile(batch.event_tile_kind)
            + self.aka(batch.event_aka)
            + self.tsumogiri(batch.event_tsumogiri)
            + self.kyoku_pos(batch.event_kyoku_pos)
            + self.meld_tile(batch.event_meld_tile).sum(dim=2)
            + self.meld_aka(batch.event_meld_aka).sum(dim=2)
        )
        x = self.norm(x)
        x = self.dropout(x)
        return x * batch.event_valid_mask.unsqueeze(-1).to(x.dtype)


class StructuredTileStateEncoder(nn.Module):
    """Shared per-tile MLP plus a fixed-order tile summary and global projection."""

    def __init__(self, d_model: int, d_tile: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_tile = int(d_tile)
        self.tile_mlp = nn.Sequential(
            nn.Linear(TILE_FEATURE_DIM, d_tile),
            nn.GELU(),
            nn.LayerNorm(d_tile),
            nn.Linear(d_tile, d_tile),
        )
        self.tile_summary = nn.Linear(TILE_KIND_COUNT * d_tile, d_model)
        self.global_mlp = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_tile: Tensor, x_global: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # [N,34,C] -> [N,34,d_tile]
        per_tile = self.tile_mlp(x_tile)
        summary = self.tile_summary(per_tile.reshape(per_tile.size(0), -1))
        summary = self.dropout(torch.tanh(summary))
        global_vec = self.global_mlp(x_global)
        return per_tile, summary, global_vec


class FactorizedTileEmbedding(nn.Module):
    """``E_suit + E_rank + alpha * E_identity`` plus honor/aka residuals."""

    def __init__(self, d_model: int, identity_scale: float = 1.0,
                 embedding: str = "factorized") -> None:
        super().__init__()
        if embedding not in {"factorized", "identity"}:
            raise VNextModelError("embedding must be factorized or identity")
        self.embedding = embedding
        self.identity_scale = float(identity_scale)
        self.suit = nn.Embedding(4, d_model)
        self.rank = nn.Embedding(9, d_model)
        self.identity = nn.Embedding(TILE_KIND_COUNT, d_model)
        self.honor = nn.Embedding(7, d_model)
        self.aka = nn.Embedding(4, d_model)

    def forward(self, kind: Tensor, aka: Tensor) -> Tensor:
        if kind.dtype is not torch.long or aka.dtype is not torch.long:
            raise VNextModelError("tile kind/aka must be long tensors")
        if kind.shape != aka.shape:
            raise VNextModelError("tile kind/aka shapes differ")
        if bool((kind < 0).any()) or bool((kind >= TILE_KIND_COUNT).any()):
            raise VNextModelError("tile kind outside 0..33")
        if bool((aka < 0).any()) or bool((aka >= 4).any()):
            raise VNextModelError("aka index outside 0..3")
        if self.embedding == "identity":
            base = self.identity(kind)
        else:
            suit = (kind // 9).clamp(max=3)
            rank = kind % 9
            base = (self.suit(suit) + self.rank(rank)
                    + self.identity_scale * self.identity(kind))
            honor = kind >= 27
            honor_vec = self.honor((kind - 27).clamp(min=0)) * honor.unsqueeze(-1)
            base = base + honor_vec
        return base + self.aka(aka)


class DecisionQueryCrossAttention(nn.Module):
    """Multi-head cross-attention with a strict per-decision prefix mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads:
            raise VNextModelError("d_model must be divisible by cross-attention heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)

    def forward(self, query: Tensor, sequence_states: Tensor,
                decision_to_sequence: Tensor, prefix_offsets: Tensor,
                event_valid_mask: Tensor) -> Tensor:
        n = query.size(0)
        if sequence_states.dim() != 3:
            raise VNextModelError("sequence_states must be [B,L,d]")
        if decision_to_sequence.shape != (n,) or prefix_offsets.shape != (n,):
            raise VNextModelError("decision index tensors must be [N]")
        b, length, width = sequence_states.shape
        if width != self.d_model:
            raise VNextModelError("sequence state width mismatch")
        if bool((decision_to_sequence < 0).any()) \
                or bool((decision_to_sequence >= b).any()):
            raise VNextModelError("decision sequence index outside batch")
        if bool((prefix_offsets < 0).any()) or bool((prefix_offsets >= length).any()):
            raise VNextModelError("prefix offset outside sequence")
        q = self.q(query).view(n, self.n_heads, 1, self.head_dim)
        kv = self.kv(sequence_states.index_select(0, decision_to_sequence))
        kv = kv.view(n, length, 2, self.n_heads, self.head_dim).permute(
            2, 0, 3, 1, 4)
        key, value = kv.unbind(0)  # [N,H,L,D]
        positions = torch.arange(length, device=query.device).unsqueeze(0)
        mask = (positions <= prefix_offsets.unsqueeze(1))
        mask = mask & event_valid_mask.index_select(0, decision_to_sequence)
        # Every decision has BOS (offset >= 0), so no query row is all-masked.
        attended = F.scaled_dot_product_attention(
            q, key, value, attn_mask=mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(attended.reshape(n, self.d_model))


class TileScoredHead(nn.Module):
    """Phase-conditioned dot-product scoring over 37 tile candidates."""

    def __init__(self, d_model: int, d_tile: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.query = nn.Linear(d_model, d_model)
        self.phase = nn.Embedding(3, d_model)
        self.key = nn.Linear(d_tile, d_model)
        self.aka = nn.Embedding(4, d_tile)
        self.residual = nn.Sequential(
            nn.Linear(4 + 3, 32), nn.GELU(), nn.Linear(32, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: Tensor, phases: Tensor, per_tile: Tensor,
                candidate_features: Tensor) -> Tensor:
        # candidate base kinds: 0..33 normal, 34..36 base five kinds.
        base = torch.cat([
            torch.arange(34, device=z.device),
            torch.tensor([AKA_KINDS[0], AKA_KINDS[1], AKA_KINDS[2]],
                         device=z.device),
        ])
        base = base.unsqueeze(0).expand(z.size(0), -1)
        tile_vec = per_tile.gather(
            1, base.unsqueeze(-1).expand(-1, -1, per_tile.size(-1)))
        aka_idx = torch.zeros((z.size(0), TILE_CANDIDATES), dtype=torch.long,
                              device=z.device)
        for colour in range(3):
            aka_idx[:, 34 + colour] = colour + 1
        tile_vec = tile_vec + self.aka(aka_idx)
        keys = self.key(tile_vec)
        query = self.query(z) + self.phase(phases)
        scores = torch.einsum("nd,nkd->nk", query, keys) / math.sqrt(self.d_model)
        phase_one_hot = F.one_hot(phases, num_classes=3).to(candidate_features.dtype)
        residual_in = torch.cat([
            candidate_features,
            phase_one_hot.unsqueeze(1).expand(-1, TILE_CANDIDATES, -1),
        ], dim=-1)
        scores = scores + self.residual(residual_in).squeeze(-1)
        return self.dropout(scores)


class OtherHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.phase = nn.Embedding(3, d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 9),
        )

    def forward(self, z: Tensor, phases: Tensor) -> Tensor:
        return self.net(z + self.phase(phases))


class Linear46Head(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, ACTION_SPACE)

    def forward(self, z: Tensor) -> Tensor:
        return self.linear(z)


class VNextPolicy(nn.Module):
    """Complete VNext model; ``forward`` accepts only a VNextModelBatch."""

    def __init__(self, config: VNextConfig | None = None) -> None:
        super().__init__()
        self.config = config or VNextConfig()
        c = self.config
        self.event_embedding = VNextEventEmbedding(c.d_model, dropout=c.dropout)
        self.event_backbone = LocalTransformer(
            d_model=c.d_model, n_layers=c.n_layers, n_heads=c.n_heads,
            ffn_hidden=c.ffn_hidden, dropout=c.dropout,
            max_positions=c.max_positions, rope_base=c.rope_base,
            rmsnorm_eps=c.rmsnorm_eps,
        )
        self.state_encoder = StructuredTileStateEncoder(
            c.d_model, d_tile=64, dropout=c.dropout)
        self.phase = nn.Embedding(3, c.d_model)
        self.query = nn.Sequential(
            nn.Linear(3 * c.d_model, c.d_model), nn.GELU(),
            nn.LayerNorm(c.d_model),
        )
        self.cross = nn.ModuleList([
            DecisionQueryCrossAttention(c.d_model, c.cross_attn_heads,
                                        dropout=c.dropout)
            for _ in range(c.cross_attn_layers)
        ])
        self.fuse = nn.Sequential(
            nn.Linear(4 * c.d_model, c.d_model), nn.GELU(),
            nn.LayerNorm(c.d_model),
        )
        self.tile_embedding = FactorizedTileEmbedding(
            c.d_model, identity_scale=c.tile_identity_scale,
            embedding=c.tile_embedding)
        self.tile_head = TileScoredHead(c.d_model, 64, dropout=c.dropout)
        self.other_head = OtherHead(c.d_model, dropout=c.dropout)
        self.linear_head = Linear46Head(c.d_model)
        self.critic = (DecisionRankCritic(c.d_model, dropout=c.dropout)
                       if c.critic_enabled else None)

    def event_states(self, batch: VNextModelBatch) -> Tensor:
        if self.config.model_mode == "state_only":
            return torch.zeros(
                (batch.batch_size, batch.seq_len, self.config.d_model),
                dtype=batch.x_tile.dtype, device=batch.x_tile.device)
        x = self.event_embedding(batch)
        return self.event_backbone(x, batch.event_valid_mask)

    def forward(self, batch: VNextModelBatch) -> VNextPolicyOutput:
        if not isinstance(batch, VNextModelBatch):
            raise TypeError(
                f"VNextPolicy.forward accepts only VNextModelBatch, got "
                f"{type(batch).__name__}")
        device = next(self.parameters()).device
        if batch.event_valid_mask.device != device:
            raise VNextModelError(
                f"batch device {batch.event_valid_mask.device} != model device {device}")
        batch = batch.to(device)
        h = self.event_states(batch)
        per_tile, tile_summary, global_vec = self.state_encoder(
            batch.x_tile, batch.x_global)
        q = self.query(torch.cat([
            tile_summary, global_vec, self.phase(batch.phases)], dim=-1))
        context = torch.zeros_like(q)
        for layer in self.cross:
            context = context + layer(
                q, h, batch.decision_to_sequence, batch.prefix_token_offsets,
                batch.event_valid_mask)
        prefix = h[batch.decision_to_sequence, batch.prefix_token_offsets]
        z = self.fuse(torch.cat([q, context, prefix, global_vec], dim=-1))
        if self.config.head_type == "linear46":
            logits = self.linear_head(z)
        else:
            tile_logits = self.tile_head(
                z, batch.phases, per_tile, batch.candidate_features)
            other_logits = self.other_head(z, batch.phases)
            logits = torch.cat([tile_logits, other_logits], dim=-1)
        if logits.size(-1) != ACTION_SPACE:  # pragma: no cover
            raise AssertionError("VNext policy head width is not 46")
        # Keep policy arithmetic in FP32 even under BF16 autocast; the output
        # contract and the masked -inf/log-softmax must share one dtype.
        logits = logits.float()
        masked = logits.masked_fill(~batch.legal_mask, float("-inf"))
        logprobs = torch.log_softmax(masked, dim=-1)
        if self.critic is not None:
            critic_input = (z.detach()
                            if self.config.critic_grad_mode == "detach_actor"
                            else z)
            rank_logits = self.critic(critic_input)
        else:
            rank_logits = None
        output = VNextPolicyOutput(
            logits=logits, masked_logits=masked, logprobs=logprobs,
            legal_mask=batch.legal_mask, rank_logits=rank_logits)
        output.validate()
        return output

    def forward_prefix(self, batch: VNextModelBatch) -> VNextPolicyOutput:
        """Reference single-prefix path (same code path, named for parity tests)."""
        return self.forward(batch)


def build_vnext_model(config: VNextConfig | None = None) -> VNextPolicy:
    return VNextPolicy(config)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


__all__ = [
    "DecisionQueryCrossAttention", "FactorizedTileEmbedding", "Linear46Head",
    "OtherHead", "StructuredTileStateEncoder", "TileScoredHead",
    "VNextEventEmbedding", "VNextModelError", "VNextPolicy", "build_vnext_model",
    "count_parameters",
]
