"""Causal local transformer backbone for the PR-13 MVP-S model.

Architecture (roadmap Recipe E / model card):

* Pre-RMSNorm residual blocks (``x = x + attn(norm(x)); x = x + ffn(norm(x))``);
* multi-head causal self-attention with RoPE, computed by
  ``torch.nn.functional.scaled_dot_product_attention`` (SDPA);
* SwiGLU feed-forward (``w2(silu(w1 x) * w3 x)``), hidden width ~2.75 × d_model;
* dropout defaults to **0.0** so prefix-parity tests are exact;
* key padding mask supported; no sequence truncation — a sequence longer than
  ``max_positions`` raises ``ValueError`` instead of being cut.

Input is ``[B, L, d_model]`` plus a bool ``[B, L]`` padding mask (True = real token).
``max_positions`` bounds the RoPE table only; it is a configuration invariant, not a
silent cut.  With left-aligned (right-padded) masks the ordinary causal path is used;
for arbitrary/holey key masks a combined causal+padding boolean mask is built so a
padded key never receives attention weight.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_MAX_POSITIONS = 4096


class TransformerShapeError(ValueError):
    """A shape / length / mask contract violation (never silently truncated)."""


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm without mean subtraction or bias."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_float = x_float * torch.rsqrt(variance + self.eps)
        return (x_float.to(dtype) * self.weight)


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding applied to ``[B, H, L, head_dim]`` q/k."""

    def __init__(self, head_dim: int, max_positions: int = DEFAULT_MAX_POSITIONS,
                 base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be positive and even, got {head_dim}")
        if max_positions <= 0:
            raise ValueError(f"max_positions must be positive, got {max_positions}")
        if base <= 1.0:
            raise ValueError(f"RoPE base must be > 1, got {base}")
        self.head_dim = int(head_dim)
        self.max_positions = int(max_positions)
        self.base = float(base)
        inv_freq = 1.0 / (self.base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        ))
        # non-persistent buffers keep checkpoints identical to the module weights.
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("_cos", torch.empty(0), persistent=False)
        self.register_buffer("_sin", torch.empty(0), persistent=False)

    def _ensure_cache(self, length: int, device: torch.device) -> None:
        if (self._cos.numel() >= length * self.head_dim
                and self._cos.device == device):
            return
        positions = torch.arange(self.max_positions, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos()
        self._sin = emb.sin()

    def forward(self, q: Tensor, k: Tensor,
                position_ids: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Apply RoPE at ``position_ids`` (default: ``0..L-1``).

        ``position_ids`` is a ``[B, L]`` long tensor.  Explicit positions let a
        decision-marker token sit at its cursor position instead of at the end of
        the concatenated sequence (PR-38), which keeps the marker's attention
        geometry identical to the legacy single-prefix forward.  Positions are
        bounded by ``max_positions`` and never silently clamped.
        """
        length = q.size(-2)
        if length > self.max_positions:
            raise TransformerShapeError(
                f"sequence length {length} exceeds RoPE max_positions "
                f"{self.max_positions}; refusing to truncate"
            )
        if position_ids is None:
            self._ensure_cache(length, q.device)
            cos = self._cos[:length].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
            sin = self._sin[:length].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
            return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        if not isinstance(position_ids, Tensor) or position_ids.dtype is not torch.long:
            raise TransformerShapeError("position_ids must be a long torch.Tensor")
        if position_ids.dim() != 2 \
                or tuple(position_ids.shape) != (q.size(0), length):
            raise TransformerShapeError(
                f"position_ids shape {tuple(position_ids.shape)} != "
                f"{(q.size(0), length)}"
            )
        if position_ids.device != q.device:
            raise TransformerShapeError("position_ids and query must share a device")
        minimum = int(position_ids.min())
        maximum = int(position_ids.max())
        if minimum < 0 or maximum >= self.max_positions:
            raise TransformerShapeError(
                f"position_ids must be in [0, {self.max_positions - 1}], got "
                f"[{minimum}, {maximum}]; refusing to wrap or clamp"
            )
        self._ensure_cache(maximum + 1, q.device)
        flat = position_ids.reshape(-1)
        cos = self._cos.index_select(0, flat).view(
            q.size(0), length, self.head_dim).to(dtype=q.dtype).unsqueeze(1)
        sin = self._sin.index_select(0, flat).view(
            q.size(0), length, self.head_dim).to(dtype=q.dtype).unsqueeze(1)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 max_positions: int = DEFAULT_MAX_POSITIONS,
                 rope_base: float = 10_000.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model {d_model} is not divisible by n_heads {n_heads}"
            )
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.rope = RotaryEmbedding(self.head_dim, max_positions=max_positions,
                                    base=rope_base)

    def forward(self, x: Tensor, attn_mask: Tensor | None,
                is_causal: bool,
                position_ids: Tensor | None = None) -> Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(batch, length, 3, self.n_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        ).unbind(0)
        q, k = self.rope(q, k, position_ids)
        dropout_p = self.dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, self.d_model)
        return self.out_proj(attended)


class SwiGLU(nn.Module):
    """``w2(silu(w1(x)) * w3(x))`` feed-forward block."""

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError(f"ffn hidden must be positive, got {hidden}")
        self.w1 = nn.Linear(d_model, hidden, bias=True)  # gate
        self.w3 = nn.Linear(d_model, hidden, bias=True)  # up
        self.w2 = nn.Linear(hidden, d_model, bias=True)  # down
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """One Pre-RMSNorm block: attention then SwiGLU, both residual."""

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int,
                 dropout: float = 0.0, rmsnorm_eps: float = 1e-6,
                 max_positions: int = DEFAULT_MAX_POSITIONS,
                 rope_base: float = 10_000.0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=rmsnorm_eps)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout,
                                        max_positions=max_positions, rope_base=rope_base)
        self.ffn_norm = RMSNorm(d_model, eps=rmsnorm_eps)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout=dropout)

    def forward(self, x: Tensor, attn_mask: Tensor | None, is_causal: bool,
                position_ids: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), attn_mask, is_causal, position_ids)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def _is_left_aligned_mask(padding_mask: Tensor) -> bool:
    lengths = padding_mask.sum(dim=1)
    positions = torch.arange(padding_mask.size(1), device=padding_mask.device)
    return bool(torch.equal(padding_mask, positions.unsqueeze(0) < lengths.unsqueeze(1)))


class LocalTransformer(nn.Module):
    """Stack of Pre-RMSNorm causal blocks with SDPA attention and RoPE."""

    def __init__(self, d_model: int = 256, n_layers: int = 6, n_heads: int = 4,
                 ffn_hidden: int = 704, dropout: float = 0.0,
                 max_positions: int = DEFAULT_MAX_POSITIONS,
                 rope_base: float = 10_000.0,
                 rmsnorm_eps: float = 1e-6) -> None:
        super().__init__()
        if d_model <= 0 or n_layers <= 0 or n_heads <= 0:
            raise ValueError("d_model, n_layers and n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by n_heads {n_heads}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.ffn_hidden = int(ffn_hidden)
        self.dropout = float(dropout)
        self.max_positions = int(max_positions)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_hidden, dropout=dropout,
                             rmsnorm_eps=rmsnorm_eps, max_positions=max_positions,
                             rope_base=rope_base)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model, eps=rmsnorm_eps)

    def attention_mask(self, padding_mask: Tensor) -> tuple[Tensor | None, bool]:
        """Return ``(attn_mask, is_causal)`` for the given key-padding mask.

        Full or left-aligned (right-padded) masks take SDPA's fast causal path;
        any other mask gets an explicit boolean causal ∧ padding mask that also
        keeps every query's self key (so an all-padding query row cannot yield NaN).
        """
        if bool(padding_mask.all()) or _is_left_aligned_mask(padding_mask):
            return None, True
        length = padding_mask.size(1)
        causal = torch.ones(length, length, dtype=torch.bool,
                            device=padding_mask.device).tril()
        mask = causal.unsqueeze(0).unsqueeze(0) & padding_mask.unsqueeze(1).unsqueeze(2)
        eye = torch.eye(length, dtype=torch.bool, device=padding_mask.device)
        return mask | eye.unsqueeze(0).unsqueeze(0), False

    def forward(self, x: Tensor, padding_mask: Tensor, *,
                attn_mask: Tensor | None = None,
                position_ids: Tensor | None = None) -> Tensor:
        """Run the backbone.

        ``attn_mask`` (bool, True = may attend) overrides the causal/padding mask
        computed from ``padding_mask``; it is used by the PR-38 multi-decision
        forward so decision markers can read only their own prefix and itself.
        ``position_ids`` overrides the RoPE positions (see :class:`RotaryEmbedding`).
        """
        if not isinstance(x, Tensor) or x.dim() != 3:
            raise TransformerShapeError(
                f"input must be [B, L, d_model], got "
                f"{tuple(x.shape) if isinstance(x, Tensor) else type(x).__name__}"
            )
        batch, length, width = x.shape
        if width != self.d_model:
            raise TransformerShapeError(
                f"input width {width} != d_model {self.d_model}"
            )
        if length == 0 or batch == 0:
            raise TransformerShapeError(f"empty input {tuple(x.shape)}")
        if length > self.max_positions:
            raise TransformerShapeError(
                f"sequence length {length} > max_positions {self.max_positions}; "
                f"refuse to truncate"
            )
        if not isinstance(padding_mask, Tensor) or padding_mask.dtype is not torch.bool:
            raise TransformerShapeError("padding_mask must be a bool torch.Tensor")
        if tuple(padding_mask.shape) != (batch, length):
            raise TransformerShapeError(
                f"padding_mask shape {tuple(padding_mask.shape)} != {(batch, length)}"
            )
        if padding_mask.device != x.device:
            raise TransformerShapeError("padding_mask and input must share a device")
        if not bool(padding_mask.any(dim=1).all()):
            raise TransformerShapeError("every row needs at least one real token")

        if attn_mask is None:
            mask, is_causal = self.attention_mask(padding_mask)
        else:
            if not isinstance(attn_mask, Tensor) or attn_mask.dtype is not torch.bool:
                raise TransformerShapeError(
                    "attn_mask must be a bool torch.Tensor (True = may attend)"
                )
            if attn_mask.dim() == 4:
                if attn_mask.size(0) != batch \
                        or attn_mask.size(-2) != length \
                        or attn_mask.size(-1) != length:
                    raise TransformerShapeError(
                        f"attn_mask shape {tuple(attn_mask.shape)} must be "
                        f"[{batch}, 1|H, {length}, {length}]"
                    )
            elif attn_mask.dim() == 3:
                if tuple(attn_mask.shape) != (batch, length, length):
                    raise TransformerShapeError(
                        f"attn_mask shape {tuple(attn_mask.shape)} != "
                        f"{(batch, length, length)}"
                    )
            else:
                raise TransformerShapeError(
                    f"attn_mask must be [B, L, L] or [B, 1|H, L, L], got "
                    f"{tuple(attn_mask.shape)}"
                )
            if attn_mask.device != x.device:
                raise TransformerShapeError(
                    "attn_mask and input must share a device"
                )
            # An all-False query row would yield NaN; fail closed instead.
            if not bool(attn_mask.any(dim=-1).all()):
                raise TransformerShapeError(
                    "attn_mask leaves a query row with no visible key"
                )
            mask, is_causal = attn_mask, False
        if position_ids is not None:
            if not isinstance(position_ids, Tensor) \
                    or position_ids.dtype is not torch.long \
                    or position_ids.dim() != 2 \
                    or tuple(position_ids.shape) != (batch, length):
                raise TransformerShapeError(
                    "position_ids must be a long [B, L] tensor"
                )
            if position_ids.device != x.device:
                raise TransformerShapeError(
                    "position_ids and input must share a device"
                )
        for block in self.blocks:
            x = block(x, mask, is_causal, position_ids)
        return self.final_norm(x)


def scaled_dot_product_attention_available() -> bool:
    """True when this torch build has the SDPA kernel used by the backbone."""
    return hasattr(F, "scaled_dot_product_attention")


__all__ = [
    "CausalSelfAttention", "DEFAULT_MAX_POSITIONS", "LocalTransformer", "RMSNorm",
    "RotaryEmbedding", "SwiGLU", "TransformerBlock", "TransformerShapeError",
    "scaled_dot_product_attention_available",
]
