"""Decision-rank Critic and gradient-isolation helpers (PR-07).

The Critic predicts the current seat's final 1..4 rank from the Actor-visible
representation ``z``.  It has no future/hidden input.  Two gradient modes are
explicit:

* ``joint``: the Critic loss may shape the shared Actor trunk;
* ``detach_actor``: the Critic sees ``z.detach()`` and its loss cannot put any
  gradient on Actor parameters.

The helper functions are independent of the model class so tests can inspect the
actual parameter gradients.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contracts import ContractError, PT_TABLE_5D, expected_pt_from_rank_probs


class CriticGradientModeError(ValueError):
    pass


class DecisionRankCritic(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )

    def forward(self, z: Tensor) -> Tensor:
        if z.dim() != 2:
            raise ContractError("critic input must be [N,d]")
        return self.net(z)


def critic_features(z: Tensor, mode: str) -> Tensor:
    if mode == "joint":
        return z
    if mode == "detach_actor":
        return z.detach()
    raise CriticGradientModeError(
        f"critic gradient mode must be joint or detach_actor, got {mode!r}")


def critic_ce(rank_logits: Tensor, final_rank: Tensor,
              weights: Tensor | None = None) -> Tensor:
    """Weighted CE over the four final-rank classes (labels 0..3)."""
    if rank_logits.dim() != 2 or rank_logits.size(1) != 4:
        raise ContractError("rank_logits must be [N,4]")
    if final_rank.shape != (rank_logits.size(0),):
        raise ContractError("final_rank must be [N]")
    if final_rank.dtype is not torch.long:
        raise ContractError("final_rank must be long")
    if bool((final_rank < 0).any()) or bool((final_rank > 3).any()):
        raise ContractError("final_rank outside 0..3")
    losses = F.cross_entropy(rank_logits, final_rank, reduction="none")
    if weights is None:
        return losses.mean()
    if weights.shape != final_rank.shape:
        raise ContractError("weights must be [N]")
    if bool((weights < 0).any()) or float(weights.sum()) <= 0:
        raise ContractError("weights must be non-negative and sum > 0")
    return (losses * weights).sum() / weights.sum()


def expected_pt(rank_logits: Tensor,
                pt_table: tuple[int, int, int, int] = PT_TABLE_5D) -> Tensor:
    return expected_pt_from_rank_probs(torch.softmax(rank_logits, dim=-1), pt_table)


def actor_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return total ** 0.5


__all__ = [
    "CriticGradientModeError", "DecisionRankCritic", "actor_grad_norm",
    "critic_ce", "critic_features", "expected_pt",
]
