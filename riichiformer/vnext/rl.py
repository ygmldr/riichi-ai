"""RL behavior/reward/GAE contracts and CPU toy objectives (PR-09/10).

This module contains no environment loop and never starts a GPU job.  It fixes
the *executable* behavior distribution semantics (mask + guard + sampling
transforms), the two non-double-counting GRP reward forms, and a two-timescale
GAE recurrence.  AWR and PPO helpers are deliberately small enough to test on a
two-action toy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .contracts import (
    ACTION_SPACE, ContractError, GuardSpec, PT_TABLE_5D, PT_TABLE_5D_ID,
    PT_TABLES, RLDecisionRow,
)


class RLError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Executable behavior distribution
# ---------------------------------------------------------------------------


def _top_p_mask(probs: Tensor, top_p: float) -> Tensor:
    if top_p >= 1.0:
        return torch.ones_like(probs, dtype=torch.bool)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = sorted_probs.cumsum(-1)
    keep_sorted = cumulative - sorted_probs < top_p
    keep_sorted[0] = True
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[sorted_idx] = keep_sorted
    return keep


def behavior_probs(raw_logits: Tensor, legal_mask: Tensor, *,
                   guard: GuardSpec | None = None) -> tuple[Tensor, int, bool]:
    """Return the true final ``mu(a|o)`` after mask/guard/sampling transforms.

    The returned ``(probs, executed_action, forced)`` uses the same distribution
    for sampling and for the stored behavior log-probability.  Illegal actions
    always have exactly zero mass; a forced action has probability 1.
    """
    guard = guard or GuardSpec()
    if raw_logits.dim() != 1 or raw_logits.numel() != ACTION_SPACE:
        raise RLError("raw_logits must be a 46-vector")
    if legal_mask.shape != (ACTION_SPACE,) or legal_mask.dtype is not torch.bool:
        raise RLError("legal_mask must be a bool 46-vector")
    effective = legal_mask if guard.effective_mask is None else (
        legal_mask & torch.tensor(guard.effective_mask, dtype=torch.bool))
    if not bool(effective.any()):
        raise RLError("effective mask is empty")
    if guard.forced_action is not None:
        if not bool(effective[guard.forced_action]):
            raise RLError("forced action is not legal/effective")
        probs = torch.zeros(ACTION_SPACE)
        probs[guard.forced_action] = 1.0
        return probs, int(guard.forced_action), True
    logits = raw_logits.float().masked_fill(~effective, float("-inf"))
    probs = torch.softmax(logits / guard.temperature, dim=-1)
    keep = _top_p_mask(probs, guard.top_p) & effective
    probs = torch.where(keep, probs, torch.zeros_like(probs))
    total = probs.sum()
    if float(total) <= 0:
        raise RLError("guard produced zero probability mass")
    probs = probs / total
    if guard.epsilon_mix > 0:
        uniform = effective.to(probs.dtype) / float(effective.sum())
        probs = (1.0 - guard.epsilon_mix) * probs + guard.epsilon_mix * uniform
    # Fail closed on any numerical mass outside the effective mask.
    probs = torch.where(effective, probs, torch.zeros_like(probs))
    probs = probs / probs.sum()
    action = int(torch.multinomial(probs, 1).item())
    return probs, action, False


# ---------------------------------------------------------------------------
# Reward forms
# ---------------------------------------------------------------------------


def terminal_pt(rank: int, pt_table: Sequence[int] = PT_TABLE_5D) -> float:
    if not isinstance(rank, int) or not 0 <= rank < 4:
        raise RLError("rank must be int 0..3")
    return float(tuple(pt_table)[rank])


def mortal_delta_rewards(phi_by_kyoku: Sequence[Sequence[float]],
                         terminal_pt_value: float) -> list[tuple[float, ...]]:
    """Per-kyoku ``Phi_{k+1}-Phi_k`` with terminal ``Phi=R_final``.

    ``phi_by_kyoku[k]`` is the four-seat GRP potential at the start of kyoku k.
    The returned list has one tuple per kyoku; no separate terminal reward is
    paid, so ``sum(rewards)[seat] == terminal_pt_value - phi_0[seat]``.
    """
    if not phi_by_kyoku:
        raise RLError("need at least one kyoku potential")
    phis = [tuple(float(v) for v in values) for values in phi_by_kyoku]
    if any(len(values) != 4 for values in phis):
        raise RLError("each Phi boundary must have 4 seats")
    if any(abs(sum(values)) > 1e-4 for values in phis):
        raise RLError("zero-sum PT requires each Phi boundary to sum to 0")
    out: list[tuple[float, ...]] = []
    for index in range(len(phis) - 1):
        out.append(tuple(phis[index + 1][seat] - phis[index][seat]
                         for seat in range(4)))
    last = tuple(float(terminal_pt_value) - phis[-1][seat] for seat in range(4))
    out.append(last)
    return out


def pbrs_rewards(phi_by_kyoku: Sequence[Sequence[float]],
                 terminal_pt_value: float) -> list[tuple[float, ...]]:
    """Exact PBRS with terminal Phi=0 plus the true terminal PT once.

    For a pure terminal objective every non-terminal reward is the potential
    difference; the final kyoku transition also includes the real PT.  This is
    algebraically the same net return as :func:`mortal_delta_rewards`, but the
    recorded reward components differ and the two must never be combined.
    """
    if not phi_by_kyoku:
        raise RLError("need at least one kyoku potential")
    phis = [tuple(float(v) for v in values) for values in phi_by_kyoku]
    if any(len(values) != 4 for values in phis):
        raise RLError("each Phi boundary must have 4 seats")
    out: list[tuple[float, ...]] = []
    for index in range(len(phis) - 1):
        out.append(tuple(phis[index + 1][seat] - phis[index][seat]
                         for seat in range(4)))
    out.append(tuple(float(terminal_pt_value) - phis[-1][seat]
                     for seat in range(4)))
    # The terminal transition includes the true R_final and sets Phi_terminal=0;
    # the expression above already telescopes to R - Phi_0, so adding R again is
    # exactly the double-counting bug this API refuses.
    return out


def check_no_double_terminal(has_terminal_reward: bool, has_delta_terminal: bool) -> None:
    if has_terminal_reward and has_delta_terminal:
        raise RLError("double terminal reward: choose PBRS/terminal or Mortal delta")


# ---------------------------------------------------------------------------
# Two-timescale GAE
# ---------------------------------------------------------------------------


def two_timescale_gae(rewards: Sequence[float], values: Sequence[float],
                      next_values: Sequence[float], *,
                      terminals: Sequence[bool] | None = None,
                      kyoku_boundaries: Sequence[bool] | None = None,
                      gamma: float = 1.0, lambda_intra: float = 1.0,
                      lambda_inter: float = 1.0) -> list[float]:
    """Backward GAE with a different lambda across a kyoku boundary.

    ``next_values[t]`` is the value used for the bootstrap at transition ``t``;
    the caller may pass ``V(o_{t+1})`` or a GRP boundary value explicitly.  A
    terminal transition has no bootstrap.  ``kyoku_boundaries[t]`` marks a
    transition that leaves the kyoku (but not the match).
    """
    n = len(rewards)
    if len(values) != n or len(next_values) != n:
        raise RLError("rewards/values/next_values lengths differ")
    terminals = list(terminals or [False] * n)
    boundaries = list(kyoku_boundaries or [False] * n)
    if len(terminals) != n or len(boundaries) != n:
        raise RLError("terminals/boundaries lengths differ")
    if not 0.0 <= gamma <= 1.0:
        raise RLError("gamma must be in [0,1]")
    for name, value in (("lambda_intra", lambda_intra),
                        ("lambda_inter", lambda_inter)):
        if not 0.0 <= value <= 1.0:
            raise RLError(f"{name} must be in [0,1]")
    advantages = [0.0] * n
    running = 0.0
    for index in range(n - 1, -1, -1):
        bootstrap = 0.0 if terminals[index] else gamma * float(next_values[index])
        delta = float(rewards[index]) + bootstrap - float(values[index])
        lam = lambda_inter if boundaries[index] else lambda_intra
        if terminals[index]:
            running = delta
        else:
            running = delta + gamma * lam * running
        advantages[index] = running
    return advantages


# ---------------------------------------------------------------------------
# Toy objectives
# ---------------------------------------------------------------------------


def awr_loss(logprobs: Tensor, actions: Tensor, advantages: Tensor, *,
             beta: float = 1.0, weight_clip: float = 10.0,
             reference_logprobs: Tensor | None = None,
             kl_weight: float = 0.0) -> Tensor:
    if logprobs.dim() != 2 or logprobs.size(1) < 2:
        raise RLError("logprobs must be [N,A] with A >= 2")
    if actions.shape != (logprobs.size(0),):
        raise RLError("actions must be [N]")
    if advantages.shape != actions.shape:
        raise RLError("advantages must be [N]")
    if beta <= 0 or weight_clip <= 0:
        raise RLError("beta/weight_clip must be positive")
    selected = logprobs[torch.arange(logprobs.size(0)), actions]
    weights = torch.exp(advantages / beta).clamp(max=weight_clip)
    loss = -(weights * selected).mean()
    if reference_logprobs is not None:
        if reference_logprobs.shape != logprobs.shape:
            raise RLError("reference_logprobs shape mismatch")
        # Reverse KL(pi_bc || pi_theta) on the sampled actions.
        loss = loss + kl_weight * (
            reference_logprobs.gather(1, actions.unsqueeze(1)).squeeze(1)
            - selected).mean()
    return loss


def ppo_loss(new_logprobs: Tensor, old_logprobs: Tensor, actions: Tensor,
             advantages: Tensor, *, clip: float = 0.2) -> Tensor:
    if new_logprobs.shape != old_logprobs.shape or new_logprobs.dim() != 2:
        raise RLError("new/old logprobs shapes differ")
    if new_logprobs.size(1) < 2:
        raise RLError("logprobs need at least two actions")
    if actions.shape != (new_logprobs.size(0),):
        raise RLError("actions must be [N]")
    if advantages.shape != actions.shape:
        raise RLError("advantages must be [N]")
    if not 0.0 < clip < 1.0:
        raise RLError("clip must be in (0,1)")
    idx = torch.arange(actions.numel())
    new = new_logprobs[idx, actions]
    old = old_logprobs[idx, actions]
    ratio = torch.exp(new - old)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * advantages
    return -torch.minimum(unclipped, clipped).mean()


@dataclass(frozen=True)
class LeagueManifest:
    league_id: str
    rules_hash: str
    opponent_checkpoint_shas: tuple[str, ...]
    engine_guard_sha: str
    sampling_seed: int
    version: int = 1

    def __post_init__(self) -> None:
        if not self.league_id or not self.rules_hash or not self.engine_guard_sha:
            raise RLError("league identifiers must be non-empty")
        if len(self.opponent_checkpoint_shas) != 3:
            raise RLError("default league is 1 learner + 3 frozen opponents")
        if any(len(value) != 64 for value in self.opponent_checkpoint_shas):
            raise RLError("opponent checkpoint SHAs must be 64 hex chars")
        if self.version < 1:
            raise RLError("league version must be >= 1")


__all__ = [
    "LeagueManifest", "RLError", "awr_loss", "behavior_probs",
    "check_no_double_terminal", "mortal_delta_rewards", "pbrs_rewards",
    "ppo_loss", "terminal_pt", "two_timescale_gae",
]
