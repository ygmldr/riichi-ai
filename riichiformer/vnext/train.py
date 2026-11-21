"""VNext BC training primitives and census (PR-08).

The trainer is deliberately small and CPU-first.  It implements masked BC plus
optional low-weight final-rank Critic CE and reports the additive loss numerator
so gradient accumulation is normalized over the whole window rather than each
microbatch.  It never starts a GPU epoch and never reads a target tensor inside
the model forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from ..schemas.action_v3 import ACTION_SPACE, ActionPhase
from ..schemas.schema_v3 import load_schema
from ..train.checkpoint import config_hash, save_checkpoint
from .contracts import (
    RL_CONTRACT_ID,
    VNextConfig,
    VNextModelBatch,
    VNextPolicyOutput,
    VNextTargets,
    contract_hash,
)
from .model import VNextPolicy


class VNextTrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class BCLoss:
    loss: Tensor
    numerator: Tensor
    valid: int
    total: int
    weight: float


def _target_weight(target: Mapping) -> float | None:
    if not bool(target.get("supervision_valid", False)):
        return None
    if target.get("forced") is True or target.get("loss_weight") == 0:
        return None
    value = target.get("loss_weight", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VNextTrainingError(f"loss_weight must be numeric, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise VNextTrainingError("loss_weight must be finite and >= 0")
    return value if value > 0 else None


def masked_bc_loss(output: VNextPolicyOutput,
                   targets: Sequence[Mapping]) -> BCLoss:
    if len(targets) != output.logits.size(0):
        raise VNextTrainingError("targets length != output decisions")
    weighted: list[Tensor] = []
    weights: list[float] = []
    for index, target in enumerate(targets):
        weight = _target_weight(target)
        if weight is None:
            continue
        action = target.get("expert_action")
        if isinstance(action, bool) or not isinstance(action, int) \
                or not 0 <= action < ACTION_SPACE:
            raise VNextTrainingError(f"target[{index}] action is invalid")
        if not bool(output.legal_mask[index, action]):
            raise VNextTrainingError(
                f"target[{index}] action {action} is outside the legal mask")
        logprob = output.logprobs[index, action]
        if not bool(torch.isfinite(logprob)):
            raise VNextTrainingError(f"target[{index}] logprob is non-finite")
        weighted.append(-logprob * weight)
        weights.append(weight)
    if not weighted:
        zero = output.logits.sum() * 0.0
        return BCLoss(zero, zero, 0, len(targets), 0.0)
    numerator = torch.stack(weighted).sum()
    total_weight = float(sum(weights))
    return BCLoss(numerator / total_weight, numerator, len(weighted),
                  len(targets), total_weight)


@dataclass(frozen=True)
class VNextTrainConfig:
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    rank_loss_weight: float = 0.1
    max_grad_norm: float = 1.0
    accumulation_count: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise VNextTrainingError("invalid optimizer hyperparameters")
        if self.rank_loss_weight < 0:
            raise VNextTrainingError("rank_loss_weight must be >= 0")
        if self.accumulation_count < 1:
            raise VNextTrainingError("accumulation_count must be >= 1")


def make_optimizer(model: VNextPolicy, config: VNextTrainConfig):
    """AdamW with no weight decay on norms, biases and embeddings."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or "embedding" in name or "norm" in name or "bias" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [{"params": decay, "weight_decay": config.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def rank_loss(output: VNextPolicyOutput,
              targets: Sequence[Mapping]) -> tuple[Tensor, int] | None:
    """Return ``(sum_rank_ce, count)`` over supervised decision rows.

    The label is the Critic class for that decision row; it is **never** used as
    an index into the decision axis.  A sum is returned so accumulation can use
    the additive numerator, just like BC.
    """
    if output.rank_logits is None:
        return None
    rows: list[int] = []
    labels: list[int] = []
    for index, target in enumerate(targets):
        if not bool(target.get("supervision_valid", False)):
            continue
        rank = target.get("rank_final")
        if isinstance(rank, bool) or not isinstance(rank, int) \
                or not 0 <= rank < 4:
            raise VNextTrainingError(f"target[{index}] rank_final is invalid")
        rows.append(index)
        labels.append(rank)
    if not rows:
        return None
    row_idx = torch.tensor(rows, dtype=torch.long,
                           device=output.rank_logits.device)
    label_idx = torch.tensor(labels, dtype=torch.long,
                             device=output.rank_logits.device)
    logits = output.rank_logits.index_select(0, row_idx)
    return F.cross_entropy(logits, label_idx, reduction="sum"), len(rows)


def train_step(model: VNextPolicy, batch: VNextModelBatch,
               targets: Sequence[Mapping], *, optimizer,
               config: VNextTrainConfig, accumulation_step: int = 0,
               window_weight: float | None = None) -> dict:
    """One BC update; returns additive loss and gradient bookkeeping."""
    if not isinstance(model, VNextPolicy):
        raise VNextTrainingError("model must be VNextPolicy")
    if config.accumulation_count > 1 and window_weight is None:
        raise VNextTrainingError(
            "accumulation_count > 1 requires window_weight")
    model.train()
    if accumulation_step % config.accumulation_count == 0:
        optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    bc = masked_bc_loss(output, targets)
    rank = rank_loss(output, targets)
    total_numerator = bc.numerator
    if rank is not None and config.rank_loss_weight > 0:
        total_numerator = total_numerator + config.rank_loss_weight * rank[0]
    if bc.valid == 0 and rank is None:
        return {
            "loss": 0.0, "bc_numerator": 0.0, "rank_numerator": None,
            "valid": 0, "total": bc.total, "weight": 0.0,
            "optimizer_stepped": False,
        }
    if window_weight is not None and float(window_weight) > 0:
        denominator = float(window_weight)
    elif bc.weight > 0:
        denominator = bc.weight
    elif rank is not None:
        # Rank-only batch: use supervised rank rows as the normalization count.
        denominator = float(rank[1])
    else:
        denominator = 0.0
    if denominator <= 0:
        raise VNextTrainingError("BC/rank window has no valid weight")
    loss = total_numerator / denominator
    if not bool(torch.isfinite(loss)):
        raise VNextTrainingError("VNext loss became NaN/Inf")
    if bc.valid > 0 or rank is not None:
        loss.backward()
    boundary = (accumulation_step + 1) % config.accumulation_count == 0
    stepped = False
    if boundary and any(p.grad is not None for p in model.parameters()):
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        stepped = True
    return {
        "loss": float(loss.detach()),
        "bc_numerator": float(bc.numerator.detach()),
        "rank_numerator": (None if rank is None
                           else float(rank[0].detach()) * config.rank_loss_weight),
        "valid": bc.valid, "total": bc.total, "weight": bc.weight,
        "optimizer_stepped": stepped,
    }


def save_vnext_checkpoint(path: str | Path, *, model: VNextPolicy,
                          optimizer, config: VNextTrainConfig,
                          counters: Mapping[str, int], provenance: Mapping,
                          extra: Mapping | None = None) -> str:
    config_dict = config.__dict__.copy()
    model_config = model.config.as_dict()
    schema = load_schema()
    hashes = {
        "schema_hash": schema["schema_hash"],
        "action_map_hash": schema["action_map_hash"],
        "normalizer_hash": schema["normalizer_hash"],
        "vnext_contract_hash": contract_hash(model.config),
    }
    info = save_checkpoint(
        path, model=model, optimizer=optimizer,
        counters=dict(counters), hashes=hashes, config=config_dict,
        seed=int(config.seed), model_config=model_config,
        provenance=provenance, extra=extra)
    return info.sha256


@dataclass
class CensusReport:
    decisions: int
    supervised: int
    phases: dict[str, int]
    ranks: dict[str, int]
    actions: dict[str, int]
    games: int
    povs: int


def census(batches: Iterable) -> CensusReport:
    """Count decisions/phases/actions from VNext BC batches (no model forward)."""
    decisions = supervised = 0
    phases: dict[str, int] = {}
    ranks: dict[str, int] = {}
    actions: dict[str, int] = {}
    games: set[str] = set()
    povs: set[int] = set()
    phase_names = tuple(phase.value for phase in ActionPhase)
    for item in batches:
        phase_ids = item.batch.phases.tolist()
        for index, (target, game, pov, kyoku) in enumerate(zip(
                item.targets, item.game_ids, item.povs, item.kyoku_ids)):
            decisions += 1
            games.add(str(game))
            povs.add(int(pov))
            if bool(target.get("supervision_valid", False)):
                supervised += 1
            phase = (phase_names[phase_ids[index]]
                     if 0 <= phase_ids[index] < len(phase_names) else "unknown")
            phases[phase] = phases.get(phase, 0) + 1
            rank = target.get("rank_final")
            ranks[str(rank)] = ranks.get(str(rank), 0) + 1
            action = target.get("expert_action")
            actions[str(action)] = actions.get(str(action), 0) + 1
    return CensusReport(decisions=decisions, supervised=supervised,
                        phases=phases, ranks=ranks, actions=actions,
                        games=len(games), povs=len(povs))


__all__ = [
    "BCLoss", "CensusReport", "RL_CONTRACT_ID", "VNextTrainConfig",
    "VNextTrainingError", "census", "make_optimizer", "masked_bc_loss",
    "rank_loss", "save_vnext_checkpoint", "train_step",
]
