#!/usr/bin/env python3
"""PR-14 masked behavior-cloning loss and one-step trainer.

Only supervised, in-mask expert decisions contribute.  Censored/None targets and forced
actions remain context in the batch but are excluded from the policy loss.  Targets are
plain dictionaries from PR-12; this module imports no target/oracle class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from ..models.batch import ModelBatch
from ..models.policy import Policy, PolicyOutput
from ..schemas.action_v3 import ACTION_SPACE


class TrainingError(RuntimeError):
    """A training input/label/loss violates the BC contract."""


def _target_weight(target: Mapping) -> float | None:
    if not bool(target.get("supervision_valid", False)):
        return None
    if target.get("forced") is True or target.get("loss_weight") == 0:
        return None
    weight = target.get("loss_weight", 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise TrainingError(f"loss_weight must be numeric, got {weight!r}")
    weight = float(weight)
    if not torch.isfinite(torch.tensor(weight)) or weight < 0:
        raise TrainingError(f"loss_weight must be finite and >= 0, got {weight!r}")
    return weight if weight > 0 else None


@dataclass(frozen=True)
class BCLossResult:
    loss: Tensor
    valid_decisions: int
    total_decisions: int
    total_weight: float

    def __post_init__(self):
        if self.valid_decisions < 0 or self.valid_decisions > self.total_decisions:
            raise TrainingError("invalid valid/total decision counts")
        if not bool(torch.isfinite(self.loss).all()):
            raise TrainingError("BC loss is NaN/Inf")


def masked_bc_loss(output: PolicyOutput, targets: Sequence[Mapping], *,
                   legal_mask: Tensor | None = None) -> BCLossResult:
    """``-sum(weight * logprob[expert]) / sum(weight)`` over supervised rows."""
    if not isinstance(output, PolicyOutput):
        raise TrainingError(f"expected PolicyOutput, got {type(output).__name__}")
    if not isinstance(targets, Sequence) or len(targets) != output.logits.size(0):
        raise TrainingError("targets length must equal output decisions")
    mask = legal_mask if legal_mask is not None else output.legal_mask
    if mask is None:
        raise TrainingError("legal_mask is required (output or argument)")
    if not isinstance(mask, Tensor) or tuple(mask.shape) != tuple(output.logits.shape):
        raise TrainingError("legal_mask must match logits shape")
    if not bool(mask.dtype is torch.bool):
        raise TrainingError("legal_mask must be torch.bool")
    if not bool(torch.isfinite(output.logits).all()):
        raise TrainingError("model produced NaN/Inf logits")
    if not bool(torch.isfinite(output.logprobs[mask]).all()):
        raise TrainingError("model produced NaN/Inf logprob on a legal action")
    weighted: list[Tensor] = []
    weights: list[float] = []
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise TrainingError(f"target[{index}] must be a mapping")
        weight = _target_weight(target)
        if weight is None:
            continue
        action = target.get("expert_action")
        if (isinstance(action, bool) or not isinstance(action, int)
                or not 0 <= action < ACTION_SPACE):
            raise TrainingError(f"target[{index}] expert_action invalid: {action!r}")
        if not bool(mask[index, action]):
            raise TrainingError(
                f"target[{index}] expert_action {action} is outside the legal mask")
        logprob = output.logprobs[index, action]
        if not bool(torch.isfinite(logprob)):
            raise TrainingError(f"target[{index}] logprob is NaN/Inf")
        weighted.append(-logprob * weight)
        weights.append(weight)
    if not weighted:
        zero = output.logits.sum() * 0.0
        return BCLossResult(zero, 0, len(targets), 0.0)
    total_weight = float(sum(weights))
    loss = torch.stack(weighted).sum() / total_weight
    if not bool(torch.isfinite(loss)):
        raise TrainingError("BC loss became NaN/Inf")
    return BCLossResult(loss, len(weighted), len(targets), total_weight)


def training_step(model: Policy, batch: ModelBatch, targets: Sequence[Mapping], *,
                  optimizer: torch.optim.Optimizer, scaler=None,
                  scheduler=None, max_grad_norm: float = 1.0,
                  bf16: bool = False, accumulation_step: int = 0,
                  accumulation_count: int = 1) -> BCLossResult:
    """One optimizer step over one already-padded decision batch."""
    if not isinstance(model, Policy):
        raise TrainingError("model must be Policy")
    if not isinstance(batch, ModelBatch):
        raise TrainingError("batch must be ModelBatch")
    if accumulation_count < 1:
        raise TrainingError("accumulation_count must be >= 1")
    if accumulation_step < 0:
        raise TrainingError("accumulation_step must be >= 0")
    model.train()
    if accumulation_step % accumulation_count == 0:
        optimizer.zero_grad(set_to_none=True)
    device_type = "cuda" if batch.padding_mask.is_cuda else "cpu"
    autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                              enabled=bool(bf16))
    with autocast:
        output = model(batch)
        result = masked_bc_loss(output, targets)
    if result.valid_decisions == 0:
        return result
    scaled = result.loss
    boundary = (accumulation_step + 1) % accumulation_count == 0
    if scaler is not None:
        scaler.scale(scaled).backward()
        if boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
    else:
        scaled.backward()
        if boundary:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
    if boundary and scheduler is not None:
        scheduler.step()
    return result


__all__ = ["BCLossResult", "TrainingError", "masked_bc_loss", "training_step"]
