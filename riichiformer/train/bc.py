#!/usr/bin/env python3
"""PR-14 masked behavior-cloning loss and one-step trainer.

Only supervised, in-mask expert decisions contribute.  Censored/None targets and forced
actions remain context in the batch but are excluded from the policy loss.  Targets are
plain dictionaries from PR-12; this module imports no target/oracle class.

Gradient accumulation (PR-39)
-----------------------------
A microbatch must never normalize by its **own** weight sum: with
``accumulation_count > 1`` that is not equivalent to one large batch.  The loss
numerator (``sum_i weight_i * -logprob_i``) is additive, so the correct update
divides every microbatch numerator by the **window** weight sum
(``window_weight_total``, computed from the targets before the first forward) and
backward-propagates ``numerator / window_total``.  :func:`training_step`
implements that and fails closed when ``accumulation_count > 1`` is requested
without the window total (or with an explicit ``loss_scale``).  An all-censored
microbatch performs no backward but still triggers the window's optimizer step
when earlier microbatches contributed gradients.

Both forward paths share that body: :func:`training_step` consumes a
:class:`~riichiformer.models.batch.ModelBatch` (one prefix row per decision) and
:func:`training_step_multi` consumes a
:class:`~riichiformer.models.batch.MultiDecisionBatch` (one shared event row per
stream plus one marker per decision).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from ..models.batch import ModelBatch, MultiDecisionBatch
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
    """Local weighted mean loss plus the (additive) numerator used for accumulation."""

    loss: Tensor
    valid_decisions: int
    total_decisions: int
    total_weight: float
    #: ``sum(weight * -logprob)`` graph-connected tensor (None when no supervision).
    numerator: Tensor | None = None

    def __post_init__(self):
        if self.valid_decisions < 0 or self.valid_decisions > self.total_decisions:
            raise TrainingError("invalid valid/total decision counts")
        if not bool(torch.isfinite(self.loss).all()):
            raise TrainingError("BC loss is NaN/Inf")
        if self.numerator is not None \
                and not bool(torch.isfinite(self.numerator).all()):
            raise TrainingError("BC loss numerator is NaN/Inf")


def valid_weight_sum(targets: Sequence[Mapping]) -> float:
    """Sum of valid label weights (no forward pass, no target action check)."""
    if not isinstance(targets, Sequence):
        raise TrainingError("targets must be a sequence")
    total = 0.0
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise TrainingError(f"target[{index}] must be a mapping")
        weight = _target_weight(target)
        if weight is not None:
            total += weight
    return total


def window_weight_total(targets_per_step: Sequence[Sequence[Mapping]]) -> float:
    """Sum of valid label weights over one accumulation window.

    This is the shared denominator that makes ``k`` microbatch backward passes
    equal one large batch.  It is computed from the target metadata only, so it
    can be computed *before* the first forward pass.
    """
    if not isinstance(targets_per_step, Sequence) or not targets_per_step:
        raise TrainingError("targets_per_step must be a non-empty sequence")
    totals = [valid_weight_sum(targets) for targets in targets_per_step]
    total = float(sum(totals))
    if not torch.isfinite(torch.tensor(total)) or total < 0.0:
        raise TrainingError(f"window weight total is invalid: {total!r}")
    return total


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
        return BCLossResult(zero, 0, len(targets), 0.0, numerator=zero)
    total_weight = float(sum(weights))
    numerator = torch.stack(weighted).sum()
    loss = numerator / total_weight
    if not bool(torch.isfinite(loss)):
        raise TrainingError("BC loss became NaN/Inf")
    return BCLossResult(loss, len(weighted), len(targets), total_weight,
                        numerator=numerator)


def _check_accumulation_args(accumulation_step: int, accumulation_count: int,
                             accumulation_weight_total: float | None,
                             loss_scale: float | None
                             ) -> tuple[float | None, float | None]:
    """Validate the PR-39 accumulation contract shared by both forward paths."""
    if accumulation_count < 1:
        raise TrainingError("accumulation_count must be >= 1")
    if accumulation_step < 0:
        raise TrainingError("accumulation_step must be >= 0")
    if accumulation_weight_total is not None:
        if isinstance(accumulation_weight_total, bool) \
                or not isinstance(accumulation_weight_total, (int, float)):
            raise TrainingError("accumulation_weight_total must be numeric")
        accumulation_weight_total = float(accumulation_weight_total)
        if not torch.isfinite(torch.tensor(accumulation_weight_total)) \
                or accumulation_weight_total < 0:
            raise TrainingError(
                "accumulation_weight_total must be finite and >= 0")
    if loss_scale is not None:
        if accumulation_weight_total is not None:
            raise TrainingError(
                "pass either accumulation_weight_total or loss_scale, not both")
        if isinstance(loss_scale, bool) or not isinstance(loss_scale, (int, float)):
            raise TrainingError("loss_scale must be numeric")
        loss_scale = float(loss_scale)
        if not torch.isfinite(torch.tensor(loss_scale)) or loss_scale <= 0:
            raise TrainingError("loss_scale must be finite and > 0")
    return accumulation_weight_total, loss_scale


def _training_step_core(model: Policy, batch, targets: Sequence[Mapping], forward, *,
                        optimizer: torch.optim.Optimizer, scaler=None,
                        scheduler=None, max_grad_norm: float = 1.0,
                        bf16: bool = False, accumulation_step: int = 0,
                        accumulation_count: int = 1,
                        accumulation_weight_total: float | None = None,
                        loss_scale: float | None = None) -> BCLossResult:
    """Shared forward/backward/optimizer body for prefix and multi batches.

    ``forward`` receives the batch and returns a :class:`PolicyOutput`; both paths
    therefore use exactly the same loss, accumulation and optimizer semantics.
    """
    model.train()
    if accumulation_step % accumulation_count == 0:
        optimizer.zero_grad(set_to_none=True)
    device_type = "cuda" if batch.padding_mask.is_cuda else "cpu"
    autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                              enabled=bool(bf16))
    with autocast:
        output = forward(batch)
        result = masked_bc_loss(output, targets)
    boundary = (accumulation_step + 1) % accumulation_count == 0
    has_gradient = any(parameter.grad is not None
                       for parameter in model.parameters())
    if result.valid_decisions > 0:
        numerator = result.numerator
        if numerator is None:  # pragma: no cover - masked_bc_loss always sets it
            raise TrainingError("BC loss numerator missing for supervised rows")
        if loss_scale is not None:
            scaled = numerator * loss_scale
        else:
            denominator = accumulation_weight_total
            if denominator is None:
                if accumulation_count != 1:
                    raise TrainingError(
                        "accumulation_count > 1 requires accumulation_weight_total "
                        "(window weight sum) or an explicit loss_scale; refusing to "
                        "normalize by this microbatch's own weight")
                denominator = result.total_weight
            if denominator <= 0.0:
                raise TrainingError(
                    "accumulation weight total must be > 0 for a supervised "
                    f"microbatch, got {denominator}")
            scaled = numerator / denominator
    else:
        scaled = None
    if scaled is not None:
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()
        has_gradient = True
    if boundary and has_gradient:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return result


def training_step(model: Policy, batch: ModelBatch, targets: Sequence[Mapping], *,
                  optimizer: torch.optim.Optimizer, scaler=None,
                  scheduler=None, max_grad_norm: float = 1.0,
                  bf16: bool = False, accumulation_step: int = 0,
                  accumulation_count: int = 1,
                  accumulation_weight_total: float | None = None,
                  loss_scale: float | None = None) -> BCLossResult:
    """One microbatch of an optimizer step with exact accumulation weighting.

    ``accumulation_weight_total`` is the sum of valid label weights over the whole
    accumulation window (:func:`window_weight_total`).  Every microbatch
    back-propagates ``numerator / accumulation_weight_total``, so the accumulated
    gradient equals one large batch instead of the average of per-microbatch means.
    With ``accumulation_count == 1`` the local weight sum is used when the total
    is omitted.  ``loss_scale`` is an explicit alternative multiplier (the caller
    owns the normalization); it cannot be combined with the window total.
    """
    if not isinstance(model, Policy):
        raise TrainingError("model must be Policy")
    if not isinstance(batch, ModelBatch):
        raise TrainingError("batch must be ModelBatch")
    accumulation_weight_total, loss_scale = _check_accumulation_args(
        accumulation_step, accumulation_count, accumulation_weight_total, loss_scale)
    return _training_step_core(
        model, batch, targets, model, optimizer=optimizer, scaler=scaler,
        scheduler=scheduler, max_grad_norm=max_grad_norm, bf16=bf16,
        accumulation_step=accumulation_step,
        accumulation_count=accumulation_count,
        accumulation_weight_total=accumulation_weight_total, loss_scale=loss_scale)


def training_step_multi(model: Policy, batch: MultiDecisionBatch,
                        targets: Sequence[Mapping], *,
                        optimizer: torch.optim.Optimizer, scaler=None,
                        scheduler=None, max_grad_norm: float = 1.0,
                        bf16: bool = False, accumulation_step: int = 0,
                        accumulation_count: int = 1,
                        accumulation_weight_total: float | None = None,
                        loss_scale: float | None = None) -> BCLossResult:
    """PR-38 counterpart of :func:`training_step` for a shared-event batch.

    The multi-decision batch is run through :meth:`Policy.forward_multi` (one
    transformer pass per stream, one marker per decision) and every decision is
    matched to its target row in marker order.  Loss, weight validation,
    gradient-accumulation normalization (``window_total``/``loss_scale``) and
    optimizer/scheduler stepping are **the same code path** as
    :func:`training_step`; only the forward differs.  Models must be in the
    default ``dropout == 0`` regime for multi/prefix parity, but training itself
    has no extra restriction.
    """
    if not isinstance(model, Policy):
        raise TrainingError("model must be Policy")
    if not isinstance(batch, MultiDecisionBatch):
        raise TrainingError("batch must be MultiDecisionBatch")
    accumulation_weight_total, loss_scale = _check_accumulation_args(
        accumulation_step, accumulation_count, accumulation_weight_total, loss_scale)
    return _training_step_core(
        model, batch, targets, model.forward_multi, optimizer=optimizer,
        scaler=scaler, scheduler=scheduler, max_grad_norm=max_grad_norm,
        bf16=bf16, accumulation_step=accumulation_step,
        accumulation_count=accumulation_count,
        accumulation_weight_total=accumulation_weight_total, loss_scale=loss_scale)


__all__ = [
    "BCLossResult", "TrainingError", "masked_bc_loss", "training_step",
    "training_step_multi", "valid_weight_sum", "window_weight_total",
]
