#!/usr/bin/env python3
"""PR-15 offline diagnostics: calibrated NLL/Top-k and explicit bucket reporting.

Only supervised, non-forced targets contribute.  Phase and action-family are mutually
exclusive partitions (their counts sum to the supervised total); riichi/call/kan/south/
score-diff/shanten are overlapping slices and are labelled as such.  NLL/Top-k are
offline imitation metrics, not arena strength or Tenhou rank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import Tensor

from ..data.encode import StreamMeta, encode_batch
from ..models.batch import ModelBatch
from ..models.policy import Policy
from ..schemas.action_v3 import ACTION_SPACE, ActionPhase, decode_action
from ..schemas.decision_v3 import DecisionInput


class OfflineEvalError(RuntimeError):
    """Malformed metric input or evaluation contract violation."""


def _valid_target(target: Mapping) -> bool:
    return (bool(target.get("supervision_valid", False))
            and target.get("expert_action") is not None
            and not bool(target.get("forced", False))
            and target.get("loss_weight", 1.0) != 0)


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _mean_ci(values: Sequence[float], z: float = 1.96) -> tuple[float, float] | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return (mean, mean)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half = z * math.sqrt(variance / len(values))
    return (mean - half, mean + half)


@dataclass(frozen=True)
class RowCategory:
    phase: str
    action_family: str
    riichi: str = "none"
    involved_call: bool = False
    kan: bool = False
    south: bool = False
    score_diff: str = "even"
    shanten: str = "unknown"


@dataclass
class BucketStats:
    count: int = 0
    nll: float = 0.0
    top1: int = 0
    top3: int = 0
    _nlls: list[float] = field(default_factory=list)

    def add(self, nll: float, top1: bool, top3: bool) -> None:
        self.count += 1
        self._nlls.append(nll)
        self.nll += nll
        self.top1 += int(top1)
        self.top3 += int(top3)

    def as_dict(self) -> dict:
        if self.count == 0:
            return {"count": 0, "nll": None, "top1": None, "top3": None,
                    "top1_ci": None, "top3_ci": None, "nll_ci": None}
        return {
            "count": self.count,
            "nll": self.nll / self.count,
            "top1": self.top1 / self.count,
            "top3": self.top3 / self.count,
            "top1_ci": _wilson(self.top1, self.count),
            "top3_ci": _wilson(self.top3, self.count),
            "nll_ci": _mean_ci(self._nlls),
        }


def _action_family(target: Mapping, phase: ActionPhase) -> str:
    action = target.get("expert_action")
    if action is None:
        return "none"
    try:
        kind = decode_action(int(action), phase).spec.kind.value
    except Exception:
        return "invalid"
    return "discard" if kind in ("discard", "discard_aka") else kind


def _score_diff_bucket(score: int) -> str:
    diff = int(score) - 25000
    if diff <= -10000:
        return "behind_10k+"
    if diff < 0:
        return "behind_1_9999"
    if diff == 0:
        return "even"
    if diff < 10000:
        return "ahead_1_9999"
    return "ahead_10k+"


def category_from_decision(decision: DecisionInput, target: Mapping) -> RowCategory:
    phase = decision.phase
    family = _action_family(target, phase)
    self_state = decision.players_public[0]
    public = decision.global_public
    south = public.bakaze.kind == 28 and public.kyoku >= 3
    return RowCategory(
        phase=phase.value,
        action_family=family,
        riichi=self_state.riichi.value,
        involved_call=family in ("chi", "pon", "kan"),
        kan=(family == "kan" or phase == ActionPhase.KAN_SELECTION),
        south=south,
        score_diff=_score_diff_bucket(self_state.score),
        shanten="unknown",
    )


def _check_tensors(logprobs: Tensor, legal_mask: Tensor) -> tuple[int, int]:
    if not isinstance(logprobs, Tensor) or logprobs.dim() != 2 \
            or logprobs.size(1) != ACTION_SPACE:
        raise OfflineEvalError(f"logprobs must be [N,{ACTION_SPACE}]")
    if not isinstance(legal_mask, Tensor) or legal_mask.dtype is not torch.bool \
            or tuple(legal_mask.shape) != tuple(logprobs.shape):
        raise OfflineEvalError("legal_mask must be bool [N,46] matching logprobs")
    n = int(logprobs.size(0))
    if n <= 0:
        raise OfflineEvalError("need at least one decision")
    if not bool(torch.isfinite(logprobs[legal_mask]).all()):
        raise OfflineEvalError("legal logprobs contain NaN/Inf")
    if not bool(legal_mask.any(dim=1).all()):
        raise OfflineEvalError("every row needs a legal action")
    return n, int(legal_mask.sum().item())


def evaluate_rows(logprobs: Tensor, legal_mask: Tensor, targets: Sequence[Mapping],
                  categories: Sequence[RowCategory] | None = None) -> dict:
    """Evaluate fixed log-probs/masks/targets; deterministic and target-isolated."""
    n, _ = _check_tensors(logprobs, legal_mask)
    if len(targets) != n:
        raise OfflineEvalError("targets length must equal rows")
    if categories is not None and len(categories) != n:
        raise OfflineEvalError("categories length must equal rows")
    phase_buckets: dict[str, BucketStats] = {}
    family_buckets: dict[str, BucketStats] = {}
    overlap = {name: BucketStats() for name in
               ("riichi", "involved_call", "kan", "south", "score_diff", "shanten")}
    overlap_counts: dict[str, int] = {name: 0 for name in overlap}
    all_stats = BucketStats()
    nlls: list[float] = []
    top1 = top3 = 0
    supervised = excluded = 0
    for row, target in enumerate(targets):
        if not _valid_target(target):
            excluded += 1
            continue
        action = int(target["expert_action"])
        if not 0 <= action < ACTION_SPACE:
            raise OfflineEvalError(f"target[{row}] action outside 0..45")
        if not bool(legal_mask[row, action]):
            raise OfflineEvalError(f"target[{row}] action {action} is outside legal mask")
        logprob = logprobs[row, action]
        if not bool(torch.isfinite(logprob)):
            raise OfflineEvalError(f"target[{row}] logprob is NaN/Inf")
        legal_probs = logprobs[row].masked_fill(~legal_mask[row], float("-inf"))
        order = torch.argsort(legal_probs, descending=True)
        rank = int((order == action).nonzero(as_tuple=False)[0].item())
        nll = float(-logprob)
        nlls.append(nll)
        top1 += int(rank == 0)
        top3 += int(rank < 3)
        supervised += 1
        all_stats.add(nll, rank == 0, rank < 3)
        if categories is not None:
            category = categories[row]
            phase_buckets.setdefault(category.phase, BucketStats()).add(
                nll, rank == 0, rank < 3)
            family_buckets.setdefault(category.action_family, BucketStats()).add(
                nll, rank == 0, rank < 3)
            if category.riichi != "none":
                overlap["riichi"].add(nll, rank == 0, rank < 3)
                overlap_counts["riichi"] += 1
            if category.involved_call:
                overlap["involved_call"].add(nll, rank == 0, rank < 3)
                overlap_counts["involved_call"] += 1
            if category.kan:
                overlap["kan"].add(nll, rank == 0, rank < 3)
                overlap_counts["kan"] += 1
            if category.south:
                overlap["south"].add(nll, rank == 0, rank < 3)
                overlap_counts["south"] += 1
            overlap["score_diff"].add(nll, rank == 0, rank < 3)
            overlap_counts["score_diff"] += 1
            overlap["shanten"].add(nll, rank == 0, rank < 3)
            overlap_counts["shanten"] += 1
    if supervised == 0:
        raise OfflineEvalError("no supervised decisions to evaluate")
    phase_counts = {name: bucket.count for name, bucket in phase_buckets.items()}
    family_counts = {name: bucket.count for name, bucket in family_buckets.items()}
    if categories is not None:
        if sum(phase_counts.values()) != supervised:
            raise OfflineEvalError("phase buckets do not sum to supervised")
        if sum(family_counts.values()) != supervised:
            raise OfflineEvalError("action-family buckets do not sum to supervised")
    result = {
        "total_decisions": n,
        "supervised": supervised,
        "excluded": excluded,
        "nll": sum(nlls) / supervised,
        "top1": top1 / supervised,
        "top3": top3 / supervised,
        "top1_ci": _wilson(top1, supervised),
        "top3_ci": _wilson(top3, supervised),
        "nll_ci": _mean_ci(nlls),
        "phase_buckets": {name: bucket.as_dict() for name, bucket in
                          sorted(phase_buckets.items())},
        "action_family_buckets": {name: bucket.as_dict() for name, bucket in
                                  sorted(family_buckets.items())},
        "overlap": {name: bucket.as_dict() for name, bucket in
                    sorted(overlap.items())},
        "overlap_is_exclusive": False,
        "phase_sum": sum(phase_counts.values()),
        "action_family_sum": sum(family_counts.values()),
    }
    return result


def evaluate_reader(reader, model: Policy, *, batch_size: int = 32,
                    limit: int | None = None) -> dict:
    """Evaluate a PR-12 shard with a model using already-batched decisions."""
    if not isinstance(model, Policy):
        raise OfflineEvalError("model must be Policy")
    if batch_size <= 0:
        raise OfflineEvalError("batch_size must be positive")
    items = []
    logprob_rows: list[Tensor] = []
    mask_rows: list[Tensor] = []
    targets: list[dict] = []
    categories: list[RowCategory] = []
    model.eval()
    total = 0
    with torch.no_grad():
        for decision, target, meta, _game, _pov, _index in reader.iter_decisions():
            items.append((decision, target, meta))
            if len(items) >= batch_size:
                _eval_chunk(items, model, logprob_rows, mask_rows, targets, categories)
                total += len(items)
                items = []
                if limit is not None and total >= limit:
                    break
        if items:
            _eval_chunk(items, model, logprob_rows, mask_rows, targets, categories)
            total += len(items)
    if not logprob_rows:
        raise OfflineEvalError("shard yielded no decisions")
    if limit is not None and len(logprob_rows) > limit:
        logprob_rows = logprob_rows[:limit]
        mask_rows = mask_rows[:limit]
        targets = targets[:limit]
        categories = categories[:limit]
    result = evaluate_rows(torch.cat(logprob_rows, dim=0),
                           torch.cat(mask_rows, dim=0), targets, categories)
    result["shard"] = str(reader.path)
    result["source_id"] = reader.metadata.source_id
    result["train_allowed"] = reader.metadata.train_allowed
    result["purpose"] = reader.metadata.purpose
    result["projection_version"] = reader.metadata.projection_version
    result["warning"] = ("NLL/Top1 are offline imitation metrics, NOT arena strength "
                         "or Tenhou rank.")
    return result


def _eval_chunk(items, model, logprob_rows, mask_rows, targets, categories):
    decisions = [item[0] for item in items]
    metas = [item[2] for item in items]
    encoded = encode_batch(decisions, metas)
    batch = encoded.to_model_batch()
    output = model(batch)
    if not bool(torch.isfinite(output.logits).all()):
        raise OfflineEvalError("model logits contain NaN/Inf")
    logprob_rows.append(output.logprobs.detach().cpu())
    mask_rows.append(batch.legal_mask.detach().cpu())
    for decision, target, _meta in items:
        targets.append(dict(target))
        categories.append(category_from_decision(decision, target))


__all__ = [
    "BucketStats", "OfflineEvalError", "RowCategory", "category_from_decision",
    "evaluate_reader", "evaluate_rows",
]
