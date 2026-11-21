#!/usr/bin/env python3
"""PR-45 offline diagnostics on the read-only validation stream.

Metrics (all natural-log, all *diagnostic*):

* ``ce``/``nll`` over supervised, legal, non-censored targets only;
* ``top1``/``top3`` against the native legal mask (strict ranking: an action is
  top-k only if fewer than k legal actions score strictly higher);
* illegal-target and non-finite-logprob rates (must be zero);
* an exclusive phase partition (self-turn / call-response / kan-selection) and an
  explicit action-family partition;
* overlapping scenario slices derived from labels and public state (riichi, call,
  kan, win, kyuushu, pass, opponent-riichi defence proxy, last-turn proxy, south
  hand, score-behind);
* sequence coverage: prefix cutoff vs full stream length, ``max_events``
  pressure, how many decisions still had future events.

These numbers measure imitation of the corpus.  A low CE on weak self-play bots
only means "reproduces weak bots"; strength needs the arena, not CE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
from torch import Tensor

from ..schemas.action_v3 import ACTION_SPACE

__all__ = [
    "MetricError",
    "compute_metrics",
    "phase_partition",
]

PHASES = ("SELF_TURN", "CALL_RESPONSE", "KAN_SELECTION")
_FAMILY_RANGES = (
    ("discard", 0, 37),
    ("riichi", 37, 38),
    ("chi", 38, 41),
    ("pon", 41, 42),
    ("kan", 42, 43),
    ("agari", 43, 44),
    ("kyuushu", 44, 45),
    ("pass", 45, 46),
)
_OVERLAP_NAMES = (
    "riichi_label", "call_label", "kan_label", "win_label", "kyuushu_label",
    "pass_label", "opponent_riichi_defence_proxy", "last_turn_proxy",
    "south_kyoku", "score_behind", "own_riichi",
)


class MetricError(RuntimeError):
    """A metric contract violation (illegal target / non-finite output)."""


def _wilson(successes: int, total: int, z: float = 1.96):
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _mean_ci(values: Sequence[float], z: float = 1.96):
    if not values:
        return None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return [mean, mean]
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half = z * math.sqrt(variance / len(values))
    return [mean - half, mean + half]


def _percentile(values: Sequence[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


@dataclass
class Bucket:
    count: int = 0
    nll_sum: float = 0.0
    top1: int = 0
    top3: int = 0
    nlls: list[float] = field(default_factory=list)

    def add(self, nll: float, top1: bool, top3: bool) -> None:
        self.count += 1
        self.nll_sum += nll
        self.nlls.append(nll)
        self.top1 += int(top1)
        self.top3 += int(top3)

    def as_dict(self) -> dict:
        if not self.count:
            return {"count": 0, "nll": None, "top1": None, "top3": None}
        return {
            "count": self.count,
            "nll": self.nll_sum / self.count,
            "top1": self.top1 / self.count,
            "top3": self.top3 / self.count,
            "top1_ci": _wilson(self.top1, self.count),
            "top3_ci": _wilson(self.top3, self.count),
        }


def _check_inputs(logprobs: Tensor, legal_mask: Tensor, targets: Sequence[Mapping],
                  contexts: Sequence, cutoffs: Sequence[int],
                  lengths: Sequence[int]) -> int:
    if not isinstance(logprobs, Tensor) or logprobs.dim() != 2 \
            or logprobs.size(1) != ACTION_SPACE:
        raise MetricError(f"logprobs must be [N,{ACTION_SPACE}]")
    if not isinstance(legal_mask, Tensor) or legal_mask.dtype is not torch.bool \
            or tuple(legal_mask.shape) != tuple(logprobs.shape):
        raise MetricError("legal_mask must be bool [N,46] matching logprobs")
    n = int(logprobs.size(0))
    if n <= 0:
        raise MetricError("need at least one decision")
    if not (len(targets) == len(contexts) == len(cutoffs) == len(lengths) == n):
        raise MetricError("targets/contexts/cutoffs/lengths must match rows")
    total_legal = int(legal_mask.sum().item())
    if total_legal <= 0:
        raise MetricError("legal mask is empty")
    finite_legal = torch.isfinite(logprobs[legal_mask])
    if not bool(finite_legal.all()):
        bad = int((~finite_legal).sum().item())
        raise MetricError(f"{bad} legal logprobs are NaN/Inf")
    return n


def compute_metrics(*, logprobs: Tensor, legal_mask: Tensor,
                    targets: Sequence[Mapping], contexts: Sequence,
                    cutoffs: Sequence[int], lengths: Sequence[int],
                    max_events: int | None = None) -> dict:
    n = _check_inputs(logprobs, legal_mask, targets, contexts, cutoffs, lengths)
    phases = {name: Bucket() for name in PHASES}
    families = {name: Bucket() for name, _lo, _hi in _FAMILY_RANGES}
    overlaps = {name: Bucket() for name in _OVERLAP_NAMES}
    overall = Bucket()
    nlls: list[float] = []
    supervised = excluded = 0
    illegal = 0
    nonfinite_rows = 0
    masked = logprobs.masked_fill(~legal_mask, float("-inf"))
    for row in range(n):
        target = targets[row]
        if not bool(target.get("supervision_valid", False)) \
                or target.get("expert_action") is None:
            excluded += 1
            continue
        action = int(target["expert_action"])
        if not 0 <= action < ACTION_SPACE:
            illegal += 1
            continue
        if not bool(legal_mask[row, action]):
            illegal += 1
            continue
        logprob = logprobs[row, action]
        if not bool(torch.isfinite(logprob)):
            nonfinite_rows += 1
            continue
        row_probs = masked[row]
        rank = int((row_probs > logprob).sum().item())
        nll = float(-logprob)
        nlls.append(nll)
        supervised += 1
        overall.add(nll, rank == 0, rank < 3)
        context = contexts[row]
        phase = getattr(context, "phase", "UNKNOWN")
        phases.setdefault(phase, Bucket()).add(nll, rank == 0, rank < 3)
        family = next((name for name, lo, hi in _FAMILY_RANGES if lo <= action < hi),
                      "other")
        families.setdefault(family, Bucket()).add(nll, rank == 0, rank < 3)
        flags = {
            "riichi_label": action == 37,
            "call_label": 38 <= action <= 42,
            "kan_label": action == 42 or phase == "KAN_SELECTION",
            "win_label": action == 43,
            "kyuushu_label": action == 44,
            "pass_label": action == 45,
            "opponent_riichi_defence_proxy": getattr(context, "opponent_riichi", 0) > 0,
            "last_turn_proxy": getattr(context, "tiles_left", 999) <= 12,
            "south_kyoku": bool(getattr(context, "is_south", False)),
            "score_behind": bool(getattr(context, "behind", False)),
            "own_riichi": getattr(context, "own_riichi", "none") != "none",
        }
        for name, enabled in flags.items():
            if enabled:
                overlaps[name].add(nll, rank == 0, rank < 3)
    if illegal:
        raise MetricError(
            f"{illegal} supervised target(s) outside the legal mask: data contract "
            "violation, not a metric to report as zero")
    if supervised == 0:
        raise MetricError("no supervised decisions to evaluate")
    phase_counts = {name: bucket.count for name, bucket in phases.items()}
    family_counts = {name: bucket.count for name, bucket in families.items()}
    if sum(phase_counts.values()) != supervised:
        raise MetricError("phase partition does not sum to supervised decisions")
    if sum(family_counts.values()) != supervised:
        raise MetricError("family partition does not sum to supervised decisions")
    over_cap = [index for index, cutoff in enumerate(cutoffs)
                if max_events is not None and int(cutoff) > max_events]
    coverage = {
        "decisions": n,
        "supervised": supervised,
        "excluded_censored": excluded,
        "max_prefix": max(int(value) for value in cutoffs),
        "mean_prefix": sum(int(value) for value in cutoffs) / n,
        "p95_prefix": _percentile([int(value) for value in cutoffs], 0.95),
        "max_stream": max(int(value) for value in lengths),
        "mean_stream": sum(int(value) for value in lengths) / n,
        "mean_prefix_fraction": sum(int(cutoff) / max(1, int(length))
                                    for cutoff, length in zip(cutoffs, lengths)) / n,
        "decisions_with_future_events": sum(
            1 for cutoff, length in zip(cutoffs, lengths) if int(cutoff) < int(length)),
        "decisions_at_stream_end": sum(
            1 for cutoff, length in zip(cutoffs, lengths) if int(cutoff) >= int(length)),
        "max_events": max_events,
        "decisions_over_max_events": len(over_cap),
        "truncated": len(over_cap) > 0,
    }
    if over_cap:
        raise MetricError(
            f"{len(over_cap)} decisions exceed max_events={max_events}; the loader "
            "must fail closed instead of truncating")
    return {
        "total_decisions": n,
        "supervised": supervised,
        "excluded": excluded,
        "illegal_supervised": illegal,
        "illegal_rate": illegal / n,
        "nonfinite_logprob_rows": nonfinite_rows,
        "nonfinite_rate": nonfinite_rows / n,
        "nll": overall.as_dict()["nll"],
        "ce": overall.as_dict()["nll"],
        "ce_units": "natural log; ce == nll",
        "nll_ci": _mean_ci(nlls),
        "top1": overall.as_dict()["top1"],
        "top3": overall.as_dict()["top3"],
        "top1_ci": overall.as_dict()["top1_ci"],
        "top3_ci": overall.as_dict()["top3_ci"],
        "phase_buckets": {name: phases[name].as_dict() for name in PHASES},
        "phase_partition_sum": sum(phase_counts.values()),
        "action_family_buckets": {name: bucket.as_dict()
                                  for name, bucket in sorted(families.items())},
        "action_family_sum": sum(family_counts.values()),
        "overlap": {name: overlaps[name].as_dict() for name in _OVERLAP_NAMES},
        "overlap_is_exclusive": False,
        "sequence_coverage": coverage,
        "rank_rule": "strict: top-k requires fewer than k legal actions to score higher",
    }


def phase_partition(phase: str) -> bool:
    """True when ``phase`` is one of the frozen exclusive phases."""
    return phase in PHASES
