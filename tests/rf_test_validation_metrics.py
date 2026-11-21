#!/usr/bin/env python3
"""PR-45 metrics contracts: CE/top-k, partitions, coverage, fail-closed inputs."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from riichiformer.schemas.action_v3 import ACTION_SPACE  # noqa: E402
from riichiformer.validation.metrics import (  # noqa: E402
    MetricError,
    compute_metrics,
)


@dataclass(frozen=True)
class _Context:
    phase: str
    opponent_riichi: int = 0
    tiles_left: int = 70
    is_south: bool = False
    behind: bool = False
    own_riichi: str = "none"


def _fixture():
    # Four decisions:
    #   0: self-turn discard, target 0, two legal actions, target ranked first
    #   1: call-response pass, target 45, censored (excluded)
    #   2: kan-selection, target 6, three legal actions, target ranked second
    #   3: self-turn riichi, target 37, two legal actions, target ranked first
    rows = [
        ([0, 3], 0, True),
        ([38], 45, False),
        ([5, 6, 7], 6, True),
        ([37, 8], 37, True),
    ]
    n = len(rows)
    legal = torch.zeros((n, ACTION_SPACE), dtype=torch.bool)
    logprobs = torch.full((n, ACTION_SPACE), -float("inf"))
    targets = []
    contexts = []
    for index, (actions, target, supervised) in enumerate(rows):
        legal[index, actions] = True
        values = torch.arange(len(actions), dtype=torch.float32) * -1.0
        probs = torch.softmax(values, dim=0)
        logprobs[index, actions] = probs.log()
        targets.append({"expert_action": target, "supervision_valid": supervised,
                        "censor_reason": None if supervised else "preempted_by_call",
                        "rank_final": 1})
        contexts.append(_Context(
            phase=("KAN_SELECTION" if index == 2 else
                   "CALL_RESPONSE" if index == 1 else "SELF_TURN"),
            opponent_riichi=1 if index == 3 else 0,
            tiles_left=10 if index == 1 else 70,
            is_south=index == 2,
            behind=index == 0,
            own_riichi="accepted" if index == 3 else "none"))
    return logprobs, legal, targets, contexts


def test_metrics_partitions_and_values():
    logprobs, legal, targets, contexts = _fixture()
    result = compute_metrics(
        logprobs=logprobs, legal_mask=legal, targets=targets, contexts=contexts,
        cutoffs=[10, 20, 30, 40], lengths=[100, 20, 30, 40], max_events=64)
    assert result["total_decisions"] == 4
    assert result["supervised"] == 3
    assert result["excluded"] == 1
    assert result["illegal_supervised"] == 0
    assert result["nonfinite_rate"] == 0.0
    expected_ce = sum(
        -float(logprobs[index, int(target["expert_action"])])
        for index, target in enumerate(targets)
        if target["supervision_valid"]) / 3
    assert result["ce"] == pytest.approx(expected_ce)
    assert result["nll"] == result["ce"]
    # rows 0 and 3 are ranked first, row 2 is ranked second -> top1 = 2/3
    assert result["top1"] == pytest.approx(2 / 3)
    assert result["top3"] == pytest.approx(1.0)
    assert result["phase_partition_sum"] == result["supervised"]
    assert result["action_family_sum"] == result["supervised"]
    assert set(result["phase_buckets"]) == {"SELF_TURN", "CALL_RESPONSE",
                                            "KAN_SELECTION"}
    assert result["phase_buckets"]["CALL_RESPONSE"]["count"] == 0
    overlap = result["overlap"]
    assert overlap["riichi_label"]["count"] == 1
    assert overlap["kan_label"]["count"] == 1
    assert overlap["south_kyoku"]["count"] == 1
    assert overlap["score_behind"]["count"] == 1
    assert overlap["opponent_riichi_defence_proxy"]["count"] == 1
    assert overlap["last_turn_proxy"]["count"] == 0  # censored row does not count
    assert result["overlap_is_exclusive"] is False
    coverage = result["sequence_coverage"]
    assert coverage["decisions_with_future_events"] == 1
    assert coverage["decisions_at_stream_end"] == 3
    assert coverage["truncated"] is False
    assert coverage["max_prefix"] == 40


def test_metrics_rejects_illegal_target():
    logprobs, legal, targets, contexts = _fixture()
    legal[0, 0] = False  # the supervised target is no longer legal
    with pytest.raises(MetricError, match="outside the legal mask"):
        compute_metrics(logprobs=logprobs, legal_mask=legal, targets=targets,
                        contexts=contexts, cutoffs=[1, 2, 3, 4],
                        lengths=[10, 10, 10, 10], max_events=64)


def test_metrics_rejects_nonfinite_legal_logprob():
    logprobs, legal, targets, contexts = _fixture()
    logprobs[0, 3] = float("nan")
    with pytest.raises(MetricError, match="NaN/Inf"):
        compute_metrics(logprobs=logprobs, legal_mask=legal, targets=targets,
                        contexts=contexts, cutoffs=[1, 2, 3, 4],
                        lengths=[10, 10, 10, 10], max_events=64)


def test_metrics_rejects_prefix_over_max_events():
    logprobs, legal, targets, contexts = _fixture()
    with pytest.raises(MetricError, match="max_events"):
        compute_metrics(logprobs=logprobs, legal_mask=legal, targets=targets,
                        contexts=contexts, cutoffs=[10, 20, 30, 40],
                        lengths=[100, 100, 100, 100], max_events=32)


def test_metrics_rejects_no_supervised():
    logprobs, legal, targets, contexts = _fixture()
    for target in targets:
        target["supervision_valid"] = False
    with pytest.raises(MetricError, match="no supervised"):
        compute_metrics(logprobs=logprobs, legal_mask=legal, targets=targets,
                        contexts=contexts, cutoffs=[1, 2, 3, 4],
                        lengths=[10, 10, 10, 10], max_events=64)


def test_metrics_rejects_shape_mismatch():
    logprobs, legal, targets, contexts = _fixture()
    with pytest.raises(MetricError):
        compute_metrics(logprobs=logprobs[:, :45], legal_mask=legal[:, :45],
                        targets=targets, contexts=contexts,
                        cutoffs=[1, 2, 3, 4], lengths=[10, 10, 10, 10])
    with pytest.raises(MetricError):
        compute_metrics(logprobs=logprobs, legal_mask=legal, targets=targets[:2],
                        contexts=contexts, cutoffs=[1, 2, 3, 4],
                        lengths=[10, 10, 10, 10])
