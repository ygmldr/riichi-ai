#!/usr/bin/env python3
"""PR-15 synthetic offline diagnostics tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.decision import build_decisions  # noqa: E402
from riichiformer.data.loader import ShardReader  # noqa: E402
from riichiformer.data.shard import (  # noqa: E402
    ACTION_HASH, NORMALIZER_HASH, SCHEMA_HASH, ShardMetadata, ShardWriter,
    make_game_record,
)
from riichiformer.eval.offline import (  # noqa: E402
    OfflineEvalError, RowCategory, evaluate_reader, evaluate_rows,
)
from riichiformer.models.policy import MVPConfig, Policy  # noqa: E402
from scripts.rf_offline_eval import main as eval_main  # noqa: E402


HAND0 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
         "1p", "2p", "3p", "4p"]
HAND1 = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
         "E", "S", "W", "N"]
HAND2 = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
         "7m", "8m", "9m", "1p"]
HAND3 = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
         "1s", "2s", "3s", "4s", "5s"]


def raw_four():
    return [
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1p",
         "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
         "scores": [25000] * 4, "tehais": [HAND0, HAND1, HAND2, HAND3]},
        {"type": "tsumo", "actor": 0, "pai": "5mr"},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "1s"},
        {"type": "dahai", "actor": 1, "pai": "1s", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "P"},
        {"type": "dahai", "actor": 2, "pai": "P", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "2p"},
        {"type": "dahai", "actor": 3, "pai": "2p", "tsumogiri": True},
        {"type": "end_kyoku"}, {"type": "end_game"},
    ]


def metadata(*, allowed=True):
    return ShardMetadata(
        purpose="train" if allowed else "audit",
        source_id="synthetic_allowed",
        rights_status="ALLOWED" if allowed else "UNKNOWN",
        train_allowed=allowed,
        canonical_manifest_sha256="a" * 64,
        canonical_index_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
        split_version="rf-split/2",
        schema_version="rf-schema-v3/2", schema_hash=SCHEMA_HASH,
        action_map_hash=ACTION_HASH, normalizer_hash=NORMALIZER_HASH,
    )


def build_shard(path: Path, games=2, *, allowed=True):
    raw = raw_four()
    samples = build_decisions(raw)
    writer = ShardWriter(path, metadata(allowed=allowed))
    for index in range(games):
        record = make_game_record(
            raw, samples, game_id=f"g{index:02d}",
            component_id=f"comp-{index}", component_sha256="d" * 64,
            split="train", source_id="synthetic_allowed",
            train_allowed=allowed,
            rights_status="ALLOWED" if allowed else "UNKNOWN")
        writer.add_game(record)
    writer.finalize()


def tiny_model() -> Policy:
    return Policy(MVPConfig(layers=1, ffn=32, state_hidden=32, max_positions=2048))


class CoreMetricTests(unittest.TestCase):
    def setUp(self):
        logits = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        padded = torch.full((5, 46), -100.0)
        padded[:, :3] = logits
        self.logprobs = torch.log_softmax(padded, dim=1)
        self.mask = torch.zeros((5, 46), dtype=torch.bool)
        self.mask[:, :3] = True
        self.targets = [
            {"expert_action": 0, "supervision_valid": True},
            {"expert_action": 0, "supervision_valid": True},
            {"expert_action": 0, "supervision_valid": True},
            {"expert_action": 0, "supervision_valid": False,
             "censor_reason": "preempted_by_call"},
            {"expert_action": 0, "supervision_valid": True, "forced": True},
        ]
        self.categories = [
            RowCategory("SELF_TURN", "discard", riichi="none"),
            RowCategory("SELF_TURN", "discard", riichi="accepted"),
            RowCategory("CALL_RESPONSE", "pon", involved_call=True),
            RowCategory("CALL_RESPONSE", "pass"),
            RowCategory("SELF_TURN", "discard", kan=True),
        ]

    def test_known_nll_top1_top3(self):
        result = evaluate_rows(self.logprobs, self.mask, self.targets,
                               self.categories)
        self.assertEqual(result["total_decisions"], 5)
        self.assertEqual(result["supervised"], 3)
        self.assertEqual(result["excluded"], 2)
        # row0 top1, row1 rank1, row2 rank2 -> top1=1/3, top3=3/3
        self.assertAlmostEqual(result["top1"], 1 / 3, places=6)
        self.assertAlmostEqual(result["top3"], 1.0, places=6)
        expected_nll = sum(
            -float(self.logprobs[row, 0]) for row in range(3)) / 3
        self.assertAlmostEqual(result["nll"], expected_nll, places=6)
        self.assertEqual(result["phase_sum"], 3)
        self.assertEqual(result["action_family_sum"], 3)
        self.assertFalse(result["overlap_is_exclusive"])

    def test_censored_and_forced_excluded(self):
        result = evaluate_rows(self.logprobs, self.mask, self.targets)
        self.assertEqual(result["supervised"], 3)
        self.assertEqual(result["excluded"], 2)

    def test_label_outside_mask_fails(self):
        targets = [dict(target) for target in self.targets]
        targets[0]["expert_action"] = 45  # illegal in this synthetic mask
        with self.assertRaises(OfflineEvalError):
            evaluate_rows(self.logprobs, self.mask, targets)

    def test_nan_and_empty_supervised_fail(self):
        bad = self.logprobs.clone()
        bad[0, 0] = float("nan")
        with self.assertRaises(OfflineEvalError):
            evaluate_rows(bad, self.mask, self.targets)
        with self.assertRaises(OfflineEvalError):
            evaluate_rows(self.logprobs, self.mask,
                          [dict(target, supervision_valid=False,
                                censor_reason="x") for target in self.targets])

    def test_deterministic(self):
        first = evaluate_rows(self.logprobs, self.mask, self.targets, self.categories)
        second = evaluate_rows(self.logprobs, self.mask, self.targets, self.categories)
        self.assertEqual(first, second)


class ShardEvalTests(unittest.TestCase):
    def test_reader_metrics_and_partitions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.rfshard"
            build_shard(path, games=2)
            reader = ShardReader(path, purpose="diagnostic")
            result = evaluate_reader(reader, tiny_model(), batch_size=4)
            self.assertGreater(result["supervised"], 0)
            self.assertEqual(result["phase_sum"], result["supervised"])
            self.assertEqual(result["action_family_sum"], result["supervised"])
            self.assertTrue(torch.isfinite(torch.tensor(result["nll"])))
            self.assertIn("NOT arena strength", result["warning"])
            self.assertEqual(result["source_id"], "synthetic_allowed")

    def test_cli_json_and_blocked_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.rfshard"
            build_shard(path, games=1)
            out = Path(temp) / "metrics.json"
            code = eval_main(["--shard", str(path), "--out", str(out),
                              "--format", "json", "--limit", "2", "--batch-size", "2"])
            self.assertEqual(code, 0)
            data = json.loads(out.read_text())
            self.assertIn("nll", data)
            self.assertIn("warning", data)

            blocked = Path(temp) / "audit.rfshard"
            build_shard(blocked, games=1, allowed=False)
            refused = eval_main(["--shard", str(blocked), "--format", "json"])
            self.assertEqual(refused, 2)
            allowed = eval_main(["--shard", str(blocked), "--allow-blocked-audit",
                                 "--format", "json", "--limit", "1"])
            self.assertEqual(allowed, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
