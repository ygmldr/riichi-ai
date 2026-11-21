#!/usr/bin/env python3
"""PR-08 VNext data adapter + BC trainer smoke tests (CPU only)."""

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
from riichiformer.vnext.contracts import VNextConfig  # noqa: E402
from riichiformer.vnext.data import VNextBatchAdapter, targets_from_dicts  # noqa: E402
from riichiformer.vnext.model import VNextPolicy  # noqa: E402
from riichiformer.vnext.train import (  # noqa: E402
    VNextTrainConfig, make_optimizer, masked_bc_loss, save_vnext_checkpoint,
    train_step,
)
from riichiformer.vnext.runtime import load_vnext_checkpoint  # noqa: E402
from riichiformer.vnext.contracts import contract_hash  # noqa: E402


def raw_long() -> list[dict]:
    hand0 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
             "1p", "2p", "3p", "4p"]
    hand1 = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
             "E", "S", "W", "N"]
    hand2 = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
             "7m", "8m", "9m", "1p"]
    hand3 = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
             "1s", "2s", "3s", "4s", "5s"]
    return [
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1p",
         "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
         "scores": [25000] * 4, "tehais": [hand0, hand1, hand2, hand3]},
        {"type": "tsumo", "actor": 0, "pai": "5mr"},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        {"type": "end_kyoku"}, {"type": "end_game"},
    ]


class FakeReader:
    def __init__(self):
        self.samples = build_decisions(raw_long())

    def iter_decisions(self):
        for index, sample in enumerate(self.samples):
            legal = [i for i, value in enumerate(sample.input.legal_mask) if value]
            yield (sample.input,
                   {"expert_action": legal[0], "supervision_valid": True,
                    "censor_reason": None, "rank_final": index % 4,
                    "loss_weight": 1.0},
                   None, "game", 0, index)


def config() -> VNextConfig:
    return VNextConfig(d_model=64, n_layers=1, n_heads=4, ffn_hidden=128,
                       cross_attn_heads=4, dropout=0.0, max_positions=64)


class AdapterTests(unittest.TestCase):
    def test_batches_keep_targets_separate(self):
        adapter = VNextBatchAdapter(FakeReader(), batch_size=2)
        batches = list(adapter.iter_batches())
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(batch.batch.n_decisions, 2)
        self.assertEqual(len(batch.targets), 2)
        self.assertNotIn("target", batch.batch.__dataclass_fields__)

    def test_target_tensor_contract(self):
        targets = targets_from_dicts(
            ({"expert_action": 1, "rank_final": 2, "supervision_valid": True,
              "loss_weight": 1.0},))
        self.assertEqual(int(targets.action[0]), 1)
        self.assertEqual(int(targets.final_rank[0]), 2)
        self.assertTrue(bool(targets.supervise_mask[0]))


class TrainTests(unittest.TestCase):
    def test_two_steps_and_loss(self):
        adapter = VNextBatchAdapter(FakeReader(), batch_size=2)
        batch = list(adapter.iter_batches())[0]
        model = VNextPolicy(config())
        trainer = VNextTrainConfig(learning_rate=1e-3, accumulation_count=1)
        optimizer = make_optimizer(model, trainer)
        result = train_step(model, batch.batch, batch.targets,
                            optimizer=optimizer, config=trainer)
        self.assertGreater(result["valid"], 0)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
        self.assertTrue(result["optimizer_stepped"])
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        result2 = train_step(model, batch.batch, batch.targets,
                             optimizer=optimizer, config=trainer,
                             accumulation_step=1)
        self.assertTrue(torch.isfinite(torch.tensor(result2["loss"])))

    def test_checkpoint_roundtrip_checks_contract(self):
        adapter = VNextBatchAdapter(FakeReader(), batch_size=2)
        batch = list(adapter.iter_batches())[0]
        model = VNextPolicy(config())
        trainer = VNextTrainConfig(learning_rate=1e-3)
        optimizer = make_optimizer(model, trainer)
        root = Path(tempfile.mkdtemp(prefix="rf_vnext_ckpt_"))
        path = root / "vnext.pt"
        provenance = {
            "source_id": "synthetic",
            "split_manifest_sha256": "1" * 64,
            "canonical_manifest_sha256": "2" * 64,
            "canonical_index_sha256": "3" * 64,
            "research_policy": "licensed_only",
            "research_policy_hash": "",
        }
        save_vnext_checkpoint(
            path, model=model, optimizer=optimizer, config=trainer,
            counters={"step": 0, "tokens": 0, "decisions": 0},
            provenance=provenance)
        loaded = load_vnext_checkpoint(
            path, expected_contract_hash=contract_hash(model.config))
        self.assertTrue(all(torch.equal(value, loaded.state_dict()[name])
                            for name, value in model.state_dict().items()))
        with self.assertRaises(Exception):
            load_vnext_checkpoint(path, expected_contract_hash="0" * 64)

    def test_all_censored_is_zero_without_update(self):
        adapter = VNextBatchAdapter(FakeReader(), batch_size=2)
        item = list(adapter.iter_batches())[0]
        targets = tuple({**target, "supervision_valid": False,
                         "censor_reason": "test"} for target in item.targets)
        model = VNextPolicy(config())
        optimizer = make_optimizer(model, VNextTrainConfig())
        result = train_step(model, item.batch, targets, optimizer=optimizer,
                            config=VNextTrainConfig())
        self.assertEqual(result["valid"], 0)
        self.assertFalse(result["optimizer_stepped"])
        self.assertFalse(any(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
