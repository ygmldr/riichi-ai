#!/usr/bin/env python3
"""PR-05 VNext runtime parity/phase tests (cacheless reference adapter)."""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.decision import build_decisions  # noqa: E402
from riichiformer.schemas.action_v3 import ActionPhase  # noqa: E402
from riichiformer.vnext.contracts import VNextConfig  # noqa: E402
from riichiformer.vnext.model import VNextPolicy  # noqa: E402
from riichiformer.vnext.runtime import (  # noqa: E402
    VNextPolicyAdapter, VNextRuntimeError,
)


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


def config() -> VNextConfig:
    return VNextConfig(d_model=64, n_layers=1, n_heads=4, ffn_hidden=128,
                       cross_attn_heads=4, dropout=0.0, max_positions=64)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = build_decisions(raw_long())
        cls.model = VNextPolicy(config()).eval()

    def test_act_is_legal_and_prefix_shared(self):
        adapter = VNextPolicyAdapter(self.model, match_id="m", pov=0)
        action = adapter.act(self.samples[0].input, decision_id="a")
        self.assertTrue(self.samples[0].input.legal_mask[action])
        self.assertEqual(adapter.stream.n_tokens, 3)
        action2 = adapter.act(self.samples[1].input, decision_id="b")
        self.assertTrue(self.samples[1].input.legal_mask[action2])
        self.assertEqual([index.prefix_token_offset
                          for index, _state in adapter.stream.decisions], [2, 4])

    def test_reset_kyoku_drops_history(self):
        adapter = VNextPolicyAdapter(self.model, match_id="m", pov=0)
        adapter.act(self.samples[0].input)
        old = adapter.stream
        adapter.reset_kyoku()
        self.assertIsNone(adapter.stream)
        adapter.act(self.samples[0].input)
        self.assertIsNot(adapter.stream, old)
        self.assertEqual(adapter.stream.n_tokens, 3)

    def test_kan_selection_requires_kan_phase(self):
        sample = self.samples[0]
        mask = tuple(i == 4 for i in range(46))
        decision = dataclasses.replace(sample.input, phase=ActionPhase.KAN_SELECTION,
                                       legal_mask=mask)
        adapter = VNextPolicyAdapter(self.model, match_id="m", pov=0)
        with self.assertRaises(VNextRuntimeError):
            adapter.select_kan(sample.input, kan_mask=mask)
        action = adapter.select_kan(decision, kan_mask=mask)
        self.assertEqual(action, 4)

    def test_act_batch_shape(self):
        adapter = VNextPolicyAdapter(self.model, match_id="m", pov=0)
        adapter.observe(self.samples[0].input)
        batch = adapter.stream.to_batch()
        actions = adapter.act_batch(batch)
        self.assertEqual(tuple(actions.shape), (1,))
        self.assertTrue(0 <= int(actions[0]) < 46)


if __name__ == "__main__":
    unittest.main()
