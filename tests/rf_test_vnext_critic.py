#!/usr/bin/env python3
"""PR-07 Decision Critic and gradient-isolation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.vnext.critic import (  # noqa: E402
    DecisionRankCritic, actor_grad_norm, critic_ce, critic_features, expected_pt,
)


class CriticGradientTests(unittest.TestCase):
    def test_detach_actor_blocks_gradient(self):
        trunk = nn.Linear(8, 8)
        critic = DecisionRankCritic(8)
        x = torch.randn(4, 8)
        z = trunk(x)
        logits = critic(critic_features(z, "detach_actor"))
        loss = critic_ce(logits, torch.tensor([0, 1, 2, 3]))
        loss.backward()
        self.assertEqual(actor_grad_norm(trunk), 0.0)
        self.assertGreater(actor_grad_norm(critic), 0.0)

    def test_joint_mode_reaches_actor(self):
        trunk = nn.Linear(8, 8)
        critic = DecisionRankCritic(8)
        z = trunk(torch.randn(4, 8))
        loss = critic_ce(critic(critic_features(z, "joint")),
                         torch.tensor([0, 1, 2, 3]))
        loss.backward()
        self.assertGreater(actor_grad_norm(trunk), 0.0)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            critic_features(torch.zeros(1, 4), "future")

    def test_weighted_ce_and_expected_pt(self):
        logits = torch.tensor([[100.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 100.0]])
        labels = torch.tensor([0, 3])
        weights = torch.tensor([1.0, 3.0])
        loss = critic_ce(logits, labels, weights)
        self.assertLess(float(loss), 1e-3)
        self.assertAlmostEqual(float(expected_pt(logits)[0]), 75.0, places=3)
        self.assertAlmostEqual(float(expected_pt(logits)[1]), -105.0, places=3)


if __name__ == "__main__":
    unittest.main()
