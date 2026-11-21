#!/usr/bin/env python3
"""PR-09/10 RL behavior, reward and GAE contract tests (CPU)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.vnext.contracts import PT_TABLE_5D, GuardSpec  # noqa: E402
from riichiformer.vnext.rl import (  # noqa: E402
    LeagueManifest, RLError, awr_loss, behavior_probs, check_no_double_terminal,
    mortal_delta_rewards, pbrs_rewards, ppo_loss, terminal_pt,
    two_timescale_gae,
)


class BehaviorTests(unittest.TestCase):
    def test_illegal_mass_is_zero_and_sum_one(self):
        torch.manual_seed(0)
        logits = torch.zeros(46)
        legal = torch.zeros(46, dtype=torch.bool)
        legal[[37, 45]] = True
        probs, action, forced = behavior_probs(logits, legal)
        self.assertTrue(bool(legal[action]))
        self.assertFalse(forced)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)
        self.assertEqual(float(probs[~legal].abs().sum()), 0.0)
        self.assertFalse(bool((probs < 0).any()))

    def test_forced_action_is_one(self):
        logits = torch.randn(46)
        legal = torch.ones(46, dtype=torch.bool)
        probs, action, forced = behavior_probs(
            logits, legal, guard=GuardSpec(forced_action=37))
        self.assertTrue(forced)
        self.assertEqual(action, 37)
        self.assertEqual(float(probs[37]), 1.0)
        self.assertEqual(float(probs.sum()), 1.0)

    def test_temperature_top_p_and_epsilon_are_real_distribution(self):
        logits = torch.randn(46)
        legal = torch.ones(46, dtype=torch.bool)
        probs, _action, _forced = behavior_probs(
            logits, legal,
            guard=GuardSpec(temperature=0.7, top_p=0.9, epsilon_mix=0.05))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertTrue(bool((probs >= 0).all()))


class RewardTests(unittest.TestCase):
    def test_pt_table(self):
        self.assertEqual(terminal_pt(0), 75.0)
        self.assertEqual(terminal_pt(3), -105.0)
        self.assertEqual(sum(PT_TABLE_5D), 0)

    def test_delta_telescopes_to_terminal_minus_phi0(self):
        phis = [[10.0, -5.0, 0.0, -5.0], [20.0, -10.0, 0.0, -10.0]]
        rewards = mortal_delta_rewards(phis, 75.0)
        for seat in range(4):
            self.assertAlmostEqual(sum(row[seat] for row in rewards),
                                   75.0 - phis[0][seat], places=6)

    def test_pbrs_net_matches_delta(self):
        phis = [[10.0, -5.0, 0.0, -5.0], [20.0, -10.0, 0.0, -10.0]]
        left = mortal_delta_rewards(phis, 75.0)
        right = pbrs_rewards(phis, 75.0)
        for seat in range(4):
            self.assertAlmostEqual(sum(row[seat] for row in left),
                                   sum(row[seat] for row in right), places=6)

    def test_pbrs_rejects_non_zero_sum_potential(self):
        with self.assertRaises(RLError):
            pbrs_rewards([[1.0, 0.0, 0.0, 0.0]], 75.0)

    def test_double_terminal_is_rejected(self):
        with self.assertRaises(RLError):
            check_no_double_terminal(True, True)


class GAETests(unittest.TestCase):
    def test_hand_computed_terminal_case(self):
        rewards = [0.0, 0.0, 10.0]
        values = [1.0, 2.0, 3.0]
        next_values = [2.0, 3.0, 0.0]
        terminals = [False, False, True]
        gae = two_timescale_gae(rewards, values, next_values,
                                terminals=terminals, gamma=1.0,
                                lambda_intra=1.0, lambda_inter=1.0)
        # delta = [1, 1, 7]; A2=7, A1=1+7=8, A0=1+8=9
        self.assertEqual(gae, [9.0, 8.0, 7.0])

    def test_boundary_lambda_changes_only_cross_kyoku(self):
        rewards = [0.0, 1.0]
        values = [0.0, 0.0]
        next_values = [0.0, 0.0]
        terminals = [False, True]
        boundaries = [True, False]
        gae0 = two_timescale_gae(rewards, values, next_values,
                                 terminals=terminals, kyoku_boundaries=boundaries,
                                 lambda_intra=1.0, lambda_inter=0.0)
        gae1 = two_timescale_gae(rewards, values, next_values,
                                 terminals=terminals, kyoku_boundaries=boundaries,
                                 lambda_intra=1.0, lambda_inter=1.0)
        self.assertNotEqual(gae0[0], gae1[0])
        self.assertEqual(gae0[1], gae1[1])

    def test_terminal_ignores_bootstrap(self):
        gae = two_timescale_gae([0.0], [1.0], [99.0], terminals=[True])
        self.assertEqual(gae, [-1.0])


class ToyObjectiveTests(unittest.TestCase):
    def test_awr_prefers_positive_advantage(self):
        logits = torch.tensor([[math.log(0.5), math.log(0.5)]]).repeat(2, 1)
        actions = torch.tensor([0, 1])
        advantages = torch.tensor([1.0, -1.0])
        loss = awr_loss(logits, actions, advantages, beta=1.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_awr_clip_prevents_inf(self):
        logits = torch.zeros(1, 2)
        loss = awr_loss(logits, torch.tensor([0]), torch.tensor([1000.0]),
                        beta=1.0, weight_clip=10.0)
        self.assertTrue(torch.isfinite(loss))

    def test_ppo_rho_one_has_zero_advantage(self):
        logits = torch.log(torch.tensor([[0.4, 0.6]]))
        loss = ppo_loss(logits, logits, torch.tensor([1]), torch.tensor([1.0]))
        self.assertAlmostEqual(float(loss), -1.0, places=6)

    def test_league_manifest_shape(self):
        manifest = LeagueManifest(
            league_id="l", rules_hash="r", opponent_checkpoint_shas=("a" * 64,
                                                                    "b" * 64,
                                                                    "c" * 64),
            engine_guard_sha="g", sampling_seed=1)
        self.assertEqual(manifest.version, 1)
        with self.assertRaises(RLError):
            LeagueManifest("l", "r", ("a" * 64,), "g", 1)


if __name__ == "__main__":
    unittest.main()
