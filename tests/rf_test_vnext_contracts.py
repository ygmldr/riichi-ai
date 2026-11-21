#!/usr/bin/env python3
"""PR-02 VNext contract tests: hashes, visibility, PT/GRP/rollout invariants."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.vnext.contracts import (  # noqa: E402
    ACTION_SPACE, GRPBoundary, GuardSpec, PT_TABLE_5D, PT_TABLE_7D,
    RLDecisionRow, VNextConfig, VNextDecisionIndex, VNextTargets,
    VNextVisibleState, assert_forward_safe, contract_hash, contract_manifest,
    expected_pt_from_rank_probs, permutation_index, phi_from_marginals,
    rank_marginals, relative_pt_from_rf_pt,
)


class ContractHashTests(unittest.TestCase):
    def test_action_space_and_manifest(self):
        self.assertEqual(ACTION_SPACE, 46)
        manifest = contract_manifest()
        self.assertEqual(manifest["action_space"], 46)
        self.assertEqual(manifest["model_arch_id"], "rf-vnext-crossattn/1")
        self.assertEqual(manifest["encoder_contract_id"], "current_kyoku_bos/1")
        self.assertEqual(len(manifest["grp_permutations"]), 24)
        self.assertEqual(contract_hash(), contract_hash())
        changed = VNextConfig(cross_attn_layers=0)
        self.assertNotEqual(contract_hash(), contract_hash(changed))
        self.assertEqual(changed.architecture_hash, changed.architecture_hash)

    def test_config_rejects_unknown_or_invalid(self):
        with self.assertRaises(ValueError):
            VNextConfig(d_model=255)
        with self.assertRaises(ValueError):
            VNextConfig(cross_attn_layers=3)
        with self.assertRaises(ValueError):
            VNextConfig.from_mapping({"unknown": True})

    def test_forward_safe_guard(self):
        assert_forward_safe(("x_tile", "legal_mask"), owner="ok")
        with self.assertRaises(ValueError):
            assert_forward_safe(("expert_action",), owner="bad")


class VisibleStateTests(unittest.TestCase):
    def test_empty_state_is_valid_and_masked(self):
        state = VNextVisibleState.empty()
        self.assertEqual(tuple(state.tile_features.shape), (34, 26))
        self.assertEqual(tuple(state.candidate_features.shape), (37, 4))
        self.assertTrue(bool(state.legal_mask[37]))
        self.assertFalse(bool(state.legal_mask[0]))

    def test_aka_candidate_is_distinct(self):
        state = VNextVisibleState.empty()
        cf = state.candidate_features
        self.assertEqual(float(cf[4, 0]), 0.0)  # normal 5m
        self.assertEqual(float(cf[34, 0]), 1.0)  # aka 5m
        self.assertNotEqual(float(cf[4, 1]), float(cf[34, 1]))

    def test_empty_mask_rejected(self):
        state = VNextVisibleState.empty()
        with self.assertRaises(ValueError):
            VNextVisibleState(
                tile_features=state.tile_features,
                global_features=state.global_features,
                candidate_features=state.candidate_features,
                legal_mask=torch.zeros(46, dtype=torch.bool), phase=0)


class TargetIsolationTests(unittest.TestCase):
    def test_targets_are_a_separate_object(self):
        targets = VNextTargets(
            action=torch.tensor([0]), final_rank=torch.tensor([1]),
            supervise_mask=torch.tensor([True]), weight=torch.tensor([1.0]))
        self.assertEqual(int(targets.final_rank[0]), 1)
        self.assertNotIn("target", VNextVisibleState.__dataclass_fields__)
        self.assertNotIn("target", VNextConfig.__dataclass_fields__)


class PTTests(unittest.TestCase):
    def test_pt_tables_are_zero_sum(self):
        self.assertEqual(sum(PT_TABLE_5D), 0)
        self.assertEqual(sum(PT_TABLE_7D), 0)
        self.assertEqual(relative_pt_from_rf_pt(30.0), 40.0)

    def test_rank_marginals_and_phi(self):
        probs = torch.zeros((1, 24))
        probs[0, permutation_index((0, 1, 2, 3))] = 1.0
        marginal = rank_marginals(probs)
        self.assertTrue(torch.allclose(marginal.sum(-1), torch.ones(1, 4)))
        self.assertTrue(torch.allclose(marginal.sum(1), torch.ones(1, 4)))
        phi = phi_from_marginals(marginal)
        self.assertAlmostEqual(float(phi.sum()), 0.0, places=6)
        self.assertAlmostEqual(float(phi[0, 0]), 75.0)
        self.assertAlmostEqual(float(phi[0, 3]), -105.0)
        self.assertTrue(torch.allclose(
            expected_pt_from_rank_probs(torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
            torch.tensor([-105.0])))


class GRPBoundaryTests(unittest.TestCase):
    def test_boundary_validates_total(self):
        b = GRPBoundary("m", 0, 0, 0, 0, (25000, 25000, 25000, 25000), "rules")
        self.assertEqual(b.input_vector[3], 250.0)
        with self.assertRaises(ValueError):
            GRPBoundary("m", 0, 0, 0, 0, (25000, 25000, 25000, 0), "rules")


class GuardAndRolloutTests(unittest.TestCase):
    def test_guard_spec(self):
        g = GuardSpec(temperature=0.5, top_p=0.9, epsilon_mix=0.01)
        self.assertEqual(g.version, "guard/1")
        with self.assertRaises(ValueError):
            GuardSpec(temperature=0.0)

    def test_rollout_row_recomputes_logprob(self):
        probs = [0.0] * 46
        probs[37] = 0.5
        probs[45] = 0.5
        legal = tuple(p > 0 for p in probs)
        row = RLDecisionRow.from_probabilities(
            match_id="m", kyoku_id=0, seat=0, decision_id="d", decision_order=0,
            phase=0, observation_ref="obs", prefix_token_offset=0,
            legal_mask_46=legal, effective_mask_46=legal,
            raw_logits_46=tuple(0.0 for _ in range(46)),
            behavior_probs_46=tuple(probs), executed_action=37,
            behavior_logprob=0.0, guard_reason=None, forced=False,
            policy_sha="p", opponent_shas_by_seat=("a", "b", "c", "d"),
            rules_hash="r", grp_sha="g", reward_spec_id="r0",
            pt_table_id="pt5d/75,30,0,-105", kyoku_done=False, match_done=False)
        self.assertAlmostEqual(row.behavior_logprob, math.log(0.5))
        with self.assertRaises(ValueError):
            RLDecisionRow.from_probabilities(
                **{**row.__dict__, "behavior_probs_46": tuple([1.0] + [0.0] * 45),
                   "executed_action": 0})


if __name__ == "__main__":
    unittest.main()
