#!/usr/bin/env python3
"""PR-06 independent 24-way GRP tests."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.vnext.contracts import (  # noqa: E402
    GRP_PERMUTATIONS, GRPBoundary, permutation_index,
)
from riichiformer.vnext.grp import (  # noqa: E402
    GRP24, GRPConfig, boundaries_tensor, calibration_report, grp_loss,
    grp_state_sha, load_grp_checkpoint, permute_input_scores, permute_prob_perm,
    save_grp_checkpoint,
)


def boundaries(n: int = 3) -> torch.Tensor:
    values = []
    for index in range(n):
        values.append([float(index % 4), float(index % 3), 0.0,
                       250.0 + index, 250.0 - index, 250.0, 250.0])
    return torch.tensor(values, dtype=torch.float32)


class PermutationTests(unittest.TestCase):
    def test_unique_table_and_inverse(self):
        self.assertEqual(len(GRP_PERMUTATIONS), 24)
        for index, perm in enumerate(GRP_PERMUTATIONS):
            self.assertEqual(permutation_index(perm), index)
        with self.assertRaises(ValueError):
            permutation_index((0, 0, 1, 2))

    def test_marginals_sum_to_one(self):
        model = GRP24(GRPConfig(hidden=16, layers=1))
        with torch.no_grad():
            probs = model.probability(boundaries(4))
            marginals = model.marginals(boundaries(4))
            phi = model.phi(boundaries(4))
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(4)))
        self.assertTrue(torch.allclose(marginals.sum(-1), torch.ones(4, 4)))
        self.assertTrue(torch.allclose(marginals.sum(1), torch.ones(4, 4)))
        self.assertTrue(torch.allclose(phi.sum(-1), torch.zeros(4), atol=1e-4))

    def test_seat_probability_transform(self):
        probs = torch.zeros((1, 24))
        probs[0, permutation_index((0, 1, 2, 3))] = 1.0
        order = (1, 0, 3, 2)
        transformed = permute_prob_perm(probs, order)
        self.assertEqual(int(transformed.argmax()), permutation_index((1, 0, 3, 2)))
        self.assertEqual(permute_input_scores([10, 20, 30, 40], order),
                         (20, 10, 40, 30))

    def test_permutation_transform_matches_joint_distribution(self):
        torch.manual_seed(2)
        probs = torch.softmax(torch.randn(3, 24), dim=-1)
        order = (2, 0, 3, 1)
        transformed = permute_prob_perm(probs, order)
        self.assertTrue(torch.allclose(transformed.sum(-1), torch.ones(3)))
        for old_index, old_perm in enumerate(GRP_PERMUTATIONS):
            new_perm = tuple(old_perm[order[i]] for i in range(4))
            new_index = permutation_index(new_perm)
            self.assertAlmostEqual(float(transformed[0, new_index]),
                                   float(probs[0, old_index]), places=6)

    def test_boundary_input(self):
        b = GRPBoundary("m", 2, 1, 1, 0, (26000, 25000, 24000, 25000), "r")
        tensor = boundaries_tensor([b])
        self.assertEqual(tuple(tensor.shape), (1, 7))
        self.assertEqual(float(tensor[0, 0]), 1.0)


class GRPLossTests(unittest.TestCase):
    def test_loss_prefers_true_permutation(self):
        logits = torch.zeros((1, 24))
        target = torch.tensor([(0, 1, 2, 3)])
        idx = permutation_index(target[0].tolist())
        logits[0, idx] = 20.0
        loss = grp_loss(logits, target)
        self.assertLess(float(loss), 1e-4)
        logits[0, idx] = -20.0
        self.assertGreater(float(grp_loss(logits, target)), 10.0)

    def test_calibration_perfect(self):
        probs = torch.zeros((2, 24))
        labels = torch.tensor([(0, 1, 2, 3), (3, 2, 1, 0)])
        for row, label in enumerate(labels.tolist()):
            probs[row, permutation_index(label)] = 1.0
        report = calibration_report(probs, labels)
        self.assertLess(report.nll, 1e-6)
        self.assertLess(report.brier, 1e-6)
        self.assertLess(report.ece, 1e-6)
        self.assertEqual(report.count, 2)

    def test_independent_checkpoint_roundtrip(self):
        model = GRP24(GRPConfig(hidden=8, layers=1))
        root = Path(tempfile.mkdtemp(prefix="rf_grp_ckpt_"))
        path = root / "grp.pt"
        manifest = save_grp_checkpoint(
            path, model, normalizer_hash="1" * 64, rules_hash="2" * 64,
            permutation_map_hash="3" * 64)
        self.assertEqual(manifest["format"], "rf-grp-checkpoint/1")
        loaded, sidecar = load_grp_checkpoint(path)
        self.assertEqual(grp_state_sha(loaded), grp_state_sha(model))
        self.assertEqual(sidecar["rules_hash"], "2" * 64)

    def test_state_hash_stable(self):
        model = GRP24(GRPConfig(hidden=8, layers=1))
        before = grp_state_sha(model)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        clone = GRP24(GRPConfig(hidden=8, layers=1))
        clone.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertEqual(before, grp_state_sha(clone))


if __name__ == "__main__":
    unittest.main()
