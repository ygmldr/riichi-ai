#!/usr/bin/env python3
"""PR-46A frozen M5D-R0 anchor tests: identity, config, seats, no hidden inputs.

The identity tests load the real user-confirmed checkpoint on CPU (the environment
for this lane).  They never modify it, never write under ``checkpoints/`` and never
touch ``runs/``.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from libriichi.consts import obs_shape  # noqa: E402
from riichiformer.eval import mortal_anchor as M  # noqa: E402


class CheckpointIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity = M.checkpoint_identity(M.DEFAULT_CHECKPOINT_PATH)

    def test_5d_checkpoint_identity(self):
        identity = self.identity
        self.assertEqual(identity.sha256, M.DEFAULT_CHECKPOINT_SHA256)
        self.assertEqual(identity.steps, 396_000)
        self.assertEqual(identity.version, 4)
        self.assertEqual(identity.conv_channels, 192)
        self.assertEqual(identity.num_blocks, 40)
        self.assertEqual(identity.model_arch,
                         "resnet{conv_channels=192,num_blocks=40}")
        self.assertIs(identity.use_event_branch, False)
        self.assertEqual(identity.model_config_hash,
                         M.model_config_hash(M.EXPECTED_MODEL_CONFIG))
        M.validate_checkpoint_identity(identity)

    def test_identity_rejects_drifted_values(self):
        with self.assertRaises(M.AnchorMismatch):
            M.validate_checkpoint_identity(self.identity, expected_steps=395_999)
        with self.assertRaises(M.AnchorMismatch):
            M.validate_checkpoint_identity(
                self.identity, expected_sha256="00" * 32)
        with self.assertRaises(M.AnchorMismatch):
            M.validate_checkpoint_identity(
                self.identity, expected_conv_channels=128)

    def test_model_config_hash_ignores_unrelated_config(self):
        # The arch hash must cover exactly the architecture subset, not data paths,
        # tensorboard dirs or 1v3 settings that differ per checkout.
        config = copy.deepcopy(M.EXPECTED_MODEL_CONFIG)
        base = M.model_config_hash(config)
        config["model"]["event_layers"] = 4
        self.assertEqual(base, M.model_config_hash(config))
        config["model"]["use_event_branch"] = True
        self.assertNotEqual(base, M.model_config_hash(config))


class LoaderAndSeatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = M.load_mortal_checkpoint(M.DEFAULT_CHECKPOINT_PATH)

    def test_loader_is_cpu_only_and_weights_are_finite(self):
        self.loaded.assert_cpu()
        import torch
        for module in (self.loaded.brain, self.loaded.dqn):
            for parameter in module.parameters():
                self.assertEqual(parameter.device.type, "cpu")
                self.assertTrue(torch.isfinite(parameter).all())

    def test_three_anchor_seats_are_identical(self):
        seats = M.build_champion_seats(self.loaded, count=3)
        self.assertEqual(len(seats), 3)
        reference = list(seats[0].engine.brain.parameters())
        for seat in seats[1:]:
            for left, right in zip(reference, seat.engine.brain.parameters()):
                self.assertIs(left, right)
        for seat in seats:
            self.assertFalse(seat.is_oracle)
            self.assertEqual(seat.version, 4)
            self.assertEqual(seat.inference_mode, "deterministic_argmax")
            self.assertEqual(seat.identity.sha256, M.DEFAULT_CHECKPOINT_SHA256)

    def test_react_batch_is_deterministic_argmax_and_rejects_hidden_obs(self):
        adapter = M.MortalCheckpointAdapter(self.loaded, seat_index=0)
        channels = obs_shape(4)[0]
        obs = np.zeros((channels, 34), dtype=np.float32)
        mask = np.ones(46, dtype=bool)
        first = adapter.react_batch([obs], [mask], None)
        second = adapter.react_batch([obs], [mask], None)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(len(first[0]), 1)
        self.assertTrue(0 <= first[0][0] < 46)
        self.assertTrue(bool(first[2][0][first[0][0]]))
        with self.assertRaises(M.AnchorError):
            adapter.react_batch([obs], [mask], [obs])


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = M.build_baseline_registry(
            python_driver_sha="e76022600018e9adc6536ccbb904db8c9dc498b3")

    def test_registry_fields_are_complete_and_frozen(self):
        registry = self.registry
        for key in (
            "anchor_id", "role", "name", "checkpoint_path", "checkpoint_sha256",
            "steps", "model_arch", "model_config_hash", "python_driver_git_sha",
            "libriichi_git_sha", "libriichi_build_sha256", "patch_manifest_hash",
            "ruleset_id", "tenhou_approx_rules_hash", "inference_mode",
            "action_guard", "seed_policy", "permission", "registry_hash",
        ):
            self.assertIn(key, registry)
        self.assertEqual(registry["anchor_id"], "M5D-R0")
        self.assertEqual(registry["role"], "frozen_opponent")
        self.assertEqual(registry["permission"], "local_research_only")
        self.assertEqual(registry["inference_mode"], "deterministic_argmax")
        self.assertEqual(registry["seed_policy"], "per_game_and_seat")
        self.assertEqual(registry["checkpoint_sha256"], M.DEFAULT_CHECKPOINT_SHA256)
        self.assertEqual(registry["steps"], 396_000)
        self.assertEqual(registry["model_arch"],
                         "resnet{conv_channels=192,num_blocks=40}")
        self.assertIs(registry["model_config"]["model"]["use_event_branch"], False)
        self.assertEqual(registry["python_driver_git_sha"],
                         "e76022600018e9adc6536ccbb904db8c9dc498b3")
        self.assertEqual(len(registry["libriichi_build_sha256"]), 64)
        self.assertEqual(len(registry["patch_manifest_hash"]), 64)
        self.assertEqual(len(registry["tenhou_approx_rules_hash"]), 64)

    def test_registry_hash_is_canonical_and_stable(self):
        again = M.build_baseline_registry(
            python_driver_sha="e76022600018e9adc6536ccbb904db8c9dc498b3")
        self.assertEqual(self.registry["registry_hash"], again["registry_hash"])
        self.assertEqual(self.registry["registry_hash"],
                         M.registry_hash(self.registry))
        self.assertNotIn("registry_hash", M.canonical_registry_payload(self.registry))

        tampered = copy.deepcopy(self.registry)
        tampered["steps"] = 395_999
        self.assertNotEqual(tampered["registry_hash"], M.registry_hash(tampered))
        self.assertNotEqual(self.registry["registry_hash"], M.registry_hash(tampered))

    def test_registry_validation_rejects_tampering(self):
        validated = M.validate_registry(self.registry)
        self.assertEqual(validated["anchor"]["anchor_id"], "M5D-R0")
        for field, value in (
            ("anchor_id", "M5D-R1"),
            ("role", "challenger"),
            ("permission", "public"),
            ("inference_mode", "sample"),
            ("seed_policy", "single"),
            ("steps", 1),
            ("checkpoint_sha256", "00" * 32),
        ):
            broken = copy.deepcopy(self.registry)
            broken[field] = value
            broken["registry_hash"] = M.registry_hash(broken)
            with self.subTest(field=field), self.assertRaises(M.AnchorMismatch):
                M.validate_registry(broken, check_files=False)
        broken = copy.deepcopy(self.registry)
        broken["registry_hash"] = "00" * 32
        with self.assertRaises(M.AnchorMismatch):
            M.validate_registry(broken, check_files=False)

    def test_committed_yaml_matches_recomputation(self):
        path = REPO_ROOT / "configs" / "riichiformer" / "m5d_baseline.yaml"
        if not path.is_file():
            self.skipTest("registry YAML not generated yet")
        loaded = M.load_baseline_registry(path)
        self.assertEqual(loaded["registry_hash"], self.registry["registry_hash"])
        self.assertEqual(loaded["checkpoint_sha256"], M.DEFAULT_CHECKPOINT_SHA256)


if __name__ == "__main__":
    unittest.main()
