#!/usr/bin/env python3
"""PR-48 B0 wrapper tests: bounded runs, strict resume, guards and selection.

CPU only, tiny synthetic shards, one-layer 64-wide model.  No full epoch, no
CUDA, no real B0 corpus, no arena.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rf_test_b0_protocol import (  # noqa: E402
    build_shard, settings as train_settings, tiny_config,
)
from rf_train_b0 import main as b0_main  # noqa: E402
from riichiformer.train.b0 import (  # noqa: E402
    B0_TRAINING_PATH, B0Protocol, B0SelectionRule, B0StopPolicy,
    build_protocol_from_files, checkpoint_provenance,
)
from riichiformer.data.loader import ShardReader  # noqa: E402
from riichiformer.data.shard import (  # noqa: E402
    ACTION_HASH, NORMALIZER_HASH, SCHEMA_HASH,
)
from riichiformer.models.policy import Policy  # noqa: E402
from riichiformer.train.checkpoint import save_checkpoint  # noqa: E402
from riichiformer.train.optimizer import (  # noqa: E402
    make_optimizer, set_seed,
)


def make_protocol(root: Path, *, name: str = "EXP-B0-CLI", max_steps: int = 10,
                  validation_steps=(10,), candidate_steps=(10,),
                  seeds=(11, 22), ema_decay: float | None = 0.9,
                  lr: float = 1e-3) -> B0Protocol:
    train = root / f"{name}-train.rfshard"
    val = root / f"{name}-val.rfshard"
    build_shard(train, 24, f"g{name[-3:]}")
    build_shard(val, 8, f"v{name[-3:]}", validation=True)
    protocol = build_protocol_from_files(
        experiment_id=name, train_shard=str(train), validation_shard=str(val),
        model_config=tiny_config(),
        train=train_settings(max_steps=max_steps, ema_decay=ema_decay, lr=lr),
        seeds=seeds, validation_steps=validation_steps,
        candidate_steps=candidate_steps,
        selection=B0SelectionRule(weights="raw"),
        stop_policy=B0StopPolicy(max_wall_seconds=120.0),
        created_utc="2026-11-21T00:00:00Z")
    protocol.write_json(root / f"{name}-prereg.json")
    return protocol


def invoke(*argv) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = b0_main([str(item) for item in argv])
    return code, stdout.getvalue(), stderr.getvalue()


def load_payload(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def assert_state_equal(test: unittest.TestCase, left, right, path: str = "") -> None:
    if isinstance(left, dict):
        test.assertEqual(set(left), set(right), f"{path} keys")
        for key in left:
            assert_state_equal(test, left[key], right[key], f"{path}/{key}")
    elif isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right), f"{path} tensor differs")
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right), f"{path} length")
        for index, (one, other) in enumerate(zip(left, right)):
            assert_state_equal(test, one, other, f"{path}[{index}]")
    else:
        test.assertEqual(left, right, f"{path} scalar")


class B0WrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.protocol = make_protocol(cls.root)
        cls.prereg = cls.root / "EXP-B0-CLI-prereg.json"
        cls.candidate_rel = Path("EXP-B0-CLI") / "run_seed11" / "candidates" / \
            "step_000010.pt"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _train(self, out: Path, *extra) -> tuple[int, str, str]:
        return invoke("--prereg", self.prereg, "--train",
                      "--max-steps", "10", "--out-dir", out, *extra)

    # ------------------------------------------------------------ code guard
    def test_no_implicit_epoch_or_training(self):
        out = self.root / "guard"
        code, _stdout, stderr = invoke("--prereg", self.prereg,
                                       "--max-steps", "4", "--out-dir", out)
        self.assertEqual(code, 2)
        self.assertIn("requires --train", stderr)
        code, _stdout, stderr = invoke("--prereg", self.prereg, "--train",
                                       "--out-dir", out)
        self.assertEqual(code, 2)
        self.assertIn("bounded --max-steps", stderr)
        code, _stdout, stderr = invoke("--prereg", self.prereg, "--train",
                                       "--max-steps", "10", "--out-dir", out)
        self.assertEqual(code, 2)
        self.assertIn("allow-full-budget", stderr)
        self.assertFalse((out / "EXP-B0-CLI" / "run_seed11" / "candidates"
                          / "step_000010.pt").exists())
        self.assertFalse((out / "EXP-B0-CLI" / "manifest.json").exists())

    def test_prereg_is_mandatory(self):
        code, _stdout, stderr = invoke("--train", "--max-steps", "1",
                                       "--out-dir", self.root / "noprereg")
        self.assertEqual(code, 2)
        self.assertIn("--prereg is required", stderr)

    def test_dry_run_writes_nothing(self):
        out = self.root / "dry"
        code, stdout, _stderr = invoke("--prereg", self.prereg, "--dry-run",
                                       "--out-dir", out)
        self.assertEqual(code, 0)
        self.assertIn("B0_DRY_RUN_OK", stdout)
        self.assertIn("training_path=multi_decisions", stdout)
        self.assertFalse(out.exists())

    def test_unknown_seed_and_device_refused(self):
        out = self.root / "badseed"
        code, _stdout, stderr = invoke(
            "--prereg", self.prereg, "--train", "--max-steps", "4",
            "--out-dir", out, "--seed", "99")
        self.assertEqual(code, 2)
        self.assertIn("not one of the frozen prereg seeds", stderr)
        code, _stdout, stderr = invoke(
            "--prereg", self.prereg, "--train", "--max-steps", "4",
            "--out-dir", out, "--device", "cuda")
        self.assertEqual(code, 2)
        self.assertIn("prereg-frozen device", stderr)
        code, _stdout, stderr = invoke(
            "--prereg", self.prereg, "--train", "--max-steps", "4",
            "--out-dir", out, "--seed", "11", "--seed", "22",
            "--resume", self.root / "missing.pt")
        self.assertEqual(code, 2)
        self.assertIn("exactly one --seed", stderr)

    # ------------------------------------------------------------- same seed
    def test_same_seed_10_steps_is_bitwise_identical(self):
        out_a, out_b = self.root / "seed_a", self.root / "seed_b"
        code, _stdout, stderr = self._train(out_a, "--allow-full-budget",
                                            "--seed", "11")
        self.assertEqual(code, 0, stderr)
        code, _stdout, stderr = self._train(out_b, "--allow-full-budget",
                                            "--seed", "11")
        self.assertEqual(code, 0, stderr)
        left = load_payload(out_a / self.candidate_rel)
        right = load_payload(out_b / self.candidate_rel)
        for name, tensor in left["model"].items():
            self.assertTrue(torch.equal(tensor, right["model"][name]), name)
        assert_state_equal(self, left["optimizer"], right["optimizer"])
        assert_state_equal(self, left["ema"], right["ema"])
        print("PR48_SAME_SEED model/optimizer/EMA identical over 10 CPU steps")

    def test_different_seed_weights_differ(self):
        out_a, out_b = self.root / "diff_a", self.root / "diff_b"
        self.assertEqual(self._train(out_a, "--allow-full-budget",
                                     "--seed", "11")[0], 0)
        self.assertEqual(self._train(out_b, "--allow-full-budget",
                                     "--seed", "22")[0], 0)
        left = load_payload(out_a / "EXP-B0-CLI" / "run_seed11" /
                            "candidates" / "step_000010.pt")["model"]
        right = load_payload(out_b / "EXP-B0-CLI" / "run_seed22" /
                             "candidates" / "step_000010.pt")["model"]
        self.assertFalse(all(torch.equal(value, right[name])
                             for name, value in left.items()))

    # ---------------------------------------------------------------- resume
    def test_strict_resume_equals_uninterrupted(self):
        protocol = make_protocol(
            self.root, name="EXP-B0-RESUME", max_steps=10,
            validation_steps=(6, 10), candidate_steps=(6, 10))
        prereg = self.root / "EXP-B0-RESUME-prereg.json"
        uninterrupted = self.root / "uninterrupted"
        resumed = self.root / "resumed"
        base_args = ("--prereg", prereg, "--train", "--seed", "11")
        code, _stdout, stderr = invoke(*base_args, "--out-dir", uninterrupted,
                                       "--max-steps", "10",
                                       "--allow-full-budget")
        self.assertEqual(code, 0, stderr)
        code, _stdout, stderr = invoke(*base_args, "--out-dir", resumed,
                                       "--max-steps", "6")
        self.assertEqual(code, 0, stderr)
        checkpoint = (resumed / "EXP-B0-RESUME" / "run_seed11" / "candidates"
                      / "step_000006.pt")
        code, _stdout, stderr = invoke(*base_args, "--out-dir", resumed,
                                       "--max-steps", "10",
                                       "--allow-full-budget",
                                       "--resume", checkpoint)
        self.assertEqual(code, 0, stderr)
        left = load_payload(uninterrupted / "EXP-B0-RESUME" / "run_seed11"
                            / "candidates" / "step_000010.pt")
        right = load_payload(resumed / "EXP-B0-RESUME" / "run_seed11"
                             / "candidates" / "step_000010.pt")
        for name, tensor in left["model"].items():
            self.assertTrue(torch.equal(tensor, right["model"][name]), name)
        for key in ("optimizer", "scheduler", "ema"):
            assert_state_equal(self, left[key], right[key], key)
        self.assertEqual(left["counters"], right["counters"])
        left_selection = json.loads(
            (uninterrupted / "EXP-B0-RESUME" / "selection.json").read_text())
        right_selection = json.loads(
            (resumed / "EXP-B0-RESUME" / "selection.json").read_text())
        self.assertEqual(left_selection["candidate_id"],
                         right_selection["candidate_id"])
        self.assertEqual(left_selection["considered"],
                         right_selection["considered"])
        self.assertEqual(len(right_selection["considered"]), 2)
        print("PR48_RESUME resumed step-10 model/optimizer/scheduler/EMA == "
              "uninterrupted; selection registry preserved")

    def test_legacy_and_cross_path_resume_refused(self):
        protocol = make_protocol(self.root, name="EXP-B0-LEGACY",
                                 max_steps=10, validation_steps=(5, 10),
                                 candidate_steps=(5, 10))
        prereg = self.root / "EXP-B0-LEGACY-prereg.json"
        metadata = ShardReader(protocol.train_shard, purpose="audit").metadata
        set_seed(5)
        model = Policy(protocol.model_config_object())
        config = protocol.train.training_config(11)
        optimizer = make_optimizer(model, config)
        legacy = self.root / "legacy.pt"
        save_checkpoint(
            legacy, model=model, optimizer=optimizer,
            counters={"step": 1, "tokens": 0, "decisions": 0},
            hashes={"schema_hash": SCHEMA_HASH, "action_map_hash": ACTION_HASH,
                    "normalizer_hash": NORMALIZER_HASH},
            config=config, seed=11, model_config=protocol.model_config)
        out = self.root / "legacy_out"
        code, _stdout, stderr = invoke(
            "--prereg", prereg, "--train", "--max-steps", "10",
            "--allow-full-budget", "--out-dir", out, "--seed", "11",
            "--resume", legacy)
        self.assertEqual(code, 1)
        self.assertIn("provenance", stderr)
        cross = self.root / "cross.pt"
        provenance = checkpoint_provenance(protocol, metadata)
        provenance["training_path"] = "prefixes"
        save_checkpoint(
            cross, model=model, optimizer=optimizer,
            counters={"step": 1, "tokens": 0, "decisions": 0},
            hashes={"schema_hash": SCHEMA_HASH, "action_map_hash": ACTION_HASH,
                    "normalizer_hash": NORMALIZER_HASH},
            config=config, seed=11, model_config=protocol.model_config,
            provenance=provenance, extra={"training_path": "prefixes"})
        code, _stdout, stderr = invoke(
            "--prereg", prereg, "--train", "--max-steps", "10",
            "--allow-full-budget", "--out-dir", out, "--seed", "11",
            "--resume", cross)
        self.assertEqual(code, 1)
        self.assertIn("training_path", stderr)

    def test_config_and_data_hash_mismatch_refused(self):
        protocol = make_protocol(self.root, name="EXP-B0-HASH",
                                 max_steps=10, validation_steps=(6, 10),
                                 candidate_steps=(6, 10), lr=9e-4)
        other_prereg = self.root / "EXP-B0-HASH-prereg.json"
        other_out = self.root / "hash_other"
        code, _stdout, stderr = invoke(
            "--prereg", other_prereg, "--train", "--max-steps", "6",
            "--out-dir", other_out, "--seed", "11")
        self.assertEqual(code, 0, stderr)
        # The same checkpoint under a different (frozen) config must refuse.
        base = make_protocol(self.root, name="EXP-B0-BASE", max_steps=10,
                             validation_steps=(6, 10), candidate_steps=(6, 10),
                             lr=1e-3)
        base_prereg = self.root / "EXP-B0-BASE-prereg.json"
        code, _stdout, stderr = invoke(
            "--prereg", base_prereg, "--train", "--max-steps", "10",
            "--allow-full-budget", "--out-dir", self.root / "base_out",
            "--seed", "11", "--resume",
            other_out / "EXP-B0-HASH" / "run_seed11" / "candidates"
            / "step_000006.pt")
        self.assertEqual(code, 1)
        self.assertIn("mismatch", stderr)
        # A tampered prereg shard hash is refused before any training.
        document = json.loads(base_prereg.read_text(encoding="utf-8"))
        document["train_shard_sha256"] = "0" * 64
        tampered = self.root / "tampered-prereg.json"
        tampered.write_text(json.dumps(document), encoding="utf-8")
        code, _stdout, stderr = invoke("--prereg", tampered, "--dry-run")
        self.assertEqual(code, 2)
        self.assertIn("sha256 mismatch", stderr)

    # -------------------------------------------------------------- selection
    def test_two_seed_run_selects_one_winner_and_manifest(self):
        out = self.root / "two_seed"
        code, stdout, stderr = invoke(
            "--prereg", self.prereg, "--train", "--max-steps", "10",
            "--allow-full-budget", "--out-dir", out, "--verbose")
        self.assertEqual(code, 0, stderr)
        experiment = out / "EXP-B0-CLI"
        selection = json.loads((experiment / "selection.json").read_text())
        manifest = json.loads((experiment / "manifest.json").read_text())
        self.assertTrue(selection["complete"])
        self.assertIn(selection["candidate_id"], selection["considered"])
        self.assertEqual(selection["weights"], "raw")
        self.assertEqual(selection["metric"], "validation_ce")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["prereg"]["fingerprint"],
                         self.protocol.fingerprint())
        self.assertNotIn("full_b0_epoch", manifest["not_run"])
        self.assertIn("arena_select", manifest["not_run"])
        self.assertIn("strength_vs_m5d_r0", manifest["not_run"])
        self.assertEqual(manifest["strength_claim"], "none")
        self.assertEqual(len(manifest["candidates"]), 2)
        self.assertTrue((experiment / "PREREG.md").is_file())
        self.assertTrue((experiment / "protocol.json").is_file())
        self.assertIn("B0_SELECTED", stdout)

    def test_validate_only_evaluates_without_training(self):
        out = self.root / "validate_run"
        code, _stdout, stderr = self._train(out, "--allow-full-budget",
                                            "--seed", "11")
        self.assertEqual(code, 0, stderr)
        checkpoint = out / self.candidate_rel
        before = sorted((out / "EXP-B0-CLI" / "run_seed11" / "candidates")
                        .iterdir())
        validate_out = self.root / "validate_out"
        code, stdout, stderr = invoke(
            "--prereg", self.prereg, "--validate-only",
            "--checkpoint", checkpoint, "--out-dir", validate_out)
        self.assertEqual(code, 0, stderr)
        self.assertIn("B0_VALIDATE", stdout)
        record = json.loads((validate_out / "validation_only.json").read_text())
        self.assertEqual(record["checkpoint_step"], 10)
        self.assertEqual(record["seed"], 11)
        self.assertTrue(math.isfinite(record["raw"]["ce"]))
        self.assertIsNotNone(record["ema"])
        self.assertIn("strength_vs_m5d_r0", record["not_run"])
        after = sorted((out / "EXP-B0-CLI" / "run_seed11" / "candidates")
                       .iterdir())
        self.assertEqual([path.name for path in before],
                         [path.name for path in after])

    def test_private_shard_policy_authentication(self):
        from rf_test_training import (
            build_private_shard, private_policy, write_policy_config,
        )
        root = self.root / "private"
        root.mkdir()
        policy = private_policy()
        policy_config = write_policy_config(root, policy)
        train = root / "private_train.rfshard"
        validation = root / "private_validation.rfshard"
        build_private_shard(train, policy, games=16)
        build_private_shard(validation, policy, games=6, validation=True)
        protocol = build_protocol_from_files(
            experiment_id="EXP-B0-PRIVATE", train_shard=str(train),
            validation_shard=str(validation), model_config=tiny_config(),
            train=train_settings(max_steps=2, ema_decay=0.9),
            seeds=(11, 22), validation_steps=(2,), candidate_steps=(2,),
            selection=B0SelectionRule(weights="raw"),
            stop_policy=B0StopPolicy(max_wall_seconds=120.0),
            policy_config=str(policy_config), policy_hash=policy.policy_hash,
            created_utc="2026-11-21T00:00:00Z")
        prereg = root / "prereg.json"
        protocol.write_json(prereg)
        code, stdout, stderr = invoke("--prereg", prereg, "--dry-run")
        self.assertEqual(code, 0, stderr)
        self.assertIn("B0_DRY_RUN_OK", stdout)
        # A prereg without operator-policy authentication cannot open it.
        document = protocol.as_dict()
        document["policy_config"] = ""
        document["policy_hash"] = ""
        unauthenticated = root / "unauthenticated.json"
        unauthenticated.write_text(json.dumps(document), encoding="utf-8")
        code, _stdout, stderr = invoke("--prereg", unauthenticated, "--dry-run")
        self.assertEqual(code, 2)
        self.assertIn("unauthenticated private shard", stderr)
        # A wrong policy hash is refused before any step.
        document = protocol.as_dict()
        document["policy_hash"] = "e" * 64
        wrong = root / "wrong.json"
        wrong.write_text(json.dumps(document), encoding="utf-8")
        code, _stdout, stderr = invoke("--prereg", wrong, "--dry-run")
        self.assertEqual(code, 2)
        self.assertIn("policy", stderr)
        # The authenticated private path trains and emits a candidate.
        code, stdout, stderr = invoke(
            "--prereg", prereg, "--train", "--max-steps", "2",
            "--allow-full-budget", "--out-dir", root / "out", "--seed", "11")
        self.assertEqual(code, 0, stderr)
        self.assertIn("B0_CANDIDATE", stdout)
        self.assertTrue((root / "out" / "EXP-B0-PRIVATE" / "run_seed11"
                        / "candidates" / "step_000002.pt").is_file())

    def test_template_requires_max_steps(self):
        model_config = self.root / "tiny.yaml"
        model_config.write_text(
            "model:\n  name: b0_tiny\n  hidden: 64\n  layers: 1\n"
            "  heads: 4\n  ffn: 32\n  state_hidden: 32\n"
            "  max_positions: 2048\n", encoding="utf-8")
        code, _stdout, stderr = invoke(
            "--emit-prereg-template", "--experiment-id", "EXP-B0-TEMPLATE",
            "--shard", self.protocol.train_shard,
            "--validation-shard", self.protocol.validation_shard,
            "--model-config", model_config,
            "--out", self.root / "template" / "no-max.json")
        self.assertEqual(code, 2)
        self.assertIn("requires --max-steps", stderr)

    def test_emit_prereg_template_roundtrips(self):
        model_config = self.root / "tiny.yaml"
        model_config.write_text(
            "model:\n  name: b0_tiny\n  hidden: 64\n  layers: 1\n"
            "  heads: 4\n  ffn: 32\n  state_hidden: 32\n"
            "  max_positions: 2048\n", encoding="utf-8")
        prereg = self.root / "template" / "prereg.json"
        code, stdout, stderr = invoke(
            "--emit-prereg-template", "--experiment-id", "EXP-B0-TEMPLATE",
            "--shard", self.protocol.train_shard,
            "--validation-shard", self.protocol.validation_shard,
            "--model-config", model_config, "--out", prereg,
            "--max-steps", "20", "--validation-every", "5",
            "--candidate-every", "10", "--seed", "11", "--seed", "22")
        self.assertEqual(code, 0, stderr)
        self.assertIn("B0_PREREG_TEMPLATE", stdout)
        self.assertTrue(prereg.is_file())
        self.assertTrue(prereg.with_suffix(".md").is_file())
        loaded = B0Protocol.load_json(prereg)
        self.assertEqual(loaded.train.max_steps, 20)
        self.assertIn(20, loaded.candidate_steps)
        self.assertTrue(set(loaded.candidate_steps) <= set(loaded.validation_steps))
        code, stdout, _stderr = invoke("--prereg", prereg, "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("B0_DRY_RUN_OK", stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
