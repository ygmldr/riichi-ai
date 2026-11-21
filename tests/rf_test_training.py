#!/usr/bin/env python3
"""PR-14 synthetic-only BC trainer/checkpoint tests."""

from __future__ import annotations

import contextlib
import dataclasses
import io
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
    ACTION_HASH, NORMALIZER_HASH, SCHEMA_HASH, ShardMetadata, ShardRightsError,
    ShardWriter, make_game_record,
)
from riichiformer.models.policy import MVPConfig, Policy, PolicyOutput  # noqa: E402
from riichiformer.policy.config import PolicyConfig  # noqa: E402
from riichiformer.schemas.action_v3 import ACTION_SPACE  # noqa: E402
from riichiformer.train.bc import TrainingError, masked_bc_loss, training_step  # noqa: E402
from riichiformer.train.checkpoint import (  # noqa: E402
    CheckpointError, CheckpointMismatch, load_checkpoint, save_checkpoint,
)
from riichiformer.train.optimizer import (  # noqa: E402
    TrainingConfig, make_optimizer, make_scheduler, set_seed,
)
from scripts.rf_train import main as train_main  # noqa: E402


HAND0 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
         "1p", "2p", "3p", "4p"]
HAND1 = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
         "E", "S", "W", "N"]
HAND2 = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
         "7m", "8m", "9m", "1p"]
HAND3 = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
         "1s", "2s", "3s", "4s", "5s"]


def raw_four() -> list[dict]:
    return [
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1p",
         "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
         "scores": [25000] * 4,
         "tehais": [HAND0, HAND1, HAND2, HAND3]},
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


def metadata(*, purpose="train", train_allowed=True, rights="ALLOWED"):
    return ShardMetadata(
        purpose=purpose, source_id="synthetic_allowed", rights_status=rights,
        train_allowed=train_allowed,
        canonical_manifest_sha256="a" * 64,
        canonical_index_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
        split_version="rf-split/2",
        schema_version="rf-schema-v3/2", schema_hash=SCHEMA_HASH,
        action_map_hash=ACTION_HASH, normalizer_hash=NORMALIZER_HASH,
    )


def build_shard(path: Path, games: int = 4, *, allowed: bool = True) -> None:
    raw = raw_four()
    samples = build_decisions(raw)
    writer = ShardWriter(
        path, metadata(train_allowed=allowed,
                       rights="ALLOWED" if allowed else "UNKNOWN",
                       purpose="train" if allowed else "audit"))
    for index in range(games):
        record = make_game_record(
            raw, samples, game_id=f"g{index:02d}",
            component_id=f"comp-{index}", component_sha256="d" * 64,
            split="train", source_id="synthetic_allowed",
            train_allowed=allowed,
            rights_status="ALLOWED" if allowed else "UNKNOWN")
        writer.add_game(record)
    writer.finalize()


def tiny_model(seed: int = 7) -> Policy:
    set_seed(seed)
    return Policy(MVPConfig(layers=1, ffn=32, state_hidden=32, max_positions=2048))


def batches(path: Path, size: int = 4):
    return list(ShardReader(path, purpose="train").iter_batches(batch_size=size))


class BCLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "train.rfshard"
        build_shard(cls.path)
        cls.batch = batches(cls.path, 4)[0]
        cls.model = tiny_model()
        cls.model.eval()
        with torch.no_grad():
            cls.output = cls.model(cls.batch.batch)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_censored_and_forced_targets_excluded(self):
        targets = [dict(target) for target in self.batch.targets]
        full = masked_bc_loss(self.output, targets)
        self.assertEqual(full.valid_decisions, len(targets))
        targets[0]["supervision_valid"] = False
        targets[0]["censor_reason"] = "preempted_by_call"
        targets[1]["forced"] = True
        reduced = masked_bc_loss(self.output, targets)
        self.assertEqual(reduced.valid_decisions, len(targets) - 2)
        mask = self.batch.batch.legal_mask
        expected = 0.0
        count = 0
        for row, target in enumerate(targets):
            if not target.get("supervision_valid") or target.get("forced"):
                continue
            expected += float(-self.output.logprobs[row, target["expert_action"]])
            count += 1
        self.assertAlmostEqual(float(reduced.loss), expected / count, places=6)

    def test_label_outside_mask_fails(self):
        targets = [dict(target) for target in self.batch.targets]
        row = 0
        illegal = next(index for index in range(ACTION_SPACE)
                       if not bool(self.batch.batch.legal_mask[row, index]))
        targets[row]["expert_action"] = illegal
        with self.assertRaises(TrainingError):
            masked_bc_loss(self.output, targets)

    def test_nan_logprob_fails(self):
        target_action = self.batch.targets[0]["expert_action"]
        with torch.no_grad():
            self.output.logprobs[0, target_action] = float("nan")
        with self.assertRaises(TrainingError):
            masked_bc_loss(self.output, self.batch.targets)
        # restore the output for other tests
        with torch.no_grad():
            self.output = self.model(self.batch.batch)

    def test_all_censored_is_zero_and_no_step(self):
        targets = [dict(target, supervision_valid=False,
                        censor_reason="preempted_by_call") for target in self.batch.targets]
        result = masked_bc_loss(self.output, targets)
        self.assertEqual(result.valid_decisions, 0)
        self.assertEqual(float(result.loss), 0.0)
        model = tiny_model()
        optimizer = make_optimizer(model, TrainingConfig(seed=7))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        step = training_step(model, self.batch.batch, targets, optimizer=optimizer)
        self.assertEqual(step.valid_decisions, 0)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))


class TrainingLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "train.rfshard"
        build_shard(cls.path, games=6)
        cls.batches = batches(cls.path, 4)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _train(self, model, config, max_steps=2):
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, config)
        losses = []
        for batch in self.batches[:max_steps]:
            result = training_step(model, batch.batch, batch.targets,
                                   optimizer=optimizer, scheduler=scheduler,
                                   max_grad_norm=config.grad_clip)
            losses.append(float(result.loss.detach()))
        return model, optimizer, scheduler, losses

    def test_two_same_seed_short_runs_identical(self):
        config = TrainingConfig(seed=123, lr=1e-3, warmup_steps=1)
        model_a = tiny_model(123)
        model_b = tiny_model(123)
        _, _, _, losses_a = self._train(model_a, config)
        _, _, _, losses_b = self._train(model_b, config)
        self.assertEqual(losses_a, losses_b)
        for name, value in model_a.state_dict().items():
            self.assertTrue(torch.equal(value, model_b.state_dict()[name]), name)

    def test_resume_next_batch_loss_identical(self):
        config = TrainingConfig(seed=321, lr=1e-3, warmup_steps=2)
        model_a = tiny_model(321)
        optimizer_a = make_optimizer(model_a, config)
        scheduler_a = make_scheduler(optimizer_a, config)
        for batch in self.batches[:2]:
            training_step(model_a, batch.batch, batch.targets,
                          optimizer=optimizer_a, scheduler=scheduler_a)
        with tempfile.TemporaryDirectory() as temp:
            ckpt = Path(temp) / "step2.pt"
            save_checkpoint(
                ckpt, model=model_a, optimizer=optimizer_a, scheduler=scheduler_a,
                counters={"step": 2, "tokens": 0, "decisions": 0},
                hashes={"schema_hash": SCHEMA_HASH, "action_map_hash": ACTION_HASH,
                        "normalizer_hash": NORMALIZER_HASH},
                config=config, seed=config.seed)
            third = self.batches[2]
            training_step(model_a, third.batch, third.targets,
                          optimizer=optimizer_a, scheduler=scheduler_a)
            expected = model_a.state_dict()["policy_head.linear.weight"].clone()
            model_b = tiny_model(321)
            optimizer_b = make_optimizer(model_b, config)
            scheduler_b = make_scheduler(optimizer_b, config)
            info, payload = load_checkpoint(
                ckpt, model=model_b, optimizer=optimizer_b, scheduler=scheduler_b,
                expected_hashes={"schema_hash": SCHEMA_HASH,
                                 "action_map_hash": ACTION_HASH,
                                 "normalizer_hash": NORMALIZER_HASH})
            self.assertEqual(info.step, 2)
            training_step(model_b, third.batch, third.targets,
                          optimizer=optimizer_b, scheduler=scheduler_b)
            self.assertTrue(torch.equal(
                model_b.state_dict()["policy_head.linear.weight"], expected))

    def test_loss_decreases_on_synthetic_batch(self):
        model = tiny_model(99)
        config = TrainingConfig(seed=99, lr=3e-3, warmup_steps=0)
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, config)
        batch = self.batches[0]
        losses = []
        for _ in range(12):
            result = training_step(model, batch.batch, batch.targets,
                                   optimizer=optimizer, scheduler=scheduler)
            losses.append(float(result.loss.detach()))
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        print(f"PR14_LOSS first={losses[0]:.6f} last={losses[-1]:.6f} "
              f"params={sum(p.numel() for p in model.parameters())}")

    def test_checkpoint_roundtrip_and_mismatch(self):
        model = tiny_model(5)
        config = TrainingConfig(seed=5)
        optimizer = make_optimizer(model, config)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            info = save_checkpoint(
                path, model=model, optimizer=optimizer,
                counters={"step": 1, "tokens": 12, "decisions": 4},
                hashes={"schema_hash": SCHEMA_HASH, "action_map_hash": ACTION_HASH,
                        "normalizer_hash": NORMALIZER_HASH},
                config=config, seed=config.seed)
            self.assertEqual(len(info.sha256), 64)
            model2 = tiny_model(8)
            loaded, _payload = load_checkpoint(
                path, model=model2, optimizer=make_optimizer(model2, config),
                expected_hashes={"schema_hash": SCHEMA_HASH,
                                 "action_map_hash": ACTION_HASH,
                                 "normalizer_hash": NORMALIZER_HASH})
            self.assertEqual(loaded.step, 1)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, model2.state_dict()[name]))
            with self.assertRaises(CheckpointMismatch):
                load_checkpoint(
                    path, model=model2,
                    expected_hashes={"schema_hash": "0" * 64,
                                     "action_map_hash": ACTION_HASH,
                                     "normalizer_hash": NORMALIZER_HASH})


class RightsAndCLITests(unittest.TestCase):
    def test_blocked_shard_refused_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "audit.rfshard"
            build_shard(path, games=1, allowed=False)
            with self.assertRaises(ShardRightsError):
                ShardReader(path, purpose="train")
            code = train_main(["--shard", str(path), "--dry-run"])
            self.assertEqual(code, 2)

    def test_cli_dry_run_on_allowed_synthetic_shard(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.rfshard"
            build_shard(path, games=2)
            self.assertEqual(train_main(["--shard", str(path), "--dry-run",
                                         "--steps", "1"]), 0)

    def test_frozen_output_path_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.rfshard"
            build_shard(path, games=1)
            code = train_main(["--shard", str(path), "--train", "--out",
                               str(REPO_ROOT / "checkpoints" / "forbidden.pt"),
                               "--max-steps", "1"])
            self.assertEqual(code, 2)


def private_policy(allowlist=("tenhou_old",),
                   reference="unit-test-private-operator") -> PolicyConfig:
    return PolicyConfig(
        profile="operator_private_research",
        operator_reference=reference,
        allowed_local_sources=frozenset(allowlist),
        scope="local_nonpublic_only",
    )


def write_policy_config(directory: Path, policy: PolicyConfig) -> Path:
    path = directory / f"policy-{policy.policy_hash[:8]}.json"
    path.write_text(json.dumps(policy.to_dict()), encoding="utf-8")
    return path


def private_metadata(policy: PolicyConfig) -> ShardMetadata:
    return ShardMetadata(
        purpose="train", source_id="tenhou_old", rights_status="RESTRICTED",
        train_allowed=False,
        canonical_manifest_sha256="a" * 64,
        canonical_index_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
        split_version="rf-split/2",
        schema_version="rf-schema-v3/2", schema_hash=SCHEMA_HASH,
        action_map_hash=ACTION_HASH, normalizer_hash=NORMALIZER_HASH,
        research_policy="operator_private_research",
        research_policy_hash=policy.policy_hash,
        policy_version=policy.policy_version,
        train_eligible_under_policy=True,
    )


def build_private_shard(path: Path, policy: PolicyConfig, games: int = 2) -> None:
    raw = raw_four()
    samples = build_decisions(raw)
    writer = ShardWriter(path, private_metadata(policy))
    for index in range(games):
        record = make_game_record(
            raw, samples, game_id=f"p{index:02d}",
            component_id=f"pcomp-{index}", component_sha256="d" * 64,
            split="train", source_id="tenhou_old", train_allowed=False,
            rights_status="RESTRICTED")
        writer.add_game(record)
    writer.finalize()


class PrivatePolicyCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.policy = private_policy()
        self.config = write_policy_config(self.dir, self.policy)
        self.shard = self.dir / "private.rfshard"
        build_private_shard(self.shard, self.policy)

    def tearDown(self):
        self.temp.cleanup()

    def _dry(self, *extra):
        return train_main(["--shard", str(self.shard), "--dry-run", "--steps", "1",
                           *extra])

    def test_private_shard_refused_without_authentication(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code = self._dry()
        self.assertEqual(code, 2)
        self.assertIn("expected policy hash", err.getvalue())

    def test_private_shard_dry_run_with_policy_config(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = self._dry("--policy-config", str(self.config))
        self.assertEqual(code, 0)
        self.assertIn("PR14_POLICY_NOTE", out.getvalue())
        self.assertIn("external_license_status=RESTRICTED", out.getvalue())

    def test_private_shard_dry_run_with_expected_hash(self):
        code = self._dry("--expected-policy-hash", self.policy.policy_hash)
        self.assertEqual(code, 0)

    def test_private_shard_wrong_expected_hash_refused(self):
        code = self._dry("--expected-policy-hash", "e" * 64)
        self.assertEqual(code, 2)

    def test_private_shard_wrong_config_hash_refused(self):
        other = write_policy_config(
            self.dir, private_policy(reference="different-operator-reference"))
        code = self._dry("--policy-config", str(other))
        self.assertEqual(code, 2)

    def test_private_shard_source_outside_allowlist_refused(self):
        # Adversarial fixture: correct policy hash, but a source outside that
        # policy's allowlist (as a differently-built/tampered shard could claim).
        meta = dataclasses.replace(
            private_metadata(self.policy), source_id="selfplay_own",
            rights_status="UNKNOWN", external_license_status="UNKNOWN")
        path = self.dir / "outside-allowlist.rfshard"
        raw = raw_four()
        samples = build_decisions(raw)
        writer = ShardWriter(path, meta)
        writer.add_game(make_game_record(
            raw, samples, game_id="outside", component_id="comp-outside",
            component_sha256="d" * 64, split="train", source_id="selfplay_own",
            train_allowed=False, rights_status="UNKNOWN"))
        writer.finalize()
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code = train_main(["--shard", str(path), "--dry-run",
                               "--policy-config", str(self.config)])
        self.assertEqual(code, 2)
        self.assertIn("outside the expected policy allowlist", err.getvalue())

    def test_cli_expected_hash_must_match_given_config(self):
        code = self._dry("--policy-config", str(self.config),
                         "--expected-policy-hash", "e" * 64)
        self.assertEqual(code, 2)

    def test_licensed_shard_refused_by_private_config(self):
        licensed = self.dir / "licensed.rfshard"
        build_shard(licensed, games=1)
        code = train_main(["--shard", str(licensed), "--dry-run",
                           "--policy-config", str(self.config)])
        self.assertEqual(code, 2)

    def test_private_train_and_resume_provenance(self):
        ckpt = self.dir / "private.pt"
        with contextlib.redirect_stdout(io.StringIO()):
            code = train_main([
                "--shard", str(self.shard), "--train", "--out", str(ckpt),
                "--max-steps", "1", "--batch-size", "4",
                "--policy-config", str(self.config)])
        self.assertEqual(code, 0)
        self.assertTrue(ckpt.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            code = train_main([
                "--shard", str(self.shard), "--train", "--resume", str(ckpt),
                "--out", str(ckpt), "--max-steps", "2", "--batch-size", "4",
                "--policy-config", str(self.config)])
        self.assertEqual(code, 0)
        other_policy = private_policy(reference="different-operator-reference")
        other_config = write_policy_config(self.dir, other_policy)
        other_shard = self.dir / "private-other.rfshard"
        build_private_shard(other_shard, other_policy)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code = train_main([
                "--shard", str(other_shard), "--train", "--resume", str(ckpt),
                "--out", str(ckpt), "--max-steps", "3", "--batch-size", "4",
                "--policy-config", str(other_config)])
        self.assertEqual(code, 1)
        self.assertIn("provenance mismatch for research_policy_hash", err.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
