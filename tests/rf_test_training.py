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
from riichiformer.models.batch import ModelBatch  # noqa: E402
from riichiformer.models.policy import MVPConfig, Policy, PolicyOutput  # noqa: E402
from riichiformer.policy.config import PolicyConfig  # noqa: E402
from riichiformer.schemas.action_v3 import ACTION_SPACE  # noqa: E402
from riichiformer.train.bc import (  # noqa: E402
    TrainingError, masked_bc_loss, training_step, valid_weight_sum,
    window_weight_total,
)
from riichiformer.train.checkpoint import (  # noqa: E402
    CheckpointError, CheckpointMismatch, config_hash, init_weights_only,
    load_checkpoint, save_checkpoint,
)
from riichiformer.train.model_config import (  # noqa: E402
    DEFAULT_MODEL_CONFIG_PATH, MVP_S_REFERENCE_PARAMETERS, ModelConfigError,
    resolve_mvp_config,
)
from riichiformer.train.optimizer import (  # noqa: E402
    TrainingConfig, make_optimizer, make_scheduler, set_seed,
)

EVENT_PAD_FILLS = {"event_type_id": 16, "event_actor_rel": 5,
                   "event_tile_kind": 34, "event_aka": 0,
                   "event_tsumogiri": 0, "event_phase": 3,
                   "event_kyoku_pos": 0}
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


def clone_state(model: Policy) -> dict:
    return {name: value.clone() for name, value in model.state_dict().items()}


def concat_batches(batches) -> ModelBatch:
    """Concatenate ModelBatches into one large batch (padding + row offsets)."""
    max_len = max(batch.seq_len for batch in batches)
    batch_size = sum(batch.batch_size for batch in batches)
    tensors = {}
    for name, fill in EVENT_PAD_FILLS.items():
        rows = []
        for batch in batches:
            padded = torch.full((batch.batch_size, max_len), fill, dtype=torch.long)
            padded[:, :batch.seq_len] = getattr(batch, name)
            rows.append(padded)
        tensors[name] = torch.cat(rows, dim=0)
    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    state_features = []
    legal_mask = []
    decision_batch_index = []
    decision_time_index = []
    decision_phase = []
    row_offset = 0
    for batch in batches:
        padding_mask[row_offset:row_offset + batch.batch_size, :batch.seq_len] = True
        state_features.append(batch.state_features)
        legal_mask.append(batch.legal_mask)
        decision_batch_index.append(batch.decision_batch_index + row_offset)
        decision_time_index.append(batch.decision_time_index)
        decision_phase.append(batch.decision_phase)
        row_offset += batch.batch_size
    return ModelBatch(
        **tensors,
        padding_mask=padding_mask,
        state_features=torch.cat(state_features, dim=0),
        legal_mask=torch.cat(legal_mask, dim=0),
        decision_batch_index=torch.cat(decision_batch_index, dim=0),
        decision_time_index=torch.cat(decision_time_index, dim=0),
        decision_phase=torch.cat(decision_phase, dim=0),
    )


def varied_targets(batch, *, all_censored=False, weights=(2.0, 0.5, 1.0, 1.5)):
    targets = []
    for index, target in enumerate(batch.targets):
        item = dict(target)
        if all_censored:
            item["supervision_valid"] = False
            item["censor_reason"] = "preempted_by_call"
        elif item.get("supervision_valid"):
            item["loss_weight"] = float(weights[index % len(weights)])
        targets.append(item)
    return targets


def max_parameter_diff(left: Policy, right: Policy) -> float:
    worst = 0.0
    for (name, value), (_, other) in zip(left.state_dict().items(),
                                         right.state_dict().items()):
        worst = max(worst, float((value - other).abs().max()))
    return worst


class GradientAccumulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        path = Path(cls.temp.name) / "accum.rfshard"
        build_shard(path, games=6)
        cls.batches = batches(path, 4)[:3]
        cls.micro_targets = [
            varied_targets(cls.batches[0], all_censored=True),
            varied_targets(cls.batches[1]),
            varied_targets(cls.batches[2]),
        ]
        # Force one validated row to be excluded as a forced action.
        for target in cls.micro_targets[1]:
            if target.get("supervision_valid") and not target.get("forced"):
                target["forced"] = True
                break
        cls.big_batch = concat_batches([batch.batch for batch in cls.batches])
        cls.big_targets = [target for step in cls.micro_targets for target in step]
        cls.window_total = window_weight_total(cls.micro_targets)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _big_step(self, config):
        model = tiny_model(config.seed)
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, config)
        result = training_step(model, self.big_batch, self.big_targets,
                               optimizer=optimizer, scheduler=scheduler,
                               max_grad_norm=config.grad_clip)
        return model, optimizer, scheduler, result

    def _accumulated(self, config, *, use_loss_scale=False):
        model = tiny_model(config.seed)
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, config)
        results = []
        for step, (batch, targets) in enumerate(zip(self.batches, self.micro_targets)):
            results.append(training_step(
                model, batch.batch, targets, optimizer=optimizer,
                scheduler=scheduler, max_grad_norm=config.grad_clip,
                accumulation_step=step, accumulation_count=3,
                accumulation_weight_total=None if use_loss_scale else self.window_total,
                loss_scale=(1.0 / self.window_total) if use_loss_scale else None))
        return model, optimizer, scheduler, results

    def test_window_weight_total_matches_big_batch(self):
        self.assertGreater(self.window_total, 0.0)
        self.assertAlmostEqual(valid_weight_sum(self.big_targets),
                               self.window_total, places=9)
        self.assertEqual(valid_weight_sum([dict(self.big_targets[0],
                                                supervision_valid=False)]), 0.0)
        # The all-censored microbatch contributes exactly zero weight.
        self.assertEqual(valid_weight_sum(self.micro_targets[0]), 0.0)

    def test_gradient_sums_match_single_large_batch(self):
        # The additive numerator is the exact object accumulation shares; compare
        # gradients of sum(micro numerators)/window_total against the big batch.
        big_model = tiny_model(77)
        big_model.train()
        big_params = [parameter for parameter in big_model.parameters()
                      if parameter.requires_grad]
        big_result = masked_bc_loss(big_model(self.big_batch), self.big_targets)
        big_grads = torch.autograd.grad(big_result.numerator / self.window_total,
                                        big_params, allow_unused=True)
        model = tiny_model(77)
        model.train()
        micro_params = [parameter for parameter in model.parameters()
                        if parameter.requires_grad]
        accumulated = None
        for batch, targets in zip(self.batches, self.micro_targets):
            result = masked_bc_loss(model(batch.batch), targets)
            if result.numerator is None:
                continue
            grads = torch.autograd.grad(result.numerator / self.window_total,
                                        micro_params, allow_unused=True)
            accumulated = grads if accumulated is None \
                else [left + right for left, right in zip(accumulated, grads)]
        worst = 0.0
        for left, right in zip(big_grads, accumulated):
            if left is None or right is None:
                continue
            worst = max(worst, float((left - right).abs().max()))
        self.assertLess(worst, 1e-6, f"gradient sum mismatch {worst:.3e}")
        print(f"PR39_ACCUM_GRAD max_abs_diff={worst:.3e} "
              f"window_total={self.window_total:.3f}")

    def test_accumulation_matches_single_large_batch(self):
        config = TrainingConfig(seed=77, lr=1e-3, warmup_steps=0)
        big_model, _big_opt, _big_sched, big_result = self._big_step(config)
        model, _opt, _sched, results = self._accumulated(config)
        accumulated_loss = sum(float(result.numerator.detach())
                               for result in results if result.numerator is not None)
        self.assertAlmostEqual(
            accumulated_loss / self.window_total,
            float(big_result.loss.detach()), places=6)
        within = max_parameter_diff(big_model, model)
        # AdamW normalizes per coordinate: a sub-1e-7 gradient-sum difference is
        # amplified where the coordinate's second moment is near zero.  The
        # measured 2.2e-6 on a 1e-3 update is 0.22% of one step; the gradient-sum
        # test above is the tight equivalence assertion.
        self.assertLess(within, 1e-5, f"parameter delta mismatch {within:.3e}")
        print(f"PR39_ACCUM max_param_diff={within:.3e} "
              f"window_total={self.window_total:.3f} "
              f"big_loss={float(big_result.loss.detach()):.6f}")

    def test_explicit_loss_scale_matches_big_batch(self):
        config = TrainingConfig(seed=78, lr=1e-3, warmup_steps=0)
        big_model, _, _, _ = self._big_step(config)
        model, _, _, _ = self._accumulated(config, use_loss_scale=True)
        within = max_parameter_diff(big_model, model)
        self.assertLess(within, 1e-6, f"loss_scale parameter mismatch {within:.3e}")

    def test_all_censored_microbatch_still_steps_at_boundary(self):
        config = TrainingConfig(seed=79, lr=1e-3, warmup_steps=0)
        model = tiny_model(config.seed)
        before = clone_state(model)
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, config)
        step = training_step(model, self.batches[0].batch, self.micro_targets[0],
                             optimizer=optimizer, accumulation_step=0,
                             accumulation_count=2,
                             accumulation_weight_total=self.window_total)
        self.assertEqual(step.valid_decisions, 0)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertIsNone(optimizer.param_groups[0]["params"][0].grad)
        # Second microbatch has supervision; the boundary step must execute.
        training_step(model, self.batches[1].batch, self.micro_targets[1],
                      optimizer=optimizer, scheduler=scheduler,
                      accumulation_step=1, accumulation_count=2,
                      accumulation_weight_total=self.window_total)
        changed = any(not torch.equal(value, before[name])
                      for name, value in model.state_dict().items())
        self.assertTrue(changed)

    def test_missing_window_total_fails_closed(self):
        model = tiny_model(80)
        optimizer = make_optimizer(model, TrainingConfig(seed=80))
        with self.assertRaises(TrainingError):
            training_step(model, self.batches[1].batch, self.micro_targets[1],
                          optimizer=optimizer, accumulation_step=0,
                          accumulation_count=2)
        with self.assertRaises(TrainingError):
            training_step(model, self.batches[1].batch, self.micro_targets[1],
                          optimizer=optimizer, accumulation_step=0,
                          accumulation_count=2,
                          accumulation_weight_total=self.window_total,
                          loss_scale=1.0)
        with self.assertRaises(TrainingError):
            training_step(model, self.batches[1].batch, self.micro_targets[1],
                          optimizer=optimizer, loss_scale=0.0)
        with self.assertRaises(TrainingError):
            training_step(model, self.batches[1].batch, self.micro_targets[1],
                          optimizer=optimizer, accumulation_step=0,
                          accumulation_count=2, accumulation_weight_total=0.0)

    def test_all_censored_window_never_steps(self):
        model = tiny_model(81)
        before = clone_state(model)
        optimizer = make_optimizer(model, TrainingConfig(seed=81))
        scheduler = make_scheduler(optimizer, TrainingConfig(seed=81))
        zero_targets = [varied_targets(batch, all_censored=True)
                        for batch in self.batches]
        total = window_weight_total(zero_targets)
        self.assertEqual(total, 0.0)
        for step, (batch, targets) in enumerate(zip(self.batches, zero_targets)):
            result = training_step(
                model, batch.batch, targets, optimizer=optimizer,
                scheduler=scheduler, accumulation_step=step,
                accumulation_count=3,
                accumulation_weight_total=total if total > 0 else None)
            self.assertEqual(result.valid_decisions, 0)
        # Every microbatch was censored: no backward, no step, no parameter drift.
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertEqual(scheduler.last_epoch, 0)


class ModelConfigResolutionTests(unittest.TestCase):
    def test_default_is_the_frozen_six_layer_mvp_s(self):
        resolved = resolve_mvp_config()
        self.assertFalse(resolved.tiny_smoke)
        self.assertTrue(resolved.reference_verified)
        self.assertEqual(resolved.config.layers, 6)
        self.assertEqual(resolved.parameter_count, MVP_S_REFERENCE_PARAMETERS)
        self.assertEqual(resolved.parameter_count, 5_217_582)
        self.assertEqual(resolved.config.ffn, 704)
        self.assertEqual(resolved.config.state_hidden, 512)
        self.assertEqual(resolved.config.dropout, 0.0)
        self.assertEqual(len(resolved.config_hash), 64)
        self.assertEqual(resolved.source_path, DEFAULT_MODEL_CONFIG_PATH)
        provenance = resolved.as_provenance()
        self.assertEqual(provenance["model_parameters"], 5_217_582)
        self.assertFalse(provenance["model_tiny_smoke"])
        self.assertEqual(provenance["model_config_hash"], resolved.config_hash)
        self.assertIn("parameters=5217582", resolved.describe())

    def test_tiny_smoke_is_explicit_and_separate(self):
        resolved = resolve_mvp_config(tiny_smoke=True)
        self.assertTrue(resolved.tiny_smoke)
        self.assertFalse(resolved.reference_verified)
        self.assertIsNone(resolved.source_path)
        self.assertLess(resolved.parameter_count, 1_000_000)
        self.assertEqual(resolved.config.layers, 1)
        with self.assertRaises(ModelConfigError):
            resolve_mvp_config(
                model_config_path=str(DEFAULT_MODEL_CONFIG_PATH), tiny_smoke=True)
        # The default path never silently returns a small model.
        self.assertGreater(resolve_mvp_config().parameter_count, 4_000_000)

    def test_explicit_model_config_is_allowed_and_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "small.yaml"
            path.write_text(
                "model:\n"
                "  name: research_small\n"
                "  architecture: local_causal\n"
                "  hidden: 128\n"
                "  layers: 2\n"
                "  heads: 2\n"
                "  ffn: 256\n"
                "  state_encoder: mlp\n"
                "  state_hidden: 128\n"
                "  policy_head: linear46\n"
                "  memory: false\n"
                "  dropout: 0.0\n"
                "  rope_base: 10000.0\n"
                "  rmsnorm_eps: 1.0e-6\n"
                "  max_positions: 512\n", encoding="utf-8")
            resolved = resolve_mvp_config(path)
            self.assertFalse(resolved.tiny_smoke)
            self.assertFalse(resolved.reference_verified)
            self.assertEqual(resolved.config.layers, 2)
            self.assertEqual(resolved.source_path, path)
            self.assertGreater(resolved.parameter_count, 0)
            self.assertLess(resolved.parameter_count, 1_000_000)

    def test_drifted_or_missing_default_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "configs" / "riichiformer"
            directory.mkdir(parents=True)
            (directory / "mvp_s.yaml").write_text(
                "model:\n"
                "  name: mvp_s\n"
                "  architecture: local_causal\n"
                "  hidden: 256\n"
                "  layers: 1\n"
                "  heads: 4\n"
                "  ffn: 64\n"
                "  state_encoder: mlp\n"
                "  state_hidden: 64\n"
                "  policy_head: linear46\n"
                "  memory: false\n"
                "  dropout: 0.0\n"
                "  rope_base: 10000.0\n"
                "  rmsnorm_eps: 1.0e-6\n"
                "  max_positions: 2048\n", encoding="utf-8")
            with self.assertRaises(ModelConfigError):
                resolve_mvp_config(repo_root=root)
        with self.assertRaises(ModelConfigError):
            resolve_mvp_config("/nonexistent/model.yaml")


def provenance_dict() -> dict:
    return {
        "source_id": "synthetic_allowed",
        "split_manifest_sha256": "c" * 64,
        "canonical_manifest_sha256": "a" * 64,
        "canonical_index_sha256": "b" * 64,
        "research_policy": "operator_private_research",
        "research_policy_hash": "d" * 64,
        "rights_status": "UNKNOWN",
    }


def rewrite_payload(path: Path, mutate) -> None:
    """Test-only helper: mutate a checkpoint payload and refresh the sidecar hash."""
    import hashlib
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mutate(payload)
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = Path(str(path) + ".json")
    import json
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    manifest["sha256"] = digest
    sidecar.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")) + "\n", encoding="utf-8")


class CheckpointResumeTests(unittest.TestCase):
    def _checkpoint(self, path: Path, *, config, model_config=None,
                    provenance=None, with_optimizer=True):
        model = tiny_model(config.seed)
        optimizer = make_optimizer(model, config) if with_optimizer else None
        scheduler = make_scheduler(optimizer, config) if optimizer is not None else None
        if optimizer is not None:
            # Materialize AdamW state so strict resume has real state to restore.
            for parameter in model.parameters():
                parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        info = save_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            counters={"step": 3, "tokens": 12, "decisions": 5},
            hashes={"schema_hash": SCHEMA_HASH, "action_map_hash": ACTION_HASH,
                    "normalizer_hash": NORMALIZER_HASH},
            config=config, seed=config.seed, model_config=model_config,
            provenance=provenance)
        return model, info

    def test_config_model_config_and_provenance_roundtrip(self):
        config = TrainingConfig(seed=11)
        model_config = MVPConfig(layers=1, ffn=32, state_hidden=32,
                                 max_positions=512)
        provenance = provenance_dict()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ckpt.pt"
            saved, info = self._checkpoint(
                path, config=config, model_config=model_config,
                provenance=provenance)
            self.assertEqual(info.config_hash, config_hash(config))
            self.assertEqual(info.model_config_hash, config_hash(model_config))
            self.assertEqual(info.provenance, provenance)
            model = tiny_model(99)
            optimizer = make_optimizer(model, config)
            scheduler = make_scheduler(optimizer, config)
            loaded, payload = load_checkpoint(
                path, model=model, optimizer=optimizer, scheduler=scheduler,
                expected_hashes={"schema_hash": SCHEMA_HASH,
                                 "action_map_hash": ACTION_HASH,
                                 "normalizer_hash": NORMALIZER_HASH},
                expected_config=config,
                expected_config_hash=config_hash(config),
                expected_model_config=model_config,
                expected_model_config_hash=config_hash(model_config),
                expected_provenance=provenance,
                strict_resume=True)
            self.assertEqual(loaded.step, 3)
            self.assertEqual(loaded.config_hash, config_hash(config))
            self.assertEqual(loaded.model_config_hash, config_hash(model_config))
            self.assertEqual(loaded.provenance, provenance)
            self.assertEqual(payload["model_config"],
                             dataclasses.asdict(model_config))
            for name, value in saved.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]))
            # Optimizer state was restored, not just the weights.
            self.assertTrue(optimizer.state)

    def test_config_hash_and_model_config_mismatch_refuse(self):
        config = TrainingConfig(seed=12)
        model_config = MVPConfig(layers=1, ffn=32, state_hidden=32)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ckpt.pt"
            self._checkpoint(path, config=config, model_config=model_config,
                             provenance=provenance_dict())
            model = tiny_model(1)
            with self.assertRaises(CheckpointMismatch):
                load_checkpoint(path, model=model, expected_config_hash="0" * 64)
            with self.assertRaises(CheckpointMismatch):
                load_checkpoint(path, model=model,
                                expected_config=TrainingConfig(seed=12, lr=1e-4))
            with self.assertRaises(CheckpointMismatch):
                load_checkpoint(
                    path, model=model,
                    expected_model_config=MVPConfig(layers=2, ffn=32,
                                                    state_hidden=32))
            with self.assertRaises(CheckpointMismatch):
                load_checkpoint(path, model=model,
                                expected_model_config_hash="1" * 64)
            with self.assertRaises(CheckpointMismatch):
                load_checkpoint(
                    path, model=model,
                    expected_provenance=dict(provenance_dict(),
                                             research_policy_hash="e" * 64))
            with self.assertRaises(CheckpointMismatch):
                load_checkpoint(path, model=model,
                                expected_provenance={"source_id": "other"})

    def test_strict_resume_requires_state(self):
        config = TrainingConfig(seed=13)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ckpt.pt"
            self._checkpoint(path, config=config, with_optimizer=False)
            model = tiny_model(2)
            optimizer = make_optimizer(model, config)
            with self.assertRaises(CheckpointMismatch):
                load_checkpoint(path, model=model, optimizer=optimizer,
                                strict_resume=True)
            # Non-strict loading still works for legacy/partial checkpoints.
            info, _payload = load_checkpoint(path, model=model)
            self.assertEqual(info.step, 3)

            def drop_rng(payload):
                payload["rng"] = None

            rewrite_payload(path, drop_rng)
            with self.assertRaises(CheckpointMismatch):
                load_checkpoint(path, model=tiny_model(2), strict_resume=True)

            def drop_config(payload):
                payload["config"] = None
                payload["config_hash"] = None

            rewrite_payload(path, drop_config)
            with self.assertRaises(CheckpointMismatch):
                load_checkpoint(path, model=tiny_model(2), strict_resume=True)

    def test_init_weights_only_is_not_resume(self):
        config = TrainingConfig(seed=14, lr=1e-2, warmup_steps=0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ckpt.pt"
            trained, info = self._checkpoint(
                path, config=config,
                model_config=MVPConfig(layers=1, ffn=32, state_hidden=32))
            fresh = tiny_model(4242)
            before = clone_state(fresh)
            optimizer = make_optimizer(fresh, config)
            scheduler = make_scheduler(optimizer, config)
            self.assertFalse(any(torch.equal(value, before[name])
                                 for name, value in trained.state_dict().items()
                                 if "policy_head" in name))
            torch.manual_seed(1234)
            rng_before = torch.get_rng_state().clone()
            loaded = init_weights_only(
                path, model=fresh,
                expected_hashes={"schema_hash": SCHEMA_HASH,
                                 "action_map_hash": ACTION_HASH,
                                 "normalizer_hash": NORMALIZER_HASH},
                expected_model_config=MVPConfig(layers=1, ffn=32,
                                                state_hidden=32))
            self.assertEqual(loaded.step, 3)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            self.assertEqual(optimizer.state, {})
            self.assertEqual(scheduler.last_epoch, 0)
            for name, value in trained.state_dict().items():
                self.assertTrue(torch.equal(value, fresh.state_dict()[name]))
            # A different architecture cannot be warm-started silently.
            wrong = tiny_model(7)
            with self.assertRaises(CheckpointMismatch):
                init_weights_only(
                    path, model=wrong,
                    expected_model_config=MVPConfig(layers=2, ffn=32,
                                                    state_hidden=32))

    def test_legacy_checkpoint_without_model_config_still_loads(self):
        config = TrainingConfig(seed=15)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy.pt"
            self._checkpoint(path, config=config)
            model = tiny_model(3)
            optimizer = make_optimizer(model, config)
            info, _payload = load_checkpoint(
                path, model=model, optimizer=optimizer,
                expected_hashes={"schema_hash": SCHEMA_HASH,
                                 "action_map_hash": ACTION_HASH,
                                 "normalizer_hash": NORMALIZER_HASH})
            self.assertEqual(info.step, 3)
            self.assertIsNone(info.model_config)
            with self.assertRaises(CheckpointMismatch):
                load_checkpoint(path, model=tiny_model(3),
                                expected_model_config=MVPConfig(
                                    layers=1, ffn=32, state_hidden=32))


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

    def test_cli_default_model_is_six_layer_mvp_s(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.rfshard"
            ckpt = Path(temp) / "default.pt"
            build_shard(path, games=1)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = train_main(["--shard", str(path), "--train", "--out",
                                   str(ckpt), "--max-steps", "0"])
            self.assertEqual(code, 0)
            text = out.getvalue()
            self.assertIn("PR14_MODEL", text)
            self.assertIn("layers=6", text)
            self.assertIn(f"parameters={MVP_S_REFERENCE_PARAMETERS}", text)

    def test_cli_tiny_model_is_explicit_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.rfshard"
            ckpt = Path(temp) / "tiny.pt"
            build_shard(path, games=1)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = train_main(["--shard", str(path), "--train", "--out",
                                   str(ckpt), "--max-steps", "0",
                                   "--tiny-smoke"])
            self.assertEqual(code, 0)
            text = out.getvalue()
            self.assertIn("kind=tiny_smoke", text)
            self.assertIn("parameters=502510", text)
            self.assertNotIn(f"parameters={MVP_S_REFERENCE_PARAMETERS}", text)


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
