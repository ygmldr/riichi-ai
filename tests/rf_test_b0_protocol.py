#!/usr/bin/env python3
"""PR-48 B0 protocol tests: fingerprints, validation, selection, resume, stops.

CPU only.  Fixtures are tiny synthetic ``rf-shard/3`` files (4 decisions per
game) and a one-layer 64-wide model; no GPU, no real B0 corpus, no full epoch.
"""

from __future__ import annotations

import copy
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

from riichiformer.data.decision import build_decisions  # noqa: E402
from riichiformer.data.loader import ShardReader  # noqa: E402
from riichiformer.data.shard import (  # noqa: E402
    ACTION_HASH, NORMALIZER_HASH, SCHEMA_HASH, ShardMetadata, ShardWriter,
    make_game_record,
)
from riichiformer.models.batch import MultiDecisionBatch  # noqa: E402
from riichiformer.models.policy import MVPConfig, Policy  # noqa: E402
from riichiformer.train.b0 import (  # noqa: E402
    B0_TRAINING_PATH, B0Candidate, B0Protocol, B0ProtocolError, B0Refusal,
    B0SelectionRule, B0StopMonitor, B0StopPolicy, B0StopTriggered,
    B0TrainSettings, ModelEMA, ValidationMetrics, build_protocol_from_files,
    checkpoint_provenance, evaluate_candidate, evaluate_multi_batches,
    not_run_items, resume_checkpoint, run_accumulated_step, save_candidate,
    verify_data_fingerprints, verify_model_fingerprint, write_manifest,
)
from riichiformer.train.bc import (  # noqa: E402
    masked_bc_loss, window_weight_total,
)
from riichiformer.train.checkpoint import (  # noqa: E402
    CheckpointMismatch, config_hash, save_checkpoint,
)
from riichiformer.train.optimizer import (  # noqa: E402
    make_optimizer, make_scheduler, set_seed,
)

EVENT_PAD_FILLS = {"event_type_id": 16, "event_actor_rel": 5,
                   "event_tile_kind": 34, "event_aka": 0,
                   "event_tsumogiri": 0, "event_phase": 3,
                   "event_kyoku_pos": 0}

HAND0 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
         "1p", "2p", "3p", "4p"]
HAND1 = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
         "E", "S", "W", "N"]
HAND2 = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
         "7m", "8m", "9m", "1p"]
HAND3 = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
         "1s", "2s", "3s", "4s", "5s"]


def raw_game() -> list[dict]:
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
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]


def metadata() -> ShardMetadata:
    return ShardMetadata(
        purpose="train", source_id="synthetic_allowed",
        rights_status="ALLOWED", train_allowed=True,
        canonical_manifest_sha256="a" * 64,
        canonical_index_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
        split_version="rf-split/2",
        schema_version="rf-schema-v3/2", schema_hash=SCHEMA_HASH,
        action_map_hash=ACTION_HASH, normalizer_hash=NORMALIZER_HASH)


def validation_metadata() -> ShardMetadata:
    return ShardMetadata(
        purpose="diagnostic", source_id="synthetic_validation",
        rights_status="UNKNOWN", train_allowed=False,
        canonical_manifest_sha256="a" * 64,
        canonical_index_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
        split_version="rf-split/2",
        schema_version="rf-schema-v3/2", schema_hash=SCHEMA_HASH,
        action_map_hash=ACTION_HASH, normalizer_hash=NORMALIZER_HASH,
        research_policy="licensed_only",
        research_policy_hash="",
        train_eligible_under_policy=False)


def build_shard(path: Path, games: int, prefix: str, *, validation: bool = False) -> None:
    raw = raw_game()
    samples = build_decisions(raw)
    meta = validation_metadata() if validation else metadata()
    split = "validation" if validation else "train"
    allowed = False if validation else True
    writer = ShardWriter(path, meta)
    for index in range(games):
        writer.add_game(make_game_record(
            raw, samples, game_id=f"{prefix}{index:03d}",
            component_id=f"{prefix}-comp-{index}",
            component_sha256="d" * 64, split=split,
            source_id=meta.source_id, train_allowed=allowed,
            rights_status="UNKNOWN" if validation else "ALLOWED"))
    writer.finalize()


def tiny_config(name: str = "b0_test") -> MVPConfig:
    return MVPConfig(name=name, hidden=64, layers=1, heads=4, ffn=32,
                     state_hidden=32, max_positions=2048)


def settings(**overrides) -> B0TrainSettings:
    values = dict(
        lr=1e-3, weight_decay=0.01, beta1=0.9, beta2=0.999, warmup_steps=0,
        grad_clip=1.0, grad_accumulation=1, batch_size=2, max_steps=6,
        bf16=False, ema_decay=0.9, training_path=B0_TRAINING_PATH,
        device="cpu")
    values.update(overrides)
    return B0TrainSettings(**values)


def make_protocol(tmp: Path, *, max_steps: int = 6,
                  validation_steps=(2, 4, 6), candidate_steps=(4, 6),
                  seeds=(11, 22), grad_accumulation: int = 1,
                  ema_decay: float | None = 0.9,
                  selection: B0SelectionRule | None = None,
                  train_games: int = 24,
                  val_games: int = 8) -> B0Protocol:
    train_shard = tmp / "train.rfshard"
    val_shard = tmp / "val.rfshard"
    build_shard(train_shard, train_games, "g")
    build_shard(val_shard, val_games, "v", validation=True)
    return build_protocol_from_files(
        experiment_id="EXP-B0-TEST",
        train_shard=str(train_shard), validation_shard=str(val_shard),
        model_config=tiny_config(),
        train=settings(max_steps=max_steps, grad_accumulation=grad_accumulation,
                       ema_decay=ema_decay),
        seeds=seeds, validation_steps=validation_steps,
        candidate_steps=candidate_steps,
        selection=selection if selection is not None else B0SelectionRule(),
        stop_policy=B0StopPolicy(max_wall_seconds=120.0),
        created_utc="2026-11-21T00:00:00Z")


def batches_of(protocol: B0Protocol, *, batch_size: int = 2):
    reader = ShardReader(protocol.train_shard, purpose="train")
    try:
        return reader, list(reader.iter_multi_decision_batches(
            batch_size=batch_size, max_events=2048, max_positions=2048))
    finally:
        reader.close()


def validation_batches(protocol: B0Protocol, *, batch_size: int = 2):
    reader = ShardReader(protocol.validation_shard, purpose="audit")
    try:
        return list(reader.iter_multi_decision_batches(
            batch_size=batch_size, max_events=2048, max_positions=2048))
    finally:
        reader.close()


def concat_multi(batches) -> MultiDecisionBatch:
    """Concatenate multi-decision batches along the stream-row axis."""
    max_len = max(batch.seq_len for batch in batches)
    rows = sum(batch.batch_size for batch in batches)
    tensors = {}
    for name, fill in EVENT_PAD_FILLS.items():
        parts = []
        for batch in batches:
            padded = torch.full((batch.batch_size, max_len), fill, dtype=torch.long)
            padded[:, :batch.seq_len] = getattr(batch, name)
            parts.append(padded)
        tensors[name] = torch.cat(parts, dim=0)
    padding_mask = torch.zeros((rows, max_len), dtype=torch.bool)
    state_features = []
    decision_batch_index = []
    decision_marker_index = []
    decision_cutoff_idx = []
    decision_phase = []
    legal_mask = []
    row_offset = 0
    for batch in batches:
        padding_mask[row_offset:row_offset + batch.batch_size, :batch.seq_len] = (
            batch.padding_mask)
        state_features.append(batch.state_features)
        decision_batch_index.append(batch.decision_batch_index + row_offset)
        decision_marker_index.append(batch.decision_marker_index)
        decision_cutoff_idx.append(batch.decision_cutoff_idx)
        decision_phase.append(batch.decision_phase)
        legal_mask.append(batch.legal_mask)
        row_offset += batch.batch_size
    return MultiDecisionBatch(
        **tensors, padding_mask=padding_mask,
        state_features=torch.cat(state_features, dim=0),
        decision_batch_index=torch.cat(decision_batch_index, dim=0),
        decision_marker_index=torch.cat(decision_marker_index, dim=0),
        decision_cutoff_idx=torch.cat(decision_cutoff_idx, dim=0),
        decision_phase=torch.cat(decision_phase, dim=0),
        legal_mask=torch.cat(legal_mask, dim=0))


def metrics(step: int, ce: float, *, top1: float = 0.5, top3: float = 0.8,
            decisions: int = 8, supervised: int = 8) -> ValidationMetrics:
    return ValidationMetrics(
        step=step, ce=ce, top1=top1, top3=top3,
        legal_rate=1.0, nonfinite_rate=0.0, decisions=decisions,
        supervised=supervised, legal=supervised, nonfinite=0,
        tokens=decisions * 4, weight=float(supervised))


def candidate(seed: int, step: int, *, raw_ce: float, ema_ce: float,
              experiment: str = "EXP-B0-TEST") -> B0Candidate:
    return B0Candidate(
        experiment_id=experiment, seed=seed, step=step,
        checkpoint_path=f"/tmp/{experiment}/seed{seed}/step{step:06d}.pt",
        checkpoint_sha256=f"{seed:02d}".rjust(64, "0"),
        raw=metrics(step, raw_ce), ema=metrics(step, ema_ce),
        tokens=step * 8, decisions=step * 8)


def state_bytes(value) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


class ProtocolValidationTests(unittest.TestCase):
    def test_training_path_is_explicit_and_multi_only(self):
        with self.assertRaises(B0ProtocolError):
            settings(training_path="prefixes")
        with self.assertRaises(TypeError):
            B0TrainSettings(
                lr=1e-3, weight_decay=0.01, beta1=0.9, beta2=0.999,
                warmup_steps=0, grad_clip=1.0, grad_accumulation=1,
                batch_size=2, max_steps=6, bf16=False, ema_decay=None)
        self.assertEqual(settings().training_path, B0_TRAINING_PATH)

    def test_exactly_two_distinct_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol = make_protocol(Path(temp))
            with self.assertRaises(B0ProtocolError):
                dataclasses.replace(protocol, seeds=(11,))
            with self.assertRaises(B0ProtocolError):
                dataclasses.replace(protocol, seeds=(11, 11))
            with self.assertRaises(B0ProtocolError):
                dataclasses.replace(protocol, seeds=(11, 22, 33))

    def test_step_schedule_invariants(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol = make_protocol(Path(temp))
            with self.assertRaises(B0ProtocolError):
                dataclasses.replace(protocol, candidate_steps=(4,))
            with self.assertRaises(B0ProtocolError):
                dataclasses.replace(protocol, validation_steps=(2, 4))
            with self.assertRaises(B0ProtocolError):
                dataclasses.replace(protocol, validation_steps=(4, 2, 6))
            with self.assertRaises(B0ProtocolError):
                dataclasses.replace(protocol, candidate_steps=(4, 8))

    def test_fingerprints_cover_config_and_model(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol = make_protocol(Path(temp))
            base = protocol.fingerprint()
            config = protocol.config_fingerprint()
            self.assertEqual(base, protocol.fingerprint())
            changed_lr = dataclasses.replace(
                protocol, train=dataclasses.replace(protocol.train, lr=2e-3))
            self.assertNotEqual(base, changed_lr.fingerprint())
            self.assertNotEqual(config, changed_lr.config_fingerprint())
            new_model = dataclasses.asdict(tiny_config(name="b0_test2"))
            changed_model = dataclasses.replace(
                protocol, model_config=new_model,
                model_config_hash=config_hash(new_model))
            self.assertNotEqual(base, changed_model.fingerprint())
            self.assertNotEqual(config, changed_model.config_fingerprint())
            changed_seed = dataclasses.replace(protocol, seeds=(11, 33))
            self.assertNotEqual(base, changed_seed.fingerprint())
            self.assertEqual(config, changed_seed.config_fingerprint())
            cosmetic = dataclasses.replace(protocol, notes="reviewed",
                                           created_utc="2030-01-01T00:00:00Z")
            self.assertEqual(base, cosmetic.fingerprint())

    def test_json_roundtrip_and_markdown(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol = make_protocol(Path(temp))
            path = Path(temp) / "prereg.json"
            protocol.write_json(path)
            loaded = B0Protocol.load_json(path)
            self.assertEqual(protocol.fingerprint(), loaded.fingerprint())
            self.assertEqual(protocol.as_dict(), loaded.as_dict())
            markdown = protocol.prereg_markdown(base_sha="e760226")
            self.assertIn("EXP-B0-TEST", markdown)
            self.assertIn("multi_decisions", markdown)
            self.assertIn(protocol.train_shard_sha256, markdown)
            self.assertIn("validation_ce", markdown)
            self.assertIn("NOT_RUN", markdown)
            self.assertIn("window-weight", markdown)
            path_md = Path(temp) / "PREREG.md"
            protocol.write_prereg_markdown(path_md, base_sha="e760226")
            self.assertEqual(path_md.read_text(encoding="utf-8").strip(),
                             markdown.strip())

    def test_unknown_prereg_keys_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol = make_protocol(Path(temp))
            document = protocol.as_dict()
            document["extra_key"] = 1
            with self.assertRaises(B0ProtocolError):
                B0Protocol.from_dict(document)


class DataFingerprintTests(unittest.TestCase):
    def test_validation_shard_must_not_be_train_purpose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.rfshard"
            val = root / "val_trainpurpose.rfshard"
            build_shard(train, 8, "g")
            build_shard(val, 4, "v")  # deliberately train-purpose/split
            protocol = build_protocol_from_files(
                experiment_id="EXP-B0-TEST",
                train_shard=str(train), validation_shard=str(val),
                model_config=tiny_config(), train=settings(max_steps=4),
                seeds=(11, 22), validation_steps=(2, 4),
                candidate_steps=(4,), selection=B0SelectionRule(),
                stop_policy=B0StopPolicy(max_wall_seconds=120.0),
                created_utc="2026-11-21T00:00:00Z")
            with self.assertRaises(B0Refusal):
                verify_data_fingerprints(protocol)

    def test_wrong_shard_and_manifest_hashes_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol = make_protocol(Path(temp))
            self.assertIn("train", verify_data_fingerprints(protocol))
            with self.assertRaises(B0Refusal):
                verify_data_fingerprints(dataclasses.replace(
                    protocol, train_shard_sha256="0" * 64))
            with self.assertRaises(B0Refusal):
                verify_data_fingerprints(dataclasses.replace(
                    protocol, train_manifest_sha256="1" * 64))
            with self.assertRaises(B0Refusal):
                verify_data_fingerprints(dataclasses.replace(
                    protocol, validation_shard_sha256="2" * 64))

    def test_parameter_count_and_config_hash_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol = make_protocol(Path(temp))
            self.assertEqual(verify_model_fingerprint(protocol),
                             protocol.parameter_count)
            with self.assertRaises(B0Refusal):
                verify_model_fingerprint(dataclasses.replace(
                    protocol, parameter_count=protocol.parameter_count + 1))
            with self.assertRaises(B0ProtocolError):
                dataclasses.replace(protocol, model_config_hash="3" * 64)


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.protocol = make_protocol(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_metrics_are_finite_and_structural(self):
        set_seed(3)
        model = Policy(tiny_config())
        result = evaluate_multi_batches(
            model, validation_batches(self.protocol), step=2)
        self.assertGreater(result.decisions, 0)
        self.assertGreater(result.supervised, 0)
        self.assertEqual(result.legal, result.supervised)
        self.assertEqual(result.nonfinite, 0)
        self.assertEqual(result.legal_rate, 1.0)
        self.assertEqual(result.nonfinite_rate, 0.0)
        self.assertTrue(math.isfinite(result.ce))
        self.assertGreaterEqual(result.top1, 0.0)
        self.assertLessEqual(result.top1, 1.0)
        self.assertGreaterEqual(result.top3, result.top1 - 1e-12)
        self.assertGreater(result.tokens, 0)
        self.assertTrue(model.training)

    def test_validation_does_not_update_optimizer_or_ema(self):
        set_seed(4)
        model = Policy(tiny_config())
        config = self.protocol.train.training_config(self.protocol.seeds[0])
        optimizer = make_optimizer(model, config)
        scheduler = make_scheduler(optimizer, config)
        ema = ModelEMA(model, 0.9)
        reader, train_batches = batches_of(self.protocol)
        batch = train_batches[0]
        result = run_accumulated_step(
            model, [batch.batch], [batch.targets], optimizer=optimizer,
            scheduler=scheduler, settings=self.protocol.train, ema=ema,
            tokens=batch.tokens)
        self.assertTrue(result.optimizer_stepped)
        self.assertEqual(ema.updates, 1)
        before_model = {name: value.clone()
                        for name, value in model.state_dict().items()}
        before_optimizer = state_bytes(optimizer.state_dict())
        before_ema = state_bytes(ema.state_dict())
        evaluation = evaluate_candidate(
            model, lambda: validation_batches(self.protocol), step=2, ema=ema)
        self.assertEqual(evaluation.raw.decisions,
                         evaluation.ema.decisions)
        self.assertTrue(model.training)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before_model[name]), name)
        self.assertEqual(before_optimizer, state_bytes(optimizer.state_dict()))
        self.assertEqual(before_ema, state_bytes(ema.state_dict()))
        self.assertEqual(ema.updates, 1)


class SelectionTests(unittest.TestCase):
    def test_lower_ce_wins_over_step_and_seed(self):
        rule = B0SelectionRule()
        candidates = [
            candidate(11, 4, raw_ce=1.30, ema_ce=1.28),
            candidate(22, 6, raw_ce=1.20, ema_ce=1.25),
        ]
        selection = rule.select(candidates)
        self.assertEqual(selection.candidate_id, candidates[1].candidate_id)
        self.assertEqual(selection.seed, 22)
        self.assertEqual(selection.step, 6)
        self.assertEqual(selection.tie_break_used, ())

    def test_tie_break_prefers_earlier_step_then_lower_seed(self):
        rule = B0SelectionRule()
        later = candidate(11, 6, raw_ce=1.0, ema_ce=1.0)
        earlier = candidate(11, 4, raw_ce=1.0, ema_ce=1.0)
        selection = rule.select([later, earlier])
        self.assertEqual(selection.candidate_id, earlier.candidate_id)
        self.assertEqual(selection.tie_break_used,
                         ("other_weights_ce", "earlier_step"))
        same_step_high = candidate(22, 4, raw_ce=1.0, ema_ce=1.0)
        selection = rule.select([same_step_high, earlier])
        self.assertEqual(selection.candidate_id, earlier.candidate_id)
        self.assertEqual(selection.tie_break_used,
                         ("other_weights_ce", "earlier_step", "lower_seed"))

    def test_ema_source_is_frozen_and_recorded(self):
        rule = B0SelectionRule(weights="ema")
        raw_best = candidate(11, 4, raw_ce=1.00, ema_ce=1.30)
        ema_best = candidate(22, 6, raw_ce=1.20, ema_ce=1.10)
        selection = rule.select([raw_best, ema_best])
        self.assertEqual(selection.weights, "ema")
        self.assertEqual(selection.seed, 22)
        self.assertAlmostEqual(selection.value, 1.10)
        self.assertAlmostEqual(selection.other_value, 1.20)
        self.assertEqual(selection.as_dict()["rule"]["weights"], "ema")

    def test_missing_ema_metrics_fail_closed(self):
        rule = B0SelectionRule(weights="ema")
        broken = dataclasses.replace(
            candidate(11, 4, raw_ce=1.0, ema_ce=1.0), ema=None)
        with self.assertRaises(B0ProtocolError):
            rule.select([broken])

    def test_rule_is_frozen(self):
        with self.assertRaises(B0ProtocolError):
            B0SelectionRule(weights="raw", metric="loss")
        with self.assertRaises(B0ProtocolError):
            B0SelectionRule(tie_break=("earlier_step",))
        with self.assertRaises(B0ProtocolError):
            B0SelectionRule(weights="average")


class AccumulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.protocol = make_protocol(
            Path(cls.temp.name), grad_accumulation=2,
            validation_steps=(2, 4, 6), candidate_steps=(4, 6))
        reader, batches = batches_of(cls.protocol)
        cls.batches = batches[:2]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _micro_settings(self) -> B0TrainSettings:
        return dataclasses.replace(self.protocol.train, grad_accumulation=2)

    def test_accumulated_step_matches_one_large_batch(self):
        micro_settings = self._micro_settings()
        window_total = window_weight_total(
            [batch.targets for batch in self.batches])
        set_seed(77)
        micro_model = Policy(tiny_config())
        micro_optimizer = make_optimizer(
            micro_model, micro_settings.training_config(11))
        result = run_accumulated_step(
            micro_model, [batch.batch for batch in self.batches],
            [batch.targets for batch in self.batches],
            optimizer=micro_optimizer, settings=micro_settings,
            tokens=sum(batch.tokens for batch in self.batches))
        self.assertTrue(result.optimizer_stepped)
        self.assertAlmostEqual(result.weight_total, window_total, places=9)

        big_batch = concat_multi([batch.batch for batch in self.batches])
        big_targets = [target for batch in self.batches
                       for target in batch.targets]
        set_seed(77)
        big_model = Policy(tiny_config())
        big_optimizer = make_optimizer(
            big_model, micro_settings.training_config(11))
        big_result = run_accumulated_step(
            big_model, [big_batch], [big_targets], optimizer=big_optimizer,
            settings=dataclasses.replace(micro_settings, grad_accumulation=1),
            accumulation_weight_total=window_total,
            tokens=sum(batch.tokens for batch in self.batches))
        self.assertAlmostEqual(result.loss, big_result.loss, places=6)
        # AdamW amplifies gradient-sum round-off; the gradient assertion below
        # is the tight one (same standard as the PR-39 accumulation tests).
        worst = 0.0
        for (name, left), (_, right) in zip(micro_model.state_dict().items(),
                                            big_model.state_dict().items()):
            worst = max(worst, float((left - right).abs().max()))
        self.assertLess(worst, 1e-4, f"parameter mismatch {worst:.3e}")

        # Gradient sums: sum(micro numerators)/window_total == big numerator/...
        micro_model.zero_grad(set_to_none=True)
        micro_params = [p for p in micro_model.parameters() if p.requires_grad]
        accumulated = None
        for batch in self.batches:
            micro_model.zero_grad(set_to_none=True)
            loss = masked_bc_loss(micro_model.forward_multi(batch.batch), batch.targets)
            grads = torch.autograd.grad(loss.numerator / window_total,
                                        micro_params, allow_unused=True)
            accumulated = grads if accumulated is None else [
                (None if left is None or right is None else left + right)
                for left, right in zip(accumulated, grads)]
        big_model.zero_grad(set_to_none=True)
        big_params = [p for p in big_model.parameters() if p.requires_grad]
        big_loss = masked_bc_loss(big_model.forward_multi(big_batch), big_targets)
        big_grads = torch.autograd.grad(big_loss.numerator / window_total,
                                        big_params, allow_unused=True)
        worst_grad = 0.0
        for left, right in zip(accumulated, big_grads):
            if left is None or right is None:
                continue
            worst_grad = max(worst_grad, float((left - right).abs().max()))
        self.assertLess(worst_grad, 1e-5,
                        f"gradient sum mismatch {worst_grad:.3e}")
        print(f"PR48_ACCUM max_grad_diff={worst_grad:.3e} "
              f"max_param_diff={worst:.3e} window_total={window_total:.3f}")

    def test_all_censored_window_never_steps(self):
        micro_settings = self._micro_settings()
        set_seed(78)
        model = Policy(tiny_config())
        optimizer = make_optimizer(model, micro_settings.training_config(11))
        before = {name: value.clone()
                  for name, value in model.state_dict().items()}
        targets = []
        for batch in self.batches:
            rows = []
            for item in batch.targets:
                row = dict(item)
                row["supervision_valid"] = False
                row["censor_reason"] = "preempted_by_call"
                rows.append(row)
            targets.append(rows)
        result = run_accumulated_step(
            model, [batch.batch for batch in self.batches], targets,
            optimizer=optimizer, settings=micro_settings)
        self.assertFalse(result.optimizer_stepped)
        self.assertEqual(result.valid_decisions, 0)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.protocol = make_protocol(cls.root)
        reader, batches = batches_of(cls.protocol)
        cls.metadata = reader.metadata
        cls.seed = cls.protocol.seeds[0]
        set_seed(9)
        cls.model = Policy(tiny_config())
        cls.config = cls.protocol.train.training_config(cls.seed)
        cls.optimizer = make_optimizer(cls.model, cls.config)
        cls.scheduler = make_scheduler(cls.optimizer, cls.config)
        cls.ema = ModelEMA(cls.model, 0.9)
        batch = batches[0]
        run_accumulated_step(
            cls.model, [batch.batch], [batch.targets],
            optimizer=cls.optimizer, scheduler=cls.scheduler,
            settings=cls.protocol.train, ema=cls.ema)
        validation = validation_batches(cls.protocol)
        evaluation = evaluate_candidate(
            cls.model, lambda: validation, step=4, ema=cls.ema)
        cls.candidate = save_candidate(
            cls.root / "candidate.pt", protocol=cls.protocol,
            metadata=cls.metadata, model=cls.model, optimizer=cls.optimizer,
            scheduler=cls.scheduler, ema=cls.ema, seed=cls.seed, step=4,
            tokens=12, decisions=4, evaluation=evaluation)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _target_model(self):
        model = Policy(tiny_config())
        optimizer = make_optimizer(model, self.config)
        return model, optimizer

    def test_candidate_checkpoint_carries_metrics_and_provenance(self):
        self.assertEqual(self.candidate.raw.step, 4)
        manifest = json.loads(Path(
            str(self.candidate.checkpoint_path) + ".json").read_text())
        self.assertEqual(manifest["hashes"]["shard_sha256"],
                         self.protocol.train_shard_sha256)
        self.assertTrue(manifest["has_ema"])
        self.assertEqual(self.candidate.checkpoint_sha256, manifest["sha256"])

    def test_strict_resume_restores_state(self):
        model, optimizer = self._target_model()
        scheduler = make_scheduler(optimizer, self.config)
        ema = ModelEMA(model, 0.9)
        info, payload = resume_checkpoint(
            self.candidate.checkpoint_path, protocol=self.protocol,
            metadata=self.metadata, seed=self.seed, model=model,
            optimizer=optimizer, scheduler=scheduler, ema=ema)
        self.assertEqual(info.step, 4)
        self.assertEqual(ema.updates, self.ema.updates)
        self.assertTrue(torch.equal(
            ema.shadow["policy_head.linear.weight"],
            self.ema.shadow["policy_head.linear.weight"]))
        self.assertEqual(payload["extra"]["training_path"], B0_TRAINING_PATH)

    def test_legacy_checkpoint_without_training_path_is_refused(self):
        model, optimizer = self._target_model()
        legacy = self.root / "legacy.pt"
        save_checkpoint(
            legacy, model=model, optimizer=optimizer,
            counters={"step": 1, "tokens": 0, "decisions": 0},
            hashes={"schema_hash": SCHEMA_HASH, "action_map_hash": ACTION_HASH,
                    "normalizer_hash": NORMALIZER_HASH},
            config=self.config, seed=self.seed)
        with self.assertRaises(CheckpointMismatch):
            resume_checkpoint(
                legacy, protocol=self.protocol, metadata=self.metadata,
                seed=self.seed, model=model, optimizer=optimizer)

    def _save_b0_with(self, path: Path, *, provenance=None, config=None,
                      extra=None):
        model, optimizer = self._target_model()
        save_checkpoint(
            path, model=model, optimizer=optimizer,
            counters={"step": 2, "tokens": 0, "decisions": 0},
            hashes={"schema_hash": SCHEMA_HASH, "action_map_hash": ACTION_HASH,
                    "normalizer_hash": NORMALIZER_HASH},
            config=config if config is not None else self.config,
            seed=self.seed, model_config=self.protocol.model_config,
            provenance=(provenance if provenance is not None
                        else checkpoint_provenance(self.protocol, self.metadata)),
            extra=extra if extra is not None else {
                "training_path": B0_TRAINING_PATH, "shard_sha256":
                self.protocol.train_shard_sha256})
        return model, optimizer

    def test_cross_path_checkpoint_is_refused(self):
        path = self.root / "prefix.pt"
        provenance = checkpoint_provenance(self.protocol, self.metadata)
        provenance["training_path"] = "prefixes"
        model, optimizer = self._save_b0_with(
            path, provenance=provenance,
            extra={"training_path": "prefixes"})
        with self.assertRaises(CheckpointMismatch):
            resume_checkpoint(
                path, protocol=self.protocol, metadata=self.metadata,
                seed=self.seed, model=model, optimizer=optimizer)

    def test_config_mismatch_is_refused(self):
        path = self.root / "config.pt"
        other_config = dataclasses.replace(self.config, lr=9e-4)
        model, optimizer = self._save_b0_with(path, config=other_config)
        with self.assertRaises(CheckpointMismatch):
            resume_checkpoint(
                path, protocol=self.protocol, metadata=self.metadata,
                seed=self.seed, model=model, optimizer=optimizer)

    def test_shard_mismatch_is_refused(self):
        path = self.root / "shard.pt"
        provenance = checkpoint_provenance(self.protocol, self.metadata)
        provenance["shard_sha256"] = "9" * 64
        model, optimizer = self._save_b0_with(path, provenance=provenance)
        with self.assertRaises(CheckpointMismatch):
            resume_checkpoint(
                path, protocol=self.protocol, metadata=self.metadata,
                seed=self.seed, model=model, optimizer=optimizer)

    def test_other_seed_is_refused(self):
        model, optimizer = self._target_model()
        with self.assertRaises(CheckpointMismatch):
            resume_checkpoint(
                self.candidate.checkpoint_path, protocol=self.protocol,
                metadata=self.metadata, seed=self.protocol.seeds[1],
                model=model, optimizer=optimizer)


class StopAndManifestTests(unittest.TestCase):
    def test_nonfinite_and_illegal_budgets(self):
        policy = B0StopPolicy(max_wall_seconds=10.0, max_nonfinite_events=0,
                              max_illegal_events=1)
        monitor = B0StopMonitor(policy, clock=lambda: 0.0)
        with self.assertRaises(B0StopTriggered) as caught:
            monitor.record_nonfinite(3, 1)
        self.assertEqual(caught.exception.event.code, "nonfinite")
        monitor.record_illegal(4, 1)
        with self.assertRaises(B0StopTriggered):
            monitor.record_illegal(5, 1)

    def test_wall_and_regression_budgets(self):
        now = [0.0]
        policy = B0StopPolicy(
            max_wall_seconds=5.0, validation_regression_patience=0,
            validation_regression_delta=0.01)
        monitor = B0StopMonitor(policy, clock=lambda: now[0])
        now[0] = 6.0
        with self.assertRaises(B0StopTriggered) as caught:
            monitor.check_wall(1)
        self.assertEqual(caught.exception.event.code, "wall_budget")
        monitor = B0StopMonitor(policy, clock=lambda: 0.0)
        monitor.observe_validation(metrics(2, 1.0), step=2)
        with self.assertRaises(B0StopTriggered) as caught:
            monitor.observe_validation(metrics(4, 1.5), step=4)
        self.assertEqual(caught.exception.event.code, "validation_regression")
        self.assertEqual(monitor.best_ce, 1.0)
        self.assertEqual(monitor.state_dict()["regression_count"], 1)

    def test_manifest_lists_not_run_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = make_protocol(root)
            prereg = root / "prereg.json"
            protocol.write_json(prereg)
            selection = B0SelectionRule().select([
                candidate(11, 4, raw_ce=1.1, ema_ce=1.2),
                candidate(22, 6, raw_ce=1.0, ema_ce=1.1),
            ], complete=False)
            path = root / "manifest.json"
            manifest = write_manifest(
                path, protocol=protocol, selection=selection,
                status="partial",
                runs=[{"seed": 11, "steps": 4, "device": "cpu"}],
                prereg_path=prereg)
            self.assertEqual(manifest["format"], "rf-b0-manifest/1")
            self.assertEqual(manifest["prereg"]["fingerprint"],
                             protocol.fingerprint())
            self.assertEqual(manifest["strength_claim"], "none")
            for item in ("full_b0_epoch", "arena_select", "arena_confirm",
                         "strength_vs_m5d_r0", "gpu_training"):
                self.assertIn(item, manifest["not_run"])
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text())["not_run"],
                             manifest["not_run"])

    def test_completed_full_budget_drops_full_epoch_not_run(self):
        with tempfile.TemporaryDirectory() as temp:
            protocol = make_protocol(Path(temp))
            items = not_run_items(
                protocol, [{"steps": protocol.train.max_steps,
                            "device": "cuda"}])
            self.assertNotIn("full_b0_epoch", items)
            self.assertNotIn("gpu_training", items)
            self.assertIn("arena_select", items)


if __name__ == "__main__":
    unittest.main(verbosity=2)
