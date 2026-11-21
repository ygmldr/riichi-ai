#!/usr/bin/env python3
"""PR-38 loader/trainer integration: multi-decision shard batches + CLI.

The fixtures are synthetic ``rf-shard/3`` files built with the existing
``build_decisions`` / ``ShardWriter`` helpers.  No GPU is used.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.decision import build_decisions  # noqa: E402
from riichiformer.data.encode import (  # noqa: E402
    EVENT_TENSOR_FIELDS, StreamMeta, encode_decision,
)
from riichiformer.data.loader import (  # noqa: E402
    MultiShardBatch, ShardBatch, ShardLoadError, ShardReader,
)
from riichiformer.data.shard import (  # noqa: E402
    ACTION_HASH, NORMALIZER_HASH, SCHEMA_HASH, ShardMetadata, ShardRightsError,
    ShardWriter, make_game_record,
)
from riichiformer.models.policy import MVPConfig, Policy, PolicyOutput  # noqa: E402
from riichiformer.train.bc import (  # noqa: E402
    TrainingError, masked_bc_loss, training_step, training_step_multi,
)
from riichiformer.train.optimizer import TrainingConfig, make_optimizer  # noqa: E402
from scripts.rf_train import main as train_main  # noqa: E402

HAND0 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
         "1p", "2p", "3p", "4p"]
HAND1 = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
         "E", "S", "W", "N"]
HAND2 = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
         "7m", "8m", "9m", "1p"]
HAND3 = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
         "1s", "2s", "3s", "4s", "5s"]
CALL_HAND = ["5m", "5m", "1s", "2s", "3s", "4s", "6s", "7s", "8s",
             "E", "S", "W", "N"]


def start_kyoku(*, kyoku=1, oya=0, dora="1p", hands=(HAND0, HAND1, HAND2, HAND3)):
    return {"type": "start_kyoku", "bakaze": "E", "dora_marker": dora,
            "kyoku": kyoku, "honba": 0, "kyotaku": 0, "oya": oya,
            "scores": [25000] * 4,
            "tehais": [list(hand) for hand in hands]}


def four_pov_game() -> list[dict]:
    """Calls, a second draw and a second kyoku (9 decisions over 4 POVs)."""
    hands = (HAND0, CALL_HAND, HAND2, HAND3)
    return [
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        start_kyoku(hands=hands),
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "pon", "actor": 1, "target": 0, "pai": "5m",
         "consumed": ["5m", "5m"]},
        {"type": "dahai", "actor": 1, "pai": "1s", "tsumogiri": False},
        {"type": "tsumo", "actor": 2, "pai": "8p"},
        {"type": "dahai", "actor": 2, "pai": "8p", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "1s"},
        {"type": "dahai", "actor": 3, "pai": "1s", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "2m"},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": True},
        {"type": "end_kyoku"},
        start_kyoku(kyoku=2, oya=1, dora="9p", hands=hands),
        {"type": "tsumo", "actor": 1, "pai": "2s"},
        {"type": "dahai", "actor": 1, "pai": "2s", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "1p"},
        {"type": "dahai", "actor": 2, "pai": "1p", "tsumogiri": True},
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


def build_shard(path: Path, games: int = 2) -> None:
    raw = four_pov_game()
    samples = build_decisions(raw)
    writer = ShardWriter(path, metadata())
    for index in range(games):
        writer.add_game(make_game_record(
            raw, samples, game_id=f"g{index:02d}",
            component_id=f"comp-{index}", component_sha256="d" * 64,
            split="train", source_id="synthetic_allowed", train_allowed=True,
            rights_status="ALLOWED"))
    writer.finalize()


def tiny_policy(seed: int = 17) -> Policy:
    torch.manual_seed(seed)
    model = Policy(MVPConfig(layers=2, ffn=64, state_hidden=64,
                             max_positions=2048))
    model.eval()
    return model


def legacy_decisions(path: Path) -> dict:
    reader = ShardReader(path, purpose="train")
    try:
        return {
            (game_id, pov, decision_index): (decision, target, meta)
            for decision, target, meta, game_id, pov, decision_index
            in reader.iter_decisions()
        }
    finally:
        reader.close()


def read_batches(path: Path, **kwargs):
    reader = ShardReader(path, purpose="train")
    try:
        batches = list(reader.iter_multi_decision_batches(**kwargs))
    finally:
        reader.close()
    return batches


def sqlite_update(path: Path, statement: str, params=()) -> None:
    conn = sqlite3.connect(path)
    conn.execute(statement, params)
    conn.commit()
    conn.close()


class MultiLoaderBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "multi.rfshard"
        build_shard(cls.path, games=2)
        cls.legacy = legacy_decisions(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_targets_grouping_accounting_and_legacy_targets(self):
        batches = read_batches(self.path, batch_size=8)
        self.assertIsInstance(batches[0], MultiShardBatch)
        self.assertEqual(sum(len(batch.targets) for batch in batches), 18)
        legacy_per_pov = Counter((game_id, pov) for (game_id, pov, _index)
                                 in self.legacy)
        seen = Counter()
        for batch in batches:
            self.assertEqual(batch.batch.n_decisions, len(batch.targets))
            self.assertEqual(len(batch.metas), len(batch.targets))
            self.assertEqual(len(batch.lengths), len(batch.targets))
            self.assertEqual(len(batch.game_ids), len(batch.targets))
            self.assertEqual(len(batch.povs), len(batch.targets))
            self.assertEqual(len(batch.decision_indices), len(batch.targets))
            self.assertEqual(len(batch.decision_stream_index), len(batch.targets))
            self.assertEqual(batch.batch.n_decisions,
                             int(batch.batch.marker_padding_mask.sum()))
            self.assertEqual(batch.tokens,
                             batch.event_tokens + batch.marker_tokens)
            self.assertEqual(batch.stream_lengths,
                             tuple(int(value) for value in batch.batch.lengths))
            self.assertEqual(
                [int(value) for value in batch.batch.decision_batch_index],
                list(batch.decision_stream_index))
            # A batch never mixes games; a POV never yields two rows.
            self.assertEqual(len(set(batch.stream_game_ids)), 1)
            self.assertEqual(len(set(zip(batch.stream_game_ids,
                                         batch.stream_povs))),
                             batch.batch.batch_size)
            for row in range(batch.batch.batch_size):
                positions = [index for index, stream in
                             enumerate(batch.decision_stream_index)
                             if stream == row]
                self.assertTrue(positions)
                game_id = batch.stream_game_ids[row]
                pov = batch.stream_povs[row]
                seen[(game_id, pov)] += len(positions)
                indices = [batch.decision_indices[index] for index in positions]
                self.assertEqual(indices, sorted(indices))
                self.assertEqual({batch.game_ids[index] for index in positions},
                                 {game_id})
                self.assertEqual({batch.povs[index] for index in positions},
                                 {pov})
                cutoffs = [int(batch.batch.decision_cutoff_idx[index])
                           for index in positions]
                self.assertEqual(cutoffs, sorted(cutoffs))
                for index in positions:
                    key = (batch.game_ids[index], batch.povs[index],
                           batch.decision_indices[index])
                    legacy_decision, legacy_target, _meta = self.legacy[key]
                    self.assertEqual(batch.targets[index], legacy_target)
                    self.assertEqual(int(batch.batch.decision_cutoff_idx[index]),
                                     legacy_decision.before_event_idx)
        self.assertEqual(seen, legacy_per_pov)
        # POVs with <= batch_size decisions are never split across batches.
        placements = defaultdict(set)
        for number, batch in enumerate(batches):
            for row in range(batch.batch.batch_size):
                placements[(batch.stream_game_ids[row],
                            batch.stream_povs[row])].add(number)
        self.assertTrue(all(len(value) == 1 for value in placements.values()))

    def test_chunking_keeps_order_and_full_stream_prefix(self):
        batches = read_batches(self.path, batch_size=2, max_decisions=2)
        self.assertEqual(sum(len(batch.targets) for batch in batches), 18)
        per_pov = defaultdict(list)
        for number, batch in enumerate(batches):
            for row in range(batch.batch.batch_size):
                positions = [index for index, stream in
                             enumerate(batch.decision_stream_index)
                             if stream == row]
                self.assertLessEqual(len(positions), 2)
                cutoffs = [int(batch.batch.decision_cutoff_idx[index])
                           for index in positions]
                # A chunk's event prefix ends exactly at its last cutoff.
                self.assertEqual(int(batch.batch.lengths[row]), max(cutoffs))
                per_pov[(batch.stream_game_ids[row],
                         batch.stream_povs[row])].append((
                             number,
                             [batch.decision_indices[index] for index in positions],
                             cutoffs))
        for key, chunks in per_pov.items():
            indices = [index for _number, index_list, _cutoffs in chunks
                       for index in index_list]
            self.assertEqual(indices, list(range(len(indices))), key)
            cutoffs = [cutoff for _number, _index, cutoff_list in chunks
                       for cutoff in cutoff_list]
            self.assertEqual(cutoffs, sorted(cutoffs), key)
        # max_decisions lowers the per-stream marker cap, never truncates.
        single = read_batches(self.path, batch_size=8, max_decisions=1)
        self.assertEqual(sum(len(batch.targets) for batch in single), 18)
        for batch in single:
            counts = Counter(batch.decision_stream_index)
            self.assertTrue(all(count == 1 for count in counts.values()))

    def test_event_and_position_caps_are_enforced(self):
        # The longest cutoff in this fixture is 17, so 17/18 are the smallest
        # caps that can hold every single decision.
        capped = read_batches(self.path, batch_size=8, max_events=17,
                              max_positions=18)
        self.assertEqual(sum(len(batch.targets) for batch in capped), 18)
        for batch in capped:
            for row in range(batch.batch.batch_size):
                positions = [index for index, stream in
                             enumerate(batch.decision_stream_index)
                             if stream == row]
                joined = int(batch.batch.lengths[row]) + len(positions)
                self.assertLessEqual(joined, 18)
                self.assertLessEqual(int(batch.batch.lengths[row]), 17)
        # A single decision prefix beyond a cap is a hard error, not truncation.
        with self.assertRaises(ShardLoadError):
            read_batches(self.path, batch_size=8, max_events=16)
        with self.assertRaises(ShardLoadError):
            read_batches(self.path, batch_size=8, max_positions=17)

    def test_multi_pov_rows_do_not_mix_streams(self):
        batch = read_batches(self.path, batch_size=16)[0]
        self.assertEqual(batch.batch.batch_size, 4)
        for row in range(batch.batch.batch_size):
            positions = [index for index, stream in
                         enumerate(batch.decision_stream_index) if stream == row]
            game_id = batch.stream_game_ids[row]
            pov = batch.stream_povs[row]
            for index in positions:
                key = (game_id, pov, batch.decision_indices[index])
                decision, _target, meta = self.legacy[key]
                encoded = encode_decision(decision, StreamMeta.coerce(meta))
                self.assertEqual(int(batch.batch.decision_cutoff_idx[index]),
                                 decision.before_event_idx)
                self.assertTrue(torch.equal(batch.batch.state_features[index],
                                            encoded.state_features))
                self.assertTrue(torch.equal(batch.batch.legal_mask[index],
                                            encoded.legal_mask))
                self.assertEqual(int(batch.batch.decision_phase[index]),
                                 encoded.phase_index)
                cutoff = decision.before_event_idx
                for name in EVENT_TENSOR_FIELDS:
                    if name == "event_phase":
                        continue
                    self.assertTrue(torch.equal(
                        getattr(batch.batch, name)[row, :cutoff],
                        getattr(encoded, name)), f"{name} row={row}")

    def test_legacy_iter_batches_is_unchanged(self):
        reader = ShardReader(self.path, purpose="train")
        try:
            legacy = list(reader.iter_batches(batch_size=4))
        finally:
            reader.close()
        self.assertTrue(legacy)
        self.assertTrue(all(isinstance(batch, ShardBatch) for batch in legacy))
        self.assertEqual(sum(len(batch.targets) for batch in legacy), 18)
        self.assertTrue(all(batch.batch.n_decisions == len(batch.targets)
                            for batch in legacy))
        multi = read_batches(self.path, batch_size=16)
        self.assertEqual(sum(len(batch.targets) for batch in multi), 18)


class MultiForwardTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "multi.rfshard"
        build_shard(cls.path, games=2)
        cls.batch = read_batches(cls.path, batch_size=16)[0]
        cls.legacy = legacy_decisions(cls.path)
        cls.model = tiny_policy()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def _legacy_output(self) -> PolicyOutput:
        logits = []
        masks = []
        with torch.no_grad():
            for index in range(self.batch.batch.n_decisions):
                key = (self.batch.game_ids[index], self.batch.povs[index],
                       self.batch.decision_indices[index])
                decision, _target, meta = self.legacy[key]
                encoded = encode_decision(decision, StreamMeta.coerce(meta))
                logits.append(self.model(encoded.to_model_batch()).logits)
                masks.append(encoded.legal_mask)
        logits_t = torch.cat(logits, dim=0)
        legal = torch.stack(masks, dim=0)
        masked = logits_t.masked_fill(~legal, float("-inf"))
        return PolicyOutput(
            logits=logits_t, masked_logits=masked,
            logprobs=torch.log_softmax(masked, dim=-1), legal_mask=legal)

    def test_forward_and_loss_parity_with_legacy_prefixes(self):
        with torch.no_grad():
            multi = self.model.forward_multi(self.batch.batch)
        legacy = self._legacy_output()
        self.assertEqual(tuple(multi.logits.shape), tuple(legacy.logits.shape))
        logit_diff = float((multi.logits - legacy.logits).abs().max())
        self.assertLessEqual(logit_diff, 1e-5)
        multi_loss = masked_bc_loss(multi, self.batch.targets)
        legacy_loss = masked_bc_loss(legacy, self.batch.targets)
        loss_diff = abs(float(multi_loss.loss) - float(legacy_loss.loss))
        self.assertLessEqual(loss_diff, 1e-5)
        self.assertEqual(multi_loss.valid_decisions,
                         legacy_loss.valid_decisions)
        self.assertTrue(torch.isfinite(multi_loss.loss))
        print(f"PR38_LOADER_PARITY decisions={multi.logits.size(0)} "
              f"logit_max_abs={logit_diff:.3e} loss_abs={loss_diff:.3e} "
              f"loss={float(multi_loss.loss):.6f}")

    def test_training_step_multi_loss_and_gradients(self):
        model = tiny_policy(seed=29)
        optimizer = make_optimizer(model, TrainingConfig(seed=29, lr=1e-3))
        result = training_step_multi(
            model, self.batch.batch, self.batch.targets, optimizer=optimizer)
        self.assertGreater(result.valid_decisions, 0)
        self.assertEqual(result.total_decisions, len(self.batch.targets))
        self.assertTrue(torch.isfinite(result.loss))
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))
        with self.assertRaises(TrainingError):
            training_step_multi(
                model, self.batch.batch, self.batch.targets, optimizer=optimizer,
                accumulation_count=2)

    def test_training_step_prefix_path_still_accepts_model_batch(self):
        model = tiny_policy(seed=31)
        optimizer = make_optimizer(model, TrainingConfig(seed=31))
        prefix_batch = encode_decision(
            self.legacy[(self.batch.game_ids[0], self.batch.povs[0],
                         self.batch.decision_indices[0])][0],
            StreamMeta.coerce(self.legacy[(self.batch.game_ids[0],
                                           self.batch.povs[0],
                                           self.batch.decision_indices[0])][2])
        ).to_model_batch()
        result = training_step(
            model, prefix_batch, [self.batch.targets[0]], optimizer=optimizer)
        self.assertTrue(torch.isfinite(result.loss))

    def test_deep_snapshot_verification_accepts_valid_shard(self):
        batches = read_batches(self.path, batch_size=16,
                               verify_state_snapshots=True)
        self.assertEqual(sum(len(batch.targets) for batch in batches), 18)


class MultiLoaderRejectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.path = self.dir / "base.rfshard"
        build_shard(self.path, games=1)

    def tearDown(self):
        self.temp.cleanup()

    def _copy(self, name="copy.rfshard") -> Path:
        target = self.dir / name
        shutil.copy2(self.path, target)
        shutil.copy2(Path(str(self.path) + ".manifest.json"),
                     Path(str(target) + ".manifest.json"))
        return target

    def _iter(self, path: Path, **kwargs):
        reader = ShardReader(path, purpose="train", verify_file_hash=False)
        try:
            return list(reader.iter_multi_decision_batches(**kwargs))
        finally:
            reader.close()

    def test_tampered_before_event_idx_fails_closed(self):
        for delta in (1, -1):
            path = self._copy(f"tamper{delta}.rfshard")
            conn = sqlite3.connect(path)
            row = conn.execute(
                "SELECT decision_id,before_event_idx FROM inputs "
                "WHERE pov=0 ORDER BY decision_index LIMIT 1").fetchone()
            conn.execute("UPDATE inputs SET before_event_idx=? WHERE decision_id=?",
                         (row[1] + delta, row[0]))
            conn.commit()
            conn.close()
            with self.assertRaises(ShardLoadError):
                self._iter(path)

    def test_tampered_cutoff_fails_in_deep_audit_mode(self):
        path = self._copy("tamper-deep.rfshard")
        conn = sqlite3.connect(path)
        row = conn.execute(
            "SELECT decision_id,before_event_idx FROM inputs "
            "WHERE pov=1 ORDER BY decision_index LIMIT 1").fetchone()
        conn.execute("UPDATE inputs SET before_event_idx=? WHERE decision_id=?",
                     (row[1] + 1, row[0]))
        conn.commit()
        conn.close()
        with self.assertRaises(ShardLoadError):
            self._iter(path, verify_state_snapshots=True)

    def test_missing_target_fails(self):
        path = self._copy("missing-target.rfshard")
        conn = sqlite3.connect(path)
        decision_id = conn.execute(
            "SELECT decision_id FROM inputs ORDER BY decision_index LIMIT 1"
        ).fetchone()[0]
        conn.execute("DELETE FROM targets WHERE decision_id=?", (decision_id,))
        conn.commit()
        conn.close()
        with self.assertRaises(ShardLoadError):
            self._iter(path)

    def test_missing_input_fails(self):
        path = self._copy("missing-input.rfshard")
        conn = sqlite3.connect(path)
        decision_id = conn.execute(
            "SELECT decision_id FROM inputs ORDER BY decision_index LIMIT 1"
        ).fetchone()[0]
        conn.execute("DELETE FROM inputs WHERE decision_id=?", (decision_id,))
        conn.commit()
        conn.close()
        with self.assertRaises(ShardLoadError):
            self._iter(path)

    def test_non_train_split_or_source_is_refused(self):
        for name, statement in (
                ("split", "UPDATE games SET split='val_holdout'"),
                ("source", "UPDATE games SET source_id='tenhou_old'")):
            path = self._copy(f"{name}.rfshard")
            sqlite_update(path, statement)
            with self.assertRaises(ShardRightsError):
                self._iter(path)

    def test_bad_batch_arguments_fail_closed(self):
        for kwargs in ({"batch_size": 0}, {"batch_size": True},
                       {"max_decisions": 0}, {"max_events": 0},
                       {"max_positions": 0}, {"max_events": True}):
            with self.assertRaises(ShardLoadError):
                self._iter(self.path, **kwargs)


class MultiDecisionCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.path = self.dir / "cli.rfshard"
        build_shard(self.path, games=2)

    def tearDown(self):
        self.temp.cleanup()

    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = train_main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_multi_reports_batches_decisions_tokens(self):
        code, text, err = self._run([
            "--shard", str(self.path), "--dry-run", "--multi-decision",
            "--batch-size", "16", "--steps", "1"])
        self.assertEqual(code, 0, err)
        self.assertIn("PR14_MULTI_DRY_RUN batches=1", text)
        self.assertIn("decisions=9", text)
        self.assertIn("tokens=", text)
        self.assertNotIn("PR14_DRY_RUN shard=", text)

    def test_default_dry_run_stays_on_prefix_path(self):
        code, text, err = self._run(["--shard", str(self.path), "--dry-run",
                                     "--steps", "1"])
        self.assertEqual(code, 0, err)
        self.assertIn("PR14_DRY_RUN shard=", text)
        self.assertNotIn("PR14_MULTI_DRY_RUN", text)

    def test_multi_train_checkpoint_and_resume_same_path(self):
        checkpoint = self.dir / "multi.pt"
        code, text, err = self._run([
            "--shard", str(self.path), "--train", "--out", str(checkpoint),
            "--multi-decision", "--tiny-smoke", "--batch-size", "16",
            "--max-steps", "1"])
        self.assertEqual(code, 0, err)
        self.assertIn("PR14_MULTI_MODEL", text)
        self.assertIn("training_path=multi_decisions", text)
        self.assertIn("PR14_MULTI_DONE", text)
        self.assertTrue(checkpoint.exists())
        code, text, err = self._run([
            "--shard", str(self.path), "--train", "--out", str(checkpoint),
            "--multi-decision", "--tiny-smoke", "--batch-size", "16",
            "--max-steps", "2", "--resume", str(checkpoint)])
        self.assertEqual(code, 0, err)

    def test_cross_path_resume_is_refused(self):
        multi = self.dir / "multi.pt"
        code, _text, err = self._run([
            "--shard", str(self.path), "--train", "--out", str(multi),
            "--multi-decision", "--tiny-smoke", "--batch-size", "16",
            "--max-steps", "1"])
        self.assertEqual(code, 0, err)
        code, _text, err = self._run([
            "--shard", str(self.path), "--train", "--out", str(multi),
            "--tiny-smoke", "--batch-size", "16", "--max-steps", "2",
            "--resume", str(multi)])
        self.assertEqual(code, 1)
        self.assertIn("training_path", err)
        prefix = self.dir / "prefix.pt"
        code, _text, err = self._run([
            "--shard", str(self.path), "--train", "--out", str(prefix),
            "--tiny-smoke", "--batch-size", "16", "--max-steps", "1"])
        self.assertEqual(code, 0, err)
        code, _text, err = self._run([
            "--shard", str(self.path), "--train", "--out", str(prefix),
            "--multi-decision", "--tiny-smoke", "--batch-size", "16",
            "--max-steps", "2", "--resume", str(prefix)])
        self.assertEqual(code, 1)
        self.assertIn("training_path", err)

    def test_legacy_checkpoint_without_training_path_is_prefix_only(self):
        prefix = self.dir / "legacy-prefix.pt"
        code, _text, err = self._run([
            "--shard", str(self.path), "--train", "--out", str(prefix),
            "--tiny-smoke", "--batch-size", "16", "--max-steps", "1"])
        self.assertEqual(code, 0, err)
        payload = torch.load(prefix, map_location="cpu", weights_only=False)
        payload["extra"] = {}
        torch.save(payload, prefix)
        digest = hashlib.sha256(prefix.read_bytes()).hexdigest()
        sidecar = Path(str(prefix) + ".json")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        manifest["sha256"] = digest
        sidecar.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")), encoding="utf-8")
        # A legacy (pre-training_path) checkpoint is a prefix checkpoint: it may
        # resume the prefix path, but must not silently become multi.
        code, _text, err = self._run([
            "--shard", str(self.path), "--train", "--out", str(prefix),
            "--tiny-smoke", "--batch-size", "16", "--max-steps", "2",
            "--resume", str(prefix)])
        self.assertEqual(code, 0, err)
        code, _text, err = self._run([
            "--shard", str(self.path), "--train", "--out", str(prefix),
            "--multi-decision", "--tiny-smoke", "--batch-size", "16",
            "--max-steps", "2", "--resume", str(prefix)])
        self.assertEqual(code, 1)
        self.assertIn("training_path", err)

    def test_multi_max_events_above_model_capacity_is_refused(self):
        code, _text, err = self._run([
            "--shard", str(self.path), "--train",
            "--out", str(self.dir / "too-long.pt"), "--multi-decision",
            "--tiny-smoke", "--batch-size", "16", "--max-steps", "1",
            "--multi-max-events", "8192"])
        self.assertEqual(code, 2)
        self.assertIn("exceeds model max_positions", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
