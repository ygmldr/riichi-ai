#!/usr/bin/env python3
"""PR-11 offline/online encoder byte parity and contract tests."""

from __future__ import annotations

import dataclasses
import gzip
import json
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.decision import build_decisions  # noqa: E402
from riichiformer.data.encode import (  # noqa: E402
    EncodedDecision,
    EncoderError,
    StreamMeta,
    encode_batch,
    encode_decision,
    encoded_hash,
    encoded_payload,
)
from riichiformer.models.batch import ModelBatch  # noqa: E402
from riichiformer.models.state_encoder import STATE_FEATURE_DIM  # noqa: E402
from riichiformer.runtime.encoder import OnlineEncoder  # noqa: E402
from riichiformer.schemas.action_v3 import ACTION_SPACE  # noqa: E402


HAND0 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
         "1p", "2p", "3p", "4p"]
HAND1 = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
         "E", "S", "W", "N"]
HAND2 = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
         "7m", "8m", "9m", "1p"]
HAND3 = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
         "1s", "2s", "3s", "4s", "5s"]


def raw_long() -> list[dict]:
    return [
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1p",
         "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
         "scores": [25000] * 4,
         "tehais": [HAND0, HAND1, HAND2, HAND3]},
        {"type": "tsumo", "actor": 0, "pai": "5mr"},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        {"type": "end_kyoku"}, {"type": "end_game"},
    ]


def synthetic_samples():
    return build_decisions(raw_long())


def dataset_dir() -> Path | None:
    for root in (REPO_ROOT / "datasets", REPO_ROOT.parent.parent / "datasets"):
        directory = root / "extracted" / "2025"
        if directory.is_dir():
            return directory
    return None


def read_log(path: Path) -> list[dict]:
    data = path.read_bytes()
    if data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def assert_encoded_equal(test: unittest.TestCase, left: EncodedDecision,
                         right: EncodedDecision) -> None:
    test.assertEqual(left.meta, right.meta)
    test.assertEqual(left.decision_time_index, right.decision_time_index)
    test.assertEqual(left.phase_index, right.phase_index)
    for field in ("event_type_id", "event_actor_rel", "event_tile_kind", "event_aka",
                  "event_tsumogiri", "event_phase", "event_kyoku_pos",
                  "padding_mask", "state_features", "legal_mask"):
        test.assertTrue(torch.equal(getattr(left, field), getattr(right, field)), field)
    test.assertEqual(encoded_hash(left), encoded_hash(right))


class SyntheticParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = synthetic_samples()

    def test_offline_online_byte_equal(self):
        sample = self.samples[0]
        meta = StreamMeta.from_header(sample.header)
        offline = encode_decision(sample.input, meta)
        online = OnlineEncoder().encode(sample.input, sample.header)
        assert_encoded_equal(self, offline, online)
        self.assertEqual(encoded_payload(offline), encoded_payload(online))

    def test_batch_padding_matches_single_rows(self):
        sample0, sample1 = self.samples[0], self.samples[1]
        meta = StreamMeta.from_header(sample0.header)
        batch = encode_batch([sample0.input, sample1.input], [meta, meta])
        self.assertEqual(batch.event_type_id.shape, (2, sample1.input.before_event_idx))
        self.assertEqual(batch.padding_mask.tolist(),
                         [[True] * sample0.input.before_event_idx
                          + [False] * (sample1.input.before_event_idx
                                       - sample0.input.before_event_idx),
                          [True] * sample1.input.before_event_idx])
        for row, sample in enumerate((sample0, sample1)):
            single = encode_decision(sample.input, meta)
            for field in ("event_type_id", "event_actor_rel", "event_tile_kind",
                          "event_aka", "event_tsumogiri", "event_phase",
                          "event_kyoku_pos"):
                self.assertTrue(torch.equal(
                    getattr(batch, field)[row, :single.event_count],
                    getattr(single, field)), field)
            self.assertTrue(torch.equal(batch.state_features[row],
                                        single.state_features))
            self.assertTrue(torch.equal(batch.legal_mask[row], single.legal_mask))
            self.assertEqual(int(batch.decision_time_index[row]),
                             single.decision_time_index)
        model_batch = batch.to_model_batch()
        self.assertIsInstance(model_batch, ModelBatch)
        self.assertEqual(tuple(model_batch.state_features.shape),
                         (2, STATE_FEATURE_DIM))
        self.assertEqual(model_batch.legal_mask.shape, (2, ACTION_SPACE))
        self.assertEqual(
            model_batch.decision_time_index.tolist(),
            [sample0.input.before_event_idx - 1,
             sample1.input.before_event_idx - 1])

    def test_state_layout_and_no_target_fields(self):
        encoded = encode_decision(self.samples[0].input,
                                  StreamMeta.from_header(self.samples[0].header))
        self.assertEqual(tuple(encoded.state_features.shape), (STATE_FEATURE_DIM,))
        self.assertTrue(torch.isfinite(encoded.state_features).all())
        self.assertEqual(tuple(encoded.legal_mask.shape), (ACTION_SPACE,))
        self.assertTrue(encoded.legal_mask.any())
        payload = encoded_payload(encoded)
        for forbidden in ("target", "label", "rank_final", "expert_action",
                          "supervision_valid", "oracle"):
            self.assertNotIn(forbidden, payload)
        model_batch = encoded.to_model_batch()
        for field in dataclasses.fields(model_batch):
            for forbidden in ("target", "label", "rank", "oracle", "expert"):
                self.assertNotIn(forbidden, field.name)

    def test_schema_hash_and_version_mismatch_fail(self):
        sample = self.samples[0]
        meta = StreamMeta.from_header(sample.header)
        base = meta.as_dict()
        bad_hash = dict(base, schema_hash="0" * 64)
        with self.assertRaises(EncoderError):
            encode_decision(sample.input, bad_hash)
        bad_version = dict(base, schema_version="rf-schema-v3/1")
        with self.assertRaises(EncoderError):
            encode_decision(sample.input, bad_version)
        bad_projection = dict(base, projection_version="rf-pov-v3/1")
        with self.assertRaises(EncoderError):
            encode_decision(sample.input, bad_projection)
        # DecisionInput itself rejects a cut-off that would expose a future event;
        # encode_decision also re-checks exact-prefix equality (defense in depth).
        self.assertEqual(sample.input.before_event_idx, len(sample.input.events))

    def test_no_silent_truncation(self):
        sample = self.samples[0]
        with self.assertRaises(EncoderError):
            encode_decision(
                sample.input, StreamMeta.from_header(sample.header),
                max_events=len(sample.input.events) - 1)
        with self.assertRaises(EncoderError):
            OnlineEncoder(max_events=len(sample.input.events) - 1).encode(
                sample.input, sample.header)

    def test_encoded_hash_is_deterministic(self):
        sample = self.samples[0]
        meta = StreamMeta.from_header(sample.header)
        first = encoded_hash(encode_decision(sample.input, meta))
        second = encoded_hash(OnlineEncoder().encode(sample.input, sample.header))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


class RealLogParityTests(unittest.TestCase):
    logs_seen = 0
    comparisons = 0
    povs_seen: set[int] = set()

    @classmethod
    def setUpClass(cls):
        directory = dataset_dir()
        if directory is None:
            raise unittest.SkipTest("no local 2025 dataset")
        files = sorted(directory.glob("*.mjson"))[:12]
        for path in files:
            raw = read_log(path)
            try:
                samples = build_decisions(raw)
            except Exception:  # noqa: BLE001 - invalid source is not an encoder failure
                continue
            cls.logs_seen += 1
            by_pov = {}
            for sample in samples:
                by_pov.setdefault(sample.header.pov, []).append(sample)
            for pov, values in by_pov.items():
                cls.povs_seen.add(pov)
                picks = {0, len(values) // 2, len(values) - 1}
                for index in sorted(picks):
                    sample = values[index]
                    meta = StreamMeta.from_header(sample.header)
                    offline = encode_decision(sample.input, meta)
                    online = OnlineEncoder().encode(sample.input, sample.header)
                    assert_encoded_equal(unittest.TestCase(), offline, online)
                    cls.comparisons += 1

    def test_real_logs_all_povs_byte_equal(self):
        self.assertGreaterEqual(self.logs_seen, 10)
        self.assertEqual(self.povs_seen, {0, 1, 2, 3})
        self.assertGreaterEqual(self.comparisons, 40)
        print(f"PR11_PARITY logs={self.logs_seen} comparisons={self.comparisons} "
              f"povs={sorted(self.povs_seen)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
