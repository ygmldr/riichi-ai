#!/usr/bin/env python3
"""PR-11 causal/prefix, padding, visibility and model-integration tests."""

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
    EncoderError,
    StreamMeta,
    encode_batch,
    encode_decision,
    encoded_hash,
)
from riichiformer.models.batch import ModelBatch  # noqa: E402
from riichiformer.models.policy import MVPConfig, Policy  # noqa: E402
from riichiformer.models.state_encoder import STATE_FEATURE_LAYOUT  # noqa: E402


HAND0 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
         "1p", "2p", "3p", "4p"]
HAND1 = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
         "E", "S", "W", "N"]
HAND2 = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
         "7m", "8m", "9m", "1p"]
HAND3 = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
         "1s", "2s", "3s", "4s", "5s"]


def raw_short(hands=(HAND0, HAND1, HAND2, HAND3)) -> list[dict]:
    return [
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1p",
         "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
         "scores": [25000] * 4,
         "tehais": [list(hand) for hand in hands]},
        {"type": "tsumo", "actor": 0, "pai": "5mr"},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
        {"type": "end_kyoku"}, {"type": "end_game"},
    ]


def raw_long() -> list[dict]:
    raw = raw_short()
    raw[4:4] = [
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
    ]
    return raw


def first_samples(raw):
    samples = build_decisions(raw)
    by_pov = {}
    for sample in samples:
        by_pov.setdefault(sample.header.pov, sample)
    return by_pov


def tiny_policy() -> Policy:
    torch.manual_seed(11)
    policy = Policy(MVPConfig(layers=2, ffn=64, state_hidden=64,
                              max_positions=2048))
    policy.eval()
    return policy


class CausalPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.short = first_samples(raw_short())[0]
        cls.long = [sample for sample in build_decisions(raw_long())
                    if sample.header.pov == 0][-1]
        cls.long_first = first_samples(raw_long())[0]
        cls.short_encoded = encode_decision(
            cls.short.input, StreamMeta.from_header(cls.short.header))
        cls.long_encoded = encode_decision(
            cls.long_first.input, StreamMeta.from_header(cls.long_first.header))

    def test_appending_future_events_does_not_change_earlier_encoding(self):
        self.assertEqual(self.short.input.before_event_idx, 3)
        self.assertEqual(self.long_first.input.before_event_idx, 3)
        self.assertEqual(encoded_hash(self.short_encoded),
                         encoded_hash(self.long_encoded))

    def test_batch_row_independence_and_prefix_logits(self):
        meta = StreamMeta.from_header(self.short.header)
        long_meta = StreamMeta.from_header(self.long.header)
        batch = encode_batch([self.short.input, self.long.input], [meta, long_meta])
        model_batch = batch.to_model_batch()
        policy = tiny_policy()
        with torch.no_grad():
            batched = policy(model_batch)
            single = policy(self.short_encoded.to_model_batch())
        self.assertEqual(tuple(batched.logits.shape), (2, 46))
        # Batch vs single matmul may differ at float round-off level; the prefix
        # content must be identical up to that numerical tolerance.
        self.assertTrue(torch.allclose(batched.logits[0], single.logits[0],
                                       atol=1e-5, rtol=1e-5))
        batched.validate()
        single.validate()

    def test_changing_future_events_on_a_longer_row_does_not_change_earlier_row(self):
        meta = StreamMeta.from_header(self.short.header)
        batch = encode_batch([self.short.input, self.long.input], [meta, meta])
        model_batch = batch.to_model_batch()
        fields = {}
        for field in dataclasses.fields(model_batch):
            value = getattr(model_batch, field.name)
            if isinstance(value, torch.Tensor):
                fields[field.name] = value.clone()
        fields["schema_version"] = model_batch.schema_version
        # Change only row 1's already-visible future tokens; row 0 must not move.
        fields["event_type_id"][1, 2] = 5
        fields["event_tile_kind"][1, 2] = 34
        fields["event_aka"][1, 2] = 0
        mutated = ModelBatch(**fields)
        policy = tiny_policy()
        with torch.no_grad():
            before = policy(model_batch)
            after = policy(mutated)
        self.assertTrue(torch.equal(before.logits[0], after.logits[0]))
        self.assertTrue(torch.isfinite(after.logits).all())

    def test_padding_slots_do_not_change_valid_logits(self):
        meta = StreamMeta.from_header(self.short.header)
        batch = encode_batch([self.short.input, self.long.input], [meta, meta])
        model_batch = batch.to_model_batch()
        self.assertFalse(bool(model_batch.padding_mask[0, 3:].any()))
        fields = {}
        for field in dataclasses.fields(model_batch):
            value = getattr(model_batch, field.name)
            if isinstance(value, torch.Tensor):
                fields[field.name] = value.clone()
        fields["schema_version"] = model_batch.schema_version
        # Arbitrary in-vocabulary ids in row 0's padded tail.
        fields["event_type_id"][0, 3:] = 16
        fields["event_actor_rel"][0, 3:] = 5
        fields["event_tile_kind"][0, 3:] = 34
        fields["event_aka"][0, 3:] = 0
        fields["event_tsumogiri"][0, 3:] = 0
        fields["event_phase"][0, 3:] = 3
        fields["event_kyoku_pos"][0, 3:] = 511
        mutated = ModelBatch(**fields)
        policy = tiny_policy()
        with torch.no_grad():
            before = policy(model_batch)
            after = policy(mutated)
        self.assertTrue(torch.equal(before.logits[0], after.logits[0]))
        self.assertEqual(float(after.probs[0, ~model_batch.legal_mask[0]].abs().max()), 0.0)

    def test_opponent_concealed_hand_perturbation_is_unchanged(self):
        hands = (
            HAND0,
            ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
             "E", "S", "W", "P"],
            ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
             "7m", "8m", "9m", "2p"],
            ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
             "1s", "2s", "3s", "4s", "6s"],
        )
        base = first_samples(raw_short())[0]
        variant = first_samples(raw_short(hands))[0]
        base_encoded = encode_decision(base.input, StreamMeta.from_header(base.header))
        variant_encoded = encode_decision(
            variant.input, StreamMeta.from_header(variant.header))
        self.assertEqual(encoded_hash(base_encoded), encoded_hash(variant_encoded))

    def test_truncation_raises(self):
        length = len(self.short.input.events)
        with self.assertRaises(EncoderError):
            encode_decision(self.short.input,
                            StreamMeta.from_header(self.short.header),
                            max_events=length - 1)

    def test_model_integration_finite_logits_and_illegal_zero(self):
        meta = StreamMeta.from_header(self.short.header)
        model_batch = encode_batch([self.short.input], [meta]).to_model_batch()
        policy = tiny_policy()
        with torch.no_grad():
            output = policy(model_batch)
        output.validate()
        self.assertEqual(tuple(output.logits.shape), (1, 46))
        self.assertTrue(torch.isfinite(output.logits).all())
        illegal = ~model_batch.legal_mask
        self.assertTrue(torch.equal(output.probs[illegal],
                                    torch.zeros_like(output.probs[illegal])))
        self.assertTrue(torch.isinf(output.masked_logits[illegal]).all())
        for field in dataclasses.fields(model_batch):
            for forbidden in ("target", "label", "rank", "oracle", "expert"):
                self.assertNotIn(forbidden, field.name)

    def test_owns_hand_change_is_visible(self):
        hands = (["9s"] + HAND0[1:], HAND1, HAND2, HAND3)
        changed = first_samples(raw_short(hands))[0]
        base_encoded = encode_decision(
            self.short.input, StreamMeta.from_header(self.short.header))
        changed_encoded = encode_decision(
            changed.input, StreamMeta.from_header(changed.header))
        self.assertNotEqual(encoded_hash(base_encoded), encoded_hash(changed_encoded))




def _dataset_dir():
    for root in (REPO_ROOT / "datasets", REPO_ROOT.parent.parent / "datasets"):
        directory = root / "extracted" / "2025"
        if directory.is_dir():
            return directory
    return None


def _read_log(path):
    data = path.read_bytes()
    if data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


class FixedKanWallTests(unittest.TestCase):
    def test_encoded_state_tiles_left_matches_fixed_kan_input(self):
        directory = _dataset_dir()
        if directory is None:
            self.skipTest("no local 2025 dataset")
        preferred = directory / "2025010100gm-00a9-0000-029e0f35.mjson"
        candidates = ([preferred] if preferred.exists() else []) + sorted(
            directory.glob("*.mjson"))[:10]
        kan_games = checked = 0
        global_offset = next(slot.offset for slot in STATE_FEATURE_LAYOUT
                             if slot.name == "global")
        tiles_left_index = global_offset + 4 + 1 + 1 + 1
        for path in candidates:
            raw = _read_log(path)
            if not any(event.get("type") in ("ankan", "kakan", "daiminkan")
                       for event in raw):
                continue
            try:
                samples = build_decisions(raw)
            except Exception:
                continue
            for sample in samples:
                prefix = list(sample.input.events)
                start = max((index for index, event in enumerate(prefix)
                             if event.EVENT_TYPE == "start_kyoku"), default=0)
                expected = 70 - sum(
                    1 for event in prefix[start:] if event.EVENT_TYPE == "tsumo")
                self.assertEqual(sample.input.global_public.tiles_left, expected)
                encoded = encode_decision(
                    sample.input, StreamMeta.from_header(sample.header))
                self.assertAlmostEqual(
                    float(encoded.state_features[tiles_left_index]), expected / 70.0,
                    places=6,
                )
                checked += 1
            kan_games += 1
            if kan_games >= 2:
                break
        self.assertGreater(kan_games, 0, "no real kan game for wall regression")
        self.assertGreater(checked, 0)
        print(f"PR11_KAN_WALL kan_games={kan_games} decisions_checked={checked}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
