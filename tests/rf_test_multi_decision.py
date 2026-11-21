#!/usr/bin/env python3
"""PR-38 multi-decision shared-forward tests (causal parity, markers, extremes).

The central claim under test is: one shared transformer pass over a full POV
stream with explicit ``DECISION`` markers produces **the same** logits and the
same gradients as the legacy per-decision prefix forward, at a fraction of the
event-token recomputation.  The parity assertions therefore compare against the
*existing* ``Policy.forward`` on ``encode_decision(...).to_model_batch()`` rows,
not against a second copy of the new code.

Floating-point policy (dropout = 0, eval): logits ``atol=1e-5 / rtol=1e-5``
(the same tolerance the PR-13 prefix-invariance tests already use), gradients
``atol=2e-6`` and relative ``2e-5`` per tensor.  The measured maxima are printed
so a reviewer can see the margin; bitwise equality is never claimed for the
joint pass because SDPA reduction order differs between kernel paths.

Run with ``PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release``.
"""

from __future__ import annotations

import dataclasses
import gzip
import inspect
import json
import pathlib
import sys
import time
import unittest
from dataclasses import replace as dataclass_replace

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.decision import build_decisions  # noqa: E402
from riichiformer.data.encode import (  # noqa: E402
    EncoderError,
    StreamMeta,
    encode_decision,
    encode_multi_decision,
    encode_multi_decision_batch,
    multi_decision_hash,
    multi_decision_payload,
)
from riichiformer.data.multi_decision import (  # noqa: E402
    DecisionMarker,
    MultiDecisionStream,
    prefix_recompute_profile,
    streams_from_samples,
)
from riichiformer.models.batch import (  # noqa: E402
    EVENT_TENSOR_FIELDS,
    TSUMO_EVENT_TYPE,
    ModelBatch,
    ModelBatchError,
    MultiDecisionBatch,
)
from riichiformer.models.policy import (  # noqa: E402
    MVPConfig,
    Policy,
    multi_decision_attention_mask,
)
from riichiformer.models.state_encoder import STATE_FEATURE_DIM  # noqa: E402
from riichiformer.models.local_transformer import TransformerShapeError  # noqa: E402
from riichiformer.schemas.action_v3 import ACTION_SPACE, ActionPhase  # noqa: E402
from riichiformer.train.bc import masked_bc_loss  # noqa: E402

HAND0 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
         "1p", "2p", "3p", "4p"]
HAND1 = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
         "E", "S", "W", "N"]
HAND2 = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
         "7m", "8m", "9m", "1p"]
HAND3 = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
         "1s", "2s", "3s", "4s", "5s"]

#: Tight parity tolerances, documented rather than implicit.
LOGIT_ATOL = 1e-5
LOGIT_RTOL = 1e-5
GRAD_ATOL = 2e-6
GRAD_RTOL = 2e-5


def start_kyoku(*, kyoku=1, oya=0, dora="1p",
                hands=(HAND0, HAND1, HAND2, HAND3), scores=None) -> dict:
    return {"type": "start_kyoku", "bakaze": "E", "dora_marker": dora,
            "kyoku": kyoku, "honba": 0, "kyotaku": 0, "oya": oya,
            "scores": list(scores or [25000] * 4),
            "tehais": [list(hand) for hand in hands]}


def raw_game(*, hands=(HAND0, HAND1, HAND2, HAND3), middle=()) -> list[dict]:
    raw = [{"type": "start_game", "names": ["a", "b", "c", "d"]},
           start_kyoku(hands=hands)]
    raw.extend(middle)
    raw.extend([{"type": "end_kyoku"}, {"type": "end_game"}])
    return raw


CALL_HAND = ["5m", "5m", "1s", "2s", "3s", "4s", "6s", "7s", "8s",
             "E", "S", "W", "N"]


def four_pov_game(hands=None) -> list[dict]:
    """Calls, a second draw and a second kyoku in one valid game."""
    hands = tuple(hands or (HAND0, CALL_HAND, HAND2, HAND3))
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


def kan_game() -> list[dict]:
    """A kan-selection opportunity shares its cutoff with the action decision."""
    hands = (["1m", "1m", "1m", "1m", "2m", "2m", "2m", "2m",
              "3m", "4m", "5m", "6m", "7m"], HAND1,
             ["P", "F", "C", "9p", "9p", "3m", "4m", "5m", "6m",
              "7m", "8m", "9m", "1p"], HAND3)
    return raw_game(hands=hands, middle=[
        {"type": "tsumo", "actor": 0, "pai": "8m"},
        {"type": "ankan", "actor": 0, "consumed": ["1m", "1m", "1m", "1m"]},
        {"type": "dahai", "actor": 0, "pai": "8m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "1s"},
        {"type": "dahai", "actor": 1, "pai": "1s", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "9p"},
        {"type": "dahai", "actor": 2, "pai": "9p", "tsumogiri": True},
    ])


def pass_game(*, preempted: bool = False) -> list[dict]:
    """Genuine pass (action 45) or a call-preempted censored decision."""
    call_hand = ["5m", "5m", "1s", "2s", "3s", "4s", "6s", "7s", "8s",
                 "E", "S", "W", "N"]
    if preempted:
        hands = (HAND0, call_hand,
                 ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                  "3m", "4m", "5s", "5s"], HAND3)
        middle = [
            {"type": "tsumo", "actor": 0, "pai": "1m"},
            {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
            {"type": "pon", "actor": 1, "target": 0, "pai": "5m",
             "consumed": ["5m", "5m"]},
            {"type": "dahai", "actor": 1, "pai": "1s", "tsumogiri": False},
        ]
    else:
        hands = (HAND0, call_hand, HAND2, HAND3)
        middle = [
            {"type": "tsumo", "actor": 0, "pai": "1m"},
            {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
            {"type": "tsumo", "actor": 1, "pai": "1s"},
            {"type": "dahai", "actor": 1, "pai": "1s", "tsumogiri": True},
        ]
    return raw_game(hands=hands, middle=middle)


_CACHE: dict[str, list] = {}


def samples_for(name: str) -> list:
    if name not in _CACHE:
        raw = {"four": four_pov_game, "kan": kan_game,
               "pass": lambda: pass_game(preempted=False),
               "censor": lambda: pass_game(preempted=True)}[name]()
        _CACHE[name] = build_decisions(raw)
    return _CACHE[name]


def streams_for(name: str) -> list[MultiDecisionStream]:
    return streams_from_samples(samples_for(name))


def streams_batch(streams, *, verify_prefix: str = "exact") -> MultiDecisionBatch:
    return encode_multi_decision_batch(streams, verify_prefix=verify_prefix)


def ordered_samples(streams, samples):
    """DecisionSample order matching ``encode_multi_decision_batch`` ordering."""
    by_pov: dict[int, list] = {}
    for sample in samples:
        by_pov.setdefault(sample.header.pov, []).append(sample)
    ordered = []
    for stream in streams:
        ordered.extend(by_pov[stream.pov])
    return ordered


def target_dicts(samples, *, weights=None, forced=(), censored=()) -> list[dict]:
    targets = []
    for index, sample in enumerate(samples):
        valid = bool(sample.target.supervision_valid)
        targets.append({
            "supervision_valid": valid,
            "expert_action": sample.target.expert_action,
            "loss_weight": 1.0 if weights is None else float(weights[index]),
            "forced": index in forced,
        })
    for index in censored:
        targets[index]["supervision_valid"] = False
    return targets


def legacy_prefix_batch(sample) -> ModelBatch:
    return encode_decision(
        sample.input, StreamMeta.from_header(sample.header)).to_model_batch()


def small_policy(seed: int = 13, **overrides) -> Policy:
    defaults = dict(layers=2, ffn=64, state_hidden=64, max_positions=2048)
    defaults.update(overrides)
    torch.manual_seed(seed)
    model = Policy(MVPConfig(**defaults))
    model.eval()
    return model


class MultiDecisionEncodingTests(unittest.TestCase):
    def test_prefix_tensor_parity_with_legacy_encoder(self):
        samples = samples_for("four")
        streams = streams_for("four")
        batch = streams_batch(streams)
        # The stream event-phase field is decision-neutral by contract.
        self.assertFalse(bool(batch.event_phase[batch.padding_mask].any()))
        offset = 0
        for row, stream in enumerate(streams):
            for index, sample in enumerate(
                    [s for s in samples if s.header.pov == stream.pov]):
                legacy = encode_decision(
                    sample.input, StreamMeta.from_header(sample.header))
                cutoff = batch.decision_cutoff_idx[offset + index]
                self.assertEqual(int(cutoff), sample.input.before_event_idx)
                for name in EVENT_TENSOR_FIELDS:
                    if name == "event_phase":
                        continue
                    self.assertTrue(torch.equal(
                        getattr(batch, name)[row, :int(cutoff)],
                        getattr(legacy, name)), f"{name} pov={stream.pov}")
                self.assertTrue(torch.equal(
                    batch.state_features[offset + index], legacy.state_features))
                self.assertTrue(torch.equal(
                    batch.legal_mask[offset + index], legacy.legal_mask))
                self.assertEqual(
                    int(batch.decision_phase[offset + index]), legacy.phase_index)
            offset += len([s for s in samples if s.header.pov == stream.pov])
        self.assertEqual(offset, batch.n_decisions)

    def test_duplicate_cutoff_kan_selection_gets_two_markers(self):
        streams = streams_for("kan")
        pov0 = next(stream for stream in streams if stream.pov == 0)
        cutoffs = pov0.cutoffs
        self.assertEqual(len(cutoffs) - len(set(cutoffs)), 1)
        batch = streams_batch(streams)
        shared = next(cutoff for cutoff in cutoffs if cutoffs.count(cutoff) > 1)
        duplicate_values = [int(value) for value in batch.decision_cutoff_idx]
        rows = batch.decision_batch_index == 0
        markers = batch.decision_marker_index[rows]
        self.assertEqual(sorted(int(m) for m in markers), list(range(int(markers.max()) + 1)))
        phases = [batch.decision_phases[int(i)]
                  for i in torch.nonzero(rows, as_tuple=False).flatten()]
        self.assertIn(ActionPhase.SELF_TURN, phases)
        self.assertIn(ActionPhase.KAN_SELECTION, phases)
        self.assertEqual(shared, int(duplicate_values[1]))
        # Both markers read the same cursor event but keep distinct phases.
        sources = batch.decision_source_index[rows]
        self.assertEqual(int(sources[0]), int(sources[1]))
        self.assertEqual(len(duplicate_values), batch.n_decisions)
        model = small_policy()
        with torch.no_grad():
            output = model.forward_multi(batch)
        output.validate()
        self.assertEqual(tuple(output.logits.shape), (batch.n_decisions, ACTION_SPACE))

    def test_pass_and_censored_decisions_are_encoded(self):
        samples = samples_for("pass")
        streams = streams_for("pass")
        batch = streams_batch(streams)
        ordered = ordered_samples(streams, samples)
        phases = [batch.decision_phases[i] for i in range(batch.n_decisions)]
        self.assertIn(ActionPhase.CALL_RESPONSE, phases)
        call_rows = [i for i, phase in enumerate(phases)
                     if phase is ActionPhase.CALL_RESPONSE]
        self.assertTrue(call_rows)
        for row in call_rows:
            self.assertTrue(bool(batch.legal_mask[row, 45]))
        censored_samples = samples_for("censor")
        streams = streams_for("censor")
        batch = streams_batch(streams)
        ordered = ordered_samples(streams, censored_samples)
        censored = [index for index, sample in enumerate(ordered)
                    if not sample.target.supervision_valid]
        self.assertTrue(censored)
        for index in censored:
            self.assertEqual(ordered[index].target.censor_reason,
                             "preempted_by_call")
            self.assertTrue(bool(batch.legal_mask[index, 45]))
        model = small_policy()
        targets = target_dicts(ordered)
        with torch.no_grad():
            output = model.forward_multi(batch)
        result = masked_bc_loss(output, targets)
        self.assertLess(result.valid_decisions, len(targets))

    def test_multi_kyoku_stream(self):
        streams = streams_for("four")
        batch = streams_batch(streams)
        kyoku_ids = set()
        for stream in streams:
            for event in stream.events:
                kyoku_ids.add(event.kyoku_idx)
            self.assertEqual(stream.events[0].seq_idx, 0)
            self.assertEqual(stream.events[-1].seq_idx, stream.length - 1)
            positions = [event.seq_idx for event in stream.events]
            self.assertEqual(positions, list(range(stream.length)))
        self.assertGreater(len(kyoku_ids), 1)
        self.assertGreater(batch.n_decisions, 1)
        # Cross-kyoku decisions remain in the same causal main sequence: at
        # least one decision cutoff lies after the second kyoku's start event.
        from riichiformer.schemas.event_v3 import StartKyokuEvent
        for stream in streams:
            starts = [index for index, event in enumerate(stream.events)
                      if isinstance(event, StartKyokuEvent)]
            if len(starts) > 1:
                self.assertTrue(any(cutoff > starts[1]
                                    for cutoff in stream.cutoffs))

    def test_padding_and_select_rows(self):
        streams = streams_for("four")
        batch = streams_batch(streams)
        lengths = [stream.length for stream in streams]
        self.assertEqual(batch.seq_len, max(lengths))
        expected = torch.zeros((len(streams), batch.seq_len), dtype=torch.bool)
        for row, length in enumerate(lengths):
            expected[row, :length] = True
        self.assertTrue(torch.equal(batch.padding_mask, expected))
        markers = batch.marker_padding_mask
        counts = torch.bincount(batch.decision_batch_index,
                                minlength=batch.batch_size)
        for row, count in enumerate(counts.tolist()):
            self.assertEqual(int(markers[row].sum()), count)
            self.assertTrue(bool(markers[row, :count].all()))
        subset = batch.select_rows(torch.tensor([1, 3]))
        self.assertEqual(subset.batch_size, 2)
        self.assertEqual(subset.n_decisions,
                         int(counts[1] + counts[3]))
        model = small_policy()
        with torch.no_grad():
            full = model.forward_multi(batch)
            part = model.forward_multi(subset)
        rows = ((batch.decision_batch_index == 1)
                | (batch.decision_batch_index == 3))
        self.assertTrue(torch.allclose(part.logits, full.logits[rows],
                                       atol=LOGIT_ATOL, rtol=LOGIT_RTOL))

    def test_length_extremes_short_and_long_streams(self):
        samples = samples_for("four")
        base_stream = next(stream for stream in streams_for("four")
                           if stream.pov == 0)
        meta = base_stream.meta
        # A one-event stream with a synthetic marker at cutoff 1.
        one_event = (base_stream.events[0],)
        legal = tuple(index == 0 for index in range(ACTION_SPACE))
        marker = DecisionMarker(
            cutoff_idx=1, phase=ActionPhase.SELF_TURN, legal_mask=legal,
            state_features=torch.full((STATE_FEATURE_DIM,), 0.1))
        short = MultiDecisionStream(events=one_event, decisions=(marker,),
                                    meta=meta)
        # A long (400 event) stream built by extending a real stream; the copied
        # events are the same own-draw event so the visibility contract holds.
        source_event = next(event for event in base_stream.events
                            if type(event).__name__ == "TsumoEvent")
        long_events = list(base_stream.events)
        while len(long_events) < 400:
            long_events.append(dataclass_replace(
                source_event, seq_idx=len(long_events)))
        long_decisions = tuple(
            DecisionMarker(cutoff_idx=cutoff, phase=ActionPhase.SELF_TURN,
                           legal_mask=legal,
                           state_features=torch.full((STATE_FEATURE_DIM,), 0.2))
            for cutoff in (1, 200, 400))
        long_stream = MultiDecisionStream(events=tuple(long_events),
                                          decisions=long_decisions, meta=meta)
        batch = encode_multi_decision_batch([short, long_stream])
        self.assertEqual(batch.seq_len, 400)
        self.assertEqual(batch.marker_capacity, 3)
        self.assertEqual(batch.n_decisions, 4)
        for row, expected_markers in ((0, 1), (1, 3)):
            self.assertEqual(int(batch.marker_padding_mask[row].sum()),
                             expected_markers)
        model = small_policy(max_positions=1024)
        with torch.no_grad():
            output = model.forward_multi(batch)
        output.validate()
        self.assertEqual(tuple(output.logits.shape), (4, ACTION_SPACE))
        # The joined length is bounded by max_positions; refusal, not truncation.
        tiny = small_policy(max_positions=64)
        with self.assertRaises(TransformerShapeError):
            tiny.forward_multi(batch)

    def test_contract_negatives(self):
        samples = samples_for("four")
        streams = streams_for("four")
        batch = streams_batch(streams)
        fields = {}
        for field in dataclasses.fields(batch):
            value = getattr(batch, field.name)
            fields[field.name] = value.clone() if isinstance(value, torch.Tensor) \
                else value
        bad = dict(fields)
        bad["event_phase"] = fields["event_phase"].clone()
        bad["event_phase"][0, 0] = 1
        with self.assertRaises(ModelBatchError):
            MultiDecisionBatch(**bad)
        bad = dict(fields)
        bad["decision_cutoff_idx"] = fields["decision_cutoff_idx"].clone()
        bad["decision_cutoff_idx"][0] = 0
        with self.assertRaises(ModelBatchError):
            MultiDecisionBatch(**bad)
        bad = dict(fields)
        bad["decision_cutoff_idx"] = fields["decision_cutoff_idx"].clone()
        bad["decision_cutoff_idx"][0] = batch.seq_len + 1
        with self.assertRaises(ModelBatchError):
            MultiDecisionBatch(**bad)
        bad = dict(fields)
        bad["decision_marker_index"] = fields["decision_marker_index"].clone()
        bad["decision_marker_index"][0] = 1
        with self.assertRaises(ModelBatchError):
            MultiDecisionBatch(**bad)
        bad = dict(fields)
        bad["legal_mask"] = fields["legal_mask"].clone()
        bad["legal_mask"][0] = torch.zeros(ACTION_SPACE, dtype=torch.bool)
        with self.assertRaises(ModelBatchError):
            MultiDecisionBatch(**bad)
        # A decision may not point at a padded event token.
        bad = dict(fields)
        bad["decision_batch_index"] = fields["decision_batch_index"].clone()
        bad["decision_cutoff_idx"] = fields["decision_cutoff_idx"].clone()
        shortest = int(torch.argmin(batch.lengths))
        rows = torch.nonzero(batch.decision_batch_index == shortest,
                             as_tuple=False).flatten()
        bad["decision_cutoff_idx"][rows[-1]] = batch.seq_len
        with self.assertRaises(ModelBatchError):
            MultiDecisionBatch(**bad)

    def test_encoder_negatives(self):
        samples = samples_for("four")
        streams = streams_for("four")
        stream = streams[0]
        pov_samples = [s for s in samples if s.header.pov == stream.pov]
        good = pov_samples[0].input
        events = stream.events
        # A cutoff that is not the prefix length can never enter the stream.
        marker = DecisionMarker(
            cutoff_idx=len(events) + 1, phase=ActionPhase.SELF_TURN,
            legal_mask=tuple(i == 0 for i in range(ACTION_SPACE)),
            state_features=torch.zeros(STATE_FEATURE_DIM))
        with self.assertRaises(EncoderError):
            encode_multi_decision(events, [marker],
                                  StreamMeta.from_header(stream.meta))
        # Decisions must be in causal order.
        later = DecisionMarker(
            cutoff_idx=len(events), phase=ActionPhase.SELF_TURN,
            legal_mask=marker.legal_mask, state_features=marker.state_features)
        earlier = DecisionMarker(
            cutoff_idx=1, phase=ActionPhase.SELF_TURN,
            legal_mask=marker.legal_mask, state_features=marker.state_features)
        with self.assertRaises(EncoderError):
            encode_multi_decision(events, [later, earlier],
                                  StreamMeta.from_header(stream.meta))
        # A decision whose visible prefix differs from the stream prefix fails
        # closed in exact mode (structural mode cannot see a tile change).
        from riichiformer.schemas.event_v3 import DahaiEvent
        tampered = next((event for event in good.events
                         if isinstance(event, DahaiEvent)), None)
        if tampered is not None:
            altered_event = dataclass_replace(
                tampered, tile=type(tampered.tile)(kind=0))
            altered = dataclass_replace(
                good, events=good.events[:altered_event.seq_idx]
                + (altered_event,)
                + good.events[altered_event.seq_idx + 1:])
            with self.assertRaises(EncoderError):
                encode_multi_decision(events, [altered],
                                      StreamMeta.from_header(stream.meta),
                                      verify_prefix="exact")
        # Truncation is refused, not silent.
        with self.assertRaises(EncoderError):
            encode_multi_decision(events, [good],
                                  StreamMeta.from_header(stream.meta),
                                  max_events=len(events) - 1)
        # Unknown event objects are refused.
        with self.assertRaises(EncoderError):
            encode_multi_decision(list(events) + ["not an event"], [good],
                                  StreamMeta.from_header(stream.meta))

    def test_no_target_fields_and_deterministic_hash(self):
        streams = streams_for("four")
        batch = streams_batch(streams)
        forbidden = ("target", "label", "oracle", "rank", "expert", "censor",
                     "supervision")
        for field in dataclasses.fields(MultiDecisionBatch):
            for word in forbidden:
                self.assertNotIn(word, field.name)
        signature = inspect.signature(encode_multi_decision)
        for name in signature.parameters:
            for word in forbidden:
                self.assertNotIn(word, name)
        payload = multi_decision_payload(batch)
        for name in payload:
            for word in forbidden:
                self.assertNotIn(word, name)
        again = streams_batch(streams_for("four"))
        self.assertEqual(multi_decision_hash(batch), multi_decision_hash(again))
        self.assertEqual(len(multi_decision_hash(batch)), 64)
        moved = batch.to("cpu")
        self.assertEqual(multi_decision_hash(moved), multi_decision_hash(batch))

    def test_visibility_rejects_revealed_opponent_tile(self):
        streams = streams_for("four")
        batch = streams_batch(streams)
        fields = {field.name: getattr(batch, field.name)
                  for field in dataclasses.fields(batch)}
        for name in ("event_type_id", "event_actor_rel", "event_tile_kind",
                     "event_aka"):
            fields[name] = getattr(batch, name).clone()
        # Reveal one known tile on an opponent tsumo token (any stream row).
        found = None
        for row in range(batch.batch_size):
            tsumo = torch.nonzero(
                (batch.event_type_id[row] == TSUMO_EVENT_TYPE)
                & (batch.event_actor_rel[row] > 0)
                & batch.padding_mask[row], as_tuple=False).flatten()
            if tsumo.numel():
                found = (row, int(tsumo[0]))
                break
        if found is None:
            self.skipTest("fixture has no opponent tsumo")
        row, column = found
        fields["event_tile_kind"][row, column] = 0
        fields["event_aka"][row, column] = 0
        with self.assertRaises(ModelBatchError):
            MultiDecisionBatch(**fields)
        fields["event_tile_kind"][row, column] = 34
        fields["event_aka"][row, column] = 1
        with self.assertRaises(ModelBatchError):
            MultiDecisionBatch(**fields)

    def test_opponent_hidden_hand_variants_do_not_change_encoding(self):
        # Opponent haipai are never read: two different concealed hands with the
        # same public events must produce the same stream hash for every POV that
        # does not own the changed hand.
        base_samples = samples_for("four")
        variant_hands = (HAND0, CALL_HAND,
                         ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
                          "7m", "8m", "9m", "2p"], HAND3)
        variant_samples = build_decisions(four_pov_game(hands=variant_hands))
        base_streams = {stream.pov: stream
                        for stream in streams_from_samples(base_samples)}
        variant_streams = {stream.pov: stream
                           for stream in streams_from_samples(variant_samples)}
        for pov in (0, 1, 3):  # seat 2's own concealed hand is what changed
            self.assertEqual(
                multi_decision_hash(streams_batch([base_streams[pov]])),
                multi_decision_hash(streams_batch([variant_streams[pov]])),
                f"pov {pov}",
            )
        # The owning POV *does* see its own haipai change.
        self.assertNotEqual(
            multi_decision_hash(streams_batch([base_streams[2]])),
            multi_decision_hash(streams_batch([variant_streams[2]])),
        )

    def test_profiler_counts_recomputation(self):
        streams = streams_for("four")
        profile = prefix_recompute_profile(streams)
        self.assertEqual(profile["streams"], len(streams))
        self.assertEqual(profile["prefix_tokens"],
                         sum(sum(stream.cutoffs) for stream in streams))
        self.assertEqual(profile["shared_tokens"],
                         sum(stream.length + len(stream.decisions)
                             for stream in streams))
        self.assertEqual(profile["prefix_tokens"] - profile["shared_tokens"],
                         profile["saved_tokens"])
        for row in profile["per_stream"]:
            self.assertGreaterEqual(row["decisions"], 1)
            self.assertGreaterEqual(row["events"], 1)
        print(f"PR38_PROFILE {json.dumps({k: v for k, v in profile.items() if k != 'per_stream'}, sort_keys=True)}")


class MultiDecisionForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = samples_for("four")
        cls.streams = streams_for("four")
        cls.batch = streams_batch(cls.streams)
        cls.ordered = ordered_samples(cls.streams, cls.samples)
        cls.model = small_policy()
        cls.weights = [0.5 + 0.25 * (index % 4)
                       for index in range(len(cls.ordered))]
        cls.targets = target_dicts(cls.ordered, weights=cls.weights)
        with torch.no_grad():
            cls.joint = cls.model.forward_multi(cls.batch)

    def test_joint_logits_match_legacy_prefix(self):
        max_diff = 0.0
        for index, sample in enumerate(self.ordered):
            with torch.no_grad():
                legacy = self.model(legacy_prefix_batch(sample))
            diff = float((legacy.logits[0] - self.joint.logits[index]).abs().max())
            max_diff = max(max_diff, diff)
            self.assertTrue(torch.allclose(
                legacy.logits[0], self.joint.logits[index],
                atol=LOGIT_ATOL, rtol=LOGIT_RTOL),
                f"decision {index}: |diff|={diff}")
        self.assertLess(max_diff, LOGIT_ATOL)
        print(f"PR38_LOGIT_PARITY max_abs_diff={max_diff:.3e} "
              f"decisions={len(self.ordered)}")
        self.joint.validate()
        self.assertEqual(tuple(self.joint.selected_marker_offsets.shape),
                         (self.batch.n_decisions,))
        self.assertTrue(torch.equal(self.joint.decision_cutoff_idx,
                                    self.batch.decision_cutoff_idx))

    def test_joint_gradients_match_legacy_prefix(self):
        model = small_policy(seed=17)
        params = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
        joint_output = model.forward_multi(self.batch)
        joint_result = masked_bc_loss(joint_output, self.targets)
        window_total = sum(
            float(target["loss_weight"]) for target in self.targets
            if target["supervision_valid"] and not target["forced"])
        joint_loss = joint_result.numerator / window_total
        joint_grads = torch.autograd.grad(joint_loss, params, allow_unused=True)
        numerator = None
        for index, sample in enumerate(self.ordered):
            target = self.targets[index]
            if not target["supervision_valid"] or target["forced"]:
                continue
            prefix_output = model(legacy_prefix_batch(sample))
            term = -prefix_output.logprobs[0, target["expert_action"]] \
                * float(target["loss_weight"])
            numerator = term if numerator is None else numerator + term
        prefix_loss = numerator / window_total
        prefix_grads = torch.autograd.grad(prefix_loss, params, allow_unused=True)
        self.assertAlmostEqual(float(joint_loss.detach()),
                               float(prefix_loss.detach()), places=6)
        max_abs = max_rel = 0.0
        for (name, _parameter), joint_grad, prefix_grad in zip(
                model.named_parameters(), joint_grads, prefix_grads):
            if joint_grad is None or prefix_grad is None:
                continue
            diff = float((joint_grad - prefix_grad).abs().max())
            scale = max(float(prefix_grad.abs().max()), 1e-8)
            max_abs = max(max_abs, diff)
            max_rel = max(max_rel, diff / scale)
            self.assertTrue(
                torch.allclose(joint_grad, prefix_grad,
                               atol=GRAD_ATOL, rtol=GRAD_RTOL),
                f"gradient mismatch {name}: abs={diff:.3e} rel={diff / scale:.3e}")
        self.assertLess(max_abs, GRAD_ATOL)
        self.assertLess(max_rel, GRAD_RTOL)
        print(f"PR38_GRAD_PARITY max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
              f"decisions={len(self.ordered)} loss={float(joint_loss.detach()):.6f}")

    def test_kan_duplicate_cutoff_parity(self):
        # The kan-selection opportunity shares its cutoff with the action it
        # re-labels: two markers read the same cursor event but keep their own
        # phase/state.  Logits *and* gradients must still match the per-prefix
        # legacy forward for both decisions.
        samples = samples_for("kan")
        streams = streams_for("kan")
        batch = streams_batch(streams)
        ordered = ordered_samples(streams, samples)
        duplicated = (len(batch.decision_cutoff_idx)
                      - len(set(int(value) for value in batch.decision_cutoff_idx)))
        self.assertEqual(duplicated, 1)
        model = small_policy(seed=21)
        with torch.no_grad():
            joint = model.forward_multi(batch)
        max_diff = 0.0
        for index, sample in enumerate(ordered):
            with torch.no_grad():
                legacy = model(legacy_prefix_batch(sample))
            diff = float((legacy.logits[0] - joint.logits[index]).abs().max())
            max_diff = max(max_diff, diff)
            self.assertTrue(torch.allclose(legacy.logits[0], joint.logits[index],
                                           atol=LOGIT_ATOL, rtol=LOGIT_RTOL))
        targets = target_dicts(ordered)
        params = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
        weight = sum(float(target["loss_weight"]) for target in targets
                     if target["supervision_valid"] and not target["forced"])
        result = masked_bc_loss(model.forward_multi(batch), targets)
        joint_grads = torch.autograd.grad(result.numerator / weight, params,
                                          allow_unused=True)
        numerator = None
        for index, sample in enumerate(ordered):
            target = targets[index]
            if not target["supervision_valid"] or target["forced"]:
                continue
            prefix_output = model(legacy_prefix_batch(sample))
            term = -prefix_output.logprobs[0, target["expert_action"]] \
                * float(target["loss_weight"])
            numerator = term if numerator is None else numerator + term
        prefix_grads = torch.autograd.grad(numerator / weight, params,
                                           allow_unused=True)
        max_grad = max(float((left - right).abs().max())
                       for left, right in zip(joint_grads, prefix_grads)
                       if left is not None and right is not None)
        self.assertLess(max_diff, LOGIT_ATOL)
        self.assertLess(max_grad, GRAD_ATOL)
        print(f"PR38_KAN_PARITY max_logit_diff={max_diff:.3e} "
              f"max_grad_diff={max_grad:.3e} duplicate_cutoffs={duplicated}")

    def test_single_decision_stream_matches_legacy_forward(self):
        # Truncate one POV stream to the first decision: this is the strictest
        # joint-vs-legacy case (the marker is the only decision token).
        stream = self.streams[0]
        sample = next(sample for sample in self.samples
                      if sample.header.pov == stream.pov)
        cutoff = sample.input.before_event_idx
        prefix_stream = MultiDecisionStream.from_decision_inputs(
            events=stream.events[:cutoff], decisions=(sample.input,),
            meta=stream.meta)
        batch = encode_multi_decision_batch([prefix_stream], verify_prefix="exact")
        with torch.no_grad():
            joint = self.model.forward_multi(batch)
            legacy = self.model(legacy_prefix_batch(sample))
        self.assertTrue(torch.allclose(joint.logits[0], legacy.logits[0],
                                       atol=LOGIT_ATOL, rtol=LOGIT_RTOL))
        self.assertTrue(torch.equal(joint.legal_mask, legacy.legal_mask))

    def test_future_event_mutation_does_not_change_past_decisions(self):
        baseline = self.joint
        cutoff = int(self.batch.decision_cutoff_idx[0])
        fields = {field.name: getattr(self.batch, field.name)
                  for field in dataclasses.fields(self.batch)}
        for name in ("event_type_id", "event_actor_rel", "event_tile_kind",
                     "event_aka", "event_tsumogiri", "event_kyoku_pos"):
            fields[name] = getattr(self.batch, name).clone()
        # Mutate every event token after the first decision's cutoff; the first
        # marker may not read them, and no marker reads another marker's state.
        for name in ("event_type_id", "event_actor_rel", "event_tile_kind",
                     "event_aka", "event_tsumogiri", "event_kyoku_pos"):
            fields[name][0, cutoff:] = 0
        mutated = MultiDecisionBatch(**fields)
        with torch.no_grad():
            output = self.model.forward_multi(mutated)
        self.assertTrue(torch.equal(output.logits[0], baseline.logits[0]))
        # A later decision on the same row may change (it reads the mutated events).
        row0 = torch.nonzero(self.batch.decision_batch_index == 0,
                             as_tuple=False).flatten()
        if row0.numel() > 1:
            self.assertFalse(torch.equal(output.logits[row0[-1]],
                                         baseline.logits[row0[-1]]))

    def test_marker_state_is_decision_local(self):
        baseline = self.joint
        for decision in (0, self.batch.n_decisions - 1):
            fields = {field.name: getattr(self.batch, field.name)
                      for field in dataclasses.fields(self.batch)}
            fields["state_features"] = self.batch.state_features.clone()
            fields["state_features"][decision] += 3.0
            mutated = MultiDecisionBatch(**fields)
            with torch.no_grad():
                output = self.model.forward_multi(mutated)
            changed = (~torch.isclose(output.logits, baseline.logits,
                                      atol=1e-7).any(dim=-1))
            expected = torch.zeros(self.batch.n_decisions, dtype=torch.bool)
            expected[decision] = True
            self.assertTrue(torch.equal(changed, expected),
                            f"state of decision {decision} leaked to "
                            f"{torch.nonzero(changed, as_tuple=False).flatten().tolist()}")

    def test_multi_decision_mask_contract(self):
        mask = multi_decision_attention_mask(self.batch)
        total = self.batch.seq_len + self.batch.marker_capacity
        self.assertEqual(tuple(mask.shape), (self.batch.batch_size, 1, total, total))
        capacity = self.batch.marker_capacity
        for row in range(self.batch.batch_size):
            length = int(self.batch.lengths[row])
            # Event queries never see marker keys; event causality is strict.
            self.assertFalse(bool(mask[row, 0, :length, self.batch.seq_len:].any()))
            for query in range(length):
                allowed = torch.nonzero(mask[row, 0, query], as_tuple=False).flatten()
                self.assertTrue(bool((allowed < query + 1).all()))
            # Marker queries see their prefix and themselves, never a peer.
            for slot in range(capacity):
                marker_index = self.batch.seq_len + slot
                allowed = torch.nonzero(mask[row, 0, marker_index],
                                        as_tuple=False).flatten()
                own = (allowed == marker_index)
                self.assertEqual(int(own.sum()), 1)
                event_keys = allowed[allowed < self.batch.seq_len]
                if slot < int(self.batch.marker_padding_mask[row].sum()):
                    cursor = int(self.batch.decision_source_index[
                        (self.batch.decision_batch_index == row)
                        & (self.batch.decision_marker_index == slot)][0])
                    self.assertTrue(bool((event_keys <= cursor - 1).all()))

    def test_deterministic_eval_and_train_no_dropout(self):
        with torch.no_grad():
            again = self.model.forward_multi(self.batch)
        self.assertTrue(torch.equal(again.logits, self.joint.logits))
        self.model.train()
        try:
            with torch.no_grad():
                train_first = self.model.forward_multi(self.batch)
                train_second = self.model.forward_multi(self.batch)
        finally:
            self.model.eval()
        self.assertTrue(torch.equal(train_first.logits, train_second.logits))
        self.assertEqual(self.model.config.dropout, 0.0)

    def test_legacy_forward_still_accepts_model_batch_only(self):
        with self.assertRaises(TypeError):
            self.model(self.batch)
        with self.assertRaises(TypeError):
            self.model.forward_multi(self.samples[0].input)
        self.assertEqual(list(inspect.signature(Policy.forward).parameters),
                         ["self", "batch"])
        # The legacy path is byte-identical to itself after the new mode exists.
        legacy = self.model(legacy_prefix_batch(self.ordered[0]))
        again = self.model(legacy_prefix_batch(self.ordered[0]))
        self.assertTrue(torch.equal(legacy.logits, again.logits))
        self.assertIsNone(legacy.selected_marker_offsets)
        self.assertIsNone(legacy.decision_cutoff_idx)


def dataset_dir() -> pathlib.Path | None:
    for root in (REPO_ROOT / "datasets", REPO_ROOT.parent.parent / "datasets"):
        directory = root / "extracted" / "2025"
        if directory.is_dir():
            return directory
    return None


class CpuBenchmarkTests(unittest.TestCase):
    """CPU-only measurement of prefix recomputation vs the shared pass."""

    def test_cpu_prefix_vs_shared_forward(self):
        directory = dataset_dir()
        if directory is None:
            self.skipTest("no local 2025 dataset for a real-stream CPU benchmark")
        stream = None
        for path in sorted(directory.glob("*.mjson"))[:6]:
            data = path.read_bytes()
            if data.startswith(b"\x1f\x8b"):
                data = gzip.decompress(data)
            raw = [json.loads(line) for line in data.decode("utf-8").splitlines()
                   if line.strip()]
            try:
                samples = build_decisions(raw)
            except Exception:  # noqa: BLE001 - invalid source, try the next log
                continue
            candidate = max(streams_from_samples(samples),
                            key=lambda item: (len(item.cutoffs), item.length))
            if stream is None or len(candidate.cutoffs) > len(stream.cutoffs):
                stream = candidate
            if len(stream.cutoffs) >= 40:
                break
        if stream is None:
            self.skipTest("no buildable real log for the benchmark")
        torch.manual_seed(20261121)
        model = Policy(MVPConfig())
        model.eval()
        batch = encode_multi_decision_batch([stream], verify_prefix="structural")
        decisions = list(stream.decisions)
        prefix_tokens = sum(
            len(decision.events) for decision in decisions)
        shared_tokens = stream.length + len(decisions)
        with torch.no_grad():
            model.forward_multi(batch)  # warmup
            started = time.perf_counter()
            model.forward_multi(batch)
            shared_seconds = time.perf_counter() - started
            for decision in decisions:
                row = encode_decision(
                    decision, StreamMeta.from_header(stream.meta)).to_model_batch()
                model(row)
            started = time.perf_counter()
            for decision in decisions:
                row = encode_decision(
                    decision, StreamMeta.from_header(stream.meta)).to_model_batch()
                model(row)
            prefix_seconds = time.perf_counter() - started
        report = {
            "device": "cpu",
            "parameters": sum(p.numel() for p in model.parameters()),
            "streams": 1,
            "events": stream.length,
            "decisions": len(decisions),
            "prefix_tokens": prefix_tokens,
            "shared_tokens": shared_tokens,
            "token_ratio": prefix_tokens / shared_tokens,
            "prefix_forward_seconds": prefix_seconds,
            "shared_forward_seconds": shared_seconds,
            "prefix_tokens_per_second": prefix_tokens / prefix_seconds,
            "shared_tokens_per_second": shared_tokens / shared_seconds,
            "prefix_decisions_per_second": len(decisions) / prefix_seconds,
            "shared_decisions_per_second": len(decisions) / shared_seconds,
            "speedup": prefix_seconds / shared_seconds,
        }
        print(f"PR38_BENCH_JSON {json.dumps(report, sort_keys=True)}")
        self.assertGreater(report["token_ratio"], 1.0)
        self.assertTrue(all(value > 0 for value in (
            report["prefix_tokens_per_second"],
            report["shared_tokens_per_second"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
