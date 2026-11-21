#!/usr/bin/env python3
"""PR-03/04 VNext encoder + model tests (CPU, synthetic native decisions)."""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.decision import build_decisions  # noqa: E402
from riichiformer.schemas.action_v3 import ActionPhase  # noqa: E402
from riichiformer.vnext.contracts import VNextConfig  # noqa: E402
from riichiformer.vnext.model import (  # noqa: E402
    FactorizedTileEmbedding, VNextPolicy, count_parameters,
)
from riichiformer.vnext.observable import (  # noqa: E402
    VNextBatchBuilder, VNextEventStream, encode_stream, encode_visible_state,
)
from riichiformer.schemas.event_v3 import (  # noqa: E402
    ChiEvent, TileRef, UNKNOWN_TILE,
)
from riichiformer.schemas.decision_v3 import MeldKind, PublicMeld  # noqa: E402


def raw_long() -> list[dict]:
    hand0 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
             "1p", "2p", "3p", "4p"]
    hand1 = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
             "E", "S", "W", "N"]
    hand2 = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
             "7m", "8m", "9m", "1p"]
    hand3 = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
             "1s", "2s", "3s", "4s", "5s"]
    return [
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1p",
         "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
         "scores": [25000] * 4, "tehais": [hand0, hand1, hand2, hand3]},
        {"type": "tsumo", "actor": 0, "pai": "5mr"},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        {"type": "end_kyoku"}, {"type": "end_game"},
    ]


def small_config(**kwargs) -> VNextConfig:
    base = dict(d_model=64, n_layers=1, n_heads=4, ffn_hidden=128,
                cross_attn_heads=4, dropout=0.0, max_positions=64)
    base.update(kwargs)
    return VNextConfig(**base)


class StreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = build_decisions(raw_long())

    def test_bos_and_current_kyoku_only(self):
        stream = encode_stream(self.samples[0].input, match_id="m", pov=0)
        self.assertEqual(stream.n_tokens, 3)  # BOS + start_kyoku + tsumo
        self.assertEqual(stream.event_type_id[0], 16)
        self.assertEqual(stream.event_kyoku_pos, [0, 0, 1])
        self.assertEqual(stream.decisions[0][0].prefix_token_offset, 2)

    def test_same_prefix_two_phases(self):
        sample = self.samples[0]
        stream = encode_stream(sample.input, match_id="m")
        kan_mask = tuple(i < 37 and sample.input.legal_mask[i] for i in range(46))
        # A legal KAN_SELECTION decision needs at least one candidate.
        if not any(kan_mask):
            self.skipTest("synthetic hand has no tile candidate")
        kan = dataclasses.replace(
            sample.input, phase=ActionPhase.KAN_SELECTION, legal_mask=kan_mask)
        stream.add_decision(kan, decision_id="kan", decision_order=1)
        batch = stream.to_batch()
        self.assertEqual(batch.prefix_token_offsets.tolist(), [2, 2])
        self.assertEqual(batch.phases.tolist(), [0, 2])
        self.assertEqual(int(batch.legal_mask.sum()),
                         sum(sample.input.legal_mask) + sum(kan_mask))

    def test_visible_state_invariants(self):
        state = encode_visible_state(self.samples[0].input)
        self.assertEqual(tuple(state.tile_features.shape), (34, 26))
        self.assertEqual(tuple(state.global_features.shape), (38,))
        self.assertEqual(float(state.tile_features[4, 0]), 0.5)
        self.assertEqual(float(state.tile_features[4, 6]), 1.0)  # own aka 5m
        self.assertEqual(float(state.tile_features[13, 6]), 0.0)  # not 5p row
        self.assertEqual(float(state.tile_features[4, 0]), 0.5)

    def test_opponent_ankan_is_not_counted_as_public_meld(self):
        sample = self.samples[0]
        players = list(sample.input.players_public)
        players[1] = dataclasses.replace(
            players[1],
            melds=(PublicMeld(MeldKind.ANKAN,
                              (TileRef(0), TileRef(0), TileRef(0), TileRef(0)),
                              from_rel=4),))
        mutated = dataclasses.replace(sample.input, players_public=tuple(players))
        state = encode_visible_state(mutated)
        # Channel 21 is opponent_meld_shimocha_over_4; ankan is rule-dependent
        # and must not be exposed as public meld tiles.
        self.assertEqual(float(state.tile_features[0, 21]), 0.0)

    def test_chi_consumed_tiles_are_preserved(self):
        stream = VNextEventStream(match_id="m", kyoku_id=0, pov=0)
        event = ChiEvent(
            seq_idx=0, kyoku_idx=0, actor_rel=1, target_rel=0,
            tile=TileRef(2), consumed=(TileRef(0), TileRef(1),
                                       UNKNOWN_TILE, UNKNOWN_TILE))
        stream.append_event(event)
        self.assertEqual(stream.event_meld_tile[-1], (0, 1, 34))
        self.assertEqual(stream.event_meld_aka[-1], (0, 0, 0))

    def test_batch_builder_two_streams(self):
        a = encode_stream(self.samples[0].input, match_id="a")
        b = encode_stream(self.samples[1].input, match_id="b")
        batch = VNextBatchBuilder([a, b]).to_batch()
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.n_decisions, 2)
        self.assertTrue(bool(batch.event_valid_mask[0, :a.n_tokens].all()))


class MultiKyokuTests(unittest.TestCase):
    def test_second_kyoku_does_not_inherit_old_river(self):
        h0 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
              "1p", "2p", "3p", "4p"]
        h1 = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
              "E", "S", "W", "N"]
        h2 = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
              "7m", "8m", "9m", "1p"]
        h3 = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
              "1s", "2s", "3s", "4s", "5s"]
        raw = [
            {"type": "start_game", "names": ["a", "b", "c", "d"]},
            {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1p",
             "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
             "scores": [25000] * 4, "tehais": [h0, h1, h2, h3]},
            {"type": "tsumo", "actor": 1, "pai": "1m"},
            {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": True},
            {"type": "end_kyoku"},
            {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1p",
             "kyoku": 2, "honba": 0, "kyotaku": 0, "oya": 1,
             "scores": [25000] * 4, "tehais": [h0, h1, h2, h3]},
            {"type": "tsumo", "actor": 1, "pai": "9m"},
            {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
            {"type": "tsumo", "actor": 0, "pai": "9m"},
            {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
            {"type": "end_kyoku"}, {"type": "end_game"},
        ]
        samples = build_decisions(raw)
        second = [sample for sample in samples
                  if sample.input.events[-1].kyoku_idx == 1][-1]
        state = encode_visible_state(second.input)
        # 1m kind 0 was discarded by the POV's kamicha in kyoku 1; kyoku 2 must not
        # inherit it, while 9m (kind 8) was discarded in kyoku 2.
        self.assertEqual(float(state.tile_features[0, 14]), 0.0)
        self.assertEqual(float(state.tile_features[8, 14]), 0.25)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        samples = build_decisions(raw_long())
        cls.stream = encode_stream(samples[0].input, match_id="m")
        cls.stream.add_decision(samples[1].input, decision_id="d1",
                                decision_order=1)
        cls.batch = cls.stream.to_batch()
        cls.model = VNextPolicy(small_config()).eval()

    def test_forward_shapes_and_mask(self):
        output = self.model(self.batch)
        self.assertEqual(tuple(output.logits.shape), (2, 46))
        self.assertEqual(tuple(output.rank_logits.shape), (2, 4))
        self.assertEqual(int(torch.isinf(output.masked_logits[~self.batch.legal_mask]).sum()),
                         int((~self.batch.legal_mask).sum()))
        self.assertTrue(torch.allclose(
            torch.logsumexp(output.logprobs, dim=-1),
            torch.zeros(2), atol=1e-5, rtol=1e-5))
        self.assertTrue(bool(torch.isfinite(
            output.masked_logits[self.batch.legal_mask]).all()))

    def test_future_token_does_not_change_past_decision(self):
        changed = dataclasses.replace(
            self.batch,
            event_tile_kind=self.batch.event_tile_kind.clone())
        # token 4 is after the first decision's prefix token 2.
        changed.event_tile_kind[0, 4] = 33
        with torch.no_grad():
            before = self.model(self.batch).logits[0]
            after = self.model(changed).logits[0]
        self.assertTrue(torch.allclose(before, after, atol=1e-6, rtol=1e-6))

    def test_shared_batch_matches_single_decision(self):
        single = self.batch.select_decisions(torch.tensor([0]))
        with torch.no_grad():
            shared = self.model(self.batch).logits[0]
            alone = self.model(single).logits[0]
        self.assertTrue(torch.allclose(shared, alone, atol=1e-5, rtol=1e-5))

    def test_decision_order_does_not_change_rows(self):
        reordered = self.batch.select_decisions(torch.tensor([1, 0]))
        with torch.no_grad():
            original = self.model(self.batch).logits
            flipped = self.model(reordered).logits
        self.assertTrue(torch.allclose(original, flipped.flip(0), atol=1e-6,
                                       rtol=1e-6))

    def test_cross_attention_ablation_and_linear_head(self):
        for config in (small_config(cross_attn_layers=0),
                       small_config(head_type="linear46"),
                       small_config(model_mode="state_only")):
            model = VNextPolicy(config).eval()
            with torch.no_grad():
                out = model(self.batch)
            self.assertEqual(tuple(out.logits.shape), (2, 46))
            out.validate()

    def test_factorized_and_aka_are_distinct(self):
        emb = FactorizedTileEmbedding(64, embedding="factorized")
        normal = emb(torch.tensor([4]), torch.tensor([0]))
        aka = emb(torch.tensor([4]), torch.tensor([1]))
        self.assertFalse(torch.allclose(normal, aka))
        identity = FactorizedTileEmbedding(64, embedding="identity")
        self.assertEqual(identity(torch.tensor([4]), torch.tensor([0])).shape,
                         normal.shape)

    def test_backward_reaches_actor(self):
        model = VNextPolicy(small_config())
        out = model(self.batch)
        loss = out.logprobs[self.batch.legal_mask].sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in grads))

    def test_detach_critic_cuts_actor_gradient(self):
        model = VNextPolicy(small_config(critic_grad_mode="detach_actor"))
        output = model(self.batch)
        output.rank_logits.sum().backward()
        actor_grads = [p.grad for name, p in model.named_parameters()
                       if not name.startswith("critic") and p.grad is not None]
        critic_grads = [p.grad for name, p in model.named_parameters()
                        if name.startswith("critic") and p.grad is not None]
        self.assertFalse(any(float(g.abs().sum()) > 0 for g in actor_grads))
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in critic_grads))

    def test_eval_rope_cache_does_not_break_training_autograd(self):
        # A cache first built inside inference_mode() used to retain the
        # inference flag; switching the same model to train() then failed in
        # RoPE backward.  This is the exact sequence used by profile/smoke code.
        model = VNextPolicy(small_config()).eval()
        with torch.inference_mode():
            model(self.batch)
        model.train()
        output = model(self.batch)
        output.logits.sum().backward()
        self.assertTrue(any(parameter.grad is not None
                            for parameter in model.parameters()))

    def test_bos_only_prefix_works(self):
        from riichiformer.vnext.contracts import (
            BOS_EVENT_ID, PAD_EVENT_ID, TILE_FEATURE_DIM, VNextModelBatch,
            VNextVisibleState,
        )
        state = VNextVisibleState.empty()
        batch = VNextModelBatch(
            event_type_id=torch.tensor([[BOS_EVENT_ID]], dtype=torch.long),
            event_actor_rel=torch.tensor([[4]], dtype=torch.long),
            event_tile_kind=torch.tensor([[34]], dtype=torch.long),
            event_aka=torch.tensor([[0]], dtype=torch.long),
            event_tsumogiri=torch.tensor([[0]], dtype=torch.long),
            event_kyoku_pos=torch.tensor([[0]], dtype=torch.long),
            event_meld_tile=torch.tensor([[[34, 34, 34]]], dtype=torch.long),
            event_meld_aka=torch.tensor([[[0, 0, 0]]], dtype=torch.long),
            event_valid_mask=torch.tensor([[True]]),
            decision_to_sequence=torch.tensor([0], dtype=torch.long),
            prefix_token_offsets=torch.tensor([0], dtype=torch.long),
            phases=torch.tensor([0], dtype=torch.long),
            x_tile=state.tile_features.unsqueeze(0),
            x_global=state.global_features.unsqueeze(0),
            candidate_features=state.candidate_features.unsqueeze(0),
            legal_mask=state.legal_mask.unsqueeze(0),
        )
        output = self.model(batch)
        self.assertEqual(tuple(output.logits.shape), (1, 46))
        output.validate()

    def test_autocast_policy_output_is_fp32(self):
        if not hasattr(torch, "autocast"):
            self.skipTest("autocast unavailable")
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = self.model(self.batch)
        self.assertEqual(output.logits.dtype, torch.float32)
        self.assertEqual(output.masked_logits.dtype, torch.float32)
        output.validate()

    def test_parameter_count_is_reported(self):
        self.assertGreater(count_parameters(self.model), 100_000)


if __name__ == "__main__":
    unittest.main()
