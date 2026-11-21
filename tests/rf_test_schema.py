#!/usr/bin/env python3
"""PR-08 schema v3 tests: event allowlist, 46-action contract, target isolation, hashes.

All tests are deterministic and use only synthetic data (no datasets, no network).
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, fields, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.schemas import (  # noqa: E402
    action_v3,
    decision_v3,
    event_v3,
    schema_v3,
    target_v3,
)


def hand() -> list[str]:
    return ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
            "1p", "2p", "3p", "4p"]


def raw_start_game(**overrides) -> dict:
    raw = {"type": "start_game", "names": ["p0", "p1", "p2", "p3"],
           "aka_flag": True, "kyoku_first": 0, "id": 123, "seed": [1, 2],
           "log": "https://example.invalid/log"}
    raw.update(overrides)
    return raw


def raw_start_kyoku(**overrides) -> dict:
    raw = {
        "type": "start_kyoku", "bakaze": "E", "dora_marker": "1p",
        "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [hand(), hand(), hand(), hand()],
    }
    raw.update(overrides)
    return raw


def build_input(*, phase=action_v3.ActionPhase.SELF_TURN,
                mask_indices=(0, 4), before_event_idx=2,
                events=None, tile_state=None, players=None, global_state=None):
    if events is None:
        events = event_v3.events_from_mjai(
            [raw_start_game(), raw_start_kyoku()], pov=0
        )
    if tile_state is None:
        own_counts = [0] * 34
        for text in hand():
            own_counts[event_v3.tile_from_text(text).kind] += 1
        zones = ([0] * 34, [0] * 34, [0] * 34)
        zones[2][event_v3.tile_from_text("1p").kind] = 1
        public_counts = [sum(zone[kind] for zone in zones) for kind in range(34)]
        tile_state = decision_v3.VisibleTileState(
            own_counts=tuple(own_counts), own_aka=(0, 0, 0),
            public_counts=tuple(public_counts), public_aka=(0, 0, 0),
            public_counts_by_zone=tuple(tuple(zone) for zone in zones),
            dora_indicators=(event_v3.tile_from_text("1p"),),
            ruleset_id=decision_v3.RulesetId.TENHOU_4P_RANKED,
        )
    if players is None:
        players = tuple(
            decision_v3.PlayerPublicState(
                seat_rel=seat, score=25000,
                wind=event_v3.tile_from_text(("E", "S", "W", "N")[seat]),
                is_oya=seat == 0,
                riichi=decision_v3.RiichiState.NONE, river_count=0, melds=(),
            )
            for seat in range(4)
        )
    if global_state is None:
        global_state = decision_v3.GlobalPublicState(
            bakaze=event_v3.tile_from_text("E"), kyoku=1, honba=0, kyotaku=0,
            tiles_left=69, self_wind=event_v3.tile_from_text("E"), oya_rel=0,
            ruleset_id=decision_v3.RulesetId.TENHOU_4P_RANKED,
            ruleset_deviation=(), source_kind=decision_v3.SourceKind.HUMAN_LOG,
        )
    return decision_v3.DecisionInput(
        events=events, tile_state=tile_state, players_public=players,
        global_public=global_state,
        legal_mask=action_v3.mask_from_indices(mask_indices),
        phase=phase, before_event_idx=before_event_idx,
    )


class TileAndActionTests(unittest.TestCase):
    def test_tile_kind_and_aka_identity(self):
        normal_5m = event_v3.TileRef(4, None)
        aka_5m = event_v3.TileRef(4, 0)
        self.assertNotEqual(normal_5m, aka_5m)
        self.assertEqual(aka_5m.text, "5mr")
        self.assertEqual(normal_5m.text, "5m")
        self.assertEqual(event_v3.tile_from_text("5pr"), event_v3.TileRef(13, 1))
        self.assertEqual(event_v3.tile_from_text("5sr"), event_v3.TileRef(22, 2))
        self.assertEqual(event_v3.tile_from_text("?"), event_v3.UNKNOWN_TILE)
        self.assertTrue(event_v3.UNKNOWN_TILE.is_unknown)
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.TileRef(4, 1)  # wrong colour for 5m
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.TileRef(0, 0)  # aka not on 5m/5p/5s
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.TileRef(34, 0)  # UNKNOWN cannot be aka
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.tile_from_text("5z")
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.tile_from_text("1mr")

    def test_action_map_shape_and_hash(self):
        self.assertEqual(action_v3.ACTION_SPACE, 46)
        self.assertEqual([spec.index for spec in action_v3.ACTION_MAP], list(range(46)))
        self.assertEqual(len({spec.name for spec in action_v3.ACTION_MAP}), 46)
        self.assertRegex(action_v3.ACTION_MAP_HASH, r"^[0-9a-f]{64}$")
        first = action_v3.canonical_action_map()
        second = action_v3.canonical_action_map()
        self.assertEqual(first, second)
        self.assertEqual(
            action_v3.ACTION_MAP_HASH,
            hashlib.sha256(action_v3.canonical_json(first).encode()).hexdigest(),
        )

    def test_action_phase_dual_semantics(self):
        discard = action_v3.decode_action(4, action_v3.ActionPhase.SELF_TURN)
        kan = action_v3.decode_action(4, action_v3.ActionPhase.KAN_SELECTION)
        self.assertEqual(discard.semantic, "discard")
        self.assertEqual(kan.semantic, "kan_tile")
        self.assertEqual(discard.tile_kind, 4)
        self.assertEqual(kan.tile_kind, 4)
        aka_discard = action_v3.decode_action(34, action_v3.ActionPhase.SELF_TURN)
        aka_kan = action_v3.decode_action(34, action_v3.ActionPhase.KAN_SELECTION)
        self.assertEqual((aka_discard.tile_kind, aka_discard.aka), (4, 0))
        self.assertEqual((aka_kan.tile_kind, aka_kan.aka), (4, 0))
        self.assertEqual(aka_discard.semantic, "discard_aka")
        self.assertEqual(aka_kan.semantic, "kan_tile_aka")

    def test_chi_and_special_actions(self):
        for index, semantic, chi in (
            (38, "chi_low", action_v3.ChiType.LOW),
            (39, "chi_mid", action_v3.ChiType.MID),
            (40, "chi_high", action_v3.ChiType.HIGH),
        ):
            decoded = action_v3.decode_action(index, action_v3.ActionPhase.CALL_RESPONSE)
            self.assertEqual(decoded.semantic, semantic)
            self.assertEqual(decoded.chi_type, chi)
        self.assertEqual(action_v3.decode_action(37, action_v3.ActionPhase.SELF_TURN).semantic,
                         "riichi")
        self.assertEqual(action_v3.decode_action(41, action_v3.ActionPhase.CALL_RESPONSE).semantic,
                         "pon")
        self.assertEqual(action_v3.decode_action(42, action_v3.ActionPhase.SELF_TURN).semantic,
                         "kan_decide")
        self.assertEqual(action_v3.decode_action(42, action_v3.ActionPhase.CALL_RESPONSE).semantic,
                         "kan_decide")
        self.assertEqual(action_v3.decode_action(43, action_v3.ActionPhase.SELF_TURN).semantic,
                         "tsumo_agari")
        self.assertEqual(action_v3.decode_action(43, action_v3.ActionPhase.CALL_RESPONSE).semantic,
                         "ron_agari")
        self.assertEqual(action_v3.decode_action(44, action_v3.ActionPhase.SELF_TURN).semantic,
                         "ryukyoku")
        self.assertEqual(action_v3.decode_action(45, action_v3.ActionPhase.CALL_RESPONSE).semantic,
                         "pass")

    def test_action_phase_illegal_combinations(self):
        for index, phase in (
            (4, action_v3.ActionPhase.CALL_RESPONSE),
            (37, action_v3.ActionPhase.CALL_RESPONSE),
            (38, action_v3.ActionPhase.SELF_TURN),
            (44, action_v3.ActionPhase.CALL_RESPONSE),
            (45, action_v3.ActionPhase.SELF_TURN),
            (37, action_v3.ActionPhase.KAN_SELECTION),
        ):
            with self.assertRaises(action_v3.IllegalActionForPhase):
                action_v3.decode_action(index, phase)
        with self.assertRaises(action_v3.IllegalActionIndex):
            action_v3.decode_action(46, action_v3.ActionPhase.SELF_TURN)
        with self.assertRaises(action_v3.IllegalActionIndex):
            action_v3.decode_action(-1, action_v3.ActionPhase.SELF_TURN)

    def test_mask_validation(self):
        valid = action_v3.mask_from_indices([0, 37, 43])
        self.assertEqual(action_v3.validate_mask_for_phase(valid, "SELF_TURN"), valid)
        self.assertEqual(action_v3.mask_indices(valid), (0, 37, 43))
        with self.assertRaises(action_v3.MaskShapeError):
            action_v3.validate_mask_for_phase((False,) * 46, "SELF_TURN")
        with self.assertRaises(action_v3.MaskShapeError):
            action_v3.validate_mask_for_phase((False,) * 45, "SELF_TURN")
        with self.assertRaises(action_v3.IllegalActionForPhase):
            action_v3.validate_mask_for_phase(
                action_v3.mask_from_indices([45]), "SELF_TURN")
        with self.assertRaises(action_v3.IllegalActionForPhase):
            action_v3.validate_mask_for_phase(
                action_v3.mask_from_indices([0]), "CALL_RESPONSE")
        with self.assertRaises(action_v3.IllegalActionForPhase):
            action_v3.validate_mask_for_phase(
                action_v3.mask_from_indices([37]), "KAN_SELECTION")

    def test_kan_selection_accepts_0_to_36_only(self):
        for index in range(37):
            decoded = action_v3.decode_action(index, action_v3.ActionPhase.KAN_SELECTION)
            self.assertIn(decoded.semantic, ("kan_tile", "kan_tile_aka"))


class EventRoundTripTests(unittest.TestCase):
    def _roundtrip(self, raw, pov=0, kyoku_idx=0):
        event = event_v3.event_from_mjai(raw, pov=pov, seq_idx=7, kyoku_idx=kyoku_idx)
        data = event.to_dict()
        self.assertEqual(event_v3.event_from_dict(data), event)
        return event, data

    def test_all_event_types_roundtrip(self):
        raws = [
            raw_start_game(),
            raw_start_kyoku(),
            {"type": "tsumo", "actor": 0, "pai": "5mr"},
            {"type": "tsumo", "actor": 2, "pai": "?"},
            {"type": "dahai", "actor": 1, "pai": "3s", "tsumogiri": False},
            {"type": "chi", "actor": 0, "target": 3, "pai": "2m",
             "consumed": ["1m", "3m"]},
            {"type": "pon", "actor": 1, "target": 0, "pai": "5m",
             "consumed": ["5m", "5m"]},
            {"type": "daiminkan", "actor": 1, "target": 0, "pai": "5m",
             "consumed": ["5m", "5m", "5m"]},
            {"type": "kakan", "actor": 1, "pai": "5mr",
             "consumed": ["5m", "5m", "5m"]},
            {"type": "ankan", "actor": 0, "consumed": ["5m", "5m", "5m", "5m"]},
            {"type": "reach", "actor": 2},
            {"type": "reach_accepted", "actor": 2},
            {"type": "dora", "dora_marker": "3p"},
            {"type": "hora", "actor": 1, "target": 0,
             "deltas": [-2900, 4900, 0, 0], "ura_markers": ["4m"]},
            {"type": "ryukyoku", "deltas": [-1500, 1500, 1500, -1500]},
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ]
        seen = set()
        for raw in raws:
            event, data = self._roundtrip(raw)
            seen.add(event.EVENT_TYPE)
            self.assertRegex(data["timing"], r"^[A-Z_]+$")
        self.assertEqual(seen, set(event_v3.EVENT_CLASSES))
        self.assertEqual(len(event_v3.EVENT_CLASSES), 16)

    def test_unknown_event_type_raises(self):
        with self.assertRaises(event_v3.UnknownEventType):
            event_v3.event_from_mjai({"type": "not_a_mjai_event"}, pov=0,
                                     seq_idx=0, kyoku_idx=0)
        with self.assertRaises(event_v3.UnknownEventType):
            event_v3.event_from_dict({"type": "not_a_mjai_event"})
        with self.assertRaises(event_v3.UnknownEventType):
            event_v3.events_from_mjai([{"type": "wat"}], pov=0)

    def test_from_dict_rejects_extra_and_missing_fields(self):
        event = event_v3.event_from_mjai(
            {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True},
            pov=0, seq_idx=0, kyoku_idx=0)
        data = event.to_dict()
        extra = dict(data)
        extra["names"] = ["leak"]
        with self.assertRaises(event_v3.SchemaFieldError):
            event_v3.event_from_dict(extra)
        missing = dict(data)
        missing.pop("tile")
        with self.assertRaises(event_v3.SchemaFieldError):
            event_v3.event_from_dict(missing)
        wrong_timing = dict(data)
        wrong_timing["timing"] = "SETTLEMENT"
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.event_from_dict(wrong_timing)

    def test_start_game_drops_private_fields(self):
        event, data = self._roundtrip(raw_start_game())
        self.assertIsInstance(event, event_v3.StartGameEvent)
        dumped = json.dumps(data)
        for forbidden in ("names", "player_names", "id", "seed", "log"):
            self.assertNotIn(forbidden, data)
            self.assertNotIn(f'"{forbidden}"', dumped)
        self.assertEqual(set(f.name for f in fields(event)),
                         {"seq_idx", "kyoku_idx", "aka_flag", "kyoku_first"})

    def test_start_kyoku_stores_only_own_hand(self):
        tehais = [
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
             "1p", "2p", "3p", "4p"],
            ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
             "1m", "9m", "1p", "9p"],
            ["E", "S", "W", "N", "P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m"],
            ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s"],
        ]
        event, data = self._roundtrip(raw_start_kyoku(tehais=tehais), pov=1)
        self.assertEqual(event.own_haipai, tuple(
            event_v3.tile_from_text(text) for text in tehais[1]))
        self.assertNotIn("tehais", data)
        dumped = json.dumps(data)
        for other_hand in tehais[0], tehais[2], tehais[3]:
            self.assertNotIn(json.dumps([event_v3.tile_from_text(t).to_dict()
                                         for t in other_hand])[1:-1], dumped)
        self.assertEqual(event.scores_rel, (25000, 25000, 25000, 25000))

    def test_start_kyoku_scores_rotate_with_pov(self):
        raw = raw_start_kyoku(scores=[22000, 24000, 26000, 28000], oya=2)
        event = event_v3.event_from_mjai(raw, pov=1, seq_idx=0, kyoku_idx=0)
        self.assertEqual(event.scores_rel, (24000, 26000, 28000, 22000))
        self.assertEqual(event.oya_rel, 1)

    def test_opponent_tsumo_unknown_own_tsumo_known(self):
        opponent = event_v3.event_from_mjai(
            {"type": "tsumo", "actor": 2, "pai": "7s"}, pov=0, seq_idx=0, kyoku_idx=0)
        own = event_v3.event_from_mjai(
            {"type": "tsumo", "actor": 0, "pai": "7s"}, pov=0, seq_idx=0, kyoku_idx=0)
        self.assertTrue(opponent.tile.is_unknown)
        self.assertEqual(own.tile, event_v3.tile_from_text("7s"))

    def test_opponent_ankan_hidden_by_default_and_revealable(self):
        raw = {"type": "ankan", "actor": 2, "consumed": ["5m", "5m", "5m", "5m"]}
        hidden = event_v3.event_from_mjai(raw, pov=0, seq_idx=0, kyoku_idx=0)
        self.assertTrue(all(tile.is_unknown for tile in hidden.consumed))
        revealed = event_v3.event_from_mjai(
            raw, pov=0, seq_idx=0, kyoku_idx=0, reveal_opponent_ankan=True)
        self.assertTrue(all(tile == event_v3.tile_from_text("5m")
                            for tile in revealed.consumed))
        own = event_v3.event_from_mjai(
            {"type": "ankan", "actor": 0, "consumed": ["5m", "5m", "5m", "5m"]},
            pov=0, seq_idx=0, kyoku_idx=0)
        self.assertTrue(all(not tile.is_unknown for tile in own.consumed))

    def test_meld_padding_and_validation(self):
        chi, _ = self._roundtrip({"type": "chi", "actor": 0, "target": 3,
                                  "pai": "2m", "consumed": ["1m", "3m"]})
        self.assertEqual(chi.consumed[2:], (event_v3.UNKNOWN_TILE,) * 2)
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.event_from_mjai({"type": "chi", "actor": 0, "target": 0,
                                     "pai": "2m", "consumed": ["1m", "3m"]},
                                    pov=0, seq_idx=0, kyoku_idx=0)
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.event_from_mjai({"type": "chi", "actor": 0, "target": 3,
                                     "pai": "5m", "consumed": ["1m", "3m"]},
                                    pov=0, seq_idx=0, kyoku_idx=0)
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.event_from_mjai({"type": "pon", "actor": 1, "target": 0,
                                     "pai": "5m", "consumed": ["5p", "5m"]},
                                    pov=0, seq_idx=0, kyoku_idx=0)
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.AnkanEvent(seq_idx=0, kyoku_idx=0, actor_rel=0,
                                consumed=(event_v3.tile_from_text("5m"),
                                          event_v3.tile_from_text("5m"),
                                          event_v3.UNKNOWN_TILE,
                                          event_v3.tile_from_text("5m")))

    def test_dahai_rejects_unknown_tile(self):
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.event_from_mjai(
                {"type": "dahai", "actor": 0, "pai": "?", "tsumogiri": True},
                pov=0, seq_idx=0, kyoku_idx=0)

    def test_hora_deltas_are_verbatim(self):
        event, _ = self._roundtrip(
            {"type": "hora", "actor": 0, "target": 3,
             "deltas": [-2900, 0, 0, 2900], "ura_markers": ["5pr"]})
        self.assertEqual(event.deltas, (-2900, 0, 0, 2900))
        self.assertEqual(event.ura_markers[0], event_v3.TileRef(13, 1))

    def test_settlement_deltas_order_and_rotation(self):
        absolute = (1000, -1000, 200, -200)
        expected = {
            1: (-1000, 200, -200, 1000),
            2: (200, -200, 1000, -1000),
            3: (-200, 1000, -1000, 200),
        }
        self.assertEqual(event_v3.rotate_deltas_to_relative(absolute, 0), absolute)
        self.assertEqual(event_v3.rotate_deltas_to_absolute(absolute, 0), absolute)
        for pov, relative in expected.items():
            self.assertEqual(
                event_v3.rotate_deltas_to_relative(absolute, pov), relative)
            self.assertEqual(
                event_v3.rotate_deltas_to_absolute(relative, pov), absolute)
            hora = event_v3.event_from_mjai(
                {"type": "hora", "actor": 0, "target": 0,
                 "deltas": list(absolute), "ura_markers": []},
                pov=pov, seq_idx=0, kyoku_idx=0)
            self.assertEqual(hora.deltas_order, event_v3.DELTAS_ORDER_ABSOLUTE)
            self.assertEqual(hora.to_dict()["deltas_order"], "absolute")
            rotated = replace(
                hora, deltas=relative,
                deltas_order=event_v3.DELTAS_ORDER_RELATIVE)
            self.assertEqual(rotated.deltas, relative)
            self.assertEqual(rotated.deltas_order, "relative")
            self.assertEqual(rotated.to_dict()["deltas_order"], "relative")
            self.assertEqual(
                event_v3.rotate_deltas_to_absolute(rotated.deltas, pov), absolute)
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.rotate_deltas_to_relative((1, 2, 3), 0)
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.rotate_deltas_to_relative((1, 2, 3, 4), 4)
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.rotate_deltas_to_absolute(("a", 2, 3, 4), 1)
        hora = event_v3.event_from_mjai(
            {"type": "hora", "actor": 0, "target": 0,
             "deltas": list(absolute), "ura_markers": []},
            pov=0, seq_idx=0, kyoku_idx=0)
        with self.assertRaises(event_v3.SchemaValueError):
            replace(hora, deltas_order="sideways")
        ryukyoku = event_v3.event_from_mjai(
            {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
            pov=0, seq_idx=0, kyoku_idx=0)
        self.assertEqual(ryukyoku.deltas_order, "absolute")
        with self.assertRaises(event_v3.SchemaValueError):
            replace(ryukyoku, deltas_order="relative-ish")

    def test_settlement_deltas_required(self):
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.event_from_mjai(
                {"type": "hora", "actor": 0, "target": 0, "ura_markers": []},
                pov=0, seq_idx=0, kyoku_idx=0)
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.event_from_mjai(
                {"type": "ryukyoku"}, pov=0, seq_idx=0, kyoku_idx=0)

    def test_start_kyoku_aka_count(self):
        bad_hand = ["5mr", "5mr"] + hand()[2:]
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.event_from_mjai(
                raw_start_kyoku(tehais=[bad_hand, hand(), hand(), hand()]),
                pov=0, seq_idx=0, kyoku_idx=0)

    def test_start_kyoku_score_invariant(self):
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.event_from_mjai(
                raw_start_kyoku(kyotaku=1), pov=0, seq_idx=0, kyoku_idx=0)

    def test_event_type_ids_and_specs(self):
        specs = event_v3.event_type_specs()
        self.assertEqual(set(specs), set(event_v3.EVENT_CLASSES))
        self.assertEqual(sorted(event_v3.EVENT_TYPE_IDS.values()), list(range(16)))
        for type_name, cls in event_v3.EVENT_CLASSES.items():
            spec = specs[type_name]
            self.assertEqual(spec["type_id"], cls.TYPE_ID)
            self.assertEqual(spec["timing"], cls.TIMING.value)
            spec_fields = {item["name"] for item in spec["fields"]}
            self.assertEqual(spec_fields, set(cls.FIELDS))
            self.assertEqual(spec_fields, set(cls.FIELD_VISIBILITY))
            for item in spec["fields"]:
                self.assertIn(item["visibility"], event_v3.VISIBILITIES)

    def test_events_from_mjai_indices(self):
        raw = [raw_start_game(), raw_start_kyoku(), {"type": "end_kyoku"},
               raw_start_kyoku(), {"type": "end_kyoku"}, {"type": "end_game"}]
        events = event_v3.events_from_mjai(raw, pov=0)
        self.assertEqual([event.seq_idx for event in events], [0, 1, 2, 3, 4, 5])
        self.assertEqual([event.kyoku_idx for event in events], [-1, 0, 0, 1, 1, 1])

    def test_event_before_start_kyoku_rejected(self):
        with self.assertRaises(event_v3.SchemaValueError):
            event_v3.events_from_mjai([{"type": "tsumo", "actor": 0, "pai": "1m"}], pov=0)


class DecisionInputTests(unittest.TestCase):
    def test_valid_input_and_forward_dict(self):
        decision = build_input()
        forward = decision.to_forward_dict()
        self.assertEqual(set(forward), set(decision_v3.DECISION_INPUT_FIELDS))
        self.assertEqual(forward["phase"], "SELF_TURN")
        self.assertEqual(forward["legal_mask"][0], True)
        self.assertEqual(forward["legal_mask"][4], True)
        decision.assert_forward_safe()
        dumped = json.dumps(forward)
        for forbidden in decision_v3.FORBIDDEN_FORWARD_KEYS:
            self.assertNotIn(f'"{forbidden}"', dumped)
        self.assertEqual(decision.legal_actions, (0, 4))

    def test_input_has_no_target_fields(self):
        input_fields = {field.name for field in fields(decision_v3.DecisionInput)}
        self.assertEqual(input_fields, set(decision_v3.DECISION_INPUT_FIELDS))
        for forbidden in ("expert_action", "supervision_valid", "censor_reason",
                          "rank_final", "oracle_labels"):
            self.assertNotIn(forbidden, input_fields)

    def test_target_cannot_enter_forward_input(self):
        with self.assertRaises(TypeError):
            build_input(rank_final=1)

        def forward(model_input):
            if not isinstance(model_input, decision_v3.DecisionInput):
                raise TypeError("forward() only accepts DecisionInput")
            return model_input.to_forward_dict()

        target = target_v3.DecisionTarget(expert_action=0, supervision_valid=True)
        with self.assertRaises(TypeError):
            forward(target)

    def test_rogue_event_subclass_rejected(self):
        @dataclass(frozen=True)
        class RogueStartKyoku(event_v3.StartKyokuEvent):
            extra: str = "leak"

        @dataclass(frozen=True)
        class RogueStartKyokuNoExtra(event_v3.StartKyokuEvent):
            pass

        decision = build_input()
        base_event = decision.events[1]
        kwargs = {field.name: getattr(base_event, field.name)
                  for field in fields(base_event)}
        for rogue_cls in (RogueStartKyoku, RogueStartKyokuNoExtra):
            extra = {"extra": "leak"} if rogue_cls is RogueStartKyoku else {}
            rogue = rogue_cls(**kwargs, **extra)
            events = (decision.events[0], rogue)
            with self.assertRaises(decision_v3.DecisionSchemaError):
                build_input(events=events, before_event_idx=2)

    def test_target_leak_guard(self):
        with self.assertRaises(decision_v3.TargetLeakError):
            decision_v3.assert_forward_safe({"nested": [{"rank_final": 1}]})
        with self.assertRaises(decision_v3.TargetLeakError):
            decision_v3.assert_forward_safe({"names": ["p0"]})

    def test_future_and_unsorted_events_rejected(self):
        events = event_v3.events_from_mjai(
            [raw_start_game(), raw_start_kyoku(),
             {"type": "tsumo", "actor": 0, "pai": "1m"}], pov=0)
        with self.assertRaises(decision_v3.DecisionSchemaError):
            build_input(events=events, before_event_idx=2)
        with self.assertRaises(decision_v3.DecisionSchemaError):
            build_input(events=tuple(reversed(events[:2])), before_event_idx=2)

    def test_empty_events_and_before_index(self):
        with self.assertRaises(decision_v3.DecisionSchemaError):
            build_input(events=(), before_event_idx=1)
        with self.assertRaises(decision_v3.DecisionSchemaError):
            build_input(before_event_idx=0)

    def test_mask_phase_mismatch(self):
        for phase, mask in (
            (action_v3.ActionPhase.SELF_TURN, (45,)),
            (action_v3.ActionPhase.CALL_RESPONSE, (0,)),
            (action_v3.ActionPhase.KAN_SELECTION, (37,)),
        ):
            with self.assertRaises(decision_v3.DecisionSchemaError):
                build_input(phase=phase, mask_indices=mask)
        with self.assertRaises(decision_v3.DecisionSchemaError):
            build_input(mask_indices=())
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.DecisionInput(
                events=build_input().events, tile_state=build_input().tile_state,
                players_public=build_input().players_public,
                global_public=build_input().global_public,
                legal_mask=(True,) * 45, phase=action_v3.ActionPhase.SELF_TURN,
                before_event_idx=2,
            )

    def test_players_and_global_consistency(self):
        players = list(build_input().players_public)
        players[1] = decision_v3.PlayerPublicState(
            seat_rel=1, score=25000, wind=event_v3.tile_from_text("S"),
            is_oya=True, riichi=decision_v3.RiichiState.NONE,
            river_count=0, melds=())
        with self.assertRaises(decision_v3.DecisionSchemaError):
            build_input(players=tuple(players))
        # wrong dealer: no oya at all
        no_oya = tuple(
            decision_v3.PlayerPublicState(
                seat_rel=seat, score=25000,
                wind=event_v3.tile_from_text(("E", "S", "W", "N")[seat]),
                is_oya=False,
                riichi=decision_v3.RiichiState.NONE, river_count=0, melds=())
            for seat in range(4)
        )
        with self.assertRaises(decision_v3.DecisionSchemaError):
            build_input(players=no_oya)

    def test_score_total_invariant(self):
        bad_global = decision_v3.GlobalPublicState(
            bakaze=event_v3.tile_from_text("E"), kyoku=1, honba=0, kyotaku=1,
            tiles_left=69, self_wind=event_v3.tile_from_text("E"), oya_rel=0,
            ruleset_id=decision_v3.RulesetId.TENHOU_4P_RANKED,
            ruleset_deviation=(), source_kind=decision_v3.SourceKind.HUMAN_LOG,
        )
        with self.assertRaises(decision_v3.DecisionSchemaError):
            build_input(global_state=bad_global)

    def test_wind_consistency(self):
        players = list(build_input().players_public)
        players[1] = decision_v3.PlayerPublicState(
            seat_rel=1, score=25000, wind=event_v3.tile_from_text("W"),
            is_oya=False, riichi=decision_v3.RiichiState.NONE,
            river_count=0, melds=())
        with self.assertRaises(decision_v3.DecisionSchemaError):
            build_input(players=tuple(players))

    def test_ruleset_mismatch(self):
        tile_state = build_input().tile_state
        arena_global = decision_v3.GlobalPublicState(
            bakaze=event_v3.tile_from_text("E"), kyoku=1, honba=0, kyotaku=0,
            tiles_left=69, self_wind=event_v3.tile_from_text("E"), oya_rel=0,
            ruleset_id=decision_v3.RulesetId.MORTAL_ARENA_PINNED,
            ruleset_deviation=(), source_kind=decision_v3.SourceKind.ARENA_SELFPLAY,
        )
        with self.assertRaises(decision_v3.DecisionSchemaError):
            build_input(tile_state=tile_state, global_state=arena_global)

    def test_schema_version_mismatch(self):
        with self.assertRaises(TypeError):
            build_input(schema_version="rf-schema-v2/0")
        decision = build_input()
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.DecisionInput(
                events=decision.events, tile_state=decision.tile_state,
                players_public=decision.players_public,
                global_public=decision.global_public, legal_mask=decision.legal_mask,
                phase=decision.phase, before_event_idx=decision.before_event_idx,
                schema_version="rf-schema-v2/0",
            )


class VisibleStateTests(unittest.TestCase):
    def _base(self):
        decision = build_input()
        return decision.tile_state

    def test_zone_helper_and_base(self):
        state = self._base()
        self.assertEqual(state.zone_counts(decision_v3.PublicZone.DORA_INDICATOR)[9], 1)
        self.assertEqual(state.zone_counts(decision_v3.PublicZone.RIVER)[9], 0)
        with self.assertRaises(decision_v3.DecisionSchemaError):
            state.zone_counts("river")

    def test_physical_overflow(self):
        state = self._base()
        own = list(state.own_counts)
        zones = [list(zone) for zone in state.public_counts_by_zone]
        own[0] = 1
        zones[0][0] = 4
        public = [sum(zone[kind] for zone in zones) for kind in range(34)]
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.VisibleTileState(
                own_counts=tuple(own), own_aka=state.own_aka,
                public_counts=tuple(public), public_aka=state.public_aka,
                public_counts_by_zone=tuple(tuple(zone) for zone in zones),
                dora_indicators=state.dora_indicators,
                ruleset_id=state.ruleset_id,
            )

    def test_zone_sum_mismatch(self):
        state = self._base()
        public = list(state.public_counts)
        public[0] = 1
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.VisibleTileState(
                own_counts=state.own_counts, own_aka=state.own_aka,
                public_counts=tuple(public), public_aka=state.public_aka,
                public_counts_by_zone=state.public_counts_by_zone,
                dora_indicators=state.dora_indicators,
                ruleset_id=state.ruleset_id,
            )

    def test_indicator_zone_mismatch(self):
        state = self._base()
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.VisibleTileState(
                own_counts=state.own_counts, own_aka=state.own_aka,
                public_counts=state.public_counts, public_aka=state.public_aka,
                public_counts_by_zone=(state.public_counts_by_zone[0],
                                       state.public_counts_by_zone[1],
                                       (0,) * 34),
                dora_indicators=state.dora_indicators,
                ruleset_id=state.ruleset_id,
            )

    def test_aka_invariants(self):
        state = self._base()
        own = list(state.own_counts)
        own[4] = 0
        # own_aka=1 but own five count 0
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.VisibleTileState(
                own_counts=tuple(own), own_aka=(1, 0, 0),
                public_counts=state.public_counts, public_aka=state.public_aka,
                public_counts_by_zone=state.public_counts_by_zone,
                dora_indicators=state.dora_indicators,
                ruleset_id=state.ruleset_id,
            )
        # two aka of one colour across own+public
        own[4] = 1
        public = list(state.public_counts)
        public[4] = 1
        zones = [list(zone) for zone in state.public_counts_by_zone]
        zones[0][4] = 1
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.VisibleTileState(
                own_counts=tuple(own), own_aka=(1, 0, 0),
                public_counts=tuple(public), public_aka=(1, 0, 0),
                public_counts_by_zone=tuple(tuple(zone) for zone in zones),
                dora_indicators=state.dora_indicators,
                ruleset_id=state.ruleset_id,
            )

    def test_aka_indicator_must_be_in_public_aka(self):
        state = self._base()
        zones = [list(zone) for zone in state.public_counts_by_zone]
        # replace the normal 1p indicator with an aka 5p indicator
        zones[2][9] = 0
        zones[2][13] = 1
        public = [sum(zone[kind] for zone in zones) for kind in range(34)]
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.VisibleTileState(
                own_counts=state.own_counts, own_aka=state.own_aka,
                public_counts=tuple(public), public_aka=(0, 0, 0),
                public_counts_by_zone=tuple(tuple(zone) for zone in zones),
                dora_indicators=(event_v3.TileRef(13, 1),),
                ruleset_id=state.ruleset_id,
            )

    def test_meld_validation(self):
        chi = decision_v3.PublicMeld(
            meld_kind=decision_v3.MeldKind.CHI,
            tiles=(event_v3.tile_from_text("1m"), event_v3.tile_from_text("2m"),
                   event_v3.tile_from_text("3m")),
            from_rel=3,
        )
        self.assertTrue(chi.is_open)
        ankan_hidden = decision_v3.PublicMeld(
            meld_kind=decision_v3.MeldKind.ANKAN,
            tiles=(event_v3.UNKNOWN_TILE,) * 4, from_rel=4)
        self.assertFalse(ankan_hidden.is_open)
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.PublicMeld(
                meld_kind=decision_v3.MeldKind.CHI,
                tiles=(event_v3.UNKNOWN_TILE,) * 3, from_rel=3)
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.PublicMeld(
                meld_kind=decision_v3.MeldKind.CHI,
                tiles=(event_v3.tile_from_text("1m"),), from_rel=3)
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.PublicMeld(
                meld_kind=decision_v3.MeldKind.CHI,
                tiles=(event_v3.tile_from_text("1m"), event_v3.tile_from_text("3m"),
                       event_v3.tile_from_text("4m")), from_rel=3)
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.PublicMeld(
                meld_kind=decision_v3.MeldKind.CHI,
                tiles=(event_v3.tile_from_text("1m"), event_v3.tile_from_text("2p"),
                       event_v3.tile_from_text("3m")), from_rel=3)
        valid_kan = decision_v3.PublicMeld(
            meld_kind=decision_v3.MeldKind.DAIMINKAN,
            tiles=(event_v3.tile_from_text("5m"),) * 4, from_rel=1)
        self.assertTrue(valid_kan.is_open)
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.PublicMeld(
                meld_kind=decision_v3.MeldKind.DAIMINKAN,
                tiles=(event_v3.tile_from_text("5m"), event_v3.tile_from_text("5m"),
                       event_v3.tile_from_text("5m"), event_v3.tile_from_text("6m")),
                from_rel=1)
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.PublicMeld(
                meld_kind=decision_v3.MeldKind.ANKAN,
                tiles=(event_v3.tile_from_text("5m"), event_v3.tile_from_text("5m"),
                       event_v3.tile_from_text("5m"), event_v3.tile_from_text("6m")),
                from_rel=4)


class GlobalStateTests(unittest.TestCase):
    def _global(self, **overrides):
        values = dict(
            bakaze=event_v3.tile_from_text("E"), kyoku=1, honba=0, kyotaku=0,
            tiles_left=69, self_wind=event_v3.tile_from_text("E"), oya_rel=0,
            ruleset_id=decision_v3.RulesetId.TENHOU_4P_RANKED,
            ruleset_deviation=(), source_kind=decision_v3.SourceKind.HUMAN_LOG,
        )
        values.update(overrides)
        return decision_v3.GlobalPublicState(**values)

    def test_h04_deviation_metadata(self):
        arena = self._global(
            ruleset_id=decision_v3.RulesetId.MORTAL_ARENA_PINNED,
            ruleset_deviation=(decision_v3.RulesetDeviation.R02,
                               decision_v3.RulesetDeviation.R25),
            source_kind=decision_v3.SourceKind.ARENA_SELFPLAY,
        )
        self.assertTrue(arena.has_deviation("R02"))
        self.assertTrue(arena.has_deviation("R25"))
        self.assertFalse(arena.has_deviation("R26"))
        self.assertFalse(arena.has_deviation("R99"))
        with self.assertRaises(decision_v3.DecisionSchemaError):
            self._global(ruleset_deviation=(decision_v3.RulesetDeviation.R02,))
        with self.assertRaises(decision_v3.DecisionSchemaError):
            self._global(ruleset_deviation=("R02",))

    def test_wind_and_oya_consistency(self):
        with self.assertRaises(decision_v3.DecisionSchemaError):
            self._global(oya_rel=1, self_wind=event_v3.tile_from_text("E"))
        with self.assertRaises(decision_v3.DecisionSchemaError):
            self._global(oya_rel=0, self_wind=event_v3.tile_from_text("S"))
        with self.assertRaises(decision_v3.DecisionSchemaError):
            self._global(tiles_left=71)
        with self.assertRaises(decision_v3.DecisionSchemaError):
            self._global(source_kind="human_log")
        with self.assertRaises(decision_v3.DecisionSchemaError):
            self._global(
                ruleset_id=decision_v3.RulesetId.MORTAL_ARENA_PINNED,
                source_kind=decision_v3.SourceKind.HUMAN_LOG,
            )

    def test_player_state_validation(self):
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.PlayerPublicState(
                seat_rel=0, score=25000, wind=event_v3.tile_from_text("E"),
                is_oya=True, riichi="none", river_count=0, melds=())
        with self.assertRaises(decision_v3.DecisionSchemaError):
            decision_v3.PlayerPublicState(
                seat_rel=0, score=25000, wind=event_v3.tile_from_text("E"),
                is_oya=True, riichi=decision_v3.RiichiState.NONE,
                river_count=0, melds=(object(),))


class TargetTests(unittest.TestCase):
    def test_valid_supervised_and_censored_targets(self):
        supervised = target_v3.DecisionTarget(expert_action=0, supervision_valid=True)
        supervised.validate_against_input(build_input())
        censored = target_v3.DecisionTarget(
            expert_action=None, supervision_valid=False,
            censor_reason="interrupted_by_other_ron")
        self.assertFalse(censored.supervision_valid)
        self.assertEqual(censored.to_dict()["censor_reason"],
                         "interrupted_by_other_ron")

    def test_target_consistency_rules(self):
        with self.assertRaises(target_v3.TargetSchemaError):
            target_v3.DecisionTarget(expert_action=None, supervision_valid=True)
        with self.assertRaises(target_v3.TargetSchemaError):
            target_v3.DecisionTarget(expert_action=0, supervision_valid=True,
                                     censor_reason="nope")
        with self.assertRaises(target_v3.TargetSchemaError):
            target_v3.DecisionTarget(expert_action=None, supervision_valid=False)
        with self.assertRaises(target_v3.TargetSchemaError):
            target_v3.DecisionTarget(expert_action=46, supervision_valid=True)
        with self.assertRaises(target_v3.TargetSchemaError):
            target_v3.DecisionTarget(expert_action=0, supervision_valid=True,
                                     rank_final=5)
        with self.assertRaises(target_v3.TargetSchemaError):
            target_v3.DecisionTarget(expert_action=0, supervision_valid=True,
                                     oracle_labels=[1, 2])

    def test_target_must_be_in_mask(self):
        target = target_v3.DecisionTarget(expert_action=44, supervision_valid=True)
        with self.assertRaises(target_v3.TargetNotInMask):
            target.validate_against_input(build_input())

    def test_decision_input_and_target_are_separate_modules(self):
        self.assertFalse(hasattr(decision_v3, "DecisionTarget"))
        self.assertFalse(hasattr(target_v3, "DecisionInput"))


class SchemaFileTests(unittest.TestCase):
    def setUp(self):
        self.schema = schema_v3.load_schema()
        self.schema_path = schema_v3.SCHEMA_PATH

    def _write(self, schema):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", encoding="utf-8", delete=False)
        with handle:
            handle.write(schema_v3.canonical_json(schema) + "\n")
        self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)

    def _rehash(self, schema):
        schema.pop("schema_hash", None)
        schema["action_map_hash"] = schema_v3.compute_action_map_hash(schema)
        schema["normalizer_hash"] = schema_v3.compute_normalizer_hash(schema)
        schema["schema_hash"] = schema_v3.compute_schema_hash(schema)
        return schema

    def test_load_and_hashes(self):
        self.assertEqual(self.schema["schema_version"], "rf-schema-v3/2")
        self.assertEqual(action_v3.SCHEMA_VERSION, event_v3.SCHEMA_VERSION)
        self.assertEqual(action_v3.SCHEMA_VERSION, decision_v3.SCHEMA_VERSION)
        self.assertEqual(action_v3.SCHEMA_VERSION, target_v3.SCHEMA_VERSION)
        self.assertEqual(self.schema["schema_hash"],
                         schema_v3.compute_schema_hash(self.schema))
        self.assertEqual(self.schema["action_map_hash"],
                         action_v3.ACTION_MAP_HASH)
        self.assertEqual(self.schema["normalizer_hash"],
                         schema_v3.compute_normalizer_hash(self.schema))
        payload = {key: value for key, value in self.schema.items()
                   if key != "schema_hash"}
        manual = hashlib.sha256(
            schema_v3.canonical_json(payload).encode("utf-8")).hexdigest()
        self.assertEqual(manual, self.schema["schema_hash"])

    def test_schema_file_is_canonical_and_whitespace_insensitive(self):
        text = self.schema_path.read_text(encoding="utf-8")
        self.assertEqual(text, schema_v3.canonical_json(self.schema) + "\n")
        pretty = json.dumps(self.schema, ensure_ascii=False, indent=2)
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", encoding="utf-8", delete=False)
        with handle:
            handle.write(pretty)
        self.addCleanup(Path(handle.name).unlink)
        self.assertEqual(schema_v3.load_schema(Path(handle.name))["schema_hash"],
                         self.schema["schema_hash"])

    def test_tampered_schema_hash_rejected(self):
        tampered = copy.deepcopy(self.schema)
        tampered["schema_hash"] = "0" * 64
        with self.assertRaises(schema_v3.SchemaHashMismatch):
            schema_v3.load_schema(self._write(tampered))

    def test_tampered_event_field_rejected(self):
        tampered = copy.deepcopy(self.schema)
        tampered["event_types"]["dahai"]["fields"][0]["dtype"] = "u32"
        with self.assertRaises(schema_v3.SchemaHashMismatch):
            schema_v3.load_schema(self._write(tampered))

    def test_tampered_action_map_rejected(self):
        tampered = copy.deepcopy(self.schema)
        tampered["action_map"]["entries"][0]["name"] = "discard_9m"
        with self.assertRaises(schema_v3.SchemaHashMismatch):
            schema_v3.load_schema(self._write(tampered))

    def test_tampered_normalizer_rejected(self):
        tampered = copy.deepcopy(self.schema)
        tampered["normalizer"]["score"]["scale"] = 1
        with self.assertRaises(schema_v3.SchemaHashMismatch):
            schema_v3.load_schema(self._write(tampered))

    def test_expected_checkpoint_hashes(self):
        with self.assertRaises(schema_v3.SchemaHashMismatch):
            schema_v3.verify_schema(
                self.schema, expected_hashes={"schema_hash": "f" * 64})
        with self.assertRaises(schema_v3.SchemaHashMismatch):
            schema_v3.verify_schema(
                self.schema,
                expected_hashes={"action_map_hash": "f" * 64})

    def test_expected_hashes_require_all_three_keys(self):
        hashes = {key: self.schema[key] for key in schema_v3.HASH_KEYS}
        self.assertEqual(schema_v3.verify_schema(
            self.schema, expected_hashes=hashes)["schema_hash"],
            self.schema["schema_hash"])
        with self.assertRaises(schema_v3.SchemaHashMismatch):
            schema_v3.verify_schema(self.schema, expected_hashes={})
        for key in schema_v3.HASH_KEYS:
            with self.assertRaises(schema_v3.SchemaHashMismatch):
                schema_v3.verify_schema(
                    self.schema, expected_hashes={key: self.schema[key]})
        with self.assertRaises(schema_v3.SchemaHashMismatch):
            schema_v3.verify_schema(
                self.schema, expected_hashes={**hashes, "unknown_hash": "0" * 64})

    def test_rehashed_phase_legal_actions_drift_rejected(self):
        tampered = copy.deepcopy(self.schema)
        tampered["phase_legal_actions"]["SELF_TURN"] = [45]
        self._rehash(tampered)
        path = self._write(tampered)
        with self.assertRaises(schema_v3.SchemaHashMismatch):
            schema_v3.load_schema(path, verify_code=True)
        schema_v3.load_schema(path, verify_code=False)

    def test_code_consistency_catches_rehashed_drift(self):
        tampered = copy.deepcopy(self.schema)
        tampered["action_map"]["entries"][0]["name"] = "discard_9m"
        self._rehash(tampered)
        # Internally self-consistent, but no longer matches the code.
        path = self._write(tampered)
        with self.assertRaises(schema_v3.SchemaHashMismatch):
            schema_v3.load_schema(path, verify_code=True)
        schema_v3.load_schema(path, verify_code=False)

    def test_schema_metadata(self):
        schema = self.schema
        self.assertEqual(len(schema["event_types"]), 16)
        self.assertEqual(schema["action_map"]["action_space"], 46)
        self.assertEqual(len(schema["action_map"]["entries"]), 46)
        self.assertEqual(schema["decision_input_fields"],
                         list(decision_v3.DECISION_INPUT_FIELDS))
        self.assertEqual(schema["decision_target_fields"],
                         list(decision_v3.DECISION_TARGET_FIELDS))
        self.assertEqual(set(schema["decision_target_fields"]),
                         {field.name for field in fields(target_v3.DecisionTarget)})
        self.assertEqual(set(schema["forward_input_forbidden_fields"]),
                         set(decision_v3.FORBIDDEN_FORWARD_KEYS))
        self.assertEqual(set(schema["ruleset_deviations"]), {"R02", "R25", "R26"})
        self.assertIn("triple_ron", schema["ruleset_deviations"]["R02"]["description"])
        self.assertIn("kyotaku", schema["ruleset_deviations"]["R25"]["description"])
        self.assertIn("four_riichi", schema["ruleset_deviations"]["R26"]["description"])
        self.assertTrue(schema["score_protocol"]["hora_deltas_include_kyotaku"])
        self.assertFalse(schema["score_protocol"]["ryukyoku_deltas_include_kyotaku"])
        self.assertEqual(schema["candidate_schema_version"], "rf-candidate-v3/0")
        self.assertEqual(schema["belief_label_schema_version"], "rf-belief-v3/0")
        self.assertEqual(set(schema["rulesets"]),
                         {"tenhou_4p_ranked", "mortal_arena_pinned"})
        self.assertEqual(schema["rulesets"]["tenhou_4p_ranked"]["allowed_deviations"], [])

    def test_event_specs_match_dataclasses(self):
        schema = self.schema
        self.assertEqual(schema["event_types"], event_v3.event_type_specs())
        common_names = {item["name"] for item in schema["common_event_fields"]}
        self.assertEqual(common_names, {"type", "seq_idx", "kyoku_idx", "timing"})
        for type_name, spec in schema["event_types"].items():
            cls = event_v3.EVENT_CLASSES[type_name]
            for item in spec["fields"]:
                self.assertIn(item["dtype"], schema["dtypes"])
                self.assertIn(item["name"], cls.FIELDS)


class ImportIsolationTests(unittest.TestCase):
    def _run(self, code: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", code], cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=60,
        )

    def test_decision_module_does_not_import_target(self):
        code = (
            "import sys; sys.path.insert(0, %r);"
            "import riichiformer.schemas.decision_v3 as d;"
            "print(int('riichiformer.schemas.target_v3' in sys.modules));"
            "print(int(hasattr(d, 'DecisionTarget')))" % str(REPO_ROOT)
        )
        result = self._run(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["0", "0"])

    def test_target_module_imports_independently(self):
        code = (
            "import sys; sys.path.insert(0, %r);"
            "import riichiformer.schemas.target_v3 as t;"
            "print(int(hasattr(t, 'DecisionInput')))" % str(REPO_ROOT)
        )
        result = self._run(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
