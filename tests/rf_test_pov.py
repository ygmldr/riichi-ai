#!/usr/bin/env python3
"""PR-09 POV projector tests: allowlist, hidden-hand symmetry, timing, serialization.

All tests are synthetic and deterministic.  The optional real-log test reads a few local
Tenhou logs (read-only) and skips when the dataset is absent.
"""

from __future__ import annotations

import copy
import gzip
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data import pov  # noqa: E402
from riichiformer.schemas import action_v3, decision_v3, event_v3  # noqa: E402

HAND0 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
         "1p", "2p", "3p", "4p"]
HAND1 = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
         "E", "S", "W", "N"]
HAND2 = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
         "7m", "8m", "9m", "1p"]
HAND3 = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
         "1s", "2s", "3s", "4s", "5s"]
HAND1B = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
          "E", "S", "W", "P"]
HAND2B = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
          "7m", "8m", "9m", "2p"]
HAND3B = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
          "1s", "2s", "3s", "4s", "6s"]
HAND0B = ["9s", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
          "1p", "2p", "3p", "4p"]
AKA_HAND0 = ["5mr", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
             "1p", "2p", "3p", "4p"]


def raw_start_game(**overrides) -> dict:
    raw = {"type": "start_game", "names": ["p0", "p1", "p2", "p3"],
           "aka_flag": True, "kyoku_first": 0, "id": 7, "seed": [3, 4],
           "log": "https://example.invalid/log"}
    raw.update(overrides)
    return raw


def raw_start_kyoku(*, hands=None, scores=None, oya=0, kyotaku=0,
                    kyoku=1, honba=0, dora_marker="1p") -> dict:
    return {
        "type": "start_kyoku", "bakaze": "E", "dora_marker": dora_marker,
        "kyoku": kyoku, "honba": honba, "kyotaku": kyotaku, "oya": oya,
        "scores": list(scores or [25000, 25000, 25000, 25000]),
        "tehais": [list(hand) for hand in (hands or (HAND0, HAND1, HAND2, HAND3))],
    }


def base_stream() -> list[dict]:
    return [
        raw_start_game(),
        raw_start_kyoku(),
        {"type": "tsumo", "actor": 0, "pai": "5mr"},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
        {"type": "chi", "actor": 1, "target": 0, "pai": "5mr",
         "consumed": ["4m", "6m"]},
        {"type": "dahai", "actor": 1, "pai": "7p", "tsumogiri": False},
        {"type": "pon", "actor": 2, "target": 1, "pai": "7p",
         "consumed": ["7p", "7p"]},
        {"type": "dahai", "actor": 2, "pai": "9s", "tsumogiri": False},
        {"type": "reach", "actor": 3},
        {"type": "reach_accepted", "actor": 3},
        {"type": "dora", "dora_marker": "3p"},
        {"type": "hora", "actor": 3, "target": 2,
         "deltas": [1000, -1000, 0, 0], "ura_markers": ["3p"]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]


def hidden_base_stream() -> list[dict]:
    raw = copy.deepcopy(base_stream())
    raw.insert(3, {"type": "tsumo", "actor": 1, "pai": "7s"})
    raw.insert(4, {"type": "ankan", "actor": 1,
                   "consumed": ["5m", "5m", "5m", "5m"]})
    return raw


def hidden_variant_stream() -> list[dict]:
    raw = hidden_base_stream()
    start = next(item for item in raw if item["type"] == "start_kyoku")
    start["tehais"] = [HAND0, HAND1B, HAND2B, HAND3B]
    for item in raw:
        if item["type"] == "tsumo" and item["actor"] == 1:
            item["pai"] = "9p"
        elif item["type"] == "ankan" and item["actor"] == 1:
            item["consumed"] = ["6m", "6m", "6m", "6m"]
    return raw


def owned_stream() -> list[dict]:
    raw = copy.deepcopy(base_stream())
    start = next(item for item in raw if item["type"] == "start_kyoku")
    start["tehais"] = [HAND0B, HAND1, HAND2, HAND3]
    return raw


def context(pov_index: int, seq_idx: int = 0, kyoku_idx: int = 0,
            policy=None) -> pov.ProjectionContext:
    return pov.ProjectionContext(
        seq_idx=seq_idx, kyoku_idx=kyoku_idx,
        policy=policy if policy is not None else pov.DEFAULT_POLICY,
    )


def encode_stream(pov_index: int, raw_events, before_event_idx=None,
                  policy=None) -> bytes:
    projector = pov.POVProjector(
        pov_index, policy=policy if policy is not None else pov.DEFAULT_POLICY,
    )
    projector.extend(raw_events)
    return projector.encode_visible(before_event_idx)


def project_all(pov_index: int, raw_events=None,
                policy: pov.ProjectionPolicy = pov.DEFAULT_POLICY):
    return pov.project_stream(raw_events or base_stream(), pov_index, policy=policy)


class ProjectionCoreTests(unittest.TestCase):
    def test_all_event_types_project(self):
        raw = [
            raw_start_game(), raw_start_kyoku(),
            {"type": "tsumo", "actor": 0, "pai": "5mr"},
            {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
            {"type": "chi", "actor": 1, "target": 0, "pai": "5mr",
             "consumed": ["4m", "6m"]},
            {"type": "pon", "actor": 2, "target": 1, "pai": "7p",
             "consumed": ["7p", "7p"]},
            {"type": "daiminkan", "actor": 1, "target": 0, "pai": "3s",
             "consumed": ["3s", "3s", "3s"]},
            {"type": "kakan", "actor": 1, "pai": "5sr",
             "consumed": ["5s", "5s", "5s"]},
            {"type": "ankan", "actor": 1,
             "consumed": ["6m", "6m", "6m", "6m"]},
            {"type": "reach", "actor": 2},
            {"type": "reach_accepted", "actor": 2},
            {"type": "dora", "dora_marker": "3p"},
            {"type": "hora", "actor": 3, "target": 1,
             "deltas": [0, -1000, 0, 1000], "ura_markers": ["4p"]},
            {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
            {"type": "end_kyoku"}, {"type": "end_game"},
        ]
        events = project_all(0, raw)
        self.assertTrue(events)
        self.assertEqual([event.EVENT_TYPE for event in events],
                         [item["type"] for item in raw])
        self.assertEqual(len({event.EVENT_TYPE for event in events}), 16)

    def test_unknown_event_fails_closed_and_keeps_state(self):
        with self.assertRaises(event_v3.UnknownEventType):
            pov.project_event({"type": "not_mjai"}, 0, context(0))
        projector = pov.POVProjector(0)
        with self.assertRaises(event_v3.UnknownEventType):
            projector.project({"type": "not_mjai"})
        self.assertEqual(projector.events, ())
        self.assertFalse(projector.closed)

    def test_context_and_rotation_validation(self):
        with self.assertRaises(pov.ProjectionError):
            pov.ProjectionContext(seq_idx=-1, kyoku_idx=0)
        with self.assertRaises(pov.ProjectionError):
            pov.ProjectionContext(seq_idx=0, kyoku_idx=-2)
        with self.assertRaises(pov.ProjectionError):
            pov.ProjectionContext(seq_idx=0, kyoku_idx=0, policy="default")
        with self.assertRaises(pov.ProjectionError):
            pov.rotate_deltas_to_relative((1, 2, 3), 0)
        with self.assertRaises(pov.ProjectionError):
            pov.rotate_deltas_to_relative((1, 2, 3, 4), 4)

    def test_order_errors(self):
        with self.assertRaises(pov.ProjectionOrderError):
            pov.project_event({"type": "tsumo", "actor": 0, "pai": "1m"}, 0,
                              context(0, kyoku_idx=-1))
        with self.assertRaises(pov.ProjectionOrderError):
            pov.project_event(raw_start_kyoku(), 0, context(0, kyoku_idx=-1))
        projector = pov.POVProjector(0)
        with self.assertRaises(pov.ProjectionOrderError):
            projector.project({"type": "dahai", "actor": 0, "pai": "1m",
                               "tsumogiri": False})
        self.assertEqual(projector.events, ())

    def test_stream_assigns_indices_and_kyoku_boundaries(self):
        raw = [
            raw_start_game(),
            raw_start_kyoku(),
            {"type": "end_kyoku"},
            raw_start_kyoku(kyoku=2),
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ]
        events = project_all(0, raw)
        self.assertEqual([event.seq_idx for event in events], list(range(6)))
        self.assertEqual([event.kyoku_idx for event in events], [-1, 0, 0, 1, 1, 1])
        self.assertEqual([event.kind for event in events],
                         [event_v3.EventKind.START_GAME, event_v3.EventKind.START_KYOKU,
                          event_v3.EventKind.END_KYOKU, event_v3.EventKind.START_KYOKU,
                          event_v3.EventKind.END_KYOKU, event_v3.EventKind.END_GAME])

    def test_projected_stream_header_attribution(self):
        projector = pov.POVProjector(0)
        projector.extend([raw_start_game(), raw_start_kyoku()])
        header = projector.header
        self.assertEqual(header.pov, 0)
        self.assertEqual(header.projection_version, pov.POV_PROJECTION_VERSION)
        self.assertEqual(header.schema_version, event_v3.SCHEMA_VERSION)
        self.assertEqual(header.ruleset_id, "tenhou_4p_ranked")
        self.assertEqual(header.source_kind, "human_log")
        self.assertFalse(header.reveal_opponent_ankan)
        stream = projector.stream()
        self.assertIsInstance(stream, pov.ProjectedStream)
        self.assertEqual(stream.header, header)
        self.assertEqual(stream.count, 2)
        payload = json.loads(projector.encode_visible())
        self.assertEqual(payload["header"]["pov"], 0)
        self.assertEqual(payload["header"]["projection_version"],
                         pov.POV_PROJECTION_VERSION)
        self.assertEqual(len(payload["events"]), 2)
        # A POV/raw mix-up is detectable from the header even for an empty stream.
        empty_headers = [pov.POVProjector(index).header.as_dict() for index in range(4)]
        self.assertEqual({item["pov"] for item in empty_headers}, {0, 1, 2, 3})
        self.assertEqual(len({json.dumps(item, sort_keys=True)
                              for item in empty_headers}), 4)
        arena = pov.POVProjector(
            0, policy=pov.ProjectionPolicy(
                ruleset_id=decision_v3.RulesetId.MORTAL_ARENA_PINNED,
                source_kind=decision_v3.SourceKind.ARENA_SELFPLAY))
        self.assertEqual(arena.header.ruleset_id, "mortal_arena_pinned")
        self.assertEqual(arena.header.source_kind, "arena_selfplay")

    def test_malformed_opponent_ankan_is_masked_before_parsing(self):
        raw = [raw_start_game(), raw_start_kyoku(),
               {"type": "ankan", "actor": 1}]  # no consumed at all
        events = project_all(0, raw)
        ankan = next(event for event in events
                     if isinstance(event, event_v3.AnkanEvent))
        self.assertTrue(all(tile.is_unknown for tile in ankan.consumed))
        with self.assertRaises(event_v3.SchemaValueError):
            project_all(0, [raw_start_game(), raw_start_kyoku(),
                            {"type": "ankan", "actor": 0}])

    def test_extend_rolls_back_on_error(self):
        projector = pov.POVProjector(0)
        projector.extend([raw_start_game(), raw_start_kyoku()])
        before_events = projector.events
        before_payload = projector.encode_visible()
        with self.assertRaises(event_v3.UnknownEventType):
            projector.extend([
                {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
                {"type": "not_mjai"},
            ])
        self.assertEqual(projector.events, before_events)
        self.assertEqual(projector.encode_visible(), before_payload)

    def test_projector_closes_after_end_game_and_resets(self):
        projector = pov.POVProjector(0)
        projector.extend(base_stream())
        self.assertTrue(projector.closed)
        with self.assertRaises(pov.ProjectionClosed):
            projector.project({"type": "end_game"})
        before = projector.encode_visible()
        projector.reset()
        self.assertFalse(projector.closed)
        self.assertEqual(projector.events, ())
        projector.extend(base_stream())
        self.assertEqual(projector.encode_visible(), before)

    def test_policy_requires_evidence_for_opponent_ankan(self):
        with self.assertRaises(pov.ProjectionPolicyError):
            pov.ProjectionPolicy(reveal_opponent_ankan=True)
        with self.assertRaises(pov.ProjectionPolicyError):
            pov.ProjectionPolicy(reveal_opponent_ankan=True, ankan_evidence="   ")
        with self.assertRaises(pov.ProjectionPolicyError):
            pov.ProjectionPolicy(reveal_opponent_ankan=False,
                                 ankan_evidence="unused")
        with self.assertRaises(pov.ProjectionPolicyError):
            pov.ProjectionPolicy(source_kind=decision_v3.SourceKind.HUMAN_LOG,
                                 ruleset_id=decision_v3.RulesetId.MORTAL_ARENA_PINNED)
        policy = pov.ProjectionPolicy(
            ruleset_id=decision_v3.RulesetId.MORTAL_ARENA_PINNED,
            source_kind=decision_v3.SourceKind.ARENA_SELFPLAY,
            reveal_opponent_ankan=True,
            ankan_evidence="H-04 R05: ankan is published immediately in this source",
        )
        raw = [raw_start_game(), raw_start_kyoku(),
               {"type": "ankan", "actor": 1,
                "consumed": ["6m", "6m", "6m", "6m"]}]
        events = project_all(0, raw, policy=policy)
        ankan = next(event for event in events if isinstance(event, event_v3.AnkanEvent))
        self.assertTrue(all(not tile.is_unknown for tile in ankan.consumed))

    def test_default_opponent_ankan_is_hidden(self):
        raw = [raw_start_game(), raw_start_kyoku(),
               {"type": "ankan", "actor": 1,
                "consumed": ["6m", "6m", "6m", "6m"]}]
        ankan = next(event for event in project_all(0, raw)
                     if isinstance(event, event_v3.AnkanEvent))
        self.assertTrue(all(tile.is_unknown for tile in ankan.consumed))

    def test_start_game_drops_private_fields(self):
        events = project_all(0)
        start = events[0]
        self.assertIsInstance(start, event_v3.StartGameEvent)
        data = start.to_dict()
        self.assertEqual(set(data),
                         {"type", "seq_idx", "kyoku_idx", "timing",
                          "aka_flag", "kyoku_first"})
        dumped = json.dumps(data)
        for forbidden in ("names", "id", "seed", "log"):
            self.assertNotIn(f'"{forbidden}"', dumped)

    def test_start_kyoku_has_only_own_hand(self):
        events = project_all(1)
        start = next(event for event in events
                     if isinstance(event, event_v3.StartKyokuEvent))
        self.assertEqual(start.own_haipai,
                         tuple(event_v3.tile_from_text(t) for t in HAND1))
        data = start.to_dict()
        self.assertNotIn("tehais", data)
        self.assertEqual(len(data["own_haipai"]), 13)

    def test_0006_opponent_tsumo_pai_leak_regression(self):
        # Old patch 0006 exposed opponent tsumo tiles in the event stream.  The PR-09
        # projection must keep only actor/seq for opponents and UNKNOWN tile/aka.
        raw = {"type": "tsumo", "actor": 1, "pai": "7s"}
        snapshot = dict(raw)
        event = pov.project_event(raw, 0, context(0, kyoku_idx=0))
        self.assertEqual(raw, snapshot)  # raw dict not mutated
        self.assertIsInstance(event, event_v3.TsumoEvent)
        self.assertTrue(event.tile.is_unknown)
        self.assertEqual(event.tile.to_dict(), {"kind": 34, "aka": None})
        self.assertEqual(set(event_v3.TsumoEvent.FIELDS), {"actor_rel", "tile"})
        serialized = json.dumps(event.to_dict())
        self.assertNotIn("7s", serialized)
        self.assertNotIn('"pai"', serialized)
        self.assertEqual(event.tile.kind, 34)
        opponent_aka = pov.project_event(
            {"type": "tsumo", "actor": 2, "pai": "5mr"}, 0,
            context(0, kyoku_idx=0))
        self.assertTrue(opponent_aka.tile.is_unknown)
        self.assertIsNone(opponent_aka.tile.aka)
        own = pov.project_event({"type": "tsumo", "actor": 0, "pai": "7s"}, 0,
                                context(0, kyoku_idx=0))
        self.assertEqual(own.tile, event_v3.tile_from_text("7s"))

    def test_no_forbidden_keys_in_projected_stream(self):
        payload = json.dumps([event.to_dict() for event in project_all(0)])
        for forbidden in ("names", "player_names", "account_id", "wall_seed",
                          "seed", "log", "tehais", "pai", "rank_final",
                          "oracle_labels", "expert_action", "supervision_valid"):
            self.assertNotIn(f'"{forbidden}"', payload)
        self.assertNotIn("p0", payload)  # no player name text


class VisibilityPropertyTests(unittest.TestCase):
    def test_four_pov_symmetry(self):
        raw = base_stream()
        streams = {pov_index: project_all(pov_index, raw) for pov_index in range(4)}
        for pov_index, events in streams.items():
            self.assertEqual(len(events), len(raw))
            for event in events:
                raw_event = raw[event.seq_idx]
                self.assertEqual(event.EVENT_TYPE, raw_event["type"])
                if hasattr(event, "actor_rel"):
                    self.assertEqual((event.actor_rel + pov_index) % 4,
                                     raw_event["actor"])
                if hasattr(event, "target_rel"):
                    self.assertEqual((event.target_rel + pov_index) % 4,
                                     raw_event["target"])
                if isinstance(event, event_v3.StartKyokuEvent):
                    self.assertEqual(
                        event.own_haipai,
                        tuple(event_v3.tile_from_text(t)
                              for t in raw_event["tehais"][pov_index]))
                    self.assertEqual(
                        event.scores_rel,
                        tuple(raw_event["scores"][(pov_index + offset) % 4]
                              for offset in range(4)))
                    self.assertEqual(event.oya_rel, (raw_event["oya"] - pov_index) % 4)
                elif isinstance(event, event_v3.TsumoEvent):
                    if raw_event["actor"] == pov_index:
                        self.assertEqual(event.tile,
                                         event_v3.tile_from_text(raw_event["pai"]))
                    else:
                        self.assertTrue(event.tile.is_unknown)
                elif isinstance(event, event_v3.HoraEvent):
                    self.assertEqual(
                        pov.rotate_deltas_to_absolute(event.deltas, pov_index),
                        tuple(raw_event["deltas"]))
                    self.assertEqual([tile.text for tile in event.ura_markers],
                                     raw_event["ura_markers"])
                elif isinstance(event, event_v3.DahaiEvent):
                    self.assertEqual(event.tile.text, raw_event["pai"])
                    self.assertEqual(event.tsumogiri, raw_event["tsumogiri"])
                elif isinstance(event, event_v3.ChiEvent):
                    self.assertEqual(event.tile.text, raw_event["pai"])
                    self.assertEqual([tile.text for tile in event.consumed[:2]],
                                     raw_event["consumed"])
                elif isinstance(event, event_v3.PonEvent):
                    self.assertEqual(event.tile.text, raw_event["pai"])
                    self.assertEqual([tile.text for tile in event.consumed[:2]],
                                     raw_event["consumed"])
                elif isinstance(event, event_v3.DoraEvent):
                    self.assertEqual(event.dora_marker.text, raw_event["dora_marker"])

    def test_hidden_opponent_perturbation_is_byte_identical(self):
        base = project_all(0, hidden_base_stream())
        variant = project_all(0, hidden_variant_stream())
        self.assertEqual([event.to_dict() for event in base],
                         [event.to_dict() for event in variant])
        self.assertEqual(encode_stream(0, hidden_base_stream()),
                         encode_stream(0, hidden_variant_stream()))

    def test_hidden_ankan_perturbation_is_byte_identical(self):
        raw_a = [raw_start_game(), raw_start_kyoku(),
                 {"type": "ankan", "actor": 1,
                  "consumed": ["5m", "5m", "5m", "5m"]},
                 {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
                 {"type": "end_game"}]
        raw_b = copy.deepcopy(raw_a)
        raw_b[2]["consumed"] = ["6m", "6m", "6m", "6m"]
        self.assertEqual(project_all(0, raw_a), project_all(0, raw_b))
        self.assertEqual(encode_stream(0, raw_a), encode_stream(0, raw_b))

    def test_own_hand_perturbation_is_visible(self):
        base = project_all(0, base_stream())
        changed = project_all(0, owned_stream())
        base_start = next(event for event in base
                          if isinstance(event, event_v3.StartKyokuEvent))
        changed_start = next(event for event in changed
                             if isinstance(event, event_v3.StartKyokuEvent))
        self.assertNotEqual(base_start.own_haipai, changed_start.own_haipai)
        self.assertNotEqual(encode_stream(0, base_stream()),
                            encode_stream(0, owned_stream()))

    def test_future_truncation_does_not_change_past(self):
        raw = base_stream()
        projector = pov.POVProjector(0)
        prefix = projector.extend(raw[:4])
        before = projector.encode_visible(4)
        self.assertEqual(before, encode_stream(0, raw[:4]))
        projector.extend(raw[4:])
        self.assertEqual(projector.encode_visible(4), before)
        self.assertEqual(projector.visible_events(4), prefix)
        self.assertEqual(projector.visible_events(0), ())
        self.assertEqual(projector.visible_events(1000), projector.events)
        with self.assertRaises(pov.ProjectionError):
            projector.event_at(1000)

    def test_projected_deltas_are_marked_relative(self):
        raw = base_stream()
        for pov_index in range(4):
            events = project_all(pov_index, raw)
            for event in events:
                if isinstance(event, (event_v3.HoraEvent, event_v3.RyukyokuEvent)):
                    self.assertEqual(event.deltas_order, event_v3.DELTAS_ORDER_RELATIVE)
                    self.assertEqual(event.to_dict()["deltas_order"], "relative")
                    self.assertEqual(
                        pov.rotate_deltas_to_absolute(event.deltas, pov_index),
                        tuple(raw[event.seq_idx]["deltas"]))
        raw_hora = event_v3.event_from_mjai(
            {"type": "hora", "actor": 0, "target": 0,
             "deltas": [0, 0, 0, 0], "ura_markers": []},
            pov=0, seq_idx=0, kyoku_idx=0)
        self.assertEqual(raw_hora.deltas_order, event_v3.DELTAS_ORDER_ABSOLUTE)

    def test_ryukyoku_deltas_rotate(self):
        raw = [raw_start_game(), raw_start_kyoku(),
               {"type": "ryukyoku", "deltas": [-1500, 1500, 1500, -1500]},
               {"type": "end_game"}]
        for pov_index in range(4):
            event = next(event for event in project_all(pov_index, raw)
                         if isinstance(event, event_v3.RyukyokuEvent))
            self.assertEqual(
                pov.rotate_deltas_to_absolute(event.deltas, pov_index),
                (-1500, 1500, 1500, -1500))


class TimingTests(unittest.TestCase):
    def test_kan_and_ura_timing(self):
        raw = [raw_start_game(), raw_start_kyoku(),
               {"type": "ankan", "actor": 0,
                "consumed": ["5m", "5m", "5m", "5m"]},
               {"type": "dora", "dora_marker": "2p"},
               {"type": "dahai", "actor": 0, "pai": "5s", "tsumogiri": False},
               {"type": "hora", "actor": 0, "target": 0,
                "deltas": [1000, -1000, 0, 0], "ura_markers": ["3p"]},
               {"type": "end_kyoku"}, {"type": "end_game"}]
        projector = pov.POVProjector(0)
        events = projector.extend(raw)
        types = [event.EVENT_TYPE for event in events]
        self.assertEqual(types, ["start_game", "start_kyoku", "ankan", "dora",
                                 "dahai", "hora", "end_kyoku", "end_game"])
        start = next(event for event in events
                     if isinstance(event, event_v3.StartKyokuEvent))
        self.assertEqual(start.dora_marker.text, "1p")
        dora = next(event for event in events
                    if isinstance(event, event_v3.DoraEvent))
        self.assertEqual(dora.dora_marker.text, "2p")
        hora_seq = next(event.seq_idx for event in events
                        if isinstance(event, event_v3.HoraEvent))
        before_hora = projector.encode_visible(hora_seq)
        after_hora = projector.encode_visible(hora_seq + 1)
        self.assertNotIn(b"ura_markers", before_hora)
        self.assertIn(b"ura_markers", after_hora)
        self.assertIn(b'"deltas_order":"relative"', after_hora)
        hora = projector.event_at(hora_seq)
        self.assertEqual([tile.text for tile in hora.ura_markers], ["3p"])
        # end_kyoku/end_game are terminal until their own position.
        end_kyoku_seq = types.index("end_kyoku")
        end_game_seq = types.index("end_game")
        self.assertNotIn("end_kyoku",
                         [event.EVENT_TYPE for event in projector.visible_events(end_kyoku_seq)])
        self.assertNotIn("end_game",
                         [event.EVENT_TYPE for event in projector.visible_events(end_game_seq)])

    def test_r26_four_riichi_extra_draw_order_is_preserved(self):
        # H-04 R26: the pinned arena accepts the fourth riichi, draws one extra tile, and
        # only then emits the abortive ryukyoku.  The projector must preserve that order
        # and must not normalize it to the Tenhou instant.
        raw = [raw_start_game(), raw_start_kyoku(),
               {"type": "reach", "actor": 0}, {"type": "reach_accepted", "actor": 0},
               {"type": "reach", "actor": 1}, {"type": "reach_accepted", "actor": 1},
               {"type": "reach", "actor": 2}, {"type": "reach_accepted", "actor": 2},
               {"type": "reach", "actor": 3}, {"type": "reach_accepted", "actor": 3},
               {"type": "tsumo", "actor": 0, "pai": "1m"},
               {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
               {"type": "end_kyoku"}, {"type": "end_game"}]
        events = project_all(0, raw)
        types = [event.EVENT_TYPE for event in events]
        self.assertEqual(types[-4:], ["tsumo", "ryukyoku", "end_kyoku", "end_game"])
        # The extra draw is between the fourth accepted riichi and the abort.
        self.assertLess(types.index("tsumo"), types.index("ryukyoku"))
        self.assertEqual(types.count("reach_accepted"), 4)
        ryukyoku = next(event for event in events
                        if isinstance(event, event_v3.RyukyokuEvent))
        self.assertEqual(ryukyoku.deltas, (0, 0, 0, 0))

    def test_ruleset_event_order_is_not_rewritten(self):
        # R02: the arena settles three ron; the projector must not inject an abort.
        raw = [raw_start_game(), raw_start_kyoku(),
               {"type": "hora", "actor": 1, "target": 0,
                "deltas": [0, 1000, 0, -1000], "ura_markers": []},
               {"type": "hora", "actor": 2, "target": 0,
                "deltas": [0, 0, 1000, -1000], "ura_markers": []},
               {"type": "hora", "actor": 3, "target": 0,
                "deltas": [0, 0, 0, 0], "ura_markers": []},
               {"type": "end_kyoku"}, {"type": "end_game"}]
        events = project_all(0, raw)
        types = [event.EVENT_TYPE for event in events]
        self.assertEqual(types.count("hora"), 3)
        self.assertNotIn("ryukyoku", types)


class PublicLedgerTests(unittest.TestCase):
    """Tests the physical-tile migration rule; PR-10 owns the production ledger."""

    @staticmethod
    def _add(counts, akas, tile):
        counts[tile.kind] += 1
        if tile.aka is not None:
            akas[tile.aka] += 1

    @staticmethod
    def _remove(counts, akas, tile):
        counts[tile.kind] -= 1
        if counts[tile.kind] < 0:
            raise AssertionError(f"removed unobserved tile kind {tile.kind}")
        if tile.aka is not None:
            akas[tile.aka] -= 1
            if akas[tile.aka] < 0:
                raise AssertionError(f"removed unobserved aka colour {tile.aka}")

    def _ledger(self, events):
        river = [0] * 34
        river_aka = [0, 0, 0]
        meld = [0] * 34
        meld_aka = [0, 0, 0]
        indicators = [0] * 34
        for event in events:
            if isinstance(event, event_v3.DahaiEvent):
                self._add(river, river_aka, event.tile)
            elif isinstance(event, (event_v3.ChiEvent, event_v3.PonEvent,
                                    event_v3.DaiminkanEvent)):
                self._remove(river, river_aka, event.tile)
                self._add(meld, meld_aka, event.tile)
                for tile in event.consumed:
                    if not tile.is_unknown:
                        self._add(meld, meld_aka, tile)
            elif isinstance(event, event_v3.KakanEvent):
                # the three consumed tiles were already counted by the earlier pon
                self._add(meld, meld_aka, event.tile)
            elif isinstance(event, event_v3.AnkanEvent):
                if all(not tile.is_unknown for tile in event.consumed):
                    for tile in event.consumed:
                        self._add(meld, meld_aka, tile)
            elif isinstance(event, event_v3.DoraEvent):
                indicators[event.dora_marker.kind] += 1
        return river, river_aka, meld, meld_aka, indicators

    def test_aka_migrates_from_river_to_meld_without_double_count(self):
        raw = [raw_start_game(), raw_start_kyoku(hands=(AKA_HAND0, HAND1, HAND2, HAND3)),
               {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
               {"type": "chi", "actor": 1, "target": 0, "pai": "5mr",
                "consumed": ["4m", "6m"]},
               {"type": "end_game"}]
        events = project_all(0, raw)
        river, river_aka, meld, meld_aka, _ = self._ledger(events)
        self.assertEqual(river[4], 0)  # called 5m left the river
        self.assertEqual(river_aka[0], 0)
        self.assertEqual(meld[4], 1)   # called aka 5m is now in the meld
        self.assertEqual(meld_aka[0], 1)
        self.assertEqual(meld[3], 1)   # 4m
        self.assertEqual(meld[5], 1)   # 6m
        self.assertEqual(sum(river_aka) + sum(meld_aka), 1)
        for kind in range(34):
            self.assertLessEqual(river[kind] + meld[kind], 4)

    def test_normal_five_pon_is_not_aka(self):
        raw = [raw_start_game(), raw_start_kyoku(hands=(AKA_HAND0, HAND1, HAND2, HAND3)),
               {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
               {"type": "pon", "actor": 1, "target": 0, "pai": "5m",
                "consumed": ["5m", "5m"]},
               {"type": "end_game"}]
        river, river_aka, meld, meld_aka, _ = self._ledger(project_all(0, raw))
        self.assertEqual(river[4], 0)
        self.assertEqual(meld[4], 3)
        self.assertEqual(sum(meld_aka), 0)
        self.assertEqual(sum(river_aka), 0)


def dataset_year_dir(year: str = "2025") -> Path | None:
    for root in (REPO_ROOT / "datasets", REPO_ROOT.parent.parent / "datasets"):
        directory = root / "extracted" / year
        if directory.is_dir():
            return directory
    return None


def read_events(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


class RealLogProjectionTests(unittest.TestCase):
    def test_real_logs_project_for_four_povs(self):
        directory = dataset_year_dir()
        if directory is None:
            self.skipTest("no local 2025 dataset")
        files = sorted(directory.glob("*.mjson"))[:10]
        if not files:
            self.skipTest("no local 2025 logs")
        for path in files:
            raw = read_events(path)
            for pov_index in range(4):
                projector = pov.POVProjector(pov_index)
                events = projector.extend(raw)
                self.assertEqual(len(events), len(raw), path.name)
                self.assertTrue(projector.closed, path.name)
                for event in events:
                    if isinstance(event, event_v3.StartKyokuEvent):
                        self.assertEqual(
                            event.own_haipai,
                            tuple(event_v3.tile_from_text(t)
                                  for t in raw[event.seq_idx]["tehais"][pov_index]),
                            path.name,
                        )
                self.assertEqual(projector.encode_visible(),
                                 encode_stream(pov_index, raw))


if __name__ == "__main__":
    unittest.main(verbosity=2)
