#!/usr/bin/env python3
"""PR-16 arena adapter tests: action table, masks, multi-kan, mock loop, guards.

All tests are synthetic/offline.  No dataset, checkpoint, libriichi call or frozen
evaluation directory is touched; the mock arena writes only under ``tmp`` dirs.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.runtime import mjai_adapter as A  # noqa: E402
from riichiformer.runtime.agent import (  # noqa: E402
    AgentError,
    EmptyPolicyOutput,
    IllegalAgentAction,
    MockPolicy,
    RiichiFormerAgent,
    TruncatedStream,
    VisibilityError,
)
from riichiformer.schemas.action_v3 import (  # noqa: E402
    ACTION_SPACE,
    ActionPhase,
)
from riichiformer.schemas.decision_v3 import (  # noqa: E402
    DecisionInput,
    MeldKind,
    PlayerPublicState,
    PublicMeld,
    RiichiState,
    RulesetId,
    VisibleTileState,
)
from riichiformer.schemas.event_v3 import (  # noqa: E402
    TsumoEvent,
    tile_from_text,
)
from scripts import rf_eval  # noqa: E402


def _tile(text: str):
    return tile_from_text(text)


def _mask(*actions: int) -> tuple[bool, ...]:
    mask = [False] * ACTION_SPACE
    for action in actions:
        mask[action] = True
    return tuple(mask)


def _hand(*texts: str) -> tuple:
    return tuple(_tile(text) for text in texts)


def _kan_ankan(kind: int = 4) -> A.KanOption:
    tiles = (A.TileRef(kind),) * 4
    return A.KanOption("ankan", tiles[0], tiles)


class ActionTableTests(unittest.TestCase):
    def test_all_46_actions_decode_in_their_phase(self):
        # A context per action; every decode must be an MJAI dict and reverse-map.
        for action in range(37):
            phase = ActionPhase.SELF_TURN
            action_tile = A.action_tile(action)
            base = [_tile(text) for text in
                    ("1p", "2p", "3p", "4p", "6p", "7p", "8p", "9p",
                     "1s", "2s", "3s", "4s", "6s")]
            hand = (action_tile,) + tuple(base)
            ctx = A.ActionContext(seat=1, phase=phase, legal_mask=_mask(action),
                                  hand=hand,
                                  last_self_tsumo=action_tile if action != 37 else None)
            event = A.action_to_mjai(action, ctx)
            self.assertEqual(A.mjai_to_action(event, phase), action)
            if action < 34:
                self.assertEqual(event["type"], "dahai")
                self.assertEqual(event["actor"], 1)
                self.assertIn("tsumogiri", event)
            else:
                self.assertEqual(event["pai"], {34: "5mr", 35: "5pr", 36: "5sr"}[action])
        # KAN_SELECTION reuses 0..36 as a tile choice (ankan case).
        kan = _kan_ankan(4)
        for action in range(37):
            ctx = A.ActionContext(seat=1, phase=ActionPhase.KAN_SELECTION,
                                  legal_mask=_mask(action), kan_options=(kan,))
            if action == 4:
                event = A.action_to_mjai(action, ctx)
                self.assertEqual(event["type"], "ankan")
                self.assertEqual(event["actor"], 1)
                self.assertEqual(A.mjai_to_action(event, ActionPhase.KAN_SELECTION), 4)
            else:
                with self.assertRaises(A.IllegalActionError):
                    A.action_to_mjai(action, ctx)

    def test_special_actions(self):
        hand = _hand("2m", "3m", "4m", "4m", "4m", "5m", "6m", "7m",
                     "8m", "9m", "1p", "2p", "3p", "5s")
        cases = {
            37: ActionPhase.SELF_TURN,
            38: ActionPhase.CALL_RESPONSE,
            39: ActionPhase.CALL_RESPONSE,
            40: ActionPhase.CALL_RESPONSE,
            41: ActionPhase.CALL_RESPONSE,
            42: ActionPhase.SELF_TURN,
            43: ActionPhase.SELF_TURN,
            44: ActionPhase.SELF_TURN,
            45: ActionPhase.CALL_RESPONSE,
        }
        for action, phase in cases.items():
            with self.subTest(action=action):
                kwargs = dict(seat=0, phase=phase, legal_mask=_mask(action), hand=hand)
                if action in (38, 39, 40, 41):
                    kwargs.update(last_kawa_tile=_tile("4m"), last_kawa_actor=3)
                if action == 42:
                    kwargs["kan_options"] = (_kan_ankan(4),)
                if action == 43:
                    kwargs["can_tsumo_agari"] = phase is ActionPhase.SELF_TURN
                    kwargs["can_ron_agari"] = phase is ActionPhase.CALL_RESPONSE
                    kwargs["ron_target"] = 3
                if action == 44:
                    kwargs["can_ryukyoku"] = True
                event = A.action_to_mjai(action, A.ActionContext(**kwargs))
                self.assertEqual(A.mjai_to_action(event, phase), action)

        # chi low/mid/high concrete consumed tiles
        for action, consumed in ((38, ("5m", "6m")), (39, ("3m", "5m")),
                                 (40, ("2m", "3m"))):
            ctx = A.ActionContext(seat=0, phase=ActionPhase.CALL_RESPONSE,
                                  legal_mask=_mask(action), hand=hand,
                                  last_kawa_tile=_tile("4m"), last_kawa_actor=3)
            event = A.action_to_mjai(action, ctx)
            self.assertEqual(event["type"], "chi")
            self.assertEqual(event["target"], 3)
            self.assertEqual(tuple(event["consumed"]), consumed)

        # pon consumes two matching tiles and targets the discarder
        ctx = A.ActionContext(seat=0, phase=ActionPhase.CALL_RESPONSE,
                              legal_mask=_mask(41), hand=hand,
                              last_kawa_tile=_tile("4m"), last_kawa_actor=3)
        event = A.action_to_mjai(41, ctx)
        self.assertEqual(event["type"], "pon")
        self.assertEqual(event["target"], 3)
        self.assertEqual(event["pai"], "4m")
        self.assertEqual(len(event["consumed"]), 2)

        # ryukyoku / pass / double ron / tsumo
        self.assertEqual(A.action_to_mjai(44, A.ActionContext(
            seat=0, phase=ActionPhase.SELF_TURN, legal_mask=_mask(44),
            can_ryukyoku=True)), {"type": "ryukyoku"})
        self.assertEqual(A.action_to_mjai(45, A.ActionContext(
            seat=0, phase=ActionPhase.CALL_RESPONSE, legal_mask=_mask(45))),
            {"type": "none"})
        ron = A.action_to_mjai(43, A.ActionContext(
            seat=2, phase=ActionPhase.CALL_RESPONSE, legal_mask=_mask(43),
            can_ron_agari=True, ron_target=0))
        self.assertEqual(ron, {"type": "hora", "actor": 2, "target": 0})
        tsumo = A.action_to_mjai(43, A.ActionContext(
            seat=0, phase=ActionPhase.SELF_TURN, legal_mask=_mask(43),
            can_tsumo_agari=True))
        self.assertEqual(tsumo, {"type": "hora", "actor": 0, "target": 0})

    def test_aka_and_normal_five_are_distinct(self):
        self.assertEqual(A.action_tile(4), A.TileRef(4, None))
        self.assertEqual(A.action_tile(34), A.TileRef(4, 0))
        self.assertEqual(A.action_for_tile(A.TileRef(4, 0)), 34)
        self.assertEqual(A.action_for_tile(A.TileRef(13, 1)), 35)
        self.assertEqual(A.action_for_tile(A.TileRef(22, 2)), 36)
        with self.assertRaises(A.AdapterError):
            A.action_for_tile(A.TileRef(34))

    def test_phase_and_mask_rejection(self):
        hand = _hand("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m",
                     "9m", "1p", "2p", "3p", "4p", "5s")
        # Mask says legal, phase says no.
        with self.assertRaises(A.IllegalActionError):
            A.action_to_mjai(45, A.ActionContext(
                seat=0, phase=ActionPhase.SELF_TURN, legal_mask=_mask(45)))
        with self.assertRaises(A.IllegalActionError):
            A.action_to_mjai(0, A.ActionContext(
                seat=0, phase=ActionPhase.CALL_RESPONSE, legal_mask=_mask(0)))
        # Mask says no.
        with self.assertRaises(A.IllegalActionError):
            A.action_to_mjai(1, A.ActionContext(
                seat=0, phase=ActionPhase.SELF_TURN, legal_mask=_mask(0), hand=hand))
        # Out of range / non-int.
        for bad in (-1, 46, 1.0, True, "1"):
            with self.assertRaises(A.IllegalActionError):
                A.action_to_mjai(bad, A.ActionContext(
                    seat=0, phase=ActionPhase.SELF_TURN, legal_mask=_mask(0)))
        # Empty legal mask is a construction error (never a silent None).
        with self.assertRaises(A.AdapterError):
            A.ActionContext(seat=0, phase=ActionPhase.SELF_TURN,
                            legal_mask=(False,) * ACTION_SPACE)
        # Discard of a tile not in hand (C is absent from the 13-tile hand).
        with self.assertRaises(A.IllegalActionError):
            A.action_to_mjai(33, A.ActionContext(
                seat=0, phase=ActionPhase.SELF_TURN, legal_mask=_mask(33), hand=hand))

    def test_multi_kan_requires_explicit_second_stage(self):
        ankan = _kan_ankan(4)
        tiles = (_tile("6m"),) * 3
        kakan = A.KanOption("kakan", _tile("6m"), tiles)
        mask = _mask(42)
        ctx = A.ActionContext(seat=0, phase=ActionPhase.SELF_TURN,
                              legal_mask=mask, kan_options=(ankan, kakan))
        with self.assertRaises(A.AmbiguousKanError):
            A.action_to_mjai(42, ctx)
        chosen = A.action_to_mjai(42, ctx.with_kan_choice(_tile("6m")))
        self.assertEqual(chosen, {"type": "kakan", "actor": 0, "pai": "6m"})
        chosen = A.action_to_mjai(42, ctx.with_kan_choice(_tile("5m")))
        self.assertEqual(chosen["type"], "ankan")
        # KAN_SELECTION routes each tile action to its own option.
        selection = A.ActionContext(seat=0, phase=ActionPhase.KAN_SELECTION,
                                    legal_mask=_mask(4, 5), kan_options=(ankan, kakan))
        self.assertEqual(A.action_to_mjai(4, selection)["type"], "ankan")
        self.assertEqual(A.action_to_mjai(5, selection)["type"], "kakan")
        with self.assertRaises(A.IllegalActionError):
            A.action_to_mjai(7, selection)
        with self.assertRaises(A.IllegalActionError):
            A.select_kan_option(())
        with self.assertRaises(A.AmbiguousKanError):
            A.select_kan_option((ankan, kakan))


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = rf_eval._meta(0)

    def _decision(self):
        tiles = rf_eval._hand_tiles(__import__("random").Random(3), 14)
        return rf_eval._decision(tiles, ActionPhase.SELF_TURN, draw=tiles[13])

    def _agent(self, policy=None, **kwargs):
        from riichiformer.data.encode import encode_decision
        return RiichiFormerAgent(
            0, policy or MockPolicy(7), encoder=encode_decision,
            stream_meta=self.meta, **kwargs)

    def test_act_returns_legal_event_and_hash(self):
        agent = self._agent()
        event = agent.act(self._decision())
        self.assertIn(event["type"], ("dahai", "reach", "ryukyoku", "ankan", "kakan"))
        self.assertTrue(agent.records)
        self.assertIsNotNone(agent.records[-1].encoded_hash)
        self.assertEqual(len(agent.records[-1].encoded_hash), 64)
        self.assertEqual(agent.stats.encoded, 1)
        # get_reaction returns compact JSON and is parseable.
        agent.set_scene(0, state=self._decision())
        self.assertIsInstance(json.loads(agent.get_reaction()), dict)

    def test_set_scene_compatible_interface(self):
        agent = self._agent()
        decision = self._decision()
        agent.set_scene(0, log='{"type":"start_game"}\n', state=decision)
        reaction = json.loads(agent.get_reaction())
        self.assertIn("type", reaction)
        agent.set_scene(0, state=None)
        with self.assertRaises(AgentError):
            agent.get_reaction()

    def test_policy_failures_are_loud(self):
        decision = self._decision()
        with self.assertRaises(EmptyPolicyOutput):
            self._agent(lambda _d: None).act(decision)
        with self.assertRaises(IllegalAgentAction):
            self._agent(lambda _d: "not an action").act(decision)
        with self.assertRaises(IllegalAgentAction):
            self._agent(lambda _d: 999).act(decision)
        # Mask-illegal action: the crafted mask only offers riichi (37).
        crafted = replace(decision, legal_mask=_mask(37))
        with self.assertRaises(IllegalAgentAction):
            self._agent(lambda _d: 0).act(crafted)
        # Multi-kan without select_kan.
        kan = rf_eval._hand_tiles(__import__("random").Random(4), 13)
        ctx_options = (A.KanOption("ankan", A.TileRef(4), (A.TileRef(4),) * 4),
                       A.KanOption("kakan", A.TileRef(6), (A.TileRef(6),) * 3))
        decision42 = rf_eval._decision(kan, ActionPhase.SELF_TURN, draw=kan[12])
        # Use a direct adapter context test for the ambiguous path; the agent's
        # path needs two real options, which the mock scene does not construct.
        with self.assertRaises(A.AmbiguousKanError):
            A.action_to_mjai(42, A.ActionContext(
                seat=0, phase=ActionPhase.SELF_TURN, legal_mask=_mask(42),
                kan_options=ctx_options))

    def test_truncation_is_loud(self):
        agent = self._agent(max_events=1)
        with self.assertRaises(TruncatedStream):
            agent.act(self._decision())
        self.assertEqual(agent.stats.truncated, 1)

    def test_opponent_concealed_tsumo_is_rejected(self):
        decision = self._decision()
        leak = TsumoEvent(seq_idx=len(decision.events), kyoku_idx=0,
                          actor_rel=1, tile=_tile("7s"))
        leaked = replace(decision, events=tuple(decision.events) + (leak,),
                         before_event_idx=len(decision.events) + 1)
        with self.assertRaises(VisibilityError):
            self._agent().act(leaked)
        # The same event with UNKNOWN is accepted (the POV-correct form).
        unknown = TsumoEvent(seq_idx=len(decision.events), kyoku_idx=0,
                             actor_rel=1, tile=A.UNKNOWN_TILE)
        safe = replace(decision, events=tuple(decision.events) + (unknown,),
                       before_event_idx=len(decision.events) + 1)
        self.assertIn(self._agent().act(safe)["type"], ("dahai", "reach", "ryukyoku",
                                                        "ankan", "kakan"))

    def test_reset_game_and_kyoku(self):
        agent = self._agent()
        agent.act(self._decision())
        agent.reset_kyoku()
        self.assertEqual(agent._kyoku, 0)
        agent.reset_game()
        self.assertEqual(agent.records, [])
        self.assertEqual(agent.stats.decisions, 0)
        self.assertIsNone(agent.last_encoded_hash)

    def test_agent_multi_kan_selects_explicit_option(self):
        base = self._decision()
        own = [_tile("5m")] * 4 + [_tile("6m")] + [
            _tile(text) for text in ("1m", "2m", "3m", "7m", "8m", "9m",
                                     "1p", "2p", "3p")]
        counts = [0] * 34
        for tile in own:
            counts[tile.kind] += 1
        public = [0] * 34
        public[5] = 3  # the 6m pon is public
        public[33] = 1  # dora indicator (C), absent from own
        zeros = (0,) * 34
        meld_zone = [0] * 34
        meld_zone[5] = 3
        indicator_zone = [0] * 34
        indicator_zone[33] = 1
        tile_state = VisibleTileState(
            own_counts=tuple(counts), own_aka=(0, 0, 0),
            public_counts=tuple(public), public_aka=(0, 0, 0),
            public_counts_by_zone=((0,) * 34, tuple(meld_zone), tuple(indicator_zone)),
            dora_indicators=(_tile("C"),), ruleset_id=RulesetId.TENHOU_4P_RANKED)
        players = list(base.players_public)
        players[0] = replace(players[0],
                             melds=(PublicMeld(MeldKind.PON, (_tile("6m"),) * 3, 3),))
        decision = replace(base, tile_state=tile_state,
                           players_public=tuple(players), legal_mask=_mask(42))
        self.assertIn(42, [i for i, m in enumerate(decision.legal_mask) if m])
        self.assertEqual(len(decision.players_public[0].melds), 1)

        class KanPolicy:
            def __call__(self, _decision):
                return 42

            def select_kan(self, options):
                return [option for option in options if option.meld_type == "kakan"][0]

        agent = RiichiFormerAgent(0, KanPolicy(), encoder=None)
        event = agent.act(decision)
        self.assertEqual(event, {"type": "kakan", "actor": 0, "pai": "6m"})
        self.assertEqual(agent.stats.decisions, 1)
        self.assertEqual(agent.stats.ambiguous, 0)

    def test_mock_policy_is_deterministic(self):
        decision = self._decision()
        first = MockPolicy(11)
        second = MockPolicy(11)
        self.assertEqual(first(decision), second(decision))
        self.assertEqual(first(decision), second(decision))


class MockArenaTests(unittest.TestCase):
    def test_10_100_1000_smoke(self):
        for hands in (10, 100, 1000):
            with self.subTest(hands=hands):
                with tempfile.TemporaryDirectory() as out:
                    summary = rf_eval.run_mock(
                        hands=hands, seed=17, key=23, out=out, decisions_per_seat=3)
                    self.assertEqual(summary["hands"], hands)
                    self.assertGreater(summary["decisions"], 0)
                    self.assertEqual(summary["illegal"], 0)
                    self.assertEqual(summary["none"], 0)
                    self.assertEqual(summary["truncated"], 0)
                    self.assertEqual(summary["failures"], 0)
                    self.assertEqual(summary["ambiguous"], 0)
                    self.assertEqual(summary["visibility"], 0)
                    self.assertEqual(sum(summary["rank_histogram"].values()), 4 * hands)
                    self.assertEqual(len(summary["digest_sha256"]), 64)
                    self.assertFalse(summary["tenhou_equivalent"])
                    self.assertIn("R02", summary["note"])
                    self.assertIn("R25", summary["note"])
                    self.assertIn("R26", summary["note"])
                    self.assertEqual(set(summary["seat_starts"]), {"0", "1", "2", "3"})
                    self.assertTrue((Path(out) / "summary.json").is_file())
                    self.assertTrue((Path(out) / "decisions.jsonl").is_file())

    def test_deterministic_replay(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = rf_eval.run_mock(hands=10, seed=5, key=9, out=first)
            b = rf_eval.run_mock(hands=10, seed=5, key=9, out=second)
            self.assertEqual(a["digest_sha256"], b["digest_sha256"])
            self.assertEqual(a["phases"], b["phases"])
            c = rf_eval.run_mock(hands=10, seed=5, key=10, out=None)
            self.assertNotEqual(a["digest_sha256"], c["digest_sha256"])

    def test_output_guard_refuses_frozen_paths(self):
        for relative in ("runs/1v3", "runs/variance", "datasets", "checkpoints", "vendor"):
            with self.subTest(relative=relative):
                with self.assertRaises(SystemExit):
                    rf_eval.guard_out(REPO_ROOT / relative / "rf16")
        with self.assertRaises(SystemExit):
            rf_eval.guard_out(REPO_ROOT / "runs")
        with self.assertRaises(SystemExit):
            rf_eval.guard_out("")
        with tempfile.TemporaryDirectory() as out:
            self.assertEqual(rf_eval.guard_out(out), Path(out).resolve())

    def test_cli_writes_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as out:
            code = rf_eval.main(["--hands", "10", "--seed", "1", "--key", "2",
                                 "--out", out, "--quiet"])
            self.assertEqual(code, 0)
            summary = json.loads((Path(out) / "summary.json").read_text())
            self.assertEqual(summary["hands"], 10)
            self.assertEqual(summary["illegal"], 0)
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(SystemExit):
                rf_eval.main(["--hands", "10", "--seed", "1", "--key", "2",
                              "--out", str(REPO_ROOT / "runs" / "1v3" / "rf16")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
