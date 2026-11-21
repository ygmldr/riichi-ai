#!/usr/bin/env python3
"""PR-10 decision snapshot/censor tests.

Synthetic tests pin the phase/pass/censor/kan semantics; the dataset test builds a
stratified 10k-decision sample from read-only local Tenhou logs (skipped if absent).
Run with PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release.
"""

from __future__ import annotations

import gzip
import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.decision import (  # noqa: E402
    DecisionBuildConfig,
    DecisionBuildError,
    build_decisions,
)
from riichiformer.schemas.action_v3 import ActionPhase  # noqa: E402
from riichiformer.schemas.decision_v3 import RulesetDeviation, RulesetId, SourceKind  # noqa: E402


HAND0 = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
         "1p", "2p", "3p", "4p"]
HAND1 = ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
         "E", "S", "W", "N"]
HAND2 = ["P", "F", "C", "1m", "2m", "3m", "4m", "5m", "6m",
         "7m", "8m", "9m", "1p"]
HAND3 = ["2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
         "1s", "2s", "3s", "4s", "5s"]


def raw_game(*, hands=None, middle=None, dora="1p") -> list[dict]:
    raw = [
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        {"type": "start_kyoku", "bakaze": "E", "dora_marker": dora,
         "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0,
         "scores": [25000] * 4,
         "tehais": [list(h) for h in (hands or (HAND0, HAND1, HAND2, HAND3))]},
    ]
    raw.extend(middle or [])
    raw.extend([{"type": "end_kyoku"}, {"type": "end_game"}])
    return raw


def decision_at(samples, pov, before):
    for sample in samples:
        if sample.header.pov == pov and sample.input.before_event_idx == before:
            return sample
    return None


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


class SyntheticDecisionTests(unittest.TestCase):
    def test_minimal_aka_discard_snapshot(self):
        raw = raw_game(middle=[
            {"type": "tsumo", "actor": 0, "pai": "5mr"},
            {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
        ])
        samples = build_decisions(raw)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.header.pov, 0)
        self.assertEqual(sample.header.projection_version, "rf-pov-v3/2")
        self.assertEqual(sample.header.schema_version, "rf-schema-v3/2")
        self.assertEqual(sample.header.ruleset_id, "tenhou_4p_ranked")
        self.assertEqual(sample.header.source_kind, "human_log")
        self.assertEqual(sample.input.phase, ActionPhase.SELF_TURN)
        self.assertEqual(sample.input.before_event_idx, 3)
        self.assertEqual(sample.target.expert_action, 34)  # aka 5m
        self.assertTrue(sample.target.supervision_valid)
        self.assertIsNone(sample.target.censor_reason)
        self.assertIn(sample.target.rank_final, (1, 2, 3, 4))
        self.assertTrue(sample.input.legal_mask[34])
        for event in sample.input.events:
            self.assertLess(event.seq_idx, sample.input.before_event_idx)
        forward = sample.input.to_forward_dict()
        for forbidden in ("expert_action", "rank_final", "censor_reason", "oracle_labels"):
            self.assertNotIn(forbidden, forward)
        record = sample.as_record()
        self.assertEqual(set(record), {"header", "decision_index", "input", "target"})
        self.assertEqual(record["header"], sample.header.as_dict())
        self.assertEqual(record["target"]["expert_action"], 34)

    def test_normal_five_discard(self):
        raw = raw_game(middle=[
            {"type": "tsumo", "actor": 0, "pai": "1m"},
            {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        ])
        samples = build_decisions(raw)
        sample = decision_at(samples, 0, 3)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.target.expert_action, 4)  # normal 5m
        self.assertTrue(sample.target.supervision_valid)

    def test_genuine_pass_vs_preempted_call(self):
        genuine = raw_game(
            hands=(HAND0, ["5m", "5m", "1s", "2s", "3s", "4s", "6s", "7s", "8s",
                           "E", "S", "W", "N"], HAND2, HAND3),
            middle=[
                {"type": "tsumo", "actor": 0, "pai": "1m"},
                {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
                {"type": "tsumo", "actor": 1, "pai": "1s"},
                {"type": "dahai", "actor": 1, "pai": "1s", "tsumogiri": True},
            ])
        pass_sample = decision_at(build_decisions(genuine), 1, 4)
        self.assertIsNotNone(pass_sample)
        self.assertEqual(pass_sample.input.phase, ActionPhase.CALL_RESPONSE)
        self.assertEqual(pass_sample.target.expert_action, 45)
        self.assertTrue(pass_sample.target.supervision_valid)
        self.assertIsNone(pass_sample.target.censor_reason)

        preempted = raw_game(
            hands=(HAND0, ["5m", "5m", "1s", "2s", "3s", "4s", "6s", "7s", "8s",
                           "E", "S", "W", "N"],
                   ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                    "3m", "4m", "5s", "5s"], HAND3),
            middle=[
                {"type": "tsumo", "actor": 0, "pai": "1m"},
                {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
                {"type": "pon", "actor": 1, "target": 0, "pai": "5m",
                 "consumed": ["5m", "5m"]},
                {"type": "dahai", "actor": 1, "pai": "1s", "tsumogiri": False},
            ])
        censored = decision_at(build_decisions(preempted), 2, 4)
        self.assertIsNotNone(censored)
        self.assertEqual(censored.input.phase, ActionPhase.CALL_RESPONSE)
        self.assertEqual(censored.target.expert_action, 45)
        self.assertFalse(censored.target.supervision_valid)
        self.assertEqual(censored.target.censor_reason, "preempted_by_call")
        self.assertTrue(censored.input.legal_mask[45])

    def test_preempted_by_ron_is_censored(self):
        hands = (HAND0,
                 ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                  "3m", "4m", "5s", "5s"],
                 ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                  "3m", "4m", "5s", "5s"], HAND3)
        raw = raw_game(hands=hands, middle=[
            {"type": "tsumo", "actor": 0, "pai": "1m"},
            {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
            {"type": "hora", "actor": 1, "target": 0,
             "deltas": [0, 1000, 0, -1000], "ura_markers": []},
            {"type": "end_kyoku"},
        ])
        # raw_game appends end_kyoku/end_game, so drop the duplicate for the loader.
        raw = raw[:-3] + [{"type": "end_kyoku"}, {"type": "end_game"}]
        samples = build_decisions(raw)
        sample = decision_at(samples, 2, 4)
        self.assertIsNotNone(sample)
        self.assertIsNone(sample.target.expert_action)
        self.assertFalse(sample.target.supervision_valid)
        self.assertEqual(sample.target.censor_reason, "preempted_by_ron")

    def test_double_ron(self):
        hands = (HAND0,
                 ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                  "3m", "4m", "5s", "5s"],
                 ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
                  "3m", "4m", "5s", "5s"], HAND3)
        raw = raw_game(hands=hands, middle=[
            {"type": "tsumo", "actor": 0, "pai": "1m"},
            {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
            {"type": "hora", "actor": 1, "target": 0,
             "deltas": [0, 1000, 0, -1000], "ura_markers": []},
            {"type": "hora", "actor": 2, "target": 0,
             "deltas": [0, 0, 1000, -1000], "ura_markers": []},
        ])
        samples = build_decisions(raw)
        ron = [s for s in samples
               if s.input.before_event_idx == 4 and s.target.expert_action == 43]
        self.assertEqual([s.header.pov for s in ron], [1, 2])
        self.assertTrue(all(s.target.supervision_valid for s in ron))

    def test_multi_kan_candidates(self):
        hands = (["1m", "1m", "1m", "1m", "2m", "2m", "2m", "2m",
                  "3m", "4m", "5m", "6m", "7m"], HAND1,
                 ["P", "F", "C", "9p", "9p", "3m", "4m", "5m", "6m",
                  "7m", "8m", "9m", "1p"], HAND3)
        raw = raw_game(hands=hands, middle=[
            {"type": "tsumo", "actor": 0, "pai": "8m"},
            {"type": "ankan", "actor": 0,
             "consumed": ["1m", "1m", "1m", "1m"]},
            {"type": "dahai", "actor": 0, "pai": "8m", "tsumogiri": True},
        ])
        samples = build_decisions(raw)
        kan = [s for s in samples if s.input.phase == ActionPhase.KAN_SELECTION]
        self.assertEqual(len(kan), 1)
        sample = kan[0]
        self.assertEqual(sample.input.legal_mask.count(True), 2)  # 1m or 2m
        self.assertEqual(sample.target.expert_action, 0)
        self.assertTrue(sample.target.supervision_valid)

    def test_riichi_declaration_then_later_discard(self):
        raw = raw_game(middle=[
            {"type": "tsumo", "actor": 0, "pai": "1m"},
            {"type": "reach", "actor": 0},
            {"type": "reach_accepted", "actor": 0},
            {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        ])
        samples = build_decisions(raw)
        pov0 = [s for s in samples if s.header.pov == 0]
        self.assertEqual(pov0[0].target.expert_action, 37)
        self.assertEqual(pov0[0].input.phase, ActionPhase.SELF_TURN)
        self.assertEqual(pov0[1].target.expert_action, 0)
        self.assertTrue(pov0[1].target.supervision_valid)
        self.assertEqual(pov0[1].input.global_public.kyotaku, 1)
        self.assertEqual(pov0[1].input.players_public[0].score, 24000)

    def test_chi_and_pon_response_actions(self):
        hands = (HAND0,
                 ["1m", "2m", "4m", "5m", "6m", "7m", "8m", "9m",
                  "1s", "2s", "3s", "4s", "5s"],
                 ["5m", "5m", "4m", "4m", "C", "1p", "2p", "3p", "4p",
                  "5p", "6p", "7p", "8p"], HAND3)
        raw = raw_game(hands=hands, middle=[
            {"type": "tsumo", "actor": 0, "pai": "1p"},
            {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
            {"type": "chi", "actor": 1, "target": 0, "pai": "3m",
             "consumed": ["1m", "2m"]},
            {"type": "dahai", "actor": 1, "pai": "4m", "tsumogiri": False},
            {"type": "pon", "actor": 2, "target": 1, "pai": "4m",
             "consumed": ["4m", "4m"]},
            {"type": "dahai", "actor": 2, "pai": "P", "tsumogiri": False},
        ])
        samples = build_decisions(raw)
        chi = decision_at(samples, 1, 4)
        self.assertIsNotNone(chi)
        self.assertIn(chi.target.expert_action, (38, 39, 40))
        self.assertEqual(chi.input.phase, ActionPhase.CALL_RESPONSE)
        self.assertTrue(chi.target.supervision_valid)
        # The pon window is before the pon event, i.e. before index 6.
        pon = decision_at(samples, 2, 6)
        self.assertIsNotNone(pon)
        self.assertEqual(pon.target.expert_action, 41)
        self.assertTrue(pon.target.supervision_valid)

    def test_config_validation(self):
        with self.assertRaises(DecisionBuildError):
            DecisionBuildConfig(ruleset_id=RulesetId.TENHOU_4P_RANKED,
                                ruleset_deviation=(RulesetDeviation.R02,))
        with self.assertRaises(DecisionBuildError):
            DecisionBuildConfig(ruleset_id=RulesetId.MORTAL_ARENA_PINNED,
                                source_kind=SourceKind.HUMAN_LOG)
        from riichiformer.data import pov
        with self.assertRaises(DecisionBuildError):
            DecisionBuildConfig(
                source_kind=SourceKind.ARENA_SELFPLAY,
                ruleset_id=RulesetId.MORTAL_ARENA_PINNED,
                policy=pov.ProjectionPolicy(
                    ruleset_id=RulesetId.TENHOU_4P_RANKED,
                    source_kind=SourceKind.HUMAN_LOG))



    def test_kan_wall_count_matches_tsumo_prefix(self):
        directory = dataset_dir()
        if directory is None:
            self.skipTest("no local 2025 dataset")
        candidates = []
        preferred = directory / "2025010100gm-00a9-0000-029e0f35.mjson"
        if preferred.exists():
            candidates.append(preferred)
        candidates.extend(sorted(directory.glob("*.mjson"))[:10])
        checked = 0
        kan_seen = 0
        for path in candidates:
            raw = read_log(path)
            if not any(event.get("type") in ("ankan", "kakan", "daiminkan")
                       for event in raw):
                continue
            try:
                samples = build_decisions(raw)
            except Exception:
                continue
            for sample in samples:
                prefix = list(sample.input.events)
                start = max(
                    (index for index, event in enumerate(prefix)
                     if event.EVENT_TYPE == "start_kyoku"),
                    default=0,
                )
                expected = 70 - sum(
                    1 for event in prefix[start:] if event.EVENT_TYPE == "tsumo")
                self.assertEqual(sample.input.global_public.tiles_left, expected,
                                 f"{path.name} before {sample.input.before_event_idx}")
                checked += 1
            kan_seen += 1
            if kan_seen >= 2:
                break
        self.assertGreater(kan_seen, 0, "no real kan game found for wall regression")
        self.assertGreater(checked, 0)
        print(f"PR10_KAN_WALL kan_games={kan_seen} decisions_checked={checked}")

class RealLogSampleTests(unittest.TestCase):
    samples: list = []
    errors: list = []
    games = 0
    double_ron_games = 0
    riichi_then_discard = 0

    @classmethod
    def setUpClass(cls):
        directory = dataset_dir()
        if directory is None:
            raise unittest.SkipTest("no local 2025 dataset")
        pool = []
        for path in sorted(directory.glob("*.mjson")):
            raw = read_log(path)
            try:
                batch = build_decisions(raw)
            except Exception as exc:  # noqa: BLE001
                cls.errors.append((path.name, type(exc).__name__, str(exc)[:160]))
                continue
            cls.games += 1
            # double ron: >=2 seats with 43 at the same action boundary
            ron = defaultdict(set)
            for sample in batch:
                if sample.target.expert_action == 43 and sample.target.supervision_valid:
                    ron[sample.input.before_event_idx].add(sample.header.pov)
            if any(len(seats) >= 2 for seats in ron.values()):
                cls.double_ron_games += 1
            # riichi declaration followed by a later discard for that POV
            by_pov = defaultdict(list)
            for sample in batch:
                by_pov[sample.header.pov].append(sample)
            for values in by_pov.values():
                seen_riichi = False
                for sample in values:
                    if sample.target.expert_action == 37:
                        seen_riichi = True
                    elif seen_riichi and sample.target.expert_action is not None \
                            and sample.target.expert_action <= 36:
                        cls.riichi_then_discard += 1
                        break
            pool.extend(batch)
            if len(pool) >= 10500:
                break
        targets = {
            ActionPhase.SELF_TURN.value: 8000,
            ActionPhase.CALL_RESPONSE.value: 1990,
            ActionPhase.KAN_SELECTION.value: 10,
        }
        buckets = defaultdict(list)
        for index, sample in enumerate(pool):
            buckets[sample.input.phase.value].append(index)
        selected: list[int] = []
        for phase, count in targets.items():
            selected.extend(buckets[phase][:count])
        seen = set(selected)
        if len(selected) < 10000:
            for index in range(len(pool)):
                if index not in seen:
                    selected.append(index)
                    seen.add(index)
                    if len(selected) == 10000:
                        break
        selected = selected[:10000]
        cls.samples = [pool[index] for index in selected]

    def test_10k_decision_sample_counts(self):
        self.assertEqual(len(self.samples), 10000)
        self.assertEqual(self.errors, [])
        phases = Counter(sample.input.phase.value for sample in self.samples)
        self.assertEqual(phases[ActionPhase.SELF_TURN.value], 8000)
        self.assertEqual(phases[ActionPhase.CALL_RESPONSE.value], 1990)
        self.assertEqual(phases[ActionPhase.KAN_SELECTION.value], 10)
        print(f"PR10_SAMPLE games={self.games} decisions={len(self.samples)} "
              f"errors={len(self.errors)} double_ron_games={self.double_ron_games} "
              f"riichi_then_discard={self.riichi_then_discard} phases={dict(phases)}")

    def test_zero_illegal_and_censor_reasons(self):
        valid = invalid = illegal = 0
        reasons = Counter()
        actions = Counter()
        for sample in self.samples:
            actions[sample.target.expert_action] += 1
            if sample.target.supervision_valid:
                valid += 1
                action = sample.target.expert_action
                if action is None or not sample.input.legal_mask[action]:
                    illegal += 1
                else:
                    sample.target.validate_against_input(sample.input)
            else:
                invalid += 1
                reasons[sample.target.censor_reason] += 1
        self.assertGreater(valid, 0)
        self.assertGreater(invalid, 0)
        self.assertEqual(illegal, 0)
        self.assertGreater(reasons["preempted_by_ron"], 0)
        self.assertGreater(reasons["preempted_by_call"], 0)
        self.assertGreater(actions[45], 0)
        self.assertGreater(actions[37], 0)
        self.assertGreater(actions[42], 0)
        self.assertGreater(actions[43], 0)
        self.assertTrue(all(sample.target.rank_final in (1, 2, 3, 4)
                            for sample in self.samples))
        print(f"PR10_LABELS valid={valid} invalid={invalid} illegal={illegal} "
              f"censor={dict(reasons)} actions={dict(actions)}")

    def test_header_attribution_on_every_record(self):
        for sample in self.samples:
            record = sample.as_record()
            self.assertEqual(set(record["header"]),
                             {"pov", "projection_version", "schema_version",
                              "ruleset_id", "source_kind", "reveal_opponent_ankan"})
            self.assertEqual(record["header"]["pov"], sample.header.pov)
            self.assertEqual(record["header"]["projection_version"], "rf-pov-v3/2")
            self.assertNotIn("rank_final", record["input"])
            self.assertNotIn("expert_action", record["input"])

    def test_physical_tile_conservation(self):
        for sample in self.samples:
            state = sample.input.tile_state
            for kind in range(34):
                self.assertLessEqual(
                    state.own_counts[kind] + state.public_counts[kind], 4)
            for color in range(3):
                self.assertLessEqual(state.own_aka[color] + state.public_aka[color], 1)
            for kind in range(34):
                self.assertEqual(
                    sum(zone[kind] for zone in state.public_counts_by_zone),
                    state.public_counts[kind])


if __name__ == "__main__":
    unittest.main(verbosity=2)
