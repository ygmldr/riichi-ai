#!/usr/bin/env python3
"""Tests for PR-04 libriichi replay adapter.

The integration test uses one local dataset log (never copied into the repo) and is
skipped when libriichi/datasets are unavailable.  Unit tests cover score accounting.
"""

from __future__ import annotations

import gzip
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "vendor" / "mortal" / "target" / "release"))

from riichiformer.data import replay  # noqa: E402


def have_libriichi() -> bool:
    try:
        import libriichi  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def first_dataset_log():
    for year in ("2026", "2025", "2024", "2023"):
        directory = REPO_ROOT / "datasets" / "extracted" / year
        if not directory.is_dir():
            continue
        files = sorted(directory.glob("*.mjson"))
        if files:
            return files[0]
    return None


class ScoreTests(unittest.TestCase):
    def test_reach_and_kyotaku_are_accounted(self):
        events = [
            {"type": "start_game", "names": ["a", "b", "c", "d"]},
            {"type": "start_kyoku", "scores": [25000] * 4, "kyotaku": 0},
            {"type": "reach_accepted", "actor": 0},
            {"type": "hora", "actor": 0, "deltas": [-1000, 0, 0, 0]},
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ]
        self.assertEqual(replay.final_scores_from_events(events), [24000, 25000, 25000, 25000])

    def test_ryukyoku_and_next_kyoku_authoritative_scores(self):
        events = [
            {"type": "start_kyoku", "scores": [25000] * 4, "kyotaku": 0},
            {"type": "ryukyoku", "deltas": [-3000, 1000, 1000, 1000]},
            {"type": "end_kyoku"},
            {"type": "start_kyoku", "scores": [22000, 26000, 26000, 26000], "kyotaku": 0},
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ]
        self.assertEqual(replay.final_scores_from_events(events), [22000, 26000, 26000, 26000])

    def test_no_start_kyoku_returns_none(self):
        self.assertIsNone(replay.final_scores_from_events([{"type": "start_game"}]))


@unittest.skipUnless(have_libriichi(), "libriichi is not importable")
class ReplayIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = first_dataset_log()
        if cls.path is None:
            raise unittest.SkipTest("no local dataset log")
        raw = cls.path.read_bytes()
        cls.text = (gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw).decode("utf-8")

    def test_real_game_replays_and_all_actions_are_legal(self):
        result = replay.replay_game(self.text, oracle=False)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.seats), 4)
        self.assertGreater(sum(seat["decisions"] for seat in result.seats), 0)
        for seat in result.seats:
            self.assertEqual(seat["invalid_actions"], [])
            self.assertGreater(seat["min_legal_count"], 0)
        self.assertIsNotNone(result.final_scores)
        self.assertEqual(len(result.final_scores), 4)

    def test_masks_are_46_wide(self):
        result = replay.replay_game(self.text, oracle=False, include_masks=True)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.masks), 4)
        for seat_masks in result.masks:
            for mask in seat_masks:
                self.assertEqual(len(mask), replay.ACTION_SPACE)

    def test_bad_json_returns_error_not_exception(self):
        result = replay.replay_game("{not json}\n")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_kind, "bad_json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
