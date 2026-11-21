#!/usr/bin/env python3
"""PR-10 physical tile ledger tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.tile_counter import TileCounter, TileCounterError  # noqa: E402
from riichiformer.schemas.event_v3 import tile_from_text  # noqa: E402


def tile(text: str):
    return tile_from_text(text)


class TileCounterTests(unittest.TestCase):
    def test_aka_call_migrates_river_to_meld_without_double_count(self):
        counter = TileCounter()
        for text in ("5m", "4m", "6m"):
            counter.add_own(tile(text))
        counter.add_own(tile("5mr"))
        counter.remove_own(tile("5mr"))
        counter.add_river(tile("5mr"))
        self.assertEqual(counter.river[4], 1)
        self.assertEqual(counter.river_aka[0], 1)
        counter.call_tile(tile("5mr"))
        counter.add_meld(tile("4m"))
        counter.add_meld(tile("6m"))
        self.assertEqual(counter.river[4], 0)
        self.assertEqual(counter.river_aka[0], 0)
        self.assertEqual(counter.meld[4], 1)
        self.assertEqual(counter.meld_aka[0], 1)
        self.assertEqual(counter.public_counts[4], 1)
        self.assertEqual(sum(counter.public_aka), 1)
        counter.assert_conservation()

    def test_conservation_and_aka_bounds(self):
        counter = TileCounter()
        for _ in range(4):
            counter.add_own(tile("5m"))
        with self.assertRaises(TileCounterError):
            counter.add_own(tile("5m"))
        with self.assertRaises(TileCounterError):
            counter.add_own(tile("5mr"))
        with self.assertRaises(TileCounterError):
            counter.add_river(tile("5mr"))
        with self.assertRaises(TileCounterError):
            counter.remove_river(tile("1m"))
        for _ in range(4):
            counter.add_other_public(tile("1m"))
        with self.assertRaises(TileCounterError):
            counter.add_other_public(tile("1m"))
        counter.assert_conservation()

    def test_public_overflow_across_zones(self):
        counter = TileCounter()
        for _ in range(2):
            counter.add_river(tile("3m"))
        for _ in range(2):
            counter.add_meld(tile("3m"))
        self.assertEqual(counter.public_counts[2], 4)
        with self.assertRaises(TileCounterError):
            counter.add_indicator(tile("3m"))

    def test_snapshot_zone_sums_and_indicators(self):
        counter = TileCounter()
        hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
                "1p", "2p", "3p", "4p"]
        for text in hand:
            counter.add_own(tile(text))
        counter.add_river(tile("9s"))
        counter.add_meld(tile("1p"))
        counter.add_meld(tile("2p"))
        counter.add_meld(tile("3p"))
        counter.add_indicator(tile("1p"))
        state = counter.to_visible_tile_state((tile("1p"),))
        self.assertEqual(sum(state.public_counts), 5)
        self.assertEqual(sum(state.public_counts_by_zone[0]), 1)
        self.assertEqual(sum(state.public_counts_by_zone[1]), 3)
        self.assertEqual(sum(state.public_counts_by_zone[2]), 1)
        with self.assertRaises(TileCounterError):
            counter.to_visible_tile_state((tile("2p"),))

    def test_call_without_river_tile_fails(self):
        counter = TileCounter()
        with self.assertRaises(TileCounterError):
            counter.call_tile(tile("5m"))

    def test_unknown_and_invalid_tiles_rejected(self):
        counter = TileCounter()
        with self.assertRaises(TileCounterError):
            counter.add_own(tile_from_text("?"))
        with self.assertRaises(TileCounterError):
            counter.add_own("5m")
        with self.assertRaises(TileCounterError):
            TileCounter(ruleset_id="tenhou")


if __name__ == "__main__":
    unittest.main(verbosity=2)
