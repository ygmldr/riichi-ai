#!/usr/bin/env python3
"""Unit tests for the deterministic B0 sample selector."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rf_make_b0_split import select_sample  # noqa: E402


def table() -> pa.Table:
    rows = [
        # game_id, split, representative, quarantine
        ("g1", "train", True, False),
        ("g2", "train", True, False),
        ("g3", "train", False, False),   # duplicate member, not representative
        ("g4", "train", True, True),     # quarantined
        ("g5", "val_holdout", True, False),
        ("g6", "train", True, False),
        ("g7", "train", True, False),
    ]
    return pa.table({
        "game_id": [row[0] for row in rows],
        "split": [row[1] for row in rows],
        "is_representative": [row[2] for row in rows],
        "quarantine": [row[3] for row in rows],
        "value": list(range(len(rows))),
    })


class B0SampleTests(unittest.TestCase):
    def test_filters_non_train_non_representative_and_quarantine(self):
        sample = select_sample(table(), size=4, seed=1)
        self.assertEqual(sample.num_rows, 4)
        selected = set(sample["game_id"].to_pylist())
        self.assertTrue(selected <= {"g1", "g2", "g6", "g7"})
        self.assertNotIn("g3", selected)
        self.assertNotIn("g4", selected)
        self.assertNotIn("g5", selected)

    def test_same_seed_is_reproducible(self):
        a = select_sample(table(), size=4, seed=20261121)
        b = select_sample(table(), size=4, seed=20261121)
        self.assertEqual(a["game_id"].to_pylist(), b["game_id"].to_pylist())

    def test_different_seed_changes_order(self):
        a = select_sample(table(), size=4, seed=1)["game_id"].to_pylist()
        b = select_sample(table(), size=4, seed=2)["game_id"].to_pylist()
        self.assertNotEqual(a, b)

    def test_insufficient_rows_fail_closed(self):
        with self.assertRaises(SystemExit):
            select_sample(table(), size=99, seed=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
