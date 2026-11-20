#!/usr/bin/env python3
"""Tests for PR-06 player HMAC / empirical-Bayes quality scoring."""

from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import rf_quality  # noqa: E402
from riichiformer.data import player_quality as q  # noqa: E402

KEY = b"unit-test-key-0123456789"


def app(game_id, seat, hmac_id, when, rank, year=2020, source="tenhou_old"):
    return q.Appearance(game_id=game_id, seat=seat, player_hmac=hmac_id,
                        match_time=when, rank=rank, year=year, source_id=source)


class HmacTests(unittest.TestCase):
    def test_key_required_and_strength(self):
        with self.assertRaises(q.HmacKeyMissing):
            q.hmac_key_from_env({})
        with self.assertRaises(q.HmacKeyMissing):
            q.hmac_key_from_env({"RF_PLAYER_HMAC_KEY": "short"})
        self.assertEqual(q.hmac_key_from_env({"RF_PLAYER_HMAC_KEY": "x" * 16}), b"x" * 16)

    def test_hmac_deterministic_and_distinct(self):
        a = q.player_hmac("player-a", KEY)
        b = q.player_hmac("player-a", KEY)
        c = q.player_hmac("player-b", KEY)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 64)


class ScoreTests(unittest.TestCase):
    def test_future_results_do_not_change_target_score(self):
        t0 = datetime(2020, 1, 1)
        base = [app("g1", 0, "p", t0, 2), app("g2", 0, "p", t0 + timedelta(days=1), 3),
                app("g3", 0, "p", t0 + timedelta(days=2), 4)]
        rows = q.compute_quality(base, min_games=1)
        g2 = next(row for row in rows if row["game_id"] == "g2")
        changed = [base[0], base[1], app("g3", 0, "p", t0 + timedelta(days=2), 1)]
        rows2 = q.compute_quality(changed, min_games=1)
        g2b = next(row for row in rows2 if row["game_id"] == "g2")
        self.assertEqual(g2["q"], g2b["q"])

    def test_target_own_rank_not_used(self):
        t0 = datetime(2020, 1, 1)
        games = [app("g1", 0, "p", t0, 1), app("g2", 0, "p", t0 + timedelta(days=1), 2)]
        a = next(row for row in q.compute_quality(games, min_games=1) if row["game_id"] == "g2")
        games2 = [games[0], app("g2", 0, "p", t0 + timedelta(days=1), 4)]
        b = next(row for row in q.compute_quality(games2, min_games=1) if row["game_id"] == "g2")
        self.assertEqual(a["q"], b["q"])
        self.assertNotEqual(a["target_rank_for_eval"], b["target_rank_for_eval"])

    def test_single_game_is_low_confidence(self):
        rows = q.compute_quality([app("g1", 0, "p", datetime(2020, 1, 1), 1)])
        row = rows[0]
        self.assertEqual(row["n_prior_games"], 0)
        self.assertTrue(row["insufficient_history"])
        self.assertEqual(row["confidence"], 0.0)
        self.assertAlmostEqual(row["q"], q.DEFAULT_PRIOR_MEAN_RANK)

    def test_min_games_boundary(self):
        t0 = datetime(2020, 1, 1)
        games = [app(f"g{i}", 0, "p", t0 + timedelta(days=i), 2) for i in range(50)]
        games.append(app("target", 0, "p", t0 + timedelta(days=60), 2))
        rows = q.compute_quality(games, min_games=50)
        target = next(row for row in rows if row["game_id"] == "target")
        self.assertEqual(target["n_prior_games"], 50)
        self.assertFalse(target["insufficient_history"])

    def test_history_game_ids_excludes_non_train(self):
        t0 = datetime(2020, 1, 1)
        allowed = {
            "g1", "g2", "g3",
        }
        games = [app("g1", 0, "p", t0, 1), app("g2", 0, "p", t0 + timedelta(days=1), 2),
                 app("g3", 0, "p", t0 + timedelta(days=2), 3),
                 app("val", 0, "p", t0 + timedelta(days=3), 4),
                 app("target", 0, "p", t0 + timedelta(days=4), 2)]
        a = next(row for row in q.compute_quality(games, min_games=1, history_game_ids=allowed)
                 if row["game_id"] == "target")
        changed = [g for g in games if g.game_id != "val"]
        changed.append(app("val", 0, "p", t0 + timedelta(days=3), 1))
        b = next(row for row in q.compute_quality(changed, min_games=1, history_game_ids=allowed)
                 if row["game_id"] == "target")
        self.assertEqual(a["q"], b["q"])

    def test_missing_time_marks_non_temporal(self):
        games = [app("g1", 0, "p", None, 1), app("g2", 0, "p", None, 2)]
        rows = q.compute_quality(games, min_games=1)
        self.assertTrue(all(not row["temporal"] for row in rows))

    def test_top25_only_marks_eligible_best_players(self):
        t0 = datetime(2020, 1, 1)
        games = []
        for i in range(60):
            games.append(app(f"a{i}", 0, "good", t0 + timedelta(days=i), 1))
            games.append(app(f"a{i}", 1, "bad", t0 + timedelta(days=i), 4))
        games.append(app("target", 0, "good", t0 + timedelta(days=100), 1))
        games.append(app("target", 1, "bad", t0 + timedelta(days=100), 4))
        games.append(app("target", 2, "new", t0 + timedelta(days=100), 2))
        rows = q.compute_quality(games, min_games=50)
        target = [row for row in rows if row["game_id"] == "target"]
        self.assertEqual(len(target), 3)
        self.assertTrue(next(row for row in target if row["seat"] == 0)["top25_seat"])
        self.assertFalse(next(row for row in target if row["seat"] == 1)["top25_seat"])
        self.assertFalse(next(row for row in target if row["seat"] == 2)["top25_seat"])

    def test_ranks_and_time_parser(self):
        self.assertEqual(q.ranks_from_scores([25000, 25000, 30000, 20000]), [2, 3, 1, 4])
        self.assertEqual(q.match_time_from_relpath("2024010101gm-00a9-0000-aa.mjson"),
                         datetime(2024, 1, 1, 1))
        self.assertIsNone(q.match_time_from_relpath("nodate.mjson"))

    def test_assert_no_names(self):
        q.assert_no_names([{"game_id": "g", "player_hmac": "abc"}])
        with self.assertRaises(ValueError):
            q.assert_no_names([{"game_id": "g", "player_name": "x"}])


class ExtractTests(unittest.TestCase):
    def test_extract_has_only_hmac(self):
        events = [
            {"type": "start_game", "names": ["aa", "bb", "cc", "dd"]},
            {"type": "start_kyoku", "scores": [25000] * 4},
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ]
        raw = gzip.compress(("\n".join(json.dumps(e) for e in events) + "\n").encode(), mtime=0)
        rows, error = q.extract_appearances(raw, "2024010101gm-00a9-0000-1234abcd.mjson",
                                            "tenhou:2024010101gm-00a9-0000-1234abcd",
                                            2024, "tenhou_recent", KEY)
        self.assertIsNone(error)
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(len(row.player_hmac), 64)
            self.assertFalse(hasattr(row, "name"))
        self.assertEqual({row.rank for row in rows}, {1, 2, 3, 4})

    def test_invalid_names(self):
        events = [{"type": "start_game", "names": ["a"]}, {"type": "end_game"}]
        raw = gzip.compress(("\n".join(json.dumps(e) for e in events) + "\n").encode(), mtime=0)
        rows, error = q.extract_appearances(raw, "x.mjson", "g", 2024, "s", KEY)
        self.assertEqual(rows, [])
        self.assertEqual(error, "invalid_names")


@unittest.skipUnless(Path("datasets/extracted/2024").is_dir(), "local dataset required")
class CliTests(unittest.TestCase):
    def test_basic_mode_without_key_and_linked_mode(self):
        manifest = REPO_ROOT / "artifacts" / "riichiformer" / "split" / "split_manifest_L.parquet"
        if not manifest.is_file():
            self.skipTest("PR-05 manifest missing")
        with tempfile.TemporaryDirectory() as tmp:
            out, report = Path(tmp) / "quality", Path(tmp) / "audit.md"
            env = {k: v for k, v in os.environ.items() if k != "RF_PLAYER_HMAC_KEY"}
            with mock.patch.dict(os.environ, env, clear=True):
                code = rf_quality.main([
                    "--manifest", str(manifest), "--out", str(out), "--report", str(report),
                    "--sample", "4", "--workers", "2", "--repo-root", str(REPO_ROOT),
                ])
            self.assertEqual(code, 0)
            summary = json.loads((out / "quality_summary.json").read_text())
            self.assertFalse(summary["player_linked"])
            self.assertEqual(summary["rows"], 0)
            self.assertTrue(report.is_file())
            self.assertNotIn("player-a", report.read_text())

            with mock.patch.dict(os.environ, {"RF_PLAYER_HMAC_KEY": "x" * 32}):
                code = rf_quality.main([
                    "--manifest", str(manifest), "--out", str(out), "--report", str(report),
                    "--sample", "4", "--workers", "2", "--repo-root", str(REPO_ROOT),
                ])
            self.assertEqual(code, 0)
            summary = json.loads((out / "quality_summary.json").read_text())
            self.assertTrue(summary["player_linked"])
            self.assertEqual(summary["rows"], 16)
            table = pq.read_table(out / "quality_scores.parquet")
            self.assertIn("player_hmac", table.column_names)
            self.assertNotIn("name", " ".join(table.column_names).lower())
            self.assertEqual(table.num_rows, 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
