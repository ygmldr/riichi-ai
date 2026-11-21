#!/usr/bin/env python3
"""Tests for PR-07 / PR-07R drift measurement.

The original nine tests from the PR-07 draft are kept unchanged in intent;
PR-07R adds negative coverage for the fixed bugs:

* tsumo ``hora`` events carry ``target == actor`` and must not count as deal-ins;
* malformed/incomplete logs fail closed instead of being aggregated as parsed;
* a reach turn bucket uses the declaring player's own discard index;
* the CLI report contains the exact command, seed, cap, strata, confounders and
  the explicit "recent = stronger is NOT a conclusion" warning;
* parse failures and CSV/summary row counts are visible and consistent.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import rf_drift  # noqa: E402
from riichiformer.data import canonicalize, drift  # noqa: E402


def sample_events():
    return [
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0,
         "oya": 0, "scores": [25000] * 4},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 1, "pai": "2m"},
        {"type": "reach", "actor": 2},
        {"type": "hora", "actor": 0, "target": 1, "deltas": [8000, -8000, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "start_kyoku", "bakaze": "S", "kyoku": 4, "honba": 1, "kyotaku": 0,
         "oya": 3, "scores": [30000, 20000, 25000, 25000]},
        {"type": "tsumo", "actor": 3, "pai": "3m"},
        {"type": "dahai", "actor": 3, "pai": "3m"},
        {"type": "ryukyoku", "deltas": [-3000, 1000, 1000, 1000]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]


def ndjson(events):
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode()


def write_manifest(root: Path, rows: list[dict]) -> Path:
    manifest = root / "audit.parquet"
    pq.write_table(pa.Table.from_pylist(rows), manifest)
    return manifest


def run_cli(manifest: Path, out: Path, report: Path, extra: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = rf_drift.main([
            "--manifest", str(manifest), "--out", str(out), "--report", str(report),
            "--repo-root", str(REPO_ROOT), *extra,
        ])
    return code, stdout.getvalue(), stderr.getvalue()


class UnitTests(unittest.TestCase):
    def test_windows_and_turn_buckets(self):
        self.assertEqual(drift.window_for(2019), "2019_2022")
        self.assertEqual(drift.window_for(2022), "2019_2022")
        self.assertEqual(drift.window_for(2023), "2023_2024")
        self.assertEqual(drift.window_for(2025), "2025")
        self.assertEqual(drift.window_for(2026), "2026")
        self.assertEqual(drift.window_for(2009), "pre_2019")
        self.assertEqual(drift.turn_bucket(5), "0-5")
        self.assertEqual(drift.turn_bucket(6), "6-11")
        self.assertEqual(drift.turn_bucket(12), "12+")

    def test_wilson_and_mean_ci(self):
        lo, hi = drift.wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertEqual(drift.wilson_interval(0, 0), (None, None))
        lo0, hi0 = drift.wilson_interval(0, 10)
        self.assertEqual(lo0, 0.0)
        self.assertGreater(hi0, 0)
        mean, lo_m, hi_m = drift.mean_ci([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertLess(lo_m, mean)
        self.assertGreater(hi_m, mean)

    def test_rate_interval_allows_more_than_one_per_kyoku(self):
        lo, hi = drift.rate_interval(185, 100)
        self.assertGreater(hi, 1.0)
        self.assertLess(lo, 1.85)
        self.assertEqual(drift.rate_interval(0, 0), (None, None))

    def test_percentile(self):
        values = list(range(1, 1001))
        self.assertEqual(drift.percentile(values, 0.50), 500)
        self.assertEqual(drift.percentile(values, 0.90), 900)
        self.assertEqual(drift.percentile(values, 0.99), 990)
        self.assertEqual(drift.percentile(values, 0.999), 999)
        self.assertIsNone(drift.percentile([], 0.5))

    def test_parse_game_metrics(self):
        record = drift.parse_game(ndjson(sample_events()),
                                  "2019/2019010100gm-00a9-0000-aaaa1111.mjson",
                                  "tenhou:x", 2019, hmac_key=None)
        self.assertEqual(record.events, 14)
        self.assertEqual(record.kyoku_count, 2)
        self.assertEqual(record.reach_count, 1)
        self.assertEqual(record.hora_count, 1)
        self.assertEqual(record.ryukyoku_count, 1)
        self.assertEqual(record.deal_ins, 1)
        self.assertEqual(record.hora_points, [8000])
        self.assertEqual(record.month, 1)
        self.assertTrue(record.kyoku_rows[1]["is_last"])
        self.assertFalse(record.kyoku_rows[0]["is_last"])
        self.assertIsNone(record.decisions)

    def test_aggregate_rates_and_coverage(self):
        g1 = drift.parse_game(ndjson(sample_events()), "2019/x.mjson", "g1", 2019)
        g2 = drift.parse_game(ndjson(sample_events()), "2026/y.mjson", "g2", 2026)
        cohorts = drift.aggregate_cohorts([g1, g2])
        window_rows = {row["cohort"]: row for row in cohorts if row["scope"] == "window"}
        self.assertEqual(window_rows["2019_2022"]["games"], 1)
        self.assertEqual(window_rows["2026"]["games"], 1)
        self.assertAlmostEqual(window_rows["2019_2022"]["reach_per_kyoku"], 0.5)
        self.assertAlmostEqual(window_rows["2019_2022"]["hora_per_kyoku"], 0.5)
        coverage = drift.cohort_windows_with_coverage([g1, g2])
        self.assertIn("2026", coverage)
        self.assertEqual(coverage["2026"]["games"], 1)
        self.assertTrue(coverage["2026"]["note"])

    def test_strata_last_and_turn_shares(self):
        records = [drift.parse_game(ndjson(sample_events()), "2025/x.mjson", "g", 2025)]
        rows = drift.aggregate_strata(records)
        last = [row for row in rows if row["stratum_type"] == "window_bakaze_last"]
        self.assertTrue(any("all_last" in row["stratum"] for row in last))
        turn = [row for row in rows if row["stratum_type"] == "window_bakaze_turn"]
        shares = [row["reach_share"] for row in turn if row["reach_share"]]
        self.assertTrue(shares)
        self.assertAlmostEqual(sum(shares), 1.0, places=6)

    def test_hmac_players_optional(self):
        record = drift.parse_game(ndjson(sample_events()), "2025/x.mjson", "g", 2025,
                                  hmac_key=b"x" * 32)
        self.assertEqual(len(record.player_hmacs), 4)
        basic = drift._basic_fields([record])
        self.assertEqual(basic["unique_players"], 4)

    # ---- PR-07R regression coverage -------------------------------------

    def test_tsumo_hora_is_not_a_deal_in(self):
        events = [
            {"type": "start_game", "names": ["a", "b", "c", "d"]},
            {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
             "kyotaku": 0, "oya": 0, "scores": [25000] * 4},
            # tenhou/mjai tsumo: target == actor
            {"type": "hora", "actor": 1, "target": 1,
             "deltas": [-2000, 7000, -2500, -2500]},
            # ron: target is the discarder, therefore a deal-in
            {"type": "hora", "actor": 2, "target": 0, "deltas": [-8000, 0, 8000, 0]},
            # absent target is also tsumo
            {"type": "hora", "actor": 3, "deltas": [-1000, -1000, -1000, 3000]},
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ]
        record = drift.parse_game(ndjson(events), "2025/x.mjson", "g", 2025)
        self.assertEqual(record.hora_count, 3)
        self.assertEqual(record.tsumo_count, 2)
        self.assertEqual(record.ron_count, 1)
        self.assertEqual(record.deal_ins, 1)
        self.assertEqual(record.hora_points, [7000, 8000, 3000])

    def test_invalid_sequence_fails_closed(self):
        with self.assertRaises(canonicalize.CanonicalizeError) as ctx:
            drift.parse_game(ndjson(sample_events()[:-1]), "2025/x.mjson", "g", 2025)
        self.assertEqual(ctx.exception.reason, "missing_end_game")

    def test_reach_turn_uses_declaring_players_own_discards(self):
        events = [
            {"type": "start_game", "names": ["a", "b", "c", "d"]},
            {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
             "kyotaku": 0, "oya": 0, "scores": [25000] * 4},
            {"type": "dahai", "actor": 0, "pai": "1m"},
            {"type": "dahai", "actor": 1, "pai": "2m"},
            {"type": "dahai", "actor": 2, "pai": "3m"},
            {"type": "dahai", "actor": 3, "pai": "4m"},
            {"type": "dahai", "actor": 1, "pai": "5m"},
            {"type": "dahai", "actor": 2, "pai": "6m"},
            {"type": "reach", "actor": 0},  # actor 0's own first discard turn
            {"type": "dahai", "actor": 0, "pai": "7m"},
            {"type": "reach_accepted", "actor": 0},
            {"type": "end_kyoku"},
            {"type": "end_game"},
        ]
        record = drift.parse_game(ndjson(events), "2025/x.mjson", "g", 2025)
        self.assertEqual(record.kyoku_rows[0]["reach_turns"], [1])
        rows = drift.aggregate_strata([record])
        turn = {row["stratum"].split("|")[-1]: row
                for row in rows if row["stratum_type"] == "window_bakaze_turn"}
        self.assertEqual(turn["0-5"]["reach"], 1)
        self.assertEqual(turn["6-11"]["reach"], 0)

    def test_action_counts_and_termination(self):
        record = drift.parse_game(ndjson(sample_events()), "2025/x.mjson", "g", 2025)
        self.assertEqual(record.dahai_count, 3)
        self.assertEqual(record.chi_count, 0)
        self.assertEqual(record.pon_count, 0)
        self.assertEqual(record.kan_count, 0)
        self.assertEqual(record.reach_accepted_count, 0)
        self.assertEqual(record.final_kyoku_label, "S4")
        self.assertEqual(record.final_event, "ryukyoku")
        self.assertTrue(record.reached_south4)
        window = {row["cohort"]: row for row in drift.aggregate_cohorts([record])}
        self.assertAlmostEqual(window["2025"]["reached_south4_share"], 1.0)
        self.assertAlmostEqual(window["2025"]["final_ryukyoku_share"], 1.0)

    def test_reached_south4_requires_s4_or_west_extension(self):
        def termination_events(bakaze, kyoku):
            return [
                {"type": "start_game", "names": ["a", "b", "c", "d"]},
                {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
                 "kyotaku": 0, "oya": 0, "scores": [25000] * 4},
                {"type": "hora", "actor": 0, "target": 1,
                 "deltas": [8000, -8000, 0, 0]},
                {"type": "end_kyoku"},
                {"type": "start_kyoku", "bakaze": bakaze, "kyoku": kyoku,
                 "honba": 0, "kyotaku": 0, "oya": 0, "scores": [33000] * 4},
                {"type": "hora", "actor": 0, "target": 1,
                 "deltas": [8000, -8000, 0, 0]},
                {"type": "end_kyoku"},
                {"type": "end_game"},
            ]

        expected = {
            ("S", 1): False,
            ("S", 2): False,
            ("S", 3): False,
            ("S", 4): True,
            ("W", 1): True,
        }
        records = []
        for (bakaze, kyoku), reached in expected.items():
            record = drift.parse_game(
                ndjson(termination_events(bakaze, kyoku)),
                f"2025/{bakaze}{kyoku}.mjson", f"g-{bakaze}{kyoku}", 2025)
            self.assertEqual(record.final_kyoku_label, f"{bakaze}{kyoku}")
            self.assertIs(record.reached_south4, reached,
                          f"last kyoku {bakaze}{kyoku} must be reached_south4={reached}")
            records.append(record)
        basic = drift._basic_fields(records)
        self.assertEqual(basic["reached_south4_games"], 2)
        self.assertAlmostEqual(
            {row["cohort"]: row for row in drift.aggregate_cohorts(records)}["2025"][
                "reached_south4_share"], 0.4)


class CliTests(unittest.TestCase):
    def test_cli_no_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            rows = []
            for year in (2019, 2026):
                directory = data / str(year)
                directory.mkdir(parents=True)
                name = f"{year}010100gm-00a9-0000-{year}abc.mjson"
                (directory / name).write_bytes(gzip.compress(ndjson(sample_events()), mtime=0))
                rows.append({"root": str(data), "relpath": f"{year}/{name}", "year": year,
                             "kind": "leaf", "invalid_reason": None})
            manifest = write_manifest(root, rows)
            out = root / "drift"
            report = root / "DRIFT.md"
            code, stdout, stderr = run_cli(
                manifest, out, report,
                ["--per-year", "2", "--workers", "2", "--no-replay"])
            self.assertEqual(code, 0, stderr)
            self.assertTrue((out / "drift_cohorts.csv").is_file())
            self.assertTrue((out / "drift_strata.csv").is_file())
            summary = json.loads((out / "drift_summary.json").read_text())
            self.assertEqual(summary["games_parsed"], 2)
            self.assertEqual(summary["games_failed"], 0)
            self.assertEqual(summary["teacher_audit"], "NOT_RUN_no_authorized_teacher_protocol")
            self.assertEqual(summary["ai_provenance"], "all_unknown")
            self.assertEqual(summary["schema"], "rf-drift-summary/2")
            self.assertFalse(summary["with_replay"])
            text = report.read_text()
            self.assertIn("2026", text)
            self.assertNotIn("names", text)
            self.assertIn("--per-year 2", text)
            self.assertIn("seed: `20261121`", text)
            self.assertIn("per-year cap: `2`", text)
            self.assertIn("recent = stronger", text)
            self.assertIn("NOT a conclusion", text)
            self.assertIn("## Confounders", text)
            self.assertIn("## Game termination", text)
            self.assertIn("## Window rates (95% CI)", text)
            self.assertIn("not imported (`--no-replay`)", text)
            self.assertNotIn("decisions computed: True", text)
            self.assertNotIn("tenhou_4p_hanchan_aka_v1", text)
            cohorts = list(csv.DictReader((out / "drift_cohorts.csv").open()))
            strata = list(csv.DictReader((out / "drift_strata.csv").open()))
            self.assertTrue(any(row["scope"] == "window" for row in cohorts))
            self.assertEqual(len(cohorts), len(summary["cohorts"]))
            self.assertEqual(len(strata), len(summary["strata"]))
            for row in summary["cohorts"]:
                if row["scope"] == "window":
                    self.assertIn(
                        f"| `{row['cohort']}` | {row['games']} | {row['kyoku']} | "
                        f"{row['decisions']} |", text)
            for name, meta in summary["artifacts"].items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(),
                                 meta["sha256"])
            self.assertEqual(hashlib.sha256(report.read_bytes()).hexdigest(),
                             summary["report"]["sha256"])

    def test_cli_counts_parse_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data" / "2025"
            data.mkdir(parents=True)
            good = data / "2025010100gm-00a9-0000-00000001.mjson"
            bad_gzip = data / "2025010100gm-00a9-0000-00000002.mjson"
            bad_seq = data / "2025010100gm-00a9-0000-00000003.mjson"
            good.write_bytes(gzip.compress(ndjson(sample_events()), mtime=0))
            bad_gzip.write_bytes(b"\x1f\x8bnot-a-real-gzip-stream")
            bad_seq.write_bytes(gzip.compress(ndjson(sample_events()[:-1]), mtime=0))
            rows = [
                {"root": str(data.parent), "relpath": f"2025/{path.name}",
                 "year": 2025, "kind": "leaf", "invalid_reason": None}
                for path in (good, bad_gzip, bad_seq)
            ]
            manifest = write_manifest(root, rows)
            out = root / "drift"
            report = root / "DRIFT.md"
            code, stdout, stderr = run_cli(
                manifest, out, report,
                ["--per-year", "3", "--workers", "2", "--no-replay"])
            self.assertEqual(code, 0, stderr)
            summary = json.loads((out / "drift_summary.json").read_text())
            self.assertEqual(summary["games_selected"], 3)
            self.assertEqual(summary["games_parsed"], 1)
            self.assertEqual(summary["games_failed"], 2)
            self.assertEqual(summary["errors"].get("parse:gzip_decode"), 1)
            self.assertEqual(summary["errors"].get("parse:missing_end_game"), 1)
            text = report.read_text()
            self.assertIn("gzip_decode", text)
            self.assertIn("missing_end_game", text)
            self.assertIn("| 2025 | 3 | 1 | 2 |", text)

    def test_cli_hmac_never_serialises_player_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data" / "2025"
            data.mkdir(parents=True)
            name = "2025010100gm-00a9-0000-00000001.mjson"
            (data / name).write_bytes(gzip.compress(ndjson(sample_events()), mtime=0))
            manifest = write_manifest(root, [
                {"root": str(data.parent), "relpath": f"2025/{name}",
                 "year": 2025, "kind": "leaf", "invalid_reason": None}])
            out = root / "drift"
            report = root / "DRIFT.md"
            with mock.patch.dict(os.environ,
                                 {"RF_PLAYER_HMAC_KEY": "test-key-0123456789"}):
                code, _stdout, stderr = run_cli(
                    manifest, out, report,
                    ["--per-year", "1", "--workers", "1", "--no-replay"])
            self.assertEqual(code, 0, stderr)
            summary = json.loads((out / "drift_summary.json").read_text())
            self.assertTrue(summary["hmac_players_available"])
            window = [row for row in summary["cohorts"] if row["scope"] == "window"][0]
            self.assertEqual(window["unique_players"], 4)
            serialised = json.dumps(summary)
            self.assertNotIn("player_hmacs", serialised)
            self.assertNotIn("names", serialised)
            self.assertNotIn("names", report.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
