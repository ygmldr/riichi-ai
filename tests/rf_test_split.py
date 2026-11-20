#!/usr/bin/env python3
"""Tests for PR-05 split registry (`riichiformer/data/split.py`, `rf_build_split.py`)."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import rf_build_split  # noqa: E402
from riichiformer.data import split as sp  # noqa: E402

TENHOU = "2026010100gm-00a9-0000-{suffix}"


def item(year: int | None, name: str, source="tenhou_recent"):
    return {"root": "datasets/extracted", "relpath": f"extracted/{year}/{name}",
            "year": year, "source_id": source, "kind": "leaf", "invalid_reason": None}


def tenhou(year: int, month=1, suffix="0aa8cefa"):
    day = f"{year:04d}{month:02d}01"
    return f"{day}00gm-00a9-0000-{suffix}"


class AssignmentTests(unittest.TestCase):
    def test_protocol_L_boundaries(self):
        self.assertEqual(sp.assign_split("L", 2009, 1)[0], "train")
        self.assertEqual(sp.assign_split("L", 2022, 12)[0], "train")
        self.assertEqual(sp.assign_split("L", 2023, 6)[0], "val_holdout")
        self.assertEqual(sp.assign_split("L", 2024, 1)[0], "external_diagnostic")
        self.assertEqual(sp.assign_split("L", 2026, 1)[0], "external_diagnostic")
        self.assertEqual(sp.assign_split("L", 2008, 1)[0], "undated")

    def test_protocol_R_months(self):
        self.assertEqual(sp.assign_split("R", 2024, 12)[0], "train")
        self.assertEqual(sp.assign_split("R", 2025, 6)[0], "val")
        self.assertEqual(sp.assign_split("R", 2025, 7)[0], "select")
        self.assertEqual(sp.assign_split("R", 2025, None)[0], "undated")
        self.assertEqual(sp.assign_split("R", 2026, 1)[0], "confirm_blocked_previously_used")

    def test_hash_fallback_is_deterministic_and_marked(self):
        a = sp.assign_split("R", 2025, 1, mode="hash_fallback", game_id="tenhou:x")
        b = sp.assign_split("R", 2025, 1, mode="hash_fallback", game_id="tenhou:x")
        self.assertEqual(a, b)
        self.assertEqual(a[1], "hash_fallback_not_temporal")

    def test_month_and_game_id_parsing(self):
        self.assertEqual(sp.month_from_relpath("extracted/2025/" + tenhou(2025, 7)), 7)
        self.assertIsNone(sp.month_from_relpath("extracted/2025/not-a-date.mjson"))
        gid, verified = sp.game_id_from_relpath("extracted/2025/" + tenhou(2025, 7) + ".mjson")
        self.assertTrue(verified)
        self.assertTrue(gid.startswith("tenhou:"))
        gid2, verified2 = sp.game_id_from_relpath("extracted/2025/mystery.mjson")
        self.assertFalse(verified2)
        self.assertTrue(gid2.startswith("sha256:"))


class RegistryTests(unittest.TestCase):
    def test_rows_have_rights_and_previous_use_flags(self):
        rows = sp.build_split_rows([item(2024, tenhou(2024, 3))], sp.SplitConfig("R"))
        row = rows[0]
        self.assertEqual(row["split"], "train")
        self.assertEqual(row["rights_status"], "unknown")
        self.assertTrue(row["blocked_source"])
        self.assertFalse(row["train_allowed"])
        self.assertTrue(row["previously_used_in_research"])
        old = sp.build_split_rows([item(2010, tenhou(2010, 3))], sp.SplitConfig("R"))[0]
        self.assertFalse(old["previously_used_in_research"])

    def test_unknown_id_fails_closed(self):
        rows = sp.build_split_rows([item(2020, "mystery.mjson")], sp.SplitConfig("L"))
        self.assertEqual(rows[0]["split"], "unknown_id")
        self.assertFalse(rows[0]["train_allowed"])

    def test_duplicate_game_id_is_counted(self):
        name = tenhou(2020, 1)
        rows = sp.build_split_rows([item(2020, name), item(2020, name)], sp.SplitConfig("L"))
        self.assertEqual({row["split"] for row in rows}, {"train"})
        self.assertTrue(all(row["duplicate_group_size"] == 2 for row in rows))

    def test_no_overlap_detection(self):
        rows = sp.build_split_rows([
            item(2022, tenhou(2022, 1)),
            item(2023, tenhou(2023, 1)),
        ], sp.SplitConfig("L"))
        sp.assert_no_overlap(rows)
        broken = [dict(rows[0]), dict(rows[0], split="val_holdout")]
        with self.assertRaises(sp.SplitError):
            sp.assert_no_overlap(broken)

    def test_registry_hash_is_stable_and_config_sensitive(self):
        rows = sp.build_split_rows([item(2023, tenhou(2023, 1))], sp.SplitConfig("L"))
        h1 = sp.registry_hash(rows, sp.SplitConfig("L"))
        h2 = sp.registry_hash(rows, sp.SplitConfig("L"))
        self.assertEqual(h1, h2)
        h3 = sp.registry_hash(rows, sp.SplitConfig("L", mode="hash_fallback"))
        self.assertNotEqual(h1, h3)
        h4 = sp.registry_hash(rows, sp.SplitConfig("L"), rights_snapshot={"x": "allowed"})
        self.assertNotEqual(h1, h4)

    def test_manifest_roundtrip(self):
        rows = sp.build_split_rows([item(2025, tenhou(2025, 7))], sp.SplitConfig("R"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "L.parquet"
            self.assertEqual(sp.write_manifest(rows, path), 1)
            back = sp.read_manifest(path)
            self.assertEqual(back[0]["game_id"], rows[0]["game_id"])
            report = sp.assert_manifest_no_overlap(path)
            self.assertEqual(report["rows"], 1)


class CliTests(unittest.TestCase):
    def _inventory(self, path: Path, rows):
        pq.write_table(pa.Table.from_pylist(rows), path)

    def test_build_and_assert(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"root": "datasets", "relpath": f"extracted/2023/{tenhou(2023, 1)}.mjson",
                 "year": 2023, "source_id": "tenhou_recent", "kind": "leaf", "invalid_reason": None},
                {"root": "datasets", "relpath": f"extracted/2025/{tenhou(2025, 7)}.mjson",
                 "year": 2025, "source_id": "tenhou_recent", "kind": "leaf", "invalid_reason": None},
            ]
            inventory = root / "audit.parquet"
            self._inventory(inventory, rows)
            out = root / "split"
            doc = root / "SPLIT_REGISTRY.md"
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = rf_build_split.main([
                    "build", "--inventory", str(inventory), "--out", str(out),
                    "--doc", str(doc), "--repo-root", str(REPO_ROOT),
                ])
            self.assertEqual(code, 0, stderr.getvalue())
            summary = json.loads((out / "split_summary.json").read_text())
            counts = {p["protocol"]: p["counts"] for p in summary["protocols"]}
            self.assertEqual(counts["L"], {"external_diagnostic": 1, "val_holdout": 1})
            self.assertEqual(counts["R"], {"select": 1, "train": 1})
            self.assertTrue(doc.is_file())
            self.assertTrue(summary["training_blocked"])

            out_buf = io.StringIO()
            with redirect_stdout(out_buf):
                code = rf_build_split.main([
                    "assert-no-overlap",
                    "--manifest", str(out / "split_manifest_L.parquet"),
                    "--manifest", str(out / "split_manifest_R.parquet"),
                ])
            self.assertEqual(code, 0, out_buf.getvalue())
            self.assertIn("OK", out_buf.getvalue())

    def test_assert_detects_fabricated_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.parquet"
            base = sp.build_split_rows(
                [item(2020, tenhou(2020, 1))], sp.SplitConfig("L")
            )[0]
            sp.write_manifest([dict(base), dict(base, split="val_holdout")], path)
            code = rf_build_split.main(["assert-no-overlap", "--manifest", str(path)])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
