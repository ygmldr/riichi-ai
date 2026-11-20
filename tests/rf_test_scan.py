#!/usr/bin/env python3
"""Tests for PR-02 read-only audit (`riichiformer/data/scan.py`, `scripts/rf_data_audit.py`).

Fixtures are tiny synthetic MJAI logs under a temp dir.  No real dataset is touched.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rf_data_audit  # noqa: E402
from riichiformer.data import scan  # noqa: E402
from riichiformer.schemas import provenance  # noqa: E402


def mjai_events(kyoku: int = 1) -> list[dict]:
    events = [{"type": "start_game", "names": ["a", "b", "c", "d"], "seed": 1}]
    for _ in range(kyoku):
        events.append({"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
                       "kyotaku": 0, "scores": [25000] * 4, "dora_marker": "1m",
                       "tehais": [["1m"] * 13 for _ in range(4)]})
        events.append({"type": "tsumo", "actor": 0, "pai": "1m"})
        events.append({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False})
        events.append({"type": "end_kyoku", "deltas": [0, 0, 0, 0]})
    events.append({"type": "end_game"})
    return events


def ndjson(events) -> bytes:
    return ("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n").encode()


def write_gzip(path: Path, events=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(ndjson(events or mjai_events()), mtime=0))


def write_plain(path: Path, events=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ndjson(events or mjai_events()))


def write_zip(path: Path, members: dict[str, bytes]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def corrupt_zip_crc(path: Path, marker: bytes = b"UNIQUE_PAYLOAD"):
    raw = bytearray(path.read_bytes())
    pos = raw.find(marker)
    assert pos >= 0, "marker not found in zip payload"
    raw[pos] = ord("X")
    path.write_bytes(bytes(raw))


class DetectTests(unittest.TestCase):
    def test_sniff_magic(self):
        self.assertEqual(scan.sniff_magic(b"\x1f\x8b\x08\x00rest"), "gzip")
        self.assertEqual(scan.sniff_magic(b"PK\x03\x04rest"), "zip")
        self.assertEqual(scan.sniff_magic(b"  {\"type\":1}"), "json")
        self.assertEqual(scan.sniff_magic(b""), "empty")
        self.assertEqual(scan.sniff_magic(b"\x00\x01\x02"), "unknown")

    def test_detection_does_not_trust_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gz_named_plain = root / "a.mjson"          # gzip bytes, plain-looking name
            plain_named_gz = root / "b.mjson.gz"      # plain bytes, gzip-looking name
            write_gzip(gz_named_plain)
            write_plain(plain_named_gz)
            self.assertEqual(scan.detect_compression(gz_named_plain), "gzip")
            self.assertEqual(scan.detect_compression(plain_named_gz), "json")

    def test_year_and_source(self):
        self.assertEqual(scan.year_from_relpath("extracted/2023/2023-01-01/x.mjson"), 2023)
        self.assertIsNone(scan.year_from_relpath("extracted/unknown/x.mjson"))
        # Tenhou hex suffixes must not be mistaken for a year (regression).
        self.assertEqual(
            scan.year_from_relpath("2026/2026010213gm-00a9-0000-e2011cd6.mjson"), 2026
        )
        self.assertIsNone(scan.year_from_relpath("data/log-e2011cd6.mjson"))
        self.assertEqual(provenance.source_for_year(2009).value, "tenhou_old")
        self.assertEqual(provenance.source_for_year(2022).value, "tenhou_old")
        self.assertEqual(provenance.source_for_year(2023).value, "tenhou_recent")
        self.assertEqual(provenance.source_for_year(2026).value, "tenhou_recent")
        self.assertEqual(provenance.source_for_year(2030).value, "unknown")

    def test_parse_mjai_sample_counts_kyoku(self):
        parsed = scan.parse_mjai_sample(ndjson(mjai_events(kyoku=2)))
        self.assertTrue(parsed["ok"], parsed)
        self.assertEqual(parsed["kyoku"], 2)
        self.assertEqual(parsed["events"], 1 + 2 * 4 + 1)

    def test_parse_mjai_sample_rejects_missing_end(self):
        events = mjai_events(1)
        events = [e for e in events if e["type"] != "end_game"]
        parsed = scan.parse_mjai_sample(ndjson(events))
        self.assertFalse(parsed["ok"])
        self.assertFalse(parsed["has_end_game"])

    def test_rights_unknown_blocks_training(self):
        self.assertFalse(provenance.rights_allow_new_training("unknown"))
        self.assertFalse(provenance.rights_allow_new_training("restricted"))
        self.assertFalse(provenance.rights_allow_new_training("reviewed"))
        self.assertTrue(provenance.rights_allow_new_training("allowed"))
        rights = provenance.default_source_rights()
        self.assertTrue(all(not r.allows_new_training for r in rights))
        self.assertIn("source_rights_unknown:tenhou_old",
                      provenance.rights_block_reasons(rights))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rf_scan_test_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))

    def test_mixed_root_compression_and_bad_files(self):
        root = self.tmp / "data"
        write_gzip(root / "2009" / "good.mjson")
        write_gzip(root / "2009" / "corrupt.mjson.gz")  # overwritten below
        (root / "2009" / "corrupt.mjson.gz").write_bytes(b"\x1f\x8b\x08\x00broken")
        write_plain(root / "2023" / "plain.mjson")
        write_plain(root / "2026" / "mislabeled.mjson.gz")  # plain bytes, gz name
        (root / "2026" / "notes.txt").write_text("not a log")

        out = self.tmp / "out"
        summary = scan.run_audit([root], out, sample_per_year=10, hash_mode="sample",
                                 repo=REPO_ROOT)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["leaf_files"], 4)          # .txt skipped
        self.assertEqual(summary["zip_files"], 0)
        self.assertTrue(summary["training_blocked"])
        self.assertIn("source_rights_unknown:tenhou_old", summary["block_reasons"])
        years = {(r["year"], r["source_id"]): r for r in summary["yearly_rows"]}
        self.assertEqual(years[(2009, "tenhou_old")]["files"], 2)
        self.assertEqual(years[(2009, "tenhou_old")]["gzip"], 2)
        self.assertEqual(years[(2023, "tenhou_recent")]["plaintext_json"], 1)
        self.assertEqual(years[(2026, "tenhou_recent")]["plaintext_json"], 1)

        bad = [json.loads(line) for line in
               (out / "bad_files.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.assertTrue(any("corrupt.mjson.gz" in item["path"] for item in bad))
        samples = [json.loads(line) for line in
                   (out / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
        good = [s for s in samples if s["ok"]]
        self.assertTrue(good)
        self.assertTrue(all(s["kyoku"] >= 1 for s in good))

        import pyarrow.parquet as pq

        table = pq.read_table(out / "source_audit.parquet")
        self.assertEqual(table.num_rows, 4)
        self.assertNotIn("player", " ".join(table.column_names).lower())
        rows = table.to_pylist()
        mislabeled = [r for r in rows if r["relpath"].endswith("mislabeled.mjson.gz")][0]
        self.assertEqual(mislabeled["compression"], "json")

        yearly = list(csv.DictReader((out / "yearly_counts.csv").open(encoding="utf-8")))
        self.assertEqual(len(yearly), 3)
        rights = list(csv.DictReader((out / "source_rights.csv").open(encoding="utf-8")))
        self.assertTrue(rights)
        tenhou = [row for row in rights if row["source_id"].startswith("tenhou_")]
        self.assertTrue(tenhou)
        self.assertTrue(all(row["rights_status"] == "unknown" for row in tenhou))
        self.assertTrue(all(row["allows_new_training"] == "False" for row in rights))

    def test_empty_root_is_ok_but_blocks_training(self):
        root = self.tmp / "empty"
        root.mkdir()
        summary = scan.run_audit([root], self.tmp / "out", sample_per_year=10)
        self.assertEqual(summary["status"], "EMPTY")
        self.assertTrue(summary["training_blocked"])
        self.assertIn("no_game_logs_found", summary["block_reasons"])
        import pyarrow.parquet as pq

        self.assertEqual(pq.read_table(self.tmp / "out" / "source_audit.parquet").num_rows, 0)

    def test_valid_zip_is_counted_and_sampled(self):
        root = self.tmp / "data"
        root.mkdir()
        write_zip(root / "2010.zip", {
            "2010/a.mjson": ndjson(mjai_events(1)),
            "2010/b.mjson": ndjson(mjai_events(2)),
        })
        summary = scan.run_audit([root], self.tmp / "out", sample_per_year=5)
        self.assertEqual(summary["zip_files"], 1)
        self.assertEqual(summary["zip_members"], 2)
        self.assertEqual(summary["status"], "PASS")
        row = next(r for r in summary["yearly_rows"] if r["year"] == 2010)
        self.assertEqual(row["zip_containers"], 1)
        self.assertEqual(row["zip_members"], 2)
        self.assertGreaterEqual(row["sampled"], 1)
        self.assertTrue(any(s.get("container") == "zip" for s in
                            [json.loads(x) for x in (self.tmp / "out" / "samples.jsonl")
                             .read_text().splitlines() if x]))

    def test_duplicate_names_in_different_dirs_are_both_counted(self):
        root = self.tmp / "data"
        write_gzip(root / "2009" / "a" / "same.mjson")
        write_plain(root / "2009" / "b" / "same.mjson")
        summary = scan.run_audit([root], self.tmp / "out", sample_per_year=5,
                                 hash_mode="all")
        self.assertEqual(summary["leaf_files"], 2)
        row = next(r for r in summary["yearly_rows"] if r["year"] == 2009)
        self.assertEqual(row["files"], 2)
        self.assertEqual(row["gzip"], 1)
        self.assertEqual(row["plaintext_json"], 1)
        import pyarrow.parquet as pq

        rows = pq.read_table(self.tmp / "out" / "source_audit.parquet").to_pylist()
        self.assertEqual(len({r["relpath"] for r in rows}), 2)
        self.assertTrue(all(r["sha256"] for r in rows))  # hash_mode=all

    def test_multiple_roots_dir_and_zip(self):
        root = self.tmp / "data"
        write_gzip(root / "2012" / "a.mjson")
        zpath = self.tmp / "2015.zip"
        write_zip(zpath, {"2015/b.mjson": ndjson(mjai_events(1))})
        summary = scan.run_audit([root, zpath], self.tmp / "out", sample_per_year=5)
        self.assertEqual(summary["leaf_files"], 1)
        self.assertEqual(summary["zip_files"], 1)
        self.assertEqual(summary["zip_members"], 1)
        self.assertEqual({r["year"] for r in summary["yearly_rows"]}, {2012, 2015})

    def test_corrupt_zip_crc_is_reported(self):
        root = self.tmp / "data"
        root.mkdir()
        zpath = root / "2011.zip"
        write_zip(zpath, {"2011/a.mjson": b"UNIQUE_PAYLOAD" + ndjson(mjai_events(1))})
        corrupt_zip_crc(zpath)
        summary = scan.run_audit([root], self.tmp / "out", sample_per_year=5)
        self.assertEqual(summary["zip_files"], 1)
        bad = [json.loads(line) for line in
               (self.tmp / "out" / "bad_files.jsonl").read_text().splitlines() if line]
        self.assertTrue(any("zip_crc" in item["reason"] for item in bad), bad)
        row = json.loads((self.tmp / "out" / "audit_summary.json").read_text())
        self.assertTrue(row["training_blocked"])

    def test_cli_empty_exit_zero_and_all_missing_exit_two(self):
        root = self.tmp / "empty"
        root.mkdir()
        out = self.tmp / "out"
        report = self.tmp / "report.md"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = rf_data_audit.main([
                "--roots", str(root), "--out", str(out), "--report", str(report),
                "--sample-per-year", "5", "--repo-root", str(REPO_ROOT),
            ])
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertIn("status=EMPTY", stdout.getvalue())
        self.assertTrue(report.is_file())
        self.assertTrue((out / "source_audit.parquet").is_file())

        err = io.StringIO()
        with redirect_stderr(err):
            code = rf_data_audit.main([
                "--roots", str(self.tmp / "nope"), "--out", str(out),
                "--repo-root", str(REPO_ROOT),
            ])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
