#!/usr/bin/env python3
"""Tests for PR-03 canonicalisation and duplicate grouping."""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from riichiformer.data import canonicalize as canon  # noqa: E402
from riichiformer.data import dedup  # noqa: E402
import rf_build_canonical  # noqa: E402

TENHOU_A = "2026/2026010100gm-00a9-0000-0aa8cefa.mjson"
TENHOU_B = "2026/2026010101gm-00a9-0000-1111aaaa.mjson"


def events(kyoku=1, final="end_game", tweak=None):
    out = [{"type": "start_game", "names": ["a", "b", "c", "d"], "seed": 7}]
    for _ in range(kyoku):
        out.append({"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
                    "kyotaku": 0, "scores": [25000] * 4, "dora_marker": "1m",
                    "tehais": [["1m"] * 13 for _ in range(4)]})
        out.append({"type": "tsumo", "actor": 0, "pai": "1m"})
        out.append({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False})
        out.append({"type": "end_kyoku", "deltas": [0, 0, 0, 0]})
    out.append({"type": final})
    if tweak:
        for index, event in enumerate(out):
            if event["type"] == tweak["type"]:
                out[index] = tweak
                break
        else:  # pragma: no cover - test helper misuse
            raise AssertionError(f"no {tweak['type']} event to tweak")
    return out


def ndjson(evs) -> bytes:
    return ("\n".join(json.dumps(e, ensure_ascii=False) for e in evs) + "\n").encode()


class CanonicalizeTests(unittest.TestCase):
    def test_gzip_and_plain_have_same_canonical_hash(self):
        payload = ndjson(events())
        rec_gz = canon.canonicalize_bytes(gzip.compress(payload, mtime=0), "2026/a.mjson")
        rec_plain = canon.canonicalize_bytes(payload, "2026/a.mjson")
        self.assertEqual(rec_gz["canonical_sha256"], rec_plain["canonical_sha256"])
        self.assertNotEqual(rec_gz["original_sha256"], rec_plain["original_sha256"])
        self.assertEqual(rec_gz["events"], rec_plain["events"])
        self.assertEqual(rec_gz["kyoku_count"], 1)

    def test_changing_a_public_event_changes_canonical_hash(self):
        a = canon.canonicalize_bytes(ndjson(events()), "2026/a.mjson")
        b = canon.canonicalize_bytes(
            ndjson(events(tweak={"type": "dahai", "actor": 0, "pai": "9p", "tsumogiri": False})),
            "2026/a.mjson",
        )
        self.assertNotEqual(a["canonical_sha256"], b["canonical_sha256"])

    def test_platform_key_and_game_id(self):
        rec = canon.canonicalize_bytes(ndjson(events()), TENHOU_A)
        self.assertEqual(rec["platform_game_key"], "2026010100gm-00a9-0000-0aa8cefa")
        self.assertEqual(rec["game_id"], "tenhou:2026010100gm-00a9-0000-0aa8cefa")
        plain = canon.canonicalize_bytes(ndjson(events()), "2026/not-a-tenhou-id.mjson")
        self.assertIsNone(plain["platform_game_key"])
        self.assertTrue(plain["game_id"].startswith("sha256:"))

    def test_missing_end_game_is_rejected(self):
        evs = events(final="end_kyoku")
        with self.assertRaises(canon.CanonicalizeError) as ctx:
            canon.canonicalize_bytes(ndjson(evs), "2026/a.mjson")
        self.assertEqual(ctx.exception.reason, "missing_end_game")

    def test_unpaired_kyoku_is_rejected(self):
        evs = [
            {"type": "start_game", "names": ["a", "b", "c", "d"]},
            {"type": "end_kyoku", "deltas": [0, 0, 0, 0]},
            {"type": "end_game"},
        ]
        with self.assertRaises(canon.CanonicalizeError) as ctx:
            canon.canonicalize_bytes(ndjson(evs), "2026/a.mjson")
        self.assertEqual(ctx.exception.reason, "unpaired_end_kyoku")

    def test_bad_json_is_rejected_with_line(self):
        raw = ndjson(events()) + b"{not json}\n"
        with self.assertRaises(canon.CanonicalizeError) as ctx:
            canon.canonicalize_bytes(raw, "2026/a.mjson")
        self.assertEqual(ctx.exception.reason, "bad_json")
        self.assertIn("line", ctx.exception.detail)

    def test_empty_and_unknown_compression(self):
        with self.assertRaises(canon.CanonicalizeError):
            canon.canonicalize_bytes(b"", "2026/a.mjson")
        with self.assertRaises(canon.CanonicalizeError):
            canon.canonicalize_bytes(b"\x00\x01", "2026/a.mjson")

    def test_invalid_record_preserves_reason(self):
        rec = canon.invalid_record("2026/a.mjson", b"x", canon.CanonicalizeError("bad_json", "line 1"))
        self.assertEqual(rec["invalid_reason"], "bad_json: line 1")
        self.assertIsNone(rec["canonical_sha256"])
        self.assertEqual(rec["original_sha256"], canon._sha256_bytes(b"x"))

    def test_canonicalize_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "2026" / "a.mjson.gz"
            path.parent.mkdir(parents=True)
            path.write_bytes(gzip.compress(ndjson(events()), mtime=0))
            rec = canon.canonicalize_file(path, relpath="2026/a.mjson.gz", year=2026)
            self.assertEqual(rec["compression"], "gzip")
            self.assertEqual(rec["year"], 2026)
            self.assertEqual(rec["kyoku_count"], 1)


class DedupTests(unittest.TestCase):
    def _rec(self, relpath, evs=None):
        return canon.canonicalize_bytes(ndjson(evs or events()), relpath,
                                        source_id="tenhou_recent", year=2026)

    def test_same_content_different_paths_is_one_group(self):
        rows = dedup.index_rows([
            self._rec("2026/a.mjson"),
            self._rec("2026/copy/b.mjson"),
        ])
        groups = {row["duplicate_group"] for row in rows}
        self.assertEqual(len(groups), 1)
        self.assertEqual(rows[0]["duplicate_count"], 2)
        self.assertEqual(sum(1 for row in rows if row["is_representative"]), 1)
        self.assertFalse(any(row["collision_review"] for row in rows))

    def test_platform_key_priority_over_content(self):
        rows = dedup.index_rows([
            self._rec(TENHOU_A),
            self._rec(TENHOU_A.replace("2026010100", "2026010101")),
        ])
        # Same content but different Tenhou ids: two groups + cross-key review flag.
        self.assertEqual(len({row["duplicate_group"] for row in rows}), 2)
        self.assertTrue(all(row["cross_key_duplicate"] for row in rows))

    def test_same_platform_key_different_content_is_collision(self):
        other = events(tweak={"type": "dahai", "actor": 0, "pai": "9p", "tsumogiri": False})
        rows = dedup.index_rows([self._rec(TENHOU_A), self._rec(TENHOU_A, other)])
        self.assertEqual(len({row["duplicate_group"] for row in rows}), 1)
        self.assertTrue(all(row["collision_review"] for row in rows))

    def test_invalid_rows_are_not_grouped(self):
        bad = canon.invalid_record("2026/x.mjson", b"", canon.CanonicalizeError("empty_log"))
        rows = dedup.index_rows([self._rec("2026/a.mjson"), bad])
        invalid = next(row for row in rows if row["invalid_reason"])
        self.assertIsNone(invalid["duplicate_group"])
        self.assertFalse(invalid["is_representative"])

    def test_index_parquet_roundtrip_and_schema(self):
        rows = dedup.index_rows([self._rec("2026/a.mjson")])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.parquet"
            n = dedup.write_index_parquet(rows, path)
            self.assertEqual(n, 1)
            back = dedup.read_index_parquet(path)
            self.assertEqual(back[0]["game_id"], rows[0]["game_id"])
            self.assertEqual(list(back[0].keys()), dedup.INDEX_COLUMNS)

    def test_stratified_sample_is_deterministic(self):
        rows = []
        for year in (2009, 2023, 2026):
            for i in range(20):
                rows.append({
                    "relpath": f"{year}/{i:03d}.mjson", "year": year,
                    "source_id": f"s{year}", "game_id": f"g{year}-{i}",
                })
        first = dedup.stratified_sample(rows, max_total=9, seed=20261121)
        second = dedup.stratified_sample(rows, max_total=9, seed=20261121)
        self.assertEqual([row["relpath"] for row in first], [row["relpath"] for row in second])
        self.assertEqual(len(first), 9)
        self.assertEqual({row["year"] for row in first}, {2009, 2023, 2026})


class BuildCanonicalCliTests(unittest.TestCase):
    def test_cli_sample_pipeline(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            paths = [
                ("2026", "2026010100gm-00a9-0000-0aa8cefa.mjson"),
                ("2026", "2026010101gm-00a9-0000-1111aaaa.mjson"),
                ("2009", "2009010100gm-00a9-0000-2222bbbb.mjson"),
            ]
            rows = []
            for year, name in paths:
                path = root / year / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(gzip.compress(ndjson(events()), mtime=0))
                rows.append({"root": str(root), "relpath": f"{year}/{name}", "year": int(year),
                             "source_id": "tenhou_recent" if int(year) >= 2023 else "tenhou_old",
                             "kind": "leaf", "compression": "gzip", "invalid_reason": None})
            audit = Path(tmp) / "audit.parquet"
            pq.write_table(pa.Table.from_pylist(rows), audit)
            out = Path(tmp) / "canonical"
            code = rf_build_canonical.main([
                "--audit-parquet", str(audit), "--out", str(out), "--sample", "3",
                "--seed", "20261121", "--repo-root", str(REPO_ROOT),
            ])
            self.assertEqual(code, 0)
            summary = json.loads((out / "summary.json").read_text())
            self.assertFalse(summary["full_corpus"])
            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["valid"], 3)
            # Priority is platform key: three different Tenhou ids => three groups,
            # while the shared canonical content is flagged for cross-key review.
            self.assertEqual(summary["duplicate_groups"], 3)
            self.assertEqual(summary["cross_key_duplicate_rows"], 3)
            self.assertEqual(summary["duplicate_rows"], 0)
            back = dedup.read_index_parquet(out / "canonical_log_index.parquet")
            self.assertEqual(len(back), 3)
            self.assertTrue(all(row["is_representative"] for row in back))  # each its own group


if __name__ == "__main__":
    unittest.main(verbosity=2)
