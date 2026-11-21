#!/usr/bin/env python3
"""Tests for canonicalisation + H-02 duplicate components (streaming/resume)."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from riichiformer.data import canonicalize as canon  # noqa: E402
from riichiformer.data import dedup  # noqa: E402
import rf_build_canonical  # noqa: E402

TENHOU_A = "2026/2026010100gm-00a9-0000-0aa8cefa.mjson"
TENHOU_B = "2026/2026010101gm-00a9-0000-1111aaaa.mjson"


def events(kyoku=1, final="end_game", tweak=None, pai="1m"):
    out = [{"type": "start_game", "names": ["a", "b", "c", "d"], "seed": 7}]
    for _ in range(kyoku):
        out.append({"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0,
                    "kyotaku": 0, "scores": [25000] * 4, "dora_marker": "1m",
                    "tehais": [["1m"] * 13 for _ in range(4)]})
        out.append({"type": "tsumo", "actor": 0, "pai": "1m"})
        out.append({"type": "dahai", "actor": 0, "pai": pai, "tsumogiri": False})
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


def gz(evs) -> bytes:
    return gzip.compress(ndjson(evs), mtime=0)


class CanonicalizeTests(unittest.TestCase):
    def test_gzip_and_plain_have_same_canonical_hash(self):
        payload = ndjson(events())
        rec_gz = canon.canonicalize_bytes(gzip.compress(payload, mtime=0), "2026/a.mjson")
        rec_plain = canon.canonicalize_bytes(payload, "2026/a.mjson")
        self.assertEqual(rec_gz["canonical_sha256"], rec_plain["canonical_sha256"])
        self.assertNotEqual(rec_gz["original_sha256"], rec_plain["original_sha256"])
        self.assertEqual(rec_gz["events"], rec_plain["events"])
        self.assertEqual(rec_gz["kyoku_count"], 1)
        self.assertEqual(rec_gz["rules_version"], canon.RULES_VERSION)

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
            path.write_bytes(gz(events()))
            rec = canon.canonicalize_file(path, relpath="2026/a.mjson.gz", year=2026)
            self.assertEqual(rec["compression"], "gzip")
            self.assertEqual(rec["year"], 2026)
            self.assertEqual(rec["kyoku_count"], 1)


class DedupComponentTests(unittest.TestCase):
    """Negative/edge tests for the duplicate graph (BR-04)."""

    def _rec(self, relpath, evs=None, source="tenhou_recent", year=2026):
        return canon.canonicalize_bytes(ndjson(evs or events()), relpath,
                                        source_id=source, year=year)

    def test_same_game_different_filename_compression_source_is_one_component(self):
        """Same events, different filename + gzip/plain + source => one component."""
        payload = ndjson(events())
        rows = dedup.index_rows([
            canon.canonicalize_bytes(payload, "2026/a.mjson", source_id="tenhou_recent", year=2026),
            canon.canonicalize_bytes(gzip.compress(payload, mtime=0), "2026/copy/b.mjson.gz",
                                     source_id="tenhou_old", year=2026),
        ])
        self.assertEqual(len({row["component_id"] for row in rows}), 1)
        self.assertEqual(rows[0]["component_size"], 2)
        self.assertEqual(sum(1 for row in rows if row["is_representative"]), 1)
        self.assertTrue(all(row["cross_key_duplicate"] is False for row in rows))
        self.assertFalse(any(row["quarantine"] for row in rows))
        self.assertFalse(any(row["collision_review"] for row in rows))

    def test_same_content_different_tenhou_id_is_one_component(self):
        """BR-04: cross-key duplicate is merged, not merely flagged for review."""
        rows = dedup.index_rows([
            self._rec(TENHOU_A),
            self._rec(TENHOU_B),
        ])
        self.assertEqual(len({row["component_id"] for row in rows}), 1)
        self.assertEqual(rows[0]["component_size"], 2)
        self.assertTrue(all(row["cross_key_duplicate"] for row in rows))
        self.assertEqual(sum(1 for row in rows if row["is_representative"]), 1)
        self.assertFalse(any(row["quarantine"] for row in rows))

    def test_same_id_different_content_is_conflict_and_quarantined(self):
        other = events(tweak={"type": "dahai", "actor": 0, "pai": "9p", "tsumogiri": False})
        rows = dedup.index_rows([self._rec(TENHOU_A), self._rec(TENHOU_A, other)])
        self.assertEqual(len({row["component_id"] for row in rows}), 1)
        self.assertTrue(all(row["collision_review"] for row in rows))
        self.assertTrue(all(row["quarantine"] for row in rows))
        self.assertFalse(any(row["is_representative"] for row in rows))
        self.assertIn("conflict_same_id_different_canonical_sha",
                      rows[0]["quarantine_reason"])

    def test_artificial_hash_collision_links_component_without_silent_pick(self):
        """Two unrelated files with the same canonical sha are one component.

        A real SHA-256 collision is infeasible to construct, so the records are built
        directly: the graph code must merge on the SHA edge and must not silently pick a
        representative if the payloads also disagree by identity.
        """
        base = {
            "canonical_version": canon.CANONICAL_VERSION,
            "rules_version": canon.RULES_VERSION,
            "platform_game_key": None,
            "compression": "gzip",
            "bytes": 10,
            "events": 5,
            "kyoku_count": 1,
            "start_game": True,
            "end_game": True,
            "invalid_reason": None,
            "canonical_sha256": "0" * 64,
            "canonical_text_sha256": "0" * 64,
        }
        records = [
            dict(base, relpath="2022/x.mjson", year=2022, source_id="s",
                 original_sha256="a" * 64, game_id="tenhou:2022010100gm-00a9-0000-aaaa0001"),
            dict(base, relpath="2023/y.mjson", year=2023, source_id="s",
                 original_sha256="b" * 64, game_id="tenhou:2023010100gm-00a9-0000-bbbb0002"),
        ]
        rows = dedup.index_rows(records)
        self.assertEqual(len({row["component_id"] for row in rows}), 1)
        self.assertEqual(rows[0]["component_size"], 2)
        self.assertTrue(all(row["cross_key_duplicate"] for row in rows))
        self.assertEqual(sum(1 for row in rows if row["is_representative"]), 1)
        self.assertEqual(rows[0]["canonical_sha256"], "0" * 64)

    def test_same_id_same_content_has_exactly_one_representative(self):
        rows = dedup.index_rows([self._rec(TENHOU_A), self._rec(TENHOU_A)])
        self.assertEqual(len({row["component_id"] for row in rows}), 1)
        self.assertEqual(sum(1 for row in rows if row["is_representative"]), 1)
        self.assertFalse(any(row["quarantine"] for row in rows))

    def test_invalid_row_is_quarantined_but_not_grouped_with_content_duplicate(self):
        bad = canon.invalid_record("2026/x.mjson", b"", canon.CanonicalizeError("empty_log"))
        rows = dedup.index_rows([self._rec("2026/a.mjson"), bad])
        invalid = next(row for row in rows if row["invalid_reason"])
        valid = next(row for row in rows if not row["invalid_reason"])
        self.assertNotEqual(invalid["component_id"], valid["component_id"])
        self.assertTrue(invalid["quarantine"])
        self.assertFalse(invalid["is_representative"])
        self.assertEqual(invalid["component_size"], 1)

    def test_incomplete_component_is_quarantined(self):
        """A broken copy sharing the Tenhou id quarantines the whole component."""
        bad = canon.invalid_record(TENHOU_A, b"", canon.CanonicalizeError("empty_log"))
        rows = dedup.index_rows([self._rec(TENHOU_A), bad])
        self.assertEqual(len({row["component_id"] for row in rows}), 1)
        self.assertTrue(all(row["quarantine"] for row in rows))
        self.assertFalse(any(row["is_representative"] for row in rows))
        self.assertTrue(any("incomplete_component" in (row["quarantine_reason"] or "")
                            for row in rows))

    def test_index_parquet_roundtrip_and_schema(self):
        rows = dedup.index_rows([self._rec("2026/a.mjson")])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.parquet"
            n = dedup.write_index_parquet(rows, path)
            self.assertEqual(n, 1)
            back = dedup.read_index_parquet(path)
            self.assertEqual(back[0]["game_id"], rows[0]["game_id"])
            self.assertEqual(list(back[0].keys()), dedup.INDEX_COLUMNS)
            self.assertEqual(back[0]["rules_version"], canon.RULES_VERSION)

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

    def test_failed_shard_transaction_rolls_back(self):
        """A crash/constraint error mid-shard must leave no partial rows/checkpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            store = dedup.CanonicalStore(Path(tmp) / "db.sqlite", create=True)
            record = canon.canonicalize_bytes(ndjson(events()), TENHOU_A,
                                              source_id="tenhou_recent", year=2026)
            record["root"] = "fixture"
            with self.assertRaises(dedup.DedupError):
                store.add_shard("2026:00000", 2026, [record, dict(record)], 0.0)
            self.assertEqual(store.count_rows(), 0)
            self.assertEqual(store.shard_status(), {})
            store.add_shard("2026:00000", 2026, [record], 0.0)
            self.assertEqual(store.count_rows(), 1)
            first = store.finalize_components()
            second = store.finalize_components()
            self.assertEqual(first, second)
            self.assertEqual(store.get_meta("finalized"), "1")
            store.close()

    def test_streamed_sample_matches_in_memory_sample(self):
        rows = []
        for year in (2009, 2010, 2023, 2025):
            for i in range(30):
                rows.append({
                    "relpath": f"{year}/{i:03d}.mjson", "year": year,
                    "source_id": f"s{year}", "game_id": f"g{year}-{i}",
                })
        in_memory = dedup.stratified_sample(rows, max_total=17, seed=7)
        streamed = dedup.select_streamed(rows, 17, seed=7)
        self.assertEqual([row["relpath"] for row in in_memory],
                         [row["relpath"] for row in streamed])


# --------------------------------------------------------------------------------------
# Fixtures for the CLI / resume / store tests
# --------------------------------------------------------------------------------------


def write_fixture(tmp: Path, games) -> Path:
    """``games``: list of ``(relpath, payload, year, source_id)`` -> audit parquet."""
    data = tmp / "data"
    rows = []
    for relpath, payload, year, source in games:
        path = data / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = gz(events(pai=payload))
        path.write_bytes(raw)
        rows.append({
            "root": str(data), "relpath": relpath, "year": year, "source_id": source,
            "kind": "leaf", "compression": "gzip", "bytes": len(raw), "sha256": None,
            "invalid_reason": None,
        })
    audit = tmp / "audit.parquet"
    pq.write_table(pa.Table.from_pylist(rows), audit)
    return audit


BASE_GAMES = [
    ("2022/2022010100gm-00a9-0000-aaaa0001.mjson", "1m", 2022, "tenhou_old"),
    ("2022/2022010101gm-00a9-0000-aaaa0002.mjson", "2m", 2022, "tenhou_old"),
    ("2023/2023010101gm-00a9-0000-bbbb0003.mjson", "3m", 2023, "tenhou_recent"),
    ("2024/2024010101gm-00a9-0000-cccc0004.mjson", "4m", 2024, "tenhou_recent"),
    ("2025/2025010101gm-00a9-0000-dddd0005.mjson", "5m", 2025, "tenhou_recent"),
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _main_checkout_root() -> Path:
    for root in rf_build_canonical.repository_roots(REPO_ROOT):
        if (root / "artifacts" / "riichiformer" / "split").is_dir():
            return root
    raise AssertionError("main checkout artifacts/riichiformer/split not found")


class BuildCanonicalCliTests(unittest.TestCase):
    def test_cli_sample_pipeline_and_new_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            audit = write_fixture(tmp, BASE_GAMES)
            out = tmp / "canonical"
            code = rf_build_canonical.main([
                "--audit-parquet", str(audit), "--out", str(out), "--sample", "5",
                "--seed", "20261121", "--restart", "--repo-root", str(REPO_ROOT),
            ])
            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertFalse(manifest["full_corpus"])
            self.assertEqual(manifest["rows"], 5)
            self.assertEqual(manifest["valid"], 5)
            self.assertEqual(manifest["invalid"], 0)
            self.assertEqual(manifest["duplicate_components"], 0)
            self.assertEqual(manifest["representatives"], 5)
            self.assertEqual(manifest["quarantine_rows"], 0)
            self.assertEqual(manifest["rules_version"], canon.RULES_VERSION)
            back = dedup.read_index_parquet(out / "canonical_log_index.parquet")
            self.assertEqual(len(back), 5)
            self.assertTrue(all(row["is_representative"] for row in back))
            self.assertTrue((out / "duplicate_components.parquet").is_file())
            self.assertTrue((out / "quarantine.jsonl").is_file())
            self.assertEqual((out / "quarantine.jsonl").read_text(), "")
            self.assertTrue((out / "sample_5.jsonl").is_file())

    def test_same_game_under_two_names_is_deduplicated_by_cli(self):
        games = [
            ("2022/2022010100gm-00a9-0000-aaaa0001.mjson", "1m", 2022, "tenhou_old"),
            ("2022/copy/2022010100gm-00a9-0000-aaaa0001.mjson.gz", "1m", 2022, "tenhou_old"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "canonical"
            code = rf_build_canonical.main([
                "--audit-parquet", str(write_fixture(tmp, games)), "--out", str(out),
                "--sample", "2", "--restart", "--repo-root", str(REPO_ROOT),
            ])
            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["duplicate_components"], 1)
            self.assertEqual(manifest["duplicate_rows"], 2)
            self.assertEqual(manifest["representatives"], 1)
            rows = dedup.read_index_parquet(out / "canonical_log_index.parquet")
            self.assertEqual(len({row["canonical_sha256"] for row in rows}), 1)
            self.assertEqual(sum(1 for row in rows if row["is_representative"]), 1)

    def test_conflict_is_quarantined_by_cli(self):
        games = [
            ("2022/2022010100gm-00a9-0000-aaaa0001.mjson", "1m", 2022, "tenhou_old"),
            ("2022/other/2022010100gm-00a9-0000-aaaa0001.mjson", "2m", 2022, "tenhou_old"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "canonical"
            code = rf_build_canonical.main([
                "--audit-parquet", str(write_fixture(tmp, games)), "--out", str(out),
                "--sample", "2", "--restart", "--repo-root", str(REPO_ROOT),
            ])
            self.assertEqual(code, 0)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["conflict_components"], 1)
            self.assertEqual(manifest["representatives"], 0)
            self.assertEqual(manifest["quarantine_rows"], 2)
            lines = (out / "quarantine.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("conflict_same_id_different_canonical_sha", lines[0])

    def test_interrupted_resume_is_byte_identical_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            audit = write_fixture(tmp, BASE_GAMES)
            base = ["--audit-parquet", str(audit), "--sample", "5", "--chunk-rows", "1",
                    "--repo-root", str(REPO_ROOT)]
            out_a = tmp / "resumed"
            out_b = tmp / "clean"
            self.assertEqual(rf_build_canonical.main(
                base + ["--out", str(out_a), "--max-shards", "2", "--restart"]), 0)
            self.assertFalse((out_a / "manifest.json").exists())
            self.assertEqual(rf_build_canonical.main(
                base + ["--out", str(out_a)]), 0)
            self.assertEqual(rf_build_canonical.main(
                base + ["--out", str(out_b), "--restart"]), 0)

            artifacts = [
                "canonical_log_index.parquet", "duplicate_components.parquet",
                "quarantine.jsonl", "sample_5.jsonl", "manifest.json", "summary.json",
            ]
            for name in artifacts:
                self.assertEqual(_sha(out_a / name), _sha(out_b / name),
                                 f"{name} differs between resume and clean run")

            # Re-running a finished build is a no-op with identical bytes.
            before = {name: _sha(out_a / name) for name in artifacts}
            self.assertEqual(rf_build_canonical.main(base + ["--out", str(out_a)]), 0)
            after = {name: _sha(out_a / name) for name in artifacts}
            self.assertEqual(before, after)

    def test_changed_inventory_refuses_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            audit = write_fixture(tmp, BASE_GAMES)
            out = tmp / "canonical"
            base = ["--audit-parquet", str(audit), "--sample", "5", "--repo-root",
                    str(REPO_ROOT), "--out", str(out)]
            self.assertEqual(rf_build_canonical.main(base + ["--restart"]), 0)
            # Same path, different content => different audit sha => refuse silently wrong resume.
            write_fixture(tmp, BASE_GAMES[:4])
            self.assertEqual(rf_build_canonical.main(base), 2)

    def test_refuses_protected_dirs_from_other_checkouts(self):
        """An absolute --out at the main checkout's PR artifacts must be refused
        even when --repo-root is this worktree (and vice versa)."""
        main = _main_checkout_root()
        with self.assertRaises(rf_build_canonical.BuildError):
            rf_build_canonical.assert_safe_out(
                main / "artifacts" / "riichiformer" / "split", REPO_ROOT
            )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            other = tmp / "other-worktree" / "artifacts" / "riichiformer" / "canonical"
            with self.assertRaises(rf_build_canonical.BuildError):
                rf_build_canonical.assert_safe_out(other, REPO_ROOT)
            # explicit fallback for environments without git metadata
            fake = tmp / "fake-checkout"
            self.assertIn(fake.resolve(),
                          rf_build_canonical.repository_roots(REPO_ROOT, [fake]))
            with self.assertRaises(rf_build_canonical.BuildError):
                rf_build_canonical.assert_safe_out(
                    fake / "artifacts" / "riichiformer" / "split", REPO_ROOT, [fake]
                )
            # normal H-02 output remains allowed
            rf_build_canonical.assert_safe_out(
                tmp / "allowed" / "artifacts" / "riichiformer" / "h02" / "canonical",
                REPO_ROOT,
            )

    def test_refuses_protected_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = write_fixture(Path(tmp), BASE_GAMES)
            code = rf_build_canonical.main([
                "--audit-parquet", str(audit),
                "--out", str(REPO_ROOT / "artifacts" / "riichiformer" / "canonical"),
                "--sample", "1", "--restart", "--repo-root", str(REPO_ROOT),
            ])
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
