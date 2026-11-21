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


# --------------------------------------------------------------------------------------
# H-02: duplicate-component split registry v2
# --------------------------------------------------------------------------------------

import gzip  # noqa: E402

from riichiformer.data import canonicalize as canon  # noqa: E402
from riichiformer.data import dedup  # noqa: E402


def _events(payload="1m"):
    return [
        {"type": "start_game", "names": ["a", "b", "c", "d"], "seed": 7},
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0,
         "scores": [25000] * 4, "dora_marker": "1m",
         "tehais": [["1m"] * 13 for _ in range(4)]},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": payload, "tsumogiri": False},
        {"type": "end_kyoku", "deltas": [0, 0, 0, 0]},
        {"type": "end_game"},
    ]


def _record(relpath, payload, year, source):
    raw = gzip.compress(
        ("\n".join(json.dumps(event) for event in _events(payload)) + "\n").encode(),
        mtime=0,
    )
    record = canon.canonicalize_bytes(raw, relpath, source_id=source, year=year)
    record["root"] = "fixture"
    return record


V2_GAMES = [
    ("2022/2022010100gm-00a9-0000-aaaa0001.mjson", "1m", 2022, "tenhou_old"),
    ("2023/2023010101gm-00a9-0000-bbbb0002.mjson", "1m", 2023, "tenhou_recent"),   # dup
    ("2022/2022010102gm-00a9-0000-cccc0003.mjson", "2m", 2022, "tenhou_old"),
    ("2022/sub/2022010102gm-00a9-0000-cccc0003.mjson", "3m", 2022, "tenhou_old"),   # conflict
    ("2024/2024010101gm-00a9-0000-dddd0004.mjson", "4m", 2024, "tenhou_recent"),
    ("2025/2025020101gm-00a9-0000-eeee0005.mjson", "5m", 2025, "tenhou_recent"),
    ("2025/2025080101gm-00a9-0000-ffff0006.mjson", "6m", 2025, "tenhou_recent"),
    ("2026/2026010101gm-00a9-0000-99990007.mjson", "7m", 2026, "tenhou_recent"),
]


def _make_store(tmp: Path):
    records = [_record(*game) for game in V2_GAMES]
    # unique subdirectory per call so determinism tests can rebuild from scratch
    store_dir = Path(tempfile.mkdtemp(dir=tmp))
    store = dedup.CanonicalStore(store_dir / "canonical.sqlite", create=True)
    store.set_meta({"config": {"audit_parquet_sha256": "fixture-audit"}})
    # One shard per calendar year; content is irrelevant to shard semantics.
    by_year = {}
    for record in records:
        by_year.setdefault(record["year"], []).append(record)
    for index, (year, year_records) in enumerate(sorted(by_year.items())):
        store.add_shard(f"{year}:{index:05d}", year, year_records, 0.0)
    store.finalize_components()
    return store


CANONICAL_META = {
    "manifest": "fixture/manifest.json",
    "manifest_sha256": "f" * 64,
    "audit_parquet_sha256": "fixture-audit",
    "index_sha256": "e" * 64,
    "index_rows": len(V2_GAMES),
    "canonical_version": canon.CANONICAL_VERSION,
    "index_version": dedup.INDEX_VERSION,
    "full_corpus": False,
    "representatives": 5,
    "quarantine_rows": 2,
}


class SplitV2Tests(unittest.TestCase):
    def _build(self, tmp: Path, old_manifests=None, protocols=("L", "R")):
        store = _make_store(tmp)
        try:
            return sp.build_registry_v2(
                store, tmp / "split", protocols=protocols,
                canonical_meta=CANONICAL_META, old_manifests=old_manifests or {},
            ), store
        finally:
            store.close()

    def test_cross_year_duplicate_stays_in_one_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            registry, store = self._build(tmp)
            store.close()
            rows = sp.read_manifest(tmp / "split" / "split_manifest_L_v2.parquet")
            dup_rows = [row for row in rows if row["duplicate_group_size"] == 2
                        and row["component_size"] == 2
                        and row["canonical_sha256"] and not row["quarantine"]]
            # the cross-key duplicate is the component with a 2022 and a 2023 member
            years = {row["year"] for row in dup_rows}
            self.assertEqual(years, {2022, 2023})
            self.assertEqual({row["split"] for row in dup_rows}, {"train"})
            self.assertEqual(sum(1 for row in dup_rows if row["is_representative"]), 1)
            self.assertFalse(any(row["quarantine"] for row in dup_rows))
            report = sp.verify_v2_manifest(tmp / "split" / "split_manifest_L_v2.parquet")
            self.assertTrue(report["ok"])
            self.assertGreaterEqual(report["components_checked"], 2)

    def test_conflict_component_is_quarantine_and_never_representative(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            registry, store = self._build(tmp)
            store.close()
            rows = sp.read_manifest(tmp / "split" / "split_manifest_L_v2.parquet")
            conflict = [row for row in rows
                        if row["quarantine"]
                        and "conflict_same_id_different_canonical_sha"
                        in (row["reason"] or "")]
            self.assertEqual(len(conflict), 2)
            self.assertEqual({row["split"] for row in conflict}, {sp.QUARANTINE_SPLIT})
            self.assertFalse(any(row["is_representative"] for row in conflict))
            self.assertFalse(any(row["train_allowed"] for row in conflict))
            summary = registry["protocols"][0]
            self.assertEqual(summary["quarantined_rows"], 2)
            self.assertEqual(summary["counts"].get(sp.QUARANTINE_SPLIT), 2)
            self.assertEqual(summary["representatives"], 5)
            sp.verify_v2_manifest(tmp / "split" / "split_manifest_L_v2.parquet")

    def test_quarantine_cannot_leak_into_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._build(tmp, protocols=("L",))
            path = tmp / "split" / "split_manifest_L_v2.parquet"
            rows = sp.read_manifest(path)
            leaked = False
            for row in rows:
                if row["quarantine"] and not leaked:
                    row["split"] = "train"
                    row["train_allowed"] = True
                    leaked = True
            self.assertTrue(leaked)
            bad = tmp / "leaked.parquet"
            sp.write_v2_manifest_stream(rows, bad)
            with self.assertRaises(sp.SplitError):
                sp.verify_v2_manifest(bad)

    def test_component_straddle_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._build(tmp, protocols=("L",))
            rows = sp.read_manifest(tmp / "split" / "split_manifest_L_v2.parquet")
            component = next(row["component_id"] for row in rows
                             if row["duplicate_group_size"] > 1 and not row["quarantine"])
            copy = [dict(row) for row in rows]
            for row in copy:
                if row["component_id"] == component:
                    row["split"] = "val_holdout"
                    break
            with self.assertRaises(sp.SplitError):
                sp.assert_no_component_overlap(copy)

    def test_old_vs_new_diff_reports_moved_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inventory = [
                {"root": "fixture", "relpath": relpath, "year": year,
                 "source_id": source, "kind": "leaf", "invalid_reason": None}
                for relpath, _, year, source in V2_GAMES
            ]
            old_rows = sp.build_split_rows(inventory, sp.SplitConfig("L"))
            old_path = tmp / "old_L.parquet"
            sp.write_manifest(old_rows, old_path)
            registry, store = self._build(tmp, old_manifests={"L": old_path},
                                          protocols=("L",))
            store.close()
            diff = registry["diff_vs_pr05"]["L"]
            self.assertEqual(diff["status"], "compared")
            self.assertEqual(diff["old_rows"], len(V2_GAMES))
            self.assertEqual(diff["moved_rows"], 3)
            self.assertEqual(diff["moved_by_pair"],
                             {"train->quarantine": 2, "val_holdout->train": 1})
            self.assertEqual(diff["quarantined_rows"], 2)
            self.assertIn("2023/2023010101gm-00a9-0000-bbbb0002.mjson",
                          {sample["relpath"] for sample in diff["moved_samples"]})
            self.assertTrue((tmp / "split" / "split_diff_v2.json").is_file())

    def test_registry_is_deterministic_and_does_not_touch_old_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._build(tmp, protocols=("L",))
            first = (tmp / "split" / "split_registry_v2.json").read_bytes()
            first_manifest = sp.dedup.sha256_file(
                tmp / "split" / "split_manifest_L_v2.parquet")
            self._build(tmp, protocols=("L",))
            second = (tmp / "split" / "split_registry_v2.json").read_bytes()
            second_manifest = sp.dedup.sha256_file(
                tmp / "split" / "split_manifest_L_v2.parquet")
            self.assertEqual(first, second)
            self.assertEqual(first_manifest, second_manifest)
            self.assertFalse((tmp / "split" / "split_manifest_L.parquet").exists())
            self.assertFalse((tmp / "split" / "split_manifest_R.parquet").exists())

    def test_protected_dir_from_main_checkout_is_refused(self):
        """Cross-worktree guard: an absolute --out at the real main checkout's PR
        artifacts is refused even when --repo-root is this worktree."""
        main = next(
            root for root in sp.repository_roots(REPO_ROOT)
            if (root / "artifacts" / "riichiformer" / "split").is_dir()
        )
        with self.assertRaises(sp.SplitError):
            sp.assert_safe_v2_out(main / "artifacts" / "riichiformer" / "split", REPO_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaises(sp.SplitError):
                sp.assert_safe_v2_out(
                    tmp / "other-worktree" / "artifacts" / "riichiformer" / "canonical",
                    REPO_ROOT,
                )
            # normal H-02 output remains allowed
            sp.assert_safe_v2_out(
                tmp / "artifacts" / "riichiformer" / "h02" / "split", REPO_ROOT
            )

    def test_verify_rejects_blocked_source_or_non_representative_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._build(tmp, protocols=("L",))
            rows = sp.read_manifest(tmp / "split" / "split_manifest_L_v2.parquet")

            blocked = [dict(row) for row in rows]
            target = next(row for row in blocked
                          if row["split"] == "train" and row["is_representative"])
            target.update(blocked_source=True, rights_status="unknown", train_allowed=True)
            path = tmp / "blocked.parquet"
            sp.write_v2_manifest_stream(blocked, path)
            with self.assertRaises(sp.SplitError):
                sp.verify_v2_manifest(path)

            duplicate = [dict(row) for row in rows]
            target = next(row for row in duplicate
                          if row["split"] == "train" and not row["is_representative"])
            target.update(blocked_source=False, rights_status="allowed", train_allowed=True)
            path = tmp / "non_representative.parquet"
            sp.write_v2_manifest_stream(duplicate, path)
            with self.assertRaises(sp.SplitError):
                sp.verify_v2_manifest(path)

            allowed = [dict(row) for row in rows]
            target = next(row for row in allowed
                          if row["split"] == "train" and row["is_representative"])
            target.update(blocked_source=False, rights_status="allowed", train_allowed=True)
            path = tmp / "allowed.parquet"
            sp.write_v2_manifest_stream(allowed, path)
            self.assertTrue(sp.verify_v2_manifest(path)["ok"])

    def test_cli_build_and_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = _make_store(tmp)
            db_path = store.path
            store.close()
            manifest = tmp / "canonical_manifest.json"
            manifest.write_text(json.dumps({
                "audit_parquet_sha256": "fixture-audit",
                "canonical_version": canon.CANONICAL_VERSION,
                "index_version": dedup.INDEX_VERSION,
                "full_corpus": False,
                "rows": len(V2_GAMES),
                "representatives": 5,
                "quarantine_rows": 2,
                "artifacts": {"canonical_log_index.parquet": {"sha256": "e" * 64}},
            }))
            out = tmp / "split"
            code = sp.main([
                "build-v2", "--canonical-db", str(db_path),
                "--canonical-manifest", str(manifest), "--out", str(out),
                "--no-diff", "--repo-root", str(REPO_ROOT),
            ])
            self.assertEqual(code, 0)
            self.assertTrue((out / "split_registry_v2.json").is_file())
            code = sp.main([
                "verify-v2", "--manifest", str(out / "split_manifest_L_v2.parquet"),
                str(out / "split_manifest_R_v2.parquet"),
            ])
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
