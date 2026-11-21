#!/usr/bin/env python3
"""Tests for the H-03 fail-closed source-rights gate.

The production registry is expected to contain **no** training-allowed source.
The only passing rows in these tests are synthetic fixtures that prove the gate
can open when -- and only when -- a human-reviewed, evidence-backed row exists.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "rf_check_source_rights.py"
PRODUCTION_CSV = REPO_ROOT / "configs" / "riichiformer" / "source_rights.csv"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rf_check_source_rights as gate  # noqa: E402
from riichiformer.schemas import provenance  # noqa: E402

EXPECTED_SOURCE_IDS = frozenset({
    "tenhou_raw_logs",
    "tenhou_old",
    "tenhou_recent",
    "tenhou_to_mjai_dataset",
    "tenhou_to_mjai_code",
    "mortal_code",
    "teacher_local",
    "akochan_code",
    "selfplay_own",
    "majsoul_optional",
    "third_party_mirror_hf",
})

FIELDS = [
    "source_id",
    "origin_url",
    "retrieved_at",
    "terms_url",
    "evidence_url",
    "evidence_quote",
    "storage_allowed",
    "train_allowed",
    "redistribute_allowed",
    "privacy_notes",
    "status",
    "reviewer",
    "notes",
]

BASE_ROW = {
    "source_id": "fixture_licensed_corpus",
    "origin_url": "https://example.invalid/fixture-corpus",
    "retrieved_at": "2025-01-01",
    "terms_url": "https://example.invalid/fixture-license",
    "evidence_url": "https://example.invalid/fixture-license",
    "evidence_quote": (
        "Explicit written permission for storage, training and redistribution; "
        "synthetic fixture only, never a real dataset."
    ),
    "storage_allowed": "ALLOWED",
    "train_allowed": "ALLOWED",
    "redistribute_allowed": "ALLOWED",
    "privacy_notes": "Synthetic fixture; contains no personal data.",
    "status": "ALLOWED",
    "reviewer": "fixture-human-reviewer",
    "notes": "Test fixture; not a real source and not a rights grant.",
}


def fixture_row(**overrides) -> dict[str, str]:
    row = dict(BASE_ROW)
    row.update(overrides)
    return row


def write_registry(path: Path, rows: list[dict[str, str]],
                   fields: list[str] | None = None) -> Path:
    fields = list(fields or FIELDS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def run_gate(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def read_production_rows() -> list[dict[str, str]]:
    with PRODUCTION_CSV.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class ProductionRegistryTests(unittest.TestCase):
    def test_production_registry_has_exact_known_source_ids(self):
        self.assertTrue(PRODUCTION_CSV.is_file())
        rows = read_production_rows()
        ids = {row["source_id"] for row in rows}
        self.assertEqual(len(rows), 11, f"expected 11 rows, got {len(rows)}")
        self.assertEqual(ids, EXPECTED_SOURCE_IDS)
        self.assertEqual(gate.KNOWN_SOURCE_IDS, EXPECTED_SOURCE_IDS,
                         "gate and tests disagree on the known source set")

    def test_no_production_source_can_train(self):
        rows = read_production_rows()
        offenders = [row["source_id"] for row in rows
                     if row["train_allowed"] == "ALLOWED"]
        self.assertEqual(offenders, [], "a source silently became trainable")
        status_offenders = [row["source_id"] for row in rows
                            if row["status"] == "ALLOWED"]
        self.assertEqual(status_offenders, [],
                         "production registry must not contain an ALLOWED row "
                         "until a human reviewer signs it")

    def test_all_production_purposes_are_blocked(self):
        rows = read_production_rows()
        for row in rows:
            for purpose in gate.PURPOSES:
                column = gate.PURPOSE_COLUMNS[purpose]
                self.assertNotEqual(
                    row[column], "ALLOWED",
                    f"{row['source_id']}/{purpose} is ALLOWED without sign-off",
                )

    def test_missing_known_default_is_an_error(self):
        rows = [row for row in read_production_rows()
                if row["source_id"] != "tenhou_old"]
        tmp = Path(tempfile.mkdtemp(prefix="rf_h03_missing_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, True))
        csv_path = tmp / "rights.csv"
        write_registry(csv_path, rows)
        proc = run_gate("--source", "tenhou_old", "--purpose", "train",
                        "--csv", str(csv_path))
        self.assertEqual(proc.returncode, gate.EXIT_INVALID, proc.stdout)
        self.assertIn("missing from the registry", proc.stderr)

    def test_partial_production_registry_rejected_for_any_source(self):
        rows = [row for row in read_production_rows()
                if row["source_id"] != "tenhou_old"]
        tmp = Path(tempfile.mkdtemp(prefix="rf_h03_partial_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, True))
        csv_path = tmp / "rights.csv"
        write_registry(csv_path, rows)
        # Request a *present* known source; the truncated registry must still be
        # rejected as a whole, not silently accepted for the remaining rows.
        proc = run_gate("--source", "tenhou_recent", "--purpose", "train",
                        "--csv", str(csv_path))
        self.assertEqual(proc.returncode, gate.EXIT_INVALID, proc.stdout)
        self.assertIn("partial production registry", proc.stderr)

    def test_provenance_defaults_stay_fail_closed(self):
        rights = provenance.default_source_rights()
        self.assertTrue(rights)
        for row in rights:
            self.assertFalse(row.allows_new_training,
                             f"{row.source_id} default unexpectedly allows training")
        self.assertIn("source_rights_unknown:tenhou_old",
                      provenance.rights_block_reasons(rights))

    def test_every_default_source_is_rejected_for_training(self):
        for row in provenance.default_source_rights():
            proc = run_gate("--source", row.source_id.value, "--purpose", "train")
            self.assertNotEqual(proc.returncode, 0,
                                f"{row.source_id} was not rejected: {proc.stdout}")
            # A known source is denied with 1; an unconfigured one with 2.
            # Both are fail-closed, but every existing default must be known.
            self.assertIn(proc.returncode, (gate.EXIT_DENIED, gate.EXIT_INVALID))


class GateDecisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rf_h03_gate_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))
        self.csv = self.tmp / "rights.csv"

    def test_restricted_source_rejected(self):
        proc = run_gate("--source", "tenhou_old", "--purpose", "train",
                        "--csv", str(PRODUCTION_CSV))
        self.assertEqual(proc.returncode, gate.EXIT_DENIED, proc.stderr)
        self.assertIn("DENY", proc.stdout)
        self.assertIn("train_allowed=RESTRICTED", proc.stdout)

    def test_restricted_redistribution_rejected(self):
        proc = run_gate("--source", "tenhou_raw_logs", "--purpose", "redistribute")
        self.assertEqual(proc.returncode, gate.EXIT_DENIED, proc.stderr)
        self.assertIn("redistribute_allowed=RESTRICTED", proc.stdout)

    def test_unknown_source_rejected(self):
        proc = run_gate("--source", "no_such_source", "--purpose", "train")
        self.assertEqual(proc.returncode, gate.EXIT_INVALID)
        self.assertIn("unknown source", proc.stderr)

    def test_default_csv_path_is_used(self):
        proc = run_gate("--source", "tenhou_old", "--purpose", "train")
        self.assertEqual(proc.returncode, gate.EXIT_DENIED, proc.stderr)
        self.assertIn("DENY", proc.stdout)

    def test_fixture_allowed_passes_all_purposes(self):
        write_registry(self.csv, [fixture_row()])
        for purpose in gate.PURPOSES:
            proc = run_gate("--source", "fixture_licensed_corpus",
                            "--purpose", purpose, "--csv", str(self.csv))
            self.assertEqual(proc.returncode, gate.EXIT_ALLOWED,
                             f"{purpose}: {proc.stderr} {proc.stdout}")
            self.assertIn("ALLOW", proc.stdout)

    def test_fixture_allowed_json_output(self):
        write_registry(self.csv, [fixture_row()])
        proc = run_gate("--source", "fixture_licensed_corpus", "--purpose", "train",
                        "--csv", str(self.csv), "--json")
        self.assertEqual(proc.returncode, gate.EXIT_ALLOWED, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["allowed"])
        self.assertEqual(payload["source"], "fixture_licensed_corpus")
        self.assertEqual(payload["retrieved_at"], "2025-01-01")

    def test_fixture_passes_only_for_its_stated_purpose(self):
        write_registry(self.csv, [fixture_row(
            train_allowed="UNKNOWN",
            redistribute_allowed="UNKNOWN",
            status="ALLOWED",
        )])
        storage = run_gate("--source", "fixture_licensed_corpus",
                           "--purpose", "storage", "--csv", str(self.csv))
        self.assertEqual(storage.returncode, gate.EXIT_ALLOWED, storage.stderr)

        for purpose in ("train", "redistribute"):
            proc = run_gate("--source", "fixture_licensed_corpus",
                            "--purpose", purpose, "--csv", str(self.csv))
            self.assertEqual(proc.returncode, gate.EXIT_DENIED,
                             f"{purpose}: {proc.stdout} {proc.stderr}")
            self.assertIn("DENY", proc.stdout)

    def test_json_denial_has_reason(self):
        proc = run_gate("--source", "tenhou_old", "--purpose", "train", "--json")
        self.assertEqual(proc.returncode, gate.EXIT_DENIED, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertFalse(payload["allowed"])
        self.assertIn("RESTRICTED", payload["reason"])

    def test_gate_has_no_write_side_effects(self):
        write_registry(self.csv, [fixture_row()])
        before = {p.name: p.stat().st_mtime_ns for p in self.tmp.iterdir()}
        proc = run_gate("--source", "fixture_licensed_corpus", "--purpose", "train",
                        "--csv", str(self.csv))
        self.assertEqual(proc.returncode, gate.EXIT_ALLOWED, proc.stderr)
        after = {p.name: p.stat().st_mtime_ns for p in self.tmp.iterdir()}
        self.assertEqual(before, after)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rf_h03_evidence_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))
        self.csv = self.tmp / "rights.csv"

    def assert_rejected(self, row: dict[str, str], needle: str = ""):
        write_registry(self.csv, [row])
        proc = run_gate("--source", row["source_id"], "--purpose", "train",
                        "--csv", str(self.csv))
        self.assertEqual(proc.returncode, gate.EXIT_INVALID,
                         f"accepted bad row: {row}\n{proc.stdout}\n{proc.stderr}")
        if needle:
            self.assertIn(needle, proc.stderr)

    def test_allowed_requires_evidence_url(self):
        self.assert_rejected(fixture_row(evidence_url=""))
        self.assert_rejected(fixture_row(evidence_url="not-a-url"))

    def test_allowed_requires_evidence_quote(self):
        self.assert_rejected(fixture_row(evidence_quote=""))
        self.assert_rejected(fixture_row(evidence_quote="too short"))

    def test_allowed_requires_retrieval_date(self):
        self.assert_rejected(fixture_row(retrieved_at=""))
        self.assert_rejected(fixture_row(retrieved_at="2026-13-40"))
        self.assert_rejected(fixture_row(retrieved_at="21/11/2026"))
        self.assert_rejected(fixture_row(retrieved_at="2099-01-01"), "future (UTC)")

    def test_allowed_requires_terms_url(self):
        self.assert_rejected(fixture_row(terms_url=""))
        self.assert_rejected(fixture_row(terms_url="UNKNOWN"))

    def test_allowed_requires_privacy_notes(self):
        self.assert_rejected(fixture_row(privacy_notes=""))

    def test_allowed_requires_human_reviewer(self):
        self.assert_rejected(fixture_row(reviewer=""))
        self.assert_rejected(fixture_row(
            reviewer="H-03 Dev-C (AI agent; evidence draft; no human sign-off)"))
        self.assert_rejected(fixture_row(reviewer="TBD"))
        self.assert_rejected(fixture_row(reviewer="pending human review"))

    def test_allowed_rejects_ai_automation_reviewer_tokens(self):
        for reviewer in ("AI", "LLM", "GPT", "ChatGPT", "bot", "automated",
                         "OpenAI", "AI reviewer", "an LLM", "GPT-4 reviewer",
                         "ChatGPT-assisted review", "a bot", "automated reviewer"):
            with self.subTest(reviewer=reviewer):
                self.assert_rejected(fixture_row(reviewer=reviewer),
                                     "AI/automation marker")

    def test_human_reviewer_with_ai_substring_is_not_rejected(self):
        # Token matching, not naive substring matching: "Kaito" contains "ai"
        # but is a human name, not an AI marker.
        write_registry(self.csv, [fixture_row(reviewer="Dr. Kaito Yamada")])
        proc = run_gate("--source", "fixture_licensed_corpus", "--purpose", "train",
                        "--csv", str(self.csv))
        self.assertEqual(proc.returncode, gate.EXIT_ALLOWED,
                         f"{proc.stdout} {proc.stderr}")

    def test_status_must_match_purpose_cells(self):
        # Any ALLOWED purpose requires status=ALLOWED.
        self.assert_rejected(fixture_row(status="UNKNOWN"))
        self.assert_rejected(fixture_row(status="RESTRICTED"))
        self.assert_rejected(fixture_row(
            storage_allowed="UNKNOWN",
            train_allowed="ALLOWED",
            redistribute_allowed="UNKNOWN",
            status="UNKNOWN",
        ))
        # status=ALLOWED is meaningless if no purpose is allowed.
        self.assert_rejected(fixture_row(
            storage_allowed="UNKNOWN",
            train_allowed="UNKNOWN",
            redistribute_allowed="UNKNOWN",
            status="ALLOWED",
        ))

    def test_bad_enum_rejected(self):
        self.assert_rejected(fixture_row(status="MAYBE"))
        self.assert_rejected(fixture_row(train_allowed="YES"))
        self.assert_rejected(fixture_row(storage_allowed="allowed"))

    def test_missing_required_column_rejected(self):
        fields = [name for name in FIELDS if name != "train_allowed"]
        write_registry(self.csv, [fixture_row()], fields=fields)
        proc = run_gate("--source", "fixture_licensed_corpus", "--purpose", "train",
                        "--csv", str(self.csv))
        self.assertEqual(proc.returncode, gate.EXIT_INVALID)
        self.assertIn("missing required columns", proc.stderr)

    def test_duplicate_source_id_rejected(self):
        write_registry(self.csv, [fixture_row(), fixture_row()])
        proc = run_gate("--source", "fixture_licensed_corpus", "--purpose", "train",
                        "--csv", str(self.csv))
        self.assertEqual(proc.returncode, gate.EXIT_INVALID)
        self.assertIn("duplicate source_id", proc.stderr)

    def test_legacy_registry_schema_rejected(self):
        legacy_fields = [
            "source_id", "display_name", "terms_url", "rights_status",
            "rights_scope", "reviewed_at_utc", "evidence", "notes",
            "allows_new_training",
        ]
        write_registry(self.csv, [{
            "source_id": "tenhou_old",
            "display_name": "legacy",
            "terms_url": "https://tenhou.net/term.html",
            "rights_status": "unknown",
            "rights_scope": "audit",
            "reviewed_at_utc": "",
            "evidence": "legacy artifact",
            "notes": "",
            "allows_new_training": "False",
        }], fields=legacy_fields)
        proc = run_gate("--source", "tenhou_old", "--purpose", "train",
                        "--csv", str(self.csv))
        self.assertEqual(proc.returncode, gate.EXIT_INVALID)
        self.assertIn("missing required columns", proc.stderr)

    def test_extra_cells_rejected(self):
        with self.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            writer.writerow([BASE_ROW[name] for name in FIELDS] + ["surprise"])
        proc = run_gate("--source", "fixture_licensed_corpus", "--purpose", "train",
                        "--csv", str(self.csv))
        self.assertEqual(proc.returncode, gate.EXIT_INVALID)
        self.assertIn("too many fields", proc.stderr)

    def test_non_utf8_registry_rejected(self):
        self.csv.write_bytes(b"\xff\xfe\x00\x00")
        proc = run_gate("--source", "tenhou_old", "--purpose", "train",
                        "--csv", str(self.csv))
        self.assertEqual(proc.returncode, gate.EXIT_INVALID)
        self.assertIn("UTF-8", proc.stderr)

    def test_empty_and_header_only_registries_rejected(self):
        self.csv.write_text("", encoding="utf-8")
        proc = run_gate("--source", "tenhou_old", "--purpose", "train",
                        "--csv", str(self.csv))
        self.assertEqual(proc.returncode, gate.EXIT_INVALID)

        with self.csv.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(FIELDS)
        proc = run_gate("--source", "tenhou_old", "--purpose", "train",
                        "--csv", str(self.csv))
        self.assertEqual(proc.returncode, gate.EXIT_INVALID)
        self.assertIn("no data rows", proc.stderr)

    def test_missing_registry_file_rejected(self):
        proc = run_gate("--source", "tenhou_old", "--purpose", "train",
                        "--csv", str(self.tmp / "absent.csv"))
        self.assertEqual(proc.returncode, gate.EXIT_INVALID)
        self.assertIn("registry not found", proc.stderr)


class LibraryTests(unittest.TestCase):
    """Direct API tests: the CLI is not the only supported entry point."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rf_h03_lib_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))

    def test_load_registry_returns_all_rows(self):
        registry = gate.load_registry(PRODUCTION_CSV)
        self.assertIn("tenhou_old", registry)
        self.assertEqual(registry["tenhou_old"]["train_allowed"], "RESTRICTED")

    def test_check_unknown_source_raises(self):
        registry = gate.load_registry(PRODUCTION_CSV)
        with self.assertRaises(gate.RegistryError):
            gate.check(registry, "no_such_source", "train")

    def test_check_rejects_bad_purpose(self):
        registry = gate.load_registry(PRODUCTION_CSV)
        with self.assertRaises(gate.RegistryError):
            gate.check(registry, "tenhou_old", "publish")

    def test_check_missing_known_source_raises(self):
        registry = gate.load_registry(PRODUCTION_CSV)
        del registry["tenhou_old"]
        with self.assertRaises(gate.RegistryError) as ctx:
            gate.check(registry, "tenhou_old", "train")
        self.assertIn("missing from the registry", str(ctx.exception))

    def test_check_revalidates_hand_built_row_status(self):
        # A hand-built dict must not smuggle train_allowed=ALLOWED past
        # status=UNKNOWN just because it bypassed load_registry().
        row = fixture_row(
            storage_allowed="UNKNOWN",
            train_allowed="ALLOWED",
            redistribute_allowed="UNKNOWN",
            status="UNKNOWN",
        )
        registry = {row["source_id"]: row}
        with self.assertRaises(gate.RegistryError):
            gate.check(registry, row["source_id"], "train")

    def test_check_accepts_consistent_allowed_row(self):
        row = fixture_row(
            storage_allowed="UNKNOWN",
            train_allowed="ALLOWED",
            redistribute_allowed="UNKNOWN",
            status="ALLOWED",
        )
        registry = {row["source_id"]: row}
        allowed, reason, _ = gate.check(registry, row["source_id"], "train")
        self.assertTrue(allowed, reason)

    def test_check_allowed_fixture(self):
        csv_path = self.tmp / "rights.csv"
        write_registry(csv_path, [fixture_row()])
        registry = gate.load_registry(csv_path)
        allowed, reason, row = gate.check(registry, "fixture_licensed_corpus", "train")
        self.assertTrue(allowed, reason)
        self.assertIsNotNone(row)
        self.assertEqual(row["evidence_url"], BASE_ROW["evidence_url"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
