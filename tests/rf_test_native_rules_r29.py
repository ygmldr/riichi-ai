#!/usr/bin/env python3
"""PR-40R — R29 native ippatsu fixture and rebuild-evidence contract.

This is the Python half of PR-40R.  The executable native half is
``patches/0015-rule-r29-ippatsu-fixture.patch`` (a ``#[cfg(test)] mod r29_tests``
appended to ``vendor/mortal/libriichi/src/state/update.rs``), and the rule matrix /
rebuild record is ``docs/riichiformer/v3_3/PR40R_NATIVE_RULES.md``.

The tests assert three independent things:

1. the tracked patch is present, is append-only, and still applies after 0014
   (the apply-check itself is recorded in the document/log; this file checks the
   patch shape and the source anchors it creates);
2. the compiled native test binary really passes the four R29 tests when it is
   available (it is not counted as a pass when the binary has not been built);
3. the report has an explicit, non-ambiguous R29 status and the matrix does not
   silently regress R02/R25/R26 to prose-only claims.

Run with::

    PYTHONPATH=vendor/mortal/target/release \
      /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q \
      tests/rf_test_native_rules_r29.py
"""

from __future__ import annotations

import os
import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UPDATE_RS = REPO_ROOT / "vendor" / "mortal" / "libriichi" / "src" / "state" / "update.rs"
PATCH_0014 = REPO_ROOT / "patches" / "0014-rule-r02-r25-r26.patch"
PATCH_0015 = REPO_ROOT / "patches" / "0015-rule-r29-ippatsu-fixture.patch"
DOC = REPO_ROOT / "docs" / "riichiformer" / "v3_3" / "PR40R_NATIVE_RULES.md"
LOG_DIR = DOC.parent / "logs"

BUILD_SHA = "582e36734b10c3d3b1230a834eacc494f94e40f6d642c712eadc23f6cabfaf4e"
R29_TESTS = (
    "r29_first_kakan_keeps_ippatsu_through_the_chankan_probe",
    "r29_kakan_window_closes_at_the_next_event_before_the_rinshan_tsumo",
    "r29_second_consecutive_kakan_has_no_ippatsu",
    "r29_own_discard_and_other_calls_close_the_window",
)
RULE_TESTS = (
    "r02_triple_ron_aborts_without_settlement",
    "r25_last_dealer_stop_excludes_just_paid_kyotaku",
    "r26_fourth_riichi_aborts_before_the_next_draw",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def matrix_rows(text: str) -> dict[str, list[str]]:
    """Parse ``| Rxx | ... |`` rows from the rule-matrix section."""
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        if re.match(r"^\|\s*R\d{2}\s*\|", line):
            cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
            rows[cells[0]] = cells
    return rows


class PatchContractTests(unittest.TestCase):
    def test_0015_is_present_and_append_only(self):
        self.assertTrue(PATCH_0015.is_file(), f"missing {PATCH_0015}")
        patch = read(PATCH_0015)
        self.assertTrue(
            patch.startswith(
                "diff --git a/libriichi/src/state/update.rs b/libriichi/src/state/update.rs\n"
            ),
            "0015 must be rooted at libriichi/src/state/update.rs",
        )
        self.assertIn("@@ -971,3 +971,188 @@", patch)
        added = [
            line
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        removed = [
            line
            for line in patch.splitlines()
            if line.startswith("-") and not line.startswith("---")
        ]
        self.assertEqual(len(added), 185, "0015 must add exactly the test module")
        self.assertEqual(removed, [], "0015 must not remove or rewrite production lines")
        for name in R29_TESTS:
            with self.subTest(test=name):
                self.assertIn(f"fn {name}()", patch)
        self.assertIn("mod r29_tests", patch)
        self.assertIn("#[cfg(test)]", patch)

    def test_0014_still_supplies_r02_r25_r26(self):
        self.assertTrue(PATCH_0014.is_file(), f"missing {PATCH_0014}")
        patch = read(PATCH_0014)
        for name in RULE_TESTS:
            with self.subTest(test=name):
                self.assertIn(f"fn {name}()", patch)
        self.assertIn("n_hora >= 3", patch)
        self.assertIn("last_dealer_stop", patch)
        self.assertIn("accepted_riichis == 4", patch)


class NativeSourceContractTests(unittest.TestCase):
    def test_source_anchors_still_match_the_reported_lines(self):
        lines = read(UPDATE_RS).splitlines()
        self.assertGreater(len(lines), 684)
        # 1-based line numbers from the report: 59/604/606/684.
        self.assertEqual(lines[58].strip(), "self.at_ippatsu = false;")
        self.assertEqual(lines[603].strip(), "self.chankan_chance = Some(());")
        self.assertEqual(lines[605].strip(), "self.at_ippatsu = false;")
        self.assertEqual(lines[683].strip(), "self.at_ippatsu = true;")
        self.assertIn("chankan_chance.take().is_some()", "\n".join(lines[55:62]))

    def test_fixture_asserts_the_clause_not_just_the_flag(self):
        text = read(UPDATE_RS)
        tail = text[text.index("#[cfg(test)]\nmod r29_tests"):]
        for name in R29_TESTS:
            with self.subTest(test=name):
                self.assertIn(f"fn {name}()", tail)
        # ippatsu + chankan = 3 han 40 fu dealer 7700; chankan-only = 3900.
        self.assertIn("assert_eq!(point.ron, 7700)", tail)
        self.assertIn("assert_eq!(point.ron, 3900)", tail)
        self.assertIn("agari_points(true, &[])", tail)
        self.assertIn("連続する加槓の２回目には一発は付かない", tail)
        self.assertIn("Event::Dora", tail)
        self.assertIn("Event::Tsumo", tail)


class DocStatusTests(unittest.TestCase):
    def test_rule_matrix_has_explicit_statuses(self):
        self.assertTrue(DOC.is_file(), f"missing {DOC}")
        rows = matrix_rows(read(DOC))
        for rid in ("R02", "R25", "R26"):
            with self.subTest(rule=rid):
                self.assertIn(rid, rows)
                self.assertEqual(rows[rid][6], "tenhou-parity-verified")
        self.assertIn("R29", rows)
        self.assertEqual(rows["R29"][6], "MATCH")
        self.assertNotIn("UNVERIFIED", " ".join(rows["R29"]))
        for name in R29_TESTS:
            with self.subTest(test=name):
                self.assertIn(name, read(DOC))

    def test_doc_records_the_rebuild_and_scope(self):
        text = read(DOC)
        self.assertIn(BUILD_SHA, text)
        self.assertIn("0cff2b52982be5b1163aa9a62fb01f03ce91e0d2", text)
        self.assertIn("0015-rule-r29-ippatsu-fixture.patch", text)
        self.assertIn("BoardState::handle_hora", text)
        self.assertIn("agari_points", text)
        self.assertIn("PlayerState", text)
        self.assertIn("tenhou_approx", text)
        # The report must not overclaim a full-board scripted certification.
        self.assertIn("NOT_RUN", text)
        self.assertIn("Full `Board`-level scripted R29 game", text)


class NativeExecutionTests(unittest.TestCase):
    def _native_test_binary(self) -> Path | None:
        deps = REPO_ROOT / "vendor" / "mortal" / "target" / "debug" / "deps"
        if not deps.is_dir():
            return None
        candidates = [
            p
            for p in deps.glob("riichi-*")
            if p.is_file()
            and p.suffix == ""
            and os.access(p, os.X_OK)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def test_compiled_r29_tests_pass(self):
        binary = self._native_test_binary()
        if binary is None:
            self.skipTest(
                "native test binary not built; run "
                "`cargo test -p libriichi --lib --no-default-features` first"
            )
        proc = subprocess.run(
            [str(binary), "r29_tests"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        output = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, output)
        self.assertIn("test result: ok. 4 passed; 0 failed", output)
        for name in R29_TESTS:
            with self.subTest(test=name):
                self.assertIn(f"state::update::r29_tests::{name}", output)

    def test_recorded_native_logs_match_the_report(self):
        full = read(LOG_DIR / "PR40R_cargo_test_full.log")
        self.assertIn("test result: ok. 35 passed; 0 failed", full)
        for name in RULE_TESTS:
            with self.subTest(test=name):
                self.assertIn(name, full)
        r29 = read(LOG_DIR / "PR40R_cargo_test_r29.log")
        self.assertIn("test result: ok. 4 passed; 0 failed", r29)
        for name in R29_TESTS:
            with self.subTest(test=name):
                self.assertIn(f"state::update::r29_tests::{name}", r29)
        build = read(LOG_DIR / "PR40R_release_build.log")
        self.assertIn("# exit: 0", build)
        self.assertIn(BUILD_SHA, build)


if __name__ == "__main__":
    unittest.main(verbosity=2)
