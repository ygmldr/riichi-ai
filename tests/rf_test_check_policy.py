#!/usr/bin/env python3
"""PR-37 rf_check_policy CLI tests: one decision, explicit 0/1/2 exit codes."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "rf_check_policy.py"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rf_check_policy as cli  # noqa: E402
from riichiformer.policy.config import POLICY_VERSION  # noqa: E402

def policy_mapping(**overrides):
    data = {
        "policy_version": POLICY_VERSION,
        "profile": "operator_private_research",
        "scope": "local_nonpublic_only",
        "operator_reference": "unit-test-check-policy-operator",
        "allowed_local_sources": ["tenhou_old", "selfplay_own"],
        "release_allowed": False,
        "network_upload_allowed": False,
    }
    data.update(overrides)
    return data


class PolicyCheckCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.config = self.dir / "policy.json"
        self.config.write_text(json.dumps(policy_mapping()), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _run_json(self, args):
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main(args)
        return code, json.loads(buffer.getvalue())

    def test_private_distribute_denies_and_preserves_status(self):
        code, payload = self._run_json([
            "--source", "tenhou_old", "--action", "distribute",
            "--profile", "operator_private_research", "--json"])
        self.assertEqual(code, cli.EXIT_DENIED)
        self.assertFalse(payload["operational_allow"])
        self.assertFalse(payload["allow"])
        self.assertEqual(payload["external_license_status"], "RESTRICTED")
        self.assertEqual(payload["source_rights_status"], "RESTRICTED")
        self.assertEqual(payload["selected_profile"],
                         "operator_private_research")
        self.assertEqual(payload["basis"], "not_authorized_for_distribution")

    def test_private_training_with_config_is_allowed(self):
        code, payload = self._run_json([
            "--source", "tenhou_old", "--action", "private_train",
            "--config", str(self.config), "--json"])
        self.assertEqual(code, cli.EXIT_ALLOWED)
        self.assertTrue(payload["operational_allow"])
        self.assertEqual(payload["basis"],
                         "operator_attestation_not_external_license")
        self.assertEqual(payload["external_license_status"], "RESTRICTED")
        self.assertEqual(payload["config_path"], str(self.config))
        self.assertEqual(len(payload["policy_hash"]), 64)

    def test_private_training_without_config_fails_closed_as_deny(self):
        code, payload = self._run_json([
            "--source", "tenhou_old", "--action", "private_train",
            "--profile", "operator_private_research", "--json"])
        self.assertEqual(code, cli.EXIT_DENIED)
        self.assertFalse(payload["operational_allow"])
        self.assertEqual(payload["basis"], "not_covered")

    def test_private_training_source_outside_config_allowlist_denies(self):
        config = self.dir / "narrow.json"
        config.write_text(json.dumps(policy_mapping(
            allowed_local_sources=["selfplay_own"])), encoding="utf-8")
        code, payload = self._run_json([
            "--source", "tenhou_old", "--action", "private_train",
            "--config", str(config), "--json"])
        self.assertEqual(code, cli.EXIT_DENIED)
        self.assertEqual(payload["basis"], "not_covered")

    def test_licensed_training_on_production_row_denies(self):
        code, payload = self._run_json([
            "--source", "tenhou_old", "--action", "private_train",
            "--profile", "licensed_only", "--json"])
        self.assertEqual(code, cli.EXIT_DENIED)
        self.assertEqual(payload["basis"], "unverified_external")

    def test_fixture_licensed_row_can_train_under_licensed_only(self):
        registry = self.dir / "rights.csv"
        registry.write_text(
            "source_id,status,train_allowed,redistribute_allowed,evidence_url\n"
            "fixture_licensed,ALLOWED,ALLOWED,ALLOWED,"
            "https://example.invalid/fixture-license\n",
            encoding="utf-8")
        code, payload = self._run_json([
            "--source", "fixture_licensed", "--action", "private_train",
            "--profile", "licensed_only", "--source-rights", str(registry),
            "--json"])
        self.assertEqual(code, cli.EXIT_ALLOWED)
        self.assertEqual(payload["basis"], "verified_external")
        self.assertEqual(payload["evidence_ref"],
                         "https://example.invalid/fixture-license")

    def test_private_profile_cannot_distribute_a_licensed_row(self):
        registry = self.dir / "rights.csv"
        registry.write_text(
            "source_id,status,train_allowed,redistribute_allowed\n"
            "fixture_licensed,ALLOWED,ALLOWED,ALLOWED\n",
            encoding="utf-8")
        code, payload = self._run_json([
            "--source", "fixture_licensed", "--action", "distribute",
            "--profile", "operator_private_research",
            "--source-rights", str(registry), "--json"])
        self.assertEqual(code, cli.EXIT_DENIED)
        self.assertEqual(payload["external_license_status"], "ALLOWED")
        self.assertEqual(payload["basis"], "not_authorized_for_distribution")

    def test_unknown_source_is_invalid_exit_2(self):
        code = cli.main(["--source", "no_such_source", "--action", "private_train",
                         "--profile", "operator_private_research"])
        self.assertEqual(code, cli.EXIT_INVALID)

    def test_unknown_action_is_rejected_by_argparse(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--source", "tenhou_old", "--action", "publish",
                      "--profile", "operator_private_research"])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_missing_profile_and_config_is_invalid_exit_2(self):
        code = cli.main(["--source", "tenhou_old", "--action", "private_train"])
        self.assertEqual(code, cli.EXIT_INVALID)

    def test_profile_config_mismatch_is_invalid_exit_2(self):
        code = cli.main(["--source", "tenhou_old", "--action", "private_train",
                         "--profile", "licensed_only",
                         "--config", str(self.config)])
        self.assertEqual(code, cli.EXIT_INVALID)

    def test_invalid_config_file_is_invalid_exit_2(self):
        bad = self.dir / "bad.yaml"
        bad.write_text("profile: [unterminated\n", encoding="utf-8")
        code = cli.main(["--source", "tenhou_old", "--action", "private_train",
                         "--config", str(bad)])
        self.assertEqual(code, cli.EXIT_INVALID)

    def test_malformed_registry_is_invalid_exit_2(self):
        registry = self.dir / "rights.csv"
        registry.write_text("source_id\n tenhou_old\n", encoding="utf-8")
        code = cli.main(["--source", "tenhou_old", "--action", "audit",
                         "--profile", "licensed_only",
                         "--source-rights", str(registry)])
        self.assertEqual(code, cli.EXIT_INVALID)

    def test_missing_registry_is_invalid_exit_2(self):
        code = cli.main(["--source", "tenhou_old", "--action", "audit",
                         "--profile", "licensed_only",
                         "--source-rights", str(self.dir / "absent.csv")])
        self.assertEqual(code, cli.EXIT_INVALID)

    def test_exit_codes_are_distinct(self):
        self.assertEqual(
            (cli.EXIT_ALLOWED, cli.EXIT_DENIED, cli.EXIT_INVALID), (0, 1, 2))

    def test_module_entry_point_wiring(self):
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--source", "tenhou_old",
             "--action", "distribute", "--profile", "operator_private_research",
             "--json"],
            capture_output=True, text=True, cwd=str(REPO_ROOT))
        self.assertEqual(proc.returncode, cli.EXIT_DENIED, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["external_license_status"], "RESTRICTED")
        self.assertFalse(payload["operational_allow"])
        self.assertEqual(proc.stderr, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
