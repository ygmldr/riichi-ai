#!/usr/bin/env python3
"""Smoke tests for the real-data VNext trainer CLI (no dataset read)."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import rf_train_vnext as T  # noqa: E402


class _FakeReader:
    def __init__(self):
        self.metadata = SimpleNamespace(
            purpose="train", source_id="tenhou_old",
            research_policy="operator_private_research",
            research_policy_hash="a" * 64,
            train_eligible_under_policy=True, train_allowed=False,
            rights_status="RESTRICTED", split_manifest_sha256="b" * 64,
            canonical_manifest_sha256="c" * 64,
            canonical_index_sha256="d" * 64,
        )
        self.closed = False

    def close(self):
        self.closed = True


class TrainerCliTests(unittest.TestCase):
    def test_dry_run_does_not_open_a_model(self):
        fake = _FakeReader()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(T, "_open_reader", return_value=fake), \
                contextlib.redirect_stdout(io.StringIO()):
            code = T.main([
                "--train-shard", "synthetic.rfshard",
                "--policy-config", "configs/riichiformer/local_research_policy.yaml",
                "--out", str(Path(tmp) / "run"), "--dry-run",
            ])
            self.assertEqual(code, 0)
            self.assertTrue(fake.closed)

    def test_refuses_frozen_output(self):
        with self.assertRaises(SystemExit):
            T.main([
                "--train-shard", "synthetic.rfshard",
                "--out", str(REPO_ROOT / "datasets" / "vnext"),
                "--dry-run",
            ])


if __name__ == "__main__":
    unittest.main()
