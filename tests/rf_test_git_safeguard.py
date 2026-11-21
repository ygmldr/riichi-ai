#!/usr/bin/env python3
"""Tests for `scripts/rf_git_safeguard.py` (H-00).

Uses throw-away repositories only; never touches the real worktree, network,
datasets, checkpoints or frozen eval runs.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rf_git_safeguard as guard  # noqa: E402


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C.UTF-8"},
    )


def snapshot_cli(*args: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = guard.main(list(args))
    return code, out.getvalue(), err.getvalue()


class GitRepoCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        run_git(self.repo, "init", "-q", "-b", "main")
        run_git(self.repo, "config", "user.email", "test@example.invalid")
        run_git(self.repo, "config", "user.name", "Test")
        (self.repo / ".gitignore").write_text(
            "/data/\ndatasets/\nartifacts/riichiformer/\nworktrees/\n*.pyc\n",
            encoding="utf-8",
        )
        (self.repo / "riichiformer" / "data").mkdir(parents=True)
        (self.repo / "riichiformer" / "data" / "replay.py").write_text("x = 1\n", encoding="utf-8")
        (self.repo / "README.md").write_text("hello\n", encoding="utf-8")
        run_git(self.repo, "add", ".")
        run_git(self.repo, "commit", "-q", "-m", "base")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def snapshot(self, name: str = "snap", *, remote: bool = False) -> Path:
        out = self.root / name
        args = ["snapshot", "--repo", str(self.repo), "--out", str(out)]
        if not remote:
            args.append("--no-remote")
        code, stdout, stderr = snapshot_cli(*args)
        self.assertEqual(code, 0, stderr + stdout)
        return out


class SnapshotTests(GitRepoCase):
    def test_snapshot_copies_dirty_and_untracked_but_skips_sensitive_and_ignored(self) -> None:
        (self.repo / "README.md").write_text("changed\n", encoding="utf-8")
        (self.repo / "notes with space").mkdir()
        (self.repo / "notes with space" / "todo.txt").write_text("keep me\n", encoding="utf-8")
        (self.repo / ".env").write_text("SECRET=1\n", encoding="utf-8")
        (self.repo / "data").mkdir()
        (self.repo / "data" / "ignored.json.gz").write_text("ignored\n", encoding="utf-8")
        before = (self.repo / "README.md").read_bytes()

        out = self.snapshot()
        baseline = json.loads((out / "baseline.json").read_text(encoding="utf-8"))
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(baseline["head"], run_git(self.repo, "rev-parse", "HEAD").stdout.decode().strip())
        self.assertEqual(baseline["branch"], "main")
        self.assertIn("README.md", baseline["dirty_tracked"])
        self.assertIn("notes with space/todo.txt", baseline["untracked"])
        self.assertIn(".env", baseline["untracked"])
        self.assertIn(".env", baseline["skipped_sensitive"])
        self.assertTrue(baseline["ignore_checks"]["riichiformer/data/replay.py"]["ok"])
        self.assertEqual((self.repo / "README.md").read_bytes(), before)

        copied = (out / "files" / "worktree" / "README.md").read_text(encoding="utf-8")
        self.assertEqual(copied, "changed\n")
        self.assertTrue((out / "files" / "worktree" / "notes with space" / "todo.txt").is_file())
        self.assertFalse((out / "files" / "worktree" / ".env").exists())
        self.assertFalse((out / "files" / "worktree" / "data" / "ignored.json.gz").exists())
        self.assertTrue(any(e["backup"].endswith("notes with space/todo.txt") for e in manifest["entries"]))

        code, stdout, stderr = snapshot_cli("verify", str(out))
        self.assertEqual(code, 0, stderr + stdout)
        self.assertTrue(json.loads(stdout)["ok"])

    def test_snapshot_refuses_nonempty_output(self) -> None:
        out = self.root / "occupied"
        out.mkdir()
        (out / "x").write_text("x", encoding="utf-8")
        code, _, stderr = snapshot_cli(
            "snapshot", "--repo", str(self.repo), "--out", str(out), "--no-remote"
        )
        self.assertEqual(code, 2)
        self.assertIn("not empty", stderr)

    def test_verify_detects_tampering(self) -> None:
        out = self.snapshot()
        target = out / "files" / "worktree" / "README.md"
        self.assertFalse(target.exists())
        (self.repo / "README.md").write_text("dirty\n", encoding="utf-8")
        out = self.snapshot("snap2")
        target = out / "files" / "worktree" / "README.md"
        self.assertTrue(target.is_file())
        target.write_text("tampered\n", encoding="utf-8")
        code, _, stderr = snapshot_cli("verify", str(out))
        self.assertEqual(code, 2)
        self.assertIn("sha256 mismatch", stderr)

    def test_vendor_submodule_dirty_state_is_captured(self) -> None:
        vendor = self.repo / "vendor" / "mortal"
        vendor.mkdir(parents=True)
        run_git(vendor, "init", "-q", "-b", "main")
        run_git(vendor, "config", "user.email", "test@example.invalid")
        run_git(vendor, "config", "user.name", "Test")
        (vendor / "lib.rs").write_text("fn a() {}\n", encoding="utf-8")
        run_git(vendor, "add", ".")
        run_git(vendor, "commit", "-q", "-m", "vendor base")
        run_git(self.repo, "add", "vendor/mortal")
        run_git(self.repo, "commit", "-q", "-m", "add gitlink")
        (vendor / "lib.rs").write_text("fn b() {}\n", encoding="utf-8")
        (vendor / "new.rs").write_text("fn c() {}\n", encoding="utf-8")

        out = self.snapshot()
        baseline = json.loads((out / "baseline.json").read_text(encoding="utf-8"))
        self.assertTrue(baseline["vendor"]["present"])
        self.assertIn("lib.rs", baseline["vendor"]["dirty_tracked"])
        self.assertIn("new.rs", baseline["vendor"]["untracked"])
        self.assertEqual((out / "files" / "vendor" / "lib.rs").read_text(encoding="utf-8"), "fn b() {}\n")
        self.assertTrue((out / "files" / "vendor" / "new.rs").is_file())
        self.assertIn("fn b() {}", (out / "vendor" / "worktree.diff").read_text(encoding="utf-8"))

    def test_orphan_history_has_no_common_ancestor(self) -> None:
        run_git(self.repo, "checkout", "-q", "--orphan", "frozen")
        run_git(self.repo, "rm", "-rf", "--cached", ".")
        (self.repo / "only.txt").write_text("frozen\n", encoding="utf-8")
        run_git(self.repo, "add", "only.txt")
        run_git(self.repo, "commit", "-q", "-m", "orphan")
        frozen = run_git(self.repo, "rev-parse", "HEAD").stdout.decode().strip()
        info = guard.ancestry(self.repo, frozen)
        self.assertEqual(info["head"], frozen)
        self.assertIsNotNone(info["merge_base"])
        self.assertTrue(info["is_ancestor"])

        run_git(self.repo, "checkout", "-q", "-f", "main")
        info = guard.ancestry(self.repo, frozen)
        self.assertIsNone(info["merge_base"])
        self.assertFalse(info["is_ancestor"])
        self.assertFalse(info["other_is_ancestor"])


class IgnoreCheckTests(GitRepoCase):
    def test_anchored_data_rule_does_not_hide_first_party_package(self) -> None:
        code, stdout, stderr = snapshot_cli("ignored-check", "--repo", str(self.repo))
        self.assertEqual(code, 0, stderr + stdout)
        self.assertTrue(json.loads(stdout)["ok"])

    def test_unanchored_data_rule_is_detected(self) -> None:
        (self.repo / ".gitignore").write_text(
            "data/\ndatasets/\nartifacts/riichiformer/\nworktrees/\n",
            encoding="utf-8",
        )
        code, stdout, stderr = snapshot_cli("ignored-check", "--repo", str(self.repo))
        self.assertEqual(code, 3, stderr + stdout)
        failed = json.loads(stdout)["failed"]
        self.assertIn("riichiformer/data/replay.py", failed)

    def test_worktrees_are_ignored(self) -> None:
        code, stdout, _ = snapshot_cli("ignored-check", "--repo", str(self.repo))
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stdout)["checks"]["worktrees/probe"]["actual_ignored"])


if __name__ == "__main__":
    unittest.main()
