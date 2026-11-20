#!/usr/bin/env python3
"""Tests for `scripts/rf_preflight.py` (RiichiFormer V2 PR-00).

Run either way:
  python3 -m pytest -q tests/rf_test_preflight.py
  python3 tests/rf_test_preflight.py          # stdlib unittest, no third-party deps

The fixtures are throw-away git repositories under a temp dir. They never touch
the real repo, real checkpoints or the network. Every test that could write is
asserted to write only the requested JSON artefact.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rf_preflight as rf  # noqa: E402

PREFIX = "RF_PROBE_JSON "


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=60
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


class FakeRepo:
    """A minimal but real git repo with an embedded submodule."""

    def __init__(
        self,
        root: Path,
        *,
        with_submodule: bool = True,
        patches=("0001", "0002", "0003"),
        checkpoints: bool = True,
        optional: bool = True,
    ):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.sub_sha = None
        self._init_git()
        _write(root / ".gitignore", "artifacts/\n__pycache__/\n")
        _write(
            root / ".gitmodules",
            '[submodule "vendor/mortal"]\n\tpath = vendor/mortal\n'
            "\turl = https://github.com/Equim-chan/Mortal.git\n",
        )
        _write(root / "configs" / "config.toml", "[1v3]\nstate_file = 'checkpoints/mortal.pth'\n")
        _write(root / "scripts" / "eval_report.py", "RANK_PTS = (90, 45, 0, -135)\n")
        for name in patches:
            _write(root / "patches" / f"{name}-fixture.patch", f"Subject: patch {name}\n")
        if checkpoints:
            _write(root / "checkpoints" / "mortal.pth", "fake-checkpoint-bytes\n")
            _write(root / "checkpoints" / "best.pth", "fake-best-bytes\n")
            _write(root / "checkpoints" / "baseline.pth", "fake-baseline-bytes\n")
            _write(root / "checkpoints" / "grp.pth", "fake-grp-bytes\n")
            _write(root / "checkpoints" / "file_index.pth", "index\n")
        if optional:
            for name in ("README.md", "RUNBOOK.md", "NOTICE.md", "LICENSE"):
                _write(root / name, f"{name} fixture\n")
        if with_submodule:
            self._add_submodule()
        self.commit("fixture")

    def _init_git(self):
        _git(self.root, "init", "-q", "-b", "main")
        _git(self.root, "config", "user.email", "test@example.invalid")
        _git(self.root, "config", "user.name", "fixture")

    def _add_submodule(self):
        sub = self.root / "vendor" / "mortal"
        sub.mkdir(parents=True, exist_ok=True)
        _write(sub / "mortal" / "model.py", "# fixture mortal\n[patch 0001] marker\n")
        _write(sub / "libriichi" / "src" / "mjai" / "event.rs", "// fixture\n")
        _git(sub, "init", "-q", "-b", "main")
        _git(sub, "config", "user.email", "test@example.invalid")
        _git(sub, "config", "user.name", "fixture")
        _git(sub, "add", "-A")
        _git(sub, "commit", "-q", "-m", "sub")
        self.sub_sha = _git(sub, "rev-parse", "HEAD")

    def commit(self, message: str):
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-q", "-m", message)

    def head(self) -> str:
        return _git(self.root, "rev-parse", "HEAD")


def run_preflight(repo: Path, *extra, skip_runtime: bool = True, skip_load: bool = True):
    output = repo / "artifacts" / "riichiformer" / "preflight.json"
    argv = ["--repo-root", str(repo), "--output", str(output)]
    if skip_runtime:
        argv.append("--skip-runtime")
    if skip_load:
        argv.append("--skip-checkpoint-load")
    argv.extend(extra)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = rf.main(argv)
    data = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else None
    return code, data, buf.getvalue(), output


def snapshot_tree(root: Path):
    """hash every file except .git/ and the artefacts directory."""
    out = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] == ".git":
            continue
        out[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def have_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def have_git() -> bool:
    return shutil.which("git") is not None


@unittest.skipUnless(have_git(), "git is required")
class PreflightArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="rf_preflight_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def new_repo(self, name="repo", **kwargs) -> FakeRepo:
        return FakeRepo(self.tmp / name, **kwargs)

    # ------------------------------------------------------------------ #
    def test_artifact_contains_required_keys(self):
        repo = self.new_repo()
        code, data, out, _ = run_preflight(repo.root)
        self.assertEqual(code, 0, out)
        for key in (
            "status",
            "git_sha",
            "submodule_sha",
            "torch",
            "cuda",
            "gpu_name",
            "vram",
            "git_dirty",
            "baseline_paths",
            "warnings",
            "blockers",
            "gate",
        ):
            self.assertIn(key, data)
        self.assertEqual(data["git_sha"], repo.head())
        self.assertEqual(data["submodule_sha"], repo.sub_sha)
        self.assertFalse(data["git_dirty"])
        self.assertIn("G0", data["gate"])
        self.assertEqual(data["schema"], rf.SCHEMA_VERSION)

    def test_missing_required_baseline_is_blocked_and_not_created(self):
        repo = self.new_repo()
        (repo.root / "configs" / "config.toml").unlink()
        code, data, out, _ = run_preflight(repo.root)
        self.assertEqual(code, 2, out)
        self.assertEqual(data["status"], "BLOCKED")
        self.assertEqual(data["gate"]["G0"], "BLOCKED")
        codes = [b["code"] for b in data["blockers"]]
        self.assertIn("BASELINE_REQUIRED_MISSING", codes)
        # the preflight must never "repair" the freeze by creating the file
        self.assertFalse((repo.root / "configs" / "config.toml").exists())

    def test_missing_checkpoint_is_unavailable_never_fabricated(self):
        repo = self.new_repo(checkpoints=False)
        code, data, out, _ = run_preflight(repo.root)
        self.assertEqual(code, 0, out)
        self.assertEqual(data["status"], "WARN")
        self.assertNotEqual(data["status"], "BLOCKED")
        self.assertIn("checkpoints/mortal.pth", data["baseline_paths"])
        self.assertEqual(data["baseline_paths"]["checkpoints/mortal.pth"]["status"], "NOT_FOUND")
        self.assertFalse((repo.root / "checkpoints").exists())
        self.assertIn("CHECKPOINT_MISSING", [w["code"] for w in data["warnings"]])

    def test_tracked_dirty_is_recorded_as_warning(self):
        repo = self.new_repo()
        with open(repo.root / "README.md", "a", encoding="utf-8") as fh:
            fh.write("user edit\n")
        code, data, out, _ = run_preflight(repo.root)
        self.assertTrue(data["git_dirty"])
        self.assertTrue(data["git_tracked_dirty"])
        self.assertIn("GIT_DIRTY_TRACKED", [w["code"] for w in data["warnings"]])
        self.assertEqual(code, 0, out)
        # the user's edit must survive untouched
        self.assertIn("user edit", (repo.root / "README.md").read_text(encoding="utf-8"))

    def test_porcelain_first_line_preserves_leading_status(self):
        repo = self.new_repo()
        with open(repo.root / ".gitignore", "a", encoding="utf-8") as fh:
            fh.write("*.bak\n")
        _, data, _, _ = run_preflight(repo.root)
        # ".gitignore" sorts first, so its porcelain line starts with a space;
        # trimming the whole stdout used to corrupt the status column and path.
        entry = next(e for e in data["git_dirty_entries"] if e["path"] == ".gitignore")
        self.assertEqual(entry["status"], " M")

    def test_untracked_only_is_not_tracked_dirty(self):
        repo = self.new_repo()
        _write(repo.root / "scratch.tmp", "x\n")
        _, data, _, _ = run_preflight(repo.root)
        self.assertTrue(data["git_dirty"])
        self.assertFalse(data["git_tracked_dirty"])
        self.assertNotIn("GIT_DIRTY_TRACKED", [w["code"] for w in data["warnings"]])
        self.assertEqual(data["git_untracked_count"], 1)

    def test_only_output_file_is_written(self):
        repo = self.new_repo()
        before = snapshot_tree(repo.root)
        _, _, _, _ = run_preflight(repo.root)
        after = snapshot_tree(repo.root)
        self.assertEqual(
            set(after) - set(before), {"artifacts/riichiformer/preflight.json"}
        )
        self.assertEqual(sorted(os.listdir(repo.root / "artifacts" / "riichiformer")),
                         ["preflight.json"])
        for rel, digest in before.items():
            self.assertEqual(after[rel], digest, f"{rel} changed during preflight")

    def test_missing_submodule_is_blocked(self):
        repo = self.new_repo(with_submodule=False)
        _write(repo.root / "vendor" / ".keep", "")
        _git(repo.root, "add", "-A")
        _git(repo.root, "commit", "-q", "-m", "no submodule")
        code, data, out = run_preflight(repo.root)[:3]
        self.assertEqual(code, 2, out)
        self.assertIn("SUBMODULE_MISSING", [b["code"] for b in data["blockers"]])

    def test_patch_inventory_and_documented_gap(self):
        repo = self.new_repo(patches=tuple(f"{n:04d}" for n in list(range(1, 11)) + [12, 13]))
        _, data, _, _ = run_preflight(repo.root)
        patches = data["patches"]
        self.assertEqual(patches["count"], 12)
        self.assertEqual(patches["missing_numbers"], [11])
        self.assertEqual(patches["documented_gaps"], [11])
        self.assertNotIn("PATCH_GAP", [w["code"] for w in data["warnings"]])
        self.assertEqual(
            [f["path"] for f in patches["files"]][-3:],
            [
                "patches/0010-fixture.patch",
                "patches/0012-fixture.patch",
                "patches/0013-fixture.patch",
            ],
        )

    def test_submodule_patch_markers_are_counted(self):
        repo = self.new_repo()
        _, data, _, _ = run_preflight(repo.root)
        self.assertEqual(data["submodule"]["patch_marker_count"], 1)
        self.assertFalse(data["submodule"]["dirty"])
        self.assertEqual(data["submodule_sha"], repo.sub_sha)

    def test_submodule_dirty_is_warned(self):
        repo = self.new_repo()
        _write(repo.root / "vendor" / "mortal" / "mortal" / "model.py", "changed\n")
        _, data, _, _ = run_preflight(repo.root)
        self.assertTrue(data["submodule"]["dirty"])
        self.assertIn("SUBMODULE_DIRTY", [w["code"] for w in data["warnings"]])

    def test_strict_flag_promotes_warnings_to_exit_1(self):
        repo = self.new_repo(checkpoints=False)
        code, data, out = run_preflight(repo.root, "--strict")[:3]
        self.assertEqual(data["status"], "WARN")
        self.assertEqual(code, 1, out)

    def test_stdout_mode_keeps_json_parseable_and_summary_on_stderr(self):
        repo = self.new_repo()
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            code = rf.main(
                [
                    "--repo-root",
                    str(repo.root),
                    "--output",
                    "-",
                    "--skip-runtime",
                    "--skip-checkpoint-load",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(out_buf.getvalue())
        self.assertEqual(payload["schema"], rf.SCHEMA_VERSION)
        self.assertIn("rf_preflight: status=", err_buf.getvalue())

    def test_artefact_core_is_deterministic(self):
        repo = self.new_repo()
        _, first, _, _ = run_preflight(repo.root)
        _, second, _, _ = run_preflight(repo.root)
        volatile = {"generated_at_utc", "duration_seconds", "nvidia_smi", "environment"}
        for key in first:
            if key in volatile:
                continue
            self.assertEqual(first[key], second[key], f"non-deterministic field: {key}")
        self.assertEqual(first["baseline_files_hash"], second["baseline_files_hash"])

    def test_baseline_files_hash_covers_checkpoints_and_patches(self):
        repo = self.new_repo()
        _, data, _, _ = run_preflight(repo.root)
        self.assertTrue(data["baseline_files_hash"])
        # a frozen file changing must change the aggregate hash
        with open(repo.root / "patches" / "0001-fixture.patch", "a", encoding="utf-8") as fh:
            fh.write("tamper\n")
        _, data2, _, _ = run_preflight(repo.root)
        self.assertNotEqual(data["baseline_files_hash"], data2["baseline_files_hash"])

    def test_real_runtime_probe_shape(self):
        repo = self.new_repo()
        runtime = rf.collect_runtime(sys.executable)
        self.assertTrue(runtime["probe_ok"], runtime.get("error"))
        for key in ("executable", "python_version", "torch"):
            self.assertIn(key, runtime)
        if runtime["torch"]:
            self.assertIn("cuda_available", runtime["torch"])
            self.assertIn("device_count", runtime["torch"])

    def test_checkpoint_is_loaded_and_config_flag_reported(self):
        if not have_torch():
            self.skipTest("torch is not installed in this interpreter")
        import torch

        repo = self.new_repo(checkpoints=False)
        ckpt_dir = repo.root / "checkpoints"
        ckpt_dir.mkdir()
        model = {"proj.weight": torch.zeros(4, 3), "event_branch.proj.weight": torch.zeros(2, 2)}
        torch.save(
            {
                "mortal": model,
                "current_dqn": model,
                "aux_net": {"aux.weight": torch.zeros(1, 2)},
                "config": {"model": {"use_event_branch": True, "d_model": 256}},
                "steps": 17,
            },
            ckpt_dir / "mortal.pth",
        )
        repo.commit("add checkpoint")
        code, data, out, _ = run_preflight(
            repo.root, "--torch-interpreter", sys.executable, skip_load=False
        )
        self.assertEqual(code, 0, out)
        rec = data["checkpoints"]["readability"]["checkpoints/mortal.pth"]
        self.assertTrue(rec["readable"], rec)
        self.assertEqual(rec["state_dict_key"], "mortal")
        self.assertEqual(rec["steps"], 17)
        self.assertEqual(rec["config_control"], {})
        self.assertTrue(rec["use_event_branch"])
        self.assertEqual(rec["param_count"], 16)
        self.assertTrue(rec["state_dict_has_event_weights"])
        self.assertTrue(rec["aux_net_has_event_weights"] is False)
        self.assertEqual(rec["config_d_model"], 256)

    def test_corrupt_checkpoint_warns_without_crashing(self):
        if not have_torch():
            self.skipTest("torch is not installed in this interpreter")
        repo = self.new_repo(checkpoints=False)
        (repo.root / "checkpoints").mkdir()
        (repo.root / "checkpoints" / "mortal.pth").write_bytes(b"definitely not a pickle")
        repo.commit("corrupt checkpoint")
        code, data, out, _ = run_preflight(
            repo.root, "--torch-interpreter", sys.executable, skip_load=False
        )
        self.assertEqual(code, 0, out)
        rec = data["checkpoints"]["readability"]["checkpoints/mortal.pth"]
        self.assertFalse(rec["readable"])
        self.assertIn("CHECKPOINT_UNREADABLE", [w["code"] for w in data["warnings"]])
        self.assertEqual(data["status"], "WARN")

    def test_sha_and_hash_helpers(self):
        target = self.tmp / "abc.txt"
        target.write_bytes(b"abc")
        self.assertEqual(
            rf.sha256_file(target),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        self.assertEqual(rf.compute_status([], []), "PASS")
        self.assertEqual(rf.compute_status([], [{"code": "X", "message": "x"}]), "WARN")
        self.assertEqual(rf.compute_status([{"code": "X", "message": "x"}], []), "BLOCKED")
        self.assertEqual(rf.gate_state("WARN"), "PASS_WITH_WARNINGS")

    def test_probe_json_parser_ignores_noise(self):
        self.assertEqual(
            rf._parse_probe("import warnings\nnoise\n" + PREFIX + '{"a": 1}\n'), {"a": 1}
        )
        self.assertIsNone(rf._parse_probe("no json here"))

    def test_interpreter_preference(self):
        fake = self.tmp / "fake_repo"
        venv_python = fake / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("#!/bin/sh\n", encoding="utf-8")
        venv_python.chmod(venv_python.stat().st_mode | stat.S_IXUSR)
        interp, source = rf.pick_interpreter(fake)
        self.assertEqual(interp, str(venv_python))
        self.assertEqual(source, "project-venv")
        interp2, source2 = rf.pick_interpreter(fake, "/opt/python")
        self.assertEqual((interp2, source2), ("/opt/python", "explicit"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
