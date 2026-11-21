#!/usr/bin/env python3
"""H-00 read-only Git safeguard and baseline capture for RiichiFormer V3.1.

The tool never cleans, resets, stashes, checks out or rewrites the source
repository.  It only reads Git state and copies user-owned dirty/untracked files
into an isolated backup directory (by default under the already-ignored
``artifacts/riichiformer/git_safeguard/``).

Commands
--------
snapshot
    Capture remotes/branches/HEAD/status/submodule/vendor state, ``.gitignore``
    checks, a content-addressed copy of dirty + untracked files, a full
    ``git bundle`` and a ``manifest.json`` with SHA-256 for every byte written.
verify
    Re-hash a snapshot and fail if any recorded file or the bundle changed.
ignored-check
    Assert first-party code is not silently ignored while data/artefacts are.

Exit code 0 means every requested check ran and passed.  A failing check exits
non-zero with a readable diagnostic; nothing is repaired automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = REPO_ROOT / "artifacts" / "riichiformer" / "git_safeguard"
SENSITIVE_PARTS = (
    ".env",
    "player_map",
    "player_hmac",
    ".salt",
    ".key",
    "private/",
)
MAX_COPY_BYTES = 20 * 1024 * 1024


class SafeguardError(RuntimeError):
    """A requested check failed; the caller must stop and inspect the state."""


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "LC_ALL": "C.UTF-8",
    })
    return env


def git(repo: Path, *args: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=_env(),
    )
    if check and proc.returncode != 0:
        raise SafeguardError(
            f"git {' '.join(args)} failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc


def _text(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout.decode("utf-8", "replace")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_z(raw: bytes) -> list[str]:
    return [part.decode("utf-8", "surrogateescape") for part in raw.split(b"\0") if part]


def _is_sensitive(rel: str) -> bool:
    low = rel.lower()
    return any(part in low for part in SENSITIVE_PARTS)


def _safe_relative(rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        raise SafeguardError(f"unsafe path returned by git: {rel!r}")
    return path


def ancestry(repo: Path, other: str) -> dict[str, Any]:
    """Return merge-base facts without assuming the object is an ancestor."""

    head = _text(git(repo, "rev-parse", "HEAD")).strip()
    mb = git(repo, "merge-base", other, "HEAD", check=False)
    return {
        "head": head,
        "merge_base": _text(mb).strip() if mb.returncode == 0 else None,
        "is_ancestor": git(repo, "merge-base", "--is-ancestor", other, "HEAD", check=False).returncode == 0,
        "other_is_ancestor": git(repo, "merge-base", "--is-ancestor", "HEAD", other, check=False).returncode == 0,
    }


class Snapshot:
    """A single immutable capture under one output directory."""

    def __init__(self, repo: Path, out: Path, include_remote: bool = True):
        self.repo = repo.resolve()
        self.out = out.resolve()
        self.include_remote = include_remote
        self.entries: list[dict[str, Any]] = []
        self.skipped_sensitive: list[str] = []
        self.skipped_large: list[dict[str, Any]] = []

    # -- helpers ---------------------------------------------------------
    def _write_text(self, name: str, value: str) -> None:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        self.entries.append({
            "kind": "metadata",
            "source": name,
            "backup": name,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        })

    def _copy(self, src: Path, rel: str, kind: str, backup_prefix: str) -> None:
        if _is_sensitive(rel):
            self.skipped_sensitive.append(rel)
            return
        size = src.stat().st_size
        if size > MAX_COPY_BYTES:
            self.skipped_large.append({"source": rel, "bytes": size, "sha256": _sha256_file(src)})
            return
        dest = self.out / "files" / backup_prefix / _safe_relative(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        self.entries.append({
            "kind": kind,
            "source": rel,
            "backup": str(dest.relative_to(self.out).as_posix()),
            "sha256": _sha256_file(dest),
            "bytes": dest.stat().st_size,
        })

    def _copy_paths(self, repo: Path, rels: Iterable[str], kind: str, prefix: str) -> None:
        for rel in sorted(set(rels)):
            src = repo / rel
            if src.is_file():
                self._copy(src, rel, kind, prefix)

    # -- capture ---------------------------------------------------------
    def capture(self) -> dict[str, Any]:
        if self.out == self.repo or self.repo in self.out.parents:
            # Allowed: artefacts are ignored.  Still refuse writing inside .git.
            pass
        if ".git" in self.out.parts:
            raise SafeguardError("refusing to write a snapshot inside .git")
        if self.out.exists() and any(self.out.iterdir()):
            raise SafeguardError(f"output directory is not empty: {self.out}")

        facts: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repo": str(self.repo),
            "git_version": _text(git(self.repo, "--version")).strip(),
        }

        facts["head"] = _text(git(self.repo, "rev-parse", "HEAD")).strip()
        facts["branch"] = _text(git(self.repo, "rev-parse", "--abbrev-ref", "HEAD")).strip()
        facts["remotes"] = _text(git(self.repo, "remote", "-v")).strip()
        facts["branches"] = _text(git(self.repo, "branch", "-avv")).strip()
        status_proc = git(self.repo, "status", "--porcelain=v1", check=False)
        facts["status_porcelain"] = _text(status_proc).strip()
        facts["status_exit_code"] = status_proc.returncode
        facts["ls_tree_vendor"] = _text(
            git(self.repo, "ls-tree", "HEAD", "vendor/mortal", check=False)
        ).strip()
        facts["submodule_status"] = _text(
            git(self.repo, "submodule", "status", "--recursive", check=False)
        ).strip()
        facts["root_gitignore"] = (self.repo / ".gitignore").read_text(
            encoding="utf-8", errors="replace"
        )

        # Public remote main may be a filtered history unrelated to the local
        # development branch.  Never fetch/merge: just record what the remote
        # advertises and whether the local objects can prove ancestry.
        remote_main_sha = None
        if self.include_remote:
            rem = git(self.repo, "ls-remote", "origin", "refs/heads/main", check=False, timeout=30)
            if rem.returncode == 0 and rem.stdout.strip():
                remote_main_sha = _text(rem).split()[0]
            else:
                facts["remote_main_error"] = rem.stderr.decode("utf-8", "replace").strip()
        facts["remote_main_sha"] = remote_main_sha
        if remote_main_sha:
            facts["ancestry_with_remote_main"] = ancestry(self.repo, remote_main_sha)
            facts["merge_base_with_remote_main"] = facts["ancestry_with_remote_main"]["merge_base"]
            facts["remote_main_is_ancestor"] = facts["ancestry_with_remote_main"]["is_ancestor"]
        else:
            facts["ancestry_with_remote_main"] = None
            facts["merge_base_with_remote_main"] = None
            facts["remote_main_is_ancestor"] = None

        local_main = git(self.repo, "rev-parse", "--verify", "main", check=False)
        facts["local_main_sha"] = _text(local_main).strip() if local_main.returncode == 0 else None

        # Dirty tracked files (worktree + index) and untracked files.
        dirty = _split_z(git(self.repo, "diff", "--name-only", "-z", check=False).stdout)
        staged = _split_z(git(self.repo, "diff", "--cached", "--name-only", "-z", check=False).stdout)
        untracked = _split_z(
            git(self.repo, "ls-files", "--others", "--exclude-standard", "-z", check=False).stdout
        )
        # ``git diff`` omits submodule working-tree contents; vendor is handled
        # separately below so a dirty submodule is never mistaken for clean.
        facts["dirty_tracked"] = sorted(set(dirty) | set(staged))
        facts["untracked"] = sorted(set(untracked))

        facts["worktree_diff"] = _text(git(self.repo, "diff", "--no-ext-diff", check=False))
        facts["worktree_diff_cached"] = _text(
            git(self.repo, "diff", "--cached", "--no-ext-diff", check=False)
        )

        self._copy_paths(self.repo, dirty + staged, "dirty_tracked", "worktree")
        self._copy_paths(self.repo, untracked, "untracked", "worktree")

        # Vendor submodule: record patched/untracked state verbatim.
        vendor = self.repo / "vendor" / "mortal"
        facts["vendor"] = {"path": str(vendor), "present": vendor.is_dir()}
        if vendor.is_dir() and (vendor / ".git").exists():
            vstatus = git(vendor, "status", "--porcelain=v1", check=False)
            vdirty = _split_z(git(vendor, "diff", "--name-only", "-z", check=False).stdout)
            vstaged = _split_z(git(vendor, "diff", "--cached", "--name-only", "-z", check=False).stdout)
            vuntracked = _split_z(
                git(vendor, "ls-files", "--others", "--exclude-standard", "-z", check=False).stdout
            )
            vdiff = _text(git(vendor, "diff", "--no-ext-diff", check=False))
            vdiff_cached = _text(git(vendor, "diff", "--cached", "--no-ext-diff", check=False))
            facts["vendor"].update({
                "head": _text(git(vendor, "rev-parse", "HEAD", check=False)).strip(),
                "status_porcelain": _text(vstatus).strip(),
                "dirty_tracked": sorted(set(vdirty) | set(vstaged)),
                "untracked": sorted(set(vuntracked)),
            })
            self._write_text("vendor/worktree.diff", vdiff)
            self._write_text("vendor/worktree.diff.cached", vdiff_cached)
            self._write_text("vendor/status.txt", facts["vendor"]["status_porcelain"] + "\n")
            self._copy_paths(vendor, vdirty + vstaged, "vendor_dirty_tracked", "vendor")
            self._copy_paths(vendor, vuntracked, "vendor_untracked", "vendor")

        # Ignore semantics that historically hid first-party code.
        facts["ignore_checks"] = ignore_checks(self.repo)

        # Full local object backup (all refs; dirty files are copied above).
        bundle = self.out / "repo.bundle"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        proc = git(self.repo, "bundle", "create", str(bundle), "--all", check=False)
        if proc.returncode != 0:
            raise SafeguardError(
                "git bundle create failed: " + proc.stderr.decode("utf-8", "replace").strip()
            )
        verify = git(self.repo, "bundle", "verify", str(bundle), check=False)
        facts["bundle"] = {
            "path": "repo.bundle",
            "sha256": _sha256_file(bundle),
            "bytes": bundle.stat().st_size,
            "verify_exit_code": verify.returncode,
            "verify_output": _text(verify).strip() or verify.stderr.decode("utf-8", "replace").strip(),
        }
        self.entries.append({
            "kind": "git_bundle",
            "source": "(all refs)",
            "backup": "repo.bundle",
            "sha256": facts["bundle"]["sha256"],
            "bytes": facts["bundle"]["bytes"],
        })

        facts["skipped_sensitive"] = sorted(self.skipped_sensitive)
        facts["skipped_large"] = self.skipped_large
        facts["counts"] = {
            "dirty_tracked": len(facts["dirty_tracked"]),
            "untracked": len(facts["untracked"]),
            "vendor_dirty_tracked": len(facts["vendor"].get("dirty_tracked", [])),
            "vendor_untracked": len(facts["vendor"].get("untracked", [])),
            "copied_files": sum(1 for e in self.entries if e["kind"] != "metadata"),
        }

        self._write_text("baseline.json", json.dumps(facts, indent=2, sort_keys=True) + "\n")
        self._write_text("status_porcelain.txt", facts["status_porcelain"] + "\n")
        self._write_text("untracked_files.txt", "\n".join(facts["untracked"]) + ("\n" if facts["untracked"] else ""))
        self._write_text("dirty_tracked_files.txt", "\n".join(facts["dirty_tracked"]) + ("\n" if facts["dirty_tracked"] else ""))

        manifest = {
            "format": 1,
            "generated_at": facts["generated_at"],
            "repo": facts["repo"],
            "head": facts["head"],
            "branch": facts["branch"],
            "entries": sorted(self.entries, key=lambda item: item["backup"]),
            "bundle_sha256": facts["bundle"]["sha256"],
        }
        (self.out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # ``manifest.json`` is not self-referential; its hash is printed and
        # recorded by the caller, not inside itself.
        return {"facts": facts, "manifest": manifest, "manifest_sha256": _sha256_file(self.out / "manifest.json")}


def ignore_checks(repo: Path) -> dict[str, dict[str, Any]]:
    """Evaluate ignore rules for first-party code vs data/artefacts.

    ``git check-ignore`` exits 0 when a path *is* ignored, 1 when it is not and
    128 on error.  The historical ``data/`` pattern ignored
    ``riichiformer/data/`` and made a whole package disappear from commits.
    """

    probes = {
        "riichiformer/data/replay.py": False,
        "riichiformer/schemas/provenance.py": False,
        "riichiformer/models/local_transformer.py": False,
        "artifacts/riichiformer/probe": True,
        "data/probe": True,
        "datasets/probe": True,
        "worktrees/probe": True,
    }
    result: dict[str, dict[str, Any]] = {}
    for rel, expected in probes.items():
        proc = git(repo, "check-ignore", "--no-index", "-v", "--", rel, check=False)
        ignored = proc.returncode == 0
        result[rel] = {
            "expected_ignored": expected,
            "actual_ignored": ignored,
            "rule": _text(proc).strip() or None,
            "ok": ignored == expected,
        }
    return result


def verify_snapshot(out: Path) -> dict[str, Any]:
    out = out.resolve()
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        raise SafeguardError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked = 0
    for entry in manifest.get("entries", []):
        path = out / entry["backup"]
        if not path.is_file():
            failures.append(f"missing: {entry['backup']}")
            continue
        digest = _sha256_file(path)
        checked += 1
        if digest != entry["sha256"]:
            failures.append(f"sha256 mismatch: {entry['backup']} ({digest} != {entry['sha256']})")
        if path.stat().st_size != entry.get("bytes"):
            failures.append(f"size mismatch: {entry['backup']}")
    bundle = out / "repo.bundle"
    if not bundle.is_file():
        failures.append("missing: repo.bundle")
    elif _sha256_file(bundle) != manifest.get("bundle_sha256"):
        failures.append("sha256 mismatch: repo.bundle")
    result = {
        "ok": not failures,
        "checked": checked,
        "failures": failures,
        "manifest_sha256": _sha256_file(manifest_path),
    }
    if failures:
        raise SafeguardError("snapshot verification failed:\n" + "\n".join(failures))
    return result


def _default_out() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUT_ROOT / stamp


def cmd_snapshot(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    out = Path(args.out).resolve() if args.out else _default_out()
    snap = Snapshot(repo, out, include_remote=not args.no_remote)
    result = snap.capture()
    verification = verify_snapshot(out)
    facts = result["facts"]
    print(json.dumps({
        "ok": True,
        "out": str(out),
        "head": facts["head"],
        "branch": facts["branch"],
        "remote_main_sha": facts.get("remote_main_sha"),
        "merge_base_with_remote_main": facts.get("merge_base_with_remote_main"),
        "counts": facts["counts"],
        "bundle_sha256": facts["bundle"]["sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "verify": verification,
        "ignore_checks_ok": all(item["ok"] for item in facts["ignore_checks"].values()),
    }, indent=2))
    if not all(item["ok"] for item in facts["ignore_checks"].values()):
        print("ERROR: ignore-rule expectation failed", file=sys.stderr)
        return 3
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result = verify_snapshot(Path(args.snapshot))
    print(json.dumps(result, indent=2))
    return 0


def cmd_ignored_check(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    checks = ignore_checks(repo)
    failed = {key: value for key, value in checks.items() if not value["ok"]}
    print(json.dumps({"ok": not failed, "checks": checks, "failed": failed}, indent=2))
    return 0 if not failed else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_snap = sub.add_parser("snapshot", help="capture and verify an immutable backup")
    p_snap.add_argument("--repo", default=str(REPO_ROOT))
    p_snap.add_argument("--out", default=None, help="exact output directory (must not exist)")
    p_snap.add_argument("--no-remote", action="store_true", help="skip git ls-remote (offline)")
    p_snap.set_defaults(func=cmd_snapshot)

    p_verify = sub.add_parser("verify", help="re-hash an existing snapshot")
    p_verify.add_argument("snapshot")
    p_verify.set_defaults(func=cmd_verify)

    p_ign = sub.add_parser("ignored-check", help="assert first-party code is not ignored")
    p_ign.add_argument("--repo", default=str(REPO_ROOT))
    p_ign.set_defaults(func=cmd_ignored_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SafeguardError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
