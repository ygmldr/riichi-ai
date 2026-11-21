#!/usr/bin/env python3
"""RiichiFormer V2 PR-00: read-only baseline freeze & runtime preflight.

What this script does (and only does):
  1. Records the real git state (HEAD, branch, remote, tracked/untracked dirt).
  2. Records the ``vendor/mortal`` submodule SHA, dirty entries and the number of
     ``[patch NNNN]`` markers actually present in the checked-out tree.
  3. Hashes every frozen baseline artefact (old config, eval_report, patches,
     README/RUNBOOK/NOTICE/LICENSE, checkpoints) with SHA-256 + size + mtime.
  4. Probes Python / PyTorch / CUDA / GPU / Rust / libriichi **in a subprocess**
     so the report is honest even when the caller is a bare system Python.
  5. Optionally loads each checkpoint on CPU to prove it is actually readable
     (keys, config flags such as ``use_event_branch``, tensor/param counts).
  6. Emits one JSON artefact under ``artifacts/riichiformer/``.

Hard safety rules (also encoded in the tests):
  * read-only: the only file written is ``--output`` (plus its parent directory);
  * never downloads, never installs, never starts training, never touches the
    network, never creates a fake ``baseline.pth``/``mortal.pth``;
  * a missing artefact is reported as ``NOT_FOUND`` / ``UNAVAILABLE`` and turns
    into a warning or a blocker -- it is never silently invented;
  * exit code is 0 when there is no blocker (warnings may still be present),
    2 when blocked, 3 on internal error. ``--strict`` also fails on warnings.

Status vocabulary:
  PASS     every check is clean;
  WARN     non-blocking issue (dirty worktree, missing optional checkpoint,
           no CUDA/Rust toolchain on this host, unreadable optional weight ...);
  BLOCKED  a required baseline file / the git repo / the submodule is missing,
           so G0 cannot be attested at all.

Gate mapping: ``gate.G0`` is ``PASS`` / ``PASS_WITH_WARNINGS`` / ``BLOCKED``.

Usage:
  python3 scripts/rf_preflight.py --output artifacts/riichiformer/preflight.json
  python3 scripts/rf_preflight.py --output -            # JSON to stdout
  python3 scripts/rf_preflight.py --skip-checkpoint-load
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA_VERSION = "rf-preflight/1"
DEFAULT_OUTPUT = "artifacts/riichiformer/preflight.json"

# Files whose absence blocks G0 (the freeze itself cannot be described).
REQUIRED_FILES = (
    "configs/config.toml",
    "scripts/eval_report.py",
    ".gitmodules",
)
# Files that should be recorded but whose absence is only a warning.
OPTIONAL_FILES = (
    "README.md",
    "RUNBOOK.md",
    "NOTICE.md",
    "LICENSE",
    "docs/EXPERIMENTS.md",
    "docs/exp-gates.md",
    "docs/FORK_ROADMAP.md",
    "docs/riichiformer/BASELINE_LOCK.md",
    "docs/riichiformer/EVAL_POOL_REGISTRY.md",
)
# Checkpoints whose hashes must be recorded (per PR-00 card).
KEY_CHECKPOINTS = (
    "checkpoints/mortal.pth",
    "checkpoints/best.pth",
    "checkpoints/baseline.pth",
    "checkpoints/grp.pth",
)
PATCH_GAP = (11,)  # patches/0011 is documented as non-existent; not a warning.

MAX_WARNING_SAMPLES = 12


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_from_ts(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256; never loads the whole file into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def run_cmd(args, cwd=None, timeout=60, strip=True):
    """Run a command read-only and return a JSON-safe record.

    ``strip=False`` is required for ``git status --porcelain``: the first
    character of the first line is a status column, not whitespace to trim.
    """
    rec = {"cmd": list(args), "returncode": None, "stdout": "", "stderr": "", "ok": False}
    try:
        proc = subprocess.run(
            [str(a) for a in args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        rec["returncode"] = proc.returncode
        rec["stdout"] = proc.stdout.strip() if strip else proc.stdout
        rec["stderr"] = proc.stderr.strip()
        rec["ok"] = proc.returncode == 0
    except FileNotFoundError as exc:
        rec["stderr"] = f"FileNotFoundError: {exc}"
    except subprocess.TimeoutExpired as exc:
        rec["stderr"] = f"TimeoutExpired after {timeout}s: {exc}"
    except OSError as exc:  # pragma: no cover - host specific
        rec["stderr"] = f"{type(exc).__name__}: {exc}"
    return rec


def git(repo: Path, *args, timeout=60, strip=True):
    return run_cmd(["git", *args], cwd=repo, timeout=timeout, strip=strip)


def _truncate(text: str, limit: int = 400) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "..."

def file_entry(repo: Path, rel: str, kind: str, required: bool, note: str | None = None):
    """Record one frozen file. Missing files are *reported*, never created."""
    path = repo / rel
    entry = {
        "path": rel,
        "kind": kind,
        "required": bool(required),
        "exists": path.is_file(),
        "bytes": None,
        "sha256": None,
        "mtime_utc": None,
        "status": "NOT_FOUND",
    }
    if note:
        entry["note"] = note
    if not path.is_file():
        return entry
    try:
        st = path.stat()
        entry["bytes"] = st.st_size
        entry["mtime_utc"] = iso_from_ts(st.st_mtime)
        entry["sha256"] = sha256_file(path)
        entry["status"] = "OK"
    except OSError as exc:  # pragma: no cover - permission issues
        entry["status"] = "UNREADABLE"
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


# --------------------------------------------------------------------------- #
# git / submodule / patches
# --------------------------------------------------------------------------- #
def collect_git(repo: Path):
    out = {
        "repo_root": str(repo),
        "is_git_repo": False,
        "head": None,
        "branch": None,
        "remote": None,
        "dirty": False,
        "tracked_dirty": False,
        "tracked_dirty_entries": [],
        "untracked_entries": [],
        "porcelain": "",
        "commands": {},
    }
    head = git(repo, "rev-parse", "HEAD")
    out["commands"]["rev-parse"] = {k: head[k] for k in ("cmd", "returncode", "ok", "stderr")}
    if not head["ok"]:
        return out
    out["is_git_repo"] = True
    out["head"] = head["stdout"].splitlines()[0].strip()
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    out["branch"] = branch["stdout"].strip() or None
    remote = git(repo, "config", "--get", "remote.origin.url")
    out["remote"] = remote["stdout"].strip() or None
    st = git(repo, "status", "--porcelain", "--untracked-files=normal", strip=False)
    out["commands"]["status"] = {k: st[k] for k in ("cmd", "returncode", "ok", "stderr")}
    lines = [ln for ln in st["stdout"].splitlines() if ln.strip()]
    out["porcelain"] = st["stdout"]
    tracked, untracked = [], []
    for line in lines:
        code, _, rest = line[:2], line[2:3], line[3:]
        if code == "??":
            untracked.append(rest)
        else:
            tracked.append({"status": code, "path": rest})
    out["dirty"] = bool(lines)
    out["tracked_dirty"] = bool(tracked)
    out["tracked_dirty_entries"] = tracked[:MAX_WARNING_SAMPLES]
    out["untracked_entries"] = untracked[:MAX_WARNING_SAMPLES]
    out["untracked_count"] = len(untracked)
    out["tracked_dirty_count"] = len(tracked)
    return out


def collect_submodule(repo: Path):
    out = {
        "path": "vendor/mortal",
        "declared": False,
        "index_sha": None,
        "head_sha": None,
        "dirty": False,
        "dirty_entries": [],
        "patch_marker_count": 0,
        "patch_marker_files": [],
        "commands": {},
    }
    gm = repo / ".gitmodules"
    if gm.is_file():
        try:
            text = gm.read_text(encoding="utf-8", errors="replace")
            out["declared"] = "vendor/mortal" in text
        except OSError:  # pragma: no cover
            pass
    tree = git(repo, "ls-tree", "HEAD", "vendor/mortal")
    out["commands"]["ls-tree"] = {k: tree[k] for k in ("cmd", "returncode", "ok", "stderr")}
    if tree["ok"] and tree["stdout"]:
        parts = tree["stdout"].split()
        if len(parts) >= 3:
            out["index_sha"] = parts[2]
    sub = repo / "vendor" / "mortal"
    head = git(sub, "rev-parse", "HEAD")
    out["commands"]["submodule-head"] = {
        k: head[k] for k in ("cmd", "returncode", "ok", "stderr")
    }
    if head["ok"]:
        out["head_sha"] = head["stdout"].splitlines()[0].strip()
    st = git(sub, "status", "--porcelain")
    out["commands"]["submodule-status"] = {k: st[k] for k in ("cmd", "returncode", "ok", "stderr")}
    if st["ok"]:
        entries = [ln for ln in st["stdout"].splitlines() if ln.strip()]
        out["dirty"] = bool(entries)
        out["dirty_entries"] = entries[:MAX_WARNING_SAMPLES]
        out["dirty_count"] = len(entries)
    # Count in-code [patch NNNN] markers; skip target/ and .git/ (build noise).
    try:
        for base in ("mortal", "libriichi/src"):
            for p in (sub / base).rglob("*"):
                if not p.is_file() or p.suffix not in (".py", ".rs"):
                    continue
                if "target" in p.parts:
                    continue
                try:
                    body = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                n = body.count("[patch ")
                if n:
                    out["patch_marker_count"] += n
                    out["patch_marker_files"].append(
                        {"path": str(p.relative_to(repo)), "markers": n}
                    )
    except OSError:  # pragma: no cover
        pass
    out["patch_marker_files"] = sorted(
        out["patch_marker_files"], key=lambda item: item["path"]
    )[:40]
    return out


def collect_patches(repo: Path):
    pdir = repo / "patches"
    files = sorted(pdir.glob("*.patch")) if pdir.is_dir() else []
    entries = []
    numbers = []
    for f in files:
        entry = file_entry(repo, str(f.relative_to(repo)), "patch", required=False)
        entries.append(entry)
        digits = "".join(ch for ch in f.name[:4] if ch.isdigit())
        if len(digits) == 4:
            numbers.append(int(digits))
    expected = list(range(1, max(numbers + [13]) + 1)) if numbers else []
    missing = [n for n in expected if n not in numbers]
    return {
        "dir": "patches",
        "present": bool(files),
        "count": len(entries),
        "numbers": numbers,
        "missing_numbers": missing,
        "documented_gaps": [n for n in PATCH_GAP if n in missing],
        "files": entries,
    }


# --------------------------------------------------------------------------- #
# runtime probes (subprocess, so the caller interpreter does not matter)
# --------------------------------------------------------------------------- #
_TORCH_PROBE = r"""
import json, sys
out = {
    "executable": sys.executable,
    "python_version": sys.version.split()[0],
    "python_implementation": sys.implementation.name,
    "torch": None,
}
try:
    import torch
except Exception as exc:  # noqa: BLE001 - report, never crash
    out["torch_import_error"] = type(exc).__name__ + ": " + str(exc)[:300]
else:
    info = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        info["cudnn_version"] = None
    if info["cuda_available"]:
        try:
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "major": int(props.major),
                "minor": int(props.minor),
                "multi_processor_count": int(props.multi_processor_count),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        except Exception as exc:
            info["gpu_error"] = type(exc).__name__ + ": " + str(exc)[:300]
    out["torch"] = info
print("RF_PROBE_JSON " + json.dumps(out))
"""

_LIBRIICHI_PROBE = r"""
import json, sys
release_dir = sys.argv[1]
sys.path.insert(0, release_dir)
out = {"ok": False, "release_dir": release_dir}
try:
    import libriichi
    out["ok"] = True
    out["module_file"] = getattr(libriichi, "__file__", None)
    consts = getattr(libriichi, "consts", None)
    if consts is not None:
        for name in ("ACTION_SPACE", "OBS_SHAPE", "N_PLAYERS"):
            if hasattr(consts, name):
                value = getattr(consts, name)
                out[name] = list(value) if isinstance(value, (tuple, list)) else value
except Exception as exc:  # noqa: BLE001
    out["error"] = type(exc).__name__ + ": " + str(exc)[:400]
print("RF_PROBE_JSON " + json.dumps(out))
"""

_CHECKPOINT_PROBE = r"""
import gc, json, sys
import torch

def describe(obj):
    rec = {"readable": True, "top_level_type": type(obj).__name__, "keys": [],
           "config": None, "state_dict_key": None, "tensor_count": 0,
           "param_count": 0, "has_optimizer": False, "use_event_branch": None,
           "state_dict_has_event_weights": None, "error": None}
    if isinstance(obj, dict):
        rec["keys"] = [str(k) for k in list(obj.keys())[:25]]
        cfg = obj.get("config")
        if isinstance(cfg, dict):
            model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
            ctrl = cfg.get("control") if isinstance(cfg.get("control"), dict) else {}
            rec["config_control"] = {
                k: ctrl.get(k) for k in ("version", "online", "batch_size", "enable_amp") if k in ctrl
            }
            rec["config_model"] = {
                k: model_cfg[k]
                for k in ("use_event_branch", "d_model", "n_blocks", "num_blocks", "conv_channels")
                if k in model_cfg
            }
            if cfg.get("resnet") is not None:
                rec["config_resnet"] = cfg["resnet"]
            if "use_event_branch" in model_cfg:
                rec["use_event_branch"] = bool(model_cfg["use_event_branch"])
            if "d_model" in model_cfg:
                rec["config_d_model"] = model_cfg.get("d_model")
            if "n_blocks" in model_cfg:
                rec["config_n_blocks"] = model_cfg.get("n_blocks")
            if "num_blocks" in model_cfg:
                rec["config_n_blocks"] = model_cfg.get("num_blocks")
        rec["has_optimizer"] = any(k in obj for k in ("optimizer", "optim"))
        rec["steps"] = obj.get("steps") if isinstance(obj.get("steps"), int) else None
        rec["timestamp"] = obj.get("timestamp") if isinstance(obj.get("timestamp"), (int, float)) else None
        rec["tag"] = obj.get("tag") if isinstance(obj.get("tag"), str) else None
        sd = None
        # Mortal checkpoints use `mortal`/`current_dqn`/`aux_net`; older or
        # grafted checkpoints may use `model`/`state_dict`/`policy`/`q_net`.
        for key in ("mortal", "model", "state_dict", "policy", "q_net", "current_dqn"):
            candidate = obj.get(key)
            if isinstance(candidate, dict) and candidate and all(
                torch.is_tensor(v) for v in list(candidate.values())[:8]
            ):
                sd = candidate
                rec["state_dict_key"] = key
                break
            if hasattr(candidate, "state_dict"):
                sd = candidate.state_dict()
                rec["state_dict_key"] = key + ".state_dict()"
                break
        if sd is None and obj and all(torch.is_tensor(v) for v in list(obj.values())[:8]):
            sd = obj
            rec["state_dict_key"] = "<root>"
        if sd is not None:
            rec["tensor_count"] = len(sd)
            rec["param_count"] = int(sum(v.numel() for v in sd.values() if torch.is_tensor(v)))
            rec["state_dict_keys_sample"] = [str(k) for k in list(sd)[:12]]
            rec["state_dict_has_event_weights"] = any("event" in str(k).lower() for k in sd)
        aux = obj.get("aux_net")
        if isinstance(aux, dict) and aux:
            rec["aux_net_tensor_count"] = len(aux)
            rec["aux_net_has_event_weights"] = any("event" in str(k).lower() for k in aux)
    elif hasattr(obj, "state_dict"):
        rec["state_dict_key"] = "<module>"
        sd = obj.state_dict()
        rec["tensor_count"] = len(sd)
        rec["param_count"] = int(sum(v.numel() for v in sd.values()))
        rec["state_dict_has_event_weights"] = any("event" in str(k).lower() for k in sd)
    return rec

result = {}
for path in sys.argv[1:]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        result[path] = {"readable": False, "error": type(exc).__name__ + ": " + str(exc)[:400]}
        continue
    try:
        result[path] = describe(obj)
    except Exception as exc:  # noqa: BLE001
        result[path] = {"readable": True, "error": "describe failed: " + str(exc)[:300]}
    finally:
        del obj
        gc.collect()
print("RF_PROBE_JSON " + json.dumps(result))
"""


def _parse_probe(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("RF_PROBE_JSON "):
            try:
                return json.loads(line[len("RF_PROBE_JSON ") :])
            except json.JSONDecodeError:  # pragma: no cover
                return None
    return None


def pick_interpreter(repo: Path, explicit: str | None = None) -> tuple[str, str]:
    """Prefer an explicit interpreter, then the project venv, then the caller."""
    if explicit:
        return explicit, "explicit"
    venv = repo / ".venv" / "bin" / "python"
    if venv.is_file() and os.access(venv, os.X_OK):
        return str(venv), "project-venv"
    return sys.executable, "caller"


def collect_runtime(interpreter: str, timeout: int = 180):
    proc = run_cmd([interpreter, "-c", _TORCH_PROBE], timeout=timeout)
    parsed = _parse_probe(proc["stdout"])
    if parsed is None:
        return {
            "probe_ok": False,
            "interpreter": interpreter,
            "error": _truncate(proc["stderr"] or proc["stdout"] or "probe produced no JSON"),
            "torch": None,
        }
    parsed["probe_ok"] = True
    return parsed


def collect_libriichi(repo: Path, interpreter: str, timeout: int = 120):
    release_dir = repo / "vendor" / "mortal" / "target" / "release"
    rec = {
        "release_dir": "vendor/mortal/target/release",
        "so_path": "vendor/mortal/target/release/libriichi.so",
        "so_exists": (release_dir / "libriichi.so").is_file(),
        "so_bytes": None,
        "so_sha256": None,
        "importable": None,
        "details": None,
        "error": None,
    }
    so = release_dir / "libriichi.so"
    if rec["so_exists"]:
        try:
            rec["so_bytes"] = so.stat().st_size
            rec["so_sha256"] = sha256_file(so)
        except OSError as exc:  # pragma: no cover
            rec["error"] = f"{type(exc).__name__}: {exc}"
    if not rec["so_exists"]:
        return rec
    proc = run_cmd([interpreter, "-c", _LIBRIICHI_PROBE, str(release_dir)], timeout=timeout)
    parsed = _parse_probe(proc["stdout"])
    if parsed is None:
        rec["importable"] = False
        rec["error"] = _truncate(proc["stderr"] or "probe produced no JSON")
        return rec
    rec["importable"] = bool(parsed.get("ok"))
    rec["details"] = parsed
    rec["error"] = parsed.get("error")
    return rec


def collect_toolchains(repo: Path):
    out = {"rustc": None, "cargo": None, "searched": [], "found": False}
    candidates = []
    for name in ("rustc", "cargo"):
        how = shutil.which(name)
        if how:
            candidates.append((name, how))
        home = Path.home() / ".cargo" / "bin" / name
        if home.is_file():
            candidates.append((name, str(home)))
    out["searched"] = sorted({p for _, p in candidates})
    versions = {}
    for name, path in candidates:
        if name in versions:
            continue
        proc = run_cmd([path, "--version"], timeout=20)
        versions[name] = {
            "path": path,
            "version": proc["stdout"] or None,
            "ok": proc["ok"],
            "error": None if proc["ok"] else _truncate(proc["stderr"]),
        }
    out["rustc"] = versions.get("rustc")
    out["cargo"] = versions.get("cargo")
    out["found"] = bool(out["rustc"] or out["cargo"])
    return out


def collect_nvidia_smi():
    """nvidia-smi is often absent from PATH inside WSL but exists under /usr/lib/wsl."""
    candidates = []
    how = shutil.which("nvidia-smi")
    if how:
        candidates.append(how)
    candidates.append("/usr/lib/wsl/lib/nvidia-smi")
    rec = {"available": False, "path": None, "gpus": [], "driver_version": None, "error": None}
    for candidate in candidates:
        if not Path(candidate).is_file():
            continue
        proc = run_cmd(
            [
                candidate,
                "--query-gpu=name,memory.total,memory.used,driver_version,uuid,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            timeout=20,
        )
        if not proc["ok"]:
            rec["error"] = _truncate(proc["stderr"])
            continue
        rows = []
        for line in proc["stdout"].splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 6:
                continue
            rows.append(
                {
                    "name": cols[0],
                    "memory_total_mib": _maybe_int(cols[1]),
                    "memory_used_mib": _maybe_int(cols[2]),
                    "driver_version": cols[3],
                    "uuid": cols[4],
                    "compute_cap": cols[5],
                }
            )
        rec.update({"available": bool(rows), "path": candidate, "gpus": rows})
        if rows:
            rec["driver_version"] = rows[0]["driver_version"]
            break
    return rec


def _maybe_int(text: str):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def collect_environment(repo: Path):
    env = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "wsl": "microsoft" in platform.release().lower(),
        "disk": None,
    }
    try:
        usage = shutil.disk_usage(repo)
        env["disk"] = {
            "mount": str(repo),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
            "free_gib": round(usage.free / (1 << 30), 1),
        }
    except OSError:  # pragma: no cover
        pass
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    env["ram_total_gib"] = round(int(line.split()[1]) / (1 << 20), 1)
                    break
    except (OSError, ValueError, IndexError):  # pragma: no cover
        pass
    return env


def collect_datasets(repo: Path):
    """Report presence only -- never recurse into multi-GB corpus directories."""
    rec = {"path": "datasets", "exists": False, "top_level_entries": 0, "note": ""}
    d = repo / "datasets"
    if d.is_dir():
        try:
            rec["exists"] = True
            rec["top_level_entries"] = len(list(d.iterdir()))
            rec["note"] = "presence only; contents are audited in PR-02, not here"
        except OSError as exc:  # pragma: no cover
            rec["note"] = f"unreadable: {type(exc).__name__}"
    return rec


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def collect_checkpoints(repo: Path, interpreter: str, load: bool = True, timeout: int = 900):
    cdir = repo / "checkpoints"
    entries = []
    keys = list(KEY_CHECKPOINTS)
    if cdir.is_dir():
        for p in sorted(cdir.glob("mortal.pth.at-*")):
            keys.append(str(p.relative_to(repo)))
        for p in sorted(cdir.glob("*.pth")):
            rel = str(p.relative_to(repo))
            if rel not in keys and "file_index" in p.name:
                keys.append(rel)
    loaded = {}
    loadable_rels = []
    for rel in keys:
        exists = (repo / rel).is_file()
        kind = "index" if "file_index" in rel else "checkpoint"
        entry = file_entry(repo, rel, kind, required=False)
        if exists and kind == "checkpoint" and load:
            loadable_rels.append(rel)
        entries.append(entry)
    if cdir.is_dir():
        for p in sorted(cdir.iterdir()):
            rel = str(p.relative_to(repo))
            if p.is_file() and p.suffix != ".pth":
                entries.append(file_entry(repo, rel, "checkpoint-aux", required=False))
    if loadable_rels:
        proc = run_cmd(
            [interpreter, "-c", _CHECKPOINT_PROBE, *[str(repo / r) for r in loadable_rels]],
            cwd=repo,
            timeout=timeout,
        )
        parsed = _parse_probe(proc["stdout"])
        if parsed is None:
            loaded = {
                rel: {
                    "readable": None,
                    "error": "checkpoint probe failed: "
                    + _truncate(proc["stderr"] or "no JSON output"),
                }
                for rel in loadable_rels
            }
        else:
            for rel in loadable_rels:
                abs_path = str(repo / rel)
                loaded[rel] = parsed.get(abs_path, {"readable": None, "error": "probe omitted file"})
    return {
        "dir": "checkpoints",
        "load_performed": bool(loadable_rels),
        "load_skipped": (not load) or (not loadable_rels),
        "interpreter": interpreter,
        "entries": entries,
        "readability": loaded,
    }


# --------------------------------------------------------------------------- #
# assessment
# --------------------------------------------------------------------------- #
def compute_status(blockers, warnings) -> str:
    if blockers:
        return "BLOCKED"
    if warnings:
        return "WARN"
    return "PASS"


def gate_state(status: str) -> str:
    return {"PASS": "PASS", "WARN": "PASS_WITH_WARNINGS", "BLOCKED": "BLOCKED"}[status]


def assess(git_info, submodule, patches, runtime, libriichi, toolchains, nvsmi,
           checkpoints, baseline_entries, load_skipped):
    blockers = []
    warnings = []
    checks = []

    def blocker(code, message):
        blockers.append({"code": code, "message": message})

    def warn(code, message):
        warnings.append({"code": code, "message": message})

    # --- repo -----------------------------------------------------------------
    if not git_info["is_git_repo"]:
        blocker("GIT_NOT_REPO", f"{git_info['repo_root']} is not a git repository / no HEAD")
        checks.append({"name": "git", "status": "FAIL", "detail": "no git HEAD"})
    else:
        checks.append(
            {
                "name": "git",
                "status": "PASS",
                "detail": f"HEAD={git_info['head']} branch={git_info['branch']}",
            }
        )
        if git_info["tracked_dirty"]:
            names = ", ".join(e["path"] for e in git_info["tracked_dirty_entries"][:5])
            warn(
                "GIT_DIRTY_TRACKED",
                f"{git_info['tracked_dirty_count']} tracked path(s) modified: {names}; "
                "recorded only -- never reset or overwrite user work",
            )
        if git_info.get("untracked_count"):
            # New PR files are expected; record but do not fail.
            checks.append(
                {
                    "name": "untracked",
                    "status": "PASS",
                    "detail": f"{git_info['untracked_count']} untracked path(s) recorded",
                }
            )

    # --- submodule ------------------------------------------------------------
    if submodule["declared"] and not submodule["head_sha"]:
        blocker(
            "SUBMODULE_MISSING",
            "vendor/mortal is declared in .gitmodules but is not initialized "
            "(run `git submodule update --init`, do not re-clone over it)",
        )
        checks.append({"name": "submodule", "status": "FAIL", "detail": "not initialized"})
    elif submodule["head_sha"]:
        state = "dirty" if submodule["dirty"] else "clean"
        checks.append(
            {
                "name": "submodule",
                "status": "PASS",
                "detail": f"{submodule['head_sha']} ({state}); "
                f"{submodule['patch_marker_count']} [patch NNNN] markers",
            }
        )
        if submodule["dirty"]:
            warn(
                "SUBMODULE_DIRTY",
                f"vendor/mortal working tree has {submodule.get('dirty_count', 0)} changed path(s) "
                f"and {submodule['patch_marker_count']} patch markers; expected when patches are "
                "applied -- freeze the exact diff in BASELINE_LOCK.md",
            )
        elif submodule["patch_marker_count"] == 0:
            warn(
                "SUBMODULE_PATCHES_UNVERIFIED",
                "submodule is clean but no [patch NNNN] markers were found; patches may not be applied",
            )
    else:
        warn("SUBMODULE_ABSENT", "no vendor/mortal checkout was found")

    # --- patches --------------------------------------------------------------
    if not patches["present"]:
        blocker("PATCHES_MISSING", "patches/*.patch is empty or missing; the fork history cannot be attested")
        checks.append({"name": "patches", "status": "FAIL", "detail": "0 patch files"})
    else:
        gaps = [n for n in patches["missing_numbers"] if n not in patches["documented_gaps"]]
        checks.append(
            {
                "name": "patches",
                "status": "PASS",
                "detail": f"{patches['count']} patch files; missing={patches['missing_numbers']}",
            }
        )
        if gaps:
            warn("PATCH_GAP", f"patches are not contiguous; unexpected gaps: {gaps}")

    # --- baseline files -------------------------------------------------------
    required_missing = [e["path"] for e in baseline_entries if e["required"] and not e["exists"]]
    if required_missing:
        blocker("BASELINE_REQUIRED_MISSING", "required baseline file(s) missing: " + ", ".join(required_missing))
        checks.append({"name": "baseline_files", "status": "FAIL", "detail": ", ".join(required_missing)})
    else:
        checks.append(
            {
                "name": "baseline_files",
                "status": "PASS",
                "detail": f"{sum(1 for e in baseline_entries if e['exists'])}/{len(baseline_entries)} present",
            }
        )
    optional_missing = [e["path"] for e in baseline_entries if (not e["required"]) and (not e["exists"])]
    if optional_missing:
        warn("BASELINE_OPTIONAL_MISSING", "optional baseline artefact(s) absent: " + ", ".join(optional_missing))

    # --- runtime --------------------------------------------------------------
    if not runtime.get("probe_ok"):
        warn("RUNTIME_PROBE_FAILED", f"runtime probe failed: {runtime.get('error')}")
        checks.append({"name": "python", "status": "WARN", "detail": "probe failed"})
    else:
        checks.append(
            {
                "name": "python",
                "status": "PASS",
                "detail": f"{runtime.get('python_version')} ({runtime.get('executable')})",
            }
        )
        torch_info = runtime.get("torch")
        if not torch_info:
            warn(
                "TORCH_UNAVAILABLE",
                f"PyTorch is not importable from {runtime.get('interpreter')}: "
                f"{runtime.get('torch_import_error')}",
            )
            checks.append({"name": "torch", "status": "WARN", "detail": "not installed"})
            checks.append({"name": "cuda", "status": "WARN", "detail": "unknown (no torch)"})
        else:
            checks.append(
                {
                    "name": "torch",
                    "status": "PASS",
                    "detail": f"torch {torch_info.get('version')} (cuda {torch_info.get('cuda_version')})",
                }
            )
            if not torch_info.get("cuda_available"):
                warn("CUDA_UNAVAILABLE", "torch imported but torch.cuda.is_available() is False")
                checks.append({"name": "cuda", "status": "WARN", "detail": "unavailable"})
            else:
                gpu = torch_info.get("gpu") or {}
                checks.append(
                    {
                        "name": "cuda",
                        "status": "PASS",
                        "detail": f"{gpu.get('name')} {round((gpu.get('total_memory_bytes') or 0) / (1 << 20))} MiB",
                    }
                )
                if gpu.get("major") is not None and (gpu["major"], gpu["minor"]) != (12, 0):
                    warn(
                        "GPU_NOT_5070TI",
                        f"reported compute capability {gpu['major']}.{gpu['minor']} on {gpu.get('name')}; "
                        "the design target is RTX 5070 Ti (sm_120)",
                    )
                if not gpu.get("bf16_supported", False):
                    warn("BF16_UNAVAILABLE", "torch.cuda.is_bf16_supported() is False")

    if runtime.get("probe_ok") and runtime.get("torch") and not nvsmi.get("available"):
        warn("NVSMI_UNAVAILABLE", "nvidia-smi could not be executed; GPU facts come from torch only")
    if nvsmi.get("available") and runtime.get("torch") and runtime["torch"].get("cuda_available"):
        torch_mib = round((runtime["torch"].get("gpu", {}).get("total_memory_bytes") or 0) / (1 << 20))
        smi_mib = nvsmi["gpus"][0].get("memory_total_mib")
        if torch_mib and smi_mib and abs(torch_mib - smi_mib) > 512:
            warn(
                "VRAM_MISMATCH",
                f"torch reports {torch_mib} MiB but nvidia-smi reports {smi_mib} MiB total VRAM",
            )

    # --- rust / libriichi -----------------------------------------------------
    if libriichi["importable"]:
        detail = "libriichi import OK"
        consts = (libriichi.get("details") or {})
        if consts.get("ACTION_SPACE") is not None:
            detail += f" ACTION_SPACE={consts['ACTION_SPACE']}"
        checks.append({"name": "libriichi", "status": "PASS", "detail": detail})
    else:
        warn(
            "LIBRIICHI_IMPORT_FAILED",
            "libriichi.so is missing or not importable: "
            + str(libriichi.get("error") or "not found"),
        )
        checks.append({"name": "libriichi", "status": "WARN", "detail": "not importable"})

    if toolchains["found"]:
        checks.append(
            {
                "name": "rust",
                "status": "PASS",
                "detail": (toolchains.get("rustc") or {}).get("version")
                or (toolchains.get("cargo") or {}).get("version")
                or "found",
            }
        )
    else:
        warn(
            "RUST_TOOLCHAIN_UNAVAILABLE",
            "rustc/cargo not found on PATH or ~/.cargo/bin; libriichi cannot be rebuilt "
            "on this host (Rust patches 0003/0006/0010 require it)",
        )
        checks.append({"name": "rust", "status": "WARN", "detail": "toolchain not found"})

    # --- checkpoints ----------------------------------------------------------
    ck_entries = {e["path"]: e for e in checkpoints["entries"]}
    missing_key = [rel for rel in KEY_CHECKPOINTS if not ck_entries.get(rel, {}).get("exists")]
    for rel in missing_key:
        warn("CHECKPOINT_MISSING", f"{rel} is NOT_FOUND (UNAVAILABLE); no fake weight is created")
    unreadable = [
        rel
        for rel, rec in checkpoints["readability"].items()
        if rec and rec.get("readable") is False
    ]
    for rel in unreadable:
        warn(
            "CHECKPOINT_UNREADABLE",
            f"{rel} could not be torch.load'ed: {checkpoints['readability'][rel].get('error')}",
        )
    unknown_load = [
        rel for rel, rec in checkpoints["readability"].items() if rec and rec.get("readable") is None
    ]
    for rel in unknown_load:
        warn(
            "CHECKPOINT_LOAD_UNKNOWN",
            f"{rel} readability is unknown: {checkpoints['readability'][rel].get('error')}",
        )
    if load_skipped:
        checks.append({"name": "checkpoints", "status": "PASS", "detail": "hashes only (load skipped)"})
    elif unreadable or unknown_load:
        checks.append(
            {"name": "checkpoints", "status": "WARN", "detail": f"unreadable: {unreadable + unknown_load}"}
        )
    else:
        checks.append(
            {"name": "checkpoints", "status": "PASS", "detail": "all present checkpoints load on CPU"}
        )
    if checkpoints["readability"]:
        for rel, rec in checkpoints["readability"].items():
            if rec.get("readable") and rec.get("use_event_branch") is not None:
                # Informational: old CNN weights must not be called "leaky" here.
                checks.append(
                    {
                        "name": f"config_flag:{rel}",
                        "status": "PASS",
                        "detail": f"use_event_branch={rec['use_event_branch']} "
                        f"event_weights={rec.get('state_dict_has_event_weights')}",
                    }
                )

    status = compute_status(blockers, warnings)
    return {
        "status": status,
        "gate": {
            "G0": gate_state(status),
            "components": {
                "repo_frozen": "PASS" if git_info["is_git_repo"] and git_info["head"] else "FAIL",
                "submodule_pinned": "PASS" if submodule.get("head_sha") else "FAIL",
                "baseline_files": "FAIL" if required_missing else "PASS",
                "checkpoints_frozen": "WARN" if (missing_key or unreadable or unknown_load) else "PASS",
                "runtime": "PASS" if runtime.get("probe_ok") and (runtime.get("torch") or {}).get("cuda_available") else "WARN",
            },
            "note": "G0 attests the freeze evidence only; it does not permit training",
        },
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
    }


# --------------------------------------------------------------------------- #
# build / write
# --------------------------------------------------------------------------- #
def build_report(repo: Path, interpreter: str | None = None, load_checkpoints: bool = True,
                 skip_runtime: bool = False, timeout: int = 900):
    started = time.time()
    repo = Path(repo).resolve()
    interp, interp_source = pick_interpreter(repo, interpreter)

    baseline_entries = []
    for rel in REQUIRED_FILES:
        baseline_entries.append(file_entry(repo, rel, "baseline-required", required=True))
    for rel in OPTIONAL_FILES:
        baseline_entries.append(file_entry(repo, rel, "baseline-optional", required=False))

    git_info = collect_git(repo)
    submodule = collect_submodule(repo)
    patches = collect_patches(repo)

    runtime = (
        {"probe_ok": False, "interpreter": interp, "error": "runtime probe skipped by flag", "torch": None}
        if skip_runtime
        else collect_runtime(interp)
    )
    libriichi = collect_libriichi(repo, interp)
    toolchains = collect_toolchains(repo)
    nvsmi = collect_nvidia_smi()
    checkpoints = collect_checkpoints(repo, interp, load=load_checkpoints)
    environment = collect_environment(repo)
    datasets = collect_datasets(repo)

    # Attach patch + checkpoint entries to baseline_paths, keyed by relative path.
    baseline_paths = {e["path"]: e for e in baseline_entries}
    for entry in patches["files"]:
        baseline_paths[entry["path"]] = entry
    for entry in checkpoints["entries"]:
        if entry["path"] not in baseline_paths:
            baseline_paths[entry["path"]] = entry

    hashable = sorted(
        (rel, e["sha256"])
        for rel, e in baseline_paths.items()
        if e.get("sha256") and e.get("kind") in ("baseline-required", "patch", "checkpoint")
    )
    baseline_files_hash = hashlib.sha256(
        "".join(f"{rel}:{digest}\n" for rel, digest in hashable).encode("utf-8")
    ).hexdigest() if hashable else None

    assessment = assess(
        git_info,
        submodule,
        patches,
        runtime,
        libriichi,
        toolchains,
        nvsmi,
        checkpoints,
        list(baseline_paths.values()),
        load_skipped=(not load_checkpoints) or not checkpoints["load_performed"],
    )

    torch_info = runtime.get("torch") or {}
    gpu = torch_info.get("gpu") or {}
    smi_gpu = (nvsmi.get("gpus") or [{}])[0]
    report = {
        "schema": SCHEMA_VERSION,
        "generated_at_utc": utc_now_iso(),
        "hostname": environment["hostname"],
        "repo_root": str(repo),
        "status": assessment["status"],
        "gate": assessment["gate"],
        "git_sha": git_info["head"],
        "git_branch": git_info["branch"],
        "git_remote": git_info["remote"],
        "git_dirty": git_info["dirty"],
        "git_tracked_dirty": git_info["tracked_dirty"],
        "git_dirty_entries": git_info["tracked_dirty_entries"],
        "git_untracked_count": git_info.get("untracked_count", 0),
        "git_porcelain": git_info["porcelain"],
        "submodule_sha": submodule["head_sha"],
        "submodule": submodule,
        "patches": patches,
        "python": {
            "caller": sys.version.split()[0],
            "probe_interpreter": interp,
            "probe_interpreter_source": interp_source,
            "probe": {
                "ok": runtime.get("probe_ok"),
                "python_version": runtime.get("python_version"),
                "executable": runtime.get("executable"),
                "error": runtime.get("error"),
            },
        },
        "torch": (
            {
                "version": torch_info.get("version"),
                "cuda_version": torch_info.get("cuda_version"),
                "cudnn_version": torch_info.get("cudnn_version"),
                "interpreter": runtime.get("executable") or interp,
            }
            if torch_info
            else None
        ),
        "cuda": {
            "available": torch_info.get("cuda_available"),
            "version": torch_info.get("cuda_version"),
            "device_count": torch_info.get("device_count"),
        },
        "gpu_name": gpu.get("name") or smi_gpu.get("name"),
        "vram": (
            {
                "bytes": gpu.get("total_memory_bytes"),
                "mib": round((gpu.get("total_memory_bytes") or 0) / (1 << 20)),
                "source": "torch",
            }
            if gpu.get("total_memory_bytes")
            else (
                {
                    "mib": smi_gpu.get("memory_total_mib"),
                    "bytes": (smi_gpu.get("memory_total_mib") or 0) << 20,
                    "source": "nvidia-smi",
                }
                if smi_gpu.get("memory_total_mib")
                else None
            )
        ),
        "compute_capability": (
            f"{gpu.get('major')}.{gpu.get('minor')}" if gpu.get("major") is not None else smi_gpu.get("compute_cap")
        ),
        "bf16_supported": gpu.get("bf16_supported"),
        "nvidia_smi": nvsmi,
        "rust": toolchains,
        "libriichi": libriichi,
        "checkpoints": checkpoints,
        "baseline_paths": baseline_paths,
        "baseline_files_hash": baseline_files_hash,
        "baseline_files_hash_recipe": (
            "sha256 of sorted 'relpath:sha256\\n' over required baseline files, "
            "patches/*.patch and checkpoints/*.pth"
        ),
        "datasets": datasets,
        "environment": environment,
        "blockers": assessment["blockers"],
        "warnings": assessment["warnings"],
        "checks": assessment["checks"],
        "commands_used": {
            "git": git_info["commands"],
            "submodule": submodule["commands"],
        },
        "duration_seconds": round(time.time() - started, 2),
        "safety": {
            "read_only": True,
            "network_used": False,
            "created_baseline_weights": False,
            "writes": ["--output only"],
        },
    }
    return report


def write_output(report: dict, output: str) -> str | None:
    """Atomically write the JSON artefact; return its SHA-256 (None for stdout)."""
    if output == "-":
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return None
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".rf_preflight.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return sha256_file(path)


def print_summary(report: dict, artifact_sha: str | None, stream=None):
    """Human summary. When JSON goes to stdout, the summary goes to stderr."""
    out = stream or sys.stdout

    def say(*parts):
        print(*parts, file=out)

    say(
        "rf_preflight: status={status} G0={gate}".format(
            status=report["status"], gate=report["gate"]["G0"]
        )
    )
    say(
        "  git={sha} branch={branch} dirty={dirty} tracked_dirty={td}".format(
            sha=(report["git_sha"] or "None")[:12],
            branch=report["git_branch"],
            dirty=report["git_dirty"],
            td=report["git_tracked_dirty"],
        )
    )
    say(
        "  submodule={sha} patches={n} torch={torch} cuda={cuda} gpu={gpu}".format(
            sha=(report["submodule_sha"] or "None")[:12],
            n=report["patches"]["count"],
            torch=(report["torch"] or {}).get("version"),
            cuda=(report["cuda"] or {}).get("available"),
            gpu=report["gpu_name"],
        )
    )
    if report["blockers"]:
        for item in report["blockers"]:
            say(f"  BLOCKER {item['code']}: {item['message']}")
    else:
        say("  blockers: none")
    if report["warnings"]:
        for item in report["warnings"]:
            say(f"  WARN {item['code']}: {item['message']}")
    else:
        say("  warnings: none")
    say(f"  baseline_files_hash={report['baseline_files_hash']}")
    say(f"  artifact_sha256={artifact_sha or 'stdout'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="RiichiFormer V2 read-only preflight (PR-00). Writes one JSON artefact."
    )
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="JSON path, or '-' for stdout")
    ap.add_argument("--torch-interpreter", default=None, help="override torch/libriichi probe interpreter")
    ap.add_argument("--skip-checkpoint-load", action="store_true", help="hash only; do not torch.load")
    ap.add_argument("--skip-runtime", action="store_true", help="skip torch/CUDA probe entirely")
    ap.add_argument("--strict", action="store_true", help="exit 1 when the status is WARN")
    ap.add_argument("--timeout", type=int, default=900, help="checkpoint probe timeout (s)")
    args = ap.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    if not repo.is_dir():
        print(f"rf_preflight: repo root not found: {repo}", file=sys.stderr)
        return 3
    try:
        report = build_report(
            repo,
            interpreter=args.torch_interpreter,
            load_checkpoints=not args.skip_checkpoint_load,
            skip_runtime=args.skip_runtime,
            timeout=args.timeout,
        )
        artifact_path = None if args.output == "-" else str(Path(args.output))
        if artifact_path is not None:
            report["output_path"] = artifact_path
        artifact_sha = write_output(report, args.output)
        print_summary(report, artifact_sha, stream=sys.stderr if args.output == "-" else sys.stdout)
    except Exception as exc:  # noqa: BLE001 - never fake success
        print(f"rf_preflight: internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    if report["status"] == "BLOCKED":
        return 2
    if report["status"] == "WARN" and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
