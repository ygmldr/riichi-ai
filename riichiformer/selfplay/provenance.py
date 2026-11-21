#!/usr/bin/env python3
"""PR-41 self-play provenance: ruleset labels, generator fingerprint, output guards.

Two rulesets are supported and **never conflated**:

``libriichi_tenhou_approx/1``
    Games really played by the pinned local ``libriichi`` arena
    (``OneVsThree.py_vs_py``).  The engine implements Tenhou-like 4-player rules, but
    PR-40 still tracks known deviations (R02/R25/R26/R29), so this is
    *Tenhou-approximate*, **not** Tenhou-parity verified and not an official rank.

``synthetic_abstract_hanchan/1``
    A procedural, draw-and-discard-only fallback with tenpai/noten payments.  It has
    no calls, no wins, no hand scoring and no wall model beyond a seeded shuffle.  It
    exists so the rollout/provenance/seed plumbing can run without libriichi; it is
    **not** mahjong-equivalent and must never be reported as such.

Every generated file is written under the caller's explicit ``--out``.  Paths whose
resolved components include ``datasets``, ``runs``, ``checkpoints`` or ``vendor`` are
refused, as is the repository root.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..schemas.action_v3 import ACTION_MAP_HASH, SCHEMA_VERSION, canonical_json

__all__ = [
    "SELFPLAY_VERSION",
    "PROVENANCE_FORMAT",
    "RULESET_LIBRIICHI",
    "RULESET_SYNTHETIC",
    "RULESETS",
    "FORBIDDEN_COMPONENTS",
    "ProvenanceError",
    "RULESET_METADATA",
    "assert_known_ruleset",
    "generator_fingerprint",
    "guard_aux_path",
    "guard_out",
    "ruleset_metadata",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "source_metadata",
    "utc_now",
    "write_json",
    "build_provenance",
]

SELFPLAY_VERSION = "rf-selfplay/1"
PROVENANCE_FORMAT = "rf-selfplay-provenance/1"

RULESET_LIBRIICHI = "libriichi_tenhou_approx/1"
RULESET_SYNTHETIC = "synthetic_abstract_hanchan/1"

#: Output path components that must never receive generated self-play data.
FORBIDDEN_COMPONENTS = ("datasets", "runs", "checkpoints", "vendor")

RULESETS = (RULESET_LIBRIICHI, RULESET_SYNTHETIC)

RULESET_METADATA: dict[str, dict[str, Any]] = {
    RULESET_LIBRIICHI: {
        "ruleset_id": RULESET_LIBRIICHI,
        "rules_engine": "libriichi.arena.OneVsThree.py_vs_py",
        "tenhou_equivalent": False,
        "parity_status": "tenhou_approximate",
        "real_mahjong_rules": True,
        "calls": True,
        "kan": True,
        "wins": True,
        "known_deviations": ["R02", "R25", "R26", "R29"],
        "deviation_note": (
            "pinned Mortal libriichi arena; R02/R25/R26 are settled by that engine. "
            "R29 ippatsu timing is MATCH at the PlayerState level (PR-40R, executable "
            "native fixture); a full Board-scripted R29 game is NOT_RUN and the "
            "per-decision schema still has no R29 member. Not Tenhou-parity verified."
        ),
        "strength_claim": "none",
    },
    RULESET_SYNTHETIC: {
        "ruleset_id": RULESET_SYNTHETIC,
        "rules_engine": "none (procedural fallback)",
        "tenhou_equivalent": False,
        "parity_status": "not_mahjong_equivalent",
        "real_mahjong_rules": False,
        "calls": False,
        "kan": False,
        "wins": False,
        "known_deviations": [],
        "deviation_note": (
            "seeded draw/discard loop with tenpai/noten payments and riichi sticks only; "
            "no calls, no wins, no hand/scoring model, no Tenhou wall model."
        ),
        "strength_claim": "none",
    },
}


class ProvenanceError(ValueError):
    """A provenance/ruleset/output-contract violation (always fail closed)."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def assert_known_ruleset(ruleset_id: str) -> str:
    if ruleset_id not in RULESET_METADATA:
        raise ProvenanceError(
            f"unknown ruleset_id {ruleset_id!r}; known: {sorted(RULESET_METADATA)}")
    return ruleset_id


def ruleset_metadata(ruleset_id: str) -> dict:
    assert_known_ruleset(ruleset_id)
    return dict(RULESET_METADATA[ruleset_id])


# ---------------------------------------------------------------------------
# output guard
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def guard_out(out: str | Path, *, repo_root: str | Path | None = None) -> Path:
    """Resolve the mandatory output directory and refuse frozen locations."""
    if out is None or str(out).strip() == "":
        raise ProvenanceError("--out is required (refusing to guess a path)")
    resolved = Path(out).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else _repo_root()
    if resolved == root:
        raise ProvenanceError(f"refusing repository root as an output directory: {resolved}")
    for part in resolved.parts:
        if part in FORBIDDEN_COMPONENTS:
            raise ProvenanceError(
                f"refusing output path {resolved}: component {part!r} is a frozen "
                f"location ({', '.join(FORBIDDEN_COMPONENTS)})")
    if resolved.exists() and not resolved.is_dir():
        raise ProvenanceError(f"output path exists and is not a directory: {resolved}")
    return resolved


def guard_aux_path(path: str | Path, *, repo_root: str | Path | None = None) -> Path:
    """Like :func:`guard_out` but for a file (e.g. the shared seed ledger)."""
    if path is None or str(path).strip() == "":
        raise ProvenanceError("auxiliary output path is required")
    resolved = Path(path).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else _repo_root()
    if resolved == root:
        raise ProvenanceError(f"refusing repository root as an output file: {resolved}")
    for part in resolved.parts:
        if part in FORBIDDEN_COMPONENTS:
            raise ProvenanceError(
                f"refusing output path {resolved}: component {part!r} is a frozen "
                f"location ({', '.join(FORBIDDEN_COMPONENTS)})")
    return resolved


# ---------------------------------------------------------------------------
# generator fingerprint
# ---------------------------------------------------------------------------
def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def generator_fingerprint(*, package_dir: str | Path | None = None) -> dict:
    """Content hash of every self-play module plus the frozen schema/action hashes.

    The digest changes whenever a bot, the rollout bridge, the seed ledger or this
    provenance module changes, so a game record can be tied to the exact generator
    source that produced it (not only to a git commit).
    """
    directory = Path(package_dir).resolve() if package_dir is not None else _package_dir()
    modules: dict[str, str] = {}
    for path in sorted(directory.glob("*.py")):
        modules[path.name] = sha256_file(path)
    if not modules:
        raise ProvenanceError(f"no generator modules found in {directory}")
    payload = "".join(f"{name}:{digest}\n" for name, digest in sorted(modules.items()))
    return {
        "format": "rf-selfplay-generator/1",
        "package": "riichiformer.selfplay",
        "version": SELFPLAY_VERSION,
        "directory": str(directory),
        "code_sha256": sha256_text(payload),
        "modules": modules,
        "schema_version": SCHEMA_VERSION,
        "action_map_sha256": ACTION_MAP_HASH,
    }


# ---------------------------------------------------------------------------
# source metadata
# ---------------------------------------------------------------------------
def source_metadata(*, ruleset_id: str | None = None) -> dict:
    """Truthful source ledger for locally generated self-play, without licence claims."""
    return {
        "source_kind": "selfplay_own",
        "source_rows": [
            {
                "source_id": "selfplay_own",
                "source_kind": "selfplay_own",
                "display_name": "local self-play generated by project heuristic bots",
                "rights_status": "own_generated",
                "external_license": "none",
                "third_party_logs": False,
                "third_party_weights": False,
                "network_access": False,
                "online_platform_play": False,
                "train_allowed_by_project": True,
                "publish_allowed": False,
                "note": (
                    "generated offline by this repository; project-private research only; "
                    "no third-party licence is granted or implied; no Tenhou/Majsoul play"
                ),
            }
        ],
        "ruleset_id": ruleset_id,
        "operator_policy": "private_research_only",
    }


# ---------------------------------------------------------------------------
# provenance report
# ---------------------------------------------------------------------------
def _libriichi_metadata() -> dict | None:
    try:
        import libriichi  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
    try:
        import libriichi

        path = getattr(libriichi, "__file__", None)
        return {
            "available": True,
            "module": "libriichi",
            "version": getattr(libriichi, "__version__", None),
            "path": path,
            "sha256": sha256_file(path) if path and Path(path).is_file() else None,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:400]}


def build_provenance(*, ruleset_id: str, engine: str, command: Sequence[str] | None = None,
                     generator: Mapping[str, Any] | None = None,
                     extra: Mapping[str, Any] | None = None) -> dict:
    assert_known_ruleset(ruleset_id)
    fingerprint = dict(generator) if generator is not None else generator_fingerprint()
    payload = {
        "format": PROVENANCE_FORMAT,
        "created_utc": utc_now(),
        "task": "PR-41/PR-42",
        "version": SELFPLAY_VERSION,
        "ruleset": ruleset_metadata(ruleset_id),
        "engine": engine,
        "generator": fingerprint,
        "source": source_metadata(ruleset_id=ruleset_id),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": list(sys.argv if command is None else command),
        "libriichi": _libriichi_metadata(),
        "notes": [
            "local offline self-play only; no network, no online platform bot play",
            "heuristic bots have no checkpoint and claim no strength",
            "not a Tenhou ranking, not an official dan/rank and not a Mortal comparison",
        ],
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(target)
    return target


def write_jsonl(path: str | Path, rows) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    tmp.replace(target)
    return target


@dataclass
class ProvenanceReport:
    """Small container used by the scripts and tests."""

    ruleset_id: str
    engine: str
    generator: dict = field(default_factory=generator_fingerprint)
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return build_provenance(ruleset_id=self.ruleset_id, engine=self.engine,
                                generator=self.generator, extra=self.extra)
