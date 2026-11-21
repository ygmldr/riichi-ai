#!/usr/bin/env python3
"""Checkpoint protocol (PR-14 + PR-39): strict resume and init-weights-only.

Two deliberately separate entry points:

``load_checkpoint``
    **Strict resume.**  Verifies the sidecar SHA-256, format, the three frozen
    hashes, optional source/split/policy/canonical provenance, the training
    config hash, and the model config hash; restores model, optimizer,
    scheduler, scaler, EMA and RNG.  ``strict_resume=True`` additionally
    requires the optimizer/scheduler/EMA/RNG state to be present instead of
    silently skipping it.
``init_weights_only``
    **Initialization only.**  Loads the model state dict (strict=True) and
    nothing else: no optimizer, scheduler, scaler, EMA or RNG restore.  This is
    the path for warm-starting a new run, never a substitute for resume.

``save_checkpoint`` stores the training config (with ``config_hash``), the model
architecture config (``model_config``/``model_config_hash``) and optional
provenance (source id, split manifest, research policy, canonical manifest/index
hashes).  The legacy three-hash contract and the existing checkpoint format
string are unchanged, so older checkpoints still load.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


class CheckpointError(RuntimeError):
    """Malformed checkpoint."""


class CheckpointMismatch(CheckpointError):
    """Hash/version/schema/source mismatch; never loaded silently."""


#: Provenance keys a strict-resume checkpoint should carry.
PROVENANCE_REQUIRED_KEYS: tuple[str, ...] = (
    "source_id",
    "split_manifest_sha256",
    "canonical_manifest_sha256",
    "canonical_index_sha256",
    "research_policy",
    "research_policy_hash",
)

#: The frozen three-hash contract every checkpoint must carry.
REQUIRED_HASH_KEYS: tuple[str, ...] = (
    "schema_hash", "action_map_hash", "normalizer_hash",
)

CHECKPOINT_FORMAT = "rf-checkpoint/1"


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_dict(config) -> dict:
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise CheckpointError("config must be a dataclass or mapping")


def canonical_config_dict(config) -> dict:
    """Canonical JSON-able dict of a dataclass/mapping config (public helper)."""
    return _config_dict(config)


def config_hash(config) -> str:
    """SHA-256 of the canonical config dict (training or model config)."""
    return hashlib.sha256(_canonical(_config_dict(config)).encode()).hexdigest()


def _validate_provenance(provenance: Mapping) -> dict:
    if not isinstance(provenance, Mapping):
        raise CheckpointError("provenance must be a mapping")
    missing = [key for key in PROVENANCE_REQUIRED_KEYS if key not in provenance]
    if missing:
        raise CheckpointError(f"provenance missing keys {missing}")
    out = {}
    for key, value in provenance.items():
        if not isinstance(key, str) or not key:
            raise CheckpointError(f"provenance key must be a non-empty string: {key!r}")
        if value is None:
            raise CheckpointError(f"provenance[{key!r}] must not be None")
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise CheckpointError(
                f"provenance[{key!r}] must be a string or int, got "
                f"{type(value).__name__}"
            )
        if isinstance(value, str) and not value:
            # A licensed_only shard has no operator-policy hash by design; every
            # other provenance string (source/split/canonical hashes) is non-empty.
            if key == "research_policy_hash" \
                    and provenance.get("research_policy") == "licensed_only":
                out[key] = ""
                continue
            raise CheckpointError(f"provenance[{key!r}] must not be empty")
        out[key] = value
    return out


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def _restore_rng(state: Mapping) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _check_rng_state(state, *, strict: bool) -> None:
    if state is None:
        if strict:
            raise CheckpointMismatch("checkpoint has no RNG state for a strict resume")
        return
    if not isinstance(state, Mapping):
        raise CheckpointMismatch("checkpoint RNG state is malformed")
    missing = [key for key in ("python", "numpy", "torch") if key not in state]
    if missing:
        raise CheckpointMismatch(f"checkpoint RNG state missing {missing}")


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    sha256: str
    step: int
    tokens: int
    decisions: int
    seed: int
    config: dict
    hashes: dict
    has_ema: bool
    config_hash: str | None = None
    model_config: dict | None = None
    model_config_hash: str | None = None
    provenance: dict | None = None


def _load_payload(path: str | Path) -> tuple[Path, str, dict]:
    """Shared integrity checks: file hash, sidecar, format, three hashes."""
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"checkpoint not found: {path}")
    sidecar = Path(str(path) + ".json")
    if not sidecar.exists():
        raise CheckpointError(f"checkpoint sidecar missing: {sidecar}")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    actual_sha = _sha_file(path)
    if manifest.get("sha256") != actual_sha:
        raise CheckpointMismatch("checkpoint file hash does not match sidecar")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointError(f"cannot load checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointMismatch("checkpoint payload is not a mapping")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointMismatch("unsupported checkpoint format")
    hashes = payload.get("hashes", {})
    required = set(REQUIRED_HASH_KEYS)
    if not required <= set(hashes):
        raise CheckpointMismatch("checkpoint lacks required hashes")
    return path, actual_sha, payload


def _check_expected_hashes(payload: Mapping, expected_hashes: Mapping | None) -> None:
    if expected_hashes is None:
        return
    required = set(REQUIRED_HASH_KEYS)
    if set(expected_hashes) != required:
        raise CheckpointMismatch("expected_hashes must contain exactly 3 keys")
    hashes = payload.get("hashes", {})
    for key in REQUIRED_HASH_KEYS:
        if expected_hashes[key] != hashes.get(key):
            raise CheckpointMismatch(f"{key} mismatch")


def _check_internal_config_hash(payload: Mapping, *, strict: bool) -> None:
    stored = payload.get("config")
    stored_hash = payload.get("config_hash")
    if stored is None:
        if strict:
            raise CheckpointMismatch(
                "checkpoint has no training config for a strict resume")
        return
    if not isinstance(stored, Mapping):
        raise CheckpointMismatch("checkpoint config is malformed")
    if stored_hash is None:
        if strict:
            raise CheckpointMismatch("checkpoint has no config_hash")
        return
    if config_hash(stored) != stored_hash:
        raise CheckpointMismatch(
            "checkpoint config_hash does not match its stored training config")


def _check_model_config(payload: Mapping, *,
                        expected_model_config=None,
                        expected_model_config_hash: str | None = None) -> None:
    stored = payload.get("model_config")
    stored_hash = payload.get("model_config_hash")
    if expected_model_config is not None:
        if stored is None:
            raise CheckpointMismatch(
                "checkpoint carries no model_config to compare against")
        expected = _config_dict(expected_model_config)
        if _canonical(stored) != _canonical(expected):
            raise CheckpointMismatch("checkpoint model_config mismatch")
        expected_hash = config_hash(expected)
        if stored_hash is not None and stored_hash != expected_hash:
            raise CheckpointMismatch("checkpoint model_config_hash mismatch")
    if expected_model_config_hash is not None:
        if stored_hash is None:
            raise CheckpointMismatch(
                "checkpoint carries no model_config_hash to compare against")
        if stored_hash != expected_model_config_hash:
            raise CheckpointMismatch("checkpoint model_config_hash mismatch")


def _provenance_value(payload: Mapping, key: str):
    provenance = payload.get("provenance") or {}
    if isinstance(provenance, Mapping) and key in provenance:
        return provenance[key]
    return (payload.get("hashes") or {}).get(key)


def _check_expected_provenance(payload: Mapping,
                               expected_provenance: Mapping | None,
                               *, strict: bool) -> None:
    if expected_provenance is None:
        return
    if not isinstance(expected_provenance, Mapping):
        raise CheckpointMismatch("expected_provenance must be a mapping")
    for key, expected_value in expected_provenance.items():
        if _provenance_value(payload, key) is None:
            raise CheckpointMismatch(
                f"checkpoint provenance missing for {key} (strict resume)")
        if _provenance_value(payload, key) != expected_value:
            raise CheckpointMismatch(f"checkpoint provenance mismatch for {key}")
    if strict:
        missing = [key for key in PROVENANCE_REQUIRED_KEYS
                   if _provenance_value(payload, key) is None]
        if missing:
            raise CheckpointMismatch(
                f"checkpoint provenance missing keys {missing} for strict resume")


def save_checkpoint(path: str | Path, *, model, optimizer, scheduler=None, scaler=None,
                    counters: Mapping, hashes: Mapping, config, seed: int,
                    ema=None, extra: Mapping | None = None,
                    model_config=None, provenance: Mapping | None = None,
                    ) -> CheckpointInfo:
    path = Path(path)
    required = set(REQUIRED_HASH_KEYS)
    if not required <= set(hashes):
        raise CheckpointError(f"checkpoint hashes missing {sorted(required - set(hashes))}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise CheckpointError("seed must be int")
    counters = {key: int(counters[key]) for key in ("step", "tokens", "decisions")}
    config_dict = _config_dict(config)
    model_config_dict = _config_dict(model_config) if model_config is not None else None
    provenance_dict = _validate_provenance(provenance) if provenance is not None else None
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "rng": _rng_state(),
        "counters": counters,
        "hashes": dict(hashes),
        "config": config_dict,
        "config_hash": config_hash(config_dict),
        "model_config": model_config_dict,
        "model_config_hash": (config_hash(model_config_dict)
                              if model_config_dict is not None else None),
        "provenance": provenance_dict,
        "seed": int(seed),
        "extra": dict(extra) if extra else {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    sha = _sha_file(tmp)
    os.replace(tmp, path)
    manifest = {
        "format": CHECKPOINT_FORMAT, "sha256": sha,
        "counters": counters, "hashes": dict(hashes),
        "config_hash": payload["config_hash"],
        "model_config_hash": payload["model_config_hash"],
        "provenance_hash": (hashlib.sha256(
            _canonical(provenance_dict).encode()).hexdigest()
            if provenance_dict is not None else None),
        "seed": int(seed),
        "has_ema": ema is not None,
    }
    tmp_manifest = Path(str(path) + ".json.tmp")
    tmp_manifest.write_text(_canonical(manifest) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, Path(str(path) + ".json"))
    return CheckpointInfo(
        path=path, sha256=sha, step=counters["step"],
        tokens=counters["tokens"], decisions=counters["decisions"],
        seed=seed, config=config_dict, hashes=dict(hashes),
        has_ema=ema is not None, config_hash=payload["config_hash"],
        model_config=model_config_dict,
        model_config_hash=payload["model_config_hash"],
        provenance=provenance_dict)


def _info_from_payload(path: Path, sha: str, payload: Mapping) -> CheckpointInfo:
    counters = payload.get("counters") or {}
    return CheckpointInfo(
        path=path, sha256=sha, step=int(counters.get("step", 0)),
        tokens=int(counters.get("tokens", 0)),
        decisions=int(counters.get("decisions", 0)),
        seed=int(payload.get("seed", 0)),
        config=payload.get("config") or {},
        hashes=payload.get("hashes") or {},
        has_ema=payload.get("ema") is not None,
        config_hash=payload.get("config_hash"),
        model_config=payload.get("model_config"),
        model_config_hash=payload.get("model_config_hash"),
        provenance=payload.get("provenance"))


def load_checkpoint(path: str | Path, *, model, optimizer=None, scheduler=None,
                    scaler=None, ema=None, expected_hashes: Mapping | None = None,
                    expected_provenance: Mapping | None = None,
                    expected_config_hash: str | None = None,
                    expected_config=None,
                    expected_model_config=None,
                    expected_model_config_hash: str | None = None,
                    strict_resume: bool = False):
    """Strict resume: full provenance/config checks, then all training state.

    ``strict_resume=True`` refuses a checkpoint that lacks optimizer state (when
    an optimizer is supplied), scheduler state (when a scheduler is supplied),
    EMA state (when an EMA module is supplied), RNG state, counters, config or
    config hash, instead of silently continuing with partial state.
    """
    path, actual_sha, payload = _load_payload(path)
    _check_expected_hashes(payload, expected_hashes)
    _check_internal_config_hash(payload, strict=strict_resume)
    if expected_config is not None:
        stored = payload.get("config")
        if stored is None or _canonical(stored) != _canonical(_config_dict(expected_config)):
            raise CheckpointMismatch("checkpoint training config mismatch")
    if expected_config_hash is not None:
        stored_hash = payload.get("config_hash")
        if stored_hash is None:
            raise CheckpointMismatch("checkpoint has no config_hash")
        if stored_hash != expected_config_hash:
            raise CheckpointMismatch("checkpoint config_hash mismatch")
    _check_model_config(payload, expected_model_config=expected_model_config,
                        expected_model_config_hash=expected_model_config_hash)
    _check_expected_provenance(payload, expected_provenance, strict=strict_resume)
    if strict_resume:
        if optimizer is not None and payload.get("optimizer") is None:
            raise CheckpointMismatch(
                "checkpoint has no optimizer state for a strict resume")
        if scheduler is not None and payload.get("scheduler") is None:
            raise CheckpointMismatch(
                "checkpoint has no scheduler state for a strict resume")
        if ema is not None and payload.get("ema") is None:
            raise CheckpointMismatch(
                "checkpoint has no EMA state for a strict resume")
        _check_rng_state(payload.get("rng"), strict=True)
        if not payload.get("counters"):
            raise CheckpointMismatch("checkpoint has no counters for a strict resume")
    else:
        _check_rng_state(payload.get("rng"), strict=False)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointMismatch(f"model state mismatch: {exc}") from exc
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
    if payload.get("rng") is not None:
        _restore_rng(payload["rng"])
    return _info_from_payload(path, actual_sha, payload), payload


def init_weights_only(path: str | Path, *, model,
                      expected_hashes: Mapping | None = None,
                      expected_model_config=None,
                      expected_model_config_hash: str | None = None,
                      ) -> CheckpointInfo:
    """Load **only** the model weights; never optimizer/scheduler/EMA/RNG.

    This is a warm-start path, deliberately separate from :func:`load_checkpoint`.
    It still verifies the file hash, format, three frozen hashes and the optional
    model-config identity; it does not restore any training state and never
    advances counters.
    """
    path, actual_sha, payload = _load_payload(path)
    _check_expected_hashes(payload, expected_hashes)
    _check_model_config(payload, expected_model_config=expected_model_config,
                        expected_model_config_hash=expected_model_config_hash)
    if "model" not in payload or not isinstance(payload["model"], Mapping):
        raise CheckpointMismatch("checkpoint has no model state dict")
    try:
        model.load_state_dict(payload["model"], strict=True)
    except Exception as exc:  # noqa: BLE001
        raise CheckpointMismatch(f"model state mismatch: {exc}") from exc
    return _info_from_payload(path, actual_sha, payload)


__all__ = [
    "CHECKPOINT_FORMAT", "CheckpointError", "CheckpointInfo", "CheckpointMismatch",
    "PROVENANCE_REQUIRED_KEYS", "REQUIRED_HASH_KEYS", "canonical_config_dict",
    "config_hash", "init_weights_only", "load_checkpoint", "save_checkpoint",
]
