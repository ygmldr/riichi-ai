#!/usr/bin/env python3
"""PR-14 checkpoint protocol: full optimizer/RNG/BF16/EMA + strict hashes."""

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


def save_checkpoint(path: str | Path, *, model, optimizer, scheduler=None, scaler=None,
                    counters: Mapping, hashes: Mapping, config, seed: int,
                    ema=None, extra: Mapping | None = None) -> CheckpointInfo:
    path = Path(path)
    required = {"schema_hash", "action_map_hash", "normalizer_hash"}
    if not required <= set(hashes):
        raise CheckpointError(f"checkpoint hashes missing {sorted(required - set(hashes))}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise CheckpointError("seed must be int")
    counters = {key: int(counters[key]) for key in ("step", "tokens", "decisions")}
    config_dict = _config_dict(config)
    payload = {
        "format": "rf-checkpoint/1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "rng": _rng_state(),
        "counters": counters,
        "hashes": dict(hashes),
        "config": config_dict,
        "config_hash": hashlib.sha256(_canonical(config_dict).encode()).hexdigest(),
        "seed": int(seed),
        "extra": dict(extra) if extra else {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    sha = _sha_file(tmp)
    os.replace(tmp, path)
    manifest = {
        "format": "rf-checkpoint/1", "sha256": sha,
        "counters": counters, "hashes": dict(hashes),
        "config_hash": payload["config_hash"], "seed": int(seed),
        "has_ema": ema is not None,
    }
    tmp_manifest = Path(str(path) + ".json.tmp")
    tmp_manifest.write_text(_canonical(manifest) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, Path(str(path) + ".json"))
    return CheckpointInfo(path=path, sha256=sha, step=counters["step"],
                          tokens=counters["tokens"], decisions=counters["decisions"],
                          seed=seed, config=config_dict, hashes=dict(hashes),
                          has_ema=ema is not None)


def load_checkpoint(path: str | Path, *, model, optimizer=None, scheduler=None,
                    scaler=None, ema=None, expected_hashes: Mapping | None = None):
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
    if payload.get("format") != "rf-checkpoint/1":
        raise CheckpointMismatch("unsupported checkpoint format")
    hashes = payload.get("hashes", {})
    required = {"schema_hash", "action_map_hash", "normalizer_hash"}
    if not required <= set(hashes):
        raise CheckpointMismatch("checkpoint lacks required hashes")
    if expected_hashes is not None:
        if set(expected_hashes) != required:
            raise CheckpointMismatch("expected_hashes must contain exactly 3 keys")
        for key in required:
            if expected_hashes[key] != hashes.get(key):
                raise CheckpointMismatch(f"{key} mismatch")
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
    counters = payload.get("counters", {})
    config = payload.get("config", {})
    return CheckpointInfo(
        path=path, sha256=actual_sha, step=int(counters.get("step", 0)),
        tokens=int(counters.get("tokens", 0)), decisions=int(counters.get("decisions", 0)),
        seed=int(payload.get("seed", 0)), config=config, hashes=hashes,
        has_ema=payload.get("ema") is not None), payload


__all__ = ["CheckpointError", "CheckpointInfo", "CheckpointMismatch",
           "load_checkpoint", "save_checkpoint"]
