#!/usr/bin/env python3
"""PR-39 model-config resolution: MVP-S by default, tiny only when explicit.

The PR-14 CLI defaulted to a one-layer 64-wide model, which silently trained a
"tiny smoke" architecture while reporting it as training.  This module is the
single library entry point that resolves a model config:

``resolve_mvp_config()``
    loads ``configs/riichiformer/mvp_s.yaml`` and verifies it still builds the
    frozen six-layer MVP-S (**5,217,582** parameters).  A drifted default config
    is an error, never a silent smaller model.
``resolve_mvp_config(tiny_smoke=True)``
    returns the explicit tiny smoke config (one layer, 64-wide FFN/state MLP)
    and records ``tiny_smoke=True``.  It is never selected implicitly and
    cannot be combined with an explicit model-config path.

The resolved object carries the model-config hash and parameter count so the
trainer can store them in a checkpoint and a strict resume can verify them.
``scripts/rf_train.py`` is owned by PR-37; wiring ``--model-config`` /
``--tiny-smoke`` to this function is its follow-up, not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch import nn

from ..models.policy import MVPConfig, Policy, count_parameters
from .checkpoint import config_hash

#: Frozen reference architecture (six-layer MVP-S).
MVP_S_REFERENCE_LAYERS: int = 6
MVP_S_REFERENCE_PARAMETERS: int = 5_217_582
MVP_S_REFERENCE_NAME: str = "mvp_s"

#: Explicit small-model choice; only reachable through ``tiny_smoke=True``.
TINY_SMOKE_CONFIG = MVPConfig(
    name="tiny_smoke", layers=1, ffn=64, state_hidden=64, max_positions=2048,
)
TINY_SMOKE_PARAMETER_LIMIT: int = 1_000_000

#: Repository-relative default model config.
DEFAULT_MODEL_CONFIG_RELPATH = Path("configs") / "riichiformer" / "mvp_s.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CONFIG_PATH = _REPO_ROOT / DEFAULT_MODEL_CONFIG_RELPATH


class ModelConfigError(ValueError):
    """A model config is missing, drifted, or is a silent tiny model."""


@dataclass(frozen=True)
class ResolvedModelConfig:
    """A resolved architecture plus the provenance a checkpoint should store."""

    config: MVPConfig
    parameter_count: int
    config_hash: str
    source_path: Path | None
    tiny_smoke: bool
    reference_verified: bool

    def as_provenance(self) -> dict:
        """Metadata suitable for a checkpoint's ``extra``/provenance block."""
        return {
            "model_config_name": self.config.name,
            "model_config_hash": self.config_hash,
            "model_parameters": self.parameter_count,
            "model_tiny_smoke": bool(self.tiny_smoke),
        }

    def describe(self) -> str:
        source = str(self.source_path) if self.source_path is not None else "<builtin>"
        kind = "tiny_smoke" if self.tiny_smoke else "mvp_s"
        return (
            f"model_config={self.config.name} kind={kind} "
            f"layers={self.config.layers} hidden={self.config.hidden} "
            f"ffn={self.config.ffn} state_hidden={self.config.state_hidden} "
            f"parameters={self.parameter_count} sha256={self.config_hash[:12]} "
            f"source={source}"
        )


_COUNT_CACHE: dict[str, int] = {}


def _parameter_count(config: MVPConfig) -> int:
    key = config_hash(config)
    cached = _COUNT_CACHE.get(key)
    if cached is None:
        model: nn.Module = Policy(config)
        cached = count_parameters(model)
        _COUNT_CACHE[key] = cached
    return cached


def _default_path(repo_root: Path | str | None) -> Path:
    if repo_root is None:
        return DEFAULT_MODEL_CONFIG_PATH
    return Path(repo_root) / DEFAULT_MODEL_CONFIG_RELPATH


def resolve_mvp_config(model_config_path: str | Path | None = None, *,
                       tiny_smoke: bool = False,
                       repo_root: str | Path | None = None,
                       verify_parameters: bool = True,
                       ) -> ResolvedModelConfig:
    """Resolve the training model config (MVP-S unless tiny is explicit).

    Parameters
    ----------
    model_config_path:
        Explicit YAML path.  Omitted -> the frozen ``mvp_s.yaml``.  May not be
        combined with ``tiny_smoke=True``.
    tiny_smoke:
        Explicit opt-in to the one-layer smoke model.  Never inferred from a
        missing config, a CLI default or a parameter budget.
    repo_root:
        Override the repository root used to locate the default config.
    verify_parameters:
        Instantiating the model to count parameters is on by default; the
        reference default is refused unless it still counts exactly
        ``MVP_S_REFERENCE_PARAMETERS``.
    """
    if not isinstance(tiny_smoke, bool):
        raise ModelConfigError("tiny_smoke must be bool")
    if tiny_smoke:
        if model_config_path is not None:
            raise ModelConfigError(
                "tiny_smoke is an explicit alternative to a model config; "
                "passing both is ambiguous"
            )
        config = TINY_SMOKE_CONFIG
        source_path = None
        reference = False
    else:
        if model_config_path is None:
            path = _default_path(repo_root)
            reference = True
        else:
            path = Path(model_config_path)
            reference = False
        if not path.is_file():
            raise ModelConfigError(f"model config not found: {path}")
        try:
            config = MVPConfig.from_yaml(path)
        except Exception as exc:  # noqa: BLE001 - re-raise as this module's error
            raise ModelConfigError(f"invalid model config {path}: {exc}") from exc
        source_path = path

    if verify_parameters:
        parameters = _parameter_count(config)
    else:
        parameters = 0
    if reference and verify_parameters:
        if config.layers != MVP_S_REFERENCE_LAYERS \
                or parameters != MVP_S_REFERENCE_PARAMETERS:
            raise ModelConfigError(
                "default model config drifted from the frozen six-layer MVP-S: "
                f"layers={config.layers} (expected {MVP_S_REFERENCE_LAYERS}), "
                f"parameters={parameters} (expected {MVP_S_REFERENCE_PARAMETERS}); "
                "pass an explicit model_config_path for an intentional variant"
            )
    if tiny_smoke:
        if config != TINY_SMOKE_CONFIG:
            raise ModelConfigError(
                "tiny_smoke config drifted from the frozen tiny smoke shape"
            )
        if verify_parameters and parameters > TINY_SMOKE_PARAMETER_LIMIT:
            raise ModelConfigError(
                f"tiny_smoke model has {parameters} parameters, above the "
                f"{TINY_SMOKE_PARAMETER_LIMIT} smoke limit"
            )
    if verify_parameters and parameters <= 0:
        raise ModelConfigError("model has no trainable parameters")
    return ResolvedModelConfig(
        config=config,
        parameter_count=parameters,
        config_hash=config_hash(config),
        source_path=source_path,
        tiny_smoke=tiny_smoke,
        reference_verified=bool(reference),
    )


__all__ = [
    "DEFAULT_MODEL_CONFIG_PATH", "DEFAULT_MODEL_CONFIG_RELPATH",
    "MVP_S_REFERENCE_LAYERS", "MVP_S_REFERENCE_NAME", "MVP_S_REFERENCE_PARAMETERS",
    "ModelConfigError", "ResolvedModelConfig", "TINY_SMOKE_CONFIG",
    "TINY_SMOKE_PARAMETER_LIMIT", "resolve_mvp_config",
]
