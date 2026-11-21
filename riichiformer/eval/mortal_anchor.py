#!/usr/bin/env python3
"""PR-46A frozen Mortal-5D anchor: checkpoint adapter + baseline registry.

The anchor is ``M5D-R0``: a user-confirmed, self-trained Mortal-5D checkpoint.

* checkpoint: ``/home/ygsldr/riichi-ai/checkpoints/mortal.pth``
* SHA-256:   ``0a44613cb9c7edae1057cf6a2c160c3971c78130e18e07f28bb43e8f62abe532``
* steps:     396,000
* model:     pure CNN ``resnet{conv_channels=192,num_blocks=40}``, Mortal DQN v4
* events:    ``[model].use_event_branch = false``
* inference: deterministic argmax on the native 46-wide legal mask, CPU

This module **validates** those facts on load (file SHA, step count, config, weight
shapes, event-branch flag) and exposes a ``react_batch``-compatible engine for the
native ``libriichi.arena.OneVsThree.py_vs_py`` driver.  It does not download, train,
mutate or publish anything.  ``permission: local_research_only`` is a scope statement,
not a licence grant; no third-party right is created or implied.

Canonical hashing: the registry is serialised as JSON with sorted keys and compact
separators; ``registry_hash`` is the SHA-256 of that canonical payload with the hash
field itself removed.  Recomputing it must reproduce the stored value.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "ANCHOR_ID",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_CHECKPOINT_SHA256",
    "DEFAULT_STEPS",
    "DEFAULT_VERSION",
    "DEFAULT_CONV_CHANNELS",
    "DEFAULT_NUM_BLOCKS",
    "EXPECTED_MODEL_CONFIG",
    "RULESET_ID",
    "AnchorError",
    "AnchorMismatch",
    "CheckpointIdentity",
    "LoadedMortalCheckpoint",
    "MortalCheckpointAdapter",
    "build_champion_seats",
    "build_baseline_registry",
    "canonical_registry_payload",
    "registry_hash",
    "load_baseline_registry",
    "write_baseline_registry",
    "validate_registry",
    "load_mortal_checkpoint",
    "checkpoint_identity",
    "sha256_file",
    "sha256_text",
    "canonical_json",
    "patch_manifest",
    "patch_manifest_hash",
    "libriichi_git_sha",
    "libriichi_build_sha256",
    "python_driver_git_sha",
    "tenhou_approx_rules_hash",
    "model_config_hash",
    "model_config_from_state",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

ANCHOR_ID = "M5D-R0"
DEFAULT_ROLE = "frozen_opponent"
DEFAULT_NAME = "mortal_selftrained_5d"
DEFAULT_CHECKPOINT_PATH = "/home/ygsldr/riichi-ai/checkpoints/mortal.pth"
DEFAULT_CHECKPOINT_SHA256 = (
    "0a44613cb9c7edae1057cf6a2c160c3971c78130e18e07f28bb43e8f62abe532"
)
DEFAULT_STEPS = 396_000
DEFAULT_VERSION = 4
DEFAULT_CONV_CHANNELS = 192
DEFAULT_NUM_BLOCKS = 40
RULESET_ID = "libriichi_tenhou_approx/1"

EXPECTED_MODEL_CONFIG: dict[str, Any] = {
    "control_version": DEFAULT_VERSION,
    "resnet": {"conv_channels": DEFAULT_CONV_CHANNELS,
               "num_blocks": DEFAULT_NUM_BLOCKS},
    "model": {
        "use_event_branch": False,
        "event_dim": 128,
        "event_ffn": 512,
        "event_heads": 4,
        "event_layers": 4,
    },
}

REGISTRY_CANONICALIZATION = (
    "sha256(json.dumps(payload_without_registry_hash, sort_keys=True, "
    "separators=(',',':'), ensure_ascii=False).encode('utf-8'))"
)


class AnchorError(RuntimeError):
    """The anchor cannot be loaded or violates its frozen identity."""


class AnchorMismatch(AnchorError):
    """A checkpoint/config/registry value differs from the frozen M5D-R0 identity."""


# ---------------------------------------------------------------------------
# hashing helpers
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# checkout provenance (computed, never guessed)
# ---------------------------------------------------------------------------
def _git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnchorError(f"git {' '.join(args)} failed in {repo}: {exc}") from exc
    return completed.stdout.strip()


def python_driver_git_sha(repo: Path | None = None) -> str:
    """Commit SHA of the checkout that contains this driver (recorded at build time)."""
    return _git(repo or REPO_ROOT, "rev-parse", "HEAD")


def libriichi_git_sha(repo: Path | None = None) -> str:
    """Pinned ``vendor/mortal`` submodule commit (dirty content is hashed separately)."""
    root = repo or REPO_ROOT
    return _git(root / "vendor" / "mortal", "rev-parse", "HEAD")


def libriichi_build_sha256(repo: Path | None = None) -> str:
    root = repo or REPO_ROOT
    path = root / "vendor" / "mortal" / "target" / "release" / "libriichi.so"
    if not path.is_file():
        raise AnchorMismatch(f"missing native build {path}")
    return sha256_file(path)


def patch_manifest(repo: Path | None = None) -> dict[str, str]:
    """Sorted ``patches/*.patch`` -> SHA-256 mapping (the applied vendor changes)."""
    root = repo or REPO_ROOT
    directory = root / "patches"
    if not directory.is_dir():
        raise AnchorError(f"missing patch directory {directory}")
    manifest = {}
    for path in sorted(directory.glob("*.patch")):
        manifest[path.name] = sha256_file(path)
    if not manifest:
        raise AnchorError("no patches found")
    return manifest


def patch_manifest_hash(repo: Path | None = None) -> str:
    return sha256_text(canonical_json(patch_manifest(repo)))


def tenhou_approx_rules_hash() -> str:
    """Canonical hash of the runtime ruleset metadata (ruleset id + deviations)."""
    from ..selfplay.provenance import ruleset_metadata

    return sha256_text(canonical_json(ruleset_metadata(RULESET_ID)))


# ---------------------------------------------------------------------------
# model config
# ---------------------------------------------------------------------------
def _require(mapping: Mapping, key: str, where: str):
    if key not in mapping:
        raise AnchorMismatch(f"checkpoint {where} is missing {key!r}")
    return mapping[key]


def model_config_from_state(state: Mapping) -> dict[str, Any]:
    """Extract the exact architecture-relevant subset from a Mortal checkpoint."""
    if not isinstance(state, Mapping):
        raise AnchorMismatch("checkpoint payload is not a mapping")
    config = _require(state, "config", "payload")
    if not isinstance(config, Mapping):
        raise AnchorMismatch("checkpoint config is not a mapping")
    control = _require(config, "control", "config")
    resnet = _require(config, "resnet", "config")
    model = _require(config, "model", "config")
    version = _require(control, "version", "config.control")
    conv_channels = _require(resnet, "conv_channels", "config.resnet")
    num_blocks = _require(resnet, "num_blocks", "config.resnet")
    use_event_branch = _require(model, "use_event_branch", "config.model")
    if isinstance(version, bool) or not isinstance(version, int):
        raise AnchorMismatch(f"control.version must be int, got {version!r}")
    for name, value in (("conv_channels", conv_channels), ("num_blocks", num_blocks)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AnchorMismatch(f"resnet.{name} must be a positive int, got {value!r}")
    if not isinstance(use_event_branch, bool):
        raise AnchorMismatch(
            f"model.use_event_branch must be bool, got {use_event_branch!r}")
    return {
        "control_version": version,
        "resnet": {"conv_channels": conv_channels, "num_blocks": num_blocks},
        "model": {
            "use_event_branch": use_event_branch,
            "event_dim": int(model.get("event_dim", 0)),
            "event_ffn": int(model.get("event_ffn", 0)),
            "event_heads": int(model.get("event_heads", 0)),
            "event_layers": int(model.get("event_layers", 0)),
        },
    }


def model_config_hash(model_config: Mapping) -> str:
    return sha256_text(canonical_json(dict(model_config)))


def _arch(model_config: Mapping) -> str:
    resnet = model_config["resnet"]
    return (f"resnet{{conv_channels={resnet['conv_channels']},"
            f"num_blocks={resnet['num_blocks']}}}")


# ---------------------------------------------------------------------------
# checkpoint identity / load
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckpointIdentity:
    path: str
    sha256: str
    bytes: int
    steps: int
    version: int
    conv_channels: int
    num_blocks: int
    use_event_branch: bool
    model_arch: str
    model_config: dict[str, Any]
    model_config_hash: str

    def as_dict(self) -> dict:
        return {
            "checkpoint_path": self.path,
            "checkpoint_sha256": self.sha256,
            "checkpoint_bytes": self.bytes,
            "steps": self.steps,
            "version": self.version,
            "model_arch": self.model_arch,
            "model_config": self.model_config,
            "model_config_hash": self.model_config_hash,
            "use_event_branch": self.use_event_branch,
        }


def checkpoint_identity(path: str | Path) -> CheckpointIdentity:
    """Hash + config-only identity; does not import torch or build the network."""
    path = Path(path)
    if not path.is_file():
        raise AnchorMismatch(f"checkpoint not found: {path}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency here
        raise AnchorError(f"torch is required to identify the checkpoint: {exc}") from exc
    try:
        state = torch.load(str(path), weights_only=True, map_location="cpu")
    except Exception as exc:
        raise AnchorError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise AnchorError(f"checkpoint {path} payload is not a dict")
    for key in ("mortal", "current_dqn"):
        if key not in state:
            raise AnchorMismatch(f"checkpoint {path} is missing {key!r}")
    model_config = model_config_from_state(state)
    steps = state.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise AnchorMismatch(f"checkpoint steps must be int, got {steps!r}")
    identity = CheckpointIdentity(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        steps=steps,
        version=model_config["control_version"],
        conv_channels=model_config["resnet"]["conv_channels"],
        num_blocks=model_config["resnet"]["num_blocks"],
        use_event_branch=model_config["model"]["use_event_branch"],
        model_arch=_arch(model_config),
        model_config=model_config,
        model_config_hash=model_config_hash(model_config),
    )
    return identity


def validate_checkpoint_identity(
    identity: CheckpointIdentity,
    *,
    expected_sha256: str = DEFAULT_CHECKPOINT_SHA256,
    expected_steps: int = DEFAULT_STEPS,
    expected_version: int = DEFAULT_VERSION,
    expected_conv_channels: int = DEFAULT_CONV_CHANNELS,
    expected_num_blocks: int = DEFAULT_NUM_BLOCKS,
    require_no_event_branch: bool = True,
) -> None:
    checks = (
        ("sha256", identity.sha256, str(expected_sha256).lower()),
        ("steps", identity.steps, int(expected_steps)),
        ("version", identity.version, int(expected_version)),
        ("conv_channels", identity.conv_channels, int(expected_conv_channels)),
        ("num_blocks", identity.num_blocks, int(expected_num_blocks)),
    )
    for name, actual, expected in checks:
        if actual != expected:
            raise AnchorMismatch(
                f"M5D-R0 {name} mismatch: {actual!r} != expected {expected!r}")
    if require_no_event_branch and identity.use_event_branch is not False:
        raise AnchorMismatch(
            "M5D-R0 must be a pure CNN checkpoint: "
            f"[model].use_event_branch={identity.use_event_branch!r}")
    if identity.model_config_hash != model_config_hash(EXPECTED_MODEL_CONFIG):
        raise AnchorMismatch(
            "M5D-R0 model_config_hash "
            f"{identity.model_config_hash} != frozen "
            f"{model_config_hash(EXPECTED_MODEL_CONFIG)}")


def _load_mortal_modules():
    """Import vendor ``model.py``/``engine.py`` by path (no ``MORTAL_CFG`` needed)."""
    try:
        import libriichi  # noqa: F401
    except Exception as exc:
        raise AnchorError(
            "libriichi is not importable; run with "
            "PYTHONPATH=vendor/mortal/target/release "
            f"({type(exc).__name__}: {exc})") from exc
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise AnchorError(f"torch is required: {exc}") from exc
    directory = REPO_ROOT / "vendor" / "mortal" / "mortal"
    # `model.py`/`engine.py` import the sibling `event_data` by its bare name.
    # Load that one module explicitly and restore the previous import state,
    # rather than putting the whole vendor directory on sys.path where it could
    # shadow project modules for later callers.
    event_path = directory / "event_data.py"
    previous_event = sys.modules.get("event_data")
    if event_path.is_file():
        event_spec = importlib.util.spec_from_file_location(
            "event_data", event_path)
        if event_spec is None or event_spec.loader is None:
            raise AnchorError(f"cannot build an import spec for {event_path}")
        event_module = importlib.util.module_from_spec(event_spec)
        sys.modules["event_data"] = event_module
        try:
            event_spec.loader.exec_module(event_module)
        except Exception:
            if previous_event is None:
                sys.modules.pop("event_data", None)
            else:
                sys.modules["event_data"] = previous_event
            raise
    modules = {}
    try:
        for name in ("model", "engine"):
            path = directory / f"{name}.py"
            if not path.is_file():
                raise AnchorError(f"missing vendor module {path}")
            module_name = f"_rf46_mortal_{name}"
            if module_name in sys.modules:
                modules[name] = sys.modules[module_name]
                continue
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise AnchorError(f"cannot build an import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                del sys.modules[module_name]
                raise
            modules[name] = module
    finally:
        if previous_event is None:
            sys.modules.pop("event_data", None)
        else:
            sys.modules["event_data"] = previous_event
    return modules["model"], modules["engine"]


@dataclass
class LoadedMortalCheckpoint:
    identity: CheckpointIdentity
    brain: Any
    dqn: Any
    model_module: Any
    engine_module: Any
    parameter_count: int

    @property
    def version(self) -> int:
        return self.identity.version

    def assert_cpu(self) -> None:
        for name, module in (("brain", self.brain), ("dqn", self.dqn)):
            for parameter in module.parameters():
                if parameter.device.type != "cpu":
                    raise AnchorError(f"{name} left the CPU ({parameter.device})")


def load_mortal_checkpoint(
    path: str | Path = DEFAULT_CHECKPOINT_PATH,
    *,
    expected_sha256: str = DEFAULT_CHECKPOINT_SHA256,
    expected_steps: int = DEFAULT_STEPS,
    expected_version: int = DEFAULT_VERSION,
    expected_conv_channels: int = DEFAULT_CONV_CHANNELS,
    expected_num_blocks: int = DEFAULT_NUM_BLOCKS,
    require_no_event_branch: bool = True,
) -> LoadedMortalCheckpoint:
    """Validate the exact file identity, then build ``Brain``/``DQN`` on CPU only."""
    import torch

    identity = checkpoint_identity(path)
    validate_checkpoint_identity(
        identity,
        expected_sha256=expected_sha256,
        expected_steps=expected_steps,
        expected_version=expected_version,
        expected_conv_channels=expected_conv_channels,
        expected_num_blocks=expected_num_blocks,
        require_no_event_branch=require_no_event_branch,
    )
    model_module, engine_module = _load_mortal_modules()
    # Re-read after the identity check so the validated bytes and the built weights are
    # provably the same file (no TOCTOU between hash and load).
    state = torch.load(str(path), weights_only=True, map_location=torch.device("cpu"))
    if sha256_file(path) != identity.sha256:
        raise AnchorMismatch("checkpoint changed between hash and model construction")
    brain = model_module.Brain(
        version=identity.version,
        conv_channels=identity.conv_channels,
        num_blocks=identity.num_blocks,
        is_oracle=False,
    ).to(torch.device("cpu")).eval()
    dqn = model_module.DQN(version=identity.version).to(torch.device("cpu")).eval()
    try:
        brain.load_state_dict(state["mortal"], strict=True)
        dqn.load_state_dict(state["current_dqn"], strict=True)
    except Exception as exc:
        raise AnchorMismatch(f"M5D-R0 state_dict does not match its arch: {exc}") from exc
    for module in (brain, dqn):
        for parameter in module.parameters():
            if not torch.isfinite(parameter).all():
                raise AnchorMismatch("M5D-R0 contains a non-finite parameter")
    parameter_count = sum(parameter.numel() for parameter in brain.parameters()) + sum(
        parameter.numel() for parameter in dqn.parameters())
    loaded = LoadedMortalCheckpoint(
        identity=identity, brain=brain, dqn=dqn,
        model_module=model_module, engine_module=engine_module,
        parameter_count=parameter_count)
    loaded.assert_cpu()
    return loaded


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------
class MortalCheckpointAdapter:
    """A frozen M5D-R0 seat engine, ``react_batch``-compatible with libriichi.

    ``libriichi.agent.MortalBatchAgent`` reads ``engine_type='mortal'``, ``name``,
    ``is_oracle``, ``version``, ``enable_quick_eval``,
    ``enable_rule_based_agari_guard`` and calls
    ``react_batch(obs, masks, invisible_obs)``.  One adapter serves every anchor
    seat of one role (the native arena batches the seats through one object).
    """

    engine_type = "mortal"
    is_oracle = False

    def __init__(
        self,
        loaded: LoadedMortalCheckpoint,
        *,
        name: str = DEFAULT_NAME,
        seat_index: int = 0,
        enable_quick_eval: bool = False,
        enable_rule_based_agari_guard: bool = False,
    ):
        if not isinstance(loaded, LoadedMortalCheckpoint):
            raise AnchorError("expected LoadedMortalCheckpoint")
        if not isinstance(name, str) or not name:
            raise AnchorError("name must be a non-empty string")
        if isinstance(seat_index, bool) or not isinstance(seat_index, int) \
                or not 0 <= seat_index <= 3:
            raise AnchorError(f"seat_index must be 0..3, got {seat_index!r}")
        self.loaded = loaded
        self.name = name
        self.seat_index = seat_index
        self.version = loaded.version
        self.enable_quick_eval = bool(enable_quick_eval)
        self.enable_rule_based_agari_guard = bool(enable_rule_based_agari_guard)
        self.inference_mode = "deterministic_argmax"
        self.engine = loaded.engine_module.MortalEngine(
            loaded.brain,
            loaded.dqn,
            is_oracle=False,
            version=loaded.version,
            device=__import__("torch").device("cpu"),
            stochastic_latent=False,
            enable_amp=False,
            enable_quick_eval=self.enable_quick_eval,
            enable_rule_based_agari_guard=self.enable_rule_based_agari_guard,
            name=self.name,
            boltzmann_epsilon=0.0,
            boltzmann_temp=1.0,
            top_p=1.0,
        )
        import inspect
        self._engine_accepts_events = (
            "events" in inspect.signature(self.engine.react_batch).parameters)

    # -- native BatchAgent surface ----------------------------------------
    def react_batch(self, obs, masks, invisible_obs=None, events=None):
        if invisible_obs is not None:
            raise AnchorError("M5D-R0 is not an oracle and must not receive hidden obs")
        # New native builds pass public event tokens as a fourth argument; the
        # frozen CNN anchor has no event branch, so the Brain ignores them.  Keep
        # the old three-argument build working when events is None.
        if self._engine_accepts_events:
            return self.engine.react_batch(obs, masks, invisible_obs, events=events)
        if events is not None:
            raise AnchorError(
                "native build passed event tokens but this engine has no events API")
        return self.engine.react_batch(obs, masks, invisible_obs)

    # -- introspection -----------------------------------------------------
    @property
    def identity(self) -> CheckpointIdentity:
        return self.loaded.identity

    def as_dict(self) -> dict:
        return {
            "anchor_id": ANCHOR_ID,
            "name": self.name,
            "seat_index": self.seat_index,
            "engine_type": self.engine_type,
            "version": self.version,
            "inference_mode": self.inference_mode,
            "enable_quick_eval": self.enable_quick_eval,
            "enable_rule_based_agari_guard": self.enable_rule_based_agari_guard,
            "stochastic_latent": False,
            "boltzmann_epsilon": 0.0,
            "device": "cpu",
            "checkpoint_sha256": self.identity.sha256,
            "model_config_hash": self.identity.model_config_hash,
            "parameter_count": self.loaded.parameter_count,
        }

    @classmethod
    def from_checkpoint(
        cls, path: str | Path = DEFAULT_CHECKPOINT_PATH, **kwargs
    ) -> "MortalCheckpointAdapter":
        loaded = load_mortal_checkpoint(path)
        adapter_kwargs = {key: kwargs.pop(key) for key in
                          ("name", "seat_index", "enable_quick_eval",
                           "enable_rule_based_agari_guard") if key in kwargs}
        if kwargs:
            raise AnchorError(f"unexpected adapter options: {sorted(kwargs)}")
        return cls(loaded, **adapter_kwargs)

    @classmethod
    def from_registry(cls, registry: Mapping, **kwargs) -> "MortalCheckpointAdapter":
        validated = validate_registry(registry)
        anchor = validated["anchor"]
        loaded = load_mortal_checkpoint(
            anchor["checkpoint_path"],
            expected_sha256=anchor["checkpoint_sha256"],
            expected_steps=anchor["steps"],
            expected_version=anchor["model_config"]["control_version"],
            expected_conv_channels=anchor["model_config"]["resnet"]["conv_channels"],
            expected_num_blocks=anchor["model_config"]["resnet"]["num_blocks"],
            require_no_event_branch=True,
        )
        return cls(loaded, **kwargs)


def build_champion_seats(
    adapter_or_loaded: MortalCheckpointAdapter | LoadedMortalCheckpoint,
    *,
    count: int = 3,
) -> tuple[MortalCheckpointAdapter, ...]:
    """Build ``count`` identical anchor seat engines sharing the frozen weights.

    The native ``OneVsThree`` API expects a single Python object for the champion
    role, and that object is batch-driven for all three seats; these per-seat
    wrappers exist so identity/mask tests can assert that every anchor seat is the
    same frozen model.  They share ``Brain``/``DQN`` tensors (no weight copies, no
    extra checkpoint reads).
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise AnchorError("count must be an int >= 1")
    if isinstance(adapter_or_loaded, MortalCheckpointAdapter):
        loaded = adapter_or_loaded.loaded
        name = adapter_or_loaded.name
        quick = adapter_or_loaded.enable_quick_eval
        guard = adapter_or_loaded.enable_rule_based_agari_guard
    elif isinstance(adapter_or_loaded, LoadedMortalCheckpoint):
        loaded = adapter_or_loaded
        name = DEFAULT_NAME
        quick = False
        guard = False
    else:
        raise AnchorError("expected MortalCheckpointAdapter or LoadedMortalCheckpoint")
    seats = tuple(
        MortalCheckpointAdapter(
            loaded, name=name, seat_index=index,
            enable_quick_eval=quick, enable_rule_based_agari_guard=guard)
        for index in range(count)
    )
    for seat in seats:
        for parameter, reference in zip(seat.loaded.brain.parameters(),
                                        loaded.brain.parameters()):
            if parameter is not reference:
                raise AnchorError("champion seats do not share the frozen weights")
    return seats


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def canonical_registry_payload(registry: Mapping) -> dict:
    if not isinstance(registry, Mapping):
        raise AnchorError("registry must be a mapping")
    payload = json.loads(json.dumps(dict(registry), sort_keys=True))
    payload.pop("registry_hash", None)
    return payload


def registry_hash(registry: Mapping) -> str:
    return sha256_text(canonical_json(canonical_registry_payload(registry)))


def _file_sha(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def build_baseline_registry(
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    python_driver_sha: str | None = None,
    repo: Path | None = None,
    extra_driver_files: Sequence[str] = (),
) -> dict:
    """Compute the complete M5D-R0 registry from the real files on disk.

    ``python_driver_sha`` is recorded explicitly (the frozen base commit) so the
    registry does not change merely because a later commit exists.  Per-file
    SHA-256s pin the exact driver source as a second, content-addressed identity.
    """
    root = (repo or REPO_ROOT).resolve()
    identity = checkpoint_identity(checkpoint_path)
    validate_checkpoint_identity(identity)
    driver_sha = python_driver_sha or python_driver_git_sha(root)
    driver_files = {}
    for relative in ("riichiformer/eval/mortal_anchor.py",
                     "riichiformer/eval/relative_pt.py",
                     "riichiformer/eval/m5d_arena.py",
                     "scripts/rf_eval_m5d.py"):
        digest = _file_sha(root / relative)
        if digest is not None:
            driver_files[relative] = digest
    for name in extra_driver_files:
        digest = _file_sha(root / str(name))
        if digest is not None:
            driver_files[str(name)] = digest
    registry = {
        "registry_schema": "rf-m5d-baseline-registry/1",
        "anchor_id": ANCHOR_ID,
        "role": DEFAULT_ROLE,
        "name": DEFAULT_NAME,
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": identity.sha256,
        "checkpoint_bytes": identity.bytes,
        "steps": identity.steps,
        "model_arch": identity.model_arch,
        "model_config": identity.model_config,
        "model_config_hash": identity.model_config_hash,
        "python_driver_git_sha": driver_sha,
        "python_driver_files": driver_files,
        "libriichi_git_sha": libriichi_git_sha(root),
        "libriichi_build_sha256": libriichi_build_sha256(root),
        "patch_manifest": patch_manifest(root),
        "patch_manifest_hash": patch_manifest_hash(root),
        "ruleset_id": RULESET_ID,
        "tenhou_approx_rules_hash": tenhou_approx_rules_hash(),
        "engine": {
            "engine_type": "mortal",
            "native_driver": "libriichi.arena.OneVsThree.py_vs_py",
            "version": identity.version,
            "is_oracle": False,
            "device": "cpu",
            "amp": False,
            "compile": False,
        },
        "inference_mode": "deterministic_argmax",
        "action_guard": "native_46_action_mask_only",
        "rule_based_agari_guard": False,
        "stochastic_latent": False,
        "boltzmann_epsilon": 0.0,
        "seed_policy": "per_game_and_seat",
        "permission": "local_research_only",
        "permission_note": (
            "Research scope only. This registry grants no licence, no redistribution "
            "right and no permission to publish the checkpoint or game logs."),
        "canonicalization": REGISTRY_CANONICALIZATION,
    }
    registry["registry_hash"] = registry_hash(registry)
    return registry


def validate_registry(registry: Mapping, *, check_files: bool = True) -> dict:
    """Verify a registry's structure, frozen identity and canonical hash."""
    if not isinstance(registry, Mapping):
        raise AnchorError("registry must be a mapping")
    data = json.loads(json.dumps(dict(registry), sort_keys=True))
    required = {
        "anchor_id", "role", "name", "checkpoint_path", "checkpoint_sha256",
        "steps", "model_arch", "model_config", "model_config_hash",
        "python_driver_git_sha", "libriichi_git_sha", "libriichi_build_sha256",
        "patch_manifest_hash", "ruleset_id", "tenhou_approx_rules_hash",
        "inference_mode", "action_guard", "seed_policy", "permission",
        "registry_hash",
    }
    missing = sorted(required - set(data))
    if missing:
        raise AnchorMismatch(f"registry is missing fields: {missing}")
    if data["anchor_id"] != ANCHOR_ID:
        raise AnchorMismatch(f"anchor_id {data['anchor_id']!r} != {ANCHOR_ID!r}")
    if data["role"] != DEFAULT_ROLE:
        raise AnchorMismatch(f"role {data['role']!r} != {DEFAULT_ROLE!r}")
    if data["permission"] != "local_research_only":
        raise AnchorMismatch(
            f"permission {data['permission']!r} != 'local_research_only'")
    if data["ruleset_id"] != RULESET_ID:
        raise AnchorMismatch(f"ruleset_id {data['ruleset_id']!r} != {RULESET_ID!r}")
    if data["inference_mode"] != "deterministic_argmax":
        raise AnchorMismatch(
            f"inference_mode {data['inference_mode']!r} != 'deterministic_argmax'")
    if data["seed_policy"] != "per_game_and_seat":
        raise AnchorMismatch(
            f"seed_policy {data['seed_policy']!r} != 'per_game_and_seat'")
    if data.get("rule_based_agari_guard") is not False:
        raise AnchorMismatch("M5D-R0 action guard must be the native mask only")
    if data["model_config"]["model"]["use_event_branch"] is not False:
        raise AnchorMismatch("[model].use_event_branch must be false for M5D-R0")
    expected_config = model_config_hash(data["model_config"])
    if data["model_config_hash"] != expected_config:
        raise AnchorMismatch("registry model_config_hash does not match model_config")
    if data["model_config_hash"] != model_config_hash(EXPECTED_MODEL_CONFIG):
        raise AnchorMismatch("registry model_config_hash is not the frozen M5D-R0 hash")
    if data["model_arch"] != _arch(data["model_config"]):
        raise AnchorMismatch("registry model_arch does not match model_config")
    if data["steps"] != DEFAULT_STEPS:
        raise AnchorMismatch(f"registry steps {data['steps']} != {DEFAULT_STEPS}")
    if data["checkpoint_sha256"].lower() != DEFAULT_CHECKPOINT_SHA256:
        raise AnchorMismatch("registry checkpoint SHA is not the frozen M5D-R0 SHA")
    if data["tenhou_approx_rules_hash"] != tenhou_approx_rules_hash():
        raise AnchorMismatch("registry rules hash does not match the runtime ruleset")
    if data["registry_hash"] != registry_hash(data):
        raise AnchorMismatch("registry canonical hash does not match its payload")
    if check_files:
        path = Path(data["checkpoint_path"])
        if not path.is_file():
            raise AnchorMismatch(f"registered checkpoint is missing: {path}")
        if sha256_file(path) != data["checkpoint_sha256"]:
            raise AnchorMismatch("registered checkpoint SHA does not match the file")
        if libriichi_build_sha256() != data["libriichi_build_sha256"]:
            raise AnchorMismatch("registered libriichi build SHA does not match the .so")
        if patch_manifest_hash() != data["patch_manifest_hash"]:
            raise AnchorMismatch("registered patch manifest hash drifted")
        if libriichi_git_sha() != data["libriichi_git_sha"]:
            raise AnchorMismatch("registered libriichi git SHA drifted")
    anchor = {
        "anchor_id": data["anchor_id"],
        "role": data["role"],
        "name": data["name"],
        "checkpoint_path": data["checkpoint_path"],
        "checkpoint_sha256": data["checkpoint_sha256"],
        "steps": data["steps"],
        "model_arch": data["model_arch"],
        "model_config": data["model_config"],
        "model_config_hash": data["model_config_hash"],
        "inference_mode": data["inference_mode"],
        "action_guard": data["action_guard"],
        "seed_policy": data["seed_policy"],
        "permission": data["permission"],
    }
    return {"anchor": anchor, "registry": data, "registry_hash": data["registry_hash"]}


def load_baseline_registry(path: str | Path, *, check_files: bool = True) -> dict:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise AnchorError(f"PyYAML is required to read {path}: {exc}") from exc
    path = Path(path)
    if not path.is_file():
        raise AnchorMismatch(f"baseline registry not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    validate_registry(payload, check_files=check_files)
    return payload


def write_baseline_registry(path: str | Path, registry: Mapping) -> str:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise AnchorError(f"PyYAML is required to write {path}: {exc}") from exc
    validate_registry(registry, check_files=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(registry), handle, sort_keys=True, allow_unicode=True)
    return sha256_file(path)


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="write the frozen M5D-R0 registry")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "configs" / "riichiformer" / "m5d_baseline.yaml"))
    parser.add_argument("--python-driver-sha", default=None)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args(argv)
    registry = build_baseline_registry(
        checkpoint_path=args.checkpoint,
        python_driver_sha=args.python_driver_sha,
    )
    if args.print:
        print(json.dumps(registry, indent=2, sort_keys=True))
    digest = write_baseline_registry(args.out, registry)
    print(f"M5D_REGISTRY {args.out} sha256={digest} "
          f"registry_hash={registry['registry_hash']} "
          f"checkpoint_sha256={registry['checkpoint_sha256']} "
          f"steps={registry['steps']} arch={registry['model_arch']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
