#!/usr/bin/env python3
"""PR-48 frozen B0 training protocol.

This module is the **code-only** half of PR-48.  It freezes, validates and
serialises the B0 protocol *before* any training result exists; it never starts
a full epoch by itself and it does not own a CLI (see ``scripts/rf_train_b0.py``).
Nothing here makes a strength claim: loss and validation CE select checkpoints,
they do not measure playing strength.

Frozen pieces
-------------
``B0TrainSettings``
    Every hyper-parameter that determines an optimizer trajectory.  The
    training path must be named explicitly and must be ``multi_decisions``
    (PR-38 shared-event path); the legacy per-decision prefix path is not a B0
    protocol and can never become the silent default.
``B0Protocol``
    The preregistration: experiment id, exactly two model RNG seeds, train and
    validation shard paths + file/manifest SHA-256, the full model config, the
    resolved parameter count, policy authentication, training settings, the
    fixed validation schedule, retained candidate steps, the single selection
    rule and the stop policy.  ``fingerprint()`` hashes the whole prereg and
    ``config_fingerprint()`` hashes the full training + model config.
``B0SelectionRule`` / ``B0Candidate`` / ``B0Selection``
    One selection metric (validation CE), one frozen weights source
    (``raw`` or ``ema``) and a deterministic tie-break chain.  Selection is a
    pure function of candidate records, so the same records always produce the
    same winner.
``B0StopPolicy`` / ``B0StopMonitor``
    OOM, non-finite, illegal action, wall-clock, VRAM and validation-regression
    budgets.  A triggered stop raises :class:`B0StopTriggered` with a
    serialisable event; it never silently continues.

Validation
----------
:func:`evaluate_multi_batches` runs a forward-only pass (``model.eval()``,
``torch.no_grad()``) over the **validation shard** using a separate read-only
loader.  It returns weighted CE, top-1, top-3, legal-action rate and
non-finite-output rate.  It takes no optimizer, performs no backward pass and
restores the previous ``train``/``eval`` mode, so validation data can never
update the optimizer, scheduler or EMA.

Resume
------
:func:`resume_checkpoint` is a strict resume: it verifies the checkpoint side
car SHA-256, the frozen schema hashes, the full training config, the model
config, the shard SHA-256/manifest SHA-256, the experiment/protocol/config
fingerprints, ``training_path`` and the seed, then restores model, optimizer,
scheduler, scaler, EMA and RNG.  A legacy checkpoint without B0 provenance or
with ``training_path=prefixes`` is refused instead of being relabelled.

Run manifests
-------------
:func:`write_prereg_markdown` writes the human preregistration template and
:func:`write_manifest` writes the machine-readable reference manifest.  Both
carry ``NOT_RUN`` items (full epoch / arena / strength) so a bounded CPU smoke
cannot be mistaken for the full B0 result.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from ..data.loader import MultiShardBatch, ShardReader
from ..data.shard import file_sha256, read_manifest
from ..models.policy import MVPConfig, Policy, count_parameters
from ..schemas.action_v3 import ACTION_SPACE
from .bc import TrainingError, training_step_multi, window_weight_total
from .checkpoint import (
    PROVENANCE_REQUIRED_KEYS,
    CheckpointInfo,
    CheckpointMismatch,
    config_hash,
    load_checkpoint,
    save_checkpoint,
)
from .optimizer import (
    TrainingConfig,
    make_optimizer,
    make_scheduler,
    set_seed,
)

#: Preregistration schema version (changing it invalidates old preregs).
B0_PREREG_VERSION = "rf-b0-prereg/1"
#: Run-manifest schema version.
B0_MANIFEST_VERSION = "rf-b0-manifest/1"
#: The only training path a B0 protocol accepts (PR-38 shared-forward).
B0_TRAINING_PATH = "multi_decisions"
#: Candidate metric names accepted by the frozen selection rule.
B0_SELECTION_METRICS = ("validation_ce",)
#: Weight sources a selection rule may freeze.
B0_WEIGHT_SOURCES = ("raw", "ema")
#: Canonical deterministic tie-break chain (after the primary metric).
B0_TIE_BREAK_CHAIN = ("other_weights_ce", "earlier_step", "lower_seed", "candidate_id")
#: Values that are always ``NOT_RUN`` in a B0 manifest at this stage.
B0_ALWAYS_NOT_RUN = ("arena_select", "arena_confirm", "strength_vs_m5d_r0")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPERIMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class B0ProtocolError(ValueError):
    """The preregistration is malformed or internally inconsistent."""


class B0Refusal(RuntimeError):
    """A B0 request violates the frozen protocol (budget, path, data)."""


class B0StopTriggered(RuntimeError):
    """A frozen stop condition fired; the run must not continue."""

    def __init__(self, event: "B0StopEvent"):
        super().__init__(f"B0 stop [{event.code}]: {event.detail}")
        self.event = event


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha256(value) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def _finite_or_none(value: float):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _metric_key(value: float) -> float:
    """Sort key for a metric where a non-finite value must never win."""
    return float(value) if math.isfinite(value) else math.inf


def _as_tuple(values) -> tuple:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise B0ProtocolError(f"expected a list/tuple, got {type(values).__name__}")
    return tuple(values)


def _require_sha(value, label: str) -> str:
    if not _is_sha256(value):
        raise B0ProtocolError(f"{label} must be a 64-character lowercase sha256")
    return value


def _require_nonempty(value, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise B0ProtocolError(f"{label} must be a non-empty string")
    return value


# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class B0TrainSettings:
    """Every frozen optimizer/dataloader knob of a B0 run.

    ``training_path`` has **no default**: a caller must state
    ``training_path="multi_decisions"`` explicitly, so the legacy prefix path
    can never be selected by omission.
    """

    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    warmup_steps: int
    grad_clip: float
    grad_accumulation: int
    batch_size: int
    max_steps: int
    bf16: bool
    ema_decay: float | None
    training_path: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.training_path != B0_TRAINING_PATH:
            raise B0ProtocolError(
                f"B0 requires training_path={B0_TRAINING_PATH!r} explicitly; got "
                f"{self.training_path!r} (the legacy prefix path is not a B0 protocol)")
        if not (isinstance(self.lr, (int, float)) and not isinstance(self.lr, bool)
                and self.lr > 0 and math.isfinite(self.lr)):
            raise B0ProtocolError("lr must be finite and > 0")
        if not (0 <= self.weight_decay < 1):
            raise B0ProtocolError("weight_decay must be in [0, 1)")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise B0ProtocolError("betas must be in [0, 1)")
        for name, value, minimum in (
                ("warmup_steps", self.warmup_steps, 0),
                ("grad_accumulation", self.grad_accumulation, 1),
                ("batch_size", self.batch_size, 1),
                ("max_steps", self.max_steps, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise B0ProtocolError(f"{name} must be an int >= {minimum}")
        if not (self.grad_clip > 0 and math.isfinite(self.grad_clip)):
            raise B0ProtocolError("grad_clip must be finite and > 0")
        if not isinstance(self.bf16, bool):
            raise B0ProtocolError("bf16 must be bool")
        if self.ema_decay is not None and not (0 <= self.ema_decay < 1):
            raise B0ProtocolError("ema_decay must be None or in [0, 1)")
        if self.device not in ("cpu", "cuda"):
            raise B0ProtocolError("device must be cpu or cuda")

    @property
    def effective_batch_decisions(self) -> int:
        return int(self.batch_size) * int(self.grad_accumulation)

    def as_dict(self) -> dict:
        return {
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "beta1": float(self.beta1),
            "beta2": float(self.beta2),
            "warmup_steps": int(self.warmup_steps),
            "grad_clip": float(self.grad_clip),
            "grad_accumulation": int(self.grad_accumulation),
            "batch_size": int(self.batch_size),
            "max_steps": int(self.max_steps),
            "bf16": bool(self.bf16),
            "ema_decay": None if self.ema_decay is None else float(self.ema_decay),
            "training_path": self.training_path,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "B0TrainSettings":
        expected = set(cls.__dataclass_fields__)
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown or missing:
            raise B0ProtocolError(
                f"B0TrainSettings keys mismatch: unknown={sorted(unknown)} "
                f"missing={sorted(missing)}")
        return cls(**{key: value[key] for key in expected})

    def training_config(self, seed: int) -> TrainingConfig:
        """Low-level PR-14 config for one seed (also stored in the checkpoint)."""
        return TrainingConfig(
            seed=seed, lr=float(self.lr), weight_decay=float(self.weight_decay),
            beta1=float(self.beta1), beta2=float(self.beta2),
            warmup_steps=int(self.warmup_steps), grad_clip=float(self.grad_clip),
            grad_accumulation=int(self.grad_accumulation), bf16=bool(self.bf16),
            ema_decay=None if self.ema_decay is None else float(self.ema_decay),
            device=self.device)


# ---------------------------------------------------------------------------
# Stop policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class B0StopPolicy:
    """Frozen stop budgets; any breach ends the run with a recorded event."""

    max_wall_seconds: float
    max_nonfinite_events: int = 0
    max_illegal_events: int = 0
    max_peak_vram_mb: float | None = None
    validation_regression_patience: int = 2
    validation_regression_delta: float = 0.02
    stop_on_oom: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_wall_seconds, bool) or not isinstance(
                self.max_wall_seconds, (int, float)) \
                or not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0:
            raise B0ProtocolError("max_wall_seconds must be finite and > 0")
        for name, value in (("max_nonfinite_events", self.max_nonfinite_events),
                            ("max_illegal_events", self.max_illegal_events),
                            ("validation_regression_patience",
                             self.validation_regression_patience)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise B0ProtocolError(f"{name} must be an int >= 0")
        if self.max_peak_vram_mb is not None:
            if isinstance(self.max_peak_vram_mb, bool) or not isinstance(
                    self.max_peak_vram_mb, (int, float)) \
                    or not math.isfinite(self.max_peak_vram_mb) \
                    or self.max_peak_vram_mb <= 0:
                raise B0ProtocolError("max_peak_vram_mb must be None or > 0")
        if not (isinstance(self.validation_regression_delta, (int, float))
                and not isinstance(self.validation_regression_delta, bool)
                and math.isfinite(self.validation_regression_delta)
                and self.validation_regression_delta >= 0):
            raise B0ProtocolError("validation_regression_delta must be finite and >= 0")
        if not isinstance(self.stop_on_oom, bool):
            raise B0ProtocolError("stop_on_oom must be bool")

    def as_dict(self) -> dict:
        return {
            "max_wall_seconds": float(self.max_wall_seconds),
            "max_nonfinite_events": int(self.max_nonfinite_events),
            "max_illegal_events": int(self.max_illegal_events),
            "max_peak_vram_mb": (None if self.max_peak_vram_mb is None
                                 else float(self.max_peak_vram_mb)),
            "validation_regression_patience": int(self.validation_regression_patience),
            "validation_regression_delta": float(self.validation_regression_delta),
            "stop_on_oom": bool(self.stop_on_oom),
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "B0StopPolicy":
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise B0ProtocolError(
                f"B0StopPolicy keys mismatch: unknown={sorted(set(value) - expected)} "
                f"missing={sorted(expected - set(value))}")
        return cls(**{key: value[key] for key in expected})


@dataclass(frozen=True)
class B0StopEvent:
    """One recorded stop-condition event (serialisable for the manifest)."""

    code: str
    detail: str
    step: int
    metrics: Mapping | None = None

    def as_dict(self) -> dict:
        return {"code": self.code, "detail": self.detail, "step": int(self.step),
                "metrics": dict(self.metrics or {})}


class B0StopMonitor:
    """Accumulates stop-condition events and raises on a budget breach."""

    def __init__(self, policy: B0StopPolicy, *, device: str = "cpu",
                 clock: Callable[[], float] = time.monotonic,
                 started_seconds: float | None = None):
        if not isinstance(policy, B0StopPolicy):
            raise B0ProtocolError("policy must be B0StopPolicy")
        if device not in ("cpu", "cuda"):
            raise B0ProtocolError("device must be cpu or cuda")
        self.policy = policy
        self.device = device
        self._clock = clock
        self.started = float(started_seconds if started_seconds is not None else clock())
        self.events: list[B0StopEvent] = []
        self.best_ce: float | None = None
        self.regression_count = 0
        self.nonfinite_total = 0
        self.illegal_total = 0

    # -- accounting -----------------------------------------------------
    @property
    def elapsed_seconds(self) -> float:
        return float(self._clock() - self.started)

    def record_nonfinite(self, step: int, count: int, *, detail: str = "") -> None:
        if count <= 0:
            return
        self.nonfinite_total += int(count)
        self.events.append(B0StopEvent(
            "nonfinite", detail or f"{count} non-finite validation/output values",
            step, {"count": int(count), "total": self.nonfinite_total}))
        if self.nonfinite_total > self.policy.max_nonfinite_events:
            raise B0StopTriggered(self.events[-1])

    def record_illegal(self, step: int, count: int, *, detail: str = "") -> None:
        if count <= 0:
            return
        self.illegal_total += int(count)
        self.events.append(B0StopEvent(
            "illegal_action", detail or f"{count} supervised target(s) outside the "
            "legal mask", step, {"count": int(count), "total": self.illegal_total}))
        if self.illegal_total > self.policy.max_illegal_events:
            raise B0StopTriggered(self.events[-1])

    def check_wall(self, step: int) -> None:
        elapsed = self.elapsed_seconds
        if elapsed > self.policy.max_wall_seconds:
            event = B0StopEvent(
                "wall_budget", f"elapsed {elapsed:.3f}s > budget "
                f"{self.policy.max_wall_seconds:.3f}s", step,
                {"elapsed_seconds": elapsed,
                 "budget_seconds": float(self.policy.max_wall_seconds)})
            self.events.append(event)
            raise B0StopTriggered(event)

    def check_vram(self, step: int) -> None:
        budget = self.policy.max_peak_vram_mb
        if budget is None or self.device != "cuda" or not torch.cuda.is_available():
            return
        peak = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        if peak > budget:
            event = B0StopEvent(
                "vram_budget", f"peak VRAM {peak:.1f} MB > budget {budget:.1f} MB",
                step, {"peak_vram_mb": peak, "budget_vram_mb": float(budget)})
            self.events.append(event)
            raise B0StopTriggered(event)

    def record_oom(self, step: int, *, detail: str = "") -> None:
        if not self.policy.stop_on_oom:
            return
        event = B0StopEvent("oom", detail or "CUDA out of memory", step, {})
        self.events.append(event)
        raise B0StopTriggered(event)

    def observe_validation(self, metrics: "ValidationMetrics", *, step: int) -> None:
        """Track the validation-CE regression budget over scheduled checks."""
        if not math.isfinite(metrics.ce):
            event = B0StopEvent(
                "nonfinite", "validation CE is not finite", step,
                {"nonfinite": int(metrics.nonfinite)})
            self.events.append(event)
            raise B0StopTriggered(event)
        if self.best_ce is None or metrics.ce < self.best_ce:
            self.best_ce = float(metrics.ce)
            self.regression_count = 0
            return
        if metrics.ce > self.best_ce + self.policy.validation_regression_delta:
            self.regression_count += 1
            self.events.append(B0StopEvent(
                "validation_regression",
                f"CE {metrics.ce:.6f} > best {self.best_ce:.6f} + "
                f"{self.policy.validation_regression_delta:.6f} "
                f"({self.regression_count} consecutive)",
                step, {"ce": float(metrics.ce), "best_ce": float(self.best_ce),
                       "regression_count": self.regression_count}))
            if self.regression_count > self.policy.validation_regression_patience:
                raise B0StopTriggered(self.events[-1])
        else:
            self.regression_count = 0

    # -- serialisation ---------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "best_ce": self.best_ce,
            "regression_count": int(self.regression_count),
            "events": [event.as_dict() for event in self.events],
        }


# ---------------------------------------------------------------------------
# Selection rule and candidate records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class B0SelectionRule:
    """The **only** model-selection rule: minimum validation CE, fixed weights.

    ``weights`` freezes whether the raw or the EMA weights are compared.  The
    other source's CE, then the earlier step, then the lower seed, then the
    candidate id resolve exact ties.  The rule is a pure function of the
    candidate records, so it cannot be changed after seeing results.
    """

    weights: str = "raw"
    metric: str = "validation_ce"
    tie_break: tuple[str, ...] = B0_TIE_BREAK_CHAIN

    def __post_init__(self) -> None:
        if self.weights not in B0_WEIGHT_SOURCES:
            raise B0ProtocolError(
                f"selection weights must be one of {B0_WEIGHT_SOURCES}, got "
                f"{self.weights!r}")
        if self.metric not in B0_SELECTION_METRICS:
            raise B0ProtocolError(
                f"selection metric must be {B0_SELECTION_METRICS}, got {self.metric!r}")
        if self.tie_break != B0_TIE_BREAK_CHAIN:
            raise B0ProtocolError(
                "the frozen tie-break chain is "
                f"{B0_TIE_BREAK_CHAIN}; got {self.tie_break!r}")

    @property
    def other_weights(self) -> str:
        return "ema" if self.weights == "raw" else "raw"

    def as_dict(self) -> dict:
        return {"weights": self.weights, "metric": self.metric,
                "tie_break": list(self.tie_break)}

    @classmethod
    def from_dict(cls, value: Mapping) -> "B0SelectionRule":
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise B0ProtocolError(
                f"B0SelectionRule keys mismatch: unknown="
                f"{sorted(set(value) - expected)} "
                f"missing={sorted(expected - set(value))}")
        return cls(weights=value["weights"], metric=value["metric"],
                   tie_break=_as_tuple(value["tie_break"]))

    def _key(self, candidate: "B0Candidate") -> tuple:
        primary = _metric_key(candidate.metrics_for(self.weights).ce)
        secondary = _metric_key(candidate.metrics_for(self.other_weights).ce)
        return (primary, secondary, int(candidate.step), int(candidate.seed),
                candidate.candidate_id)

    def select(self, candidates: Sequence["B0Candidate"], *,
               complete: bool = True) -> "B0Selection":
        """Deterministic winner over candidate records (never over results)."""
        candidates = list(candidates)
        if not candidates:
            raise B0ProtocolError("selection needs at least one candidate")
        experiments = {candidate.experiment_id for candidate in candidates}
        if len(experiments) != 1:
            raise B0ProtocolError(
                f"selection candidates span experiments {sorted(experiments)}")
        for candidate in candidates:
            candidate.metrics_for(self.weights)  # fail closed if the source is absent
            candidate.metrics_for(self.other_weights)
        ordered = sorted(candidates, key=self._key)
        if not math.isfinite(_metric_key(ordered[0].metrics_for(self.weights).ce)):
            raise B0ProtocolError("no candidate has a finite validation CE")
        winner = ordered[0]
        tie_break_used: list[str] = []
        if len(ordered) > 1:
            first, second = ordered
            primary = _metric_key(first.metrics_for(self.weights).ce)
            runner_primary = _metric_key(second.metrics_for(self.weights).ce)
            if primary == runner_primary:
                first_key = self._key(first)
                second_key = self._key(second)
                for index, name in enumerate(B0_TIE_BREAK_CHAIN):
                    if first_key[index + 1] != second_key[index + 1]:
                        tie_break_used = list(B0_TIE_BREAK_CHAIN[:index + 1])
                        break
        runner_up = ordered[1] if len(ordered) > 1 else None
        return B0Selection(
            rule=self, candidate_id=winner.candidate_id,
            experiment_id=winner.experiment_id, seed=int(winner.seed),
            step=int(winner.step), weights=self.weights, metric=self.metric,
            value=float(winner.metrics_for(self.weights).ce),
            other_weights=self.other_weights,
            other_value=float(winner.metrics_for(self.other_weights).ce),
            runner_up_id=(runner_up.candidate_id if runner_up is not None else None),
            runner_up_value=(float(runner_up.metrics_for(self.weights).ce)
                             if runner_up is not None else None),
            considered=tuple(candidate.candidate_id for candidate in ordered),
            tie_break_used=tuple(tie_break_used), complete=bool(complete))


@dataclass(frozen=True)
class ValidationMetrics:
    """Forward-only validation metrics over the independent validation shard."""

    step: int
    ce: float
    top1: float
    top3: float
    legal_rate: float
    nonfinite_rate: float
    decisions: int
    supervised: int
    legal: int
    nonfinite: int
    tokens: int
    weight: float

    def __post_init__(self) -> None:
        for name in ("step", "decisions", "supervised", "legal", "nonfinite",
                     "tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise B0ProtocolError(f"{name} must be an int >= 0")
        if not self.legal <= self.supervised <= self.decisions:
            raise B0ProtocolError(
                "validation counts must satisfy legal <= supervised <= decisions")
        if not self.nonfinite <= self.decisions:
            raise B0ProtocolError("nonfinite count must be <= decisions")
        for name in ("ce", "top1", "top3", "legal_rate", "nonfinite_rate", "weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise B0ProtocolError(f"{name} must be numeric")
        if self.weight < 0 or not math.isfinite(self.weight):
            raise B0ProtocolError("weight must be finite and >= 0")
        if not (0.0 <= self.legal_rate <= 1.0):
            raise B0ProtocolError("legal_rate must be in [0, 1]")
        if not (0.0 <= self.nonfinite_rate <= 1.0):
            raise B0ProtocolError("nonfinite_rate must be in [0, 1]")
        if math.isfinite(self.ce) and self.ce < 0:
            raise B0ProtocolError("CE must be >= 0 when finite")
        for name in ("top1", "top3"):
            value = getattr(self, name)
            if math.isfinite(value) and not (0.0 <= value <= 1.0):
                raise B0ProtocolError(f"{name} must be in [0, 1] when finite")

    @property
    def finite(self) -> bool:
        return self.nonfinite == 0 and math.isfinite(self.ce)

    def as_dict(self) -> dict:
        return {
            "step": int(self.step),
            "ce": float(self.ce),
            "top1": float(self.top1),
            "top3": float(self.top3),
            "legal_rate": float(self.legal_rate),
            "nonfinite_rate": float(self.nonfinite_rate),
            "decisions": int(self.decisions),
            "supervised": int(self.supervised),
            "legal": int(self.legal),
            "nonfinite": int(self.nonfinite),
            "tokens": int(self.tokens),
            "weight": float(self.weight),
        }

    def as_json_dict(self) -> dict:
        data = self.as_dict()
        for key in ("ce", "top1", "top3"):
            data[key] = _finite_or_none(data[key])
        return data

    @classmethod
    def from_dict(cls, value: Mapping) -> "ValidationMetrics":
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise B0ProtocolError(
                f"ValidationMetrics keys mismatch: unknown="
                f"{sorted(set(value) - expected)} "
                f"missing={sorted(expected - set(value))}")
        data = dict(value)
        for key in ("ce", "top1", "top3"):
            data[key] = math.nan if data[key] is None else float(data[key])
        return cls(**data)


@dataclass(frozen=True)
class B0Candidate:
    """One retained checkpoint with its raw and EMA validation metrics."""

    experiment_id: str
    seed: int
    step: int
    checkpoint_path: str
    checkpoint_sha256: str
    raw: ValidationMetrics
    ema: ValidationMetrics | None = None
    tokens: int = 0
    decisions: int = 0

    def __post_init__(self) -> None:
        _require_nonempty(self.experiment_id, "experiment_id")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) \
                or self.seed < 0:
            raise B0ProtocolError("seed must be an int >= 0")
        if isinstance(self.step, bool) or not isinstance(self.step, int) \
                or self.step < 1:
            raise B0ProtocolError("step must be an int >= 1")
        _require_nonempty(self.checkpoint_path, "checkpoint_path")
        _require_sha(self.checkpoint_sha256, "checkpoint_sha256")
        if not isinstance(self.raw, ValidationMetrics):
            raise B0ProtocolError("raw must be ValidationMetrics")
        if self.raw.step != self.step:
            raise B0ProtocolError("raw validation step must equal the candidate step")
        if self.ema is not None:
            if not isinstance(self.ema, ValidationMetrics):
                raise B0ProtocolError("ema must be ValidationMetrics")
            if self.ema.step != self.step:
                raise B0ProtocolError(
                    "ema validation step must equal the candidate step")
        for name in ("tokens", "decisions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise B0ProtocolError(f"{name} must be an int >= 0")

    @property
    def candidate_id(self) -> str:
        return f"{self.experiment_id}:seed{self.seed}:step{self.step:06d}"

    def metrics_for(self, weights: str) -> ValidationMetrics:
        if weights == "raw":
            return self.raw
        if weights == "ema":
            if self.ema is None:
                raise B0ProtocolError(
                    f"candidate {self.candidate_id} has no EMA validation metrics")
            return self.ema
        raise B0ProtocolError(f"unknown weights source {weights!r}")

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "experiment_id": self.experiment_id,
            "seed": int(self.seed),
            "step": int(self.step),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "raw": self.raw.as_json_dict(),
            "ema": None if self.ema is None else self.ema.as_json_dict(),
            "tokens": int(self.tokens),
            "decisions": int(self.decisions),
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "B0Candidate":
        expected = {"experiment_id", "seed", "step", "checkpoint_path",
                    "checkpoint_sha256", "raw", "ema", "tokens", "decisions"}
        if set(value) != expected:
            raise B0ProtocolError(
                f"B0Candidate keys mismatch: unknown={sorted(set(value) - expected)} "
                f"missing={sorted(expected - set(value))}")
        return cls(
            experiment_id=value["experiment_id"], seed=value["seed"],
            step=value["step"], checkpoint_path=value["checkpoint_path"],
            checkpoint_sha256=value["checkpoint_sha256"],
            raw=ValidationMetrics.from_dict(value["raw"]),
            ema=(ValidationMetrics.from_dict(value["ema"])
                 if value["ema"] is not None else None),
            tokens=value["tokens"], decisions=value["decisions"])


@dataclass(frozen=True)
class B0Selection:
    """The recorded selection outcome (one winner, one frozen rule)."""

    rule: B0SelectionRule
    candidate_id: str
    experiment_id: str
    seed: int
    step: int
    weights: str
    metric: str
    value: float
    other_weights: str
    other_value: float
    runner_up_id: str | None
    runner_up_value: float | None
    considered: tuple[str, ...]
    tie_break_used: tuple[str, ...]
    complete: bool

    def as_dict(self) -> dict:
        return {
            "rule": self.rule.as_dict(),
            "candidate_id": self.candidate_id,
            "experiment_id": self.experiment_id,
            "seed": int(self.seed),
            "step": int(self.step),
            "weights": self.weights,
            "metric": self.metric,
            "value": float(self.value),
            "other_weights": self.other_weights,
            "other_value": float(self.other_value),
            "runner_up_id": self.runner_up_id,
            "runner_up_value": (None if self.runner_up_value is None
                                else float(self.runner_up_value)),
            "considered": list(self.considered),
            "tie_break_used": list(self.tie_break_used),
            "complete": bool(self.complete),
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "B0Selection":
        expected = {"rule", "candidate_id", "experiment_id", "seed", "step",
                    "weights", "metric", "value", "other_weights", "other_value",
                    "runner_up_id", "runner_up_value", "considered",
                    "tie_break_used", "complete"}
        if set(value) != expected:
            raise B0ProtocolError(
                f"B0Selection keys mismatch: unknown={sorted(set(value) - expected)} "
                f"missing={sorted(expected - set(value))}")
        return cls(
            rule=B0SelectionRule.from_dict(value["rule"]),
            candidate_id=value["candidate_id"], experiment_id=value["experiment_id"],
            seed=value["seed"], step=value["step"], weights=value["weights"],
            metric=value["metric"], value=value["value"],
            other_weights=value["other_weights"], other_value=value["other_value"],
            runner_up_id=value["runner_up_id"],
            runner_up_value=value["runner_up_value"],
            considered=tuple(value["considered"]),
            tie_break_used=tuple(value["tie_break_used"]),
            complete=bool(value["complete"]))


# ---------------------------------------------------------------------------
# The frozen protocol
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class B0Protocol:
    """A complete preregistered B0 protocol (serialised before any run)."""

    experiment_id: str
    seeds: tuple[int, int]
    train_shard: str
    train_shard_sha256: str
    train_manifest_sha256: str
    validation_shard: str
    validation_shard_sha256: str
    validation_manifest_sha256: str
    model_config: dict
    model_config_hash: str
    parameter_count: int
    policy_config: str
    policy_hash: str
    train: B0TrainSettings
    validation_steps: tuple[int, ...]
    candidate_steps: tuple[int, ...]
    selection: B0SelectionRule
    stop_policy: B0StopPolicy
    created_utc: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        _require_nonempty(self.experiment_id, "experiment_id")
        if not _EXPERIMENT_RE.match(self.experiment_id):
            raise B0ProtocolError(
                "experiment_id must match [A-Za-z0-9][A-Za-z0-9._-]*")
        seeds = _as_tuple(self.seeds)
        if len(seeds) != 2:
            raise B0ProtocolError(
                f"B0 freezes exactly two model RNG seeds, got {len(seeds)}")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
               for seed in seeds):
            raise B0ProtocolError("seeds must be non-negative ints")
        if seeds[0] == seeds[1]:
            raise B0ProtocolError("the two model RNG seeds must differ")
        object.__setattr__(self, "seeds", (int(seeds[0]), int(seeds[1])))
        for label, value in (("train_shard", self.train_shard),
                             ("validation_shard", self.validation_shard)):
            if not isinstance(value, str) or not value:
                raise B0ProtocolError(f"{label} must be a non-empty string")
        if not isinstance(self.policy_config, str):
            raise B0ProtocolError("policy_config must be a string")
        if self.policy_hash:
            _require_sha(self.policy_hash, "policy_hash")
        if Path(self.train_shard).resolve() == Path(self.validation_shard).resolve():
            raise B0ProtocolError(
                "the validation shard must be a different file from the train shard")
        for label, value in (("train_shard_sha256", self.train_shard_sha256),
                             ("train_manifest_sha256", self.train_manifest_sha256),
                             ("validation_shard_sha256",
                              self.validation_shard_sha256),
                             ("validation_manifest_sha256",
                              self.validation_manifest_sha256),
                             ("model_config_hash", self.model_config_hash)):
            _require_sha(value, label)
        if self.train_shard_sha256 == self.validation_shard_sha256:
            raise B0ProtocolError(
                "train and validation shards must not share one file sha256")
        if not isinstance(self.model_config, Mapping) or not self.model_config:
            raise B0ProtocolError("model_config must be a non-empty mapping")
        model_config = dict(self.model_config)
        try:
            MVPConfig.from_mapping(model_config)
        except Exception as exc:  # noqa: BLE001 - re-raise as protocol error
            raise B0ProtocolError(f"invalid model_config: {exc}") from exc
        if config_hash(model_config) != self.model_config_hash:
            raise B0ProtocolError(
                "model_config_hash does not match the encoded model_config")
        if isinstance(self.parameter_count, bool) \
                or not isinstance(self.parameter_count, int) \
                or self.parameter_count <= 0:
            raise B0ProtocolError("parameter_count must be a positive int")
        if not isinstance(self.train, B0TrainSettings):
            raise B0ProtocolError("train must be B0TrainSettings")
        if not isinstance(self.selection, B0SelectionRule):
            raise B0ProtocolError("selection must be B0SelectionRule")
        if not isinstance(self.stop_policy, B0StopPolicy):
            raise B0ProtocolError("stop_policy must be B0StopPolicy")
        if not isinstance(self.created_utc, str) or not isinstance(self.notes, str):
            raise B0ProtocolError("created_utc/notes must be strings")
        self._validate_steps()

    def _validate_steps(self) -> None:
        max_steps = int(self.train.max_steps)
        validation = _as_tuple(self.validation_steps)
        candidates = _as_tuple(self.candidate_steps)
        for label, steps in (("validation_steps", validation),
                             ("candidate_steps", candidates)):
            if not steps:
                raise B0ProtocolError(f"{label} must not be empty")
            if any(isinstance(step, bool) or not isinstance(step, int) or step < 1
                   for step in steps):
                raise B0ProtocolError(f"{label} must contain positive ints")
            if any(left >= right for left, right in zip(steps, steps[1:])):
                raise B0ProtocolError(f"{label} must be strictly increasing")
            if steps[-1] > max_steps:
                raise B0ProtocolError(
                    f"{label} {steps[-1]} exceeds train.max_steps {max_steps}")
        if max_steps not in candidates:
            raise B0ProtocolError(
                "candidate_steps must include the final train.max_steps")
        missing = [step for step in candidates if step not in validation]
        if missing:
            raise B0ProtocolError(
                f"every candidate step needs a validation record; missing {missing}")
        object.__setattr__(self, "validation_steps", tuple(int(v) for v in validation))
        object.__setattr__(self, "candidate_steps", tuple(int(c) for c in candidates))

    # -- derived ---------------------------------------------------------
    def fingerprint(self) -> str:
        """SHA-256 of the entire prereg (``created_utc``/``notes`` excluded)."""
        return _sha(_canonical(self._fingerprint_payload()))

    def config_fingerprint(self) -> str:
        """SHA-256 of the full training settings + full model config."""
        return _sha(_canonical({"train": self.train.as_dict(),
                                "model": dict(self.model_config)}))

    def _fingerprint_payload(self) -> dict:
        data = self.as_dict()
        data.pop("created_utc", None)
        data.pop("notes", None)
        return data

    def model_config_object(self) -> MVPConfig:
        return MVPConfig.from_mapping(self.model_config)

    def training_config(self, seed: int) -> TrainingConfig:
        if seed not in self.seeds:
            raise B0ProtocolError(
                f"seed {seed} is not one of the frozen prereg seeds {self.seeds}")
        return self.train.training_config(int(seed))

    def is_candidate_step(self, step: int) -> bool:
        return int(step) in self.candidate_steps

    def is_validation_step(self, step: int) -> bool:
        return int(step) in self.validation_steps

    # -- serialisation ---------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "prereg_version": B0_PREREG_VERSION,
            "experiment_id": self.experiment_id,
            "seeds": [int(seed) for seed in self.seeds],
            "train_shard": self.train_shard,
            "train_shard_sha256": self.train_shard_sha256,
            "train_manifest_sha256": self.train_manifest_sha256,
            "validation_shard": self.validation_shard,
            "validation_shard_sha256": self.validation_shard_sha256,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            "model_config": dict(self.model_config),
            "model_config_hash": self.model_config_hash,
            "parameter_count": int(self.parameter_count),
            "policy_config": self.policy_config,
            "policy_hash": self.policy_hash,
            "train": self.train.as_dict(),
            "validation_steps": [int(v) for v in self.validation_steps],
            "candidate_steps": [int(step) for step in self.candidate_steps],
            "selection": self.selection.as_dict(),
            "stop_policy": self.stop_policy.as_dict(),
            "created_utc": self.created_utc,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "B0Protocol":
        expected = {
            "prereg_version", "experiment_id", "seeds", "train_shard",
            "train_shard_sha256", "train_manifest_sha256", "validation_shard",
            "validation_shard_sha256", "validation_manifest_sha256",
            "model_config", "model_config_hash", "parameter_count",
            "policy_config", "policy_hash", "train", "validation_steps",
            "candidate_steps", "selection", "stop_policy", "created_utc", "notes",
        }
        if set(value) != expected:
            raise B0ProtocolError(
                f"B0Protocol keys mismatch: unknown="
                f"{sorted(set(value) - expected)} "
                f"missing={sorted(expected - set(value))}")
        if value["prereg_version"] != B0_PREREG_VERSION:
            raise B0ProtocolError(
                f"unsupported prereg_version {value['prereg_version']!r}")
        return cls(
            experiment_id=value["experiment_id"],
            seeds=_as_tuple(value["seeds"]),
            train_shard=value["train_shard"],
            train_shard_sha256=value["train_shard_sha256"],
            train_manifest_sha256=value["train_manifest_sha256"],
            validation_shard=value["validation_shard"],
            validation_shard_sha256=value["validation_shard_sha256"],
            validation_manifest_sha256=value["validation_manifest_sha256"],
            model_config=dict(value["model_config"]),
            model_config_hash=value["model_config_hash"],
            parameter_count=value["parameter_count"],
            policy_config=value["policy_config"],
            policy_hash=value["policy_hash"],
            train=B0TrainSettings.from_dict(value["train"]),
            validation_steps=_as_tuple(value["validation_steps"]),
            candidate_steps=_as_tuple(value["candidate_steps"]),
            selection=B0SelectionRule.from_dict(value["selection"]),
            stop_policy=B0StopPolicy.from_dict(value["stop_policy"]),
            created_utc=value["created_utc"], notes=value["notes"])

    def to_json(self) -> str:
        return _canonical(self.as_dict())

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(self.to_json() + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "B0Protocol":
        path = Path(path)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise B0ProtocolError(f"prereg not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise B0ProtocolError(f"prereg is not valid JSON: {exc}") from exc
        if not isinstance(document, Mapping):
            raise B0ProtocolError("prereg must be a JSON object")
        return cls.from_dict(document)

    # -- human preregistration ------------------------------------------
    def prereg_markdown(self, *, base_sha: str = "", owner: str = "Dev-C") -> str:
        train = self.train
        model = dict(self.model_config)
        selection = self.selection.as_dict()
        stop = self.stop_policy.as_dict()
        lines = [
            f"# B0 preregistration — {self.experiment_id}",
            "",
            "Status: **PREREGISTERED / FROZEN BEFORE RESULTS**.  This file is a",
            "protocol, not a run record and not a strength result.  The full B0",
            "epoch, the arena and any M5D-R0 comparison are `NOT_RUN` unless a",
            "separate executed record says otherwise.",
            "",
            "## 1. Experiment metadata",
            "",
            f"- experiment_id: `{self.experiment_id}`",
            f"- prereg_version: `{B0_PREREG_VERSION}`",
            f"- created_utc: `{self.created_utc or 'UNSET'}`",
            f"- owner: `{owner}`",
            f"- base_sha: `{base_sha or 'UNSET'}`",
            f"- protocol_fingerprint: `{self.fingerprint()}`",
            f"- config_fingerprint: `{self.config_fingerprint()}`",
            f"- model_config_hash: `{self.model_config_hash}`",
            f"- parameter_count: `{self.parameter_count}`",
            f"- seeds: `{list(self.seeds)}` (two independent model RNG seeds; "
            f"same data, same step budget)",
            "",
            "## 2. Data",
            "",
            "| item | value |",
            "|---|---|",
            f"| train shard | `{self.train_shard}` |",
            f"| train shard sha256 | `{self.train_shard_sha256}` |",
            f"| train manifest sha256 | `{self.train_manifest_sha256}` |",
            f"| validation shard | `{self.validation_shard}` |",
            f"| validation shard sha256 | `{self.validation_shard_sha256}` |",
            f"| validation manifest sha256 | `{self.validation_manifest_sha256}` |",
            f"| policy config | `{self.policy_config}` |",
            f"| policy hash | `{self.policy_hash or 'N/A (licensed shard)'}` |",
            "",
            "The validation shard is read through a read-only audit reader and is",
            "never yielded to the training loop; no optimizer step may consume it.",
            "",
            "## 3. Model",
            "",
            "```json",
            json.dumps(model, indent=2, sort_keys=True),
            "```",
            "",
            f"Model-config hash `{self.model_config_hash}`; expected parameter",
            f"count `{self.parameter_count}` (asserted by the wrapper).",
            "",
            "## 4. Optimizer / training settings",
            "",
            "| item | value |",
            "|---|---|",
            f"| optimizer | AdamW |",
            f"| lr | `{train.lr}` |",
            f"| weight decay | `{train.weight_decay}` |",
            f"| betas | `({train.beta1}, {train.beta2})` |",
            f"| warmup steps | `{train.warmup_steps}` |",
            f"| grad clip | `{train.grad_clip}` |",
            f"| batch size (decisions) | `{train.batch_size}` |",
            f"| gradient accumulation | `{train.grad_accumulation}` "
            f"(window-weight normalized: `sum(numerator)/window_weight_total`) |",
            f"| effective decisions / optimizer step | `{train.effective_batch_decisions}` |",
            f"| precision | `{'bf16' if train.bf16 else 'fp32'}` "
            f"(grad scaler: `{'yes' if train.bf16 and train.device == 'cuda' else 'no'}`) |",
            f"| EMA decay | `{train.ema_decay}` |",
            f"| training path | `{B0_TRAINING_PATH}` (explicit; prefix path refused) |",
            f"| max optimizer steps | `{train.max_steps}` |",
            f"| device | `{train.device}` |",
            "",
            "## 5. Validation schedule and metrics",
            "",
            f"- validation steps: `{list(self.validation_steps)}`",
            "- metrics: weighted CE, top-1, top-3, legal-action rate,",
            "  non-finite-output rate, tokens/decisions",
            "- forward-only (`model.eval()`, `torch.no_grad()`): no backward pass,",
            "  no optimizer/scheduler/EMA update from validation data",
            "",
            "## 6. Candidate retention and the one selection rule",
            "",
            f"- retained candidate steps: `{list(self.candidate_steps)}`",
            "- every retained step has a validation record `(raw, EMA)`; a missing",
            "  candidate or validation record fails the run",
            "- selection metric: `validation_ce`",
            f"- selection weights source: `{selection['weights']}` (raw vs EMA is",
            "  recorded for every candidate; the frozen source decides)",
            f"- tie-break: `{' > '.join(selection['tie_break'])}` "
            "(exact ties only; deterministic)",
            "- no other model-selection rule exists in this protocol; changing the",
            "  rule requires a new experiment id and a new prereg",
            "",
            "## 7. Stop policy",
            "",
            "| condition | budget |",
            "|---|---|",
            f"| wall clock | `{stop['max_wall_seconds']}` s per seed |",
            f"| non-finite events | `{stop['max_nonfinite_events']}` |",
            f"| illegal actions | `{stop['max_illegal_events']}` |",
            f"| peak VRAM | `{stop['max_peak_vram_mb']}` MB |",
            f"| validation regression | `>{stop['validation_regression_patience']}` "
            f"consecutive checks above best + "
            f"`{stop['validation_regression_delta']}` CE |",
            f"| OOM | `{'stop' if stop['stop_on_oom'] else 'ignore'}` |",
            "",
            "A triggered stop ends the run with a recorded event (exit code 3); it",
            "is never resumed silently.",
            "",
            "## 8. Budget and what is NOT RUN",
            "",
            "- This prereg fixes the protocol.  Executing it needs an explicit",
            "  bounded `--max-steps`; the full budget additionally needs",
            "  `--allow-full-budget` and a user GPU authorization.",
            "- `NOT_RUN` until separately executed and recorded: full B0 epoch,",
            "  arena select/confirm, any M5D-R0 comparison, any strength claim.",
            "- Loss/CE only selects checkpoints; it is not evidence of strength.",
            "",
            "## 9. Frozen fingerprints",
            "",
            f"- protocol: `{self.fingerprint()}`",
            f"- training+model config: `{self.config_fingerprint()}`",
            "",
        ]
        if self.notes:
            lines.extend(["## 10. Notes", "", self.notes, ""])
        return "\n".join(lines)

    def write_prereg_markdown(self, path: str | Path, *, base_sha: str = "",
                              owner: str = "Dev-C") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(self.prereg_markdown(base_sha=base_sha, owner=owner),
                       encoding="utf-8")
        os.replace(tmp, path)
        return path


# ---------------------------------------------------------------------------
# Data / model verification
# ---------------------------------------------------------------------------
def _manifest_path(shard: str | Path) -> Path:
    return Path(str(shard) + ".manifest.json")


def build_protocol_from_files(
        *, experiment_id: str, train_shard: str, validation_shard: str,
        model_config: MVPConfig | Mapping, train: B0TrainSettings,
        seeds: Sequence[int] = (20261121, 20261122),
        validation_steps: Sequence[int] | None = None,
        candidate_steps: Sequence[int] | None = None,
        selection: B0SelectionRule | None = None,
        stop_policy: B0StopPolicy | None = None,
        policy_config: str = "", policy_hash: str = "",
        created_utc: str = "") -> B0Protocol:
    """Derive a complete protocol from real shard/model files (template path)."""
    if not isinstance(train, B0TrainSettings):
        raise B0ProtocolError("train must be B0TrainSettings")
    if isinstance(model_config, MVPConfig):
        config = model_config
    elif isinstance(model_config, Mapping):
        config = MVPConfig.from_mapping(model_config)
    else:
        raise B0ProtocolError("model_config must be MVPConfig or a mapping")
    config_dict = dataclasses.asdict(config)
    parameter_count = count_parameters(Policy(config))
    train_path = Path(train_shard)
    validation_path = Path(validation_shard)
    for label, path in (("train", train_path), ("validation", validation_path)):
        if not path.is_file():
            raise B0ProtocolError(f"{label} shard not found: {path}")
        if not _manifest_path(path).is_file():
            raise B0ProtocolError(f"{label} shard manifest not found: "
                                  f"{_manifest_path(path)}")
    validation = tuple(validation_steps) if validation_steps is not None \
        else (train.max_steps,)
    candidates = tuple(candidate_steps) if candidate_steps is not None \
        else tuple(validation)
    return B0Protocol(
        experiment_id=experiment_id,
        seeds=tuple(seeds),
        train_shard=str(train_path), train_shard_sha256=file_sha256(train_path),
        train_manifest_sha256=file_sha256(_manifest_path(train_path)),
        validation_shard=str(validation_path),
        validation_shard_sha256=file_sha256(validation_path),
        validation_manifest_sha256=file_sha256(_manifest_path(validation_path)),
        model_config=config_dict,
        model_config_hash=config_hash(config_dict),
        parameter_count=parameter_count,
        policy_config=policy_config, policy_hash=policy_hash,
        train=train, validation_steps=validation, candidate_steps=candidates,
        selection=selection if selection is not None else B0SelectionRule(),
        stop_policy=stop_policy if stop_policy is not None else B0StopPolicy(
            max_wall_seconds=14400.0),
        created_utc=created_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime()))


def verify_data_fingerprints(protocol: B0Protocol) -> dict:
    """Refuse a protocol whose shard files no longer match the prereg."""
    if not isinstance(protocol, B0Protocol):
        raise B0ProtocolError("protocol must be B0Protocol")
    result = {}
    for label, path, shard_sha, manifest_sha in (
            ("train", protocol.train_shard, protocol.train_shard_sha256,
             protocol.train_manifest_sha256),
            ("validation", protocol.validation_shard,
             protocol.validation_shard_sha256,
             protocol.validation_manifest_sha256)):
        source = Path(path)
        if not source.is_file():
            raise B0Refusal(f"{label} shard missing: {source}")
        manifest_path = _manifest_path(source)
        if not manifest_path.is_file():
            raise B0Refusal(f"{label} shard manifest missing: {manifest_path}")
        actual = file_sha256(source)
        if actual != shard_sha:
            raise B0Refusal(
                f"{label} shard sha256 mismatch: {actual} != prereg {shard_sha}")
        actual_manifest = file_sha256(manifest_path)
        if actual_manifest != manifest_sha:
            raise B0Refusal(
                f"{label} shard manifest sha256 mismatch: {actual_manifest} != "
                f"prereg {manifest_sha}")
        manifest = read_manifest(source)
        if manifest.get("shard_sha256") != shard_sha:
            raise B0Refusal(
                f"{label} shard manifest records sha256 "
                f"{manifest.get('shard_sha256')!r} != prereg {shard_sha}")
        result[label] = {"shard_sha256": actual,
                         "manifest_sha256": actual_manifest,
                         "metadata": manifest.get("metadata", {})}
    return result


def verify_model_fingerprint(protocol: B0Protocol) -> int:
    """Refuse a model config/parameter count that drifted from the prereg."""
    config = protocol.model_config_object()
    if config_hash(config) != protocol.model_config_hash:
        raise B0Refusal("model config hash drifted from the prereg")
    parameter_count = count_parameters(Policy(config))
    if parameter_count != protocol.parameter_count:
        raise B0Refusal(
            f"model parameter count {parameter_count} != prereg "
            f"{protocol.parameter_count}")
    return parameter_count


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class ModelEMA:
    """Plain exponential moving average of every floating model tensor."""

    def __init__(self, model: Policy, decay: float):
        if not isinstance(model, Policy):
            raise B0ProtocolError("EMA needs a Policy model")
        if not (0 <= decay < 1) or not math.isfinite(float(decay)):
            raise B0ProtocolError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.updates = 0
        self.shadow: dict[str, Tensor] = {}
        for name, value in model.state_dict().items():
            self.shadow[name] = value.detach().clone()

    @torch.no_grad()
    def update(self, model: Policy) -> None:
        state = model.state_dict()
        for name, value in state.items():
            if name not in self.shadow:
                raise B0ProtocolError(f"EMA shadow missing tensor {name}")
            shadow = self.shadow[name]
            if shadow.shape != value.shape:
                raise B0ProtocolError(f"EMA shape mismatch for {name}")
            detached = value.detach()
            if shadow.is_floating_point():
                shadow.mul_(self.decay).add_(detached, alpha=1.0 - self.decay)
            else:
                shadow.copy_(detached)
        self.updates += 1

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "updates": int(self.updates),
            "shadow": {name: tensor.detach().clone()
                       for name, tensor in self.shadow.items()},
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping) -> None:
        if not isinstance(state, Mapping):
            raise B0ProtocolError("EMA state must be a mapping")
        decay = state.get("decay")
        if decay is None or not (0 <= float(decay) < 1):
            raise B0ProtocolError("EMA state has an invalid decay")
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping) or set(shadow) != set(self.shadow):
            raise B0ProtocolError("EMA state shadow keys do not match the model")
        for name, tensor in shadow.items():
            if not isinstance(tensor, Tensor) or tensor.shape != self.shadow[name].shape:
                raise B0ProtocolError(f"EMA state tensor {name} is invalid")
        self.decay = float(decay)
        self.updates = int(state.get("updates", 0))
        for name, tensor in shadow.items():
            self.shadow[name].copy_(tensor)

    def copy_to(self, model: Policy) -> None:
        """Overwrite ``model`` weights with the EMA shadow (inference only)."""
        state = model.state_dict()
        if set(state) != set(self.shadow):
            raise B0ProtocolError("EMA shadow keys do not match the target model")
        model.load_state_dict({name: tensor.detach().clone()
                               for name, tensor in self.shadow.items()},
                              strict=True)

    def clone_model(self, model: Policy) -> Policy:
        clone = copy.deepcopy(model)
        self.copy_to(clone)
        return clone

    def as_metadata(self) -> dict:
        return {"ema_decay": self.decay, "ema_updates": int(self.updates)}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validation_weight(target: Mapping) -> float | None:
    """Weight of a supervised validation decision (mirrors the BC contract)."""
    if not bool(target.get("supervision_valid", False)):
        return None
    if target.get("forced") is True or target.get("loss_weight") == 0:
        return None
    weight = target.get("loss_weight", 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise B0ProtocolError(f"loss_weight must be numeric, got {weight!r}")
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise B0ProtocolError(f"loss_weight must be finite and >= 0, got {weight!r}")
    return weight if weight > 0 else None


def evaluate_multi_batches(
        model: Policy, batches: Iterable[MultiShardBatch], *,
        step: int, device: str = "cpu") -> ValidationMetrics:
    """Forward-only CE/top1/top3/legal/non-finite metrics (no optimizer).

    The model is switched to ``eval()`` and restored afterwards; no gradient is
    computed and no RNG/optimizer/EMA state is touched.
    """
    if not isinstance(model, Policy):
        raise B0ProtocolError("evaluate_multi_batches needs a Policy")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise B0ProtocolError("step must be an int >= 0")
    was_training = bool(model.training)
    model.eval()
    decisions = supervised = legal = nonfinite = tokens = 0
    ce_sum = 0.0
    weight_sum = 0.0
    top1_hits = 0
    top3_hits = 0
    try:
        with torch.no_grad():
            for shard_batch in batches:
                if not isinstance(shard_batch, MultiShardBatch):
                    raise B0ProtocolError(
                        "validation needs MultiShardBatch objects")
                batch = shard_batch.batch.to(device)
                targets = shard_batch.targets
                out = model.forward_multi(batch)
                tokens += int(shard_batch.tokens)
                decisions += len(targets)
                finite_row = (torch.isfinite(out.logits).all(dim=1)
                              & (torch.isfinite(out.logprobs)
                                 | ~out.legal_mask).all(dim=1))
                nonfinite += int((~finite_row).sum())
                top3 = out.masked_logits.topk(3, dim=-1).indices
                legal_mask = batch.legal_mask
                for index, target in enumerate(targets):
                    weight = _validation_weight(target)
                    if weight is None:
                        continue
                    supervised += 1
                    action = target.get("expert_action")
                    if isinstance(action, bool) or not isinstance(action, int) \
                            or not 0 <= action < ACTION_SPACE:
                        continue
                    if not bool(legal_mask[index, action]):
                        continue
                    legal += 1
                    if not bool(finite_row[index]):
                        continue
                    logprob = out.logprobs[index, action]
                    if not bool(torch.isfinite(logprob)):
                        continue
                    ce_sum += float(-logprob) * weight
                    weight_sum += weight
                    if int(top3[index, 0]) == action:
                        top1_hits += 1
                    if action in top3[index].tolist():
                        top3_hits += 1
    finally:
        model.train(was_training)
    ce = ce_sum / weight_sum if weight_sum > 0 else math.nan
    top1 = top1_hits / legal if legal > 0 else math.nan
    top3_acc = top3_hits / legal if legal > 0 else math.nan
    return ValidationMetrics(
        step=int(step), ce=float(ce), top1=float(top1), top3=float(top3_acc),
        legal_rate=(legal / supervised if supervised > 0 else 0.0),
        nonfinite_rate=(nonfinite / decisions if decisions > 0 else 0.0),
        decisions=decisions, supervised=supervised, legal=legal,
        nonfinite=nonfinite, tokens=tokens, weight=float(weight_sum))


@dataclass(frozen=True)
class CandidateEvaluation:
    """Raw and (optional) EMA validation metrics for one candidate step."""

    raw: ValidationMetrics
    ema: ValidationMetrics | None

    def as_dict(self) -> dict:
        return {"raw": self.raw.as_json_dict(),
                "ema": None if self.ema is None else self.ema.as_json_dict()}


def evaluate_candidate(
        model: Policy, batch_factory: Callable[[], Iterable[MultiShardBatch]], *,
        step: int, device: str = "cpu",
        ema: ModelEMA | None = None) -> CandidateEvaluation:
    """Evaluate raw weights and, when an EMA exists, a cloned EMA model.

    ``batch_factory`` must build a fresh validation iterator so the raw and EMA
    passes see exactly the same batches.
    """
    raw = evaluate_multi_batches(model, batch_factory(), step=step, device=device)
    ema_metrics = None
    if ema is not None:
        shadow = ema.clone_model(model)
        try:
            ema_metrics = evaluate_multi_batches(
                shadow, batch_factory(), step=step, device=device)
        finally:
            del shadow
    return CandidateEvaluation(raw=raw, ema=ema_metrics)


# ---------------------------------------------------------------------------
# One accumulated optimizer step
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowStepResult:
    """Aggregated result of one gradient-accumulation window."""

    loss: float
    numerator: float
    valid_decisions: int
    total_decisions: int
    weight_total: float
    microbatches: int
    optimizer_stepped: bool
    tokens: int

    def as_dict(self) -> dict:
        return {
            "loss": float(self.loss),
            "numerator": float(self.numerator),
            "valid_decisions": int(self.valid_decisions),
            "total_decisions": int(self.total_decisions),
            "weight_total": float(self.weight_total),
            "microbatches": int(self.microbatches),
            "optimizer_stepped": bool(self.optimizer_stepped),
            "tokens": int(self.tokens),
        }


def run_accumulated_step(
        model: Policy, micro_batches: Sequence, micro_targets: Sequence,
        *, optimizer: torch.optim.Optimizer, settings: B0TrainSettings,
        scheduler=None, ema: ModelEMA | None = None,
        accumulation_weight_total: float | None = None,
        tokens: int = 0) -> WindowStepResult:
    """One optimizer step over ``grad_accumulation`` microbatches.

    The denominator is the **window** weight sum (PR-39 contract), computed
    before the first forward pass, so ``k`` microbatches are numerically
    equivalent to one large batch.  An all-censored window performs no
    backward/optimizer/EMA update.
    """
    if not isinstance(settings, B0TrainSettings):
        raise B0ProtocolError("settings must be B0TrainSettings")
    micro_batches = list(micro_batches)
    micro_targets = list(micro_targets)
    if len(micro_batches) != len(micro_targets):
        raise B0ProtocolError("micro batches and targets must have equal length")
    if not micro_batches:
        raise B0ProtocolError("an accumulation window needs at least one microbatch")
    if len(micro_batches) != settings.grad_accumulation:
        raise B0ProtocolError(
            f"window has {len(micro_batches)} microbatches but grad_accumulation "
            f"is {settings.grad_accumulation}")
    expected_total = window_weight_total(micro_targets)
    if accumulation_weight_total is not None:
        if abs(float(accumulation_weight_total) - expected_total) > 1e-9:
            raise B0ProtocolError(
                "accumulation_weight_total does not match the window's target "
                f"weights ({accumulation_weight_total} != {expected_total})")
    window_total = expected_total
    numerator = 0.0
    valid = total = 0
    for index, (batch, targets) in enumerate(zip(micro_batches, micro_targets)):
        result = training_step_multi(
            model, batch, targets, optimizer=optimizer, scheduler=scheduler,
            max_grad_norm=settings.grad_clip, bf16=settings.bf16,
            accumulation_step=index, accumulation_count=len(micro_batches),
            accumulation_weight_total=window_total)
        valid += int(result.valid_decisions)
        total += int(result.total_decisions)
        if result.numerator is not None:
            numerator += float(result.numerator.detach())
    stepped = valid > 0
    if ema is not None and stepped:
        ema.update(model)
    loss = (numerator / window_total) if window_total > 0 else 0.0
    return WindowStepResult(
        loss=float(loss), numerator=float(numerator), valid_decisions=valid,
        total_decisions=total, weight_total=float(window_total),
        microbatches=len(micro_batches), optimizer_stepped=stepped,
        tokens=int(tokens))


# ---------------------------------------------------------------------------
# Checkpoint lifecycle
# ---------------------------------------------------------------------------
def checkpoint_provenance(protocol: B0Protocol, metadata) -> dict:
    """Full strict-resume provenance block for a B0 checkpoint."""
    provenance = {
        "source_id": metadata.source_id,
        "split_manifest_sha256": metadata.split_manifest_sha256,
        "canonical_manifest_sha256": metadata.canonical_manifest_sha256,
        "canonical_index_sha256": metadata.canonical_index_sha256,
        "research_policy": metadata.research_policy,
        "research_policy_hash": metadata.research_policy_hash,
        "external_license_status": metadata.external_license_status,
        "train_eligible_under_policy": (
            "true" if metadata.train_eligible_under_policy else "false"),
        "rights_status": str(metadata.rights_status).upper(),
        "training_path": B0_TRAINING_PATH,
        "shard_sha256": protocol.train_shard_sha256,
        "shard_manifest_sha256": protocol.train_manifest_sha256,
        "b0_experiment_id": protocol.experiment_id,
        "b0_protocol_fingerprint": protocol.fingerprint(),
        "b0_config_fingerprint": protocol.config_fingerprint(),
    }
    if metadata.policy_version:
        provenance["policy_version"] = metadata.policy_version
    missing = [key for key in PROVENANCE_REQUIRED_KEYS if key not in provenance]
    if missing:  # pragma: no cover - construction bug
        raise B0ProtocolError(f"provenance missing required keys {missing}")
    return provenance


def candidate_filename(step: int) -> str:
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise B0ProtocolError("candidate step must be a positive int")
    return f"step_{step:06d}.pt"


def save_candidate(
        path: str | Path, *, protocol: B0Protocol, metadata, model: Policy,
        optimizer, scheduler, ema: ModelEMA | None, seed: int, step: int,
        tokens: int, decisions: int,
        evaluation: CandidateEvaluation) -> B0Candidate:
    """Write one retained candidate checkpoint and its record."""
    if seed not in protocol.seeds:
        raise B0ProtocolError(f"seed {seed} is not in the prereg")
    if not protocol.is_candidate_step(step):
        raise B0ProtocolError(
            f"step {step} is not a preregistered candidate step")
    path = Path(path)
    provenance = checkpoint_provenance(protocol, metadata)
    hashes = {
        "schema_hash": metadata.schema_hash,
        "action_map_hash": metadata.action_map_hash,
        "normalizer_hash": metadata.normalizer_hash,
        "shard_sha256": protocol.train_shard_sha256,
        "shard_manifest_sha256": protocol.train_manifest_sha256,
    }
    extra = {
        "training_path": B0_TRAINING_PATH,
        "token_accounting": "shared_events_plus_markers",
        "shard_sha256": protocol.train_shard_sha256,
        "b0_experiment_id": protocol.experiment_id,
        "b0_protocol_fingerprint": protocol.fingerprint(),
        "b0_config_fingerprint": protocol.config_fingerprint(),
        "b0_seed": int(seed),
        "b0_step": int(step),
        "b0_parameter_count": int(protocol.parameter_count),
        "b0_validation_raw": evaluation.raw.as_json_dict(),
        "b0_validation_ema": (None if evaluation.ema is None
                              else evaluation.ema.as_json_dict()),
    }
    info = save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        counters={"step": int(step), "tokens": int(tokens),
                  "decisions": int(decisions)},
        hashes=hashes, config=protocol.training_config(seed), seed=int(seed),
        ema=ema, model_config=protocol.model_config, provenance=provenance,
        extra=extra)
    return B0Candidate(
        experiment_id=protocol.experiment_id, seed=int(seed), step=int(step),
        checkpoint_path=str(path), checkpoint_sha256=info.sha256,
        raw=evaluation.raw, ema=evaluation.ema, tokens=int(tokens),
        decisions=int(decisions))


def resume_checkpoint(
        path: str | Path, *, protocol: B0Protocol, metadata, seed: int,
        model: Policy, optimizer=None, scheduler=None, ema: ModelEMA | None = None,
        strict: bool = True) -> tuple[CheckpointInfo, dict]:
    """Strict B0 resume: provenance, config, model, shard and path identity.

    Raises :class:`CheckpointMismatch` for a legacy checkpoint without B0
    provenance, a ``training_path=prefixes`` checkpoint, another config/model or
    another train shard.  Never rewrites counters from a foreign run.
    """
    if not isinstance(protocol, B0Protocol):
        raise B0ProtocolError("protocol must be B0Protocol")
    if seed not in protocol.seeds:
        raise B0ProtocolError(f"seed {seed} is not in the prereg")
    if strict and optimizer is None:
        raise B0ProtocolError("a strict B0 resume needs the optimizer")
    provenance = checkpoint_provenance(protocol, metadata)
    info, payload = load_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, ema=ema,
        expected_hashes={
            "schema_hash": metadata.schema_hash,
            "action_map_hash": metadata.action_map_hash,
            "normalizer_hash": metadata.normalizer_hash,
        },
        expected_provenance=provenance,
        expected_config=protocol.training_config(seed),
        expected_model_config=protocol.model_config,
        expected_model_config_hash=protocol.model_config_hash,
        strict_resume=bool(strict))
    stored_path = (payload.get("extra") or {}).get("training_path")
    # Legacy checkpoints predate the field and can only be prefix checkpoints;
    # they are refused rather than relabelled onto the multi-decision path.
    if stored_path is None:
        stored_path = "prefixes"
    if stored_path != B0_TRAINING_PATH:
        raise CheckpointMismatch(
            f"checkpoint training_path {stored_path!r} != required "
            f"{B0_TRAINING_PATH!r}; cross-path resume is refused")
    if int(info.seed) != int(seed):
        raise CheckpointMismatch(
            f"checkpoint seed {info.seed} != requested seed {seed}")
    stored_extra = payload.get("extra") or {}
    for key, expected in (("shard_sha256", protocol.train_shard_sha256),
                          ("b0_protocol_fingerprint", protocol.fingerprint()),
                          ("b0_config_fingerprint", protocol.config_fingerprint())):
        if key in stored_extra and stored_extra[key] != expected:
            raise CheckpointMismatch(f"checkpoint extra[{key!r}] mismatch")
    return info, payload


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def not_run_items(protocol: B0Protocol, runs: Sequence[Mapping]) -> tuple[str, ...]:
    """``NOT_RUN`` list for a B0 manifest (never silently promoted to PASS)."""
    items = list(B0_ALWAYS_NOT_RUN)
    steps = [int(run.get("steps", 0)) for run in runs]
    if not steps or max(steps) < protocol.train.max_steps:
        items.insert(0, "full_b0_epoch")
    if any(str(run.get("device", "")) == "cpu" for run in runs) or not runs:
        items.append("gpu_training")
    return tuple(items)


def write_manifest(
        path: str | Path, *, protocol: B0Protocol, runs: Sequence[Mapping],
        selection: B0Selection | None, status: str,
        stop_events: Sequence[B0StopEvent] = (),
        candidates: Sequence[B0Candidate] = (),
        prereg_path: str | Path | None = None,
        extra: Mapping | None = None) -> dict:
    """Write the machine-readable B0 reference manifest (canonical JSON)."""
    if not isinstance(protocol, B0Protocol):
        raise B0ProtocolError("protocol must be B0Protocol")
    if status not in ("planned", "running", "completed", "partial", "stopped",
                      "incomplete", "failed"):
        raise B0ProtocolError(f"unknown manifest status {status!r}")
    prereg_sha = ""
    if prereg_path is not None:
        prereg_file = Path(prereg_path)
        if prereg_file.is_file():
            prereg_sha = file_sha256(prereg_file)
    manifest = {
        "format": B0_MANIFEST_VERSION,
        "experiment_id": protocol.experiment_id,
        "status": status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prereg": {
            "path": str(prereg_path) if prereg_path is not None else "",
            "sha256": prereg_sha,
            "fingerprint": protocol.fingerprint(),
            "config_fingerprint": protocol.config_fingerprint(),
        },
        "protocol": protocol.as_dict(),
        "model": {
            "model_config_hash": protocol.model_config_hash,
            "parameter_count": int(protocol.parameter_count),
            "model_config": dict(protocol.model_config),
        },
        "runs": [dict(run) for run in runs],
        "candidates": [candidate.as_dict() for candidate in candidates],
        "selection": None if selection is None else selection.as_dict(),
        "stop_events": [event.as_dict() for event in stop_events],
        "not_run": list(not_run_items(protocol, runs)),
        "strength_claim": "none",
        "gpu_used": any(str(run.get("device", "")) == "cuda" for run in runs),
    }
    if extra:
        manifest["extra"] = dict(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(_canonical(manifest) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return manifest


__all__ = [
    "B0_ALWAYS_NOT_RUN", "B0_MANIFEST_VERSION", "B0_PREREG_VERSION",
    "B0_SELECTION_METRICS", "B0_TIE_BREAK_CHAIN", "B0_TRAINING_PATH",
    "B0_WEIGHT_SOURCES", "B0Candidate", "B0Protocol", "B0ProtocolError",
    "B0Refusal", "B0Selection", "B0SelectionRule", "B0StopEvent",
    "B0StopMonitor", "B0StopPolicy", "B0StopTriggered", "B0TrainSettings",
    "CandidateEvaluation", "ModelEMA", "ValidationMetrics", "WindowStepResult",
    "build_protocol_from_files", "candidate_filename", "checkpoint_provenance",
    "evaluate_candidate", "evaluate_multi_batches", "not_run_items",
    "resume_checkpoint", "run_accumulated_step", "save_candidate",
    "verify_data_fingerprints", "verify_model_fingerprint", "write_manifest",
]
