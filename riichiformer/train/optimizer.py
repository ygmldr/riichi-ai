#!/usr/bin/env python3
"""PR-14 optimizer/scheduler/seed setup (AdamW, BF16, warmup, clipping)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class OptimizerError(ValueError):
    """Invalid optimization config/state."""


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20261121
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 10
    grad_clip: float = 1.0
    grad_accumulation: int = 1
    bf16: bool = False
    ema_decay: float | None = None
    device: str = "cpu"

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise OptimizerError("seed must be int")
        if not (self.lr > 0 and math.isfinite(self.lr)):
            raise OptimizerError("lr must be finite and > 0")
        if not (0 <= self.weight_decay < 1):
            raise OptimizerError("weight_decay must be in [0,1)")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise OptimizerError("betas must be in [0,1)")
        if self.warmup_steps < 0:
            raise OptimizerError("warmup_steps must be >= 0")
        if not (self.grad_clip > 0 and math.isfinite(self.grad_clip)):
            raise OptimizerError("grad_clip must be finite and > 0")
        if self.grad_accumulation < 1:
            raise OptimizerError("grad_accumulation must be >= 1")
        if self.ema_decay is not None and not (0 <= self.ema_decay < 1):
            raise OptimizerError("ema_decay must be in [0,1)")
        if self.device not in ("cpu", "cuda"):
            raise OptimizerError("device must be cpu or cuda")


def set_seed(seed: int) -> None:
    """Deterministic Python/NumPy/Torch seed control."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise OptimizerError("seed must be int")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if not isinstance(model, nn.Module):
        raise OptimizerError("model must be torch.nn.Module")
    if not isinstance(config, TrainingConfig):
        raise OptimizerError("config must be TrainingConfig")
    return torch.optim.AdamW(
        model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay)


def make_scheduler(optimizer: torch.optim.Optimizer,
                   config: TrainingConfig):
    """Linear warmup then constant LR (deterministic step-based)."""
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise OptimizerError("optimizer must be torch.optim.Optimizer")
    warmup = int(config.warmup_steps)

    def factor(step: int) -> float:
        if warmup <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def make_grad_scaler(config: TrainingConfig):
    """BF16 on CUDA does not need a GradScaler; return None for CPU/bf16 tests."""
    return None


def assert_finite_loss(loss: torch.Tensor, *, name: str = "loss") -> None:
    if not isinstance(loss, torch.Tensor) or not bool(torch.isfinite(loss).all()):
        raise OptimizerError(f"{name} is NaN/Inf")


__all__ = [
    "OptimizerError", "TrainingConfig", "assert_finite_loss", "make_grad_scaler",
    "make_optimizer", "make_scheduler", "set_seed",
]
