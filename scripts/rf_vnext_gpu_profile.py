#!/usr/bin/env python3
"""Bounded GPU throughput/VRAM profile for the VNext reference model.

User-authorised resource profile only: synthetic contract-valid tensors, a hard
timeout is the caller's responsibility, no dataset is read and no checkpoint is
promoted.  The numbers are a measurement, not an epoch projection or strength
result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rf_bench_vnext import make_batch  # noqa: E402
from riichiformer.vnext.contracts import VNextConfig  # noqa: E402
from riichiformer.vnext.model import VNextPolicy, count_parameters  # noqa: E402
from riichiformer.vnext.train import (  # noqa: E402
    VNextTrainConfig, make_optimizer, train_step,
)


def _gpu_name() -> str:
    return torch.cuda.get_device_name(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=int, default=128)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", default="logs/vnext/profile_gpu.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.decisions <= 0 or args.length <= 1 or args.iters <= 0:
        raise SystemExit("decisions/length/iters must be positive (length > 1)")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    batch = make_batch(args.decisions, args.length, seed=args.seed).to(device)
    config = VNextConfig()
    model = VNextPolicy(config).to(device)
    train_config = VNextTrainConfig(learning_rate=1e-4)
    optimizer = make_optimizer(model, train_config)
    params = count_parameters(model)
    targets = [
        {"expert_action": 37, "supervision_valid": True,
         "rank_final": index % 4, "loss_weight": 1.0}
        for index in range(args.decisions)
    ]

    model.eval()
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.iters):
            model(batch)
    torch.cuda.synchronize()
    inference = (time.perf_counter() - started) / args.iters
    inference_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    for index in range(args.warmup):
        train_step(model, batch, targets, optimizer=optimizer,
                   config=train_config, accumulation_step=index)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for index in range(args.iters):
        train_step(model, batch, targets, optimizer=optimizer,
                   config=train_config, accumulation_step=index)
    torch.cuda.synchronize()
    train = (time.perf_counter() - started) / args.iters
    train_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    event_tokens = args.decisions * args.length
    report = {
        "device": "cuda",
        "gpu": _gpu_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "decisions": args.decisions,
        "event_tokens": event_tokens,
        "parameter_count": params,
        "warmup": args.warmup,
        "iters": args.iters,
        "inference_seconds": inference,
        "inference_decisions_per_s": args.decisions / inference,
        "inference_tokens_per_s": event_tokens / inference,
        "inference_peak_vram_mb": inference_peak,
        "train_step_seconds": train,
        "train_decisions_per_s": args.decisions / train,
        "train_tokens_per_s": event_tokens / train,
        "train_peak_vram_mb": train_peak,
        "note": (
            "synthetic contract-valid batch; bounded profile only, not a "
            "full-epoch budget and no strength claim"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
