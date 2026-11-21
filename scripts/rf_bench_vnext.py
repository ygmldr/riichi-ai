#!/usr/bin/env python3
"""CPU throughput microbenchmark for the VNext model (no training, no GPU job).

This is a *measured* baseline for PR-11's budget discussion, not a prediction of
full-epoch time.  It constructs a synthetic, contract-valid batch and times
forward-only and one backward/optimizer step.  The output includes parameter
count, decisions/s, event-token/s and peak RSS.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.vnext.contracts import (  # noqa: E402
    BOS_EVENT_ID, PAD_EVENT_ID, TILE_FEATURE_DIM, VNextConfig, VNextModelBatch,
)
from riichiformer.vnext.model import VNextPolicy, count_parameters  # noqa: E402
from riichiformer.vnext.observable import VNextEventStream  # noqa: E402
from riichiformer.vnext.train import (  # noqa: E402
    VNextTrainConfig, make_optimizer, train_step,
)


def make_batch(decisions: int, length: int, *, seed: int = 0) -> VNextModelBatch:
    torch.manual_seed(seed)
    stream = VNextEventStream(match_id="bench", kyoku_id=0, pov=0)
    # BOS + length-1 ordinary visible events.
    stream.event_type_id = [BOS_EVENT_ID] + [3] * (length - 1)
    stream.event_actor_rel = [4] + [0] * (length - 1)
    stream.event_tile_kind = [34] + [0] * (length - 1)
    stream.event_aka = [0] * length
    stream.event_tsumogiri = [0] * length
    stream.event_kyoku_pos = [0] * length
    for index in range(decisions):
        from riichiformer.schemas.action_v3 import ActionPhase
        from riichiformer.vnext.contracts import VNextVisibleState

        state = VNextVisibleState.empty()
        # Same legal mask (SELF_TURN riichi) for every synthetic row.
        state = VNextVisibleState(
            tile_features=state.tile_features,
            global_features=state.global_features,
            candidate_features=state.candidate_features,
            legal_mask=state.legal_mask, phase=0)
        from riichiformer.vnext.contracts import VNextDecisionIndex
        stream.decisions.append((
            VNextDecisionIndex(
                match_id="bench", kyoku_id=0, pov=0, decision_id=f"d{index}",
                decision_order=index, phase=0, prefix_event_count=length - 1,
                prefix_token_offset=length - 1),
            state))
    return stream.to_batch()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=int, default=64)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="logs/vnext/profile_cpu.json")
    args = parser.parse_args()
    if args.decisions <= 0 or args.length <= 1 or args.iters <= 0:
        raise SystemExit("decisions/length/iters must be positive (length > 1)")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    batch = make_batch(args.decisions, args.length)
    config = VNextConfig()
    model = VNextPolicy(config)
    train_config = VNextTrainConfig(learning_rate=1e-4)
    optimizer = make_optimizer(model, train_config)
    params = count_parameters(model)
    model.eval()
    with torch.no_grad():
        model(batch)
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.iters):
            model(batch)
    inference = (time.perf_counter() - started) / args.iters
    model.train()
    started = time.perf_counter()
    for index in range(args.iters):
        target = [{"expert_action": 37, "supervision_valid": True,
                   "rank_final": index % 4, "loss_weight": 1.0}
                  for _ in range(args.decisions)]
        train_step(model, batch, target, optimizer=optimizer,
                   config=train_config, accumulation_step=index)
    train = (time.perf_counter() - started) / args.iters
    event_tokens = args.decisions * args.length
    report = {
        "device": "cpu",
        "threads": args.threads,
        "decisions": args.decisions,
        "event_tokens": event_tokens,
        "parameter_count": params,
        "inference_seconds": inference,
        "inference_decisions_per_s": args.decisions / inference,
        "inference_tokens_per_s": event_tokens / inference,
        "train_step_seconds": train,
        "train_decisions_per_s": args.decisions / train,
        "train_tokens_per_s": event_tokens / train,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "note": "synthetic contract-valid batch; not a full-epoch budget and no strength claim",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
