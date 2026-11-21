#!/usr/bin/env python3
"""Bounded synthetic GPU smoke for the VNext BC / GRP / RL objectives.

The user authorised GPU use, but the rights gate still forbids training on the
restricted corpus.  This script therefore uses only generated tensors: it proves
the CUDA path, finite losses, checkpoint roundtrip and objective plumbing.  It
is not a training result, not a convergence claim and not strength evidence.
"""

from __future__ import annotations

import argparse
import hashlib
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
from riichiformer.vnext.grp import GRP24, grp_loss  # noqa: E402
from riichiformer.vnext.model import VNextPolicy  # noqa: E402
from riichiformer.vnext.rl import (  # noqa: E402
    awr_loss, behavior_probs, mortal_delta_rewards, pbrs_rewards,
    ppo_loss, two_timescale_gae,
)
from riichiformer.vnext.runtime import load_vnext_checkpoint  # noqa: E402
from riichiformer.vnext.train import (  # noqa: E402
    VNextTrainConfig, make_optimizer, save_vnext_checkpoint, train_step,
)


def _state_hash(model: VNextPolicy) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--grp-steps", type=int, default=50)
    parser.add_argument("--decisions", type=int, default=64)
    parser.add_argument("--length", type=int, default=192)
    parser.add_argument("--out", default="runs/vnext_gpu_smoke")
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.steps <= 0 or args.grp_steps <= 0:
        raise SystemExit("steps must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    batch = make_batch(args.decisions, args.length, seed=args.seed).to(device)
    targets = [
        {"expert_action": 37, "supervision_valid": True,
         "rank_final": index % 4, "loss_weight": 1.0}
        for index in range(args.decisions)
    ]
    model = VNextPolicy(VNextConfig()).to(device)
    train_config = VNextTrainConfig(learning_rate=1e-4, seed=args.seed)
    optimizer = make_optimizer(model, train_config)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses: list[float] = []
    for step in range(args.steps):
        info = train_step(model, batch, targets, optimizer=optimizer,
                          config=train_config, accumulation_step=step)
        losses.append(float(info["loss"]))
        if not torch.isfinite(torch.tensor(info["loss"])):
            raise SystemExit(f"non-finite BC loss at step {step}: {info['loss']}")
    torch.cuda.synchronize()
    bc_seconds = time.perf_counter() - started
    bc_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty smoke dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source_id": "synthetic_fixture",
        "split_manifest_sha256": hashlib.sha256(b"synthetic-split").hexdigest(),
        "canonical_manifest_sha256": hashlib.sha256(b"synthetic-manifest").hexdigest(),
        "canonical_index_sha256": hashlib.sha256(b"synthetic-index").hexdigest(),
        "research_policy": "synthetic_no_rights",
        "research_policy_hash": hashlib.sha256(b"synthetic-policy").hexdigest(),
    }
    checkpoint = out / "synthetic_bc.pt"
    before_hash = _state_hash(model)
    checkpoint_sha = save_vnext_checkpoint(
        checkpoint, model=model, optimizer=optimizer, config=train_config,
        counters={"step": args.steps, "tokens": args.decisions * args.length * args.steps,
                  "decisions": args.decisions * args.steps},
        provenance=provenance)
    loaded = load_vnext_checkpoint(checkpoint)
    after_hash = _state_hash(loaded)
    if before_hash != after_hash:
        raise SystemExit("checkpoint roundtrip changed the model state")

    grp = GRP24().to(device)
    grp_optimizer = torch.optim.AdamW(grp.parameters(), lr=1e-3)
    boundaries = torch.randn(64, 8, 7, device=device)
    ranks = torch.stack([torch.randperm(4, device=device) for _ in range(64)])
    grp_losses: list[float] = []
    for _ in range(args.grp_steps):
        grp_optimizer.zero_grad(set_to_none=True)
        loss = grp_loss(grp(boundaries), ranks)
        loss.backward()
        grp_optimizer.step()
        grp_losses.append(float(loss.detach()))
    if not all(torch.isfinite(torch.tensor(v)) for v in grp_losses):
        raise SystemExit("non-finite GRP loss")

    output = model(batch)
    logits = output.logits[:32].detach()
    new_logits = output.logits[:32] + 0.01
    legal = batch.legal_mask[:32]
    actions: list[int] = []
    for row in range(logits.size(0)):
        _probs, action, _forced = behavior_probs(logits[row], legal[row])
        actions.append(action)
    action_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    old_logprobs = torch.log_softmax(logits, dim=-1)
    new_logprobs = torch.log_softmax(new_logits, dim=-1)
    advantages = torch.linspace(-1.0, 1.0, 32, device=device)
    awr = awr_loss(new_logprobs, action_tensor, advantages, beta=1.0)
    ppo = ppo_loss(new_logprobs, old_logprobs.detach(), action_tensor, advantages)
    rl_loss = awr + ppo
    rl_loss.backward()
    rl_finite = bool(torch.isfinite(rl_loss))

    phis = [
        (0.1, -0.2, 0.3, -0.2),
        (0.0, -0.1, 0.2, -0.1),
    ]
    pbrs = pbrs_rewards(phis, 75.0)
    delta = mortal_delta_rewards(phis, 75.0)
    gae = two_timescale_gae(
        [1.0, 0.5, 2.0, -1.0], [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.0], terminals=[False, False, False, True],
        kyoku_boundaries=[False, True, False, False])

    result = {
        "status": "PASS",
        "kind": "synthetic_gpu_smoke",
        "synthetic_fixture": True,
        "strength_claim": "NONE",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": "cuda",
        "decisions": args.decisions,
        "length": args.length,
        "bc": {
            "steps": args.steps,
            "seconds": bc_seconds,
            "steps_per_s": args.steps / bc_seconds,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "finite": True,
            "peak_vram_mb": bc_peak,
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "roundtrip_state_sha256": after_hash,
            "state_unchanged": True,
        },
        "grp": {
            "steps": args.grp_steps,
            "first_loss": grp_losses[0],
            "last_loss": grp_losses[-1],
            "finite": True,
        },
        "rl": {
            "behavior_actions": len(actions),
            "awr_loss": float(awr.detach()),
            "ppo_loss": float(ppo.detach()),
            "finite": rl_finite,
            "pbrs_shape": [len(pbrs), 4],
            "delta_shape": [len(delta), 4],
            "gae_length": len(gae),
        },
        "note": (
            "Generated tensors only; rights gate remains BLOCKED_FOR_NEW_TRAINING. "
            "No convergence, win-rate, dan or strength claim."
        ),
    }
    (out / "smoke_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
