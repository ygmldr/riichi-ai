#!/usr/bin/env python3
"""PR-45 prefix vs multi-decision parity on identical weights and seed.

``Policy.forward`` (one prefix per decision) and ``Policy.forward_multi`` (one
shared event stream + a marker per decision) must agree on the same decisions up
to floating-point reduction order.  This is a *correctness* fixture: it says
nothing about training throughput or strength.

The fixture takes the first decisions of the validation shard through both paths,
matches rows by ``(game_id, pov, decision_index)`` and reports the maximum absolute
logit/log-prob difference.  A mismatch raises.
"""

from __future__ import annotations

import torch

from ..data.encode import encode_batch
from ..models.policy import Policy

__all__ = ["ParityError", "prefix_multi_parity"]


class ParityError(RuntimeError):
    """Prefix and multi-decision forward disagree beyond tolerance."""


def prefix_multi_parity(reader, model: Policy, *, max_decisions: int = 64,
                        seed: int = 20261121, atol: float = 1e-5,
                        max_scan_decisions: int = 8192) -> dict:
    if not isinstance(model, Policy):
        raise ParityError("model must be a Policy")
    if max_decisions <= 0:
        raise ParityError("max_decisions must be positive")
    was_training = model.training
    model.eval()
    multi_logits: dict[tuple[str, int, int], torch.Tensor] = {}
    multi_logprobs: dict[tuple[str, int, int], torch.Tensor] = {}
    multi_targets: dict[tuple[str, int, int], dict] = {}
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            for item in reader.iter_batches(limit=max_decisions):
                batch = item.batch
                output = model.forward_multi(batch.batch)
                for index, key in enumerate(zip(batch.game_ids, batch.povs,
                                                batch.decision_indices)):
                    key = (str(key[0]), int(key[1]), int(key[2]))
                    multi_logits[key] = output.logits[index].detach().cpu().clone()
                    multi_logprobs[key] = output.logprobs[index].detach().cpu().clone()
                    multi_targets[key] = dict(batch.targets[index])
                if len(multi_logits) >= max_decisions:
                    break
        if not multi_logits:
            raise ParityError("validation shard yielded no decisions")

        wanted = set(multi_logits)
        prefix_decisions = []
        prefix_metas = []
        prefix_keys = []
        scanned = 0
        with torch.no_grad():
            for decision, _target, meta, game_id, pov, index in \
                    reader._reader.iter_decisions():
                scanned += 1
                key = (str(game_id), int(pov), int(index))
                if key in wanted:
                    prefix_decisions.append(decision)
                    prefix_metas.append(meta)
                    prefix_keys.append(key)
                if len(prefix_keys) == len(wanted) or scanned >= max_scan_decisions:
                    break
            missing = wanted - set(prefix_keys)
            if missing:
                raise ParityError(
                    f"prefix path did not expose {len(missing)} decisions "
                    f"(scan={scanned}); cannot compare the full fixture")
            prefix_batch = encode_batch(prefix_decisions, prefix_metas).to_model_batch()
            prefix_output = model(prefix_batch)
        max_logit_diff = 0.0
        max_logprob_diff = 0.0
        max_loss_diff = 0.0
        supervised = 0
        bad_rows = []
        for index, key in enumerate(prefix_keys):
            logit_diff = float(
                (multi_logits[key] - prefix_output.logits[index].detach().cpu())
                .abs().max().item())
            logprob_diff = float(
                (multi_logprobs[key]
                 - prefix_output.logprobs[index].detach().cpu())
                .abs().max().item())
            max_logit_diff = max(max_logit_diff, logit_diff)
            max_logprob_diff = max(max_logprob_diff, logprob_diff)
            target = multi_targets[key]
            if bool(target.get("supervision_valid")) \
                    and target.get("expert_action") is not None:
                action = int(target["expert_action"])
                loss_diff = float(abs(
                    float(multi_logprobs[key][action])
                    - float(prefix_output.logprobs[index, action])))
                max_loss_diff = max(max_loss_diff, loss_diff)
                supervised += 1
            if logit_diff > atol:
                bad_rows.append((key, logit_diff))
    finally:
        model.train(was_training)
    result = {
        "decisions": len(prefix_keys),
        "supervised": supervised,
        "logit_max_abs": max_logit_diff,
        "logprob_max_abs": max_logprob_diff,
        "loss_max_abs": max_loss_diff,
        "atol": atol,
        "seed": seed,
        "bad_rows": sorted(bad_rows, key=lambda item: -item[1])[:5],
        "ok": max_logit_diff <= atol,
        "note": ("same weights and seed; correctness fixture only, not a "
                 "throughput or strength claim"),
    }
    if not result["ok"]:
        raise ParityError(
            f"prefix vs multi-decision logits differ by {max_logit_diff:.3e} > "
            f"{atol:.1e}")
    return result
