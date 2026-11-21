#!/usr/bin/env python3
"""Bounded VNext BC trainer for real ``rf-shard/3`` data.

This is the missing PR-08 -> PR-12 bridge: it consumes the same authenticated
shards as the rest of RiichiFormer, groups decisions with
:class:`VNextBatchAdapter`, and performs masked BC plus optional Critic rank CE.

The default is a bounded, resumable pilot.  A full run must be requested
explicitly with ``--steps`` (or ``--max-hours``) and is still subject to the
caller's data policy; this script never reads raw ``.mjson`` directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.loader import ShardLoadError, ShardReader  # noqa: E402
from riichiformer.data.shard import ShardError, ShardRightsError  # noqa: E402
from riichiformer.policy.config import PolicyConfigError, load_policy_config  # noqa: E402
from riichiformer.schemas.schema_v3 import load_schema  # noqa: E402
from riichiformer.train.checkpoint import (  # noqa: E402
    CheckpointError, config_hash, load_checkpoint,
)
from riichiformer.vnext.contracts import VNextConfig  # noqa: E402
from riichiformer.vnext.data import VNextBatchAdapter  # noqa: E402
from riichiformer.vnext.model import VNextPolicy  # noqa: E402
from riichiformer.vnext.train import (  # noqa: E402
    VNextTrainConfig, make_optimizer, masked_bc_loss, rank_loss,
    save_vnext_checkpoint, train_step,
)


class VNextTrainError(RuntimeError):
    pass


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _open_reader(path: str | Path, *, purpose: str,
                 policy) -> ShardReader:
    kwargs = {}
    if policy is not None:
        kwargs["expected_policy_hash"] = policy.policy_hash
        kwargs["allowed_local_sources"] = policy.allowed_local_sources
    try:
        return ShardReader(path, purpose=purpose, **kwargs)
    except (ShardLoadError, ShardRightsError, ShardError) as exc:
        raise VNextTrainError(f"{purpose} shard refused: {exc}") from exc


def _provenance(metadata, *, train_shard_sha256: str,
                val_shard_sha256: str | None) -> dict:
    return {
        "source_id": metadata.source_id,
        "split_manifest_sha256": metadata.split_manifest_sha256,
        "canonical_manifest_sha256": metadata.canonical_manifest_sha256,
        "canonical_index_sha256": metadata.canonical_index_sha256,
        "research_policy": metadata.research_policy,
        "research_policy_hash": metadata.research_policy_hash or (
            "" if metadata.research_policy == "licensed_only" else
            "missing-policy-hash"),
        "train_shard_sha256": train_shard_sha256,
        "val_shard_sha256": val_shard_sha256 or "none",
    }


def _validation(reader: ShardReader, model: VNextPolicy, *,
                batch_size: int, device: torch.device,
                rank_weight: float, max_batches: int | None,
                rank_base: int = 1,
                max_prefix_tokens: int | None = None) -> dict:
    adapter = VNextBatchAdapter(reader, batch_size=batch_size,
                                rank_base=rank_base,
                                max_prefix_tokens=max_prefix_tokens)
    model.eval()
    bc_num = 0.0
    bc_weight = 0.0
    rank_num = 0.0
    rank_rows = 0
    decisions = 0
    batches = 0
    with torch.inference_mode():
        for item in adapter.iter_batches():
            batch = item.batch.to(device)
            output = model(batch)
            bc = masked_bc_loss(output, item.targets)
            rank = rank_loss(output, item.targets)
            bc_num += float(bc.numerator.detach())
            bc_weight += bc.weight
            if rank is not None:
                rank_num += float(rank[0].detach())
                rank_rows += int(rank[1])
            decisions += item.batch.n_decisions
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    model.train()
    bc_ce = bc_num / bc_weight if bc_weight > 0 else None
    rank_ce = rank_num / rank_rows if rank_rows > 0 else None
    total = bc_num + rank_weight * rank_num
    denominator = bc_weight if bc_weight > 0 else float(rank_rows)
    return {
        "batches": batches,
        "decisions": decisions,
        "bc_weight": bc_weight,
        "bc_ce": bc_ce,
        "rank_rows": rank_rows,
        "rank_ce": rank_ce,
        "total_loss": (total / denominator) if denominator > 0 else None,
    }


def _save(model: VNextPolicy, optimizer, train_config: VNextTrainConfig,
          out: Path, *, step: int, tokens: int, decisions: int,
          metadata, history: list[dict], extra: dict) -> Path:
    path = out / "candidates" / f"step_{step:06d}.pt"
    provenance = _provenance(
        metadata,
        train_shard_sha256=extra["train_shard_sha256"],
        val_shard_sha256=extra.get("val_shard_sha256"))
    sha = save_vnext_checkpoint(
        path, model=model, optimizer=optimizer, config=train_config,
        counters={"step": step, "tokens": tokens, "decisions": decisions},
        provenance=provenance, extra=extra)
    record = {
        "step": step, "tokens": tokens, "decisions": decisions,
        "sha256": sha, "path": str(path),
    }
    (out / "metrics.jsonl").open("a", encoding="utf-8").write(
        json.dumps(record, sort_keys=True) + "\n")
    history.append(record)
    return path


def _resume(path: str | Path, model: VNextPolicy, optimizer, config: VNextConfig,
            train_config: VNextTrainConfig, *, expected_provenance: dict,
            train_shard_sha256: str, val_shard_sha256: str | None,
            ) -> tuple[int, dict, int, dict]:
    schema = load_schema()
    hashes = {key: schema[key] for key in
              ("schema_hash", "action_map_hash", "normalizer_hash")}
    info, payload = load_checkpoint(
        path, model=model, optimizer=optimizer,
        expected_hashes=hashes,
        expected_model_config=config.as_dict(),
        expected_model_config_hash=config_hash(config.as_dict()),
        expected_provenance=expected_provenance,
        strict_resume=True,
    )
    counters = payload.get("counters") or {}
    extra = payload.get("extra") or {}
    if extra.get("train_shard_sha256") != train_shard_sha256:
        raise VNextTrainError("checkpoint was trained on a different train shard")
    if extra.get("val_shard_sha256", "none") != (val_shard_sha256 or "none"):
        raise VNextTrainError("checkpoint was validated on a different val shard")
    batches_seen = int(extra.get("batches_seen", -1))
    if batches_seen < 0:
        raise VNextTrainError(
            "checkpoint has no batches_seen; cannot resume without replaying data")
    return (
        int(counters.get("step", info.step)),
        {
            "step": int(counters.get("step", info.step)),
            "tokens": int(counters.get("tokens", 0)),
            "decisions": int(counters.get("decisions", 0)),
        },
        batches_seen,
        extra,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", required=True)
    parser.add_argument("--val-shard", default=None)
    parser.add_argument("--policy-config", default=None,
                        help="operator_private_research policy YAML")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=200,
                        help="optimizer steps; bounded pilot default")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--rank-base", type=int, choices=(0, 1), default=1,
                        help="real shards use 1..4 ranks; synthetic 0..3")
    parser.add_argument("--max-prefix-tokens", type=int, default=None,
                        help="skip decisions longer than this; defaults to 4096")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--constant-lr", action="store_true")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=0,
                        help="validate every N optimizer steps (0 disables)")
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="with --validate-only, evaluate this VNext checkpoint")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise SystemExit("--min-lr-ratio must be in [0,1]")
    if args.save_every < 0 or args.val_every < 0:
        raise SystemExit("--save-every/--val-every must be >= 0")
    out = Path(args.out).resolve()
    root = REPO_ROOT.resolve()
    if out == root:
        raise SystemExit(f"refusing to write the repository root {root}")
    frozen = [
        (root / "datasets").resolve(),
        (root / "checkpoints").resolve(),
        (root / "vendor").resolve(),
        (root / ".git").resolve(),
        (root / "runs" / "1v3").resolve(),
        (root / "runs" / "variance").resolve(),
        (root / "runs" / "exp" / "PINNED").resolve(),
    ]
    for forbidden in frozen:
        if out == forbidden or forbidden in out.parents:
            raise SystemExit(f"refusing to write frozen path {forbidden}")
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise SystemExit(f"refusing to overwrite non-empty run dir {out}")

    policy = None
    if args.policy_config:
        try:
            policy = load_policy_config(args.policy_config)
        except PolicyConfigError as exc:
            raise SystemExit(f"policy config error: {exc}") from exc

    train_reader = _open_reader(args.train_shard, purpose="train", policy=policy)
    val_reader = None
    if args.val_shard:
        val_reader = _open_reader(args.val_shard, purpose="diagnostic", policy=policy)
    try:
        meta = train_reader.metadata
        if meta.research_policy == "operator_private_research" and policy is None:
            raise SystemExit("private shard requires --policy-config")
        train_sha = str(train_reader.manifest["shard_sha256"])
        val_sha = None
        if val_reader is not None:
            val_meta = val_reader.metadata
            if val_meta.purpose != "diagnostic":
                raise SystemExit(
                    f"val shard purpose {val_meta.purpose!r} is not diagnostic")
            if val_meta.split_manifest_sha256 == meta.split_manifest_sha256:
                raise SystemExit("val shard uses the train split manifest")
            if val_reader.manifest.get("shard_sha256") == train_sha:
                raise SystemExit("val shard is the train shard")
            val_sha = str(val_reader.manifest["shard_sha256"])
        if args.dry_run:
            print(json.dumps({
                "train_shard": args.train_shard,
                "val_shard": args.val_shard,
                "metadata": {
                    "purpose": meta.purpose,
                    "source_id": meta.source_id,
                    "research_policy": meta.research_policy,
                    "research_policy_hash": meta.research_policy_hash,
                    "train_eligible_under_policy": meta.train_eligible_under_policy,
                    "train_allowed": meta.train_allowed,
                    "rights_status": meta.rights_status,
                    "split_manifest_sha256": meta.split_manifest_sha256,
                },
                "steps": args.steps, "batch_size": args.batch_size,
                "device": str(_device(args.device)),
            }, indent=2, sort_keys=True))
            return 0

        device = _device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        max_prefix_tokens = args.max_prefix_tokens or VNextConfig().max_positions
        if args.validate_only:
            if args.checkpoint is None or val_reader is None:
                raise SystemExit("--validate-only requires --checkpoint and --val-shard")
            from riichiformer.vnext.runtime import load_vnext_checkpoint
            model = load_vnext_checkpoint(args.checkpoint).to(device)
            report = _validation(
                val_reader, model, batch_size=args.batch_size, device=device,
                rank_weight=args.rank_weight, max_batches=args.val_batches,
                rank_base=args.rank_base, max_prefix_tokens=max_prefix_tokens)
            report["checkpoint"] = str(Path(args.checkpoint).resolve())
            report["checkpoint_sha256"] = hashlib.sha256(
                Path(args.checkpoint).read_bytes()).hexdigest()
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model_config = VNextConfig()
        model = VNextPolicy(model_config).to(device)
        train_config = VNextTrainConfig(
            learning_rate=args.lr, rank_loss_weight=args.rank_weight,
            seed=args.seed, accumulation_count=1)
        optimizer = make_optimizer(model, train_config)
        start_step, start_batches = 0, 0
        counters = {"step": 0, "tokens": 0, "decisions": 0}
        extra = {
            "train_shard_sha256": train_sha,
            "val_shard_sha256": val_sha or "none",
            "batches_seen": 0,
        }
        history: list[dict] = []
        if args.resume:
            expected = _provenance(
                meta, train_shard_sha256=train_sha, val_shard_sha256=val_sha)
            start_step, counters, start_batches, extra = _resume(
                args.resume, model, optimizer, model_config, train_config,
                expected_provenance=expected,
                train_shard_sha256=train_sha, val_shard_sha256=val_sha)
            extra = dict(extra)
            extra["train_shard_sha256"] = train_sha
            extra["val_shard_sha256"] = val_sha or "none"
            print(f"resumed at step {start_step} batches_seen {start_batches}")

        out.mkdir(parents=True, exist_ok=True)
        (out / "run.json").write_text(json.dumps({
            "train_shard": str(Path(args.train_shard).resolve()),
            "val_shard": (str(Path(args.val_shard).resolve())
                          if args.val_shard else None),
            "policy_config": args.policy_config,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "device": str(device),
            "seed": args.seed,
            "train_shard_sha256": train_sha,
            "val_shard_sha256": val_sha,
            "model_config": model_config.as_dict(),
            "provenance": _provenance(
                meta, train_shard_sha256=train_sha, val_shard_sha256=val_sha),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        model.train()
        adapter = VNextBatchAdapter(train_reader, batch_size=args.batch_size,
                                    rank_base=args.rank_base,
                                    max_prefix_tokens=max_prefix_tokens)
        batches = adapter.iter_batches()
        step = start_step
        batches_seen = 0
        started = time.perf_counter()
        losses: list[float] = []
        last_validation = None
        last_saved_step = -1
        saved_path = None
        exhausted = False
        no_progress = 0
        stopped = "target_reached"
        while step < args.steps:
            if args.max_hours > 0 and (time.perf_counter() - started) >= args.max_hours * 3600:
                stopped = "wall_cap"
                print(f"stopping at step {step}: max-hours reached")
                break
            try:
                item = next(batches)
            except StopIteration:
                exhausted = True
                skipped = adapter.long_prefix_skipped
                adapter = VNextBatchAdapter(
                    train_reader, batch_size=args.batch_size,
                    rank_base=args.rank_base,
                    max_prefix_tokens=max_prefix_tokens)
                adapter.long_prefix_skipped = skipped
                batches = adapter.iter_batches()
                continue
            batches_seen += 1
            extra["batches_seen"] = batches_seen
            if batches_seen <= start_batches:
                continue
            if not args.constant_lr:
                if step < args.warmup:
                    scale = (step + 1) / max(1, args.warmup)
                else:
                    progress = min(1.0, (step - args.warmup)
                                   / max(1, args.steps - args.warmup))
                    scale = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (
                        1.0 + math.cos(math.pi * progress))
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * scale
            batch = item.batch.to(device)
            info = train_step(model, batch, item.targets, optimizer=optimizer,
                              config=train_config, accumulation_step=step)
            if not info["optimizer_stepped"]:
                no_progress += 1
                if no_progress >= 1000:
                    raise VNextTrainError(
                        "1000 consecutive batches had no supervised/rank update; "
                        "refusing an all-censored or empty shard")
                continue
            no_progress = 0
            step += 1
            counters["step"] = step
            counters["decisions"] += item.batch.n_decisions
            counters["tokens"] += int(batch.event_valid_mask.sum().item())
            losses.append(float(info["loss"]))
            if not math.isfinite(losses[-1]):
                raise VNextTrainError(f"non-finite loss at step {step}")
            if args.val_every > 0 and val_reader is not None and step % args.val_every == 0:
                last_validation = _validation(
                    val_reader, model, batch_size=args.batch_size, device=device,
                    rank_weight=args.rank_weight, max_batches=args.val_batches,
                    rank_base=args.rank_base)
                with (out / "validation.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step, **last_validation},
                                            sort_keys=True) + "\n")
                print(f"step={step} val={last_validation}")
            if args.save_every > 0 and step % args.save_every == 0:
                saved_path = _save(
                    model, optimizer, train_config, out, step=step,
                    tokens=counters["tokens"],
                    decisions=counters["decisions"], metadata=meta,
                    history=history, extra=extra)
                last_saved_step = step
                print(f"saved {saved_path}")
        if step != last_saved_step:
            saved_path = _save(
                model, optimizer, train_config, out, step=step,
                tokens=counters["tokens"],
                decisions=counters["decisions"], metadata=meta,
                history=history, extra=extra)
            last_saved_step = step
        final = saved_path
        report = {
            "status": ("COMPLETED" if step >= args.steps else "STOPPED"),
            "stopped_reason": ("target_reached" if step >= args.steps else stopped),
            "steps": step, "target_steps": args.steps, "start_step": start_step,
            "batches_seen": batches_seen, "start_batches": start_batches,
            "exhausted": exhausted,
            "max_prefix_tokens": max_prefix_tokens,
            "long_prefix_skipped": adapter.long_prefix_skipped,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_mean": (sum(losses) / len(losses)) if losses else None,
            "last_validation": last_validation,
            "checkpoint": str(final) if final is not None else None,
            "wall_seconds": time.perf_counter() - started,
            "device": str(device),
        }
        (out / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        train_reader.close()
        if val_reader is not None:
            val_reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
