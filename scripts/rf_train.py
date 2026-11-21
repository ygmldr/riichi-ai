#!/usr/bin/env python3
"""PR-14 synthetic-first BC trainer CLI.

Default is dry-run.  Training requires a shard whose metadata is purpose=train,
train_allowed=True and rights_status=ALLOWED; blocked shards are rejected before a model
is constructed.  Frozen run/checkpoint/data roots are write-refused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from riichiformer.data.loader import ShardLoadError, ShardReader  # noqa: E402
from riichiformer.data.shard import ShardRightsError  # noqa: E402
from riichiformer.models.policy import MVPConfig, Policy  # noqa: E402
from riichiformer.train.bc import TrainingError, training_step  # noqa: E402
from riichiformer.train.checkpoint import (  # noqa: E402
    CheckpointMismatch, load_checkpoint, save_checkpoint,
)
from riichiformer.train.optimizer import (  # noqa: E402
    OptimizerError, TrainingConfig, make_optimizer, make_scheduler, set_seed,
)

FROZEN_PREFIXES = ("datasets", "runs/1v3", "runs/variance", "checkpoints")


class TrainerRefusal(RuntimeError):
    pass


def _check_output_path(path: Path) -> None:
    resolved = path.resolve()
    for prefix in FROZEN_PREFIXES:
        frozen = (REPO_ROOT / prefix).resolve()
        if resolved == frozen or frozen in resolved.parents:
            raise TrainerRefusal(f"refusing to write frozen path {resolved}")


def _load_reader(path: str) -> ShardReader:
    try:
        reader = ShardReader(path, purpose="train")
    except ShardRightsError as exc:
        raise TrainerRefusal(f"rights gate refused shard: {exc}") from exc
    _hashes = {
        "schema_hash": reader.metadata.schema_hash,
        "action_map_hash": reader.metadata.action_map_hash,
        "normalizer_hash": reader.metadata.normalizer_hash,
    }
    return reader


def run(args) -> int:
    if args.out:
        _check_output_path(Path(args.out))
    reader = _load_reader(args.shard)
    if args.dry_run or not args.train:
        batches = 0
        decisions = 0
        for batch in reader.iter_batches(batch_size=args.batch_size):
            batches += 1
            decisions += len(batch.targets)
            if args.steps and batches >= args.steps:
                break
        print(f"PR14_DRY_RUN shard={args.shard} batches={batches} decisions={decisions} "
              f"purpose={reader.metadata.purpose} source={reader.metadata.source_id}")
        return 0
    if not args.out:
        raise TrainerRefusal("--train requires --out")
    config = TrainingConfig(seed=args.seed, lr=args.lr, bf16=args.bf16,
                            device=args.device, warmup_steps=args.warmup)
    if args.cpu:
        config = TrainingConfig(**{**config.__dict__, "device": "cpu", "bf16": False})
    set_seed(config.seed)
    model_config = MVPConfig(layers=args.layers, ffn=args.ffn,
                             state_hidden=args.state_hidden,
                             max_positions=args.max_positions)
    model = Policy(model_config).to(config.device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    step = tokens = decisions = 0
    hashes = {"schema_hash": reader.metadata.schema_hash,
              "action_map_hash": reader.metadata.action_map_hash,
              "normalizer_hash": reader.metadata.normalizer_hash,
              "split_manifest_sha256": reader.metadata.split_manifest_sha256,
              "source_id": reader.metadata.source_id}
    if args.resume:
        expected = {key: hashes[key] for key in
                    ("schema_hash", "action_map_hash", "normalizer_hash")}
        info, _ = load_checkpoint(args.resume, model=model, optimizer=optimizer,
                                  scheduler=scheduler, expected_hashes=expected)
        step, tokens, decisions = info.step, info.tokens, info.decisions
    batches = reader.iter_batches(batch_size=args.batch_size)
    model.train()
    while step < args.max_steps:
        try:
            batch = next(batches)
        except StopIteration:
            break
        model_batch = batch.batch.to(config.device)
        result = training_step(model, model_batch, batch.targets,
                               optimizer=optimizer, scheduler=scheduler,
                               max_grad_norm=config.grad_clip, bf16=config.bf16)
        if result.valid_decisions == 0:
            continue
        step += 1
        decisions += result.valid_decisions
        tokens += int(batch.batch.padding_mask.sum())
        print(f"PR14_STEP step={step} loss={float(result.loss):.6f} "
              f"valid={result.valid_decisions}/{result.total_decisions}")
    info = save_checkpoint(
        args.out, model=model, optimizer=optimizer, scheduler=scheduler,
        counters={"step": step, "tokens": tokens, "decisions": decisions},
        hashes=hashes, config=config, seed=config.seed)
    print(f"PR14_DONE step={step} tokens={tokens} decisions={decisions} "
          f"sha256={info.sha256}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--train", action="store_true",
                        help="actually train (requires --out and ALLOWED shard)")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=0,
                        help="dry-run batch bound; 0 = all")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--ffn", type=int, default=64)
    parser.add_argument("--state-hidden", type=int, default=64)
    parser.add_argument("--max-positions", type=int, default=2048)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (TrainerRefusal, ShardLoadError, ShardRightsError) as exc:
        print(f"PR14_REFUSED: {exc}", file=sys.stderr)
        return 2
    except (TrainingError, OptimizerError, CheckpointMismatch) as exc:
        print(f"PR14_FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"PR14_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
