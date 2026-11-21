#!/usr/bin/env python3
"""PR-15 offline diagnostics CLI (diagnostic purpose only).

Refuses a shard whose source is not train_allowed unless the caller explicitly opts
into a blocked audit/diagnostic read.  Never writes datasets/runs/checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.loader import ShardLoadError, ShardReader  # noqa: E402
from riichiformer.eval.offline import OfflineEvalError, evaluate_reader  # noqa: E402
from riichiformer.models.policy import MVPConfig, Policy  # noqa: E402
from riichiformer.train.checkpoint import (  # noqa: E402
    CheckpointError, CheckpointMismatch, load_checkpoint,
)

FROZEN_PREFIXES = ("datasets", "runs/1v3", "runs/variance", "checkpoints")


class EvalRefusal(RuntimeError):
    pass


def _check_out(path: Path) -> None:
    resolved = path.resolve()
    for prefix in FROZEN_PREFIXES:
        frozen = (REPO_ROOT / prefix).resolve()
        if resolved == frozen or frozen in resolved.parents:
            raise EvalRefusal(f"refusing to write frozen path {resolved}")


def _load_model(args, reader):
    config = MVPConfig(layers=args.layers, ffn=args.ffn,
                       state_hidden=args.state_hidden,
                       max_positions=args.max_positions)
    model = Policy(config)
    if args.checkpoint:
        expected = {
            "schema_hash": reader.metadata.schema_hash,
            "action_map_hash": reader.metadata.action_map_hash,
            "normalizer_hash": reader.metadata.normalizer_hash,
        }
        info, payload = load_checkpoint(args.checkpoint, model=model,
                                        expected_hashes=expected)
        print(f"PR15_CHECKPOINT sha256={info.sha256} step={info.step} seed={info.seed}")
    else:
        print("PR15_MODEL random-init (synthetic smoke only; no strength claim)")
    return model


def _write(result: dict, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    if fmt == "json":
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    elif fmt == "csv":
        rows = []
        for key in ("total_decisions", "supervised", "excluded", "nll", "top1",
                    "top3", "top1_ci", "top3_ci", "nll_ci"):
            rows.append((key, result[key]))
        for section in ("phase_buckets", "action_family_buckets", "overlap"):
            for bucket, values in result.get(section, {}).items():
                for metric, value in values.items():
                    rows.append((f"{section}.{bucket}.{metric}", value))
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            writer.writerows(rows)
    elif fmt == "markdown":
        lines = ["# PR-15 Offline Diagnostics", "", f"Shard: `{result.get('shard')}`",
                 f"Source: `{result.get('source_id')}`",
                 f"Supervised: {result['supervised']} / {result['total_decisions']}",
                 f"NLL: {result['nll']:.6f}",
                 f"Top1: {result['top1']:.4f}", f"Top3: {result['top3']:.4f}", "",
                 "> NLL/Top1 are offline imitation metrics, NOT arena strength or Tenhou rank.", "",
                 "| bucket | count | NLL | Top1 | Top3 |", "|---|---:|---:|---:|---:|"]
        for section in ("phase_buckets", "action_family_buckets"):
            for bucket, values in result.get(section, {}).items():
                lines.append(f"| {section}.{bucket} | {values['count']} | "
                             f"{values['nll']:.6f} | {values['top1']:.4f} | "
                             f"{values['top3']:.4f} |")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        raise EvalRefusal(f"unknown format {fmt!r}")
    tmp.replace(path)


def run(args) -> int:
    if args.purpose != "diagnostic":
        raise EvalRefusal("this CLI only supports --purpose diagnostic")
    out = Path(args.out) if args.out else None
    if out:
        _check_out(out)
    reader = ShardReader(args.shard, purpose="diagnostic")
    if not reader.metadata.train_allowed and not args.allow_blocked_audit:
        raise EvalRefusal(
            "shard source is not train_allowed; pass --allow-blocked-audit only for "
            "an H-03-permitted diagnostic read")
    model = _load_model(args, reader)
    result = evaluate_reader(reader, model, batch_size=args.batch_size,
                             limit=args.limit)
    if out:
        _write(result, out, args.format)
        print(f"PR15_WROTE {out} format={args.format}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--format", default="json", choices=("json", "csv", "markdown"))
    parser.add_argument("--purpose", default="diagnostic",
                        choices=("diagnostic",))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-blocked-audit", action="store_true")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--ffn", type=int, default=64)
    parser.add_argument("--state-hidden", type=int, default=64)
    parser.add_argument("--max-positions", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.limit == 0:
        args.limit = None
    try:
        return run(args)
    except (EvalRefusal, ShardLoadError, CheckpointMismatch, CheckpointError) as exc:
        print(f"PR15_REFUSED: {exc}", file=sys.stderr)
        return 2
    except OfflineEvalError as exc:
        print(f"PR15_FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"PR15_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
