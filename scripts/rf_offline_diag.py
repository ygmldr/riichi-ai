#!/usr/bin/env python3
"""PR-45 CLI: offline diagnostics on the read-only validation shard.

Never trains: opens the shard through :class:`ValidationReader` (purpose
``diagnostic``/``audit``), forwards under ``torch.no_grad()`` and prints CE/NLL,
top1/top3, legality/non-finite rates, the exclusive phase partition, overlapping
scenario slices, sequence coverage, and (optionally) prefix-vs-multi parity.

The model is either a checkpoint (reloaded read-only) or a freshly seeded
MVP-S initialisation.  GPU is never selected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.models.policy import Policy, count_parameters  # noqa: E402
from riichiformer.train.checkpoint import (  # noqa: E402
    CheckpointError,
    load_checkpoint,
)
from riichiformer.train.model_config import (  # noqa: E402
    ModelConfigError,
    resolve_mvp_config,
)
from riichiformer.train.optimizer import set_seed  # noqa: E402
from riichiformer.validation.reader import (  # noqa: E402
    ValidationError,
    ValidationReader,
)

TASK = "PR-45"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PR-45 read-only validation diagnostics (CPU, no optimizer)")
    parser.add_argument("--shard", required=True)
    parser.add_argument("--policy-config", default=None)
    parser.add_argument("--expected-policy-hash", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--init-seed", type=int, default=20261121)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-decisions", type=int, default=0,
                        help="0 = the whole shard")
    parser.add_argument("--prefix-parity", action="store_true",
                        help="also compare prefix vs multi-decision logits on the "
                             "same weights/seed")
    parser.add_argument("--parity-decisions", type=int, default=64)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _policy_args(args) -> tuple[str | None, str | None]:
    expected = (args.expected_policy_hash or "").strip().lower() or None
    if args.policy_config:
        from riichiformer.policy.config import load_policy_config

        policy = load_policy_config(args.policy_config)
        if policy.profile == "operator_private_research":
            if expected is not None and expected != policy.policy_hash:
                raise ValidationError(
                    "--expected-policy-hash does not match --policy-config")
            expected = policy.policy_hash
    return (args.policy_config, expected)


def _load_model(args):
    resolved = resolve_mvp_config(args.model_config)
    model = Policy(resolved.config)
    if args.checkpoint:
        load_checkpoint(
            args.checkpoint, model=model,
            expected_model_config=resolved.config,
            expected_model_config_hash=resolved.config_hash)
        source = f"checkpoint:{args.checkpoint}"
    else:
        set_seed(args.init_seed)
        source = f"init_seed:{args.init_seed}"
    parameters = count_parameters(model)
    return model, resolved, parameters, source


def run(args) -> dict:
    _, expected = _policy_args(args)
    model, resolved, parameters, source = _load_model(args)
    limit = args.max_decisions if args.max_decisions > 0 else None
    reader = ValidationReader(
        args.shard, expected_policy_hash=expected, batch_size=args.batch_size,
        max_positions=resolved.config.max_positions)
    try:
        describe = reader.describe()
        result = reader.evaluate(
            model, limit=limit, prefix_parity=args.prefix_parity,
            parity_max_decisions=args.parity_decisions)
        result["task"] = TASK
        result["model"] = {
            "source": source,
            "parameter_count": parameters,
            "model_config_hash": resolved.config_hash,
            "device": "cpu",
            "optimizer_steps": 0,
        }
        result["reader"] = describe
    finally:
        reader.close()
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"PR45_DIAG shard={args.shard} decisions={result['total_decisions']} "
              f"supervised={result['supervised']} ce={result['ce']} "
              f"top1={result['top1']} top3={result['top3']} "
              f"illegal_rate={result['illegal_rate']} "
              f"nonfinite_rate={result['nonfinite_rate']} "
              f"params={parameters} model={source}")
        print(f"PR45_PHASES {json.dumps(result['phase_buckets'], sort_keys=True)}")
        print(f"PR45_SCENARIOS "
              f"{json.dumps(result['overlap'], sort_keys=True)}")
        print(f"PR45_COVERAGE {json.dumps(result['sequence_coverage'], sort_keys=True)}")
        if "prefix_multi_parity" in result:
            print(f"PR45_PARITY "
                  f"{json.dumps(result['prefix_multi_parity'], sort_keys=True)}")
        print("PR45_NOTE CE/top1 are offline imitation diagnostics on weak "
              "self-play bots; they are not strength, dan/rank or Tenhou evidence")
    return result


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValidationError, CheckpointError, ModelConfigError) as exc:
        print(f"PR45_REFUSED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"PR45_FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
