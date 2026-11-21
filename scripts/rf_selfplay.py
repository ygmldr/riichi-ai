#!/usr/bin/env python3
"""PR-41 CLI: generate local four-seat hanchan self-play with full provenance.

Examples (CPU only, no network, output directory is mandatory):

.. code-block:: bash

    export PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release

    # real local rules engine (pinned libriichi), 40 games = 10 walls x 4 rotations
    .venv/bin/python scripts/rf_selfplay.py \
        --out /tmp/rf41-selfplay --seed-start 1000 --games 40 \
        --ruleset libriichi_tenhou_approx/1 --seed-pool train \
        --seed-ledger /tmp/rf41-seeds.json

    # labelled procedural fallback, no rules engine
    .venv/bin/python scripts/rf_selfplay.py \
        --out /tmp/rf41-synth --seed-start 2000 --games 8 \
        --ruleset synthetic_abstract_hanchan/1 --seed-pool select \
        --seed-ledger /tmp/rf41-synth-seeds.json

Exit codes: 0 all invariants hold, 1 an audit/legality violation, 2 a contract error
(unknown ruleset/bot, frozen output path, seed reuse, missing libriichi).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from riichiformer.selfplay.bots import BOT_FACTORIES, default_lineup  # noqa: E402
from riichiformer.selfplay.provenance import (  # noqa: E402
    RULESETS,
    ProvenanceError,
    guard_out,
    guard_aux_path,
)
from riichiformer.selfplay.rollout import (  # noqa: E402
    RolloutConfig,
    RolloutError,
    run_selfplay,
)
from riichiformer.selfplay.seeds import (  # noqa: E402
    POOLS,
    SeedError,
    SeedLedger,
    validate_games,
)

TASK = "PR-41"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PR-41 local self-play rollout (offline, CPU, no strength claim)")
    parser.add_argument("--out", required=True,
                        help="explicit output directory (datasets/runs/checkpoints/vendor "
                             "are refused)")
    parser.add_argument("--seed-start", type=int, required=True,
                        help="first wall seed; each wall is replayed in 4 rotations")
    parser.add_argument("--games", type=int, required=True,
                        help="total games; must be a multiple of 4 (= 4 rotations/wall)")
    parser.add_argument("--ruleset", required=True, choices=RULESETS,
                        help="explicit ruleset id (never inferred)")
    parser.add_argument("--key", type=int, default=20261121,
                        help="wall key paired with every seed")
    parser.add_argument("--lineup", default=",".join(default_lineup()),
                        help="four comma-separated local bot names (seat positions)")
    parser.add_argument("--seed-pool", default="train", choices=POOLS,
                        help="train/select/confirm seed pool; pools must stay disjoint")
    parser.add_argument("--seed-ledger", default=None,
                        help="shared seed ledger JSON (default: <out>/seed_ledger.json)")
    parser.add_argument("--verify-bots", action="store_true",
                        help="cross-check the py shanten/win logic against the native "
                             "engine on every decision (slower; test use)")
    parser.add_argument("--no-game-logs", action="store_true",
                        help="do not write raw engine game logs under <out>/games")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _parse_lineup(text: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(text).split(",") if part.strip())
    unknown = [name for name in names if name not in BOT_FACTORIES]
    if unknown:
        raise RolloutError(
            f"unknown local bot(s) {unknown}; known: {sorted(BOT_FACTORIES)}")
    if len(names) != 4:
        raise RolloutError(f"--lineup needs exactly four bot names, got {len(names)}")
    return names


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_games(args.games)
        out = guard_out(args.out)
        lineup = _parse_lineup(args.lineup)
        ledger_path = guard_aux_path(
            Path(args.seed_ledger) if args.seed_ledger else out / "seed_ledger.json")
        ledger = SeedLedger.load(ledger_path)
        config = RolloutConfig(
            seed_start=args.seed_start,
            games=args.games,
            ruleset_id=args.ruleset,
            out=out,
            key=args.key,
            lineup=lineup,
            seed_pool=args.seed_pool,
            ledger=ledger,
            write_logs=not args.no_game_logs,
            verify_bots=args.verify_bots,
        )
        summary = run_selfplay(config)
    except (ProvenanceError, SeedError, RolloutError, ValueError) as exc:
        print(f"rf_selfplay: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    summary["task"] = TASK
    summary["command"] = " ".join(sys.argv)
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    audit = summary.get("audit", {})
    return 0 if audit.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
