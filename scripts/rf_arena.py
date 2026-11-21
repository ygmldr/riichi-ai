#!/usr/bin/env python3
"""PR-42 CLI: duplicate arena over registered opponents with seed-block statistics.

Examples:

.. code-block:: bash

    export PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release

    # 1v3: challenger sits in every seat once per wall; 20 walls = 80 games
    .venv/bin/python scripts/rf_arena.py \
        --out /tmp/rf42-arena --seed-start 5000 --games 80 \
        --ruleset libriichi_tenhou_approx/1 \
        --opponents shanten-ukeire,defense --seed-pool confirm \
        --seed-ledger /tmp/rf42-seeds.json

    # four-seat free-for-all
    .venv/bin/python scripts/rf_arena.py \
        --out /tmp/rf42-ffa --seed-start 6000 --games 40 \
        --ruleset synthetic_abstract_hanchan/1 \
        --opponents shanten-ukeire,defense,legal-random,shanten-ukeire

Exit codes: 0 audit clean, 1 audit/legality violation, 2 contract error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from riichiformer.eval.arena import DuplicateArena, GameAudit  # noqa: E402
from riichiformer.eval.opponents import (  # noqa: E402
    OpponentError,
    build_lineup,
    default_registry,
)
from riichiformer.selfplay.provenance import (  # noqa: E402
    RULESETS,
    ProvenanceError,
    build_provenance,
    generator_fingerprint,
    guard_out,
    guard_aux_path,
    write_json,
    write_jsonl,
)
from riichiformer.selfplay.rollout import (  # noqa: E402
    ENGINE_FOR_RULESET,
    RolloutError,
    libriichi_status,
    run_seed,
)
from riichiformer.selfplay.seeds import (  # noqa: E402
    POOLS,
    SeedError,
    SeedLedger,
    seed_range,
    validate_games,
)

TASK = "PR-42"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PR-42 duplicate arena (four-seat rotation, seed-block bootstrap)")
    parser.add_argument("--out", required=True,
                        help="explicit output directory (frozen paths are refused)")
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--games", type=int, required=True,
                        help="total games; must be a multiple of 4")
    parser.add_argument("--ruleset", required=True, choices=RULESETS)
    parser.add_argument("--opponents", required=True,
                        help="2 names (1v3) or 4 names (free-for-all) from the registry")
    parser.add_argument("--key", type=int, default=20261121)
    parser.add_argument("--seed-pool", default="confirm", choices=POOLS)
    parser.add_argument("--seed-ledger", default=None,
                        help="shared seed ledger JSON (default: <out>/seed_ledger.json)")
    parser.add_argument("--resamples", type=int, default=2000,
                        help="seed-block bootstrap resamples")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--no-game-logs", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_games(args.games)
        out = guard_out(args.out)
        registry = default_registry()
        names = tuple(part.strip() for part in str(args.opponents).split(",")
                      if part.strip())
        mode, lineup = build_lineup(names, registry=registry)
        seeds = seed_range(args.seed_start, args.games)
        ledger_path = guard_aux_path(
            Path(args.seed_ledger) if args.seed_ledger else out / "seed_ledger.json")
        ledger = SeedLedger.load(ledger_path)
        ledger.reserve_range(args.seed_pool, args.seed_start, args.games,
                             tag=f"arena:{args.ruleset}")
        out.mkdir(parents=True, exist_ok=True)
        write_logs = not args.no_game_logs
        engine = ENGINE_FOR_RULESET[args.ruleset]

        def runner(seed: int):
            return run_seed(seed, args.key, lineup, ruleset_id=args.ruleset,
                            out=out, write_logs=write_logs)

        arena = DuplicateArena(
            registry=registry, runner=runner, ruleset_id=args.ruleset,
            expected_engine=engine, key=args.key, resamples=args.resamples,
            bootstrap_seed=args.bootstrap_seed, audit=GameAudit())
        report = arena.run(seeds, lineup=lineup)
    except (ProvenanceError, SeedError, OpponentError, RolloutError, ValueError) as exc:
        print(f"rf_arena: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    ledger.consume(args.seed_pool, seeds)
    generator = generator_fingerprint()
    payload = report.as_dict()
    payload["task"] = TASK
    payload["command"] = " ".join(sys.argv)
    payload["mode_detail"] = mode
    payload["generator_sha256"] = generator["code_sha256"]
    payload["seed_pool"] = args.seed_pool
    payload["outputs"] = {"summary": "summary.json", "games": "games.jsonl",
                          "provenance": "provenance.json",
                          "seed_ledger": "seed_ledger.json"}
    provenance = build_provenance(
        ruleset_id=args.ruleset,
        engine=engine,
        generator=generator,
        extra={
            "seed_start": args.seed_start,
            "seed_count": len(seeds),
            "games": args.games,
            "key": args.key,
            "lineup": list(lineup),
            "opponents": list(names),
            "seed_pool": args.seed_pool,
            "registry": registry.as_dict(),
            "libriichi_status": libriichi_status(),
        },
    )
    write_json(out / "summary.json", payload)
    write_json(out / "provenance.json", provenance)
    write_json(out / "seed_ledger.json", ledger.as_dict())
    write_jsonl(out / "games.jsonl", list(report.game_rows))
    if not args.quiet:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["audit"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
