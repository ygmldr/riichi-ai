#!/usr/bin/env python3
"""``rf_eval_m5d``: native 1v3 duplicate evaluation against the frozen M5D-R0.

Champion: three identical ``M5D-R0`` seats under the pinned
``libriichi.arena.OneVsThree.py_vs_py`` engine.

``--candidate mortal`` is the anchor **self-control** (the same M5D-R0 plays the
candidate seat).  It measures rotation/seed bias and must be near zero; it is not an
RF result.  ``--candidate rf --checkpoint PATH`` runs an RF checkpoint policy through
the POV-safe ``mjai-log`` bridge.

The scorer is :mod:`riichiformer.eval.relative_pt`: primary ``[75,30,0,-105]``,
auxiliary ``[90,45,0,-135]``, delta ``= 4/3 * pt5_C``, whole-complete-set bootstrap.
Outputs are written only to ``--out`` (never ``runs/1v3``, ``runs/variance``,
``checkpoints``, ``datasets`` or ``vendor``).  A fresh 64-bit wall key is generated
with :func:`secrets.randbits` unless one is supplied; it is printed and recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from riichiformer.eval.m5d_arena import (  # noqa: E402
    DEFAULT_WINDOW_START,
    MAX_SMOKE_SETS,
    M5DArenaError,
    fresh_seed_key,
    plan_run,
    run_arena,
)
from riichiformer.eval.mortal_anchor import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    REPO_ROOT as ANCHOR_REPO_ROOT,
)
from riichiformer.eval.relative_pt import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
)

DEFAULT_REGISTRY = ANCHOR_REPO_ROOT / "configs" / "riichiformer" / "m5d_baseline.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PR-46B native 1v3 evaluation against frozen Mortal-5D M5D-R0")
    parser.add_argument("--candidate", choices=("mortal", "rf"), default="mortal",
                        help="mortal = anchor self-control only; rf = RF checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="RF checkpoint path (required for --candidate rf)")
    parser.add_argument("--use-ema", action="store_true",
                        help="use the checkpoint EMA weights when present")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--seed-key", default=None,
                        help="64-bit wall key decimal or hex (default: fresh secrets key)")
    parser.add_argument("--seed-start", type=int, default=DEFAULT_WINDOW_START)
    parser.add_argument("--sets", type=int, default=4,
                        help=f"complete sets (4 games each); smoke <= {MAX_SMOKE_SETS}")
    parser.add_argument("--out", required=False, default=None,
                        help="output directory (must not be a frozen path)")
    parser.add_argument("--log-dir", default=None,
                        help="optional explicit game-log directory (default OUT/games)")
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--max-events", type=int, default=4096)
    parser.add_argument("--plan-only", action="store_true",
                        help="validate the identity/plan and print JSON without playing")
    parser.add_argument("--progress", action="store_true",
                        help="show the native arena progress bar")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sets < 1 or args.sets > 100:
        parser.error("--sets must be in 1..100")
    if args.sets > MAX_SMOKE_SETS:
        print(f"RF_M5D_WARNING sets={args.sets} exceeds the smoke budget "
              f"({MAX_SMOKE_SETS}); this is a trend/select-scale run, not confirm",
              file=sys.stderr)
    key = fresh_seed_key() if args.seed_key is None else args.seed_key
    out = args.out
    if out is None:
        out = f"/tmp/rf_m5d_{args.candidate}_{str(key).replace('0x', '')[-8:]}"
    try:
        if args.plan_only:
            plan = plan_run(
                seed_key=key, seed_start=args.seed_start, seed_count=args.sets,
                candidate=args.candidate, candidate_checkpoint=args.checkpoint,
                registry_path=args.registry, out_dir=out)
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if args.candidate == "rf" and not args.checkpoint:
            parser.error("--candidate rf requires --checkpoint")
        run = run_arena(
            seed_key=key,
            seed_start=args.seed_start,
            seed_count=args.sets,
            candidate=args.candidate,
            candidate_checkpoint=args.checkpoint,
            use_ema=args.use_ema,
            registry_path=args.registry,
            out_dir=out,
            log_dir=args.log_dir,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            write_logs=True,
            max_events=args.max_events,
            progress=args.progress,
        )
    except M5DArenaError as exc:
        print(f"RF_M5D_FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    summary = run.summary()
    for item in run.report.get("per_set", []):
        print(f"RF_M5D_SET wall_seed={item['wall_seed']} "
              f"pt5_candidate={item['pt5_candidate']:.3f} "
              f"delta_pt5={item['delta_pt5']:.3f} "
              f"avg_rank={item['avg_rank']:.3f} "
              f"rotations={item['rotations']}")
    print(f"RF_M5D_SUMMARY {json.dumps(summary, sort_keys=True)}")
    if run.self_control_only:
        print("RF_M5D_NOT_RUN rf_candidate_quality=self_control_only "
              "reason=no RF checkpoint in --candidate mortal mode")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
