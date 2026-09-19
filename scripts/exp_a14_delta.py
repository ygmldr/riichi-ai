#!/usr/bin/env python3
"""[A14 / plan todo 26] paired per-set delta of the trainee vs the anchor seats.

The 1v3 protocol is zero-sum in rank and RANK_PTS, so the challenger's per-set metrics
are already differentials against the 3 anchor (champion) seats. This script recomputes
the per-set vectors and the explicit paired differences

    d_rank = r - (10 - r) / 3      # trainee set-avg rank minus anchor-seat set-avg rank
    d_pt   = p - (-p / 3)          # trainee set-avg pt   minus anchor-seat set-avg pt

and bootstraps them with eval_report.py's procedure (B=2000 resamples of sets, rng seed 0;
the module is imported and never modified). The JSON output is the "paired Delta
(trainee - anchor)" record cited by runs/exp/A14/VERDICT.md.

Usage:
  python3 scripts/exp_a14_delta.py --games-dir runs/exp/A14/eval_confirm/games \
      --seed-start 70000 --out runs/exp/A14/delta_confirm.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import eval_report as er  # noqa: E402

SEED_KEY = 0x5DFAA4CEF265CD7
EXPECTED_SETS = 3000
MDD_AVG_RANK_12K = 0.0239
MDD_PT_12K = 1.86


def per_set_stats(games_dir: Path):
    entries, ignored, non_game = er.scan_candidates(str(games_dir))
    by_seed: dict[int, dict[str, tuple[int, int]]] = {}
    for path, seed, key, split in entries:
        if key != SEED_KEY:
            raise er.GateFailure(f"{path}: seed_key {key} != {SEED_KEY}")
        fseed, fkey, _, scores = er.replay_one(path)
        if (fseed, fkey) != (seed, key):
            raise er.GateFailure(f"{path}: NDJSON seed={[fseed, fkey]} != filename ({seed}, {key})")
        seat = er.SEAT_OF_SPLIT[split]
        rank = er.rank_of(scores, seat)
        by_seed.setdefault(seed, {})[split] = (rank, er.RANK_PTS[rank - 1])
    sets: dict[int, tuple[float, float]] = {}
    for seed, splits in by_seed.items():
        missing = [s for s in er.SPLITS if s not in splits]
        if missing:
            raise er.GateFailure(f"{games_dir}: seed {seed} missing splits {missing}")
        ranks = [splits[s][0] for s in er.SPLITS]
        pts = [splits[s][1] for s in er.SPLITS]
        sets[seed] = (sum(ranks) / 4.0, sum(pts) / 4.0)
    info = {
        "games_dir": str(games_dir),
        "candidates": len(entries),
        "sets": len(sets),
        "games": 4 * len(sets),
        "ignored_files": len(ignored),
        "non_game_files": non_game,
        "seed_min": min(sets) if sets else None,
        "seed_max": max(sets) if sets else None,
    }
    return sets, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A14 paired per-set delta vs the anchor seats")
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--challenger", default="runs/exp/A14/state_final.pth")
    ap.add_argument("--champion", default="checkpoints/mortal.pth.at-148800")
    args = ap.parse_args(argv)

    sets, info = per_set_stats(Path(args.games_dir))
    if info["sets"] != EXPECTED_SETS:
        raise er.GateFailure(f"sets={info['sets']} != {EXPECTED_SETS}")
    if info["ignored_files"] != 0:
        raise er.GateFailure(f"ignored_files={info['ignored_files']} != 0")
    if not (args.seed_start <= info["seed_min"] and info["seed_max"] < args.seed_start + EXPECTED_SETS):
        raise er.GateFailure(
            f"seed window [{info['seed_min']},{info['seed_max']}] outside "
            f"[{args.seed_start},{args.seed_start + EXPECTED_SETS})"
        )

    seed_list = sorted(sets)
    r = np.array([sets[s][0] for s in seed_list], dtype=float)
    p = np.array([sets[s][1] for s in seed_list], dtype=float)
    anchor_r = (10.0 - r) / 3.0
    anchor_p = -p / 3.0
    d_rank = r - anchor_r
    d_pt = p - anchor_p

    raw_ci = er.bootstrap_ci(r.tolist(), p.tolist())
    ci = er.bootstrap_ci(d_rank.tolist(), d_pt.tolist())

    raw_pt_lo_gt_0 = raw_ci[2] > 0.0
    raw_pt_hi_lt_0 = raw_ci[3] < 0.0
    auxiliary = raw_ci[1] < 2.5
    report = {
        "comparison": "trainee - anchor seats (paired per-set, 1v3 zero-sum)",
        "protocol": {
            "seed_key": hex(SEED_KEY),
            "seed_start": args.seed_start,
            "seed_range": [args.seed_start, args.seed_start + EXPECTED_SETS],
            "sets": info["sets"],
            "games": info["games"],
            "bootstrap": {"resamples": er.BOOTSTRAP_RESAMPLES, "rng_seed": er.BOOTSTRAP_SEED},
            "challenger": args.challenger,
            "champion": args.champion,
            "eval_dir": info,
            "resolution_mdd_80pct_12k": {"avg_rank": MDD_AVG_RANK_12K, "pt": MDD_PT_12K},
        },
        "trainee": {
            "avg_rank": float(r.mean()),
            "avg_rank_ci": [raw_ci[0], raw_ci[1]],
            "avg_pt": float(p.mean()),
            "avg_pt_ci": [raw_ci[2], raw_ci[3]],
        },
        "anchor_seats": {"avg_rank": float(anchor_r.mean()), "avg_pt": float(anchor_p.mean())},
        "delta_avg_rank": float(d_rank.mean()),
        "delta_avg_rank_ci": [ci[0], ci[1]],
        "delta_avg_pt": float(d_pt.mean()),
        "delta_avg_pt_ci": [ci[2], ci[3]],
        "criteria": {
            "primary_raw_pt_ci_lo_gt_0": raw_pt_lo_gt_0,
            "primary_raw_pt_ci_hi_lt_0": raw_pt_hi_lt_0,
            "primary_raw_pt_ci_crosses_0": not (raw_pt_lo_gt_0 or raw_pt_hi_lt_0),
            "auxiliary_raw_avg_rank_ci_hi_lt_2.500": auxiliary,
            "auxiliary_delta_rank_ci_hi_lt_0": ci[1] < 0.0,
            "a14_passes": bool(raw_pt_lo_gt_0 and auxiliary),
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "delta_avg_rank", "delta_avg_rank_ci", "delta_avg_pt", "delta_avg_pt_ci", "criteria")},
        indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
