#!/usr/bin/env python3
"""A1 paired delta analysis: arm - original at the same seed window (todo 12).

Replays the three arm corpora with `scripts/eval_report.py`'s verified NDJSON
logic (imported, never modified: scan_candidates / replay_one / rank_of /
RANK_PTS / SEAT_OF_SPLIT / bootstrap_ci) and computes, per same (!) seed set:

    d_rank[set] = set_avg_rank(arm) - set_avg_rank(original)
    d_pt[set]   = set_avg_pt(arm)   - set_avg_pt(original)

CI = set-level bootstrap over the paired differences (B=2000, rng seed 0),
exactly the procedure of eval_report.py, so the comparison is apples-to-apples.

Usage:
  python3 scripts/exp_a1_delta.py [--a1-dir runs/exp/A1] [--arms shuffle recency_zero]
                                  [--out runs/exp/A1/delta_report.json]
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

# runs/variance/README.md (todo 6), 12k games, 80% power:
MDD_AVG_RANK_12K = 0.0239
MDD_PT_12K = 1.86


def per_set_stats(games_dir: Path) -> tuple[dict[int, tuple[float, float]], dict]:
    entries, ignored, non_game = er.scan_candidates(str(games_dir))
    by_seed: dict[int, dict[str, tuple[int, int]]] = {}
    for path, seed, key, split in entries:
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
        "ignored_files": len(ignored),
        "non_game_files": non_game,
    }
    return sets, info


def mean(xs) -> float:
    return float(np.mean(np.asarray(xs, dtype=float)))


def compare(ref_sets, arm_sets, arm: str, seed_list: list[int]) -> dict:
    d_rank = [arm_sets[s][0] - ref_sets[s][0] for s in seed_list]
    d_pt = [arm_sets[s][1] - ref_sets[s][1] for s in seed_list]
    ci = er.bootstrap_ci(d_rank, d_pt)
    delta_rank = mean(d_rank)
    delta_pt = mean(d_pt)
    rank_parity = ci[0] <= 0.0 <= ci[1]
    pt_parity = ci[2] <= 0.0 <= ci[3]
    return {
        "comparison": f"{arm} - original",
        "arm": arm,
        "n_sets": len(seed_list),
        "n_games": 4 * len(seed_list),
        "arm_avg_rank": mean([arm_sets[s][0] for s in seed_list]),
        "original_avg_rank": mean([ref_sets[s][0] for s in seed_list]),
        "delta_avg_rank": delta_rank,
        "delta_avg_rank_ci": [ci[0], ci[1]],
        "arm_avg_pt": mean([arm_sets[s][1] for s in seed_list]),
        "original_avg_pt": mean([ref_sets[s][1] for s in seed_list]),
        "delta_avg_pt": delta_pt,
        "delta_avg_pt_ci": [ci[2], ci[3]],
        "rank_ci_crosses_0": rank_parity,
        "pt_ci_crosses_0": pt_parity,
        "prereg_decision": (
            f"{arm}: NO detectable effect on avg_rank (95% CI crosses 0)"
            if rank_parity
            else f"{arm}: avg_rank effect (95% CI excludes 0)"
        ),
        "mdd_80pct_12k": {"avg_rank": MDD_AVG_RANK_12K, "pt": MDD_PT_12K},
        "bootstrap": {"B": er.BOOTSTRAP_RESAMPLES, "rng_seed": er.BOOTSTRAP_SEED},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A1 paired deltas vs original (set-level bootstrap)")
    ap.add_argument("--a1-dir", default=str(REPO / "runs" / "exp" / "A1"))
    ap.add_argument("--reference", default="original")
    ap.add_argument("--arms", nargs="+", default=["shuffle", "recency_zero"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    a1 = Path(args.a1_dir)
    ref_sets, ref_info = per_set_stats(a1 / f"games_{args.reference}")
    seed_list = sorted(ref_sets)
    print(f"reference {args.reference}: sets={ref_info['sets']} games={4 * ref_info['sets']}")

    report = {
        "reference": args.reference,
        "reference_info": ref_info,
        "reference_avg_rank": mean([ref_sets[s][0] for s in seed_list]),
        "reference_avg_pt": mean([ref_sets[s][1] for s in seed_list]),
        "mdd_80pct_12k": {"avg_rank": MDD_AVG_RANK_12K, "pt": MDD_PT_12K},
        "comparisons": [],
    }
    ok = True
    for arm in args.arms:
        arm_sets, arm_info = per_set_stats(a1 / f"games_{arm}")
        if sorted(arm_sets) != seed_list:
            raise er.GateFailure(f"{arm}: seed sets differ from {args.reference} (not paired)")
        comp = compare(ref_sets, arm_sets, arm, seed_list)
        comp["arm_info"] = arm_info
        report["comparisons"].append(comp)
        print(
            f"{comp['comparison']}: d_rank={comp['delta_avg_rank']:+.6f} "
            f"CI=[{comp['delta_avg_rank_ci'][0]:+.6f},{comp['delta_avg_rank_ci'][1]:+.6f}] "
            f"({'crosses 0' if comp['rank_ci_crosses_0'] else 'excludes 0'}) | "
            f"d_pt={comp['delta_avg_pt']:+.6f} "
            f"CI=[{comp['delta_avg_pt_ci'][0]:+.6f},{comp['delta_avg_pt_ci'][1]:+.6f}] "
            f"({'crosses 0' if comp['pt_ci_crosses_0'] else 'excludes 0'})"
        )
        print(f"  decision: {comp['prereg_decision']}")
        ok = ok and comp["rank_ci_crosses_0"]
    if len(seed_list) != 3000:
        raise er.GateFailure(f"expected 3000 sets, got {len(seed_list)}")
    out = Path(args.out) if args.out else a1 / "delta_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"delta_report -> {out} (all rank CIs cross 0: {ok})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
