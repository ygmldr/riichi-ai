#!/usr/bin/env python3
"""todo 22 pool-B paired delta: E3a − E2 (paired per-set bootstrap).

Replays both pool-B corpora with `scripts/eval_report.py`'s verified NDJSON logic
(imported, never modified: scan_candidates / replay_one / rank_of / RANK_PTS /
SEAT_OF_SPLIT / bootstrap_ci, B=2000, rng seed 0) and computes, per same seed set:

    d_rank[set] = set_avg_rank(E3a) - set_avg_rank(E2)
    d_pt[set]   = set_avg_pt(E3a)   - set_avg_pt(E2)

CI = set-level bootstrap over the paired differences, exactly the procedure of
eval_report.py / exp_a1_delta.py.

Pre-registered primary criterion (runs/exp/ARCH_B/PREREG.md §3):
    Δpt 95% CI lower bound > 0  => confirm event branch (keep line open)
    otherwise                   => close the river-transformer line

Usage:
  python3 scripts/exp_arch_b_delta.py \
      --a-dir runs/exp/ARCH_B/E3a/games --b-dir runs/exp/ARCH_B/E2/games \
      --out runs/exp/ARCH_B/delta_e3a_vs_e2.json
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
SEED_START = 50000
SEED_END = 53000
EXPECTED_SETS = 3000
MDD_AVG_RANK_12K = 0.0239
MDD_PT_12K = 1.86


def per_set_stats(games_dir: Path) -> tuple[dict[int, tuple[float, float]], dict]:
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


def mean(xs) -> float:
    return float(np.mean(np.asarray(xs, dtype=float)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="pool-B paired delta E3a - E2")
    ap.add_argument("--a-dir", default=str(REPO / "runs" / "exp" / "ARCH_B" / "E3a" / "games"))
    ap.add_argument("--b-dir", default=str(REPO / "runs" / "exp" / "ARCH_B" / "E2" / "games"))
    ap.add_argument("--out", default=str(REPO / "runs" / "exp" / "ARCH_B" / "delta_e3a_vs_e2.json"))
    args = ap.parse_args(argv)

    a_sets, a_info = per_set_stats(Path(args.a_dir))
    b_sets, b_info = per_set_stats(Path(args.b_dir))
    for tag, info, sets in (("E3a", a_info, a_sets), ("E2", b_info, b_sets)):
        if info["sets"] != EXPECTED_SETS:
            raise er.GateFailure(f"{tag}: sets={info['sets']} != {EXPECTED_SETS}")
        if info["ignored_files"] != 0:
            raise er.GateFailure(f"{tag}: ignored_files={info['ignored_files']} != 0")
        if not (SEED_START <= info["seed_min"] and info["seed_max"] < SEED_END):
            raise er.GateFailure(f"{tag}: seed window [{info['seed_min']},{info['seed_max']}] outside pool B")
    if sorted(a_sets) != sorted(b_sets):
        raise er.GateFailure("E3a/E2 seed sets differ (not paired)")

    seed_list = sorted(a_sets)
    d_rank = [a_sets[s][0] - b_sets[s][0] for s in seed_list]
    d_pt = [a_sets[s][1] - b_sets[s][1] for s in seed_list]
    ci = er.bootstrap_ci(d_rank, d_pt)

    delta_rank, delta_pt = mean(d_rank), mean(d_pt)
    pt_lower_gt_zero = ci[2] > 0.0
    decision = (
        "CONFIRM: E3a > E2 on Δpt (95% CI lower bound > 0) -> river-transformer line stays OPEN"
        if pt_lower_gt_zero
        else "NO CONFIRMATION: Δpt 95% CI lower bound <= 0 -> CLOSE the river-transformer line"
    )
    report = {
        "comparison": "E3a - E2 (pool B, paired per-set)",
        "protocol": {
            "seed_key": SEED_KEY,
            "seed_start": SEED_START,
            "seed_range": [SEED_START, SEED_END],
            "sets": len(seed_list),
            "games": 4 * len(seed_list),
            "champion": "checkpoints/mortal.pth.at-148800",
            "challenger_E3a": "runs/exp/E3a/state.pth",
            "challenger_E2": "runs/exp/E2/state.pth",
            "evals": {"E3a": a_info, "E2": b_info},
        },
        "E3a": {
            "avg_rank": mean([a_sets[s][0] for s in seed_list]),
            "avg_pt": mean([a_sets[s][1] for s in seed_list]),
        },
        "E2": {
            "avg_rank": mean([b_sets[s][0] for s in seed_list]),
            "avg_pt": mean([b_sets[s][1] for s in seed_list]),
        },
        "delta_avg_rank": delta_rank,
        "delta_avg_rank_ci": [ci[0], ci[1]],
        "delta_avg_pt": delta_pt,
        "delta_avg_pt_ci": [ci[2], ci[3]],
        "pt_ci_lower_gt_zero": pt_lower_gt_zero,
        "rank_ci_crosses_0": ci[0] <= 0.0 <= ci[1],
        "prereg_decision": decision,
        "mdd_80pct_12k_independent": {"avg_rank": MDD_AVG_RANK_12K, "pt": MDD_PT_12K},
        "bootstrap": {"B": er.BOOTSTRAP_RESAMPLES, "rng_seed": er.BOOTSTRAP_SEED},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"E3a: avg_rank={report['E3a']['avg_rank']:.6f} pt={report['E3a']['avg_pt']:+.6f} | "
        f"E2: avg_rank={report['E2']['avg_rank']:.6f} pt={report['E2']['avg_pt']:+.6f}"
    )
    print(
        f"d_rank={delta_rank:+.6f} CI=[{ci[0]:+.6f},{ci[1]:+.6f}] "
        f"({'crosses 0' if report['rank_ci_crosses_0'] else 'excludes 0'}) | "
        f"d_pt={delta_pt:+.6f} CI=[{ci[2]:+.6f},{ci[3]:+.6f}]"
    )
    print(f"decision: {decision}")
    print(f"delta_report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
