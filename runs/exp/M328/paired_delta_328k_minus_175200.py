#!/usr/bin/env python3
"""M328 paired delta: challenger at-328000 minus challenger at-175200.

Both challengers played the SAME 3000 seeds (window [10000,13000), seed_key
0x5dfaa4cef265cd7) against the SAME champion `checkpoints/mortal.pth.at-148800`
(name `old148800`):

  - runs/exp/M328/vs148800/games   challenger = runs/exp/PINNED/mortal.pth.at-328000
  - runs/exp/EVAL_A_175200/games   challenger = runs/exp/PINNED/mortal.pth.at-175200

Procedure: exactly `scripts/exp_a1_delta.py` (imported, never modified) — per-set
mean statistics replayed with `scripts/eval_report.py`'s verified NDJSON logic,
paired per-seed differences, set-level bootstrap B=2000 with numpy rng seed 0.

Usage:
  python3 runs/exp/M328/paired_delta_328k_minus_175200.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
import exp_a1_delta as ad  # noqa: E402

PINS = {
    "at-328000": REPO / "runs/exp/PINNED/mortal.pth.at-328000",
    "at-175200": REPO / "runs/exp/PINNED/mortal.pth.at-175200",
    "at-148800": REPO / "checkpoints/mortal.pth.at-148800",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ref_dir = REPO / "runs/exp/EVAL_A_175200/games"  # challenger at-175200
    arm_dir = REPO / "runs/exp/M328/vs148800/games"  # challenger at-328000

    ref_sets, ref_info = ad.per_set_stats(ref_dir)
    arm_sets, arm_info = ad.per_set_stats(arm_dir)
    seeds = sorted(ref_sets)
    if sorted(arm_sets) != seeds:
        raise ad.er.GateFailure("seed sets differ between the two corpora (not paired)")
    if len(seeds) != 3000:
        raise ad.er.GateFailure(f"expected 3000 sets, got {len(seeds)}")

    comp = ad.compare(ref_sets, arm_sets, "challenger_328k", seeds)
    comp["reference"] = "challenger_175200"
    comp["games_dir_arm"] = str(arm_dir.relative_to(REPO))
    comp["games_dir_reference"] = str(ref_dir.relative_to(REPO))
    comp["arm_info"] = arm_info
    comp["reference_info"] = ref_info
    comp["challenger_names"] = {"arm": "at-328000", "reference": "mortal (at-175200)"}
    comp["champion"] = "old148800 (checkpoints/mortal.pth.at-148800)"

    report = {
        "probe": "M328 paired delta: challenger at-328000 - challenger at-175200",
        "seed_key": 0x5DFAA4CEF265CD7,
        "seed_window": [10000, 13000],
        "champion": comp["champion"],
        "pins_sha256": {name: sha256(p) for name, p in PINS.items()},
        "comparison": comp,
        "mdd_80pct_12k": {"avg_rank": ad.MDD_AVG_RANK_12K, "pt": ad.MDD_PT_12K},
    }

    out = REPO / "runs/exp/M328/paired_delta_328k_minus_175200.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(comp, indent=2, sort_keys=True))
    print(f"paired_delta -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
