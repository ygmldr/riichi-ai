#!/usr/bin/env python3
"""t9: resolve the SP array orientation — is index 0 of `own_tenpai_prob` / `own_win_prob`
the nearest or the farthest turn? (Inherited open item,
`.omo/plans/2026-09-22-commentary-analysis-engine-m1-m2.md:2501`.)

Conclusion: **nearest_first**. Index 0 = the next draw after the decision point (nearest
turn); index k = the same statistic conditional on k turns having elapsed without progress.
`own_tenpai_prob[0]` / `own_win_prob[0]` are the from-now cumulative values t6 must use;
index k > 0 is NOT a per-turn marginal.

Evidence (recomputed each run, never hardcoded): (1) every readable discard-SP row of two
frozen records is non-increasing with the first slot maximal and zeros only ever form a
suffix — the prefix `encode_sp_table` writes via `take_while(p > 0)` (`obs_repr.rs:644-692`);
(2) calc.rs `nanikiru` pins `tenpai_probs[0] = 0.90023905` for a 1-shanten candidate, above
the single-turn upper bound `57 / (121 - 14)`, so slot 0 cannot be the farthest turn;
(3) calc.rs gates haitei on the last index (`j == MAX_TSUMO - 1`) and double riichi on index 0.

The test writes `runs/exp/EX1/sp_orientation.json`; PREREG sha256 via `exp_ex1_audit.load_prereg`.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import unittest
from datetime import datetime, timezone

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from exp_ex1_audit import PREREG_PATH, assert_prereg_unchanged, load_prereg  # noqa: E402

CALC_RS = REPO_ROOT / "vendor/mortal/libriichi/src/algo/sp/calc.rs"
OUT_PATH = REPO_ROOT / "runs/exp/EX1/sp_orientation.json"
LAYOUT_PATH = REPO_ROOT / "runs/analysis/obs_layout.json"
PROBE_RECORDS = (
    "runs/1v3/10219_579411421376059130_d.json.gz", "runs/1v3/10000_423244137402621143_a.json.gz",
)
SP_TURNS = 17
DISCARD_ACTION_MAX = 36  # actions 0..36 = discards (aka included); 37 = riichi
PROB_EPS = 1e-9
ORIENTATIONS = ("nearest_first", "farthest_first", "undecidable")

# calc.rs `nanikiru` fixture 3, non-`sp_reproduce_cpp_ver` branch: hand "45677m 456778p 248s",
# discard 2s, 14-tile hand + 1 dora indicator seen, 15 tsumos left, 57 required tiles (17 kinds).
FIXTURE_SEEN_TILES = 15
FIXTURE_TSUMOS_LEFT = 15
FIXTURE_NUM_REQUIRED_TILES = 57
FIXTURE_TENPAI_PROB_0 = 0.90023905
TILES_TOTAL = 136

# Structural anchors in the read-only libriichi fixture file: code gates + pinned index-0
# values (machine-consumed facts), never prose.
FIXTURE_ANCHORS = {
    "haitei_bonus_at_last_index": r"let win_haitei = self\.sup\.calc_haitei && [ij] == MAX_TSUMO - 1;",
    "double_riichi_at_first_index": r"calc_double_riichi && i == 0;",
    "not_tsumo_row_zero_is_start": r"row\[0\] = 1\.;",
    "fixture_tsumos_left_15": r"let tsumos_left = 15;",
    "fixture_num_required_tiles_57": r"assert_eq!\(c\.num_required_tiles, 57\);",
    "fixture_tenpai_prob_0": r"assert!\(feq\(c\.tenpai_probs\[0\], 0\.90023905\)\);",
    "fixture_win_prob_0": r"assert!\(feq\(c\.win_probs\[0\], 0\.34794784\)\);",
}

COUNTERS = (
    "n_readable", "n_non_increasing_tenpai", "n_non_decreasing_tenpai",
    "n_strictly_decreasing_tenpai", "n_strictly_increasing_tenpai",
    "n_first_slot_is_row_max", "n_all_ones", "n_non_increasing_win",
    "n_zeros_form_suffix", "n_horizon_exceeds_array",
)


def single_turn_upper_bound() -> float:
    """Last-index upper bound: `tsumo_prob_table` times a factor <= 1, so `57 / (121 - 14)`."""
    n_left = TILES_TOTAL - FIXTURE_SEEN_TILES
    return FIXTURE_NUM_REQUIRED_TILES / (n_left - (FIXTURE_TSUMOS_LEFT - 1))


def _probe_record(path: pathlib.Path, t_off: int) -> dict:
    """Probe the delivered SP planes of one frozen record; returns the counter dict."""
    from jiangpu.record import read_record
    from jiangpu.replay import iter_decisions, tiles_left_from_obs
    from jiangpu.tiles import deaka

    counters = dict.fromkeys(COUNTERS, 0)
    for dec in iter_decisions(read_record(str(path))):
        row0 = dec.obs[t_off]
        if float(row0.max()) == 0.0 or not 0 <= dec.logged_action <= DISCARD_ACTION_MAX:
            continue  # legitimately unencoded (shanten >= 4 / non-discard action)
        col = deaka(dec.logged_action)
        if float(row0[col]) == 0.0:
            continue

        tenpai = [float(dec.obs[t_off + i][col]) for i in range(SP_TURNS)]
        win = [float(dec.obs[t_off + SP_TURNS + i][col]) for i in range(SP_TURNS)]
        horizon = max((i for i, v in enumerate(tenpai) if v > 0.0), default=-1)
        first_zero = next((i for i, v in enumerate(tenpai) if v == 0.0), SP_TURNS)
        counters["n_readable"] += 1
        counters["n_non_increasing_tenpai"] += all(
            tenpai[i] >= tenpai[i + 1] - PROB_EPS for i in range(SP_TURNS - 1)
        )
        counters["n_non_decreasing_tenpai"] += all(
            tenpai[i] <= tenpai[i + 1] + PROB_EPS for i in range(SP_TURNS - 1)
        )
        counters["n_strictly_decreasing_tenpai"] += tenpai[0] > tenpai[horizon] + 1e-6
        counters["n_strictly_increasing_tenpai"] += tenpai[0] + 1e-6 < tenpai[horizon]
        counters["n_first_slot_is_row_max"] += tenpai[0] >= max(tenpai) - PROB_EPS
        counters["n_all_ones"] += all(abs(v - 1.0) <= PROB_EPS for v in tenpai)
        counters["n_non_increasing_win"] += all(
            win[i] >= win[i + 1] - PROB_EPS for i in range(SP_TURNS - 1)
        )
        counters["n_zeros_form_suffix"] += all(v == 0.0 for v in tenpai[first_zero:])
        if any(a <= DISCARD_ACTION_MAX for a in dec.legal_actions):
            # SP array length = tsumos_left; slots beyond it are legitimately 0 (obs_repr.rs:644).
            counters["n_horizon_exceeds_array"] += horizon > tiles_left_from_obs(dec.obs) // 4 - 1
    return counters


def _decide_orientation(totals: dict) -> str:
    """Derive the orientation from the probe counters (never hardcoded)."""
    n = totals["n_readable"]
    if n == 0:
        return "undecidable"
    if totals["n_non_decreasing_tenpai"] == n:
        return "farthest_first" if totals["n_strictly_increasing_tenpai"] > 0 else "undecidable"
    if totals["n_non_increasing_tenpai"] == n and totals["n_strictly_decreasing_tenpai"] > 0:
        return "nearest_first"
    return "undecidable"


def run_probe() -> dict:
    """Probe both frozen records and aggregate the orientation counters."""
    from jiangpu import layout

    t_off, t_count = layout.span(layout.load_layout(), "sp_tables")
    if t_count != 3 * SP_TURNS:
        raise AssertionError(f"sp_tables channel count {t_count} != 3 * {SP_TURNS}")
    totals = dict.fromkeys(COUNTERS, 0)
    records = []
    for rel in PROBE_RECORDS:
        path = REPO_ROOT / rel
        counters = _probe_record(path, t_off)
        for key in COUNTERS:
            totals[key] += counters[key]
        records.append({
            "path": rel,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_readable": counters["n_readable"],
        })
    return {
        "sp_tables_span": [t_off, t_count],
        "sp_turns": SP_TURNS,
        "records": records,
        "totals": totals,
        "orientation": _decide_orientation(totals),
    }


def _evidence(probe: dict, bound: float) -> str:
    """One evidence string: what was measured, on what, and why it decides the orientation."""
    totals = probe["totals"]
    n = totals["n_readable"]
    records = "; ".join(
        f"{r['path']} sha256={r['sha256'][:16]}… ({r['n_readable']} rows)" for r in probe["records"]
    )
    denom = TILES_TOTAL - FIXTURE_SEEN_TILES - (FIXTURE_TSUMOS_LEFT - 1)
    return (
        f"nearest_first. (1) Delivered-engine probe [{records}]: {n} readable discard-SP rows; "
        f"own_tenpai_prob is non-increasing in {totals['n_non_increasing_tenpai']}/{n} rows "
        f"(non-decreasing only in {totals['n_non_decreasing_tenpai']}, all shanten==0 fill(1.)), "
        f"first slot is the row max in {totals['n_first_slot_is_row_max']}/{n}, zeros form a "
        f"suffix in {totals['n_zeros_form_suffix']}/{n} (the take_while(p>0) prefix), horizon "
        f"never exceeds tsumos_left-1 ({totals['n_horizon_exceeds_array']} violations); a "
        f"farthest-first array would be non-decreasing with the smallest value at slot 0. "
        f"(2) calc.rs nanikiru fixture: tenpai_probs[0]={FIXTURE_TENPAI_PROB_0} for a 1-shanten "
        f"candidate whose single-turn reach-tenpai probability is bounded by "
        f"{FIXTURE_NUM_REQUIRED_TILES}/{denom}={bound:.4f}; {FIXTURE_TENPAI_PROB_0} > {bound:.4f} "
        f"so slot 0 cannot be the farthest (single) turn. (3) calc.rs gates the haitei bonus on "
        f"j == MAX_TSUMO-1 (last index = last draw) and double riichi on i == 0 (index 0 = first turn)."
    )


def build_artifact(probe: dict, prereg_sha: str, anchors: dict) -> dict:
    """Assemble the t9 artifact: conclusion + evidence + PREREG hash + probe counters."""
    return {
        "schema": "ex1-sp-orientation/v1",
        "todo": "t9",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question": "Is index 0 of own_tenpai_prob / own_win_prob the nearest or the farthest turn?",
        "orientation": probe["orientation"],
        "evidence": _evidence(probe, single_turn_upper_bound()),
        "semantics": {
            "index_0": "the next draw after the decision point (nearest turn); array length = tsumos_left",
            "index_k": "k turns elapsed, conditional on no progress — NOT a per-turn marginal",
            "index_max": "the last draw of the hand (farthest); the haitei bonus is gated here",
            "own_tenpai_prob_0": "P(reach tenpai at some later turn) — the from-now cumulative",
            "own_win_prob_0": "P(win at some later turn) under the single-player (no-opponent) model",
            "t6_rule": (
                "align slot 0 with realized 'our seat reaches tenpai / wins by the end of the "
                "hand'; never read slot k as P(event exactly at turn k)"
            ),
            "horizon": "last slot with own_tenpai_prob > 0; zeros beyond it are legitimate",
        },
        "probe": probe,
        "fixture_anchors": anchors,
        "single_turn_bound": {
            "fixture": "vendor/mortal/libriichi/src/algo/sp/calc.rs::nanikiru (non-cpp branch)",
            "num_required_tiles": FIXTURE_NUM_REQUIRED_TILES,
            "tsumos_left": FIXTURE_TSUMOS_LEFT,
            "bound": single_turn_upper_bound(),
            "fixture_tenpai_prob_0": FIXTURE_TENPAI_PROB_0,
        },
        "farthest_first_rejected_because": (
            "profiles are non-increasing with the first slot maximal; the fixture index-0 value "
            "exceeds the single-turn upper bound"
        ),
        "proposed_alternative_probe": None,
        "prereg": {"path": "scripts/exp_ex1_prereg.md", "sha256": prereg_sha},
    }


class SpOrientationTest(unittest.TestCase):
    """t9: freeze the SP array orientation from the delivered planes + the libriichi fixture."""

    @classmethod
    def setUpClass(cls):
        cls.prereg_sha = load_prereg(PREREG_PATH)
        missing = [p for p in (CALC_RS, LAYOUT_PATH) if not p.exists()]
        missing += [REPO_ROOT / rel for rel in PROBE_RECORDS if not (REPO_ROOT / rel).exists()]
        if missing:
            raise unittest.SkipTest(
                "SP orientation fixtures missing: " + ", ".join(str(p) for p in missing)
            )
        cls.calc_rs = CALC_RS.read_text(encoding="utf-8")
        cls.anchors = {name: bool(re.search(pat, cls.calc_rs)) for name, pat in FIXTURE_ANCHORS.items()}
        cls.probe = run_probe()
        cls.artifact = build_artifact(cls.probe, cls.prereg_sha, cls.anchors)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(
            json.dumps(cls.artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls):
        assert_prereg_unchanged(PREREG_PATH, cls.prereg_sha)

    def test_tenpai_and_win_profiles_are_non_increasing(self):
        totals = self.probe["totals"]
        self.assertGreaterEqual(totals["n_readable"], 100)
        self.assertEqual(len(self.probe["records"]), len(PROBE_RECORDS))
        self.assertEqual(totals["n_non_increasing_tenpai"], totals["n_readable"])
        self.assertEqual(totals["n_non_increasing_win"], totals["n_readable"])
        self.assertEqual(totals["n_first_slot_is_row_max"], totals["n_readable"])
        self.assertGreater(totals["n_strictly_decreasing_tenpai"], 0)

    def test_only_tenpai_fill_rows_are_non_decreasing(self):
        totals = self.probe["totals"]
        self.assertEqual(totals["n_non_decreasing_tenpai"], totals["n_all_ones"])
        self.assertEqual(totals["n_strictly_increasing_tenpai"], 0)

    def test_zeros_only_ever_form_a_suffix(self):
        totals = self.probe["totals"]
        self.assertEqual(totals["n_zeros_form_suffix"], totals["n_readable"])
        self.assertEqual(totals["n_horizon_exceeds_array"], 0)

    def test_calc_fixture_anchors(self):
        for name, matched in self.anchors.items():
            self.assertTrue(matched, f"calc.rs fixture anchor missing: {name}")

    def test_fixture_index0_exceeds_the_single_turn_bound(self):
        bound = single_turn_upper_bound()
        self.assertLess(bound, 0.6)
        self.assertGreater(FIXTURE_TENPAI_PROB_0, bound)

    def test_orientation_is_nearest_first(self):
        self.assertEqual(self.probe["orientation"], "nearest_first")
        self.assertIn(self.artifact["orientation"], ORIENTATIONS)
        self.assertEqual(self.artifact["orientation"], "nearest_first")

    def test_artifact_round_trips_with_prereg_hash_and_evidence(self):
        loaded = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(loaded["orientation"], "nearest_first")
        self.assertTrue(loaded["evidence"].strip())
        self.assertEqual(loaded["prereg"]["sha256"], self.prereg_sha)
        self.assertIsNone(loaded["proposed_alternative_probe"])
        self.assertEqual(loaded["probe"]["totals"], self.probe["totals"])
        for key in ("semantics", "fixture_anchors", "single_turn_bound"):
            self.assertIn(key, loaded)


if __name__ == "__main__":
    unittest.main()
