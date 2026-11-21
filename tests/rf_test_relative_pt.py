#!/usr/bin/env python3
"""PR-46B relative-PT scorer tests: profiles, ties, rotations, bootstrap, failures.

Synthetic score vectors only: no checkpoint, native engine, dataset or frozen run is
read or written.  The negative controls are the ones required by the V3.3 protocol.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from riichiformer.eval import relative_pt as R  # noqa: E402

KEY = "00112233445566aa"


def _record(seed, rotation, scores, *, seat=None, key=KEY, game_id=None):
    return R.GameRecord(
        seed_key=key, wall_seed=seed, rotation=rotation,
        candidate_seat=rotation if seat is None else seat,
        scores=tuple(scores), game_id=game_id or f"{seed}:{rotation}")


def _set(seed, scores, *, key=KEY):
    return [_record(seed, rotation, scores, key=key) for rotation in range(4)]


class ProfileTests(unittest.TestCase):
    def test_profiles_are_exact(self):
        self.assertEqual(R.TOKUJOU_5D_HANCHAN, (75, 30, 0, -105))
        self.assertEqual(R.HOUOU_7D_HANCHAN, (90, 45, 0, -135))
        self.assertEqual(sum(R.TOKUJOU_5D_HANCHAN), 0)
        self.assertEqual(sum(R.HOUOU_7D_HANCHAN), 0)
        self.assertEqual(R.PRIMARY_PROFILE, "TOKUJOU_5D_HANCHAN")
        self.assertEqual(R.SECONDARY_PROFILE, "HOUOU_7D_HANCHAN")

    def test_policy_hash_is_deterministic_and_profile_specific(self):
        a = R.score_policy_hash()
        self.assertEqual(a, R.score_policy_hash())
        self.assertEqual(len(a), 64)
        self.assertNotEqual(a, R.score_policy_hash(R.SECONDARY_PROFILE))
        payload = R.score_policy()
        self.assertEqual(payload["tie_rule"], "scores_descending_then_lower_absolute_seat")
        self.assertEqual(payload["unit"], "complete_seed_set")


class RankTests(unittest.TestCase):
    def test_rank_by_scores_breaks_ties_by_lower_absolute_seat(self):
        self.assertEqual(R.rank_by_scores((25000, 25000, 25000, 25000)), (1, 2, 3, 4))
        self.assertEqual(R.rank_by_scores((25000, 25000, 30000, 20000)), (2, 3, 1, 4))
        # 1-2 tie and 3-4 tie both resolved by seat index.
        self.assertEqual(R.rank_by_scores((30000, 30000, 20000, 20000)), (1, 2, 3, 4))
        self.assertEqual(R.rank_by_scores((20000, 30000, 20000, 30000)), (3, 1, 4, 2))

    def test_supplied_ranks_must_agree(self):
        with self.assertRaises(R.RankMismatchError):
            R.GameRecord(KEY, 0, 0, 0, (30000, 25000, 25000, 20000), ranks=(2, 1, 3, 4))
        with self.assertRaises(R.RankMismatchError):
            R.GameRecord(KEY, 0, 0, 0, (30000, 25000, 25000, 20000), ranks=(1, 1, 3, 4))
        record = R.GameRecord(KEY, 0, 0, 0, (30000, 25000, 25000, 20000),
                              ranks=(1, 2, 3, 4))
        self.assertEqual(record.ranks, (1, 2, 3, 4))


class GameScoringTests(unittest.TestCase):
    def test_all_first_and_all_last(self):
        first = R.score_game(_record(1, 0, (40000, 20000, 20000, 20000)))
        self.assertEqual(first.pt5_candidate, 75)
        self.assertEqual(first.pt5_mortal_mean, -25.0)
        self.assertEqual(first.delta_pt5, 100.0)
        self.assertEqual(first.pt7_candidate, 90)
        self.assertEqual(first.delta_pt7, 120.0)

        last = R.score_game(_record(1, 0, (10000, 30000, 30000, 30000)))
        self.assertEqual(last.pt5_candidate, -105)
        self.assertEqual(last.pt5_mortal_mean, 35.0)
        self.assertEqual(last.delta_pt5, -140.0)

    def test_second_and_third(self):
        second = R.score_game(_record(1, 1, (29000, 30000, 21000, 20000)))
        self.assertEqual(second.ranks[1], 1)
        self.assertEqual(second.pt5_candidate, 30 + 45)
        third = R.score_game(_record(1, 2, (30000, 28000, 22000, 20000)))
        self.assertEqual(third.pt5_candidate, 0)
        self.assertEqual(third.delta_pt5, 0.0)

    def test_delta_equals_four_thirds_of_raw(self):
        for rank, scores in enumerate((
                (40000, 20000, 20000, 20000),
                (20000, 40000, 20000, 20000),
                (20000, 20000, 40000, 20000),
                (20000, 20000, 20000, 40000))):
            scored = R.score_game(_record(1, 0, scores, seat=rank))
            self.assertEqual(scored.delta_pt5, (4.0 / 3.0) * scored.pt5_candidate)
            self.assertAlmostEqual(scored.pt5_candidate
                                   - sum(scored.pt5_mortal_seats) / 3.0,
                                   scored.delta_pt5, places=12)

    def test_score_sum_is_checked(self):
        with self.assertRaises(R.ScoreSumError):
            R.GameRecord(KEY, 0, 0, 0, (25000, 25000, 25000, 24000))
        with self.assertRaises(R.ScoreSumError):
            R.rank_by_scores((30000, 30000, 30000, 30000))

    def test_incomplete_game_is_refused(self):
        with self.assertRaises(R.GameIncompleteError):
            R.GameRecord(KEY, 0, 0, 0, (25000, 25000, 25000, 25000), completed=False)


class CompleteSetTests(unittest.TestCase):
    def test_rotations_form_one_set_and_set_average(self):
        records = _set(7, (32600, 7300, 31100, 29000))
        sets = R.build_complete_sets(records, seed_key=KEY, expected_wall_seeds=[7])
        self.assertEqual(len(sets), 1)
        one = sets[0]
        self.assertEqual([g.rotation for g in one.games], [0, 1, 2, 3])
        self.assertEqual(sorted(g.candidate_seat for g in one.games), [0, 1, 2, 3])
        self.assertEqual(one.set_delta_pt5,
                         sum(g.delta_pt5 for g in one.games) / 4.0)
        self.assertEqual(one.set_pt5_candidate,
                         sum(g.pt5_candidate for g in one.games) / 4.0)
        self.assertEqual(one.rank_hist[0] + one.rank_hist[1]
                         + one.rank_hist[2] + one.rank_hist[3], 4)

    def test_balanced_self_control_has_zero_mean(self):
        # Same scores for all four rotations: a perfectly balanced self-control.
        sets = R.build_complete_sets(_set(1, (30000, 30000, 20000, 20000)))
        report = R.aggregate_sets(sets, bootstrap_resamples=200)
        self.assertEqual(report["delta_pt5"], 0.0)
        self.assertEqual(report["avg_rank"], 2.5)
        self.assertAlmostEqual(report["delta_pt5_ci95"]["low"], 0.0)
        self.assertAlmostEqual(report["delta_pt5_ci95"]["high"], 0.0)

    def test_missing_game_is_incomplete(self):
        records = _set(1, (25000, 25000, 25000, 25000))[:3]
        with self.assertRaises(R.IncompleteSetError):
            R.build_complete_sets(records)

    def test_duplicate_rotation_is_refused(self):
        records = _set(1, (25000, 25000, 25000, 25000))
        records[3] = _record(1, 2, (25000, 25000, 25000, 25000))
        with self.assertRaises(R.DuplicateGameError):
            R.build_complete_sets(records)

    def test_missing_expected_seed_is_refused(self):
        records = _set(1, (25000, 25000, 25000, 25000))
        with self.assertRaises(R.MissingSeedError):
            R.build_complete_sets(records, expected_wall_seeds=[1, 2])

    def test_duplicate_expected_seed_is_refused(self):
        records = _set(1, (25000, 25000, 25000, 25000))
        with self.assertRaises(R.DuplicateSeedError):
            R.build_complete_sets(records, expected_wall_seeds=[1, 1])

    def test_wrong_seed_key_is_refused(self):
        records = _set(1, (25000, 25000, 25000, 25000), key="ff" * 8)
        with self.assertRaises(R.KeyMismatchError):
            R.build_complete_sets(records, seed_key=KEY)

    def test_mixed_key_within_a_set_is_refused(self):
        records = _set(1, (25000, 25000, 25000, 25000))
        records[2] = _record(1, 2, (25000, 25000, 25000, 25000), key="ff" * 8)
        with self.assertRaises(R.KeyMismatchError):
            R.build_complete_sets(records)

    def test_duplicate_wall_seed_between_sets_is_refused(self):
        one = R.build_complete_sets(_set(1, (25000, 25000, 25000, 25000)))
        with self.assertRaises(R.DuplicateSeedError):
            R.aggregate_sets(one + one)

    def test_missing_rotation_of_candidate_seat_is_refused(self):
        records = _set(1, (25000, 25000, 25000, 25000))
        # Four distinct rotation ids, but the candidate seats repeat: the four-seat
        # rotation contract is broken.
        records = [_record(1, rotation, (25000, 25000, 25000, 25000), seat=0)
                   for rotation in range(4)]
        with self.assertRaises(R.IncompleteSetError):
            R.build_complete_sets(records)


class BootstrapTests(unittest.TestCase):
    def test_block_bootstrap_is_deterministic(self):
        values = [-140.0, 100.0, 0.0, 40.0, -40.0, 140.0]
        first = R.block_bootstrap_sets(values, resamples=2000, seed=123)
        second = R.block_bootstrap_sets(values, resamples=2000, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first["unit"], "complete_seed_set")
        self.assertEqual(first["sets"], 6)
        self.assertLess(first["low"], first["high"])
        self.assertAlmostEqual(first["point"], sum(values) / len(values))

    def test_bootstrap_rejects_small_or_bad_parameters(self):
        with self.assertRaises(R.RelativePTError):
            R.block_bootstrap_sets([1.0], resamples=10)
        with self.assertRaises(R.RelativePTError):
            R.block_bootstrap_sets([1.0], ci=1.5)

    def test_aggregate_has_no_old_gate_and_reports_per_set(self):
        records = []
        records += _set(1, (32600, 7300, 31100, 29000))
        records += _set(2, (30000, 30000, 20000, 20000))
        sets = R.build_complete_sets(records, expected_wall_seeds=[1, 2])
        report = R.aggregate_sets(sets, bootstrap_resamples=200)
        self.assertEqual(report["n_complete_sets"], 2)
        self.assertEqual(report["n_games"], 8)
        self.assertEqual(len(report["per_set"]), 2)
        self.assertEqual(report["seed_key"], KEY)
        self.assertEqual(report["score_policy_id"], "TOKUJOU_5D_HANCHAN")
        self.assertFalse(report["official_rank"])
        self.assertNotIn("3000", str(report.get("score_policy")))
        for item in report["per_set"]:
            for key in ("pt5_candidate", "delta_pt5", "avg_rank", "rank_hist",
                        "rotations", "candidate_seats"):
                self.assertIn(key, item)
        self.assertEqual(report["delta_pt5_ci95"]["resamples"], 200)
        self.assertEqual(report["delta_pt5_ci95"]["seed"], R.BOOTSTRAP_SEED)


class SeedKeyTests(unittest.TestCase):
    def test_key_normalisation(self):
        self.assertEqual(R.canonical_seed_key(1), "0000000000000001")
        self.assertEqual(R.canonical_seed_key("0XAbC"), "0000000000000abc")
        self.assertEqual(R.canonical_seed_key(0x1234567890ABCDEF), "1234567890abcdef")
        for bad in ("", "xyz", 2**64, -1, None, 1.5):
            with self.assertRaises(R.KeyMismatchError):
                R.canonical_seed_key(bad)


if __name__ == "__main__":
    unittest.main()
