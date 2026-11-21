import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class RankDistTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jiangpu.models import load_bundle, CHECKPOINT_FIXTURE
        from jiangpu.record import read_record, FROZEN_FIXTURE
        from jiangpu.replay import iter_decisions
        cls.record = read_record(FROZEN_FIXTURE)
        cls.decisions = iter_decisions(cls.record)[:12]
        cls.bundle = load_bundle(CHECKPOINT_FIXTURE, device="cpu")

    def test_rows_sum_to_one(self):
        from jiangpu.providers import rank_dist
        out = rank_dist.compute(self.bundle, self.record, self.decisions)
        self.assertEqual(len(out), len(self.decisions))
        first = out[self.decisions[0].index]
        self.assertFalse(first.get("unavailable"), first.get("reason"))
        self.assertEqual(first["kind"], "state_level")
        self.assertEqual(first["unit"], "probability")
        self.assertEqual(len(first["matrix"]), 4)
        for row in first["matrix"]:
            self.assertEqual(len(row), 4)
            self.assertAlmostEqual(sum(row), 1.0, places=5)

    def test_expected_rank_in_range(self):
        from jiangpu.providers import rank_dist
        out = rank_dist.compute(self.bundle, self.record, self.decisions)
        for payload in out.values():
            if payload.get("unavailable"):
                continue
            self.assertGreaterEqual(payload["expected_rank"], 1.0)
            self.assertLessEqual(payload["expected_rank"], 4.0)

    def test_same_kyoku_shares_matrix(self):
        from jiangpu.providers import rank_dist
        out = rank_dist.compute(self.bundle, self.record, self.decisions)
        by_kyoku = {}
        for d in self.decisions:
            by_kyoku.setdefault(d.at_kyoku, set()).add(
                tuple(tuple(r) for r in out[d.index]["matrix"]))
        for _k, variants in by_kyoku.items():
            self.assertEqual(len(variants), 1, "同一局内应共享同一个顺位矩阵")

    def test_subject_player_recorded(self):
        from jiangpu.providers import rank_dist
        out = rank_dist.compute(self.bundle, self.record, self.decisions)
        self.assertEqual(out[self.decisions[0].index]["subject_player"],
                         self.decisions[0].player_id)


if __name__ == "__main__":
    unittest.main()
