import pathlib
import sys
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class ReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jiangpu.record import read_record, FROZEN_FIXTURE
        cls.record = read_record(FROZEN_FIXTURE)

    def test_shapes_and_logged_action_legality(self):
        from jiangpu.replay import iter_decisions
        decisions = iter_decisions(self.record)
        self.assertGreater(len(decisions), 100)
        for d in decisions[:30]:
            self.assertEqual(d.obs.shape, (1012, 34))
            self.assertEqual(d.obs.dtype, np.float32)
            self.assertEqual(d.mask.shape, (46,))
            self.assertEqual(d.mask.dtype, np.bool_)
            self.assertTrue(d.mask[d.logged_action],
                            f"decision {d.index}: logged action {d.logged_action} 不合法")
            self.assertEqual(d.legal_actions,
                             [i for i, ok in enumerate(d.mask.tolist()) if ok])

    def test_decision_indices_are_globally_unique(self):
        from jiangpu.replay import iter_decisions
        idx = [d.index for d in iter_decisions(self.record)]
        # provider 以 Decision.index 为字典键;**跨 POV 必须全局唯一**,否则四视角互相覆盖
        self.assertEqual(len(idx), len(set(idx)), "Decision.index 必须全局唯一")

    def test_four_povs_partition_the_full_decision_stream(self):
        from jiangpu.replay import iter_decisions
        counts = {p: len(iter_decisions(self.record, pov=p)) for p in range(4)}
            # 各席位行动次数本来就不等(本冻结文件实测 [167,173,160,169],总计 669)
            # —— **不要**断言相等;M3 语料是另一组数(如 [256,251,245,240]),别混用
        self.assertTrue(all(n > 0 for n in counts.values()), counts)
        self.assertEqual(sum(counts.values()), len(iter_decisions(self.record)))

    def test_public_by_kyoku_from_events(self):
        from jiangpu.replay import public_by_kyoku
        pub = public_by_kyoku(self.record)
        self.assertGreaterEqual(len(pub), 1)
        first = pub[min(pub)]
        self.assertEqual(len(first["scores"]), 4)
        self.assertEqual(sum(first["scores"]), 100000)
        self.assertIn(first["bakaze"], {"E", "S", "W", "N"})
        self.assertGreaterEqual(len(first["dora_markers"]), 1)

    def test_decision_public_attached(self):
        from jiangpu.replay import iter_decisions
        d = iter_decisions(self.record)[0]
        self.assertEqual(set(d.public), {"bakaze", "kyoku", "honba", "kyotaku",
                                         "oya", "scores", "dora_markers"})
        self.assertIsInstance(d.public["scores"], list)

    def test_tiles_left_in_range(self):
        from jiangpu.replay import iter_decisions, tiles_left_from_obs
        d = iter_decisions(self.record)[0]
        n = tiles_left_from_obs(d.obs)
        self.assertGreaterEqual(n, 0)
        self.assertLessEqual(n, 70)


if __name__ == "__main__":
    unittest.main()
