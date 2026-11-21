import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class ProvidersM1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jiangpu.models import load_bundle, CHECKPOINT_FIXTURE
        from jiangpu.record import read_record, FROZEN_FIXTURE
        from jiangpu.replay import iter_decisions
        cls.decisions = iter_decisions(read_record(FROZEN_FIXTURE))[:8]
        cls.bundle = load_bundle(CHECKPOINT_FIXTURE, device="cpu")

    def test_q_ev_shape_units_and_legal_keys(self):
        from jiangpu.providers import q_ev
        out = q_ev.compute(self.bundle, self.decisions, device="cpu")
        self.assertEqual(len(out), len(self.decisions))
        first = out[self.decisions[0].index]
        self.assertEqual(first["provider"], "q_ev")
        self.assertEqual(first["unit"], "pt_ev")
        self.assertEqual(len(first["checkpoint_sha256"]), 64)
        self.assertEqual(sorted(int(k) for k in first["q_rel"]),
                         self.decisions[0].legal_actions)
        self.assertAlmostEqual(max(first["q_rel"].values()), 0.0, places=9)

    def test_q_ev_confidence_not_fabricated(self):
        from jiangpu.providers import q_ev
        out = q_ev.compute(self.bundle, self.decisions, device="cpu")
        self.assertEqual(out[self.decisions[0].index]["confidence"], "uncalibrated")

    def test_q_ev_confidence_uses_calibration_when_given(self):
        from jiangpu.providers import q_ev
        out = q_ev.compute(self.bundle, self.decisions, device="cpu",
                           calibration={"gap_threshold": 1e9})
        self.assertEqual(out[self.decisions[0].index]["confidence"], "low")

    def test_sp_table_shapes_and_ranges(self):
        from jiangpu.providers import sp_table
        out = sp_table.compute(self.decisions)
        # r3: **不要**取 decisions[0] —— 它可能是 shanten>=4 的决策(SP 整行未编码,合法 unavailable)。
        # 必须挑一个确实可读的 payload 来断言字段形状。
        readable = [p for p in out.values()
                    if not p.get("unavailable") and p.get("subject_tile") is not None]
        self.assertTrue(readable, "样本里应至少有一个可读的 SP payload")
        first = readable[0]
        self.assertEqual(first["unit"], "normalized_ev")
        self.assertIsNone(first["checkpoint_sha256"])
        for key in ("own_tenpai_prob", "own_win_prob", "own_ev"):
            self.assertEqual(len(first[key]), sp_table.SP_TURNS)
        for v in first["own_tenpai_prob"] + first["own_win_prob"]:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_sp_table_reads_the_logged_discard_column(self):
        from jiangpu.providers import sp_table
        from jiangpu.record import read_record, FROZEN_FIXTURE
        from jiangpu.replay import iter_decisions
        # SP 平面在部分决策上**本来就不被编码**(整行全零/非候选列/立直),故不能断言
        # "第一个打牌决策必可用";但**必须至少有一个**打牌决策读到真实列,否则列读取没生效
        sample = iter_decisions(read_record(FROZEN_FIXTURE))[:60]
        payloads = sp_table.compute(sample)
        ok = [p for p in payloads.values()
              if not p.get("unavailable") and p.get("subject_tile") is not None]
        self.assertTrue(ok, "60 个决策里应至少有一个打牌决策能读到 SP 列")
        self.assertGreater(max(ok[0]["own_win_prob"]), 0.0)

    def test_sp_table_never_silently_zeroes(self):
        from jiangpu.providers import sp_table
        out = sp_table.compute(self.decisions)
        for payload in out.values():
            if payload.get("unavailable"):
                self.assertIn("reason", payload)
                self.assertTrue(payload["reason"])
            else:
                # 可用的 payload **不得**是"整段全零"——那正是 r2 抓到的静默填 0
                total = sum(payload["own_tenpai_prob"]) + sum(payload["own_win_prob"])
                self.assertGreater(total, 0.0, "可用的 SP payload 不得全零")


if __name__ == "__main__":
    unittest.main()
