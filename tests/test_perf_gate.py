import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class PerfGateTest(unittest.TestCase):
    def test_p95_under_budget_on_cpu(self):
        from jiangpu.perf_gate import P95_BUDGET_MS, run
        res = run("runs/1v3/10219_579411421376059130_d.json.gz",
                  "runs/exp/PINNED/mortal.pth.at-396000", device="cpu", limit=40)
        self.assertGreater(res["n_decisions"], 0)
        self.assertGreater(res["ms_per_decision"], 0.0)
        self.assertLess(res["p95_ms"], P95_BUDGET_MS)

    def test_budget_constant_matches_tenhou_plan(self):
        from jiangpu.perf_gate import P95_BUDGET_MS
        self.assertEqual(P95_BUDGET_MS, 500.0)


if __name__ == "__main__":
    unittest.main()
