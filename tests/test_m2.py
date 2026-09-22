import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class M2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jiangpu import aggregate
        from jiangpu.models import load_bundle, CHECKPOINT_FIXTURE
        from jiangpu.providers import q_ev, rank_dist, regret, sp_table
        from jiangpu.record import read_record, FROZEN_FIXTURE
        from jiangpu.replay import iter_decisions

        cls.aggregate = aggregate
        cls.record = read_record(FROZEN_FIXTURE)
        cls.decisions = iter_decisions(cls.record)[:80]
        cls.bundle = load_bundle(CHECKPOINT_FIXTURE, device="cpu")
        cls.q = q_ev.compute(cls.bundle, cls.decisions, device="cpu")
        cls.rd = rank_dist.compute(cls.bundle, cls.record, cls.decisions)
        cls.sp = sp_table.compute(cls.decisions)
        cls.rg = regret.compute(cls.decisions, cls.q)
        cls.out = aggregate.build(
            cls.decisions, cls.q, cls.rg, cls.rd, cls.record,
            final_scores=[25000, 25000, 25000, 25000], subject_player=0)

    def test_regret_fields(self):
        first = self.rg[self.decisions[0].index]
        self.assertIn("delta_rel", first)
        self.assertIn("is_blunder", first)
        self.assertIn(first["confidence"], {"uncalibrated", "calibrated"})

    def test_regret_delta_never_positive(self):
        for payload in self.rg.values():
            if "delta_rel" in payload:
                self.assertLessEqual(payload["delta_rel"], 1e-9)

    def test_key_moments_bounded_and_flagged(self):
        kms = self.out["key_moments"]
        self.assertLessEqual(len(kms), self.aggregate.KEY_MOMENT_TOP_K)
        for km in kms:
            self.assertIn("facts", km)
            self.assertTrue(km["flagged_by"])

    def test_key_moments_include_worst_regret_decision(self):
        worst = min((p["delta_rel"], i) for i, p in self.rg.items() if "delta_rel" in p)[1]
        self.assertIn(worst, {km["index"] for km in self.out["key_moments"]})

    def test_key_moments_facts_are_not_prose(self):
        for km in self.out["key_moments"]:
            for value in km["facts"].values():
                self.assertNotIsInstance(value, str,
                                         msg="facts 不得含自然语言句子(讲解属 W3)")

    def test_judgement_vs_outcome_present(self):
        jvo = self.out["game_summary"]["judgement_vs_outcome"]
        self.assertIn("judgement_score", jvo)
        self.assertIn("actual_rank", jvo)
        self.assertIn(jvo["divergence"],
                      {"aligned", "good_call_bad_result", "bad_call_good_result", "neutral"})

    def test_m3_metrics_declared_unavailable(self):
        metrics = {item["metric"] for item in self.out["unavailable"]}
        for name in ("opp_tenpai", "danger", "called_risk", "meld_decision_eval",
                     "dangerous_tiles"):
            self.assertIn(name, metrics)

    def test_kyoku_summaries_cover_observed_kyoku(self):
        keys = {s["kyoku"] for s in self.out["kyoku_summaries"]}
        self.assertEqual(keys, {d.at_kyoku for d in self.decisions})


if __name__ == "__main__":
    unittest.main()
