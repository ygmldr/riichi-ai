import copy
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _doc():
    return {
        "schema_version": 1, "engine_version": 1,
        "checkpoint": {"path": "x.pth", "sha256": "a" * 64, "steps": 396000},
        "record": {"path": "p.json.gz", "source": "frozen", "names": ["a"] * 4,
                   "seed": 1, "key": 2, "split": "a"},
        "decisions": [{
            "index": 0, "player_id": 0, "at_kyoku": 0, "at_turn": 1, "shanten": 3,
            "logged_action": 5, "legal_actions": [5, 6],
            "metrics": {
                "q_ev": {"provider": "q_ev", "provider_version": 1,
                         "checkpoint_sha256": "a" * 64, "unit": "pt_ev",
                         "q_rel": {"5": 0.0, "6": -0.5}, "confidence": "uncalibrated"},
                "sp_table": {"provider": "sp_table", "provider_version": 1,
                             "checkpoint_sha256": None, "unit": "normalized_ev",
                             "own_tenpai_prob": [0.1] * 17, "own_win_prob": [0.05] * 17,
                             "own_ev": [0.0] * 17},
                "rank_dist": {"provider": "rank_dist", "provider_version": 1,
                              "checkpoint_sha256": "b" * 64, "unit": "probability",
                              "kind": "state_level", "matrix": [[0.25] * 4] * 4,
                              "expected_rank": 2.5},
            },
        }],
        "kyoku_summaries": [], "game_summary": {}, "key_moments": [], "unavailable": [],
    }


class ContractTest(unittest.TestCase):
    def test_valid_passes(self):
        from jiangpu.contract import validate
        self.assertEqual(validate(_doc()), [])

    def test_missing_top_key(self):
        from jiangpu.contract import validate
        doc = _doc()
        del doc["unavailable"]
        self.assertTrue(any("unavailable" in e for e in validate(doc)))

    def test_provider_meta_required(self):
        from jiangpu.contract import validate
        doc = _doc()
        del doc["decisions"][0]["metrics"]["q_ev"]["unit"]
        self.assertTrue(any("unit" in e for e in validate(doc)))

    def test_rank_matrix_rows_must_sum_to_one(self):
        from jiangpu.contract import validate
        doc = _doc()
        doc["decisions"][0]["metrics"]["rank_dist"]["matrix"] = [[0.1] * 4] * 4
        self.assertTrue(any("row" in e.lower() for e in validate(doc)))

    def test_unavailable_needs_reason(self):
        from jiangpu.contract import validate
        doc = _doc()
        doc["unavailable"] = [{"metric": "danger"}]
        self.assertTrue(any("reason" in e for e in validate(doc)))

    def test_round_floats(self):
        from jiangpu.contract import round_floats
        self.assertEqual(round_floats({"a": 0.1234567891, "b": [1.0000004]}),
                         {"a": 0.123457, "b": [1.0]})

    def test_round_floats_does_not_touch_bools(self):
        from jiangpu.contract import round_floats
        self.assertEqual(round_floats({"ok": True}), {"ok": True})


if __name__ == "__main__":
    unittest.main()
