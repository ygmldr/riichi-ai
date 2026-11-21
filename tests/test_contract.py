import json
import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

PREREG = REPO_ROOT / "scripts" / "exp_ex1_prereg.md"
CONTRACT_MARKER = "```json ex1-metric-contract"
FORBIDDEN_IMPORT = "exp_ex1_labels"


def _prereg_contract():
    """解析 PREREG §4 的 `json ex1-metric-contract` 冻结块。"""
    text = PREREG.read_text(encoding="utf-8")
    start = text.find(CONTRACT_MARKER)
    if start < 0:
        raise AssertionError(f"PREREG 缺少冻结契约块 {CONTRACT_MARKER!r}")
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end < 0:
        raise AssertionError("PREREG 冻结契约块未闭合")
    return json.loads(text[start:end])


def _forbidden_import_hits(root):
    """AST 扫描 root/**/*.py,返回每个 `exp_ex1_labels` import 的 `path:lineno`(NO-REVEAL)。"""
    import ast
    hits = []
    for path in sorted(pathlib.Path(root).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else [a.name for a in node.names]
            else:
                continue
            if any(n == FORBIDDEN_IMPORT or n.startswith(FORBIDDEN_IMPORT + ".")
                   for n in names):
                hits.append(f"{path}:{node.lineno}")
    return hits


def _doc():
    return {
        "schema_version": 2, "engine_version": 1,
        "checkpoint": {"path": "x.pth", "sha256": "a" * 64, "steps": 396000},
        "record": {"path": "p.json.gz", "source": "frozen", "names": ["a"] * 4,
                   "seed": 1, "key": 2, "split": "a"},
        "decisions": [{
            "index": 0, "player_id": 0, "at_kyoku": 0, "at_turn": 1, "shanten": 3,
            "logged_action": 5, "legal_actions": [5, 6],
            "metrics": {
                "q_ev": {"provider": "q_ev", "provider_version": 1,
                         "checkpoint_sha256": "a" * 64, "unit": "pt_ev",
                         "no_hidden_info": True, "basis": "self_hand",
                         "q_rel": {"5": 0.0, "6": -0.5}, "confidence": "uncalibrated"},
                "sp_table": {"provider": "sp_table", "provider_version": 1,
                             "checkpoint_sha256": None, "unit": "normalized_ev",
                             "no_hidden_info": True, "basis": "self_hand",
                             "own_tenpai_prob": [0.1] * 17, "own_win_prob": [0.05] * 17,
                             "own_ev": [0.0] * 17},
                "rank_dist": {"provider": "rank_dist", "provider_version": 1,
                              "checkpoint_sha256": "b" * 64, "unit": "probability",
                              "no_hidden_info": True, "basis": "public_info_predictor",
                              "kind": "state_level", "matrix": [[0.25] * 4] * 4,
                              "expected_rank": 2.5},
            },
        }],
        "kyoku_summaries": [], "game_summary": {}, "key_moments": [], "unavailable": [],
    }


def _opp_tenpai_payload():
    return {"provider": "opp_tenpai", "provider_version": 1,
            "checkpoint_sha256": "c" * 64, "unit": "probability",
            "no_hidden_info": True, "basis": "public_info_predictor",
            "opp_tenpai_prob": [0.1, 0.2, 0.3]}


def _danger_payload():
    return {"provider": "danger", "provider_version": 1,
            "checkpoint_sha256": "c" * 64, "unit": "mixed",
            "no_hidden_info": True, "basis": "public_info_predictor",
            "unit_by_field": {"deal_in": "probability",
                              "wait_membership": "probability", "exp_loss": "pt_loss"},
            "deal_in": [[0.0] * 34 for _ in range(3)],
            "wait_membership": [[0.5] * 34 for _ in range(3)],
            "exp_loss": [[0.0] * 34 for _ in range(3)]}


class ContractTest(unittest.TestCase):
    def _errors(self, name, payload):
        from jiangpu.contract import validate
        doc = _doc()
        doc["decisions"][0]["metrics"][name] = payload
        return validate(doc)

    def test_valid_passes(self):
        from jiangpu.contract import validate
        self.assertEqual(validate(_doc()), [])

    def test_schema_version_is_2(self):
        from jiangpu import contract
        self.assertEqual(contract.SCHEMA_VERSION, 2)
        doc = _doc()
        doc["schema_version"] = 1
        self.assertTrue(any("schema_version" in e for e in contract.validate(doc)))

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

    def test_provider_requires_no_hidden_info(self):
        payload = _doc()["decisions"][0]["metrics"]["q_ev"]
        del payload["no_hidden_info"]
        self.assertTrue(any("no_hidden_info" in e for e in self._errors("q_ev", payload)))

    def test_no_hidden_info_must_be_true(self):
        payload = _doc()["decisions"][0]["metrics"]["q_ev"]
        payload["no_hidden_info"] = False
        self.assertTrue(any("no_hidden_info" in e for e in self._errors("q_ev", payload)))

    def test_provider_requires_basis(self):
        payload = _doc()["decisions"][0]["metrics"]["q_ev"]
        del payload["basis"]
        self.assertTrue(any("basis" in e for e in self._errors("q_ev", payload)))

    def test_hindsight_basis_always_invalid(self):
        payload = _opp_tenpai_payload()
        payload["basis"] = "hindsight_exact"
        self.assertTrue(any("hindsight_exact" in e for e in self._errors("opp_tenpai", payload)))

    def test_unknown_basis_invalid(self):
        payload = _opp_tenpai_payload()
        payload["basis"] = "oracle_reveal"
        self.assertTrue(any("basis" in e for e in self._errors("opp_tenpai", payload)))

    def test_no_checkpoint_provider_whitelist(self):
        from jiangpu.contract import validate
        doc = _doc()
        doc["decisions"][0]["metrics"]["own_state"] = {
            "provider": "own_state", "provider_version": 1, "checkpoint_sha256": None,
            "unit": "count", "no_hidden_info": True, "basis": "self_hand",
            "shanten": 3, "ukeire_tiles": [1, 2]}
        self.assertEqual(validate(doc), [])
        doc["decisions"][0]["metrics"]["q_ev"]["checkpoint_sha256"] = None
        self.assertTrue(any("checkpoint_sha256" in e for e in validate(doc)))

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

    def test_valid_opp_tenpai_and_danger_pass(self):
        from jiangpu.contract import validate
        doc = _doc()
        doc["decisions"][0]["metrics"]["opp_tenpai"] = _opp_tenpai_payload()
        doc["decisions"][0]["metrics"]["danger"] = _danger_payload()
        self.assertEqual(validate(doc), [])

    def test_opp_tenpai_shape_must_be_3(self):
        payload = _opp_tenpai_payload()
        payload["opp_tenpai_prob"] = [0.1, 0.2, 0.3, 0.4]     # spec:120 的 [4] 口径偏差
        self.assertTrue(any("opp_tenpai_prob" in e for e in self._errors("opp_tenpai", payload)))

    def test_opp_tenpai_range(self):
        payload = _opp_tenpai_payload()
        payload["opp_tenpai_prob"] = [0.1, 0.2, 1.1]
        self.assertTrue(any("opp_tenpai_prob" in e for e in self._errors("opp_tenpai", payload)))

    def test_danger_shape(self):
        payload = _danger_payload()
        payload["deal_in"] = [[0.0] * 34 for _ in range(2)]
        self.assertTrue(any("deal_in" in e for e in self._errors("danger", payload)))

    def test_danger_probability_ranges(self):
        payload = _danger_payload()
        payload["wait_membership"] = [[1.5] * 34 for _ in range(3)]
        self.assertTrue(any("wait_membership" in e for e in self._errors("danger", payload)))

    def test_danger_exp_loss_is_continuous_not_unit_interval(self):
        payload = _danger_payload()
        payload["exp_loss"] = [[0.5] * 34 for _ in range(3)]
        self.assertEqual(self._errors("danger", payload), [])
        for bad in (48000.1, -0.1):
            payload["exp_loss"] = [[bad] * 34 for _ in range(3)]
            self.assertTrue(any("exp_loss" in e for e in self._errors("danger", payload)),
                            f"exp_loss={bad} 应越界")

    def test_danger_unit_must_be_mixed(self):
        payload = _danger_payload()
        payload["unit"] = "probability"
        self.assertTrue(any("unit 应为" in e for e in self._errors("danger", payload)))

    def test_danger_requires_unit_by_field(self):
        payload = _danger_payload()
        del payload["unit_by_field"]
        self.assertTrue(any("unit_by_field" in e for e in self._errors("danger", payload)))

    def test_frozen_contract_matches_prereg(self):
        from jiangpu import contract
        self.assertEqual(contract.FROZEN_METRIC_CONTRACT, _prereg_contract())

    def test_provider_basis_constants(self):
        from jiangpu.providers import q_ev, rank_dist, regret, sp_table
        for mod, basis in ((q_ev, "self_hand"), (sp_table, "self_hand"),
                           (regret, "self_hand"), (rank_dist, "public_info_predictor")):
            self.assertEqual(mod.BASIS, basis, mod.PROVIDER_NAME)
            self.assertIs(mod.NO_HIDDEN_INFO, True, mod.PROVIDER_NAME)

    def test_no_reveal_import_graph(self):
        hits = _forbidden_import_hits(REPO_ROOT / "scripts" / "jiangpu")
        self.assertEqual(hits, [], f"NO-REVEAL 违规:jiangpu 不得 import {FORBIDDEN_IMPORT}")

    def test_import_scanner_catches_violation(self):
        with tempfile.TemporaryDirectory() as td:
            bad = pathlib.Path(td) / "violation.py"
            bad.write_text(f"import {FORBIDDEN_IMPORT}\n"
                           f"from {FORBIDDEN_IMPORT} import extract_labels\n",
                           encoding="utf-8")
            self.assertEqual(len(_forbidden_import_hits(td)), 2)

    def test_no_public_info_only_field_name(self):
        hits = [str(p) for p in (REPO_ROOT / "scripts" / "jiangpu").rglob("*.py")
                if "public_info_only" in p.read_text(encoding="utf-8")]
        self.assertEqual(hits, [], "字段名必须是 basis,不得使用 public_info_only")


if __name__ == "__main__":
    unittest.main()
