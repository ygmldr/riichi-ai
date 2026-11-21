"""H1 `opp_tenpai` provider: feature parity, contract, unavailable path, determinism."""
import ast
import pathlib
import sys
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

RECORD = "runs/1v3/10219_579411421376059130_d.json.gz"
CKPT = "runs/exp/PINNED/mortal.pth.at-396000"
HEAD = "runs/exp/EX1/headA/headA.pt"
HEAD_SHA = "2dcb6d6a9baadd34b8af6a5e059a23c36ebbdd1e83e41c6bb45c290a71ad1aa5"


class OppTenpaiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jiangpu.providers import opp_tenpai
        from jiangpu.record import read_record
        from jiangpu.replay import iter_decisions

        cls.opp = opp_tenpai
        cls.record = read_record(RECORD)
        cls.decisions = list(iter_decisions(cls.record))[:12]

    def test_features_match_training_extractor(self):
        import exp_ex1_headA as H
        from jiangpu import opp_tenpai_features as F

        obs = np.stack([np.asarray(d.obs) for d in self.decisions])
        at = np.array([int(d.at_turn) for d in self.decisions], dtype=np.uint8)
        a = H.public_features(obs, at, H._spans())
        b = F.features_for_decisions(self.decisions)
        self.assertEqual(a.shape, (len(self.decisions), 526))
        self.assertTrue(np.array_equal(a, b))

    def test_payload_shape_metadata_and_range(self):
        out = self.opp.compute(self.decisions, checkpoint_path=HEAD, device="cpu")
        self.assertEqual(len(out), len(self.decisions))
        for dec in self.decisions:
            payload = out[dec.index]
            self.assertEqual(payload["provider"], "opp_tenpai")
            self.assertEqual(payload["unit"], "probability")
            self.assertEqual(payload["basis"], "public_info_predictor")
            self.assertIs(payload["no_hidden_info"], True)
            self.assertEqual(payload["checkpoint_sha256"], HEAD_SHA)
            probs = payload["opp_tenpai_prob"]
            self.assertEqual(len(probs), 3)
            self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_probabilities_match_direct_model(self):
        import torch
        import exp_ex1_headA as H

        out = self.opp.compute(self.decisions, checkpoint_path=HEAD, device="cpu")
        ckpt = torch.load(HEAD, map_location="cpu", weights_only=False)
        model = H.HeadA()
        model.load_state_dict(ckpt["state_dict"])
        x = H.public_features(
            np.stack([np.asarray(d.obs) for d in self.decisions]),
            np.array([int(d.at_turn) for d in self.decisions], dtype=np.uint8), H._spans())
        pred = H.predict(model, x, np.asarray(ckpt["mu"]), np.asarray(ckpt["sd"]), "cpu")
        for i, dec in enumerate(self.decisions):
            np.testing.assert_allclose(out[dec.index]["opp_tenpai_prob"],
                                       pred["opp_tenpai"][i], atol=1e-6)

    def test_without_checkpoint_is_explicitly_unavailable(self):
        out = self.opp.compute(self.decisions, checkpoint_path=None, device="cpu")
        self.assertEqual(len(out), len(self.decisions))
        for payload in out.values():
            self.assertTrue(payload["unavailable"])
            self.assertTrue(payload["reason"])
            self.assertNotIn("opp_tenpai_prob", payload)

    def test_deterministic(self):
        a = self.opp.compute(self.decisions, checkpoint_path=HEAD, device="cpu")
        b = self.opp.compute(self.decisions, checkpoint_path=HEAD, device="cpu")
        self.assertEqual(a, b)

    def test_provider_source_has_no_oracle_import(self):
        for path in (REPO_ROOT / "scripts/jiangpu/opp_tenpai_features.py",
                     REPO_ROOT / "scripts/jiangpu/providers/opp_tenpai.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.add(node.module or "")
            self.assertFalse(any("exp_ex1_labels" in name or "exp_ex1_headA" in name
                                 for name in imported), path)

    def test_cli_with_head_checkpoint_validates_and_exposes_metric(self):
        from jiangpu.cli import build_document
        from jiangpu.contract import validate
        doc = build_document(RECORD, CKPT, device="cpu", max_decisions=4,
                             head_checkpoint=HEAD)
        self.assertEqual(validate(doc), [])
        self.assertTrue(any("opp_tenpai" in d["metrics"] for d in doc["decisions"]))
        self.assertFalse(any(item["metric"] == "opp_tenpai" for item in doc["unavailable"]))

    def test_cli_without_head_checkpoint_degrades(self):
        from jiangpu.cli import build_document
        from jiangpu.contract import validate
        doc = build_document(RECORD, CKPT, device="cpu", max_decisions=4)
        self.assertEqual(validate(doc), [])
        self.assertFalse(any("opp_tenpai" in d["metrics"] for d in doc["decisions"]))
        self.assertTrue(any(item["metric"] == "opp_tenpai" and item["reason"]
                            for item in doc["unavailable"]))


if __name__ == "__main__":
    unittest.main()
