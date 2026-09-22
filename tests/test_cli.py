import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

RECORD = "runs/1v3/10219_579411421376059130_d.json.gz"
CKPT = "runs/exp/PINNED/mortal.pth.at-396000"


class CliTest(unittest.TestCase):
    def test_build_document_is_valid(self):
        from jiangpu.cli import build_document
        from jiangpu.contract import validate
        doc = build_document(RECORD, CKPT, device="cpu", max_decisions=6)
        self.assertEqual(validate(doc), [])
        self.assertEqual(doc["record"]["split"], "d")
        self.assertEqual(len(doc["decisions"]), 6)
        self.assertEqual(doc["checkpoint"]["steps"], 396000)
        # r4: **不要**假设 decisions[0] 必有 sp_table(它可能是合法未编码的决策)——
        # 与 Task 6 同一模式:对 sp_table 只要求"至少一个决策可用"
        self.assertIn("q_ev", doc["decisions"][0]["metrics"])
        self.assertIn("rank_dist", doc["decisions"][0]["metrics"])
        self.assertTrue(any("sp_table" in d["metrics"] for d in doc["decisions"]),
                        "至少一个决策应有 sp_table")

    def test_document_is_byte_reproducible(self):
        from jiangpu.cli import build_document
        a = build_document(RECORD, CKPT, device="cpu", max_decisions=4)
        b = build_document(RECORD, CKPT, device="cpu", max_decisions=4)
        self.assertEqual(json.dumps(a, sort_keys=True, ensure_ascii=False),
                         json.dumps(b, sort_keys=True, ensure_ascii=False))

    def test_cli_writes_valid_json_and_exit_zero(self):
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "doc.json"
            proc = subprocess.run(
                [sys.executable, "scripts/analyze_record.py",
                 "--record", RECORD, "--checkpoint", CKPT,
                 "--device", "cpu", "--max-decisions", "4", "--out", str(out)],
                cwd=REPO_ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            doc = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(doc["decisions"]), 4)
            self.assertEqual(doc["schema_version"], 1)

    def test_cli_exit_1_on_bad_record(self):
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "doc.json"
            proc = subprocess.run(
                [sys.executable, "scripts/analyze_record.py",
                 "--record", "runs/1v3/does_not_exist_a.json.gz",
                 "--checkpoint", CKPT, "--device", "cpu", "--out", str(out)],
                cwd=REPO_ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
