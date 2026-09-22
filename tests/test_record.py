import gzip
import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class RecordTest(unittest.TestCase):
    def test_reads_frozen_fixture(self):
        from jiangpu.record import read_record, FROZEN_FIXTURE
        rec = read_record(FROZEN_FIXTURE)
        self.assertEqual(rec.names, ["old148800", "old148800", "old148800", "mortal"])
        self.assertEqual(rec.names[3], "mortal")   # split=d → challenger 坐 3 号席
        self.assertEqual(rec.split, "d")
        self.assertGreater(len(rec.events), 100)
        self.assertEqual(rec.events[0]["type"], "start_game")
        self.assertEqual(rec.events[-1]["type"], "end_game")
        self.assertIn("start_kyoku", {e["type"] for e in rec.events})

    def test_filename_identity_parsed(self):
        from jiangpu.record import read_record, FROZEN_FIXTURE
        rec = read_record(FROZEN_FIXTURE)
        self.assertEqual((rec.seed, rec.key), (10219, 579411421376059130))

    def test_bad_filename_rejected(self):
        from jiangpu.record import read_record, RecordError, FROZEN_FIXTURE
        with tempfile.TemporaryDirectory() as td:
            bad = pathlib.Path(td) / "not_a_record.json.gz"
            bad.write_bytes(pathlib.Path(FROZEN_FIXTURE).read_bytes())
            with self.assertRaises(RecordError):
                read_record(str(bad))

    def test_missing_end_game_rejected(self):
        from jiangpu.record import read_record, RecordError, FROZEN_FIXTURE
        with gzip.open(FROZEN_FIXTURE, "rt", encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        with tempfile.TemporaryDirectory() as td:
            bad = pathlib.Path(td) / "1_2_a.json.gz"
            with gzip.open(bad, "wt", encoding="utf-8") as gz:
                gz.write("".join(lines[:-1]))
            with self.assertRaises(RecordError):
                read_record(str(bad))

    def test_raw_text_equals_file_text(self):
        from jiangpu.record import read_record, FROZEN_FIXTURE
        rec = read_record(FROZEN_FIXTURE)
        with gzip.open(FROZEN_FIXTURE, "rt", encoding="utf-8") as fh:
            self.assertEqual(rec.raw_text, fh.read())


if __name__ == "__main__":
    unittest.main()
