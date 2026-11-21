import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class LayoutTest(unittest.TestCase):
    def test_schema_and_block_count(self):
        import json
        from jiangpu.layout import load_layout
        raw = json.loads((REPO_ROOT / "runs/analysis/obs_layout.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["schema"], "riichi-obs-layout/v1")
        self.assertEqual(raw["shape"], {"channels": 1012, "tile_kinds": 34})
        self.assertEqual(len(load_layout()), 64)

    def test_known_offsets(self):
        from jiangpu.layout import load_layout, span
        top = load_layout()
        self.assertEqual(span(top, "scores"), (7, 8))
        self.assertEqual(span(top, "dora_indicators"), (28, 7))
        self.assertEqual(span(top, "waits"), (860, 1))
        self.assertEqual(span(top, "at_furiten"), (861, 1))
        self.assertEqual(span(top, "shanten"), (862, 7))
        self.assertEqual(span(top, "tiles_left"), (717, 1))
        self.assertEqual(span(top, "discard_candidates"), (874, 5))
        self.assertEqual(span(top, "sp_ev"), (889, 2))
        self.assertEqual(span(top, "sp_required_tiles"), (891, 70))
        self.assertEqual(span(top, "sp_tables"), (961, 51))

    def test_unknown_name_raises(self):
        from jiangpu.layout import load_layout, span
        with self.assertRaises(KeyError):
            span(load_layout(), "no_such_block")


if __name__ == "__main__":
    unittest.main()
