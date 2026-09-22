import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TilesTest(unittest.TestCase):
    def test_order_matches_libriichi(self):
        from jiangpu.tiles import TILE_STRINGS
        self.assertEqual(TILE_STRINGS[0], "1m")
        self.assertEqual(TILE_STRINGS[8], "9m")
        self.assertEqual(TILE_STRINGS[9], "1p")
        self.assertEqual(TILE_STRINGS[17], "9p")
        self.assertEqual(TILE_STRINGS[18], "1s")
        self.assertEqual(TILE_STRINGS[26], "9s")
        self.assertEqual(TILE_STRINGS[27:34], ["E", "S", "W", "N", "P", "F", "C"])
        self.assertEqual(TILE_STRINGS[34:37], ["5mr", "5pr", "5sr"])
        self.assertEqual(len(TILE_STRINGS), 38)

    def test_roundtrip(self):
        from jiangpu.tiles import tile_id, tile_str
        for tid in range(37):
            self.assertEqual(tile_id(tile_str(tid)), tid)

    def test_deaka_and_aka(self):
        from jiangpu.tiles import deaka, is_aka
        self.assertEqual(deaka(34), 4)
        self.assertEqual(deaka(35), 13)
        self.assertEqual(deaka(36), 22)
        self.assertEqual(deaka(3), 3)
        self.assertTrue(is_aka(34))
        self.assertFalse(is_aka(3))
        self.assertFalse(is_aka(37))

    def test_unknown_raises(self):
        from jiangpu.tiles import tile_str, tile_id
        with self.assertRaises(KeyError):
            tile_id("0z")
        with self.assertRaises(IndexError):
            tile_str(38)


if __name__ == "__main__":
    unittest.main()
