import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class Gate1DecodeTest(unittest.TestCase):
    def test_decode_against_mask_bits(self):
        from jiangpu.gate1 import decode_meta_q
        out = decode_meta_q({"q_values": [-1.5, -0.25], "mask_bits": (1 << 0) | (1 << 3)})
        self.assertEqual(out, {0: -1.5, 3: -0.25})

    def test_decode_ascending_action_index_order(self):
        from jiangpu.gate1 import decode_meta_q
        bits = (1 << 1) | (1 << 5) | (1 << 9)
        out = decode_meta_q({"q_values": [10.0, 20.0, 30.0], "mask_bits": bits})
        self.assertEqual(list(out.keys()), [1, 5, 9])
        self.assertEqual(list(out.values()), [10.0, 20.0, 30.0])

    def test_decode_mismatch_raises(self):
        from jiangpu.gate1 import decode_meta_q
        with self.assertRaises(ValueError):
            decode_meta_q({"q_values": [1.0, 2.0], "mask_bits": 1 << 0})

    def test_decode_handles_empty(self):
        from jiangpu.gate1 import decode_meta_q
        self.assertEqual(decode_meta_q({"q_values": [], "mask_bits": 0}), {})
        self.assertEqual(decode_meta_q({}), {})

    def test_seat_of_split_matches_eval_report(self):
        from jiangpu.gate1 import seat_of_split
        self.assertEqual([seat_of_split(s) for s in "abcd"], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
