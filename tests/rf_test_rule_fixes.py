#!/usr/bin/env python3
"""PR-40 (RULES) — executable Tenhou-rule fix fixtures for R02/R25/R26/R29.

These tests are the Python half of the PR-40 evidence. The native half lives in the
`#[cfg(test)] mod rule_tests` blocks added by `patches/0014-rule-r02-r25-r26.patch`:

* ``arena::board::rule_tests::r02_triple_ron_aborts_without_settlement``
* ``arena::board::rule_tests::r26_fourth_riichi_aborts_before_the_next_draw``
* ``arena::game::rule_tests::r25_last_dealer_stop_excludes_just_paid_kyotaku``

Run the native fixtures with::

    CARGO_HOME=<worktree>/.cargo-home PYO3_PYTHON=<venv>/bin/python \
      cargo test -p libriichi --lib --no-default-features rule_tests

Run this file with::

    PYTHONPATH=vendor/mortal/target/release \
      <python> -m pytest -q tests/rf_test_rule_fixes.py

The tests below are arithmetic/event-model fixtures over recorded or explicitly
constructed inputs. They do not re-implement the arena; they pin the before/after
settlement shapes that the native fixtures assert on the real board.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC = REPO_ROOT / "docs" / "riichiformer" / "v3" / "TENHOU_RULE_PARITY.md"
BOARD_RS = REPO_ROOT / "vendor" / "mortal" / "libriichi" / "src" / "arena" / "board.rs"
GAME_RS = REPO_ROOT / "vendor" / "mortal" / "libriichi" / "src" / "arena" / "game.rs"
RESULT_RS = REPO_ROOT / "vendor" / "mortal" / "libriichi" / "src" / "arena" / "result.rs"
PATCH = REPO_ROOT / "patches" / "0014-rule-r02-r25-r26.patch"


def ranks(scores: list[int]) -> list[int]:
    """Tenhou R14 tie-break: descending score, then E1 wind order (absolute seat)."""
    order = sorted(range(4), key=lambda i: (-scores[i], i))
    out = [0, 0, 0, 0]
    for rank, seat in enumerate(order, start=1):
        out[seat] = rank
    return out


# Recorded at runs/1v3/10089_579411421376059130_b.json.gz (split b), E3 honba 2, oya 2.
# The start vector is before the two reach payments; the recorded next start is
# [32100,25500,42100,300] with honba 3 and the same oya, which is the pre-fix settlement.
TRIPLE_RON_START = [20500, 23500, 30500, 23500]
TRIPLE_RON_RIICHI = [-1000, 0, 0, -1000]  # seats 0 and 3 accepted riichi
TRIPLE_RON_OLD_SCORES = [32100, 25500, 42100, 300]
TRIPLE_RON_ABORT_SCORES = [19500, 23500, 30500, 22500]


class R02TripleRonTests(unittest.TestCase):
    def test_pre_fix_settlement_and_fixed_abort_scores(self):
        # Pre-fix: three hora rows are paid (recorded log), reproducing the next start.
        observed = [[12600, 0, 0, -8600], [0, 2000, 0, -2000], [0, 0, 11600, -11600]]
        settled = [
            TRIPLE_RON_START[i] + TRIPLE_RON_RIICHI[i] + sum(row[i] for row in observed)
            for i in range(4)
        ]
        self.assertEqual(settled, TRIPLE_RON_OLD_SCORES)
        self.assertEqual(ranks(settled), [2, 3, 1, 4])
        self.assertEqual(sum(settled), 100000)

        # Fixed: 三家和了 = abortive draw. The riichi sticks stay on the table and no
        # hora delta is applied, so honba 2 -> 3 and the dealer (seat 2) repeats.
        abort = [TRIPLE_RON_START[i] + TRIPLE_RON_RIICHI[i] for i in range(4)]
        self.assertEqual(abort, TRIPLE_RON_ABORT_SCORES)
        self.assertEqual(ranks(abort), [4, 2, 1, 3])
        # Two 1000-point sticks are off the player scores and on the table.
        self.assertEqual(sum(abort) + 4 * 1000, 100000)
        # The four sticks (2 carried + 2 accepted here) are carried, not distributed:
        # 4 * 1000 points are off the player scores and on the table.
        self.assertEqual(sum(abort), 96000)
        self.assertNotEqual(settled, abort)

    def test_recorded_log_is_documented_as_pre_fix(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("10089_579411421376059130_b.json.gz", text)
        self.assertIn("Pre-fix recorded settlement", text)
        self.assertIn("PR-40 Tenhou abort", text)
        self.assertIn("19500", text)
        self.assertIn("42100", text)


class R25LastDealerKyotakuTests(unittest.TestCase):
    @staticmethod
    def stop(raw_scores, paid, winner, kyoku=7, length=8):
        """Mirror of game.rs::last_dealer_stop (comparison vector excludes the payout)."""
        chk = list(raw_scores)
        if winner is not None:
            chk[winner] -= paid * 1000
        oya = kyoku % 4  # S4 is kyoku 7 -> absolute seat 3
        if kyoku < length - 1 or chk[oya] < 30000:
            return False, chk
        top = min(range(4), key=lambda i: (-chk[i], i))
        return top == oya, chk

    def test_kyotaku_decisive_case_changes_the_end_decision(self):
        raw = [19000, 19000, 29000, 33000]  # 4 sticks paid to oya (seat 3)
        old_stop = raw[3] >= 30000 and min(range(4), key=lambda i: -raw[i]) == 3
        new_stop, chk = self.stop(raw, 4, 3)
        self.assertEqual(chk, [19000, 19000, 29000, 29000])
        self.assertTrue(old_stop)   # old, kyotaku-inclusive: ends
        self.assertFalse(new_stop)  # fixed: 29000 < 30000, continues
        # Final scores if the old rule ends here.
        self.assertEqual(ranks(raw), [3, 4, 2, 1])
        # Constructed continuation (not observed): seat 0 rons 12000 from seat 3.
        cont = [31000, 19000, 29000, 21000]
        self.assertEqual(ranks(cont), [1, 4, 2, 3])

    def test_genuine_30000_still_stops(self):
        raw = [19000, 19000, 28000, 34000]
        stop, chk = self.stop(raw, 4, 3)
        self.assertEqual(chk, [19000, 19000, 28000, 30000])
        self.assertTrue(stop)

    def test_docs_quote_the_exclusion(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("供託料を除く", text)
        self.assertIn("scores_excluding_hora_kyotaku", text)
        self.assertIn("tenhou-parity-verified", text)


class R26FourRiichiTests(unittest.TestCase):
    def test_native_fixture_asserts_no_draw_and_wall_unchanged(self):
        # The native fixture is the executable evidence; this check prevents the test
        # from being reduced to a hand-written list again.
        board = BOARD_RS.read_text(encoding="utf-8")
        start = board.index("fn r26_fourth_riichi_aborts_before_the_next_draw()")
        end = board.index("\n    }", start)
        body = board[start:end]
        self.assertIn("assert_eq!(state.tiles_left, 10);", body)
        self.assertIn("assert_eq!(state.board.yama.len(), 10);", body)
        self.assertIn("no tile may be drawn after the fourth riichi is accepted", body)
        self.assertIn("Some(Event::EndKyoku)", body)
        self.assertIn("assert_eq!(state.end().kyotaku_left, 5);", body)

    def test_native_fixture_is_present_and_ordered(self):
        board = BOARD_RS.read_text(encoding="utf-8")
        self.assertIn("fn r26_fourth_riichi_aborts_before_the_next_draw()", board)
        none_branch = board[board.index("Event::None => {"):board.index("Event::Dahai { actor, pai, .. } => {")]
        accept = none_branch.index("self.check_riichi_accepted();")
        guard = none_branch.index("if self.accepted_riichis == 4 {", accept)
        draw = none_branch.index("let tile = if self.deal_from_rinshan.take()", accept)
        self.assertLess(accept, guard)
        self.assertLess(guard, draw)
        self.assertNotIn("self.broadcast(&tsumo);", none_branch[:guard])


class R29IppatsuTests(unittest.TestCase):
    def test_status_stays_unverified(self):
        text = DOC.read_text(encoding="utf-8")
        rows = {}
        for line in text.splitlines():
            m = re.match(r"^\|\s*(R\d{2})\s*\|", line)
            if m:
                rows[m.group(1)] = [c.strip() for c in line.strip().strip("|").split("|")]
        self.assertEqual(rows["R29"][6], "UNVERIFIED")
        self.assertIn("no executable fixture", text)
        # The implementation is wired, but wiring is not Tenhou certification.
        self.assertIn("test_r29_ippatsu_window_is_wired", text)


class ReportVocabularyTests(unittest.TestCase):
    def test_parity_labels_and_patch_exist(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("tenhou-parity-verified", text)
        self.assertIn("tenhou-approx", text)
        self.assertIn("UNVERIFIED", text)
        self.assertNotIn("官方完全一致", text)
        self.assertTrue(PATCH.is_file(), f"missing {PATCH}")
        self.assertIn("apply to the parent vendor tree", text)

    def test_native_fixtures_are_declared(self):
        board = BOARD_RS.read_text(encoding="utf-8")
        game = GAME_RS.read_text(encoding="utf-8")
        result = RESULT_RS.read_text(encoding="utf-8")
        self.assertIn("fn r02_triple_ron_aborts_without_settlement()", board)
        self.assertIn("fn r26_fourth_riichi_aborts_before_the_next_draw()", board)
        self.assertIn("fn r25_last_dealer_stop_excludes_just_paid_kyotaku()", game)
        self.assertIn("pub fn scores_excluding_hora_kyotaku", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
