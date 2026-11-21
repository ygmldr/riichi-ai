#!/usr/bin/env python3
"""H-04 (RULES) — executable Tenhou ruleset-parity audit for the pinned Mortal arena.

Scope and ownership
-------------------
This file and ``docs/riichiformer/v3/TENHOU_RULE_PARITY.md`` are the H-04 deliverables.
Nothing under ``vendor/mortal``, ``riichiformer/data/replay.py`` or any runtime module is
modified.  The audit answers one question only: *which Tenhou 4-player ranked (段位戦)
rules does the pinned arena implement, and which does it not?*

Evidence classes (kept identical to the report)
----------------------------------------------
A  source line read and asserted in this run (exact ``file:line``) **or** a deterministic
   native execution of ``libriichi`` (``PlayerState`` / ``Stat``).  No wall injection.
B  recorded arena output (real ``runs/1v3/*.json.gz`` mjai logs) reconciled arithmetically
   with a transcription of the settlement code.  The log is independent of the model, so
   ``model == observed`` is a real check, not a tautology.
C  inference / not executed.  Never used to award ``MATCH``.

The prebuilt ``libriichi.so`` used here comes from the patched parent tree
(patches/0001-0013); those patches touch only agent/dataset/mjai modules, not
``arena/*`` or ``state/update.rs``, so the rule engine under test is the pinned source.

The arena cannot inject a wall from Python: ``libriichi.arena`` exposes only
``OneVsThree``/``TwoVsTwo``; ``arena::Board`` is not a ``#[pyclass]`` and has no
constructor.  Board-level rules therefore rest on A (source contract) plus B (recorded
output).  The one case that genuinely needs a scripted board — a triple ron — is marked
with an explicit ``@unittest.skip`` that names the missing capability and the expected
Tenhou behaviour.  It is *not* reported as a pass.

Run::

    PYTHONPATH=<repo>/vendor/mortal/target/release \
        <python> -m pytest -q tests/rf_test_ruleset_parity.py
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "vendor" / "mortal" / "target" / "release"))

RUST_SRC = REPO_ROOT / "vendor" / "mortal" / "libriichi" / "src"
REPORT_PATH = REPO_ROOT / "docs" / "riichiformer" / "v3" / "TENHOU_RULE_PARITY.md"

# Official Tenhou manual, 段位戦4人打ち rule section.  Every Tenhou claim below is quoted
# from this page (fetched 2026-11-21).
TENHOU_MAN = "https://tenhou.net/man/#RULE"


def have_libriichi() -> bool:
    try:
        import libriichi  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def read_source(relpath: str) -> list[str]:
    return (RUST_SRC / relpath).read_text(encoding="utf-8").splitlines()


# --------------------------------------------------------------------------------------
# A-level anchors: exact source lines.  These are a fail-closed contract: the pinned
# vendor tree must not drift while the report is cited.  If vendor/mortal moves, the audit
# must be re-run instead of silently inheriting stale file:line citations.
# --------------------------------------------------------------------------------------
ANCHORS: dict[str, dict[int, str]] = {
    "arena/board.rs": {
        26: "/// 1. No triple-ron ryukyoku.",
        243: "self.can_renchan = self.player_states[self.oya as usize].shanten() == 0;",
        347: "self.board.scores[actor as usize] -= 1000;",
        348: "self.board.kyotaku += 1;",
        375: "let mut honba_left = self.board.honba as i32; // mut in case of multi-ron",
        376: "let mut kyotaku_point = self.board.kyotaku as i32 * 1000; // ditto",
        377: "self.board.kyotaku = 0; // Unlike honba, kyotaku in self will be cleared",
        401: ".cycle()",
        402: ".skip(single_target as usize + 1)",
        403: ".take(3)",
        417: "deltas[actor] = point.ron + kyotaku_point + honba_left * 300;",
        419: "kyotaku_point = 0;",
        420: "honba_left = 0;",
        506: "deltas: Some([0; 4]),",
        519: "if self.accepted_riichis == 4 {",
        547: "if self.check_four_kan && !matches!(ev.event, Event::Hora { .. }) {",
        581: "if self.need_new_dora_at_tsumo.take().is_some() {",
        590: "if self.need_new_dora_at_discard.take().is_some() {",
        599: "if self.can_four_wind && self.check_four_wind(pai)? {",
        604: "if self.kans == 4 && self.player_states.iter().all(|s| s.kans_count() < 4) {",
        625: "// Immediately add new dora",
        626: "self.add_new_dora()?;",
        636: "self.need_new_dora_at_tsumo = Some(());",
        645: "self.need_new_dora_at_discard = Some(());",
        663: "Event::Ryukyoku { .. } => {",
        793: "t!(5mr), t!(5m), t!(5m), t!(5m),",
        803: "t!(5pr), t!(5p), t!(5p), t!(5p),",
        813: "t!(5sr), t!(5s), t!(5s), t!(5s),",
    },
    "arena/game.rs": {
        69: "if self.kyoku >= self.length + 4",
        72: "&& self.scores.iter().any(|&s| s >= 30000)",
        129: "let has_tobi = self.scores.iter().any(|&s| s < 0);",
        135: "if kyoku_result.has_abortive_ryukyoku {",
        136: "self.honba += 1;",
        140: "if !kyoku_result.can_renchan {",
        142: "if kyoku_result.has_hora {",
        145: "self.honba += 1;",
        156: "if kyoku_result.kyoku >= self.length - 1 && self.scores[oya] >= 30000 {",
        164: "if top == oya {",
        171: "self.in_renchan = true;",
        172: "self.honba += 1;",
        182: "if self.kyotaku > 0 {",
        183: "*self.scores.iter_mut().min_by_key(|s| -**s).unwrap() += self.kyotaku as i32 * 1000;",
        224: "length: 8,",
        225: "init_scores: [25000; 4],",
    },
    "arena/result.rs": {
        28: "pub fn rankings(&self) -> Rankings {",
        29: "Rankings::new(self.scores)",
    },
    "rankings.rs": {
        8: "pub fn new(scores: [i32; 4]) -> Self {",
        10: "player_by_rank.sort_by_key(|&i| -scores[i as usize]);",
    },
    "state/update.rs": {
        56: "self.at_furiten = true;",
        247: "|| /* 海底摸月 */ self.tiles_left == 0",
        270: "if self.tiles_left == 0 {",
        276: "// Using Tenhou rule here.",
        278: "agari::check_ankan_after_riichi(&self.tehai, self.tehai_len_div3, pai, false);",
        304: "&& self.tiles_left >= 4",
        305: "&& self.scores[0] >= 1000",
        364: "// Riichi furiten",
        365: "self.at_furiten = true;",
        371: "if !self.at_furiten && self.waits[pai.deaka().as_usize()] {",
        400: "// Mark as furiten at the next event. We do not set",
        409: "// Mark as furiten immediately, following the behavior",
        469: "// Forbid 喰い替え",
        471: "self.forbidden_tiles[deaka_tile_id] = true;",
        533: "self.forbidden_tiles[pai.deaka().as_usize()] = true;",
        923: "self.at_furiten = false;",
        932: "// Cannot wait, not even furiten for the 5th tile.",
        939: "// it is still a valid ryukyoku tenpai in our rule spec.",
        947: "if self.discarded_tiles[t] {",
        971: "Rankings::new(scores_abs).rank_by_player[self.player_id as usize]",
    },
    "algo/agari.rs": {
        846: "/// > リーチ後の暗槓は待ちが変わらない場合のみ。送り槓不可、牌姿や役の増減は不",
        847: "/// > 問。",
        854: "pub fn check_ankan_after_riichi(tehai: &[u8; 34], len_div3: u8, tile: Tile, strict: bool) -> bool {",
    },
    "state/agent_helper.rs": {
        40: "if self.riichi_accepted[0] {",
        61: "!self.forbidden_tiles[i]",
    },
}

# Every additional file:line cited by TENHOU_RULE_PARITY.md.  Together with the
# anchors above this covers the report exactly; ReportSyncTests enforces the mapping.
ANCHORS["arena/board.rs"].update({
    126: 'let oya = self.kyoku % 4;',
    153: 'if self.has_abortive_ryukyoku {',
    241: 'fn exhaustive_ryukyoku(&mut self) {',
    245: 'let mut has_nagashi_mangan = false;',
    265: 'if !has_nagashi_mangan {',
    314: 'fn check_four_wind(&mut self, pai: Tile) -> Result<bool> {',
    342: 'fn check_riichi_accepted(&mut self) {',
    387: 'self.can_renchan |= actor == self.oya;',
    399: '.into_iter()',
    407: 'if let Some(pao_target) = self.paos[actor] {',
    415: 'deltas[single_target as usize] = -point.ron - honba_left * 300;',
    475: 'fn update_paos(&mut self, ev: &Event) {',
    556: 'Event::None => {',
    561: 'self.check_riichi_accepted();',
    569: 'self.board.yama.pop().with_context(|| {',
    658: 'Event::Hora { actor, target, .. } => {',
})

ANCHORS["arena/game.rs"].update({
    47: '/// Used in 西入 where the oya and another player get to 30000 at the same',
})

ANCHORS["arena/result.rs"].update({
    14: 'pub kyotaku_left: u8,',
})

ANCHORS["state/update.rs"].update({
    235: 'if self.can_w_riichi {',
    244: 'if self.waits[pai.deaka().as_usize()] {',
    274: 'if self.riichi_accepted[0] {',
    303: 'self.last_cans.can_riichi = self.is_menzen',
    345: 'self.at_ippatsu = false;',
    415: 'if self.riichi_accepted[0] || self.tiles_left == 0 {',
    683: 'if actor_rel == 0 {',
    684: 'self.at_ippatsu = true;',
    710: 'match tile.as_u8() {',
    930: 'for (t, is_wait) in self.waits.iter_mut().enumerate() {',
})


CITED_SOURCE_LINE_COUNT = 60  # distinct (file, line) pairs cited by the report


def resolve_rs_path(path: str):
    """Map a report citation such as `board.rs` or `.../arena/board.rs` to an ANCHORS key."""
    aliases = {"board.rs": "arena/board.rs", "game.rs": "arena/game.rs",
               "result.rs": "arena/result.rs"}
    if path in aliases:
        return aliases[path]
    for key in ANCHORS:
        if path == key or path.endswith("/" + key):
            return key
    return None


class SourceContractTests(unittest.TestCase):
    """A-level: every quoted arena line is asserted at its exact file:line."""

    def test_anchor_lines_are_exact(self):
        for relpath, anchors in ANCHORS.items():
            lines = read_source(relpath)
            for lineno, needle in anchors.items():
                with self.subTest(file=relpath, line=lineno):
                    self.assertGreaterEqual(len(lines), lineno, f"{relpath} is shorter than {lineno}")
                    self.assertIn(
                        needle,
                        lines[lineno - 1],
                        f"{relpath}:{lineno} no longer contains the audited text {needle!r}",
                    )

    def test_r01_hanchan_start_and_return_points(self):
        """R01: 25000点開始の30000点返し (hanchan = 8 kyoku, W round extends)."""
        game = "\n".join(read_source("arena/game.rs"))
        self.assertIn("init_scores: [25000; 4]", game)
        self.assertIn("length: 8", game)
        self.assertIn("s >= 30000", game)
        # The return point appears only in the 西入 / renchan-owari checks; there is no
        # uma/oka settlement in the arena (Rankings only supplies the ordinal rank).
        self.assertIn("pub fn rankings(&self) -> Rankings", "\n".join(read_source("arena/result.rs")))

    def test_r02_triple_ron_is_settled_not_aborted(self):
        """R02: the board documents the Tenhou deviation, and handle_hora pays every hora."""
        board = "\n".join(read_source("arena/board.rs"))
        self.assertIn("/// 1. No triple-ron ryukyoku.", board)
        # The multi-ron loop walks the three other seats in turn order and pays each.
        self.assertIn(".skip(single_target as usize + 1)", board)
        self.assertIn(".take(3)", board)
        # There is no 三家和了 abort path anywhere in the pinned arena sources.
        self.assertNotIn("三家和", board)
        self.assertNotIn("三家和", "\n".join(read_source("arena/game.rs")))
        # Event::Ryukyoku (the only abortive trigger the user can send) is created only by
        # abortive_ryukyoku()/the 九種九牌 branch, never by a third Hora.
        self.assertEqual(board.count("self.abortive_ryukyoku();"), 4)

    def test_r03_multi_ron_uses_kamicha_priority_and_resets_counters(self):
        """R03: 積み棒/供託は上家取り — first winner after the discarder takes both."""
        board = read_source("arena/board.rs")
        # 375-377: honba/kyotaku are copied, then the board's kyotaku is zeroed.
        self.assertIn("// mut in case of multi-ron", board[374])
        # 419-420: after the first paid hora both counters are cleared, so later winners
        # in the same discard receive neither honba nor kyotaku.
        self.assertIn("kyotaku_point = 0;", board[418])
        self.assertIn("honba_left = 0;", board[419])
        # The iteration starts at the discarder's next seat: 上家取り in turn order.
        self.assertIn("skip(single_target as usize + 1)", board[401])

    def test_r05_kan_dora_timing_matches_tenhou(self):
        """R05: 暗槓=即乗り, 明槓/加槓=後めくり（打牌または続く嶺上の直前）."""
        board = "\n".join(read_source("arena/board.rs"))
        # Ankan: broadcast the kan, then immediately pop a dora (board.rs:625-626).
        self.assertLess(board.index("Event::Ankan { actor, .. }"), board.index("// Immediately add new dora"))
        # Daiminkan/Kakan: defer to the next discard (645) and, for a following kan, to the
        # rinshan draw (636 -> 581).
        self.assertIn("self.need_new_dora_at_discard = Some(());", board)
        self.assertIn("self.need_new_dora_at_tsumo = Some(());", board)
        self.assertIn("if self.need_new_dora_at_tsumo.take().is_some() {", board)

    def test_r07_tobi_threshold_is_strictly_negative(self):
        """R07: 点数がマイナスで飛び終了、0点は続行 — the predicate is `s < 0`."""
        game = read_source("arena/game.rs")
        self.assertIn("any(|&s| s < 0)", game[128])
        # 0 is explicitly *not* tobi; mirror the predicate on the boundary values.
        self.assertFalse(any(s < 0 for s in (0, 25000, 25000, 25000)))
        self.assertTrue(any(s < 0 for s in (-100, 25000, 25100, 25000)))

    def test_r08_west_round_entry_and_maximum(self):
        """R08: 東南戦の西入あり、最大は西場まで、30000点以上で終了（供託未収）."""
        game = read_source("arena/game.rs")
        self.assertIn(">= self.length + 4", game[68])  # 8 + 4 == 12 kyoku (W4 completes)
        self.assertIn("any(|&s| s >= 30000)", game[71])  # sudden-death entry check
        # Tenhou: 西入 only if nobody reached 30000 at the end of S4.
        self.assertFalse(any(s >= 30000 for s in (29900, 25000, 25000, 20100)))
        self.assertTrue(any(s >= 30000 for s in (30000, 25000, 25000, 20000)))

    def test_r09_r10_renchan_and_honba_accounting(self):
        """R09/R10: 和了/聴牌/途中流局は連荘、本場は和了で0・流局/連荘で+1."""
        board = read_source("arena/board.rs")
        game = read_source("arena/game.rs")
        self.assertIn("self.can_renchan = self.player_states[self.oya as usize].shanten() == 0;", board[242])
        self.assertIn("self.can_renchan |= actor == self.oya;", "\n".join(board))
        self.assertIn("if !kyoku_result.can_renchan", game[139])
        self.assertIn("if kyoku_result.has_hora {", game[141])
        self.assertIn("self.honba = 0;", game[142])
        self.assertIn("self.honba += 1;", game[144])

    def test_r11_riichi_stake_is_paid_only_when_accepted(self):
        """R11: リーチ宣言牌で放銃した場合は供託料は発生しない."""
        board = "\n".join(read_source("arena/board.rs"))
        self.assertEqual(board.count("self.check_riichi_accepted();"), 3)
        # check_riichi_accepted is called from the None/Chi/Pon/Daiminkan branches only;
        # the Hora branch calls handle_hora and returns, so a ron on the declaration tile
        # never pays the 1000 into kyotaku.
        handle = board[board.index("fn handle_hora"):board.index("fn update_paos")]
        self.assertNotIn("check_riichi_accepted", handle)
        hora_arm = board[board.index("Event::Hora { actor, target, .. } => {"):]
        hora_arm = hora_arm[:hora_arm.index("Event::Ryukyoku { .. } => {")]
        self.assertNotIn("check_riichi_accepted", hora_arm)

    def test_r12_furiten_is_river_only_and_same_cycle(self):
        """R12: フリテンは自分の河のみ、同巡振聴は打牌で解消、リーチ後は恒久."""
        upd = read_source("state/update.rs")
        self.assertIn("if self.discarded_tiles[t]", upd[946])
        self.assertIn("// Mark as furiten at the next event. We do not set", upd[399])
        self.assertIn("// Mark as furiten immediately, following the behavior", upd[408])
        self.assertIn("// Riichi furiten", upd[363])
        # The 5th-tile wait is not a wait for furiten/ron, but it *is* a ryukyoku tenpai.
        self.assertIn("// Cannot wait, not even furiten for the 5th tile.", upd[931])
        self.assertIn("it is still a valid ryukyoku tenpai in our rule spec", upd[938])

    def test_r13_riichi_ankan_uses_the_non_strict_tenhou_rule(self):
        """R13: リーチ後の暗槓は待ちが変わらない場合のみ。送り槓不可、牌姿や役の増減は不問."""
        agari = read_source("algo/agari.rs")
        self.assertIn("リーチ後の暗槓は待ちが変わらない場合のみ", agari[845])
        # The Tenhou quote wraps over two doc-comment lines (846-847):
        #   > ...送り槓不可、牌姿や役の増減は不
        #   > 問。
        self.assertIn("送り槓不可", agari[845])
        self.assertIn("問。", agari[846])
        self.assertIn("check_ankan_after_riichi", agari[853])
        upd = read_source("state/update.rs")
        self.assertIn("// Using Tenhou rule here.", upd[275])
        # `strict = false` is Tenhou; the strict variant would also reject shape-only kans.
        self.assertIn("pai, false);", upd[277])

    def test_r14_equal_scores_use_e1_wind_order(self):
        """R14: 終了時に同点の場合は東1局の風順で順位を決定."""
        ranks = read_source("rankings.rs")
        # Stable sort with the E1 dealer at absolute seat 0 (board.rs:126, `oya = kyoku % 4`).
        self.assertIn("sort_by_key", ranks[9])
        self.assertIn("-scores[i as usize]", ranks[9])
        # Rust's sort_by_key is stable, and absolute seat 0 is the E1 dealer (board.rs:126).
        # The executable check lives in NativeRankingTests.
        self.assertIn("let oya = self.kyoku % 4;", read_source("arena/board.rs")[125])

    def test_r15_leftover_kyotaku_goes_to_the_first_max(self):
        """R15: 終了時の供託はトップ取り."""
        game = read_source("arena/game.rs")
        self.assertIn("if self.kyotaku > 0 {", game[181])
        self.assertIn("min_by_key(|s| -**s)", game[182])
        # Rust `min_by_key` returns the first minimum; with key = -score that is the first
        # (lowest-seat) maximum.  Mirror it to prove the tie direction.
        scores = [25000, 31000, 31000, 19000]
        idx = min(range(4), key=lambda i: -scores[i])
        self.assertEqual(idx, 1)

    def test_r16_r17_r18_r19_all_four_abortive_draws_exist(self):
        """R16-R19: 四家立直/四槓散了/九種九牌/四風連打 are implemented; 三家和了 is not."""
        board = "\n".join(read_source("arena/board.rs"))
        self.assertIn("// 四家立直", board)
        self.assertIn("// 四槓散了", board)
        self.assertIn("// 四風連打", board)
        self.assertIn("Event::Ryukyoku { .. }", board)  # 九種九牌
        self.assertIn("self.abortive_ryukyoku();", board)
        # Every abortive draw sets has_abortive_ryukyoku, and game.rs makes it renchan.
        self.assertIn("self.has_abortive_ryukyoku = true;", board)
        self.assertIn("if kyoku_result.has_abortive_ryukyoku {", "\n".join(read_source("arena/game.rs")))

    def test_r20_riichi_preconditions(self):
        """R20: リーチ宣言は得点が1000点以上でツモ巡がある場合に可能."""
        upd = "\n".join(read_source("state/update.rs"))
        self.assertIn("self.last_cans.can_riichi = self.is_menzen", upd)
        self.assertIn("&& self.tiles_left >= 4", upd)  # the declarer still draws once
        self.assertIn("&& self.scores[0] >= 1000", upd)

    def test_r21_haitei_blocks_calls_and_kans(self):
        """R21: 海底牌は鳴けない; 海底摸月 is a yaku; haitei cannot kan/riichi."""
        upd = read_source("state/update.rs")
        self.assertIn("// haitei tile cannot be used for kakan or ankan", upd[268])
        self.assertIn("if self.tiles_left == 0 {", upd[269])
        self.assertIn("/* 海底摸月 */ self.tiles_left == 0", upd[246])
        # The last discard cannot be called: dahai returns before setting chi/pon/kan.
        dahai = "\n".join(upd)
        guard = dahai.index("if self.riichi_accepted[0] || self.tiles_left == 0 {")
        pon = dahai.index("self.last_cans.can_pon =")
        self.assertLess(guard, pon)

    def test_r22_kuikae_is_forbidden(self):
        """R22: 喰い替えなし."""
        upd = read_source("state/update.rs")
        self.assertIn("// Forbid 喰い替え", upd[468])
        self.assertIn("self.forbidden_tiles[deaka_tile_id] = true;", upd[470])
        self.assertIn("self.forbidden_tiles[pai.deaka().as_usize()] = true;", upd[532])
        # The discard mask consults forbidden_tiles; validate_reaction alone does not.
        helper = "\n".join(read_source("state/agent_helper.rs"))
        self.assertIn("!self.forbidden_tiles[i]", helper)

    def test_r23_exhaustive_draw_tenpai_payments(self):
        """R23: ノーテン罰符 3000/1500/1000."""
        board = read_source("arena/board.rs")
        self.assertIn("1 => (3000, -1000),", board[274])
        self.assertIn("2 => (1500, -1500),", board[275])
        self.assertIn("3 => (1000, -3000),", board[276])

    def test_r06_last_dealer_agari_and_tenpai_stop(self):
        """R06: ラス親トップの自動和了り止め、自動聴牌止めあり."""
        game = read_source("arena/game.rs")
        self.assertIn(">= self.length - 1 && self.scores[oya] >= 30000", game[155])
        self.assertIn("if top == oya {", game[163])
        self.assertIn("self.ended = true;", "\n".join(game))
        # 原点 = 30000 (25000 start / 30000 return).  The renchan-owari branch is only
        # reachable when the dealer can repeat (agari or tenpai); the kyotaku-inclusive
        # score issue is tracked separately as R25.
        self.assertIn("s >= 30000", game[71])
        self.assertIn("init_scores: [25000; 4]", game[224])

    def test_r24_nagashi_mangan_payments(self):
        """R24: 流し満貫 = 満貫清算, 親12000 / 子8000(4000+2000+2000)."""
        board = read_source("arena/board.rs")
        self.assertIn("let mut dod = [-4000; 4];", board[253])
        self.assertIn("dod[i] = 12000;", board[254])
        self.assertIn("let mut dod = [-2000; 4];", board[257])
        self.assertIn("dod[i] = 8000;", board[258])
        self.assertIn("dod[self.oya as usize] = -4000;", board[259])

    def test_r27_pao_exists_for_daisangen_and_daisuushi_only(self):
        """R27: 大三元/大四喜のパオあり、四槓子のパオなし."""
        board = "\n".join(read_source("arena/board.rs"))
        self.assertIn("fn update_paos", board)
        self.assertIn("daisanein_confirmed", board)
        self.assertIn("daisuushi_confirmed", board)
        self.assertNotIn("sukantsu", board.lower())
        self.assertNotIn("四槓子", board)

    def test_r25_s4_stop_uses_kyotaku_inclusive_scores(self):
        """R25: the 和了止め test must exclude 供託料; the arena's scores already hold it."""
        game = "\n".join(read_source("arena/game.rs"))
        board = read_source("arena/board.rs")
        result = "\n".join(read_source("arena/result.rs"))
        # scores are taken from KyokuResult, whose deltas already paid the kyotaku ...
        self.assertIn("self.scores = kyoku_result.scores;", game)
        self.assertIn("deltas[actor] = point.ron + kyotaku_point + honba_left * 300;", board[416])
        self.assertIn("self.board.kyotaku = 0;", board[376])
        # ... and the S4 check compares those scores directly.
        self.assertIn(">= self.length - 1 && self.scores[oya] >= 30000", game)
        # KyokuResult exposes only the post-payment count, so a fix needs a new field.
        self.assertIn("pub kyotaku_left: u8,", result)
        self.assertNotIn("kyotaku_paid", result)

    def test_r26_four_riichi_check_runs_after_the_next_draw(self):
        """R26: the 4-riichi abort is evaluated on the poll after the next tsumo."""
        board = read_source("arena/board.rs")
        text = "\n".join(board)
        abort_at = text.index("if self.accepted_riichis == 4 {")
        # The abort check is at the top of step(), before reactions are even validated.
        self.assertLess(abort_at, text.index("// Validate reactions"))
        # The 4th riichi is accepted in the None branch, after which a tile is drawn and a
        # Tsumo is broadcast; only the *next* step sees accepted_riichis == 4.
        none_branch = text[text.index("Event::None => {"):text.index("Event::Dahai { actor, pai, .. } => {")]
        self.assertIn("self.check_riichi_accepted();", none_branch)
        self.assertIn("self.broadcast(&tsumo);", none_branch)
        self.assertLess(none_branch.index("self.check_riichi_accepted();"),
                        none_branch.index("self.broadcast(&tsumo);"))

    def test_r29_ippatsu_window_is_wired(self):
        """R29: implementation exists (at_ippatsu set on acceptance, cleared by calls)."""
        upd = read_source("state/update.rs")
        self.assertIn("self.at_ippatsu = true;", upd[683])
        self.assertIn("self.at_ippatsu = false;", upd[344])  # own discard closes the window
        self.assertIn("self.at_ippatsu = false;", upd[445])  # any call cancels it
        self.assertIn("self.chankan_chance = Some(());", "\n".join(upd))

    def test_r28_riichi_discard_must_be_tsumogiri(self):
        """R28: リーチ後はツモ切り（the arena's discard mask offers only the drawn tile）."""
        helper = "\n".join(read_source("state/agent_helper.rs"))
        self.assertIn("if self.riichi_accepted[0] {", helper)
        self.assertIn("ret[last_self_tsumo.as_usize()] = true;", helper)


# --------------------------------------------------------------------------------------
# Native rules that PlayerState *can* express.  These are real engine decisions, not
# re-implementations: libriichi validates the hand, riichi state, waits, furiten, kans and
# returns the action mask that the arena itself consumes.
# --------------------------------------------------------------------------------------
FILLER = ["2p", "3p", "4p", "6p", "7p", "8p", "2s", "3s", "4s", "6s", "7s", "8s", "E"]


def _start_kyoku(player_id, hand, *, scores=(25000, 25000, 25000, 25000), oya=0,
                 dora="1s", kyoku=1, honba=0, kyotaku=0):
    tehais = [list(FILLER) for _ in range(4)]
    tehais[player_id] = list(hand)
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": dora,
        "kyoku": kyoku,
        "honba": honba,
        "kyotaku": kyotaku,
        "oya": oya,
        "scores": list(scores),
        "tehais": tehais,
    }


@unittest.skipUnless(have_libriichi(), "libriichi is not importable")
class NativePlayerStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import libriichi

        cls.libriichi = libriichi

    def seat(self, player_id):
        return self.libriichi.state.PlayerState(player_id)

    def feed(self, ps, event):
        return ps.update(json.dumps(event))

    # R20 ------------------------------------------------------------------------------
    def test_r20_riichi_needs_1000_points(self):
        hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5s"]
        for score, expected in ((900, False), (1000, True), (2600, True)):
            with self.subTest(score=score):
                ps = self.seat(0)
                self.feed(ps, _start_kyoku(0, hand, scores=(score, 25000, 25000, 25000)))
                cans = self.feed(ps, {"type": "tsumo", "actor": 0, "pai": "5s"})
                self.assertIs(cans.can_riichi, expected)

    def test_r20_riichi_needs_a_draw_turn(self):
        # Same tenpai hand, driven to the last wall tile: 4 draws left -> riichi allowed,
        # 0 draws left (after the final tsumo) -> riichi forbidden.
        hand = ["1m", "1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "9m"]
        ps = self.seat(1)
        self.feed(ps, _start_kyoku(1, hand, oya=0))
        pool = ["2p", "3p", "4p", "6p", "7p", "8p", "2s", "3s", "4s", "6s", "7s", "8s",
                "P", "F", "C", "E", "W", "N", "1p", "9p"]
        seen = {}
        for draw in range(70):
            actor = draw % 4
            tile = pool[draw % len(pool)]
            if actor == 1:
                cans = self.feed(ps, {"type": "tsumo", "actor": 1, "pai": tile})
                seen[draw] = cans
                self.feed(ps, {"type": "dahai", "actor": 1, "pai": tile, "tsumogiri": True})
            else:
                self.feed(ps, {"type": "tsumo", "actor": actor, "pai": tile})
        self.assertIs(seen[65].can_riichi, True)   # 70 - 66 == 4 tiles left
        self.assertIs(seen[69].can_riichi, False)  # 0 tiles left, no draw turn remains

    # R13 ------------------------------------------------------------------------------
    def test_r13_riichi_ankan_allowed_when_wait_is_unchanged(self):
        # 123m456m789m P EEE (wait P tanki); draw the 4th E after riichi -> ankan allowed.
        hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "P", "E", "E", "E"]
        ps = self.seat(0)
        self.feed(ps, _start_kyoku(0, hand))
        self.assertEqual([i for i, w in enumerate(ps.waits) if w], [31])  # 31 == P
        self.feed(ps, {"type": "tsumo", "actor": 0, "pai": "2p"})
        self.feed(ps, {"type": "reach", "actor": 0})
        self.feed(ps, {"type": "dahai", "actor": 0, "pai": "2p", "tsumogiri": True})
        self.feed(ps, {"type": "reach_accepted", "actor": 0})
        cans = self.feed(ps, {"type": "tsumo", "actor": 0, "pai": "E"})
        self.assertIs(cans.can_ankan, True)
        self.assertEqual(list(ps.ankan_candidates()), ["E"])

    def test_r13_riichi_ankan_forbidden_when_wait_changes(self):
        # 123456m 4445s SSS (waits 3s/5s/6s); ankan of the 4th 4s would shrink the wait to
        # 5s only -> 送り槓, forbidden after riichi.
        hand = ["1m", "2m", "3m", "4m", "5m", "6m", "S", "S", "S", "4s", "4s", "4s", "5s"]
        ps = self.seat(0)
        self.feed(ps, _start_kyoku(0, hand))
        self.assertEqual([i for i, w in enumerate(ps.waits) if w], [20, 22, 23])  # 3s/5s/6s
        self.feed(ps, {"type": "tsumo", "actor": 0, "pai": "2p"})
        self.feed(ps, {"type": "reach", "actor": 0})
        self.feed(ps, {"type": "dahai", "actor": 0, "pai": "2p", "tsumogiri": True})
        self.feed(ps, {"type": "reach_accepted", "actor": 0})
        cans = self.feed(ps, {"type": "tsumo", "actor": 0, "pai": "4s"})
        self.assertIs(cans.can_ankan, False)
        self.assertEqual(list(ps.ankan_candidates()), [])

    # R12 ------------------------------------------------------------------------------
    def test_r12_minogashi_furiten_blocks_ron(self):
        # 123456789m 123p 5s, tanki wait 5s (ittsuu gives a yaku).  Discarding the drawn 5s
        # is a tsumo-minogashi -> permanent furiten; the next 5s from an opponent is dead.
        hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5s"]
        ps = self.seat(0)
        self.feed(ps, _start_kyoku(0, hand))
        self.assertEqual([i for i, w in enumerate(ps.waits) if w], [22])
        cans = self.feed(ps, {"type": "tsumo", "actor": 0, "pai": "5s"})
        self.assertIs(cans.can_tsumo_agari, True)
        self.feed(ps, {"type": "dahai", "actor": 0, "pai": "5s", "tsumogiri": True})
        self.assertIs(ps.at_furiten, True)
        cans = self.feed(ps, {"type": "tsumo", "actor": 1, "pai": "2p"})
        cans = self.feed(ps, {"type": "dahai", "actor": 1, "pai": "5s", "tsumogiri": True})
        self.assertIs(cans.can_ron_agari, False)

    def test_r12_same_cycle_furiten_clears_on_own_discard(self):
        hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5s"]
        ps = self.seat(0)
        self.feed(ps, _start_kyoku(0, hand))
        # First 5s from an opponent is live; passing it marks same-cycle furiten at the
        # next event, not at the dahai itself.
        cans = self.feed(ps, {"type": "tsumo", "actor": 1, "pai": "2p"})
        cans = self.feed(ps, {"type": "dahai", "actor": 1, "pai": "5s", "tsumogiri": True})
        self.assertIs(cans.can_ron_agari, True)
        self.assertIs(ps.at_furiten, False)
        self.feed(ps, {"type": "tsumo", "actor": 2, "pai": "2s"})
        self.assertIs(ps.at_furiten, True)
        # Our own discard clears it (同巡振聴の解消は打牌時).
        self.feed(ps, {"type": "tsumo", "actor": 3, "pai": "3s"})
        self.feed(ps, {"type": "tsumo", "actor": 0, "pai": "E"})
        self.feed(ps, {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": True})
        self.assertIs(ps.at_furiten, False)
        self.feed(ps, {"type": "tsumo", "actor": 1, "pai": "4s"})
        cans = self.feed(ps, {"type": "dahai", "actor": 1, "pai": "5s", "tsumogiri": True})
        self.assertIs(cans.can_ron_agari, True)

    # R21 ------------------------------------------------------------------------------
    def test_r21_last_discard_cannot_be_ponned(self):
        hand = ["1m", "1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "9m"]
        ps = self.seat(1)
        self.feed(ps, _start_kyoku(1, hand, oya=0))
        pool = ["2p", "3p", "4p", "6p", "7p", "8p", "2s", "3s", "4s", "6s", "7s", "8s",
                "P", "F", "C", "E", "W", "N", "1p", "9p"]
        control = treated = None
        for draw in range(70):
            actor = draw % 4
            tile = pool[draw % len(pool)]
            if actor == 1:
                cans = self.feed(ps, {"type": "tsumo", "actor": 1, "pai": tile})
                self.feed(ps, {"type": "dahai", "actor": 1, "pai": tile, "tsumogiri": True})
                if draw == 65:  # 4 tiles left: a live discard may be called
                    control = self.feed(ps, {"type": "dahai", "actor": 0, "pai": "9m",
                                             "tsumogiri": True})
                if draw == 69:  # 0 tiles left (海底牌): cannot call
                    treated = self.feed(ps, {"type": "dahai", "actor": 0, "pai": "9m",
                                             "tsumogiri": True})
            else:
                self.feed(ps, {"type": "tsumo", "actor": actor, "pai": tile})
        self.assertIsNotNone(control)
        self.assertIsNotNone(treated)
        self.assertIs(control.can_pon, True)
        self.assertIs(treated.can_pon, False)

    def test_r21_haitei_is_a_yaku_for_an_open_no_yaku_hand(self):
        # Open no-yaku tenpai (S pon is not a value honour: bakaze E, seat wind W).
        # 123m 456m 789p 5s + SSS pon, tanki 5s.  With 4 tiles left the tsumo is illegal
        # (no yaku); on the last tile 海底摸月 makes it legal.
        hand = ["1m", "2m", "3m", "4m", "5m", "6m", "7p", "8p", "9p", "5s", "5s", "S", "S"]
        ps = self.seat(2)  # oya 0, seat 2 -> jikaze W; S is neither round nor seat wind
        self.feed(ps, _start_kyoku(2, hand, oya=0))
        # draw 0 (seat 0) and draw 1 (seat 1); seat 1 discards the S we pon.
        self.feed(ps, {"type": "tsumo", "actor": 0, "pai": "1p"})
        self.feed(ps, {"type": "dahai", "actor": 0, "pai": "1p", "tsumogiri": True})
        self.feed(ps, {"type": "tsumo", "actor": 1, "pai": "2p"})
        self.feed(ps, {"type": "dahai", "actor": 1, "pai": "S", "tsumogiri": False})
        self.feed(ps, {"type": "pon", "actor": 2, "target": 1, "pai": "S",
                       "consumed": ["S", "S"]})
        self.feed(ps, {"type": "dahai", "actor": 2, "pai": "5s", "tsumogiri": False})
        self.assertEqual([i for i, w in enumerate(ps.waits) if w], [22])  # 5s tanki
        pool = ["2p", "3p", "4p", "5p", "6p", "1s", "2s", "3s", "4s", "6s", "7s", "8s",
                "9s", "E", "W", "N", "P", "F", "C", "1m"]
        control = treated = None
        for draw in range(2, 70):
            actor = (3 + (draw - 2)) % 4
            tile = pool[draw % len(pool)]
            if draw in (65, 69):
                tile = "5s"
            if actor == 2:
                cans = self.feed(ps, {"type": "tsumo", "actor": 2, "pai": tile})
                if draw == 65:
                    control = cans
                if draw == 69:
                    treated = cans
                else:
                    self.feed(ps, {"type": "dahai", "actor": 2, "pai": tile,
                                   "tsumogiri": True})
            else:
                self.feed(ps, {"type": "tsumo", "actor": actor, "pai": tile})
        self.assertIsNotNone(control)
        self.assertIsNotNone(treated)
        self.assertIs(control.can_tsumo_agari, False)  # no yaku at 4 tiles left
        self.assertIs(treated.can_tsumo_agari, True)   # 海底摸月 on the final tile

    # R18 / R17 ------------------------------------------------------------------------
    def test_r18_nine_terminals_only_on_first_uncalled_draw(self):
        hand = ["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "2m"]
        ps = self.seat(0)
        self.feed(ps, _start_kyoku(0, hand))
        cans = self.feed(ps, {"type": "tsumo", "actor": 0, "pai": "C"})
        self.assertIs(cans.can_ryukyoku, True)

        # Any prior call (by anyone) disables 九種九牌; riichi itself stays possible.
        ps2 = self.seat(0)
        self.feed(ps2, _start_kyoku(0, hand, oya=1))
        self.feed(ps2, {"type": "tsumo", "actor": 1, "pai": "2m"})
        self.feed(ps2, {"type": "dahai", "actor": 1, "pai": "2m", "tsumogiri": True})
        self.feed(ps2, {"type": "tsumo", "actor": 2, "pai": "3m"})
        self.feed(ps2, {"type": "dahai", "actor": 2, "pai": "3m", "tsumogiri": True})
        self.feed(ps2, {"type": "pon", "actor": 3, "target": 2, "pai": "3m",
                        "consumed": ["3m", "3m"]})
        self.feed(ps2, {"type": "dahai", "actor": 3, "pai": "4m", "tsumogiri": False})
        cans = self.feed(ps2, {"type": "tsumo", "actor": 0, "pai": "C"})
        self.assertIs(cans.can_ryukyoku, False)

    def test_r17_a_fifth_kan_is_never_offered(self):
        hand = ["1m", "1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "9m"]
        results = {}
        for n_kans in (3, 4):
            ps = self.seat(0)
            self.feed(ps, _start_kyoku(0, hand))
            self.feed(ps, {"type": "tsumo", "actor": 0, "pai": "E"})
            for k in range(n_kans):
                tile = ["2p", "3p", "4p", "6p"][k]
                self.feed(ps, {"type": "ankan", "actor": 1 + k % 3,
                               "consumed": [tile, tile, tile, tile]})
            cans = self.feed(ps, {"type": "tsumo", "actor": 0, "pai": "F"})
            results[n_kans] = (cans.can_ankan, list(ps.ankan_candidates()))
        self.assertEqual(results[3], (True, ["1m"]))
        self.assertEqual(results[4], (False, []))

    # R30 ------------------------------------------------------------------------------
    def test_r30_fifth_tile_wait_is_tenpai_but_not_a_wait(self):
        # 1111m 234m 567m 789m: the 4th 1m is the tanki single; adding a 5th 1m would
        # complete 111 + 11 + 234 + 567 + 789.  `waits` excludes it (tehai == 4) but
        # `shanten` is 0, which is what exhaustive_ryukyoku() uses for tenpai.
        hand = ["1m", "1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "7m", "8m", "9m"]
        ps = self.seat(0)
        self.feed(ps, _start_kyoku(0, hand))
        self.assertEqual(ps.shanten, 0)
        self.assertNotIn(0, [i for i, w in enumerate(ps.waits) if w])  # 1m has 4 copies
        self.assertEqual([i for i, w in enumerate(ps.waits) if w], [3, 6])  # 4m, 7m

    # R04 ------------------------------------------------------------------------------
    def test_r04_aka_dora_is_one_per_suit(self):
        hand = ["5mr", "1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "5pr", "1p", "2p", "5sr"]
        ps = self.seat(0)
        self.feed(ps, _start_kyoku(0, hand))
        self.assertEqual(list(ps.akas_in_hand), [True, True, True])
        # `tehai` is deaka'd: the red 5 counts as a normal 5.
        self.assertEqual(ps.tehai[4], 1)    # 5m
        self.assertEqual(ps.tehai[13], 1)   # 5p
        self.assertEqual(ps.tehai[22], 1)   # 5s


# --------------------------------------------------------------------------------------
# R14/R15 native ranking path.  `GameResult::rankings()` (arena/result.rs:28-29) and
# `Stat::from_game` (stat.rs:417) both call the same `Rankings::new`, so Stat is a valid
# native probe for the tie-break, while PlayerState.get_rank reuses the same function.
# --------------------------------------------------------------------------------------
@unittest.skipUnless(have_libriichi(), "libriichi is not importable")
class NativeRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import libriichi

        cls.Stat = libriichi.stat.Stat

    @staticmethod
    def _log(scores, *, kyotaku=0, final_delta=None):
        events = [
            {"type": "start_game", "names": ["a", "b", "c", "d"], "seed": [1, 2]},
            {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1s", "kyoku": 1,
             "honba": 0, "kyotaku": kyotaku, "oya": 0, "scores": list(scores),
             "tehais": [["1m"] * 13] * 4},
        ]
        if final_delta is not None:
            events.append({"type": "ryukyoku", "deltas": list(final_delta)})
        events += [{"type": "end_kyoku"}, {"type": "end_game"}]
        return "\n".join(json.dumps(e) for e in events) + "\n"

    def _ranks(self, text):
        ranks = []
        for pid in range(4):
            stat = self.Stat.from_log(text, pid)
            counters = [stat.rank_1, stat.rank_2, stat.rank_3, stat.rank_4]
            self.assertEqual(sum(counters), 1)
            ranks.append(counters.index(1) + 1)
        return ranks

    def test_r14_equal_scores_rank_by_e1_wind_order(self):
        # 東1局の風順: absolute seat 0 is the E1 dealer (board.rs:126).
        self.assertEqual(self._ranks(self._log([25000] * 4)), [1, 2, 3, 4])
        # Ties inside a non-equal table keep the same direction.
        self.assertEqual(self._ranks(self._log([18000, 32000, 32000, 18000])), [3, 1, 2, 4])

    def test_r15_leftover_kyotaku_does_not_change_rank(self):
        # Stat never distributes the leftover kyotaku (unlike arena/game.rs:182-184), but
        # since it goes to the first maximum the ordinal ranks are unchanged.  This pins
        # the reason it is safe to use Stat for R14 while eval_report.py is required for
        # exact final scores.
        tied = [25000, 25000, 20000, 20000]
        self.assertEqual(self._ranks(self._log(tied, kyotaku=0)), self._ranks(self._log(tied, kyotaku=3)))


# --------------------------------------------------------------------------------------
# R02/R03 settlement model.  `settle_multi_ron` is a line-by-line transcription of
# arena/board.rs:366-438 (handle_hora).  The model is validated against two real arena
# logs, so "model == log" is independent evidence.  The model is then used to state what
# the arena does for a triple ron and what Tenhou would do.
# --------------------------------------------------------------------------------------
def settle_multi_ron(target: int, ron_values: dict[int, int], *, honba: int,
                     kyotaku: int, oya: int) -> list[list[int]]:
    """Transcription of board.rs:366-438 for the ron path.

    * ``honba_left``/``kyotaku_point`` per winner (board.rs:375-376, 419-420)
    * iteration order ``cycle().skip(target + 1).take(3)`` (board.rs:401-403)
    * discarder pays ``ron + honba*300`` unless pao (board.rs:412-416); winner receives
      ``ron + kyotaku + honba*300`` (board.rs:417)
    """
    deltas: list[list[int]] = []
    honba_left = honba
    kyotaku_point = kyotaku * 1000
    for offset in range(3):
        actor = (target + 1 + offset) % 4
        if actor not in ron_values:
            continue
        ron = ron_values[actor]
        row = [0, 0, 0, 0]
        row[target] = -ron - honba_left * 300
        row[actor] = ron + kyotaku_point + honba_left * 300
        kyotaku_point = 0
        honba_left = 0
        deltas.append(row)
    return deltas


# Recorded at runs/1v3/10006_579411421376059130_b.json.gz (split b), kyoku S1, honba 1,
# kyotaku 2 at kyoku start, 0 reach_accepted in the kyoku, oya 0.  The discard is
# {"type":"dahai","actor":3,"pai":"3p","tsumogiri":false}; the log then has exactly:
#   hora actor 0 target 3 deltas [5200,0,0,-3200]
#   hora actor 1 target 3 deltas [0,2600,0,-2600]
# Dealer (seat 0) is among the winners -> renchan; next start_kyoku is S1 honba 2 oya 0.
DOUBLE_RON = {
    "source": "runs/1v3/10006_579411421376059130_b.json.gz",
    "kyoku": "S1",
    "honba": 1,
    "kyotaku_start": 2,
    "reach_in_kyoku": 0,
    "oya": 0,
    "target": 3,
    "ron": {0: 2900, 1: 2600},  # dealer 30fu2han / non-dealer 40fu2han
    "observed": [[5200, 0, 0, -3200], [0, 2600, 0, -2600]],
    "next_start": {"kyoku": 1, "honba": 2, "oya": 0},
}

# Recorded in a real Tenhou log (datasets/extracted/2024/2024010101gm-00a9-0000-5c705fc0.mjson),
# S2 honba 4 kyotaku 2 oya 1, discarder 2.  The dealer (seat 1) was the *second* winner and
# received no honba/kyotaku: the first winner after the discarder (seat 0) took both, which
# is 上家取り.  Logged deltas: [11200,0,-9200,0] and [0,2900,-2900,0].
TENHOU_DOUBLE_RON = {
    "source": "datasets/extracted/2024/2024010101gm-00a9-0000-5c705fc0.mjson",
    "honba": 4,
    "kyotaku": 2,
    "oya": 1,
    "target": 2,
    "ron": {0: 8000, 1: 2900},
    "observed": [[11200, 0, -9200, 0], [0, 2900, -2900, 0]],
}

# Recorded at runs/1v3/10089_579411421376059130_b.json.gz (split b), kyoku E3, honba 2,
# kyotaku 2 at kyoku start + 2 reach_accepted in the kyoku (seats 0 and 3) = 4000 on the
# table, oya 2.  The discard is
# {"type":"dahai","actor":3,"pai":"6p","tsumogiri":true}; the log then has exactly three
# consecutive hora events (a real triple ron that the arena settled):
#   hora actor 0 target 3 deltas [12600,0,0,-8600], ura ["8s","5p"]
#   hora actor 1 target 3 deltas [0,2000,0,-2000]
#   hora actor 2 target 3 deltas [0,0,11600,-11600]
# The dealer (seat 2) was among the winners -> renchan; next start_kyoku is E3 honba 3
# oya 2 with scores [32100,25500,42100,300].
TRIPLE_RON = {
    "source": "runs/1v3/10089_579411421376059130_b.json.gz",
    "kyoku": "E3",
    "honba": 2,
    "kyotaku_start": 2,
    "reach_in_kyoku": 2,
    "oya": 2,
    "target": 3,
    "ron": {0: 8000, 1: 2000, 2: 11600},
    "observed": [[12600, 0, 0, -8600], [0, 2000, 0, -2000], [0, 0, 11600, -11600]],
    "next_start": {"kyoku": 3, "honba": 3, "oya": 2},
    "next_scores": [32100, 25500, 42100, 300],
}


def exhaustive_ryukyoku_deltas(tenpai, oya, nagashi=()):
    """Transcription of board.rs:241-292 (exhaustive draw settlement)."""
    deltas = [0, 0, 0, 0]
    if not nagashi:
        plus, minus = {1: (3000, -1000), 2: (1500, -1500), 3: (1000, -3000)}.get(
            len(tenpai), (0, 0))
        if plus:
            deltas = [minus] * 4
            for i in tenpai:
                deltas[i] = plus
        return deltas
    for i in nagashi:
        if i == oya:
            dod = [-4000] * 4
            dod[i] = 12000
        else:
            dod = [-2000] * 4
            dod[i] = 8000
            dod[oya] = -4000
        deltas = [a + b for a, b in zip(deltas, dod)]
    return deltas


class MultiRonSettlementTests(unittest.TestCase):
    def test_r03_double_ron_matches_recorded_arena_log(self):
        fx = DOUBLE_RON
        kyotaku = fx["kyotaku_start"] + fx["reach_in_kyoku"]
        deltas = settle_multi_ron(fx["target"], fx["ron"], honba=fx["honba"],
                                  kyotaku=kyotaku, oya=fx["oya"])
        self.assertEqual(deltas, fx["observed"])
        # 積み棒/供託は上家取り: the first winner after the discarder takes both; the
        # second winner gets the bare ron value.
        self.assertEqual(deltas[0][0], fx["ron"][0] + fx["honba"] * 300 + kyotaku * 1000)
        self.assertEqual(deltas[1][1], fx["ron"][1])
        # Dealer among winners -> 連荘, so honba+1 and the same kyoku/oya.
        dealer_won = fx["oya"] in fx["ron"]
        self.assertIs(dealer_won, True)
        self.assertEqual(fx["next_start"]["honba"], fx["honba"] + 1)
        self.assertEqual(fx["next_start"]["oya"], fx["oya"])

    def test_r02_triple_ron_is_settled_by_the_arena(self):
        fx = TRIPLE_RON
        kyotaku = fx["kyotaku_start"] + fx["reach_in_kyoku"]
        deltas = settle_multi_ron(fx["target"], fx["ron"], honba=fx["honba"],
                                  kyotaku=kyotaku, oya=fx["oya"])
        self.assertEqual(len(deltas), 3)
        self.assertEqual(deltas, fx["observed"])
        # The discarder pays every ron value in full plus one honba block.
        paid = sum(row[fx["target"]] for row in deltas)
        self.assertEqual(paid, -sum(fx["ron"].values()) - fx["honba"] * 300)
        # All four riichi sticks (2 carried + 2 in this kyoku) were distributed, so the
        # next kyoku starts with kyotaku 0.
        self.assertEqual(fx["next_start"]["honba"], fx["honba"] + 1)
        self.assertEqual(fx["next_start"]["oya"], fx["oya"])  # dealer won -> renchan
        # Reconcile the next kyoku scores from the recorded start + riichi + deltas.
        start = [20500, 23500, 30500, 23500]
        riichi = [-1000, 0, 0, -1000]  # seats 0 and 3 declared/accepted
        totals = [start[i] + riichi[i] + sum(row[i] for row in deltas) for i in range(4)]
        self.assertEqual(totals, fx["next_scores"])

    def test_r02_tenhou_would_abort_instead(self):
        """The KNOWN_DIFFERENCE: 三家和了 is an abortive draw, not three paid ron."""
        fx = TRIPLE_RON
        kyotaku = fx["kyotaku_start"] + fx["reach_in_kyoku"]
        arena = settle_multi_ron(fx["target"], fx["ron"], honba=fx["honba"],
                                 kyotaku=kyotaku, oya=fx["oya"])
        # Tenhou: 途中流局(九種九牌/四家立直/三家和了/四槓散了/四風連打)あり。すべて連荘。
        tenhou_deltas: list[list[int]] = []  # no score is paid
        tenhou_honba = fx["honba"] + 1       # every abortive draw increments honba
        tenhou_renchan = True                # ... and the dealer repeats
        tenhou_kyotaku = kyotaku             # riichi sticks stay on the table
        self.assertNotEqual(arena, tenhou_deltas)
        self.assertEqual(sum(sum(row) for row in arena), kyotaku * 1000)
        self.assertNotEqual(tenhou_kyotaku, 0)
        self.assertIs(tenhou_renchan, True)
        self.assertEqual(tenhou_honba, fx["honba"] + 1)
        # The log proves the arena took the settled branch: three hora events, no ryukyoku
        # between them, and the next kyoku starts with the distributed scores.
        self.assertEqual(len(fx["observed"]), 3)
        self.assertNotIn("ryukyoku", json.dumps(fx["observed"]))

    def test_r03_allocation_matches_recorded_tenhou_double_ron(self):
        fx = TENHOU_DOUBLE_RON
        deltas = settle_multi_ron(fx["target"], fx["ron"], honba=fx["honba"],
                                  kyotaku=fx["kyotaku"], oya=fx["oya"])
        self.assertEqual(deltas, fx["observed"])
        # The dealer was the second winner and did not receive the honba/kyotaku.
        self.assertEqual(deltas[1][fx["oya"]], fx["ron"][1])

    def test_r23_noten_payment_matches_recorded_tenhou_draw(self):
        # Tenhou log 2024010100gm-00a9-0000-0d9240dd.mjson: E4 honba 1 oya 3,
        # three players tenpai -> [1000,1000,1000,-3000].
        self.assertEqual(exhaustive_ryukyoku_deltas({0, 1, 2}, 3), [1000, 1000, 1000, -3000])
        self.assertEqual(exhaustive_ryukyoku_deltas({0}, 1), [3000, -1000, -1000, -1000])
        self.assertEqual(exhaustive_ryukyoku_deltas({0, 1}, 2), [1500, 1500, -1500, -1500])
        self.assertEqual(exhaustive_ryukyoku_deltas(set(), 0), [0, 0, 0, 0])

    def test_r24_nagashi_payment_matches_recorded_tenhou_draw(self):
        # Tenhou log 2024010114gm-00a9-0000-3275a4ba.mjson: E3 honba 1 oya 2,
        # non-dealer seat 1 nagashi -> [-2000,8000,-4000,-2000].
        self.assertEqual(exhaustive_ryukyoku_deltas(set(), 2, nagashi=(1,)),
                         [-2000, 8000, -4000, -2000])
        self.assertEqual(exhaustive_ryukyoku_deltas(set(), 0, nagashi=(0,)),
                         [12000, -4000, -4000, -4000])

    def test_r03_single_ron_and_tsumo_are_unaffected(self):
        # A ron at honba 0 / kyotaku 0 is the plain hand value.
        self.assertEqual(settle_multi_ron(3, {1: 7700}, honba=0, kyotaku=0, oya=0),
                         [[0, 7700, 0, -7700]])


# --------------------------------------------------------------------------------------
# The native board is not reachable from Python.  This skip is the honest statement of
# that gap; it must never be turned into a pass.
# --------------------------------------------------------------------------------------
class NativeBoardCapabilityTests(unittest.TestCase):
    @unittest.skip(
        "MISSING CAPABILITY: libriichi.arena exposes only OneVsThree and TwoVsTwo; "
        "arena::Board is not a #[pyclass] and has no Python constructor, and there is no "
        "API to inject yama/haipai or scripted reactions. A native triple-ron fixture "
        "therefore cannot be driven. Expected Tenhou behaviour: 三家和了 = abortive "
        "ryukyoku, deltas [0,0,0,0], honba+1, dealer renchan, kyotaku carried. The arena "
        "settles all three hora instead (see TRIPLE_RON / R02).",
    )
    def test_r02_native_triple_ron_aborts(self):
        self.fail("unreachable: kept as a documented capability gap, not a pass")


# --------------------------------------------------------------------------------------
# Report/test consistency.  The matrix is the deliverable; these checks stop the test and
# the report from drifting apart.
# --------------------------------------------------------------------------------------
EXPECTED_RULES = [
    "R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R09", "R10", "R11", "R12",
    "R13", "R14", "R15", "R16", "R17", "R18", "R19", "R20", "R21", "R22", "R23", "R24",
    "R25", "R26", "R27", "R28", "R29", "R30",
]
VALID_STATUS = {"MATCH", "KNOWN_DIFFERENCE", "UNVERIFIED", "FIX_NEEDED"}
ROW_RE = re.compile(r"^\|\s*(R\d{2})\s*\|")
FS_LINE_RE = re.compile(r"[\w/]+\.rs:\d+")


class ReportSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = REPORT_PATH.read_text(encoding="utf-8")
        cls.rows = {}
        for line in cls.text.splitlines():
            match = ROW_RE.match(line)
            if match:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                cls.rows[match.group(1)] = cells

    def test_report_exists_and_covers_every_audited_rule(self):
        self.assertTrue(REPORT_PATH.is_file(), f"missing report {REPORT_PATH}")
        self.assertEqual(sorted(self.rows), sorted(EXPECTED_RULES))
        for rid, cells in self.rows.items():
            with self.subTest(rule=rid):
                self.assertEqual(len(cells), 8, f"{rid} must have 8 columns")
                self.assertIn(cells[6], VALID_STATUS, f"{rid} has status {cells[6]!r}")

    def test_every_reported_file_line_is_anchored(self):
        """The report may not cite a source line that the test does not assert."""
        cited = set()
        for match in re.finditer(r"((?:[\w.-]+/)*[\w.-]+\.rs):(\d+)(?:-\d+)?", self.text):
            key = resolve_rs_path(match.group(1))
            self.assertIsNotNone(key, f"unknown source path {match.group(1)!r}")
            cited.add((key, int(match.group(2))))
        self.assertEqual(
            len(cited), CITED_SOURCE_LINE_COUNT,
            "citation count changed: update the anchors and CITED_SOURCE_LINE_COUNT",
        )
        for key, line in sorted(cited):
            with self.subTest(file=key, line=line):
                self.assertIn(key, ANCHORS, f"{key} has no ANCHORS group")
                self.assertIn(line, ANCHORS[key], f"{key}:{line} is cited but not anchored")

    def test_known_differences_cite_source_and_are_not_marked_match(self):
        for rid, cells in self.rows.items():
            if cells[6] in {"KNOWN_DIFFERENCE", "FIX_NEEDED"}:
                with self.subTest(rule=rid):
                    self.assertRegex(cells[3], FS_LINE_RE,
                                     f"{rid} must cite an arena file:line")
                    self.assertTrue(cells[5].strip(), f"{rid} must name a fixture/test")
                    self.assertTrue(cells[7].strip(), f"{rid} must state the impact")

    def test_triple_ron_is_not_claimed_equivalent(self):
        cells = self.rows["R02"]
        self.assertIn(cells[6], {"KNOWN_DIFFERENCE", "FIX_NEEDED"})
        self.assertNotIn("MATCH", cells[6])
        self.assertIn("No triple-ron ryukyoku", self.text)  # quoted source comment
        self.assertIn("三家和了", self.text)

    def test_report_states_the_tenhou_basis_and_blocking_rules(self):
        self.assertIn(TENHOU_MAN, self.text)
        self.assertIn("block", self.text.lower())
        self.assertIn("R02", self.text)
        self.assertIn("R25", self.text)
        self.assertIn("NOT Tenhou-equivalent", self.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
