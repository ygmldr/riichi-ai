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
        383: "if n_hora >= 3 {",
        384: "// 三家和了: Tenhou treats three simultaneous ron as an abortive draw.",
        396: "self.kyotaku_paid = self.board.kyotaku;",
        387: "self.abortive_ryukyoku();",
        391: "self.has_hora = true;",
        439: "if kyotaku_point > 0 && self.kyotaku_winner.is_none() {",
        440: "self.kyotaku_winner = Some(actor as u8);",
        530: "fn abortive_ryukyoku(&mut self) {",
        596: "let tile = if self.deal_from_rinshan.take().is_some() {",
        588: "if self.accepted_riichis == 4 {",
        589: "// 四家立直: abort as soon as the fourth riichi is accepted,",
        249: "self.can_renchan = self.player_states[self.oya as usize].shanten() == 0;",
        353: "self.board.scores[actor as usize] -= 1000;",
        354: "self.board.kyotaku += 1;",
        394: "let mut honba_left = self.board.honba as i32; // mut in case of multi-ron",
        395: "let mut kyotaku_point = self.board.kyotaku as i32 * 1000; // ditto",
        397: "self.board.kyotaku = 0; // Unlike honba, kyotaku in self will be cleared",
        421: ".cycle()",
        422: ".skip(single_target as usize + 1)",
        423: ".take(3)",
        437: "deltas[actor] = point.ron + kyotaku_point + honba_left * 300;",
        442: "kyotaku_point = 0;",
        443: "honba_left = 0;",
        532: "deltas: Some([0; 4]),",
        545: "if self.accepted_riichis == 4 {",
        573: "if self.check_four_kan && !matches!(ev.event, Event::Hora { .. }) {",
        614: "if self.need_new_dora_at_tsumo.take().is_some() {",
        623: "if self.need_new_dora_at_discard.take().is_some() {",
        632: "if self.can_four_wind && self.check_four_wind(pai)? {",
        637: "if self.kans == 4 && self.player_states.iter().all(|s| s.kans_count() < 4) {",
        658: "// Immediately add new dora",
        659: "self.add_new_dora()?;",
        669: "self.need_new_dora_at_tsumo = Some(());",
        678: "self.need_new_dora_at_discard = Some(());",
        696: "Event::Ryukyoku { .. } => {",
        826: "t!(5mr), t!(5m), t!(5m), t!(5m),",
        836: "t!(5pr), t!(5p), t!(5p), t!(5p),",
        846: "t!(5sr), t!(5s), t!(5s), t!(5s),",
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
        155: "if last_dealer_stop(self.length, &kyoku_result) {",
        216: "fn last_dealer_stop(length: u8, kyoku_result: &KyokuResult) -> bool {",
        221: "let scores = kyoku_result.scores_excluding_hora_kyotaku();",
        222: "if scores[oya] < 30000 {",
        230: "== Some(oya)",
        161: "self.in_renchan = true;",
        162: "self.honba += 1;",
        172: "if self.kyotaku > 0 {",
        173: "*self.scores.iter_mut().min_by_key(|s| -**s).unwrap() += self.kyotaku as i32 * 1000;",
        236: "length: 8,",
        237: "init_scores: [25000; 4],",
    },
    "arena/result.rs": {
        16: "pub kyotaku_paid: u8,",
        18: "pub kyotaku_winner: Option<u8>,",
        29: "pub fn scores_excluding_hora_kyotaku(&self) -> [i32; 4] {",
        48: "pub fn rankings(&self) -> Rankings {",
        49: "Rankings::new(self.scores)",
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
    130: 'let oya = self.kyoku % 4;',
    157: 'if self.has_abortive_ryukyoku {',
    247: 'fn exhaustive_ryukyoku(&mut self) {',
    251: 'let mut has_nagashi_mangan = false;',
    271: 'if !has_nagashi_mangan {',
    320: 'fn check_four_wind(&mut self, pai: Tile) -> Result<bool> {',
    348: 'fn check_riichi_accepted(&mut self) {',
    407: 'self.can_renchan |= actor == self.oya;',
    419: '.into_iter()',
    427: 'if let Some(pao_target) = self.paos[actor] {',
    435: 'deltas[single_target as usize] = -point.ron - honba_left * 300;',
    501: 'fn update_paos(&mut self, ev: &Event) {',
    582: 'Event::None => {',
    587: 'self.check_riichi_accepted();',
    602: 'self.board.yama.pop().with_context(|| {',
    691: 'Event::Hora { actor, target, .. } => {',
})

ANCHORS["arena/game.rs"].update({
    47: '/// Used in 西入 where the oya and another player get to 30000 at the same',
})

ANCHORS["arena/result.rs"].update({
    14: 'pub kyotaku_left: u8,',
})

ANCHORS["state/update.rs"].update({
    59: 'self.at_ippatsu = false;',
    235: 'if self.can_w_riichi {',
    244: 'if self.waits[pai.deaka().as_usize()] {',
    274: 'if self.riichi_accepted[0] {',
    303: 'self.last_cans.can_riichi = self.is_menzen',
    345: 'self.at_ippatsu = false;',
    446: 'self.at_ippatsu = false;',
    415: 'if self.riichi_accepted[0] || self.tiles_left == 0 {',
    606: 'self.at_ippatsu = false;',
    683: 'if actor_rel == 0 {',
    684: 'self.at_ippatsu = true;',
    710: 'match tile.as_u8() {',
    930: 'for (t, is_wait) in self.waits.iter_mut().enumerate() {',
})


CITED_SOURCE_LINE_COUNT = 71  # distinct (file, line) pairs cited by the report


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

    def test_r02_triple_ron_aborts(self):
        """R02: 三家和了 is an abortive draw; the guard runs before any settlement."""
        board = "\n".join(read_source("arena/board.rs"))
        self.assertIn("三家和了", board)
        guard = board.index("if n_hora >= 3 {")
        # The guard precedes `has_hora`, the kyotaku/ura snapshot and every payout.
        self.assertLess(guard, board.index("self.has_hora = true;"))
        self.assertLess(guard, board.index("self.kyotaku_paid = self.board.kyotaku;"))
        self.assertLess(guard, board.index("let ura_indicators ="))
        # The old upstream deviation comment is gone.
        self.assertNotIn("No triple-ron ryukyoku", board)
        # Existing four abort sites + the triple-ron guard + the R26 immediate guard.
        self.assertEqual(board.count("self.abortive_ryukyoku();"), 6)

    def test_r03_multi_ron_uses_kamicha_priority_and_resets_counters(self):
        """R03: 積み棒/供託は上家取り — first winner after the discarder takes both."""
        board = read_source("arena/board.rs")
        # 375-377: honba/kyotaku are copied, then the board's kyotaku is zeroed.
        self.assertIn("// mut in case of multi-ron", board[393])
        # 419-420: after the first paid hora both counters are cleared, so later winners
        # in the same discard receive neither honba nor kyotaku.
        self.assertIn("kyotaku_point = 0;", board[441])
        self.assertIn("honba_left = 0;", board[442])
        # The iteration starts at the discarder's next seat: 上家取り in turn order.
        self.assertIn("skip(single_target as usize + 1)", board[421])

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
        self.assertIn("self.can_renchan = self.player_states[self.oya as usize].shanten() == 0;", board[248])
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
        self.assertIn("let oya = self.kyoku % 4;", read_source("arena/board.rs")[129])

    def test_r15_leftover_kyotaku_goes_to_the_first_max(self):
        """R15: 終了時の供託はトップ取り."""
        game = read_source("arena/game.rs")
        self.assertIn("if self.kyotaku > 0 {", game[171])
        self.assertIn("min_by_key(|s| -**s)", game[172])
        # Rust `min_by_key` returns the first minimum; with key = -score that is the first
        # (lowest-seat) maximum.  Mirror it to prove the tie direction.
        scores = [25000, 31000, 31000, 19000]
        idx = min(range(4), key=lambda i: -scores[i])
        self.assertEqual(idx, 1)

    def test_r16_r17_r18_r19_all_four_abortive_draws_exist(self):
        """R16-R19: 四家立直/四槓散了/九種九牌/四風連打 are implemented (R02 separately)."""
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
        self.assertIn("1 => (3000, -1000),", board[280])
        self.assertIn("2 => (1500, -1500),", board[281])
        self.assertIn("3 => (1000, -3000),", board[282])

    def test_r06_last_dealer_agari_and_tenpai_stop(self):
        """R06: ラス親トップの自動和了り止め、自動聴牌止めあり."""
        game = read_source("arena/game.rs")
        text = "\n".join(game)
        self.assertIn("fn last_dealer_stop", text)
        self.assertIn("let scores = kyoku_result.scores_excluding_hora_kyotaku();", text)
        self.assertIn("if scores[oya] < 30000 {", text)
        self.assertIn("== Some(oya)", text)
        self.assertIn("if last_dealer_stop(self.length, &kyoku_result) {", text)
        self.assertIn("self.ended = true;", text)
        # 原点 = 30000 (25000 start / 30000 return).  The renchan-owari branch is only
        # reachable when the dealer can repeat (agari or tenpai); R25 covers the
        # kyotaku exclusion inside `last_dealer_stop`.
        self.assertIn("s >= 30000", text)
        self.assertIn("init_scores: [25000; 4]", text)

    def test_r24_nagashi_mangan_payments(self):
        """R24: 流し満貫 = 満貫清算, 親12000 / 子8000(4000+2000+2000)."""
        board = read_source("arena/board.rs")
        self.assertIn("let mut dod = [-4000; 4];", board[259])
        self.assertIn("dod[i] = 12000;", board[260])
        self.assertIn("let mut dod = [-2000; 4];", board[263])
        self.assertIn("dod[i] = 8000;", board[264])
        self.assertIn("dod[self.oya as usize] = -4000;", board[265])

    def test_r27_pao_exists_for_daisangen_and_daisuushi_only(self):
        """R27: 大三元/大四喜のパオあり、四槓子のパオなし."""
        board = "\n".join(read_source("arena/board.rs"))
        self.assertIn("fn update_paos", board)
        self.assertIn("daisanein_confirmed", board)
        self.assertIn("daisuushi_confirmed", board)
        self.assertNotIn("sukantsu", board.lower())
        self.assertNotIn("四槓子", board)

    def test_r25_s4_stop_excludes_paid_kyotaku(self):
        """R25: 供託料を除く — compare with the just-paid kyotaku removed."""
        game = "\n".join(read_source("arena/game.rs"))
        board = "\n".join(read_source("arena/board.rs"))
        result = "\n".join(read_source("arena/result.rs"))
        # board.rs snapshots the sticks before clearing them and pays them to a winner.
        self.assertIn("self.kyotaku_paid = self.board.kyotaku;", board)
        self.assertIn("kyotaku_paid: self.kyotaku_paid,", board)
        self.assertIn("self.kyotaku_winner = Some(actor as u8);", board)
        # KyokuResult carries the paid amount and the winner.
        self.assertIn("pub kyotaku_paid: u8,", result)
        self.assertIn("pub kyotaku_winner: Option<u8>,", result)
        self.assertIn("pub fn scores_excluding_hora_kyotaku(&self) -> [i32; 4]", result)
        self.assertIn("scores[winner as usize] -= self.kyotaku_paid as i32 * 1000;", result)
        # The last-dealer decision uses the adjusted vector, not self.scores.
        self.assertIn("let scores = kyoku_result.scores_excluding_hora_kyotaku();", game)
        self.assertNotIn("&& self.scores[oya] >= 30000", game)

    def test_r26_four_riichi_aborts_before_the_next_draw(self):
        """R26: abort when the fourth riichi is accepted, before the next draw."""
        board = read_source("arena/board.rs")
        text = "\n".join(board)
        # The top-of-step check remains as a safety net for acceptance paths that a
        # four-riichi board cannot normally reach (calls are impossible with 4 riichis).
        abort_at = text.index("if self.accepted_riichis == 4 {")
        self.assertLess(abort_at, text.index("// Validate reactions"))
        # The None branch accepts the fourth riichi, then must abort before drawing.
        none_branch = text[text.index("Event::None => {"):text.index("Event::Dahai { actor, pai, .. } => {")]
        accept = none_branch.index("self.check_riichi_accepted();")
        guard = none_branch.index("if self.accepted_riichis == 4 {", accept)
        draw = none_branch.index("let tile = if self.deal_from_rinshan.take()", accept)
        self.assertIn("// 四家立直: abort as soon as the fourth riichi is accepted,", none_branch)
        self.assertLess(accept, guard)
        self.assertLess(guard, draw)
        self.assertLess(guard, none_branch.index("self.broadcast(&tsumo);"))

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

    def test_r02_recorded_log_is_the_pre_fix_settlement(self):
        """The recorded arena log is the *old* behaviour, kept as historical evidence."""
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

    def test_r02_fixed_arena_aborts_and_carries_kyotaku(self):
        """Settlement arithmetic for the abort; continuation is source-read, not run here.

        The executable continuation evidence is the Rust fixture in `board.rs`
        (it calls `handle_hora` and `end()`), not this Python arithmetic.  This test
        only pins the no-payout/score shape so a future edit cannot silently reuse the
        old triple-ron payment table.
        """
        fx = TRIPLE_RON
        kyotaku = fx["kyotaku_start"] + fx["reach_in_kyoku"]
        start = [20500, 23500, 30500, 23500]
        riichi = [-1000, 0, 0, -1000]  # seats 0 and 3 declared/accepted
        after_sticks = [start[i] + riichi[i] for i in range(4)]
        # 三家和了 is an abortive draw: no hand is paid, so the scores are exactly
        # start + the two riichi deductions; they must differ from the pre-fix
        # settlement (which paid three ron and distributed all four sticks).
        self.assertEqual(after_sticks, [19500, 23500, 30500, 22500])
        self.assertNotEqual(after_sticks, fx["next_scores"])
        # The 4 sticks (2 carried + 2 accepted here) remain on the table.
        self.assertEqual(kyotaku, 4)
        self.assertEqual(sum(after_sticks), 96000)
        # Scores are distinct, so descending rank is unambiguous here.  The report
        # labels the continuation (honba/dealer/kyotaku carry) as source-read.
        ranks = [1 + sum(1 for other in after_sticks if other > score)
                 for score in after_sticks]
        self.assertEqual(ranks, [4, 2, 1, 3])

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
        "API to inject yama/haipai or scripted reactions, so this case cannot be driven "
        "from Python. The executable coverage is the Rust unit test "
        "arena::board::rule_tests::r02_triple_ron_aborts_without_settlement (and the R25/"
        "R26 rule_tests in game.rs/board.rs); this skip is never counted as a pass.",
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

    def test_r02_is_matched_only_with_native_evidence_and_r29_stays_unverified(self):
        # The matrix supplies the verdict; this test checks the evidence that supports
        # it instead of hard-coding a green label.  The native fixture executes the
        # settlement branch; the continuation is explicitly source-read.
        self.assertIn(self.rows["R02"][6], VALID_STATUS)
        self.assertIn("r02_triple_ron_aborts_without_settlement", self.text)
        self.assertIn("settlement", self.text.lower())
        self.assertIn("source-read", self.text.lower())
        self.assertIn("tenhou-parity-verified", self.text)
        self.assertIn("三家和了", self.text)
        self.assertIn("No triple-ron ryukyoku", self.text)  # historical quote
        # R29 must not be promoted without an executable timing fixture.
        self.assertEqual(self.rows["R29"][6], "UNVERIFIED")
        self.assertIn("R29", self.text)
        # Do not use the forbidden "official complete equivalence" wording.
        self.assertNotIn("官方完全一致", self.text)

    def test_report_states_the_tenhou_basis_and_blocking_rules(self):
        self.assertIn(TENHOU_MAN, self.text)
        self.assertIn("block", self.text.lower())
        self.assertIn("R02", self.text)
        self.assertIn("R25", self.text)
        self.assertIn("NOT Tenhou-equivalent", self.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
