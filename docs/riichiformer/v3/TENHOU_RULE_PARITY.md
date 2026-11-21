# TENHOU_RULE_PARITY — H-04 ruleset-difference audit

> **V3.3 update (PR-40R, 2026-11-21):** R29 now has an executable native fixture and is
> promoted to `MATCH` in `docs/riichiformer/v3_3/PR40R_NATIVE_RULES.md`. The R29 rows and
> `UNVERIFIED` statements in this V3.1/V3.2 document are preserved as the historical state;
> they are not the current verdict. R02/R25/R26 were already promoted by PR-40 and remain
> `tenhou-parity-verified`.

- **Task**: H-04 (RULES), RiichiFormer V3.1 parallel program
- **Owner**: Dev-D (rules lane)
- **Worktree / branch**: `/home/ygsldr/riichi-ai/worktrees/h04-rules` @ `rf/h04-rules`
- **Approved base**: `59691f57bc7158dcdb2ba98eecd9d102857171f8` (tag `rf3-base`)
- **Pinned arena**: `vendor/mortal` submodule `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2` (libriichi)
- **Executable companion**: `tests/rf_test_ruleset_parity.py` (V3.1: 51 passed, 1 skipped, 197 subtests; see `docs/riichiformer/v3/logs/H04_pytest.log`)
- **V3.1 scope limit**: no change to `vendor/mortal`, `riichiformer/data/replay.py`, schema/POV or any runtime file. The V3.1 sections below are proposals only; **PR-40 (§0) applies the R02/R25/R26 fixes as `patches/0014-rule-r02-r25-r26.patch`**.

## 0. PR-40 update — R02/R25/R26 fixed, R29 still unverified

- **Task**: PR-40 (Dev-RULES), RiichiFormer V3.2 Wave 1
- **Worktree / branch**: `worktrees/v32-rules` @ `rf/v32-rules`
- **BASE_SHA**: `cbf166a356f83c739b31f0b5a0b241b954ef9e0a`
- **Pinned arena**: `vendor/mortal` submodule `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2`
- **Patch**: `patches/0014-rule-r02-r25-r26.patch` (apply to the parent vendor tree
  after `0001-0013`; the patch does not move the submodule pointer)
- **Native fixture target**: `vendor/mortal/libriichi/src/arena/{board,game,result}.rs`
  `#[cfg(test)] mod rule_tests`
- **Python regression**: `tests/rf_test_rule_fixes.py` (new);
  `tests/rf_test_ruleset_parity.py` (updated anchors and R02/R25/R26 assertions)

This section is the current status. §1–§10 retain the V3.1 (H-04) audit as history;
where they say "FIX_NEEDED", "KNOWN_DIFFERENCE" or "not applied" for R02/R25/R26, that is
the *pre-PR-40* state. The historical text is deliberately not deleted.

### 0.1 Tenhou clauses (official manual, 段位戦4人打ち)

Source: <https://tenhou.net/man/#RULE> (fetched 2026-11-21). Quoted verbatim:

- **R02 / R26 (abortive draws)**
  「途中流局(九種九牌/四家立直/三家和了/四槓散了/四風連打)あり。すべて連荘」
  This makes 三家和了 and 四家立直 abortive draws, and every abortive draw is a
  dealer repeat (`連荘`), i.e. honba+1 with the same dealer and the kyotaku left on the
  table.
- **R25 (last-dealer stop)**
  「親が連荘(和了/聴牌)し原点を超えたトップになる(和了/聴牌清算後、供託料を除く)」
  The comparison is made *after* settlement but **excluding the kyotaku**.
- **R29 (ippatsu cancellation)**
  「純正巡消しは発声＆和了打診後（加槓のみ)、嶺上ツモの前（連続する加槓の２回目には一発は付かない）」
  and the yaku notes: 「一発と槍槓（連続槓の１回目）は複合する」.

The manual lists 四家立直 as an abortive draw but does **not** separately spell out the
draw instant for R26. The PR-40 fixture pins the only event order consistent with the
listed abort: the fourth `reach_accepted` is the abort event, so the next draw must not
happen. This is called out again in §0.4 rather than hidden.

### 0.2 Anchors in the patched source

All line numbers below refer to the patched worktree vendor tree and are asserted by
`SourceContractTests.test_anchor_lines_are_exact`.

| Rule | file:line | Quoted source text / meaning |
|---|---|---|
| R02 | `vendor/mortal/libriichi/src/arena/board.rs:383` | `if n_hora >= 3 {` — guard at the top of `handle_hora` |
| R02 | `vendor/mortal/libriichi/src/arena/board.rs:384` | `// 三家和了: Tenhou treats three simultaneous ron as an abortive draw.` |
| R02 | `vendor/mortal/libriichi/src/arena/board.rs:387` | `self.abortive_ryukyoku();` — `Ryukyoku { deltas: Some([0; 4]) }`, no payout |
| R02 | `vendor/mortal/libriichi/src/arena/board.rs:391` | `self.has_hora = true;` — deliberately **after** the guard, so game.rs takes the abortive branch |
| R02 | `vendor/mortal/libriichi/src/arena/game.rs:135-137` | abortive continuation: `honba += 1`, `poll()` again; `can_renchan` is set by `board.rs` |
| R25 | `vendor/mortal/libriichi/src/arena/board.rs:396` | `self.kyotaku_paid = self.board.kyotaku;` before the table is cleared |
| R25 | `vendor/mortal/libriichi/src/arena/board.rs:439-441` | records the first winner paid the kyotaku (上家取り order) |
| R25 | `vendor/mortal/libriichi/src/arena/result.rs:16` | `pub kyotaku_paid: u8,` |
| R25 | `vendor/mortal/libriichi/src/arena/result.rs:18` | `pub kyotaku_winner: Option<u8>,` |
| R25 | `vendor/mortal/libriichi/src/arena/result.rs:29-32` | `scores[winner as usize] -= self.kyotaku_paid as i32 * 1000;` |
| R25 | `vendor/mortal/libriichi/src/arena/game.rs:216-230` | `last_dealer_stop` uses `scores_excluding_hora_kyotaku()`; the raw `self.scores[oya]` comparison is gone |
| R26 | `vendor/mortal/libriichi/src/arena/board.rs:587-594` | `check_riichi_accepted()` then an immediate `accepted_riichis == 4` abort, **before** the draw |
| R26 | `vendor/mortal/libriichi/src/arena/board.rs:596` | `let tile = if self.deal_from_rinshan.take().is_some() {` — the draw the abort now precedes |
| R26 | `vendor/mortal/libriichi/src/arena/board.rs:545-549` | the old top-of-`step` guard is retained as a safety net |
| R29 | `vendor/mortal/libriichi/src/state/update.rs:684` | `self.at_ippatsu = true;` on own `reach_accepted` (unchanged) |
| R29 | `vendor/mortal/libriichi/src/state/update.rs:345` | own discard closes the window |
| R29 | `vendor/mortal/libriichi/src/state/update.rs:446` | a call cancels it |
| R29 | `vendor/mortal/libriichi/src/state/update.rs:606` | kakan/other-actor branch: only a live chankan keeps it; otherwise clears |
| R29 | `vendor/mortal/libriichi/src/state/update.rs:59` | a pending `chankan_chance` is cleared at the next event |

### 0.3 Executable fixtures and results (before/after)

The Rust fixtures are real executions of the patched arena, not transcriptions:

```text
CARGO_HOME=<worktree>/.cargo-home PYO3_PYTHON=<venv>/bin/python \
  cargo test -p libriichi --lib --no-default-features rule_tests
# running 3 tests
# r02_triple_ron_aborts_without_settlement ... ok
# r25_last_dealer_stop_excludes_just_paid_kyotaku ... ok
# r26_fourth_riichi_aborts_before_the_next_draw ... ok
# test result: ok. 3 passed; 0 failed
```

The Python regression is `tests/rf_test_rule_fixes.py`; the existing parity suite is
`tests/rf_test_ruleset_parity.py`. Exact commands and exit codes are recorded in
`docs/riichiformer/v3_2/PR40_RULES.md`.

**R02 — recorded pre-fix triple ron** (`runs/1v3/10089_579411421376059130_b.json.gz`,
E3 honba 2, oya 2, discarder 3; start scores `[20500,23500,30500,23500]`, two
`reach_accepted` in the kyoku, so 4 kyotaku on the table):

| Path | seat 0 | seat 1 | seat 2 (oya) | seat 3 | honba | kyotaku | ranks |
|---|---:|---:|---:|---:|---:|---:|---|
| Pre-fix recorded settlement (three hora paid) | 32100 | 25500 | 42100 | 300 | 3 | 0 | `[2,3,1,4]` |
| PR-40 Tenhou abort (native settlement fixture; continuation source-read) | 19500 | 23500 | 30500 | 22500 | 3 | 4 | `[4,2,1,3]` |

The abort row is start minus the two riichi sticks (`-1000` seats 0 and 3), no hora
deltas, honba 2→3, same oya, 4 sticks carried. The recorded row is preserved as
historical evidence of the old deviation, not as current behaviour.

**R25 — S4 dealer agari stop** (kyoku 7 ⇒ oya is absolute seat 3; 4 kyotaku paid to the
dealer). Two score vectors are shown so the before/after decision is unambiguous:

| Case | raw scores (after settlement) | kyotaku-free comparison | old decision | fixed decision |
|---|---|---|---|---|
| kyotaku decisive | `[19000,19000,29000,33000]` | `[19000,19000,29000,29000]` | ends: oya 33000 ≥ 30000 and top | continues: oya 29000 < 30000 |
| genuine 30000+ | `[19000,19000,28000,34000]` | `[19000,19000,28000,30000]` | ends | ends (30000, oya top) |

Final scores if the first case ends under the old rule: `[19000,19000,29000,33000]`,
ranks `[3,4,2,1]`. Under the fixed rule the hand continues; a *constructed* continuation
(seat 0 rons 12000 from seat 3, clearly not an observed game) gives
`[31000,19000,29000,21000]`, ranks `[1,4,2,3]`, which demonstrates why the decision can
change the final rank. The fixture asserts the decision, not the constructed continuation.

**R26 — fourth riichi event order**:

| | log around the fourth acceptance | next tile drawn? | tiles_left |
|---|---|---|---|
| Pre-fix | `reach_accepted` → `tsumo` → (next step) `ryukyoku` | yes | 9 |
| PR-40 | `reach_accepted` → `ryukyoku` | no | 10 |

The native test sets `accepted_riichis = 3`, `riichi_to_be_accepted = Some(0)`, a 10-tile
wall, drives one `Event::None` poll, and asserts: `Poll::End`, `accepted_riichis == 4`,
`tiles_left == 10`, `yama.len() == 10`, no `Tsumo` after `ReachAccepted`, and an
`EndKyoku` tail. No score changes in either version.

**R29 — still UNVERIFIED.** The implementation sites above are present, but there is no
executable fixture for the consecutive-kakan / chankan nuance yet (the arena board is not
injectable from Python and a `PlayerState`-only fixture cannot observe `at_ippatsu`; it is
a private field). No patch is claimed for R29.

### 0.4 Parity labels (required reporting vocabulary)

- **tenhou-parity-verified** — an official clause is quoted and an executable native
  fixture exercises the exact branch. Used for **R02** and **R25**. For **R26** it applies
  to the abort-before-draw event order; the manual does not separately spell out the draw
  instant (see §0.1).
- **tenhou-approx** — the arena behaves in the Tenhou direction but the exact nuance has
  no executable certification. **R29** stays here, and any run that reports it must still
  carry the R29 caveat.
- **UNVERIFIED** — no executable evidence either way. **R29** remains UNVERIFIED.

This document does **not** claim "official complete equivalence". R29 is a live caveat;
R22 (external-driver kuikae) remains an INFO gap; and the R26 manual-timing caveat above
is explicit.

## 1. Historical V3.1 verdict (preserved; superseded for R02/R25/R26 by §0)

> **Historical snapshot only.** This section is the V3.1 (H-04) verdict as written before
> PR-40. R02/R25/R26 are now MATCH under the executable fixtures in §0; R29 remains
> UNVERIFIED. The text is kept verbatim so the audit trail is not rewritten.

**NOT Tenhou-equivalent.** The pinned arena implements essentially all of Tenhou's 4-player
ranked (段位戦) rules, including the parts that are easy to get wrong (kan-dora timing,
riichi-then-ankan wait preservation, furiten bookkeeping, noten payments, 上家取り on
double ron, tie-break by E1 wind order). Two rule-level differences have real score or
continuation impact, and one is log-only:

1. **R02 — triple ron is settled, not aborted.** Tenhou's 三家和了 is an abortive draw;
   the arena pays all three ron winners in full (documented upstream as
   `/// 1. No triple-ron ryukyoku.`). This is the one rule that can materially change a
   game's score and its continuation rules. Status **FIX_NEEDED**.
2. **R25 — the last-dealer 和了止め test includes the just-paid kyotaku.** Tenhou says
   "供託料を除く"; the arena compares scores that already contain the kyotaku payout.
   Rare, but it can end the hanchan one hand early or late. Status **FIX_NEEDED**.
3. **R26 — the 4-riichi abort is evaluated after one more draw.** Tenhou aborts when the
   fourth riichi is accepted; the arena draws and broadcasts one extra tile (the next
   player cannot win on it). Log/observation-level only, no score change. Status
   **KNOWN_DIFFERENCE**.
4. **R29 — ippatsu cancellation timing is not proven.** The implementation is wired, but
   the exact Tenhou nuance for consecutive kakan / chankan is not exercised. Status
   **UNVERIFIED**.

Everything else in the matrix below is **MATCH**, with the evidence level shown per row.
This means: an evaluation run on this arena is valid as a *relative-strength* measurement,
but it must not be described as "Tenhou-equivalent", and PR-16/PR-17 reports must carry the
historical R02 caveat (see §9 and §0).

## 2. Sources and method

### 2.1 Tenhou source

All Tenhou statements below are quoted from the official manual, rule section
"ルール / 段位戦4人打ち": <https://tenhou.net/man/#RULE> (fetched 2026-11-21).
The page also defines the abortive draws, the 補足 sections for sudden death and
riichi-then-ankan, and the yaku list used for 海底摸月 / 河底撈魚 / 嶺上開花.

Supplementary Tenhou evidence comes from a read-only sample of 3,000 real Tenhou 2024
game logs (`datasets/extracted/2024/*.mjson`, first 3,000 sorted filenames,
gzip-wrapped mjai). In that sample: 4,349 ordinary noten draws, 560 abortive draws
(all-zero `ryukyoku.deltas`), 7 nagashi mangan. Those logs are cited where they confirm a
settlement shape (B evidence); they are **not** required by the test file.

### 2.2 Pinned arena source

`vendor/mortal/libriichi/src/arena/{board.rs,game.rs,result.rs}`, `rankings.rs`,
`state/update.rs`, `state/agent_helper.rs`, `algo/agari.rs`. This document cites **71
distinct source lines**. Every full `file.rs:line` citation in the report tables is
present in the test's `ANCHORS` table; shorthand citations (for example R29's
`345/446/560/606`) are explicitly not independently anchored and are not claimed as
executable evidence. Full citations are asserted at
that exact line by `SourceContractTests.test_anchor_lines_are_exact`.
`ReportSyncTests.test_every_reported_file_line_is_anchored` parses this file and fails if a
citation is added without an anchor (or if the count changes without updating
`CITED_SOURCE_LINE_COUNT`). This is a fail-closed contract: if the vendor tree moves, the
test fails and the audit must be re-run. Range citations (`file:line-line`) are anchored at
their first line; the anchors table also pins additional single lines used by the tests.

The V3.1 prebuilt `libriichi.so` used for native fixtures came from the patched parent
tree (`patches/0001-0013`). PR-40 builds its own
`vendor/mortal/target/release/libriichi.so` in the `v32-rules` worktree from the clean pin
plus `patches/0014-rule-r02-r25-r26.patch`; the Rust board fixtures execute that source
directly. In both cases the rule engine is the pinned arena source. The patches modify `agent/mortal.rs`, `dataset/gameplay.rs`,
`mjai/mod.rs`, `mjai/tokens.rs` and several Python files under `mortal/`; **no patch
touches `arena/*`, `state/update.rs`, `state/agent_helper.rs`, `algo/agari.rs` or
`rankings.rs`**, so the rule engine executed by `PlayerState`/`Stat` is the pinned source
(the Python-side patches are irrelevant to it).

### 2.3 Evidence levels

| Level | Meaning | Examples |
|---|---|---|
| **A** | Source line verified in this run, or a deterministic native execution of `libriichi` (`PlayerState` mask, `Stat` ranking). | `board.rs:421-423`; furiten fixtures; ankan-after-riichi fixtures |
| **B** | Recorded game output reconciled with a transcription of the code. The log is independent of the model, so `model == observed` is a real check. | real arena triple ron; real Tenhou double ron; real Tenhou noten / nagashi payments |
| **C** | Inference, not executed. Never used to award MATCH. | R29 timing, some Tenhou sub-nuances |

### 2.4 Reproduction

```bash
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q tests/rf_test_ruleset_parity.py
```

The suite is deterministic, offline, and does not read `datasets/` or `runs/` at run time
(the recorded episodes are embedded as constants with their source paths).

## 3. Rule matrix

Columns: **rule**, **Tenhou behaviour (quote + URL)**, **pinned arena behaviour + file:line**,
**evidence**, **fixture/test**, **status**, **competitive impact**.
`[man#RULE]` = <https://tenhou.net/man/#RULE>.

| ID | Rule | Tenhou behaviour ([man#RULE](https://tenhou.net/man/#RULE)) | Pinned arena (file:line) | Ev | Fixture / test | Status | Competitive impact |
|---|---|---|---|---|---|---|---|
| R01 | 開始点・返し点・半荘長 | 「25000点開始の30000点返し」「ウマ10-20」 | `arena/game.rs:236` `length: 8,`; `arena/game.rs:237` `init_scores: [25000; 4],` | A | `SourceContractTests.test_r01_hanchan_start_and_return_points` | MATCH | none; uma/oka are ranking-tool conventions, the arena emits only ordinal rank |
| R02 | 三家和了（triple ron） | 「途中流局(九種九牌/四家立直/三家和了/四槓散了/四風連打)あり。すべて連荘」 — three ron = abortive draw, no score, honba+1, dealer renchan, kyotaku carried | `arena/board.rs:383-389` (`if n_hora >= 3` guard in `handle_hora`, before `has_hora`/kyotaku/ura); abort path `arena/board.rs:530-536`; continuation `arena/game.rs:135-137` | A+native | `arena::board::rule_tests::r02_triple_ron_aborts_without_settlement`; `MultiRonSettlementTests.test_r02_fixed_arena_aborts_and_carries_kyotaku`; historical recorded log `runs/1v3/10089_579411421376059130_b.json.gz` | MATCH | tenhou-parity-verified for the **settlement branch** (the native fixture calls the private `handle_hora` and `end()`); honba/dealer/kyotaku continuation is **source-read** at `arena/game.rs:135-137`, not executed by that fixture; old distortion removed |
| R03 | ダブロン/多栄の支払いと積み棒・供託の上家取り | 「ダブロンあり。積み棒/供託は上家取り。」「ダブロンは親が含まれれば連荘」 | `arena/board.rs:394-397`, `421-423`, `437-443` | A+B | `test_r03_double_ron_matches_recorded_arena_log`; `test_r03_allocation_matches_recorded_tenhou_double_ron` | MATCH | none; allocation verified against a real Tenhou double ron (dealer as second winner got no honba/kyotaku) |
| R04 | 赤ドラ 各1枚 | 「赤ありルールでの枚数：各1枚」 | `arena/board.rs:826` (5mr), `arena/board.rs:836` (5pr), `arena/board.rs:846` (5sr); `state/update.rs:710-724` | A | `NativePlayerStateTests.test_r04_aka_dora_is_one_per_suit` | MATCH | none |
| R05 | カンドラ時点 | 「カンドラは、暗槓は即乗り、明槓/加槓は後めくり（打牌または続く嶺上の直前）」 | `arena/board.rs:658-659` immediate for ankan; `arena/board.rs:623`+`678` for daiminkan/kakan; `arena/board.rs:669`+`614` for a following kan | A | `SourceContractTests.test_r05_kan_dora_timing_matches_tenhou` | MATCH | none; the `dora` event is logged immediately before the discard event, which is the same instant |
| R06 | ラス親トップの和了止め・聴牌止め | 「ラス親トップの自動和了り止め、自動聴牌止めあり」; 補足「親が原点以上のトップなら和了止め/聴牌止め」 | `arena/game.rs:216-230` (`last_dealer_stop`, kyotaku-exclusive); call at `arena/game.rs:155-158` | A+native | `SourceContractTests.test_r06_last_dealer_agari_and_tenpai_stop`; `arena::game::rule_tests::r25_last_dealer_stop_excludes_just_paid_kyotaku` | MATCH | none; R25 exclusion is fixed in the same helper |
| R07 | トビ | 「飛び終了あり。点数がマイナスで飛び終了、マイナス点数も集計、0点は続行」 | `arena/game.rs:129-133` (`s < 0`) | A | `test_r07_tobi_threshold_is_strictly_negative` | MATCH | none |
| R08 | 西入・サドンデス・最大局 | 「東南戦の西入あり…30000点(供託未収)以上になった時点で終了、ただし親の連荘がある場合は連荘を優先する」「最大局は…西場まで」 | `arena/game.rs:69-73` (`length + 4`, `>= 30000`), `arena/game.rs:47-54` (`in_renchan`) | A | `test_r08_west_round_entry_and_maximum` | MATCH | none |
| R09 | 親の連荘 | 「ノーテン親流れ、聴牌連荘」「ダブロンは親が含まれれば連荘」「途中流局…すべて連荘」 | `arena/board.rs:249` (tenpai), `arena/board.rs:407` (`can_renchan` or-assigned when `actor == oya`), `arena/board.rs:157-159` (abortive), `arena/game.rs:135-148` | A+B | `test_r09_r10_renchan_and_honba_accounting`; real Tenhou abortive log E2 honba 1→2 with the same oya | MATCH | none |
| R10 | 本場 | 「積み棒」; 300点/本 (ron +300, tsumo +100 each) | `arena/board.rs:435-443` (first winner only); `arena/game.rs:136`, `145`, `162` | A+B | `test_r09_r10_...`; `test_r03_allocation_matches_recorded_tenhou_double_ron` (honba 4 = 1200) | MATCH | none |
| R11 | 供託（リーチ棒） | 「リーチ宣言は得点が1000点以上…」「リーチ宣言牌で放銃した場合は供託料は発生しない」 | `arena/board.rs:348-357`; `check_riichi_accepted` is absent from the Hora path (`arena/board.rs:691-694`) | A | `test_r11_riichi_stake_is_paid_only_when_accepted` | MATCH | none |
| R12 | 振聴 | 「フリテンは河に捨てられた牌でのみ判定(加槓・抜きでさらした非純手牌を除く)」「同巡振聴の解消は打牌時、逆巡副露では解消しない」「リーチ後の当たり牌見逃しは以降振聴」 | `state/update.rs:923-951`, `363-365`, `394-412` | A | `test_r12_minogashi_furiten_blocks_ron`; `test_r12_same_cycle_furiten_clears_on_own_discard` | MATCH | none |
| R13 | リーチ後の暗槓 | 「リーチ後の暗槓は待ちが変わらない場合のみ。送り槓不可、牌姿や役の増減は不問。」 | `algo/agari.rs:846-854`; `state/update.rs:274-278` calls it with `strict=false` under the comment `// Using Tenhou rule here.` | A | `test_r13_riichi_ankan_allowed_when_wait_is_unchanged`; `test_r13_riichi_ankan_forbidden_when_wait_changes` | MATCH | none |
| R14 | 同点順位 | 「終了時に同点の場合は東1局の風順で順位を決定」 | `rankings.rs:8-10` (stable `sort_by_key`), `arena/result.rs:48-49`, `arena/board.rs:130` (E1 dealer is absolute seat 0) | A | `NativeRankingTests.test_r14_equal_scores_rank_by_e1_wind_order` | MATCH | none |
| R15 | 終局時の供託 | 「終了時の供託はトップ取り」「終了時の得点の1000点未満の四捨五入を廃止」 | `arena/game.rs:172-174` (first maximum); no rounding anywhere in the settlement | A | `test_r15_leftover_kyotaku_goes_to_the_first_max`; `NativeRankingTests.test_r15_leftover_kyotaku_does_not_change_rank` | MATCH | none |
| R16 | 四家立直 | 「四家立直」 (abortive draw, すべて連荘) | `arena/board.rs:545-549` (top-of-step safety net); immediate abort at `arena/board.rs:587-594` (R26) | A+native | `test_r16_r17_r18_r19_all_four_abortive_draws_exist`; `arena::board::rule_tests::r26_fourth_riichi_aborts_before_the_next_draw` | MATCH | none; no extra draw after the fourth acceptance (R26) |
| R17 | 四槓散了 | 「四槓散了」 (abortive; a single player's four kans is 四槓子, not an abort) | `arena/board.rs:573-577`; `arena/board.rs:637-640` (`all(s.kans_count() < 4)`) | A | `test_r16_r17_r18_r19_...`; `test_r17_a_fifth_kan_is_never_offered` | MATCH | none |
| R18 | 九種九牌 | 「九種九牌はポン/カン/チー/暗槓のない第1ツモで幺九牌が9種以上」 | `state/update.rs:235-237`; `arena/board.rs:696-700` | A | `test_r18_nine_terminals_only_on_first_uncalled_draw` | MATCH | none |
| R19 | 四風連打 | 「四風連打」 | `arena/board.rs:320-346`, `632-635` | A | `test_r16_r17_r18_r19_...` | MATCH | none |
| R20 | リーチ条件 | 「リーチ宣言は得点が1000点以上でツモ巡がある場合に可能」 | `state/update.rs:303-306` (`is_menzen`, `tiles_left >= 4`, `scores[0] >= 1000`) | A | `test_r20_riichi_needs_1000_points`; `test_r20_riichi_needs_a_draw_turn` | MATCH | none |
| R21 | 海底・河底 | 「海底牌は鳴けない」; 海底摸月 / 河底撈魚 are yaku | `state/update.rs:244-272` (haitei yaku, no kan, no riichi), `state/update.rs:415-417` (no calls on the last discard) | A | `test_r21_last_discard_cannot_be_ponned`; `test_r21_haitei_is_a_yaku_for_an_open_no_yaku_hand` | MATCH | none |
| R22 | 喰い替え | 「喰い替えなし」 | `state/update.rs:469-487`, `531-534`; `state/agent_helper.rs:61` | A | `test_r22_kuikae_is_forbidden` | MATCH | none through the arena's own mask; `validate_reaction` alone would not reject a kuikae supplied by an external driver (INFO, unreachable from a normal agent) |
| R23 | ノーテン罰符 | 3000 / 1500 / 1000; the man page states 「ノーテン親流れ、聴牌連荘」 | `arena/board.rs:271-292` | A+B | `test_r23_exhaustive_draw_tenpai_payments`; `test_r23_noten_payment_matches_recorded_tenhou_draw` (real Tenhou `[1000,1000,1000,-3000]`) | MATCH | none |
| R24 | 流し満貫 | 「流し満貫あり。聴牌清算を満貫清算に代替。親の聴牌で連荘。」「流し満貫は、自分の河が鳴かれていない場合に成立。自分が鳴いた場合でも成立。」 | `arena/board.rs:251-269` (親 12000 / 子 8000); `arena/board.rs:249` (renchan by dealer tenpai) | A+B | `test_r24_nagashi_mangan_payments`; `test_r24_nagashi_payment_matches_recorded_tenhou_draw` (real Tenhou `[-2000,8000,-4000,-2000]`, next kyoku honba+1) | MATCH | none; the dealer-renchan sub-case follows from 「親の聴牌で連荘」 and is consistent with the observed honba increment |
| R25 | ラス親和了止めの供託除外 | 「親が連荘(和了/聴牌)し原点を超えたトップになる(和了/聴牌清算後、供託料を除く)」 | `arena/board.rs:396` snapshots the paid sticks; `arena/board.rs:439-441` records the winner; `arena/result.rs:16-32` excludes them; `arena/game.rs:216-230` uses the adjusted vector | A+native | `arena::game::rule_tests::r25_last_dealer_stop_excludes_just_paid_kyotaku`; `R25LastDealerKyotakuTests.test_kyotaku_decisive_case_changes_the_end_decision` | MATCH | tenhou-parity-verified; can change the S4 end decision when the kyotaku is decisive |
| R26 | 四家立直の流局時点 | 「四家立直」 — abortive; the fourth acceptance is the abort event, so the next draw must not happen | `arena/board.rs:587-594` aborts immediately after `check_riichi_accepted()`; no tile is drawn before the abort. The top-of-step guard remains at `arena/board.rs:545-549` | A+native | `arena::board::rule_tests::r26_fourth_riichi_aborts_before_the_next_draw` (no `Tsumo`, wall unchanged); `SourceContractTests.test_r26_four_riichi_aborts_before_the_next_draw` | MATCH | tenhou-parity-verified for abort-before-draw; manual lists 四家立直 but does not separately spell out the draw instant (§0.1) |
| R27 | パオ | 「大三元/大四喜のパオあり。複合役満を含む得点を、ツモ＝全額・ロン＝折半で支払う。積み棒は包。」「四槓子のパオなし」 | `arena/board.rs:427-433`, `arena/board.rs:501-527` (only daisangen/daisuushi) | A | `test_r27_pao_exists_for_daisangen_and_daisuushi_only` | MATCH | none |
| R28 | リーチ後の打牌 | 「リーチ後はロン/ツモ/暗槓以外でツモ切り」 | `state/agent_helper.rs:40-46` (only the drawn tile is offered) | A | `test_r28_riichi_discard_must_be_tsumogiri` | MATCH | none |
| R29 | 一発の消え方 | 「純正巡消しは発声＆和了打診後（加槓のみ)、嶺上ツモの前（連続する加槓の２回目には一発は付かない）」 | `state/update.rs:684` (set on acceptance), `345` (own discard), `446` (call), `560` (daiminkan), `606` (kakan) | A+C | `test_r29_ippatsu_window_is_wired` | UNVERIFIED | low: affects only the 1-han ippatsu in rare consecutive-kan / chankan sequences; no executable fixture exists without a scripted board |
| R30 | 形式聴牌（5枚目待ち） | 「形式聴牌あり。5枚目を待つ聴牌あり」 | `state/update.rs:930-951` (5th tile excluded from `waits`, still a ryukyoku tenpai by `shanten`), `arena/board.rs:249` (`shanten() == 0`) | A | `test_r30_fifth_tile_wait_is_tenpai_but_not_a_wait` | MATCH | none |

## 4. Known differences and fix sketches (V3.1 historical; not applied then — see §0 for the PR-40 fixes)

### 4.1 R02 — triple ron is settled instead of aborted (V3.1 FIX_NEEDED; fixed in PR-40)

Upstream says, in the `Board` documentation:

```text
vendor/mortal/libriichi/src/arena/board.rs L26 (V3.1 pin)
/// 1. No triple-ron ryukyoku.
```

`handle_hora` walks the three other seats in turn order and pays each Hora reaction:

```text
vendor/mortal/libriichi/src/arena/board.rs:419-443
            points
                .into_iter()
                .enumerate()
                .cycle()
                .skip(single_target as usize + 1)
                .take(3)
                .filter_map(|(actor, v)| v.map(|point| (actor, point)))
                .for_each(|(actor, point)| {
                    ...
                    deltas[actor] = point.ron + kyotaku_point + honba_left * 300;
                    kyotaku_point = 0;
                    honba_left = 0;
```

There is no `三家和了` branch in the arena: the only abortive triggers are four riichi,
four kans, four winds and nine terminals. Tenhou's rule is explicit that 三家和了 is one of
the five abortive draws, and that every abortive draw is renchan.

**Observed, not inferred** (B evidence): `runs/1v3/10089_579411421376059130_b.json.gz`
contains a real triple ron at E3, honba 2, kyotaku 2 + 2 reach_accepted, dealer seat 2:

```text
hora actor 0 target 3 deltas [12600, 0, 0, -8600]  ura ["8s","5p"]
hora actor 1 target 3 deltas [0, 2000, 0, -2000]
hora actor 2 target 3 deltas [0, 0, 11600, -11600]
next start_kyoku: E3 honba 3 oya 2 scores [32100, 25500, 42100, 300]
```

The next kyoku score vector reconciles exactly: start `[20500,23500,30500,23500]`, riichi
`-1000` for seats 0 and 3, plus the three hora rows = `[32100,25500,42100,300]`. The dealer
(seat 2) was a winner, so honba went 2→3 and the dealer repeated. In Tenhou the same discard
would have been an abortive draw: no score, honba 2→3, dealer repeat, 4 riichi sticks still
on the table.

**Minimal patch sketch** (vendor change, NOT applied):

```rust
// board.rs::handle_hora, at the top, before `self.has_hora = true;`
let n_hora = reactions
    .iter()
    .filter(|ev| matches!(ev.event, Event::Hora { .. }))
    .count();
if n_hora >= 3 {
    // 三家和了: abortive draw, no score, kyotaku carried, honba +1 via game.rs
    self.abortive_ryukyoku();   // logs Ryukyoku { deltas: Some([0; 4]) }
    return Ok(Poll::End);
}
```

`has_hora` must stay `false` so `game.rs:135-137` takes the abortive branch (honba +1 and
dealer renchan) and `board.poll` sets `can_renchan = true`. A regression test needs a
scripted board; the capability gap is documented in `NativeBoardCapabilityTests`.

### 4.2 R25 — last-dealer stop sees the kyotaku payout (V3.1 FIX_NEEDED; fixed in PR-40)

Tenhou's supplementary rule says the 和了止め / 聴牌止め comparison excludes the kyotaku.
The arena, by contrast, compares `kyoku_result.scores` (`arena/game.rs L156 (V3.1 pin)`), and those
scores already contain the kyotaku: `arena/board.rs:397` sets `self.board.kyotaku = 0` and
`arena/board.rs:437` pays `kyotaku_point` to the winner before `poll()` adds `kyoku_deltas`
to `board.scores`. The 聴牌止め path is unaffected (an exhaustive draw leaves the kyotaku
outside `board.scores`), so the discrepancy is specifically a dealer **agari** at S4.

**Minimal patch sketch** (vendor change, NOT applied):

```rust
// result.rs: expose what was paid, e.g.
pub kyotaku_paid: u8,          // 0 unless a hora cleared the table
pub kyotaku_winner: Option<u8>,
// game.rs: build a kyotaku-free vector for the renchan-owari test
let mut chk = kyoku_result.scores;
if let Some(w) = kyoku_result.kyotaku_winner {
    chk[w as usize] -= kyoku_result.kyotaku_paid as i32 * 1000;
}
if kyoku_result.kyoku >= self.length - 1 && chk[oya] >= 30000 {
    let top = chk.iter().enumerate().min_by_key(|&(_, &s)| -s).map(|(i, _)| i).unwrap();
    if top == oya { self.ended = true; return Ok(()); }
}
```

`KyokuResult` currently exposes only `kyotaku_left` (`arena/result.rs:14`), which is 0 after
a hora, so the paid amount must be added; the test asserts that this field is absent today.

### 4.3 R26 — the 4-riichi abort happens after one more draw (V3.1 KNOWN_DIFFERENCE; fixed in PR-40)

`arena/board.rs:545-548` checks `accepted_riichis == 4` at the top of `step()`, before
reaction validation. But the fourth riichi is only accepted inside the `Event::None` branch
(`check_riichi_accepted()` at `arena/board.rs:587`), and that same branch then draws a tile
and broadcasts a `Tsumo` (`arena/board.rs:602-619`). The abort therefore fires on the *next*
`step()`, after one tile has been revealed. The drawn player cannot win on it, because the
abort check precedes reaction handling. No score is affected; the log contains one tile that
Tenhou would not have dealt, which is why this blocks exact log/observation equivalence but
not score equivalence. The V3.1 sketch said no patch; PR-40 now implements the fix in
`patches/0014-rule-r02-r25-r26.patch` (the sketch is retained as historical): fixing it required moving the 4-riichi
decision ahead of the draw (a structural change to `poll`).

### 4.4 R29 — ippatsu cancellation (UNVERIFIED)

`at_ippatsu` is set on `reach_accepted` (`state/update.rs:684`), cleared on the player's own
discard (`state/update.rs:345`), on any call (`446`, `515`), on daiminkan/kakan
(`560`, `606`) and on ankan (`638`). The `chankan_chance` path (`601-604`) keeps ippatsu
alive for a chankan and then clears it at the next event. The exact Tenhou nuance
("純正巡消しは発声＆和了打診後（加槓のみ)、嶺上ツモの前（連続する加槓の２回目には一発は付かない）")
would need a scripted board to verify; no executable fixture exists, so this stays
UNVERIFIED rather than being claimed as MATCH.

## 5. Native fixture coverage (what the engine can and cannot express)

`libriichi.arena` exposes only `OneVsThree` and `TwoVsTwo`; `arena::Board` is not a
`#[pyclass]` and has no constructor, so a Python test cannot inject `yama`, `haipai`, or a
scripted set of reactions. The executable part of this audit therefore uses:

- **`PlayerState`** (native per-seat rule engine): riichi preconditions, riichi-then-ankan
  (both directions), furiten and same-cycle furiten, haitei (no calls, no kan/riichi,
  海底摸月 on an open no-yaku hand), the fifth-kan prohibition, nine terminals, aka dora,
  the 5th-tile-wait tenpai rule. Each fixture drives a fixed event stream and asserts the
  returned `ActionCandidate`, i.e. the same mask the arena consumes.
- **`Stat`** (native ranking path sharing `Rankings::new` with `GameResult::rankings`):
  equal-score tie-break and the invariance of the ordinal rank to the leftover kyotaku.
- **Transcribed settlement models** validated against recorded game output: double/triple
  ron from real arena logs, double ron from a real Tenhou log, and Tenhou noten/nagashi
  payments. The model is a line-by-line transcription of the quoted source, and the
  observed values are independent of it.

The board-level cases are covered from **inside the crate** by PR-40's
`#[cfg(test)] mod rule_tests`, which can construct `BoardState`, inject a wall and call
`handle_hora`/`poll` directly:

- `arena::board::rule_tests::r02_triple_ron_aborts_without_settlement`
- `arena::board::rule_tests::r26_fourth_riichi_aborts_before_the_next_draw`
- `arena::game::rule_tests::r25_last_dealer_stop_excludes_just_paid_kyotaku`

The Python-side `NativeBoardCapabilityTests.test_r02_native_triple_ron_aborts` remains an
explicit `@unittest.skip` for the *Python* injection gap. It is counted as **skipped**,
never as passed, and it now points at the Rust fixtures instead of claiming the arena
settles triple ron.

## 6. Golden fixtures (expected / actual)

### 6.1 Triple ron, real arena output (B)

Source: `runs/1v3/10089_579411421376059130_b.json.gz`, E3, honba 2, kyotaku 2 + 2
reach_accepted, oya 2, discarder 3, discard `6p` (tsumogiri).

| Winner | Arena `hora.deltas` | Hand value (derived) | Honba | Kyotaku |
|---|---|---|---|---|
| seat 0 | `[12600, 0, 0, -8600]` | 8000 | 600 | 4000 |
| seat 1 | `[0, 2000, 0, -2000]` | 2000 | 0 | 0 |
| seat 2 | `[0, 0, 11600, -11600]` | 11600 | 0 | 0 |
| **Tenhou expected** | no deltas at all (abortive) | — | honba+1 = 3 | carried = 4 |

Discarder total: `-8600 - 2000 - 11600 = -22200 = -(8000+2000+11600) - 600`. Winner total:
`12600+2000+11600 = 26200`, difference 4000 = the kyotaku paid from escrow. The next kyoku
(`E3 honba 3 oya 2`, scores `[32100,25500,42100,300]`) confirms the renchan and the exact
score reconciliation.

### 6.2 Double ron, real arena output (B)

Source: `runs/1v3/10006_579411421376059130_b.json.gz`, S1, honba 1, kyotaku 2, oya 0,
discarder 3, discard `3p`.

| Winner | Arena `hora.deltas` | Hand value | Honba | Kyotaku |
|---|---|---|---|---|
| seat 0 (dealer) | `[5200, 0, 0, -3200]` | 2900 (dealer 30fu2han) | 300 | 2000 |
| seat 1 | `[0, 2600, 0, -2600]` | 2600 | 0 | 0 |

The dealer was a winner, so the next `start_kyoku` is S1 honba 2 oya 0 — 連荘. Tenhou's
"ダブロンは親が含まれれば連荘" and "積み棒/供託は上家取り" both hold.

### 6.3 Double ron, real Tenhou output (B)

Source: `datasets/extracted/2024/2024010101gm-00a9-0000-5c705fc0.mjson`, S2, honba 4,
kyotaku 2, oya 1, discarder 2. Logged `hora.deltas`:
`[11200,0,-9200,0]` and `[0,2900,-2900,0]`. The dealer (seat 1) was the *second* winner and
received no honba/kyotaku; the first winner after the discarder (seat 0) took 2000 kyotaku
and 1200 honba. This is the same allocation the arena produces (R03), and it is the
independent counterpart to §6.2.

### 6.4 Noten and nagashi payments, real Tenhou output (B)

- Noten: `datasets/extracted/2024/2024010100gm-00a9-0000-0d9240dd.mjson`, E4 honba 1 oya 3,
  three players tenpai → `[1000,1000,1000,-3000]`; next kyoku S1 honba 2 oya 0 (dealer
  rotated, honba+1 on a draw).
- Nagashi: `datasets/extracted/2024/2024010114gm-00a9-0000-3275a4ba.mjson`, E3 honba 1 oya 2,
  non-dealer seat 1 nagashi → `[-2000,8000,-4000,-2000]`; next kyoku E4 honba 2 oya 3.

Both shapes are reproduced exactly by the transcription of `arena/board.rs:247-298`
asserted in `exhaustive_ryukyoku_deltas` tests.

## 7. Rules that block a "Tenhou-equivalent" claim (V3.1 historical; see §0)

> **Current status:** R02/R25/R26 are fixed and tenhou-parity-verified in §0; only R29
> remains UNVERIFIED. The list below is the V3.1 blocker list, preserved as history.

Hard blockers (score or continuation):

1. **R02** triple ron — the arena pays three ron and may reset honba where Tenhou aborts and
   repeats the dealer with the kyotaku carried. Any evaluation containing a triple ron is not
   Tenhou-distributed.
2. **R25** last-dealer agari stop — the end-of-game decision can differ by one hand because
   the kyotaku is inside the compared scores.

Blocking exact log/observation equivalence, not scores:

3. **R26** four-riichi abort timing — one extra tile is revealed in the kyoku log.

Unverified (cannot be used to claim equivalence either way):

4. **R29** ippatsu cancellation timing.

Everything else in §3 is MATCH under the stated evidence level. In particular, the
high-frequency rules that dominate strength measurement — riichi legality, furiten,
riichi-then-ankan, kan-dora timing, noten payments, dealer repeat, honba, kyotaku, double
ron and final kyotaku — are verified, most of them executably.

## 8. Recommendations for downstream lanes

- **PR-16 / PR-17 reports**: state "relative strength on the pinned Mortal arena", never
  "Tenhou-equivalent". If a run contains a triple ron, it should be reported as
  out-of-distribution rather than silently included. The frequency is low (1 in 2,004
  recorded arena games), so a 12k-game evaluation is unlikely to be materially biased, but
  the claim must be qualified.
- **PR-32 risk-oracle / PR-10 censoring**: double-ron settlement is now verified executably
  (§6.2); triple-ron outcomes must not be used as a Tenhou-faithful label. The oracle should
  treat "3 simultaneous ron" as an unsupported branch or apply the Tenhou abort rule
  explicitly.
- **PR-12 shards / PR-07R drift**: the arena's mjai logs are self-consistent (verified
  against §6.1/§6.2), so off-policy replay of arena logs is sound. Do not assume, however,
  that a *Tenhou* log and an arena log share the same continuation distribution for
  three-ron hands.

## 9. Cross-lane notes (not H-04 ownership)

- **BR-01 / H-01**: `riichiformer/data/replay.py::final_scores_from_events()` adds
  `1000 * kyotaku` to the hora actor although the arena's `hora.deltas` already include the
  kyotaku (`arena/board.rs:437`). That is a score-accounting bug in a log consumer, not an
  arena rule difference; H-01 owns the fix. `scripts/eval_report.py` does not double count
  (it only restores the leftover kyotaku at `end_game`), and this audit's golden fixtures
  use the raw `hora.deltas`.
- **`libriichi.stat.Stat`** never distributes the final leftover kyotaku, unlike
  `arena/game.rs:172-174`. Its `rank` is nevertheless consistent with the arena because
  the kyotaku goes to the first maximum; its `point` field can be off by
  `1000 * kyotaku_left`. `NativeRankingTests.test_r15_leftover_kyotaku_does_not_change_rank`
  pins that reasoning.
- **`scripts/eval_report.py`** documents the same tie-break (`rank_of`, stable sort) and the
  same final-kyotaku rule (`scores.index(max(scores))` = first maximum). It matches the
  arena for R14/R15.

## 10. Logs

Full pytest output: `docs/riichiformer/v3/logs/H04_pytest.log`.

```text
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q tests/rf_test_ruleset_parity.py
```

V3.1 run (2026-11-21, Python 3.12.3, pytest 9.2.0):
**51 passed, 1 skipped, 197 subtests passed in 0.08s**. The skip is
`NativeBoardCapabilityTests::test_r02_native_triple_ron_aborts`, which documents the
Python board-injection gap and is not counted as a pass.

PR-40 run (same date, worktree `v32-rules`):
`tests/rf_test_ruleset_parity.py` → **51 passed, 1 skipped, 224 subtests passed**;
`tests/rf_test_rule_fixes.py` → **10 passed**; full `tests/rf_test_*.py` → **504 passed,
2 skipped, 269 subtests**; native
`cargo test -p libriichi --lib --no-default-features rule_tests` → **3 passed**.
See `docs/riichiformer/v3_2/PR40_RULES.md` for the exact commands and exit codes.
