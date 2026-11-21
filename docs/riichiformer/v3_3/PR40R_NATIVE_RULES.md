# PR-40R — native rules rebuild and R29 ippatsu fixture

- **Task**: PR-40R (Dev-D, Rules/QA), RiichiFormer V3.3 Wave 2
- **Worktree / branch**: `/home/ygsldr/riichi-ai/worktrees/v33-d` @ `rf/pr40r-r29`
- **BASE_SHA**: `e76022600018e9adc6536ccbb904db8c9dc498b3`
- **Vendor pin**: `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2` (libriichi)
- **Patches**: `patches/0014-rule-r02-r25-r26.patch` (R02/R25/R26),
  `patches/0015-rule-r29-ippatsu-fixture.patch` (R29, this task)
- **Native rebuild**: clean clone of the pinned vendor in `/tmp/v33d_vendor`,
  `cargo build -p libriichi --lib --release`, CPU only
- **Build sha256**: `582e36734b10c3d3b1230a834eacc494f94e40f6d642c712eadc23f6cabfaf4e`
- **Python**: `/home/ygsldr/riichi-ai/.venv/bin/python` 3.12.3, pytest 9.2.0
- **Date**: 2026-11-21 (Asia/Shanghai)
- **Arena rule scope**: `tenhou_approx`. This document does not claim "official
  complete equivalence"; R29 is certified only for the event-stream window described in §2.

## 1. Rule matrix

| ID | Rule | Tenhou behaviour (quote + [man#RULE](https://tenhou.net/man/#RULE)) | Pinned arena | Evidence | Fixture / command | Status | Scope / caveat |
|---|---|---|---|---|---|---|---|
| R02 | 三家和了 | 「途中流局(九種九牌/四家立直/三家和了/四槓散了/四風連打)あり。すべて連荘」 — three simultaneous ron is an abortive draw, no payment, honba+1, dealer renchan, kyotaku carried | `arena/board.rs:383` `if n_hora >= 3` guard before `has_hora`/kyotaku/ura; continuation `arena/game.rs:135-137` | native + source | `arena::board::rule_tests::r02_triple_ron_aborts_without_settlement` | `tenhou-parity-verified` | The settlement branch is executed; the honba/dealer continuation is source-read from `game.rs:135-137` and is not executed by that fixture. |
| R25 | ラス親の和了止め・供託除外 | 「親が連荘(和了/聴牌)し原点を超えたトップになる(和了/聴牌清算後、供託料を除く)」 | `arena/board.rs:396` snapshots the paid sticks; `arena/result.rs:16-32` exposes the kyotaku-free vector; `arena/game.rs:216-230` uses it | native | `arena::game::rule_tests::r25_last_dealer_stop_excludes_just_paid_kyotaku` | `tenhou-parity-verified` | The all-last dealer-agari decision is executed; the surrounding `Game` loop is not scripted. |
| R26 | 四家立直の流局時点 | 「四家立直」 — abortive draw. The fourth acceptance is the abort event; the next draw must not happen | `arena/board.rs:587-594` aborts immediately after `check_riichi_accepted()`, before the draw; top-of-step safety net at `arena/board.rs:545-549` | native | `arena::board::rule_tests::r26_fourth_riichi_aborts_before_the_next_draw` | `tenhou-parity-verified` | The manual lists 四家立直 but does not separately spell out the draw instant; the fixture pins the only event order consistent with the abort and asserts no `Tsumo` after the fourth acceptance. |
| R29 | 一発の消え方（加槓・槍槓） | 「純正巡消しは発声＆和了打診後（加槓のみ)、嶺上ツモの前（連続する加槓の２回目には一発は付かない）」; yaku note 「一発と槍槓（連続槓の１回目）は複合する」 | `state/update.rs:684` sets `at_ippatsu` on own `ReachAccepted`; `state/update.rs:58-60` clears it when a pending `chankan_chance` is consumed at the next event; `state/update.rs:596-608` keeps it only for a waited opponent kakan and clears it otherwise | native | `state::update::r29_tests::{r29_first_kakan_keeps_ippatsu_through_the_chankan_probe, r29_kakan_window_closes_at_the_next_event_before_the_rinshan_tsumo, r29_second_consecutive_kakan_has_no_ippatsu, r29_own_discard_and_other_calls_close_the_window}` | `MATCH` | The fixture drives `PlayerState` (the same state machine the arena uses) with the arena's kakan event order and calls `agari_points` at the chankan probe, exactly where `BoardState::handle_hora` calls it. It does not script a full `Board` game; see §2 and §5. |

Status vocabulary: `tenhou-parity-verified` = the clause is quoted and an executable native
fixture exercises the exact branch; `MATCH` = the pinned state machine's event order agrees
with the clause in the fixture; `UNVERIFIED` / `KNOWN_DIFFERENCE` would be used if the
timing could not be established. R29 no longer carries `UNVERIFIED`.

## 2. R29 executable fixture

`patches/0015-rule-r29-ippatsu-fixture.patch` adds only `#[cfg(test)] mod r29_tests` at the
end of `vendor/mortal/libriichi/src/state/update.rs` (185 added lines). It does not change
production behaviour and does not expose `at_ippatsu` to Python; the tests are inside the
module and read the private field directly.

The fixture starts a kyoku as dealer with player 0 holding
`123m 456p 789s 123p 5p` (a 5p tanki wait, no base yaku). Player 0 discards once to clear
`can_w_riichi`, then declares a normal riichi; `ReachAccepted` sets `at_ippatsu`. The tests
then drive the exact event order `Board::step` emits for a kakan: `Kakan` → `Dora` (new kan
dora) → `Tsumo` (rinshan) → `Dahai`.

| Test | Event sequence | Assertion |
|---|---|---|
| `r29_first_kakan_keeps_ippatsu_through_the_chankan_probe` | opponent `Kakan` on the waited 5p | `at_ippatsu` is still true, `chankan_chance` is active, and `agari_points(true, &[])` is `7700` — riichi + ippatsu + chankan, 3 han 40 fu dealer. Losing either ippatsu or chankan would be 2 han / 3900. |
| `r29_kakan_window_closes_at_the_next_event_before_the_rinshan_tsumo` | waited `Kakan` → `Dora` → `Tsumo` | `at_ippatsu` is true after the kakan, false after the `Dora` event, and still false at the rinshan `Tsumo`. This is the clause's 「嶺上ツモの前」 timing. |
| `r29_second_consecutive_kakan_has_no_ippatsu` | first `Kakan` on 9p (not a wait) → `Dora` → rinshan `Tsumo`/`Dahai` → second `Kakan` on 5p | The first kakan clears the window immediately (no chankan chance); the second consecutive kakan *is* a chankan chance, but `at_ippatsu` is false and `agari_points` is `3900` (riichi + chankan only, 2 han 40 fu dealer). This is 「連続する加槓の２回目には一発は付かない」. |
| `r29_own_discard_and_other_calls_close_the_window` | own draw + discard; separately an opponent `Pon` | Both clear `at_ippatsu`. |

The chankan-probe assertion is not a synthetic scoring call: `BoardState::handle_hora`
calls `player_states[actor].agari_points(is_ron, &ura_indicators)` while the `Kakan`
event's `chankan_chance` and `at_ippatsu` are still live, and only afterwards settles the
`Hora` event. Calling `agari_points` at that instant is therefore the production path.

## 3. Clean rebuild provenance

The lane worktree's `vendor/mortal` is a source-only copy (no usable submodule checkout), so
the rebuild was made from a real clone of the pinned module and the resulting `.so` was
copied back to the lane:

```text
git clone --no-hardlinks /home/ygsldr/riichi-ai/.git/modules/vendor/mortal /tmp/v33d_vendor
git -C /tmp/v33d_vendor checkout --detach 0cff2b52982be5b1163aa9a62fb01f03ce91e0d2
git -C /tmp/v33d_vendor rev-parse HEAD
# 0cff2b52982be5b1163aa9a62fb01f03ce91e0d2
git -C /tmp/v33d_vendor status --short            # empty after checkout
git -C /tmp/v33d_vendor apply .../patches/0014-rule-r02-r25-r26.patch
git -C /tmp/v33d_vendor apply --check .../patches/0015-rule-r29-ippatsu-fixture.patch
git -C /tmp/v33d_vendor apply .../patches/0015-rule-r29-ippatsu-fixture.patch
```

After both patches the clone's `git status --short` is:

```text
 M libriichi/src/arena/board.rs
 M libriichi/src/arena/game.rs
 M libriichi/src/arena/result.rs
 M libriichi/src/state/update.rs
```

The submodule pointer was **not** committed. The lane worktree still reports
`m vendor/mortal` (modified content) by design.

Toolchain and build:

```text
rustc 1.97.1 (8bab26f4f 2026-07-14) [x86_64-unknown-linux-gnu]
cargo 1.97.1 (c980f4866 2026-06-30)
CARGO_HOME=/home/ygsldr/riichi-ai/.cargo-v33
PYO3_PYTHON=/home/ygsldr/riichi-ai/.venv/bin/python
CARGO_BUILD_JOBS=4 cargo build --offline -p libriichi --lib --release
# exit 0; initial clean build 50.28 s; cached provenance re-run 0.07 s, same sha
# sha256(target/release/libriichi.so) =
#   582e36734b10c3d3b1230a834eacc494f94e40f6d642c712eadc23f6cabfaf4e
```

The same `.so` was copied to
`vendor/mortal/target/release/libriichi.so` (sha256
`582e36734b10c3d3b1230a834eacc494f94e40f6d642c712eadc23f6cabfaf4e`). A second
independent build by the Lead in `/tmp/v33d_check` produced
`cee3e18e994ba08bf1dcbe60f4caadbb30795e6d7ce592afc0cc1dc09f54c0a5`; the ELF differs
because release debug info embeds the absolute source path, not because the source differs.
The stable identity in this repository is the patch series plus the pinned vendor commit.

## 4. Command log

| # | Command | Exit | Duration | Result |
|---|---|---|---|---|
| 1 | `git clone --no-hardlinks .git/modules/vendor/mortal /tmp/v33d_vendor && git checkout --detach 0cff2b5…` | 0 | < 1 s | HEAD `0cff2b5…`, clean |
| 2 | `git apply 0014-rule-r02-r25-r26.patch` | 0 | < 1 s | 3 files, +241/−16 |
| 3 | `git apply --check/apply 0015-rule-r29-ippatsu-fixture.patch` | 0 | < 1 s | 1 file, +185 |
| 4 | release build (above) | 0 | 50.28 s clean | `.so` sha `582e367…` |
| 5 | `CARGO_HOME=… cargo test --offline -p libriichi --lib --no-default-features` in `worktrees/v33-d/vendor/mortal` | 0 | 44.99 s first run (2.51 s cached) | **35 passed, 0 failed** |
| 6 | same command with filter `r29_tests` in the clean clone | 0 | 16.16 s first run (0.12 s test time) | **4 passed, 0 failed** |
| 7 | same command with filter `rule_tests` | 0 | ~1 s | **3 passed, 0 failed** (R02/R25/R26) |
| 8 | `PYTHONPATH=vendor/mortal/target/release .venv/bin/python -m pytest -q tests/rf_test_native_rules_r29.py tests/rf_test_rule_fixes.py tests/rf_test_ruleset_parity.py` | 0 | see `logs/PR40R_python_tests.log` | targeted Python suite green |

Exact logs: [`logs/PR40R_release_build.log`](logs/PR40R_release_build.log),
[`logs/PR40R_cargo_test_full.log`](logs/PR40R_cargo_test_full.log),
[`logs/PR40R_cargo_test_r29.log`](logs/PR40R_cargo_test_r29.log),
[`logs/PR40R_python_tests.log`](logs/PR40R_python_tests.log).

## 5. NOT_RUN / remaining caveat

- Full `Board`-level scripted R29 game: **NOT_RUN**. `arena::Board` is not injectable from
  Python and the fixture deliberately stays at the `PlayerState` level, which is the layer
  that owns `at_ippatsu`. The production `handle_hora` call order is covered by source
  inspection and by calling `agari_points` at the same instant.
- Full 12k-game arena, GPU, and full-corpus rebuild: **NOT_RUN**; this task is CPU-only.
- Parent `vendor/mortal` (root worktree) rebuild: **NOT_RUN**; only the lane worktree and
  the clean clone were written.
- No commit of the `vendor/mortal` submodule pointer.
- R29's certification is scoped to the quoted kakan/chankan window. It is not a claim that
  every ippatsu interaction in every rule variant is certified.
