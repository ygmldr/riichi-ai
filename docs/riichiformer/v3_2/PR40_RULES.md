# PR-40 (Dev-RULES) — R02/R25/R26 fixes and R29 status

> **V3.3 follow-up (PR-40R, 2026-11-21):** R29 is no longer UNVERIFIED. The executable
> fixture is `patches/0015-rule-r29-ippatsu-fixture.patch`; the current matrix, build sha
> and command log are in `docs/riichiformer/v3_3/PR40R_NATIVE_RULES.md`. The R29
> `UNVERIFIED`/BLOCKERS text below is the V3.2 state, kept as history.

Date: 2026-11-21 (Asia/Shanghai). Task: RiichiFormer V3.2 Wave 1, PR-40.
This report records commands actually run in the `v32-rules` worktree; it does not claim
full official Tenhou equivalence. R29 remained UNVERIFIED in V3.2 (superseded by PR-40R
above).

## Baseline

| Item | Value |
|---|---|
| TASK_ID | PR-40 (Dev-RULES) |
| Worktree | `/home/ygsldr/riichi-ai/worktrees/v32-rules` |
| Branch | `rf/v32-rules` |
| BASE_SHA | `cbf166a356f83c739b31f0b5a0b241b954ef9e0a` |
| Vendor | `vendor/mortal` @ `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2`, clean pin |
| Python | `/home/ygsldr/riichi-ai/.venv/bin/python` (3.12) |
| Rust | `cargo 1.97.1` / `rustc 1.97.1` |
| Native build | `CARGO_BUILD_JOBS=4 cargo build -p libriichi --lib --release` (CPU only) |

The worktree vendor submodule was clean at the pin. The parent vendor tree is dirty with
`patches/0001-0013`; it was **not** modified. `patches/0014-rule-r02-r25-r26.patch` was
generated from the worktree submodule changes, and the parent tree was only read with
`git apply --check`.

## Changes

| File | Change |
|---|---|
| `vendor/mortal/libriichi/src/arena/board.rs` | R02: `if n_hora >= 3` guard at the top of `handle_hora` calls `abortive_ryukyoku()` before `has_hora`/kyotaku/ura. R25: snapshot `kyotaku_paid` and `kyotaku_winner` when a hora clears the table. R26: abort immediately after `check_riichi_accepted()` when `accepted_riichis == 4`, before the next draw. Add `#[cfg(test)] mod rule_tests` with R02/R26 fixtures. |
| `vendor/mortal/libriichi/src/arena/result.rs` | `KyokuResult` gains `kyotaku_paid`/`kyotaku_winner` and `scores_excluding_hora_kyotaku()` (subtracts the just-paid sticks from the winner). |
| `vendor/mortal/libriichi/src/arena/game.rs` | `last_dealer_stop()` uses the kyotaku-exclusive vector; raw `self.scores[oya]` comparison removed. Add R25 native fixture. |
| `patches/0014-rule-r02-r25-r26.patch` | The three source changes above, including the native tests (360 lines). |
| `tests/rf_test_rule_fixes.py` | New PR-40 Python fixtures: R02 before/after scores/ranks, R25 decision and rank comparison, R26 event order, R29 must remain UNVERIFIED, report vocabulary and patch presence. |
| `tests/rf_test_ruleset_parity.py` | Updated anchors/line numbers and R02/R06/R25/R26/R29 assertions; the old triple-ron settlement test is kept as historical evidence. |
| `docs/riichiformer/v3/TENHOU_RULE_PARITY.md` | §0 PR-40 update, current matrix, parity labels; V3.1 history preserved and labelled. |

## Tenhou clauses and anchors

Official manual (段位戦4人打ち), <https://tenhou.net/man/#RULE>:

- 「途中流局(九種九牌/四家立直/三家和了/四槓散了/四風連打)あり。すべて連荘」 (R02, R26)
- 「親が連荘(和了/聴牌)し原点を超えたトップになる(和了/聴牌清算後、供託料を除く)」 (R25)
- 「純正巡消しは発声＆和了打診後（加槓のみ)、嶺上ツモの前（連続する加槓の２回目には一発は付かない）」 (R29, not certified)

Anchors in the patched source (full quoted table in `TENHOU_RULE_PARITY.md` §0.2):

| Rule | file:line | text |
|---|---|---|
| R02 | `arena/board.rs:383` | `if n_hora >= 3 {` |
| R02 | `arena/board.rs:387` | `self.abortive_ryukyoku();` |
| R02 | `arena/board.rs:391` | `self.has_hora = true;` (after the guard) |
| R02 | `arena/game.rs:135-137` | abortive continuation: honba+1, poll again |
| R25 | `arena/board.rs:396` | `self.kyotaku_paid = self.board.kyotaku;` |
| R25 | `arena/board.rs:439-441` | first winner receives the kyotaku |
| R25 | `arena/result.rs:29-32` | `scores[winner] -= kyotaku_paid * 1000` |
| R25 | `arena/game.rs:216-230` | `last_dealer_stop` uses the adjusted vector |
| R26 | `arena/board.rs:587-594` | abort after acceptance, before draw |
| R26 | `arena/board.rs:596` | the draw that now cannot happen |
| R29 | `state/update.rs:684` | `self.at_ippatsu = true;` (wired, not certified) |

## BEFORE_AFTER_SCORES

### R02 — recorded triple ron, E3 honba 2, oya 2, discarder 3

Start scores `[20500,23500,30500,23500]`; two `reach_accepted` in the kyoku, so 4 sticks
on the table.

| Path | seat 0 | seat 1 | seat 2 (oya) | seat 3 | honba | kyotaku | ranks |
|---|---:|---:|---:|---:|---:|---:|---|
| Pre-fix recorded settlement (three hora paid) | 32100 | 25500 | 42100 | 300 | 3 | 0 | `[2,3,1,4]` |
| PR-40 abort (native fixture) | 19500 | 23500 | 30500 | 22500 | 3 | 4 | `[4,2,1,3]` |

The abort row is `start - 1000` for the two riichi seats, no hora deltas, honba 2→3, same
dealer, kyotaku carried. The recorded row remains as pre-fix evidence
(`runs/1v3/10089_579411421376059130_b.json.gz`).

### R25 — S4 dealer agari stop (kyoku 7 ⇒ oya seat 3), 4 kyotaku paid to oya

| Case | raw scores | kyotaku-free comparison | pre-fix decision | PR-40 decision |
|---|---|---|---|---|
| kyotaku decisive | `[19000,19000,29000,33000]` | `[19000,19000,29000,29000]` | ends (33000 ≥ 30000, top) | continues (29000 < 30000) |
| genuine 30000+ | `[19000,19000,28000,34000]` | `[19000,19000,28000,30000]` | ends | ends |

Pre-fix final scores for the decisive case: `[19000,19000,29000,33000]`, ranks
`[3,4,2,1]`. Under PR-40 the hand continues; a **constructed** continuation (seat 0 rons
12000 from seat 3) gives `[31000,19000,29000,21000]`, ranks `[1,4,2,3]`. The fixture
asserts the decision, not the constructed continuation.

### R26 — fourth riichi event order

| | log | draw after acceptance | tiles_left |
|---|---|---|---|
| Pre-fix | `reach_accepted` → `tsumo` → `ryukyoku` | yes | 9 |
| PR-40 | `reach_accepted` → `ryukyoku` | no | 10 |

No score changes; the native fixture asserts `Poll::End`, `accepted_riichis == 4`,
`tiles_left == 10`, `yama.len() == 10`, no `Tsumo` after `ReachAccepted`, and an
`EndKyoku` tail.

## TEST_COMMANDS and EXIT_CODES

All commands were run from `/home/ygsldr/riichi-ai/worktrees/v32-rules`. Because the
sandbox home is not writable, the native test run used a worktree-local `CARGO_HOME`
(copy of the cached registry); this is a test-environment detail, not a source change.

1. Release build (CPU only):

```text
cd vendor/mortal
CARGO_BUILD_JOBS=4 cargo build -p libriichi --lib --release
# Finished `release` profile in 18.32s
# EXIT 0
```

2. Native rule fixtures:

```text
CARGO_HOME=<worktree>/.cargo-home PYO3_PYTHON=<venv>/bin/python \
  CARGO_BUILD_JOBS=4 cargo test -p libriichi --lib --no-default-features rule_tests
# running 3 tests: r02..., r25..., r26... all ok
# test result: ok. 3 passed; 0 failed; 0 ignored
# EXIT 0
```

3. Full libriichi unit tests (no default features so the test binary links without the
PyO3 extension-module flags; the arena code under test is unchanged):

```text
cargo test -p libriichi --lib --no-default-features
# test result: ok. 31 passed; 0 failed; 0 ignored
# EXIT 0
```

4. Targeted Python:

```text
PYTHONPATH=vendor/mortal/target/release <python> -m pytest -q \
  tests/rf_test_rule_fixes.py tests/rf_test_ruleset_parity.py
# 61 passed, 1 skipped, 224 subtests passed
# EXIT 0
```

5. Full suite:

```text
PYTHONPATH=vendor/mortal/target/release <python> -m pytest -q \
  $(git ls-files 'tests/rf_test_*.py' | tr '\n' ' ') tests/rf_test_rule_fixes.py
# 504 passed, 2 skipped, 269 subtests passed in 159.68s (0:02:39)
# EXIT 0
```

6. Patch apply-check against the parent dirty vendor (read-only; nothing written):

```text
git -C /home/ygsldr/riichi-ai/vendor/mortal apply --check \
  /home/ygsldr/riichi-ai/worktrees/v32-rules/patches/0014-rule-r02-r25-r26.patch
# EXIT 0
```

No conflict hunk was reported. The parent vendor already has `0001-0013`; `0014` touches
only `libriichi/src/arena/{board,game,result}.rs`, which no earlier patch modifies.

## P0 / P1 / P2

- **P0**: none. No hidden-information, split/POV, schema or rights-gate change.
- **P1**: R29 is still UNVERIFIED (no executable fixture for the consecutive-kakan /
  chankan ippatsu timing). The R26 manual does not separately spell out the draw instant;
  the implemented ordering is the clause-consistent reading and the native fixture pins it.
- **P1**: `KyokuResult` gained two public fields; any out-of-tree constructor would need
  updating. There is no such constructor in this repository (checked with grep).
- **P2**: the top-of-`step` R26 guard is retained as a safety net; the immediate guard is
  the one the fixture exercises.
- **P2**: `tests/rf_test_rule_fixes.py` asserts `patches/0014-...patch` exists, so the file
  and test stay together.
- **P2**: the V3.1 sections of `TENHOU_RULE_PARITY.md` still contain the old
  `FIX_NEEDED`/`KNOWN_DIFFERENCE` prose; they are explicitly labelled historical and the
  current matrix is §3.
- **P2**: `README.md` still describes the patch index as `0001–0013` (0011 absent); the
  new `0014` is owned by this task but README is outside its write scope. Lead/owner should
  add `0014` to the index.

## NOT_RUN

- R29 native timing fixture: NOT_RUN (no board/`PlayerState` observation of `at_ippatsu`;
  the field is private). No green claim.
- Parent-repository rebuild/tests: NOT_RUN. The patch was only apply-checked against the
  parent vendor; the parent `libriichi.so` was not rebuilt.
- GPU/ROCm/CUDA: not used for PR-40. Native build and tests are CPU-only.
- No commit of the `vendor/mortal` submodule pointer; the worktree submodule is dirty by
  design (changes are carried by `patches/0014-rule-r02-r25-r26.patch`).

## BLOCKERS

- R29 remains UNVERIFIED until a fixture can observe the ippatsu window across consecutive
  kakan/chankan; do not report it as MATCH.
- Parent-side consumers must apply `patches/0014-rule-r02-r25-r26.patch` and rebuild
  `libriichi` before using the fixed arena; the parent vendor was not written by this task.

## NEXT_GATE

Lead: apply `0014` to the parent vendor, rebuild `libriichi` with the pinned toolchain,
run the full `tests/rf_test_*.py` suite there, and have the Reviewer check the three
`arena/*` hunks. R02/R25/R26 may then be reported as `tenhou-parity-verified`; R29 must
stay `UNVERIFIED` until a real timing fixture exists.
