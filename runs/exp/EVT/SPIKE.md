# 18-A Spike — public event stream at the dataset decision point

Plan: `.omo/plans/fork-roadmap-steps-1-10.md` todo 18 (18-A).
Date: 2026-09-17. Submodule pin: `0cff2b5` + patches 0002/0003/0004/0005/0012.

## Question

Can the dataset side obtain, for every decision point it records, the **public**
event sequence from the start of the current kyoku up to and including the event
that triggered the decision — with no hidden information and no future leakage?

## Verdict: yes

The dataset loader already walks the whole mjai log in order and already keeps
per-decision snapshots of derived per-event quantities. The event stream is one
more snapshot of the same kind.

## Code path (dataset side)

All line numbers below refer to the patched tree (`patches/0003` + `0006`
applied); the pre-patch equivalents from the plan are given in parentheses.

1. **Iteration.** `Gameplay::load_events_by_player` walks the log with a
   sliding window: `for wnd in events.windows(4)` — `gameplay.rs:287`
   (plan: `:268`). `cur = &wnd[0]` (`:313`), so every event of the log is
   processed exactly once, in file order, except the final three events
   (which are `hora/ryukyoku -> end_kyoku -> end_game`; no decision can occur
   after them).
2. **State update.** `state.update(cur)` — `gameplay.rs:354` (plan: `:335`).
   `PlayerState::update` resets `last_cans` and recomputes the action
   candidates for the POV; the decision exists iff `cans.can_act()`.
3. **Decision registration.** `add_entry` — `gameplay.rs:459-470`
   (plan: `:422-442`). It pushes one element to each per-decision vector:
   `obs`, `actions`, `masks`, `at_kyoku`, `apply_gamma`, `at_turns`,
   `shantens` (+ `invisible_obs` when oracle). The `shantens` field
   (`gameplay.rs:58`) / push (`gameplay.rs:465`) pair is the snapshot pattern
   the plan points at (`:58` / `:431` / `:226-228`): a per-event value is
   pushed into a per-decision vector at the exact moment the decision is
   emitted.
4. **Event snapshot (new, patch 0006).** The token buffer
   (`events_buf`, `gameplay.rs:63`) is appended for every `cur` right after
   `state.update` bookkeeping (`gameplay.rs:330-343`), and `add_entry`
   snapshots it (`self.events.push(self.events_buf.clone())`,
   `gameplay.rs:470`). The buffer is cleared by `StartKyoku`
   (`gameplay.rs:332-334`), so every decision stream starts with that
   kyoku's `start_kyoku`.
5. **Event shapes.** `mjai/event.rs:20-120` defines the enum. The only field
   that carries hidden information is `StartKyoku.tehais` (`event.rs:42`) —
   the tokenizer never reads it; it emits the public `dora_marker` only.
   `Hora.ura_markers` / `Hora.deltas` and `Ryukyoku.deltas` are likewise not
   read. Everything else (`tsumo/dahai/chi/pon/kan/dora/reach/hora/...`) is
   public by the mjai protocol.

## Token spec (single source of truth: `libriichi/src/mjai/tokens.rs`)

Shape `(T, 5)` `int16`, columns `(type_id, tile, actor, turn, flags)`,
`T <= MAX_EVENT_TOKENS = 384`.

| col | values |
|-----|--------|
| `type_id` | 0 `none`, 1 `start_game`, 2 `start_kyoku`, 3 `tsumo`, 4 `dahai`, 5 `chi`, 6 `pon`, 7 `daiminkan`, 8 `kakan`, 9 `ankan`, 10 `dora`, 11 `reach`, 12 `reach_accepted`, 13 `hora`, 14 `ryukyoku`, 15 `end_kyoku`, 16 `end_game` (all <= 19) |
| `tile` | `deaka()` id 0..=33, or 34 = none. `StartKyoku`/`Dora` carry the dora marker; `ankan` carries `consumed[0]` |
| `actor` | absolute seat 0..=3, or 4 = none |
| `turn` | POV-relative: number of POV `Tsumo` events so far in this kyoku, including the current event when it is a POV tsumo. Equals `PlayerState::at_turn()` after the event, but is derived from the log alone so the arena path (which has no per-event historical state) reproduces it exactly |
| `flags` | bit0 = representative tile is aka; bit1 = `dahai` tsumogiri; bit2 = a `consumed` tile is aka |

`turn` is the only POV-dependent column; everything else is a pure function of
the event. The cap 384 is a safety bound: the longest stream observed in
12 full hanchan was `T = 154`.

## Empirical verification (dataset side)

`runs/exp/EVT/dataset_smoke.py` (12 full hanchan, 29,926 events, 18,267
decisions over all 4 POVs) asserts, using an **independent Python
re-implementation** of the spec (`runs/exp/EVT/event_verify.py`) and an
independent `PlayerState` replay:

- `len(events) == len(obs) == len(at_kyoku)` for every game/POV;
- every stream is byte-identical to the independent encoding of
  `raw_kyoku_events[:T]` (content, count, no fabrication);
- first token is `start_kyoku` (type 2) with the kyoku's dora marker;
- last token is a decision trigger (`tsumo/dahai/chi/pon/kan/reach/...`);
- `T` is strictly increasing within a kyoku (no future events);
- `T <= 384` (observed max 154).

Two spec bugs were found and fixed by this harness before it went green:
`start_kyoku` did not set the aka flag of an aka dora marker, and
`chi/pon/daiminkan/kakan` did not set the aka flag of an aka `pai`. A third,
rarer one — `ankan` whose representative tile `consumed[0]` is aka — was caught
by the 18-C arena run (game slot 22 of the 40-game smoke) and is now pinned by
a synthetic log in `dataset_smoke.py` (`synthetic.ankan_flags == [5]`).
Coverage in the final dataset run: 14,645 aka-representative tokens and 981
consumed-aka tokens verified byte-exact.

## 18-C outcome (same spec, arena side)

`patches/0010` feeds `ctx.log` through `encode_event_log` and appends the
result as the 4th `react_batch` argument; the engine packs it with the same
`event_data.collate_events`. `runs/exp/EVT/EVAL_FEED.json` (40 games, both
engines `enable_quick_eval=False` so no decision bypasses the engine):
25,177/25,177 engine-received samples matched the ground-truth decision points
of the written game logs — non-empty/non-zero (①), token count == public
events up to that decision (②, plus md5 of the token bytes), and
non-decreasing within each kyoku (③). A `enable_quick_eval=True` sanity run
(4 games, 2,234 engine samples) confirms the bypassed decisions are exactly
those that never reach Python (the encoder is only invoked after the
quick-eval early return in `set_scene`).

## Decision-enumeration difference between dataset and arena (documented)

The dataset only records a decision when it can also determine a label. When a
POV's `can_act` is real but **pre-empted** by a faster call or another player's
ron, the dataset skips the point:

- `can_chi()` true, next event is another player's `pon/daiminkan/chi` ->
  skipped;
- `can_ron_agari` true, next event is another player's `hora` -> skipped.

Measured over the 12 hanchan: 18,238 dataset decision points vs 18,298 arena
`can_act` points; all 60 skipped points are pre-empted (`pon` 34, `hora` 26).
The arena evaluates the pre-empted points too (it asks every player who can
act, then the board picks the highest-priority reaction). Consequence for todo
19: the event branch will see token streams at *more* decision points during
eval/self-play than in the offline dataset; the token stream itself is the
same shape and spec on both sides.

## Consequence for 18-C

The arena's `BoardState.log` (`arena/board.rs:81`) is exactly the same kind of
public per-kyoku sequence: `haipai()` pushes `start_kyoku` + first `tsumo`
(`board.rs:223,236`), every committed reaction is appended
(`board.rs:594-655`), and the log is taken at kyoku end (`board.rs:126` in
`arena/game.rs`). `agent/mortal.rs:200-203` receives it as `ctx.log` but
discards it. Patch 0010 therefore only has to feed `ctx.log` through the same
`tokens.rs` encoder and append the result to `react_batch`'s arguments.
