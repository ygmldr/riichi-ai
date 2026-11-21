# ARENA_ADAPTER — PR-16 mock-first arena inference adapter

**Status**: PR-16 draft (Arena owner). Not frozen until Lead accepts PR-14/PR-16.
**Task**: PR-16 — arena inference adapter + rule-consistency contract (mock-first)
**Worktree/branch**: `/home/ygsldr/riichi-ai/worktrees/arena` @ `rf/pr16-arena`
**Base**: `de85040` (includes PR-08/09/10/11/13; the PR-10 `tiles_left` correction is
being integrated separately and this PR does not read or re-derive that field).
**Owned files**: `riichiformer/runtime/agent.py`,
`riichiformer/runtime/mjai_adapter.py`, `scripts/rf_eval.py`,
`tests/rf_test_arena.py`, this doc, `docs/riichiformer/v3/logs/PR16_pytest.log`.
**Tests**: 17 passed, 17 subtests, exit 0 (see `PR16_pytest.log`).

## 0. Scope and claim

This PR is deliberately **mock-first**:

- it implements the 46-action decode/encode contract, a legal-mask-guarded
  `RiichiFormerAgent`, and a deterministic 4-seat mock hand loop;
- it does **not** run a real rules engine, a real checkpoint, the Mortal 1v3
  bridge, datasets, or a frozen evaluation pool;
- the mock loop's scores/ranks are synthetic labels for plumbing only.  They are
  **not** a strength measurement and must not be reported as one.
- Real checkpoint evaluation is gated on the PR-14 checkpoint and PR-17's
  pre-registered baseline gate.  Until then this adapter advertises **relative
  strength only** and makes **no Tenhou-equivalent claim**.

Rule-consistency flags carried by every mock run (H-04, `TENHOU_RULE_PARITY.md`):

| flag | pinned arena behaviour | consequence for PR-16 reporting |
| --- | --- | --- |
| **R02** | triple ron is settled by the arena's payment loop (all winners paid), not by a Tenhou triple-ron abort | multi-ron outcomes are relative-strength data, not Tenhou-equivalent |
| **R25** | last-dealer stop uses kyotaku-inclusive scores | end-of-game rank may differ from Tenhou at the S4 boundary |
| **R26** | four-riichi abort is log-only / timing differs; the arena draws one extra tile | exhaustive-draw hands are not byte-identical to Tenhou |

The summary JSON carries `tenhou_equivalent: false` and the three flags in its
`note`; tests assert all three are present.

## 1. Action decode contract (`runtime/mjai_adapter.py`)

The frozen action map is 46-wide (`riichiformer/schemas/action_v3.py`):

| index | meaning | phase(s) |
| --- | --- | --- |
| 0..33 | discard tile kind (1m..9m, 1p..9p, 1s..9s, E,S,W,N,P,F,C) | SELF_TURN; KAN_SELECTION |
| 34..36 | discard red five 5mr/5pr/5sr | SELF_TURN; KAN_SELECTION |
| 37 | riichi | SELF_TURN |
| 38/39/40 | chi when called tile is low/middle/high of the run | CALL_RESPONSE |
| 41 | pon | CALL_RESPONSE |
| 42 | kan declaration (concrete ankan/kakan/daiminkan chosen explicitly) | SELF_TURN, CALL_RESPONSE |
| 43 | agari (tsumo in SELF_TURN, ron in CALL_RESPONSE) | both |
| 44 | ryukyoku (kyuushu kyuuhai) | SELF_TURN |
| 45 | pass | CALL_RESPONSE |

`KAN_SELECTION` reuses **0..36 as the tile choice** for one kan candidate.  A
declaration (42) with several candidates is ambiguous unless the caller supplies
`ActionContext.kan_choice`; the agent asks the policy's optional
`select_kan(options)` hook and therefore makes the second stage explicit.  The
agent also validates the decoded event by mapping it back to the action index and
comparing (`mjai_to_action(event, phase) == action`).

`ActionContext` is the complete, testable context: absolute POV seat, phase,
native 46-mask, own hand (normal/aka identities), last discard + absolute actor,
own last tsumo, chi options, kan options, agari/ryukyoku availability.  `action_to_mjai`
fails closed on:

- action outside 0..45, bool/non-int, or a masked action;
- an action not in the phase's legal family (`PHASE_LEGAL_ACTIONS`);
- a discard/chi/pon tile that is not physically in the hand;
- a kan with no candidate, or with several candidates and no explicit choice;
- a KAN_SELECTION index that is not one of the supplied kan candidates;
- agari/ryukyoku without the corresponding mask/facts.

It never returns `None` and never returns a masked action.  Normal fives and aka
fives are distinct (`4` vs `34`; `13` vs `35`; `22` vs `36`).

`context_from_decision(decision, seat)` builds this context from a PR-10
`DecisionInput`: the visible prefix supplies the last discard, own tsumo and
public melds; the native legal mask supplies legality; relative `actor_rel` values
are converted to absolute MJAI actors.  Kan candidates are restricted by phase
(ankan/kakan on the own turn, daiminkan only in CALL_RESPONSE).

## 2. Agent (`runtime/agent.py`)

```python
agent = RiichiFormerAgent(
    seat,                      # 0..3, absolute
    policy,                    # callable(decision: DecisionInput) -> int
    encoder=encode_decision,   # optional PR-11 encoder
    stream_meta=meta,          # required when encoder is enabled
    max_events=4096,
)
agent.reset_game()
agent.reset_kyoku()
event = agent.act(decision)              # mock/direct path
agent.set_scene(seat, log, decision)     # Mortal-compatible scene path
reaction_json = agent.get_reaction()     # compact JSON
```

Before every action the agent checks: exact visible prefix
(`before_event_idx == len(events)`), 46-wide non-empty mask, valid phase, no
opponent concealed tile in the prefix (opponent `TsumoEvent` must be UNKNOWN;
opponent `AnkanEvent` must be fully UNKNOWN), the policy output, the native mask,
and the decoded event round-trip.  Failure modes are explicit exceptions:

| exception | trigger |
| --- | --- |
| `EmptyPolicyOutput` | policy returns `None`/empty; empty mask |
| `IllegalAgentAction` | non-int/out-of-range/masked action, ambiguous kan, non-reversible event |
| `TruncatedStream` | encoder rejects the prefix (e.g. `max_events`) or is configured without metadata |
| `VisibilityError` | opponent concealed tsumo/ankan tile in the POV prefix |
| `AgentError` | no scene, wrong phase/mask width, policy raised |

`MockPolicy(seed)` samples uniformly from the legal actions with a deterministic
xorshift32 stream and implements `select_kan`.  The agent stores no target, label,
rank, oracle, raw log or hidden opponent data; per-game state is cleared by
`reset_game()` and per-kyoku state by `reset_kyoku()`.  No Local KV cache exists in
this mock agent (the full hanchan context remains PR-14's responsibility).

## 3. Mock arena CLI (`scripts/rf_eval.py`)

```bash
.venv/bin/python scripts/rf_eval.py \
  --hands 1000 --seed 10000 --key 987654321 \
  --out /tmp/rf16_smoke/1000
```

`--out` is mandatory; `runs/1v3`, `runs/variance`, the `runs` root, `datasets`,
`checkpoints` and `vendor` are refused (resolved paths, so symlinks are caught).
`--hands` accepts 10/100/1000 for staged smoke; `--seed`/`--key` are explicit.
`--decisions-per-seat` defaults to 2 (one SELF_TURN discard + one CALL_RESPONSE
pass); `3` adds a KAN_SELECTION on every third hand.  `--no-encoder` exists
only for debugging; the normal path runs the PR-11 encoder on every decision.

The loop rotates the starting seat by hand, builds a seeded synthetic 14-tile hand
and schema-valid `DecisionInput` per seat, runs the agent, validates the decoded
event, and writes `summary.json` + `decisions.jsonl`.  It reports:

- `illegal` — masked/out-of-phase/non-reversible actions;
- `none` — `None`/empty policy outputs (never silently accepted);
- `truncated` — encoder rejections;
- `failures` / `ambiguous` / `visibility` — the remaining explicit failures;
- seat-start and seat-decision counts, phase counts, a synthetic rank histogram,
  and a SHA-256 digest of the ordered decision lines (deterministic replay).

Smoke results (standard CLI, encoder on, `--seed 10000 --key 987654321`):

| hands | decisions | illegal | None | truncation | failures | digest (first 12) |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 80 | 0 | 0 | 0 | 0 | `e2bf6b54bab6` |
| 100 | 800 | 0 | 0 | 0 | 0 | `d61a597baec1` |
| 1000 | 8000 | 0 | 0 | 0 | 0 | `99713fdb44ed` |

All three exited 0.  The 10-hand started-seat counts are 3/3/2/2 and the 100/1000
counts are 25/25/25/25 and 250/250/250/250.  With `--decisions-per-seat 3` the
1000-hand smoke exercises 9,336 decisions with 0 failures (all three phases from
the test's kan scenes included).  These are plumbing counts, not win rates.

## 4. Tests (`tests/rf_test_arena.py`)

17 tests / 17 subtests:

- every action 0..36 in SELF_TURN and KAN_SELECTION, plus 37..45 special actions,
  with reverse mapping; aka/normal five distinction;
- chi low/mid/high concrete consumed tiles, pon, ankan/kakan/daiminkan;
- illegal mask, out-of-range, bool/non-int, phase mismatch, empty mask, tile not
  in hand;
- multi-kan ambiguity and explicit `kan_choice` / KAN_SELECTION routing, including
  the agent's `select_kan` path;
- agent legal-mask/no-None/truncation/no-opponent-tsumo-leak guards, scene
  interface, reset semantics, deterministic mock policy;
- 10/100/1000 mock smoke (encoder on), rank-histogram completeness, R02/R25/R26
  note, all four starting seats, output files;
- deterministic replay with the same seed/key and a different key;
- output guard for `runs/1v3`, `runs/variance`, `datasets`, `checkpoints`,
  `vendor`, `runs`, empty path; CLI exit 0 and refused frozen path.

## 5. NOT implemented / next steps

- No real 1v3/libriichi bridge and no real checkpoint policy; the
  `set_scene`/`get_reaction` interface is ready for it but PR-14 must supply the
  model and the arena harness must call it.
- No real rank distribution or Elo: the mock ranks are synthetic.
- No multi-kyoku hanchan state / Local KV; PR-14 owns per-hand context.
- No real dataset, frozen pool, network, training or evaluation write.
- The PR-10 `tiles_left` correction is not a dependency of this PR; the agent
  consumes `DecisionInput` as provided.
