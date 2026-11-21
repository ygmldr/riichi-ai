# V3.3 independent review checklist

Reviewer is read-only and must not edit lane code. Every verdict needs a file:line
or a command log. A green check without a real execution is `NOT_RUN`, not `PASS`.

## A. PR-46 M5D / arena
- [ ] Anchor file is the user-confirmed `checkpoints/mortal.pth`; SHA, steps, config,
      `use_event_branch=false`, and 4-seat identical-weight loading are checked.
- [ ] Registry contains all required fields and does not call local research a licence.
- [ ] Native `OneVsThree.py_vs_py` really runs (not the old mock arena); champion is
      three identical M5D-R0 seats; four rotations per wall seed are present.
- [ ] Candidate observation is POV-only; no opponent concealed hand/future event reaches
      the model. A hidden-tile test is present and fails closed.
- [ ] PT profiles: primary 5D `[75,30,0,-105]`, secondary 7D `[90,45,0,-135]`;
      `delta = 4/3 * pt5_C`; zero-sum; tie rule; raw PT also reported.
- [ ] CI is a cluster bootstrap over complete seed sets, not games; incomplete sets fail.
- [ ] Negative controls: all-first/all-last, tie, missing game, duplicate seed, wrong
      key, score sum, self-control near zero, frozen-output refusal.
- [ ] Smoke is 4–20 sets CPU, with real wall time/games/illegal counts; no `runs/1v3` write.

## B. PR-45/47 validation
- [ ] Validation corpus is 100–200 real native games or a documented smaller real corpus;
      500 B0 train games and historical pools are excluded.
- [ ] `game_id`/seed/generator hashes disjoint; intersection report is a real command.
- [ ] Shard purpose is diagnostic/audit; the train loader cannot optimize it.
- [ ] CE/top1/top3/illegal/non-finite/phase/truncation metrics are computed on valid
      supervision only; no strength claim from CE.
- [ ] E2E: selfplay → four POV → decisions → shard → multi loader → 2-step CPU train →
      checkpoint → reload/inference; every stage has a hash/log.
- [ ] Negative controls: future event/tampered prefix/label injection/hidden tile fails.
- [ ] No GPU, no full corpus, no fake 200-game claim if only 40 ran.

## C. PR-48 B0 protocol
- [ ] `--multi-decision` is explicit; legacy prefix cannot become the default silently.
- [ ] Two seeds/config fingerprints/data hashes/budget; one frozen selection rule.
- [ ] Validation does not touch optimizer/scheduler/EMA/RNG; val CE is the only selector.
- [ ] Strict resume equals uninterrupted on a small CPU fixture; cross-path resume refused.
- [ ] No full 6,191-step epoch, no 12k arena, no candidate promotion/strength claim.
- [ ] Checkpoint candidates, EMA and retention rules are documented and tested.

## D. PR-40R R29
- [ ] Rebuild is from a real clean `0cff2b5` clone, not the dirty parent; patch order
      `0014` then `0015`, build sha recorded.
- [ ] R02/R25/R26 native fixtures still pass after rebuild.
- [ ] R29 has an executable fixture that actually observes the ippatsu window, or is
      explicitly `UNVERIFIED`/`KNOWN_DIFFERENCE`; no source-only green.
- [ ] No submodule pointer change; root vendor untouched.

## E. Cross-cutting
- [ ] Tests are not tautologies and do not weaken a fail-closed check to pass.
- [ ] No test writes to `runs/1v3`, `runs/variance`, `checkpoints/`, `datasets/`, root vendor.
- [ ] No GPU process was started; `nvidia-smi` before/after is recorded.
- [ ] Claims distinguish `CODE_PASS`, `SMOKE`, `PILOT`, `STRENGTH_NOT_RUN`.
- [ ] Remote `a27df099` was not merged over `e760226`; no push of unfiltered history.
