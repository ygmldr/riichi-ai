# OFFLINE_DIAGNOSTICS — PR-15 NLL / Top-k and bucket report

**Status**: PR-15 synthetic-only. Real Tenhou diagnostics are **NOT_RUN** because H-03
blocks the current H-02 sources; synthetic ALLOWED shards are used.
- **Task**: PR-15 — offline diagnostics and hard-case distribution
- **Worktree/branch**: `/home/ygsldr/riichi-ai/worktrees/offline` @ `rf/pr15-offline`
- **Base**: `88479f97834687819c9652dd77179bab48b04855` (PR-12 shard/loader, PR-13 model, PR-14 trainer)
- **Owned files**: `riichiformer/eval/offline.py`, `scripts/rf_offline_eval.py`, `tests/rf_test_offline.py`, this doc, `docs/riichiformer/v3/logs/PR15_pytest.log`
- **Tests**: 7 passed, exit 0 (synthetic, CPU).

## 1. Metrics

For supervised decisions only:

```text
NLL   = mean(-log p_model(expert_action))
Top1  = fraction(argmax_legal logprob == expert_action)
Top3  = fraction(expert_action in top 3 legal logprobs)
```

Conditions:
- `supervision_valid=True` and `expert_action is not None`;
- `forced=True` / `loss_weight=0` excluded from the policy metric (context only);
- expert action must be in the 46-wide legal mask (outside-mask is an evaluation error);
- legal logprobs must be finite; NaN/Inf fails closed;
- no per-game cherry-picking: all rows from the selected shard prefix/limit are evaluated,
  and the shard/limit are reported with the result.

95% intervals:
- Top1/Top3: Wilson interval over supervised decisions;
- NLL: normal mean interval (`mean ± 1.96·sd/√n`); single-row intervals collapse to the
  point.

## 2. Buckets

### Mutually exclusive partitions (counts sum to supervised)

| dimension | buckets |
|---|---|
| `phase` | `SELF_TURN`, `CALL_RESPONSE`, `KAN_SELECTION` |
| `action_family` | `discard`, `riichi`, `chi`, `pon`, `kan`, `agari`, `ryukyoku`, `pass` |

The evaluator asserts `sum(phase counts) == sum(action_family counts) == supervised`.
A bucket report contains count, NLL, Top1, Top3, and their 95% intervals.

### Overlapping slices (explicitly **not** a partition)

| slice | rule |
|---|---|
| `riichi` | POV self `RiichiState` is declared/accepted |
| `involved_call` | expert action family is chi/pon/kan |
| `kan` | expert action family is kan or phase is `KAN_SELECTION` |
| `south` | `bakaze == South` and `kyoku >= 3` |
| `score_diff` | self score − 25000 bucket: ≤−10000, −1..−9999, 0, 1..9999, ≥10000 |
| `shanten` | **not available** in PR-12 `DecisionInput`; reported as `unknown` (P1 gap) |

`overlap_is_exclusive=false` is in the JSON. Do not sum these slices; a row may appear in
multiple.

## 3. CLI

```bash
# synthetic shard with an explicit checkpoint (recommended for real metrics)
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  .venv/bin/python scripts/rf_offline_eval.py \
  --shard /tmp/synthetic.rfshard --checkpoint /tmp/ckpt.pt \
  --out /tmp/offline.json --format json --purpose diagnostic

# random model smoke (synthetic only; no strength claim)
... --shard /tmp/synthetic.rfshard --format markdown --out /tmp/offline.md

# blocked audit shard: only with the explicit H-03 diagnostic opt-in
... --shard /tmp/audit.rfshard --allow-blocked-audit --purpose diagnostic
```

- `--purpose diagnostic` is required by this CLI; `--format json|csv|markdown`.
- The CLI opens the shard with `ShardReader(purpose="diagnostic")`; if
  `train_allowed=False`, it refuses unless `--allow-blocked-audit` is explicit. This is
  the H-03 gate; the current real H-02 sources remain blocked, so real-data status is
  **NOT_RUN**.
- Default output is JSON to stdout; `--out` writes atomically. Frozen
  datasets/run/checkpoint roots are write-refused.
- `--limit` bounds rows; the result carries shard/source/purpose/projection metadata.

## 4. Synthetic evidence

`tests/rf_test_offline.py` (7 passed):
- fixed logprobs/labels: known NLL, Top1=1/3, Top3=1.0; censored/forced excluded;
- legal-label check and NaN/Inf fail closed;
- phase and action-family partitions sum to supervised; overlap is explicitly non-summed;
- deterministic repeated evaluation;
- synthetic shard + tiny PR-13 model reader evaluation; JSON CLI output;
- blocked audit shard refused without opt-in, allowed with `--allow-blocked-audit`.

Repro:

```bash
cd /home/ygsldr/riichi-ai/worktrees/offline
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q -s tests/rf_test_offline.py
```

## 5. What this does NOT claim

- **NLL/Top1/Top3 are offline imitation metrics, not arena strength and not Tenhou rank.**
- No real Tenhou/log diagnostic has been run; H-03 rights remain blocked.
- No training, no checkpoint training, no `datasets/`, `runs/`, or `checkpoints/` writes.
- Shanten bucket is `unknown` because PR-12 `DecisionInput` deliberately carries no
  shanten probe; a future offline evaluator may add a native shanten source without
  changing model inputs.
- CIs describe the selected synthetic/sample rows only; they are not a population
  estimate or an arena confidence interval.
