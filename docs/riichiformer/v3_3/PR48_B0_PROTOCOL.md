# PR-48 — frozen B0 training protocol (code + CPU smoke only)

Date: 2026-11-21 (Asia/Shanghai). Base: `e76022600018e9adc6536ccbb904db8c9dc498b3`
(`rf3.2-w2`), branch `rf/pr48-b0`, worktree `worktrees/v33-c`.

This record covers **protocol code, tests and a bounded CPU smoke**. It is not a
B0 training result, not an arena result and not a strength result. The full
6,191-step B0 epoch, any CUDA run, the 12k arena and any M5D-R0 comparison are
`NOT_RUN` and stay `NOT_RUN` until the user authorizes GPU compute.

## 1. Status

| Item | Status |
|---|---|
| `CODE_PASS` | **PASS** — 40 new tests + 60 related existing tests pass on CPU |
| `CPU_SMOKE_PASS` | **PASS** — 2 seeds × 2 steps, frozen six-layer MVP-S, synthetic shards |
| `B0_FULL_TRAIN` | **NOT_RUN** — needs user GPU authorization, an independent validation shard and a written budget |
| `B0_VALIDATION` | **NOT_RUN** — no independent 100–200 game validation shard exists yet (Dev-B / PR-45) |
| `ARENA_SELECT` / `ARENA_CONFIRM` | **NOT_RUN** — PR-46 / Dev-A dependency |
| `M5D_R0_COMPARISON` | **NOT_RUN** — `checkpoints/mortal.pth` `0a44613c…` is not a B0 training target |
| `STRENGTH` | `NOT_RUN` / `STRENGTH_NOT_RUN` — loss and validation CE are selection signals, never strength |
| `GPU_LOCK_USED` | **no** — CPU only; no CUDA call in this lane |

No loss value in this document is evidence that the model plays better. A lower
validation CE only means the offline imitation objective improved on the fixed
validation shard.

## 2. Files

| File | Role |
|---|---|
| `riichiformer/train/b0.py` | frozen protocol dataclasses, fingerprints, prereg/manifest writers, validation metrics, EMA, accumulation step, candidate save/selection, strict resume, stop monitor |
| `scripts/rf_train_b0.py` | prereg-gated CLI: dry-run, bounded two-seed training, candidate/validation/selection/manifest emission, validate-only, prereg-template emission |
| `tests/rf_test_b0_protocol.py` | 28 CPU tests (fingerprints, schedule, validation isolation, accumulation equivalence, selection, resume refusals, stops, manifest) |
| `tests/rf_test_b0_cli.py` | 12 CPU tests (10-step same-seed identity, seed divergence, strict resume == uninterrupted, guards, dry-run/validate-only, template, two-seed selection) |
| `docs/riichiformer/v3_3/PR48_B0_PROTOCOL.md` | this record |

`scripts/rf_train.py`, `riichiformer/data/loader.py`, `data/shard.py`,
`models/**` and PR-38 loader semantics are **unchanged**. The B0 trainer is a
separate, protocol-frozen wrapper; the legacy CLI keeps its existing defaults.

## 3. Frozen protocol

`B0Protocol` is a frozen dataclass serialised as `rf-b0-prereg/1`. It carries:

* `experiment_id` — path-safe id; any tuning change needs a **new id and prereg**;
* exactly **two distinct model RNG seeds** (independent initial weights/optimizer
  RNG, same data, same step budget);
* train and validation shard absolute paths plus **file SHA-256 and manifest
  SHA-256** for both;
* full model config dict + `model_config_hash` + resolved `parameter_count`;
* policy authentication (`policy_config`, `policy_hash`) for
  `operator_private_research` shards (empty for licensed shards);
* `train` (`B0TrainSettings`): AdamW `lr/weight_decay/betas`, warmup, grad clip,
  batch size (decisions), **gradient accumulation**, BF16 flag, EMA decay,
  `max_steps`, device, and `training_path`;
* `validation_steps` and `candidate_steps` (fixed, strictly increasing);
* one `selection` rule and one `stop_policy`.

Two fingerprints:

```
protocol_fingerprint = sha256(canonical(full prereg minus created_utc/notes))
config_fingerprint   = sha256(canonical({"train": <settings>, "model": <full model config>}))
```

### 3.1 `--multi-decision` is not a default

`B0TrainSettings.training_path` has **no default** and must be
`training_path="multi_decisions"`. Constructing settings with `"prefixes"` (the
legacy per-decision prefix path) raises `B0ProtocolError`; omitting the field is
a `TypeError`. A B0 run can therefore never silently fall back to the prefix
path, and PR-38 cross-path resume rules still apply at checkpoint level.

### 3.2 Data fingerprint invariants

* train and validation shards must be different files with different SHA-256;
* both file hashes and both manifest hashes must match the prereg before a run;
* the shard manifest's `shard_sha256` must agree with the file hash;
* the validation shard is opened with `ShardReader(purpose="audit")` only —
  the training loop never receives it, so no optimizer step can consume
  validation data;
* the model config hash and parameter count are re-verified before any step.

## 4. Exact prereg template

Generate it from real files (hashes and parameter count are computed, never
typed by hand):

```bash
.venv/bin/python scripts/rf_train_b0.py --emit-prereg-template \
  --experiment-id EXP-B0-<unique> \
  --shard      /abs/train.rfshard \
  --validation-shard /abs/validation.rfshard \
  --policy-config configs/riichiformer/local_research_policy.example.yaml \
  --model-config configs/riichiformer/mvp_s_8192.yaml \
  --out        /abs/b0_prereg.json \
  --max-steps 6191 --validation-every 250 --candidate-every 1000 \
  --seed 20261121 --seed 20261122 --device cpu
```

The emitted JSON has exactly these keys.  The block shows the full-B0 shape
(`max_steps=6191`, `device="cuda"`); the executed 2-step CPU smoke used the same
key set with `max_steps=2`, `device="cpu"` and its own paths.  `…`/`<…>` marks
values the command computes or the operator supplies:

```json
{
  "prereg_version": "rf-b0-prereg/1",
  "experiment_id": "EXP-B0-<unique>",
  "seeds": [20261121, 20261122],
  "train_shard": "/abs/train.rfshard",
  "train_shard_sha256": "<sha256 of the shard file>",
  "train_manifest_sha256": "<sha256 of train.rfshard.manifest.json>",
  "validation_shard": "/abs/validation.rfshard",
  "validation_shard_sha256": "<sha256 of the validation shard file>",
  "validation_manifest_sha256": "<sha256 of its manifest>",
  "model_config": {
    "name": "mvp_s_8192", "architecture": "local_causal", "hidden": 256,
    "layers": 6, "heads": 4, "ffn": 704, "state_encoder": "mlp",
    "state_hidden": 512, "policy_head": "linear46", "memory": false,
    "dropout": 0.0, "rope_base": 10000.0, "rmsnorm_eps": 1e-06,
    "max_positions": 8192
  },
  "model_config_hash": "ea0d9285f2ede2f4e2277f4444c823f633f27712d0e3e69fc3a590d9216c4360",
  "parameter_count": 5217582,
  "policy_config": "/abs/local_research_policy.example.yaml",
  "policy_hash": "<policy config hash>",
  "train": {
    "lr": 0.0003, "weight_decay": 0.01, "beta1": 0.9, "beta2": 0.999,
    "warmup_steps": 10, "grad_clip": 1.0, "grad_accumulation": 1,
    "batch_size": 64, "max_steps": 6191, "bf16": false, "ema_decay": 0.999,
    "training_path": "multi_decisions", "device": "cuda"
  },
  "validation_steps": [250, 500, "…", 6000, 6191],
  "candidate_steps": [1000, 2000, "…", 6000, 6191],
  "selection": {
    "weights": "raw",
    "metric": "validation_ce",
    "tie_break": ["other_weights_ce", "earlier_step", "lower_seed",
                  "candidate_id"]
  },
  "stop_policy": {
    "max_wall_seconds": 14400.0, "max_nonfinite_events": 0,
    "max_illegal_events": 0, "max_peak_vram_mb": 15000.0,
    "validation_regression_patience": 2,
    "validation_regression_delta": 0.02, "stop_on_oom": true
  },
  "created_utc": "<UTC timestamp>",
  "notes": ""
}
```

Invariants enforced by `B0Protocol.__post_init__` (a malformed prereg never
starts a run):

1. exactly two distinct non-negative seeds;
2. `validation_steps` and `candidate_steps` are strictly increasing, positive,
   and end at or below `max_steps`;
3. `max_steps ∈ candidate_steps ⊆ validation_steps`;
4. train/validation shard files and hashes differ;
5. `model_config_hash == sha256(canonical(model_config))`, model config parses
   as a valid `MVPConfig`, `parameter_count > 0`;
6. `training_path == "multi_decisions"`;
7. unknown JSON keys are refused (`B0Protocol.from_dict` is exact).

Reference hashes for the frozen model configs:

| config | `max_positions` | parameters | `model_config_hash` |
|---|---:|---:|---|
| `configs/riichiformer/mvp_s.yaml` | 4096 | 5,217,582 | `771b3bc0fea5c70a8699b7c8cd8727adfc338dc2af394c8993b8e4d56f9eddcf` |
| `configs/riichiformer/mvp_s_8192.yaml` | 8192 | 5,217,582 | `ea0d9285f2ede2f4e2277f4444c823f633f27712d0e3e69fc3a590d9216c4360` |

A prereg is frozen by committing the JSON **before** the first result. The
wrapper copies it into `<out>/<experiment_id>/protocol.json` and its SHA-256 into
`manifest.json.prereg.sha256`; changing any field requires a new experiment id.

## 5. Optimizer, accumulation, precision, EMA

* AdamW (`betas=(beta1,beta2)`, `weight_decay`), linear warmup then constant LR,
  grad-norm clipping before the step (PR-14/PR-39 code path).
* Gradient accumulation uses the **window weight sum** as the denominator
  (`window_weight_total`), computed over all microbatches before the first
  forward: every microbatch back-propagates `numerator / window_total`. This is
  numerically equivalent to one large batch, not an average of per-microbatch
  means. An all-censored window performs no backward/optimizer/EMA update.
* BF16 is recorded in the protocol; on CPU the default smoke is fp32. BF16 must
  only be enabled after a finite-loss check (PR-44 rule). `GradScaler` is
  `None` for BF16 on CUDA by the PR-14 design and the protocol records
  `bf16`/`scaler=no` in the human prereg.
* EMA (`ema_decay`, optional) updates after an optimizer step only, over all
  floating tensors; its shadow and `updates` counter are checkpointed and
  strictly restored on resume.
* No training-time early stop on loss: the budget is fixed steps, and the only
  stopping conditions are the frozen stop policy.

## 6. Validation schedule and metrics

Validation runs at the fixed `validation_steps` (not at an adaptive/every-N
schedule) on the independent validation shard:

| metric | definition |
|---|---|
| `ce` | weight-sum-normalized cross-entropy over supervised, legal, finite decisions |
| `top1` / `top3` | expert action in the top-1/top-3 masked logits (legal supervised rows) |
| `legal_rate` | legal supervised / supervised |
| `nonfinite_rate` | decisions with a non-finite legal logit/logprob / decisions |
| `decisions`, `supervised`, `legal`, `nonfinite`, `tokens`, `weight` | accounting |

`evaluate_multi_batches` is forward-only: `model.eval()`, `torch.no_grad()`, no
optimizer argument, and it restores the previous train/eval mode. A unit test
snapshots model tensors, optimizer state and EMA state before/after validation
and asserts byte-equality. Raw and EMA weights are both evaluated from the same
batch iterator; EMA evaluation uses a deep-copied model so the live model is
never mutated.

## 7. Candidate retention and the single selection rule

* Candidates are retained **only** at `candidate_steps`, one checkpoint per
  `(seed, step)`, containing raw model + optimizer + scheduler + EMA + RNG +
  counters + provenance + the step's raw/EMA validation metrics.
* A missing candidate or a missing validation record fails the run; a step that
  is not preregistered can never be saved or selected.
* Exactly one selection rule exists:

```
metric   = validation_ce
weights  = "raw" | "ema"      # frozen before results
key      = (ce[weights], ce[other_weights], step, seed, candidate_id)
winner   = min(key)
```

  The tie-break chain is frozen in `B0_TIE_BREAK_CHAIN`; the selection record
  states which tie-break was used and both raw/EMA CE of the winner. A
  non-finite CE can never win.
* Selection is a pure function over candidate records; it cannot see the
  training loss, wall time, or any held-out arena result. No second selection
  rule, no post-hoc metric choice, no "best of both seeds" retune.

## 8. Strict resume provenance

`resume_checkpoint` is the only resume path:

1. checkpoint file SHA-256 must match its sidecar;
2. frozen schema/action/normalizer hashes, full `TrainingConfig`
   (`expected_config`, including seed/device/accumulation/bf16/EMA) and full
   model config/hash must match;
3. provenance must match `checkpoint_provenance(protocol, metadata)`:
   source/split/canonical hashes, research policy/hash, rights facts,
   `training_path="multi_decisions"`, train shard SHA-256 + manifest SHA-256,
   experiment id, protocol fingerprint and config fingerprint;
4. optimizer/scheduler/EMA/RNG/counters must be present (`strict_resume=True`);
5. the checkpoint seed must equal the requested seed, and `extra.training_path`
   must be `multi_decisions`.

Refused, never relabelled:

| case | behaviour |
|---|---|
| legacy checkpoint without `training_path`/B0 provenance | `CheckpointMismatch`, exit 1 |
| prefix (`training_path="prefixes"`) checkpoint resumed on B0 | `CheckpointMismatch`, exit 1 |
| different `lr`/model config/device/EMA/accumulation | `CheckpointMismatch`, exit 1 |
| different train shard SHA-256/manifest SHA-256 | `CheckpointMismatch`, exit 1 |
| different experiment/protocol/config fingerprint | `CheckpointMismatch`, exit 1 |
| different seed | `CheckpointMismatch`, exit 1 |
| resumed step ≥ requested `--max-steps` | refusal, exit 2 |

A resumed run re-consumes the same deterministic loader windows before step
`k+1`, so validation/data order after resume is the uninterrupted order.

## 9. Stop policy and budgets

The stop policy is part of the prereg; every trigger writes a `B0StopEvent`
(code/detail/step/metrics) into the run record and manifest. Exit code **3**
means "stopped by the frozen protocol".

| Stop condition | Default template | Notes |
|---|---|---|
| non-finite output/loss | 0 events | cumulative; any breach stops |
| illegal supervised action | 0 events | cumulative; validation measures `supervised - legal` |
| wall clock | 14,400 s per seed | per-run, `time.monotonic` |
| peak VRAM | 15,000 MB | CUDA only; ignored on CPU |
| validation regression | >2 consecutive checks above best + 0.02 CE | tracks the selected weights source |
| OOM | stop | recorded; never retried silently |

Budgets are **not** fabricated: the only measured values in this lane are the
CPU smoke numbers in §10. The full-epoch GPU wall time, VRAM and I/O numbers are
unmeasured here; PR-44's ~300 s / 2,000 steps is not a budget until it is
re-measured with the final config on the authorized GPU.

I/O budget of candidate retention (measured file size): **83,635,901 bytes
(~79.8 MiB) per 5,217,582-parameter candidate** (model + AdamW state + EMA +
scheduler + sidecar). Seven candidates per seed (1000…6000 + 6191) × 2 seeds =
**14 candidates ≈ 1.17 GB**. This must be confirmed against the Lead's disk
budget before the full run.

## 10. Executed evidence (CPU only)

Command set (all from `worktrees/v33-c`, `PYTHONPATH=vendor/mortal/target/release`):

```bash
# 1. protocol tests (28 tests)
.venv/bin/python -m pytest -q tests/rf_test_b0_protocol.py
# 28 passed in 12.38s (exit 0)

# 2. wrapper tests (12 tests)
.venv/bin/python -m pytest -q tests/rf_test_b0_cli.py
# 12 passed in 11.39s (exit 0; 6.47 s on an unloaded host)

# 3. related existing suites (no regressions)
.venv/bin/python -m pytest -q tests/rf_test_training.py tests/rf_test_multi_loader.py
# 60 passed in 21.15s total (39 passed 13.61 s; 21 passed 7.54 s), exit 0

# 4. all four files together
.venv/bin/python -m pytest -q tests/rf_test_b0_protocol.py tests/rf_test_b0_cli.py \
  tests/rf_test_training.py tests/rf_test_multi_loader.py
# 100 passed, 1 pre-existing warning in 22.16s (exit 0; wall 25.86 s)
```

Measured properties from the tests:

| Measurement | Value |
|---|---|
| `max_grad_diff` large-batch vs 2-microbatch accumulation | **1.192e-07** |
| `max_param_diff` after one AdamW step, same comparison | **1.486e-06** |
| same-seed 10-step run | model/optimizer/EMA tensors bitwise identical |
| different seed, same data/steps | weights differ |
| strict resume at step 6 → 10 | model/optimizer/scheduler/EMA equal to uninterrupted |
| validation vs optimizer/EMA | no change (snapshotted bytes identical) |

Bounded CPU smoke with the **frozen six-layer MVP-S** (`mvp_s_8192.yaml`,
**5,217,582 parameters**) on tiny synthetic shards (6 train / 4 validation
games, batch 2 decisions, 2 steps, EMA 0.999, validation + candidate at step 2):

```bash
.venv/bin/python scripts/rf_train_b0.py --emit-prereg-template \
  --experiment-id EXP-B0-CPU-SMOKE --shard /tmp/b0_mvps_smoke/train.rfshard \
  --validation-shard /tmp/b0_mvps_smoke/val.rfshard \
  --model-config configs/riichiformer/mvp_s_8192.yaml \
  --out /tmp/b0_mvps_smoke/prereg.json --max-steps 2 --validation-every 2 \
  --candidate-every 2 --seed 11 --seed 22 --device cpu --batch-size 2
# parameters=5217582; exit 0; 1.13 s

.venv/bin/python scripts/rf_train_b0.py --prereg /tmp/b0_mvps_smoke/prereg.json \
  --train --max-steps 2 --allow-full-budget --verbose \
  --out-dir /tmp/b0_mvps_smoke/out
# exit 0; wall 2.65 s; max RSS 959,600 KB (~0.94 GiB)
# seed 11: wall 0.32 s, raw_ce=2.455446, ema_ce=3.070752, candidate 83,635,901 B
# seed 22: wall 0.32 s, raw_ce=1.905810, ema_ce=2.396673, candidate 83,635,901 B
# selection: seed22 step 2 (raw), complete=true
```

These two CE values are **pipeline signals on a synthetic shard**, not model
quality and not strength. They are not comparable to PR-44's pilot loss.

CPU wrapper behaviour measured: `--dry-run` ~1–2 s (verifies hashes/model/auth,
no step), `--validate-only` forward-only, `--emit-prereg-template` ~1.1 s with
the six-layer config. CPU training runs pin one Torch thread: with the tiny
test model, 7 threads were ~60× slower than 1 thread on this host (thread-pool
contention), and a fixed thread count keeps the same-seed/resume comparisons
reproducible. CUDA runs are unaffected.

## 11. Code-only vs run

| Deliverable | State |
|---|---|
| protocol dataclasses, fingerprints, prereg/manifest writers | **CODE** (tested) |
| explicit multi-decision requirement, two seeds, schedule | **CODE** (tested) |
| gradient accumulation / EMA / validation isolation | **CODE** (tested) |
| candidate retention, one selection rule, tie-break | **CODE** (tested) |
| strict cross-path resume + provenance refusals | **CODE** (tested) |
| stop policy (non-finite/illegal/wall/VRAM/regression/OOM) | **CODE** (tested) |
| bounded CPU smoke with frozen MVP-S | **RUN** (2 seeds × 2 steps, synthetic data) |
| same-seed 10-step / resume equality | **RUN in tests** (tiny model) |
| real B0 validation shard (100–200 games, disjoint seed/game range) | **NOT_RUN** (PR-45 dependency) |
| full 6,191-step B0 epoch (two seeds) | **NOT_RUN** (no GPU authorization) |
| B0 candidate evaluate → arena select | **NOT_RUN** (PR-46 dependency) |
| arena confirm / M5D-R0 ΔPT / strength | **NOT_RUN** |

## 12. How the full B0 would be executed (NOT RUN)

```bash
# Preconditions: user GPU authorization, frozen prereg committed, real
# independent validation shard, disk budget confirmed, single GPU lock held.
.venv/bin/python scripts/rf_train_b0.py \
  --prereg /abs/b0_prereg.json --train \
  --max-steps 6191 --allow-full-budget --allow-cuda \
  --out-dir /abs/b0_run
```

This command has **not** been run. The wrapper refuses it today because
`--max-steps 6191` equals the prereg budget and needs `--allow-full-budget`;
without a user GPU authorization it must not be passed. There is also no
independent validation shard in this worktree, so a real B0 prereg cannot be
frozen yet.

## 13. P0 / P1 / P2

**P0:** none found. No P0-class correctness issue is claimed fixed here; no
GPU/real-data claim is made.

**P1:**

1. **No independent B0 validation shard yet.** The protocol requires a separate
   validation shard with its own SHA-256; PR-44's 500-game shard has no disjoint
   validation split. Full B0 is blocked on Dev-B/PR-45 (or a Lead-approved
   split) plus a frozen prereg. Do not reuse PR-44 training games for
   validation.
2. **Validation cost not benchmarked at B0 scale.** Each validation step runs
   two forward passes (raw + EMA) over the whole validation shard. The number of
   scheduled checks and the validation shard size determine the real cost; only
   the 4-game synthetic shard was measured on CPU. No GPU estimate is asserted.
3. **Retention I/O must be budgeted.** ~1.17 GB for 14 candidates at the default
   1000-step retention. The Lead must confirm disk before the full run.
4. **PR-44 pilot checkpoints are not B0 candidates.** `f3879cb8…`/`3aa54cbe…`
   are 2,000-step pilot artifacts without validation metrics; the protocol
   cannot select them. The stopped 6,191-step job has no checkpoint and is not
   B0.

**P2:**

1. Selection uses validation CE only; there is deliberately no arena metric in
   B0 selection (arena is a separate frozen gate).
2. The validation-regression monitor tracks the frozen selection weights source;
   the other source's curve is recorded but not monitored.
3. `B0StopMonitor` state is per-run and is not restored on resume (a stopped run
   is terminal by design; resume is only for an explicitly saved candidate).
4. Manifests are canonical JSON but not signed; the local-operator threat model
   is trusted.
5. The `candidate_id` tie-break is unreachable inside one experiment because the
   id already encodes seed and step; it is kept for deterministic completeness.
6. A hard refusal/failure aborts before `manifest.json` is written; already
   written candidate files may remain in the run directory.  A stopped run (exit
   3) does write the manifest with the stop events.

## 14. Next dependency

1. Dev-B / PR-45: independent 100–200 game validation shard with disjoint
   game/seed hashes and its manifest.
2. Lead: confirm the disk/IO budget and authorize a bounded CUDA smoke
   (2 steps, default MVP-S) before the full epoch.
3. User: authorize the full B0 GPU budget and wall/VRAM limits; only then freeze
   the real prereg and run `--max-steps 6191 --allow-full-budget --allow-cuda`.
4. Dev-A / PR-46: arena select/confirm and M5D-R0 ΔPT after a B0 winner exists.

Until those happen: **full B0 train `NOT_RUN`; arena `NOT_RUN`; strength
`NOT_RUN`.**
