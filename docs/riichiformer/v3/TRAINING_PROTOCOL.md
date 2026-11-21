# TRAINING_PROTOCOL — PR-14 BC Trainer + checkpoint protocol

**Status**: PR-14 synthetic-only implementation. No real Tenhou/log training is allowed
until H-03 marks a source `train_allowed=ALLOWED`; the current H-02 corpus is blocked.
- **Task**: PR-14 — masked BC trainer, optimizer setup, checkpoint protocol
- **Worktree/branch**: `/home/ygsldr/riichi-ai/worktrees/trainer` @ `rf/pr14-trainer`
- **Base**: `af7cfbea6aa578c67bf62324485b2bbf830fb4d7` (PR-12 shard/loader + PR-13 model + fixed PR-10/11)
- **Owned files**: `riichiformer/train/bc.py`, `riichiformer/train/optimizer.py`, `riichiformer/train/checkpoint.py`, `scripts/rf_train.py`, `tests/rf_test_training.py`, this doc, `docs/riichiformer/v3/logs/PR14_pytest.log`
- **Tests**: 11 passed, exit 0 (CPU, synthetic shard only).

## 1. Loss

For a `ShardBatch` (one `ModelBatch` over N decisions) and plain target dictionaries:

```text
valid_i = supervision_valid_i AND expert_action_i is not None
          AND NOT forced_i AND loss_weight_i > 0
weighted_i = loss_weight_i * (-logprobs_i[expert_action_i])
loss = sum(valid_i * weighted_i) / sum(valid_i * loss_weight_i)
```

- Censored (`supervision_valid=False`) and `expert_action=None` contribute **nothing**;
  they remain in the batch as context only.
- `forced=True` / `loss_weight=0` actions may be context but are excluded from policy loss.
- Every supervised action is checked against the 46-wide legal mask; an outside-mask
  label raises `TrainingError` before backward.
- `logits` and legal log-probs are checked finite; a NaN/Inf loss raises.
- If no valid decision exists, the result has `valid_decisions=0`, a finite zero loss, and
  the trainer performs no optimizer step.
- Multiple decisions per game share the shard/loader batch; the full prefix is not
  re-encoded per decision by the trainer.

## 2. Optimizer / determinism

`TrainingConfig` (defaults): seed, AdamW `lr=3e-4`, weight decay 0.01, betas
(0.9,0.999), linear warmup then constant scheduler, grad clip 1.0, grad accumulation
(default 1), optional BF16 autocast, optional EMA decay, device cpu/cuda.

`set_seed` seeds Python, NumPy, Torch (and CUDA if present) and requests deterministic
algorithms. `training_step` supports `accumulation_step/accumulation_count`: gradients
are zeroed at the start of an accumulation window, clipped/stepped only at the boundary.
BF16 is implemented with `torch.autocast`; the PR-13 model already uses SDPA attention.
No GradScaler is needed for BF16; the checkpoint protocol still carries a scaler slot.

## 3. Checkpoint protocol

`save_checkpoint(path, model=..., optimizer=..., scheduler=..., scaler=..., ema=...,
counters={"step","tokens","decisions"}, hashes=..., config=..., seed=...)` writes:

- model/optimizer/scheduler/scaler/EMA state dicts;
- Python/NumPy/Torch/CUDA RNG states;
- step/token/decision counters;
- schema/action/normalizer (+ optional split/source) hashes;
- config dict + `config_hash` + seed;
- format `rf-checkpoint/1`, atomic `.tmp` replace, sidecar JSON with file SHA-256.

`load_checkpoint` verifies the file SHA-256, format, required hashes, strict model
`load_state_dict`, and optional expected hashes; it restores optimizer/scheduler/scaler/
EMA/RNG and returns counters/config. A hash or model mismatch raises `CheckpointMismatch`;
nothing is silently migrated. `created_utc`-style time is not part of checkpoint identity.

## 4. CLI / rights gate

```bash
# dry-run (default): validates rights and counts batches, no model step/write
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  .venv/bin/python scripts/rf_train.py --shard /tmp/synthetic.rfshard --dry-run

# synthetic training smoke (requires an ALLOWED purpose=train shard)
... --shard /tmp/synthetic.rfshard --train --out /tmp/ckpt.pt --max-steps 10 \
    --batch-size 4 --seed 20261121 --cpu

# resume
... --shard /tmp/synthetic.rfshard --train --out /tmp/ckpt.pt --resume /tmp/ckpt.pt
```

Before constructing a model, the CLI opens the shard with
`ShardReader(purpose="train")`, which refuses `purpose=audit|diagnostic` shards and any
`train_allowed=False`/non-ALLOWED source. This is the H-03 source-rights gate equivalent
(the PR-12 metadata is populated from the H-03 registry). Output under `datasets/`,
`runs/1v3`, `runs/variance`, or `checkpoints/` is refused. Default is dry-run; `--train`
is required to write a checkpoint.

## 5. Synthetic smoke evidence

CPU, tiny MVP-S (`layers=1`, hidden 256, ffn 32, state hidden 32, 464,142 parameters),
synthetic ALLOWED shard built from four-POV games:

- `PR14_LOSS first=2.379641 last=0.000445 params=464142` over 12 steps on one batch.
- Two same-seed short runs: losses and state dicts exactly equal.
- Resume from step-2 checkpoint: next-step weight exactly equal to uninterrupted run;
  RNG/optimizer/scheduler restored.
- Censored/forced targets excluded; label outside legal mask raises; NaN log-prob raises;
  all-censored batch performs no optimizer step.
- Blocked/audit shard refused by `ShardReader(purpose="train")` and CLI exit 2 before any
  model construction; frozen output path refused exit 2.
- Checkpoint roundtrip equal; wrong schema hash raises `CheckpointMismatch`.

Repro:

```bash
cd /home/ygsldr/riichi-ai/worktrees/trainer
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q -s tests/rf_test_training.py
```

## 6. What is NOT trained / gaps

- **No real Tenhou/log training**: H-03 currently marks the H-02 sources
  `train_allowed=RESTRICTED/UNKNOWN`; only synthetic ALLOWED shards are used.
- No real corpus, no `runs/`, `checkpoints/`, or `datasets/` writes; no checkpoint is
  promoted as a model artifact.
- CPU-only smoke; no GPU/BF16 peak-VRAM or throughput claim.
- Grad accumulation is supported by `training_step`; the CLI currently uses the default
  accumulation of 1 for the synthetic smoke.
- EMA is saved/loaded by the checkpoint API but not configured in the CLI smoke.
- No distributed training, mixed-precision GradScaler, or long-run stability claim.
