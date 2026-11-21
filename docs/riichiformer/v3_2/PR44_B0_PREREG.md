# PR-44 B0 pilot — preregistration

Date: 2026-11-21 (Asia/Shanghai). This is a **pilot preregistration**, not a completed
B0 reference model and not a strength result. It exists so the first real training run
has fixed data, config, seeds, budgets and stop rules before any GPU result is seen.

## 1. Why a pilot, not full B0

The V3.2 Wave 1 gate (`PR43_GATE.md`) left two prerequisites open:

1. `PR-38` multi-decision forward is not yet consumed by the loader/trainer; the prefix
   path re-encodes and re-forwards every decision prefix.
2. The B0 corpus has not been fixed.

A measured prefix-path step (batch 64, 6-layer MVP-S, 100-game probe shard, RTX 5070 Ti)
took **~20 s/step** for 64 decisions / ~69k prefix tokens. At that rate a 2,000-step
pilot would take ~11 GPU hours, which is not a sensible use of the single GPU before
the shared-forward path is integrated. The pilot therefore waits for PR-38 loader
integration and then uses `--multi-decision`.

## 2. Fixed corpus

| Item | Value |
|---|---|
| Source split | `artifacts/riichiformer/h02/split_full/split_manifest_L_v2.parquet` |
| Source SHA-256 | `5032d050d52f3c43cc4d054a13891e0bf51e9c11747f1511fad1e2d925c6d1b7` |
| Sampling | SHA-256 of `f"{sample_seed}:{game_id}"`, representative/non-quarantine/train only |
| `sample_seed` | `20261121` |
| `sample_size` | 500 games |
| Sampled manifest | `/tmp/b0_sample500.parquet` |
| Sampled manifest SHA-256 | `f6560a179d24bf05824e12a537bffd399d3b2e9488d4423b09560cb2d5271fa4` |
| Sample composition | 500 `tenhou_old` rows, years 2010–2022 (protocol-L train is old-log-only) |
| Command | `scripts/rf_make_b0_split.py --sample-size 500 --sample-seed 20261121` |

The sample is deterministic and independent of game-id order. It is **not** a random
sample of all Tenhou years: protocol L's train partition contains only `tenhou_old`
(2009–2022); recent logs are `external_diagnostic`/`val_holdout`. Any claim about
2023–2026 data would need protocol R or a different prereg.

The sampled rows keep their original `rights_status=UNKNOWN` / `train_allowed=false`.
Training is under the operator-private research profile; this is not a third-party
licence and the sample may not be published.

## 3. Shard and policy

- Builder: `scripts/rf_build_shards.py` with
  `--research-policy operator_private_research --policy-config
  configs/riichiformer/local_research_policy.example.yaml --limit 500 --write`.
- Format: `rf-shard/3`; expected ~327k decisions; record the exact shard sha256 and
  decision count in the run summary.
- Policy hash: `72ac377f2fa44297c18228aeeba59160de48f67c65bc3b3b242a3e9b889251b0`.
- Required provenance in the checkpoint: source/split/canonical hashes,
  `operator_private_research`, policy hash, `external_license_status=UNKNOWN`.

## 4. Model / optimizer / budget

| Item | Value |
|---|---|
| Model | `configs/riichiformer/mvp_s.yaml`, 6 layers, hidden 256, FFN 704, state 512 |
| Parameters | 5,217,582 (must be printed/asserted) |
| Optimizer | AdamW, lr `3e-4`, weight decay `0.01`, betas `(0.9, 0.999)`, grad clip `1.0` |
| Batch | 64 decisions (multi-decision path), no accumulation for the pilot |
| Seeds | `20261121`, `20261122` |
| Steps per seed | 2,000 (pilot), fixed; no early stop on loss |
| Precision | fp32 first; BF16 only after a finite-loss check |
| Device | single RTX 5070 Ti, exclusive; no concurrent eval |

Stop immediately on NaN/Inf, a non-finite loss, a policy/provenance mismatch, or a
data batch that violates the cutoff/POV checks. A crash does not get resumed silently;
resume only via the strict `--resume` path.

## 5. Metrics (diagnostic, not strength)

- training BC loss per step, tokens/s, decisions/s, wall time, peak VRAM;
- model config hash, parameter count, checkpoint sha256 per seed;
- final train loss over a fixed held-out subset of the same shard (diagnostic only;
  protocol-L val_holdout is not mixed into train);
- legal-action rate is structural (masked loss); no arena result in this pilot.

No win rate, no dan/rank estimate, no Mortal comparison, no “九段/天凤位” claim. A
loss decrease on 500 old-log games is a pipeline signal, not evidence of strength.

## 6. Stop / promotion rule

The pilot is promoted to a **candidate B0** only if:

1. both seeds complete the fixed budget without NaN/Inf or provenance mismatch;
2. the multi-decision vs prefix parity tests pass on the same model and the CUDA
   parity run (flash-SDP disabled or documented tolerance) passes;
3. the shard/corpus/policy hashes are recorded and reproducible;
4. an independent reviewer accepts the code and run bookkeeping.

Even then it is an internal B0 reference, not a Tenhou rank. PR-45 diagnostics and
PR-46 duplicate arena are separate gates.

## 7. Current status

- Sample manifest: **created** (`f6560a17…`).
- 500-game shard: **building** (background job; exact sha/decisions to be filled in).
- Multi-decision loader/trainer: **in progress** on `rf/v32-b0`.
- B0 pilot training: **NOT_RUN** until the multi path and a GPU lock exist.
- Full B0 / strength: **NOT RUN**.
