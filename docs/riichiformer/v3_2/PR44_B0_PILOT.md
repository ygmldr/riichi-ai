# PR-44 B0 pilot — executed record

Date: 2026-11-21 (Asia/Shanghai). This is the **executed pilot** of
`PR44_B0_PREREG.md`, not a completed B0 reference and not a strength result.

## 1. Base and prerequisites

| Item | Value |
|---|---|
| Integration branch | `rf/v32-integration` |
| Code HEAD at run | `ccbf27e` (after multi-decision loader/trainer merge) |
| Full suite | **707 passed, 2 skipped, 281 subtests** (182.20 s; patched vendor `.so`) |
| Multi path | `--multi-decision`; adaptive `max_events`/`max_positions` |
| Model | six-layer MVP-S, **5,217,582** parameters, `max_positions=4096` |
| Device | RTX 5070 Ti 16 GB, exclusive; CPU loader |

The PR-43 P1 "multi-decision not wired into loader/trainer" is closed by the
`PR38_LOADER_INTEGRATION.md` work: `iter_multi_decision_batches`,
`training_step_multi`, CLI flags, adaptive long-POV chunking, exact prefix and
observable-state checks.

## 2. Corpus and shard

| Item | Value |
|---|---|
| Sample | 500 games, deterministic `sha256("20261121:"+game_id)` |
| Sampled manifest | `/tmp/b0_sample500.parquet` |
| Sample SHA-256 | `f6560a179d24bf05824e12a537bffd399d3b2e9488d4423b09560cb2d5271fa4` |
| Source split manifest | L v2, SHA-256 `5032d050d52f3c43cc4d054a13891e0bf51e9c11747f1511fad1e2d925c6d1b7` |
| Shard | `/tmp/b0_500.rfshard`, `rf-shard/3` |
| Shard SHA-256 | `f474a3f1d78f0cec55f5d54a5089f51b5a47ccf33feff98248243e77b80f3a58` |
| Games / decisions | 500 / **333,224** |
| Max POV stream | 2,142 events (all ≤ 4096; no long-context variant needed) |
| Policy | `operator_private_research`, hash `72ac377f2fa44297c18228aeeba59160de48f67c65bc3b3b242a3e9b889251b0` |
| Rights | `rights_status=UNKNOWN`, `external_license_status=UNKNOWN`, `train_allowed=false` (unchanged facts) |
| Composition | 500 `tenhou_old` rows, years 2010–2022; protocol-L train is old-log-only |

The shard was built with the private policy config and the deterministic sample;
it is a local research artifact, not a redistributable dataset.

## 3. Loader accounting

Full dry-run (CPU, no optimizer):

```bash
python scripts/rf_train.py --shard /tmp/b0_500.rfshard --multi-decision \
  --dry-run --batch-size 64 --steps 0 \
  --policy-config configs/riichiformer/local_research_policy.example.yaml
```

Observed: `batches=6191 decisions=333224 tokens=5437312`, exit 0, wall 231.39 s.
The token count is the joined shared-event + marker sequence summed over rows; the
legacy prefix path would materialise/re-forward a prefix per decision.

## 4. Two-seed pilot

Command template (run twice, changing `--seed`/`--out`):

```bash
python scripts/rf_train.py --shard /tmp/b0_500.rfshard --multi-decision --train \
  --out /tmp/b0_pilot_seed<seed>.pt --device cuda --max-steps 500 \
  --batch-size 64 --seed <seed> \
  --policy-config configs/riichiformer/local_research_policy.example.yaml
```

| Seed | Steps | Decisions | Tokens | Wall | First loss | Last loss | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| 20261121 | 500 | 26,918 | 432,147 | 86.0 s | 1.812336 | 1.239297 | `f21c8a3e2a20f8d4ca1706673e5881f9f2178976a4889512666d013fed55f3a8` |
| 20261122 | 500 | 26,918 | 432,147 | 87.1 s | 1.800469 | 1.213882 | `ac7346d109570a32a6ab7345ede6b3cea36b4a6ddc83005ac852bf651fe5e65b` |

Both runs: 6-layer MVP-S, 5,217,582 params, finite loss every step, checkpoint
model-config hash `771b3bc0…`, policy/provenance recorded. Peak `nvidia-smi`
memory observed 4,649 MiB (includes the idle baseline; the isolated delta was not
measured). This is ~8% of one epoch (6,191 batches), so it is a **pilot**, not a
full B0.

Strict multi-path resume: seed 20261121 checkpoint → step 501, loss `1.102307`,
tokens `432,672`, decisions `26,982`, checkpoint SHA-256
`8b9d2c152c563980e06a0e73535c81ba0927394329cedb2a79ee7dd859ce9dbd`.
Legacy checkpoints without `training_path` remain loadable; cross-path resume is
refused by the integration tests.

## 5. What this does and does not show

Shows:
- the multi-decision loader/trainer works on real H-02 data and the GPU;
- both seeds complete a fixed 500-step budget without NaN/Inf or provenance
  mismatch, with the same token/decision accounting;
- strict resume and checkpoint provenance work on the multi path.

Does **not** show:
- a complete B0 epoch, convergence, validation loss, or model quality;
- any arena strength, dan/rank, Mortal comparison, or “九段/天凤位”;
- permission to publish/redistribute the data or weights. External rights remain
  `UNKNOWN`; the operator-private profile is not a licence.

## 6. Remaining B0 work

1. Run the preregistered 2,000-step budget (or a full epoch) for two seeds, with a
   fixed validation subset and checkpoint selection rule.
2. Add a validation/eval loop; the current CLI only logs training loss.
3. Record the run in a B0 reference manifest (data/model/optimizer/seed hashes,
   step/token/decision counts) and submit to independent review.
4. Keep `STRENGTH_NOT_RUN` until PR-45/46 diagnostics/arena are executed.
