# V3.2 remote branch — Lead acceptance (training chain)

Date: 2026-11-21 (Asia/Shanghai). Status: **PASS_WITH_P1** for the private-research
training chain. This document records commands actually run on the user's WSL2 machine;
it does not claim model strength, Tenhou parity, or third-party permission.

## Baseline

| Item | Value |
|---|---|
| Remote branch | `origin/rf/v3.2-policy-parallel-plan` |
| Remote SHA | `a27df099e9b395a742ebc3a239e003d0f8896b92` |
| Local dev branch | `rf/v3.2-local` |
| Independent worktree | `worktrees/v3.2-policy` |
| Public filtered base | `riichiformer-v3-filtered` `a4492e4d4e9a90bab9fb66f0b269ba77a97ea712` |
| Python | `/home/ygsldr/riichi-ai/.venv/bin/python` (3.12, torch 2.11.0+cu128) |
| libriichi | `/home/ygsldr/riichi-ai/vendor/mortal/target/release/libriichi.so` |
| Vendor source | pinned `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2` (submodule initialized in the worktree) |
| GPU | RTX 5070 Ti 16 GB, CUDA available |

The worktree was created directly from the remote branch, not by merging or copying the
old V3.2 pack code:

```bash
git fetch origin refs/heads/rf/v3.2-policy-parallel-plan:refs/remotes/origin/rf/v3.2-policy-parallel-plan
git worktree add -b rf/v3.2-local worktrees/v3.2-policy origin/rf/v3.2-policy-parallel-plan
git -C worktrees/v3.2-policy submodule update --init vendor/mortal
```

## Test suite

```bash
cd worktrees/v3.2-policy
PYTHONPATH=/home/ygsldr/riichi-ai/vendor/mortal/target/release \
  /home/ygsldr/riichi-ai/.venv/bin/python -m pytest -q \
  $(git ls-files 'tests/rf_test_*.py' | tr '\n' ' ')
```

Result: **494 passed, 2 skipped, 242 subtests passed** in 145.41 s, exit 0.

The first run (before the submodule was initialized) failed 25 `SourceContractTests`
because `vendor/mortal/libriichi/src/**` was absent from the new worktree. Those were
environment failures, not source failures; after `git submodule update --init` the same
suite passed. The two skips are the pre-existing explicit skips documented in
`tests/rf_test_ruleset_parity.py` (board injection unavailable).

## Private-research shard chain (real H-02 full data)

Canonical: `artifacts/riichiformer/h02/canonical` (2,512,424 rows, 109 shards,
index sha `21d9077f…`). Split: `artifacts/riichiformer/h02/split_full/`.

Dry-run (protocol L):

```bash
.venv/bin/python scripts/rf_build_shards.py \
  --canonical-dir .../h02/canonical \
  --split-manifest .../h02/split_full/split_manifest_L_v2.parquet \
  --purpose train --limit 10 --out /tmp/rf32_private10.rfshard \
  --research-policy operator_private_research --dry-run
```

Observed: `rows=10 sources=tenhou_old rights=UNKNOWN train_allowed=False
research_policy=operator_private_research policy_hash=3e515bd9… canonical=rf-canon-store/1`,
exit 0.

Write: exit 0, `games=10 decisions=8504 written=10 skipped=0`,
shard sha256 `268082d5cff32cfd94f6f23232b6c8f217ab83203f9cb21e15baefcee163eccf`,
12 MB. Sidecar manifest and metadata agree:

- `format_version = rf-shard/3`
- `purpose = train`
- `research_policy = operator_private_research`
- `research_policy_hash = 3e515bd90d9a7739590faca91f27eb9cdbb5e9f89ee35aac983504079d302806`
- `rights_status = UNKNOWN`, `train_allowed = false` (facts preserved)
- schema `12503d36…`, action `6cf1efc5…`, normalizer `37df861b…`
- `ShardReader(path, purpose="train")` opened the shard and yielded a 4-decision
  `ShardBatch` (`ModelBatch` event tensors shape `[4, 19]`, state `[4, 172]`,
  legal mask `[4, 46]`).

Negative control: the same rows with `--research-policy licensed_only` exit **2**
(`training shard selected blocked row … train_allowed=False`). A copy of the shard whose
metadata was rewritten to `licensed_only` is refused by `ShardReader` with
`ShardRightsError: licensed training shards require train_allowed=True and
rights_status=ALLOWED`.

## GPU training and strict resume

Two-step six-layer smoke (the exact handoff command shape):

```bash
.venv/bin/python scripts/rf_train.py --shard /tmp/rf32_private10.rfshard --train \
  --out /tmp/rf32_private10-smoke.pt --device cuda --max-steps 2 --batch-size 4 \
  --layers 6 --ffn 704 --state-hidden 512
```

Observed exit 0, 19.49 s wall, max RSS 2.05 GB:
`step=1 loss=2.193345`, `step=2 loss=1.711688`, `tokens=204 decisions=8`,
checkpoint sha `a41320f08f6fc120814ce9e8e93d9b45e432e623920fb0e1d3f690924bdccf14`.
Loaded model parameter count: **5,217,582** (MVP-S six-layer shape).

Resume from that checkpoint with the same shard, `--max-steps 3`: exit 0,
`step=3 loss=0.325555 tokens=253 decisions=12`, checkpoint sha
`64abc1a6648c9d18249e4e8e73696f2557f70e677fd4b0f8f0d40de5b61d92c4`. Optimizer state and
scheduler step were loaded (`scheduler.last_epoch=3`).

Negative control: a second 10-game shard built from the protocol-R split has a different
`split_manifest_sha256`. Resuming the protocol-L checkpoint against it exits **1** with
`PR14_FAILED: checkpoint provenance mismatch for split_manifest_sha256`; no output
checkpoint is written.

## P1 findings (not blockers for parallel dispatch, but must be fixed)

1. **Private policy hash is not tied to a policy config.** `ShardReader` accepts any
   64-character `research_policy_hash`; it never compares it to a configured profile or
   allowlist. A forged/adjusted shard can claim `operator_private_research` and load.
   Resume provenance only catches a change *after* the first checkpoint.
2. **Builder default is the permissive profile.** `scripts/rf_build_shards.py` defaults
   `--research-policy` to `operator_private_research`. The PR-37 spec requires strict
   `licensed_only` by default and an explicit private opt-in. This is a fail-open default.
3. **Policy hash derivation is builder-local and not independently reproducible.** The
   hash is computed from selected source ids plus a hardcoded operator reference; there
   is no committed policy config, operator attestation artifact, or `--policy-config`.
4. **Stale trainer text.** `scripts/rf_train.py` still says training requires an ALLOWED
   shard and documents only the old gate; `--init-weights-only` does not exist yet.
5. **Aggregate rights status.** Mixed-source shards store `rights_status=MIXED`/
   `UNKNOWN` at metadata level; per-game status is retained, so this is a reporting
   limitation rather than a leak.

No P0 (hidden-information leak, split/quarantine bypass, or third-party permission
rewrite) was observed in this chain.

## Not run / not claimed

- No full-corpus training, no B0 reference model, no checkpoint promotion.
- No arena strength, no Tenhou-equivalent rank, no claim about Mortal or 九段.
- No third-party rights were granted or changed; `source_rights.csv` remains 11 rows,
  0 ALLOWED.
- No upload, publication, or online Tenhou/雀魂 AI play.

Next: PR-38/39/40/41/42 development in independent worktrees from this accepted base;
PR-37 follow-up must close P1 #1–#3 before any claim that private shards are
policy-authenticated.
