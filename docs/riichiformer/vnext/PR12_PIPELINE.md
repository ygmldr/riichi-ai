# VNext Real-Data Pipeline (operator-authorized local research)

This is the exact pipeline used for the 1,000-game pilot and the 2009 full-year
build.  It does not rewrite `source_rights.csv`; the local policy is an operator
routing decision, not an upstream licence.

## 1. Audit

```bash
PYTHONPATH="$PWD" /home/ygsldr/riichi-ai/.venv/bin/python \
  scripts/rf_data_audit.py \
  --roots /home/ygsldr/riichi-ai/datasets/extracted/2009 \
  --out artifacts/riichiformer/audit_2009 \
  --report artifacts/riichiformer/audit_2009/report.md \
  --sample-per-year 2000 --hash-mode sample
```

## 2. Canonical index

```bash
PYTHONPATH="$PWD" /home/ygsldr/riichi-ai/.venv/bin/python \
  scripts/rf_build_canonical.py \
  --audit-parquet artifacts/riichiformer/audit_2009/source_audit.parquet \
  --all --out artifacts/riichiformer/h02/year2009_canonical
```

The pilot used `--sample 4000` instead of `--all`.

## 3. Split v2

```bash
PYTHONPATH="$PWD" /home/ygsldr/riichi-ai/.venv/bin/python \
  -m riichiformer.data.split build-v2 \
  --canonical-db artifacts/riichiformer/h02/year2009_canonical/canonical_index.sqlite \
  --canonical-manifest artifacts/riichiformer/h02/year2009_canonical/manifest.json \
  --out artifacts/riichiformer/h02/year2009_split_v2 --protocol L --no-diff
```

For a val-only shard, filter the v2 manifest to `split == "val_holdout"` and build
with `--purpose diagnostic`.

## 4. Shards

```bash
export PYTHONPATH="$PWD/vendor/mortal/target/release:$PWD/vendor/mortal:$PWD"
/home/ygsldr/riichi-ai/.venv/bin/python scripts/rf_build_shards.py \
  --canonical-dir artifacts/riichiformer/h02/year2009_canonical \
  --split-manifest artifacts/riichiformer/h02/year2009_split_v2/split_manifest_L_v2.parquet \
  --protocol L --out artifacts/riichiformer/h02/year2009_shards/train.rfshard \
  --purpose train --limit 6897 \
  --research-policy operator_private_research \
  --policy-config configs/riichiformer/local_research_policy.yaml --write
```

`libriichi` must be importable or every game is skipped as `ReplayUnavailable`.
The builder prints `PR12_PROGRESS` every 100 games and `PR12_DONE` with decision
count/shard SHA.

## 5. Train / validate

```bash
/home/ygsldr/riichi-ai/.venv/bin/python scripts/rf_train_vnext.py \
  --train-shard artifacts/riichiformer/h02/year2009_shards/train.rfshard \
  --val-shard artifacts/riichiformer/h02/pilot_shards/val.rfshard \
  --policy-config configs/riichiformer/local_research_policy.yaml \
  --out runs/vnext_bc_2009 --device cuda --steps 71100 --batch-size 64 \
  --rank-base 1 --warmup 1000 --save-every 5000 --val-every 5000 --val-batches 50
```

`--validate-only --checkpoint ...` evaluates without training.

## 6. Census

```bash
/home/ygsldr/riichi-ai/.venv/bin/python scripts/rf_vnext_census.py \
  --shard <shard> --policy-config configs/riichiformer/local_research_policy.yaml
```

The 2.51M-match corpus is not built as one shard: measured build/training rates
make one epoch ~47 days.  The 2009 full-year run is the current bounded scope.
