#!/usr/bin/env bash
# Wait for the full-year 2009 shard, then launch the preregistered BC run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SHARD="artifacts/riichiformer/h02/year2009_shards/train.rfshard"
MANIFEST="${SHARD}.manifest.json"
VAL="artifacts/riichiformer/h02/pilot_shards/val.rfshard"
POLICY="configs/riichiformer/local_research_policy.yaml"
OUT="runs/vnext_bc_2009"
LOG="runs/vnext_bc_2009.log"

while [ ! -f "$MANIFEST" ]; do
    echo "[wait] $(date -Is) waiting for $MANIFEST"
    sleep 60
done

DECISIONS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision_count"])' "$MANIFEST")"
STEPS="$((DECISIONS / 64))"
echo "[launch] $(date -Is) decisions=$DECISIONS steps=$STEPS" | tee -a "$LOG"

PYTHON="${PYTHON:-/home/ygsldr/riichi-ai/.venv/bin/python}"
export PYTHONPATH="$ROOT/vendor/mortal/target/release:$ROOT/vendor/mortal:$ROOT"
exec "$PYTHON" scripts/rf_train_vnext.py \
    --train-shard "$SHARD" \
    --val-shard "$VAL" \
    --policy-config "$POLICY" \
    --out "$OUT" \
    --device cuda \
    --steps "$STEPS" \
    --batch-size 64 \
    --rank-base 1 \
    --warmup 1000 \
    --min-lr-ratio 0.1 \
    --save-every 5000 \
    --val-every 5000 \
    --val-batches 50 \
    --max-hours 5 2>&1 | tee -a "$LOG"
