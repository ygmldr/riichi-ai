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
GAMES="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["game_count"])' "$MANIFEST")"
if [ "$GAMES" -ne 6897 ]; then
    echo "[abort] $(date -Is) expected 6897 games, manifest has $GAMES" | tee -a "$LOG"
    exit 1
fi
if [ "$DECISIONS" -le 0 ]; then
    echo "[abort] $(date -Is) manifest has no decisions" | tee -a "$LOG"
    exit 1
fi
STEPS="$((DECISIONS / 64))"
echo "[launch] $(date -Is) games=$GAMES decisions=$DECISIONS steps=$STEPS" | tee -a "$LOG"

PYTHON="${PYTHON:-/home/ygsldr/riichi-ai/.venv/bin/python}"
export PYTHONPATH="$ROOT/vendor/mortal/target/release:$ROOT/vendor/mortal:$ROOT"
if ! "$PYTHON" scripts/rf_vnext_census.py \
        --shard "$SHARD" --policy-config "$POLICY" \
        --out docs/riichiformer/vnext/logs/VNEXT_CENSUS_2009.json 2>&1 | tee -a "$LOG"; then
    echo "[warn] $(date -Is) census failed; continuing to training" | tee -a "$LOG"
fi
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
