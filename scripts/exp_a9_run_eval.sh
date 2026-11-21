#!/usr/bin/env bash
# A9 pool-A evaluation: 12,000 games, seed [10000,13000), own log_dir, event debug on.
# Launch detached with:
#   (setsid nohup bash scripts/exp_a9_run_eval.sh > runs/exp/A9/eval.log 2>&1 & echo $!)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/activate_env.sh" >/dev/null
cd "$REPO_ROOT/vendor/mortal/mortal"

CFG="$REPO_ROOT/runs/exp/A9/eval.toml"
GAMES="$REPO_ROOT/runs/exp/A9/games"
DEBUG=/tmp/opencode/a9_evt_debug.jsonl

[ -f "$CFG" ] || { echo "missing $CFG" >&2; exit 1; }
[ -f "$REPO_ROOT/runs/exp/A9/state.pth" ] || { echo "missing A9 state.pth" >&2; exit 1; }
if [ -d "$GAMES" ] && [ -n "$(ls -A "$GAMES" 2>/dev/null)" ]; then
    echo "REFUSING: $GAMES is not empty" >&2
    exit 1
fi
mkdir -p "$GAMES"
rm -f "$DEBUG"

export MORTAL_CFG="$CFG"
export MORTAL_EVT_DEBUG="$DEBUG"
echo "[A9 eval] start $(date -Is) cfg=$CFG games_dir=$GAMES debug=$DEBUG"
exec python3 one_vs_three.py
