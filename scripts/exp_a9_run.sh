#!/usr/bin/env bash
# Launch A9 training (detached by the caller) with the 20k-step watchdog.
#   bash scripts/exp_a9_run.sh real
#   bash scripts/exp_a9_run.sh smoke
# Real mode refuses to start if runs/exp/A9/state.pth exists (no resume: A9 is a
# from-scratch arm) and starts scripts/exp_a9_watchdog.py for the declared budget.
set -euo pipefail
MODE="${1:-real}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/activate_env.sh" >/dev/null
cd "$REPO_ROOT/vendor/mortal/mortal"

case "$MODE" in
    real)
        CFG="$REPO_ROOT/runs/exp/A9/train.toml"
        STATE="$REPO_ROOT/runs/exp/A9/state.pth"
        PIDFILE="$REPO_ROOT/runs/exp/A9/train.pid"
        WLOG="$REPO_ROOT/runs/exp/A9/watchdog.log"
        if [ -e "$STATE" ]; then
            echo "REFUSING: $STATE already exists (A9 is from-scratch, no resume)" >&2
            exit 1
        fi
        if [ ! -f "$CFG" ]; then
            echo "missing $CFG (run scripts/exp_a9_make_configs.py first)" >&2
            exit 1
        fi
        echo "$$" > "$PIDFILE"
        python3 "$REPO_ROOT/scripts/exp_a9_watchdog.py" \
            --log "$REPO_ROOT/runs/exp/A9/train.log" \
            --state "$STATE" --steps 20000 --pid "$$" --out "$WLOG" &
        ;;
    smoke)
        CFG="/tmp/opencode/a9smoke/smoke.toml"
        PIDFILE="/tmp/opencode/a9smoke/train.pid"
        if [ ! -f "$CFG" ]; then
            echo "missing $CFG (run scripts/exp_a9_make_configs.py first)" >&2
            exit 1
        fi
        echo "$$" > "$PIDFILE"
        ;;
    *)
        echo "unknown mode $MODE" >&2
        exit 2
        ;;
esac

export MORTAL_CFG="$CFG"
echo "[A9 run] mode=$MODE pid=$$ cfg=$CFG"
exec python3 "$REPO_ROOT/scripts/exp_a9_train.py"
