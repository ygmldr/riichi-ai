#!/usr/bin/env bash
# duplicate mahjong 1v3 评估 —— P3 验收
# 用法: bash scripts/run_eval.sh [config.toml]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:-$REPO_ROOT/configs/config.toml}"
# 归一化为绝对路径: 后续 cd vendor/mortal/mortal 后,MORTAL_CFG 按进程 CWD 解析(同 prepare_data_on_gpu.sh 模式)
case "$CFG" in
    /*) ;;
    *) CFG="$(pwd)/$CFG" ;;
esac
LOG_DIR="$REPO_ROOT/runs/1v3"
LOG_FILE="$LOG_DIR/eval_$(date +%Y%m%d_%H%M).log"

source "$REPO_ROOT/scripts/activate_env.sh"
cd "$REPO_ROOT/vendor/mortal/mortal"
export MORTAL_CFG="$CFG"

mkdir -p "$REPO_ROOT/checkpoints"
mkdir -p "$LOG_DIR"
echo ">>> 1v3 评估开始: $(date) | 日志: $LOG_FILE"
python3 one_vs_three.py 2>&1 | tee "$LOG_FILE"
echo ">>> 1v3 评估结束: $(date) | 日志: $LOG_FILE"
