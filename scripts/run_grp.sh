#!/usr/bin/env bash
# GRP(顺位预测)训练 —— P2
# 用法: bash scripts/run_grp.sh [config.toml 路径]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:-$REPO_ROOT/configs/config.toml}"
# 归一化为绝对路径: 后续 cd vendor/mortal/mortal 后,MORTAL_CFG 按进程 CWD 解析(同 prepare_data_on_gpu.sh 模式)
case "$CFG" in
    /*) ;;
    *) CFG="$(pwd)/$CFG" ;;
esac

source "$REPO_ROOT/scripts/activate_env.sh"
cd "$REPO_ROOT/vendor/mortal/mortal"

mkdir -p "$REPO_ROOT/checkpoints"
mkdir -p "$REPO_ROOT/runs"
echo ">>> GRP 训练开始: $(date)"
export MORTAL_CFG="$CFG"
nohup python3 train_grp.py > "$REPO_ROOT/runs/grp_train.log" 2>&1 &
echo "PID: $! | 日志: $REPO_ROOT/runs/grp_train.log"
echo "监控: tail -f $REPO_ROOT/runs/grp_train.log"
