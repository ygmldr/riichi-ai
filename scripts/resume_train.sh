#!/usr/bin/env bash
# 从检查点续训(修改 config.toml 的 state_file 指向已存模型后运行)
# 用法: bash scripts/resume_train.sh [config.toml]
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

# train.py 启动即加载 baseline 检查点(TestPlayer),缺失则续训必然崩溃,
# 与 run_train.sh 的引导逻辑保持一致。
BASELINE_FILE="$(MORTAL_CFG="$CFG" python3 - <<'PYEOF'
from config import config
print(config['baseline']['test']['state_file'])
PYEOF
)"
if [ ! -f "$BASELINE_FILE" ]; then
    echo "ERROR: baseline 检查点不存在: $BASELINE_FILE" >&2
    echo "train.py 启动即加载它(TestPlayer),缺失则续训必然崩溃。" >&2
    echo "请先准备 baseline: cp checkpoints/mortal.pth checkpoints/baseline.pth" >&2
    echo "(首次引导见 RUNBOOK「baseline.pth 引导」)" >&2
    exit 1
fi

export MORTAL_CFG="$CFG"
exec python3 train.py
