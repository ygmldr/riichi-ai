#!/usr/bin/env bash
# 离线 DQN 训练 —— P3
# 用法: bash scripts/run_train.sh [config.toml]
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

# C1: train.py 启动时无条件构造 TestPlayer()(vendor/mortal/mortal/train.py:97),
# 立即 torch.load config['baseline']['test']['state_file'](player.py:19);
# baseline 检查点缺失时 DQN 训练必然崩溃,必须先引导生成。
BASELINE_FILE="$(MORTAL_CFG="$CFG" python3 - <<'PYEOF'
from config import config
print(config['baseline']['test']['state_file'])
PYEOF
)"
if [ ! -f "$BASELINE_FILE" ]; then
    echo "ERROR: baseline 检查点不存在: $BASELINE_FILE" >&2
    echo "train.py 启动即加载它(TestPlayer),缺失则训练必然崩溃;脚本不会静默创建空文件。" >&2
    echo "请先准备 baseline(二选一):" >&2
    echo "  A. 随机初始化引导(首次): 按 RUNBOOK「baseline.pth 引导」执行引导命令," >&2
    echo "     首轮训练完成后 cp checkpoints/mortal.pth checkpoints/baseline.pth 更新;" >&2
    echo "  B. 已有训练产物: cp checkpoints/mortal.pth checkpoints/baseline.pth" >&2
    exit 1
fi

echo ">>> 离线 DQN 训练开始: $(date)"
export MORTAL_CFG="$CFG"
nohup python3 train.py > "$REPO_ROOT/runs/dqn_train.log" 2>&1 &
echo "PID: $! | 日志: $REPO_ROOT/runs/dqn_train.log"
echo "监控: tail -f $REPO_ROOT/runs/dqn_train.log | grep -E 'steps|loss|save'"
