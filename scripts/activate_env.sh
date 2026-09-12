#!/usr/bin/env bash
# 激活 Mortal 训练环境(优先 conda mortal,退回 .venv)
# 用法: source scripts/activate_env.sh
# 必须在仓库根或脚本目录下运行
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v conda &> /dev/null; then
    if conda env list | grep -qE "^\s*mortal\s"; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate mortal
        echo ">>> 环境: conda mortal ($(python --version 2>&1))"
        return 0 2>/dev/null || exit 0
    fi
fi

if [ -d "$REPO_ROOT/.venv" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
    echo ">>> 环境: .venv ($(python --version 2>&1))"
else
    echo "ERROR: 未找到 mortal conda 环境或 .venv,请先运行 scripts/setup_gpu_env.sh" >&2
    exit 1
fi

# libriichi(PyO3 cdylib)构建产物目录,供 import libriichi。
# release 比 debug 快约 18 倍(特征提取实测 0.78ms vs 19ms/点),优先 release。
for LIB in release debug; do
    LIBRIICHI_PATH="$REPO_ROOT/vendor/mortal/target/$LIB"
    if [ -f "$LIBRIICHI_PATH/libriichi.so" ]; then
        export PYTHONPATH="$LIBRIICHI_PATH${PYTHONPATH:+:$PYTHONPATH}"
        echo ">>> PYTHONPATH 已加入 libriichi($LIB): $LIBRIICHI_PATH"
        break
    fi
done
