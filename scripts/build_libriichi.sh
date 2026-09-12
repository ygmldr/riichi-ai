#!/usr/bin/env bash
# 构建 Mortal 的 Rust 引擎 libriichi(PyO3 cdylib)
# 用法: bash scripts/build_libriichi.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/vendor/mortal"

if ! command -v cargo &> /dev/null; then
    echo "ERROR: cargo 未安装。请先安装 Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
    exit 1
fi

echo ">>> cargo build -p libriichi --lib"
cargo build -p libriichi --lib

SO_PATH="$(pwd)/target/debug/libriichi.so"
if [ ! -f "$SO_PATH" ]; then
    echo "ERROR: 构建产物不存在: $SO_PATH"
    exit 1
fi

echo ">>> 验证 import libriichi"
PYTHONPATH="$(pwd)/target/debug" python3 -c "
import libriichi
print('libriichi OK:', libriichi.__file__)
from libriichi.consts import ACTION_SPACE, obs_shape
assert ACTION_SPACE == 46, ACTION_SPACE
assert obs_shape(4) == (1012, 34), obs_shape(4)
print('ACTION_SPACE =', ACTION_SPACE, '| obs_shape(4) =', obs_shape(4))
"
echo ">>> libriichi 构建完成: $SO_PATH"
