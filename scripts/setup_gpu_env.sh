#!/usr/bin/env bash
# 5070Ti 训练机一次性环境安装(需 sudo / 手动确认驱动已装)
# 前置: NVIDIA 驱动已装且支持 CUDA 12.8+(nvidia-smi 正常)
# 环境管理: 优先用已存在的 conda mortal 环境;否则 venv(系统 python3.12)。
# 说明: 不主动用 conda create(依赖 conda 镜像源网络,清华/浙大源常连不上)。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo ">>> [0/4] 前置检查: NVIDIA 驱动"
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: 未找到 nvidia-smi,请先安装 NVIDIA 驱动(CUDA 12.8+ 以支持 Blackwell sm_120)"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo ">>> [1/4] 环境准备"
USE_CONDA=0
if command -v conda &> /dev/null && conda env list 2>/dev/null | grep -qE "^\s*mortal\s"; then
    echo "检测到 conda 环境 mortal,使用它"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate mortal
    USE_CONDA=1
else
    echo "未检测到 conda mortal 环境,使用 venv(系统 python3.12)"
    if ! command -v python3.12 &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv
    fi
    rm -rf .venv
    python3.12 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo ">>> [2/4] PyTorch (Blackwell 需 cu128+)"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo ">>> [3/4] Mortal Python 依赖"
pip install tqdm toml tensorboard numpy

echo ">>> [4/4] 构建 libriichi"
bash scripts/build_libriichi.sh

echo ">>> 验证 GPU 可用"
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA 不可用'
print('GPU:', torch.cuda.get_device_name(0))
print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)
"
echo ">>> 环境就绪(USE_CONDA=$USE_CONDA)"
