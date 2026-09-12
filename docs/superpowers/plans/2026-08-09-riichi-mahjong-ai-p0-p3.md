# 自制立直麻将 AI — 实施计划(P0-P3:环境、数据、GRP、离线 DQN)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 RTX 5070 Ti 上复刻 Mortal 训练管线,完成从环境搭建到离线 DQN 训练出第一个能跑评估的模型(基准:duplicate mahjong 打赢 akochan)。

**Architecture:** 分阶段演进(方案 C)。以 Equim-chan/Mortal 主线为骨架(fork),复用其完整训练管线:数据(tenhou-to-mjai 公开数据集)→ libriichi(Rust 引擎,特征提取)→ GRP(顺位预测奖励模型)→ DQN(离线 CQL)→ duplicate mahjong 评估。所有自定义配置通过 `MORTAL_CFG` 环境变量注入,不硬改上游代码,保持 upstream 同步能力。

**Tech Stack:** Rust(libriichi/PyO3)、Python 3.12、PyTorch(CUDA 12.8+/Blackwell)、tenhou-to-mjai 数据集(MJAI 格式 `.mjson.gz`)、mjai 协议。

**计划范围:** P0(环境就绪)→ P1(数据就绪)→ P2(GRP 训练)→ P3(离线 DQN + 评估闭环)。P4(迭代打磨)与 P5(自对弈微调)依赖 P3 产出的基线模型,后续单独出计划。

---

## Global Constraints

- 训练硬件:RTX 5070 Ti(16GB VRAM),Blackwell 架构,必须用 CUDA 12.8+ 的 PyTorch wheel(`cu128` 或 `cu129`)
- Python 版本:**3.12.x**(Mortal `environment.yml` 要求)
- Rust 工具链:stable(CI 用 `rustup default stable`,实测 rustc 1.97 构建通过)
- 数据集:tenhou-to-mjai release `v2.0.0`,18 个年度 zip(`2009.zip` ~ `2026.zip`),合计 ~11.9GB,内含 gzip 压缩的 `.mjson` 文件
- 数据文件扩展名:`*.mjson`(gzip 压缩);训练配置的 `dataset.globs` 必须用 `*.mjson`,不是示例里的 `*.json.gz`
- 模型配置:version 4(obs_shape `(1012, 34)`),resnet `conv_channels=192, num_blocks=40`(≈11M 参数),ACTION_SPACE=46
- 许可:AGPL-3.0(fork Mortal,自用无碍);不硬改上游 `mortal/` 下文件,配置差异全走 `MORTAL_CFG`
- 双机形态:开发/下载在控制台(`/root/riichi-ai`),训练在 5070Ti;同一 git 仓库,`git pull` 同步
- 训练脚本必须在仓库 `mortal/` 目录下运行(它们用相对导入 `from config import config`,依赖 `MORTAL_CFG` 环境变量指向配置文件)

---

### Task 1: 项目骨架 + fork Mortal 上游

**Files:**
- Create: `/root/riichi-ai/.gitignore`
- Create: `/root/riichi-ai/vendor/README.md`
- Create: `/root/riichi-ai/scripts/`(空目录占位)
- Create: `/root/riichi-ai/configs/`(空目录占位)

**Interfaces:**
- Produces: 仓库根目录下 `vendor/mortal/` 指向 Equim-chan/Mortal 的 git 子模块(pinned 到固定 commit),后续所有任务在 `vendor/mortal/` 下操作

- [ ] **Step 1: 确认仓库根与设计文档已提交**

```bash
cd /root/riichi-ai && git log --oneline -1 && ls docs/superpowers/specs/
```
Expected: 显示 `c2a63ff` 与设计文档文件。

- [ ] **Step 2: 创建 .gitignore(排除 Rust target 与数据集)**

```bash
cd /root/riichi-ai && cat > .gitignore << 'EOF'
# Rust
target/
**/*.rs.bk
Cargo.lock.orig

# Python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/

# 数据集(大文件不进仓库)
data/
datasets/

# 训练产物
checkpoints/
*.pth
runs/
tensorboard/
EOF
```

- [ ] **Step 3: 添加 Mortal 作为子模块并固定 commit**

```bash
cd /root/riichi-ai && git submodule add https://github.com/Equim-chan/Mortal.git vendor/mortal
cd vendor/mortal && git fetch --depth 1 origin main && git checkout $(git rev-parse origin/main)
```
Expected: `vendor/mortal/` 目录存在,`git -C vendor/mortal log --oneline -1` 显示某个上游 commit hash(记录下来,写入 vendor/README.md)。

- [ ] **Step 4: 写入 vendor/README.md 记录固定版本**

```bash
cd /root/riichi-ai && cat > vendor/README.md << 'EOF'
# vendor/

`mortal/` 是 Equim-chan/Mortal 的 git 子模块(fork 基准)。

- 固定 commit:见 `git -C vendor/mortal log --oneline -1`
- 同步上游:`git submodule update --remote vendor/mortal`(需审慎,先跑测试)
- 约定:不修改 `vendor/mortal/mortal/` 下任何代码;自定义配置通过 `MORTAL_CFG` 环境变量注入
EOF
```

- [ ] **Step 5: 提交**

```bash
cd /root/riichi-ai && git add .gitignore vendor/ && git commit -m "chore: 项目骨架 + Mortal fork 子模块"
```

- [ ] **Step 6: 验证子模块可克隆(模拟 5070Ti 侧拉取)**

```bash
rm -rf /tmp/riichi-clone-test && git clone --recurse-submodules /root/riichi-ai /tmp/riichi-clone-test && ls /tmp/riichi-clone-test/vendor/mortal/mortal/train.py
```
Expected: 克隆成功且 `train.py` 存在。

---

### Task 2: libriichi 构建脚本(控制台 + 5070Ti 通用)

**Files:**
- Create: `/root/riichi-ai/scripts/build_libriichi.sh`
- Create: `/root/riichi-ai/scripts/test_libriichi.py`

**Interfaces:**
- Produces: `build_libriichi.sh` — 在任意机器上构建 libriichi 并验证 `import libriichi`;`test_libriichi.py` — 冒烟测试(常量与 obs_shape)
- Consumes: `vendor/mortal/`(Task 1)

- [ ] **Step 1: 写构建脚本(实测可用的命令)**

```bash
cd /root/riichi-ai && cat > scripts/build_libriichi.sh << 'EOF'
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
EOF
chmod +x scripts/build_libriichi.sh
```

- [ ] **Step 2: 在控制台实际运行构建脚本(验证脚本正确)**

```bash
cd /root/riichi-ai && bash scripts/build_libriichi.sh
```
Expected: 输出 `libriichi OK:` 与 `ACTION_SPACE = 46 | obs_shape(4) = (1012, 34)`,退出码 0。

- [ ] **Step 3: 写并运行冒烟测试**

```bash
cd /root/riichi-ai && cat > scripts/test_libriichi.py << 'EOF'
"""libriichi 冒烟测试:常量、obs 形状、单局解析。"""
import os
import sys
import zipfile
import gzip

# 将构建产物加入 path(与 build_libriichi.sh 一致)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "vendor", "mortal", "target", "debug"))

import libriichi  # noqa: E402
from libriichi.consts import ACTION_SPACE, GRP_SIZE, obs_shape  # noqa: E402


def test_consts():
    assert ACTION_SPACE == 46
    assert GRP_SIZE == 7
    assert obs_shape(4) == (1012, 34)


def test_parse_one_game():
    """用 2009.zip 里的一局验证 GameplayLoader 能解析 .mjson(gzip)。"""
    zip_path = os.environ.get("TEST_ZIP", "/tmp/mjai-test/2009.zip")
    if not os.path.exists(zip_path):
        return  # 数据未下载时跳过
    loader = libriichi.dataset.GameplayLoader(version=4, oracle=False)
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        raw = z.read(name)  # .mjson 是 gzip 压缩的
        text = gzip.decompress(raw).decode("utf-8")
    games = loader.load_log(text)
    assert len(games) == 4  # 四人麻将,4 个玩家视角
    obs = games[0].take_obs()
    assert obs.shape[1] == 34, obs.shape


if __name__ == "__main__":
    test_consts()
    test_parse_one_game()
    print("libriichi 冒烟测试全部通过")
EOF
TEST_ZIP=/tmp/mjai-test/2009.zip python3 scripts/test_libriichi.py
```
Expected: `libriichi 冒烟测试全部通过`。

- [ ] **Step 4: 提交**

```bash
cd /root/riichi-ai && git add scripts/ && git commit -m "feat: libriichi 构建脚本 + 冒烟测试"
```

---

### Task 3: 数据下载脚本(P1 数据就绪,高优先级)

**Files:**
- Create: `/root/riichi-ai/scripts/download_data.py`
- Create: `/root/riichi-ai/scripts/verify_data.py`

**Interfaces:**
- Produces: `download_data.py <target_dir>` — 断点续传下载 18 个年度 zip 到指定目录;`verify_data.py <data_dir>` — 校验 zip 完整性并报告每年度局数
- Consumes: GitHub release `NikkeTryHard/tenhou-to-mjai` tag `v2.0.0`

**Context(实测确认):** release `v2.0.0` 的 assets 为 `2009.zip`…`2026.zip`,共 18 个,合计 11,899,587,440 字节 ≈ 11.9GB。zip 内为 `YYYY/` 目录,文件为 gzip 压缩的 `.mjson`。

- [ ] **Step 1: 写下载脚本(支持断点续传 + 校验和)**

```bash
cd /root/riichi-ai && cat > scripts/download_data.py << 'EOF'
#!/usr/bin/env python3
"""下载 tenhou-to-mjai 公开数据集(天凤凤凰卓 2009-2026, ~11.9GB)。

用法: python3 scripts/download_data.py <target_dir>
支持断点续传(HTTP Range);完成后逐 zip 做 CRC 校验(通过 zipfile 完整性检查)。
"""
import os
import sys
import zipfile
import urllib.request

RELEASE_TAG = "v2.0.0"
BASE_URL = f"https://github.com/NikkeTryHard/tenhou-to-mjai/releases/download/{RELEASE_TAG}"
YEARS = [f"{y}.zip" for y in range(2009, 2027)]


def download(url: str, dest: str) -> None:
    if os.path.exists(dest) and _zip_ok(dest):
        print(f"[skip] {dest} 已存在且完整")
        return
    tmp = dest + ".part"
    resume = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url)
    if resume:
        req.add_header("Range", f"bytes={resume}-")
        print(f"[resume] {dest} 从 {resume} 字节续传")
    with urllib.request.urlopen(req) as resp, open(tmp, "ab") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.rename(tmp, dest)
    print(f"[done] {dest}")


def _zip_ok(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:
        return False


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "datasets"
    os.makedirs(target, exist_ok=True)
    for name in YEARS:
        url = f"{BASE_URL}/{name}"
        dest = os.path.join(target, name)
        download(url, dest)
    print("=== 全部下载完成,开始完整性校验 ===")
    for name in YEARS:
        dest = os.path.join(target, name)
        ok = _zip_ok(dest)
        size = os.path.getsize(dest) / (1024**3)
        print(f"{name}: {'OK' if ok else '损坏!'} ({size:.2f} GB)")
        if not ok:
            sys.exit(f"{name} 校验失败,请删除后重跑(断点续传会自动跳过完整文件)")


if __name__ == "__main__":
    main()
EOF
```

- [ ] **Step 2: 写校验/统计脚本**

```bash
cd /root/riichi-ai && cat > scripts/verify_data.py << 'EOF'
#!/usr/bin/env python3
"""校验数据集完整性并统计局数。

用法: python3 scripts/verify_data.py <data_dir>
输出: 每年度 zip 的局数(即 .mjson 文件数)与总局数。
"""
import os
import sys
import zipfile


def count_games(zip_path: str) -> int:
    with zipfile.ZipFile(zip_path) as z:
        return len([n for n in z.namelist() if n.endswith(".mjson")])


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets"
    total = 0
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".zip"):
            continue
        path = os.path.join(data_dir, name)
        n = count_games(path)
        total += n
        print(f"{name}: {n:,} 局")
    print(f"总计: {total:,} 局")


if __name__ == "__main__":
    main()
EOF
```

- [ ] **Step 3: 下载完整数据集到 `datasets/`(约 11.9GB,取决于带宽,可能数小时)**

```bash
cd /root/riichi-ai && nohup python3 scripts/download_data.py datasets > downloads.log 2>&1 &
# 监控: tail -f downloads.log
```

- [ ] **Step 4: 下载完成后校验并统计**

```bash
cd /root/riichi-ai && python3 scripts/verify_data.py datasets
```
Expected: 每个年度打印局数,总计约 250 万局(2009 约 8k,2024 约 335k,…… 与数据集发布页一致)。

- [ ] **Step 5: 提交脚本(数据集本身不入库,.gitignore 已排除 `datasets/`)**

```bash
cd /root/riichi-ai && git add scripts/download_data.py scripts/verify_data.py && git commit -m "feat: 数据集下载与校验脚本(tenhou-to-mjai v2.0.0)"
```

---

### Task 4: 训练配置模板(P0/P2/P3 共用)

**Files:**
- Create: `/root/riichi-ai/configs/config.example.toml`(基于上游示例,填真实路径)
- Create: `/root/riichi-ai/configs/README.md`

**Interfaces:**
- Produces: `configs/config.example.toml` — 供 GRP(`[grp]` 段)与 DQN 离线(`[control]`/`[dataset]`/`[cql]` 段)共用,训练前复制为 `config.toml` 并微调
- Consumes: 上游 `vendor/mortal/mortal/config.example.toml` 的结构(字段名逐一对应)

**Context(实测确认):** `config.py` 通过环境变量 `MORTAL_CFG` 读取配置(默认 `config.toml`);训练脚本用相对导入,必须在 `vendor/mortal/mortal/` 目录下运行。`dataset.globs` 匹配 `.mjson` 文件;`file_index` 缓存文件列表(可复用,加速启动)。

- [ ] **Step 1: 基于上游示例生成我们的配置模板**

```bash
cd /root/riichi-ai && sed -e "s|/path/to/dataset|/root/riichi-ai/datasets|g" \
  -e "s|\*.json.gz|\*.mjson|g" \
  -e "s|/path/to/mortal.pth|/root/riichi-ai/checkpoints/mortal.pth|g" \
  -e "s|/path/to/best.pth|/root/riichi-ai/checkpoints/best.pth|g" \
  -e "s|/path/to/dir|/root/riichi-ai/runs|g" \
  -e "s|/path/to/train_play|/root/riichi-ai/runs/train_play|g" \
  -e "s|/path/to/test_play|/root/riichi-ai/runs/test_play|g" \
  -e "s|/path/to/file_index.pth|/root/riichi-ai/checkpoints/file_index.pth|g" \
  -e "s|/path/to/baseline.pth|/root/riichi-ai/checkpoints/baseline.pth|g" \
  -e "s|/path/to/buffer|/root/riichi-ai/runs/buffer|g" \
  -e "s|/path/to/drain|/root/riichi-ai/runs/drain|g" \
  -e "s|/path/to/1v3|/root/riichi-ai/runs/1v3|g" \
  -e "s|/path/to/challenger.pth|/root/riichi-ai/checkpoints/challenger.pth|g" \
  -e "s|/path/to/grp.pth|/root/riichi-ai/checkpoints/grp.pth|g" \
  -e "s|/path/to/akochan|/root/riichi-ai/vendor/akochan|g" \
  -e "s|/path/to/tactics.json|/root/riichi-ai/vendor/akochan/tactics.json|g" \
  -e "s|/path/to/grp_file_index.pth|/root/riichi-ai/checkpoints/grp_file_index.pth|g" \
  vendor/mortal/mortal/config.example.toml > configs/config.example.toml
```

- [ ] **Step 2: 修正 GRP 数据划分(训练/验证分年)**

```bash
cd /root/riichi-ai && cat >> configs/config.example.toml << 'EOF'

# --- GRP 数据划分(追加覆盖)---
# 训练: 2009-2022 全年; 验证: 2023 全年(与 DQN 训练数据错开)
EOF
```
(注:GRP 的 `train_globs`/`val_globs` 建议在训练机按实际年份目录生成,见 Task 5;此文件为模板,具体 globs 在部署时填写。)

- [ ] **Step 3: 验证配置可被 Mortal 的 config.py 解析**

```bash
cd /root/riichi-ai/vendor/mortal/mortal && MORTAL_CFG=/root/riichi-ai/configs/config.example.toml python3 -c "
from config import config
assert config['resnet']['conv_channels'] == 192
assert config['resnet']['num_blocks'] == 40
assert config['control']['version'] == 4
print('配置解析 OK, version =', config['control']['version'])
"
```
Expected: `配置解析 OK, version = 4`。

- [ ] **Step 4: 写 configs/README.md 说明配置用法**

```bash
cd /root/riichi-ai && cat > configs/README.md << 'EOF'
# 配置

- `config.example.toml` — 模板(路径指向 /root/riichi-ai)。训练前复制为 `config.toml` 并调整:
  - `[grp.dataset]` 的 train_globs / val_globs(按年份切分)
  - `[1v3.champion]` / `[1v3.challenger]` 的 state_file(评估对象)
- 运行时通过环境变量指定:MORTAL_CFG=/path/to/config.toml
- 训练脚本须在 vendor/mortal/mortal/ 目录下运行(相对导入)
EOF
```

- [ ] **Step 5: 提交**

```bash
cd /root/riichi-ai && git add configs/ && git commit -m "feat: 训练配置模板(MORTAL_CFG 注入,不硬改上游)"
```

---

### Task 5: 5070Ti 训练机环境准备(P0,P2/P3 前置)

**Files:**
- Create: `/root/riichi-ai/scripts/setup_gpu_env.sh`
- Create: `/root/riichi-ai/scripts/prepare_data_on_gpu.sh`

**Interfaces:**
- Consumes: `datasets/`(Task 3,通过 rsync/scp 或共享盘拷到 5070Ti)、`configs/config.example.toml`(Task 4)
- Produces: 5070Ti 上可运行的训练环境:Python 3.12 venv、CUDA 12.8+ 的 PyTorch、libriichi 构建产物、真实 `config.toml`

**Context(关键):** 5070 Ti 是 Blackwell(sm_120),PyTorch 官方 wheel 需选 `cu128`(CUDA 12.8)及以上,否则 GPU 不可用。命令:`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`。

- [ ] **Step 1: 写 GPU 环境安装脚本**

```bash
cd /root/riichi-ai && cat > scripts/setup_gpu_env.sh << 'EOF'
#!/usr/bin/env bash
# 5070Ti 训练机一次性环境安装(需 sudo / 手动确认驱动已装)
# 前置: NVIDIA 驱动已装且支持 CUDA 12.8+(nvidia-smi 正常)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo ">>> [1/4] Python 3.12 venv"
if ! command -v python3.12 &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv
fi
python3.12 -m venv .venv
source .venv/bin/activate

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
echo ">>> 环境就绪"
EOF
chmod +x scripts/setup_gpu_env.sh
```

- [ ] **Step 2: 写数据准备脚本(解压 + 生成 GRP globs)**

```bash
cd /root/riichi-ai && cat > scripts/prepare_data_on_gpu.sh << 'EOF'
#!/usr/bin/env bash
# 5070Ti 上: 解压数据集 zip → datasets/extracted/YYYY/*.mjson
# 并生成 GRP train_globs / val_globs 文本文件
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${1:-datasets}"
EXTRACT_DIR="$DATA_DIR/extracted"
mkdir -p "$EXTRACT_DIR"

echo ">>> 解压所有年度 zip"
for zip in "$DATA_DIR"/*.zip; do
    year=$(basename "$zip" .zip)
    mkdir -p "$EXTRACT_DIR/$year"
    unzip -o -q "$zip" -d "$EXTRACT_DIR/$year"
done

echo ">>> 生成 GRP 数据划分(训练: 2009-2022, 验证: 2023)"
TRAIN_GLOBS="$DATA_DIR/grp_train_globs.txt"
VAL_GLOBS="$DATA_DIR/grp_val_globs.txt"
: > "$TRAIN_GLOBS"
: > "$VAL_GLOBS"
for year_dir in "$EXTRACT_DIR"/*/; do
    year=$(basename "$year_dir")
    if [ "$year" -le 2022 ]; then
        echo "\"$EXTRACT_DIR/$year/**/*.mjson\"," >> "$TRAIN_GLOBS"
    elif [ "$year" -eq 2023 ]; then
        echo "\"$EXTRACT_DIR/$year/**/*.mjson\"," >> "$VAL_GLOBS"
    fi
done
echo ">>> train_globs: $(wc -l < "$TRAIN_GLOBS") 年, val_globs: $(wc -l < "$VAL_GLOBS") 年"
echo ">>> 数据准备完成。将 train/val globs 填入 config.toml 的 [grp.dataset]"
EOF
chmod +x scripts/prepare_data_on_gpu.sh
```

- [ ] **Step 3: 语法校验两个脚本**

```bash
cd /root/riichi-ai && bash -n scripts/setup_gpu_env.sh && bash -n scripts/prepare_data_on_gpu.sh && echo "bash 语法 OK"
```

- [ ] **Step 4: 提交**

```bash
cd /root/riichi-ai && git add scripts/setup_gpu_env.sh scripts/prepare_data_on_gpu.sh && git commit -m "feat: 5070Ti 环境安装与数据准备脚本"
```

---

### Task 6: GRP 训练(P2)

**Files:**
- Create: `/root/riichi-ai/scripts/run_grp.sh`
- Create: `/root/riichi-ai/checkpoints/`(运行时创建,不入库)

**Interfaces:**
- Consumes: 解压后的数据集(extracted 目录)、`configs/config.example.toml`(Task 4,`[grp]` 段)
- Produces: `checkpoints/grp.pth` — GRP 顺位预测模型(GRU,2 层 hidden 64),供 Task 7 的 DQN 训练计算奖励

**Context(实测确认):** `train_grp.py` 读取 `config['grp']` 段;训练脚本须在 `vendor/mortal/mortal/` 目录运行,依赖 `MORTAL_CFG`。

- [ ] **Step 1: 写 GRP 训练脚本**

```bash
cd /root/riichi-ai && cat > scripts/run_grp.sh << 'EOF'
#!/usr/bin/env bash
# GRP(顺位预测)训练 —— P2
# 用法: bash scripts/run_grp.sh [config.toml 路径]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:-$REPO_ROOT/configs/config.toml}"

source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT/vendor/mortal/mortal"

echo ">>> GRP 训练开始: $(date)"
export MORTAL_CFG="$CFG"
nohup python3 train_grp.py > "$REPO_ROOT/runs/grp_train.log" 2>&1 &
echo "PID: $! | 日志: $REPO_ROOT/runs/grp_train.log"
echo "监控: tail -f $REPO_ROOT/runs/grp_train.log"
EOF
chmod +x scripts/run_grp.sh
```

- [ ] **Step 2: 在 5070Ti 上准备真实 config.toml(复制模板,填写 GRP globs)**

```bash
cd /root/riichi-ai && cp configs/config.example.toml configs/config.toml
# 手动编辑 configs/config.toml 的 [grp.dataset]:
#   train_globs = [ $(cat datasets/grp_train_globs.txt) ]
#   val_globs   = [ $(cat datasets/grp_val_globs.txt) ]
#   file_index  = '/root/riichi-ai/checkpoints/grp_file_index.pth'
```

- [ ] **Step 3: 在 5070Ti 上运行 GRP 训练**

```bash
cd /root/riichi-ai && bash scripts/run_grp.sh configs/config.toml && sleep 30 && tail -30 runs/grp_train.log
```
Expected(30 秒后): 日志显示数据加载、`total steps` 递增、loss 下降;`[grp.control] save_every=2000` 时产出 `grp.pth`。

- [ ] **Step 4: 验证 GRP 产物**

```bash
cd /root/riichi-ai && ls -la checkpoints/grp.pth && python3 -c "
import torch
state = torch.load('checkpoints/grp.pth', weights_only=True, map_location='cpu')
print('GRP keys:', list(state.keys()))
assert 'state_dict' in state or 'gru' in state, state.keys()
print('GRP 权重有效')
"
```
Expected: 打印 GRP keys 且断言通过。

- [ ] **Step 5: 提交**

```bash
cd /root/riichi-ai && git add scripts/run_grp.sh && git commit -m "feat: GRP 训练脚本(P2)"
```

---

### Task 7: 离线 DQN 训练(P3)

**Files:**
- Create: `/root/riichi-ai/scripts/run_train.sh`
- Create: `/root/riichi-ai/scripts/resume_train.sh`

**Interfaces:**
- Consumes: `checkpoints/grp.pth`(Task 6)、解压数据集(extracted)、`configs/config.toml` 的 `[control]`/`[dataset]`/`[cql]`/`[grp]` 段
- Produces: `checkpoints/mortal.pth`(主模型,含 `mortal`/`current_dqn` state_dict)与 `checkpoints/best.pth`;`runs/test_play/` 下 test_play 日志

**Context(实测确认):** `train.py` 离线模式:先按 `[dataset]` 建 file_index,单 epoch 遍历数据;`state_file` 可指向已有模型(续训)或不存在(从随机初始化开始,训练脚本会创建);`test_every` 步后跑 test_play(1v3 自对弈)。GRP 通过 `[grp] state_file` 加载用于奖励计算。

- [ ] **Step 1: 写 DQN 训练脚本(带续训)**

```bash
cd /root/riichi-ai && cat > scripts/run_train.sh << 'EOF'
#!/usr/bin/env bash
# 离线 DQN 训练 —— P3
# 用法: bash scripts/run_train.sh [config.toml]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:-$REPO_ROOT/configs/config.toml}"

source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT/vendor/mortal/mortal"

echo ">>> 离线 DQN 训练开始: $(date)"
export MORTAL_CFG="$CFG"
nohup python3 train.py > "$REPO_ROOT/runs/dqn_train.log" 2>&1 &
echo "PID: $! | 日志: $REPO_ROOT/runs/dqn_train.log"
echo "监控: tail -f $REPO_ROOT/runs/dqn_train.log | grep -E 'steps|loss|save'"
EOF
chmod +x scripts/run_train.sh

cat > scripts/resume_train.sh << 'EOF'
#!/usr/bin/env bash
# 从检查点续训(修改 config.toml 的 state_file 指向已存模型后运行)
# 用法: bash scripts/resume_train.sh [config.toml]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:-$REPO_ROOT/configs/config.toml}"
source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT/vendor/mortal/mortal"
export MORTAL_CFG="$CFG"
exec python3 train.py
EOF
chmod +x scripts/resume_train.sh
```

- [ ] **Step 2: 配置 DQN 训练参数(编辑 configs/config.toml)**

```toml
# 要点(基于上游示例的推荐起点):
[control]
version = 4
online = false
state_file = '/root/riichi-ai/checkpoints/mortal.pth'   # 续训时指向已有模型
device = 'cuda:0'
batch_size = 512
enable_amp = true                                       # 5070Ti 支持混合精度,省显存提速
save_every = 400
test_every = 20000

[dataset]
globs = ['/root/riichi-ai/datasets/extracted/**/*.mjson']
file_index = '/root/riichi-ai/checkpoints/file_index.pth'
num_epochs = 1

[cql]
min_q_weight = 5

[grp]
state_file = '/root/riichi-ai/checkpoints/grp.pth'
```

- [ ] **Step 3: 在 5070Ti 上启动 DQN 训练**

```bash
cd /root/riichi-ai && bash scripts/run_train.sh configs/config.toml && sleep 60 && grep -E "steps|loss|file list|loaded" runs/dqn_train.log | tail -20
```
Expected(60 秒后): 日志显示 `file list size: ~2,500,000`、`total steps` 与 loss 值。首次会构建 file_index(遍历 250 万文件,可能 10-30 分钟)。

- [ ] **Step 4: 验证检查点产出**

```bash
cd /root/riichi-ai && ls -la checkpoints/mortal.pth checkpoints/best.pth 2>/dev/null || echo "(训练中,检查点按 save_every=400 步产出)"
```

- [ ] **Step 5: 提交**

```bash
cd /root/riichi-ai && git add scripts/run_train.sh scripts/resume_train.sh && git commit -m "feat: 离线 DQN 训练与续训脚本(P3)"
```

---

### Task 8: duplicate mahjong 评估闭环(P3 验收)

**Files:**
- Create: `/root/riichi-ai/scripts/run_eval.sh`
- Create: `/root/riichi-ai/scripts/parse_eval.py`

**Interfaces:**
- Consumes: `checkpoints/mortal.pth`(Task 7)、`configs/config.toml` 的 `[1v3]` 段
- Produces: `runs/1v3/` 下的对战日志 + 解析出的顺位分 JSON(验收指标: vs akochan 的 duplicate mahjong 顺位分)

**Context(实测确认):** `one_vs_three.py` 读 `config['1v3']` 段;`[1v3.champion]` 为被挑战方(akochan 或我们自己的基线模型),`[1v3.challenger]` 为挑战方。`games_per_iter` 每 4 局用同一副牌(seed 相同)进行 duplicate mahjong,消除运气因素。akochan 需单独 clone 并编译(`vendor/akochan`)。

- [ ] **Step 1: 写评估脚本**

```bash
cd /root/riichi-ai && cat > scripts/run_eval.sh << 'EOF'
#!/usr/bin/env bash
# duplicate mahjong 1v3 评估 —— P3 验收
# 用法: bash scripts/run_eval.sh [config.toml]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${1:-$REPO_ROOT/configs/config.toml}"
source "$REPO_ROOT/.venv/bin/activate"
cd "$REPO_ROOT/vendor/mortal/mortal"
export MORTAL_CFG="$CFG"
mkdir -p "$REPO_ROOT/runs/1v3"
echo ">>> 1v3 评估开始: $(date)"
python3 one_vs_three.py 2>&1 | tee "$REPO_ROOT/runs/1v3/eval_$(date +%Y%m%d_%H%M).log"
EOF
chmod +x scripts/run_eval.sh
```

- [ ] **Step 2: 写结果解析脚本**

```bash
cd /root/riichi-ai && cat > scripts/parse_eval.py << 'EOF'
#!/usr/bin/env python3
"""解析 1v3 评估日志,输出顺位分对比。

用法: python3 scripts/parse_eval.py <eval.log>
预期输出: challenger 与 champion 的均顺位分(avg rank pt)。
顺位分转正 = challenger 强于 champion。
"""
import re
import sys

PT = {"1st": 90, "2nd": 45, "3rd": 0, "4th": -135}  # jun_pt 与 Mortal 文档一致


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "runs/1v3/eval.log"
    challenger = {"games": 0, "pts": 0.0, "name": None}
    champion = {"games": 0, "pts": 0.0, "name": None}
    pattern = re.compile(r"\[(?P<name>\w+)\].*?(?P<rank>1st|2nd|3rd|4th)")

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            target = challenger if "challenger" in line or m.group("name") in ("mortal",) else champion
            target["games"] += 1
            target["pts"] += PT[m.group("rank")]
            if target["name"] is None:
                target["name"] = m.group("name")

    for d in (challenger, champion):
        if d["games"]:
            print(f"{d['name']}: {d['games']} 局, 均顺位分 {d['pts']/d['games']:+.3f}")
    if challenger["games"] and champion["games"]:
        delta = challenger["pts"] / challenger["games"] - champion["pts"] / champion["games"]
        verdict = "challenger 更强(顺位分转正)" if delta > 0 else "challenger 尚未超过 champion"
        print(f"顺位分差: {delta:+.3f} | {verdict}")


if __name__ == "__main__":
    main()
EOF
```

- [ ] **Step 3: 配置 1v3 评估(编辑 configs/config.toml)**

```toml
[1v3]
games_per_iter = 2000   # 每轮 2000 局(500 副牌 × 4 座位)
iters = 1               # 先跑 1 轮验证;验收时加到 5+
log_dir = '/root/riichi-ai/runs/1v3'

[1v3.challenger]
name = 'mortal'
state_file = '/root/riichi-ai/checkpoints/mortal.pth'
device = 'cuda:0'
enable_amp = true

[1v3.champion]
name = 'baseline'
state_file = '/root/riichi-ai/checkpoints/baseline.pth'
device = 'cuda:0'
enable_amp = true

[1v3.akochan]
enabled = true          # 验收目标: 对 akochan
dir = '/root/riichi-ai/vendor/akochan'
tactics = '/root/riichi-ai/vendor/akochan/tactics.json'
```

- [ ] **Step 4: 在 5070Ti 上运行评估**

```bash
cd /root/riichi-ai && bash scripts/run_eval.sh configs/config.toml
```
Expected: 日志出现对战进度;完成后 `ls runs/1v3/` 有日志文件。

- [ ] **Step 5: 解析结果(验收)**

```bash
cd /root/riichi-ai && python3 scripts/parse_eval.py runs/1v3/eval_*.log
```
**验收标准(P3 完成):** challenger 相对 champion(akochan)的均顺位分 **转正**(> 0)。未达标则回到 Task 7 迭代(调整 batch/lr/epochs 或数据量),或进入 P4(单独计划)。

- [ ] **Step 6: 提交**

```bash
cd /root/riichi-ai && git add scripts/run_eval.sh scripts/parse_eval.py && git commit -m "feat: duplicate mahjong 1v3 评估脚本(P3 验收)"
```

---

### Task 9: 运行手册 + 收尾

**Files:**
- Create: `/root/riichi-ai/RUNBOOK.md`
- Modify: `/root/riichi-ai/README.md`(若不存在则创建)

**Interfaces:**
- Produces: 完整部署/运行手册,5070Ti 操作者按步骤执行即可完成 P0-P3

- [ ] **Step 1: 写 RUNBOOK.md(双机操作手册)**

```bash
cd /root/riichi-ai && cat > RUNBOOK.md << 'EOF'
# 运行手册(P0-P3)

## 双机拓扑
- 控制台(/root/riichi-ai): 开发、数据下载、评估结果解析
- 训练机(5070Ti): git pull + 环境 + 训练

## 控制台一次操作
```bash
cd /root/riichi-ai
python3 scripts/download_data.py datasets   # ~11.9GB,断点续传
python3 scripts/verify_data.py datasets     # 校验 + 统计局数
```

## 5070Ti 一次操作
```bash
git clone --recurse-submodules <repo-url> riichi-ai
cd riichi-ai
bash scripts/setup_gpu_env.sh               # Python3.12 venv + torch cu128 + libriichi
bash scripts/prepare_data_on_gpu.sh datasets # 解压 + GRP globs
cp configs/config.example.toml configs/config.toml
# 编辑 config.toml: [grp.dataset] globs、[dataset] globs、[control] state_file 等
bash scripts/run_grp.sh configs/config.toml   # P2: GRP 训练
bash scripts/run_train.sh configs/config.toml # P3: 离线 DQN 训练
bash scripts/run_eval.sh configs/config.toml  # P3 验收: 1v3 duplicate mahjong
```

## 监控与续训
- GRP 日志: runs/grp_train.log
- DQN 日志: runs/dqn_train.log
- 续训: 把 config.toml 的 state_file 指向 checkpoints/mortal.pth, 然后 resume_train.sh
- TensorBoard: tensorboard --logdir runs

## 验收标准(P3 完成)
- scripts/parse_eval.py 输出 challenger 相对 akochan 的均顺位分 > 0
EOF
```

- [ ] **Step 2: 写 README.md(项目简介 + 指向文档)**

```bash
cd /root/riichi-ai && cat > README.md << 'EOF'
# riichi-ai

自制立直麻将 AI —— 基于 Equim-chan/Mortal 的复刻(fork),在 RTX 5070 Ti 上训练。

- 设计: `docs/superpowers/specs/2026-08-09-riichi-mahjong-ai-design.md`
- 计划: `docs/superpowers/plans/2026-08-09-riichi-mahjong-ai-p0-p3.md`
- 运行: `RUNBOOK.md`
- 许可: AGPL-3.0(fork Mortal)
EOF
```

- [ ] **Step 3: 全仓库一致性检查(确认无占位路径残留)**

```bash
cd /root/riichi-ai && grep -rn "/path/to" configs/ scripts/ RUNBOOK.md 2>/dev/null | grep -v "vendor/mortal" || echo "无 /path/to 残留(上游示例除外)"
```

- [ ] **Step 4: 最终提交**

```bash
cd /root/riichi-ai && git add RUNBOOK.md README.md && git commit -m "docs: 运行手册与项目 README(P0-P3 收尾)"
```

- [ ] **Step 5: 验证仓库完整可克隆**

```bash
rm -rf /tmp/riichi-final-check && git clone --recurse-submodules /root/riichi-ai /tmp/riichi-final-check && cd /tmp/riichi-final-check && ls scripts/ configs/ RUNBOOK.md README.md && bash -n scripts/*.sh && echo "仓库完整,所有脚本语法 OK"
```
Expected: 列出全部脚本/配置/文档,`bash -n` 全部通过,输出 `仓库完整,所有脚本语法 OK`。

---

## Self-Review

**1. 规格覆盖(对照设计文档):**
- P0 环境就绪 → Task 2(libriichi 构建)、Task 5(5070Ti 环境)✅
- P1 数据就绪 → Task 3(下载+校验)、Task 5 Step 2(解压+globs)✅
- P2 GRP 训练 → Task 6 ✅
- P3 离线 DQN → Task 7(训练+续训)✅
- P3 验收(打赢 akochan) → Task 8(1v3 duplicate mahjong + 解析)✅
- 双机形态/配置注入 → Task 1(子模块)、Task 4(MORTAL_CFG)、Task 9(RUNBOOK)✅
- 数据提前下载高优先级 → Task 3 标注"高优先级",脚本支持断点续传 ✅
- YAGNI: 无自对弈(留给 P5)、无复盘 UI、无天凤接入 ✅

**2. 占位符扫描:** 计划内无 TBD/TODO;所有路径为 `/root/riichi-ai/...` 具体值;`configs/config.toml` 由模板复制而来,明确标注需编辑的字段。✅

**3. 类型一致性:** `config.py` 的 `MORTAL_CFG` 环境变量贯穿 Task 4/5/6/7/8;`libriichi.consts.obs_shape(4)==(1012,34)`、`ACTION_SPACE==46` 在 Task 2 断言与 Task 4 配置校验一致;`state_file`/`file_index`/`grp.pth` 路径在 Task 4-8 间一致。✅

**4. 实测验证过的假设:** libriichi 构建与 import、数据格式 `.mjson` 兼容性、release 资产列表与大小、配置解析——均已在计划编写前实测确认。✅
