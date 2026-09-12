#!/usr/bin/env bash
# 5070Ti 上: 解压数据集 zip → datasets/extracted/YYYY/*.mjson
# 并生成 GRP train_globs / val_globs 文本文件
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${1:-datasets}"
# 归一化为绝对路径: train_grp.py 的 glob() 按进程 CWD 解析相对 globs,
# 训练时 CWD 是 vendor/mortal/mortal,相对路径会解析失败导致找不到数据
case "$DATA_DIR" in
    /*) ;;
    *) DATA_DIR="$REPO_ROOT/$DATA_DIR" ;;
esac
EXTRACT_DIR="$DATA_DIR/extracted"
mkdir -p "$EXTRACT_DIR"

shopt -s nullglob
ZIPS=("$DATA_DIR"/*.zip)
shopt -u nullglob
if [ "${#ZIPS[@]}" -eq 0 ]; then
    echo "ERROR: 在 $DATA_DIR 中未找到任何 .zip 文件"
    exit 1
fi

echo ">>> 解压所有年度 zip"
for zip in "${ZIPS[@]}"; do
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
