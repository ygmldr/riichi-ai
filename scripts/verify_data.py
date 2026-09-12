#!/usr/bin/env python3
"""校验数据集完整性并统计每年度局数。

用法: python3 scripts/verify_data.py <data_dir>
输出: 每年度 zip 的局数(即 .mjson 文件数)与总局数;损坏的 zip 会明确报出。
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
        try:
            n = count_games(path)
        except zipfile.BadZipFile as e:
            print(f"{name}: 损坏!({e})")
            sys.exit(f"{name} 不是有效 zip 文件,请重新下载")
        total += n
        print(f"{name}: {n:,} 局")
    print(f"总计: {total:,} 局")


if __name__ == "__main__":
    main()
