#!/usr/bin/env python3
"""把目录下所有非 gzip 的 .mjson 文件就地压缩为 gzip。

背景: tenhou-to-mjai 数据集里 2025/2026 年度的 zip 内是**未压缩明文 JSON**,
其余年度为 gzip;而 libriichi 的 load_gz_log_files 严格要求 gzip。不规范化的话,
训练会在随机读到这两年文件时崩溃(`invalid gzip header`,2026-09-14 实际发生过)。

用法: python3 scripts/normalize_gzip.py <dir> [<dir> ...]
说明: 已是 gzip 的文件跳过; 原地替换(先写临时文件再原子替换), 可重复执行。
"""
import gzip
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

GZIP_MAGIC = b"\x1f\x8b"


def fix(path: str) -> int:
    with open(path, "rb") as f:
        if f.read(2) == GZIP_MAGIC:
            return 0
    with open(path, "rb") as f:
        data = f.read()
    tmp = path + ".gztmp"
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        f.write(data)
    os.replace(tmp, path)
    return 1


def main():
    roots = sys.argv[1:]
    if not roots:
        sys.exit(__doc__)
    files = []
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            files += [os.path.join(dirpath, n) for n in filenames if n.endswith(".mjson")]
    print(f"检查 {len(files):,} 个 .mjson ...", flush=True)
    if not files:
        return
    fixed = done = 0
    t0 = time.time()
    workers = min(16, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(fix, files, chunksize=256):
            done += 1
            fixed += r
            if done % 100000 == 0:
                print(f"{done:,}/{len(files):,} fixed={fixed:,} {time.time()-t0:.0f}s", flush=True)
    print(f"完成: 共 {len(files):,} 个, 其中 {fixed:,} 个转为 gzip, 用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
