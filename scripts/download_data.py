#!/usr/bin/env python3
"""下载 tenhou-to-mjai 公开数据集(天凤凤凰卓 2009-2026, ~11.9GB)。

用法: python3 scripts/download_data.py <target_dir>
使用 curl 子进程作为下载后端:urllib 对 GitHub 302 重定向 + Range 续传不可靠
(黑洞连接挂起无 socket 超时,文件停止增长),curl 的 -C - 断点续传与
--retry 自动重试实测可靠(11.5MB/s)。完成后逐 zip 做完整性校验(zipfile.testzip)。
"""
import os
import subprocess
import sys
import zipfile

RELEASE_TAG = "v2.0.0"
BASE_URL = f"https://github.com/NikkeTryHard/tenhou-to-mjai/releases/download/{RELEASE_TAG}"
YEARS = [f"{y}.zip" for y in range(2009, 2027)]


def _zip_ok(path: str) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except Exception:
        return False


def download(url: str, dest: str) -> None:
    name = os.path.basename(dest)
    if os.path.exists(dest) and _zip_ok(dest):
        print(f"[skip] {name} 已存在且完整")
        return
    # -C -: 断点续传(已存在的 partial 文件自动续传); --fail: HTTP 错误即退出非零
    cmd = ["curl", "-sL", "--fail", "--retry", "5", "--retry-delay", "3",
           "-C", "-", "-o", dest, url]
    print(f"[get] {name}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        # 不崩溃退出:打 warn 继续,由末尾统一校验兜底
        print(f"[warn] {name} curl 退出码 {proc.returncode}(可重跑续传)")
        return
    print(f"[done] {name}")


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
        size = (os.path.getsize(dest) / (1024**3)) if os.path.exists(dest) else 0.0
        print(f"{name}: {'OK' if ok else '损坏!'} ({size:.2f} GB)")
        if not ok:
            sys.exit(f"{name} 校验失败,请删除后重跑(断点续传会自动跳过完整文件)")


if __name__ == "__main__":
    main()
