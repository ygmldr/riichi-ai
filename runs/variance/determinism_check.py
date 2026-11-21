#!/usr/bin/env python3
"""Determinism cross-check: pool_1 vs EVAL_A_175200.

对每个 .json.gz 逐行解析 NDJSON, 归一化掉运行时计时字段 `eval_time_ns`,
再对归一化后的内容做 sha256; 逐文件比较两侧哈希。
输出: identical / differ 计数与首个差异文件名。
"""
import gzip
import hashlib
import json
import os
import re
import sys

A = "runs/variance/pool_1/games"
B = "runs/exp/EVAL_A_175200/games"
TIMING_KEYS = {"eval_time_ns"}

if len(sys.argv) == 3:
    A, B = sys.argv[1], sys.argv[2]


def norm_hash(path):
    h = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            ev = json.loads(line)
            meta = ev.get("meta")
            if isinstance(meta, dict):
                for k in TIMING_KEYS:
                    meta.pop(k, None)
            h.update(json.dumps(ev, sort_keys=True, separators=(",", ":")).encode())
            h.update(b"\n")
    return h.hexdigest()


names = sorted(
    n for n in os.listdir(A) if n.endswith(".json.gz") and os.path.exists(os.path.join(B, n))
)
print(f"comparing {len(names)} files (normalized: drop {sorted(TIMING_KEYS)})")
identical = 0
diffs = []
for i, name in enumerate(names):
    ha = norm_hash(os.path.join(A, name))
    hb = norm_hash(os.path.join(B, name))
    if ha == hb:
        identical += 1
    else:
        diffs.append(name)
    if (i + 1) % 2000 == 0:
        print(f"  {i + 1}/{len(names)} identical={identical} differ={len(diffs)}", flush=True)
print(f"RESULT: files={len(names)} identical={identical} differ={len(diffs)}")
for n in diffs[:10]:
    print(f"  DIFF: {n}")
sys.exit(0 if not diffs else 1)
