#!/usr/bin/env python3
"""1v3 评估报告工具: 解析 NDJSON 牌谱 -> (seed, seed_key) set 级 bootstrap 95% CI。

实测数据格式(2026-09, 以 runs/1v3/ 为准):
  - 文件名 `<seed>_<seed_key十进制>_<split>.json.gz`, split ∈ {a,b,c,d}
    (libriichi/src/arena/one_vs_three.rs:212-216)
  - 每文件恰 1 局, 且是 gzip 包裹的 NDJSON —— 每行一个 JSON 对象, 逐行 `json.loads`
    (libriichi/src/arena/result.rs:32-51); 绝不能整文件 `json.loads`。
  - split -> 挑战者(challenger)座位: a=0, b=1, c=2, d=3 (one_vs_three.rs:144-171)
  - 座位-座位名对照: `start_game.names` —— challenger 占 `seat_of_split(split)`,
    其余 3 席 champion; 挑战者名字即本行 arm 的名字。
  - 最终分数重放: `start_kyoku.scores` 为权威基线; `reach_accepted` 扣 1000 且供托 +1
    (board.rs:342-349); `hora.deltas` / `ryukyoku.deltas` 累加; `hora` 时供托被和牌者
    取走(清零, board.rs:376-377); 对局结束时若供托 > 0, 加给**当前第一名**
    (arena/game.rs:182-184 的 `min_by_key(|s| -**s)` 取的是最大分, 并列取首个最大)。
  - 顺位: 分数降序的**序数排名**(stable sort, 同分按座位号小者在前,
    libriichi/src/rankings.rs:9-10)——同分不并列为同一名次。
  - 顺位分: RANK_PTS = (90, 45, 0, -135), 与 scripts/parse_eval.py 一致。

Bootstrap(确定性):
  - set = 同一 `(seed, seed_key)` 的 4 个 split, 即 4 局 (4 人各坐 challenger 一次)。
  - 有放回重采样 **set**(不是单局), B=2000 次; RNG = `numpy.random.default_rng(0)`
    (PCG64, 跨版本稳定); 95% CI = 重采样统计量的 2.5/97.5 百分位
    (numpy.percentile 默认线性插值)。B 是重采样次数, 不是 set 数。
  - 不完整 set(缺 split)的局计入点估计, 但不参与 bootstrap(无法作为 4 局整体重采样)。

门控断言: 见 `EXACT_GATE`。当 `--games-dir` basename 为 `1v3`、`--seed-key` 为
`0x80a7b9e2dec96fa`、`--seed-range` 为 `10000-10500` 时, 脚本自检并断言 ground truth
(来自 runs/1v3/eval_20260915_1835.log: challenger=mortal live vs champion=old148800)。
任何断言失败 -> 打印实际值并以非零码退出, 绝不静默放过。

用法:
  python3 scripts/eval_report.py --games-dir runs/1v3 \
      --seed-key 0x80a7b9e2dec96fa --seed-range 10000-10500 \
      --checkpoint at-175200 --source-dir runs/exp/EVAL_A_175200 \
      --out runs/exp/EVAL_A_175200/report.csv
"""

import argparse
import csv
import gzip
import json
import os
import re
import sys
from collections import Counter

import numpy as np

RANK_PTS = (90, 45, 0, -135)
SPLITS = ("a", "b", "c", "d")
SEAT_OF_SPLIT = {name: i for i, name in enumerate(SPLITS)}
FILE_RE = re.compile(r"^(?P<seed>\d+)_(?P<key>\d+)_(?P<split>[abcd])\.json\.gz$")

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0

# 精确复现门。ground truth 独立验证自 runs/1v3/eval_20260915_1835.log 与
# scripts/parse_eval.py: `challenger rankings: [508 504 527 461] (2.4705, 3.0825pt)`。
# arm = challenger `mortal`(当时 live checkpoints/mortal.pth) vs champion `old148800`
# (checkpoints/mortal.pth.at-148800); NDJSON `start_game.names` 逐局可验证。
EXACT_GATE = {
    "dir_basename": "1v3",
    "seed_key": 0x80A7B9E2DEC96FA,
    "seed_range": (10000, 10500),
    "sets": 500,
    "games": 2000,
    "dist": [508, 504, 527, 461],
    "avg_rank": 2.4705,
    "pt": 3.0825,
    "ignored_files": 4,
    "challenger": "mortal",
    "champion": "old148800",
    "source": "runs/1v3/eval_20260915_1835.log",
}


class GateFailure(SystemExit):
    def __init__(self, msg):
        super().__init__(f"ASSERTION FAILED: {msg}")


def check(cond, msg):
    if not cond:
        raise GateFailure(msg)


def parse_seed_key(text):
    """`int(x, 0)`: 同时接受十进制与 0xHEX(与文件名内十进制整数比较)。"""
    try:
        return int(text, 0)
    except ValueError:
        raise SystemExit(f"--seed-key 需为 int(x, 0) 可解析的整数(如 0x80a7b9e2dec96fa): {text!r}")


def parse_seed_range(text):
    """`A-B` -> 半开区间 [A, B)。"""
    parts = text.split("-")
    if len(parts) != 2:
        raise SystemExit(f"--seed-range 需形如 A-B(半开区间 [A,B)): {text!r}")
    try:
        lo, hi = (int(p, 0) for p in parts)
    except ValueError:
        raise SystemExit(f"--seed-range 需形如 A-B(int(x,0)): {text!r}")
    if hi <= lo:
        raise SystemExit(f"--seed-range 上界须大于下界(半开 [A,B)): {text!r}")
    return (lo, hi)


def scan_candidates(games_dir):
    """返回 (entries, ignored, non_game_count)。

    entries: [(path, seed, key, split)] 通过过滤;
    ignored: [(name, reason)] `*.json.gz` 但被过滤(含无法解析文件名者);
    非 `*.json.gz` 文件(如 .log)不计入 ignored_files, 只计 non_game_count。
    """
    entries, ignored, non_game = [], [], 0
    for name in sorted(os.listdir(games_dir)):
        full = os.path.join(games_dir, name)
        if not os.path.isfile(full):
            continue
        if not name.endswith(".json.gz"):
            non_game += 1
            continue
        m = FILE_RE.match(name)
        if m is None:
            ignored.append((name, "文件名不符合 <seed>_<key>_<a|b|c|d>.json.gz"))
            continue
        entries.append((full, int(m["seed"]), int(m["key"]), m["split"]))
    return entries, ignored, non_game


def replay_one(path):
    """逐行解析 NDJSON 并重放分数; 返回 (seed, key, names, final_scores)。"""
    seed = key = names = None
    scores = None
    kyotaku = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)  # 逐行 —— 绝不对整文件 json.loads
            etype = ev["type"]
            if etype == "start_game":
                seed, key = ev["seed"]
                names = list(ev["names"])
            elif etype == "start_kyoku":
                scores = list(ev["scores"])
                kyotaku = int(ev["kyotaku"])
            elif etype == "reach_accepted":
                scores[ev["actor"]] -= 1000
                kyotaku += 1
            elif etype == "hora":
                deltas = ev.get("deltas") or [0, 0, 0, 0]
                scores = [s + d for s, d in zip(scores, deltas)]
                kyotaku = 0  # 供托被和牌者取走(board.rs:376-377)
            elif etype == "ryukyoku":
                deltas = ev.get("deltas") or [0, 0, 0, 0]
                scores = [s + d for s, d in zip(scores, deltas)]
            elif etype == "end_game":
                if kyotaku > 0:
                    # 终局供托补偿: 加给首个最高分者 (game.rs:182-184, min_by_key(-s) == argmax)
                    scores[scores.index(max(scores))] += kyotaku * 1000
                kyotaku = 0
    if seed is None or key is None or scores is None or names is None:
        raise GateFailure(f"{path}: 缺少 start_game/start_kyoku, 不是完整牌谱")
    return seed, key, names, scores


def rank_of(scores, seat):
    """名次 1..4, 分数降序的序数排名, 同分座位号小者在前(与 rankings.rs 稳定排序一致)。"""
    return sorted(range(4), key=lambda i: -scores[i]).index(seat) + 1


def bootstrap_ci(set_rank_means, set_pt_means, b=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """对 set 有放回重采样 b 次; 返回 (rank_lo, rank_hi, pt_lo, pt_hi)。"""
    if not set_rank_means:
        return None
    ranks = np.asarray(set_rank_means, dtype=float)
    pts = np.asarray(set_pt_means, dtype=float)
    n_sets = ranks.shape[0]
    rng = np.random.default_rng(seed)  # 固定 seed 0 -> 确定性
    idx = rng.integers(0, n_sets, size=(b, n_sets))  # b 次重采样(不是 b 个 set)
    rank_lo, rank_hi = np.percentile(ranks[idx].mean(axis=1), [2.5, 97.5])
    pt_lo, pt_hi = np.percentile(pts[idx].mean(axis=1), [2.5, 97.5])
    return float(rank_lo), float(rank_hi), float(pt_lo), float(pt_hi)


def analyze_key(key, seed_splits, source_dir, checkpoint, ignored_total):
    """解析一个 seed_key 下全部 set(每个 set = 同 seed 的 4 split), 返回 CSV 行 + arm 信息。"""
    ranks, pts = [], []
    set_rank_means, set_pt_means = [], []
    challenger_names, champion_names = Counter(), Counter()
    sets = 0

    for seed in sorted(seed_splits):
        split_paths = seed_splits[seed]
        sets += 1
        games_in_set = []
        for split in SPLITS:
            path = split_paths.get(split)
            if path is None:
                continue
            fseed, fkey, names, scores = replay_one(path)
            if (fseed, fkey) != (seed, key):
                raise GateFailure(
                    f"{path}: NDJSON seed={[fseed, fkey]} 与文件名 (seed={seed}, key={key}) 不一致"
                )
            seat = SEAT_OF_SPLIT[split]
            rank = rank_of(scores, seat)
            ranks.append(rank)
            pts.append(RANK_PTS[rank - 1])
            games_in_set.append(rank)
            challenger_names[names[seat]] += 1
            champion_names[tuple(sorted(names[:seat] + names[seat + 1 :]))] += 1
        if len(games_in_set) == 4:  # 只有完整 set 可参与 set 级重采样
            set_rank_means.append(sum(games_in_set) / 4)
            set_pt_means.append(sum(RANK_PTS[r - 1] for r in games_in_set) / 4)

    games = len(ranks)
    dist = [ranks.count(r) for r in (1, 2, 3, 4)]
    avg_rank = sum(r * n for r, n in zip((1, 2, 3, 4), dist)) / games
    pt = sum(p * n for p, n in zip(RANK_PTS, dist)) / games

    # 自洽断言(任何过滤条件/数据都成立)
    check(sum(dist) == games, f"分布 {dist} 不等于局数 {games}")
    check(games > 0, f"seed_key={key} 内 0 局")
    check(abs(avg_rank - sum(ranks) / games) < 1e-9, "avg_rank 与原始 ranks 不一致")
    check(abs(pt - sum(pts) / games) < 1e-9, "pt 与原始 pts 不一致")

    ci = bootstrap_ci(set_rank_means, set_pt_means)
    row = {
        "source_dir": source_dir,
        "checkpoint": checkpoint,
        "seed_key": key,
        "seed_start": min(seed_splits),
        "sets": sets,
        "games": games,
        "incomplete_sets": sets - len(set_rank_means),
        "rank1": dist[0],
        "rank2": dist[1],
        "rank3": dist[2],
        "rank4": dist[3],
        "rank_dist": "[%d %d %d %d]" % tuple(dist),
        "avg_rank": avg_rank,
        "avg_rank_ci_lo": ci[0] if ci else float("nan"),
        "avg_rank_ci_hi": ci[1] if ci else float("nan"),
        "pt": pt,
        "pt_ci_lo": ci[2] if ci else float("nan"),
        "pt_ci_hi": ci[3] if ci else float("nan"),
        "ignored_files": ignored_total,
    }
    arm = {
        "challenger": challenger_names.most_common(1)[0][0],
        "challenger_counter": challenger_names,
        "champion_counter": champion_names,
        "ci": ci,
        "bootstrap_sets": len(set_rank_means),
    }
    return row, arm


CSV_COLUMNS = (
    "source_dir",
    "checkpoint",
    "seed_key",
    "seed_start",
    "sets",
    "games",
    "incomplete_sets",
    "rank1",
    "rank2",
    "rank3",
    "rank4",
    "rank_dist",
    "avg_rank",
    "avg_rank_ci_lo",
    "avg_rank_ci_hi",
    "pt",
    "pt_ci_lo",
    "pt_ci_hi",
    "ignored_files",
)


def parse_args(argv):
    ap = argparse.ArgumentParser(description="1v3 NDJSON 牌谱 -> set 级 bootstrap 报告")
    ap.add_argument("--games-dir", required=True, help="含 *.json.gz 牌谱的目录")
    ap.add_argument("--seed-key", help="seed_key 过滤, int(x,0), 如 0x80a7b9e2dec96fa; 省略则按 key 分组全报告")
    ap.add_argument("--seed-range", help="半开 seed 区间 A-B, 如 10000-10500")
    ap.add_argument("--checkpoint", default="-", help="checkpoint 标签(写入 CSV)")
    ap.add_argument("--source-dir", help="来源目录标签(默认即 --games-dir)")
    ap.add_argument("--out", default="report.csv", help="输出 report.csv 路径(默认 ./report.csv)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    games_dir = args.games_dir
    source_dir = args.source_dir or games_dir
    key_filter = parse_seed_key(args.seed_key) if args.seed_key is not None else None
    range_filter = parse_seed_range(args.seed_range) if args.seed_range else None

    check(os.path.isdir(games_dir), f"--games-dir 不存在: {games_dir}")

    entries, ignored, non_game = scan_candidates(games_dir)
    matched = []
    for path, seed, key, split in entries:
        if key_filter is not None and key != key_filter:
            ignored.append((os.path.basename(path), f"seed_key {key} != {key_filter}"))
            continue
        if range_filter is not None and not (range_filter[0] <= seed < range_filter[1]):
            ignored.append(
                (os.path.basename(path), f"seed {seed} 不在 [{range_filter[0]},{range_filter[1]})")
            )
            continue
        matched.append((path, seed, key, split))
    ignored.sort()

    print("=== eval_report ===")
    print(f"games_dir={games_dir}  source_dir={source_dir}  checkpoint={args.checkpoint}")
    key_desc = f"{key_filter} (0x{key_filter:x})" if key_filter is not None else "ALL"
    range_desc = f"[{range_filter[0]},{range_filter[1]}) half-open" if range_filter else "ALL"
    print(f"filter: seed_key={key_desc}  seed_range={range_desc}")
    print(
        f"candidates(*.json.gz)={len(entries)} matched={len(matched)} "
        f"ignored_files={len(ignored)} non-game files skipped={non_game}"
    )
    for name, reason in ignored[:10]:
        print(f"  ignored: {name}  ({reason})")
    if len(ignored) > 10:
        print(f"  ... {len(ignored) - 10} more ignored")
    check(len(matched) > 0, "过滤后 0 局, 检查 --games-dir/--seed-key/--seed-range")

    groups = {}  # seed_key -> seed -> {split: path}
    for path, seed, key, split in matched:
        groups.setdefault(key, {}).setdefault(seed, {})[split] = path
    incomplete_total = sum(
        1 for seed_splits in groups.values() for sp in seed_splits.values() if len(sp) < 4
    )

    rows, arms = [], []
    for key, seed_splits in sorted(groups.items()):
        row, arm = analyze_key(key, seed_splits, source_dir, args.checkpoint, len(ignored))
        rows.append(row)
        arms.append(arm)

    total_sets = sum(r["sets"] for r in rows)
    total_games = sum(r["games"] for r in rows)
    print(
        f"sets={total_sets} games={total_games} incomplete_sets={incomplete_total} "
        f"seed_key_groups={len(rows)}"
    )
    for row, arm in zip(rows, arms):
        tag = f"seed_key={row['seed_key']} seed_start={row['seed_start']}: "
        print(f"{tag}dist(1/2/3/4)={row['rank_dist']} avg_rank={row['avg_rank']:.4f} pt={row['pt']:+.4f}")
        ci = arm["ci"]
        if ci is not None:
            print(
                f"  bootstrap B={BOOTSTRAP_RESAMPLES} (resample sets, rng seed={BOOTSTRAP_SEED}, "
                f"bootstrap_sets={arm['bootstrap_sets']}): "
                f"avg_rank 95%CI=[{ci[0]:.4f}, {ci[1]:.4f}]  pt 95%CI=[{ci[2]:+.4f}, {ci[3]:+.4f}]"
            )
        else:
            print(f"  bootstrap B={BOOTSTRAP_RESAMPLES}: 无完整 set, CI n/a")
        for name, cnt in arm["challenger_counter"].items():
            print(f"  arm: challenger='{name}' x{cnt} vs champion={dict(arm['champion_counter'])}")

    # 精确复现门: 仅当调用完全匹配 ground-truth 条件时启用
    gate = EXACT_GATE
    gate_applies = (
        os.path.basename(os.path.abspath(games_dir)) == gate["dir_basename"]
        and key_filter == gate["seed_key"]
        and range_filter == gate["seed_range"]
    )
    if gate_applies:
        check(len(rows) == 1, f"精确门预期 1 个 group, 实际 {len(rows)}")
        row, arm = rows[0], arms[0]
        check(total_sets == gate["sets"], f"sets={total_sets} != {gate['sets']}")
        check(total_games == gate["games"], f"games={total_games} != {gate['games']}")
        check(row["rank_dist"] == "[%d %d %d %d]" % tuple(gate["dist"]), f"dist={row['rank_dist']} != {gate['dist']}")
        check(round(row["avg_rank"], 4) == gate["avg_rank"], f"avg_rank={row['avg_rank']:.6f} != {gate['avg_rank']}")
        check(round(row["pt"], 4) == gate["pt"], f"pt={row['pt']:.6f} != {gate['pt']}")
        check(len(ignored) == gate["ignored_files"], f"ignored_files={len(ignored)} != {gate['ignored_files']}")
        check(arm["challenger"] == gate["challenger"], f"challenger={arm['challenger']} != {gate['challenger']}")
        champ_names = {n for names, _ in arm["champion_counter"].items() for n in names}
        check(champ_names == {gate["champion"]}, f"champion={champ_names} != {{{gate['champion']}}}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{row[k]:.6f}" if k in ("avg_rank", "pt") else row[k]) for k in CSV_COLUMNS})

    print("assertions: self-consistency OK")
    if gate_applies:
        print(
            f"assertions: EXACT GATE OK (arm challenger='{gate['challenger']}' live vs "
            f"champion='{gate['champion']}'; source {gate['source']})"
        )
    else:
        print("assertions: exact gate not applicable for this invocation")
    print(f"CSV -> {args.out} ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
