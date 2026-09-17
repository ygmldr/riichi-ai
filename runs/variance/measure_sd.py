#!/usr/bin/env python3
"""方差标定复算工具 —— set 级 SD 与"给定 n 局可分辨效应"表(todo 6)。

两种模式:
  A. --from-csv <report.csv>: 仅从 eval_report.py 产出的 CSV 复算(CI 反推 SE);
  B. --games-dir <dir>: 直接重放牌谱, 精确计算 set 级样本 SD(ddof=1)。

统计口径(与 scripts/eval_report.py 一致):
  - 1 set = 同一 (seed, seed_key) 的 4 个 split(a/b/c/d), 即 4 局;
  - set 级统计量 = 该 set 4 局的均值(avg_rank 或 RANK_PTS 得分);
  - 独立重采样单元是 set, 所以 n 局 = n/4 个 set。

公式(README 中同步声明):
  σ_set  = set 级统计量的样本 SD(每 set 已对 4 局取均值)
  SE(n)  = σ_set / sqrt(n/4) = 2·σ_set / sqrt(n)          # n 局均值的标准误
  ±95%   = z_{0.975} · SE(n),  z_{0.975} = 1.959964       # 95% 置信半宽
  Δ_MDD  = (z_{0.975} + z_{0.8416}) · SE(n) = 2.801585 · SE(n)   # 80% 功效双侧最小可分辨效应
CSV 反推(模式 A): bootstrap 百分位区间宽度 ≈ 2·z·SE(sets),
  故 σ_set ≈ (ci_hi - ci_lo)·sqrt(sets) / (2·z)。

用法:
  python3 runs/variance/measure_sd.py --from-csv runs/variance/pool_1/report.csv
  python3 runs/variance/measure_sd.py --pooled-csv runs/variance/pool_1/report.csv \
      runs/variance/pool_2/report.csv runs/variance/pool_3/report.csv
  python3 runs/variance/measure_sd.py --games-dir runs/variance/pool_1/games \
      --seed-key 0x5dfaa4cef265cd7 --seed-range 10000-13000
"""

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import eval_report  # noqa: E402  (scripts/eval_report.py, 只读复用)

Z95 = 1.959964  # z_{0.975}
Z80 = 0.841621  # z_{0.80}
NS = (6100, 12000, 24000)  # 计划要求的 n 局


def table(sigma_set):
    """返回 [(n, se, half95, mdd80), ...]。sigma_set 为 set 级(4 局均值)样本 SD。"""
    rows = []
    for n in NS:
        se = 2.0 * sigma_set / (n**0.5)
        rows.append((n, se, Z95 * se, (Z95 + Z80) * se))
    return rows


def print_table(label, sigma_set):
    print(f"\n[{label}] sigma_set={sigma_set:.6f}")
    print("| n games | SE(n) | ±95% CI | MDD(80% power) |")
    print("|---|---|---|---|")
    for n, se, half, mdd in table(sigma_set):
        print(f"| {n} | {se:.5f} | ±{half:.5f} | {mdd:.5f} |")


def from_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sets = int(row["sets"])
            for stat, lo_key, hi_key in (
                ("avg_rank", "avg_rank_ci_lo", "avg_rank_ci_hi"),
                ("pt", "pt_ci_lo", "pt_ci_hi"),
            ):
                lo, hi = float(row[lo_key]), float(row[hi_key])
                # bootstrap 百分位区间 ≈ mean ± z·SE, SE = σ_set/sqrt(sets)
                sigma = (hi - lo) * (sets**0.5) / (2.0 * Z95)
                print(
                    f"{path}: set={row.get('source_dir', '?')} sets={sets} "
                    f"stat={stat} mean={row[stat]} ci=[{lo:.4f}, {hi:.4f}] "
                    f"-> sigma_set≈{sigma:.6f}"
                )
                print_table(f"{os.path.basename(path)} {stat}", sigma)


def from_games(games_dir, seed_key, seed_range):
    entries, ignored, non_game = eval_report.scan_candidates(games_dir)
    key = eval_report.parse_seed_key(seed_key) if seed_key else None
    rng = eval_report.parse_seed_range(seed_range) if seed_range else None
    by_seed = {}
    for path, seed, k, split in entries:
        if key is not None and k != key:
            continue
        if rng is not None and not (rng[0] <= seed < rng[1]):
            continue
        by_seed.setdefault(seed, {})[split] = path

    set_ranks, set_pts, games = [], [], 0
    for seed, splits in sorted(by_seed.items()):
        rs = []
        for split in eval_report.SPLITS:
            path = splits.get(split)
            if path is None:
                continue
            _, _, _, scores = eval_report.replay_one(path)
            rank = eval_report.rank_of(scores, eval_report.SEAT_OF_SPLIT[split])
            rs.append(rank)
        games += len(rs)
        if len(rs) == 4:
            set_ranks.append(sum(rs) / 4)
            set_pts.append(sum(eval_report.RANK_PTS[r - 1] for r in rs) / 4)

    sets = len(set_ranks)
    print(
        f"{games_dir}: sets={sets} games={games} ignored={len(ignored)} non_game={non_game} "
        f"seed_key={seed_key} seed_range={seed_range}"
    )
    if sets < 2:
        raise SystemExit("set 数不足, 无法计算 SD")
    for label, vals in (("avg_rank", set_ranks), ("pt", set_pts)):
        sd = statistics.stdev(vals)  # ddof=1
        se = sd / (sets**0.5)
        print(f"{label}: mean={statistics.mean(vals):.6f} sd_set={sd:.6f} se_mean={se:.6f}")
        print_table(f"{os.path.basename(games_dir)} {label}", sd)


def pooled_from_csv(paths):
    """多个池的 CSV -> 逐池 CI 反推 σ -> 合并 σ = sqrt(mean(σ_i^2)) -> 可分辨效应表。

    合并口径: 三个池相互独立、各自 3000 sets, 故用等权 RMS(方差的算术平均)。
    """
    for stat, lo_key, hi_key in (
        ("avg_rank", "avg_rank_ci_lo", "avg_rank_ci_hi"),
        ("pt", "pt_ci_lo", "pt_ci_hi"),
    ):
        sigmas = []
        for path in paths:
            with open(path, newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            sets = int(row["sets"])
            lo, hi = float(row[lo_key]), float(row[hi_key])
            sigma = (hi - lo) * (sets**0.5) / (2.0 * Z95)
            sigmas.append(sigma)
            print(f"  {path}: sets={sets} stat={stat} mean={row[stat]} ci=[{lo:.4f}, {hi:.4f}] sigma≈{sigma:.6f}")
        pooled = (sum(s * s for s in sigmas) / len(sigmas)) ** 0.5
        print(f"pooled sigma_set({stat}) = sqrt(mean(sigma_i^2)) = {pooled:.6f}")
        print_table(f"pooled {len(paths)} pools {stat}", pooled)
        print("READ: n = (z95+z80)^2 * 4 * sigma^2 / Delta^2 = "
              f"{(Z95 + Z80) ** 2 * 4 * pooled * pooled:.4f} / Delta^2")


def main():
    ap = argparse.ArgumentParser(description="set 级 SD 与可分辨效应表复算")
    ap.add_argument("--from-csv")
    ap.add_argument("--pooled-csv", nargs="+")
    ap.add_argument("--games-dir")
    ap.add_argument("--seed-key")
    ap.add_argument("--seed-range")
    args = ap.parse_args()
    if args.pooled_csv:
        pooled_from_csv(args.pooled_csv)
    elif args.from_csv:
        from_csv(args.from_csv)
    elif args.games_dir:
        from_games(args.games_dir, args.seed_key, args.seed_range)
    else:
        ap.error("需要 --from-csv / --pooled-csv / --games-dir")


if __name__ == "__main__":
    main()
