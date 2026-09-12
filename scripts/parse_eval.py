#!/usr/bin/env python3
"""解析 1v3 duplicate mahjong 评估日志,输出 challenger / champion 顺位分对比。

用法: python3 scripts/parse_eval.py <eval.log> [eval2.log ...]
默认(无参数): runs/1v3/eval.log

数据来源: one_vs_three.py 每轮迭代打印一行(challenger 的顺位分布):
    challenger rankings: [n1 n2 n3 n4] (avg_rank, avg_pt)
其中 n1..n4 = 该轮 challenger 拿到 1st..4th 的局数(每 4 局使用同一副牌,
即 duplicate mahjong,消除运气因素)。

champion 说明: stdout 只打印 challenger 的分布。由于每局 4 个座位恰好占据
1st..4th 各一次,champion(每局 3 个座位)在 rank r 的局数 = 总局数 - challenger
在 rank r 的局数,故 champion 的均顺位分可由 challenger 分布精确反推。

顺位分(jun_pt,与 Mortal 文档一致): 1st=90, 2nd=45, 3rd=0, 4th=-135。
challenger 相对 champion 的均顺位分 > 0 => challenger 更强(验收标准)。
"""
import re
import sys

PT = {"1st": 90, "2nd": 45, "3rd": 0, "4th": -135}
RANKS = ["1st", "2nd", "3rd", "4th"]

# one_vs_three.py 真实输出格式:
#   challenger rankings: [45 32 11 12] (1.63, 25.2pt)
CHALLENGER_RE = re.compile(
    r"challenger\s+rankings:\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]"
)


def main():
    paths = sys.argv[1:] or ["runs/1v3/eval.log"]
    counts = [0, 0, 0, 0]  # challenger 累计顺位局数: 1st..4th
    total = 0              # challenger 累计总局数

    for path in paths:
        try:
            f = open(path, encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"跳过无法读取的日志: {path} ({e})")
            continue
        with f:
            for line in f:
                m = CHALLENGER_RE.search(line)
                if not m:
                    continue
                row = [int(m.group(i + 1)) for i in range(4)]
                for i in range(4):
                    counts[i] += row[i]
                total += sum(row)

    if total == 0:
        print(f"未在 {', '.join(paths)} 中找到 challenger rankings 行(共 {total} 局)。")
        sys.exit(1)

    chal_pts = sum(counts[i] * PT[RANKS[i]] for i in range(4))
    chal_avg = chal_pts / total

    # champion 组(3 座位/局)分布: rank r 局数 = 总局数 - challenger 在 r 的局数
    cham_counts = [total - c for c in counts]
    cham_total = total * 3
    cham_pts = sum(cham_counts[i] * PT[RANKS[i]] for i in range(4))
    cham_avg = cham_pts / cham_total

    print(f"challenger(mortal): {total} 局, 均顺位分 {chal_avg:+.3f}")
    print(f"champion(3 座位/局): {cham_total} 座位局, 均顺位分 {cham_avg:+.3f}")
    print(f"  顺位分布 challenger: 1st={counts[0]} 2nd={counts[1]} 3rd={counts[2]} 4th={counts[3]}")
    print(f"  顺位分布 champion  : 1st={cham_counts[0]} 2nd={cham_counts[1]} 3rd={cham_counts[2]} 4th={cham_counts[3]}")

    delta = chal_avg - cham_avg
    verdict = "challenger 更强(顺位分转正)" if delta > 0 else "challenger 尚未超过 champion"
    print(f"顺位分差(challenger - champion): {delta:+.3f} | {verdict}")


if __name__ == "__main__":
    main()
