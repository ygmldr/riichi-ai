#!/usr/bin/env python3
"""解析 1v3 评估日志, 输出顺位分对比。

用法: python3 scripts/parse_eval.py <eval.log> [--akochan]

日志来源: one_vs_three.py 每个 iter 输出一行:
    challenger rankings: [508 504 527 461] (2.4705, 3.0825pt)
四元组依次为 challenger 取得 1/2/3/4 名的局数。

默认: challenger=我方模型, champion=对比模型(baseline/旧快照)。
--akochan: 日志来自 akochan 基准评估(akochan 坐 challenger x1, 我方模型坐 3 个
champion 席位); 输出改按此视角, 以"我方每席位均分 > 0"为超过 akochan 的判据。
"""
import math
import re
import sys

RANK_PTS = (90, 45, 0, -135)
LINE_RE = re.compile(
    r"challenger rankings:\s*\[(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]\s*"
    r"\(([-\d.]+),\s*([-\d.]+)pt\)"
)


def main():
    argv = sys.argv[1:]
    akochan_mode = "--akochan" in argv
    args = [a for a in argv if not a.startswith("--")]
    log_path = args[0] if args else "runs/1v3/eval.log"
    iters = []
    with open(log_path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                iters.append([int(m.group(i)) for i in range(1, 5)])

    if not iters:
        sys.exit(f"未在 {log_path} 中找到 'challenger rankings' 行(确认日志来自 one_vs_three.py)")

    total = [sum(col) for col in zip(*iters)]
    games = sum(total)
    pit = sum(n * pts for n, pts in zip(total, RANK_PTS)) / games
    rank = sum(n * r for n, r in zip(total, (1, 2, 3, 4))) / games

    # 零和换算: 每局 1 个 challenger 席位 + 3 个 champion 席位, 顺位分/顺位在两侧互补
    cham_pit = -pit / 3
    cham_rank = (10 - rank) / 3
    delta = pit - cham_pit

    print(f"iters={len(iters)} 局数={games} 分布(1/2/3/4 名)={total}")
    if akochan_mode:
        print(f"akochan(挑战者席位): 均顺位 {rank:.4f} | 均顺位分 {pit:+.4f}")
        print(f"我方模型(3 席位)   : 均顺位 {cham_rank:.4f} | 均顺位分 {cham_pit:+.4f}")
        verdict = "我方强于 akochan(验收通过)" if cham_pit > 0 else "我方暂未超过 akochan"
        print(f"我方每席位均分 {cham_pit:+.4f} (>0 即强于 akochan) | {verdict}")
        print("参照: 上游 Mortal 3.1 时 akochan 每席位 -5.67pt(v3.1 +1.89); v4 推算 -8~-9pt")
    else:
        print(f"challenger: 均顺位 {rank:.4f} | 均顺位分 {pit:+.4f}")
        print(f"champion  : 均顺位 {cham_rank:.4f} | 均顺位分 {cham_pit:+.4f}")
        verdict = "challenger 更强(验收通过)" if delta > 0 else "challenger 未超过 champion"
        print(f"顺位分差: {delta:+.4f} | {verdict}")

    # 粗略噪声参考(按局独立近似, 未校正 duplicate 相关性)
    var = sum(n * (p - pit) ** 2 for n, p in zip(total, RANK_PTS)) / games
    se = math.sqrt(var / games) * 4 / 3
    print(f"(粗略 95% CI: {delta - 1.96 * se:+.2f} ~ {delta + 1.96 * se:+.2f}; 局独立近似, 仅供参考)")


if __name__ == "__main__":
    main()
