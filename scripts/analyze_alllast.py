#!/usr/bin/env python3
"""#111 quantification（只测不修）: all-last 超车相关统计。

用法:
    python3 scripts/analyze_alllast.py runs/1v3 [runs/other ...] [--out PATH]

输入: 显式来源目录, 每个 `*.json.gz` 是逐行 NDJSON(mjai 事件流, 一行一个 JSON)。
输出: `runs/analysis/alllast.json`(按来源目录分组) + stdout 摘要。

回答 docs/FORK_ROADMAP.md §A.3「档 0 量化」:
  ① all-last 检测: 存在 `bakaze=="S" && kyoku==4` 的局;
  ② all-last 终局四家分数(事件重放 + 供托补偿);
  ③ 末局和牌者 / 放铳者 / 和牌者 deltas;
  ④ 指标 (a): all-last 结束时的分差分布;
  ⑤ 指标 (b): 末局由"赛前第 4 名"和了的局中, 和了后仍是第 4 的比例。

歧义处理: 1576 个 all-last 局里有 71 个在 S4 后进入西入(西1..西4), 此时"牌谱末局"
不是 S4。两种口径都产出, 用名字区分(measure only, 不下结论):
  * `metrics` / `last_round` / `final_scores`: 牌谱末局与终局分(= 任务 ②③ 字面口径);
  * `metrics_at_all_last_round_end` / `all_last_round` / `all_last_end_scores`:
    最后一个 S4 局结束时(供托仅在该局真的终局时才结算);
  * `metrics_subset_final_round_is_s4` / `metrics_subset_went_to_west`: 子集敏感性。

重放规则(依据 vendor/mortal/libriichi pin 0cff2b5):
  * `start_kyoku.scores` 是每局开局的权威分数;
  * `reach_accepted` 扣 1000 点(不体现在任何 deltas 里) → 必须显式扣;
  * `hora.deltas` 已含全部供托(余额×1000)给和牌者; `ryukyoku.deltas` 不含
    (供托留存到下一局);
  * 终局若仍有供托, `arena/game.rs:182-184` 把它加给最高分者(同分取小座位号);
  * 顺位 = libriichi `Rankings::new`: 按 `-score` 稳定排序(同分小座位号在前)。

`scripts/parse_eval.py` 解析的是 eval.log 文本行(顺位分), 与事件流重放无关,
故此处不共用; 顺位序与 libriichi 保持一致以免口径分歧。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "runs/analysis/alllast.json")
DEFAULT_PATTERN = "*.json.gz"
SELF_CHECK_REL = "runs/1v3/10037_579411421376059130_a.json.gz"
SELF_CHECK_LAST_ROUND = {"bakaze": "S", "kyoku": 4, "honba": 1}
SELF_CHECK_HORAS = [{"actor": 1, "target": 3, "deltas": [0, 2300, 0, -2300]}]
SELF_CHECK_FINAL = [18900, 39700, 8800, 32600]
SMALL_SAMPLE_THRESHOLD = 500
GAP_THRESHOLDS = (1000, 2000, 3900, 8000, 12000, 16000, 24000)
FILENAME_RE = re.compile(r"^(\d+)_(\d+)_([a-z])\.json\.gz$")
DEFINITIONS = {
    "all_last_game": 'log contains a round with bakaze == "S" and kyoku == 4',
    "last_round": "the final round in the log; for 71/1576 all-last games this is a West round (西入)",
    "all_last_round": "the last bakaze == S / kyoku == 4 round; equals last_round when the game ended there",
    "rank": "libriichi Rankings::new: stable sort by -score; ties -> lower seat index ranks higher",
    "final_scores": (
        "last start_kyoku.scores - 1000*reach_accepted + all hora/ryukyoku deltas "
        "(+ leftover kyotaku*1000 to the top-score player when the game ends, per arena/game.rs:182-184)"
    ),
    "all_last_end_scores": (
        "same replay applied to the last S4 round; leftover pot is awarded only when that round ended the game"
    ),
    "pot_resolved": (
        "riichi/pot count resolved by a round (kyotaku_start + reach_accepted in that round, 0 after any hora); "
        "paid to the winner via hora.deltas, or to the top-score player if the round ended the game"
    ),
    "metric_a": "distribution of score gaps at the moment all-last ends (adjacent ranks and each rank's gap to 1st)",
    "metric_b": (
        "among all-last games whose last hand was won by the player who was 4th at the "
        "last round's start scores, fraction that still finished 4th"
    ),
}


class DecodeError(Exception):
    """A source file could not be decoded as a valid mjai NDJSON game log."""


def decode_events(path: str):
    """Yield mjai events from a .json.gz NDJSON file, line by line (never whole-file)."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DecodeError(f"line {lineno}: invalid JSON: {exc}") from exc
                if not isinstance(ev, dict) or not isinstance(ev.get("type"), str):
                    raise DecodeError(f"line {lineno}: not an mjai event object")
                yield ev
    except (OSError, EOFError, UnicodeDecodeError, ValueError) as exc:
        raise DecodeError(f"cannot read gzip: {exc}") from exc


def _int_field(ev: dict, key: str, where: str, lo: int = 0, hi: int = 3) -> int:
    val = ev.get(key)
    if not isinstance(val, int) or not lo <= val <= hi:
        raise DecodeError(f"{where}: bad field {key}={val!r}")
    return val


def _deltas_field(ev: dict, where: str) -> list:
    val = ev.get("deltas")
    if not isinstance(val, list) or len(val) != 4 or not all(isinstance(x, int) for x in val):
        raise DecodeError(f"{where}: bad deltas={val!r}")
    return list(val)


def rank_by_player(scores) -> list:
    """1-based rank per seat, matching libriichi Rankings (ties -> lower seat index)."""
    order = sorted(range(4), key=lambda i: (-scores[i], i))
    ranks = [0] * 4
    for rank, seat in enumerate(order, 1):
        ranks[seat] = rank
    return ranks


def player_by_rank(scores) -> list:
    """Seat indices ordered 1st -> 4th, matching libriichi Rankings."""
    return sorted(range(4), key=lambda i: (-scores[i], i))


def _round_deltas_sum(round_: dict) -> list:
    total = [0, 0, 0, 0]
    for hora in round_["horas"]:
        for i in range(4):
            total[i] += hora["deltas"][i]
    if round_["ryukyoku"] is not None:
        for i in range(4):
            total[i] += round_["ryukyoku"][i]
    return total


def replay_round(round_: dict) -> list:
    """Scores after a round, excluding any leftover pot (next start_kyoku is the truth)."""
    scores = list(round_["start_scores"])
    for seat in range(4):
        scores[seat] -= 1000 * round_["reach_accepted"][seat]
    deltas = _round_deltas_sum(round_)
    return [scores[i] + deltas[i] for i in range(4)]


def resolve_round(round_: dict, award_pot: bool) -> dict:
    """Scores after a round; `award_pot` mirrors game.rs's end-of-game pot compensation."""
    scores = replay_round(round_)
    pot = 0 if round_["horas"] else round_["kyotaku"] + sum(round_["reach_accepted"])
    awarded_to = None
    if pot > 0 and award_pot:
        awarded_to = max(range(4), key=lambda i: (scores[i], -i))
        scores[awarded_to] += 1000 * pot
    return {
        "scores": scores,
        "round_deltas": _round_deltas_sum(round_),
        "pot_resolved": pot,
        "pot_awarded": pot if award_pot else 0,
        "pot_awarded_to": awarded_to,
    }


def round_view(round_: dict, resolved: dict, scores: list) -> dict:
    """Compact public view of one round (scores = resolved final scores)."""
    horas = round_["horas"]
    winner = horas[0]["actor"] if horas else None
    rank_before = rank_by_player(round_["start_scores"])
    return {
        "label": f"{round_['bakaze']}{round_['kyoku']}",
        "bakaze": round_["bakaze"],
        "kyoku": round_["kyoku"],
        "honba": round_["honba"],
        "oya": round_["oya"],
        "kyotaku_start": round_["kyotaku"],
        "start_scores": round_["start_scores"],
        "rank_before_by_player": rank_before,
        "reach_accepted": round_["reach_accepted"],
        "outcome": "hora" if horas else ("ryukyoku" if round_["ryukyoku"] is not None else "none"),
        "horas": horas,
        "multi_ron": len(horas) > 1,
        "ryukyoku_deltas": round_["ryukyoku"],
        "round_deltas": resolved["round_deltas"],
        "pot_resolved": resolved["pot_resolved"],
        "pot_awarded": resolved["pot_awarded"],
        "pot_awarded_to": resolved["pot_awarded_to"],
        "winner": winner,
        "winner_rank_before": rank_before[winner] if winner is not None else None,
        "winner_rank_after": rank_by_player(scores)[winner] if winner is not None else None,
    }


def metric_view(scores: list, round_data: dict) -> dict:
    return {
        "final_scores": scores,
        "player_by_rank": player_by_rank(scores),
        "rank_by_player": rank_by_player(scores),
        "last_round": round_data,
    }


def analyze_stream(path: str, events) -> dict:
    """Decode one game log into a compact record (single pass, no event retention)."""
    game = {
        "file": os.path.basename(path),
        "seed": None,
        "key": None,
        "split": None,
        "names": None,
        "n_rounds": 0,
        "filename_mismatch": False,
    }
    match = FILENAME_RE.match(game["file"])
    file_seed, file_key, file_split = (
        (int(match.group(1)), int(match.group(2)), match.group(3)) if match else (None, None, None)
    )
    game["split"] = file_split
    rounds = []
    cur = None
    closed = None  # last finalized round's replayed scores, cross-checked vs the next start_kyoku
    closed_round_index = 0
    mismatches = []
    all_last_round = None
    n_start_game = 0
    n_end_game = 0

    for ev in events:
        etype = ev["type"]
        if etype == "start_game":
            n_start_game += 1
            names = ev.get("names")
            game["names"] = names if isinstance(names, list) and len(names) == 4 else None
            seed = ev.get("seed")
            if isinstance(seed, list) and len(seed) == 2:
                game["seed"], game["key"] = seed[0], seed[1]
            if game["seed"] is None:
                game["seed"], game["key"] = file_seed, file_key
            game["filename_mismatch"] = file_seed is not None and (
                file_seed != game["seed"] or file_key != game["key"]
            )
        elif etype == "start_kyoku":
            if cur is not None:
                raise DecodeError("start_kyoku inside an open round")
            scores = ev.get("scores")
            if not isinstance(scores, list) or len(scores) != 4:
                raise DecodeError(f"start_kyoku: bad scores={scores!r}")
            honba = ev.get("honba")
            kyotaku = ev.get("kyotaku")
            if not isinstance(honba, int) or not isinstance(kyotaku, int):
                raise DecodeError(f"start_kyoku: bad honba/kyotaku={honba!r}/{kyotaku!r}")
            if closed is not None and list(scores) != closed and len(mismatches) < 5:
                mismatches.append({"round": closed_round_index, "expected": list(scores), "derived": list(closed)})
            cur = {
                "bakaze": ev.get("bakaze"),
                "kyoku": _int_field(ev, "kyoku", "start_kyoku", 1, 4),
                "honba": honba,
                "kyotaku": kyotaku,
                "oya": _int_field(ev, "oya", "start_kyoku"),
                "start_scores": list(scores),
                "reach_accepted": [0, 0, 0, 0],
                "horas": [],
                "ryukyoku": None,
            }
        elif etype == "reach_accepted":
            if cur is None:
                raise DecodeError("reach_accepted outside a round")
            cur["reach_accepted"][_int_field(ev, "actor", "reach_accepted")] += 1
        elif etype == "hora":
            if cur is None:
                raise DecodeError("hora outside a round")
            cur["horas"].append(
                {
                    "actor": _int_field(ev, "actor", "hora"),
                    "target": _int_field(ev, "target", "hora"),
                    "deltas": _deltas_field(ev, "hora"),
                }
            )
        elif etype == "ryukyoku":
            if cur is not None:
                cur["ryukyoku"] = _deltas_field(ev, "ryukyoku")
        elif etype == "end_kyoku":
            if cur is None:
                raise DecodeError("end_kyoku without an open round")
            if cur["bakaze"] == "S" and cur["kyoku"] == 4:
                all_last_round = cur
            rounds.append(cur)
            closed_round_index = len(rounds)
            closed = replay_round(cur)
            cur = None
        elif etype == "end_game":
            n_end_game += 1

    if cur is not None:
        raise DecodeError("unclosed round at end of stream")
    if n_start_game != 1 or n_end_game != 1:
        raise DecodeError(f"expected exactly 1 start_game/end_game, got {n_start_game}/{n_end_game}")
    if not rounds:
        raise DecodeError("no completed rounds")

    game["n_rounds"] = len(rounds)
    game["mismatch_count"] = len(mismatches)
    game["mismatch_examples"] = mismatches
    game["all_last"] = all_last_round is not None

    last = rounds[-1]
    last_resolved = resolve_round(last, award_pot=True)
    game["final_scores"] = last_resolved["scores"]
    game["last_round"] = round_view(last, last_resolved, game["final_scores"])
    game["view_final"] = metric_view(game["final_scores"], game["last_round"])

    if all_last_round is None:
        game["all_last_round"] = None
        game["all_last_end_scores"] = None
        game["view_all_last_end"] = None
        game["went_to_west"] = None
    else:
        ended_there = all_last_round is last
        s4_resolved = last_resolved if ended_there else resolve_round(all_last_round, award_pot=False)
        game["all_last_end_scores"] = s4_resolved["scores"]
        game["all_last_round"] = (
            game["last_round"] if ended_there else round_view(all_last_round, s4_resolved, s4_resolved["scores"])
        )
        game["view_all_last_end"] = metric_view(game["all_last_end_scores"], game["all_last_round"])
        game["went_to_west"] = not ended_there
    return game


def analyze_file(path: str) -> dict:
    return analyze_stream(path, decode_events(path))


def _percentile(sorted_vals: list, q: float) -> float:
    """Linear-interpolation percentile (numpy default convention)."""
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = (n - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))


def _stats(values: list) -> dict:
    if not values:
        return {"n": 0}
    vals = sorted(values)
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 2),
        "min": vals[0],
        "p10": round(_percentile(vals, 0.10), 1),
        "p25": round(_percentile(vals, 0.25), 1),
        "p50": round(_percentile(vals, 0.50), 1),
        "p75": round(_percentile(vals, 0.75), 1),
        "p90": round(_percentile(vals, 0.90), 1),
        "p99": round(_percentile(vals, 0.99), 1),
        "max": vals[-1],
    }


def _frac(numerator: int, denominator: int) -> dict:
    return {
        "n": numerator,
        "d": denominator,
        "fraction": round(numerator / denominator, 6) if denominator else None,
    }


def aggregate(views: list) -> dict:
    """Aggregate metric views (each: final_scores / player_by_rank / rank_by_player / last_round)."""
    n_games = len(views)
    if n_games == 0:
        return {"n_all_last_games": 0, "note": "no games in this slice"}
    gap_to_first = {2: [], 3: [], 4: []}
    adjacent = {"1st-2nd": [], "2nd-3rd": [], "3rd-4th": []}
    n_hora = n_ryukyoku = n_none = n_multi_ron = 0
    winner_rank_before = {1: 0, 2: 0, 3: 0, 4: 0}
    winner_rank_after = {1: 0, 2: 0, 3: 0, 4: 0}
    n_winner_finished_4th = 0
    n_winner_4th_before = n_winner_4th_before_still_4th = 0
    n_4th_before_finished_4th = 0
    transition_4th_before = {1: 0, 2: 0, 3: 0, 4: 0}
    for view in views:
        scores = view["final_scores"]
        order = view["player_by_rank"]
        for rank in (2, 3, 4):
            gap_to_first[rank].append(scores[order[0]] - scores[order[rank - 1]])
        adjacent["1st-2nd"].append(scores[order[0]] - scores[order[1]])
        adjacent["2nd-3rd"].append(scores[order[1]] - scores[order[2]])
        adjacent["3rd-4th"].append(scores[order[2]] - scores[order[3]])
        last = view["last_round"]
        if last["outcome"] == "hora":
            n_hora += 1
        elif last["outcome"] == "ryukyoku":
            n_ryukyoku += 1
        else:
            n_none += 1
        if last["multi_ron"]:
            n_multi_ron += 1
        fourth_before = last["rank_before_by_player"].index(4)
        transition_4th_before[view["rank_by_player"][fourth_before]] += 1
        if view["rank_by_player"][fourth_before] == 4:
            n_4th_before_finished_4th += 1
        if last["winner"] is not None:
            winner_rank_before[last["winner_rank_before"]] += 1
            winner_rank_after[last["winner_rank_after"]] += 1
            if last["winner_rank_after"] == 4:
                n_winner_finished_4th += 1
            if last["winner_rank_before"] == 4:
                n_winner_4th_before += 1
                if last["winner_rank_after"] == 4:
                    n_winner_4th_before_still_4th += 1
    gaps_out = {
        "gap_to_1st": {f"rank{rank}": _stats(vals) for rank, vals in gap_to_first.items()},
        "adjacent_gaps": {name: _stats(vals) for name, vals in adjacent.items()},
        "fraction_below_threshold": {
            name: {str(t): _frac(sum(1 for v in vals if v < t), len(vals)) for t in GAP_THRESHOLDS}
            for name, vals in {
                "gap_to_1st_rank4": gap_to_first[4],
                "adjacent_3rd-4th": adjacent["3rd-4th"],
                "adjacent_1st-2nd": adjacent["1st-2nd"],
            }.items()
        },
    }
    metric_b = {
        "denominator_winner_rank_before_4th": n_winner_4th_before,
        "numerator_winner_rank_before_4th_and_rank_after_4th": n_winner_4th_before_still_4th,
        "fraction_winner_was_4th_still_4th": _frac(n_winner_4th_before_still_4th, n_winner_4th_before)["fraction"],
    }
    return {
        "n_all_last_games": n_games,
        "small_sample": n_games < SMALL_SAMPLE_THRESHOLD,
        "last_round_outcomes": {"hora": n_hora, "ryukyoku": n_ryukyoku, "none": n_none},
        "n_last_round_multi_ron": n_multi_ron,
        "metric_a_final_gaps": gaps_out,
        "metric_b": metric_b,
        "winner_rank_before_counts": {str(k): v for k, v in winner_rank_before.items()},
        "winner_rank_after_counts": {str(k): v for k, v in winner_rank_after.items()},
        "n_winner_finished_4th": n_winner_finished_4th,
        "rank_transition_of_player_4th_at_last_round_start": {str(k): v for k, v in transition_4th_before.items()},
        "player_4th_at_last_round_start_finished_4th": _frac(n_4th_before_finished_4th, n_games),
    }


def run_group(source_dir: str, pattern: str) -> dict:
    files = sorted(glob(os.path.join(source_dir, pattern)))
    records, skipped = [], []
    for path in files:
        try:
            records.append(analyze_file(path))
        except DecodeError as exc:
            skipped.append({"file": os.path.basename(path), "error": str(exc)})
    all_last = [r for r in records if r["all_last"]]
    ended_at_s4 = [r for r in all_last if not r["went_to_west"]]
    went_west = [r for r in all_last if r["went_to_west"]]
    by_key, by_split, by_names = {}, {}, {}
    for rec in all_last:
        key = str(rec["key"]) if rec["key"] is not None else "unknown"
        by_key.setdefault(key, []).append(rec)
        by_split.setdefault(rec["split"] or "unknown", []).append(rec)
        by_names.setdefault("|".join(rec["names"]) if rec["names"] else "unknown", []).append(rec)
    return {
        "source_dir": source_dir,
        "pattern": pattern,
        "files_matched": len(files),
        "games_decoded": len(records),
        "skipped": {"count": len(skipped), "files": skipped},
        "all_last_games": len(all_last),
        "games_without_all_last": len(records) - len(all_last),
        "replay_mismatch_games": sum(1 for r in records if r["mismatch_count"]),
        "filename_seed_mismatch_games": sum(1 for r in records if r["filename_mismatch"]),
        "n_games_went_to_west": len(went_west),
        "by_filename_key": {k: aggregate([r["view_final"] for r in v]) for k, v in sorted(by_key.items())},
        "by_split": {k: aggregate([r["view_final"] for r in v]) for k, v in sorted(by_split.items())},
        "by_names": {k: len(v) for k, v in sorted(by_names.items())},
        "metrics": aggregate([r["view_final"] for r in all_last]),
        "metrics_at_all_last_round_end": aggregate([r["view_all_last_end"] for r in all_last]),
        "metrics_subset_final_round_is_s4": aggregate([r["view_final"] for r in ended_at_s4]),
        "metrics_subset_went_to_west": aggregate([r["view_final"] for r in went_west]),
        "games": [
            {
                "file": r["file"],
                "seed": r["seed"],
                "key": r["key"],
                "split": r["split"],
                "names": r["names"],
                "n_rounds": r["n_rounds"],
                "replay_mismatch_count": r["mismatch_count"],
                "filename_seed_mismatch": r["filename_mismatch"],
                "went_to_west": r["went_to_west"],
                "final_scores": r["final_scores"],
                "player_by_rank": r["view_final"]["player_by_rank"],
                "rank_by_player": r["view_final"]["rank_by_player"],
                "last_round": r["last_round"],
                "all_last_round": r["all_last_round"],
                "all_last_end_scores": r["all_last_end_scores"],
            }
            for r in all_last
        ],
        "_views_final": [r["view_final"] for r in all_last],
    }


def self_check() -> dict:
    """Decode the pinned sample and assert its known values; raise on mismatch."""
    path = os.path.join(REPO_ROOT, SELF_CHECK_REL)
    if not os.path.isfile(path):
        raise DecodeError(f"self-check sample missing: {SELF_CHECK_REL}")
    game = analyze_file(path)
    last = game["last_round"]
    actual = {
        "last_round": {"bakaze": last["bakaze"], "kyoku": last["kyoku"], "honba": last["honba"]},
        "horas": last["horas"],
        "final_scores": game["final_scores"],
    }
    expected = {"last_round": SELF_CHECK_LAST_ROUND, "horas": SELF_CHECK_HORAS, "final_scores": SELF_CHECK_FINAL}
    if actual != expected:
        raise AssertionError(
            f"self-check FAILED\n  expected={json.dumps(expected, ensure_ascii=False)}\n"
            f"  actual  ={json.dumps(actual, ensure_ascii=False)}"
        )
    return {"path": SELF_CHECK_REL, "expected": expected, "actual": actual}


def _print_summary(group: dict) -> None:
    metrics = group["metrics"]
    print(
        f"[group {group['source_dir']}] files={group['files_matched']} decoded={group['games_decoded']} "
        f"skipped={group['skipped']['count']} all_last={group['all_last_games']} "
        f"(no-all-last={group['games_without_all_last']}, went-to-west={group['n_games_went_to_west']}, "
        f"replay-mismatch-files={group['replay_mismatch_games']}, "
        f"filename-mismatch-files={group['filename_seed_mismatch_games']})"
    )
    if metrics.get("n_all_last_games", 0) == 0:
        return
    for label, block in (("final", metrics), ("at-S4-end", group["metrics_at_all_last_round_end"])):
        gap = block["metric_a_final_gaps"]["gap_to_1st"]
        mb = block["metric_b"]
        print(
            f"  [{label}] metric_a gaps to 1st (p50/p90): "
            f"2nd={gap['rank2']['p50']}/{gap['rank2']['p90']} "
            f"3rd={gap['rank3']['p50']}/{gap['rank3']['p90']} "
            f"4th={gap['rank4']['p50']}/{gap['rank4']['p90']} | "
            f"metric_b {mb['numerator_winner_rank_before_4th_and_rank_after_4th']}/"
            f"{mb['denominator_winner_rank_before_4th']}={mb['fraction_winner_was_4th_still_4th']}"
        )
    print(f"  last-round outcomes: {metrics['last_round_outcomes']}")
    if metrics["small_sample"]:
        print(f"  WARNING: all_last_games={metrics['n_all_last_games']} < {SMALL_SAMPLE_THRESHOLD} (preliminary only)")


def main() -> int:
    parser = argparse.ArgumentParser(description="#111 quantification: all-last overtake statistics")
    parser.add_argument("source_dirs", nargs="+", help="source directories containing *.json.gz NDJSON game logs")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output JSON path (default: {DEFAULT_OUT})")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help=f"file glob (default: {DEFAULT_PATTERN})")
    parser.add_argument("--self-check-only", action="store_true", help="run the built-in sample self-check and exit")
    args = parser.parse_args()

    dirs = [os.path.normpath(d) for d in args.source_dirs]
    if len(set(os.path.abspath(d) for d in dirs)) != len(dirs):
        parser.error("duplicate source directories")
    for d in dirs:
        if not os.path.isdir(d):
            parser.error(f"not a directory: {d}")

    started = time.monotonic()
    try:
        check = self_check()
    except (DecodeError, AssertionError) as exc:
        print(f"SELF-CHECK FAILED: {exc}", file=sys.stderr)
        return 2
    print(
        f"[self-check] OK {check['path']}: last round {check['actual']['last_round']}, "
        f"horas={json.dumps(check['actual']['horas'])}, final_scores={check['actual']['final_scores']}"
    )
    if args.self_check_only:
        return 0

    groups, totals_views = {}, []
    for source_dir in dirs:
        group = run_group(source_dir, args.pattern)
        totals_views.extend(group.pop("_views_final"))
        if group["skipped"]["count"]:
            print(f"WARNING: {group['skipped']['count']} file(s) skipped in {source_dir}", file=sys.stderr)
            for item in group["skipped"]["files"]:
                print(f"  skipped: {item['file']}: {item['error']}", file=sys.stderr)
        groups[source_dir] = group
        _print_summary(group)

    wall = time.monotonic() - started
    totals = aggregate(totals_views)
    print(
        f"[totals] all_last={totals.get('n_all_last_games', 0)} "
        f"skipped={sum(g['skipped']['count'] for g in groups.values())}"
    )
    print(f"[runtime] wall={wall:.2f}s")

    payload = {
        "schema": "alllast-quant/v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "scripts/analyze_alllast.py",
        "issue": "#111 (quantify only, no observation-encoding change)",
        "definitions": DEFINITIONS,
        "self_check": check,
        "runtime": {
            "wall_seconds": round(wall, 3),
            "source_dirs": dirs,
            "pattern": args.pattern,
            "files_matched": sum(g["files_matched"] for g in groups.values()),
            "games_decoded": sum(g["games_decoded"] for g in groups.values()),
            "skipped": sum(g["skipped"]["count"] for g in groups.values()),
        },
        "totals": totals,
        "groups": groups,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = f"{args.out}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, args.out)
    print(f"[wrote] {args.out} ({os.path.getsize(args.out)} bytes)")
    if sum(g["skipped"]["count"] for g in groups.values()):
        print("WARNING: some files were skipped; see stderr and the JSON 'skipped' fields", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
