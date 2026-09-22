"""单元 4: 分层摘要 + 关键局面 top-k + 结果论解耦。

设计原则(D4): 一局约 270 个决策点,下游(尤其 2B LLM)吃不下全量 ⇒ 必须压成
每局摘要 + 整局摘要 + key_moments(3-5 条)。key_moments 只含**结构化 facts**,
不含任何自然语言句子(讲解属 W3)。
修正说明: GRP 每局只给一个矩阵 ⇒ "逐决策顺位摆动"退化;改用局边界顺位变化,
归属到该局首个决策点。
"""
KEY_MOMENT_TOP_K = 5
M3_PLUS_METRICS = ("opp_tenpai", "danger", "called_risk", "meld_decision_eval",
                   "dangerous_tiles")


def first_decision_of_each_kyoku(decisions):
    seen = {}
    for dec in decisions:
        seen.setdefault(dec.at_kyoku, dec)
    return seen


def _kyoku_rank_swings(decisions, rank_payloads):
    """返回 {kyoku: |E[rank]_k - E[rank]_{k-1}|},归属到该局首个决策点。"""
    firsts = first_decision_of_each_kyoku(decisions)
    ordered = sorted(firsts)
    values = {}
    previous = None
    for k in ordered:
        payload = rank_payloads.get(firsts[k].index)
        if payload is None or payload.get("unavailable"):
            previous = None   # r3: 基线不可用 ⇒ 下一局不得跨过它做差(否则夸大摆动)
            continue
        current = float(payload["expected_rank"])
        values[k] = 0.0 if previous is None else abs(current - previous)
        previous = current
    return values


def _score(regret_payloads, index):
    payload = regret_payloads.get(index) or {}
    return payload.get("delta_rel")


def build(decisions, q_payloads, regret_payloads, rank_payloads, record,
          final_scores, subject_player):
    unavailable = [{"metric": name,
                    "reason": "M3+ 范围(需新建预测头或新的判断评估),M1/M2 不实现"}
                   for name in M3_PLUS_METRICS]

    # r3: is_blunder 未校准时恒为 None ⇒ n_blunders 不得谎报 0,应如实给 null
    has_calibration = any((p or {}).get("confidence") == "calibrated"
                          for p in regret_payloads.values())

    kyoku = {}
    for dec in decisions:
        kyoku.setdefault(dec.at_kyoku, []).append(dec)

    by_kyoku_regret = {}
    kyoku_summaries = []
    for k in sorted(kyoku):
        items = kyoku[k]
        deltas = [d for d in (_score(regret_payloads, d.index) for d in items)
                  if d is not None]
        by_kyoku_regret[k] = deltas
        kyoku_summaries.append({
            "kyoku": k,
            "n_decisions": len(items),
            "n_blunders": (sum(1 for d in items
                               if (regret_payloads.get(d.index) or {}).get("is_blunder") is True)
                           if has_calibration else None),
            "worst_delta_rel": round(min(deltas), 6) if deltas else None,
            "mean_delta_rel": round(sum(deltas) / len(deltas), 6) if deltas else None,
        })

    all_deltas = [d for deltas in by_kyoku_regret.values() for d in deltas]
    mean_delta = (sum(all_deltas) / len(all_deltas)) if all_deltas else 0.0
    judgement_score = max(0.0, min(1.0, 1.0 + mean_delta))

    ranks = sorted(range(4), key=lambda i: (-final_scores[i], i))
    actual_rank = ranks.index(subject_player) + 1
    firsts = first_decision_of_each_kyoku(decisions)
    k0 = min(firsts)
    start_payload = rank_payloads.get(firsts[k0].index) or {}
    expected_rank_start = start_payload.get("expected_rank")

    if expected_rank_start is None:
        divergence = "neutral"
    else:
        judged_well = judgement_score >= 0.8
        resulted_well = actual_rank <= expected_rank_start
        divergence = ("aligned" if judged_well == resulted_well
                      else ("good_call_bad_result" if judged_well
                            else "bad_call_good_result"))

    swings = _kyoku_rank_swings(decisions, rank_payloads)
    swing_by_index = {firsts[k].index: v for k, v in swings.items()}

    regret_ranked = sorted(
        ((abs(p["delta_rel"]), i) for i, p in regret_payloads.items() if "delta_rel" in p),
        reverse=True)
    swing_ranked = sorted(((v, i) for i, v in swing_by_index.items()), reverse=True)

    chosen = {}
    for value, index in regret_ranked[:KEY_MOMENT_TOP_K]:
        chosen.setdefault(index, {"index": index, "flagged_by": []})
        chosen[index]["flagged_by"].append("max_regret")
        chosen[index].setdefault("facts", {})["delta_rel"] = round(-value, 6)
    for value, index in swing_ranked[:KEY_MOMENT_TOP_K]:
        chosen.setdefault(index, {"index": index, "flagged_by": []})
        chosen[index]["flagged_by"].append("kyoku_rank_swing")
        chosen[index].setdefault("facts", {})["kyoku_rank_swing"] = round(value, 6)

    # r3: 必须按**重要性**选,不能按下标截断(`sorted(chosen)[:K]` 会丢掉最差 regret 决策,
    # 并可能整类信号被清空)。做法:两条榜单**交替**取,再去重截断 ⇒ 两类信号都必然出现。
    merged = []
    for rank in range(KEY_MOMENT_TOP_K):
        if rank < len(regret_ranked):
            merged.append(regret_ranked[rank][1])
        if rank < len(swing_ranked):
            merged.append(swing_ranked[rank][1])
    ordered = []
    for index in merged:
        if index not in ordered:
            ordered.append(index)
    key_moments = [chosen[i] for i in ordered][:KEY_MOMENT_TOP_K]

    game_summary = {
        "subject_player": int(subject_player),
        "n_decisions": len(decisions),
        "mean_delta_rel": round(mean_delta, 6),
        "judgement_vs_outcome": {
            "judgement_score": round(judgement_score, 6),
            "actual_rank": actual_rank,
            "expected_rank_start": (round(float(expected_rank_start), 6)
                                    if expected_rank_start is not None else None),
            "divergence": divergence,
        },
    }

    return {"kyoku_summaries": kyoku_summaries, "game_summary": game_summary,
            "key_moments": key_moments, "unavailable": unavailable}
