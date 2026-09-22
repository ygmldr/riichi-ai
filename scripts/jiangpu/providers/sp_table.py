"""3b: SP 表(自家按期限抵达听牌/和了的概率与和了点数期望)。

来源: v4 观测张量中 libriichi 烤入的 SP 平面。偏移取自 runs/analysis/obs_layout.json
的 block 名(obs_layout.py 不导出常量):
  sp_ev     [889, 891)
  sp_tables [961,1012) → tenpai [961,978) / win [978,995) / ev [995,1012), MAX_NUM_TURNS=17
量纲更正(实测,Oracle 复核): SP 的 tenpai/win 是概率,但 **ev 平面是归一化值**
(obs_repr.rs: `ev * ev_scale` 后 `min(1.)`;`sp_ev` 为 `value.clamp(...)/100000`、`/30000`),
**不是**和了点数。故 `unit` 如实标为 `normalized_ev`。
"""
from jiangpu import layout
from jiangpu.tiles import deaka

PROVIDER_NAME = "sp_table"
PROVIDER_VERSION = 1
UNIT = "normalized_ev"
SP_TURNS = 17


def compute(decisions):
    """SP 表切片。

    可用性判据**只看起点(turn 0)**,绝不逐巡目判(r3 修正,r4 落地):
    SP 编码器用 `take_while(p > 0)` 写入 ⇒ **越出预测 horizon 的巡目合法地为 0**,
    把它当成"未编码"会把每个决策都误判为 unavailable(实测曾 0/669 可用)。
    """
    top = layout.load_layout()
    t_off, t_count = layout.span(top, "sp_tables")
    e_off, e_count = layout.span(top, "sp_ev")
    out = {}
    for dec in decisions:
        if t_count != 3 * SP_TURNS:
            out[dec.index] = {
                "provider": PROVIDER_NAME, "provider_version": PROVIDER_VERSION,
                "checkpoint_sha256": None, "unit": UNIT, "unavailable": True,
                "reason": f"sp_tables 通道数应为 {3 * SP_TURNS},实为 {t_count}",
            }
            continue

        col = deaka(dec.logged_action) if 0 <= dec.logged_action <= 36 else None
        row0 = dec.obs[t_off]
        reason = None
        if float(row0.max()) == 0.0:
            reason = "SP 未编码该决策(起点整行为 0;如 shanten>=4)"
        elif col is None:
            if float(row0.max()) - float(row0.min()) > 1e-4:
                reason = "非打牌动作但起点行非常量(如立直决策),无单一主体牌"
        elif float(row0[col]) == 0.0:
            reason = "打出的牌是候选列但 SP 起点为 0(该候选未计算)"

        if reason is not None:
            out[dec.index] = {
                "provider": PROVIDER_NAME, "provider_version": PROVIDER_VERSION,
                "checkpoint_sha256": None, "unit": UNIT, "unavailable": True,
                "reason": reason,
            }
            continue

        c = col if col is not None else 0
        tenpai = [round(float(dec.obs[t_off + i][c]), 6) for i in range(SP_TURNS)]
        win = [round(float(dec.obs[t_off + SP_TURNS + i][c]), 6) for i in range(SP_TURNS)]
        ev = [round(float(dec.obs[t_off + 2 * SP_TURNS + i][c]), 6) for i in range(SP_TURNS)]
        sp_ev = [round(float(dec.obs[e_off + i][0]), 6) for i in range(e_count)]
        horizon = max((i for i, v in enumerate(tenpai) if v > 0.0), default=-1)
        out[dec.index] = {
            "provider": PROVIDER_NAME,
            "provider_version": PROVIDER_VERSION,
            "checkpoint_sha256": None,
            "unit": UNIT,
            # r2/r3: 本 payload 混两种量纲 ⇒ 逐字段声明(D6)
            "unit_by_field": {"own_tenpai_prob": "probability",
                              "own_win_prob": "probability",
                              "own_ev": "normalized_ev",
                              "sp_ev": "normalized_ev"},
            "subject_tile": col,
            "horizon": horizon,      # 最后一个 tenpai>0 的巡目;之后为合法 0
            "own_tenpai_prob": tenpai,
            "own_win_prob": win,
            "own_ev": ev,
            "sp_ev": sp_ev,
            "by_turn": list(range(SP_TURNS)),
        }
    return out
