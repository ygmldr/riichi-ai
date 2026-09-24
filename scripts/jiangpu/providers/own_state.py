"""3d: 自家状态(自家向听 + 自家受入/required tiles)。

来源: v4 观测张量中 libriichi 烤入的自家平面。偏移一律取自
`runs/analysis/obs_layout.json` 的 block 名(`layout.span`),不硬编码:
  shanten            [862, 869)  自家向听 one-hot 0..6(>=6 压最后一格)
  sp_required_tiles  [891, 961)  v4 单机 required-tiles 平面(70 = 2*34 + 2)
  discard_candidates [874, 879)  交叉核对 can_discard:编码器写 +0 当且仅当可打牌
  tiles_left         [717, 718)  仅用于 unavailable 的诊断理由(state.tiles_left / 69)

布局(obs_repr.rs:576-608;实测复核,见 tests/test_ex1_own_state.py):
  A. `cans.can_discard` 且 `single_player_tables()` 成功:
     +0..+33   每个弃牌候选的 required-tiles one-hot(shanten_down=false)
     +34..+67  同上(shanten_down=true;SP 计算路径 `calc_shanten_down=false`,
               故实测仅向听 >= 4 的 simple 分支可能落在这里)
     +68       受入种数最多的弃牌候选 one-hot ⇒ 读它的候选平面作为"自家受入"
  B. `!can_discard` 且 `single_player_tables()` 成功:
     +0..+68 全零,+69 = 单一(摸牌)候选的 required-tiles one-hot
     (`obs_layout.json` 的语义串写 "+68",实测为 +69 —— 本 provider 以实测为准,
     不因为文档差一而把已编码的决策误判为 unavailable)
  C. `single_player_tables()` 失败(tiles_left<4 / real_time_shanten<0 / 无剩余自摸):
     整块全零 ⇒ `unavailable: true` + 非空 reason,**绝不填 0**(MI-3 / PREREG §9.1)

受入口径:one-hot 的**点亮种数**(distinct tile kinds),不是牌张拷贝数;`ukeire_tiles`
为升序牌种索引,`subject_tile` 是所读平面的主体牌(布局 A = 受入种数最多的弃牌候选;
布局 B = None,因为主体是"下一次摸牌")。

NO-REVEAL: 只读自家观测张量的自家平面(向听 + required tiles),不触碰任何对手信息 ⇒
`basis="self_hand"`、`no_hidden_info=true`。
"""
import numpy as np

from jiangpu import layout

PROVIDER_NAME = "own_state"
PROVIDER_VERSION = 1
UNIT = "count"
UNIT_BY_FIELD = {"shanten": "shanten_level", "ukeire_kinds": "count",
                 "ukeire_tiles": "tile_index"}
NO_HIDDEN_INFO = True
BASIS = "self_hand"

SHANTEN_CAP = 6                  # one-hot 0..6;6 = ">=6"(i8 负值经 usize 转换同样压到 6)
TILE_KINDS = 34
DISCARD_CANDIDATES_PLANE = 0     # 块内相对平面:can_discard 时为弃牌候选 one-hot
MAX_REQUIRED_TILE_PLANE = 68     # 布局 A:受入种数最多的弃牌 one-hot
DRAW_REQUIRED_TILE_PLANE = 69    # 布局 B:单候选 required tiles(实测;文档写 +68)
SP_REQUIRED_TILES_COUNT = 2 * TILE_KINDS + 2
TILES_LEFT_SCALE = 69            # obs_repr.rs:282 编码为 state.tiles_left / 69


def _meta():
    return {"provider": PROVIDER_NAME, "provider_version": PROVIDER_VERSION,
            "checkpoint_sha256": None, "unit": UNIT, "unit_by_field": UNIT_BY_FIELD,
            "no_hidden_info": NO_HIDDEN_INFO, "basis": BASIS}


def _unavailable(reason):
    return {**_meta(), "unavailable": True, "reason": reason}


def _lit(row):
    """返回 one-hot 行里点亮的平面/牌种索引(升序)。"""
    return [i for i in range(len(row)) if float(row[i].max()) > 0.0]


def compute(decisions, top=None):
    """逐决策返回自家向听 + 受入;不可读时显式 unavailable(不填 0)。"""
    top = layout.load_layout() if top is None else top
    sh_off, sh_count = layout.span(top, "shanten")
    rt_off, rt_count = layout.span(top, "sp_required_tiles")
    dc_off, dc_count = layout.span(top, "discard_candidates")
    tl_off, tl_count = layout.span(top, "tiles_left")

    if sh_count != SHANTEN_CAP + 1 or rt_count != SP_REQUIRED_TILES_COUNT:
        reason = (f"布局漂移:shanten={sh_count}(应 {SHANTEN_CAP + 1}),"
                  f"sp_required_tiles={rt_count}(应 {SP_REQUIRED_TILES_COUNT})")
        return {dec.index: _unavailable(reason) for dec in decisions}

    out = {}
    for dec in decisions:
        sh_lit = _lit(dec.obs[sh_off:sh_off + sh_count])
        if len(sh_lit) != 1:
            out[dec.index] = _unavailable(f"shanten 平面非 one-hot(点亮 {sh_lit})")
            continue
        shanten = sh_lit[0]

        rt = dec.obs[rt_off:rt_off + rt_count]
        if float(rt.max()) == 0.0:
            tiles_left = int(round(float(dec.obs[tl_off:tl_off + tl_count].max())
                                   * TILES_LEFT_SCALE))
            out[dec.index] = _unavailable(
                f"sp_required_tiles 全零:SP 表未编码该决策(tiles_left={tiles_left};"
                "需 tiles_left>=4 且 real_time_shanten>=0 且剩余自摸>=1)")
            continue

        lit = _lit(rt)
        can_discard = float(dec.obs[dc_off:dc_off + dc_count][DISCARD_CANDIDATES_PLANE].max()) > 0.0
        if lit == [DRAW_REQUIRED_TILE_PLANE] and not can_discard:
            layout_kind, subject, plane = "draw", None, rt[DRAW_REQUIRED_TILE_PLANE]
        elif MAX_REQUIRED_TILE_PLANE in lit:
            subject = int(np.argmax(rt[MAX_REQUIRED_TILE_PLANE]))
            up, down = rt[subject], rt[TILE_KINDS + subject]
            up_lit, down_lit = float(up.max()) > 0.0, float(down.max()) > 0.0
            if up_lit == down_lit:
                out[dec.index] = _unavailable(
                    f"弃牌 {subject} 的候选平面不唯一(非向听戻={up_lit},向听戻={down_lit})")
                continue
            layout_kind, plane = "discard", (up if up_lit else down)
        else:
            out[dec.index] = _unavailable(f"sp_required_tiles 点亮平面 {lit} 不符合已知布局")
            continue

        ukeire_tiles = _lit(plane)
        out[dec.index] = {**_meta(), "layout": layout_kind, "subject_tile": subject,
                          "shanten": shanten, "shanten_capped": shanten == SHANTEN_CAP,
                          "ukeire_kinds": len(ukeire_tiles), "ukeire_tiles": ukeire_tiles}
    return out
