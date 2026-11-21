"""t12: 局面/巡目上下文 provider(免费,无检查点,纯公开信息)。

来源: v4 观测张量的公开状态平面。偏移一律经 `layout.span(top, "<block>")` 查
`runs/analysis/obs_layout.json`,绝不硬编码:

  scores       [7, 15)    4 家分数,每家 2 平面(相对席序 self/shimocha/toimen/kamicha;
                          主平面 clamp(score,0,100000)/100000,次平面 /30000 冗余)
  honba        [23, 24)   honba/10(编码时 clamp 到 10)
  kyotaku      [24, 25)   供托/10(编码时 clamp 到 10)
  round_index  [27, 28)   ((bakaze-E).min(1)*4 + kyoku(0-based))/7
  tiles_left   [717, 718) 牌山余枚/69
  tiles_seen   [835, 836) 每种牌被全员看见的张数/4(单平面 34 列)
  at_turn      决策元数据(replay),非 obs 切片

NO-REVEAL: 以上全部是公开量。**不**暴露 `doras_owned`——其自家面由自家手牌+副露派生
(属 self_hand),不是公开特征(M12/F6);若将来需要,必须单列并标 basis="self_hand"。
本 payload 不消费任何对手暗手派生量。

解码精确: obs 存的是 f32 的 n/cap,`round(v*cap)` 在全部取值域往返无损
(cap ∈ {4,7,10,69,100000};t12 测试逐位断言)。scores 的 clamp(负分→0,>100000→100000)
是编码器自身语义(obs_repr.rs:149-165),不是本 provider 的截断。
"""
from jiangpu import layout

PROVIDER_NAME = "context"
PROVIDER_VERSION = 1
UNIT = "mixed"          # 混合量纲 ⇒ 逐字段声明(照 sp_table 范式)
UNIT_BY_FIELD = {
    "tiles_left": "tiles",
    "honba": "count",
    "kyotaku": "count",
    "scores": "points",
    "round_index": "index",
    "tiles_seen": "count",
    "at_turn": "turn",
}
NO_HIDDEN_INFO = True
BASIS = "public_info_predictor"

TILES = 34
SCORE_CAP = 100_000

# block 名 → 编码分母(obs_repr.rs: value = clamp(n, cap)/cap;tiles_left 无 clamp)
_CAPS = {"honba": 10, "kyotaku": 10, "round_index": 7, "tiles_left": 69}
_BLOCK_NAMES = ("scores", "honba", "kyotaku", "round_index", "tiles_left", "tiles_seen")
_EXPECTED_COUNTS = {"scores": 8, "honba": 1, "kyotaku": 1, "round_index": 1,
                    "tiles_left": 1, "tiles_seen": 1}


def _meta(**extra):
    payload = {
        "provider": PROVIDER_NAME,
        "provider_version": PROVIDER_VERSION,
        "checkpoint_sha256": None,
        "unit": UNIT,
        "no_hidden_info": NO_HIDDEN_INFO,
        "basis": BASIS,
    }
    payload.update(extra)
    return payload


def _decode(value, cap):
    """obs 的 n/cap(f32) → 整数 n;round 往返无损(见模块 docstring)。"""
    return int(round(float(value) * cap))


def _spans(top):
    """block 名 → span;缺名 ⇒ KeyError(fail-loud,不做静默回退)。"""
    spans = {name: layout.span(top, name) for name in _BLOCK_NAMES}
    drift = [f"{name} 通道数应为 {_EXPECTED_COUNTS[name]},实为 {count}"
             for name, (_, count) in spans.items() if count != _EXPECTED_COUNTS[name]]
    return spans, drift


def compute(decisions):
    """公开状态/巡目上下文。结构漂移时 fail-closed(unavailable + 非空 reason,不填 0)。"""
    spans, drift = _spans(layout.load_layout())
    if drift:
        reason = "layout 漂移: " + "; ".join(drift)
        return {dec.index: _meta(unavailable=True, reason=reason) for dec in decisions}

    out = {}
    for dec in decisions:
        obs = dec.obs
        score_off = spans["scores"][0]
        out[dec.index] = _meta(
            unit_by_field=dict(UNIT_BY_FIELD),
            tiles_left=_decode(obs[spans["tiles_left"][0]][0], _CAPS["tiles_left"]),
            honba=_decode(obs[spans["honba"][0]][0], _CAPS["honba"]),
            kyotaku=_decode(obs[spans["kyotaku"][0]][0], _CAPS["kyotaku"]),
            scores=[_decode(obs[score_off + 2 * k][0], SCORE_CAP) for k in range(4)],
            round_index=_decode(obs[spans["round_index"][0]][0], _CAPS["round_index"]),
            tiles_seen=[_decode(obs[spans["tiles_seen"][0]][t], 4) for t in range(TILES)],
            at_turn=int(dec.at_turn),
        )
    return out
