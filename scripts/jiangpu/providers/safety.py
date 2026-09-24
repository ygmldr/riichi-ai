"""t13: 公开安全输入 —— `heuristic_score`(启发式分数,**绝不是**概率)。

输入全部来自 v4 观测张量的公开平面(block 名 → runs/analysis/obs_layout.json):
  kawa_overview_{self,shimocha,toimen,kamicha} [723,751)  各家牌河 tile set(4 copy-planes + 3 aka)
  riichi_sutehai_{shimocha,toimen,kamicha}      [845,854)  立直宣言牌 one-hot(+1 aka / +2 is_dora)
  riichi_declared_opponents                     [854,857)  "已宣立直"对手 one-hot(成立后仍为 1)
  riichi_accepted_opponents                     [857,860)  "立直已成立"对手 one-hot
  tiles_seen                                    [835,836)  牌种已知枚数/4(含自家手牌、副露、指示牌)
  dora_indicators                               [28,35)    宝牌指示牌 tile set(4 copy-planes + 3 aka)

规则(纯公开信息,NO-REVEAL;不读对手暗手、不读真实牌序、不读 oracle 张量):
  - 现物(genbutsu): 立直对手 j 的牌河中的任何牌都是硬安全牌。立直(含宣言当巡)使 j 处于
    furiten,不能荣和自己弃过的牌,故 j 的牌河整集即对 j 的现物。obs 不含弃牌顺序,而该判据
    只依赖"牌河集",无需顺序。
  - 筋(suji): j 弃过 s ⇒ 含 s 的两面半边死 ⇒ s±3 为筋(降危而非免危:单骑/双碰/嵌张/边张
    仍可能荣和)。仅同花色内 1..9 有效;字牌无筋。
  - 可见枚数(visible): tiles_seen[t] ∈ 0..4。满 4 ⇒ 四枚全部定位,任何对手都与该牌无关(硬事实)。
  - 残壁(unseen): 4 − visible,尚未定位的枚数(公开计数的补数)。
  - 宝牌(dora): 按 tile.rs:117-132 的 Tile::next 由指示牌推出;点值高、和了更亏 ⇒ 小降分。

分数(无量纲,越大越安全;**没有**概率语义):
  score(t) = 1000·1[visible==4] + 100·genbutsu_count + 10·suji_count + visible − dora
词典序分层(满可见 > 现物数 > 筋数 > 可见数),并列项只做次序区分。权重是启发式常数,未经
拟合,不得读作概率/期望。F3 的验收统计量 = AUC(risk = −score, bad),见 `exp_ex1_prereg.md` §6.2。

输出契约: unit == "heuristic_score"; 输出键名不得含 prob/deal_in/danger(PREREG §6.2)。
无检查点 ⇒ checkpoint_sha256=None(contract.NO_CHECKPOINT_PROVIDERS 白名单)。
"""
from jiangpu import layout
from jiangpu.tiles import deaka

PROVIDER_NAME = "safety"
PROVIDER_VERSION = 1
UNIT = "heuristic_score"
NO_HIDDEN_INFO = True
BASIS = "public_info_predictor"   # 全部特征来自公开平面(牌河/立直/可见枚数/指示牌)

N_TILES = 34
N_OPPONENTS = 3
N_RIVER_CHANNELS = 7
N_SUTEHAI_CHANNELS = 3
N_SEEN_CHANNELS = 1
N_DORA_CHANNELS = 7
DISCARD_ACTION_MAX = 36          # 0..33 普通牌, 34..36 赤5; 37=立直, 38+=鸣牌

RIVER_BLOCKS = ("kawa_overview_self", "kawa_overview_shimocha",
                "kawa_overview_toimen", "kawa_overview_kamicha")
SUTEHAI_BLOCKS = ("riichi_sutehai_shimocha", "riichi_sutehai_toimen",
                  "riichi_sutehai_kamicha")
DECLARED_BLOCK = "riichi_declared_opponents"
ACCEPTED_BLOCK = "riichi_accepted_opponents"
SEEN_BLOCK = "tiles_seen"
DORA_BLOCK = "dora_indicators"

REQUIRED_CHANNELS = {name: N_RIVER_CHANNELS for name in RIVER_BLOCKS}
REQUIRED_CHANNELS.update({name: N_SUTEHAI_CHANNELS for name in SUTEHAI_BLOCKS})
REQUIRED_CHANNELS.update({DECLARED_BLOCK: N_OPPONENTS, ACCEPTED_BLOCK: N_OPPONENTS,
                          SEEN_BLOCK: N_SEEN_CHANNELS, DORA_BLOCK: N_DORA_CHANNELS})

SCORE_VISIBLE_FOUR = 1000.0
SCORE_PER_GENBUTSU = 100.0
SCORE_PER_SUJI = 10.0
SCORE_DORA_PENALTY = 1.0


def next_tile(tile_id):
    """宝牌指示牌 → 宝牌,逐字对齐 `tile.rs:117-132` 的 Tile::next(含字牌环绕)。"""
    kind, num = divmod(tile_id, 9)
    if kind < 3:
        return kind * 9 + (num + 1) % 9
    if num < 4:
        return 27 + (num + 1) % 4
    return 27 + 4 + (num - 4 + 1) % 3


def suji_mates(tile_id):
    """tile 的筋牌 = 同花色内 ±3(字牌无筋)。"""
    kind, num = divmod(tile_id, 9)
    if kind > 2:
        return ()
    return tuple(kind * 9 + num + d for d in (-3, 3) if 0 <= num + d <= 8)


def heuristic_score(visible, genbutsu_mask, suji_mask, is_dora):
    """无量纲安全分,越大越安全;词典序 满可见 > 现物数 > 筋数 > 可见数,宝牌降分。"""
    score = SCORE_VISIBLE_FOUR if visible == 4 else 0.0
    score += SCORE_PER_GENBUTSU * bin(genbutsu_mask).count("1")
    score += SCORE_PER_SUJI * bin(suji_mask).count("1")
    score += float(visible) - (SCORE_DORA_PENALTY if is_dora else 0.0)
    return score


def _river_counts(obs, off):
    """tile set 的 4 张 copy-plane 之和 = 该牌种在牌河中的枚数(0..4)。"""
    return [sum(int(round(float(obs[off + c][t]))) for c in range(4)) for t in range(N_TILES)]


def _labeled_tile(obs, off):
    """one-hot 平面(riichi_sutehai)上的牌种;空平面返回 None。"""
    row = obs[off]
    return int(row.argmax()) if float(row.max()) > 0.0 else None


def _tiles_seen(obs, off):
    return [int(round(float(obs[off][t]) * 4)) for t in range(N_TILES)]


def _dora_flags(obs, off):
    flags = [False] * N_TILES
    for t in range(N_TILES):
        if sum(int(round(float(obs[off + c][t]))) for c in range(4)) > 0:
            flags[next_tile(t)] = True
    return flags


def _spans(top):
    spans = {name: layout.span(top, name) for name in REQUIRED_CHANNELS}
    bad = [f"{name}: 通道数应为 {REQUIRED_CHANNELS[name]},实为 {spans[name][1]}"
           for name in REQUIRED_CHANNELS if spans[name][1] != REQUIRED_CHANNELS[name]]
    return spans, bad


def _unavailable(reason):
    return {
        "provider": PROVIDER_NAME, "provider_version": PROVIDER_VERSION,
        "checkpoint_sha256": None, "unit": UNIT,
        "no_hidden_info": NO_HIDDEN_INFO, "basis": BASIS,
        "unavailable": True, "reason": reason,
    }


def compute(decisions):
    top = layout.load_layout()
    spans, bad = _spans(top)
    if bad:
        return {dec.index: _unavailable("; ".join(bad)) for dec in decisions}

    out = {}
    for dec in decisions:
        obs = dec.obs
        river = [_river_counts(obs, spans[name][0]) for name in RIVER_BLOCKS[1:]]
        sutehai = [_labeled_tile(obs, spans[name][0]) for name in SUTEHAI_BLOCKS]
        declared = [float(obs[spans[DECLARED_BLOCK][0] + j][0]) > 0.0
                    for j in range(N_OPPONENTS)]
        accepted = [float(obs[spans[ACCEPTED_BLOCK][0] + j][0]) > 0.0
                    for j in range(N_OPPONENTS)]
        threat = [d or a for d, a in zip(declared, accepted)]
        seen = _tiles_seen(obs, spans[SEEN_BLOCK][0])
        dora = _dora_flags(obs, spans[DORA_BLOCK][0])

        genbutsu = [0] * N_TILES
        suji = [0] * N_TILES
        scores = [0.0] * N_TILES
        for t in range(N_TILES):
            for j in range(N_OPPONENTS):
                if not threat[j]:
                    continue
                if river[j][t] > 0 or sutehai[j] == t:
                    genbutsu[t] |= 1 << j
                elif any(river[j][m] > 0 for m in suji_mates(t)):
                    suji[t] |= 1 << j
            scores[t] = heuristic_score(seen[t], genbutsu[t], suji[t], dora[t])

        subject = (deaka(dec.logged_action)
                   if 0 <= dec.logged_action <= DISCARD_ACTION_MAX else None)
        out[dec.index] = {
            "provider": PROVIDER_NAME,
            "provider_version": PROVIDER_VERSION,
            "checkpoint_sha256": None,
            "unit": UNIT,
            "no_hidden_info": NO_HIDDEN_INFO,
            "basis": BASIS,
            "subject_tile": subject,
            "subject_score": None if subject is None else scores[subject],
            "tile_score": scores,
            "components": {
                # bitmask, bit j = 相对对手 j(shimocha/toimen/kamicha)
                "genbutsu_opponents": genbutsu,
                "suji_opponents": suji,
                "visible_copies": seen,      # tiles_seen 原始枚数(含自家手牌/副露/指示牌)
                "unseen_copies": [4 - v for v in seen],   # 残壁: 尚未定位的枚数
                "dora_flag": dora,
            },
            "threat": {
                "declared": declared,
                "accepted": accepted,
                "declaration_tiles": sutehai,
                "n_riichi": sum(1 for v in threat if v),
            },
        }
    return out
