"""Serving-side public features for the frozen H1 `opp_tenpai` head.

This module deliberately contains **no oracle / label / training code**. It is a byte-for-byte
copy of the frozen feature layout in `scripts/exp_ex1_headA.py` (526 dims: t12 context + t13
safety), used at inference by `jiangpu.providers.opp_tenpai`. A test compares it against the
training extractor on a frozen record.

NO-REVEAL: every input here is a public observation plane; no opponent concealed hand, no true
tile order, no oracle tensor.
"""
from functools import lru_cache

import numpy as np

from jiangpu import layout
from jiangpu.providers import safety as safety_provider

N_OPPS = 3
N_TILES = 34
TILE_RANGE = np.arange(N_TILES, dtype=np.int64)

FEATURE_GROUPS = (
    ("tiles_left", 1), ("honba", 1), ("kyotaku", 1), ("scores", 4), ("round_index", 1),
    ("at_turn", 1), ("tiles_seen", N_TILES), ("dora_flag", N_TILES), ("safety_score", N_TILES),
    ("opp_river", N_OPPS * N_TILES), ("opp_genbutsu", N_OPPS * N_TILES),
    ("opp_suji", N_OPPS * N_TILES), ("threat_declared", N_OPPS), ("threat_accepted", N_OPPS),
    ("riichi_sutehai_onehot", N_OPPS * N_TILES), ("n_riichi", 1),
)
FEATURE_DIM = sum(dim for _, dim in FEATURE_GROUPS)
FEATURE_SLICES: dict[str, slice] = {}
_off = 0
for _name, _dim in FEATURE_GROUPS:
    FEATURE_SLICES[_name] = slice(_off, _off + _dim)
    _off += _dim

_NEXT_TILE = np.array([safety_provider.next_tile(t) for t in range(N_TILES)], dtype=np.int64)
_SUJI_ADJ = np.zeros((N_TILES, N_TILES), dtype=np.float32)
for _m in range(N_TILES):
    for _t in safety_provider.suji_mates(_m):
        _SUJI_ADJ[_m, _t] = 1.0

_SPAN_NAMES = ("tiles_left", "honba", "kyotaku", "scores", "round_index", "tiles_seen",
               "dora_indicators", "kawa_overview_shimocha", "kawa_overview_toimen",
               "kawa_overview_kamicha", "riichi_sutehai_shimocha", "riichi_sutehai_toimen",
               "riichi_sutehai_kamicha", "riichi_declared_opponents",
               "riichi_accepted_opponents")
_EXPECTED = {"tiles_left": 1, "honba": 1, "kyotaku": 1, "scores": 8, "round_index": 1,
             "tiles_seen": 1, "dora_indicators": 7, "kawa_overview_shimocha": 7,
             "kawa_overview_toimen": 7, "kawa_overview_kamicha": 7,
             "riichi_sutehai_shimocha": 3, "riichi_sutehai_toimen": 3,
             "riichi_sutehai_kamicha": 3, "riichi_declared_opponents": 3,
             "riichi_accepted_opponents": 3}


def feature_spec():
    return {
        "dim": FEATURE_DIM,
        "groups": [{"name": n, "dim": d,
                    "slice": [FEATURE_SLICES[n].start, FEATURE_SLICES[n].stop]}
                   for n, d in FEATURE_GROUPS],
        "source": "t12 context + t13 safety (public planes; per-opponent decomposition)",
        "basis": "public_info_predictor",
        "no_hidden_info": True,
    }


@lru_cache(maxsize=1)
def spans():
    """Layout spans; fail-loud on drift (same expected channels as the frozen extractor)."""
    top = layout.load_layout()
    out = {}
    drift = []
    for name in _SPAN_NAMES:
        off, count = layout.span(top, name)
        if count != _EXPECTED[name]:
            drift.append(f"{name}: expected {_EXPECTED[name]} planes, got {count}")
        out[name] = (off, count)
    if drift:
        raise ValueError("obs layout drift: " + "; ".join(drift))
    return out


def public_features(obs: np.ndarray, at_turns: np.ndarray, span_map: dict) -> np.ndarray:
    """Exact copy of the frozen training extractor (t14) for one POV decision stream."""
    n = obs.shape[0]
    seen = np.rint(obs[:, span_map["tiles_seen"][0]] * 4).astype(np.float32)
    tiles_left = obs[:, span_map["tiles_left"][0], 0] * 69.0
    honba = obs[:, span_map["honba"][0], 0] * 10.0
    kyotaku = obs[:, span_map["kyotaku"][0], 0] * 10.0
    so = span_map["scores"][0]
    scores = obs[:, so:so + 8:2, 0] * 100_000.0
    round_index = obs[:, span_map["round_index"][0], 0] * 7.0
    dora_off = span_map["dora_indicators"][0]
    dora_ind = obs[:, dora_off:dora_off + 4].sum(axis=1) > 0
    dora = np.zeros((n, N_TILES), dtype=np.float32)
    dora[:, _NEXT_TILE] = dora_ind.astype(np.float32)

    rivers = []
    for name in ("kawa_overview_shimocha", "kawa_overview_toimen", "kawa_overview_kamicha"):
        off = span_map[name][0]
        rivers.append(np.rint(obs[:, off:off + 4].sum(axis=1)).astype(np.float32))
    river = np.stack(rivers, axis=1)
    sutehai = []
    for name in ("riichi_sutehai_shimocha", "riichi_sutehai_toimen", "riichi_sutehai_kamicha"):
        plane = obs[:, span_map[name][0]]
        sutehai.append(np.where(plane.max(axis=1) > 0, plane.argmax(axis=1), -1))
    sutehai = np.stack(sutehai, axis=1)

    do = span_map["riichi_declared_opponents"][0]
    ao = span_map["riichi_accepted_opponents"][0]
    declared = (obs[:, do:do + 3, 0] > 0).astype(np.float32)
    accepted = (obs[:, ao:ao + 3, 0] > 0).astype(np.float32)
    threat = (declared + accepted) > 0
    threat_f = threat[:, :, None]
    genbutsu = threat_f & ((river > 0) | (sutehai[:, :, None] == TILE_RANGE[None, None, :]))
    suji = threat_f & ~genbutsu & ((river @ _SUJI_ADJ) > 0)
    gen_c = genbutsu.sum(axis=1).astype(np.float32)
    suji_c = suji.sum(axis=1).astype(np.float32)
    score = (safety_provider.SCORE_VISIBLE_FOUR * (seen == 4)
             + safety_provider.SCORE_PER_GENBUTSU * gen_c
             + safety_provider.SCORE_PER_SUJI * suji_c
             + seen
             - safety_provider.SCORE_DORA_PENALTY * dora)
    sutehai_oh = (sutehai[:, :, None] == TILE_RANGE[None, None, :]).astype(np.float32)

    feats = np.concatenate([
        (tiles_left / 69.0)[:, None],
        (honba / 10.0)[:, None],
        (kyotaku / 10.0)[:, None],
        scores / 100_000.0,
        (round_index / 7.0)[:, None],
        (at_turns.astype(np.float32) / 17.0)[:, None],
        seen / 4.0,
        dora,
        score,
        (river / 4.0).reshape(n, -1),
        genbutsu.astype(np.float32).reshape(n, -1),
        suji.astype(np.float32).reshape(n, -1),
        declared,
        accepted,
        sutehai_oh.reshape(n, -1),
        (threat.sum(axis=1, keepdims=True) / 3.0),
    ], axis=1).astype(np.float32)
    if feats.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"feature dim {feats.shape[1]} != frozen {FEATURE_DIM}")
    return feats


def features_for_decisions(decisions) -> np.ndarray:
    """Stack decision observations and return (n, 526) float32 public features."""
    if not decisions:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    obs = np.stack([np.asarray(dec.obs) for dec in decisions])
    at_turns = np.array([int(dec.at_turn) for dec in decisions], dtype=np.uint8)
    return public_features(obs, at_turns, spans())
