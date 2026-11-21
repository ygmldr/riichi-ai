import gzip
import hashlib
import json
from collections import defaultdict

import numpy as np

from libriichi.state import PlayerState

TYPE_NONE = 0
TYPE_START_GAME = 1
TYPE_START_KYOKU = 2
TYPE_TSUMO = 3
TYPE_DAHAI = 4
TYPE_CHI = 5
TYPE_PON = 6
TYPE_DAIMINKAN = 7
TYPE_KAKAN = 8
TYPE_ANKAN = 9
TYPE_DORA = 10
TYPE_REACH = 11
TYPE_REACH_ACCEPTED = 12
TYPE_HORA = 13
TYPE_RYUKYOKU = 14
TYPE_END_KYOKU = 15
TYPE_END_GAME = 16

TYPE_BY_NAME = {
    'none': TYPE_NONE,
    'start_game': TYPE_START_GAME,
    'start_kyoku': TYPE_START_KYOKU,
    'tsumo': TYPE_TSUMO,
    'dahai': TYPE_DAHAI,
    'chi': TYPE_CHI,
    'pon': TYPE_PON,
    'daiminkan': TYPE_DAIMINKAN,
    'kakan': TYPE_KAKAN,
    'ankan': TYPE_ANKAN,
    'dora': TYPE_DORA,
    'reach': TYPE_REACH,
    'reach_accepted': TYPE_REACH_ACCEPTED,
    'hora': TYPE_HORA,
    'ryukyoku': TYPE_RYUKYOKU,
    'end_kyoku': TYPE_END_KYOKU,
    'end_game': TYPE_END_GAME,
}

TILE_NONE = 34
ACTOR_NONE = 4
FLAG_AKA = 1
FLAG_TSUMOGIRI = 2
FLAG_CONSUMED_AKA = 4

TRIGGER_TYPES = {
    TYPE_TSUMO,
    TYPE_DAHAI,
    TYPE_CHI,
    TYPE_PON,
    TYPE_DAIMINKAN,
    TYPE_KAKAN,
    TYPE_ANKAN,
    TYPE_REACH,
    TYPE_REACH_ACCEPTED,
}

NON_DECISION_TRIGGERS = {'hora', 'ryukyoku', 'end_kyoku'}

_TILE_ID = {}
for _c in 'mps':
    for _n in range(1, 10):
        _TILE_ID[f'{_n}{_c}'] = len(_TILE_ID)
for _t, _i in zip('ESWN PFC'.replace(' ', ''), range(27, 34)):
    _TILE_ID[_t] = _i
_AKA_DEAKA = {'5mr': 4, '5pr': 13, '5sr': 22}


def _tile(s):
    if s in _AKA_DEAKA:
        return _AKA_DEAKA[s], True
    if s == '?':
        return TILE_NONE, False
    return _TILE_ID[s], False


def _consumed_flags(consumed):
    return FLAG_CONSUMED_AKA if any(_tile(t)[1] for t in consumed) else 0


def encode_event(ev, turn):
    t = ev['type']
    tid = TYPE_BY_NAME[t]
    if t in ('start_game', 'ryukyoku', 'end_kyoku', 'end_game'):
        return [tid, TILE_NONE, ACTOR_NONE, turn, 0]
    if t == 'start_kyoku':
        tile, aka = _tile(ev['dora_marker'])
        return [tid, tile, ACTOR_NONE, turn, FLAG_AKA if aka else 0]
    if t == 'tsumo':
        tile, aka = _tile(ev['pai'])
        return [tid, tile, ev['actor'], turn, FLAG_AKA if aka else 0]
    if t == 'dahai':
        tile, aka = _tile(ev['pai'])
        flags = (FLAG_AKA if aka else 0) | (FLAG_TSUMOGIRI if ev.get('tsumogiri') else 0)
        return [tid, tile, ev['actor'], turn, flags]
    if t in ('chi', 'pon', 'daiminkan', 'kakan'):
        tile, aka = _tile(ev['pai'])
        flags = (FLAG_AKA if aka else 0) | _consumed_flags(ev['consumed'])
        return [tid, tile, ev['actor'], turn, flags]
    if t == 'ankan':
        tile, aka = _tile(ev['consumed'][0])
        return [tid, tile, ev['actor'], turn, (FLAG_AKA if aka else 0) | _consumed_flags(ev['consumed'])]
    if t == 'dora':
        tile, aka = _tile(ev['dora_marker'])
        return [tid, tile, ACTOR_NONE, turn, FLAG_AKA if aka else 0]
    if t in ('reach', 'reach_accepted', 'hora'):
        return [tid, TILE_NONE, ev['actor'], turn, 0]
    raise ValueError(f'unknown event type {t}')


def encode_log(events, pov):
    out = []
    turn = 0
    for ev in events:
        if ev['type'] == 'tsumo' and ev['actor'] == pov:
            turn += 1
        out.append(encode_event(ev, turn))
    return np.asarray(out, dtype=np.int16)


def kyoku_segments(events):
    segments = []
    cur = None
    for ev in events:
        if ev['type'] == 'start_kyoku':
            if cur is not None:
                segments.append(cur)
            cur = []
        if cur is None:
            continue
        cur.append(ev)
        if ev['type'] == 'end_kyoku':
            segments.append(cur)
            cur = None
    if cur:
        segments.append(cur)
    return segments


def decision_points(events, pov):
    st = PlayerState(pov)
    out = []
    for i, ev in enumerate(events):
        cans = st.update(json.dumps(ev))
        if cans.can_act:
            out.append(i + 1)
    return out


def expected_decisions(events, pov):
    out = []
    for k, seg in enumerate(kyoku_segments(events)):
        for n in decision_points(seg, pov):
            tokens = encode_log(seg[:n], pov)
            out.append({
                'kyoku': k,
                'n': n,
                'first_type': int(tokens[0, 0]) if n else None,
                'last_type': int(tokens[-1, 0]) if n else None,
                'hash': hashlib.md5(tokens.tobytes()).hexdigest(),
                'tokens': tokens,
            })
    return out


def strip_consecutive_duplicates(seq):
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def read_text(filename):
    with open(filename, 'rb') as f:
        blob = f.read()
    if blob[:2] == b'\x1f\x8b':
        blob = gzip.decompress(blob)
    return blob.decode()


def read_events(filename):
    return [json.loads(line) for line in read_text(filename).splitlines() if line.strip()]


def expected_by_hash(decisions):
    table = defaultdict(list)
    for d in decisions:
        table[d['hash']].append(d)
    return table
