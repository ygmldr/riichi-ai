import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/ygsldr/riichi-ai/vendor/mortal/mortal')

import event_verify as ev  # noqa: E402
from event_data import collate_events  # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402

ALL_FILES = sorted(
    glob.glob('/home/ygsldr/riichi-ai/datasets/extracted/2022/**/*.mjson', recursive=True),
    key=os.path.getsize,
)
FILES = list(ALL_FILES[-12:])

report = {'files': [], 'collate': None, 'assertions': {}}
failures = []


def check(name, cond, detail=''):
    report['assertions'][name] = bool(cond)
    if not cond:
        failures.append(f'{name}: {detail}')


MAX_T = 0
n_decisions = 0
n_kan_dupes = 0
skipped_reasons = {}
n_skipped = 0
n_ds_points = 0
n_arena_points = 0
n_events_total = 0
n_aka_rep = 0
n_consumed_aka = 0
for fn in FILES:
    raw = ev.read_text(fn)
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    n_events_total += len(events)
    segments = ev.kyoku_segments(events)
    loader = GameplayLoader(version=4, oracle=False)
    games = loader.load_log(raw)
    file_stat = {'file': os.path.basename(fn), 'players': len(games), 'decisions': 0}
    for game in games:
        pov = game.take_player_id()
        obs = game.take_obs()
        evs = game.take_events()
        at_kyoku = game.take_at_kyoku()
        check('len(events) == len(obs) == len(actions)', len(evs) == len(obs))
        check('len(at_kyoku) == len(obs)', len(at_kyoku) == len(obs))
        ds_by_kyoku = {}
        for i, t in enumerate(evs):
            ds_by_kyoku.setdefault(at_kyoku[i], []).append(len(t))
        for k, seg in enumerate(segments):
            ds = ev.strip_consecutive_duplicates(ds_by_kyoku.get(k, []))
            truth = [n for n in ev.decision_points(seg, pov) if seg[n - 1]['type'] not in ev.NON_DECISION_TRIGGERS]
            n_ds_points += len(ds)
            n_arena_points += len(truth)
            if any(ds[i] >= ds[i + 1] for i in range(len(ds) - 1)):
                failures.append(f'dataset T not strictly increasing {fn} pov={pov} kyoku={k}: {ds}')
            if not set(ds) <= set(truth):
                failures.append(f'dataset decision not in ground truth {fn} pov={pov} kyoku={k}: {sorted(set(ds) - set(truth))}')
            missing = [n for n in truth if n not in ds]
            n_skipped += len(missing)
            for n in missing:
                nxt = seg[n]['type'] if n < len(seg) else 'GAME_END'
                skipped_reasons[nxt] = skipped_reasons.get(nxt, 0) + 1
        for i, tokens in enumerate(evs):
            T = tokens.shape[0]
            MAX_T = max(MAX_T, T)
            n_decisions += 1
            if i > 0 and len(evs[i - 1]) == T:
                n_kan_dupes += 1
            if tokens.shape[1] != 5 or tokens.dtype != np.int16:
                failures.append(f'bad shape/dtype {tokens.shape} {tokens.dtype}')
                continue
            if T < 1 or T > 384:
                failures.append(f'bad T={T}')
                continue
            seg = segments[at_kyoku[i]]
            expected_tokens = ev.encode_log(seg[:T], pov)
            if not np.array_equal(tokens, expected_tokens):
                failures.append(f'tokens mismatch {fn} pov={pov} decision={i} T={T}')
                continue
            if tokens[0, 0] != ev.TYPE_START_KYOKU:
                failures.append(f'first token not start_kyoku {fn} pov={pov} i={i}')
            n_aka_rep += int(((tokens[:, 4] & 1) != 0).sum())
            n_consumed_aka += int(((tokens[:, 4] & 4) != 0).sum())
            if int(tokens[-1, 0]) not in ev.TRIGGER_TYPES:
                failures.append(f'last token not a trigger type {fn} pov={pov} i={i} {tokens[-1]}')
            tile, aka = ev._tile(seg[0]['dora_marker'])
            if int(tokens[0, 1]) != tile:
                failures.append(f'first tile != dora marker {fn} pov={pov} i={i}')
        file_stat['decisions'] += len(evs)
    report['files'].append(file_stat)

check('no per-decision failures', not failures, '; '.join(failures[:5]))
check('max T <= 384', MAX_T <= 384, str(MAX_T))

# Synthetic log: ankan whose representative tile (consumed[0]) is the aka five.
# This exact combination is rare in the corpus but was the source of a real
# encoder divergence found by the 18-C feed gate, so it is pinned here.
syn_tehais = [
    ['5mr', '5m', '5m', '5m', '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p'],
    ['1m', '2m', '3m', '4m', '6m', '7m', '8m', '9m', '1s', '2s', '3s', '4s', '6s'],
    ['5s', '6s', '7s', '8s', '9s', 'E', 'S', 'W', 'N', 'P', 'F', 'C', '9m'],
    ['1m', '2m', '3m', '4m', '6m', '7m', '8m', '9m', '1p', '2p', '3p', '4p', '7s'],
]
syn = [
    {'type': 'start_game', 'names': ['a', 'b', 'c', 'd']},
    {'type': 'start_kyoku', 'bakaze': 'E', 'dora_marker': '1m', 'kyoku': 1, 'honba': 0,
     'kyotaku': 0, 'oya': 0, 'scores': [25000, 25000, 25000, 25000], 'tehais': syn_tehais},
    {'type': 'tsumo', 'actor': 0, 'pai': '1s'},
    {'type': 'ankan', 'actor': 0, 'consumed': ['5mr', '5m', '5m', '5m']},
    {'type': 'dora', 'dora_marker': '2p'},
    {'type': 'tsumo', 'actor': 0, 'pai': '3s'},
    {'type': 'dahai', 'actor': 0, 'pai': '1s', 'tsumogiri': False},
    {'type': 'end_kyoku'},
    {'type': 'end_game'},
]
syn_raw = '\n'.join(json.dumps(e, separators=(',', ':')) for e in syn)
syn_segments = ev.kyoku_segments(syn)
syn_games = GameplayLoader(version=4, oracle=False).load_log(syn_raw)
syn_ankan_flags = []
syn_ok = True
syn_decisions = 0
for game in syn_games:
    pov = game.take_player_id()
    syn_evs = game.take_events()
    syn_at_kyoku = game.take_at_kyoku()
    syn_decisions += len(syn_evs)
    for i, tokens in enumerate(syn_evs):
        exp = ev.encode_log(syn_segments[syn_at_kyoku[i]][:len(tokens)], pov)
        if not np.array_equal(tokens, exp):
            syn_ok = False
        for row in tokens:
            if row[0] == ev.TYPE_ANKAN:
                syn_ankan_flags.append(int(row[4]))
check('synthetic ankan-aka tokens match independent encoder', syn_ok)
check('synthetic ankan-aka flags == aka|consumed_aka', syn_ankan_flags == [5] * len(syn_ankan_flags) and len(syn_ankan_flags) > 0,
      str(syn_ankan_flags))
report['synthetic'] = {'ankan_flags': syn_ankan_flags, 'decisions': syn_decisions}

a = np.tile(np.arange(5, dtype=np.int16), (7, 1))
a[:, 0] = ev.TYPE_DAHAI
b = (np.tile(np.arange(5, dtype=np.int16), (3, 1)) + 100).astype(np.int16)
b[:, 0] = ev.TYPE_TSUMO
out = collate_events([a, b])
report['collate'] = {
    'tokens_shape': list(out['tokens'].shape),
    'tokens_dtype': str(out['tokens'].dtype),
    'lengths': out['lengths'].tolist(),
    'mask_counts': out['mask'].sum(-1).tolist(),
    'causal_shape': list(out['causal'].shape),
}
check('collate pad shape', tuple(out['tokens'].shape) == (2, 7, 5))
check('collate dtype int16', out['tokens'].dtype == torch.int16)
check('collate lengths', out['lengths'].tolist() == [7, 3])
check('collate padding zero', bool((out['tokens'][1, 3:] == 0).all()))
check('collate mask', out['mask'].sum(-1).tolist() == [7, 3])
check('collate causal tril', bool(out['causal'].tril().equal(out['causal'])))
check('collate causal strict upper false', not bool(out['causal'][0, 1]))
check('collate tokens roundtrip', np.array_equal(out['tokens'][0].numpy(), a)
      and np.array_equal(out['tokens'][1, :3].numpy(), b))

report['totals'] = {
    'decisions': n_decisions,
    'kan_select_duplicates': n_kan_dupes,
    'max_T': MAX_T,
    'dataset_decision_points': n_ds_points,
    'arena_ground_truth_points': n_arena_points,
    'skipped_preempted_points': n_skipped,
    'skipped_by_next_event_type': skipped_reasons,
    'total_events': n_events_total,
    'aka_representative_tokens': n_aka_rep,
    'consumed_aka_tokens': n_consumed_aka,
    'failures': failures,
}
with open(os.path.join(HERE, 'dataset_smoke.json'), 'w') as f:
    json.dump(report, f, indent=2)

print(json.dumps(report['totals'], indent=2))
print('assertions:', json.dumps(report['assertions'], indent=2))
print('FILES:', [os.path.basename(x) for x in FILES])
sys.exit(1 if failures else 0)
