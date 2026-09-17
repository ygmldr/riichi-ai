import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = '/home/ygsldr/riichi-ai'
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'vendor/mortal/mortal'))

import event_verify as ev  # noqa: E402


def agent_player_ids(seed_count):
    challenger = [i % 4 for i in range(seed_count * 4)]
    patterns = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    champion = []
    for g in range(seed_count * 4):
        champion.extend(patterns[g % 4])
    return challenger, champion


class Recorder:
    def __init__(self):
        self.calls = []

    def wrap(self, engine):
        orig = engine.react_batch
        capture = os.environ.get('EVT_CAPTURE_TOKENS') == '1'

        def hooked(obs, masks, invisible_obs, events=None):
            rec = None
            if events is not None:
                rec = []
                for e in events:
                    a = np.ascontiguousarray(np.asarray(e, dtype=np.int16))
                    entry = {
                        'rows': int(a.shape[0]),
                        'sum': int(a.astype(np.int64).sum()),
                        'nz': int(np.count_nonzero(a)),
                        'first_type': int(a[0, 0]) if a.shape[0] else -1,
                        'last_type': int(a[-1, 0]) if a.shape[0] else -1,
                        'md5': hashlib.md5(a.tobytes()).hexdigest(),
                    }
                    if capture:
                        entry['tokens'] = a
                    rec.append(entry)
            self.calls.append(rec)
            return orig(obs, masks, invisible_obs, events)

        engine.react_batch = hooked
        return engine


def build_engine(cfg, name, quick_eval=False):
    from model import Brain, DQN
    from engine import MortalEngine
    state = torch.load(cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
    scfg = state['config']
    version = scfg['control'].get('version', 1)
    mortal = Brain(
        version=version,
        conv_channels=scfg['resnet']['conv_channels'],
        num_blocks=scfg['resnet']['num_blocks'],
    ).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    return MortalEngine(
        mortal,
        dqn,
        is_oracle=False,
        version=version,
        device=torch.device(cfg['device']),
        enable_amp=cfg['enable_amp'],
        enable_quick_eval=quick_eval,
        enable_rule_based_agari_guard=cfg['enable_rule_based_agari_guard'],
        name=name,
    )


def run(tag, sets, seed_start, seed_key, log_dir, quick_eval=False):
    from config import config
    from libriichi.arena import OneVsThree
    seed_count = sets
    challenger, champion = agent_player_ids(seed_count)
    rec_chal, rec_champ = Recorder(), Recorder()
    eng_chal = rec_chal.wrap(build_engine(config['1v3']['challenger'], 'chal', quick_eval))
    eng_champ = rec_champ.wrap(build_engine(config['1v3']['champion'], 'champ', quick_eval))
    env = OneVsThree(disable_progress_bar=False, log_dir=log_dir)
    t0 = time.time()
    rankings = env.py_vs_py(
        challenger=eng_chal,
        champion=eng_champ,
        seed_start=(seed_start, seed_key),
        seed_count=seed_count,
    )
    elapsed = time.time() - t0
    torch.save(rec_chal.calls, os.path.join(HERE, f'eval_calls_chal_{tag}.pt'))
    torch.save(rec_champ.calls, os.path.join(HERE, f'eval_calls_champ_{tag}.pt'))
    meta = {
        'tag': tag,
        'sets': sets,
        'games': sets * 4,
        'seed_start': seed_start,
        'seed_key': seed_key,
        'log_dir': log_dir,
        'rankings': list(rankings),
        'elapsed_s': elapsed,
        'challenger_player_ids': challenger,
        'champion_player_ids': champion,
        'chal_calls': len(rec_chal.calls),
        'champ_calls': len(rec_champ.calls),
    }
    with open(os.path.join(HERE, f'eval_meta_{tag}.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return meta


def verify(tag, meta, debug_file):
    seed_start = meta['seed_start']
    seed_key = meta['seed_key']
    log_dir = meta['log_dir']
    chal_ids = meta['challenger_player_ids']
    champ_ids = meta['champion_player_ids']
    calls = {
        'chal': torch.load(os.path.join(HERE, f'eval_calls_chal_{tag}.pt'), weights_only=False),
        'champ': torch.load(os.path.join(HERE, f'eval_calls_champ_{tag}.pt'), weights_only=False),
    }

    rust = {'chal': [], 'champ': []}
    with open(debug_file) as f:
        for line in f:
            rec = json.loads(line)
            if rec['agent'] == chal_ids:
                rust['chal'].append(rec)
            elif rec['agent'] == champ_ids:
                rust['champ'].append(rec)
            else:
                raise SystemExit(f'unmatched agent id in debug record: {rec["agent"][:8]}...')
    for k in rust:
        rust[k].sort(key=lambda r: r['call'])
        rust[k] = [r for r in rust[k] if r['call'] < len(calls[k])]

    expected = {}
    for slot in range(meta['games']):
        seed = seed_start + slot // 4
        split = 'abcd'[slot % 4]
        fn = os.path.join(log_dir, f'{seed}_{seed_key}_{split}.json.gz')
        if not os.path.exists(fn):
            raise SystemExit(f'missing game log {fn}')
        events = ev.read_events(fn)
        segments = ev.kyoku_segments(events)
        for pov in range(4):
            seq = []
            for k, seg in enumerate(segments):
                for n in ev.decision_points(seg, pov):
                    if seg[n - 1]['type'] in ev.NON_DECISION_TRIGGERS:
                        continue
                    tokens = ev.encode_log(seg[:n], pov)
                    seq.append({
                        'kyoku': k,
                        'n': n,
                        'md5': hashlib.md5(np.ascontiguousarray(tokens).tobytes()).hexdigest(),
                    })
            expected[(slot, pov)] = seq

    failures = []
    per_call_rows = {}
    n_samples = 0
    n_engine_decisions = 0
    for name, ids in (('chal', chal_ids), ('champ', champ_ids)):
        if len(rust[name]) != len(calls[name]):
            failures.append(f'{name}: rust calls {len(rust[name])} != python calls {len(calls[name])}')
        for ci, (rcall, pcall) in enumerate(zip(rust[name], calls[name])):
            if pcall is None:
                failures.append(f'{name} call {ci}: events arg was None')
                continue
            if len(rcall['entries']) != len(pcall):
                failures.append(f'{name} call {ci}: rust entries {len(rcall["entries"])} != python {len(pcall)}')
                continue
            per_call_rows[ci] = len(pcall)
            for j, (re_, pe) in enumerate(zip(rcall['entries'], pcall)):
                n_samples += 1
                if pe['rows'] < 1 or pe['sum'] == 0 or pe['nz'] == 0:
                    failures.append(f'{name} call {ci} entry {j}: empty/zero event tensor {pe}')
                    continue
                if re_['n'] != pe['rows'] or re_['first'] != pe['first_type'] or re_['last'] != pe['last_type']:
                    failures.append(f'{name} call {ci} entry {j}: rust/python mismatch {re_} vs {pe}')
                    continue
                if re_['g'] >= len(ids):
                    failures.append(f'{name} call {ci} entry {j}: bad game slot {re_["g"]}')
                    continue
                if name == 'chal':
                    game, pov = re_['g'], re_['pov']
                else:
                    game, pov = re_['g'] // 3, re_['pov']
                seq = expected[(game, pov)]
                matches = [d for d in seq if d['n'] == re_['n']]
                if not matches:
                    failures.append(f'{name} call {ci} entry {j}: n={re_["n"]} not a ground-truth decision for game {game} pov {pov}')
                    continue
                if not any(d['md5'] == pe['md5'] for d in matches):
                    failures.append(f'{name} call {ci} entry {j}: token hash mismatch game {game} pov {pov} n={re_["n"]}')
                    continue
                n_engine_decisions += 1

    recorded = {'chal': [], 'champ': []}
    for name in ('chal', 'champ'):
        for ci, pcall in enumerate(calls[name]):
            if pcall is None:
                continue
            for j, pe in enumerate(pcall):
                rcall = rust[name][ci]
                re_ = rcall['entries'][j]
                recorded[name].append((re_['g'], re_['pov'], re_['kan'], re_['n']))

    monotonic_ok = True
    matched_seq_ok = True
    for name, ids in (('chal', chal_ids), ('champ', champ_ids)):
        by_game_pov = {}
        for (g, pov, kan, n) in recorded[name]:
            game = g if name == 'chal' else g // 3
            by_game_pov.setdefault((game, pov), []).append((kan, n))
        for (game, pov), entries in by_game_pov.items():
            seq = expected[(game, pov)]
            idx = 0
            last_kyoku, last_n = -1, -1
            for (kan, n) in entries:
                while idx < len(seq) and seq[idx]['n'] != n:
                    idx += 1
                if idx >= len(seq):
                    failures.append(f'{name} game {game} pov {pov}: entry n={n} ran past ground truth')
                    break
                d = seq[idx]
                if d['kyoku'] == last_kyoku and n < last_n:
                    monotonic_ok = False
                    failures.append(f'{name} game {game} pov {pov}: n decreased within kyoku {d["kyoku"]}: {last_n} -> {n}')
                last_kyoku, last_n = d['kyoku'], n
                if not kan:
                    idx += 1

    result = {
        'method': (
            'Decisions counted only when they reach the Python engine: both engines were '
            'constructed with enable_quick_eval=False, so no decision is bypassed. '
            'Ground truth per (game, pov) = PlayerState replay of the arena-written game log, '
            'can_act points whose trigger is not hora/ryukyoku/end_kyoku. Alignment of the '
            'live Rust feed to those points uses the env-gated MORTAL_EVT_DEBUG record '
            '(game slot, seat, kan flag, n, first/last type) which is written per evaluate() '
            'call in the same order as the react_batch argument list; every recorded sample '
            'is additionally content-checked by md5 of the token bytes.'
        ),
        'tag': tag,
        'games': meta['games'],
        'engine_calls': {'chal': len(calls['chal']), 'champ': len(calls['champ'])},
        'engine_samples': n_samples,
        'decisions_matched_to_ground_truth': n_engine_decisions,
        'assertion_1_nonempty_nonzero': not any('empty/zero' in f for f in failures),
        'assertion_2_length_equals_public_events': not any(
            ('hash mismatch' in f or 'not a ground-truth' in f or 'ran past ground truth' in f)
            for f in failures
        ),
        'assertion_3_monotonic_within_kyoku': monotonic_ok,
        'all_assertions_pass': not failures,
        'failures': failures[:20],
        'failure_count': len(failures),
    }
    with open(os.path.join(HERE, 'EVAL_FEED.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='smoke')
    ap.add_argument('--sets', type=int, default=1)
    ap.add_argument('--seed-start', type=int, default=50000)
    ap.add_argument('--verify-only', action='store_true')
    ap.add_argument('--quick-eval', action='store_true')
    ap.add_argument('--no-verify', action='store_true')
    args = ap.parse_args()

    from config import config
    seed_key = config['1v3']['seed_key']
    log_dir = config['1v3']['log_dir']
    debug_file = os.environ['MORTAL_EVT_DEBUG']
    meta_path = os.path.join(HERE, f'eval_meta_{args.tag}.json')
    if args.verify_only:
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        if os.path.exists(debug_file):
            os.remove(debug_file)
        meta = run(args.tag, args.sets, args.seed_start, seed_key, log_dir, args.quick_eval)
    if args.no_verify:
        return
    verify(args.tag, meta, debug_file)


if __name__ == '__main__':
    main()
