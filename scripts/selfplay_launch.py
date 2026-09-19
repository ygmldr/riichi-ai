#!/usr/bin/env python3
"""Online self-play orchestration (plan todo 24).

    python3 scripts/selfplay_launch.py prepare [--k 2] [--games 100]
    python3 scripts/selfplay_launch.py launch
    python3 scripts/selfplay_launch.py status
    python3 scripts/selfplay_launch.py stop

`prepare` writes one config copy per worker under `runs/exp/SELFPLAY/` with a
distinct `[baseline.train].state_file` (frozen opponent snapshot) and a distinct
`[train_play.default].log_dir` (`runs/exp/SELFPLAY/worker_<k>/`), plus scratch
server/trainer configs and `worker_map.json`. `launch` starts `server.py`, then
`train.py` (it must submit the first param before workers can fetch one), then
K x `client.py`, each with its own `MORTAL_CFG`, detached (`start_new_session`).

Destructive-op guard: `player.py` rmtree's `log_dir` at the start of every
session, so both `prepare` and `launch` refuse any log_dir that is not a fresh
directory under `runs/exp/SELFPLAY/` (must not exist or be empty).
"""

import argparse
import datetime
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import tomllib
from copy import deepcopy

REPO = pathlib.Path('/home/ygsldr/riichi-ai')
SELFPLAY = REPO / 'runs/exp/SELFPLAY'
MORTAL_DIR = REPO / 'vendor/mortal/mortal'
LIBRIICHI = REPO / 'vendor/mortal/target/release'
REPO_CFG = REPO / 'configs/config.toml'
LOCK = pathlib.Path('/tmp/opencode/train.lock')
PIDS = SELFPLAY / 'pids.json'
LOGS = SELFPLAY / 'logs'
POOL = (
    {'id': 'at-175200', 'state_file': REPO / 'runs/exp/PINNED/mortal.pth.at-175200'},
    {'id': 'at-148800', 'state_file': REPO / 'checkpoints/mortal.pth.at-148800'},
)
LIVE = {
    'state_file': REPO / 'checkpoints/mortal.pth',
    'best_state_file': REPO / 'checkpoints/best.pth',
    'tensorboard_dir': REPO / 'runs',
    'test_play_log_dir': REPO / 'runs/test_play',
    'train_play_log_dir': REPO / 'runs/train_play',
}


def load_cfg(path):
    with open(path, 'rb') as f:
        return tomllib.load(f)


def is_under(path, root):
    return pathlib.Path(path).resolve().is_relative_to(root.resolve())


def assert_fresh_log_dir(path, worker):
    p = pathlib.Path(path)
    if not is_under(p, SELFPLAY) or p.resolve() == SELFPLAY.resolve():
        raise SystemExit(f'REFUSING worker_{worker}: log_dir not under {SELFPLAY}: {p}')
    if p.exists() and any(p.iterdir()):
        raise SystemExit(
            f'REFUSING worker_{worker}: log_dir is not fresh (non-empty, '
            f'training player would rmtree it): {p}'
        )
    state = 'missing' if not p.exists() else 'empty'
    print(f'  worker_{worker} log_dir OK ({state}, fresh): {p}')


def assert_worker_cfg(path, cfg, expect_games):
    baseline = cfg['baseline']['train']['state_file']
    log_dir = cfg['train_play']['default']['log_dir']
    if not pathlib.Path(baseline).is_file():
        raise SystemExit(f'REFUSING: opponent snapshot missing: {baseline}')
    if cfg['train_play']['default']['games'] != expect_games:
        raise SystemExit(f'REFUSING: {path}: games={cfg["train_play"]["default"]["games"]} != {expect_games}')
    return pathlib.Path(baseline), pathlib.Path(log_dir)


def assert_distinct(items, what):
    values = [str(v) for v in items]
    if len(set(values)) != len(values):
        raise SystemExit(f'REFUSING: {what} are not pairwise distinct: {values}')
    print(f'  {what} pairwise distinct: {values}')


def cmd_prepare(args):
    import toml

    cfg = load_cfg(REPO_CFG)
    if cfg['control']['online'] is not True:
        raise SystemExit(f'REFUSING: {REPO_CFG} [control].online is not true')
    if cfg['freeze_bn']['mortal'] is not True:
        raise SystemExit(f'REFUSING: {REPO_CFG} [freeze_bn].mortal is not true')
    if args.games % 4 != 0:
        raise SystemExit(f'REFUSING: --games {args.games} must be a multiple of 4 (seed_count = games//4)')
    if args.k > len(POOL):
        raise SystemExit(f'REFUSING: K={args.k} > pool size {len(POOL)} (every worker needs a distinct snapshot)')
    print(f'PREPARE: repo config online=true, freeze_bn.mortal=true; K={args.k}, games/session={args.games}')

    SELFPLAY.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    workers = []
    for k in range(args.k):
        log_dir = SELFPLAY / f'worker_{k}'
        member = POOL[k]
        assert_fresh_log_dir(log_dir, k)
        wcfg = deepcopy(cfg)
        wcfg['baseline']['train']['state_file'] = str(member['state_file'])
        wcfg['train_play']['default']['log_dir'] = str(log_dir)
        wcfg['train_play']['default']['games'] = args.games
        cfg_path = SELFPLAY / f'worker_{k}.toml'
        with open(cfg_path, 'w', encoding='utf-8') as f:
            toml.dump(wcfg, f)
        workers.append({
            'worker': k,
            'config': str(cfg_path),
            'pool_member_id': member['id'],
            'opponent_state_file': str(member['state_file']),
            'log_dir': str(log_dir),
            'games_per_session': args.games,
        })
        print(f'  wrote {cfg_path}: opponent={member["id"]} log_dir={log_dir}')

    assert_distinct([w['log_dir'] for w in workers], 'worker log_dirs')
    assert_distinct([w['opponent_state_file'] for w in workers], 'worker opponent snapshots')

    for k, w in enumerate(workers):
        cfg_w = load_cfg(w['config'])
        baseline, log_dir = assert_worker_cfg(w['config'], cfg_w, args.games)
        assert str(baseline) == w['opponent_state_file'], f'worker_{k} baseline mismatch'
        assert str(log_dir) == w['log_dir'], f'worker_{k} log_dir mismatch'

    trainer_dir = SELFPLAY / 'trainer'
    tcfg = deepcopy(cfg)
    tcfg['control']['state_file'] = str(trainer_dir / 'state.pth')
    tcfg['control']['best_state_file'] = str(trainer_dir / 'best.pth')
    tcfg['control']['tensorboard_dir'] = str(trainer_dir / 'tb')
    tcfg['test_play']['log_dir'] = str(trainer_dir / 'test_play')
    for key in ('state_file', 'best_state_file', 'tensorboard_dir'):
        value = pathlib.Path(tcfg['control'][key])
        if value.resolve() == LIVE[key].resolve() or not is_under(value, SELFPLAY):
            raise SystemExit(f'REFUSING trainer {key}: {value} (must be a scratch path under {SELFPLAY})')
    if not is_under(tcfg['test_play']['log_dir'], SELFPLAY):
        raise SystemExit(f'REFUSING trainer test_play.log_dir: {tcfg["test_play"]["log_dir"]}')
    tcfg_path = SELFPLAY / 'trainer.toml'
    with open(tcfg_path, 'w', encoding='utf-8') as f:
        toml.dump(tcfg, f)
    print(f'  wrote {tcfg_path}: state/tb all under {trainer_dir}')

    scfg = deepcopy(cfg)
    scfg['online']['server']['buffer_dir'] = str(SELFPLAY / 'server/buffer')
    scfg['online']['server']['drain_dir'] = str(SELFPLAY / 'server/drain')
    for key in ('buffer_dir', 'drain_dir'):
        if not is_under(scfg['online']['server'][key], SELFPLAY):
            raise SystemExit(f'REFUSING server {key}: {scfg["online"]["server"][key]}')
    scfg_path = SELFPLAY / 'server.toml'
    with open(scfg_path, 'w', encoding='utf-8') as f:
        toml.dump(scfg, f)
    print(f'  wrote {scfg_path}: buffer/drain under {SELFPLAY / "server"}')

    worker_map = {
        'experiment': 'SELFPLAY',
        'todo': 24,
        'created_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'k': args.k,
        'games_per_session': args.games,
        'trainee': {'seat_count': 1, 'engine_name': 'trainee', 'player_names': ['trainee']},
        'opponent_seats_per_game': 3,
        'pool': [
            {'id': m['id'], 'state_file': str(m['state_file']), 'bytes': m['state_file'].stat().st_size}
            for m in POOL
        ],
        'workers': workers,
    }
    with open(SELFPLAY / 'worker_map.json', 'w', encoding='utf-8') as f:
        json.dump(worker_map, f, indent=2)
        f.write('\n')
    print(f'  wrote {SELFPLAY / "worker_map.json"}')
    print('PREPARE OK')


def require_lock():
    if not (LOCK / 'owner.txt').is_file():
        raise SystemExit(f'REFUSING launch: {LOCK} not held (acquire it first)')
    print(f'  lock held: {(LOCK / "owner.txt").read_text().strip()}')


def wait_port(host, port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def wait_log(path, needle, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.is_file() and needle in path.read_text(errors='replace'):
            return True
        time.sleep(1)
    return False


def start(name, script, cfg_path):
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['MORTAL_CFG'] = str(pathlib.Path(cfg_path).resolve())
    env['PYTHONPATH'] = f'{LIBRIICHI}{os.pathsep}{env["PYTHONPATH"]}' if env.get('PYTHONPATH') else str(LIBRIICHI)
    log = LOGS / f'{name}.log'
    with open(log, 'ab') as out:
        proc = subprocess.Popen(
            [sys.executable, script],
            cwd=MORTAL_DIR,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f'  started {name}: pid={proc.pid} pgid={proc.pid} cfg={env["MORTAL_CFG"]} log={log}')
    return proc.pid


def cmd_launch(args):
    require_lock()
    cfg = load_cfg(REPO_CFG)
    if cfg['control']['online'] is not True or cfg['freeze_bn']['mortal'] is not True:
        raise SystemExit('REFUSING: repo config online/freeze_bn flags are not set')
    if not (SELFPLAY / 'worker_map.json').is_file():
        raise SystemExit('REFUSING: run `prepare` first (worker_map.json missing)')

    workers = json.loads((SELFPLAY / 'worker_map.json').read_text())['workers']
    if args.k is not None:
        workers = workers[:args.k]
    print(f'LAUNCH: K={len(workers)} workers')
    expected_games = json.loads((SELFPLAY / 'worker_map.json').read_text())['games_per_session']
    baselines = []
    log_dirs = []
    for w in workers:
        assert_fresh_log_dir(w['log_dir'], w['worker'])
        baseline, log_dir = assert_worker_cfg(w['config'], load_cfg(w['config']), expected_games)
        baselines.append(baseline)
        log_dirs.append(log_dir)
    assert_distinct(log_dirs, 'worker log_dirs')
    assert_distinct(baselines, 'worker opponent snapshots')

    remote = cfg['online']['remote']
    pids = {
        'started_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'configs': {
            'server': str(SELFPLAY / 'server.toml'),
            'trainer': str(SELFPLAY / 'trainer.toml'),
            'workers': [w['config'] for w in workers],
        },
    }
    pids['server'] = start('server', 'server.py', SELFPLAY / 'server.toml')
    if not wait_port(remote['host'], remote['port'], 60):
        raise SystemExit(f'REFUSING: server did not listen on {remote["host"]}:{remote["port"]} within 60s')
    print(f'  server listening on {remote["host"]}:{remote["port"]}')

    pids['trainer'] = start('trainer', 'train.py', SELFPLAY / 'trainer.toml')
    if not wait_log(LOGS / 'trainer.log', 'param has been submitted', 300):
        raise SystemExit('REFUSING: trainer did not submit its first param within 300s')
    print('  trainer submitted the first param')

    for w in workers:
        pids[f'worker_{w["worker"]}'] = start(f'worker_{w["worker"]}', 'client.py', w['config'])

    with open(PIDS, 'w', encoding='utf-8') as f:
        json.dump(pids, f, indent=2)
        f.write('\n')
    print(f'  wrote {PIDS}')
    print('LAUNCH OK')


def process_rss(pid):
    try:
        out = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], capture_output=True, text=True, check=True)
        return int(out.stdout.strip()) // 1024
    except (subprocess.CalledProcessError, ValueError):
        return None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def log_count(path, needle):
    if not pathlib.Path(path).is_file():
        return 0
    return pathlib.Path(path).read_text(errors='replace').count(needle)


def cmd_status(args):
    pids = json.loads(PIDS.read_text())
    status = {'pids': {}, 'workers': [], 'server_dirs': {}}
    for role, pid in pids.items():
        if not isinstance(pid, int):
            continue
        status['pids'][role] = {'pid': pid, 'alive': alive(pid), 'rss_mib': process_rss(pid)}
    for w in json.loads((SELFPLAY / 'worker_map.json').read_text())['workers']:
        log_dir = pathlib.Path(w['log_dir'])
        files = sorted(log_dir.glob('*.json.gz')) if log_dir.is_dir() else []
        log = LOGS / f'worker_{w["worker"]}.log'
        status['workers'].append({
            'worker': w['worker'],
            'opponent': w['pool_member_id'],
            'log_dir': w['log_dir'],
            'current_session_files': len(files),
            'sessions_submitted': log_count(log, 'logs have been submitted'),
            'rss_mib': process_rss(pids.get(f'worker_{w["worker"]}', -1)),
        })
    for key in ('buffer', 'drain'):
        d = SELFPLAY / 'server' / key
        status['server_dirs'][key] = len(list(d.glob('*.json.gz'))) if d.is_dir() else 0
    gpu = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
        capture_output=True, text=True,
    ).stdout.strip()
    status['gpu_compute_apps'] = gpu or '(none)'
    print(json.dumps(status, indent=2))


def cmd_stop(args):
    pids = json.loads(PIDS.read_text())
    targets = [(role, pid) for role, pid in pids.items() if isinstance(pid, int)]
    for role, pid in targets:
        if not alive(pid):
            print(f'  {role} pid={pid} already dead')
            continue
        try:
            os.killpg(pid, signal.SIGTERM)
            print(f'  {role} pid={pid}: SIGTERM to pgid {pid}')
        except ProcessLookupError:
            print(f'  {role} pid={pid}: gone before SIGTERM')
    deadline = time.time() + 20
    while time.time() < deadline:
        if not any(alive(pid) for _, pid in targets):
            break
        time.sleep(0.5)
    for role, pid in targets:
        if alive(pid):
            os.killpg(pid, signal.SIGKILL)
            print(f'  {role} pid={pid}: SIGKILL after timeout')
    leftovers = subprocess.run(
        ['pgrep', '-af', r'python3 (server|client|train)\.py'], capture_output=True, text=True,
    ).stdout.strip()
    print(f'  pgrep leftovers: {leftovers or "(none)"}')
    pids['stopped_at'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    with open(PIDS, 'w', encoding='utf-8') as f:
        json.dump(pids, f, indent=2)
        f.write('\n')
    if leftovers:
        raise SystemExit('STOP INCOMPLETE: processes remain')
    print('STOP OK')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p_prepare = sub.add_parser('prepare')
    p_prepare.add_argument('--k', type=int, default=2)
    p_prepare.add_argument('--games', type=int, default=100)
    p_prepare.set_defaults(func=cmd_prepare)
    p_launch = sub.add_parser('launch')
    p_launch.add_argument('--k', type=int, default=None)
    p_launch.set_defaults(func=cmd_launch)
    sub.add_parser('status').set_defaults(func=cmd_status)
    sub.add_parser('stop').set_defaults(func=cmd_stop)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
