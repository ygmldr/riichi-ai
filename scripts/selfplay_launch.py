#!/usr/bin/env python3
"""Online self-play orchestration + frozen opponent pool (plan todos 24/25).

    python3 scripts/selfplay_launch.py validate [--pool POOL]
    python3 scripts/selfplay_launch.py sample  --cycle 0 [--k 2] [--write]
    python3 scripts/selfplay_launch.py prepare --cycle 0 --wave 0 [--k 2] [--games 4]
    python3 scripts/selfplay_launch.py launch
    python3 scripts/selfplay_launch.py status
    python3 scripts/selfplay_launch.py stop

`runs/exp/SELFPLAY/pool.json` (todo 25) is the single source of truth for the
frozen opponent pool: members (path + bytes + sha256), active/retired status,
explicit sampling weights (sum = 1.0), and the per-cycle assignment schedule
produced by the weighted-without-replacement sampler. `prepare` validates every
active member (missing file or sha256 mismatch -> hard SystemExit, never a
silent downgrade) and materialises one config per worker under
`cycle_<g>/wave_<w>/`; `worker_map.json` (root = latest wave + a copy inside
the wave dir) records worker -> member -> log_dir. `launch` re-validates the
pool, checks the prepared map is not stale, then starts server -> trainer ->
K x `client.py`, each with its own `MORTAL_CFG`, detached (`start_new_session`).
No server protocol change: membership lives entirely in per-worker configs and
the pool file.

Destructive-op guard: `player.py` rmtree's `log_dir` at the start of every
session, so both `prepare` and `launch` refuse any log_dir that is not a fresh
directory under `runs/exp/SELFPLAY/` (must not exist or be empty).
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
from copy import deepcopy

REPO = pathlib.Path('/home/ygsldr/riichi-ai')
SELFPLAY = REPO / 'runs/exp/SELFPLAY'
POOL_FILE = SELFPLAY / 'pool.json'
MORTAL_DIR = REPO / 'vendor/mortal/mortal'
LIBRIICHI = REPO / 'vendor/mortal/target/release'
REPO_CFG = REPO / 'configs/config.toml'
LOCK = pathlib.Path('/tmp/opencode/train.lock')
PIDS = SELFPLAY / 'pids.json'
LIVE = {
    'state_file': REPO / 'checkpoints/mortal.pth',
    'best_state_file': REPO / 'checkpoints/best.pth',
    'tensorboard_dir': REPO / 'runs',
    'test_play_log_dir': REPO / 'runs/test_play',
    'train_play_log_dir': REPO / 'runs/train_play',
}


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def load_cfg(path):
    with open(path, 'rb') as f:
        return tomllib.load(f)


def is_under(path, root):
    return pathlib.Path(path).resolve().is_relative_to(root.resolve())


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- pool (todo 25)

def load_pool(pool_path):
    pool_path = pathlib.Path(pool_path)
    if not pool_path.is_file():
        raise SystemExit(f'REFUSING: opponent pool file missing: {pool_path}')
    with open(pool_path, encoding='utf-8') as f:
        return json.load(f)


def active_members(pool):
    return [m for m in pool['members'] if m.get('status') == 'active']


def validate_pool(pool, pool_path):
    """Hard-check every active member (file + bytes + sha256) and the weights.

    Any problem aborts (SystemExit != 0): a missing or altered snapshot must
    fail startup loudly, never silently downgrade to another opponent.
    Returns {id: member} for the active set.
    """
    pool_path = pathlib.Path(pool_path)
    active = active_members(pool)
    if not active:
        raise SystemExit(f'REFUSING: pool has no active members: {pool_path}')
    by_id = {m['id']: m for m in active}
    if len(by_id) != len(active):
        raise SystemExit(f'REFUSING: duplicate member ids in {pool_path}')

    weights = pool['weighting']['weights']
    errors = []
    print(f'POOL VALIDATE: {pool_path}')
    total = 0.0
    for mid in weights:
        if mid not in by_id:
            errors.append(f'weights has id not active in members: {mid}')
    for m in active:
        if m['id'] not in weights:
            errors.append(f'active member has no weight: {m["id"]}')
        else:
            total += weights[m['id']]
        p = pathlib.Path(m['path'])
        if not p.is_file():
            errors.append(f'missing snapshot: {m["id"]}: {p}')
            print(f'  [FAIL] {m["id"]}: file missing: {p}')
            continue
        size = p.stat().st_size
        if size != m['bytes']:
            errors.append(f'bytes mismatch: {m["id"]}: {p} file={size} pool={m["bytes"]}')
            print(f'  [FAIL] {m["id"]}: bytes {size} != recorded {m["bytes"]}: {p}')
            continue
        digest = sha256_file(p)
        if digest != m['sha256']:
            errors.append(f'sha256 mismatch: {m["id"]}: {p} file={digest} pool={m["sha256"]}')
            print(f'  [FAIL] {m["id"]}: sha256 {digest} != recorded {m["sha256"]}: {p}')
            continue
        print(f'  [OK] {m["id"]:12s} {size:>10d} B  sha256 {digest[:12]}...  '
              f'steps={m.get("steps")}  {p}')
    if abs(total - 1.0) > 1e-6:
        errors.append(f'weights sum {total!r} != 1.0 (tolerance 1e-6)')
        print(f'  [FAIL] weights sum {total!r} != 1.0')
    retired = [m['id'] for m in pool['members'] if m.get('status') == 'retired']
    if retired:
        print(f'  retired (kept on disk, not sampled): {retired}')
    if errors:
        for e in errors:
            print(f'  error: {e}')
        raise SystemExit(f'REFUSING: pool validation failed ({len(errors)} issue(s)): {pool_path}')
    print(f'  active members: {len(active)}  weights sum={total:.6f}  POOL OK')
    return by_id


def sample_cycle(pool, cycle, k):
    """Deterministic cycle schedule: weighted without replacement.

    Weights are the explicit pool probabilities. A cycle draws members one by
    one (probability w_i / sum(w_remaining), remove, repeat) until every active
    member is drawn exactly once; the drawn sequence is split into consecutive
    waves of k workers. So within a wave members are distinct and every active
    member appears exactly once per cycle.
    """
    active = active_members(pool)
    weights = pool['weighting']['weights']
    rng = random.Random(f'{pool["sampling"]["seed"]}:cycle:{cycle}')
    remaining = [m['id'] for m in active]
    drawn = []
    while remaining:
        total = sum(weights[i] for i in remaining)
        r = rng.random() * total
        acc = 0.0
        pick = remaining[-1]
        for mid in remaining:
            acc += weights[mid]
            if r < acc:
                pick = mid
                break
        drawn.append(pick)
        remaining.remove(pick)
    if len(set(drawn)) != len(active):
        raise SystemExit('BUG: sampler produced duplicates')
    waves = [drawn[i:i + k] for i in range(0, len(drawn), k)]
    return waves


def cmd_validate(args):
    pool = load_pool(args.pool)
    validate_pool(pool, args.pool)


def cmd_sample(args):
    pool = load_pool(args.pool)
    validate_pool(pool, args.pool)
    waves = sample_cycle(pool, args.cycle, args.k)
    key = f'cycle_{args.cycle}'
    print(json.dumps({key: waves, 'k': args.k}, indent=2))
    if args.write:
        assign = pool.setdefault('assign', {})
        old = assign.get(key)
        assign[key] = waves
        pool['sampled_at'] = now_iso()
        pool['cycle_length_waves'] = -(-len(active_members(pool)) // args.k)
        with open(args.pool, 'w', encoding='utf-8') as f:
            json.dump(pool, f, indent=2)
            f.write('\n')
        print(f'  wrote {key} to {args.pool} (was: {old})')


# ---------------------------------------------------------------- guards

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


def assert_distinct(items, what):
    values = [str(v) for v in items]
    if len(set(values)) != len(values):
        raise SystemExit(f'REFUSING: {what} are not pairwise distinct: {values}')
    print(f'  {what} pairwise distinct: {values}')


# ---------------------------------------------------------------- prepare

def cmd_prepare(args):
    import toml

    pool = load_pool(args.pool)
    by_id = validate_pool(pool, args.pool)
    cfg = load_cfg(REPO_CFG)
    if cfg['control']['online'] is not True:
        raise SystemExit(f'REFUSING: {REPO_CFG} [control].online is not true')
    if cfg['freeze_bn']['mortal'] is not True:
        raise SystemExit(f'REFUSING: {REPO_CFG} [freeze_bn].mortal is not true')
    if args.games % 4 != 0:
        raise SystemExit(f'REFUSING: --games {args.games} must be a multiple of 4 (seed_count = games//4)')

    key = f'cycle_{args.cycle}'
    assign = pool.get('assign', {}).get(key)
    if assign is None:
        raise SystemExit(
            f'REFUSING: no assign for {key} in {args.pool}; '
            f'run `sample --cycle {args.cycle} --k {args.k} --write` first'
        )
    if args.wave >= len(assign):
        raise SystemExit(f'REFUSING: wave {args.wave} >= cycle_length_waves {len(assign)} for {key}')
    picks = assign[args.wave]
    if len(picks) != args.k:
        raise SystemExit(f'REFUSING: {key} wave {args.wave} has {len(picks)} members != k={args.k}')
    if len(set(picks)) != len(picks):
        raise SystemExit(f'REFUSING: {key} wave {args.wave} has duplicate members: {picks}')
    unknown = [p for p in picks if p not in by_id]
    if unknown:
        raise SystemExit(f'REFUSING: assign names non-active members: {unknown}')

    wave_dir = SELFPLAY / f'cycle_{args.cycle}' / f'wave_{args.wave}'
    logs_dir = wave_dir / 'logs'
    print(f'PREPARE: {key} wave {args.wave} k={args.k} games/session={args.games} dir={wave_dir}')

    workers = []
    for slot, mid in enumerate(picks):
        member = by_id[mid]
        log_dir = wave_dir / f'worker_{slot}'
        assert_fresh_log_dir(log_dir, slot)
        wcfg = deepcopy(cfg)
        wcfg['baseline']['train']['state_file'] = member['path']
        wcfg['train_play']['default']['log_dir'] = str(log_dir)
        wcfg['train_play']['default']['games'] = args.games
        # client.py never reads [control].state_file; keep live checkpoints out of worker configs anyway (task MUST-NOT).
        wcfg['control']['state_file'] = str(wave_dir / f'worker_{slot}.unused-control-state.pth')
        wcfg['control']['best_state_file'] = str(wave_dir / f'worker_{slot}.unused-control-best.pth')
        wcfg['control']['tensorboard_dir'] = str(wave_dir / f'worker_{slot}.unused-control-tb')
        cfg_path = wave_dir / f'worker_{slot}.toml'
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, 'w', encoding='utf-8') as f:
            toml.dump(wcfg, f)
        workers.append({
            'worker': slot,
            'config': str(cfg_path),
            'pool_member_id': member['id'],
            'opponent_state_file': member['path'],
            'opponent_sha256': member['sha256'],
            'log_dir': str(log_dir),
            'games_per_session': args.games,
        })
        print(f'  worker_{slot} <- {member["id"]} (sha256 {member["sha256"][:12]}...) log_dir={log_dir}')

    assert_distinct([w['log_dir'] for w in workers], 'worker log_dirs')
    assert_distinct([w['opponent_state_file'] for w in workers], 'worker opponent snapshots')

    for w in workers:
        cfg_w = load_cfg(w['config'])
        baseline = pathlib.Path(cfg_w['baseline']['train']['state_file'])
        log_dir = pathlib.Path(cfg_w['train_play']['default']['log_dir'])
        if str(baseline) != w['opponent_state_file']:
            raise SystemExit(f'worker_{w["worker"]} baseline mismatch: {baseline}')
        if str(log_dir) != w['log_dir']:
            raise SystemExit(f'worker_{w["worker"]} log_dir mismatch: {log_dir}')
        if cfg_w['train_play']['default']['games'] != args.games:
            raise SystemExit(f'worker_{w["worker"]} games mismatch')
        for key in ('state_file', 'best_state_file', 'tensorboard_dir'):
            value = pathlib.Path(cfg_w['control'][key])
            if value.resolve() == LIVE[key].resolve() or not is_under(value, SELFPLAY):
                raise SystemExit(f'worker_{w["worker"]} control.{key} is not scratch: {value}')
        if sha256_file(baseline) != w['opponent_sha256']:
            raise SystemExit(f'worker_{w["worker"]} opponent sha256 drifted after validation: {baseline}')

    trainer_dir = wave_dir / 'trainer'
    tcfg = deepcopy(cfg)
    tcfg['control']['state_file'] = str(trainer_dir / 'state.pth')
    tcfg['control']['best_state_file'] = str(trainer_dir / 'best.pth')
    tcfg['control']['tensorboard_dir'] = str(trainer_dir / 'tb')
    tcfg['test_play']['log_dir'] = str(trainer_dir / 'test_play')
    for k in ('state_file', 'best_state_file', 'tensorboard_dir'):
        value = pathlib.Path(tcfg['control'][k])
        if value.resolve() == LIVE[k].resolve() or not is_under(value, SELFPLAY):
            raise SystemExit(f'REFUSING trainer {k}: {value} (must be a scratch path under {SELFPLAY})')
    if not is_under(tcfg['test_play']['log_dir'], SELFPLAY):
        raise SystemExit(f'REFUSING trainer test_play.log_dir: {tcfg["test_play"]["log_dir"]}')
    tcfg_path = wave_dir / 'trainer.toml'
    with open(tcfg_path, 'w', encoding='utf-8') as f:
        toml.dump(tcfg, f)

    scfg = deepcopy(cfg)
    scfg['online']['server']['buffer_dir'] = str(wave_dir / 'server/buffer')
    scfg['online']['server']['drain_dir'] = str(wave_dir / 'server/drain')
    scfg['control']['state_file'] = str(wave_dir / 'unused-server-control-state.pth')
    scfg['control']['best_state_file'] = str(wave_dir / 'unused-server-control-best.pth')
    scfg['control']['tensorboard_dir'] = str(wave_dir / 'unused-server-control-tb')
    for key in ('buffer_dir', 'drain_dir'):
        if not is_under(scfg['online']['server'][key], SELFPLAY):
            raise SystemExit(f'REFUSING server {key}: {scfg["online"]["server"][key]}')
    for key in ('state_file', 'best_state_file', 'tensorboard_dir'):
        value = pathlib.Path(scfg['control'][key])
        if value.resolve() == LIVE[key].resolve() or not is_under(value, SELFPLAY):
            raise SystemExit(f'REFUSING server control.{key}: {value}')
    scfg_path = wave_dir / 'server.toml'
    with open(scfg_path, 'w', encoding='utf-8') as f:
        toml.dump(scfg, f)

    worker_map = {
        'experiment': 'SELFPLAY',
        'todo': 25,
        'created_at': now_iso(),
        'cycle': args.cycle,
        'wave': args.wave,
        'pool_source': str(args.pool),
        'pool_source_sha256': sha256_file(args.pool),
        'k': args.k,
        'games_per_session': args.games,
        'trainee': {'seat_count': 1, 'engine_name': 'trainee', 'player_names': ['trainee']},
        'opponent_seats_per_game': 3,
        'logs_dir': str(logs_dir),
        'configs': {'server': str(scfg_path), 'trainer': str(tcfg_path)},
        'assign': picks,
        'pool': [
            {
                'id': m['id'], 'state_file': m['path'], 'bytes': m['bytes'],
                'sha256': m['sha256'], 'steps': m.get('steps'),
                'status': m['status'], 'weight': pool['weighting']['weights'][m['id']],
            }
            for m in active_members(pool)
        ],
        'workers': workers,
    }
    root_map = SELFPLAY / 'worker_map.json'
    archive = SELFPLAY / 'worker_map.todo24.json'
    if root_map.is_file() and not archive.is_file():
        shutil.copy2(root_map, archive)
        print(f'  archived previous worker_map.json -> {archive}')
    for target in (root_map, wave_dir / 'worker_map.json'):
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(worker_map, f, indent=2)
            f.write('\n')
        print(f'  wrote {target}')
    print('PREPARE OK')


# ---------------------------------------------------------------- launch

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


def start(name, script, cfg_path, logs_dir):
    logs_dir = pathlib.Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['MORTAL_CFG'] = str(pathlib.Path(cfg_path).resolve())
    env['PYTHONPATH'] = f'{LIBRIICHI}{os.pathsep}{env["PYTHONPATH"]}' if env.get('PYTHONPATH') else str(LIBRIICHI)
    log = logs_dir / f'{name}.log'
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
    pool_path = pathlib.Path(args.pool)
    by_id = validate_pool(load_pool(pool_path), pool_path)
    if args.k is not None:
        print(f'  NOTE: --k {args.k} limits the prepared workers; the pool assign is unchanged')

    map_path = SELFPLAY / 'worker_map.json'
    if not map_path.is_file():
        raise SystemExit('REFUSING: run `prepare` first (worker_map.json missing)')
    worker_map = json.loads(map_path.read_text())
    if worker_map.get('pool_source_sha256') != sha256_file(pool_path):
        raise SystemExit(
            f'REFUSING: {map_path} was prepared against a different pool.json '
            f'(prepared {worker_map.get("pool_source_sha256")}, current {sha256_file(pool_path)}); re-run prepare'
        )

    workers = worker_map['workers']
    if args.k is not None:
        workers = workers[:args.k]
    print(f'LAUNCH: cycle={worker_map.get("cycle")} wave={worker_map.get("wave")} K={len(workers)}')
    expected_games = worker_map['games_per_session']
    baselines = []
    log_dirs = []
    for w in workers:
        if w['pool_member_id'] not in by_id:
            raise SystemExit(f'REFUSING: worker_{w["worker"]} names non-active pool member {w["pool_member_id"]}')
        member = by_id[w['pool_member_id']]
        if member['path'] != w['opponent_state_file'] or member['sha256'] != w['opponent_sha256']:
            raise SystemExit(f'REFUSING: worker_{w["worker"]} opponent no longer matches pool.json')
        assert_fresh_log_dir(w['log_dir'], w['worker'])
        cfg_w = load_cfg(w['config'])
        baseline = pathlib.Path(cfg_w['baseline']['train']['state_file'])
        log_dir = pathlib.Path(cfg_w['train_play']['default']['log_dir'])
        if str(baseline) != w['opponent_state_file'] or str(log_dir) != w['log_dir']:
            raise SystemExit(f'REFUSING: worker_{w["worker"]} config drifted from worker_map.json')
        if cfg_w['train_play']['default']['games'] != expected_games:
            raise SystemExit(f'REFUSING: worker_{w["worker"]} games != {expected_games}')
        baselines.append(baseline)
        log_dirs.append(log_dir)
    assert_distinct(log_dirs, 'worker log_dirs')
    assert_distinct(baselines, 'worker opponent snapshots')

    remote = cfg['online']['remote']
    logs_dir = worker_map['logs_dir']
    pids = {
        'started_at': now_iso(),
        'run': {'cycle': worker_map.get('cycle'), 'wave': worker_map.get('wave')},
        'pool_source': str(pool_path),
        'configs': {
            'server': worker_map['configs']['server'],
            'trainer': worker_map['configs']['trainer'],
            'workers': [w['config'] for w in workers],
        },
    }
    pids['server'] = start('server', 'server.py', worker_map['configs']['server'], logs_dir)
    if not wait_port(remote['host'], remote['port'], 60):
        raise SystemExit(f'REFUSING: server did not listen on {remote["host"]}:{remote["port"]} within 60s')
    print(f'  server listening on {remote["host"]}:{remote["port"]}')

    pids['trainer'] = start('trainer', 'train.py', worker_map['configs']['trainer'], logs_dir)
    if not wait_log(pathlib.Path(logs_dir) / 'trainer.log', 'param has been submitted', 300):
        raise SystemExit('REFUSING: trainer did not submit its first param within 300s')
    print('  trainer submitted the first param')

    for w in workers:
        pids[f'worker_{w["worker"]}'] = start(f'worker_{w["worker"]}', 'client.py', w['config'], logs_dir)

    for target in (PIDS, pathlib.Path(logs_dir).parent / 'pids.json'):
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(pids, f, indent=2)
            f.write('\n')
        print(f'  wrote {target}')
    print('LAUNCH OK')


# ---------------------------------------------------------------- status / stop

def is_pid_role(role):
    """Only these keys in pids.json are PIDs. Never treat metadata (cycle=0,
    wave=0, ...) as a PID: killpg(0) would signal our own process group."""
    return role in ('server', 'trainer') or (role.startswith('worker_') and role[7:].isdigit())


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
    worker_map = json.loads((SELFPLAY / 'worker_map.json').read_text())
    logs_dir = pathlib.Path(worker_map['logs_dir'])
    status = {
        'run': pids.get('run', {}),
        'pids': {}, 'workers': [], 'server_dirs': {},
    }
    for role, pid in pids.items():
        if not isinstance(pid, int) or not is_pid_role(role):
            continue
        status['pids'][role] = {'pid': pid, 'alive': alive(pid), 'rss_mib': process_rss(pid)}
    for w in worker_map['workers']:
        log_dir = pathlib.Path(w['log_dir'])
        files = sorted(log_dir.glob('*.json.gz')) if log_dir.is_dir() else []
        log = logs_dir / f'worker_{w["worker"]}.log'
        status['workers'].append({
            'worker': w['worker'],
            'opponent': w['pool_member_id'],
            'opponent_sha256': w['opponent_sha256'][:12],
            'log_dir': w['log_dir'],
            'current_session_files': len(files),
            'sessions_submitted': log_count(log, 'logs have been submitted'),
            'rss_mib': process_rss(pids.get(f'worker_{w["worker"]}', -1)),
        })
    for key in ('buffer', 'drain'):
        d = pathlib.Path(worker_map['configs']['server']).parent / 'server' / key
        status['server_dirs'][key] = len(list(d.glob('*.json.gz'))) if d.is_dir() else 0
    gpu = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
        capture_output=True, text=True,
    ).stdout.strip()
    status['gpu_compute_apps'] = gpu or '(none)'
    print(json.dumps(status, indent=2))


def cmd_stop(args):
    pids = json.loads(PIDS.read_text())
    targets = [
        (role, pid) for role, pid in pids.items()
        if isinstance(pid, int) and pid > 1 and is_pid_role(role)
    ]
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
    pids['stopped_at'] = now_iso()
    with open(PIDS, 'w', encoding='utf-8') as f:
        json.dump(pids, f, indent=2)
        f.write('\n')
    if leftovers:
        raise SystemExit('STOP INCOMPLETE: processes remain')
    print('STOP OK')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    p_validate = sub.add_parser('validate')
    p_validate.add_argument('--pool', default=str(POOL_FILE))
    p_validate.set_defaults(func=cmd_validate)

    p_sample = sub.add_parser('sample')
    p_sample.add_argument('--cycle', type=int, required=True)
    p_sample.add_argument('--k', type=int, default=2)
    p_sample.add_argument('--pool', default=str(POOL_FILE))
    p_sample.add_argument('--write', action='store_true')
    p_sample.set_defaults(func=cmd_sample)

    p_prepare = sub.add_parser('prepare')
    p_prepare.add_argument('--cycle', type=int, required=True)
    p_prepare.add_argument('--wave', type=int, required=True)
    p_prepare.add_argument('--k', type=int, default=2)
    p_prepare.add_argument('--games', type=int, default=100)
    p_prepare.add_argument('--pool', default=str(POOL_FILE))
    p_prepare.set_defaults(func=cmd_prepare)

    p_launch = sub.add_parser('launch')
    p_launch.add_argument('--k', type=int, default=None)
    p_launch.add_argument('--pool', default=str(POOL_FILE))
    p_launch.set_defaults(func=cmd_launch)

    sub.add_parser('status').set_defaults(func=cmd_status)
    sub.add_parser('stop').set_defaults(func=cmd_stop)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
