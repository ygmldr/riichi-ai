#!/usr/bin/env python3
"""[plan todo 26 / A14] minimal online self-play RL run + evaluation driver.

    python3 scripts/exp_a14_run.py prepare          # cycle-1 configs, post-process, seed state
    python3 scripts/exp_a14_run.py train            # needs /tmp/opencode/train.lock; detached in practice
    python3 scripts/exp_a14_run.py eval --arm trend  | confirm
    python3 scripts/exp_a14_run.py status

Pre-registration: `runs/exp/A14/PREREG.md` (must exist before `train`).
Pool: `runs/exp/SELFPLAY/pool.json` (cycle 1, sampled with `sample --cycle 1 --write`).
All run artefacts live under `runs/exp/A14/`; `checkpoints/mortal.pth` is never a target.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

import toml

REPO = pathlib.Path('/home/ygsldr/riichi-ai')
sys.path.insert(0, str(REPO / 'scripts'))
import selfplay_launch as sl

A14 = REPO / 'runs/exp/A14'
SF_ROOT = A14 / 'selfplay'
POOL = REPO / 'runs/exp/SELFPLAY/pool.json'
PREREG = A14 / 'PREREG.md'
CYCLE = 1
WAVES = (0, 1)
K = 2
GAMES = 2000
INIT_STATE = REPO / 'checkpoints/mortal.pth.at-148800'
CHAMPION = REPO / 'checkpoints/mortal.pth.at-148800'
TRAINER_STATE = SF_ROOT / 'cycle_1/wave_0/trainer/state.pth'
FINAL_STATE = A14 / 'state_final.pth'
SEED_KEY = 0x5DFAA4CEF265CD7
SEED_KEY_ARG = '0x5dfaa4cef265cd7'
EVAL_GAMES = 12000
EVALS = {
    'trend': A14 / 'eval_poolA',
    'confirm': A14 / 'eval_confirm',
}
SEED_START = {'trend': 10000, 'confirm': 70000}
TRAIN_CAP_S = 130 * 60
WAVE_CAP_S = 75 * 60
CATCHUP_CAP_S = 30 * 60
EVAL_CAP_S = 90 * 60
LIVE_MTIMES = {
    REPO / 'checkpoints/mortal.pth': '2026-09-17 02:03:09',
    REPO / 'checkpoints/best.pth': '2026-09-16 02:59:45',
}
FROZEN_CORPUS = REPO / 'runs/1v3'


def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def wave_dir(wave):
    return SF_ROOT / f'cycle_{CYCLE}' / f'wave_{wave}'


def worker_log(wave, slot):
    return wave_dir(wave) / 'logs' / f'worker_{slot}.log'


def trainer_log():
    return wave_dir(0) / 'logs' / 'trainer.log'


def check_invariants(tag):
    frozen = len(list(FROZEN_CORPUS.glob('*.json.gz')))
    ok = frozen == 2004
    log(f'INVARIANT[{tag}] runs/1v3/*.json.gz = {frozen} (want 2004) {"OK" if ok else "FAIL"}')
    for path, want in LIVE_MTIMES.items():
        got = datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        same = got == want
        ok = ok and same
        log(f'INVARIANT[{tag}] {path.name} mtime {got} (want {want}) {"OK" if same else "FAIL"}')
    if not ok:
        raise SystemExit(f'INVARIANT FAILURE at {tag}')
    return frozen


def submitted_games(log_path):
    if not log_path.is_file():
        return 0
    text = log_path.read_text(errors='replace')
    starts = [(m.start(), int(m.group(1))) for m in re.finditer(r'start \d+ sets, (\d+) hanchans', text)]
    submits = [m.start() for m in re.finditer(r'logs have been submitted', text)]
    total = 0
    for s in submits:
        prev = [n for pos, n in starts if pos < s]
        if prev:
            total += prev[-1]
    return total


def trainer_stats():
    path = trainer_log()
    text = path.read_text(errors='replace') if path.is_file() else ''
    sizes = [int(x.replace(',', '')) for x in re.findall(r'file list size: ([\d,]+)', text)]
    last_drain_pos = text.rfind('file list size:')
    submit_positions = [m.start() for m in re.finditer(r'param has been submitted', text)]
    return {
        'drains': len(sizes),
        'drained_games': sum(sizes),
        'last_drain_pos': last_drain_pos,
        'epoch_end_after_last_drain': bool(submit_positions) and submit_positions[-1] > last_drain_pos,
        'steps_lines': [int(x.replace(',', '')) for x in re.findall(r'total steps: ([\d,]+)', text)],
    }


def is_zombie(pid):
    try:
        return pathlib.Path(f'/proc/{pid}/stat').read_text().split()[2] == 'Z'
    except (FileNotFoundError, IndexError):
        return False


def dead(pid):
    return (not sl.alive(pid)) or is_zombie(pid)


def spawn(name, script, cfg_path, logs_dir):
    logs_dir = pathlib.Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['MORTAL_CFG'] = str(pathlib.Path(cfg_path).resolve())
    env['PYTHONPATH'] = f'{sl.LIBRIICHI}:{env["PYTHONPATH"]}' if env.get('PYTHONPATH') else str(sl.LIBRIICHI)
    out = open(logs_dir / f'{name}.log', 'ab')
    return subprocess.Popen(
        [sys.executable, script], cwd=sl.MORTAL_DIR, env=env, stdin=subprocess.DEVNULL,
        stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
    )


def stop_pgid(pid, name=''):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    deadline = time.time() + 10
    while time.time() < deadline and not dead(pid):
        time.sleep(0.5)
    log(f'  killed {name or "pgid"} {pid} (dead={dead(pid)})')


def wait_worker_submissions(wave, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        n = [submitted_games(worker_log(wave, slot)) for slot in range(K)]
        if all(x > 0 for x in n):
            return True, n
        if any(dead(p) for p in worker_pids(wave)):
            log(f'  WARNING wave {wave}: a worker died before submitting (games so far {n})')
            return False, n
        time.sleep(3)
    return False, [submitted_games(worker_log(wave, slot)) for slot in range(K)]


def worker_pids(wave):
    return [int(wave_pids()[f'worker_{slot}']) for slot in range(K)]


def wave_pids():
    return json.loads((sf_pids()).read_text())


def sf_pids():
    return SF_ROOT / 'pids.json'


def postprocess_trainer(path):
    cfg = toml.load(path)
    cfg['control']['enable_compile'] = False
    cfg['control']['save_every'] = 100
    cfg['rl']['use_pg'] = True
    cfg['rl'].setdefault('lam_v', 1.0)
    cfg['rl'].setdefault('lam_kl', 0.05)
    cfg['model']['use_event_branch'] = False
    if pathlib.Path(cfg['rl']['anchor_state_file']).resolve() != INIT_STATE.resolve():
        raise SystemExit(f'REFUSING: trainer anchor != {INIT_STATE}: {cfg["rl"]["anchor_state_file"]}')
    for key in ('state_file', 'best_state_file', 'tensorboard_dir'):
        if not sl.is_under(cfg['control'][key], A14):
            raise SystemExit(f'REFUSING trainer {key} outside A14: {cfg["control"][key]}')
    with open(path, 'w', encoding='utf-8') as f:
        toml.dump(cfg, f)
    return cfg


def postprocess_worker(path):
    cfg = toml.load(path)
    cfg['model']['use_event_branch'] = False
    if cfg['train_play']['default']['games'] != GAMES:
        raise SystemExit(f'REFUSING: worker games != {GAMES}: {path}')
    if not sl.is_under(cfg['train_play']['default']['log_dir'], A14):
        raise SystemExit(f'REFUSING worker log_dir outside A14: {cfg["train_play"]["default"]["log_dir"]}')
    with open(path, 'w', encoding='utf-8') as f:
        toml.dump(cfg, f)


def cmd_prepare(args):
    if not PREREG.is_file():
        raise SystemExit(f'REFUSING: PREREG missing: {PREREG}')
    log(f'PREPARE cycle {CYCLE} waves {WAVES} K={K} games/session={GAMES} root={SF_ROOT}')
    sl.set_root(str(SF_ROOT))
    for wave in WAVES:
        ns = argparse.Namespace(cycle=CYCLE, wave=wave, k=K, games=GAMES, pool=str(POOL), root=str(SF_ROOT))
        sl.cmd_prepare(ns)
        postprocess_trainer(wave_dir(wave) / 'trainer.toml')
        for spec in json.loads((SF_ROOT / 'worker_map.json').read_text())['workers']:
            postprocess_worker(pathlib.Path(spec['config']))
        log(f'  wave {wave}: configs post-processed (use_pg=true, CNN, save_every=100, compile off)')
    stale_archive = SF_ROOT / 'worker_map.todo24.json'
    if stale_archive.is_file():
        stale = json.loads(stale_archive.read_text())
        if (stale.get('cycle'), stale.get('wave')) == (CYCLE, 0):
            stale_archive.unlink()
            log('  removed the wave-0 map that prepare archived under the misleading todo24 name')
    TRAINER_STATE.parent.mkdir(parents=True, exist_ok=True)
    src_sha, dst_sha = sha256_file(INIT_STATE), None
    if TRAINER_STATE.exists():
        dst_sha = sha256_file(TRAINER_STATE)
        if dst_sha != src_sha:
            raise SystemExit(f'REFUSING: existing {TRAINER_STATE} is not the init copy; move it away first')
    else:
        shutil.copy2(INIT_STATE, TRAINER_STATE)
        dst_sha = sha256_file(TRAINER_STATE)
    if dst_sha != src_sha:
        raise SystemExit('REFUSING: init-state copy sha256 mismatch')
    summary = {
        'prepared_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'cycle': CYCLE, 'k': K, 'games_per_session': GAMES, 'waves': list(WAVES),
        'init_state': str(INIT_STATE), 'init_state_sha256': src_sha,
        'trainer_state': str(TRAINER_STATE), 'trainer_state_sha256': dst_sha,
        'prereg_sha256': sha256_file(PREREG), 'pool_sha256': sha256_file(POOL),
    }
    (A14 / 'prepare_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    check_invariants('prepare')
    log('PREPARE OK')


def launch_all():
    ns = argparse.Namespace(k=None, pool=str(POOL), root=str(SF_ROOT))
    sl.cmd_launch(ns)
    return json.loads(sf_pids().read_text())


def wait_worker_session(wave):
    while True:
        budget = TRAIN_CAP_S - (time.time() - TRAIN_T0)
        if budget <= 0:
            log(f'  TRAIN CAP reached while waiting for wave {wave}')
            return False
        ok, counts = wait_worker_submissions(wave, min(WAVE_CAP_S, budget))
        if ok:
            return True
        log(f'  wave {wave} did not finish its sessions within the cap: {counts}')
        return False


TRAIN_T0 = 0.0


def wait_catchup(total_submitted):
    deadline = time.time() + CATCHUP_CAP_S
    while time.time() < deadline:
        stats = trainer_stats()
        buffer_n = len(list((wave_dir(0) / 'server' / 'buffer').glob('*.json.gz')))
        if (stats['drained_games'] >= total_submitted and buffer_n == 0
                and stats['epoch_end_after_last_drain']):
            log(f'  catch-up: drained={stats["drained_games"]} submitted={total_submitted} buffer=0; dwell 60s')
            time.sleep(60)
            stats2 = trainer_stats()
            if stats2['drained_games'] >= total_submitted and not list(
                    (wave_dir(0) / 'server' / 'buffer').glob('*.json.gz')):
                return stats2
        else:
            log(f'  catch-up wait: drained={stats["drained_games"]}/{total_submitted} '
                f'buffer={buffer_n} idle_marker={stats["epoch_end_after_last_drain"]}')
        time.sleep(10)
    log('  catch-up cap reached')
    return trainer_stats()


def cmd_train(args):
    global TRAIN_T0
    sl.set_root(str(SF_ROOT))
    sl.require_lock()
    check_invariants('train-start')
    if not TRAINER_STATE.is_file():
        raise SystemExit(f'REFUSING: run prepare first ({TRAINER_STATE} missing)')
    w0_map = json.loads((wave_dir(0) / 'worker_map.json').read_text())
    if (w0_map['cycle'], w0_map['wave']) != (CYCLE, 0):
        raise SystemExit(f'REFUSING: wave-0 map is {w0_map["cycle"]}/{w0_map["wave"]}')
    if w0_map['configs']['trainer'] != str(wave_dir(0) / 'trainer.toml'):
        raise SystemExit('REFUSING: wave-0 trainer config mismatch')
    w0_map['pool_source_sha256'] = sha256_file(POOL)
    sf_root_map = SF_ROOT / 'worker_map.json'
    sf_root_map.write_text(json.dumps(w0_map, indent=2) + '\n')
    pids = launch_all()
    TRAIN_T0 = time.time()
    summary = {'started_at': datetime.now().astimezone().isoformat(timespec='seconds'),
               'wave_pids': pids, 'sessions': {}, 'warnings': []}
    for wave in WAVES:
        if wave == 1:
            if time.time() - TRAIN_T0 >= TRAIN_CAP_S:
                summary['warnings'].append('train cap reached before wave 1; generation incomplete')
                break
            wmap = json.loads((wave_dir(1) / 'worker_map.json').read_text())
            if (wmap['cycle'], wmap['wave']) != (CYCLE, 1):
                raise SystemExit(f'REFUSING: wave-1 map is {wmap["cycle"]}/{wmap["wave"]}')
            new_pids = {}
            for spec in wmap['workers']:
                sl.assert_fresh_log_dir(spec['log_dir'], spec['worker'])
                name = f'worker_{spec["worker"]}'
                new_pids[name] = sl.start(name, 'client.py', spec['config'], wmap['logs_dir'])
            pids.update(new_pids)
            sf_pids().write_text(json.dumps(pids, indent=2) + '\n')
            sf_root_map.write_text(json.dumps(wmap, indent=2) + '\n')
            log(f'  wave 1 workers started: {new_pids} (opponents {[s["pool_member_id"] for s in wmap["workers"]]})')
        log(f'wave {wave}: waiting for both worker sessions '
            f'(members {json.loads((wave_dir(wave) / "worker_map.json").read_text())["assign"]})')
        cap_hit = not wait_worker_session(wave)
        games = [submitted_games(worker_log(wave, slot)) for slot in range(K)]
        log(f'  wave {wave} submitted games per worker: {games} (cap_hit={cap_hit})')
        for slot in range(K):
            stop_pgid(int(pids[f'worker_{slot}']), f'wave{wave}_worker_{slot}')
        summary['sessions'][f'wave_{wave}'] = {
            'games': games, 'cap_hit': cap_hit,
            'local_logs': [len(list((wave_dir(wave) / f'worker_{slot}').glob('*.json.gz'))) for slot in range(K)],
            'ended_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        }
    total = sum(sum(v['games']) for v in summary['sessions'].values())
    log(f'total submitted games: {total} (target {4 * GAMES})')
    stats = wait_catchup(total)
    sl.cmd_stop(argparse.Namespace(root=str(SF_ROOT)))
    if not TRAINER_STATE.is_file():
        raise SystemExit('REFUSING: trainer never saved a state file')
    shutil.copy2(TRAINER_STATE, FINAL_STATE)
    summary['final_state'] = {
        'path': str(FINAL_STATE), 'sha256': sha256_file(FINAL_STATE),
        'source': str(TRAINER_STATE), 'source_sha256': sha256_file(TRAINER_STATE),
    }
    summary['trainer'] = {
        'drained_games': stats['drained_games'], 'drains': stats['drains'],
        'steps_seen': stats['steps_lines'][-3:],
    }
    summary['finished_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
    summary['elapsed_s'] = round(time.time() - TRAIN_T0, 1)
    (A14 / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    shutil.copy2(trainer_log(), A14 / 'train.log')
    log(f'TRAIN DONE: {json.dumps(summary["sessions"], ensure_ascii=False)}')
    log(f'final state: {summary["final_state"]}')
    check_invariants('train-end')


def build_eval_cfg(arm, dst):
    cfg = toml.load(wave_dir(0) / 'trainer.toml')
    cfg['control'].update({
        'online': False,
        'enable_compile': False,
        'state_file': str(EVALS[arm] / 'unused-control-state.pth'),
        'best_state_file': str(EVALS[arm] / 'unused-control-best.pth'),
        'tensorboard_dir': str(EVALS[arm] / 'unused-control-tb'),
    })
    cfg['test_play']['log_dir'] = str(EVALS[arm] / 'unused-test-play')
    cfg['train_play']['default']['log_dir'] = str(EVALS[arm] / 'unused-train-play')
    cfg['1v3'].update({
        'seed_key': SEED_KEY,
        'seed_start': SEED_START[arm],
        'games_per_iter': EVAL_GAMES,
        'iters': 1,
        'log_dir': str(EVALS[arm] / 'games'),
    })
    cfg['1v3']['challenger'].update({
        'state_file': str(FINAL_STATE), 'name': 'a14', 'enable_rule_based_agari_guard': True,
    })
    cfg['1v3']['champion'].update({'state_file': str(CHAMPION), 'enable_rule_based_agari_guard': True})
    with open(dst, 'w', encoding='utf-8') as f:
        toml.dump(cfg, f)
    return cfg


def cmd_eval(args):
    arm = args.arm
    sl.set_root(str(SF_ROOT))
    sl.require_lock()
    if not FINAL_STATE.is_file():
        raise SystemExit(f'REFUSING: final state missing: {FINAL_STATE}')
    check_invariants(f'eval-{arm}-start')
    EVALS[arm].mkdir(parents=True, exist_ok=True)
    games_dir = EVALS[arm] / 'games'
    if games_dir.exists() and any(games_dir.iterdir()):
        raise SystemExit(f'REFUSING: games dir not fresh: {games_dir}')
    cfg_path = A14 / f'eval_{arm}.toml'
    build_eval_cfg(arm, cfg_path)
    log(f'EVAL {arm}: seed_start={SEED_START[arm]} games={EVAL_GAMES} cfg={cfg_path}')
    proc = spawn(f'eval_{arm}', 'one_vs_three.py', cfg_path, EVALS[arm] / 'logs')
    deadline = time.time() + EVAL_CAP_S
    while time.time() < deadline and proc.poll() is None:
        time.sleep(15)
    rc = proc.poll()
    if rc is None:
        stop_pgid(proc.pid, f'eval_{arm}')
        raise SystemExit(f'REFUSING: eval {arm} hit the {EVAL_CAP_S}s cap')
    if rc != 0:
        tail = (EVALS[arm] / 'logs' / f'eval_{arm}.log').read_text(errors='replace')[-2000:]
        raise SystemExit(f'REFUSING: eval {arm} exited rc={rc}; log tail:\n{tail}')
    log(f'  arena pid {proc.pid} finished rc=0')
    check_invariants(f'eval-{arm}-end')
    report = EVALS[arm] / ('report.csv' if arm == 'confirm' else 'report_poolA.csv')
    subprocess.run([
        sys.executable, str(REPO / 'scripts/eval_report.py'),
        '--games-dir', str(games_dir), '--seed-key', SEED_KEY_ARG,
        '--seed-range', f'{SEED_START[arm]}-{SEED_START[arm] + 3000}',
        '--checkpoint', 'a14_final', '--source-dir', 'runs/exp/A14',
        '--out', str(report),
    ], check=True, cwd=REPO)
    delta_out = A14 / f'delta_{arm}.json'
    subprocess.run([
        sys.executable, str(REPO / 'scripts/exp_a14_delta.py'),
        '--games-dir', str(games_dir), '--seed-start', str(SEED_START[arm]),
        '--out', str(delta_out),
    ], check=True, cwd=REPO)
    import csv
    with open(report, encoding='utf-8') as f:
        row = next(csv.DictReader(f))
    if (int(row['sets']), int(row['games']), int(row['incomplete_sets'])) != (3000, 12000, 0):
        raise SystemExit(f'REFUSING: {arm} report is not 3000 sets / 12000 games / 0 incomplete: {row}')
    log(f'EVAL {arm} DONE: report={report} delta={delta_out} '
        f'({row["rank_dist"]} rank {row["avg_rank"]} [{row["avg_rank_ci_lo"]},{row["avg_rank_ci_hi"]}] '
        f'pt {row["pt"]} [{row["pt_ci_lo"]},{row["pt_ci_hi"]}])')


def cmd_status(args):
    info = {'now': datetime.now().astimezone().isoformat(timespec='seconds')}
    for name in ('server', 'trainer', 'worker_0', 'worker_1'):
        try:
            pid = int(json.loads(sf_pids().read_text())[name])
            info[name] = {'pid': pid, 'alive': sl.alive(pid)}
        except (FileNotFoundError, KeyError):
            info[name] = None
    info['trainer_stats'] = trainer_stats()
    info['submitted'] = {f'wave{wave}': [submitted_games(worker_log(wave, slot)) for slot in range(K)]
                         for wave in WAVES}
    info['gpu'] = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
        capture_output=True, text=True).stdout.strip()
    print(json.dumps(info, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare').set_defaults(func=cmd_prepare)
    sub.add_parser('train').set_defaults(func=cmd_train)
    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--arm', choices=sorted(EVALS), required=True)
    p_eval.set_defaults(func=cmd_eval)
    sub.add_parser('status').set_defaults(func=cmd_status)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
