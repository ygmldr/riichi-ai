#!/usr/bin/env python3
"""Step-budget watchdog: stop a training process after its declared step budget.

`vendor/mortal/mortal/train.py` loops `while True: train_epoch()` and never
stops at a step count on its own (the arms are budgeted by this watchdog, as
E2/E3a were). Usage:

    python3 scripts/exp_a7a8_watchdog.py --log runs/exp/A7/train.log \
        --state runs/exp/A7/state.pth --steps 20000 --pid <train-pid> \
        --out runs/exp/A7/watchdog.json

Polls `--log` for the `total steps: 20,000` save marker, waits a grace period
for the checkpoint write to finish, terminates the process group (setsid
launcher), then records the checkpoint's `steps` + mtime in `--out`.
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import time


def ts():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def find_children(pid):
    out = subprocess.run(['ps', '-eo', 'pid=,ppid=,args='], capture_output=True, text=True).stdout
    kids = {pid}
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        p, pp = int(parts[0]), int(parts[1])
        if pp in kids:
            kids.add(p)
    return sorted(kids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True)
    ap.add_argument('--state', required=True)
    ap.add_argument('--steps', type=int, required=True)
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--grace', type=int, default=45)
    ap.add_argument('--poll', type=int, default=30)
    args = ap.parse_args()

    marker = f'total steps: {args.steps:,}'
    result = {
        'pid': args.pid, 'target_steps': args.steps, 'marker': marker,
        'started_watch_at': ts(), 'marker_seen_at': None,
        'state_file': args.state, 'state_steps': None, 'state_mtime': None,
        'killed_at': None, 'killed_pids': [], 'exit': None,
    }
    print(f'WATCHDOG start pid={args.pid} steps={args.steps} log={args.log}', flush=True)

    while True:
        alive = os.path.exists(f'/proc/{args.pid}')
        if os.path.exists(args.log):
            with open(args.log, 'r', errors='replace') as f:
                if marker in f.read():
                    result['marker_seen_at'] = ts()
                    break
        if not alive:
            result['exit'] = 'process died before marker'
            break
        time.sleep(args.poll)

    if result['marker_seen_at']:
        print(f'WATCHDOG marker seen at {result["marker_seen_at"]}, grace {args.grace}s', flush=True)
        time.sleep(args.grace)
        pids = find_children(args.pid)
        try:
            pgid = os.getpgid(args.pid)
        except ProcessLookupError:
            pgid = None
        if pgid == args.pid:
            os.killpg(pgid, signal.SIGTERM)
            result['killed_pids'] = [f'pgid:{pgid}']
        else:
            for p in reversed(pids):
                try:
                    os.kill(p, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            result['killed_pids'] = pids
        time.sleep(10)
        left = [p for p in find_children(args.pid) if os.path.exists(f'/proc/{p}')]
        if left:
            for p in reversed(left):
                try:
                    os.kill(p, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            result['killed_pids'] += [f'KILL:{p}' for p in left]
        result['killed_at'] = ts()
        try:
            import torch
            state = torch.load(args.state, weights_only=True, map_location='cpu')
            result['state_steps'] = int(state['steps'])
            result['state_mtime'] = datetime.datetime.fromtimestamp(
                os.path.getmtime(args.state)).astimezone().isoformat(timespec='seconds')
        except Exception as e:  # noqa: BLE001 - recorded, not raised
            result['state_read_error'] = repr(e)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f'WATCHDOG done: {json.dumps(result)}', flush=True)


if __name__ == '__main__':
    main()
