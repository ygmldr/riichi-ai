#!/usr/bin/env python3
"""A9 budget watchdog: stop training once the declared 20,000 steps are saved.

The vendor training loop runs to the end of one corpus epoch (millions of steps);
the declared budget lives in `runs/exp/A9/PREREG.md`. This watchdog tails the
training log, and once the save block for the target step has been confirmed in
`state.pth` (`steps == target`) it SIGTERMs the training PID (SIGKILL after a
grace period). It exits silently if the training process dies first.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time
from pathlib import Path

CUM_STEP_RE = re.compile(r"total steps: ([0-9,]+)")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def last_logged_step(log_path: Path) -> int:
    if not log_path.is_file():
        return -1
    step = -1
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = CUM_STEP_RE.search(line)
            if m:
                step = int(m.group(1).replace(",", ""))
    return step


def state_steps(state_path: Path):
    import torch

    if not state_path.is_file():
        return None
    try:
        state = torch.load(state_path, weights_only=True, map_location="cpu")
    except Exception:
        return None
    return int(state.get("steps", -1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--kill-grace", type=float, default=90.0)
    args = ap.parse_args()

    log_path = Path(args.log).resolve()
    state_path = Path(args.state).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def note(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [A9 watchdog pid={os.getpid()}] {msg}"
        print(line, flush=True)
        with open(out_path, "a") as f:
            f.write(line + "\n")

    note(f"armed: pid={args.pid} target={args.steps} log={log_path} state={state_path}")
    deadline = time.time() + 24 * 3600
    while time.time() < deadline:
        if not pid_alive(args.pid):
            note(f"training pid {args.pid} exited before target; watchdog done")
            return 0
        logged = last_logged_step(log_path)
        if logged >= args.steps:
            saved = state_steps(state_path)
            note(f"log reached {logged}; state.pth steps={saved}")
            if saved is not None and saved >= args.steps:
                note(f"target {args.steps} saved; sending SIGTERM to {args.pid}")
                try:
                    os.kill(args.pid, signal.SIGTERM)
                except ProcessLookupError:
                    note("pid already gone")
                    return 0
                kill_deadline = time.time() + args.kill_grace
                while time.time() < kill_deadline and pid_alive(args.pid):
                    time.sleep(2.0)
                if pid_alive(args.pid):
                    note("still alive after grace; SIGKILL")
                    try:
                        os.kill(args.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                time.sleep(3.0)
                note(f"done: pid alive={pid_alive(args.pid)}")
                return 0
        time.sleep(args.poll)
    note("watchdog timed out after 24h")
    return 2


if __name__ == "__main__":
    sys.exit(main())
