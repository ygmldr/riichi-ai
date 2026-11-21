#!/usr/bin/env python3
"""A9 training launcher: monkey-patch `dataloader.collate_batch`, then run train.

Run via `scripts/exp_a9_run.sh` (which sets cwd, MORTAL_CFG and PYTHONPATH); the
vendor tree is left untouched. Safety is re-asserted here immediately before
`train.main()`: the event branch must be enabled and no control/log path may
point at the live checkpoints/runs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR_MORTAL = REPO / "vendor" / "mortal" / "mortal"
SCRIPTS = REPO / "scripts"

LIVE_FILES = (
    "/home/ygsldr/riichi-ai/checkpoints/mortal.pth",
    "/home/ygsldr/riichi-ai/checkpoints/best.pth",
)
LIVE_DIR_PREFIXES = (
    "/home/ygsldr/riichi-ai/runs/test_play",
    "/home/ygsldr/riichi-ai/runs/train_play",
)


def main() -> None:
    cfg_path = os.environ.get("MORTAL_CFG")
    assert cfg_path, "MORTAL_CFG must be set by scripts/exp_a9_run.sh"
    cfg_path = Path(cfg_path).resolve()
    assert cfg_path.is_file(), cfg_path
    os.environ["MORTAL_CFG"] = str(cfg_path)
    for p in (str(SCRIPTS), str(VENDOR_MORTAL)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import exp_a9_shuffle as a9

    print(f"[A9 launch] MORTAL_CFG={cfg_path}", flush=True)
    print(f"[A9 launch] perm fingerprint {a9.perms_fingerprint()}", flush=True)
    print("[A9 launch] perms " + json.dumps(a9.perms_summary(), sort_keys=True), flush=True)

    import dataloader

    patched, _original = a9.patch_dataloader()
    assert getattr(dataloader, "_a9_patched", False)
    assert dataloader.collate_batch is patched
    print("[A9 launch] dataloader.collate_batch patched (vendor tree untouched)", flush=True)

    from config import config

    assert config["model"]["use_event_branch"] is True, "A9 is an event arm"
    ctrl = config["control"]
    for key in ("state_file", "best_state_file", "tensorboard_dir"):
        value = str(ctrl[key])
        assert value not in LIVE_FILES, f"{key} is a LIVE path: {value}"
    test_play = str(config["test_play"]["log_dir"])
    train_play = str(config["train_play"]["default"]["log_dir"])
    for value in (test_play, train_play):
        assert not any(value.startswith(p) for p in LIVE_DIR_PREFIXES), value
    print(
        f"[A9 launch] safety state={ctrl['state_file']} best={ctrl['best_state_file']} "
        f"tb={ctrl['tensorboard_dir']}",
        flush=True,
    )
    print(f"[A9 launch] safety test_play={test_play} train_play={train_play}", flush=True)

    import train

    train.main()


if __name__ == "__main__":
    main()
