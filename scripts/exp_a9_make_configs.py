#!/usr/bin/env python3
"""Generate A9's train/eval/smoke configs WITHOUT touching any live path.

Bases:
  train/eval  <- runs/exp/E3a/{train,eval}.toml (todo 19, same declared conditions)
  smoke       <- A9 train.toml with /tmp/opencode/a9smoke/ paths, small batch

Every generated config is re-read and asserted: `[model].use_event_branch=true`,
and `state_file`/`best_state_file`/`tensorboard_dir`/`test_play`/`train_play`/`1v3`
log dirs never equal the live values. The eval config also asserts pool A
(`seed_start=10000`), 12000 games, the pinned champion and the A9 challenger.
"""

from __future__ import annotations

import argparse
import os
import tomllib

import toml

REPO = "/home/ygsldr/riichi-ai"
A9 = f"{REPO}/runs/exp/A9"
E3A = f"{REPO}/runs/exp/E3a"
SMOKE_ROOT = "/tmp/opencode/a9smoke"
LIVE = {
    "state_file": f"{REPO}/checkpoints/mortal.pth",
    "best_state_file": f"{REPO}/checkpoints/best.pth",
    "tensorboard_dir": f"{REPO}/runs",
    "test_play_log_dir": f"{REPO}/runs/test_play",
    "train_play_log_dir": f"{REPO}/runs/train_play",
}
PINNED = f"{REPO}/runs/exp/PINNED/mortal.pth.at-175200"
OLD = f"{REPO}/checkpoints/mortal.pth.at-148800"
SEED_KEY = 0x5DFAA4CEF265CD7
SEED_START = 10000


def replace_prefix(node, old: str, new: str):
    if isinstance(node, dict):
        return {k: replace_prefix(v, old, new) for k, v in node.items()}
    if isinstance(node, list):
        return [replace_prefix(v, old, new) for v in node]
    if isinstance(node, str):
        return node.replace(old, new)
    return node


def die(msg: str) -> None:
    raise SystemExit(f"SAFETY: {msg}")


def check_safety(cfg: dict, kind: str) -> None:
    ctrl = cfg["control"]
    if ctrl["state_file"] == LIVE["state_file"]:
        die("state_file still live")
    if ctrl["best_state_file"] == LIVE["best_state_file"]:
        die("best_state_file still live")
    if ctrl["tensorboard_dir"] == LIVE["tensorboard_dir"]:
        die("tensorboard_dir still live")
    if cfg["test_play"]["log_dir"] == LIVE["test_play_log_dir"]:
        die("test_play log_dir still live")
    if cfg["train_play"]["default"]["log_dir"] == LIVE["train_play_log_dir"]:
        die("train_play log_dir still live")
    if cfg["model"].get("use_event_branch") is not True:
        die(f"{kind}: use_event_branch must be true for A9")
    for key in ("state_file", "best_state_file", "tensorboard_dir"):
        value = str(ctrl[key])
        if kind == "smoke":
            if not value.startswith(SMOKE_ROOT):
                die(f"smoke {key} not under {SMOKE_ROOT}: {value}")
        elif not value.startswith(A9 + "/"):
            die(f"{kind} {key} not under {A9}: {value}")
    for key in ("log_dir",):
        value = str(cfg["test_play"][key])
        if kind == "smoke":
            if not value.startswith(SMOKE_ROOT):
                die(f"smoke test_play not under {SMOKE_ROOT}: {value}")
        elif not value.startswith(A9 + "/"):
            die(f"{kind} test_play not under {A9}: {value}")
    value = str(cfg["train_play"]["default"]["log_dir"])
    if kind == "smoke":
        if not value.startswith(SMOKE_ROOT):
            die(f"smoke train_play not under {SMOKE_ROOT}: {value}")
    elif not value.startswith(A9 + "/"):
        die(f"{kind} train_play not under {A9}: {value}")


def generate_train() -> dict:
    with open(f"{E3A}/train.toml", "rb") as f:
        cfg = tomllib.load(f)
    cfg = replace_prefix(cfg, E3A, A9)
    cfg["model"]["use_event_branch"] = True
    check_safety(cfg, "train")
    ctrl = cfg["control"]
    for key, expected in (
        ("batch_size", 1024),
        ("opt_step_every", 1),
        ("save_every", 2000),
        ("test_every", 100000),
        ("enable_compile", False),
        ("enable_amp", True),
    ):
        if ctrl[key] != expected:
            die(f"train.control.{key}={ctrl[key]!r} != E3a's {expected!r}")
    if cfg["dataset"]["num_workers"] != 3:
        die("train.dataset.num_workers != 3")
    if cfg["1v3"]["seed_start"] != SEED_START or cfg["1v3"]["seed_key"] != SEED_KEY:
        die("train 1v3 seed window changed")
    if cfg["1v3"]["log_dir"] != f"{A9}/games":
        die(f"train 1v3.log_dir={cfg['1v3']['log_dir']!r}")
    if cfg["optim"]["scheduler"] != {"peak": 0.0001, "final": 0.0001, "warm_up_steps": 0, "max_steps": 0}:
        die("scheduler differs from E3a")
    if cfg["resnet"] != {"conv_channels": 192, "num_blocks": 40}:
        die("resnet differs from E3a")
    if len(cfg["dataset"]["globs"]) != 14 or "2023" in "".join(cfg["dataset"]["globs"]):
        die("train dataset globs must be 2009-2022 with a 2023 holdout")
    if cfg["dataset"]["val_globs"] != [f"{REPO}/datasets/extracted/2023/**/*.mjson"]:
        die("val_globs must be the 2023 holdout")
    return cfg


def generate_eval() -> dict:
    with open(f"{E3A}/eval.toml", "rb") as f:
        cfg = tomllib.load(f)
    cfg = replace_prefix(cfg, E3A, A9)
    cfg["model"]["use_event_branch"] = True
    cfg["1v3"]["challenger"]["state_file"] = f"{A9}/state.pth"
    cfg["1v3"]["challenger"]["name"] = "a9"
    check_safety(cfg, "eval")
    if cfg["1v3"]["seed_start"] != SEED_START or cfg["1v3"]["seed_key"] != SEED_KEY:
        die("eval must use pool A (seed_start=10000, registered key)")
    if cfg["1v3"]["games_per_iter"] != 12000 or cfg["1v3"]["iters"] != 1:
        die("eval must be 12000 games x 1 iter")
    if cfg["1v3"]["log_dir"] != f"{A9}/games":
        die(f"eval 1v3.log_dir={cfg['1v3']['log_dir']!r}")
    if cfg["1v3"]["champion"]["state_file"] != OLD:
        die("eval champion must be the at-148800 pin")
    if cfg["1v3"]["challenger"]["state_file"] != f"{A9}/state.pth":
        die("eval challenger must be A9's own state.pth")
    if cfg["1v3"]["akochan"]["enabled"] is not False:
        die("akochan must stay disabled")
    return cfg


def generate_smoke() -> dict:
    with open(f"{A9}/train.toml", "rb") as f:
        cfg = tomllib.load(f)
    cfg = replace_prefix(cfg, A9, SMOKE_ROOT)
    cfg["control"]["batch_size"] = 256
    cfg["control"]["save_every"] = 50
    cfg["control"]["test_every"] = 100000
    cfg["dataset"]["num_workers"] = 2
    cfg["dataset"]["file_batch_size"] = 2
    cfg["dataset"]["file_index"] = f"{SMOKE_ROOT}/file_index.pth"
    cfg["dataset"]["val_globs"] = [f"{REPO}/datasets/extracted/2023/**/*.mjson"]
    cfg["model"]["use_event_branch"] = True
    check_safety(cfg, "smoke")
    return cfg


def dump(name: str, cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        toml.dump(cfg, f)
    with open(path, "rb") as f:
        reread = tomllib.load(f)
    check_safety(reread, name)
    print(f"OK {name} -> {path} (use_event_branch={reread['model']['use_event_branch']!r})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=A9)
    args = ap.parse_args()
    paths = {
        "train": f"{args.out_dir}/train.toml",
        "eval": f"{args.out_dir}/eval.toml",
        "smoke": f"{SMOKE_ROOT}/smoke.toml",
    }
    os.makedirs(SMOKE_ROOT, exist_ok=True)
    dump("train", generate_train(), paths["train"])
    dump("eval", generate_eval(), paths["eval"])
    dump("smoke", generate_smoke(), paths["smoke"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
