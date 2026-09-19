#!/usr/bin/env python3
"""todo 22 pool-B eval configs: per-arm temp config under runs/exp/ARCH_B/<arm>/.

从各臂自己的 `runs/exp/<arm>/eval.toml` 深拷贝,仅做:
  1. 路径重定向: runs/exp/<arm>/ → runs/exp/ARCH_B/<arm>/(control state/best/tb、
     test_play、train_play、1v3.log_dir);
  2. `[1v3].challenger.state_file` 指回该臂真实 state.pth(只读输入,本脚本不改它);
  3. `[1v3].seed_start = 50000`(池 B);
其余字段(seed_key/games_per_iter/iters/champion/model/...)与源逐位一致。

安全断言(任一失败即非 0 退出):
  - `[control].state_file/best_state_file/tensorboard_dir`、`[test_play].log_dir`、
    `[train_play.default].log_dir` 全部在 `runs/exp/ARCH_B/<arm>/` 下,且**均非 live 路径**;
  - champion ≡ `checkpoints/mortal.pth.at-148800`;challenger.state_file 存在且非 live;
  - seed_start=50000、seed_key=0x5dfaa4cef265cd7、12000 局 × 1 iter。

用法: python3 scripts/exp_arch_b_make_configs.py [--arm E3a E2]
"""

from __future__ import annotations

import argparse
import os
import tomllib

import toml

REPO = "/home/ygsldr/riichi-ai"
ARCH = f"{REPO}/runs/exp/ARCH_B"
ARM_ROOTS = {"E3a": f"{REPO}/runs/exp/E3a", "E2": f"{REPO}/runs/exp/E2"}
LIVE = {
    "state_file": f"{REPO}/checkpoints/mortal.pth",
    "best_state_file": f"{REPO}/checkpoints/best.pth",
    "tensorboard_dir": f"{REPO}/runs",
    "test_play_log_dir": f"{REPO}/runs/test_play",
    "train_play_log_dir": f"{REPO}/runs/train_play",
}
CHAMPION = f"{REPO}/checkpoints/mortal.pth.at-148800"
SEED_KEY = 0x5DFAA4CEF265CD7
SEED_START = 50000
EXPECT = {
    "E3a": {"use_event_branch": True, "conv_channels": 192, "name": "mortal"},
    "E2": {"use_event_branch": False, "conv_channels": 202, "name": "e2"},
}


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


def check(arm: str, cfg: dict) -> None:
    root = f"{ARCH}/{arm}"
    ctrl = cfg["control"]
    # 1) four keys must live under ARCH_B/<arm>/ and must not be live paths
    for key in ("state_file", "best_state_file", "tensorboard_dir"):
        value = str(ctrl[key])
        if not value.startswith(root + "/"):
            die(f"{arm} control.{key} not under {root}: {value}")
        if value == LIVE[key]:
            die(f"{arm} control.{key} still live: {value}")
    if not str(cfg["test_play"]["log_dir"]).startswith(root + "/"):
        die(f"{arm} test_play.log_dir not under {root}: {cfg['test_play']['log_dir']}")
    if str(cfg["test_play"]["log_dir"]) == LIVE["test_play_log_dir"]:
        die(f"{arm} test_play.log_dir still live")
    if not str(cfg["train_play"]["default"]["log_dir"]).startswith(root + "/"):
        die(f"{arm} train_play.default.log_dir not under {root}")
    if str(cfg["train_play"]["default"]["log_dir"]) == LIVE["train_play_log_dir"]:
        die(f"{arm} train_play.default.log_dir still live")
    # 2) pool B protocol
    v13 = cfg["1v3"]
    if v13["seed_start"] != SEED_START:
        die(f"{arm} seed_start={v13['seed_start']} != {SEED_START}")
    if v13["seed_key"] != SEED_KEY:
        die(f"{arm} seed_key changed")
    if v13["games_per_iter"] != 12000 or v13["iters"] != 1:
        die(f"{arm} expected 12000 games x 1 iter")
    if v13["log_dir"] != f"{root}/games":
        die(f"{arm} 1v3.log_dir={v13['log_dir']!r}")
    if v13["champion"]["state_file"] != CHAMPION:
        die(f"{arm} champion must be at-148800 pin")
    if v13["champion"]["name"] != "old148800":
        die(f"{arm} champion name changed")
    chal = str(v13["challenger"]["state_file"])
    if chal != f"{ARM_ROOTS[arm]}/state.pth":
        die(f"{arm} challenger.state_file={chal!r}")
    if chal in (LIVE["state_file"], f"{ARCH}/{arm}/state.pth"):
        die(f"{arm} challenger.state_file is live or redirected: {chal}")
    if not os.path.isfile(chal):
        die(f"{arm} challenger state does not exist: {chal}")
    if v13["challenger"]["name"] != EXPECT[arm]["name"]:
        die(f"{arm} challenger name={v13['challenger']['name']!r}")
    if v13["akochan"]["enabled"] is not False:
        die(f"{arm} akochan must stay disabled")
    # 3) model identity unchanged vs the arm's own eval.toml
    if cfg["model"].get("use_event_branch") is not EXPECT[arm]["use_event_branch"]:
        die(f"{arm} use_event_branch changed")
    if cfg["resnet"]["conv_channels"] != EXPECT[arm]["conv_channels"]:
        die(f"{arm} resnet.conv_channels changed")


def gen(arm: str) -> dict:
    src = f"{ARM_ROOTS[arm]}/eval.toml"
    with open(src, "rb") as f:
        cfg = tomllib.load(f)
    cfg = replace_prefix(cfg, ARM_ROOTS[arm], f"{ARCH}/{arm}")
    cfg["1v3"]["challenger"]["state_file"] = f"{ARM_ROOTS[arm]}/state.pth"
    cfg["1v3"]["seed_start"] = SEED_START
    check(arm, cfg)
    return cfg


def dump(arm: str, cfg: dict) -> str:
    out = f"{ARCH}/{arm}/eval.toml"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        toml.dump(cfg, f)
    with open(out, "rb") as f:
        reread = tomllib.load(f)
    check(arm, reread)
    print(
        f"OK {arm} -> {out} (seed_start={reread['1v3']['seed_start']}, "
        f"log_dir={reread['1v3']['log_dir']}, challenger={reread['1v3']['challenger']['state_file']}, "
        f"use_event_branch={reread['model'].get('use_event_branch')})"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", default=["E3a", "E2"])
    args = ap.parse_args()
    for arm in args.arm:
        if arm not in ARM_ROOTS:
            die(f"unknown arm {arm}")
    outs = [dump(arm, gen(arm)) for arm in args.arm]
    print(f"LIVE-PATH ASSERTION: none of {outs} points at {LIVE}")
    # distinct log dirs across arms
    dirs = [gen(arm)["1v3"]["log_dir"] for arm in args.arm]
    if len(set(dirs)) != len(dirs):
        die(f"log dirs not distinct: {dirs}")
    print(f"DISTINCT LOG DIRS: OK {dirs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
