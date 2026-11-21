#!/usr/bin/env python3
"""A1 (plan todo 12): inference-time river-order interventions on the 1v3 arena.

Question: does the pinned challenger actually *use* river (discard-pile) ORDER, or
is the river window just a bag of tiles? Falsification arms (challenger only; the
champion is never wrapped, so the opponent is identical across arms):

  original      -- observations pass through untouched (paired reference)
  shuffle       -- before obs reach `brain`, every river WINDOW block has its item
                   SLOTS permuted with a FIXED registered seed; the per-block slot
                   multiset (None/empty slots are their own class) is preserved
  recency_zero  -- the 4 recency plane groups are zeroed (self [131],
                   shimocha [324,327), toimen [519,522), kamicha [714,717))

Layout source of truth: `runs/analysis/obs_layout.json` (todo 11). Offsets are
read from that file at import; they are NEVER re-derived here. `kawa_overview_*`
(order-insensitive tile-set summaries) are NOT touched.

In-script self-checks, asserted on EVERY patched observation:
  * each permuted block keeps its exact slot multiset (bit-packed item vectors;
    per-channel value multiset (np.sort along the slot axis) and per-channel
    nonzero counts;
  * a non-uniform block is asserted to DIFFER after the shuffle (proof the shuffle
    applied); a uniform block (all-empty window) is asserted UNCHANGED;
  * blast radius: the set of channels that differ from the original is contained
    in the target region (window blocks for `shuffle`, recency planes for
    `recency_zero`); recency planes are asserted all-zero after zeroing.

Usage:
  python3 scripts/exp_a1_shuffle.py --selftest          # no arena, no torch
  python3 scripts/exp_a1_shuffle.py --arm <arm> [--config PATH]

`--arm` requires `[1v3].log_dir` in the config to equal
`runs/exp/A1/games_<arm>/` (config edits stay outside this script, atomically).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VENDOR_MORTAL = REPO / "vendor" / "mortal" / "mortal"
LAYOUT_JSON = REPO / "runs" / "analysis" / "obs_layout.json"
A1_DIR = REPO / "runs" / "exp" / "A1"

ARMS = ("original", "shuffle", "recency_zero")

# Registered in runs/exp/A1/PREREG.md BEFORE any arena run. Do not change.
PERM_SEED = 0xA15EED  # 10575597
OBS_CHANNELS = 1012
TILE_KINDS = 34


def arm_games_dir(arm: str) -> Path:
    return A1_DIR / f"games_{arm}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_layout(path: Path) -> tuple[list[tuple[int, int, int, str]], list[tuple[int, int, str]]]:
    """Read the verified todo-11 layout: window blocks and recency planes."""
    layout = json.loads(path.read_text(encoding="utf-8"))
    assert layout["schema"] == "riichi-obs-layout/v1", layout.get("schema")
    assert layout["shape"] == {"channels": OBS_CHANNELS, "tile_kinds": TILE_KINDS}, layout["shape"]
    assert layout["totals"]["covered_channels"] == OBS_CHANNELS
    assert layout["totals"]["gaps"] == []

    top = {b["name"]: b for b in layout["blocks"]}
    assert len(top) == 64, len(top)

    windows: list[tuple[int, int, int, str]] = []
    for b in layout["river_blocks"]["window_blocks"]:
        ref = top[b["name"]]
        assert ref["group"] == "river" and ref["is_river"] is True, b["name"]
        # item slot = item_channels contiguous channels -> a (items, item_channels) matrix
        assert b["item_stride"] == b["item_channels"], b
        assert b["count"] == b["items"] * b["item_channels"], b
        assert ref["offset"] == b["offset"] and ref["count"] == b["count"], b
        assert b["offset"] + b["count"] <= OBS_CHANNELS
        windows.append((b["offset"], b["items"], b["item_channels"], b["name"]))

    recency: list[tuple[int, int, str]] = []
    for p in layout["river_blocks"]["recency_planes"]:
        ref = top[p["name"]]
        assert ref["group"] == "river" and ref["is_recency_plane"] is True, p["name"]
        assert ref["offset"] == p["offset"] and ref["count"] == p["count"], p
        assert p["count"] >= 1 and p["offset"] + p["count"] <= OBS_CHANNELS, p
        recency.append((p["offset"], p["count"], p["name"]))

    assert len(windows) == 8, windows
    assert len(recency) == 4, recency
    return windows, recency


WINDOWS, RECENCY = load_layout(LAYOUT_JSON)

PATH_COUNTS = {"binary": 0, "fallback": 0}


def check_slot_multiset(before: np.ndarray, after: np.ndarray, name: str) -> None:
    """Exact slot-multiset equality for (items, item_channels, 34) item matrices.

    An item SLOT is a whole (item_channels, 34) plane group; None/empty slots are
    all-zero groups and participate as their own class. Checks the full item
    multiset plus, per channel/tile element, the value multiset / nonzero counts.
    Binary data (all river-window planes are 0/1 flags) takes a bit-packed
    lexicographic fast path; anything else falls back to np.unique(axis=0).
    """
    b = before.reshape(before.shape[0], -1)
    a = after.reshape(after.shape[0], -1)
    is_binary = bool(b.min() >= 0.0 and b.max() <= 1.0 and a.min() >= 0.0 and a.max() <= 1.0)
    if is_binary:
        PATH_COUNTS["binary"] += 1
        pb = np.packbits(b != 0, axis=1)
        pa = np.packbits(a != 0, axis=1)
        vb = np.sort(pb.view(f"V{pb.shape[1]}").ravel())
        va = np.sort(pa.view(f"V{pa.shape[1]}").ravel())
        assert np.array_equal(vb, va), f"{name}: slot multiset changed by permutation"
        # binary values: the per-element multiset is fixed by the count of ones
        assert np.array_equal(before.sum(axis=0), after.sum(axis=0)), (
            f"{name}: per-element counts changed"
        )
    else:
        PATH_COUNTS["fallback"] += 1
        ub, cb = np.unique(b, axis=0, return_counts=True)
        ua, ca = np.unique(a, axis=0, return_counts=True)
        assert ub.shape == ua.shape and np.array_equal(ub, ua) and np.array_equal(cb, ca), (
            f"{name}: slot multiset changed by permutation"
        )
        assert np.array_equal(np.sort(before, axis=0), np.sort(after, axis=0)), (
            f"{name}: per-element value multiset changed"
        )
        assert np.array_equal((before != 0).sum(axis=0), (after != 0).sum(axis=0)), (
            f"{name}: per-element nonzero counts changed"
        )


def apply_slot_permutation(before: np.ndarray, perm: np.ndarray, name: str) -> np.ndarray:
    """Permute item slots of a (items, item_channels, 34) block; assert invariants.

    Forces a non-identity arrangement when the multiset is non-uniform, so callers
    can assert "the shuffle applied" without a rare no-op draw.
    """
    n = before.shape[0]
    perm = np.asarray(perm, dtype=np.int64).copy()
    assert perm.shape == (n,), (name, perm.shape)
    assert np.array_equal(np.sort(perm), np.arange(n)), f"{name}: perm is not a permutation"
    after = before[perm].copy()
    uniform = bool(np.all(before == before[0]))
    if not uniform and np.array_equal(after, before):
        j = int(np.flatnonzero(np.any(before != before[0], axis=(1, 2)))[0])
        after[[0, j]] = after[[j, 0]]
    check_slot_multiset(before, after, name)
    if uniform:
        # all slots are the same item class (e.g. all-empty river) -> must be a no-op
        assert np.array_equal(after, before), f"{name}: uniform block changed"
    else:
        assert not np.array_equal(after, before), f"{name}: shuffle did not apply"
    return after


class ArmEngine:
    """Delegating wrapper around MortalEngine; patches obs before they reach `brain`."""

    def __init__(self, inner, arm: str, perm_seed: int = PERM_SEED):
        assert arm in ARMS, arm
        self._inner = inner
        self._arm = arm
        self._perm_seed = int(perm_seed)
        self._call_idx = 0
        self._window_mask = np.zeros(OBS_CHANNELS, dtype=bool)
        for off, n, c, _ in WINDOWS:
            self._window_mask[off : off + n * c] = True
        self._recency_mask = np.zeros(OBS_CHANNELS, dtype=bool)
        for off, cnt, _ in RECENCY:
            self._recency_mask[off : off + cnt] = True
        self.stats = {
            "arm": arm,
            "perm_seed": self._perm_seed if arm == "shuffle" else None,
            "react_batch_calls": 0,
            "obs_patched": 0,
            "obs_changed": 0,
            "blocks_checked": 0,
            "blocks_changed": 0,
            "blocks_uniform": 0,
            "recency_planes_zeroed_obs": 0,
        }

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def _patch_shuffle(self, o, call_idx: int, obs_idx: int) -> tuple[np.ndarray, bool]:
        orig = np.asarray(o)
        assert orig.shape == (OBS_CHANNELS, TILE_KINDS), orig.shape
        arr = orig.copy()
        rng = np.random.default_rng([self._perm_seed, call_idx, obs_idx])
        n_changed = 0
        for b_idx, (off, n, c, name) in enumerate(WINDOWS):
            block = arr[off : off + n * c].reshape(n, c, TILE_KINDS)
            before = block.copy()
            perm = rng.permutation(n)
            after = apply_slot_permutation(before, perm, f"{name}@call{call_idx}.obs{obs_idx}")
            block[:] = after
            self.stats["blocks_checked"] += 1
            if np.array_equal(after, before):
                self.stats["blocks_uniform"] += 1
            else:
                self.stats["blocks_changed"] += 1
                n_changed += 1
        diff = np.flatnonzero(np.any(arr != orig, axis=1))
        assert np.all(self._window_mask[diff]), (
            f"shuffle touched non-window channels: {diff[~self._window_mask[diff]][:8]}"
        )
        # a non-uniform block must show up as a differing channel; an all-empty river
        # (all 8 windows uniform) legitimately yields an identical observation
        if n_changed > 0:
            assert diff.size > 0, "non-uniform river windows did not change any channel"
        return arr, n_changed > 0

    def _patch_recency_zero(self, o) -> tuple[np.ndarray, bool]:
        orig = np.asarray(o)
        assert orig.shape == (OBS_CHANNELS, TILE_KINDS), orig.shape
        arr = orig.copy()
        any_nonzero = False
        for off, cnt, name in RECENCY:
            seg = arr[off : off + cnt]
            any_nonzero = any_nonzero or bool(np.any(seg != 0))
            seg[:] = 0.0
            assert not np.any(arr[off : off + cnt]), f"{name}: recency plane not zeroed"
        diff = np.flatnonzero(np.any(arr != orig, axis=1))
        assert np.all(self._recency_mask[diff]), "recency_zero touched non-recency channels"
        if any_nonzero:
            self.stats["recency_planes_zeroed_obs"] += 1
        return arr, any_nonzero

    def react_batch(self, obs, masks, invisible_obs):
        if self._arm == "original":
            return self._inner.react_batch(obs, masks, invisible_obs)
        call_idx = self._call_idx
        self._call_idx += 1
        if self._arm == "shuffle":
            results = [self._patch_shuffle(o, call_idx, i) for i, o in enumerate(obs)]
        else:
            results = [self._patch_recency_zero(o) for o in obs]
        patched = [r[0] for r in results]
        self.stats["react_batch_calls"] += 1
        self.stats["obs_patched"] += len(obs)
        self.stats["obs_changed"] += sum(1 for _, changed in results if changed)
        return self._inner.react_batch(patched, masks, invisible_obs)


# --------------------------------------------------------------------------- #
# self-test                                                                   #
# --------------------------------------------------------------------------- #
class _DummyInner:
    engine_type = "mortal"
    name = "dummy"
    is_oracle = False
    version = 4
    enable_quick_eval = True
    enable_rule_based_agari_guard = True

    def __init__(self):
        self.seen = []

    def react_batch(self, obs, masks, invisible_obs):
        self.seen.append([np.asarray(o).copy() for o in obs])
        return [0] * len(obs), [[0.0]] * len(obs), [list(m) for m in masks], [True] * len(obs)


def _selftest_obs(rng: np.random.Generator, uniform_block: int | None = None, binary: bool = True) -> np.ndarray:
    obs = np.zeros((OBS_CHANNELS, TILE_KINDS), dtype=np.float32)
    hi = 2 if binary else 5
    for b_idx, (off, n, c, _) in enumerate(WINDOWS):
        if b_idx == uniform_block:
            continue  # all-zero uniform window
        items = rng.integers(0, hi, size=(n, c, TILE_KINDS)).astype(np.float32)
        obs[off : off + n * c] = items.reshape(n * c, TILE_KINDS)
    for off, cnt, _ in RECENCY:
        obs[off : off + cnt, rng.integers(0, TILE_KINDS, size=cnt)] = 1.0
    return obs


def selftest() -> int:
    rng = np.random.default_rng(12345)
    checks = 0

    # 1) explicit perms (binary fast path + non-binary fallback): identity on
    # non-uniform must be forced to change; uniform blocks must no-op
    for b_idx, (off, n, c, name) in enumerate(WINDOWS):
        for is_binary, perm in (
            (True, np.arange(n)),
            (True, np.arange(n)[::-1]),
            (True, rng.permutation(n)),
            (False, np.arange(n)),
            (False, rng.permutation(n)),
        ):
            hi = 2 if is_binary else 5
            before = rng.integers(0, hi, size=(n, c, TILE_KINDS)).astype(np.float32)
            apply_slot_permutation(before, perm, f"{name}-explicit")
            checks += 1
    uniform = np.zeros((18, 8, TILE_KINDS), dtype=np.float32)
    after = apply_slot_permutation(uniform, np.arange(18)[::-1], "uniform")
    assert np.array_equal(after, uniform)
    checks += 1
    # negative control: a deliberately corrupted multiset must be caught
    before = rng.integers(0, 2, size=(18, 8, TILE_KINDS)).astype(np.float32)
    bad = before[rng.permutation(18)].copy()
    bad[0] += 1.0
    caught = False
    try:
        check_slot_multiset(before, bad, "negative-control")
    except AssertionError:
        caught = True
    assert caught, "slot-multiset check did not detect a corrupted block"
    checks += 1

    # 2) engine wrapper: original passthrough, shuffle invariants + determinism
    masks = [np.zeros(46, dtype=bool) for _ in range(2)]
    inner = _DummyInner()
    eng = ArmEngine(inner, "shuffle")
    obs = [_selftest_obs(rng), _selftest_obs(rng, uniform_block=3)]
    eng.react_batch(obs, masks, None)
    inner2 = _DummyInner()
    eng2 = ArmEngine(inner2, "shuffle")
    eng2.react_batch(obs, masks, None)
    for a, b in zip(inner.seen[0], inner2.seen[0]):
        assert np.array_equal(a, b), "shuffle not deterministic for identical call sequence"
    checks += 1
    # different call -> different arrangement (at least one block)
    inner3 = _DummyInner()
    eng3 = ArmEngine(inner3, "shuffle")
    eng3.react_batch(obs, masks, None)  # call 0 == eng call 0
    eng3.react_batch(obs, masks, None)  # call 1
    assert not np.array_equal(inner3.seen[1][0], inner3.seen[0][0]), "call-index variation missing"
    checks += 1
    st = eng.stats
    assert st["blocks_checked"] == 16 and st["blocks_changed"] >= 15 and st["blocks_uniform"] == 1, st
    assert st["obs_changed"] == 2, st

    inner4 = _DummyInner()
    eng4 = ArmEngine(inner4, "original")
    eng4.react_batch(obs, masks, None)
    for a, b in zip(obs, inner4.seen[0]):
        assert np.array_equal(np.asarray(a), b), "original arm modified the observation"
    assert eng4.stats["obs_changed"] == 0
    checks += 1

    # 3) recency_zero
    inner5 = _DummyInner()
    eng5 = ArmEngine(inner5, "recency_zero")
    eng5.react_batch(obs, masks, None)
    for o in inner5.seen[0]:
        for off, cnt, name in RECENCY:
            assert not np.any(o[off : off + cnt]), f"{name}: not zeroed"
    assert eng5.stats["recency_planes_zeroed_obs"] == 2, eng5.stats
    checks += 1

    # 4) layout read-back sanity
    assert WINDOWS[0][:3] == (35, 6, 4)
    assert WINDOWS[1][:3] == (59, 18, 4)
    assert WINDOWS[7][:3] == (570, 18, 8)
    assert RECENCY == [(131, 1, "self_kawa_recency"),
                       (324, 3, "opp1_shimocha_kawa_recency"),
                       (519, 3, "opp2_toimen_kawa_recency"),
                       (714, 3, "opp3_kamicha_kawa_recency")], RECENCY
    checks += 1

    print(f"SELFTEST OK: {checks} checks, {st['blocks_checked']} blocks, "
          f"{st['blocks_changed']} changed / {st['blocks_uniform']} uniform (asserted)")
    return 0


# --------------------------------------------------------------------------- #
# arena run                                                                   #
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(f"[A1 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_engine(engine_cfg: dict, torch):
    """Mirror vendor/mortal/mortal/one_vs_three.py's engine construction."""
    from model import Brain, DQN
    from engine import MortalEngine

    state = torch.load(engine_cfg["state_file"], weights_only=True, map_location=torch.device("cpu"))
    scfg = state["config"]
    version = scfg["control"].get("version", 1)
    mortal = Brain(
        version=version,
        conv_channels=scfg["resnet"]["conv_channels"],
        num_blocks=scfg["resnet"]["num_blocks"],
    ).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    if engine_cfg["enable_compile"]:
        mortal.compile()
        dqn.compile()
    return MortalEngine(
        mortal,
        dqn,
        is_oracle=False,
        version=version,
        device=torch.device(engine_cfg["device"]),
        enable_amp=engine_cfg["enable_amp"],
        enable_rule_based_agari_guard=engine_cfg["enable_rule_based_agari_guard"],
        name=engine_cfg["name"],
    )


def run_arm(arm: str, config_path: Path, smoke_seeds: int = 0) -> int:
    expected_dir = arm_games_dir(arm)
    os.environ["MORTAL_CFG"] = str(config_path.resolve())
    sys.path.insert(0, str(VENDOR_MORTAL))
    os.chdir(VENDOR_MORTAL)

    import prelude  # noqa: F401  (logging/torch warmup, same as one_vs_three.py)
    import torch
    from config import config
    from libriichi.arena import OneVsThree

    cfg = config["1v3"]
    key = cfg.get("seed_key", -1)
    assert key != -1, "seed_key must be fixed in config for a paired comparison"
    games_per_iter = int(cfg["games_per_iter"])
    iters = int(cfg["iters"])
    seeds_per_iter = games_per_iter // 4
    seed_start = int(cfg.get("seed_start", 10000))
    if smoke_seeds:
        seeds_per_iter = smoke_seeds
        games_per_iter = seeds_per_iter * 4
        log_dir = Path("/tmp/opencode") / f"a1_smoke_{arm}"
        if log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True)
    else:
        assert games_per_iter == 12000 and iters == 1, (
            f"A1 PREREG fixes 12000 games x 1 iter, config says {games_per_iter}x{iters}"
        )
        log_dir = Path(cfg["log_dir"]).resolve()
        assert log_dir == expected_dir.resolve(), (
            f"config log_dir {log_dir} != {expected_dir} (edit [1v3].log_dir)"
        )
        if log_dir.exists():
            leftover = [p.name for p in log_dir.iterdir()]
            assert not leftover, f"{log_dir} not empty ({len(leftover)} entries); refusing to mix runs"
        else:
            log_dir.mkdir(parents=True)
    A1_DIR.mkdir(parents=True, exist_ok=True)

    prereg = A1_DIR / "PREREG.md"
    assert prereg.is_file(), "PREREG.md must exist before any arena run"
    meta = {
        "arm": arm,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed_key": key,
        "seed_start": seed_start,
        "seed_count": seeds_per_iter,
        "games": games_per_iter,
        "log_dir": str(log_dir),
        "config": str(config_path),
        "perm_seed": PERM_SEED if arm == "shuffle" else None,
        "prereg_sha256": sha256_file(prereg),
        "layout_sha256": sha256_file(LAYOUT_JSON),
        "layout_json": str(LAYOUT_JSON),
        "windows": [{"name": n, "offset": o, "items": i, "item_channels": c} for o, i, c, n in WINDOWS],
        "recency_planes": [{"name": n, "offset": o, "count": c} for o, c, n in RECENCY],
    }
    _log(f"arm={arm} seed_key={key:#x} seeds=[{seed_start},{seed_start + seeds_per_iter}) "
         f"games={games_per_iter} log_dir={log_dir} perm_seed={meta['perm_seed']}")

    engine_cham = build_engine(cfg["champion"], torch)
    engine_chal = build_engine(cfg["challenger"], torch)
    wrapped = ArmEngine(engine_chal, arm)
    assert wrapped.engine_type == "mortal" and wrapped.name == cfg["challenger"]["name"]
    assert wrapped.version == 4 and wrapped.is_oracle is False

    t0 = time.time()
    env = OneVsThree(disable_progress_bar=False, log_dir=str(log_dir))
    rankings = env.py_vs_py(
        challenger=wrapped,
        champion=engine_cham,
        seed_start=(seed_start, key),
        seed_count=seeds_per_iter,
    )
    wall = time.time() - t0

    rankings = [int(x) for x in rankings]
    assert sum(rankings) == games_per_iter, rankings
    avg_rank = float(np.dot(rankings, np.arange(1, 5)) / games_per_iter)
    avg_pt = float(np.dot(rankings, np.array([90, 45, 0, -135])) / games_per_iter)
    n_files = sum(1 for _ in log_dir.glob("*.json.gz"))
    assert n_files == games_per_iter, f"{n_files} game logs != {games_per_iter}"

    stats = wrapped.stats
    stats["path_counts"] = dict(PATH_COUNTS)
    if not smoke_seeds:
        if arm == "shuffle":
            assert stats["blocks_changed"] > 0, "shuffle never actually applied (all blocks uniform?)"
            assert stats["obs_changed"] > 0, "no observation was changed by the shuffle"
        if arm == "recency_zero":
            assert stats["recency_planes_zeroed_obs"] > 0, "no recency plane was ever nonzero"

    meta.update(
        {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "wall_seconds": round(wall, 1),
            "rankings": rankings,
            "avg_rank": avg_rank,
            "avg_pt": avg_pt,
            "game_log_files": n_files,
            "patch_stats": stats,
            "smoke_seeds": smoke_seeds or None,
        }
    )
    out_json = (log_dir / "smoke_run.json") if smoke_seeds else (A1_DIR / f"{arm}_run.json")
    out_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    _log(f"done arm={arm} wall={wall:.1f}s rankings={rankings} avg_rank={avg_rank:.4f} pt={avg_pt:+.4f}")
    _log(f"patch_stats={json.dumps(stats)}")
    if smoke_seeds:
        shutil.rmtree(log_dir)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A1 river-order intervention arms (challenger only)")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--config", default=str(REPO / "configs" / "config.toml"))
    ap.add_argument("--selftest", action="store_true", help="run in-process patch self-checks and exit")
    ap.add_argument("--smoke", type=int, default=0, metavar="SEEDS", help="tiny real-obs arena run, no repo writes")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.arm:
        ap.error("--arm is required unless --selftest")
    return run_arm(args.arm, Path(args.config), smoke_seeds=args.smoke)


if __name__ == "__main__":
    sys.exit(main())
