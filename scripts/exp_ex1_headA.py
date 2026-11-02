#!/usr/bin/env python3
"""t14 — Route A: public-feature head family (opp_tenpai / deal_in / wait_membership / exp_loss).

Pre-registration: `scripts/exp_ex1_prereg.md` (frozen; sha256 recorded start==end). Shapes, units and
ranges are **referenced** from PREREG §4 (never re-defined here); acceptance lines are PREREG §6.3.

Input features are the t12 (`context`: tiles_left/honba/kyotaku/scores/round_index/tiles_seen/at_turn)
and t13 (`safety`: tile_score, per-opponent genbutsu/suji, visible copies, dora, riichi threat) public
planes, decomposed per opponent where the providers aggregate. **No oracle tensor, no opponent
concealed hand, no true tile order** — `basis = public_info_predictor` (NO-REVEAL, PREREG §8).
`verify-features` proves the vectorized extractor is field-exact against the shipped t12/t13 providers.

Training sample = t2's `runs/exp/EX1/train_manifest.json` **verbatim** (years 2009-2022, 20,000 games,
bounds [10_000, 40_000]); evaluation = `calib_sub` (400 games). Fixed architecture/budget, no search:

  trunk 526 -> 256 -> 256 (ReLU) -> heads: opp_tenpai(3) | deal_in(3x34) | wait(3x34) | exp_loss(3x34)
  AdamW lr=1e-3 wd=1e-5, batch 8192, epochs 4, seed 13 (shuffle control seed 0)

Outputs: `runs/exp/EX1/headA.pt` (+ `headA_shuffle.pt`), `runs/exp/EX1/headA_metrics.json`,
`runs/exp/EX1/headA_shuffle_metrics.json`, feature cache under `runs/exp/EX1/headA/`.

allow: SIZE_OK — the plan (t14) mandates this single entry point; its commit adds only this file.
The body is one pipeline (manifest -> feature cache -> bounded training -> PREREG metrics/controls).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/ (flat modules)
sys.path.insert(0, str(REPO / "vendor/mortal/target/release"))
sys.path.insert(0, str(REPO / "vendor/mortal"))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")        # deterministic cuBLAS (same-seed rerun)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from exp_ex1_audit import assert_prereg_unchanged, load_prereg  # noqa: E402
from exp_ex1_labels import (RecordError, _bootstrap_mortal, _decode_text,  # noqa: E402
                            _read_bytes, load_targets, parse_events, start_names)
from jiangpu import layout  # noqa: E402
from jiangpu.providers import context as context_provider  # noqa: E402
from jiangpu.providers import safety as safety_provider  # noqa: E402

_bootstrap_mortal()  # pin the `mortal` namespace package before exp_a2_* imports (t2/t8 recipe)

PREREG_PATH = REPO / "scripts/exp_ex1_prereg.md"
OUT_DIR = REPO / "runs/exp/EX1"
MANIFEST_PATH = OUT_DIR / "train_manifest.json"
CALIB_LABEL_DIR = OUT_DIR / "labels/calib_sub"
CACHE_DIR = OUT_DIR / "headA"
TRAIN_YEARS = tuple(range(2009, 2023))
FORBIDDEN_YEARS = (2024, 2025, 2026)
TRAIN_BOUNDS = (10_000, 40_000)
HEAD_SEED = 13          # PREREG §2 head training seed (both routes)
SHUFFLE_SEED = 0        # PREREG §2 negative-control seed
BOOTSTRAP_B = 2000      # PREREG §5
BOOTSTRAP_SEED = 7      # PREREG §5
N_BINS = 15
N_TILES = 34
N_OPPS = 3
TRAIN_BS = 8192
TRAIN_EPOCHS = 4
TRAIN_LR = 1e-3
TRAIN_WD = 1e-5
HIDDEN = 256
TRUNK_LAYERS = 2
EVAL_BATCH = 16_384
EXP_LOSS_MAX = 48_000.0  # PREREG §4 payload range; head output = 48000*sigmoid(raw)
GPU_LOCK = Path("/tmp/opencode/train.lock")
TILE_RANGE = np.arange(N_TILES, dtype=np.int64)

# Feature layout — frozen by `feature_spec()` and asserted against the vectorized extractor.
FEATURE_GROUPS = (
    ("tiles_left", 1), ("honba", 1), ("kyotaku", 1), ("scores", 4), ("round_index", 1),
    ("at_turn", 1), ("tiles_seen", N_TILES), ("dora_flag", N_TILES), ("safety_score", N_TILES),
    ("opp_river", N_OPPS * N_TILES), ("opp_genbutsu", N_OPPS * N_TILES),
    ("opp_suji", N_OPPS * N_TILES), ("threat_declared", N_OPPS), ("threat_accepted", N_OPPS),
    ("riichi_sutehai_onehot", N_OPPS * N_TILES), ("n_riichi", 1),
)
FEATURE_DIM = sum(dim for _, dim in FEATURE_GROUPS)
FEATURE_SLICES: dict[str, slice] = {}
_off = 0
for _name, _dim in FEATURE_GROUPS:
    FEATURE_SLICES[_name] = slice(_off, _off + _dim)
    _off += _dim

_NEXT_TILE = np.array([safety_provider.next_tile(t) for t in range(N_TILES)], dtype=np.int64)
_SUJI_ADJ = np.zeros((N_TILES, N_TILES), dtype=np.float32)
for _m in range(N_TILES):
    for _t in safety_provider.suji_mates(_m):
        _SUJI_ADJ[_m, _t] = 1.0


# --------------------------------------------------------------------------- small helpers


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _f(x):
    """numpy scalar -> JSON-native (NaN/inf become None so json.dump stays valid)."""
    v = float(x)
    return v if np.isfinite(v) else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _spans() -> dict[str, tuple[int, int]]:
    top = layout.load_layout()
    names = ("tiles_left", "honba", "kyotaku", "scores", "round_index", "tiles_seen",
             "dora_indicators", "kawa_overview_shimocha", "kawa_overview_toimen",
             "kawa_overview_kamicha", "riichi_sutehai_shimocha", "riichi_sutehai_toimen",
             "riichi_sutehai_kamicha", "riichi_declared_opponents", "riichi_accepted_opponents")
    expected = {"tiles_left": 1, "honba": 1, "kyotaku": 1, "scores": 8, "round_index": 1,
                "tiles_seen": 1, "dora_indicators": 7, "kawa_overview_shimocha": 7,
                "kawa_overview_toimen": 7, "kawa_overview_kamicha": 7,
                "riichi_sutehai_shimocha": 3, "riichi_sutehai_toimen": 3,
                "riichi_sutehai_kamicha": 3, "riichi_declared_opponents": 3,
                "riichi_accepted_opponents": 3}
    spans = {}
    drift = []
    for name in names:
        off, count = layout.span(top, name)
        if count != expected[name]:
            drift.append(f"{name}: expected {expected[name]} planes, got {count}")
        spans[name] = (off, count)
    if drift:
        raise ValueError("obs layout drift: " + "; ".join(drift))
    return spans


def feature_spec() -> dict:
    return {
        "dim": FEATURE_DIM,
        "groups": [{"name": n, "dim": d, "slice": [FEATURE_SLICES[n].start, FEATURE_SLICES[n].stop]}
                   for n, d in FEATURE_GROUPS],
        "source": "t12 context + t13 safety (public planes; per-opponent decomposition)",
        "basis": "public_info_predictor",
        "no_hidden_info": True,
    }


# ------------------------------------------------------------------------ feature extraction


def public_features(obs: np.ndarray, at_turns: np.ndarray, spans: dict) -> np.ndarray:
    """Vectorized t12/t13 public features for one POV game: (n, 526) float32.

    Field-exact vs `context.compute` / `safety.compute` on the same decisions (see `verify_features`).
    """
    n = obs.shape[0]
    seen = np.rint(obs[:, spans["tiles_seen"][0]] * 4).astype(np.float32)          # (n,34) int 0..4
    tiles_left = obs[:, spans["tiles_left"][0], 0] * 69.0
    honba = obs[:, spans["honba"][0], 0] * 10.0
    kyotaku = obs[:, spans["kyotaku"][0], 0] * 10.0
    so = spans["scores"][0]
    scores = obs[:, so:so + 8:2, 0] * 100_000.0                                     # 4x2 planes
    round_index = obs[:, spans["round_index"][0], 0] * 7.0
    dora_off = spans["dora_indicators"][0]
    dora_ind = obs[:, dora_off:dora_off + 4].sum(axis=1) > 0                        # (n,34) bool
    dora = np.zeros((n, N_TILES), dtype=np.float32)
    dora[:, _NEXT_TILE] = dora_ind.astype(np.float32)                               # indicator -> dora

    rivers = []
    for name in ("kawa_overview_shimocha", "kawa_overview_toimen", "kawa_overview_kamicha"):
        off = spans[name][0]
        rivers.append(np.rint(obs[:, off:off + 4].sum(axis=1)).astype(np.float32))
    river = np.stack(rivers, axis=1)                                                # (n,3,34)
    sutehai = []
    for name in ("riichi_sutehai_shimocha", "riichi_sutehai_toimen", "riichi_sutehai_kamicha"):
        plane = obs[:, spans[name][0]]
        sutehai.append(np.where(plane.max(axis=1) > 0, plane.argmax(axis=1), -1))
    sutehai = np.stack(sutehai, axis=1)                                             # (n,3) tile| -1

    do = spans["riichi_declared_opponents"][0]
    ao = spans["riichi_accepted_opponents"][0]
    declared = (obs[:, do:do + 3, 0] > 0).astype(np.float32)
    accepted = (obs[:, ao:ao + 3, 0] > 0).astype(np.float32)
    threat = (declared + accepted) > 0
    threat_f = threat[:, :, None]
    genbutsu = threat_f & ((river > 0) | (sutehai[:, :, None] == TILE_RANGE[None, None, :]))
    suji = threat_f & ~genbutsu & ((river @ _SUJI_ADJ) > 0)
    gen_c = genbutsu.sum(axis=1).astype(np.float32)
    suji_c = suji.sum(axis=1).astype(np.float32)
    score = (safety_provider.SCORE_VISIBLE_FOUR * (seen == 4)
             + safety_provider.SCORE_PER_GENBUTSU * gen_c
             + safety_provider.SCORE_PER_SUJI * suji_c
             + seen
             - safety_provider.SCORE_DORA_PENALTY * dora)
    sutehai_oh = (sutehai[:, :, None] == TILE_RANGE[None, None, :]).astype(np.float32)

    feats = np.concatenate([
        (tiles_left / 69.0)[:, None],
        (honba / 10.0)[:, None],
        (kyotaku / 10.0)[:, None],
        scores / 100_000.0,
        (round_index / 7.0)[:, None],
        (at_turns.astype(np.float32) / 17.0)[:, None],
        seen / 4.0,
        dora,
        score,
        (river / 4.0).reshape(n, -1),
        genbutsu.astype(np.float32).reshape(n, -1),
        suji.astype(np.float32).reshape(n, -1),
        declared,
        accepted,
        sutehai_oh.reshape(n, -1),
        (threat.sum(axis=1, keepdims=True) / 3.0),
    ], axis=1).astype(np.float32)
    if feats.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"feature dim {feats.shape[1]} != frozen {FEATURE_DIM}")
    return feats


def _load_game(source_path: str, seat: int, expected_actions: np.ndarray,
               expected_at_turns: np.ndarray, n_expected: int):
    """Reload one record at the label file's POV; assert the decision stream is the labelled one."""
    from libriichi.dataset import GameplayLoader
    text = _decode_text(_read_bytes(source_path), source_path)
    names = start_names(parse_events(text))
    games = GameplayLoader(version=4, oracle=False, player_names=[names[seat]]).load_log(text)
    game = next((g for g in games if g.take_player_id() == seat), None)
    if game is None:  # duplicate names: rebuild unfiltered and index by seat
        games = GameplayLoader(version=4, oracle=False).load_log(text)
        game = games[seat]
    actions = np.asarray(game.take_actions(), dtype=np.int64)
    at_turns = np.frombuffer(game.take_at_turns(), dtype=np.uint8)
    if actions.shape != expected_actions.shape or not np.array_equal(actions, expected_actions):
        raise RecordError(f"{source_path}: action stream != label file (seat {seat})")
    if len(at_turns) != n_expected or not np.array_equal(at_turns, expected_at_turns):
        raise RecordError(f"{source_path}: at_turns != label file (seat {seat})")
    obs = np.stack([np.asarray(o) for o in game.take_obs()])
    if obs.shape[0] != n_expected:
        raise RecordError(f"{source_path}: obs rows {obs.shape[0]} != {n_expected}")
    return obs, at_turns


def _extract_one(job: dict) -> dict:
    """Worker: one labelled record -> (real-decision features, compact targets)."""
    targets = load_targets(job["label_path"])
    meta = targets["meta"]
    seat = int(meta["player_id"])
    if seat != int(job["seat"]):
        raise RecordError(f"{job['label_path']}: seat {seat} != manifest {job['seat']}")
    obs, at_turns = _load_game(job["source_path"], seat, targets["actions"], targets["at_turns"],
                               int(targets["n"]))
    spans = _spans()
    real = targets["real_mask"].astype(bool)
    x = public_features(obs, at_turns, spans)[real]
    n_real = int(real.sum())
    ron_dec = targets["ron_dec"].astype(np.int64)
    if ron_dec.size and not real[ron_dec].all():
        raise RecordError(f"{job['label_path']}: ron row on a kan-select pseudo-decision")
    return {
        "file_id": int(job["file_id"]),
        "year": int(job["year"]),
        "source_path": job["source_path"],
        "label_path": job["label_path"],
        "n_decisions": int(targets["n"]),
        "n_real": n_real,
        "x": x.astype(np.float16),
        "opp_tenpai": targets["opp_tenpai_prob"][real].astype(np.uint8),
        "wait_packed": np.packbits(targets["wait_membership"][real].astype(np.uint8), axis=-1,
                                   bitorder="little"),
        "ron_dec": ron_dec,
        "ron_j": targets["ron_actor"].astype(np.int8),
        "ron_t": targets["ron_tile"].astype(np.int8),
        "ron_loss": targets["ron_loss"].astype(np.float64),
        "exp_loss_sum": float(targets["exp_loss"].sum()),
    }


def _job_list(entries: list[dict]) -> list[dict]:
    return [{"label_path": e["label_path"], "source_path": e["source_path"],
             "seat": int(e["player_id"]), "file_id": i, "year": int(e["year"])}
            for i, e in enumerate(entries)]


def _calib_jobs() -> list[dict]:
    jobs = []
    for i, label_path in enumerate(sorted(CALIB_LABEL_DIR.glob("*.npz"))):
        with np.load(label_path) as z:
            meta = json.loads(str(z["meta_json"]))
        jobs.append({"label_path": str(label_path), "source_path": meta["source_path"],
                     "seat": int(meta["player_id"]), "file_id": i, "year": int(meta["year"])})
    return jobs


def _write_shard(buf: list[dict], shard_dir: Path, idx: int) -> dict:
    x = np.concatenate([r["x"] for r in buf], axis=0)
    n = x.shape[0]
    file_id = np.concatenate([np.full(r["n_real"], r["file_id"], dtype=np.int32) for r in buf])
    # ron_dec is file-local; make it global by adding each file's real-decision offset
    offsets = np.cumsum([0] + [r["n_real"] for r in buf])[:-1]
    ron_dec = np.concatenate([r["ron_dec"] + off for r, off in zip(buf, offsets)])
    shard = {
        "n": n,
        "x": x.astype(np.float16),
        "opp_tenpai": np.concatenate([r["opp_tenpai"] for r in buf], axis=0),
        "wait_packed": np.concatenate([r["wait_packed"] for r in buf], axis=0),
        "file_id": file_id,
        "ron_dec": ron_dec.astype(np.int64),
        "ron_j": np.concatenate([r["ron_j"] for r in buf]) if buf else np.zeros(0, np.int8),
        "ron_t": np.concatenate([r["ron_t"] for r in buf]) if buf else np.zeros(0, np.int8),
        "ron_loss": np.concatenate([r["ron_loss"] for r in buf]) if buf else np.zeros(0, np.float64),
        "n_files": len(buf),
        "n_decisions": int(sum(r["n_decisions"] for r in buf)),
        "exp_loss_sum": float(sum(r["exp_loss_sum"] for r in buf)),
        "years": [r["year"] for r in buf],
    }
    np.savez(shard_dir / f"shard_{idx:05d}.npz", **shard)
    return shard


def _finalize(subset: str, shard_dir: Path, out_dir: Path, meta: dict) -> dict:
    shards = sorted(shard_dir.glob("shard_*.npz"))
    if not shards:
        raise RecordError(f"{subset}: no shards produced")
    n_total = 0
    for p in shards:
        with np.load(p) as z:
            n_total += int(z["n"])
    x_path = out_dir / f"{subset}_features.npy"
    xm = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float16, shape=(n_total, FEATURE_DIM))
    pos = 0
    parts: dict[str, list[np.ndarray]] = {k: [] for k in
                                          ("opp_tenpai", "wait_packed", "file_id", "ron_dec",
                                           "ron_j", "ron_t", "ron_loss")}
    years: list[int] = []
    n_files = 0
    n_decisions = 0
    exp_loss_sum = 0.0
    for p in shards:
        with np.load(p) as z:
            k = int(z["n"])
            xm[pos:pos + k] = z["x"]
            pos += k
            for key in parts:
                parts[key].append(z[key])
            n_files += int(z["n_files"])
            n_decisions += int(z["n_decisions"])
            exp_loss_sum += float(z["exp_loss_sum"])
            years.extend(int(y) for y in z["years"])
    xm.flush()
    del xm
    targets = {k: np.concatenate(v) for k, v in parts.items()}
    targets["n"] = n_total
    targets["n_files"] = n_files
    targets["n_decisions"] = n_decisions
    targets["exp_loss_sum"] = exp_loss_sum
    targets["years"] = np.array(sorted(set(years)), dtype=np.int64)
    used_ids = np.unique(targets["file_id"]).astype(np.int64).tolist()
    t_path = out_dir / f"{subset}_targets.npz"
    np.savez(t_path, **targets)
    meta = {**meta, "subset": subset, "n_decisions_real": n_total, "n_files": n_files,
            "n_decisions_all": n_decisions, "exp_loss_sum": exp_loss_sum,
            "file_ids": used_ids,
            "features_path": _rel(x_path), "targets_path": _rel(t_path),
            "features_sha256": _sha256(x_path), "targets_sha256": _sha256(t_path)}
    return meta


def prepare(args) -> int:
    sha_start = load_prereg(PREREG_PATH)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = _prepare_jobs(args)
    if args.limit_games is not None:
        if args.limit_games > len(jobs):
            raise SystemExit(f"FATAL: --limit-games {args.limit_games} exceeds the manifest's "
                             f"{len(jobs)} records (the full 1,968,147-file pool is never used)")
        jobs = jobs[:args.limit_games]
    if not jobs:
        raise SystemExit("FATAL: no records selected")
    _guard_years(jobs, subset=args.subset, max_games=args.max_games)
    print(f"[prepare:{args.subset}] {len(jobs)} records, jobs={args.jobs}", flush=True)

    shard_dir = out_dir / f"shards_{args.subset}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("shard_*.npz"):
        old.unlink()
    t0 = time.time()
    buf: list[dict] = []
    buf_decisions = 0
    idx = 0
    meta_extra = {"feature_spec": feature_spec()}
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=_fork_ctx()) as ex:
        for k, res in enumerate(ex.map(_extract_one, jobs, chunksize=4), 1):
            buf.append(res)
            buf_decisions += res["n_real"]
            if buf_decisions >= args.shard_decisions:
                _write_shard(buf, shard_dir, idx)
                idx += 1
                buf, buf_decisions = [], 0
            if k % 500 == 0:
                print(f"[prepare:{args.subset}] {k}/{len(jobs)} records "
                      f"({time.time() - t0:.0f}s)", flush=True)
    if buf:
        _write_shard(buf, shard_dir, idx)
    wall = time.time() - t0
    meta = _finalize(args.subset, shard_dir, out_dir, meta_extra)
    meta["wall_seconds"] = round(wall, 1)
    meta["generated_at"] = _now()
    meta["manifest_sha256"] = args.manifest_sha256
    meta["limit_games"] = args.limit_games
    meta["seed"] = HEAD_SEED
    # field-exact cross-check against the shipped t12/t13 providers
    if args.verify_sample > 0:
        meta["feature_verification"] = verify_features(jobs[:args.verify_sample], max_files=None)
        print(f"[prepare:{args.subset}] feature verification: "
              f"{meta['feature_verification']['status']}", flush=True)
    meta_path = out_dir / f"{args.subset}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n",
                         encoding="utf-8")
    sha_end = assert_prereg_unchanged(PREREG_PATH, sha_start)
    print(json.dumps({"subset": args.subset, "n_files": meta["n_files"],
                      "n_decisions_real": meta["n_decisions_real"],
                      "wall_seconds": meta["wall_seconds"],
                      "prereg_sha256": sha_start, "prereg_unchanged": sha_start == sha_end,
                      "feature_verification": meta.get("feature_verification", {}).get("status")},
                     indent=2))
    return 0


def _fork_ctx():
    import multiprocessing as mp
    return mp.get_context("fork")  # spawn breaks the mortal namespace package (t7/t8 note)


def _guard_years(jobs: list[dict], *, subset: str, max_games: int) -> None:
    if max_games > TRAIN_BOUNDS[1]:
        raise SystemExit(f"FATAL: budget guard — --max-games {max_games} exceeds the PREREG §2 "
                         f"upper bound {TRAIN_BOUNDS[1]} (the full 1,968,147-file pool is forbidden)")
    if len(jobs) > max_games:
        raise SystemExit(f"FATAL: budget guard — {len(jobs)} games > --max-games {max_games} "
                         "(PREREG §2 bounds [10000, 40000])")
    years = sorted({int(j["year"]) for j in jobs})
    if subset == "train":
        bad = [y for y in years if y not in TRAIN_YEARS]
        if bad:
            raise SystemExit(f"FATAL: training years {bad} outside {TRAIN_YEARS} "
                             "(PREREG §2: never train on 2024-2026)")
    else:
        bad = [y for y in years if y not in FORBIDDEN_YEARS]
        if bad:
            raise SystemExit(f"FATAL: {subset} years {bad} outside holdout {FORBIDDEN_YEARS}")
    for j in jobs:
        year = int(j["year"])
        if subset == "train" and year not in TRAIN_YEARS:
            raise SystemExit(f"FATAL: {j['source_path']} year {year} is not a training year")
        if subset != "train" and year not in FORBIDDEN_YEARS:
            raise SystemExit(f"FATAL: {j['source_path']} year {year} is not a holdout year")


def _prepare_jobs(args) -> list[dict]:
    if args.subset == "calib_sub":
        return _calib_jobs()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("schema") != "ex1-train-manifest/v1":
        raise SystemExit(f"FATAL: {args.manifest} is not an ex1 train manifest")
    args.manifest_sha256 = _sha256(args.manifest)
    return _job_list(manifest["files"])


# ------------------------------------------------------------------ provider cross-check


def verify_features(jobs: list[dict], max_files: int | None = 2) -> dict:
    """Field-exact comparison of the vectorized extractor vs t12/t13 providers (per record)."""
    spans = _spans()
    checked = 0
    decisions = 0
    mismatches: list[str] = []
    for job in (jobs if max_files is None else jobs[:max_files]):
        targets = load_targets(job["label_path"])
        seat = int(targets["meta"]["player_id"])
        obs, at_turns = _load_game(job["source_path"], seat, targets["actions"],
                                   targets["at_turns"], int(targets["n"]))
        x = public_features(obs, at_turns, spans)
        actions = targets["actions"]
        decs = [SimpleNamespace(index=i, obs=obs[i], at_turn=int(at_turns[i]),
                                logged_action=int(actions[i])) for i in range(len(obs))]
        ctx = context_provider.compute(decs)
        saf = safety_provider.compute(decs)
        checked += 1
        decisions += len(obs)
        for i in range(len(obs)):
            row = x[i]
            c, s = ctx[i], saf[i]
            if c["tiles_left"] != round(row[FEATURE_SLICES["tiles_left"]][0] * 69):
                mismatches.append(f"{job['source_path']}#{i}: tiles_left")
            if c["honba"] != round(row[FEATURE_SLICES["honba"]][0] * 10):
                mismatches.append(f"{job['source_path']}#{i}: honba")
            if c["kyotaku"] != round(row[FEATURE_SLICES["kyotaku"]][0] * 10):
                mismatches.append(f"{job['source_path']}#{i}: kyotaku")
            if c["round_index"] != round(row[FEATURE_SLICES["round_index"]][0] * 7):
                mismatches.append(f"{job['source_path']}#{i}: round_index")
            if c["at_turn"] != round(row[FEATURE_SLICES["at_turn"]][0] * 17):
                mismatches.append(f"{job['source_path']}#{i}: at_turn")
            if not np.array_equal(np.array(c["scores"]),
                                  np.rint(row[FEATURE_SLICES["scores"]] * 100_000).astype(int)):
                mismatches.append(f"{job['source_path']}#{i}: scores")
            seen = np.rint(row[FEATURE_SLICES["tiles_seen"]] * 4).astype(int)
            if not np.array_equal(np.array(c["tiles_seen"]), seen):
                mismatches.append(f"{job['source_path']}#{i}: tiles_seen")
            comp = s["components"]
            if not np.array_equal(np.array(comp["visible_copies"]), seen):
                mismatches.append(f"{job['source_path']}#{i}: visible_copies")
            if not np.array_equal(np.array(comp["unseen_copies"]), 4 - seen):
                mismatches.append(f"{job['source_path']}#{i}: unseen_copies")
            if not np.array_equal(np.array(comp["dora_flag"]),
                                  row[FEATURE_SLICES["dora_flag"]].astype(bool)):
                mismatches.append(f"{job['source_path']}#{i}: dora_flag")
            if not np.allclose(np.array(s["tile_score"]), row[FEATURE_SLICES["safety_score"]],
                               atol=1e-6):
                mismatches.append(f"{job['source_path']}#{i}: tile_score")
            gen = row[FEATURE_SLICES["opp_genbutsu"]].reshape(N_OPPS, N_TILES)
            suji = row[FEATURE_SLICES["opp_suji"]].reshape(N_OPPS, N_TILES)
            for t in range(N_TILES):
                if bin(int(comp["genbutsu_opponents"][t])).count("1") != int(gen[:, t].sum()):
                    mismatches.append(f"{job['source_path']}#{i}: genbutsu[{t}]")
                if bin(int(comp["suji_opponents"][t])).count("1") != int(suji[:, t].sum()):
                    mismatches.append(f"{job['source_path']}#{i}: suji[{t}]")
            if not np.array_equal(np.array(s["threat"]["declared"]),
                                  row[FEATURE_SLICES["threat_declared"]].astype(bool)):
                mismatches.append(f"{job['source_path']}#{i}: declared")
            if not np.array_equal(np.array(s["threat"]["accepted"]),
                                  row[FEATURE_SLICES["threat_accepted"]].astype(bool)):
                mismatches.append(f"{job['source_path']}#{i}: accepted")
            if s["threat"]["n_riichi"] != round(row[FEATURE_SLICES["n_riichi"]][0] * 3):
                mismatches.append(f"{job['source_path']}#{i}: n_riichi")
            sut = row[FEATURE_SLICES["riichi_sutehai_onehot"]].reshape(N_OPPS, N_TILES)
            for j in range(N_OPPS):
                want = s["threat"]["declaration_tiles"][j]
                got = int(sut[j].argmax()) if sut[j].max() > 0 else None
                if want != got:
                    mismatches.append(f"{job['source_path']}#{i}: sutehai[{j}] {want} != {got}")
            if len(mismatches) > 20:
                break
        if len(mismatches) > 20:
            break
    return {"status": "PASS" if not mismatches else "FAIL",
            "files": checked, "decisions": decisions,
            "mismatch_count": len(mismatches), "mismatches": mismatches[:20],
            "fields": ["tiles_left", "honba", "kyotaku", "scores", "round_index", "at_turn",
                       "tiles_seen", "dora_flag", "tile_score", "genbutsu", "suji",
                       "declared", "accepted", "declaration_tiles", "n_riichi"]}


# ------------------------------------------------------------------------------ metrics


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision, threshold-wise (sklearn convention: ties share one precision point)."""
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(score, dtype=np.float64)
    n_pos = float(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    ss = s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    distinct = np.where(np.diff(ss))[0]
    ends = np.concatenate([distinct, [len(ss) - 1]])
    precision = tp[ends] / np.maximum(tp[ends] + fp[ends], 1e-300)
    recall = tp[ends] / n_pos
    return float((np.diff(np.concatenate([[0.0], recall])) * precision).sum())


def quantile_ece(y: np.ndarray, prob: np.ndarray, n_bins: int = N_BINS) -> dict:
    """ECE with equal-count (quantile) bins of the prediction — for low-prevalence targets (MA-2)."""
    from exp_a2b_calibration import quantile_bins
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(prob, dtype=np.float64)
    idx, edges = quantile_bins(p, n_bins)
    n = len(y)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        rows.append({"bin": b, "n": nb, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "mean_conf": float(p[m].mean()), "acc": float(y[m].mean()),
                     "gap": float(y[m].mean() - p[m].mean())})
    vce = float(sum(r["n"] / n * abs(r["gap"]) for r in rows)) if n else float("nan")
    return {"ece": vce, "n_bins_effective": int(len(edges) - 1), "rows": rows}


def brier_skill(y: np.ndarray, prob: np.ndarray) -> float:
    from exp_a2_probe import brier_score
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(prob, dtype=np.float64)
    base = float(y.mean()) * (1.0 - float(y.mean()))
    if base <= 0:
        return float("nan")
    return float(1.0 - brier_score(y, p) / base)


def mae_skill(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    mae = float(np.abs(p - y).mean())
    mae_base = float(np.abs(y.mean() - y).mean())
    return {"mae": mae, "mae_base": mae_base,
            "mae_skill": float(1.0 - mae / mae_base) if mae_base > 0 else float("nan")}


# ---------------------------------------------------------------- cluster bootstrap (fork)


_BOOT_DATA: dict = {}
_BOOT_STATS: dict = {}


def _draw_index(bounds: np.ndarray, n: int, rng) -> np.ndarray:
    """Cluster-bootstrap row index: resample groups (files) with replacement (t5 repeat trick)."""
    counts = np.bincount(rng.integers(0, len(bounds) - 1, len(bounds) - 1),
                         minlength=len(bounds) - 1)
    return np.repeat(np.arange(n), np.repeat(counts, np.diff(bounds)))


def _boot_worker(payload):
    key, stat_name, seeds = payload
    d = _BOOT_DATA[key]
    fn = _BOOT_STATS[stat_name]
    out = np.empty(len(seeds), dtype=np.float64)
    for k, seed in enumerate(seeds):
        rng = np.random.default_rng([BOOTSTRAP_SEED, seed])
        idx = _draw_index(d["bounds"], d["n"], rng)
        out[k] = fn(d["y"], d["p"], idx)
    return key, stat_name, out


def _register_stats() -> None:
    from exp_a2_probe import auc_score, ece_score
    from exp_a2b_calibration import bin_sums, ece_from_sums, spearman, value_table

    def _auc(y, p, idx):
        return auc_score(y[idx], p[idx])

    def _bss(y, p, idx):
        return brier_skill(y[idx], p[idx])

    def _ece(y, p, idx):
        _, cnt, sconf, sy = bin_sums(p[idx], y[idx], np.zeros(len(idx), dtype=np.int64))
        return ece_from_sums(cnt, sconf, sy)

    def _qece(y, p, idx):
        return quantile_ece(y[idx], p[idx])["ece"]

    def _ap(y, p, idx):
        return average_precision(y[idx], p[idx])

    def _mae_skill(y, p, idx):
        return mae_skill(y[idx], p[idx])["mae_skill"]

    def _spearman(y, p, idx):
        return spearman(p[idx], y[idx])

    def _vce_std(y, p, idx):
        return value_table(p[idx], y[idx])["value_binned_calibration_error_over_std"]

    def _riichi_lift(y, p, idx):
        yv, pv = y[idx], p[idx]
        a = pv[yv > 0.5]
        b = pv[yv < 0.5]
        return float(a.mean() - b.mean()) if len(a) and len(b) else float("nan")

    _BOOT_STATS.update({"auc": _auc, "bss": _bss, "ece": _ece, "qece": _qece, "ap": _ap,
                        "mae_skill": _mae_skill, "spearman": _spearman, "vce_std": _vce_std,
                        "riichi_lift": _riichi_lift})


def bootstrap_ci(families: dict, specs: list[tuple[str, str]], n_boot: int, jobs: int) -> dict:
    """95% file-cluster bootstrap CIs (B=2000, PREREG §5); draws fanned out over fork workers."""
    _register_stats()
    _BOOT_DATA.clear()
    for key, fam in families.items():
        groups = np.asarray(fam["groups"], dtype=np.int64)
        uniq = np.unique(groups) if groups.size else np.zeros(0, dtype=np.int64)
        if groups.size == 0 or not np.array_equal(uniq, np.arange(len(uniq))) \
                or not np.all(np.diff(groups) >= 0):
            raise RuntimeError(f"{key}: groups must be sorted and contiguous 0..G-1")
        bounds = np.searchsorted(groups, np.arange(len(uniq) + 1))
        _BOOT_DATA[key] = {"y": np.asarray(fam["y"], dtype=np.float64),
                           "p": np.asarray(fam["p"], dtype=np.float64),
                           "bounds": bounds, "n": len(groups)}
    tasks = []
    n_chunks = max(1, jobs * 2)
    for key, stat in specs:
        seeds = np.arange(n_boot, dtype=np.int64)
        for chunk in np.array_split(seeds, n_chunks):
            if len(chunk):
                tasks.append((key, stat, chunk.tolist()))
    out: dict[tuple[str, str], list[float]] = {(k, s): [] for k, s in specs}
    with ProcessPoolExecutor(max_workers=jobs, mp_context=_fork_ctx()) as ex:
        for key, stat, vals in ex.map(_boot_worker, tasks, chunksize=1):
            out[(key, stat)].extend(vals.tolist())
    ci = {}
    for key, stat in specs:
        vals = np.asarray(out[(key, stat)], dtype=np.float64)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ci[(key, stat)] = {"ci95": [_f(lo), _f(hi)], "n_boot": int(len(vals)),
                           "bootstrap_mean": _f(vals.mean())}
    return ci


# ------------------------------------------------------------------------- families


def binary_family(y: np.ndarray, p: np.ndarray, groups: np.ndarray, *, binning: str) -> dict:
    from exp_a2_probe import auc_score, ece_score
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    pbar = float(y.mean())
    point = {
        "n": int(len(y)), "n_positive": int(y.sum()), "prevalence": _f(pbar),
        "auc": _f(auc_score(y, p)),
        "brier_skill": _f(brier_skill(y, p)),
        "average_precision": _f(average_precision(y, p)),
        "ece_equal_width_15": _f(ece_score(y, p, N_BINS)),
    }
    if binning == "quantile":
        q = quantile_ece(y, p)
        point["quantile_ece_15"] = _f(q["ece"])
        point["quantile_bins_effective"] = q["n_bins_effective"]
        point["quantile_reliability_table"] = q["rows"]
    const = np.full_like(p, pbar)
    point["base_rate_control"] = {
        "predictor": "constant prevalence of the evaluation subset",
        "auc": _f(auc_score(y, const)),
        "brier_skill": _f(brier_skill(y, const)),
        "average_precision": _f(average_precision(y, const)),
        "ece_equal_width_15": _f(ece_score(y, const, N_BINS)),
        "quantile_ece_15": _f(quantile_ece(y, const)["ece"]),
    }
    return {"point": point, "groups": groups}


def continuous_family(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict:
    from exp_a2b_calibration import spearman, value_table
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    vt = value_table(p, y)
    ms = mae_skill(y, p)
    const = np.full_like(p, float(y.mean()))
    vt_const = value_table(const, y)
    point = {
        "n": int(len(y)), "target_sum": _f(y.sum()), "target_mean": _f(y.mean()),
        "target_std": _f(y.std()), "mean_pred": _f(p.mean()),
        "mae": _f(ms["mae"]), "mae_base": _f(ms["mae_base"]), "mae_skill": _f(ms["mae_skill"]),
        "spearman": _f(spearman(p, y)),
        "value_binned_calibration_error_pts": _f(vt["value_binned_calibration_error_pts"]),
        "value_binned_calibration_error_over_std":
            _f(vt["value_binned_calibration_error_over_std"]),
        "value_bins_effective": int(vt["n_bins"]),
        "value_reliability_table": vt["rows"],
        "base_rate_control": {
            "predictor": "constant target mean of the evaluation subset",
            "mae": _f(mae_skill(y, const)["mae"]),
            "mae_skill": _f(mae_skill(y, const)["mae_skill"]),
            "spearman": _f(spearman(const, y)),
            "value_binned_calibration_error_over_std":
                _f(vt_const["value_binned_calibration_error_over_std"]),
        },
    }
    return {"point": point, "groups": groups}


# ---------------------------------------------------------------------------- model


class HeadA(nn.Module):
    """Fixed bounded architecture (no hyperparameter search, PREREG §Scope).

    `exp_loss` is a hurdle (two-part) estimator of the conditional mean for a zero-inflated
    target: `E[loss] = P(loss>0) * E[loss | loss>0]` — plain MSE on raw points is dominated by
    the rare 1000-48000-point rons and cannot reach the MAE-optimal regime.
    """

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = HIDDEN, layers: int = TRUNK_LAYERS):
        super().__init__()
        trunk: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            trunk += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.trunk = nn.Sequential(*trunk)
        self.opp_tenpai = nn.Linear(d, N_OPPS)
        self.deal_in = nn.Linear(d, N_OPPS * N_TILES)
        self.wait_membership = nn.Linear(d, N_OPPS * N_TILES)
        self.exp_occ = nn.Linear(d, N_OPPS * N_TILES)
        self.exp_val = nn.Linear(d, N_OPPS * N_TILES)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        b = h.shape[0]
        occ_logits = self.exp_occ(h).view(b, N_OPPS, N_TILES)
        val_pts = EXP_LOSS_MAX * torch.sigmoid(self.exp_val(h).view(b, N_OPPS, N_TILES))
        return {
            "opp_tenpai_logits": self.opp_tenpai(h),
            "deal_in_logits": self.deal_in(h).view(b, N_OPPS, N_TILES),
            "wait_logits": self.wait_membership(h).view(b, N_OPPS, N_TILES),
            "exp_occ_logits": occ_logits,
            "exp_val_pts": val_pts,
            "exp_loss_pts": torch.sigmoid(occ_logits) * val_pts,
        }


def _deterministic() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_cache(out_dir: Path, subset: str) -> tuple[np.ndarray, dict, dict]:
    meta_path = out_dir / f"{subset}_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    x = np.load(out_dir / f"{subset}_features.npy")
    with np.load(out_dir / f"{subset}_targets.npz") as z:
        targets = {k: z[k] for k in z.files}
    if x.shape != (targets["n"], FEATURE_DIM):
        raise RuntimeError(f"{subset}: feature shape {x.shape} != {(targets['n'], FEATURE_DIM)}")
    return x, targets, meta


def standardize_stats(x: np.ndarray, chunk: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    s = np.zeros(FEATURE_DIM, dtype=np.float64)
    ss = np.zeros(FEATURE_DIM, dtype=np.float64)
    for i in range(0, n, chunk):
        b = x[i:i + chunk].astype(np.float32)
        s += b.sum(axis=0, dtype=np.float64)
        ss += (b.astype(np.float64) ** 2).sum(axis=0)
    mu = s / n
    var = np.maximum(ss / n - mu * mu, 1e-12)
    return mu.astype(np.float32), np.sqrt(var).astype(np.float32)


def _sparse_dense(targets: dict, n: int) -> dict:
    """Dense (n,3,34) targets for deal_in (0/1) and exp_loss (points)."""
    dec = targets["ron_dec"].astype(np.int64)
    deal = np.zeros((n, N_OPPS, N_TILES), dtype=np.float64)
    exp = np.zeros((n, N_OPPS, N_TILES), dtype=np.float64)
    np.add.at(deal, (dec, targets["ron_j"].astype(np.int64), targets["ron_t"].astype(np.int64)), 1.0)
    np.add.at(exp, (dec, targets["ron_j"].astype(np.int64), targets["ron_t"].astype(np.int64)),
              targets["ron_loss"])
    return {"deal_in": deal, "exp_loss": exp}


def _flat_shuffle_targets(targets: dict, n: int, seed: int) -> dict:
    """Negative-control labels: fully exchangeable across (decision, opponent, tile) slots.

    A decision-level permutation would leave per-slot marginals learnable from the per-tile input
    features (a real, non-leakage signal), so the correct null for a slot-indexed head is the flat
    permutation of the whole (n,3,34) label tensor.
    """
    rng = np.random.default_rng(seed)
    out = {"file_id": targets["file_id"]}
    opp = targets["opp_tenpai"].reshape(-1)
    out["opp_tenpai"] = opp[rng.permutation(opp.size)].reshape(n, N_OPPS)
    wait = np.unpackbits(targets["wait_packed"], axis=-1,
                         bitorder="little")[..., :N_TILES].reshape(-1)
    perm = np.arange(wait.size, dtype=np.int32)
    rng.shuffle(perm)
    out["wait_packed"] = np.packbits(wait[perm].reshape(n, N_OPPS, N_TILES), axis=-1,
                                     bitorder="little")
    slots = (targets["ron_dec"].astype(np.int64) * (N_OPPS * N_TILES)
             + targets["ron_j"].astype(np.int64) * N_TILES + targets["ron_t"].astype(np.int64))
    new_slots = perm[slots].astype(np.int64)
    dec, rem = np.divmod(new_slots, N_OPPS * N_TILES)
    j, t = np.divmod(rem, N_TILES)
    out["ron_dec"] = dec
    out["ron_j"] = j.astype(np.int8)
    out["ron_t"] = t.astype(np.int8)
    out["ron_loss"] = targets["ron_loss"]
    return out


def train_head(x: np.ndarray, targets: dict, *, seed: int, shuffle_seed: int | None,
               epochs: int, batch: int, device: str, log: bool = True) -> tuple[HeadA, dict]:
    """Train the fixed head family; `shuffle_seed` swaps in flat-shuffled labels (PREREG §5)."""
    _deterministic()
    torch.manual_seed(seed)
    n = x.shape[0]
    mu, sd = standardize_stats(x)
    if shuffle_seed is not None:
        targets = _flat_shuffle_targets(targets, n, shuffle_seed)
    model = HeadA().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WD)
    rng = np.random.default_rng(seed)
    pos_in_batch = np.full(n, -1, dtype=np.int64)
    wall0 = time.time()
    steps = 0
    last: dict[str, float] = {}
    for epoch in range(epochs):
        order = rng.permutation(n)
        for lo in range(0, n, batch):
            sel = order[lo:lo + batch]
            b = len(sel)
            pos_in_batch[sel] = np.arange(b, dtype=np.int64)
            xb = torch.from_numpy(np.ascontiguousarray(x[sel])).to(device).float()
            xb = (xb - torch.from_numpy(mu).to(device)) / torch.from_numpy(sd).to(device)
            y_opp = torch.from_numpy(targets["opp_tenpai"][sel].astype(np.float32)).to(device)
            wait = np.unpackbits(targets["wait_packed"][sel], axis=-1,
                                 bitorder="little")[..., :N_TILES].astype(np.float32)
            y_wait = torch.from_numpy(np.ascontiguousarray(wait)).to(device)
            dec = targets["ron_dec"].astype(np.int64)
            take = pos_in_batch[dec] >= 0
            deal = np.zeros((b, N_OPPS, N_TILES), dtype=np.float32)
            exp = np.zeros((b, N_OPPS, N_TILES), dtype=np.float32)
            if take.any():
                bi = pos_in_batch[dec[take]]
                ji = targets["ron_j"][take].astype(np.int64)
                ti = targets["ron_t"][take].astype(np.int64)
                deal[bi, ji, ti] = 1.0
                exp[bi, ji, ti] = targets["ron_loss"][take].astype(np.float32)
            y_deal = torch.from_numpy(deal).to(device)
            y_exp = torch.from_numpy(exp).to(device)
            out = model(xb)
            loss_opp = torch.nn.functional.binary_cross_entropy_with_logits(
                out["opp_tenpai_logits"], y_opp)
            loss_deal = torch.nn.functional.binary_cross_entropy_with_logits(
                out["deal_in_logits"].reshape(b, -1), y_deal.reshape(b, -1))
            loss_wait = torch.nn.functional.binary_cross_entropy_with_logits(
                out["wait_logits"].reshape(b, -1), y_wait.reshape(b, -1))
            occ_target = (y_exp > 0).float()
            loss_occ = torch.nn.functional.binary_cross_entropy_with_logits(
                out["exp_occ_logits"].reshape(b, -1), occ_target.reshape(b, -1))
            pos = y_exp > 0
            loss_mag = (torch.nn.functional.mse_loss(out["exp_val_pts"][pos] / EXP_LOSS_MAX,
                                                     y_exp[pos] / EXP_LOSS_MAX)
                        if bool(pos.any()) else torch.zeros((), device=device))
            loss_exp = loss_occ + loss_mag
            loss = loss_opp + loss_deal + loss_wait + loss_exp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pos_in_batch[sel] = -1
            steps += 1
            last = {"opp_tenpai": float(loss_opp.detach()), "deal_in": float(loss_deal.detach()),
                    "wait_membership": float(loss_wait.detach()), "exp_loss": float(loss_exp.detach()),
                    "total": float(loss.detach())}
            if log and steps % 200 == 0:
                print(f"[train] epoch {epoch + 1}/{epochs} step {steps} loss={last['total']:.4f} "
                      f"(opp {last['opp_tenpai']:.4f} deal {last['deal_in']:.4f} "
                      f"wait {last['wait_membership']:.4f} exp {last['exp_loss']:.4f})", flush=True)
        if log:
            print(f"[train] epoch {epoch + 1} done ({time.time() - wall0:.0f}s)", flush=True)
    record = {"seed": seed, "shuffle_seed": shuffle_seed, "epochs": epochs, "batch_size": batch,
              "lr": TRAIN_LR, "weight_decay": TRAIN_WD, "hidden": HIDDEN,
              "trunk_layers": TRUNK_LAYERS, "params": int(sum(p.numel() for p in model.parameters())),
              "steps": steps, "n_train": int(n), "wall_seconds": round(time.time() - wall0, 1),
              "final_losses": last, "feature_mean_sha256": _sha256_bytes(mu.tobytes()),
              "feature_std_sha256": _sha256_bytes(sd.tobytes())}
    return model, record, mu, sd


def _sha256_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _weights_sha(model: HeadA) -> str:
    """Deterministic hash of the model weights (torch.save container bytes are not stable)."""
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def predict(model: HeadA, x: np.ndarray, mu: np.ndarray, sd: np.ndarray, device: str,
            batch: int = EVAL_BATCH) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list[np.ndarray]] = {k: [] for k in
                                        ("opp_tenpai", "deal_in", "wait_membership", "exp_loss")}
    with torch.inference_mode():
        for i in range(0, x.shape[0], batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch])).to(device).float()
            xb = (xb - torch.from_numpy(mu).to(device)) / torch.from_numpy(sd).to(device)
            o = model(xb)
            out["opp_tenpai"].append(torch.sigmoid(o["opp_tenpai_logits"]).float().cpu().numpy())
            out["deal_in"].append(torch.sigmoid(o["deal_in_logits"]).float().cpu().numpy())
            out["wait_membership"].append(torch.sigmoid(o["wait_logits"]).float().cpu().numpy())
            out["exp_loss"].append(o["exp_loss_pts"].float().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def predict_constant(model: HeadA, mu: np.ndarray, sd: np.ndarray, device: str,
                     n: int) -> dict[str, np.ndarray]:
    """Constant-input control: feed the standardized mean (x=0) and expand the constant output."""
    zero = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    out = predict(model, zero, np.zeros_like(mu), np.ones_like(sd), device)
    return {k: np.repeat(v, n, axis=0) for k, v in out.items()}


# ------------------------------------------------------------------------- evaluation


def evaluate_all(model: HeadA, calib: dict, mu, sd, device: str, *,
                 constant_input: bool = False) -> dict:
    x = calib["x"]
    n = x.shape[0]
    pred = (predict_constant(model, mu, sd, device, n) if constant_input
            else predict(model, x, mu, sd, device))
    y_opp = calib["targets"]["opp_tenpai"]
    dense = _sparse_dense(calib["targets"], n)
    wait = np.unpackbits(calib["targets"]["wait_packed"], axis=-1,
                         bitorder="little")[..., :N_TILES]
    groups = calib["targets"]["file_id"].astype(np.int64)
    families = {
        "opp_tenpai_prob": {"y": y_opp.reshape(-1), "p": pred["opp_tenpai"].reshape(-1),
                            "groups": np.repeat(groups, N_OPPS)},
        "deal_in": {"y": dense["deal_in"].reshape(-1), "p": pred["deal_in"].reshape(-1),
                    "groups": np.repeat(groups, N_OPPS * N_TILES)},
        "wait_membership": {"y": wait.reshape(-1).astype(np.float64),
                            "p": pred["wait_membership"].reshape(-1),
                            "groups": np.repeat(groups, N_OPPS * N_TILES)},
        "exp_loss": {"y": dense["exp_loss"].reshape(-1), "p": pred["exp_loss"].reshape(-1),
                     "groups": np.repeat(groups, N_OPPS * N_TILES)},
    }
    return {"families": families, "pred": pred, "groups": groups, "y_opp": y_opp}


def metrics_from_families(families: dict, jobs: int, n_boot: int = BOOTSTRAP_B) -> dict:
    specs = [("opp_tenpai_prob", "auc"), ("opp_tenpai_prob", "bss"), ("opp_tenpai_prob", "ece"),
             ("deal_in", "ap"), ("deal_in", "bss"), ("deal_in", "qece"),
             ("wait_membership", "ap"), ("wait_membership", "bss"), ("wait_membership", "qece"),
             ("exp_loss", "mae_skill"), ("exp_loss", "spearman"), ("exp_loss", "vce_std")]
    ci = bootstrap_ci(families, specs, n_boot, jobs) if n_boot > 0 else {}
    out = {
        "opp_tenpai_prob": binary_family(families["opp_tenpai_prob"]["y"],
                                         families["opp_tenpai_prob"]["p"],
                                         families["opp_tenpai_prob"]["groups"], binning="equal_width"),
        "deal_in": binary_family(families["deal_in"]["y"], families["deal_in"]["p"],
                                 families["deal_in"]["groups"], binning="quantile"),
        "wait_membership": binary_family(families["wait_membership"]["y"],
                                         families["wait_membership"]["p"],
                                         families["wait_membership"]["groups"], binning="quantile"),
        "exp_loss": continuous_family(families["exp_loss"]["y"], families["exp_loss"]["p"],
                                      families["exp_loss"]["groups"]),
    }
    for name, stat in specs:
        if ci:
            out[name]["point"].setdefault("ci95", {})[stat] = ci[(name, stat)]
    return out


def triage_reference(families: dict) -> dict:
    """Informational: PREREG §6.3 triage lines measured on calib_sub (official selection is t16)."""
    opp = families["opp_tenpai_prob"]["point"]
    deal = families["deal_in"]["point"]
    wait = families["wait_membership"]["point"]
    exp = families["exp_loss"]["point"]
    checks = {
        "H1": {"bss>=0.05": opp["brier_skill"] is not None and opp["brier_skill"] >= 0.05,
               "auc>=0.70": opp["auc"] is not None and opp["auc"] >= 0.70},
        "H2": {"pr_auc>=0.005": deal["average_precision"] is not None
               and deal["average_precision"] >= 0.005,
               "pr_auc>=3xprev": deal["average_precision"] is not None
               and deal["average_precision"] >= 3 * deal["prevalence"],
               "bss>=0.005": deal["brier_skill"] is not None and deal["brier_skill"] >= 0.005},
        "H3": {"pr_auc>=0.03": wait["average_precision"] is not None
               and wait["average_precision"] >= 0.03,
               "pr_auc>=2xprev": wait["average_precision"] is not None
               and wait["average_precision"] >= 2 * wait["prevalence"],
               "bss>=0.02": wait["brier_skill"] is not None and wait["brier_skill"] >= 0.02},
        "H4": {"mae_skill>=0.02": exp["mae_skill"] is not None and exp["mae_skill"] >= 0.02,
               "rho>=0.05": exp["spearman"] is not None and exp["spearman"] >= 0.05},
    }
    return {"note": "informational only — PREREG §6.5 selection happens on select_sub in t16",
            "checks": checks}


# ---------------------------------------------------------------------- gate-2 invariants


def gate2_invariants(model: HeadA, calib: dict, mu, sd, device: str, jobs: int) -> dict:
    """Spec gate-2 domain invariants (MA-6) on the calib_sub predictions."""
    x = calib["x"]
    pred = predict(model, x, mu, sd, device)
    n = x.shape[0]
    declared = x[:, FEATURE_SLICES["threat_declared"]] > 0
    accepted = x[:, FEATURE_SLICES["threat_accepted"]] > 0
    threat = declared | accepted                                    # (n,3)
    genbutsu = x[:, FEATURE_SLICES["opp_genbutsu"]].reshape(n, N_OPPS, N_TILES) > 0
    dense = _sparse_dense(calib["targets"], n)
    p_opp = pred["opp_tenpai"]
    p_deal = pred["deal_in"]

    riichi_vals = p_opp[threat]
    other_vals = p_opp[~threat]
    lift = {"n_riichi_slots": int(threat.sum()), "n_other_slots": int((~threat).sum()),
            "mean_pred_riichi": _f(riichi_vals.mean()),
            "mean_pred_no_riichi": _f(other_vals.mean()),
            "mean_truth_riichi": _f(calib["targets"]["opp_tenpai"][threat].mean()),
            "lift": _f(riichi_vals.mean() - other_vals.mean())}
    groups = calib["targets"]["file_id"].astype(np.int64)
    fam = {"riichi": {"y": threat.reshape(-1).astype(np.float64), "p": p_opp.reshape(-1),
                      "groups": np.repeat(groups, N_OPPS)}}
    ci = bootstrap_ci(fam, [("riichi", "riichi_lift")], BOOTSTRAP_B, jobs)
    lift["lift_ci95"] = ci[("riichi", "riichi_lift")]["ci95"]

    gmask = genbutsu
    gen_mean = float(p_deal[gmask].mean()) if gmask.any() else float("nan")
    non_gen_mean = float(p_deal[~gmask].mean())
    gen_truth = float(dense["deal_in"][gmask].sum())
    pair_gen, pair_non, pairs_better = [], [], 0
    for j in range(N_OPPS):
        rows = np.where(threat[:, j])[0]
        for dec in rows:
            g = gmask[dec, j]
            if not g.any() or g.all():
                continue
            mg = float(p_deal[dec, j, g].mean())
            mn = float(p_deal[dec, j, ~g].mean())
            pair_gen.append(mg)
            pair_non.append(mn)
            pairs_better += int(mg < mn)
    pair_gen_m = float(np.mean(pair_gen)) if pair_gen else float("nan")
    pair_non_m = float(np.mean(pair_non)) if pair_non else float("nan")
    genbutsu_block = {
        "n_genbutsu_slots": int(gmask.sum()),
        "mean_pred_genbutsu": _f(gen_mean), "mean_pred_non_genbutsu": _f(non_gen_mean),
        "ratio_genbutsu_over_all": _f(gen_mean / float(p_deal.mean())),
        "truth_positives_on_genbutsu": _f(gen_truth),
        "truth_positives_on_non_genbutsu": _f(dense["deal_in"][~gmask].sum()),
        "within_decision_opponent_pairs": int(len(pair_gen)),
        "pair_mean_pred_genbutsu": _f(pair_gen_m),
        "pair_mean_pred_non_genbutsu": _f(pair_non_m),
        "pairs_where_genbutsu_safer": int(pairs_better),
        "pairs_where_genbutsu_safer_fraction": _f(pairs_better / len(pair_gen) if pair_gen else float("nan")),
    }
    sum_diag = {
        "mean_sum_t_pred_deal_in_per_opponent":
            [_f(p_deal[:, j, :].sum(axis=1).mean()) for j in range(N_OPPS)],
        "mean_sum_t_truth_deal_in_per_opponent":
            [_f(dense["deal_in"][:, j, :].sum(axis=1).mean()) for j in range(N_OPPS)],
        "note": "diagnostic only (MA-6): total predicted deal-in mass per opponent per decision",
    }
    passed = bool(lift["lift"] is not None and lift["lift"] > 0
                  and lift["lift_ci95"][0] is not None and lift["lift_ci95"][0] > 0
                  and gen_truth == 0.0 and pair_gen_m < pair_non_m and gen_mean < 0.02)
    return {"opp_tenpai_riichi_lift": lift, "deal_in_genbutsu": genbutsu_block,
            "deal_in_sum_diagnostic": sum_diag,
            "checks": {"riichi_lift_positive": bool(lift["lift"] > 0),
                       "riichi_lift_ci_lower_gt_0": bool(lift["lift_ci95"][0] > 0),
                       "genbutsu_truth_exactly_zero": bool(gen_truth == 0.0),
                       "genbutsu_within_pair_safer": bool(pair_gen_m < pair_non_m),
                       "genbutsu_pred_abs_lt_0.02": bool(gen_mean < 0.02)},
            "passed": passed}


# ------------------------------------------------------------------------------ training


def _check_gpu_lock(device: str) -> None:
    if device == "cuda" and not GPU_LOCK.exists():
        raise SystemExit(f"FATAL: GPU lock {GPU_LOCK} not held — create it with `mkdir` first "
                         "(AGENTS.md GPU lock protocol; t14/t15 are serial)")


def _resolve_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 1.5e9:
        return "cuda"
    return "cpu"


def run_train(args) -> int:
    t_start = time.time()
    sha_start = load_prereg(PREREG_PATH)
    out_dir = Path(args.out_dir)
    device = _resolve_device(args.device)
    _check_gpu_lock(device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest_sha = _sha256(args.manifest)
    train_x, train_t, train_meta = _load_cache(out_dir, "train")
    calib_x, calib_t, calib_meta = _load_cache(out_dir, "calib_sub")
    if train_meta["manifest_sha256"] != manifest_sha:
        raise SystemExit("FATAL: feature cache was built from a different manifest "
                         f"({train_meta['manifest_sha256']} != {manifest_sha})")
    n_games = int(train_meta["n_files"])
    n_decisions = int(train_meta["n_decisions_real"])
    years = [int(y) for y in train_t["years"]]
    if args.smoke:
        print(f"[train] SMOKE: {n_games} games / {n_decisions} decisions (bounds not enforced)",
              flush=True)
    else:
        if not TRAIN_BOUNDS[0] <= n_games <= TRAIN_BOUNDS[1]:
            raise SystemExit(f"FATAL: budget guard — {n_games} games outside PREREG bounds "
                             f"{TRAIN_BOUNDS}")
        if any(y not in TRAIN_YEARS for y in years):
            raise SystemExit(f"FATAL: training years {years} include 2024-2026")
    used = [int(i) for i in train_meta["file_ids"]]
    want_exp = float(sum(manifest["files"][i]["exp_loss_sum"] for i in used))
    want_rons = int(sum(manifest["files"][i]["n_rons"] for i in used))
    assert float(train_t["exp_loss_sum"]) == want_exp, (
        "exp_loss labels are not t2's continuous labels "
        f"({train_t['exp_loss_sum']} != {want_exp} over the selected manifest entries)")
    assert int(train_t["ron_dec"].size) == want_rons, (
        f"deal_in rows {train_t['ron_dec'].size} != manifest n_rons {want_rons}")
    if not args.smoke:
        calib_rons = json.loads((OUT_DIR / "holdout_labels.json").read_text(encoding="utf-8"))
        want = int(calib_rons["subsets"]["calib_sub"]["n_rons"])
        assert int(calib_t["ron_dec"].size) == want, (
            f"calib_sub ron rows {calib_t['ron_dec'].size} != {want} (holdout_labels.json)")

    print(f"[train] device={device} games={n_games} decisions={n_decisions} "
          f"features={train_x.shape} exp_loss_sum={train_t['exp_loss_sum']:.0f}", flush=True)
    model, record, mu, sd = train_head(train_x, train_t, seed=args.seed, shuffle_seed=None,
                                       epochs=args.epochs, batch=args.batch, device=device)
    ckpt_path = Path(args.checkpoint_out) if args.checkpoint_out else out_dir / "headA.pt"
    weights_sha = _weights_sha(model)
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "feature_spec": feature_spec(), "train": record, "manifest_sha256": manifest_sha,
                "prereg_sha256": sha_start, "route": "A_public_features",
                "weights_sha256": weights_sha}, ckpt_path)
    ckpt_sha = _sha256(ckpt_path)
    print(f"[train] checkpoint {_rel(ckpt_path)} sha256={ckpt_sha[:12]} ({record['wall_seconds']}s)",
          flush=True)

    calib = {"x": calib_x, "targets": calib_t}
    # --- negative control 1: trained on shuffled labels (PREREG §5) -------------------
    ctrl_targets = _flat_shuffle_targets(train_t, train_x.shape[0], SHUFFLE_SEED)
    ctrl, ctrl_record, _, _ = train_head(train_x, ctrl_targets, seed=SHUFFLE_SEED,
                                         shuffle_seed=None, epochs=args.epochs,
                                         batch=args.batch, device=device, log=False)
    ctrl_calib = {"x": calib_x,
                  "targets": _flat_shuffle_targets(calib_t, calib_x.shape[0], SHUFFLE_SEED)}
    ctrl_eval = evaluate_all(ctrl, ctrl_calib, mu, sd, device)
    ctrl_metrics = metrics_from_families(ctrl_eval["families"], args.jobs,
                                         n_boot=max(2, args.bootstrap // 10))
    ctrl_ckpt = out_dir / "headA_shuffle.pt"
    ctrl_weights_sha = _weights_sha(ctrl)
    torch.save({"state_dict": ctrl.state_dict(), "mu": mu, "sd": sd, "train": ctrl_record,
                "shuffle_seed": SHUFFLE_SEED, "weights_sha256": ctrl_weights_sha}, ctrl_ckpt)
    print("[train] shuffle control done", flush=True)

    # --- main metrics on calib_sub -----------------------------------------------------
    ev = evaluate_all(model, calib, mu, sd, device)
    families = metrics_from_families(ev["families"], args.jobs, n_boot=args.bootstrap)
    for name in ("opp_tenpai_prob", "deal_in", "wait_membership"):
        br = families[name]["point"]["base_rate_control"]
        assert br["auc"] == 0.5 and abs(br["brier_skill"]) < 1e-12 \
            and abs(br["ece_equal_width_15"]) < 1e-12 \
            and abs(br["average_precision"] - families[name]["point"]["prevalence"]) < 1e-12, (
                f"{name}: base-rate control does not behave as a constant: {br}")
    exp_br = families["exp_loss"]["point"]["base_rate_control"]
    assert abs(exp_br["mae_skill"]) < 1e-12 and exp_br["spearman"] is None, exp_br
    invariants = gate2_invariants(model, calib, mu, sd, device, args.jobs)
    const_eval = evaluate_all(model, calib, mu, sd, device, constant_input=True)
    const_metrics = metrics_from_families(const_eval["families"], args.jobs, n_boot=0)

    shuffle_summary = {
        "definition": "trained with flat-shuffled labels (seed 0, fully exchangeable across "
                      "(decision, opponent, tile) slots); evaluated against calib_sub labels "
                      "shuffled the same way (A2 control convention)",
        "metrics": {name: {"auc": ctrl_metrics[name]["point"].get("auc"),
                           "brier_skill": ctrl_metrics[name]["point"].get("brier_skill"),
                           "average_precision": ctrl_metrics[name]["point"].get("average_precision"),
                           "prevalence": ctrl_metrics[name]["point"].get("prevalence"),
                           "mae_skill": ctrl_metrics[name]["point"].get("mae_skill"),
                           "spearman": ctrl_metrics[name]["point"].get("spearman")}
                    for name in ctrl_metrics},
        "gates": {
            "opp_tenpai_auc_within_0.02": abs(ctrl_metrics["opp_tenpai_prob"]["point"]["auc"] - 0.5) < 0.02,
            "deal_in_auc_within_0.02": abs(ctrl_metrics["deal_in"]["point"]["auc"] - 0.5) < 0.02,
            "wait_auc_within_0.02": abs(ctrl_metrics["wait_membership"]["point"]["auc"] - 0.5) < 0.02,
            "opp_tenpai_no_positive_bss": ctrl_metrics["opp_tenpai_prob"]["point"]["brier_skill"] < 0.01,
            "deal_in_no_positive_bss": ctrl_metrics["deal_in"]["point"]["brier_skill"] < 0.01,
            "wait_no_positive_bss": ctrl_metrics["wait_membership"]["point"]["brier_skill"] < 0.01,
            "exp_loss_no_positive_mae_skill": ctrl_metrics["exp_loss"]["point"]["mae_skill"] < 0.02,
            "exp_loss_spearman_within_0.02": abs(ctrl_metrics["exp_loss"]["point"]["spearman"]) < 0.02,
        },
        "two_sided_bss_within_0.01": {
            name: abs(ctrl_metrics[name]["point"]["brier_skill"]) < 0.01
            for name in ("opp_tenpai_prob", "deal_in", "wait_membership")},
        "bss_gate_form": "one-sided BSS<0.01 (no positive skill) + |AUC-0.5|<0.02 — t6 precedent: a "
                         "shuffled Brier score is reliability+uncertainty >= base, so a two-sided "
                         "|BSS|<0.01 gate is unattainable for any spread predictor (PREREG §5).",
        "checkpoint": {"path": _rel(ctrl_ckpt), "sha256": _sha256(ctrl_ckpt),
                       "weights_sha256": ctrl_weights_sha},
        "control_metrics_path": "runs/exp/EX1/headA_shuffle_metrics.json",
    }
    shuffle_summary["passed"] = all(shuffle_summary["gates"].values())

    metrics = {
        "schema": "ex1-headA-metrics/v1",
        "generated_at": _now(),
        "route": "A_public_features",
        "prereg": {"path": _rel(PREREG_PATH), "sha256_start": sha_start},
        "checkpoint": {"path": _rel(ckpt_path), "sha256": ckpt_sha,
                       "weights_sha256": weights_sha,
                       "bytes": ckpt_path.stat().st_size},
        "feature_spec": feature_spec(),
        "training": {
            "manifest": {"path": _rel(args.manifest), "sha256": manifest_sha,
                         "years": years, "n_games": n_games,
                         "bounds": list(TRAIN_BOUNDS), "seed": int(manifest["seed"])},
            "n_decisions_real": n_decisions,
            "n_decisions_all": int(train_meta["n_decisions_all"]),
            "exp_loss_sum": _f(train_t["exp_loss_sum"]),
            "deal_in_rows": int(train_t["ron_dec"].size),
            "opp_tenpai_positives": int(train_t["opp_tenpai"].sum()),
            "config": record,
            "smoke": bool(args.smoke),
        },
        "evaluation": {
            "subset": "calib_sub",
            "n_files": int(calib_meta["n_files"]),
            "n_decisions_real": int(calib_meta["n_decisions_real"]),
            "n_slots_opp_tenpai": int(calib_x.shape[0] * N_OPPS),
            "n_slots_tile": int(calib_x.shape[0] * N_OPPS * N_TILES),
            "families": families,
        },
        "negative_controls": {
            "trained_shuffle": shuffle_summary,
            "constant_input": {name: {k: v["point"].get(k) for k in
                                      ("auc", "brier_skill", "average_precision", "mae_skill",
                                       "spearman", "prevalence", "target_mean")}
                               for name, v in const_metrics.items()},
            "constant_input_note": "constant input (standardized mean) -> constant per output slot; "
                                   "AUC is not exactly 0.5 because the fixed per-slot biases weakly "
                                   "track per-slot prevalence. The exact constant control is "
                                   "`base_rate` (AUC exactly 0.5, BSS exactly 0).",
            "base_rate": {name: v["point"]["base_rate_control"] for name, v in families.items()},
        },
        "gate2_invariants": invariants,
        "triage_reference_calib_sub": triage_reference(families),
        "runtime": {"device": device, "seconds": round(time.time() - t_start, 1),
                    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024},
    }
    sha_end = assert_prereg_unchanged(PREREG_PATH, sha_start)
    metrics["prereg"]["sha256_end"] = sha_end
    metrics["prereg"]["immutable_during_run"] = sha_start == sha_end

    metrics_path = Path(args.metrics_out) if args.metrics_out else out_dir / "headA_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
                            encoding="utf-8")
    ctrl_metrics_path = out_dir / "headA_shuffle_metrics.json"
    ctrl_metrics_path.write_text(
        json.dumps({"schema": "ex1-headA-metrics/v1", "generated_at": _now(),
                    "route": "A_public_features_shuffle_control",
                    "prereg": {"path": _rel(PREREG_PATH), "sha256": sha_start},
                    "training": ctrl_record,
                    "evaluation": {"subset": "calib_sub", "families": ctrl_metrics},
                    "gate2_invariants": None}, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    print(f"[train] metrics -> {_rel(metrics_path)}")
    print(json.dumps({
        "opp_tenpai": {k: families["opp_tenpai_prob"]["point"][k]
                       for k in ("auc", "brier_skill", "ece_equal_width_15", "prevalence")},
        "deal_in": {k: families["deal_in"]["point"][k]
                    for k in ("average_precision", "brier_skill", "quantile_ece_15", "prevalence")},
        "wait_membership": {k: families["wait_membership"]["point"][k]
                            for k in ("average_precision", "brier_skill", "quantile_ece_15",
                                      "prevalence")},
        "exp_loss": {k: families["exp_loss"]["point"][k]
                     for k in ("mae_skill", "spearman", "value_binned_calibration_error_over_std")},
        "gate2_passed": invariants["passed"],
        "shuffle_control_passed": shuffle_summary["passed"],
    }, indent=2))
    if not shuffle_summary["passed"] and not args.smoke:
        raise SystemExit("FATAL: shuffle negative control failed (PREREG §9.7: STOP)")
    if not invariants["passed"] and not args.smoke:
        raise SystemExit("FATAL: spec gate-2 invariants failed (MA-6) — see headA_metrics.json")
    return 0


def run_eval(args) -> int:
    sha_start = load_prereg(PREREG_PATH)
    out_dir = Path(args.out_dir)
    device = _resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = HeadA()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    mu, sd = np.asarray(ckpt["mu"]), np.asarray(ckpt["sd"])
    calib_x, calib_t, calib_meta = _load_cache(out_dir, "calib_sub")
    calib = {"x": calib_x, "targets": calib_t}
    ev = evaluate_all(model, calib, mu, sd, device)
    families = metrics_from_families(ev["families"], args.jobs, n_boot=args.bootstrap)
    invariants = gate2_invariants(model, calib, mu, sd, device, args.jobs)
    payload = {"schema": "ex1-headA-metrics/v1", "generated_at": _now(),
               "route": "A_public_features", "checkpoint": {"path": _rel(args.checkpoint),
                                                            "sha256": _sha256(args.checkpoint)},
               "prereg": {"path": _rel(PREREG_PATH), "sha256_start": sha_start},
               "evaluation": {"subset": "calib_sub", "n_files": int(calib_meta["n_files"]),
                              "families": families},
               "gate2_invariants": invariants}
    out = Path(args.metrics_out) if args.metrics_out else out_dir / "headA_metrics.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    sha_end = assert_prereg_unchanged(PREREG_PATH, sha_start)
    print(json.dumps({"wrote": _rel(out), "prereg_unchanged": sha_start == sha_end,
                      "gate2_passed": invariants["passed"]}, indent=2))
    return 0


def run_compare(args) -> int:
    """Determinism check: metric blocks of two runs must be identical."""
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))

    def core(doc):
        shuffle = (doc.get("negative_controls") or {}).get("trained_shuffle") or {}
        return {"evaluation": doc.get("evaluation"), "gate2_invariants": doc.get("gate2_invariants"),
                "shuffle_metrics": shuffle.get("metrics"), "shuffle_gates": shuffle.get("gates"),
                "shuffle_weights_sha256": (shuffle.get("checkpoint") or {}).get("weights_sha256"),
                "weights_sha256": (doc.get("checkpoint") or {}).get("weights_sha256")}

    same = json.dumps(core(a), sort_keys=True) == json.dumps(core(b), sort_keys=True)
    print(json.dumps({"a": args.a, "b": args.b, "metric_blocks_identical": same}, indent=2))
    if not same:
        raise SystemExit("FATAL: same-seed rerun produced different metric blocks")
    return 0


# ------------------------------------------------------------------------------- selftest


def selftest() -> int:
    _register_stats()
    from exp_a2_probe import auc_score
    from exp_a2b_calibration import spearman, value_table

    # binary metrics
    assert auc_score([0, 1], [0.1, 0.9]) == 1.0
    assert abs(auc_score([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-12
    ap = average_precision(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.6]))
    assert abs(ap - (1.0 + 2 / 3) / 2) < 1e-12, ap
    # perfectly calibrated equal-width: p=k/1000, k positives of 1000
    ps, ys = [], []
    for k in range(1, 1000):
        ps.append(np.full(1000, k / 1000.0))
        ys.append(np.concatenate([np.ones(k), np.zeros(1000 - k)]))
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    from exp_a2_probe import ece_score
    assert ece_score(y, p, N_BINS) < 1e-6
    pbar = float(y.mean())
    want_bss = 1.0 - float((p * (1 - p)).mean()) / (pbar * (1 - pbar))
    assert abs(brier_skill(y, p) - want_bss) < 1e-9, (brier_skill(y, p), want_bss)
    # quantile ECE zero when every value is self-calibrated
    p2 = np.repeat(np.linspace(0.05, 0.95, 19), 500)
    y2 = np.concatenate([np.concatenate([np.ones(round(v * 500)), np.zeros(500 - round(v * 500))])
                         for v in np.linspace(0.05, 0.95, 19)])
    assert quantile_ece(y2, p2)["ece"] < 1e-9, quantile_ece(y2, p2)["ece"]
    # constant controls
    const = np.full_like(p, y.mean())
    assert abs(brier_skill(y, const)) < 1e-12
    assert abs(ece_score(y, const, N_BINS)) < 1e-12
    # continuous metrics
    assert abs(mae_skill(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))["mae_skill"] - 1.0) < 1e-12
    assert abs(mae_skill(np.array([0.0, 1.0, 2.0]),
                         np.full(3, 1.0))["mae_skill"]) < 1e-12
    assert abs(spearman(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) - 1.0) < 1e-12
    assert abs(value_table(np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0]))
               ["value_binned_calibration_error_pts"]) < 1e-12
    # bootstrap path: single group => every draw reproduces the point statistic
    fam = {"k": {"y": np.array([0.0, 1.0, 1.0, 0.0]), "p": np.array([0.1, 0.9, 0.8, 0.2]),
                 "groups": np.zeros(4, dtype=np.int64)}}
    ci = bootstrap_ci(fam, [("k", "auc")], n_boot=8, jobs=2)
    assert abs(ci[("k", "auc")]["ci95"][0] - 1.0) < 1e-12
    y3 = np.zeros(1000); y3[:5] = 1.0
    s3 = np.linspace(1, 0, 1000)
    assert abs(average_precision(y3, s3) - 1.0) < 1e-12
    assert abs(average_precision(y3, np.full(1000, 0.3)) - 0.005) < 1e-12
    print(json.dumps({"selftest": "PASS", "feature_dim": FEATURE_DIM,
                      "ap_known": ap, "bootstrap_single_group": ci[("k", "auc")]["ci95"]},
                     indent=2))
    return 0


# ----------------------------------------------------------------------------------- main


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="EX1 t14 route A public-feature head family")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="build the feature cache from t2 labels")
    p.add_argument("--subset", choices=["train", "calib_sub"], default="train")
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--out-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--jobs", type=int, default=min(12, os.cpu_count() or 1))
    p.add_argument("--limit-games", type=int, default=None)
    p.add_argument("--max-games", type=int, default=TRAIN_BOUNDS[1])
    p.add_argument("--shard-decisions", type=int, default=250_000)
    p.add_argument("--verify-sample", type=int, default=2)
    p.set_defaults(func=prepare, manifest_sha256=None)

    t = sub.add_parser("train", help="train + evaluate + write metrics")
    t.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    t.add_argument("--out-dir", type=Path, default=CACHE_DIR)
    t.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    t.add_argument("--seed", type=int, default=HEAD_SEED)
    t.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    t.add_argument("--batch", type=int, default=TRAIN_BS)
    t.add_argument("--jobs", type=int, default=min(12, os.cpu_count() or 1))
    t.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--metrics-out", type=Path, default=None)
    t.add_argument("--checkpoint-out", type=Path, default=None)
    t.set_defaults(func=run_train)

    e = sub.add_parser("eval", help="re-evaluate a checkpoint on calib_sub")
    e.add_argument("--checkpoint", type=Path, default=CACHE_DIR / "headA.pt")
    e.add_argument("--out-dir", type=Path, default=CACHE_DIR)
    e.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    e.add_argument("--jobs", type=int, default=min(12, os.cpu_count() or 1))
    e.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    e.add_argument("--metrics-out", type=Path, default=None)
    e.set_defaults(func=run_eval)

    v = sub.add_parser("verify-features", help="vectorized features vs t12/t13 providers")
    v.add_argument("--subset", choices=["train", "calib_sub"], default="train")
    v.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    v.add_argument("--max-files", type=int, default=2)
    v.set_defaults(func=_run_verify)

    c = sub.add_parser("compare", help="determinism check between two metrics JSONs")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=run_compare)

    s = sub.add_parser("selftest", help="synthetic metric/bootstrap checks")
    s.set_defaults(func=lambda a: selftest())
    return ap.parse_args(argv)


def _run_verify(args) -> int:
    args.manifest_sha256 = None
    jobs = _prepare_jobs(args)
    jobs = jobs[:args.max_files] if args.max_files else jobs
    result = verify_features(jobs, max_files=None)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
