#!/usr/bin/env python3
"""EX1 oracle label extraction (t2): the ONLY label source for the EX1 validity audit and
the t14/t15 head training.

Contract (frozen in `scripts/exp_ex1_prereg.md` §3/§8):
  * per-decision, per-opponent oracle decode of `shanten_j` / `waits_j` / `furiten_j` via
    `GameplayLoader(version=4, oracle=True)` + `take_invisible_obs()` (A2 layout constants and
    `exp_a2_probe.verify_and_label` verification gates, critical violations must be 0);
  * realized deal-in from the event stream: each `hora{actor=j, target=us, actor != target}` is
    paired with the immediately preceding `dahai.pai` (or `kakan.pai` for chankan); multi-ron
    counts once per `hora` event; tile index is 34 classes (aka folded);
  * continuous `exp_loss`: for `(decision, j, tile)` sum `-hora.deltas[us]` over the paired rons
    (positive = our loss), `0.0` elsewhere; missing `hora.deltas` fails loud (never a silent 0).

Opponent index `j` is relative seat order `(player_id + j + 1) % 4` = shimocha/toimen/kamicha,
for POV `player_id = SEAT_OF_SPLIT[split]` (a=0,b=1,c=2,d=3) — the same seat every engine
consumer resolves from the file's canonical name.

**NO-REVEAL**: this module is a validity/head-training label source only. It must never be
imported from `scripts/jiangpu/**` (t4 asserts the import graph); labels never enter `datasets/`.

Usage:
  python scripts/exp_ex1_labels.py train-manifest [--jobs N]   # -> runs/exp/EX1/train_manifest.json
  python scripts/exp_ex1_labels.py holdout [--jobs N]          # -> runs/exp/EX1/holdout_labels.json
  python scripts/exp_ex1_labels.py inspect PATH [--seat N]

allow: SIZE_OK — t2 mandates this exact single-file entry point; the t4 NO-REVEAL import-graph
guard matches the literal name `exp_ex1_labels`, so a split helper would evade it. The body is
one data pipeline (streams -> alignment -> oracle decode -> npz -> manifests/CLI).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "vendor/mortal/target/release"))
sys.path.insert(0, str(REPO / "vendor/mortal"))

from eval_report import SEAT_OF_SPLIT, SPLITS  # noqa: E402
from exp_ex1_audit import assert_prereg_unchanged, load_prereg  # noqa: E402

PREREG_PATH = REPO / "scripts/exp_ex1_prereg.md"
OUT_DIR = REPO / "runs/exp/EX1"
DATASETS_DIR = REPO / "datasets/extracted"
TRAIN_YEARS = tuple(range(2009, 2023))
TRAIN_SEED = 13
TRAIN_TARGET_GAMES = 20_000
TRAIN_BOUNDS = (10_000, 40_000)
TRAIN_SEED_BASE = 903_000_000
CANON_KEY = 0x455831
SUBSETS = ("calib_sub", "select_sub", "report_sub")
LABEL_SCHEMA = "ex1-labels/v1"

# --- mjai tile ids (identical to scripts/jiangpu/tiles.py; libriichi MJAI_PAI_STRINGS) ---
TILE_STRINGS = (
    [f"{n}m" for n in range(1, 10)]
    + [f"{n}p" for n in range(1, 10)]
    + [f"{n}s" for n in range(1, 10)]
    + ["E", "S", "W", "N", "P", "F", "C"]
    + ["5mr", "5pr", "5sr"]
)
TILE_ID = {s: i for i, s in enumerate(TILE_STRINGS)}
N_TILES = 34
N_OPPS = 3
OPP_STRIDE = 17
OPP_BLOCK = OPP_STRIDE * N_OPPS  # invisible rows 0..50: tehai/aka/shanten/waits/furiten only

# decision labels (gameplay.rs:377-445); >= 37 is not a tile
PASS_LABEL = 45
RYUKYOKU_LABEL = 44
KAN_LABEL = 42
RON_LABEL = 43
REACH_LABEL = 37
CHI_LABELS = frozenset((38, 39, 40))
PON_LABEL = 41
DISCARD_MAX = 36
ACTION_TYPES = frozenset(("dahai", "reach", "chi", "pon", "daiminkan", "kakan", "ankan", "hora"))
DISCARD_TYPES = ("dahai", "kakan")
BARRIERS = frozenset(("dahai", "kakan", "ankan", "daiminkan", "tsumo", "chi", "pon", "reach",
                      "start_kyoku", "end_kyoku", "ryukyoku"))


class RecordError(Exception):
    """A game record could not be decoded / labeled (missing file, bad log, bad hora)."""


@dataclass(frozen=True, slots=True)
class LabelIdentity:
    """Canonical `(seed, key, split)` of a label file, plus its source year."""

    seed: int | None = None
    key: int | None = None
    split: str | None = None
    year: int | None = None


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


# ------------------------------------------------------------------ event stream parsing


def tile_id(s: str) -> int:
    tid = TILE_ID.get(s)
    if tid is None:
        raise RecordError(f"unknown mjai tile string {s!r}")
    return tid


def deaka(tid: int) -> int:
    """Fold aka fives onto their base tile (34/35/36 -> 4/13/22), 34 classes."""
    return {34: 4, 35: 13, 36: 22}.get(tid, tid)


def _read_bytes(path) -> bytes:
    p = Path(path)
    if not p.is_file():
        raise RecordError(f"missing game file: {p}")
    try:
        return p.read_bytes()
    except OSError as exc:
        raise RecordError(f"cannot read {p}: {exc}") from exc


def _decode_text(data: bytes, path) -> str:
    try:
        return gzip.decompress(data).decode("utf-8") if data[:2] == b"\x1f\x8b" else data.decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError, ValueError) as exc:
        raise RecordError(f"cannot decode {path}: {exc}") from exc


def parse_events(text: str) -> list[dict]:
    events: list[dict] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecordError(f"line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(ev, dict) or not isinstance(ev.get("type"), str):
            raise RecordError(f"line {lineno}: not an mjai event object")
        events.append(ev)
    if not events:
        raise RecordError("empty game log")
    return events


def _actor(ev: dict, key: str, where: str) -> int:
    val = ev.get(key)
    if not isinstance(val, int) or not 0 <= val <= 3:
        raise RecordError(f"{where}: bad field {key}={val!r}")
    return val


def _deltas_field(ev: dict, where: str) -> list[int]:
    """Fail-loud deltas reader — same behaviour as scripts/analyze_alllast.py:113-117 (MINOR-1)."""
    val = ev.get("deltas")
    if not isinstance(val, list) or len(val) != 4 or not all(isinstance(x, int) for x in val):
        raise RecordError(f"{where}: bad deltas={val!r}")
    return list(val)


def start_names(events: list[dict]) -> list[str]:
    for ev in events:
        if ev["type"] == "start_game":
            names = ev.get("names")
            if isinstance(names, list) and len(names) == 4 and all(isinstance(n, str) for n in names):
                return list(names)
            raise RecordError(f"start_game: bad names={names!r}")
    raise RecordError("no start_game event")


def find_rons(events: list[dict]) -> list[dict]:
    """All ron events (actor != target) with their paired discard.

    Every `hora` is deltas-validated fail-loud. The paired discard is the nearest preceding
    `dahai`/`kakan`; an intervening barrier (ankan/daiminkan/...) is a malformed pairing.
    """
    rons: list[dict] = []
    for i, ev in enumerate(events):
        if ev["type"] != "hora":
            continue
        actor = _actor(ev, "actor", f"hora[{i}]")
        target = _actor(ev, "target", f"hora[{i}]")
        deltas = _deltas_field(ev, f"hora[{i}]")
        if actor == target:
            continue  # tsumo
        k = i - 1
        while k >= 0 and events[k]["type"] not in BARRIERS:
            k -= 1
        found = events[k]["type"] if k >= 0 else None
        if k < 0 or found not in DISCARD_TYPES:
            raise RecordError(f"hora[{i}]: no preceding dahai/kakan (found {found!r})")
        disc = events[k]
        if disc.get("actor") != target:
            raise RecordError(f"hora[{i}]: target={target} but discarder={disc.get('actor')!r}")
        if not (deltas[target] < 0 < deltas[actor]):
            raise RecordError(f"hora[{i}]: bad deltas sign {deltas} (actor={actor}, target={target})")
        rons.append({
            "hora_index": i,
            "discard_index": k,
            "actor": actor,
            "target": target,
            "tile": deaka(tile_id(disc["pai"])),
            "loss": -deltas[target],  # exp_loss = -deltas[us]; positive = our loss (PREREG §3)
        })
    return rons


# --------------------------------------------------------------- decision/event alignment


def align_decisions(actions: np.ndarray, events: list[dict], player_id: int) -> dict:
    """Map each libriichi decision index to the mjai action event it encodes.

    `actions` labels: 0..36 discard (aka 34..36; 37 riichi; 38..40 chi; 41 pon; 42 kan;
    43 ron; 44 ryukyoku; 45 pass). With `always_include_kan_select=True` every 42 is followed
    by exactly one kan-select pseudo-decision (tile 0..33) that consumes no event.
    """
    our = [(i, ev) for i, ev in enumerate(events)
           if ev.get("actor") == player_id and ev["type"] in ACTION_TYPES]
    n = int(actions.shape[0])
    dec_event = np.full(n, -1, dtype=np.int64)
    is_kan_select = np.zeros(n, dtype=np.uint8)
    kinds: dict[str, int] = {}
    j = 0
    i = 0
    while i < n:
        a = int(actions[i])
        if a in (PASS_LABEL, RYUKYOKU_LABEL):
            i += 1
            continue
        if j >= len(our):
            raise RecordError(f"decision {i} (action {a}) has no remaining {player_id}-event")
        ei, ev = our[j]
        et = ev["type"]
        if a <= DISCARD_MAX:
            if et != "dahai":
                raise RecordError(f"decision {i} action {a}: expected dahai, got {et} (event {ei})")
            if deaka(tile_id(ev["pai"])) != deaka(a):
                raise RecordError(f"decision {i} action {a}: discard mismatch {ev['pai']} (event {ei})")
        elif a == REACH_LABEL:
            if et != "reach":
                raise RecordError(f"decision {i} action 37: expected reach, got {et} (event {ei})")
        elif a in CHI_LABELS:
            if et != "chi":
                raise RecordError(f"decision {i} action {a}: expected chi, got {et} (event {ei})")
        elif a == PON_LABEL:
            if et != "pon":
                raise RecordError(f"decision {i} action 41: expected pon, got {et} (event {ei})")
        elif a == KAN_LABEL:
            if et not in ("kakan", "ankan", "daiminkan"):
                raise RecordError(f"decision {i} action 42: expected a kan, got {et} (event {ei})")
            if i + 1 >= n or int(actions[i + 1]) > DISCARD_MAX - 3:
                raise RecordError(f"decision {i} action 42: missing kan-select pseudo-decision")
            kan_tile = deaka(tile_id(ev["pai"])) if "pai" in ev else deaka(tile_id(ev["consumed"][0]))
            if int(actions[i + 1]) != kan_tile:
                raise RecordError(
                    f"decision {i} action 42: kan-select {int(actions[i + 1])} != tile {kan_tile}")
            is_kan_select[i + 1] = 1
            dec_event[i] = ei
            kinds[et] = kinds.get(et, 0) + 1
            i += 2
            j += 1
            continue
        elif a == RON_LABEL:
            if et != "hora":
                raise RecordError(f"decision {i} action 43: expected hora, got {et} (event {ei})")
        else:
            raise RecordError(f"decision {i}: unexpected action {a}")
        dec_event[i] = ei
        kinds[et] = kinds.get(et, 0) + 1
        i += 1
        j += 1
    if j != len(our):
        raise RecordError(f"{len(our) - j} of seat {player_id}'s events unclaimed by decisions")
    event_to_dec = {int(ev): k for k, ev in enumerate(dec_event) if ev >= 0}
    return {"dec_event": dec_event, "is_kan_select": is_kan_select,
            "event_to_dec": event_to_dec, "kinds": kinds}


# ------------------------------------------------------------------------ oracle decoding


def _bootstrap_mortal() -> None:
    """Pin the `mortal` namespace package to vendor/mortal, order-independent.

    `jiangpu/models.py:14-17` already puts `vendor/mortal/mortal` first on sys.path, where a
    `mortal.py` script shadows the namespace package; build the package module explicitly so
    `from mortal.model import ...` keeps working regardless of import order.
    """
    import importlib.machinery  # noqa: PLC0415
    import importlib.util  # noqa: PLC0415
    inner = str(REPO / "vendor/mortal/mortal")
    existing = sys.modules.get("mortal")
    if existing is not None and not hasattr(existing, "__path__"):
        del sys.modules["mortal"]
    if "mortal" not in sys.modules:
        spec = importlib.machinery.ModuleSpec("mortal", None, is_package=True)
        spec.submodule_search_locations = [inner]
        sys.modules["mortal"] = importlib.util.module_from_spec(spec)
    if inner not in sys.path:
        sys.path.append(inner)


def decode_oracle(obs: np.ndarray, inv: np.ndarray, own_sh: np.ndarray) -> dict:
    """Per-decision per-opponent oracle labels + A2 verification gates (critical == 0).

    Only `inv[:, :OPP_BLOCK]` is consumed: rows 51+ are the non-deterministic filler yama.
    """
    _bootstrap_mortal()
    from exp_a2_probe import (CRITICAL_COUNTERS, FURITEN_ROW, OPP_STRIDE, SHANTEN_ROWS,  # noqa: PLC0415
                              WAITS_ROW, verify_and_label)
    labels, counters = verify_and_label(obs, inv, own_sh)
    bad = {k: v for k, v in counters.items() if k in CRITICAL_COUNTERS and v > 0}
    if bad:
        raise RecordError(f"critical oracle decode violations: {bad}")

    n = inv.shape[0]
    blocks = inv[:, :OPP_BLOCK].reshape(n, N_OPPS, OPP_STRIDE, N_TILES)
    rowfill = (blocks[:, :, SHANTEN_ROWS[0]:SHANTEN_ROWS[1], :] == 1).all(axis=3)
    onehot_ok = rowfill.sum(axis=2) == 1
    shanten = np.where(onehot_ok, rowfill.argmax(axis=2), -1).astype(np.int8)
    waits = blocks[:, :, WAITS_ROW, :].astype(np.uint8)
    furiten = blocks[:, :, FURITEN_ROW, 0].astype(np.uint8)
    mism = int((waits.any(axis=2) != (shanten == 0)).sum())
    if mism != counters.get("waits_vs_shanten", -1):
        raise RecordError(f"waits/shanten cross-check drift: {mism} != {counters.get('waits_vs_shanten')}")
    if not np.array_equal(labels["opp_any_tenpai"], (shanten.min(axis=1) <= 0)):
        raise RecordError("opp_any_tenpai disagrees with per-opponent shanten decode")
    if counters.get("n_opp_obs") != N_OPPS * n:
        raise RecordError(f"n_opp_obs {counters.get('n_opp_obs')} != {N_OPPS * n}")
    return {"opp_shanten": shanten, "opp_waits": waits, "opp_furiten": furiten, "counters": counters}


# ------------------------------------------------------------------------- label file I/O


def sparse_ron_labels(rows, n_decisions: int) -> tuple[np.ndarray, np.ndarray]:
    """rows = iterable of `(decision, actor, tile, loss)` -> dense (n,3,34) deal_in / exp_loss."""
    deal_in = np.zeros((n_decisions, N_OPPS, N_TILES), dtype=np.float64)
    exp_loss = np.zeros((n_decisions, N_OPPS, N_TILES), dtype=np.float64)
    for dec, actor, tile, loss in rows:
        if not 0 <= dec < n_decisions or not 0 <= actor < N_OPPS or not 0 <= tile < N_TILES:
            raise RecordError(f"ron row out of range: {(dec, actor, tile, loss)} n={n_decisions}")
        if loss < 0:
            raise RecordError(f"ron row with negative loss: {(dec, actor, tile, loss)}")
        deal_in[dec, actor, tile] = 1.0
        exp_loss[dec, actor, tile] += float(loss)
    return deal_in, exp_loss


def load_labels(label_path) -> dict:
    """Raw compact label arrays + metadata (n_decisions == len(actions))."""
    with np.load(label_path) as z:
        raw = {k: z[k] for k in z.files if k not in ("schema", "meta_json")}
        raw["schema"] = str(z["schema"])
        raw["meta"] = json.loads(str(z["meta_json"]))
    return raw


def load_targets(label_path) -> dict:
    """Dense EX1 targets for one label file (PREREG §4 shapes, kan-select entries included)."""
    raw = load_labels(label_path)
    n = int(raw["actions"].shape[0])
    waits = np.unpackbits(raw["opp_waits_packed"], axis=-1, bitorder="little")[..., :N_TILES]
    tenpai = (raw["opp_shanten"] == 0)
    deal_in, exp_loss = sparse_ron_labels(
        zip(raw["ron_dec"], raw["ron_actor"], raw["ron_tile"], raw["ron_loss"]), n)
    out = dict(raw)
    out.update({
        "n": n,
        "opp_tenpai_prob": tenpai.astype(np.float64),
        "wait_membership": (waits * tenpai[..., None]).astype(np.float64),
        "deal_in": deal_in,
        "exp_loss": exp_loss,
        "real_mask": (raw["is_kan_select"] == 0),
    })
    return out


def label_file(path, seat: int, label_path, *, prereg_sha256: str,
               identity: LabelIdentity | None = None) -> dict:
    """Decode one game record into an `ex1-labels/v1` .npz for POV `seat`."""
    ident = identity or LabelIdentity()
    data = _read_bytes(path)
    text = _decode_text(data, path)
    events = parse_events(text)
    names = start_names(events)
    from libriichi.dataset import GameplayLoader  # noqa: PLC0415

    try:
        games = GameplayLoader(version=4, oracle=True, player_names=[names[seat]]).load_log(text)
    except RuntimeError as exc:
        raise RecordError(f"{path}: libriichi load failed: {exc}") from exc
    game = next((g for g in games if g.take_player_id() == seat), None)
    if game is None:  # duplicate names in start_game: fall back to the unfiltered build
        games = GameplayLoader(version=4, oracle=True).load_log(text)
        game = games[seat]
    actions = np.asarray(game.take_actions(), dtype=np.int64)
    # take_at_kyoku / take_at_turns are Rust Vec<u8> -> Python bytes
    at_kyoku = np.frombuffer(game.take_at_kyoku(), dtype=np.uint8)
    at_turns = np.frombuffer(game.take_at_turns(), dtype=np.uint8)
    own_shanten = np.asarray(game.take_shantens(), dtype=np.int8)
    n = int(actions.shape[0])
    if n:
        obs = np.stack([np.asarray(o) for o in game.take_obs()])
        inv = np.stack([np.asarray(o) for o in game.take_invisible_obs()])
        dec = decode_oracle(obs, inv, own_shanten)
        opp_shanten, opp_waits, opp_furiten = dec["opp_shanten"], dec["opp_waits"], dec["opp_furiten"]
        counters = dec["counters"]
    else:
        opp_shanten = np.zeros((0, N_OPPS), np.int8)
        opp_waits = np.zeros((0, N_OPPS, N_TILES), np.uint8)
        opp_furiten = np.zeros((0, N_OPPS), np.uint8)
        counters = {}

    al = align_decisions(actions, events, seat)
    rows = []
    for ron in find_rons(events):
        if ron["target"] != seat:
            continue
        dec_idx = al["event_to_dec"].get(ron["discard_index"])
        if dec_idx is None:
            raise RecordError(f"{path}: ron on seat {seat} (event {ron['hora_index']}) unmatched")
        j = (ron["actor"] - seat - 1) % 4  # PREREG §3 relative seat order, shimocha/toimen/kamicha
        if not 0 <= j < N_OPPS:
            raise RecordError(f"{path}: ron actor {ron['actor']} is not an opponent of seat {seat}")
        rows.append((dec_idx, j, ron["tile"], ron["loss"], ron["hora_index"]))
    rows.sort()
    ron_dec = np.asarray([r[0] for r in rows], dtype=np.int64)
    ron_actor = np.asarray([r[1] for r in rows], dtype=np.int8)
    ron_tile = np.asarray([r[2] for r in rows], dtype=np.int8)
    ron_loss = np.asarray([r[3] for r in rows], dtype=np.int64)
    ron_hora_ev = np.asarray([r[4] for r in rows], dtype=np.int64)
    source_sha256 = hashlib.sha256(data).hexdigest()
    meta = {
        "schema": LABEL_SCHEMA,
        "source_path": _rel(path),
        "source_sha256": source_sha256,
        "seed": ident.seed, "key": ident.key, "split": ident.split, "year": ident.year,
        "player_id": seat,
        "prereg_sha256": prereg_sha256,
        "n_decisions": n,
        "n_rons": int(ron_dec.shape[0]),
        "opp_layout": {"stride": OPP_STRIDE, "opps": N_OPPS, "block": OPP_BLOCK},
        "counters": {k: int(v) for k, v in counters.items()},
    }
    Path(label_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        label_path,
        schema=np.array(LABEL_SCHEMA),
        meta_json=np.array(json.dumps(meta, sort_keys=True)),
        actions=actions.astype(np.int16),
        at_kyoku=at_kyoku,
        at_turns=at_turns,
        own_shanten=own_shanten,
        opp_shanten=opp_shanten,
        opp_furiten=opp_furiten,
        opp_waits_packed=np.packbits(opp_waits, axis=-1, bitorder="little"),
        is_kan_select=al["is_kan_select"],
        ron_dec=ron_dec,
        ron_actor=ron_actor,
        ron_tile=ron_tile,
        ron_loss=ron_loss,
        ron_hora_ev=ron_hora_ev,
    )
    return {
        "source_path": _rel(path),
        "source_sha256": source_sha256,
        "seed": ident.seed, "key": ident.key, "split": ident.split, "year": ident.year,
        "player_id": seat,
        "label_path": _rel(label_path),
        "n_decisions": n,
        "n_rons": int(ron_dec.shape[0]),
        "n_opp_tenpai": int((opp_shanten == 0).sum()),
        "exp_loss_sum": int(ron_loss.sum()),
        "counters": {k: int(v) for k, v in counters.items()},
    }


# --------------------------------------------------------------------- batch entry point


def default_identity(index: int) -> tuple[int, int, str]:
    """Deterministic (seed, key, split): 4 consecutive picks share a seed, split rotates."""
    return TRAIN_SEED_BASE + index // 4, CANON_KEY, SPLITS[index % len(SPLITS)]


def _worker_init() -> None:  # pragma: no cover - pool plumbing
    os.environ.setdefault("RAYON_NUM_THREADS", "1")


def _label_job(job: dict, *, prereg_sha256: str, allow_errors: bool) -> dict:
    base = {k: v for k, v in job.items() if k != "identity"}  # keep results JSON-serializable
    try:
        result = label_file(job["source_path"], int(job["seat"]), job["label_path"],
                            prereg_sha256=prereg_sha256, identity=job["identity"])
        return {**base, "status": "ok", **result}
    except RecordError as exc:
        if not allow_errors:
            raise
        return {**base, "status": "error", "error": str(exc)}


def extract_labels(paths, out_dir, *, identities=None, subset: str = "train", jobs: int = 1,
                   limit=None, prereg_path=PREREG_PATH, allow_errors: bool = False,
                   progress_every: int = 500) -> dict:
    """Decode `paths` into `out_dir/labels/<subset>/<seed>_<key>_<split>.npz` label files.

    `identities` (per path, `(seed, key, split)`; optional `year` mapping via each path) pins the
    label names to the caller's manifest — t3's `corpus_index.json` for the holdout subsets, the
    train manifest for training. Without it, `default_identity(i)` is used.
    """
    all_paths = [Path(p) for p in paths]
    identities = [default_identity(i) for i in range(len(all_paths))] if identities is None \
        else list(identities)
    if len(identities) != len(all_paths):
        raise RecordError(f"identities {len(identities)} != paths {len(all_paths)}")
    if limit is not None:
        all_paths, identities = all_paths[:limit], identities[:limit]
    out_dir = Path(out_dir).resolve()
    label_dir = out_dir / "labels" / subset
    label_dir.mkdir(parents=True, exist_ok=True)
    sha = load_prereg(Path(prereg_path))

    jobs_list = []
    for p, ident in zip(all_paths, identities):
        seed, key, split = int(ident[0]), int(ident[1]), str(ident[2])
        if split not in SEAT_OF_SPLIT:
            raise RecordError(f"identity {ident!r}: split has no SEAT_OF_SPLIT mapping")
        year = ident[3] if len(ident) > 3 else next(
            (int(part) for part in p.parts if part.isdigit() and len(part) == 4), None)
        jobs_list.append({
            "source_path": str(p), "label_path": str(label_dir / f"{seed}_{key}_{split}.npz"),
            "seat": SEAT_OF_SPLIT[split],
            "identity": LabelIdentity(seed=seed, key=key, split=split, year=year),
        })
    worker = partial(_label_job, prereg_sha256=sha, allow_errors=allow_errors)
    results: list[dict] = []
    if jobs <= 1:
        for i, job in enumerate(jobs_list, 1):
            results.append(worker(job))
            if progress_every and i % progress_every == 0:
                print(f"[extract {subset}] {i}/{len(jobs_list)} files", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init) as ex:
            for i, res in enumerate(ex.map(worker, jobs_list, chunksize=4), 1):
                results.append(res)
                if progress_every and i % progress_every == 0:
                    print(f"[extract {subset}] {i}/{len(jobs_list)} files", flush=True)
    sha_end = assert_prereg_unchanged(Path(prereg_path), sha)
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] != "ok"]
    return {
        "subset": subset,
        "label_dir": _rel(label_dir),
        "prereg": {"path": _rel(prereg_path), "sha256_start": sha, "sha256_end": sha_end,
                   "immutable_during_run": sha == sha_end},
        "n_files": len(jobs_list),
        "n_ok": len(ok),
        "n_error": len(errors),
        "n_decisions": int(sum(r["n_decisions"] for r in ok)),
        "n_rons": int(sum(r["n_rons"] for r in ok)),
        "n_opp_tenpai": int(sum(r["n_opp_tenpai"] for r in ok)),
        "exp_loss_sum": int(sum(r["exp_loss_sum"] for r in ok)),
        "errors": [{"source_path": r["source_path"], "error": r["error"]} for r in errors],
        "files": ok,
    }


# ------------------------------------------------------------------------- CLI workflows


def enumerate_train_pool(datasets_dir=DATASETS_DIR) -> tuple[list[str], dict[str, int]]:
    pool: list[str] = []
    by_year: dict[str, int] = {}
    for year in TRAIN_YEARS:
        found = sorted(str(p) for p in Path(datasets_dir).joinpath(str(year)).glob("**/*.mjson"))
        if not found:
            raise RecordError(f"no .mjson under {datasets_dir}/{year} (training pool incomplete)")
        pool.extend(found)
        by_year[str(year)] = len(found)
    pool.sort()
    return pool, by_year


def build_train_manifest(out_dir=OUT_DIR, *, datasets_dir=DATASETS_DIR, seed: int = TRAIN_SEED,
                         target: int = TRAIN_TARGET_GAMES, jobs: int = 1, limit=None) -> dict:
    """Fixed-seed reservoir sample of the 2009-2022 pool + label extraction (PREREG §2)."""
    if seed < 0:
        raise RecordError("seed must be non-negative (PREREG pins 13)")
    lo, hi = TRAIN_BOUNDS
    if not lo <= target <= hi:
        raise RecordError(f"target games {target} outside the frozen bounds {TRAIN_BOUNDS}")
    _bootstrap_mortal()
    from exp_a2_probe import reservoir_sample  # noqa: PLC0415

    pool, by_year = enumerate_train_pool(datasets_dir)
    n_pick = min(target, len(pool))
    picked = reservoir_sample(pool, n_pick, seed)
    identities = [default_identity(i) for i in range(len(picked))]
    if limit is not None:
        picked, identities = picked[:limit], identities[:limit]
    summary = extract_labels(picked, out_dir, identities=identities, subset="train", jobs=jobs,
                             progress_every=200)
    manifest = {
        "schema": "ex1-train-manifest/v1",
        "seed": seed,
        "target_games": target,
        "bounds": list(TRAIN_BOUNDS),
        "years": list(TRAIN_YEARS),
        "pool": {"n_files": len(pool), "by_year": by_year},
        "identity_rule": f"seed = {TRAIN_SEED_BASE} + i//4, key = {CANON_KEY}, split = a/b/c/d by i%4",
        "pov_seat_rule": "player_id = SEAT_OF_SPLIT[split] (a=0,b=1,c=2,d=3)",
        "label_dir": summary["label_dir"],
        "labels_schema": LABEL_SCHEMA,
        "notes": [
            "labels include one kan-select pseudo-decision after every action 42 (is_kan_select)",
            "deal_in/exp_loss are sparse realized ron labels; opp_shanten -1 = undecodable",
            "this manifest is the single training sample for both head routes (t14/t15)",
        ],
        "prereg": summary["prereg"],
        "counts": {k: summary[k] for k in ("n_files", "n_ok", "n_error", "n_decisions", "n_rons",
                                           "n_opp_tenpai", "exp_loss_sum")},
        "errors": summary["errors"],
        "files": summary["files"],
    }
    out_path = Path(out_dir) / "train_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[train-manifest] {summary['n_ok']} games -> {_rel(out_path)} "
          f"({summary['n_decisions']} decisions, {summary['n_rons']} rons)")
    return manifest


def extract_holdout(out_dir=OUT_DIR, *, corpus_index_path=None, jobs: int = 1, limit=None) -> dict:
    """Label the three frozen holdout subsets, pinned to t3's `corpus_index.json`."""
    index_path = Path(corpus_index_path) if corpus_index_path else Path(OUT_DIR) / "corpus_index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecordError(f"cannot read corpus index {index_path}: {exc}") from exc
    if index.get("schema") != "ex1-corpus-index/v1":
        raise RecordError(f"unexpected corpus index schema {index.get('schema')!r}")
    subsets: dict[str, dict] = {}
    for name in SUBSETS:
        block = index["subsets"][name]
        entries = block["files"]
        paths = [REPO / f["target_path"] for f in entries]
        identities = [(f["seed"], f["key"], f["split"], f["year"]) for f in entries]
        summary = extract_labels(paths, out_dir, identities=identities, subset=name, jobs=jobs,
                                 limit=limit, progress_every=200)
        want_by_ident = {(f["seed"], f["key"], f["split"]): f["target_sha256"] for f in entries}
        if len(want_by_ident) != len(entries):
            raise RecordError(f"{name}: duplicate (seed,key,split) in corpus index")
        for r in summary["files"]:
            want = want_by_ident[(r["seed"], r["key"], r["split"])]
            if r["source_sha256"] != want:
                raise RecordError(f"{name}: {r['source_path']} sha256 {r['source_sha256']} != "
                                  f"corpus target {want}")
        subsets[name] = {"count": summary["n_ok"], "target_paths": len(entries),
                         "label_dir": summary["label_dir"], "n_decisions": summary["n_decisions"],
                         "n_rons": summary["n_rons"], "prereg_sha256": summary["prereg"]["sha256_start"]}
    out_path = Path(out_dir) / "holdout_labels.json"
    payload = {"schema": "ex1-holdout-labels/v1", "corpus_index": _rel(index_path),
               "label_root": _rel(Path(out_dir) / "labels"), "subsets": subsets}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[holdout] {sum(s['count'] for s in subsets.values())} games labelled -> {_rel(out_path)}")
    return payload


def inspect(path, seat: int) -> dict:
    """Decode one record for one seat and print a label summary (debug/QA helper)."""
    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "x.npz"
        info = label_file(path, seat, out, prereg_sha256="inspect-only")
        targets = load_targets(out)
    info["opp_tenpai_prob_sum"] = float(targets["opp_tenpai_prob"].sum())
    info["wait_membership_sum"] = float(targets["wait_membership"].sum())
    info["deal_in_sum"] = float(targets["deal_in"].sum())
    info["exp_loss_sum_dense"] = float(targets["exp_loss"].sum())
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EX1 oracle label extraction (t2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("train-manifest", help="sample 2009-2022 + write train labels")
    p1.add_argument("--jobs", type=int, default=1)
    p1.add_argument("--target", type=int, default=TRAIN_TARGET_GAMES)
    p1.add_argument("--limit", type=int, default=None)
    p1.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    p2 = sub.add_parser("holdout", help="label calib/select/report subsets from corpus_index.json")
    p2.add_argument("--jobs", type=int, default=1)
    p2.add_argument("--limit", type=int, default=None)
    p2.add_argument("--corpus-index", type=Path, default=None)
    p3 = sub.add_parser("inspect", help="decode one record for one seat")
    p3.add_argument("path", type=str)
    p3.add_argument("--seat", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "train-manifest":
            manifest = build_train_manifest(datasets_dir=args.datasets_dir, target=args.target,
                                            jobs=args.jobs, limit=args.limit)
            print(json.dumps(manifest["counts"], indent=2))
        elif args.cmd == "holdout":
            print(json.dumps(extract_holdout(corpus_index_path=args.corpus_index, jobs=args.jobs,
                                             limit=args.limit)["subsets"], indent=2))
        else:
            path = Path(args.path)
            seat = args.seat
            if seat is None:
                from eval_report import FILE_RE  # noqa: PLC0415
                match = FILE_RE.match(path.name)
                if match is None:
                    raise SystemExit(f"FATAL: {path.name} has no <seed>_<key>_<split> name; pass --seat")
                seat = SEAT_OF_SPLIT[match.group("split")]
            print(json.dumps(inspect(path, seat), indent=2, sort_keys=True))
    except RecordError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
