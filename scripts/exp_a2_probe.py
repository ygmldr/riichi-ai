#!/usr/bin/env python3
"""A2: linear probe on the frozen 1024-d representation.

Question (see runs/exp/A2/PREREG.md): does the frozen 1024-d encoder output of the
pinned policy network linearly encode opponents' hidden state (concealed hand / shanten),
which is NOT part of the network input?

Pipeline:
  1. reservoir-sample game files, load them with libriichi GameplayLoader(version=4,
     oracle=True);
  2. decode opponent labels from the oracle tensors (invisible.rs:163-209) with the
     critical decode-verification gates from the PREREG;
  3. compute phi = brain.encoder(obs) (model.py:156) with the frozen pinned Brain;
  4. fit a logistic regression on a file-level 80/20 split, report AUC / ECE / Brier,
     bootstrap CI, and a permuted-label control (gate |auc_random - 0.5| < 0.02).

Products: runs/exp/A2/results.json (+ README.md written separately).
No training of the Brain, no fine-tuning, no config edits.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
PINNED = REPO / "runs/exp/PINNED/mortal.pth.at-175200"
DATASET_GLOB = str(REPO / "datasets/extracted/**/*.mjson")
OUT_DIR = REPO / "runs/exp/A2"

sys.path.insert(0, str(REPO / "vendor/mortal/target/release"))
sys.path.insert(0, str(REPO / "vendor/mortal"))

from libriichi.dataset import GameplayLoader  # noqa: E402
from mortal.model import Brain  # noqa: E402

# --- oracle-tensor layout (verified against libriichi/src/dataset/invisible.rs:163-209) ---
OPP_STRIDE = 17
N_OPPS = 3
OPP_BLOCK = OPP_STRIDE * N_OPPS  # 51
SHANTEN_ROWS = (7, 14)  # one-hot rows [7,14)
SHANTEN_NORM_ROW = 14
HAND_COUNT_ROWS = (0, 4)
AKA_ROWS = (4, 7)
WAITS_ROW = 15
FURITEN_ROW = 16

# --- public obs planes (runs/analysis/obs_layout.json, obs_repr.rs:303-329/862) ---
FU_OFFSET = 771  # fuuro_overview shimocha/toimen/kamicha, 20 planes each
AK_OFFSET = 832  # ankan_overview shimocha/toimen/kamicha, 1 plane each
OWN_SHANTEN_OFF = 862  # own shanten one-hot, 7 planes

PRIMARY = "opp_any_tenpai"
POSITIVE_CONTROL = "ctrl_opp_any_meld_visible"
HIDDEN_LABELS = ("opp_any_tenpai", "opp_any_shanten_le1", "opp_any_shanten_le2", "opp_any_aka_in_hand")

CRITICAL_COUNTERS = (
    "shanten_onehot_rows",
    "shanten_norm_mismatch",
    "hand_count_range",
    "hand_count_nonint",
    "aka_nonbinary",
    "waits_nonbinary",
    "furiten_nonbinary",
    "own_shanten_onehot_rows",
    "own_shanten_mismatch",
)


# --------------------------------------------------------------------------- metrics


def auc_score(y_true, score) -> float:
    """Mann-Whitney AUC with tie-averaged ranks."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    uniq, inv, counts = np.unique(ss, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank = (cum - counts + 1 + cum) / 2.0  # 1-based average rank per unique value
    ranks_sorted = avg_rank[inv]
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece_score(y_true, prob, n_bins: int = 15) -> float:
    """Standard equal-width ECE: sum_b (n_b/N) |acc_b - conf_b|."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        ece += nb / len(y) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def brier_score(y_true, prob) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prob, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def bootstrap_auc_ci(y_true, score, n_boot: int, seed: int):
    y = np.asarray(y_true)
    s = np.asarray(score)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = auc_score(y[idx], s[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------------- decode + labels


def verify_and_label(obs: np.ndarray, inv: np.ndarray, own_sh: np.ndarray):
    """Decode opponent labels from oracle rows and run the PREREG verification gates.

    obs: (n, 1012, 34), inv: (n, 217, 34), own_sh: (n,) i8.
    Returns (labels: dict[str, np.ndarray], counters: dict[str, int]).
    """
    n = obs.shape[0]
    c: dict[str, int] = defaultdict(int)

    blocks = inv[:, :OPP_BLOCK].reshape(n, N_OPPS, OPP_STRIDE, 34)
    counts = blocks[:, :, HAND_COUNT_ROWS[0]:HAND_COUNT_ROWS[1], :].sum(axis=2)  # (n,3,34)
    hand_size = counts.sum(axis=2)  # (n,3)

    rowfill = (blocks[:, :, SHANTEN_ROWS[0]:SHANTEN_ROWS[1], :] == 1).all(axis=3)  # (n,3,7)
    n_rows = rowfill.sum(axis=2)
    onehot_ok = n_rows == 1
    c["shanten_onehot_rows"] += int((~onehot_ok).sum())
    shanten = np.where(onehot_ok, rowfill.argmax(axis=2), -1)

    norm = blocks[:, :, SHANTEN_NORM_ROW, 0]
    c["shanten_norm_mismatch"] += int(
        (onehot_ok & (np.abs(norm - shanten / 6.0) > 1e-6)).sum()
    )

    c["hand_count_range"] += int(((counts > 4) | (counts < 0)).sum())
    c["hand_count_nonint"] += int((np.abs(counts - np.rint(counts)) > 1e-6).sum())

    aka = blocks[:, :, AKA_ROWS[0]:AKA_ROWS[1], :]  # (n,3,3,34)
    c["aka_nonbinary"] += int(((aka != 0) & (aka != 1)).sum())
    waits = blocks[:, :, WAITS_ROW, :]  # (n,3,34)
    furi = blocks[:, :, FURITEN_ROW, 0]  # (n,3)
    c["waits_nonbinary"] += int(((waits != 0) & (waits != 1)).sum())
    c["furiten_nonbinary"] += int(((furi != 0) & (furi != 1)).sum())
    c["waits_vs_shanten"] += int((waits.any(axis=2) != (shanten == 0)).sum())
    c["n_opp_obs"] += 3 * n

    # hand size vs public melds: expected = 13 - 3*fuuro - 4*ankan (+0/+1 draw flag)
    fuuro = obs[:, FU_OFFSET:FU_OFFSET + 60].reshape(n, 3, 4, 5, 34)
    n_fuuro = (fuuro[:, :, :, :4, :] != 0).any(axis=(3, 4)).sum(axis=2)  # (n,3)
    n_ankan = (obs[:, AK_OFFSET:AK_OFFSET + 3] != 0).sum(axis=2)  # (n,3)
    expected = 13 - 3 * n_fuuro - 4 * n_ankan
    c["handsize_vs_melds"] += int(((hand_size != expected) & (hand_size != expected + 1)).sum())

    # own shanten decode (public obs planes) must equal take_shantens()
    own_oh = obs[:, OWN_SHANTEN_OFF:OWN_SHANTEN_OFF + 7]
    orf = (own_oh == 1).all(axis=2)
    ovalid = orf.sum(axis=1) == 1
    c["own_shanten_onehot_rows"] += int((~ovalid).sum())
    own_dec = np.where(ovalid, orf.argmax(axis=1), -1)
    own_ref = np.minimum(np.maximum(own_sh.astype(np.int16), 0), 6)
    c["own_shanten_mismatch"] += int((own_dec != own_ref).sum())

    min_sh = shanten.min(axis=1)
    labels = {
        "opp_any_tenpai": (min_sh <= 0),
        "opp_any_shanten_le1": (min_sh <= 1),
        "opp_any_shanten_le2": (min_sh <= 2),
        "opp_any_aka_in_hand": (aka != 0).any(axis=(1, 2, 3)),
        "ctrl_opp_any_meld_visible": (hand_size < 13).any(axis=1),
    }
    return {k: v.astype(np.uint8) for k, v in labels.items()}, dict(c)


# -------------------------------------------------------------------------- model


def load_frozen_brain(device: str):
    state = torch.load(PINNED, map_location="cpu", weights_only=True)
    cfg = state["config"]
    brain = Brain(version=cfg["control"]["version"], **cfg["resnet"])
    missing, unexpected = brain.load_state_dict(state["mortal"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
    in_ch = brain.encoder.net[0].weight.shape[1]
    if in_ch != 1012:
        raise RuntimeError(f"unexpected encoder input channels {in_ch} (expected 1012, non-oracle)")
    brain.eval().to(device)
    for p in brain.parameters():
        p.requires_grad_(False)
    return brain, state["steps"]


def embed(brain, arr: np.ndarray, chunk: int, device: str) -> np.ndarray:
    out = []
    with torch.inference_mode():
        for i in range(0, arr.shape[0], chunk):
            x = torch.from_numpy(np.ascontiguousarray(arr[i:i + chunk])).to(device, non_blocking=True)
            out.append(brain.encoder(x).float().cpu().numpy())
            del x
    return np.concatenate(out, axis=0)


# ----------------------------------------------------------------------- sampling


def reservoir_sample(paths, k: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    seen = 0
    res: list[str] = []
    for p in paths:
        seen += 1
        if len(res) < k:
            res.append(p)
        else:
            j = rng.randrange(seen)
            if j < k:
                res[j] = p
    res.sort()
    return res


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


# ---------------------------------------------------------------------- extraction


def extract(args, brain, files, loader, device):
    emb_parts: list[np.ndarray] = []
    name_list = list(HIDDEN_LABELS) + [POSITIVE_CONTROL]
    label_parts: dict[str, list[np.ndarray]] = {k: [] for k in name_list}
    own_sh_parts: list[np.ndarray] = []
    file_id_parts: list[np.ndarray] = []
    pov_parts: list[np.ndarray] = []
    counters: dict[str, int] = defaultdict(int)
    n_points = 0
    per_file = []
    t0 = time.time()

    for bstart in range(0, len(files), args.file_batch):
        batch = files[bstart:bstart + args.file_batch]
        for off, games in enumerate(loader.load_gz_log_files(batch)):
            fid = bstart + off
            n_before = n_points
            for g in games:
                pov = int(g.take_player_id())
                obs_l = g.take_obs()
                inv_l = g.take_invisible_obs()
                sh_l = g.take_shantens()
                if not (len(obs_l) == len(inv_l) == len(sh_l)):
                    raise RuntimeError(f"length mismatch file={batch[off]} pov={pov}")
                if not obs_l:
                    continue
                arr = np.stack([np.asarray(o) for o in obs_l])
                inv = np.stack([np.asarray(o) for o in inv_l])
                own_sh = np.asarray(sh_l, dtype=np.int8)

                labels, c = verify_and_label(arr, inv, own_sh)
                for k, v in c.items():
                    counters[k] += v

                phi = embed(brain, arr, args.chunk, device)
                emb_parts.append(phi)
                for k in name_list:
                    label_parts[k].append(labels[k])
                own_sh_parts.append(own_sh)
                file_id_parts.append(np.full(len(obs_l), fid, dtype=np.int32))
                pov_parts.append(np.full(len(obs_l), pov, dtype=np.uint8))
                n_points += len(obs_l)
                del arr, inv, own_sh, obs_l, inv_l, sh_l, labels, phi
            per_file.append(n_points - n_before)
        print(
            f"[extract] files {min(bstart + args.file_batch, len(files))}/{len(files)} "
            f"points {n_points} elapsed {time.time() - t0:.0f}s MemAvailable {mem_available_gb():.1f} GB",
            flush=True,
        )
        if mem_available_gb() < 4.0:
            print("[extract] WARNING: MemAvailable < 4 GB", flush=True)

    X = np.concatenate(emb_parts, axis=0) if emb_parts else np.zeros((0, 1024), np.float32)
    labels = {k: np.concatenate(v) for k, v in label_parts.items() if v}
    own_sh = np.concatenate(own_sh_parts) if own_sh_parts else np.zeros(0, np.int8)
    file_ids = np.concatenate(file_id_parts) if file_id_parts else np.zeros(0, np.int32)
    povs = np.concatenate(pov_parts) if pov_parts else np.zeros(0, np.uint8)
    return X, labels, own_sh, file_ids, povs, dict(counters), per_file


# ------------------------------------------------------------------------- fitting


def fit_predict(Xtr, ytr, Xva, args, device, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = nn.Linear(Xtr.shape[1], 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    model.train()
    for _ in range(args.epochs):
        perm = rng.permutation(n)
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            xb = torch.from_numpy(Xtr[idx]).to(device, non_blocking=True)
            yb = torch.from_numpy(ytr[idx]).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(Xva).to(device)).squeeze(1).float().cpu().numpy()
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)


def fit_and_metric(Xtr, ytr, Xva, yva, args, device) -> tuple[dict, np.ndarray]:
    p = fit_predict(Xtr, ytr, Xva, args, device, args.seed)
    out = {
        "auc": auc_score(yva, p),
        "ece": ece_score(yva, p, args.ece_bins),
        "brier": brier_score(yva, p),
        "prevalence_train": float(ytr.mean()),
        "prevalence_val": float(yva.mean()),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_pos_val": int(yva.sum()),
    }
    return out, p


def evaluate_label(y, train_mask, val_mask, Xtr, Xva, args, device, control_seeds) -> tuple[dict, np.ndarray]:
    """Probe + permuted-label controls; returns (metrics dict, val probabilities of the real fit)."""
    ytr, yva = y[train_mask], y[val_mask]
    out, p = fit_and_metric(Xtr, ytr, Xva, yva, args, device)
    ctrl_aucs = []
    for cs in control_seeds:
        yperm = y[np.random.default_rng(cs).permutation(len(y))]
        cres, _ = fit_and_metric(Xtr, yperm[train_mask], Xva, yperm[val_mask], args, device)
        ctrl_aucs.append(cres["auc"])
    out["auc_random_per_seed"] = [float(a) for a in ctrl_aucs]
    out["auc_random"] = float(np.mean(ctrl_aucs))
    out["auc_random_seeds"] = list(control_seeds)
    return out, p


# ------------------------------------------------------------------------- selftest


def selftest(args, brain, loader, device) -> None:
    files = sorted(glob.glob(DATASET_GLOB, recursive=True))[:2]
    assert files, "no dataset files"
    X, labels, own_sh, file_ids, povs, counters, per_file = extract(args, brain, files, loader, device)
    assert X.shape[1] == 1024, X.shape
    assert X.shape[0] == len(own_sh) == len(file_ids), X.shape
    for k, v in labels.items():
        assert len(v) == X.shape[0], (k, len(v), X.shape)
    bad = {k: v for k, v in counters.items() if k in CRITICAL_COUNTERS and v > 0}
    assert not bad, f"critical decode violations: {bad}"
    rate = counters["handsize_vs_melds"] / max(1, counters["n_opp_obs"])
    assert rate <= 1e-4, f"handsize_vs_melds rate {rate}"
    # metric unit checks
    assert auc_score([0, 1], [0.1, 0.9]) == 1.0
    assert auc_score([0, 1], [0.9, 0.1]) == 0.0
    assert auc_score([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert abs(auc_score([0, 1, 1, 0], [0.2, 0.8, 0.7, 0.3]) - 1.0) < 1e-12
    assert abs(ece_score([0, 1], [0.0, 1.0])) < 1e-12
    assert abs(brier_score([0, 1], [0.0, 1.0])) < 1e-12
    # tiny fit path (small batch so it actually converges on 2k points)
    rng = np.random.default_rng(0)
    Xt = rng.normal(size=(2000, 1024)).astype(np.float32)
    yt = (Xt[:, 0] + rng.normal(scale=0.1, size=2000) > 0).astype(np.float32)
    bs_backup = args.bs
    args.bs = 256
    p = fit_predict(Xt, yt, Xt[:1000], args, device, 0)
    args.bs = bs_backup
    assert auc_score(yt[:1000], p) > 0.9, auc_score(yt[:1000], p)
    print(
        f"[selftest] PASS points={X.shape[0]} files={len(files)} "
        f"counters={counters} per_file={per_file}",
        flush=True,
    )


# ----------------------------------------------------------------------------- main


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-files", type=int, default=400, help="reservoir sample size (seed 13)")
    ap.add_argument("--min-points", type=int, default=200_000)
    ap.add_argument("--fallback-k", type=int, default=100)
    ap.add_argument("--fallback-seed-start", type=int, default=14)
    ap.add_argument("--fallback-max", type=int, default=6)
    ap.add_argument("--file-batch", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=2048, help="embedding GPU batch size")
    ap.add_argument("--bs", type=int, default=4096, help="logreg mini-batch")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=13)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--control-seeds", type=int, nargs="+", default=[100, 101, 102])
    ap.add_argument("--ece-bins", type=int, default=15)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=7)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()

    if args.device == "auto":
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            device = "cuda" if free > 1.5e9 else "cpu"
        else:
            device = "cpu"
    else:
        device = args.device
    brain, ckpt_steps = load_frozen_brain(device)
    loader = GameplayLoader(4, oracle=True)
    print(f"[setup] device={device} ckpt_steps={ckpt_steps} args={vars(args)}", flush=True)

    if args.selftest:
        selftest(args, brain, loader, device)
        return 0

    # --- file sampling -------------------------------------------------------
    files = reservoir_sample(
        glob.iglob(DATASET_GLOB, recursive=True), args.n_files, args.split_seed
    )
    print(f"[sample] reservoir-sampled {len(files)} files (seed {args.split_seed})", flush=True)

    X, labels, own_sh, file_ids, povs, counters, per_file = extract(
        args, brain, files, loader, device
    )

    # --- fallback draws until min-points ------------------------------------
    seed = args.fallback_seed_start
    while len(X) < args.min_points and seed < args.fallback_seed_start + args.fallback_max:
        extra = reservoir_sample(glob.iglob(DATASET_GLOB, recursive=True), args.fallback_k, seed)
        print(f"[sample] fallback draw k={args.fallback_k} seed={seed}: {len(extra)} files", flush=True)
        X2, lab2, sh2, fid2, pov2, c2, pf2 = extract(args, brain, extra, loader, device)
        offset = len(files)
        files += extra
        X = np.concatenate([X, X2], axis=0)
        for k in labels:
            labels[k] = np.concatenate([labels[k], lab2[k]])
        own_sh = np.concatenate([own_sh, sh2])
        file_ids = np.concatenate([file_ids, fid2 + offset])
        povs = np.concatenate([povs, pov2])
        per_file += pf2
        for k, v in c2.items():
            counters[k] += v
        del X2, lab2, sh2, fid2, pov2
        seed += 1

    n = len(X)
    print(f"[extract] done: {n} decision points from {len(files)} files", flush=True)
    if n < args.min_points:
        raise SystemExit(f"FATAL: only {n} decision points (< {args.min_points})")

    # --- decode verification / degradation -----------------------------------
    degraded = any(counters.get(k, 0) > 0 for k in CRITICAL_COUNTERS) or (
        counters.get("handsize_vs_melds", 0) / max(1, counters.get("n_opp_obs", 1)) > 1e-4
    )
    label_source = "opponents (oracle decode verified)"
    if degraded:
        print("[warn] critical decode violation -> degrading labels to own shanten", flush=True)
        labels = {
            "own_tenpai": (own_sh <= 0).astype(np.uint8),
            "own_shanten_le1": (own_sh <= 1).astype(np.uint8),
            "own_shanten_le2": (own_sh <= 2).astype(np.uint8),
            POSITIVE_CONTROL: labels[POSITIVE_CONTROL],
        }
        label_source = "own_shanten (decode degraded)"
    primary = PRIMARY if PRIMARY in labels else "own_tenpai"

    # --- split / standardize --------------------------------------------------
    n_files_used = len(files)
    ids = list(range(n_files_used))
    random.Random(args.split_seed).shuffle(ids)
    n_val_files = max(1, int(round(args.val_frac * n_files_used)))
    val_files = set(ids[:n_val_files])
    train_mask = np.array([fid not in val_files for fid in file_ids], dtype=bool)
    val_mask = ~train_mask
    Xtr, Xva = X[train_mask], X[val_mask]
    mu = Xtr.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = np.maximum(Xtr.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    X -= mu  # in-place: keep peak RAM low
    X /= sd
    Xtr, Xva = np.ascontiguousarray(X[train_mask]), np.ascontiguousarray(X[val_mask])
    del X
    print(
        f"[split] train {Xtr.shape[0]} pts / val {Xva.shape[0]} pts "
        f"({n_val_files}/{n_files_used} files)",
        flush=True,
    )

    # --- probe + controls ------------------------------------------------------
    results: dict = {}
    primary_probs = None
    for name in labels:
        y = labels[name].astype(np.float32)
        cs = list(args.control_seeds) if name == primary else [args.control_seeds[0]]
        res, p = evaluate_label(y, train_mask, val_mask, Xtr, Xva, args, device, cs)
        results[name] = res
        if name == primary:
            primary_probs = p
        print(
            f"[probe] {name}: auc={res['auc']:.4f} ece={res['ece']:.4f} "
            f"auc_random={res['auc_random']:.4f} prev_val={res['prevalence_val']:.3f}",
            flush=True,
        )

    ci_lo, ci_hi = bootstrap_auc_ci(
        labels[primary][val_mask], primary_probs, args.bootstrap, args.bootstrap_seed
    )

    # --- verdict ---------------------------------------------------------------
    primary_res = results[primary]
    control_ok = abs(primary_res["auc_random"] - 0.5) < 0.02
    positive_ok = results[POSITIVE_CONTROL]["auc"] >= 0.90
    pipeline_valid = bool(control_ok and positive_ok)
    gate = (
        "allow"
        if pipeline_valid and primary_res["auc"] >= 0.65 and ci_lo > 0.5
        else "block"
    )
    verdict = {
        "pipeline_valid": pipeline_valid,
        "control_gate_pass": bool(control_ok),
        "positive_control_pass": bool(positive_ok),
        "control_gate": "|auc_random - 0.5| < 0.02",
        "gate_belief": gate,
        "evidence_tier": (
            "strong" if primary_res["auc"] >= 0.65 and ci_lo > 0.5
            else "weak" if primary_res["auc"] >= 0.60
            else "none"
        ),
        "rule": "allow iff |auc_random-0.5|<0.02 AND ctrl_auc>=0.90 AND auc>=0.65 AND CI_lo>0.5",
    }

    sha = hashlib.sha256(PINNED.read_bytes()).hexdigest()
    out = {
        "schema": "a2-linear-probe/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": primary,
        "n_samples": int(n),
        "auc": primary_res["auc"],
        "ece": primary_res["ece"],
        "auc_random": primary_res["auc_random"],
        "checkpoint": {
            "path": str(PINNED.relative_to(REPO)),
            "size_bytes": PINNED.stat().st_size,
            "sha256": sha,
            "steps": int(ckpt_steps),
        },
        "embedding": {
            "definition": "phi = brain.encoder(obs) (vendor/mortal/mortal/model.py:156)",
            "dim": 1024,
            "is_oracle": False,
            "encoder_input_channels": 1012,
            "dtype": "float32",
            "batchnorm": "eval",
        },
        "data": {
            "loader": "libriichi.dataset.GameplayLoader(version=4, oracle=True)",
            "glob": "datasets/extracted/**/*.mjson",
            "sampling": f"reservoir sample k={args.n_files}, seed={args.split_seed}; fallback k={args.fallback_k}",
            "n_files_used": n_files_used,
            "n_samples": int(n),
            "n_opponent_observations": int(3 * n),
            "points_per_file_mean": float(np.mean(per_file)),
        },
        "split": {
            "strategy": "by source file (80/20)",
            "seed": args.split_seed,
            "n_train": int(Xtr.shape[0]),
            "n_val": int(Xva.shape[0]),
            "n_val_files": n_val_files,
        },
        "probe": {
            "model": "logistic regression (nn.Linear(1024,1))",
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.wd,
            "batch_size": args.bs,
            "epochs": args.epochs,
            "seed": args.seed,
            "features": "standardized with train mean/std",
        },
        "labels": results,
        "label_source": label_source,
        "degraded": bool(degraded),
        "primary_ci95": [ci_lo, ci_hi],
        "verification": counters,
        "verdict": verdict,
        "runtime": {
            "seconds": round(time.time() - t_start, 1),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "device": device,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=False)
    print(f"[done] wrote {out_path}", flush=True)
    print(json.dumps({k: out[k] for k in ("label", "n_samples", "auc", "ece", "auc_random", "label_source")}, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
