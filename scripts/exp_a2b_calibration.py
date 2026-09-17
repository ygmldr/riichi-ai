#!/usr/bin/env python3
"""A2b: calibration audit (zero training) of three probabilistic outputs.

See runs/exp/A2b/PREREG.md (written before this script ran; its sha256 is recorded in
results.json). Objects:

  (a) A2 linear-probe probabilities  -> ECE/Brier on the held-out val split, re-derived
      with the pre-registered A2 protocol (reservoir seed 13 / k=400, split seed 13,
      probe seed 0, 100 epochs). Per-sample probabilities are persisted.
  (b) GRP 24-permutation distribution vs the game-final ranking (checkpoints/grp.pth),
      using the training-time prefix rule; top-1 confidence ECE/Brier + multiclass
      Brier + one-vs-rest and marginal reliability.
  (c) DQN Q values (pinned current_dqn) -> softmax action-confidence calibration
      against the recorded action (tau=1 primary, tau=0.05 sensitivity) plus a
      value-scale reliability table against the exact training target
      (gamma**steps_to_done * kyoku_reward).

No network is trained or fine-tuned. Products: runs/exp/A2b/{results.json, *.npz}.
README.md is written by hand after inspecting the numbers.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import resource
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "vendor/mortal/target/release"))
sys.path.insert(0, str(REPO / "vendor/mortal"))

from exp_a2_probe import (  # noqa: E402
    DATASET_GLOB,
    HIDDEN_LABELS,
    PINNED,
    POSITIVE_CONTROL,
    PRIMARY,
    auc_score,
    brier_score,
    ece_score,
    embed,
    fit_predict,
    load_frozen_brain,
    mem_available_gb,
    reservoir_sample,
    verify_and_label,
)
from libriichi.dataset import GameplayLoader, Grp  # noqa: E402
from mortal.model import Brain, DQN, GRP  # noqa: E402
from mortal.reward_calculator import RewardCalculator  # noqa: E402

OUT_DIR = REPO / "runs/exp/A2b"
GRP_CKPT = REPO / "checkpoints/grp.pth"
A2_RESULTS = REPO / "runs/exp/A2/results.json"

N_BINS = 15
TIE_EPS = 1e-6
PERMS = list(permutations(range(4)))
PERM_INDEX = {p: i for i, p in enumerate(PERMS)}

# A2 aggregate references for the reproduction check (runs/exp/A2/results.json)
A2_PRIMARY = {"auc": 0.9334087675680629, "ece": 0.009871001787941872, "brier": 0.09427791940688557}


# --------------------------------------------------------------------------- metrics


def reliability_table(prob, y, n_bins: int = N_BINS) -> list[dict]:
    """Equal-width binning reliability table; rows include empty bins."""
    p = np.asarray(prob, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    n = len(yy)
    rows = []
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        row = {
            "bin": b,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": nb,
            "weight": nb / n if n else 0.0,
            "mean_conf": float(p[m].mean()) if nb else None,
            "acc": float(yy[m].mean()) if nb else None,
            "gap": float(yy[m].mean() - p[m].mean()) if nb else None,
        }
        rows.append(row)
    return rows


def bin_sums(conf, y, groups, n_bins: int = N_BINS):
    """Per-group per-bin sufficient statistics of ECE (count, conf sum, positive sum)."""
    conf = np.asarray(conf, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    uniq, ginv = np.unique(groups, return_inverse=True)
    G, B = len(uniq), n_bins
    cnt = np.zeros((G, B), dtype=np.float64)
    sconf = np.zeros((G, B), dtype=np.float64)
    sy = np.zeros((G, B), dtype=np.float64)
    np.add.at(cnt, (ginv, b), 1.0)
    np.add.at(sconf, (ginv, b), conf)
    np.add.at(sy, (ginv, b), y)
    return uniq, cnt, sconf, sy


def ece_from_sums(cnt, sconf, sy) -> float:
    n = cnt.sum()
    if n == 0:
        return float("nan")
    n_b = cnt.sum(0)
    sconf_b = sconf.sum(0)
    sy_b = sy.sum(0)
    mask = n_b > 0
    conf_b = np.zeros_like(n_b)
    acc_b = np.zeros_like(n_b)
    conf_b[mask] = sconf_b[mask] / n_b[mask]
    acc_b[mask] = sy_b[mask] / n_b[mask]
    return float((n_b / n * np.abs(acc_b - conf_b)).sum())


def ece_cluster_ci(conf, y, groups, n_boot: int = 1000, seed: int = 7) -> list:
    """95% CI of ECE under a cluster bootstrap (resample groups with replacement)."""
    uniq, cnt, sconf, sy = bin_sums(conf, y, groups)
    rng = np.random.default_rng(seed)
    G = len(uniq)
    out = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, G, G)
        out[i] = ece_from_sums(cnt[pick], sconf[pick], sy[pick])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def top1_confidence(probs: np.ndarray, true_idx: np.ndarray):
    """probs (n,K) -> (max prob, tie-tolerant hit indicator)."""
    p = np.asarray(probs)
    t = np.asarray(true_idx)
    mx = p.max(axis=1)
    hit = ((p[np.arange(len(t)), t] >= mx - TIE_EPS)).astype(np.float64)
    return mx.astype(np.float64), hit


def multiclass_brier(probs: np.ndarray, true_idx: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    n, K = p.shape
    oh = np.zeros_like(p)
    oh[np.arange(n), np.asarray(true_idx)] = 1.0
    return float(((p - oh) ** 2).sum(axis=1).mean())


def marginal_cells(probs: np.ndarray):
    """(n,24) permutation probs -> ((n,4,4) player x rank probs, same-shape indicators)."""
    p = np.asarray(probs, dtype=np.float64)
    n = p.shape[0]
    cells = np.zeros((n, 4, 4))
    for k, perm in enumerate(PERMS):
        for player in range(4):
            cells[:, player, perm[player]] += p[:, k]
    return cells


def pooled_ovr(probs: np.ndarray, true_idx: np.ndarray):
    p = np.asarray(probs, dtype=np.float64)
    n, K = p.shape
    y = np.zeros_like(p)
    y[np.arange(n), np.asarray(true_idx)] = 1.0
    return p.ravel(), y.ravel()


def quantile_bins(values: np.ndarray, n_bins: int = N_BINS):
    v = np.asarray(values, dtype=np.float64)
    qs = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    idx = np.clip(np.searchsorted(qs, v, side="right") - 1, 0, len(qs) - 2)
    return idx, qs


def value_table(pred, target, n_bins: int = N_BINS) -> dict:
    idx, edges = quantile_bins(pred, n_bins)
    rows = []
    n = len(pred)
    for b in range(len(edges) - 1):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        rows.append({
            "bin": b,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": nb,
            "mean_pred": float(pred[m].mean()),
            "mean_realized": float(target[m].mean()),
            "bias": float(pred[m].mean() - target[m].mean()),
        })
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    err = pred - target
    std = float(target.std())
    vce = float(sum(r["n"] / n * abs(r["bias"]) for r in rows)) if n else float("nan")
    return {
        "n": int(n),
        "n_bins": int(len(edges) - 1),
        "rows": rows,
        "mean_pred": float(pred.mean()),
        "mean_realized": float(target.mean()),
        "mean_bias": float(err.mean()),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "spearman": spearman(pred, target),
        "value_binned_calibration_error_pts": vce,
        "target_std": std,
        "value_binned_calibration_error_over_std": vce / std if std > 0 else float("nan"),
    }


def average_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ss = x[order]
    uniq, inv, counts = np.unique(ss, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = (cum - counts + 1 + cum) / 2.0
    ranks = np.empty_like(avg[inv])
    ranks[order] = avg[inv]
    return ranks


def spearman(a, b) -> float:
    ra, rb = average_ranks(a), average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------------------- selftest


def selftest() -> dict:
    """Synthetic checks of the shared ECE/Brier routines (see PREREG §6)."""
    out = {}

    # 1) perfectly calibrated by construction: p=k/1000 with exactly k positives per value
    ps, ys = [], []
    for k in range(1, 1000):
        ps.append(np.full(1000, k / 1000.0))
        ys.append(np.concatenate([np.ones(k), np.zeros(1000 - k)]))
    p = np.concatenate(ps)
    y = np.concatenate(ys)
    e_perfect = ece_score(y, p, N_BINS)
    assert e_perfect < 1e-6, f"perfect-calibration ECE {e_perfect} >= 1e-6"
    out["perfect_ece"] = e_perfect
    out["perfect_brier"] = brier_score(y, p)
    out["perfect_n"] = int(len(y))

    # 2) known miscalibration: p=0.7 with 50% positives -> ECE = 0.2 exactly
    p2 = np.full(20_000, 0.7)
    y2 = (np.arange(20_000) % 2 == 0).astype(np.float64)
    e_known = ece_score(y2, p2, N_BINS)
    assert abs(e_known - 0.2) < 1e-12, f"known-miscalibration ECE {e_known} != 0.2"
    out["known_miscal_ece"] = e_known
    out["known_miscal_brier"] = brier_score(y2, p2)

    # 3) top-1 path: 24-class construction with exactly round(v*n) hits per confidence level
    conf_parts, hit_parts = [], []
    for vi in range(50, 1000, 50):
        v = vi / 1000.0
        n = 1000
        nhit = round(v * n)
        K = 24
        probs = np.full((n, K), (1.0 - v) / (K - 1))
        probs[:, 0] = v
        true_idx = np.zeros(n, dtype=np.int64)
        true_idx[nhit:] = 1
        conf_parts.append(probs)
        hit_parts.append(true_idx)
    probs = np.concatenate(conf_parts)
    true_idx = np.concatenate(hit_parts)
    conf, hit = top1_confidence(probs, true_idx)
    e_top1 = ece_score(hit, conf, N_BINS)
    assert e_top1 < 1e-6, f"top-1 perfect ECE {e_top1} >= 1e-6"
    out["top1_perfect_ece"] = e_top1
    out["top1_perfect_brier"] = brier_score(hit, conf)

    # 4) Brier sanity
    assert brier_score([0, 1], [0.0, 1.0]) == 0.0
    out["brier_sanity"] = 0.0

    # 5) cluster-CI path must reproduce the point ECE when there is a single group
    uniq, cnt, sconf, sy = bin_sums(conf, hit, np.zeros(len(hit), dtype=np.int64))
    assert abs(ece_from_sums(cnt, sconf, sy) - e_top1) < 1e-12
    out["cluster_path_ok"] = True
    out["n_samples_perfect"] = int(out["perfect_n"])
    return out


# ------------------------------------------------------------------------ extraction (a)+(c)


def load_pinned(device: str):
    """Load the pinned Brain + DQN of the same checkpoint (frozen, eval)."""
    state = torch.load(PINNED, map_location="cpu", weights_only=True)
    cfg = state["config"]
    brain = Brain(version=cfg["control"]["version"], **cfg["resnet"])
    m1, u1 = brain.load_state_dict(state["mortal"], strict=False)
    if m1 or u1:
        raise RuntimeError(f"brain checkpoint mismatch: {m1[:3]} {u1[:3]}")
    dqn = DQN(version=cfg["control"]["version"])
    m2, u2 = dqn.load_state_dict(state["current_dqn"], strict=False)
    if m2 or u2:
        raise RuntimeError(f"dqn checkpoint mismatch: {m2[:3]} {u2[:3]}")
    brain.eval().to(device)
    dqn.eval().to(device)
    for p in list(brain.parameters()) + list(dqn.parameters()):
        p.requires_grad_(False)
    cfg_grp = cfg.get("grp", {})
    return brain, dqn, cfg, cfg_grp


def load_grp(cfg_grp: dict):
    net_cfg = cfg_grp.get("network", {"hidden_size": 64, "num_layers": 2})
    grp = GRP(**net_cfg)
    state = torch.load(GRP_CKPT, map_location="cpu", weights_only=True)
    m, u = grp.load_state_dict(state["model"], strict=False)
    if m or u:
        raise RuntimeError(f"grp checkpoint mismatch: {m[:3]} {u[:3]}")
    grp.eval()
    return grp, int(state.get("steps", -1))


def extract_ac(args, brain, dqn, reward_calc, loader, files, device):
    """Single pass over the A2 sample: phi + A2 labels (a) and Q/value records (c).

    Also collects GRP round records for the shared files (cross-object subset of (b)).
    """
    name_list = list(HIDDEN_LABELS) + [POSITIVE_CONTROL]
    phi_parts: list[np.ndarray] = []
    label_parts: dict[str, list[np.ndarray]] = {k: [] for k in name_list}
    file_id_parts, pov_parts = [], []
    q_parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("q_taken", "q_max", "match", "conf_tau1", "conf_tau05",
                        "target", "kyoku_reward", "at_kyoku", "action", "steps_to_done")
    }
    grp_parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("probs", "true_idx", "file_idx", "round_idx", "player_id")}
    counters: dict[str, int] = defaultdict(int)
    per_file: list[int] = []
    n_points = 0
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
                act_l = g.take_actions()
                mask_l = g.take_masks()
                ky_l = g.take_at_kyoku()
                dones_l = g.take_dones()
                ag_l = g.take_apply_gamma()
                grp_obj = g.take_grp()
                if not obs_l:
                    continue
                arr = np.stack([np.asarray(o) for o in obs_l])
                inv = np.stack([np.asarray(o) for o in inv_l])
                own_sh = np.asarray(sh_l, dtype=np.int8)
                actions = np.asarray(act_l, dtype=np.int64)
                masks = np.stack([np.asarray(m) for m in mask_l])
                at_ky = np.frombuffer(ky_l, dtype=np.uint8) if isinstance(ky_l, bytes) else np.asarray(ky_l, np.uint8)
                dones = np.asarray(dones_l, dtype=bool)
                apply_gamma = np.asarray(ag_l, dtype=bool)

                labels, c = verify_and_label(arr, inv, own_sh)
                for k, v in c.items():
                    counters[k] += v

                # ---- (c) Q values: phi -> Mish -> Q, stats only (no full Q kept)
                phi = embed(brain, arr, args.chunk, device)
                with torch.inference_mode():
                    mish = torch.nn.functional.mish(torch.from_numpy(phi).to(device, non_blocking=True))
                    q = dqn(mish, torch.from_numpy(masks).to(device))  # (n,46) masked -inf
                    qm = q.float().cpu().numpy()
                q_taken = qm[np.arange(len(actions)), actions]
                q_max = qm.max(axis=1)
                match = (q_taken >= q_max - TIE_EPS).astype(np.uint8)
                shifted = np.where(masks, qm, -np.inf)
                shifted = shifted - shifted.max(axis=1, keepdims=True)
                with np.errstate(over="ignore", invalid="ignore"):
                    e1 = np.exp(shifted / 1.0)
                    e05 = np.exp(shifted / 0.05)
                e1[~masks] = 0.0
                e05[~masks] = 0.0
                conf_tau1 = e1.max(axis=1) / e1.sum(axis=1)
                conf_tau05 = e05.max(axis=1) / e05.sum(axis=1)

                # ---- (c) value target: exactly dataloader.py:91-122 + train.py:236
                feat = grp_obj.take_feature()
                rkb = list(np.frombuffer(grp_obj.take_rank_by_player(), dtype=np.uint8))
                kr = np.asarray(reward_calc.calc_delta_pt(pov, feat, rkb), dtype=np.float64)
                if len(kr) < int(at_ky[-1]) + 1:
                    raise RuntimeError(f"kyoku_rewards shorter than at_kyoku (file {batch[off]})")
                steps = np.zeros(len(dones), dtype=np.int64)
                for i in reversed(range(len(dones))):
                    if not dones[i]:
                        steps[i] = steps[i + 1] + int(apply_gamma[i])
                kyoku_reward = kr[at_ky]
                target = args.gamma ** steps * kyoku_reward

                # ---- (b) shared-sample GRP records + marginal-vs-reference check
                seqs = [torch.as_tensor(feat[:i + 1]) for i in range(len(feat))]
                with torch.inference_mode():
                    logits = grp_model(seqs)
                    probs64 = logits.softmax(-1)
                true_idx = PERM_INDEX[tuple(int(x) for x in rkb)]
                if len(probs64) > 0:
                    ref = reward_calc.calc_grp(feat)
                    mine = marginal_cells(probs64.numpy())
                    if not np.allclose(mine, ref.numpy(), atol=1e-9):
                        raise RuntimeError("GRP marginal mismatch vs reward_calculator.calc_grp")
                    grp_parts["probs"].append(probs64.numpy().astype(np.float32))
                    grp_parts["true_idx"].append(np.full(len(feat), true_idx, dtype=np.int16))
                    grp_parts["file_idx"].append(np.full(len(feat), fid, dtype=np.int32))
                    grp_parts["round_idx"].append(np.arange(len(feat), dtype=np.int16))
                    grp_parts["player_id"].append(np.full(len(feat), pov, dtype=np.uint8))

                phi_parts.append(phi)
                for k in name_list:
                    label_parts[k].append(labels[k])
                q_parts["q_taken"].append(q_taken.astype(np.float32))
                q_parts["q_max"].append(q_max.astype(np.float32))
                q_parts["match"].append(match)
                q_parts["conf_tau1"].append(conf_tau1.astype(np.float32))
                q_parts["conf_tau05"].append(conf_tau05.astype(np.float32))
                q_parts["target"].append(target.astype(np.float32))
                q_parts["kyoku_reward"].append(kyoku_reward.astype(np.float32))
                q_parts["at_kyoku"].append(at_ky.astype(np.uint8))
                q_parts["action"].append(actions.astype(np.int16))
                q_parts["steps_to_done"].append(steps.astype(np.uint8))

                file_id_parts.append(np.full(len(obs_l), fid, dtype=np.int32))
                pov_parts.append(np.full(len(obs_l), pov, dtype=np.uint8))
                n_points += len(obs_l)
                del arr, inv, own_sh, obs_l, inv_l, sh_l, labels, phi, mish, q, qm, feat, logits, probs64
            per_file.append(n_points - n_before)
        print(
            f"[ac] files {min(bstart + args.file_batch, len(files))}/{len(files)} "
            f"points {n_points} elapsed {time.time() - t0:.0f}s MemAvailable {mem_available_gb():.1f} GB",
            flush=True,
        )
        if mem_available_gb() < 4.0:
            print("[ac] WARNING: MemAvailable < 4 GB", flush=True)

    out = {
        "phi": np.concatenate(phi_parts, axis=0) if phi_parts else np.zeros((0, 1024), np.float32),
        "labels": {k: np.concatenate(v) for k, v in label_parts.items() if v},
        "file_ids": np.concatenate(file_id_parts) if file_id_parts else np.zeros(0, np.int32),
        "povs": np.concatenate(pov_parts) if pov_parts else np.zeros(0, np.uint8),
        "q": {k: np.concatenate(v) for k, v in q_parts.items() if v},
        "grp": {k: np.concatenate(v) for k, v in grp_parts.items() if v},
        "counters": dict(counters),
        "per_file": per_file,
    }
    return out


# ----------------------------------------------------------------------------- (b) pass


def grp_pass(args, grp, files):
    """GRP per-round distributions for the pre-registered k=10000 draw."""
    probs_parts, true_parts, fid_parts, ridx_parts = [], [], [], []
    load_errors, zero_round, bad_rank, n_games = 0, 0, 0, 0
    t0 = time.time()
    for cstart in range(0, len(files), args.grp_chunk):
        chunk = files[cstart:cstart + args.grp_chunk]
        try:
            batch = Grp.load_gz_log_files(chunk)
        except Exception as exc:  # noqa: BLE001 - recover file by file
            print(f"[grp] chunk at {cstart} failed ({exc}); retrying per file", flush=True)
            batch = []
            for f in chunk:
                try:
                    batch.append(Grp.load_gz_log_files([f])[0])
                except Exception:  # noqa: BLE001
                    load_errors += 1
                    batch.append(None)
        for off, g in enumerate(batch):
            if g is None:
                continue
            fid = cstart + off
            feat = g.take_feature()
            rkb = np.frombuffer(g.take_rank_by_player(), dtype=np.uint8)
            if feat.shape[0] == 0:
                zero_round += 1
                continue
            if len(set(int(x) for x in rkb)) != 4:
                bad_rank += 1
                continue
            n_games += 1
            true_idx = PERM_INDEX[tuple(int(x) for x in rkb)]
            seqs = [torch.as_tensor(feat[:i + 1]) for i in range(len(feat))]
            with torch.inference_mode():
                probs = grp(seqs).softmax(-1).numpy().astype(np.float32)
            probs_parts.append(probs)
            true_parts.append(np.full(len(feat), true_idx, dtype=np.int16))
            fid_parts.append(np.full(len(feat), fid, dtype=np.int32))
            ridx_parts.append(np.arange(len(feat), dtype=np.int16))
        print(
            f"[grp] files {min(cstart + args.grp_chunk, len(files))}/{len(files)} "
            f"games {n_games} rounds {sum(len(p) for p in probs_parts)} "
            f"elapsed {time.time() - t0:.0f}s",
            flush=True,
        )
    return {
        "probs": np.concatenate(probs_parts) if probs_parts else np.zeros((0, 24), np.float32),
        "true_idx": np.concatenate(true_parts) if true_parts else np.zeros(0, np.int16),
        "file_idx": np.concatenate(fid_parts) if fid_parts else np.zeros(0, np.int32),
        "round_idx": np.concatenate(ridx_parts) if ridx_parts else np.zeros(0, np.int16),
        "n_games": n_games,
        "load_errors": load_errors,
        "zero_round": zero_round,
        "bad_rank": bad_rank,
    }


# ------------------------------------------------------------------------ (b) metrics


def grp_metrics(probs, true_idx, file_idx, args) -> dict:
    conf, hit = top1_confidence(probs, true_idx)
    p_ovr, y_ovr = pooled_ovr(probs, true_idx)
    cells = marginal_cells(probs)
    out = {
        "n_rounds": int(len(probs)),
        "n_files": int(len(np.unique(file_idx))),
        "top1_accuracy": float(hit.mean()),
        "top1_mean_confidence": float(conf.mean()),
        "top1_ece": float(ece_score(hit, conf, args.ece_bins)),
        "top1_brier": float(brier_score(hit, conf)),
        "top1_ece_ci95_cluster_bootstrap": ece_cluster_ci(conf, hit, file_idx, args.bootstrap, args.bootstrap_seed),
        "multiclass_brier": multiclass_brier(probs, true_idx),
        "pooled_ovr_ece": float(ece_score(y_ovr, p_ovr, args.ece_bins)),
        "pooled_ovr_brier": float(brier_score(y_ovr, p_ovr)),
        "reliability_top1": reliability_table(conf, hit, args.ece_bins),
        "uniform_reference": {
            "top1_ece": float(abs(hit.mean() - 1.0 / 24.0)),
            "multiclass_brier": float(((np.full((len(probs), 24), 1 / 24) - np.eye(24)[true_idx]) ** 2).sum(1).mean()),
        },
    }
    # marginal (player, rank) one-vs-rest, pooled per rank row and overall
    y_cells = np.zeros_like(cells)
    ti = np.asarray(true_idx)
    for k, perm in enumerate(PERMS):
        m = (ti == k)
        if not m.any():
            continue
        for player in range(4):
            y_cells[m, player, perm[player]] = 1.0
    # each round contributes 4 players x 4 ranks with exactly four 1s -> mean 1/4
    assert abs(y_cells.mean() - 0.25) < 1e-9, f"marginal indicator mean {y_cells.mean()}"
    rows = []
    for r in range(4):
        p_r = cells[:, :, r].ravel()
        y_r = y_cells[:, :, r].ravel()
        entry = {
            "rank": r,
            "n": int(len(p_r)),
            "mean_prob": float(p_r.mean()),
            "frequency": float(y_r.mean()),
            "ece": float(ece_score(y_r, p_r, args.ece_bins)),
            "brier": float(brier_score(y_r, p_r)),
        }
        if r == 0:
            entry["reliability"] = reliability_table(p_r, y_r, args.ece_bins)
        rows.append(entry)
    out["marginal_by_rank"] = rows
    p_all, y_all = cells.ravel(), y_cells.ravel()
    out["marginal_ovr_ece"] = float(ece_score(y_all, p_all, args.ece_bins))
    out["marginal_ovr_brier"] = float(brier_score(y_all, p_all))
    return out


def frequency_reference(probs, true_idx, file_idx, args) -> dict:
    """Constant predictor: empirical permutation frequencies on a file-level 50/50 split."""
    files = np.unique(file_idx)
    fl = [int(f) for f in files]
    random.Random(args.split_seed).shuffle(fl)
    fit_files = set(fl[: len(fl) // 2])
    eval_mask = ~np.isin(file_idx, list(fit_files))
    counts = np.bincount(np.asarray(true_idx)[~eval_mask], minlength=24).astype(np.float64)
    freq = counts / counts.sum()
    p = np.tile(freq, (int(eval_mask.sum()), 1))
    t = np.asarray(true_idx)[eval_mask]
    conf, hit = top1_confidence(p, t)
    return {
        "definition": "constant predictor p_k = empirical permutation frequency (fit half), evaluated on the held-out half",
        "split": "file-level 50/50, random.Random(13)",
        "n_fit_rounds": int((~eval_mask).sum()),
        "n_eval_rounds": int(eval_mask.sum()),
        "top1_ece": float(ece_score(hit, conf, args.ece_bins)),
        "top1_brier": float(brier_score(hit, conf)),
        "multiclass_brier": multiclass_brier(p, t),
        "argmax_perm": int(freq.argmax()),
        "argmax_perm_frequency": float(freq.max()),
        "eval_accuracy_at_argmax": float(hit.mean()),
    }


def grp_metrics_by_round(probs, true_idx, round_idx, args) -> dict:
    conf, hit = top1_confidence(probs, true_idx)
    buckets = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 99)]
    out = {}
    for lo, hi in buckets:
        m = (round_idx >= lo) & (round_idx <= hi)
        if not m.any():
            continue
        out[f"round_{lo}" if lo == hi else f"round_{lo}-{hi}"] = {
            "n": int(m.sum()),
            "acc": float(hit[m].mean()),
            "mean_conf": float(conf[m].mean()),
            "ece": float(ece_score(hit[m], conf[m], args.ece_bins)),
        }
    return out


# ----------------------------------------------------------------------------- main


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-files", type=int, default=400)
    ap.add_argument("--min-points", type=int, default=200_000)
    ap.add_argument("--fallback-k", type=int, default=100)
    ap.add_argument("--fallback-seed-start", type=int, default=14)
    ap.add_argument("--fallback-max", type=int, default=6)
    ap.add_argument("--file-batch", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=13)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--ece-bins", type=int, default=N_BINS)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=7)
    ap.add_argument("--grp-files", type=int, default=10_000)
    ap.add_argument("--grp-chunk", type=int, default=1000)
    ap.add_argument("--grp-probed-first", type=int, default=2000,
                    help="first N files of the GRP draw used by the pre-run timing probe (excluded from primary)")
    ap.add_argument("--gamma", type=float, default=None,
                    help="discount for the Q target; default = checkpoint config env.gamma")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run for validation")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    global grp_model
    args = parse_args(argv)
    t_start = time.time()

    if args.selftest:
        res = selftest()
        print("[selftest] PASS " + json.dumps(res, indent=2))
        return 0

    if args.smoke:
        args.n_files = 4
        args.min_points = 0
        args.fallback_max = 0
        args.grp_files = 40
        args.grp_chunk = 40
        args.grp_probed_first = 20
        args.epochs = 3
        args.bootstrap = 50
        args.out_dir = Path("/tmp/opencode/a2b/smoke")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(OUT_DIR / "PREREG.md", args.out_dir / "PREREG.md")

    prereg = args.out_dir / "PREREG.md"
    if not prereg.exists():
        raise SystemExit(f"FATAL: {prereg} must exist before the audit runs (pre-registration)")
    prereg_sha = hashlib.sha256(prereg.read_bytes()).hexdigest()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 1.5e9 else "cpu"
    else:
        device = args.device

    brain, dqn, cfg, cfg_grp = load_pinned(device)
    grp_model, grp_steps = load_grp(cfg_grp)
    gamma = float(cfg["env"]["gamma"]) if args.gamma is None else float(args.gamma)
    args.gamma = gamma
    pts = list(cfg["env"]["pts"])
    reward_calc = RewardCalculator(grp_model, pts=pts)
    loader = GameplayLoader(4, oracle=True)
    print(
        f"[setup] device={device} gamma={gamma} pts={pts} grp_steps={grp_steps} args={vars(args)}",
        flush=True,
    )

    # ---------------- (a)+(c): A2 sample ------------------------------------
    files = reservoir_sample(glob.iglob(DATASET_GLOB, recursive=True), args.n_files, args.split_seed)
    file_list_sha = hashlib.sha256("\n".join(files).encode()).hexdigest()
    print(f"[sample] reservoir-sampled {len(files)} files (seed {args.split_seed})", flush=True)

    ac = extract_ac(args, brain, dqn, reward_calc, loader, files, device)
    X = ac.pop("phi")
    labels = ac["labels"]
    file_ids = ac["file_ids"]
    povs = ac["povs"]

    seed = args.fallback_seed_start
    while len(X) < args.min_points and seed < args.fallback_seed_start + args.fallback_max:
        extra = reservoir_sample(glob.iglob(DATASET_GLOB, recursive=True), args.fallback_k, seed)
        print(f"[sample] fallback draw k={args.fallback_k} seed={seed}", flush=True)
        ac2 = extract_ac(args, brain, dqn, reward_calc, loader, extra, device)
        offset = len(files)
        files += extra
        X = np.concatenate([X, ac2.pop("phi")], axis=0)
        for k in labels:
            labels[k] = np.concatenate([labels[k], ac2["labels"][k]])
        file_ids = np.concatenate([file_ids, ac2["file_ids"] + offset])
        povs = np.concatenate([povs, ac2["povs"]])
        for k in ac["q"]:
            ac["q"][k] = np.concatenate([ac["q"][k], ac2["q"][k]])
        for k in ac["grp"]:
            ac["grp"][k] = np.concatenate([ac["grp"][k], ac2["grp"][k]])
        for k, v in ac2["counters"].items():
            ac["counters"][k] = ac["counters"].get(k, 0) + v
        ac["per_file"] += ac2["per_file"]
        del ac2
        seed += 1

    n = len(X)
    print(f"[ac] done: {n} decision points from {len(files)} files", flush=True)
    if n < args.min_points:
        raise SystemExit(f"FATAL: only {n} decision points (< {args.min_points})")
    q = ac["q"]

    # ---------------- split + standardization (A2 protocol) ------------------
    n_files_used = len(files)
    ids = list(range(n_files_used))
    random.Random(args.split_seed).shuffle(ids)
    n_val_files = max(1, int(round(args.val_frac * n_files_used)))
    val_files = set(ids[:n_val_files])
    train_mask = np.array([fid not in val_files for fid in file_ids], dtype=bool)
    val_mask = ~train_mask
    Xtr_stats = X[train_mask]
    mu = Xtr_stats.mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = np.maximum(Xtr_stats.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    del Xtr_stats
    X -= mu  # in-place: keep peak RAM low
    X /= sd
    Xtr = np.ascontiguousarray(X[train_mask])
    Xva = np.ascontiguousarray(X[val_mask])
    del X
    print(f"[split] train {Xtr.shape[0]} pts / val {Xva.shape[0]} pts ({n_val_files}/{n_files_used} files)", flush=True)

    # ---------------- object (a): probe fits ---------------------------------
    fit_args = SimpleNamespace(bs=args.bs, epochs=args.epochs, lr=args.lr, wd=args.wd)
    probe_out = {}
    val_probs = {}
    for name in labels:
        t0 = time.time()
        y = labels[name].astype(np.float32)
        p = fit_predict(Xtr, y[train_mask], Xva, fit_args, device, args.seed)
        yva = y[val_mask].astype(np.float64)
        ece = float(ece_score(yva, p, args.ece_bins))
        entry = {
            "n_val": int(val_mask.sum()),
            "prevalence_val": float(yva.mean()),
            "auc": float(auc_score(yva, p)),
            "ece": ece,
            "brier": float(brier_score(yva, p)),
            "ece_ci95_cluster_bootstrap": ece_cluster_ci(p, yva, file_ids[val_mask], args.bootstrap, args.bootstrap_seed),
            "reliability": reliability_table(p, yva, args.ece_bins),
        }
        if name == PRIMARY:
            entry["a2_reference"] = {
                "auc": A2_PRIMARY["auc"], "ece": A2_PRIMARY["ece"], "brier": A2_PRIMARY["brier"],
                "delta_auc": entry["auc"] - A2_PRIMARY["auc"],
                "delta_ece": entry["ece"] - A2_PRIMARY["ece"],
                "delta_brier": entry["brier"] - A2_PRIMARY["brier"],
                "reproduced_within_1e-3": bool(
                    abs(entry["auc"] - A2_PRIMARY["auc"]) < 1e-3
                    and abs(entry["ece"] - A2_PRIMARY["ece"]) < 1e-3
                    and abs(entry["brier"] - A2_PRIMARY["brier"]) < 1e-3
                ),
            }
        probe_out[name] = entry
        val_probs[name] = p.astype(np.float32)
        print(
            f"[probe] {name}: auc={entry['auc']:.4f} ece={entry['ece']:.4f} "
            f"brier={entry['brier']:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    del Xtr, Xva

    # ---------------- object (c): Q calibration ------------------------------
    conf1 = q["conf_tau1"].astype(np.float64)
    conf05 = q["conf_tau05"].astype(np.float64)
    match = q["match"].astype(np.float64)
    groups = file_ids
    q_out = {
        "n_points": int(n),
        "n_files": int(n_files_used),
        "match_definition": "1[q(a_recorded) >= max_legal q - 1e-6]",
        "tau1_primary": {
            "tau": 1.0,
            "match_rate": float(match.mean()),
            "mean_confidence": float(conf1.mean()),
            "ece": float(ece_score(match, conf1, args.ece_bins)),
            "brier": float(brier_score(match, conf1)),
            "ece_ci95_cluster_bootstrap": ece_cluster_ci(conf1, match, groups, args.bootstrap, args.bootstrap_seed),
            "reliability": reliability_table(conf1, match, args.ece_bins),
        },
        "tau005_sensitivity": {
            "tau": 0.05,
            "match_rate": float(match.mean()),
            "mean_confidence": float(conf05.mean()),
            "ece": float(ece_score(match, conf05, args.ece_bins)),
            "brier": float(brier_score(match, conf05)),
            "ece_ci95_cluster_bootstrap": ece_cluster_ci(conf05, match, groups, args.bootstrap, args.bootstrap_seed),
            "reliability": reliability_table(conf05, match, args.ece_bins),
        },
        "value_scale_diagnostic": value_table(q["q_taken"].astype(np.float64), q["target"].astype(np.float64), args.ece_bins),
    }
    val_probe_subset = None
    if val_mask.any():
        val_probe_subset = {
            "n_points": int(val_mask.sum()),
            "tau1_ece": float(ece_score(match[val_mask], conf1[val_mask], args.ece_bins)),
            "tau1_brier": float(brier_score(match[val_mask], conf1[val_mask])),
        }
    q_out["val_subset"] = val_probe_subset
    print(
        f"[q] tau1 ece={q_out['tau1_primary']['ece']:.4f} brier={q_out['tau1_primary']['brier']:.4f} "
        f"match={q_out['tau1_primary']['match_rate']:.4f}; "
        f"value mae={q_out['value_scale_diagnostic']['mae']:.4f} "
        f"vce={q_out['value_scale_diagnostic']['value_binned_calibration_error_pts']:.4f}",
        flush=True,
    )

    # ---------------- object (b): GRP audit ----------------------------------
    grp_files_sample = reservoir_sample(glob.iglob(DATASET_GLOB, recursive=True), args.grp_files, args.split_seed)
    grp_list_sha = hashlib.sha256("\n".join(grp_files_sample).encode()).hexdigest()
    print(f"[grp] reservoir-sampled {len(grp_files_sample)} files (seed {args.split_seed})", flush=True)
    b = grp_pass(args, grp_model, grp_files_sample)
    keep = b["file_idx"] >= args.grp_probed_first
    b_primary = {
        "sample": f"files[{args.grp_probed_first}:] of the k={args.grp_files} seed={args.split_seed} draw",
        "n_files": int(len(np.unique(b["file_idx"][keep]))),
    }
    b_primary.update(grp_metrics(b["probs"][keep], b["true_idx"][keep], b["file_idx"][keep], args))
    b_primary["top1_by_round_index"] = grp_metrics_by_round(
        b["probs"][keep], b["true_idx"][keep], b["round_idx"][keep], args)
    b_primary["train_frequency_reference"] = frequency_reference(
        b["probs"][keep], b["true_idx"][keep], b["file_idx"][keep], args)
    b_probed = {"sample": f"files[:{args.grp_probed_first}] (pre-run timing probe subset; transparency only)"}
    b_probed.update(grp_metrics(b["probs"][~keep], b["true_idx"][~keep], b["file_idx"][~keep], args))
    b_shared = {"sample": "the 400-file A2 sample (cross-object consistency subset)"}
    b_shared.update(grp_metrics(ac["grp"]["probs"], ac["grp"]["true_idx"], ac["grp"]["file_idx"], args))
    b_out = {
        "primary": b_primary,
        "probed_subset": b_probed,
        "a2_sample_subset": b_shared,
        "loader": {
            "n_files_requested": len(grp_files_sample),
            "n_games": b["n_games"],
            "n_rounds_total": int(len(b["probs"])),
            "load_errors": b["load_errors"],
            "zero_round_games": b["zero_round"],
            "bad_rank_games": b["bad_rank"],
            "truth_alignment": "game_final_rank_by_player (exact; no degradation needed)",
        },
        "reference": "checkpoints/grp.pth (hidden 64, 2 layers, float64, eval)",
    }
    print(
        f"[grp] primary: n={b_primary['n_rounds']} top1_ece={b_primary['top1_ece']:.4f} "
        f"top1_brier={b_primary['top1_brier']:.4f} acc={b_primary['top1_accuracy']:.4f} "
        f"mc_brier={b_primary['multiclass_brier']:.4f}",
        flush=True,
    )

    # ---------------- verdicts ----------------------------------------------
    e_a = probe_out[PRIMARY]["ece"]
    e_b = b_primary["top1_ece"]
    e_c = q_out["tau1_primary"]["ece"]
    gate_three = "allow" if max(e_a, e_b, e_c) > 0.05 else "block"
    gate_two = "allow" if max(e_a, e_b) > 0.05 else "block"
    verdict = {
        "threshold_audit": "ECE > 0.05 => allow (triage / readiness gate for the belief line, todo 15)",
        "threshold_delivery": "belief-head delivery gate (todo 29): ECE <= 0.15; NOT the same as the 0.05 triage gate",
        "gate_three_default": gate_three,
        "gate_two_models": gate_two,
        "gate_three_rule": "allow iff max(ece_a_probe_primary, ece_b_grp_top1, ece_c_q_tau1) > 0.05",
        "gate_two_rule": "allow iff max(ece_a_probe_primary, ece_b_grp_top1) > 0.05",
        "independent_of_a2_verdict": True,
    }

    # ---------------- artifacts ---------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "a2b_probe_val.npz",
        **{f"prob_{k}": v for k, v in val_probs.items()},
        **{f"y_{k}": labels[k][val_mask].astype(np.uint8) for k in labels},
        file_ids=file_ids[val_mask].astype(np.int32),
        povs=povs[val_mask].astype(np.uint8),
    )
    np.savez_compressed(
        args.out_dir / "a2b_grp_rounds.npz",
        probs=b["probs"],
        true_idx=b["true_idx"].astype(np.int16),
        file_idx=b["file_idx"].astype(np.int32),
        round_idx=b["round_idx"].astype(np.int16),
        primary_from_file_idx=np.int32(args.grp_probed_first),
    )
    np.savez_compressed(
        args.out_dir / "a2b_q_records.npz",
        **{k: v for k, v in q.items()},
        file_ids=file_ids.astype(np.int32),
        val_mask=val_mask.astype(np.uint8),
    )

    policy_sha = hashlib.sha256(PINNED.read_bytes()).hexdigest()
    grp_sha = hashlib.sha256(GRP_CKPT.read_bytes()).hexdigest()
    results = {
        "schema": "a2b-calibration-audit/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "zero_training": True,
        "prereg": {"path": "runs/exp/A2b/PREREG.md", "sha256": prereg_sha},
        "checkpoints": {
            "policy": {"path": str(PINNED.relative_to(REPO)), "sha256": policy_sha},
            "grp": {"path": str(GRP_CKPT.relative_to(REPO)), "sha256": grp_sha, "steps": grp_steps},
        },
        "reward_config": {"gamma": gamma, "pts": pts,
                          "source": "pinned checkpoint config['env'] (dataloader.py:91-122, train.py:236)"},
        "objects": {
            "a_probe": {
                "definition": "A2 logistic probe re-derived on the frozen phi; ECE/Brier on the held-out val split",
                "sample": {"n_files": n_files_used, "n_points": int(n), "n_val": int(val_mask.sum()),
                           "n_val_files": n_val_files, "file_list_sha256": file_list_sha,
                           "sampling": f"reservoir k={args.n_files} seed={args.split_seed}"},
                "labels": probe_out,
                "gate_bearing_label": PRIMARY,
            },
            "b_grp": {
                "definition": "24-permutation softmax of checkpoints/grp.pth vs the game-final rank_by_player; "
                              "primary = top-1 confidence ECE/Brier on the pre-registered sample",
                **b_out,
            },
            "c_dqn": {
                "definition": "DQN Q values of the pinned checkpoint: softmax action-confidence calibration "
                              "(tau=1 primary, tau=0.05 sensitivity) + value-scale table vs the exact training target",
                **q_out,
            },
        },
        "summary_ece": {
            "a_probe_primary": e_a,
            "b_grp_top1": e_b,
            "c_q_actionconf_tau1": e_c,
            "note": "these three are the numbers that feed docs/exp-gates.md (todo 15)",
        },
        "gate": verdict,
        "verification": ac["counters"],
        "self_check": selftest(),
        "runtime": {
            "seconds": round(time.time() - t_start, 1),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "device": device,
        },
    }
    with open(args.out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {args.out_dir / 'results.json'}", flush=True)
    print(json.dumps({"summary_ece": results["summary_ece"], "gate": verdict}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
