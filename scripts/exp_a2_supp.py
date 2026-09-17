#!/usr/bin/env python3
"""A2 supplementary analysis — POST-HOC, not pre-registered.

Written after the pre-registered A2 run (runs/exp/A2/results.json) to interpret it:

1. PUBLIC-INFO BASELINE: probe a small vector of plainly public features (tiles left,
   own turn counter, opponent riichi flags, own shanten, round/honba, visible dora/kan
   count) for the primary label. Quantifies how much of the primary AUC is explainable
   from public information alone.
2. MISH(phi) PROBE: `Brain.forward` returns `Mish(phi)` and the DQN head is a linear map
   on that tensor; probing it measures what the *policy-visible* representation encodes.
3. EXACT-DETERMINANT POSITIVE CONTROL: probe the 6 raw meld-presence planes (which
   exactly determine `ctrl_opp_any_meld_visible`) to validate the label -> split -> fit ->
   metric path independently of the encoder.

Re-uses the pre-registered reservoir sample (seed 13, k=400) and split (seed 13), so the
phi reproductions are directly comparable to results.json. Output: supplementary.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_a2_probe import (  # noqa: E402
    DATASET_GLOB,
    OUT_DIR,
    POSITIVE_CONTROL,
    PRIMARY,
    GameplayLoader,
    auc_score,
    ece_score,
    embed,
    fit_predict,
    load_frozen_brain,
    reservoir_sample,
    verify_and_label,
)

PUBLIC_FEATURE_NAMES = [
    "tiles_left",
    "own_at_turns/17",
    "riichi_declared_opp{1,2,3}",
    "riichi_accepted_opp{1,2,3}",
    "own_at_furiten",
    "own_shanten_onehot_0..6",
    "round_index",
    "honba",
    "tiles_seen",
    "n_dora_indicators",
    "kyoku_onehot_0..3",
]


def public_features(obs: np.ndarray, at_turns: np.ndarray) -> np.ndarray:
    n = obs.shape[0]
    assert at_turns.shape == (n,)
    feats = np.concatenate(
        [
            obs[:, 717, 0:1],  # tiles_left /69
            (at_turns.astype(np.float32) / 17.0)[:, None],
            obs[:, 854:857].sum(axis=2),  # riichi declared (relative opponents)
            obs[:, 857:860].sum(axis=2),  # riichi accepted
            obs[:, 861].max(axis=1, keepdims=True),  # own furiten (plane all-ones)
            obs[:, 862:869].max(axis=2),  # own shanten one-hot
            obs[:, 27, 0:1],  # round_index
            obs[:, 23, 0:1],  # honba
            obs[:, 835, 0:1],  # tiles_seen
            (obs[:, 28:35].max(axis=2) > 0).sum(axis=1, keepdims=True).astype(np.float32),
            obs[:, 19:23].max(axis=2),  # kyoku one-hot
        ],
        axis=1,
    )
    return feats.astype(np.float32)


def exact_meld_features(obs: np.ndarray) -> np.ndarray:
    n = obs.shape[0]
    fuuro = obs[:, 771:831].reshape(n, 3, 4, 5, 34)
    n_fuuro = (fuuro[:, :, :, :4, :] != 0).any(axis=(3, 4)).sum(axis=2)  # (n,3)
    n_ankan = (obs[:, 832:835] != 0).sum(axis=2)  # (n,3)
    return np.concatenate([(n_fuuro > 0), (n_ankan > 0)], axis=1).astype(np.float32)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-files", type=int, default=400)
    ap.add_argument("--sample-seed", type=int, default=13)
    ap.add_argument("--split-seed", type=int, default=13)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--file-batch", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 1.5e9 else "cpu"
    else:
        device = args.device
    brain, _ = load_frozen_brain(device)
    loader = GameplayLoader(4, oracle=True)

    files = reservoir_sample(glob.iglob(DATASET_GLOB, recursive=True), args.n_files, args.sample_seed)
    print(f"[supp] {len(files)} files (seed {args.sample_seed}), device={device}", flush=True)

    phi_parts, mish_parts, pub_parts, meld_parts, label_parts = [], [], [], [], defaultdict(list)
    file_ids, own_sh_parts = [], []
    year_pos = defaultdict(lambda: [0, 0])
    n_points = 0
    for bstart in range(0, len(files), args.file_batch):
        for off, games in enumerate(loader.load_gz_log_files(files[bstart:bstart + args.file_batch])):
            fid = bstart + off
            year = Path(files[fid]).parts[-2]
            for g in games:
                obs_l = g.take_obs()
                inv_l = g.take_invisible_obs()
                sh_l = g.take_shantens()
                turns_l = g.take_at_turns()
                if not obs_l:
                    continue
                obs = np.stack([np.asarray(o) for o in obs_l])
                inv = np.stack([np.asarray(o) for o in inv_l])
                own_sh = np.asarray(sh_l, dtype=np.int8)
                # PyO3 converts Vec<u8> to bytes
                at_turns = np.frombuffer(turns_l, dtype=np.uint8).astype(np.float32)
                labels, _ = verify_and_label(obs, inv, own_sh)
                phi = embed(brain, obs, args.chunk, device)
                with torch.inference_mode():
                    mish = torch.nn.functional.mish(torch.from_numpy(phi)).numpy()
                phi_parts.append(phi)
                mish_parts.append(mish)
                pub_parts.append(public_features(obs, at_turns))
                meld_parts.append(exact_meld_features(obs))
                for k, v in labels.items():
                    label_parts[k].append(v)
                own_sh_parts.append(own_sh)
                file_ids.append(np.full(len(obs_l), fid, dtype=np.int32))
                pos = labels[PRIMARY]
                year_pos[year][0] += int(len(pos))
                year_pos[year][1] += int(pos.sum())
                n_points += len(obs_l)
                del obs, inv, phi, mish, obs_l, inv_l, sh_l, turns_l
        if (bstart // args.file_batch) % 40 == 0:
            print(f"[supp] files {bstart + len(files[bstart:bstart + args.file_batch])}/{len(files)} points {n_points} {time.time() - t0:.0f}s", flush=True)

    phi = np.concatenate(phi_parts)
    mish = np.concatenate(mish_parts)
    pub = np.concatenate(pub_parts)
    meld = np.concatenate(meld_parts)
    labels = {k: np.concatenate(v) for k, v in label_parts.items()}
    file_ids = np.concatenate(file_ids)
    del phi_parts, mish_parts, pub_parts, meld_parts
    print(f"[supp] extraction done: {n_points} points, phi {phi.shape}, pub {pub.shape}", flush=True)

    ids = list(range(len(files)))
    random.Random(args.split_seed).shuffle(ids)
    n_val_files = max(1, int(round(args.val_frac * len(files))))
    val_files = set(ids[:n_val_files])
    train_mask = np.array([fid not in val_files for fid in file_ids], dtype=bool)
    val_mask = ~train_mask
    y = {k: v.astype(np.float32) for k, v in labels.items()}
    fit_args = SimpleNamespace(bs=args.bs, epochs=args.epochs, lr=args.lr, wd=args.wd)

    def probe(X, name, control_seed=None):
        mu = X[train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
        sd = np.maximum(X[train_mask].std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
        Xs = (X - mu) / sd
        ytr, yva = y[name][train_mask], y[name][val_mask]
        p = fit_predict(Xs[train_mask], ytr, Xs[val_mask], fit_args, device, args.seed)
        out = {"auc": auc_score(yva, p), "ece": ece_score(yva, p, 15)}
        if control_seed is not None:
            yperm = y[name][np.random.default_rng(control_seed).permutation(len(y[name]))]
            pc = fit_predict(Xs[train_mask], yperm[train_mask], Xs[val_mask], yperm[val_mask], fit_args, device, args.seed)
            out["auc_random"] = auc_score(yperm[val_mask], pc)
        del Xs
        return out

    out = {
        "schema": "a2-supplementary/v1",
        "post_hoc": True,
        "note": "Written AFTER the pre-registered run; not part of PREREG.md decision gates.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample": {"n_files": len(files), "n_points": int(n_points), "sample_seed": args.sample_seed,
                   "split_seed": args.split_seed, "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum())},
        "phi_primary_reproduction": probe(phi, PRIMARY),
        "mish_phi_primary": probe(mish, PRIMARY),
        "public_features_primary": {**probe(pub, PRIMARY), "n_features": int(pub.shape[1]), "feature_names": PUBLIC_FEATURE_NAMES},
        "phi_aka_reproduction": probe(phi, "opp_any_aka_in_hand"),
        "public_features_aka": {**probe(pub, "opp_any_aka_in_hand"), "n_features": int(pub.shape[1]), "feature_names": PUBLIC_FEATURE_NAMES},
        "exact_meld_features_control": {**probe(meld, POSITIVE_CONTROL), "n_features": int(meld.shape[1]),
                                        "note": "6 raw meld-presence planes; exactly determine the label"},
        "phi_control_meld_reproduction": probe(phi, POSITIVE_CONTROL),
        "mish_phi_control_meld": probe(mish, POSITIVE_CONTROL),
        "primary_prevalence_by_year": {y_: {"n": v[0], "pos_rate": v[1] / v[0]} for y_, v in sorted(year_pos.items())},
        "runtime": {"seconds": round(time.time() - t0, 1)},
    }
    print(json.dumps({k: out[k] for k in ("phi_primary_reproduction", "mish_phi_primary",
                                          "public_features_primary", "exact_meld_features_control",
                                          "phi_control_meld_reproduction", "mish_phi_control_meld")}, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "supplementary.json", "w") as f:
        json.dump(out, f, indent=2)

    # add a reference + headline numbers to results.json (does not touch its required keys)
    rp = args.out_dir / "results.json"
    if rp.exists():
        res = json.load(open(rp))
        res["supplementary"] = {
            "path": "supplementary.json",
            "post_hoc": True,
            "public_features_primary_auc": out["public_features_primary"]["auc"],
            "mish_phi_primary_auc": out["mish_phi_primary"]["auc"],
            "exact_meld_features_control_auc": out["exact_meld_features_control"]["auc"],
        }
        with open(rp, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[supp] wrote {rp} supplementary reference", flush=True)
    print(f"[supp] wrote {args.out_dir / 'supplementary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
