#!/usr/bin/env python3
"""H1 standalone shipping study: prepare report_sub features and evaluate exactly once.

Two subcommands:

  prepare   build `runs/exp/OPP_TENPAI/report_sub_{features.npy,targets.npz,meta.json}`
            from the canonical report_sub corpus + t2 labels.
  evaluate  score the frozen H1 head on that cache, with file-cluster bootstrap CIs and controls.
            Refuses if `metrics_report.json` already exists (one-shot).

The prereg is `scripts/exp_opp_tenpai_prereg.md`; its sha256 is recorded in every artifact.
This script imports `exp_ex1_headA` (a training/audit script); `scripts/jiangpu/**` must not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "vendor/mortal/target/release"))
sys.path.insert(0, str(REPO / "vendor/mortal"))

import exp_ex1_headA as H  # noqa: E402

PREREG = REPO / "scripts/exp_opp_tenpai_prereg.md"
LABEL_DIR = REPO / "runs/exp/EX1/labels/report_sub"
CORPUS_INDEX = REPO / "runs/exp/EX1/corpus_index.json"
CHECKPOINT = REPO / "runs/exp/EX1/headA/headA.pt"
SHUFFLE = REPO / "runs/exp/EX1/headA/headA_shuffle.pt"
OUT_DIR = REPO / "runs/exp/OPP_TENPAI"

EXPECTED_HEAD_SHA = "2dcb6d6a9baadd34b8af6a5e059a23c36ebbdd1e83e41c6bb45c290a71ad1aa5"
EXPECTED_SHUFFLE_SHA = "12de9bc028f63dc175326ac4893fdc27479709a7d0d9b691bdb17ef98a8cde87"
EXPECTED_FILES = 800
EXPECTED_DECISIONS = 129_890
EXPECTED_SLOTS = EXPECTED_DECISIONS * 3


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _jobs():
    labels = sorted(LABEL_DIR.glob("*.npz"))
    if len(labels) != EXPECTED_FILES:
        raise SystemExit(f"FATAL: expected {EXPECTED_FILES} report_sub labels, got {len(labels)}")
    jobs = []
    for i, label_path in enumerate(labels):
        with np.load(label_path) as z:
            meta = json.loads(str(z["meta_json"]))
        jobs.append({"label_path": str(label_path), "source_path": meta["source_path"],
                     "seat": int(meta["player_id"]), "file_id": i,
                     "year": int(meta["year"])})
    return jobs


def prepare(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = _jobs()
    if args.limit:
        jobs = jobs[:int(args.limit)]
    t0 = time.time()
    if args.jobs <= 1:
        results = [H._extract_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(H._extract_one, jobs))
    x = np.concatenate([r["x"] for r in results], axis=0).astype(np.float16)
    file_id = np.concatenate([np.full(r["n_real"], r["file_id"], dtype=np.int32)
                              for r in results])
    y = np.concatenate([r["opp_tenpai"] for r in results], axis=0).astype(np.uint8)
    if x.shape != (y.shape[0], H.FEATURE_DIM):
        raise SystemExit(f"FATAL: feature/label rows {x.shape} vs {y.shape}")
    feats_path = out_dir / "report_sub_features.npy"
    targets_path = out_dir / "report_sub_targets.npz"
    np.save(feats_path, x)
    np.savez(targets_path, opp_tenpai=y, file_id=file_id, n=np.int64(len(y)),
             n_files=np.int64(len(results)),
             n_decisions=np.int64(sum(r["n_decisions"] for r in results)),
             years=np.array(sorted({int(r["year"]) for r in results}), dtype=np.int64),
             exp_loss_sum=np.float64(sum(r["exp_loss_sum"] for r in results)))
    meta = {
        "schema": "opp-tenpai-report-sub-cache/v1",
        "generated_at": _now(),
        "prereg": {"path": _rel(PREREG), "sha256": _sha256(PREREG)},
        "corpus_index": {"path": _rel(CORPUS_INDEX), "sha256": _sha256(CORPUS_INDEX)},
        "checkpoint": {"path": _rel(CHECKPOINT), "sha256": _sha256(CHECKPOINT)},
        "n_files": len(results), "n_decisions_real": int(len(y)),
        "n_slots": int(y.size), "years": sorted({int(r["year"]) for r in results}),
        "features_path": _rel(feats_path), "features_sha256": _sha256(feats_path),
        "targets_path": _rel(targets_path), "targets_sha256": _sha256(targets_path),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "report_sub_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": _rel(out_dir), "n_files": len(results),
                      "n_decisions": int(len(y)), "n_slots": int(y.size),
                      "seconds": meta["wall_seconds"]}, indent=2))
    return 0


def _metrics(x, y, groups, checkpoint, device, jobs):
    import torch
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = H.HeadA()
    model.load_state_dict(ckpt["state_dict"])
    mu = np.asarray(ckpt["mu"], dtype=np.float32)
    sd = np.asarray(ckpt["sd"], dtype=np.float32)
    pred = H.predict(model, x, mu, sd, device)
    p = pred["opp_tenpai"].reshape(-1)
    fam = {"opp_tenpai": {"y": y, "p": p, "groups": groups}}
    point = H.binary_family(y, p, groups, binning="equal_width")["point"]
    ci = H.bootstrap_ci(fam, [("opp_tenpai", "auc"), ("opp_tenpai", "bss"),
                              ("opp_tenpai", "ece")], 2000, jobs)
    return point, ci


def evaluate(args) -> int:
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"FATAL: one-shot report already exists at {out}; "
                         "report_sub must not be evaluated twice (prereg §4)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats_path = Path(args.features) if args.features else out_dir / "report_sub_features.npy"
    targets_path = Path(args.targets) if args.targets else out_dir / "report_sub_targets.npz"
    for path in (feats_path, targets_path):
        if not path.exists():
            raise SystemExit(f"FATAL: missing cache {path}; run `prepare` first")

    prereg_sha = _sha256(PREREG)
    ckpt_sha = _sha256(CHECKPOINT)
    shuf_sha = _sha256(SHUFFLE)
    if ckpt_sha != EXPECTED_HEAD_SHA:
        raise SystemExit(f"FATAL: checkpoint sha {ckpt_sha} != prereg frozen {EXPECTED_HEAD_SHA}")
    if shuf_sha != EXPECTED_SHUFFLE_SHA:
        raise SystemExit(f"FATAL: shuffle checkpoint sha {shuf_sha} != {EXPECTED_SHUFFLE_SHA}")

    x = np.load(feats_path)
    with np.load(targets_path) as z:
        y2 = z["opp_tenpai"].astype(np.uint8)
        file_id = z["file_id"].astype(np.int64)
        n_files = int(z["n_files"])
        n_decisions = int(z["n_decisions"])
        years = [int(v) for v in z["years"]]
    if x.shape != (y2.shape[0], H.FEATURE_DIM):
        raise SystemExit(f"FATAL: feature shape {x.shape} vs labels {y2.shape}")
    y = y2.reshape(-1).astype(np.float64)
    groups = np.repeat(file_id, H.N_OPPS)
    if len(y) != len(groups):
        raise SystemExit("FATAL: label/group length mismatch")

    point, ci = _metrics(x, y, groups, CHECKPOINT, args.device, args.jobs)
    shuf_point, _ = _metrics(x, y, groups, SHUFFLE, args.device, args.jobs)

    primary = {
        "bss_ge_0.10": point["brier_skill"] >= 0.10,
        "auc_ge_0.75": point["auc"] >= 0.75,
        "ece15_le_0.05": point["ece_equal_width_15"] <= 0.05,
    }
    robust = {
        "bss_ci_low_ge_0.05": ci[("opp_tenpai", "bss")]["ci95"][0] >= 0.05,
        "auc_ci_low_ge_0.70": ci[("opp_tenpai", "auc")]["ci95"][0] >= 0.70,
    }
    if all(primary.values()) and all(robust.values()):
        decision = "ship_calibrated"
    elif all(primary.values()):
        decision = "ship_low_confidence"
    else:
        decision = "unavailable"

    report = {
        "schema": "opp-tenpai-ship/v1",
        "generated_at": _now(),
        "one_shot": True,
        "prereg": {"path": _rel(PREREG), "sha256": prereg_sha},
        "checkpoint": {"path": _rel(CHECKPOINT), "sha256": ckpt_sha,
                       "weights_sha256": "c9585d2164e33b1766e64abb584b12db25bf0c66f4728458bb306af6be7b426e"},
        "data": {
            "subset": "report_sub", "n_files": n_files, "n_decisions": n_decisions,
            "n_slots": int(len(y)), "years": years,
            "features_path": _rel(feats_path), "features_sha256": _sha256(feats_path),
            "targets_path": _rel(targets_path), "targets_sha256": _sha256(targets_path),
            "corpus_index_sha256": _sha256(CORPUS_INDEX),
        },
        "metrics": {"opp_tenpai_prob": {
            "point": point,
            "ci95": {k: ci[("opp_tenpai", k)] for k in ("auc", "bss", "ece")},
        }},
        "controls": {
            "base_rate": point["base_rate_control"],
            "trained_shuffle": {"checkpoint_sha256": shuf_sha, "point": shuf_point},
        },
        "pass": {"primary": primary, "robustness": robust, "decision": decision},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    if _sha256(PREREG) != prereg_sha:
        raise SystemExit("FATAL: prereg changed during the run")
    (out_dir / "PREREG.md").write_text(PREREG.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"wrote": _rel(out), "decision": decision,
                      "auc": point["auc"], "bss": point["brier_skill"],
                      "ece": point["ece_equal_width_15"],
                      "ci": report["metrics"]["opp_tenpai_prob"]["ci95"]}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="H1 opp_tenpai standalone ship study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.set_defaults(func=prepare)
    e = sub.add_parser("evaluate")
    e.add_argument("--jobs", type=int, default=8)
    e.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    e.add_argument("--features", default=None)
    e.add_argument("--targets", default=None)
    e.add_argument("--out-dir", type=Path, default=OUT_DIR)
    e.add_argument("--out", type=Path, default=OUT_DIR / "metrics_report.json")
    e.set_defaults(func=evaluate)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
