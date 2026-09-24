#!/usr/bin/env python3
"""t7: rank_dist payload validity study on the engine-delivered payloads (not a model rerun).

Object: the 4x4 `matrix` + `expected_rank` shipped by `jiangpu.providers.rank_dist` through
`jiangpu.cli.build_document` (post `contract.round_floats`) over frozen `report_sub`; truth =
`Grp.take_rank_by_player()`. C-4: read-only evaluation, nothing here feeds a fit or selection.
"""
# allow: SIZE_OK - plan t7 mandates this single script (the commit touches only this file); the
# 54-line `run()` return literal is a pure-data table (frozen rank_validity.json schema), and
# code-only pure LOC is 221 (< 250).
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "vendor/mortal/target/release"))

from exp_ex1_audit import (  # noqa: E402
    _bootstrap_mortal_paths, assert_prereg_unchanged, load_prereg)

_bootstrap_mortal_paths()
from exp_a2b_calibration import (  # noqa: E402
    N_BINS, brier_score, ece_cluster_ci, ece_score, reliability_table,
    selftest as a2b_selftest, value_table)

PREREG_PATH = REPO / "scripts/exp_ex1_prereg.md"
CORPUS_INDEX = REPO / "runs/exp/EX1/corpus_index.json"
OUT_PATH = REPO / "runs/exp/EX1/rank_validity.json"
CKPT = str(REPO / "runs/exp/PINNED/mortal.pth.at-396000")
GRP_CKPT = str(REPO / "checkpoints/grp.pth")
PLAN = ".omo/plans/2026-09-23-explainability-metrics-and-validity.md"
BOOT_B, BOOT_SEED, SHUFFLE_SEED, ROW_SUM_TOL = 2000, 7, 0, 1e-4
V4_LINES = {"marginal_triage": 0.10, "marginal_delivery": 0.05, "top1_triage": 0.15,
            "top1_delivery": 0.10}  # PREREG §6.1 V4 (frozen; never relaxed post hoc)


def _worker_init() -> None:
    import torch

    torch.set_num_threads(1)


def _analyze_file(path: str) -> dict:
    """Run the real engine on one record and keep the delivered rank_dist payloads."""
    from jiangpu.cli import build_document
    from jiangpu.contract import validate
    from jiangpu.record import read_record
    from libriichi.dataset import Grp

    rec = read_record(path)
    doc = build_document(path, CKPT, device="cpu")
    truth = [int(v) for v in np.frombuffer(
        Grp.load_log(rec.raw_text).take_rank_by_player(), dtype=np.uint8)]
    payloads, unavailable = [], []
    for dec in doc["decisions"]:
        payload = (dec.get("metrics") or {}).get("rank_dist")
        if payload is None or payload.get("unavailable"):
            reason = str(payload.get("reason")) if payload else "provider 未返回该决策点"
            unavailable.append({"decision": dec["index"], "reason": reason})
            continue
        assert payload["basis"] == "public_info_predictor", payload["basis"]
        assert payload["no_hidden_info"] is True and payload["unit"] == "probability"
        payloads.append({
            "at_kyoku": int(dec["at_kyoku"]), "subject_player": int(payload["subject_player"]),
            "matrix": payload["matrix"], "expected_rank": float(payload["expected_rank"]),
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "provider_version": int(payload["provider_version"])})
    return {"path": path, "split": rec.split, "truth": truth, "payloads": payloads,
            "unavailable": unavailable, "contract_errors": validate(doc),
            "decisions": len(doc["decisions"]), "schema_version": doc["schema_version"],
            "engine_version": doc["engine_version"]}


def _collect(results: list) -> tuple:
    """Pool payloads into (matrix, truth, expected_rank, subject, file_id, first_of_kyoku)."""
    mats, truth, exp_rank, subj, fid, first = [], [], [], [], [], []
    for fi, res in enumerate(results):
        t = np.asarray(res["truth"], dtype=np.int64)
        seen = set()
        for p in res["payloads"]:
            mats.append(p["matrix"])
            truth.append(t)
            exp_rank.append(p["expected_rank"])
            subj.append(p["subject_player"])
            fid.append(fi)
            first.append(p["at_kyoku"] not in seen)
            seen.add(p["at_kyoku"])
    return (np.asarray(mats, dtype=np.float64), np.asarray(truth, dtype=np.int64),
            np.asarray(exp_rank, dtype=np.float64), np.asarray(subj, dtype=np.int64),
            np.asarray(fid, dtype=np.int64), np.asarray(first, dtype=bool))


def _block(conf: np.ndarray, hit: np.ndarray, groups: np.ndarray) -> dict:
    """Point ECE/Brier + PREREG §5 file-cluster bootstrap CI (B=2000, seed 7)."""
    return {"n": int(len(conf)), "ece": ece_score(hit, conf, N_BINS), "brier": brier_score(hit, conf),
            "ci95": ece_cluster_ci(conf, hit, groups, n_boot=BOOT_B, seed=BOOT_SEED)}


def _row_sum_errors(matrix: list) -> list:
    """Real contract row-sum branch on a minimal doc (other required keys intentionally absent)."""
    from jiangpu import contract

    return contract.validate({"decisions": [{"metrics": {"rank_dist": {"matrix": matrix}}}]})


def _selftest() -> dict:
    """PREREG-pinned synthetic checks: shared ECE routines + the real row-sum contract gate."""
    a2b = a2b_selftest()
    assert a2b["perfect_ece"] < 1e-6 and abs(a2b["known_miscal_ece"] - 0.2) < 1e-12
    assert a2b["top1_perfect_ece"] < 1e-6 and a2b["brier_sanity"] == 0.0
    good = _row_sum_errors([[0.25] * 4] * 4)
    bad = _row_sum_errors([[0.5] * 4] * 4)
    assert not any("row 和应为 1" in e for e in good), good
    err = next(e for e in bad if "row 和应为 1" in e)
    return {"a2b_ece_checks": {k: a2b[k] for k in ("perfect_ece", "known_miscal_ece",
                                                   "top1_perfect_ece", "brier_sanity")},
            "row_sum_positive_ok": True, "row_sum_negative_rejected": True,
            "row_sum_negative_error": err}


def _stats(results: list) -> dict:
    """All V4 statistics over the pooled delivered payloads (PREREG §5 definitions)."""
    mats, truth, exp_rank, subj, fid, first = _collect(results)
    n = len(mats)
    assert n > 0, "no delivered rank_dist payloads on report_sub"
    row_dev = float(np.abs(mats.sum(axis=2) - 1.0).max())
    assert row_dev < ROW_SUM_TOL, f"row-sum deviation {row_dev} >= {ROW_SUM_TOL}"

    p16 = mats.reshape(-1)  # 16-cell marginal: pooled one-vs-rest over (player, rank)
    y16 = np.stack([(truth == r) for r in range(4)], axis=2).astype(np.float64).reshape(-1)
    marginal = _block(p16, y16, np.repeat(fid, 16))
    marginal["definition"] = ("pooled one-vs-rest over all 16 (player, rank) cells of every "
                              "delivered payload; truth 1{rank_by_player[player] == rank}")
    marginal["reliability"] = reliability_table(p16, y16, N_BINS)
    marginal["per_kyoku_dedup_ece"] = ece_score(
        y16.reshape(n, 16)[first].reshape(-1), p16.reshape(n, 16)[first].reshape(-1), N_BINS)

    rows_s = mats[np.arange(n), subj]
    rank_s = truth[np.arange(n), subj]
    conf_s = rows_s.max(axis=1)
    hit_s = (rows_s.argmax(axis=1) == rank_s).astype(np.float64)
    top1 = _block(conf_s, hit_s, fid)
    top1.update({"definition": ("subject player's delivered row: argmax rank vs game-final rank; "
                                "confidence = max row probability (the `expected_rank` row)"),
                 "accuracy": float(hit_s.mean()), "mean_confidence": float(conf_s.mean()),
                 "per_kyoku_dedup_ece": ece_score(hit_s[first], conf_s[first], N_BINS)})

    conf_a = np.concatenate([mats[:, pl].max(axis=1) for pl in range(4)])
    hit_a = np.concatenate([(mats[:, pl].argmax(axis=1) == truth[:, pl]).astype(np.float64)
                            for pl in range(4)])
    top1_all = _block(conf_a, hit_a, np.concatenate([fid] * 4))
    top1_all.update({"definition": "all four delivered rows pooled: argmax rank vs that player's "
                                    "final rank",
                     "accuracy": float(hit_a.mean()), "mean_confidence": float(conf_a.mean())})

    value = value_table(exp_rank, (rank_s + 1).astype(np.float64), N_BINS)
    value["definition"] = ("expected_rank (1=1st) vs realized final rank of the subject, 15 "
                           "equal-count bins; diagnostic, not a gate")

    perm = np.random.default_rng(SHUFFLE_SEED).permutation(n)
    shuffle_marginal = ece_score(y16.reshape(n, 16)[perm].reshape(-1), p16, N_BINS)
    shuffle_top1 = ece_score(hit_s[perm], conf_s, N_BINS)
    constant_ece = ece_score(y16, np.full_like(p16, 0.25), N_BINS)
    assert shuffle_marginal > marginal["ece"] and shuffle_top1 > top1["ece"], "shuffle no power"
    assert constant_ece < 1e-9, constant_ece
    for blk in (marginal, top1, top1_all):
        assert blk["ci95"][0] <= blk["ece"] <= blk["ci95"][1], blk
    return {"n": n, "row_dev": row_dev, "marginal": marginal, "top1": top1, "top1_all": top1_all,
            "value": value, "shuffle_marginal": shuffle_marginal, "shuffle_top1": shuffle_top1,
            "constant_ece": constant_ece}


def _run_files(files: list, args, t0: float) -> list:
    results = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("fork"),
                                 initializer=_worker_init) as ex:
            for i, res in enumerate(ex.map(_analyze_file, files, chunksize=2), 1):
                results.append(res)
                if i % 50 == 0:
                    print(f"  {i}/{len(files)} files ({time.time() - t0:.0f}s)", flush=True)
    else:
        _worker_init()
        for i, path in enumerate(files, 1):
            results.append(_analyze_file(path))
            if i % 25 == 0:
                print(f"  {i}/{len(files)} files ({time.time() - t0:.0f}s)", flush=True)
    return sorted(results, key=lambda r: r["path"])


def run(args) -> dict:
    sha_start = load_prereg(PREREG_PATH)
    subset = json.loads(CORPUS_INDEX.read_text(encoding="utf-8"))["subsets"]["report_sub"]
    files = [f["target_path"] for f in subset["files"]]
    if args.limit:
        files = files[: args.limit]
    t0 = time.time()
    results = _run_files(files, args, t0)
    wall = time.time() - t0

    contract_errors = [e for r in results for e in r["contract_errors"]]
    assert not contract_errors, f"engine document failed contract.validate: {contract_errors[:3]}"
    unavailable = [(r["path"], u) for r in results for u in r["unavailable"]]
    ckpt_shas = {p["checkpoint_sha256"] for r in results for p in r["payloads"]}
    assert len(ckpt_shas) == 1, ckpt_shas
    assert len({(r["schema_version"], r["engine_version"]) for r in results}) == 1
    stats = _stats(results)
    marginal, top1, top1_all, value = (stats["marginal"], stats["top1"], stats["top1_all"],
                                       stats["value"])

    m_tri = marginal["ece"] <= V4_LINES["marginal_triage"]
    m_del = marginal["ece"] <= V4_LINES["marginal_delivery"]
    t_tri = top1["ece"] <= V4_LINES["top1_triage"]
    t_del = top1["ece"] <= V4_LINES["top1_delivery"]
    status = "delivery" if (m_del and t_del) else "triage" if (m_tri and t_tri) else "below_triage"
    digest = hashlib.sha256(json.dumps([[r["path"], r["payloads"]] for r in results],
                                       sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    grp_sha = hashlib.sha256(Path(GRP_CKPT).read_bytes()).hexdigest()
    sha_end = assert_prereg_unchanged(PREREG_PATH, sha_start)

    return {
        "schema": "ex1-rank-validity/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": PLAN,
        "prereg": {"path": "scripts/exp_ex1_prereg.md", "sha256_start": sha_start,
                   "sha256_end": sha_end, "immutable_during_run": sha_start == sha_end},
        "object": {
            "metric": "rank_dist", "provider": "rank_dist", "unit": "probability",
            "provider_version": results[0]["payloads"][0]["provider_version"],
            "engine_version": results[0]["engine_version"],
            "schema_version": results[0]["schema_version"], "kind": "state_level",
            "basis": "public_info_predictor", "no_hidden_info": True, "device": "cpu",
            "checkpoint": {"path": "runs/exp/PINNED/mortal.pth.at-396000",
                           "sha256": ckpt_shas.pop()},
            "grp_checkpoint": {"path": "checkpoints/grp.pth", "sha256": grp_sha},
            "payload_source": "engine-delivered: jiangpu.cli.build_document over report_sub "
                              "(rank_dist payload extracted post contract.round_floats)",
            "truth": "Grp.take_rank_by_player() (game-final rank, 0=1st) via the same libriichi "
                     "loader the provider uses"},
        "sample": {
            "subset": "report_sub", "subset_sha256": subset["subset_sha256"], "read_only": True,
            "subset_role": subset["role"], "used_for_selection_or_fit": False,
            "files": len(results), "files_requested": len(files), "limit": args.limit,
            "years": subset["year_distribution"], "splits": subset["splits"],
            "decisions": sum(r["decisions"] for r in results), "payloads": stats["n"],
            "rank_dist_unavailable": len(unavailable),
            "unavailable_reasons": sorted({u["reason"] for _, u in unavailable})[:5],
            "payload_sha256": digest, "row_sum_max_abs_dev": stats["row_dev"],
            "row_sum_tol": ROW_SUM_TOL, "wall_seconds": round(wall, 2), "jobs": args.jobs},
        "marginal_16cell": marginal, "top1_subject": top1, "top1_all_players": top1_all,
        "expected_rank": value,
        "negative_controls": {
            "shuffle_seed": SHUFFLE_SEED, "shuffle_marginal_ece": stats["shuffle_marginal"],
            "shuffle_top1_ece": stats["shuffle_top1"], "constant_uniform_ece": stats["constant_ece"],
            "constant_note": "constant 0.25 has ECE 0 by construction: ECE is calibration, not "
                             "discrimination -- read beside top-1 accuracy and Spearman",
            "pass": True},
        "verdict": {
            "id": "V4", "prereg_lines": V4_LINES, "status": status,
            "marginal_triage_pass": m_tri, "marginal_delivery_pass": m_del,
            "top1_triage_pass": t_tri, "top1_delivery_pass": t_del,
            "discrimination_companions": {
                "top1_accuracy": top1["accuracy"], "top1_chance": 0.25,
                "top1_all_accuracy": top1_all["accuracy"], "expected_rank_mae": value["mae"],
                "expected_rank_spearman": value["spearman"],
                "expected_rank_value_ce_over_std":
                    value["value_binned_calibration_error_over_std"]},
            "note": "lines are PREREG-frozen; ECE alone is not skill (constant predictor ECE=0). "
                    "No number here feeds any fit or selection (C-4)."},
        "assertions": {"row_sum_within_tol": True, "contract_validate_all_clean": True,
                       "payload_keys_basis_no_hidden_info": True, "single_checkpoint_sha": True,
                       "ci_contains_point_estimate": True, "shuffle_control_has_power": True,
                       "constant_control_is_constant": True},
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="EX1 t7: rank_dist payload validity study")
    ap.add_argument("--limit", type=int, default=None, help="first N report_sub files (smoke)")
    ap.add_argument("--jobs", type=int, default=8, help="worker processes (default 8)")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--selftest", action="store_true", help="synthetic checks, no corpus run")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest:
        print(json.dumps({"selftest": "PASS", **_selftest()}, indent=2))
        return 0
    if args.jobs < 1:
        raise SystemExit("FATAL: --jobs must be >= 1")
    artifact = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] wrote {out}")
    print(json.dumps({"files": artifact["sample"]["files"],
                      "payloads": artifact["sample"]["payloads"],
                      "marginal_ece": artifact["marginal_16cell"]["ece"],
                      "marginal_ci95": artifact["marginal_16cell"]["ci95"],
                      "top1_ece": artifact["top1_subject"]["ece"],
                      "top1_ci95": artifact["top1_subject"]["ci95"],
                      "row_sum_max_abs_dev": artifact["sample"]["row_sum_max_abs_dev"],
                      "status": artifact["verdict"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
