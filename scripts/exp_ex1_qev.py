#!/usr/bin/env python3
"""t5 — q_ev two-leg validity study (leg ③ deleted per BL-1).

Plan: `.omo/plans/2026-09-23-explainability-metrics-and-validity.md` t5.
PREREG: `scripts/exp_ex1_prereg.md` §3/§5/§6.1/§9 — frozen; its sha256 is recorded at start and
re-read at end (drift ⇒ FATAL, M8).

Object under audit: the shipped `q_ev` payload (`scripts/jiangpu/providers/q_ev.py`) produced by
re-inference on the frozen `report_sub` corpus with the engine's canonical checkpoint
`runs/exp/PINNED/mortal.pth.at-396000` (`scripts/jiangpu/models.py:19`). The checkpoint path and
its sha256 are stated explicitly in the artifact.

Two legs, reported separately and never merged into a composite "validity score":
  ① action consistency — `softmax(Q/tau=1)` over legal actions vs the recorded human action
     (top-1 agreement calibration; agreement is NOT correctness/optimality): ECE(15) + Brier.
  ② outcome — `risk = -logged_action_q_rel` (>= 0; larger = the model ranks the played action
     lower relative to its best legal action) vs `bad` (primary proxy: the tile played at the
     decision was ron'd by an opponent): AUC + Spearman, plus continuous realized-loss and
     rank-change diagnostics.

Leg ③ ("stronger checkpoint reference") is NOT estimated (BL-1): the engine's canonical checkpoint
IS the reference, so a reference leg would be self-comparison ⇒ `leg3_status == "not_estimable"`.

Evaluation subset = `report_sub`, read-only: no number here feeds any fit or selection (PREREG
§9.8 / C-4), and no calibration product is consumed. Labels come from the t2 oracle harness —
allowed as a validity-harness label source only (NO-REVEAL, PREREG §8).

Usage:
  source scripts/activate_env.sh
  python scripts/exp_ex1_qev.py --selftest      # PREREG-pinned synthetic checks + leg-local math
  python scripts/exp_ex1_qev.py                 # full 800-file report_sub run (~minutes, CPU)

allow: SIZE_OK — t5 mandates this exact single-file audit entry point; the body is one pipeline
(collect -> leg ① -> leg ② -> hash gates -> one frozen artifact schema). Splitting the artifact
assembly would fragment the schema and the PREREG hash discipline across files, which is worse.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections.abc import Callable

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "vendor/mortal/target/release"))
sys.path.insert(0, str(REPO / "vendor/mortal"))

import exp_ex1_labels  # noqa: E402  (the only label source; never imported by scripts/jiangpu/**)

exp_ex1_labels._bootstrap_mortal()  # t2's order-independent `mortal` namespace pin

import numpy as np  # noqa: E402

from analyze_alllast import analyze_stream, decode_events  # noqa: E402
from eval_report import SEAT_OF_SPLIT, rank_of  # noqa: E402
from exp_a2_probe import auc_score, brier_score, ece_score  # noqa: E402
from exp_a2b_calibration import (  # noqa: E402
    ece_cluster_ci,
    reliability_table,
    spearman,
    value_table,
)
from exp_ex1_audit import assert_prereg_unchanged, load_prereg  # noqa: E402
from exp_ex1_audit import selftest as audit_selftest  # noqa: E402
from exp_ex1_labels import load_targets  # noqa: E402
from jiangpu.models import CHECKPOINT_FIXTURE, checkpoint_sha256, load_bundle  # noqa: E402
from jiangpu.providers import q_ev  # noqa: E402
from jiangpu.record import read_record  # noqa: E402
from jiangpu.replay import iter_decisions  # noqa: E402

PREREG_PATH = REPO / "scripts/exp_ex1_prereg.md"
CORPUS_INDEX = REPO / "runs/exp/EX1/corpus_index.json"
LABEL_ROOT = REPO / "runs/exp/EX1/labels"
DEFAULT_OUT = REPO / "runs/exp/EX1/qev_validity.json"
SUBSET = "report_sub"
SCHEMA = "ex1-qev-validity/v1"
N_BINS = 15
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 7
SHUFFLE_SEED = 0
TIE_EPS = 1e-6
MIN_POSITIVES = 50
V1_TRIAGE, V1_DELIVERY = 0.10, 0.05
V2_TRIAGE, V2_DELIVERY = 0.55, 0.60
LEG3_REASON = (
    "no demonstrably stronger reference exists: the engine's canonical checkpoint is "
    "runs/exp/PINNED/mortal.pth.at-396000 (jiangpu/models.py:19, tests/test_cli.py:12), so a "
    "reference-checkpoint leg would compare Q against itself (degenerate, uninformative) — "
    "leg ③ deleted per BL-1"
)


def _rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _finite(x) -> float | None:
    """JSON-safe float: non-finite/undefined statistics become null, never NaN."""
    if x is None:
        return None
    v = float(x)
    return v if np.isfinite(v) else None


def _softmax_top(values) -> float:
    """Max softmax probability over a legal-action Q vector (shift-invariant; tau = 1)."""
    v = np.asarray(values, dtype=np.float64)
    e = np.exp(v - v.max())
    return float(e.max() / e.sum())


def _cluster_bootstrap_ci(groups: np.ndarray, stats: dict[str, Callable[[np.ndarray], float]],
                          n_boot: int, seed: int) -> dict[str, dict]:
    """95 % file-cluster bootstrap CI (resample source files with replacement, PREREG §5).

    One shared resampling serves every statistic. Draws that are not finite (e.g. a resample with
    a single class) are dropped; fewer than half the draws valid ⇒ CI = null.
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")   # rows grouped by file, block order
    lens = np.bincount(inv, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_boot, dtype=np.float64) for name in stats}
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        idx = np.repeat(order, np.repeat(counts, lens))
        for name, fn in stats.items():
            draws[name][b] = fn(idx)
    out = {}
    for name, vals in draws.items():
        ok = np.isfinite(vals)
        if int(ok.sum()) >= max(1, n_boot // 2):
            lo, hi = np.percentile(vals[ok], [2.5, 97.5])
            ci = [float(lo), float(hi)]
        else:
            ci = None
        out[name] = {"ci95": ci, "n_valid_draws": int(ok.sum()), "n_draws": int(n_boot)}
    return out


def _stat(point, ci: dict) -> dict:
    return {"point": _finite(point), "ci95_file_cluster": ci["ci95"],
            "n_valid_draws": ci["n_valid_draws"], "n_draws": ci["n_draws"]}


def _verdict(value, triage: float, delivery: float, lower_is_better: bool) -> dict:
    """Pre-registered triage/delivery verdict (PREREG §6.1; lines are never relaxed)."""
    if value is None:
        return {"triage_bar": triage, "delivery_bar": delivery, "triage_met": None,
                "delivery_met": None, "verdict": "not_estimable"}
    triage_met = value <= triage if lower_is_better else value >= triage
    delivery_met = value <= delivery if lower_is_better else value >= delivery
    if delivery_met:
        verdict = "passes_delivery"
    elif triage_met:
        verdict = "passes_triage_only"
    else:
        verdict = "negative_result"
    return {"triage_bar": triage, "delivery_bar": delivery, "triage_met": bool(triage_met),
            "delivery_met": bool(delivery_met), "verdict": verdict}


def _load_subset(index_path: pathlib.Path, subset: str) -> tuple[dict, list[dict]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if subset not in index.get("subsets", {}):
        raise SystemExit(f"FATAL: {index_path} has no subset '{subset}'")
    meta = index["subsets"][subset]
    return meta, list(meta["files"])


def _collect(files: list[dict], device: str, max_files: int | None) -> dict:
    """Re-infer Q over the subset and collect per-decision arrays + t2 labels (one pass)."""
    used = files[:max_files] if max_files else files
    bundle = load_bundle(CHECKPOINT_FIXTURE, device=device)
    conf, match, risk, bad, loss, bad_rank, real, file_ids = [], [], [], [], [], [], [], []
    n_decisions = n_kan = 0
    for fi, entry in enumerate(used):
        target = REPO / entry["target_path"]
        rec = read_record(str(target))
        if (rec.seed, rec.key, rec.split) != (entry["seed"], entry["key"], entry["split"]):
            raise SystemExit(f"FATAL: corpus identity mismatch for {target}")
        pov = SEAT_OF_SPLIT[rec.split]
        decs = iter_decisions(rec, pov=pov)
        labels = load_targets(LABEL_ROOT / SUBSET / f"{rec.seed}_{rec.key}_{rec.split}.npz")
        acts = np.asarray([d.logged_action for d in decs], dtype=np.int64)
        if len(decs) != int(labels["n"]) or not np.array_equal(acts, labels["actions"]):
            raise SystemExit(f"FATAL: decision/label misalignment for {target}")
        payloads = q_ev.compute(bundle, decs, device=device)
        final_rank = rank_of(analyze_stream(rec.path, decode_events(rec.path))["final_scores"], pov)
        rank_by_kyoku: dict[int, int] = {}
        for dec in decs:
            payload = payloads[dec.index]
            gap = float(payload["logged_action_q_rel"])
            conf.append(_softmax_top(list(payload["q_rel"].values())))
            match.append(gap >= -TIE_EPS)
            risk.append(-gap)
            if dec.at_kyoku not in rank_by_kyoku:
                rank_by_kyoku[dec.at_kyoku] = rank_of(dec.public["scores"], pov)
            bad_rank.append(final_rank > rank_by_kyoku[dec.at_kyoku])
        bad.extend(labels["deal_in"].sum(axis=(1, 2)) > 0)
        loss.extend(labels["exp_loss"].sum(axis=(1, 2)))
        real.extend(labels["real_mask"])
        file_ids.extend([fi] * len(decs))
        n_decisions += len(decs)
        n_kan += int((~labels["real_mask"]).sum())
        if (fi + 1) % 100 == 0 or fi + 1 == len(used):
            print(f"[qev] files {fi + 1}/{len(used)} decisions {n_decisions}", flush=True)
    return {
        "conf": np.asarray(conf, dtype=np.float64),
        "match": np.asarray(match, dtype=bool),
        "risk": np.asarray(risk, dtype=np.float64),
        "bad": np.asarray(bad, dtype=bool),
        "loss": np.asarray(loss, dtype=np.float64),
        "bad_rank": np.asarray(bad_rank, dtype=bool),
        "real": np.asarray(real, dtype=bool),
        "file_ids": np.asarray(file_ids, dtype=np.int64),
        "n_files": len(used),
        "n_decisions_total": n_decisions,
        "n_kan_select_excluded": n_kan,
        "checkpoint": {"path": _rel(pathlib.Path(CHECKPOINT_FIXTURE)), "sha256": bundle.sha256,
                       "steps": bundle.steps, "device": device},
    }


def _base_artifact(sha_start: str, args, subset_meta: dict, files_used: int,
                   n_decisions_total: int, checkpoint: dict) -> dict:
    return {
        "schema": SCHEMA,
        "plan": ".omo/plans/2026-09-23-explainability-metrics-and-validity.md",
        "prereg": {"path": _rel(PREREG_PATH), "sha256_start": sha_start,
                   "sha256_end": None, "immutable_during_run": None},
        "evaluation_subset": {
            "name": args.subset,
            "corpus_index": _rel(args.corpus_index.resolve()),
            "subset_sha256": subset_meta.get("subset_sha256"),
            "n_files": len(subset_meta["files"]),
            "files_used": files_used,
            "subset_complete": files_used == len(subset_meta["files"]),
            "n_decisions_total": n_decisions_total,
            "read_only": True,
            "used_for_fitting": False,
            "used_for_selection": False,
            "calibration_product_consumed": None,
        },
        "checkpoint": checkpoint,
        "leg3_status": "not_estimable",
        "leg3_reason": LEG3_REASON,
        "no_composite_score": True,
    }


def _not_estimable(artifact: dict, reason: str) -> dict:
    artifact.update({
        "status": "not_estimable",
        "reason": reason,
        "leg1_action_consistency": {"status": "not_estimable", "reason": reason},
        "leg2_outcome": {"status": "not_estimable", "reason": reason},
        "verdicts": {},
    })
    return artifact


def _run_study(files: list[dict], args, subset_meta: dict, sha_start: str) -> dict:
    data = _collect(files, args.device, args.max_files)
    artifact = _base_artifact(sha_start, args, subset_meta, data["n_files"],
                              data["n_decisions_total"], data["checkpoint"])
    real = data["real"]
    n_total, n_real = int(len(real)), int(real.sum())
    artifact["evaluation_subset"]["n_decisions_real"] = n_real
    artifact["evaluation_subset"]["n_kan_select_excluded"] = data["n_kan_select_excluded"]
    if n_real == 0:
        return _not_estimable(artifact, "evaluation subset contains no real (non-kan-select) decisions")

    conf, match = data["conf"][real], data["match"][real]
    risk, bad, loss = data["risk"][real], data["bad"][real], data["loss"][real]
    bad_rank, groups = data["bad_rank"][real], data["file_ids"][real]

    # ------------------------------------------------------------------ leg ① (agreement, not correctness)
    ece = float(ece_score(match, conf, N_BINS))
    ece_ci = ece_cluster_ci(conf, match, groups, args.bootstrap, BOOTSTRAP_SEED)
    ci = _cluster_bootstrap_ci(groups, {
        "brier": lambda idx: brier_score(match[idx], conf[idx]),
        "auc": lambda idx: auc_score(bad[idx], risk[idx]),
        "spearman_bad": lambda idx: spearman(risk[idx], bad[idx].astype(np.float64)),
        "spearman_loss": lambda idx: spearman(risk[idx], loss[idx]),
        "auc_rank_change": lambda idx: auc_score(bad_rank[idx], risk[idx]),
    }, args.bootstrap, BOOTSTRAP_SEED)

    rng = np.random.default_rng(SHUFFLE_SEED)
    ece_shuffled = float(ece_score(rng.permutation(match), conf, N_BINS))
    if not ece_shuffled > ece:
        raise SystemExit(
            f"FATAL: leg-1 shuffle control did not degrade ECE (shuffled {ece_shuffled:.6f} <= "
            f"real {ece:.6f}); the calibration signal is not distinguishable from the marginal "
            "confidence distribution (PREREG §5/§9.7)")

    artifact["leg1_action_consistency"] = {
        "definition": "softmax(Q/tau=1) over legal actions vs the action recorded in the log; "
                      "top-1 agreement calibration (agreement, NOT correctness/optimality)",
        "tau": 1.0,
        "n": n_real,
        "match_rate": _finite(match.mean()),
        "mean_confidence": _finite(conf.mean()),
        "ece15": {"point": _finite(ece), "ci95_file_cluster": [float(x) for x in ece_ci],
                  "n_valid_draws": args.bootstrap, "n_draws": args.bootstrap},
        "brier": _stat(brier_score(match, conf), ci["brier"]),
        "reliability": [
            {"bin": r["bin"], "lo": r["lo"], "hi": r["hi"], "n": r["n"], "weight": r["weight"],
             "mean_conf": r["mean_conf"], "acc": r["acc"], "gap": r["gap"]}
            for r in reliability_table(conf, match, N_BINS)
        ],
        "all_decisions_sensitivity": {
            "note": "kan-select pseudo-decisions included; informational only, not a PREREG line",
            "n": n_total,
            "ece15": _finite(ece_score(data["match"], data["conf"], N_BINS)),
            "brier": _finite(brier_score(data["match"], data["conf"])),
        },
        "shuffle_negative_control": {
            "seed": SHUFFLE_SEED, "ece15_shuffled": _finite(ece_shuffled),
            "delta_vs_real": _finite(ece_shuffled - ece), "passed": bool(ece_shuffled > ece),
        },
    }

    # ------------------------------------------------------------------ leg ② (outcome)
    n_pos = int(bad.sum())
    if n_pos < MIN_POSITIVES:
        artifact["leg2_outcome"] = {
            "status": "not_estimable",
            "reason": f"primary bad proxy has {n_pos} positives (< {MIN_POSITIVES}); the PREREG §3 "
                      "secondary proxy (kyoku point-delta < 0) is not materialized in the t2 label "
                      "schema, so the switch cannot be executed in t5 (PREREG §9.4 fail-closed)",
        }
        artifact["verdicts"] = {"V1": _verdict(ece, V1_TRIAGE, V1_DELIVERY, True)}
        artifact["status"] = "not_estimable"
        return artifact

    auc = float(auc_score(bad, risk))
    if not np.isfinite(auc):
        artifact["leg2_outcome"] = {"status": "not_estimable",
                                    "reason": "AUC undefined (single class)"}
        artifact["verdicts"] = {"V1": _verdict(ece, V1_TRIAGE, V1_DELIVERY, True)}
        artifact["status"] = "not_estimable"
        return artifact

    rng = np.random.default_rng(SHUFFLE_SEED)
    auc_shuffled = float(auc_score(rng.permutation(bad), risk))
    if not abs(auc_shuffled - 0.5) < 0.02:
        raise SystemExit(
            f"FATAL: shuffle negative control failed: |AUC-0.5|={abs(auc_shuffled - 0.5):.4f} "
            ">= 0.02 (PREREG §5/§9.7); no numbers are reported until the pipeline is fixed")

    artifact["leg2_outcome"] = {
        "definition": "risk = -logged_action_q_rel (>= 0; larger = the model ranks the played "
                      "action lower relative to its best legal action)",
        "n": n_real,
        "bad_proxy": {
            "used": "primary",
            "definition": "1[the tile we played at d was ron'd by any opponent]",
            "n_positive": n_pos,
            "prevalence": _finite(n_pos / n_real),
            "fallback_triggered": False,
            "fallback_min_positives": MIN_POSITIVES,
        },
        "risk_summary": {"mean": _finite(risk.mean()), "std": _finite(risk.std()),
                         "min": _finite(risk.min()), "max": _finite(risk.max()),
                         "n_zero": int((risk == 0.0).sum())},
        "auc_vs_bad": _stat(auc, ci["auc"]),
        "spearman_vs_bad": _stat(spearman(risk, bad.astype(np.float64)), ci["spearman_bad"]),
        "spearman_vs_realized_loss": _stat(spearman(risk, loss), ci["spearman_loss"]),
        "auc_vs_rank_change_diagnostic": {
            "note": "final rank worse than the rank at kyoku start; game-level diagnostic, not a "
                    "PREREG line",
            **_stat(auc_score(bad_rank, risk), ci["auc_rank_change"]),
        },
        "value_binned_calibration_diagnostic": value_table(risk, loss, N_BINS),
        "shuffle_negative_control": {
            "seed": SHUFFLE_SEED, "auc_shuffled": _finite(auc_shuffled),
            "abs_delta_from_half": _finite(abs(auc_shuffled - 0.5)), "passed": True,
        },
    }

    artifact["verdicts"] = {
        "V1": {**_verdict(ece, V1_TRIAGE, V1_DELIVERY, True),
               "primary_statistic": "ECE(15) of top-1 agreement confidence", "value": _finite(ece)},
        "V2": {**_verdict(auc, V2_TRIAGE, V2_DELIVERY, False),
               "primary_statistic": "AUC(risk, bad primary proxy)", "value": _finite(auc)},
    }
    artifact["status"] = "measured"
    artifact["notes"] = [
        "consistency is agreement with the recorded human action, never correctness/optimality",
        "no composite validity score is produced; the two legs are separate (M5)",
        "every number is recomputed from raw data on report_sub; no A2b number is reused (MA-4: no "
        "cross-experiment anchor is cited)",
        "the primary bad proxy has >= 50 positives, so the PREREG §3 secondary-proxy switch is not "
        "triggered",
        "labels are oracle-derived (t2) and are a validity-harness label source only (NO-REVEAL)",
        "checkpoint re-hashed after the run; a mid-run change is fatal",
    ]
    return artifact


def _selftest() -> dict:
    """PREREG-pinned synthetic checks (t1) + leg-local math on known-value fixtures."""
    checks = audit_selftest()
    v = np.array([0.0, -1.0, -2.0])
    conf = _softmax_top(v)
    ref = float(1.0 / np.exp(v).sum())
    assert abs(conf - ref) < 1e-12, f"softmax path {conf} != {ref}"
    y = np.concatenate([np.zeros(500, dtype=bool), np.ones(500, dtype=bool)])
    s = np.concatenate([np.zeros(500), np.ones(500)])
    assert auc_score(y, s) == 1.0, "separable AUC must be 1.0"
    rng = np.random.default_rng(SHUFFLE_SEED)
    auc_shuf = float(auc_score(rng.permutation(y), s))
    assert abs(auc_shuf - 0.5) < 0.02, f"shuffle AUC {auc_shuf} outside 0.5 +/- 0.02"
    groups = np.zeros(len(y), dtype=np.int64)
    ci = _cluster_bootstrap_ci(groups, {"auc": lambda idx: auc_score(y[idx], s[idx])}, 20,
                               BOOTSTRAP_SEED)
    assert ci["auc"]["ci95"] == [1.0, 1.0], f"single-cluster CI must collapse: {ci}"
    return {"selftest": "PASS", "audit_selftest": checks, "softmax_conf": conf,
            "auc_separable": 1.0, "auc_shuffled": auc_shuf,
            "single_cluster_ci": ci["auc"]["ci95"]}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="EX1 t5: q_ev two-leg validity study")
    ap.add_argument("--corpus-index", type=pathlib.Path, default=CORPUS_INDEX)
    ap.add_argument("--subset", default=SUBSET)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def _sanitize(obj):
    """Recursively replace non-finite floats with null so the artifact is strict JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write(path: pathlib.Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(artifact), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    sha_start = load_prereg(PREREG_PATH)

    if args.selftest:
        out = _selftest()
        sha_end = assert_prereg_unchanged(PREREG_PATH, sha_start)
        out["prereg"] = {"path": _rel(PREREG_PATH), "sha256": sha_start,
                         "unchanged": sha_start == sha_end}
        print(json.dumps(out, indent=2))
        return 0

    subset_meta, files = _load_subset(args.corpus_index.resolve(), args.subset)
    if not files:
        sha_end = assert_prereg_unchanged(PREREG_PATH, sha_start)
        artifact = _not_estimable(
            _base_artifact(sha_start, args, subset_meta, 0, 0,
                           {"path": _rel(pathlib.Path(CHECKPOINT_FIXTURE)), "sha256": None,
                            "steps": None, "device": args.device}),
            f"evaluation subset '{args.subset}' is empty (0 files) — nothing to evaluate")
        artifact["prereg"]["sha256_end"] = sha_end
        artifact["prereg"]["immutable_during_run"] = sha_start == sha_end
        _write(args.out, artifact)
        print(f"[done] {_rel(args.out)}: not_estimable (empty subset)")
        return 0

    print(f"[qev] subset {args.subset}: {len(files)} files, device {args.device}")
    artifact = _run_study(files, args, subset_meta, sha_start)

    # end-of-run gates: PREREG drift and checkpoint drift are both fatal (M8)
    sha_end = assert_prereg_unchanged(PREREG_PATH, sha_start)
    artifact["prereg"]["sha256_end"] = sha_end
    artifact["prereg"]["immutable_during_run"] = sha_start == sha_end
    sha_now = checkpoint_sha256(CHECKPOINT_FIXTURE)
    if artifact["checkpoint"]["sha256"] is not None and sha_now != artifact["checkpoint"]["sha256"]:
        raise SystemExit(f"FATAL: checkpoint changed during the run: {CHECKPOINT_FIXTURE}")
    artifact["checkpoint"]["immutable_during_run"] = True

    _write(args.out, artifact)
    print(f"[done] wrote {_rel(args.out)}: status={artifact['status']}")
    if artifact["status"] == "measured":
        print(json.dumps({"V1_ece15": artifact["verdicts"]["V1"]["value"],
                          "V1_verdict": artifact["verdicts"]["V1"]["verdict"],
                          "V2_auc": artifact["verdicts"]["V2"]["value"],
                          "V2_verdict": artifact["verdicts"]["V2"]["verdict"],
                          "shuffle_auc": artifact["leg2_outcome"]["shuffle_negative_control"]["auc_shuffled"]},
                         indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
