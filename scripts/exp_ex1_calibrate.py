#!/usr/bin/env python3
"""EX1 t10 — calibration product: fit `gap_threshold` / `regret_blunder_threshold` on calib_sub.

Plan: `.omo/plans/2026-09-23-explainability-metrics-and-validity.md` todo 10.
PREREG: `scripts/exp_ex1_prereg.md` §7 (threshold fitting protocol), §3 (bad proxies), §2 (the
fit subset is pinned to `calib_sub`; `report_sub` is never used for fitting), §5 (file-cluster
bootstrap). The PREREG sha256 is recorded at start and re-read at end (drift ⇒ FATAL, M8).

Products (both under the gitignored `runs/exp/EX1/`):

  * `calibration_product.json` — **flat top-level floats** `gap_threshold` and
    `regret_blunder_threshold`, plus a sibling `derivation` block (fit subset, rule, status, CI,
    bad proxy, sample size). The flat schema is a hard requirement: `providers/q_ev.py:39` and
    `providers/regret.py:19` do `float(calibration[key])`, so a nested object is a runtime
    `TypeError` (MA-1). Nothing here is hardcoded: both thresholds are data-derived (D7).
  * `calib_sub_curves.json` — this todo's own `calib_sub` derivation inputs (NB-5): the per-
    decision arrays (`|logged_action_q_rel|`, `delta_rel`, both `bad` proxies, file clusters), the
    two candidate-grid curves, population counts and the behavior-check transcript. t5–t8 cover
    `report_sub` only; none of their numbers is reused for fitting (C-4).

Fitting protocol (PREREG §7, verbatim):

  * `gap_threshold` — candidate grid = deciles 10..90 of `|logged_action_q_rel|` on `calib_sub`;
    smallest candidate `g` such that the outcome-AUC (V2 definition: `AUC(bad, |gap|)`) of the
    `|gap| < g` decisions lies inside `[0.45, 0.55]` **and** the `|gap| >= g` decisions have
    AUC >= 0.60. Fallback = median `|gap|`, `status = "fallback_median_no_separation"`.
  * `regret_blunder_threshold` — candidate grid = percentiles 1..20 of `delta_rel` on `calib_sub`;
    least-negative candidate `t` such that `P(bad | delta_rel <= t) >= 2 * P(bad)`. If the primary
    `bad` proxy has < 50 positives, the secondary proxy is used and the switch is recorded.
    Fallback = 1st percentile, `status = "fallback_p1_no_enrichment"`.

Behavior contract (plan t10 acceptance): with the product, `regret.compute` returns a non-`None`
`is_blunder` and `confidence == "calibrated"`; `q_ev` confidence is `low`/`high` according to
`gap_threshold` (and flips across it); with no calibration the providers behave exactly as
M1+M2 (`is_blunder is None` / `confidence == "uncalibrated"`); a nested calibration object raises
`TypeError` in the consumers.

Usage:
  source scripts/activate_env.sh
  python scripts/exp_ex1_calibrate.py --selftest       # PREREG §10 + synthetic fit-rule checks
  python scripts/exp_ex1_calibrate.py --jobs 8         # full calib_sub fit (~1-2 min, CPU)
  python scripts/exp_ex1_calibrate.py --verify         # schema + behavior on the on-disk product
  python scripts/exp_ex1_calibrate.py --max-files 8    # smoke; outputs get a _limit8 suffix

allow: SIZE_OK — t10 mandates this exact single-file entry point (one product, one derivation
artifact, one PREREG hash discipline). The body is one pipeline: collect calib_sub → fit the two
PREREG §7 rules → flat product → consumer behavior check → curves artifact. Splitting it would
fragment the schema and the hash gate across files, which is worse.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "vendor/mortal/target/release"))
sys.path.insert(0, str(REPO / "vendor/mortal"))

import mortal  # noqa: E402, F401  # pin the namespace package before its own dir joins sys.path
sys.path.append(str(REPO / "vendor/mortal/mortal"))

from eval_report import SEAT_OF_SPLIT  # noqa: E402
from exp_a2_probe import auc_score  # noqa: E402
from exp_ex1_audit import assert_prereg_unchanged, load_prereg  # noqa: E402
from exp_ex1_audit import selftest as audit_selftest  # noqa: E402
from exp_ex1_labels import RecordError, load_targets  # noqa: E402
from exp_ex1_regret import kyoku_net_deltas  # noqa: E402  (PREREG §3 secondary bad proxy)
from jiangpu.models import CHECKPOINT_FIXTURE, checkpoint_sha256, load_bundle  # noqa: E402
from jiangpu.providers import q_ev, regret  # noqa: E402
from jiangpu.record import read_record  # noqa: E402
from jiangpu.replay import iter_decisions  # noqa: E402

PREREG_PATH = REPO / "scripts/exp_ex1_prereg.md"
CORPUS_INDEX = REPO / "runs/exp/EX1/corpus_index.json"
HOLDOUT_LABELS = REPO / "runs/exp/EX1/holdout_labels.json"
OUT_DIR = REPO / "runs/exp/EX1"
PRODUCT_PATH = OUT_DIR / "calibration_product.json"
CURVES_PATH = OUT_DIR / "calib_sub_curves.json"
SUBSET = "calib_sub"
PLAN = ".omo/plans/2026-09-23-explainability-metrics-and-validity.md"
PRODUCT_SCHEMA = "ex1-calibration-product/v1"
CURVES_SCHEMA = "ex1-calib-sub-curves/v1"

GAP_DECILES = tuple(range(10, 100, 10))      # PREREG §7: deciles 10..90
GAP_LOW_BAND = (0.45, 0.55)                  # PREREG §7: low-|gap| outcome-AUC band
GAP_HIGH_MIN = 0.60                          # PREREG §7: high-|gap| outcome-AUC floor
REGRET_PERCENTILES = tuple(range(1, 21))     # PREREG §7: percentiles 1..20
REGRET_ENRICHMENT = 2.0                      # PREREG §7: P(bad|<=t) >= 2 * P(bad)
MIN_POSITIVES = 50                           # PREREG §3/§9.4
BOOTSTRAP_B = 2000                           # PREREG §5
BOOTSTRAP_SEED = 7                           # PREREG §2
BEHAVIOR_FILES = 8                           # consumer-behavior sample (calib_sub files)

GAP_RULE = (
    "smallest candidate g in deciles 10..90 of |logged_action_q_rel| on calib_sub such that "
    "AUC(bad, |gap|) is in [0.45, 0.55] for |gap| < g AND >= 0.60 for |gap| >= g; "
    "fallback = median |gap| with status fallback_median_no_separation (PREREG §7)"
)
REGRET_RULE = (
    "least-negative candidate t in percentiles 1..20 of delta_rel on calib_sub such that "
    "P(bad | delta_rel <= t) >= 2 * P(bad); fallback = 1st percentile with status "
    "fallback_p1_no_enrichment (PREREG §7)"
)


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _finite(x) -> float | None:
    """JSON-safe float: non-finite statistics become null, never NaN."""
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


# -------------------------------------------------------------------------------------- inputs


def load_subset(subset: str) -> dict:
    """Pin the fit subset to t3's corpus index + t2's holdout labels (read-only, calib_sub only).

    Mirrors `exp_ex1_regret.load_subset` (which is hardcoded to `report_sub`), parameterized so
    t10 can load its own fitting subset. Labels come from the t2 oracle harness — a validity-
    harness label source only (NO-REVEAL, PREREG §8).
    """
    index = json.loads(CORPUS_INDEX.read_text(encoding="utf-8"))
    labels = json.loads(HOLDOUT_LABELS.read_text(encoding="utf-8"))
    if index.get("schema") != "ex1-corpus-index/v1":
        raise RecordError(f"unexpected corpus index schema {index.get('schema')!r}")
    if labels.get("schema") != "ex1-holdout-labels/v1":
        raise RecordError(f"unexpected holdout labels schema {labels.get('schema')!r}")
    if subset not in index["subsets"] or subset not in labels["subsets"]:
        raise RecordError(f"subset {subset!r} missing from corpus index / holdout labels")
    block = index["subsets"][subset]
    label_meta = labels["subsets"][subset]
    label_dir = REPO / label_meta["label_dir"]
    entries = []
    for i, f in enumerate(sorted(block["files"], key=lambda x: (x["seed"], x["key"], x["split"]))):
        label_path = label_dir / f"{f['seed']}_{f['key']}_{f['split']}.npz"
        if not label_path.is_file():
            raise RecordError(f"missing label file {label_path} "
                              f"(run scripts/exp_ex1_labels.py holdout)")
        entries.append({"file_id": i, "label_path": str(label_path), "seed": f["seed"],
                        "key": f["key"], "split": f["split"], "year": f["year"],
                        "target_path": f["target_path"], "source_path": f["source_path"]})
    return {"block": block, "label_meta": label_meta, "entries": entries}


# ----------------------------------------------------------------------------- per-file worker

_BUNDLE = None
_PREREG_SHA = None


def _worker_init(checkpoint_path: str, prereg_sha256: str) -> None:
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import torch  # noqa: PLC0415

    torch.set_num_threads(1)
    global _BUNDLE, _PREREG_SHA
    _BUNDLE = load_bundle(checkpoint_path, device="cpu")
    _PREREG_SHA = prereg_sha256


def _process_job(job: dict) -> dict:
    """Replay one calib_sub file, run the engine's own q_ev/regret providers, emit flat arrays."""
    targets = load_targets(job["label_path"])
    meta = targets["meta"]
    if meta.get("prereg_sha256") != _PREREG_SHA:
        raise RecordError(f"{job['label_path']}: label PREREG sha {meta.get('prereg_sha256')} "
                          f"!= {_PREREG_SHA}")
    if (int(meta["seed"]), int(meta["key"]), meta["split"]) != (job["seed"], job["key"],
                                                                job["split"]):
        raise RecordError(f"{job['label_path']}: identity {meta['seed']}/{meta['key']}/"
                          f"{meta['split']} != corpus index")
    seat = int(meta["player_id"])
    if seat != SEAT_OF_SPLIT[meta["split"]]:
        raise RecordError(f"{job['label_path']}: seat {seat} != SEAT_OF_SPLIT[{meta['split']}]")

    record = read_record(str(REPO / meta["source_path"]))
    decisions = iter_decisions(record, pov=seat)
    n = int(targets["n"])
    if len(decisions) != n:
        raise RecordError(f"{record.path}: {len(decisions)} decisions != {n} labels")
    if not np.array_equal(targets["actions"], [d.logged_action for d in decisions]):
        raise RecordError(f"{record.path}: label actions != replayed actions")
    if not np.array_equal(targets["at_kyoku"], np.asarray([d.at_kyoku for d in decisions],
                                                          dtype=np.uint8)):
        raise RecordError(f"{record.path}: label at_kyoku != replayed at_kyoku")

    q_payloads = q_ev.compute(_BUNDLE, decisions)
    regret_payloads = regret.compute(decisions, q_payloads)

    real = targets["real_mask"].astype(bool)
    bad_primary = targets["deal_in"].any(axis=(1, 2)).astype(np.uint8)
    kyoku_delta = kyoku_net_deltas(record.events, seat)
    if n and int(targets["at_kyoku"].max()) >= len(kyoku_delta):
        raise RecordError(f"{record.path}: at_kyoku {int(targets['at_kyoku'].max())} >= "
                          f"{len(kyoku_delta)} kyoku")
    bad_secondary = np.asarray([kyoku_delta[int(k)] < 0 for k in targets["at_kyoku"]],
                               dtype=np.uint8)

    gap_abs = np.full(n, np.nan)
    delta_rel = np.full(n, np.nan)
    n_missing = 0
    for dec in decisions:
        payload = regret_payloads.get(dec.index)
        q_payload = q_payloads.get(dec.index)
        if payload is None or payload.get("unavailable") or "delta_rel" not in payload \
                or q_payload is None or "logged_action_q_rel" not in q_payload:
            n_missing += 1
            continue
        delta_rel[dec.index] = float(payload["delta_rel"])
        gap_abs[dec.index] = abs(float(q_payload["logged_action_q_rel"]))
    keep = real & ~np.isnan(gap_abs) & ~np.isnan(delta_rel)
    return {
        "file_id": job["file_id"], "seed": job["seed"], "key": job["key"], "split": job["split"],
        "year": job["year"], "n_decisions": n, "n_kept": int(keep.sum()), "n_missing": n_missing,
        "gap_abs": gap_abs[keep].astype(np.float64),
        "delta_rel": delta_rel[keep].astype(np.float64),
        "bad_primary": bad_primary[keep], "bad_secondary": bad_secondary[keep],
        "checkpoint_sha256": _BUNDLE.sha256, "checkpoint_steps": int(_BUNDLE.steps),
    }


def collect(entries: list[dict], jobs: int, checkpoint: str, prereg_sha256: str) -> dict:
    """Re-infer Q + label realized outcomes over the fit subset; concatenate flat arrays."""
    jobs_list = [dict(job) for job in entries]
    if jobs <= 1:
        _worker_init(checkpoint, prereg_sha256)
        results = [_process_job(job) for job in jobs_list]
    else:
        chunksize = max(1, len(jobs_list) // (jobs * 4))
        with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init,
                                 initargs=(checkpoint, prereg_sha256),
                                 mp_context=mp.get_context("fork")) as ex:
            results = list(ex.map(_process_job, jobs_list, chunksize=chunksize))

    if results:
        shas = {r["checkpoint_sha256"] for r in results}
        steps = {r["checkpoint_steps"] for r in results}
        if len(shas) != 1 or len(steps) != 1:
            raise RecordError(f"worker checkpoints differ: {shas} / {steps}")
        ckpt_sha, ckpt_steps = shas.pop(), steps.pop()
    else:
        ckpt_sha, ckpt_steps = checkpoint_sha256(checkpoint), None

    def concat(key, dtype):
        if not results:
            return np.empty(0, dtype=dtype)
        return np.concatenate([r[key] for r in results]).astype(dtype)

    file_ids = (np.concatenate([np.full(len(r["gap_abs"]), r["file_id"], dtype=np.int64)
                                for r in results]) if results else np.empty(0, dtype=np.int64))
    return {
        "gap_abs": concat("gap_abs", np.float64),
        "delta_rel": concat("delta_rel", np.float64),
        "bad_primary": concat("bad_primary", np.uint8),
        "bad_secondary": concat("bad_secondary", np.uint8),
        "file_ids": file_ids,
        "n_files": len(results),
        "n_decisions_total": int(sum(r["n_decisions"] for r in results)),
        "n_decisions_kept": int(sum(r["n_kept"] for r in results)),
        "n_missing_payloads": int(sum(r["n_missing"] for r in results)),
        "checkpoint": {"path": _rel(Path(checkpoint)), "sha256": ckpt_sha, "steps": ckpt_steps,
                       "device": "cpu"},
    }


# --------------------------------------------------------------------------------- fitting rules


def _auc_or_none(bad, score) -> float | None:
    if len(bad) == 0:
        return None
    value = auc_score(bad, score)
    return float(value) if np.isfinite(value) else None


def fit_gap_threshold(abs_gap, bad) -> dict:
    """PREREG §7 gap rule. Returns value/status/grid/curve/selected; never hardcodes a value."""
    grid = [float(x) for x in np.quantile(abs_gap, [p / 100.0 for p in GAP_DECILES])]
    curve = []
    for g in grid:
        low = abs_gap < g
        high = ~low
        auc_low = _auc_or_none(bad[low], abs_gap[low])
        auc_high = _auc_or_none(bad[high], abs_gap[high])
        low_in_band = bool(auc_low is not None
                           and GAP_LOW_BAND[0] <= auc_low <= GAP_LOW_BAND[1])
        high_ok = bool(auc_high is not None and auc_high >= GAP_HIGH_MIN)
        curve.append({"candidate": g, "n_low": int(low.sum()), "n_high": int(high.sum()),
                      "auc_low": _finite(auc_low), "auc_high": _finite(auc_high),
                      "low_auc_in_band": low_in_band, "high_auc_ge_min": high_ok,
                      "qualifies": bool(low_in_band and high_ok)})
    selected = next((row for row in curve if row["qualifies"]), None)
    if selected is not None:
        value, status = float(selected["candidate"]), "fitted"
    else:
        value, status = float(np.median(abs_gap)), "fallback_median_no_separation"
    return {"value": value, "status": status, "grid": grid, "curve": curve, "selected": selected}


def fit_regret_threshold(delta_rel, bad) -> dict:
    """PREREG §7 regret rule: least-negative (largest) qualifying percentile."""
    grid = [float(x) for x in np.quantile(delta_rel, [p / 100.0 for p in REGRET_PERCENTILES])]
    base = float(bad.mean()) if len(bad) else 0.0
    curve = []
    for t in grid:
        at_or_below = delta_rel <= t
        n_le = int(at_or_below.sum())
        p_le = float(bad[at_or_below].mean()) if n_le else None
        ratio = (p_le / base) if (p_le is not None and base > 0.0) else None
        curve.append({"candidate": t, "n_at_or_below": n_le,
                      "n_above": int(len(delta_rel) - n_le),
                      "p_bad_at_or_below": _finite(p_le), "p_bad_base": _finite(base),
                      "enrichment_ratio": _finite(ratio),
                      "qualifies": bool(ratio is not None and ratio >= REGRET_ENRICHMENT)})
    selected = next((row for row in reversed(curve) if row["qualifies"]), None)
    if selected is not None:
        value, status = float(selected["candidate"]), "fitted"
    else:
        value = float(np.quantile(delta_rel, REGRET_PERCENTILES[0] / 100.0))
        status = "fallback_p1_no_enrichment"
    return {"value": value, "status": status, "grid": grid, "curve": curve, "selected": selected}


# --------------------------------------------------------------------------------------- CIs


def _resample_index(rng, groups: np.ndarray) -> np.ndarray:
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")   # rows grouped by file, block order
    lens = np.bincount(inv, minlength=len(uniq))
    counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
    return np.repeat(order, np.repeat(counts, lens))


def bootstrap_ci(groups: np.ndarray, stat_fn, n_boot: int = BOOTSTRAP_B,
                 seed: int = BOOTSTRAP_SEED) -> dict:
    """95 % file-cluster bootstrap CI (PREREG §5): resample files with replacement."""
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draws[b] = stat_fn(_resample_index(rng, groups))
    ok = np.isfinite(draws)
    n_ok = int(ok.sum())
    if n_ok < max(1, n_boot // 2):
        return {"ci95": None, "n_valid_draws": n_ok, "n_draws": int(n_boot)}
    lo, hi = np.percentile(draws[ok], [2.5, 97.5])
    return {"ci95": [float(lo), float(hi)], "n_valid_draws": n_ok, "n_draws": int(n_boot)}


def build_ci(data: dict, gap_value: float, regret_value: float, n_boot: int = BOOTSTRAP_B) -> dict:
    """CIs of the statistics that drove the two selections (calib_sub only)."""
    abs_gap, delta_rel = data["gap_abs"], data["delta_rel"]
    bad, groups = data["bad"], data["file_ids"]
    g, t = float(gap_value), float(regret_value)

    def low_auc(idx):
        m = abs_gap[idx] < g
        return auc_score(bad[idx][m], abs_gap[idx][m]) if m.any() else np.nan

    def high_auc(idx):
        m = abs_gap[idx] >= g
        return auc_score(bad[idx][m], abs_gap[idx][m]) if m.any() else np.nan

    def enrichment(idx):
        y, d = bad[idx], delta_rel[idx]
        m = d <= t
        base = y.mean()
        return y[m].mean() / base if (m.any() and base > 0.0) else np.nan

    low, high = abs_gap < g, abs_gap >= g
    at_or_below = delta_rel <= t
    base = float(bad.mean()) if len(bad) else 0.0
    enrichment_point = (float(bad[at_or_below].mean()) / base
                        if at_or_below.any() and base > 0.0 else None)
    return {
        "method": "file_cluster_bootstrap",
        "bootstrap": n_boot,
        "seed": BOOTSTRAP_SEED,
        "gap_threshold": {
            "low_gap_auc": {"point": _finite(_auc_or_none(bad[low], abs_gap[low])),
                            **bootstrap_ci(groups, low_auc, n_boot)},
            "high_gap_auc": {"point": _finite(_auc_or_none(bad[high], abs_gap[high])),
                             **bootstrap_ci(groups, high_auc, n_boot)},
        },
        "regret_blunder_threshold": {
            "enrichment_ratio": {"point": _finite(enrichment_point),
                                 **bootstrap_ci(groups, enrichment, n_boot)},
        },
    }


# --------------------------------------------------------------------------------- artifacts


def build_product(sha_start: str, gap_fit: dict, regret_fit: dict, population: dict,
                  bad_proxy: dict, ci: dict, checkpoint: dict) -> dict:
    notes = [
        "thresholds are data-derived on calib_sub only; report_sub never feeds a fit (C-4)",
        "flat top-level floats are required by q_ev.py:39 / regret.py:19 float(...) (MA-1)",
        "gap semantics: confidence = 'low' iff abs(logged_action_q_rel) < gap_threshold",
        "regret semantics: is_blunder = delta_rel <= regret_blunder_threshold",
        "aggregate.py:88's judgement_score >= 0.8 constant is untouched (documented only)",
    ]
    if population.get("n_below_gap_threshold") == 0:
        notes.append(
            "gap_threshold fallback is degenerate (the |gap| mass sits at 0, so no decision has "
            "|gap| < threshold): every decision gets confidence='high' under this product — see "
            "calib_sub_curves.json behavior_check for the flip probes")
    return {
        "schema": PRODUCT_SCHEMA,
        "gap_threshold": float(gap_fit["value"]),
        "regret_blunder_threshold": float(regret_fit["value"]),
        "derivation": {
            "fit_subset": SUBSET,
            "n_decisions": population["n_decisions_kept"],
            "n_files": population["n_files"],
            "n_decisions_total": population["n_decisions_total"],
            "n_kan_select_excluded": population["n_kan_select_excluded"],
            "rule": {"gap_threshold": GAP_RULE, "regret_blunder_threshold": REGRET_RULE},
            "status": {"gap_threshold": gap_fit["status"],
                       "regret_blunder_threshold": regret_fit["status"]},
            "ci": ci,
            "bad_proxy": bad_proxy,
            "candidate_grids": {
                "gap_threshold": [round(x, 6) for x in gap_fit["grid"]],
                "regret_blunder_threshold": [round(x, 6) for x in regret_fit["grid"]],
            },
            "checkpoint": checkpoint,
            "curves": _rel(CURVES_PATH),
            "prereg": {"path": _rel(PREREG_PATH), "sha256": sha_start},
            "notes": notes,
        },
    }


def _curve_summary(fit: dict) -> dict:
    return {"grid": [round(x, 6) for x in fit["grid"]], "curve": fit["curve"],
            "selected": fit["selected"], "value": fit["value"], "status": fit["status"]}


def build_curves(sha_start: str, subset: dict, data: dict, gap_fit: dict, regret_fit: dict,
                 population: dict, bad_proxy: dict, ci: dict, checkpoint: dict,
                 behavior: dict) -> dict:
    arrays = {
        "order": "rows are real (non-kan-select) decisions in corpus order: files sorted by "
                 "(seed, key, split); decisions in replay order",
        "gap_abs": [round(float(x), 6) for x in data["gap_abs"]],
        "delta_rel": [round(float(x), 6) for x in data["delta_rel"]],
        "bad_primary": [int(x) for x in data["bad_primary"]],
        "bad_secondary": [int(x) for x in data["bad_secondary"]],
        "file_id": [int(x) for x in data["file_ids"]],
    }
    return {
        "schema": CURVES_SCHEMA,
        "plan": PLAN,
        "prereg": {"path": _rel(PREREG_PATH), "sha256_start": sha_start, "sha256_end": None,
                   "immutable_during_run": None},
        "fit_subset": SUBSET,
        "corpus_index": _rel(CORPUS_INDEX),
        "subset_sha256": subset["block"].get("subset_sha256"),
        "subset_complete": population["n_files"] == subset["block"]["count"],
        "checkpoint": checkpoint,
        "population": population,
        "bad_proxy": bad_proxy,
        "gap_threshold_curve": _curve_summary(gap_fit),
        "regret_threshold_curve": _curve_summary(regret_fit),
        "selection": {"gap_threshold": {"value": float(gap_fit["value"]),
                                        "status": gap_fit["status"]},
                      "regret_blunder_threshold": {"value": float(regret_fit["value"]),
                                                   "status": regret_fit["status"]}},
        "ci": ci,
        "arrays": arrays,
        "behavior_check": behavior,
        "notes": [
            "these arrays are the ONLY fitting inputs for t10; no report_sub number is reused",
            "delta_rel = min(logged_action_q_rel, 0) rounded to 6 dp (providers/regret.py:41)",
            "bad_primary = 1[the tile we played was ron'd]; bad_secondary = kyoku delta < 0",
        ],
    }


# ------------------------------------------------------------------------- consumer behavior


def _nested_typeerror(fn) -> bool:
    try:
        fn()
    except TypeError:
        return True
    return False


def behavior_check(product: dict, entries: list[dict], checkpoint: str,
                   n_files: int = BEHAVIOR_FILES) -> dict:
    """Exercise the real consumers against the on-disk product (plan t10 acceptance).

    Sample = the first `n_files` calib_sub records (re-indexed into one stream, since
    `iter_decisions` numbers decisions per record).
    """
    bundle = load_bundle(checkpoint, device="cpu")
    decisions: list = []
    for entry in entries[:n_files]:
        record = read_record(str(REPO / entry["target_path"]))
        decisions.extend(iter_decisions(record, pov=SEAT_OF_SPLIT[record.split]))
    decisions = [dataclasses.replace(d, index=i) for i, d in enumerate(decisions)]

    g = float(product["gap_threshold"])
    q_cal = q_ev.compute(bundle, decisions, calibration=product)
    confs = [p["confidence"] for p in q_cal.values()]
    n_low, n_high = confs.count("low"), confs.count("high")
    rule_consistent = all(
        (abs(float(p["logged_action_q_rel"])) < g) == (p["confidence"] == "low")
        for p in q_cal.values())
    flip_observed = n_low > 0 and n_high > 0

    # Probes so the flip is demonstrated even when the fallback threshold is degenerate: an
    # above-every-|gap| threshold must flip all decisions to "low"; the largest PREREG grid
    # candidate is a real threshold with decisions strictly on both sides.
    max_gap = max((abs(float(p["logged_action_q_rel"])) for p in q_cal.values()), default=0.0)
    q_probe = q_ev.compute(bundle, decisions, calibration={"gap_threshold": max_gap + 1e-6})
    probe_all_low = all(p["confidence"] == "low" for p in q_probe.values())
    flip_via_probe = bool(n_high > 0 and probe_all_low)
    grid_max = max((float(x) for x in
                    product["derivation"]["candidate_grids"]["gap_threshold"]), default=0.0)
    q_grid = q_ev.compute(bundle, decisions, calibration={"gap_threshold": grid_max})
    grid_low = sum(1 for p in q_grid.values() if p["confidence"] == "low")
    grid_flip = bool(0 < grid_low < len(q_grid))

    q_uncal = q_ev.compute(bundle, decisions, calibration=None)
    uncal_all = all(p["confidence"] == "uncalibrated" for p in q_uncal.values())

    r_cal = regret.compute(decisions, q_cal, calibration=product)
    cal_ok = all(p.get("is_blunder") is not None and p["confidence"] == "calibrated"
                 for p in r_cal.values())
    n_blunder = sum(1 for p in r_cal.values() if p.get("is_blunder") is True)
    r_uncal = regret.compute(decisions, q_uncal, calibration=None)
    runcal_ok = all(p.get("is_blunder") is None and p["confidence"] == "uncalibrated"
                    for p in r_uncal.values())

    t_gap = _nested_typeerror(lambda: q_ev.compute(
        bundle, decisions, calibration={"gap_threshold": {"nested": 0.0}}))
    t_reg = _nested_typeerror(lambda: regret.compute(
        decisions, q_cal, calibration={"regret_blunder_threshold": {"nested": 0.0}}))

    result = {
        "sample": {"n_files": min(n_files, len(entries)), "n_decisions": len(decisions)},
        "q_ev": {
            "calibrated_confidences": {"low": n_low, "high": n_high},
            "rule_consistent_with_gap_threshold": rule_consistent,
            "flip_observed_with_product": flip_observed,
            "flip_via_threshold_probe": flip_via_probe,
            "grid_probe": {"threshold": grid_max, "low": grid_low,
                           "high": len(q_grid) - grid_low, "flip_observed": grid_flip},
            "gap_threshold_degenerate": n_low == 0,
            "uncalibrated_all_uncalibrated": uncal_all,
            "nested_threshold_typeerror": t_gap,
        },
        "regret": {
            "calibrated_all_is_blunder_bool": cal_ok,
            "calibrated_all_confidence": cal_ok,
            "n_blunder": n_blunder,
            "uncalibrated_all_is_blunder_none": runcal_ok,
            "uncalibrated_all_confidence_uncalibrated": runcal_ok,
            "nested_threshold_typeerror": t_reg,
        },
        "checkpoint": {"path": _rel(Path(checkpoint)), "sha256": bundle.sha256},
    }
    result["passed"] = bool(
        rule_consistent and uncal_all and cal_ok and runcal_ok and t_gap and t_reg
        and (flip_observed or grid_flip or flip_via_probe))
    return result


def verify_product(product_path: Path, curves_path: Path, checkpoint: str,
                   n_files: int = BEHAVIOR_FILES) -> dict:
    """Schema + consistency + consumer-behavior verification of the on-disk artifacts."""
    if not product_path.is_file():
        raise SystemExit(f"FATAL: {product_path} does not exist; run the fit first")
    product = json.loads(product_path.read_text(encoding="utf-8"))
    out: dict = {"product": _rel(product_path), "schema_checks": {}}
    for key in ("gap_threshold", "regret_blunder_threshold"):
        value = product.get(key)
        out["schema_checks"][key] = {"type": type(value).__name__,
                                     "is_float": isinstance(value, float),
                                     "is_not_dict": not isinstance(value, dict)}
        if not isinstance(value, float):
            raise SystemExit(f"FATAL: {product_path}['{key}'] is {type(value).__name__}, "
                             "not a flat float (MA-1)")
    derivation = product.get("derivation", {})
    out["schema_checks"]["fit_subset"] = derivation.get("fit_subset")
    if derivation.get("fit_subset") != SUBSET:
        raise SystemExit(f"FATAL: derivation.fit_subset != {SUBSET!r}")
    if curves_path.is_file():
        curves = json.loads(curves_path.read_text(encoding="utf-8"))
        block = json.loads(CORPUS_INDEX.read_text(encoding="utf-8"))["subsets"][SUBSET]
        labels = json.loads(HOLDOUT_LABELS.read_text(encoding="utf-8"))["subsets"][SUBSET]
        population = curves["population"]
        out["curves_checks"] = {
            "n_files": population["n_files"],
            "subset_count": block["count"],
            "n_files_consistent": population["n_files"] == block["count"],
            "n_decisions_total": population["n_decisions_total"],
            "label_n_decisions": labels["n_decisions"],
            "n_decisions_consistent": population["n_decisions_total"] == labels["n_decisions"],
            "subset_sha256": curves.get("subset_sha256"),
        }
        if not out["curves_checks"]["n_files_consistent"] \
                or not out["curves_checks"]["n_decisions_consistent"]:
            raise SystemExit("FATAL: calib_sub_curves.json sample counts inconsistent with "
                             "calib_sub (NB-5)")
    else:
        out["curves_checks"] = {"exists": False}
    entries = load_subset(SUBSET)["entries"]
    out["behavior_check"] = behavior_check(product, entries, checkpoint, n_files=n_files)
    if not out["behavior_check"]["passed"]:
        raise SystemExit("FATAL: consumer behavior check failed (see behavior_check)")
    return out


# ------------------------------------------------------------------------------------ selftest


def selftest() -> dict:
    """PREREG §10 audit checks + synthetic checks of both PREREG §7 fitting rules."""
    out: dict = {"audit_selftest": audit_selftest()}
    rng = np.random.default_rng(11)

    # gap rule, separated strata: deciles q10..q30 sit in a flat 5 %-bad stratum, q70..q90 in a
    # 40 %-bad stratum -> the smallest candidate must qualify with status "fitted".
    near = np.abs(rng.uniform(0.0, 0.5, 8000))
    mid = np.abs(rng.uniform(0.5, 1.0, 8000))
    far = np.abs(rng.uniform(1.0, 3.0, 8000))
    gap = np.concatenate([near, mid, far])
    bad = np.concatenate([rng.random(8000) < 0.05, rng.random(8000) < 0.05,
                          rng.random(8000) < 0.40]).astype(np.uint8)
    fit = fit_gap_threshold(gap, bad)
    assert fit["status"] == "fitted", fit["status"]
    assert fit["selected"] is fit["curve"][0], "smallest qualifying candidate must win"
    out["gap_fit_separated"] = {"value": fit["value"], "status": fit["status"],
                                "auc_low": fit["curve"][0]["auc_low"],
                                "auc_high": fit["curve"][0]["auc_high"]}

    # gap rule, no separation: independent bad labels -> fallback to the median.
    gap2 = rng.uniform(0.0, 1.0, 20000)
    bad2 = (rng.random(20000) < 0.10).astype(np.uint8)
    fit2 = fit_gap_threshold(gap2, bad2)
    assert fit2["status"] == "fallback_median_no_separation", fit2["status"]
    assert abs(fit2["value"] - float(np.median(gap2))) < 1e-12, fit2["value"]
    out["gap_fit_fallback"] = {"value": fit2["value"], "status": fit2["status"]}

    # regret rule, enriched: 60 % at delta_rel == 0 (5 % bad) + 40 % spread (40 % bad).
    delta = np.concatenate([np.zeros(12000), -np.abs(rng.uniform(0.1, 5.0, 8000))])
    bad3 = np.concatenate([rng.random(12000) < 0.05,
                           rng.random(8000) < 0.40]).astype(np.uint8)
    fit3 = fit_regret_threshold(delta, bad3)
    assert fit3["status"] == "fitted", fit3["status"]
    assert fit3["selected"] is fit3["curve"][-1], "least-negative qualifying candidate must win"
    assert fit3["selected"]["enrichment_ratio"] >= REGRET_ENRICHMENT
    out["regret_fit_enriched"] = {"value": fit3["value"], "status": fit3["status"],
                                  "enrichment_ratio": fit3["selected"]["enrichment_ratio"]}

    # regret rule, no enrichment: bad independent of delta_rel -> fallback to the 1st percentile.
    delta2 = np.concatenate([np.zeros(12000), -np.abs(rng.uniform(0.1, 5.0, 8000))])
    bad4 = (rng.random(20000) < 0.10).astype(np.uint8)
    fit4 = fit_regret_threshold(delta2, bad4)
    assert fit4["status"] == "fallback_p1_no_enrichment", fit4["status"]
    assert abs(fit4["value"] - float(np.quantile(delta2, 0.01))) < 1e-12, fit4["value"]
    out["regret_fit_fallback"] = {"value": fit4["value"], "status": fit4["status"]}

    # nested calibration object -> TypeError in the real regret consumer (MA-1 schema proof).
    assert _nested_typeerror(lambda: regret.compute(
        [], {}, calibration={"regret_blunder_threshold": {"nested": 0.0}})), \
        "regret must reject a nested threshold with TypeError"
    out["regret_nested_typeerror"] = True
    return out


# --------------------------------------------------------------------------------------- main


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="EX1 t10: calibration product (fit on calib_sub)")
    ap.add_argument("--checkpoint", default=CHECKPOINT_FIXTURE)
    ap.add_argument("--out", type=Path, default=PRODUCT_PATH)
    ap.add_argument("--curves-out", type=Path, default=CURVES_PATH)
    ap.add_argument("--product", type=Path, default=PRODUCT_PATH,
                    help="product to verify (--verify mode)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--behavior-files", type=int, default=BEHAVIOR_FILES)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify", action="store_true")
    return ap.parse_args(argv)


def _suffix_limited(path: Path, max_files: int | None, default: Path) -> Path:
    if max_files is None or path != default:
        return path
    return path.with_name(f"{path.stem}_limit{max_files}{path.suffix}")


def main(argv=None) -> int:
    args = parse_args(argv)
    sha_start = load_prereg(PREREG_PATH)

    if args.selftest:
        out = selftest()
        sha_end = assert_prereg_unchanged(PREREG_PATH, sha_start)
        out["prereg"] = {"path": _rel(PREREG_PATH), "sha256": sha_start,
                         "unchanged": sha_start == sha_end}
        print(json.dumps(out, indent=2))
        return 0

    if args.verify:
        out = verify_product(args.product.resolve(), args.curves_out.resolve(),
                             str(Path(args.checkpoint).resolve()), args.behavior_files)
        sha_end = assert_prereg_unchanged(PREREG_PATH, sha_start)
        out["prereg"] = {"path": _rel(PREREG_PATH), "sha256": sha_start,
                         "unchanged": sha_start == sha_end}
        print(json.dumps(out, indent=2))
        return 0

    started = time.monotonic()
    checkpoint = str(Path(args.checkpoint).resolve())
    out_path = _suffix_limited(args.out, args.max_files, PRODUCT_PATH)
    curves_path = _suffix_limited(args.curves_out, args.max_files, CURVES_PATH)

    subset = load_subset(SUBSET)
    entries = subset["entries"]
    if args.max_files is not None:
        if args.max_files <= 0:
            raise RecordError("--max-files must be positive")
        entries = entries[: args.max_files]
    print(f"[t10] fit subset={SUBSET} files={len(entries)} jobs={args.jobs} "
          f"checkpoint={_rel(checkpoint)}", flush=True)

    data = collect(entries, args.jobs, checkpoint, sha_start)
    if data["n_decisions_kept"] == 0:
        raise SystemExit("FATAL: no decisions with both q_ev and regret payloads on the fit "
                         "subset — nothing to fit (PREREG §9.1 fail-closed)")

    n_primary = int(data["bad_primary"].sum())
    n_secondary = int(data["bad_secondary"].sum())
    if n_primary >= MIN_POSITIVES:
        used, bad = "primary", data["bad_primary"].astype(bool)
        switch_reason = None
    elif n_secondary >= MIN_POSITIVES:
        used, bad = "secondary", data["bad_secondary"].astype(bool)
        switch_reason = (f"primary bad proxy has {n_primary} positives (< {MIN_POSITIVES}); "
                         "the PREREG §3 secondary proxy is used")
    else:
        raise SystemExit(f"FATAL: both bad proxies below {MIN_POSITIVES} positives on calib_sub "
                         f"(primary {n_primary}, secondary {n_secondary}) — not_estimable "
                         "(PREREG §9.4 fail-closed)")
    data["bad"] = bad.astype(np.uint8)

    gap_fit = fit_gap_threshold(data["gap_abs"], bad)
    regret_fit = fit_regret_threshold(data["delta_rel"], bad)

    population = {
        "n_files": data["n_files"],
        "subset_count": subset["block"]["count"],
        "subset_complete": data["n_files"] == subset["block"]["count"],
        "n_decisions_total": data["n_decisions_total"],
        "n_decisions_kept": data["n_decisions_kept"],
        "n_kan_select_excluded": data["n_decisions_total"] - data["n_decisions_kept"],
        "n_missing_payloads": data["n_missing_payloads"],
        "n_bad_primary": n_primary,
        "n_bad_secondary": n_secondary,
        "bad_proxy_used": used,
        "n_at_gap_zero": int((data["gap_abs"] == 0.0).sum()),
        "n_at_delta_zero": int((data["delta_rel"] == 0.0).sum()),
        "n_below_gap_threshold": int((data["gap_abs"] < gap_fit["value"]).sum()),
        "n_at_or_below_regret_threshold": int((data["delta_rel"] <= regret_fit["value"]).sum()),
        "gap_abs_deciles": [round(float(x), 6) for x in
                            np.quantile(data["gap_abs"], [p / 100.0 for p in GAP_DECILES])],
        "delta_rel_percentiles": [round(float(x), 6) for x in
                                  np.quantile(data["delta_rel"],
                                              [p / 100.0 for p in REGRET_PERCENTILES])],
    }
    bad_proxy = {
        "used": used,
        "primary_definition": "1[the tile we played at d was ron'd by any opponent]",
        "secondary_definition": "1[our seat's point delta over the rest of the kyoku < 0]",
        "n_positive_primary": n_primary,
        "n_positive_secondary": n_secondary,
        "min_positives": MIN_POSITIVES,
        "switch_triggered": used == "secondary",
        "switch_reason": switch_reason,
    }
    ci = build_ci(data, gap_fit["value"], regret_fit["value"], args.bootstrap)
    product = build_product(sha_start, gap_fit, regret_fit, population, bad_proxy, ci,
                            data["checkpoint"])

    # write the product first, then exercise the consumers against the on-disk artifact.
    _write_json(out_path, product)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    for key in ("gap_threshold", "regret_blunder_threshold"):
        if not isinstance(loaded[key], float):
            raise SystemExit(f"FATAL: {key} in {_rel(out_path)} is "
                             f"{type(loaded[key]).__name__}, not a flat float (MA-1)")
    if loaded["derivation"]["fit_subset"] != SUBSET:
        raise SystemExit("FATAL: derivation.fit_subset drifted from calib_sub")
    print(f"[t10] fitted gap_threshold={loaded['gap_threshold']!r} "
          f"({gap_fit['status']}) regret_blunder_threshold="
          f"{loaded['regret_blunder_threshold']!r} ({regret_fit['status']})", flush=True)

    behavior = behavior_check(loaded, subset["entries"], checkpoint, args.behavior_files)
    if not behavior["passed"]:
        raise SystemExit(f"FATAL: consumer behavior check failed: "
                         f"{json.dumps(behavior, sort_keys=True)}")
    print(f"[t10] behavior check PASSED: q_ev low/high="
          f"{behavior['q_ev']['calibrated_confidences']} "
          f"regret n_blunder={behavior['regret']['n_blunder']}", flush=True)

    curves = build_curves(sha_start, subset, data, gap_fit, regret_fit, population, bad_proxy,
                          ci, data["checkpoint"], behavior)
    if curves["subset_complete"]:
        if population["n_files"] != subset["block"]["count"]:
            raise SystemExit("FATAL: fit file count != calib_sub count")
        if population["n_decisions_total"] != subset["label_meta"]["n_decisions"]:
            raise SystemExit(f"FATAL: calib_sub decision count {population['n_decisions_total']} "
                             f"!= holdout labels {subset['label_meta']['n_decisions']} (NB-5)")

    # end-of-run gates: PREREG drift and checkpoint drift are both fatal (M8).
    sha_end = assert_prereg_unchanged(PREREG_PATH, sha_start)
    curves["prereg"]["sha256_end"] = sha_end
    curves["prereg"]["immutable_during_run"] = sha_start == sha_end
    sha_now = checkpoint_sha256(checkpoint)
    if data["checkpoint"]["sha256"] is not None and sha_now != data["checkpoint"]["sha256"]:
        raise SystemExit(f"FATAL: checkpoint changed during the run: {checkpoint}")
    curves["checkpoint"]["immutable_during_run"] = True
    _write_json(curves_path, curves)

    print(f"[done] wrote {_rel(out_path)} and {_rel(curves_path)} "
          f"in {time.monotonic() - started:.1f}s")
    print(json.dumps({
        "gap_threshold": loaded["gap_threshold"],
        "gap_status": gap_fit["status"],
        "regret_blunder_threshold": loaded["regret_blunder_threshold"],
        "regret_status": regret_fit["status"],
        "bad_proxy_used": used,
        "n_decisions": population["n_decisions_kept"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
