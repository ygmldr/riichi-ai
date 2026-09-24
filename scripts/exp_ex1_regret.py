#!/usr/bin/env python3
"""EX1 t8 — validity study of `delta_rel` (V5) and aggregate `judgement_vs_outcome` (V6).

Plan: `.omo/plans/2026-09-23-explainability-metrics-and-validity.md` todo 8.
PREREG: `scripts/exp_ex1_prereg.md` §3 (bad proxies), §5 (statistics + negative controls),
§6.1 (V5/V6 acceptance lines), §9 (degradation rules).

Product: `runs/exp/EX1/regret_validity.json` with two separate curves (never a composite):

  * `delta_rel_of_judgement` — V5: per-decision risk `= -delta_rel` vs the realized-outcome
    `bad` proxy. `delta_rel` is the shipped `regret` provider value
    (`providers/regret.py:41`: `min(logged_action_q_rel, 0)`, rounded to 6 dp), recomputed here
    from the engine's own inference path (`q_ev.compute` + `regret.compute`). Reports AUC +
    Spearman with file-cluster bootstrap CIs, a label-shuffle negative control, and the PREREG
    §6.1 quintile monotonicity check (bad rate in the most-negative `delta_rel` quintile >= the
    least-negative quintile).
  * `judgement_score` — V6: per-game `judgement_score = clamp(1 + mean(delta_rel), 0, 1)`
    (`jiangpu/aggregate.py:74-76`, cross-checked against `aggregate.build` itself), discrimination
    of realized point loss, the PREREG §6.1 dispersion gate (std >= 0.05 AND >= 10 distinct
    values, else `not_estimable`) and a shuffle negative control.

C-4: the evaluation subset is `report_sub`, read-only; nothing computed here feeds any fit or
selection. The engine formula and its `judgement_score >= 0.8` policy constant are read, never
modified (PREREG §7 / plan t8 "Must NOT"). `leg3_status` is reported per PREREG §6.1.

Usage:
  python scripts/exp_ex1_regret.py --selftest          # synthetic checks only
  python scripts/exp_ex1_regret.py --jobs 8            # full report_sub run
  python scripts/exp_ex1_regret.py --limit 20 --jobs 4 # smoke (suffixed artifact)
  python scripts/exp_ex1_regret.py --limit 20 --jobs 4 --qa-constant-judgement  # gate QA

allow: SIZE_OK — t8 mandates this exact single-file entry point (one artifact, one script); the
body is one measurement pipeline (PREREG gate -> per-file extraction -> two curves -> artifact).
"""
from __future__ import annotations

import argparse
import json
import os
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

import mortal  # noqa: E402, F401  # pin the namespace package before its own dir joins sys.path
sys.path.append(str(REPO / "vendor/mortal/mortal"))

from analyze_alllast import analyze_stream, replay_round  # noqa: E402
from eval_report import SEAT_OF_SPLIT  # noqa: E402
from exp_a2_probe import auc_score  # noqa: E402
from exp_a2b_calibration import spearman  # noqa: E402
from exp_ex1_audit import assert_prereg_unchanged, load_prereg  # noqa: E402
from exp_ex1_labels import RecordError, _deltas_field, load_targets  # noqa: E402
from jiangpu import aggregate  # noqa: E402
from jiangpu.models import CHECKPOINT_FIXTURE, checkpoint_sha256, load_bundle  # noqa: E402
from jiangpu.providers import q_ev, regret  # noqa: E402
from jiangpu.record import read_record  # noqa: E402
from jiangpu.replay import iter_decisions  # noqa: E402

PREREG_PATH = REPO / "scripts/exp_ex1_prereg.md"
OUT_DIR = REPO / "runs/exp/EX1"
ARTIFACT_ROOT = REPO / "runs/exp/EX1"
CORPUS_INDEX = OUT_DIR / "corpus_index.json"
HOLDOUT_LABELS = OUT_DIR / "holdout_labels.json"
SUBSET = "report_sub"
PLAN = ".omo/plans/2026-09-23-explainability-metrics-and-validity.md"

BOOTSTRAP_B = 2000           # PREREG §5
BOOTSTRAP_SEED = 7           # PREREG §2
SHUFFLE_SEED = 0             # PREREG §2 (shuffle negative control)
SHUFFLE_DRAWS = 20           # a single null draw's sd is 0.02-0.03 at this class balance
SHUFFLE_BAR = 0.02           # PREREG §5: |AUC - 0.5| < 0.02
PRIMARY_MIN_POSITIVES = 50   # PREREG §3 / §9.4
DISPERSION_STD = 0.05        # PREREG §6.1 V6
DISPERSION_MIN_DISTINCT = 10 # PREREG §6.1 V6
TRIAGE_AUC = 0.55            # PREREG §6.1 V5/V6
DELIVERY_AUC = 0.60          # PREREG §6.1 V5/V6
LEG3_STATUS = "not_estimable"
LEG3_REASON = "no demonstrably stronger reference exists"


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


# ------------------------------------------------------------------ secondary bad proxy (PREREG §3)


def kyoku_net_deltas(events, seat: int) -> list[int]:
    """Per-kyoku net point delta of `seat`, indexed by `at_kyoku` (start_kyoku order).

    `analyze_alllast.replay_round` semantics: start scores - 1000 * accepted riichi + hora and
    ryukyoku deltas; leftover pot excluded (resolved only at game end). PREREG §3 secondary
    proxy: `bad = 1[our seat's point delta over the rest of the kyoku < 0]`.
    """
    rounds: list[dict] = []
    cur = None
    for i, ev in enumerate(events):
        etype = ev.get("type")
        if etype == "start_kyoku":
            if cur is not None:
                raise RecordError(f"start_kyoku inside an open round (event {i})")
            scores = ev.get("scores")
            if not isinstance(scores, list) or len(scores) != 4:
                raise RecordError(f"start_kyoku[{i}]: bad scores={scores!r}")
            cur = {"start_scores": [int(s) for s in scores], "reach_accepted": [0, 0, 0, 0],
                   "horas": [], "ryukyoku": None}
        elif etype == "reach_accepted" and cur is not None:
            actor = ev.get("actor")
            if not isinstance(actor, int) or not 0 <= actor <= 3:
                raise RecordError(f"reach_accepted[{i}]: bad actor={actor!r}")
            cur["reach_accepted"][actor] += 1
        elif etype == "hora" and cur is not None:
            cur["horas"].append({"actor": ev.get("actor"), "target": ev.get("target"),
                                 "deltas": _deltas_field(ev, f"hora[{i}]")})
        elif etype == "ryukyoku" and cur is not None:
            cur["ryukyoku"] = _deltas_field(ev, f"ryukyoku[{i}]")
        elif etype == "end_kyoku" and cur is not None:
            rounds.append(cur)
            cur = None
    if cur is not None:
        raise RecordError("unclosed round in event stream")
    return [replay_round(r)[seat] - r["start_scores"][seat] for r in rounds]


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
    """Replay one report_sub file, run the engine's own Q/regret providers, emit flat arrays."""
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
    game = analyze_stream(record.path, record.events)
    start_scores = next(ev["scores"] for ev in record.events if ev["type"] == "start_kyoku")
    game_delta = int(game["final_scores"][seat]) - int(start_scores[seat])

    # V6 input = the shipped aggregator's own number (read-only), cross-checked independently.
    agg = aggregate.build(decisions, q_payloads, regret_payloads, {}, record,
                          final_scores=game["final_scores"], subject_player=seat)
    js_engine = agg["game_summary"]["judgement_vs_outcome"]["judgement_score"]
    deltas = [p["delta_rel"] for p in regret_payloads.values() if "delta_rel" in p]
    mean_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
    js_mine = max(0.0, min(1.0, 1.0 + mean_delta))
    if abs(round(js_mine, 6) - js_engine) > 1e-6:
        raise RecordError(f"{record.path}: judgement_score {js_mine} != aggregate.build {js_engine}")

    mask = targets["real_mask"].astype(bool)
    deal_in_any = targets["deal_in"].any(axis=(1, 2)).astype(np.uint8)
    n_ron_rows = int(targets["ron_dec"].shape[0])
    kyoku_delta = kyoku_net_deltas(record.events, seat)
    if n and int(targets["at_kyoku"].max()) >= len(kyoku_delta):
        raise RecordError(f"{record.path}: at_kyoku {int(targets['at_kyoku'].max())} >= "
                          f"{len(kyoku_delta)} kyoku")
    bad_secondary = np.asarray([kyoku_delta[int(k)] < 0 for k in targets["at_kyoku"]],
                               dtype=np.uint8)

    risk = np.full(n, np.nan)
    n_missing = 0
    for dec in decisions:
        payload = regret_payloads.get(dec.index)
        if payload is None or payload.get("unavailable") or "delta_rel" not in payload:
            n_missing += 1
            continue
        risk[dec.index] = -float(payload["delta_rel"])
    keep = mask & ~np.isnan(risk)
    return {
        "file_id": job["file_id"], "seed": job["seed"], "key": job["key"], "split": job["split"],
        "year": job["year"], "n_decisions": n, "n_kept": int(keep.sum()),
        "n_missing_payloads": n_missing,
        "risk": risk[keep].astype(np.float64),
        "bad_primary": deal_in_any[keep], "bad_secondary": bad_secondary[keep],
        "judgement_score": float(js_mine), "game_delta": game_delta,
        "bad_game": int(game_delta < 0), "n_rons": int(deal_in_any.sum()),
        "n_ron_rows": n_ron_rows,
        "checkpoint_sha256": _BUNDLE.sha256, "checkpoint_steps": int(_BUNDLE.steps),
    }


# --------------------------------------------------------------------------------- statistics


def cluster_bootstrap(risk, bad, slices, n_boot: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED):
    """File-cluster bootstrap (PREREG §5): resample clusters with replacement, B=2000, seed 7."""
    rng = np.random.default_rng(seed)
    n_clusters = len(slices)
    aucs = np.empty(n_boot)
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([slices[i] for i in pick])
        rb, yb = risk[idx], bad[idx]
        aucs[b] = auc_score(yb, rb)
        rhos[b] = spearman(rb, yb)
    return aucs, rhos


def percentile_ci(values) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def shuffle_control(risk, bad, draws: int = SHUFFLE_DRAWS, seed: int = SHUFFLE_SEED) -> dict:
    """Label-shuffle negative control (PREREG §5, plan t8: |AUC - 0.5| < 0.02).

    PREREG §2 pins seed 0; implemented as one deterministic 20-draw permutation stream (a single
    draw's null sd is 0.02-0.03 at this class balance, so the mean over draws is the sound
    control statistic; the first draw is reported separately).
    """
    rng = np.random.default_rng(seed)
    aucs = np.asarray([auc_score(rng.permutation(bad), risk) for _ in range(draws)])
    mean = float(aucs.mean())
    return {
        "draws": draws, "seed": seed,
        "auc_mean": mean, "auc_std": float(aucs.std()), "auc_min": float(aucs.min()),
        "auc_max": float(aucs.max()), "auc_first_draw": float(aucs[0]),
        "abs_mean_minus_half": abs(mean - 0.5), "bar": SHUFFLE_BAR,
        "passes": bool(abs(mean - 0.5) < SHUFFLE_BAR),
        "note": "BSS is undefined for a risk score that is not a probability; the PREREG §5 "
                "control applied here is |AUC - 0.5| < 0.02 (plan t8 acceptance).",
    }


def quintile_bad_rates(delta_rel, bad, n_bins: int = 5) -> dict:
    """PREREG §6.1 V5 monotonicity: bad rate per `delta_rel` quintile, most negative first."""
    edges = np.quantile(delta_rel, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    bins = np.clip(np.searchsorted(edges, delta_rel, side="right"), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        rows.append({
            "quintile": b, "n": int(m.sum()),
            "delta_rel_lo": float(delta_rel[m].min()) if m.any() else None,
            "delta_rel_hi": float(delta_rel[m].max()) if m.any() else None,
            "bad_rate": float(bad[m].mean()) if m.any() else None,
        })
    most, least = rows[0]["bad_rate"], rows[-1]["bad_rate"]
    comparable = most is not None and least is not None
    n_at_zero = int((np.asarray(delta_rel) == 0.0).sum())
    empty_bins = [r["quintile"] for r in rows if r["bad_rate"] is None]
    note = None
    if empty_bins:
        note = (f"empty quintile bins {empty_bins}: the quantile edges collapse at "
                f"delta_rel == 0 (the played action is the Q argmax for {n_at_zero}/"
                f"{len(delta_rel)} decisions), so the PREREG §6.1 check compares the populated "
                f"extreme bins (quintile 0 vs quintile {n_bins - 1})")
    return {
        "quintiles": rows,
        "most_negative_bad_rate": most, "least_negative_bad_rate": least,
        "comparable": comparable,
        "passes": (bool(most >= least) if comparable else None),
        "n_at_zero": n_at_zero, "empty_bins": empty_bins, "note": note,
        "rule": "bad rate in the most-negative delta_rel quintile >= least-negative quintile",
    }


def dispersion_gate(values) -> dict:
    """PREREG §6.1 V6: std >= 0.05 AND >= 10 distinct values, else not_estimable."""
    std = float(np.std(values, ddof=0))
    distinct = int(len(np.unique(np.round(np.asarray(values, dtype=np.float64), 6))))
    return {"std": std, "std_ddof": 0, "min_std": DISPERSION_STD,
            "n_distinct": distinct, "min_distinct": DISPERSION_MIN_DISTINCT,
            "passed": bool(std >= DISPERSION_STD and distinct >= DISPERSION_MIN_DISTINCT)}


def verdict_for(auc) -> str:
    if auc is None:
        return "not_estimable"
    if auc < TRIAGE_AUC:
        return "negative_result"
    if auc < DELIVERY_AUC:
        return "low_confidence"
    return "delivery_met"


# ------------------------------------------------------------------------------------ selftest


def selftest() -> dict:
    """Synthetic checks of the t8 statistics with exact/known values (PREREG-pinned routines)."""
    out: dict = {}
    n = 20_000
    y = np.zeros(n, dtype=bool)
    y[: n // 10] = True
    perfect = y.astype(np.float64)
    out["auc_perfect"] = auc_score(y, perfect)
    assert out["auc_perfect"] == 1.0, out["auc_perfect"]
    out["auc_reversed"] = auc_score(y, -perfect)
    assert out["auc_reversed"] == 0.0, out["auc_reversed"]
    rng = np.random.default_rng(SHUFFLE_SEED)
    out["auc_independent"] = auc_score(y, rng.random(n))
    assert abs(out["auc_independent"] - 0.5) < 0.05, out["auc_independent"]
    out["spearman_identity"] = spearman(np.arange(n, dtype=np.float64),
                                        np.arange(n, dtype=np.float64))
    assert abs(out["spearman_identity"] - 1.0) < 1e-12, out["spearman_identity"]
    out["dispersion_constant"] = dispersion_gate(np.ones(100))
    assert out["dispersion_constant"]["passed"] is False
    out["dispersion_spread"] = dispersion_gate(np.linspace(0.0, 1.0, 100))
    assert out["dispersion_spread"]["passed"] is True
    x = np.linspace(-2.0, 0.0, 10_000)
    rows = quintile_bad_rates(x, (x < -1.0).astype(np.uint8))
    out["quintile_monotone"] = rows["passes"]
    assert rows["passes"] is True
    assert rows["most_negative_bad_rate"] > rows["least_negative_bad_rate"]
    out["verdicts"] = [verdict_for(0.50), verdict_for(0.57), verdict_for(0.61)]
    assert out["verdicts"] == ["negative_result", "low_confidence", "delivery_met"]
    events = [
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000]},
        {"type": "reach_accepted", "actor": 1},
        {"type": "hora", "actor": 1, "target": 2, "deltas": [0, 8000, -8000, 0]},
        {"type": "end_kyoku"},
    ]
    out["kyoku_delta_fixture"] = kyoku_net_deltas(events, 1)
    assert out["kyoku_delta_fixture"] == [7000], out["kyoku_delta_fixture"]
    small_risk = np.array([0.1, 0.2, 0.9, 1.0, 0.3, 0.8])
    small_bad = np.array([0, 0, 1, 1, 0, 1])
    slices = [np.arange(0, 3), np.arange(3, 6)]
    aucs, rhos = cluster_bootstrap(small_risk, small_bad, slices, n_boot=50)
    out["bootstrap_ci_ordering"] = [percentile_ci(aucs), percentile_ci(rhos)]
    for lo, hi in out["bootstrap_ci_ordering"]:
        assert lo <= hi, (lo, hi)
    return out


# --------------------------------------------------------------------------------------- main


def load_subset() -> dict:
    """Pin the evaluation subset to t3's corpus index + t2's holdout labels (read-only)."""
    index = json.loads(CORPUS_INDEX.read_text(encoding="utf-8"))
    labels = json.loads(HOLDOUT_LABELS.read_text(encoding="utf-8"))
    if index.get("schema") != "ex1-corpus-index/v1":
        raise RecordError(f"unexpected corpus index schema {index.get('schema')!r}")
    block = index["subsets"][SUBSET]
    label_dir = REPO / labels["subsets"][SUBSET]["label_dir"]
    entries = []
    for i, f in enumerate(sorted(block["files"], key=lambda x: (x["seed"], x["key"], x["split"]))):
        label_path = label_dir / f"{f['seed']}_{f['key']}_{f['split']}.npz"
        if not label_path.is_file():
            raise RecordError(f"missing label file {label_path} "
                              f"(run scripts/exp_ex1_labels.py holdout)")
        entries.append({"file_id": i, "label_path": str(label_path), "seed": f["seed"],
                        "key": f["key"], "split": f["split"], "year": f["year"]})
    return {"block": block, "entries": entries}


def _empty_curve(definition: str, reason: str) -> dict:
    return {"definition": definition, "not_estimable": True, "reason": reason,
            "auc": None, "auc_ci": None, "shuffle_control": None,
            "triage_line": TRIAGE_AUC, "delivery_line": DELIVERY_AUC,
            "verdict": "not_estimable"}


def run(args) -> int:
    started = time.monotonic()
    prereg_path = args.prereg.resolve()
    sha_start = load_prereg(prereg_path)

    subset = load_subset()
    entries = subset["entries"]
    if args.limit is not None:
        if args.limit <= 0:
            raise RecordError("--limit must be positive")
        entries = entries[: args.limit]
    checkpoint = str(args.checkpoint.resolve())
    jobs = [dict(job) for job in entries]

    print(f"[t8] subset={SUBSET} files={len(jobs)} limit={args.limit} jobs={args.jobs} "
          f"checkpoint={_rel(checkpoint)}", flush=True)
    if args.jobs <= 1:
        _worker_init(checkpoint, sha_start)
        results = [_process_job(job) for job in jobs]
    else:
        chunksize = max(1, len(jobs) // (args.jobs * 4))
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_worker_init,
                                 initargs=(checkpoint, sha_start)) as ex:
            results = []
            for i, res in enumerate(ex.map(_process_job, jobs, chunksize=chunksize), 1):
                results.append(res)
                if i % 50 == 0:
                    print(f"[t8] {i}/{len(jobs)} files", flush=True)

    if results:
        ckpt_shas = {r["checkpoint_sha256"] for r in results}
        ckpt_steps = {r["checkpoint_steps"] for r in results}
        if len(ckpt_shas) != 1 or len(ckpt_steps) != 1:
            raise RecordError(f"worker checkpoints differ: {ckpt_shas} / {ckpt_steps}")
        ckpt_sha, ckpt_steps = ckpt_shas.pop(), ckpt_steps.pop()
    else:
        ckpt_sha, ckpt_steps = checkpoint_sha256(checkpoint), None

    delta_rel_def = ("delta_rel = min(logged_action_q_rel, 0), the shipped regret provider "
                     "value (providers/regret.py:41); risk = -delta_rel (higher = worse)")
    js_def = ("judgement_score = clamp(1 + mean(delta_rel over the game's decisions), 0, 1), "
              "the shipped aggregate.judgement_vs_outcome value (aggregate.py:74-76); "
              "risk = -judgement_score (a metric with signal yields AUC > 0.5)")

    if not results:
        v5 = _empty_curve(delta_rel_def, "empty evaluation subset (PREREG §9.4)")
        v6 = _empty_curve(js_def, "empty evaluation subset (PREREG §9.4)")
        bad_proxy = {"primary": None, "secondary": None, "selected_for_v5": None,
                     "switch": "empty subset", "min_positives": PRIMARY_MIN_POSITIVES}
        games: list[dict] = []
        counts = {"n_files": 0, "n_decisions": 0, "n_kept": 0, "n_missing_payloads": 0}
        monotonicity = None
    else:
        risk = np.concatenate([r["risk"] for r in results])
        bad_primary = np.concatenate([r["bad_primary"] for r in results]).astype(bool)
        bad_secondary = np.concatenate([r["bad_secondary"] for r in results]).astype(bool)
        file_sizes = [len(r["risk"]) for r in results]
        slices, pos = [], 0
        for size in file_sizes:
            slices.append(np.arange(pos, pos + size, dtype=np.int64))
            pos += int(size)
        n_pos_primary = int(bad_primary.sum())
        n_pos_secondary = int(bad_secondary.sum())
        if n_pos_primary >= PRIMARY_MIN_POSITIVES:
            selected, bad, switch = "primary", bad_primary, None
        elif n_pos_secondary >= PRIMARY_MIN_POSITIVES:
            selected, bad = "secondary", bad_secondary
            switch = (f"primary proxy had {n_pos_primary} positives "
                      f"(< {PRIMARY_MIN_POSITIVES})")
        else:
            selected, bad = None, None
            switch = (f"both proxies < {PRIMARY_MIN_POSITIVES} positives "
                      f"(primary={n_pos_primary}, secondary={n_pos_secondary})")
        bad_proxy = {
            "primary": {"definition": "1[the tile we played at d was ron'd by any opponent] "
                                      "(deal_in.any)", "n_positives": n_pos_primary,
                        "n_ron_rows": int(sum(r["n_ron_rows"] for r in results))},
            "secondary": {"definition": "1[subject's net point delta over the kyoku "
                                        "containing d < 0] (PREREG §3; replay_round semantics)",
                          "n_positives": n_pos_secondary},
            "selected_for_v5": selected, "switch": switch,
            "min_positives": PRIMARY_MIN_POSITIVES,
            "note": "n_positives counts distinct decisions (the AUC unit); n_ron_rows counts "
                    "hora events — multi-ron on one discard adds rows, not decisions.",
        }

        v5 = {"definition": delta_rel_def, "risk_direction": "risk = -delta_rel",
              "n_decisions": int(len(risk)), "n_files": len(results),
              "bad_proxy_used": selected, "n_positives": (int(bad.sum()) if bad is not None
                                                          else None),
              "triage_line": TRIAGE_AUC, "delivery_line": DELIVERY_AUC}
        monotonicity = None
        if selected is None:
            v5.update({"not_estimable": True, "reason": switch, "auc": None, "auc_ci": None,
                       "spearman": None, "spearman_ci": None, "shuffle_control": None,
                       "monotonicity": None, "verdict": "not_estimable"})
        else:
            monotonicity = quintile_bad_rates(-risk, bad)
            if not monotonicity["comparable"]:
                v5.update({"not_estimable": True, "auc": None, "auc_ci": None,
                           "spearman": None, "spearman_ci": None, "shuffle_control": None,
                           "monotonicity": monotonicity, "verdict": "not_estimable",
                           "reason": "monotonicity check not computable: an extreme delta_rel "
                                     "quintile is empty (PREREG §6.1 / §9.1 fail-closed)"})
            else:
                point_auc = auc_score(bad, risk)
                point_rho = spearman(risk, bad)
                boot_auc, boot_rho = cluster_bootstrap(risk, bad, slices)
                v5.update({
                    "not_estimable": False, "reason": None,
                    "auc": float(point_auc), "auc_ci": percentile_ci(boot_auc),
                    "spearman": float(point_rho), "spearman_ci": percentile_ci(boot_rho),
                    "shuffle_control": shuffle_control(risk, bad),
                    "monotonicity": monotonicity,
                    "verdict": verdict_for(float(point_auc)),
                })

        judgement_score = np.asarray([r["judgement_score"] for r in results], dtype=np.float64)
        if args.qa_constant_judgement:
            judgement_score = np.ones_like(judgement_score)
        bad_game = np.asarray([r["bad_game"] for r in results], dtype=bool)
        risk_js = -judgement_score
        gate = dispersion_gate(judgement_score)
        n_pos_game = int(bad_game.sum())
        n_neg_game = int((~bad_game).sum())
        v6 = {"definition": js_def, "n_games": int(len(judgement_score)),
              "n_positives": n_pos_game, "n_negatives": n_neg_game,
              "dispersion": gate, "triage_line": TRIAGE_AUC, "delivery_line": DELIVERY_AUC}
        if not gate["passed"]:
            v6.update({"not_estimable": True, "auc": None, "auc_ci": None,
                       "shuffle_control": None, "verdict": "not_estimable",
                       "reason": f"dispersion gate failed: std={gate['std']:.6f} "
                                 f"(min {DISPERSION_STD}), n_distinct={gate['n_distinct']} "
                                 f"(min {DISPERSION_MIN_DISTINCT}) (PREREG §6.1 V6)"})
        elif n_pos_game < PRIMARY_MIN_POSITIVES or n_neg_game < PRIMARY_MIN_POSITIVES:
            v6.update({"not_estimable": True, "auc": None, "auc_ci": None,
                       "shuffle_control": None, "verdict": "not_estimable",
                       "reason": f"game-level bad proxy has < {PRIMARY_MIN_POSITIVES} positives "
                                 f"or negatives (pos={n_pos_game}, neg={n_neg_game})"})
        else:
            game_slices = [np.arange(i, i + 1, dtype=np.int64) for i in range(len(judgement_score))]
            point_auc = auc_score(bad_game, risk_js)
            boot_auc, _ = cluster_bootstrap(risk_js, bad_game, game_slices)
            v6.update({
                "not_estimable": False, "reason": None,
                "auc": float(point_auc), "auc_ci": percentile_ci(boot_auc),
                "auc_unnegated_diagnostic": float(auc_score(bad_game, judgement_score)),
                "shuffle_control": shuffle_control(risk_js, bad_game),
                "verdict": verdict_for(float(point_auc)),
            })

        games = [{"file_id": r["file_id"], "seed": r["seed"], "key": r["key"],
                  "split": r["split"], "year": r["year"],
                  "judgement_score": r["judgement_score"], "game_delta": r["game_delta"],
                  "bad_game": bool(r["bad_game"]), "n_rons": r["n_rons"],
                  "n_decisions": r["n_decisions"]} for r in results]
        counts = {
            "n_files": len(results),
            "n_decisions": int(sum(r["n_decisions"] for r in results)),
            "n_kept": int(len(risk)),
            "n_missing_payloads": int(sum(r["n_missing_payloads"] for r in results)),
        }

    sha_end = assert_prereg_unchanged(prereg_path, sha_start)
    payload = {
        "schema": "ex1-regret-validity/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": PLAN,
        "script": _rel(Path(__file__)),
        "prereg": {"path": _rel(prereg_path), "sha256_start": sha_start, "sha256_end": sha_end,
                   "immutable_during_run": sha_start == sha_end},
        "evaluation_subset": {
            "name": SUBSET,
            "role": "read-only evaluation; no number here feeds any fit or selection (C-4)",
            "corpus_index": _rel(CORPUS_INDEX),
            "subset_sha256": subset["block"]["subset_sha256"],
            "target_paths": subset["block"]["target"],
            "n_files": len(results),
            "limit": args.limit,
            "selection_or_fit_performed": False,
        },
        "checkpoint": {"path": _rel(checkpoint), "sha256": ckpt_sha, "steps": ckpt_steps},
        "counts": counts,
        "bad_proxy": bad_proxy,
        "delta_rel_of_judgement": v5,
        "judgement_score": v6,
        "monotonicity": monotonicity,
        "leg3_status": LEG3_STATUS,
        "leg3_reason": LEG3_REASON,
        "games": games,
        "runtime": {"wall_seconds": round(time.monotonic() - started, 3), "jobs": args.jobs,
                    "qa_constant_judgement": bool(args.qa_constant_judgement)},
    }
    errors = verify_artifact(payload)
    if errors:
        raise SystemExit("FATAL: artifact invariants failed:\n  " + "\n  ".join(errors))

    out_path = (args.out.resolve() if args.out is not None
                else ARTIFACT_ROOT / _default_out_name(args))
    if ARTIFACT_ROOT.resolve() not in out_path.parents:
        raise SystemExit(f"FATAL: refusing to write outside {_rel(ARTIFACT_ROOT)}: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    print(f"[t8] wrote {_rel(out_path)}")
    print(json.dumps({
        "delta_rel_of_judgement": {"auc": v5.get("auc"), "auc_ci": v5.get("auc_ci"),
                                   "verdict": v5.get("verdict"),
                                   "shuffle_passes": (v5.get("shuffle_control") or {}).get("passes"),
                                   "monotonicity_passes": (v5.get("monotonicity") or {}).get("passes")},
        "judgement_score": {"auc": v6.get("auc"), "auc_ci": v6.get("auc_ci"),
                            "verdict": v6.get("verdict"),
                            "not_estimable": v6.get("not_estimable"),
                            "dispersion": v6.get("dispersion"),
                            "shuffle_passes": (v6.get("shuffle_control") or {}).get("passes")},
        "counts": counts,
    }, indent=2))
    return 0


def _default_out_name(args) -> str:
    parts = ["regret_validity"]
    if args.limit is not None:
        parts.append(f"limit{args.limit}")
    if args.qa_constant_judgement:
        parts.append("qa-constant-judgement")
    return "_".join(parts) + ".json"


def verify_artifact(payload: dict) -> list[str]:
    """In-script acceptance assertions (plan t8 acceptance criteria)."""
    errors: list[str] = []
    if payload["evaluation_subset"]["name"] != SUBSET or \
            payload["evaluation_subset"]["selection_or_fit_performed"]:
        errors.append("evaluation subset invariant violated (C-4)")
    if payload.get("leg3_status") != LEG3_STATUS or not payload.get("leg3_reason"):
        errors.append("leg3_status/leg3_reason missing (PREREG §6.1)")
    for name in ("delta_rel_of_judgement", "judgement_score"):
        block = payload[name]
        if block.get("not_estimable"):
            if not block.get("reason"):
                errors.append(f"{name}: not_estimable without a reason")
            if block.get("auc") is not None or block.get("auc_ci") is not None:
                errors.append(f"{name}: not_estimable but auc/auc_ci present")
            continue
        auc, ci = block.get("auc"), block.get("auc_ci")
        if auc is None or ci is None or not np.isfinite(auc):
            errors.append(f"{name}: missing/non-finite auc or auc_ci")
        elif not (ci[0] <= ci[1]):
            errors.append(f"{name}: auc_ci not ordered: {ci}")
        control = block.get("shuffle_control") or {}
        if not control.get("passes"):
            errors.append(f"{name}: shuffle control failed "
                          f"(|AUC-0.5|={control.get('abs_mean_minus_half')})")
    v6 = payload["judgement_score"]
    if v6.get("dispersion", {}).get("passed") is False and not v6.get("not_estimable"):
        errors.append("V6: dispersion gate failed but not_estimable is not true")
    v5 = payload["delta_rel_of_judgement"]
    if not v5.get("not_estimable"):
        mono = v5.get("monotonicity") or {}
        if len(mono.get("quintiles", [])) != 5 or not isinstance(mono.get("passes"), bool):
            errors.append("V5: monotonicity check missing/incomplete")
    return errors


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="EX1 t8: delta_rel (V5) + judgement_score (V6) "
                                             "validity on report_sub")
    ap.add_argument("--prereg", type=Path, default=PREREG_PATH)
    ap.add_argument("--checkpoint", type=Path, default=Path(CHECKPOINT_FIXTURE))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--qa-constant-judgement", action="store_true",
                    help="QA only: force judgement_score to a constant to exercise the "
                         "dispersion-gate not_estimable path (artifact name is suffixed)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest:
        sha = load_prereg(args.prereg.resolve())
        checks = selftest()
        print(json.dumps({"selftest": "PASS", "prereg_sha256": sha, **checks}, indent=2))
        return 0
    try:
        return run(args)
    except RecordError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
