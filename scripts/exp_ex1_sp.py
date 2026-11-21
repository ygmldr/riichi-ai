#!/usr/bin/env python3
"""EX1 t6: validity study of the shipped `sp_table` payload (`own_tenpai_prob` /
`own_win_prob` / `own_ev`).

Object = the payload `scripts/jiangpu/providers/sp_table.py` delivers for the v4 obs SP
planes. This script calls the shipped provider verbatim (`sp_table.compute(decisions)` on
the engine's own replay path) — the SP slicing and the delivered array orientation are
never modified, only *audit alignment* is resolved here.

Orientation (t9, `runs/exp/EX1/sp_orientation.json`): **`nearest_first`**. Slot 0 is the
from-now cumulative probability (`P(reach tenpai / win at some later turn)`); slot k is NOT
`P(event exactly at turn k)`; zeros past the horizon are legitimate. This script therefore
aligns **slot 0 only** with realized hand-level outcomes, and suppresses all aligned
statistics (`not_estimable=true` + reason) if the orientation artifact is undecidable.

Evaluation subset = `report_sub` (t3 `corpus_index.json`), read-only: no number computed
here feeds any fit or selection (PREREG §9.8 / C-4).

Realized labels (hand-level; replay + t2 label file; oracle labels are a validity-harness
input only, never an engine input):
  * own tenpai: `own_shanten[d] == 0` (already tenpai), or any later decision of ours in the
    same kyoku with shanten 0, or a win (a win implies tenpai). Conservative: tenpai first
    reached on the last discard of a kyoku that ends before our next decision is missed.
  * own win: a `hora{actor == us}` event at or after the decision's action event.
  * own value: our seat's point delta from the decision to the end of the kyoku
    (`hora`/`ryukyoku` deltas, minus 1000 per accepted riichi — the stick is not in
    `deltas`), divided by the decision's max EV recovered from the shipped `sp_ev` planes
    (`sp_ev[0] * 100000`, `obs_repr.rs:564-574,632-638`), so prediction (`own_ev = ev /
    max_ev`) and realized target share one unit. Rows with `max_ev < 1` (payload
    `ev_scale == 0`) or a saturated `sp_ev` are excluded from the value stats and counted.
  * `own_ev` is reported as a value-binned table, never as ECE (PREREG §5).

allow: SIZE_OK — t6 mandates this single entry point (`scripts/exp_ex1_sp.py`); the body is
one pipeline (replay -> shipped payload -> realized alignment -> PREREG statistics ->
artifact) and splitting it would scatter the PREREG hash discipline across files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "vendor/mortal/target/release"))
sys.path.insert(0, str(REPO / "vendor/mortal"))

import exp_ex1_labels as exl  # noqa: E402
from eval_report import SEAT_OF_SPLIT  # noqa: E402
from exp_ex1_audit import assert_prereg_unchanged, load_prereg  # noqa: E402
from exp_ex1_labels import RecordError, align_decisions, load_targets  # noqa: E402
from jiangpu.providers import sp_table  # noqa: E402
from jiangpu.record import read_record  # noqa: E402
from jiangpu.replay import iter_decisions  # noqa: E402

exl._bootstrap_mortal()  # pin the `mortal` namespace package before exp_a2_* import it

from exp_a2_probe import auc_score, brier_score, ece_score  # noqa: E402
from exp_a2b_calibration import N_BINS, reliability_table, spearman, value_table  # noqa: E402

PREREG_PATH = REPO / "scripts/exp_ex1_prereg.md"
PLAN = ".omo/plans/2026-09-23-explainability-metrics-and-validity.md"
OUT_DIR = REPO / "runs/exp/EX1"
CORPUS_INDEX = OUT_DIR / "corpus_index.json"
ORIENTATION_ARTIFACT = OUT_DIR / "sp_orientation.json"
SUBSET = "report_sub"
BOOT_B = 2000
BOOT_SEED = 7
SHUFFLE_SEED = 0
MIN_POSITIVES = 50


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clean(obj):
    """JSON-safe view: non-finite floats -> None, numpy scalars -> Python scalars."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


# ------------------------------------------------------------------------ realized labels


def _kyoku_bounds(events):
    """(start, end) event-index bounds of the k-th kyoku (k-th `start_kyoku`)."""
    starts = [i for i, ev in enumerate(events) if ev["type"] == "start_kyoku"]
    return [(s, starts[k + 1] if k + 1 < len(starts) else len(events))
            for k, s in enumerate(starts)]


def _scoring_events(events, seat):
    """(index, our point delta) for every event that moves our seat's score."""
    idx, delta = [], []
    for i, ev in enumerate(events):
        if ev["type"] == "reach_accepted" and ev.get("actor") == seat:
            idx.append(i)
            delta.append(-1000)  # riichi stick is not carried in `deltas`
        elif ev["type"] in ("hora", "ryukyoku"):
            d = ev.get("deltas")
            if not isinstance(d, list) or len(d) != 4 or not all(isinstance(x, int) for x in d):
                raise RecordError(f"event {i} ({ev['type']}): bad deltas={d!r}")
            idx.append(i)
            delta.append(int(d[seat]))
    return np.asarray(idx, np.int64), np.asarray(delta, np.float64)


def align_file(events, seat, actions, own_shanten, at_kyoku):
    """Per-decision realized tenpai / win / value (see module docstring).

    Also re-runs `align_decisions` (t2) so every label/engine event mapping is re-verified.
    """
    anchors = np.asarray(align_decisions(np.asarray(actions), events, seat)["dec_event"], np.int64)
    sc_idx, sc_delta = _scoring_events(events, seat)
    prefix = np.concatenate([[0.0], np.cumsum(sc_delta)])
    wins = np.asarray([i for i, ev in enumerate(events)
                       if ev["type"] == "hora" and ev.get("actor") == seat], np.int64)
    kyoku = np.asarray(at_kyoku)
    tenpai_now = np.asarray(own_shanten) == 0
    n = len(anchors)
    real_tenpai = np.zeros(n, bool)
    real_win = np.zeros(n, bool)
    real_pts = np.zeros(n, np.float64)
    for k, (_start, end) in enumerate(_kyoku_bounds(events)):
        idx = np.flatnonzero(kyoku == k)
        if len(idx) == 0:
            continue
        real_tenpai[idx] = np.cumsum(tenpai_now[idx][::-1])[::-1] > 0
        for d in idx:
            a = int(anchors[d])
            if a < 0:
                continue
            lo = int(np.searchsorted(sc_idx, a, side="left"))   # events at or after the action
            hi = int(np.searchsorted(sc_idx, end, side="left"))  # events before the next kyoku
            real_pts[d] = float(prefix[hi] - prefix[lo])
            real_win[d] = bool(((wins >= a) & (wins < end)).any())
    return {"anchors": anchors, "real_tenpai": real_tenpai,
            "real_win": real_win, "real_pts": real_pts}


def recover_max_ev(payload):
    """(max_ev, status) from the shipped `sp_ev` planes; status in {ok, lt1, saturated}."""
    sp_ev0, sp_ev1 = float(payload["sp_ev"][0]), float(payload["sp_ev"][1])
    if sp_ev0 < 1.0 - 1e-6:
        max_ev = sp_ev0 * 100000.0
        return max_ev, ("ok" if max_ev >= 1.0 else "lt1")
    if sp_ev1 < 1.0 - 1e-6:
        return sp_ev1 * 30000.0, "ok"
    return None, "saturated"


# --------------------------------------------------------------------------------- per file


def process_file(entry, labels_dir):
    """Replay one canonical record, run the shipped provider, align slot 0 to realized labels."""
    target = REPO / entry["target_path"]
    label_path = labels_dir / f"{entry['seed']}_{entry['key']}_{entry['split']}.npz"
    record = read_record(str(target))
    seat = SEAT_OF_SPLIT[record.split]
    decisions = iter_decisions(record, pov=seat)
    labels = load_targets(label_path)
    actions = labels["actions"]
    n = len(decisions)
    if labels["meta"]["player_id"] != seat:
        raise RecordError(f"{target.name}: label seat {labels['meta']['player_id']} != {seat}")
    if actions.shape[0] != n or not np.array_equal(
            actions, np.asarray([d.logged_action for d in decisions], dtype=actions.dtype)):
        raise RecordError(f"{target.name}: label actions != replay actions ({actions.shape[0]} vs {n})")
    if not np.array_equal(labels["at_kyoku"], np.asarray([d.at_kyoku for d in decisions], np.uint8)):
        raise RecordError(f"{target.name}: label at_kyoku != replay at_kyoku")
    aligned = align_file(record.events, seat, actions, labels["own_shanten"], labels["at_kyoku"])
    payloads = sp_table.compute(decisions)

    keys = ("pred_tenpai", "pred_win", "pred_ev", "real_tenpai", "real_win", "real_pts",
            "max_ev", "scale_ok")
    rows = {k: [] for k in keys}
    reasons: Counter = Counter()
    horizon = []
    inv_ok: Counter = Counter()
    profile = np.zeros((3, sp_table.SP_TURNS), np.float64)
    n_available = 0
    for i, dec in enumerate(decisions):
        payload = payloads.get(dec.index)
        if payload is None:
            reasons["provider_missing"] += 1
            continue
        if payload.get("unavailable"):
            reasons[f"unavailable: {payload.get('reason', 'unknown')}"] += 1
            continue
        n_available += 1
        tenpai = np.asarray(payload["own_tenpai_prob"], np.float64)
        profile[0] += tenpai
        profile[1] += np.asarray(payload["own_win_prob"], np.float64)
        profile[2] += np.asarray(payload["own_ev"], np.float64)
        horizon.append(int(payload["horizon"]))
        if np.all(np.diff(tenpai) <= 1e-9):
            inv_ok["non_increasing"] += 1
        if np.all(np.diff(tenpai) >= -1e-9):
            inv_ok["non_decreasing"] += 1
        if tenpai[0] >= tenpai.max() - 1e-9:
            inv_ok["first_slot_max"] += 1
        pos = np.flatnonzero(tenpai > 0.0)
        if len(pos) == 0 or not (tenpai[pos[-1] + 1:] > 0.0).any():
            inv_ok["zeros_suffix"] += 1
        if int(labels["is_kan_select"][i]):
            reasons["excluded: kan_select"] += 1
            continue
        if not 0 <= dec.logged_action <= 36:
            reasons["excluded: non_discard_action"] += 1
            continue
        max_ev, status = recover_max_ev(payload)
        rows["pred_tenpai"].append(float(tenpai[0]))
        rows["pred_win"].append(float(payload["own_win_prob"][0]))
        rows["pred_ev"].append(float(payload["own_ev"][0]))
        rows["real_tenpai"].append(float(aligned["real_tenpai"][i]))
        rows["real_win"].append(float(aligned["real_win"][i]))
        rows["real_pts"].append(float(aligned["real_pts"][i]))
        rows["max_ev"].append(float(max_ev) if max_ev is not None else -1.0)
        rows["scale_ok"].append(status == "ok")
    return {
        "n": n,
        "n_available": n_available,
        "reasons": dict(reasons),
        "horizon": np.asarray(horizon, np.int64),
        "inv_ok": dict(inv_ok),
        "profile": profile,
        "rows": {k: np.asarray(v, np.float64 if k != "scale_ok" else bool) for k, v in rows.items()},
    }


# ---------------------------------------------------------------------------------- metrics


def _bss(y, pred) -> float:
    pbar = float(np.mean(y))
    if not 0.0 < pbar < 1.0:
        return float("nan")
    return 1.0 - brier_score(y, pred) / (pbar * (1.0 - pbar))


def _cluster_ci_multi(groups, fn, n_boot=BOOT_B, seed=BOOT_SEED):
    """95% file-cluster bootstrap CI (resample files with replacement) for every key of fn."""
    _uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_group = [np.flatnonzero(inv == g) for g in range(int(inv.max()) + 1)] if len(inv) else []
    rng = np.random.default_rng(seed)
    acc: dict = defaultdict(list)
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_by_group), len(idx_by_group))
        idx = np.concatenate([idx_by_group[p] for p in pick])
        for k, v in fn(idx).items():
            acc[k].append(float(v))
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in acc.items()}


def _binary_field(pred, y, groups):
    n = len(y)
    pos = int(y.sum())
    pbar = pos / n if n else float("nan")
    field = {
        "n": n,
        "positives": pos,
        "prevalence": pbar,
        "mean_pred": float(pred.mean()) if n else None,
        "point": {
            "ece": ece_score(y, pred, N_BINS) if n else None,
            "brier": brier_score(y, pred) if n else None,
            "brier_base": pbar * (1.0 - pbar) if n else None,
            "brier_skill": _bss(y, pred) if n else None,
            "auc": auc_score(y, pred) if n else None,
        },
        "reliability_table": reliability_table(pred, y, N_BINS) if n else [],
        "ci95": {},
        "not_estimable": pos < MIN_POSITIVES,
        "not_estimable_reason": None if pos >= MIN_POSITIVES else
        f"<{MIN_POSITIVES} positives on {SUBSET} ({pos}) (PREREG §9.4)",
    }
    if pos >= MIN_POSITIVES:
        field["ci95"] = _cluster_ci_multi(groups, lambda idx: {
            "ece": ece_score(y[idx], pred[idx], N_BINS),
            "brier": brier_score(y[idx], pred[idx]),
            "brier_skill": _bss(y[idx], pred[idx]),
            "auc": auc_score(y[idx], pred[idx]),
        })
    return field


def _ev_field(pred, target, groups, pts, n_excluded):
    n = len(pred)
    vt = value_table(pred, target, N_BINS) if n else {}
    field = {
        "n": n,
        "n_excluded_scale": n_excluded,
        "point": {k: vt.get(k) for k in
                  ("mae", "rmse", "spearman", "mean_pred", "mean_realized", "mean_bias",
                   "target_std", "value_binned_calibration_error_pts",
                   "value_binned_calibration_error_over_std")} if n else {},
        "value_table": vt,
        "realized_points": ({"mean": float(pts.mean()), "std": float(pts.std()),
                             "min": float(pts.min()), "max": float(pts.max())} if n else {}),
        "ci95": {},
    }
    if n >= MIN_POSITIVES:
        def stats(idx):
            v = value_table(pred[idx], target[idx], N_BINS)
            return {"mae": v["mae"], "spearman": v["spearman"],
                    "value_binned_calibration_error_over_std":
                        v["value_binned_calibration_error_over_std"]}
        field["ci95"] = _cluster_ci_multi(groups, stats)
    return field


def _negative_controls(pred_t, y_t, pred_w, y_w):
    rng = np.random.default_rng(SHUFFLE_SEED)
    out = {
        "shuffle_seed": SHUFFLE_SEED,
        "constant_input": {},
        "note": ("global label shuffle: AUC must collapse to 0.5 (asserted two-sided). The Brier "
                 "skill under a shuffle is expected NEGATIVE (Brier decomposition: shuffled Brier = "
                 "reliability + uncertainty >= base rate), so the gate is one-sided: no positive "
                 "skill may appear (BSS < 0.01)."),
    }
    for name, pred, y in (("own_tenpai_prob", pred_t, y_t), ("own_win_prob", pred_w, y_w)):
        perm = rng.permutation(len(y))
        auc_s, bss_s = auc_score(y[perm], pred), _bss(y[perm], pred)
        out[name] = {
            "auc": auc_s, "brier_skill": bss_s,
            "auc_ok": abs(auc_s - 0.5) < 0.02,
            "bss_no_positive_skill_ok": bss_s < 0.01,
            "bss_two_sided_ok": abs(bss_s) < 0.01,
        }
        const = np.full(len(y), float(y.mean()))
        out["constant_input"][name] = {
            "ece": ece_score(y, const, N_BINS), "auc": auc_score(y, const),
            "brier_skill": _bss(y, const),
        }
    return out


# ---------------------------------------------------------------------------------- selftest


def selftest() -> dict:
    """PREREG-pinned synthetic checks + the t6 realized-alignment contract."""
    from exp_ex1_audit import selftest as audit_selftest  # noqa: PLC0415

    events = [
        {"type": "start_kyoku"},
        {"type": "tsumo", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": "reach", "actor": 0},
        {"type": "reach_accepted", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "2m"},
        {"type": "tsumo", "actor": 0},
        {"type": "hora", "actor": 0, "target": 0, "deltas": [2000, -1000, -500, -500]},
        {"type": "end_kyoku"},
        {"type": "start_kyoku"},
        {"type": "tsumo", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "3m"},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
    ]
    al = align_file(events, 0, np.asarray([0, 37, 1, 43, 2], np.int64),
                    np.asarray([1, 0, 0, 0, 2], np.int8), np.asarray([0, 0, 0, 0, 1], np.uint8))
    assert al["anchors"].tolist() == [2, 3, 5, 7, 11], al["anchors"].tolist()
    assert al["real_tenpai"].tolist() == [True] * 4 + [False], al["real_tenpai"].tolist()
    assert al["real_win"].tolist() == [True] * 4 + [False], al["real_win"].tolist()
    assert al["real_pts"].tolist() == [1000.0, 1000.0, 2000.0, 2000.0, 0.0], al["real_pts"].tolist()
    assert recover_max_ev({"sp_ev": [0.08, 1.0]}) == (8000.0, "ok")
    assert recover_max_ev({"sp_ev": [1.0, 1.0]}) == (None, "saturated")
    assert recover_max_ev({"sp_ev": [0.0, 0.0]})[1] == "lt1"
    return {
        "audit": audit_selftest(),
        "alignment": {"anchors": al["anchors"].tolist(), "real_pts": al["real_pts"].tolist()},
    }


# -------------------------------------------------------------------------------------- main


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="EX1 t6 sp_table payload validity study")
    ap.add_argument("--prereg", type=Path, default=PREREG_PATH)
    ap.add_argument("--corpus-index", type=Path, default=CORPUS_INDEX)
    ap.add_argument("--subset", default=SUBSET)
    ap.add_argument("--labels-dir", type=Path, default=None)
    ap.add_argument("--orientation-artifact", type=Path, default=ORIENTATION_ARTIFACT)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "sp_validity.json")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N files (QA)")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def _cat(results, key):
    parts = [r["rows"][key] for r in results]
    return np.concatenate(parts) if parts else np.zeros(0)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.selftest:
        print(json.dumps(_clean(selftest()), indent=2, sort_keys=True))
        return 0

    prereg_path = args.prereg.resolve()
    sha_start = load_prereg(prereg_path)
    orientation_path = args.orientation_artifact.resolve()
    if not orientation_path.exists():
        raise SystemExit(f"FATAL: missing orientation artifact {orientation_path} (t9 gate)")
    orient = json.loads(orientation_path.read_text(encoding="utf-8"))
    orientation = orient.get("orientation")
    orient_reason = None if orientation == "nearest_first" else (
        f"orientation={orientation!r}: only nearest_first maps slot 0 to the from-now "
        "cumulative probability (t9); aligned statistics suppressed (horizon-invariant "
        "aggregates only)")

    index = json.loads(args.corpus_index.read_text(encoding="utf-8"))
    if index.get("schema") != "ex1-corpus-index/v1":
        raise SystemExit(f"FATAL: unexpected corpus index schema {index.get('schema')!r}")
    block = index["subsets"][args.subset]
    files = list(block["files"])
    if args.limit is not None:
        files = files[: max(0, args.limit)]
    labels_dir = (args.labels_dir.resolve() if args.labels_dir
                  else OUT_DIR / "labels" / args.subset)

    results, errors = [], []
    for j, entry in enumerate(files):
        try:
            results.append(process_file(entry, labels_dir))
        except (RecordError, OSError, RuntimeError) as exc:
            errors.append({"target_path": entry["target_path"], "error": str(exc)})
        if (j + 1) % 50 == 0 or j + 1 == len(files):
            print(f"[sp] {j + 1}/{len(files)} files ({len(errors)} errors)", flush=True)
    if errors:
        for e in errors[:10]:
            print(f"FATAL: {e['target_path']}: {e['error']}", file=sys.stderr)
        raise SystemExit(f"FATAL: {len(errors)} file(s) failed; no artifact written")

    pred_t, pred_w = _cat(results, "pred_tenpai"), _cat(results, "pred_win")
    real_t, real_w = _cat(results, "real_tenpai"), _cat(results, "real_win")
    pred_ev, real_pts = _cat(results, "pred_ev"), _cat(results, "real_pts")
    max_ev, scale_ok = _cat(results, "max_ev"), _cat(results, "scale_ok")
    groups = np.concatenate([np.full(len(r["rows"]["pred_tenpai"]), j) for j, r in enumerate(results)]) \
        if results else np.zeros(0)
    n_decisions = int(sum(r["n"] for r in results))
    n_available = int(sum(r["n_available"] for r in results))
    reasons: Counter = Counter()
    inv_ok: Counter = Counter()
    profile = np.zeros((3, sp_table.SP_TURNS), np.float64)
    for r in results:
        reasons.update(r["reasons"])
        inv_ok.update(r["inv_ok"])
        profile += r["profile"]
    horizons = np.concatenate([r["horizon"] for r in results]) if results else np.zeros(0, np.int64)
    n_aligned = int(len(pred_t))

    if n_aligned:
        assert n_available > 0, "available payload count must be > 0"
        if not (pred_t.max() > 0 and pred_w.max() > 0 and pred_ev.max() > 0):
            raise SystemExit("FATAL: usable payloads are all-zero (F3 invariant)")
    top_reason = orient_reason or (
        None if n_aligned else f"no usable sp_table payloads on {args.subset} (n_aligned=0)")

    n_excluded_scale = int((~scale_ok).sum()) if len(scale_ok) else 0
    scale_status: Counter = Counter()
    if len(max_ev):
        for ok, ev in zip(scale_ok.tolist(), max_ev.tolist()):
            if not ok:
                scale_status["saturated" if ev < 0 else "lt1"] += 1

    out = {
        "schema": "ex1-sp-validity/v1",
        "todo": "t6",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": PLAN,
        "prereg": {"path": _rel(prereg_path), "sha256_start": sha_start},
        "evaluation_subset": {
            "name": args.subset,
            "corpus_index": _rel(args.corpus_index),
            "subset_sha256": block.get("subset_sha256"),
            "n_files": len(files),
            "n_files_ok": len(results),
            "n_files_error": 0,
            "labels_dir": _rel(labels_dir),
        },
        "orientation": {
            "artifact": _rel(orientation_path),
            "sha256": _sha256(orientation_path),
            "orientation": orientation,
            "slot_used": 0 if orientation == "nearest_first" else None,
            "alignment_rule": ("slot 0 = from-now cumulative probability; never read slot k as "
                               "P(event exactly at turn k) (t9)"),
            "not_estimable": orient_reason is not None,
            "reason": orient_reason,
        },
        "population": {
            "n_decisions": n_decisions,
            "n_available": n_available,
            "n_aligned": n_aligned,
            "n_excluded": int(sum(v for k, v in reasons.items() if k.startswith("excluded:"))),
            "exclusion_reasons": dict(reasons),
            "n_ev_scale_ok": int(scale_ok.sum()) if len(scale_ok) else 0,
            "n_ev_scale_excluded": n_excluded_scale,
            "ev_scale_exclusion_reasons": dict(scale_status),
        },
        "horizon_invariants": {
            "n_with_horizon": int((horizons >= 0).sum()) if len(horizons) else 0,
            "mean_horizon": float(horizons.mean()) if len(horizons) else None,
            "max_horizon": int(horizons.max()) if len(horizons) else None,
            "n_non_increasing_tenpai": inv_ok.get("non_increasing", 0),
            "n_non_decreasing_tenpai": inv_ok.get("non_decreasing", 0),
            "n_first_slot_is_max": inv_ok.get("first_slot_max", 0),
            "n_zeros_form_suffix": inv_ok.get("zeros_suffix", 0),
            "slot_profile_mean": (profile / n_available).tolist() if n_available else [],
        },
        "not_estimable": top_reason is not None,
        "not_estimable_reason": top_reason,
    }

    if top_reason:
        out["fields"] = {
            name: {"not_estimable": True, "reason": top_reason}
            for name in ("own_tenpai_prob", "own_win_prob", "own_ev")
        }
    else:
        ev = scale_ok  # own_ev is only auditable where max EV was recoverable from `sp_ev`
        ev_pts = real_pts[ev]
        fields = {
            "own_tenpai_prob": _binary_field(pred_t, real_t, groups),
            "own_win_prob": _binary_field(pred_w, real_w, groups),
            "own_ev": _ev_field(pred_ev[ev], ev_pts / max_ev[ev], groups[ev],
                                ev_pts, n_excluded_scale),
        }
        out["fields"] = fields
        controls = _negative_controls(pred_t, real_t, pred_w, real_w)
        out["negative_controls"] = controls
        for name in ("own_tenpai_prob", "own_win_prob"):
            if not (controls[name]["auc_ok"] and controls[name]["bss_no_positive_skill_ok"]):
                raise SystemExit(f"FATAL: shuffle negative control failed for {name} (PREREG §9.7)")
        for name, vals in controls["constant_input"].items():
            if abs(vals["ece"]) > 1e-9 or abs(vals["auc"] - 0.5) > 1e-12 or abs(vals["brier_skill"]) > 1e-9:
                raise SystemExit(f"FATAL: constant-input control is not constant for {name} (PREREG §9.7)")
        t_ece = fields["own_tenpai_prob"]["point"]["ece"]
        w_ece = fields["own_win_prob"]["point"]["ece"]
        v_ratio = fields["own_ev"]["point"].get("value_binned_calibration_error_over_std")
        out["verdicts"] = {
            "tenpai_ece_triage_met": t_ece is not None and t_ece <= 0.10,
            "tenpai_ece_delivery_met": t_ece is not None and t_ece <= 0.05,
            "win_ece_triage_met": w_ece is not None and w_ece <= 0.10,
            "win_ece_delivery_met": w_ece is not None and w_ece <= 0.05,
            "ev_value_ce_over_std_delivery_met": v_ratio is not None and v_ratio <= 0.25,
            "v3_triage_met": t_ece is not None and w_ece is not None and t_ece <= 0.10 and w_ece <= 0.10,
            "v3_delivery_met": (t_ece is not None and w_ece is not None and v_ratio is not None
                                and t_ece <= 0.05 and w_ece <= 0.05 and v_ratio <= 0.25),
        }

    out["prereg"]["sha256_end"] = assert_prereg_unchanged(prereg_path, sha_start)
    out["prereg"]["immutable_during_run"] = out["prereg"]["sha256_end"] == sha_start
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_clean(out), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] {_rel(out_path)} (n_available={n_available}, n_aligned={n_aligned}, "
          f"not_estimable={out['not_estimable']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
