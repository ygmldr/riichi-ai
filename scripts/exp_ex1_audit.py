#!/usr/bin/env python3
"""EX1 audit foundation: PREREG hash gate + frozen metric contract + metric selftest.

t1 of `.omo/plans/2026-09-23-explainability-metrics-and-validity.md`.

The frozen contract for the four new metrics (`opp_tenpai_prob`, `deal_in`, `exp_loss`,
`wait_membership`) lives in `scripts/exp_ex1_prereg.md` §4 (fenced block
`json ex1-metric-contract`). This harness:

  1. refuses to run without the PREREG (FATAL, non-zero exit);
  2. records the PREREG sha256 at start and re-reads/re-hashes it at end; drift is fatal (M8);
  3. parses the frozen contract block and asserts it equals the in-code expectation, so t4 can
     reference the same literals without re-defining shapes/units/ranges;
  4. `--selftest` exercises the shared A2b ECE/Brier/top-1 routines on synthetic data with
     PREREG-pinned known values (§10);
  5. the default run writes `runs/exp/EX1/prereg_manifest.json` (the t1 evidence artifact).

Later EX1 scripts (t5-t18) must import `load_prereg` / `assert_prereg_unchanged` from here so
every artifact carries the same hash discipline. `extract_ac` is deliberately NOT imported
(CLI-only global `grp_model`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "vendor/mortal/target/release"))
sys.path.insert(0, str(REPO / "vendor/mortal"))

PREREG_PATH = REPO / "scripts/exp_ex1_prereg.md"
OUT_DIR = REPO / "runs/exp/EX1"
PLAN = ".omo/plans/2026-09-23-explainability-metrics-and-validity.md"
CONTRACT_MARKER = "```json ex1-metric-contract"

# In-code mirror of PREREG §4. Equality with the parsed block is asserted on every run.
EXPECTED_CONTRACT = {
    "opp_tenpai": {
        "unit": "probability",
        "fields": {
            "opp_tenpai_prob": {"shape": [3], "unit": "probability", "range": [0.0, 1.0]},
        },
    },
    "danger": {
        "unit": "mixed",
        "unit_by_field": {
            "deal_in": "probability",
            "wait_membership": "probability",
            "exp_loss": "pt_loss",
        },
        "fields": {
            "deal_in": {"shape": [3, 34], "unit": "probability", "range": [0.0, 1.0]},
            "wait_membership": {"shape": [3, 34], "unit": "probability", "range": [0.0, 1.0]},
            "exp_loss": {"shape": [3, 34], "unit": "pt_loss", "range": [0.0, 48000.0]},
        },
    },
}


# ------------------------------------------------------------------------------ prereg gate


def load_prereg(path: Path) -> str:
    """Return the PREREG sha256; exit FATAL if the pre-registration is missing (A2b:709-712)."""
    if not path.exists():
        raise SystemExit(f"FATAL: {path} must exist before the EX1 audit runs (pre-registration)")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_prereg_unchanged(path: Path, sha_start: str) -> str:
    """Re-hash the PREREG at end-of-run; a changed hash is fatal (M8)."""
    sha_end = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha_end != sha_start:
        raise SystemExit(
            f"FATAL: PREREG drift detected: {path} sha256 {sha_start} -> {sha_end} (M8)"
        )
    return sha_end


def parse_contract(text: str) -> dict:
    """Extract and json.loads the fenced `json ex1-metric-contract` block from the PREREG."""
    start = text.find(CONTRACT_MARKER)
    if start < 0:
        raise SystemExit(f"FATAL: PREREG is missing the '{CONTRACT_MARKER}' frozen-contract block")
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end < 0:
        raise SystemExit("FATAL: PREREG frozen-contract block is unterminated")
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"FATAL: PREREG frozen-contract block is not valid JSON: {exc}") from exc


def contract_diff(got, expected, where: str = "contract") -> list[str]:
    """Recursive diff between the PREREG contract and EXPECTED_CONTRACT (empty == match)."""
    errors: list[str] = []
    if isinstance(expected, dict) and isinstance(got, dict):
        for key in sorted(set(expected) | set(got)):
            if key not in got:
                errors.append(f"{where}.{key}: missing in PREREG")
            elif key not in expected:
                errors.append(f"{where}.{key}: unexpected in PREREG")
            else:
                errors.extend(contract_diff(got[key], expected[key], f"{where}.{key}"))
    elif got != expected:
        errors.append(f"{where}: expected {expected!r}, got {got!r}")
    return errors


# ----------------------------------------------------------------------------------- selftest


def _bootstrap_mortal_paths() -> None:
    """Make `mortal.model` and its sibling `event_data` importable (upstream bare import).

    `vendor/mortal/mortal/` is a namespace package (no `__init__.py`) and `model.py` does a bare
    `from event_data import ...`, so that directory must itself be on sys.path. It also contains
    a `mortal.py` script that would shadow the namespace package if added first — so pin the
    namespace package before appending (mirrors the path setup in `exp_e3a_selfcheck.py:26`).
    """
    import mortal  # noqa: F401, PLC0415  # pin before its own directory joins sys.path
    sys.path.append(str(REPO / "vendor/mortal/mortal"))


def selftest() -> dict:
    """Synthetic checks of the shared A2b ECE/Brier/top-1 routines (PREREG §10).

    Imports are lazy: torch/libriichi are only needed on this path, never for the hash gate.
    """
    _bootstrap_mortal_paths()
    import numpy as np  # noqa: PLC0415

    from exp_a2b_calibration import (  # noqa: PLC0415
        N_BINS,
        bin_sums,
        brier_score,
        ece_from_sums,
        ece_score,
        top1_confidence,
    )

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
        k = 24
        probs = np.full((n, k), (1.0 - v) / (k - 1))
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
    _, cnt, sconf, sy = bin_sums(conf, hit, np.zeros(len(hit), dtype=np.int64))
    assert abs(ece_from_sums(cnt, sconf, sy) - e_top1) < 1e-12
    out["cluster_path_ok"] = True
    out["n_samples_perfect"] = int(out["perfect_n"])
    return out


# --------------------------------------------------------------------------------------- main


def _rel(path: Path) -> str:
    """Repo-relative display path (falls back to absolute for out-of-repo --prereg)."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _git_head() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return proc.stdout.strip() or None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="EX1 audit foundation (PREREG gate + selftest)")
    ap.add_argument("--prereg", type=Path, default=PREREG_PATH,
                    help="frozen pre-registration (default: scripts/exp_ex1_prereg.md)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="artifact directory (default: runs/exp/EX1)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the PREREG-pinned synthetic metric checks and exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    prereg_path = args.prereg.resolve()

    # start-of-run gate: the PREREG must exist and its hash is recorded
    sha_start = load_prereg(prereg_path)
    contract = parse_contract(prereg_path.read_text(encoding="utf-8"))
    errors = contract_diff(contract, EXPECTED_CONTRACT)
    if errors:
        raise SystemExit("FATAL: frozen metric contract drift:\n  " + "\n  ".join(errors))

    checks = selftest()

    # end-of-run gate: re-read and re-hash; drift is fatal (M8)
    sha_end = assert_prereg_unchanged(prereg_path, sha_start)

    if args.selftest:
        print(json.dumps({
            "selftest": "PASS",
            "prereg": {"path": _rel(prereg_path), "sha256": sha_start},
            "contract_ok": True,
            **checks,
        }, indent=2))
        return 0

    manifest = {
        "schema": "ex1-prereg-manifest/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": PLAN,
        "prereg": {
            "path": _rel(prereg_path),
            "sha256_start": sha_start,
            "sha256_end": sha_end,
            "bytes": prereg_path.stat().st_size,
            "immutable_during_run": sha_start == sha_end,
        },
        "frozen_contract": contract,
        "frozen_contract_ok": True,
        "selftest": checks,
        "git_head": _git_head(),
    }
    out_path = Path(args.out_dir) / "prereg_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {_rel(out_path)}")
    print(json.dumps({
        "prereg_sha256": sha_start,
        "contract_ok": True,
        "selftest": {k: checks[k] for k in
                     ("perfect_ece", "known_miscal_ece", "top1_perfect_ece", "brier_sanity")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
