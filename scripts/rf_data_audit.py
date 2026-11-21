#!/usr/bin/env python3
"""RiichiFormer V2 PR-02: read-only raw-year data audit.

Usage:
  python3 scripts/rf_data_audit.py --roots datasets/ --mode scan-only \
      --out artifacts/riichiformer/audit

Outputs (under ``--out``):
  source_audit.parquet   one row per discovered leaf log / zip container
  yearly_counts.csv      per year × source: files, compression, bytes, samples
  bad_files.jsonl        corrupt/empty/unknown files and sample parse errors
  samples.jsonl          deterministic per-year deep sample (event/kyoku counts)
  source_rights.csv      source rights registry (all `unknown` until reviewed)
  audit_summary.json     machine-readable summary (same object printed)
and a human report at ``--report`` (default docs/riichiformer/DATA_REPORT.md).

Safety: no network, no downloads, no writes to the roots, no identity extraction.
The script exits 0 for a completed audit (including `EMPTY`); `EMPTY` and unknown
source rights set ``training_blocked=true``.  Exit 2 means every root was missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from riichiformer.data.scan import run_audit  # noqa: E402

DEFAULT_OUT = "artifacts/riichiformer/audit"
DEFAULT_REPORT = "docs/riichiformer/DATA_REPORT.md"


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo / path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RiichiFormer V2 read-only data audit (PR-02)")
    ap.add_argument("--roots", nargs="+", default=["datasets/"],
                    help="local directories and/or .zip files (default: datasets/)")
    ap.add_argument("--mode", default="scan-only", choices=["scan-only"],
                    help="only scan-only is implemented in PR-02")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report", default=DEFAULT_REPORT, help="markdown report path, or '-'")
    ap.add_argument("--sample-per-year", type=int, default=200)
    ap.add_argument("--hash-mode", choices=["sample", "all", "none"], default="sample")
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    args = ap.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    roots = [_resolve(repo, value) for value in args.roots]
    out_dir = _resolve(repo, args.out)
    report = None if args.report == "-" else _resolve(repo, args.report)

    missing = [str(p) for p in roots if not p.exists()]
    if len(missing) == len(roots):
        print(f"rf_data_audit: no root exists: {missing}", file=sys.stderr)
        return 2

    try:
        summary = run_audit(
            roots=roots,
            out_dir=out_dir,
            report_path=report,
            sample_per_year=args.sample_per_year,
            hash_mode=args.hash_mode,
            repo=repo,
        )
    except Exception as exc:  # noqa: BLE001 - fail loudly, never fake a report
        print(f"rf_data_audit: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"rf_data_audit: status={summary['status']} training_blocked={summary['training_blocked']}")
    print(f"  roots={summary['roots']} missing={summary['missing_roots']}")
    print(f"  leaf_files={summary['leaf_files']} zip_files={summary['zip_files']} "
          f"zip_members={summary['zip_members']} sampled={summary['sampled']} "
          f"bad_files={summary['bad_files']} rows={summary['parquet_rows']}")
    print(f"  outputs={summary['outputs']}")
    if summary["block_reasons"]:
        print("  block_reasons=" + ", ".join(summary["block_reasons"]))
    print("  summary_sha256_source=audit_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
